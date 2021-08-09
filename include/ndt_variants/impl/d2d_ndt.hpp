#ifndef PCLEX_D2DNDT_IMPL_H_
#define PCLEX_D2DNDT_IMPL_H_

namespace pclex {
using namespace pcl;

template <typename PointSource, typename PointTarget>
D2DNDT<PointSource, PointTarget>::D2DNDT()
: target_cells_()
, resolution_(1.0f)
, step_size_(0.1)
, outlier_ratio_(0.55)
, min_points_per_voxel_(6)
, neighbor_points_(2)
, gauss_d1_()
, gauss_d2_()
, trans_probability_()
{
  reg_name_ = "D2DNDT";

  // Initializes the gaussian fitting parameters (eq. 6.8) [Magnusson 2009]
  const double gauss_c1 = 10.0 * (1 - outlier_ratio_);
  const double gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
  const double gauss_d3 = -std::log(gauss_c2);
  gauss_d1_ = -std::log(gauss_c1 + gauss_c2) - gauss_d3;
  gauss_d2_ =
      -2 * std::log((-std::log(gauss_c1 * std::exp(-0.5) + gauss_c2) - gauss_d3) /
                    gauss_d1_);

  transformation_epsilon_ = 0.1;
  max_iterations_ = 35;
}

template <typename PointSource, typename PointTarget>
void
D2DNDT<PointSource, PointTarget>::computeTransformation(PointCloudSource& output,
                                                        const Eigen::Matrix4f& guess)
{
  // Initialize source grid
  initSource();

  nr_iterations_ = 0;
  converged_ = false;

  // Initializes the gaussian fitting parameters (eq. 6.8) [Magnusson 2009]
  const double gauss_c1 = 10 * (1 - outlier_ratio_);
  const double gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
  const double gauss_d3 = -std::log(gauss_c2);
  gauss_d1_ = -std::log(gauss_c1 + gauss_c2) - gauss_d3;
  gauss_d2_ =
      -2 * std::log((-std::log(gauss_c1 * std::exp(-0.5) + gauss_c2) - gauss_d3) /
                    gauss_d1_);

  if (guess != Eigen::Matrix4f::Identity()) {
    // Initialise final transformation to the guessed one
    final_transformation_ = guess;
    // Apply guessed transformation prior to search for neighbours
    transformPointCloud(output, output, guess);
  }

  // Update source grid
  updateSource(guess);

  // Initialize Point/Covariance Gradient and Hessian
  point_jacobian_.setZero();
  point_jacobian_.block<3, 3>(0, 0).setIdentity();
  point_hessian_.setZero();
  cov_jacobian_.setZero();
  cov_hessian_.setZero();

  Eigen::Matrix<double, 6, 1> score_gradient;

  Eigen::Matrix<double, 6, 6> hessian;

  // Calculate derivates of initial transform vector, subsequent derivative calculations
  // are done in the step length determination.
  double score = computeDerivatives(score_gradient, hessian);

  while (!converged_) {
    // Store previous transformation
    previous_transformation_ = transformation_;

    // Solve for decent direction using newton method, line 23 in Algorithm 2 [Magnusson
    // 2009]
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> sv(
        hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Negative for maximization as opposed to minimization
    Eigen::Matrix<double, 6, 1> delta = sv.solve(-score_gradient);

    // Calculate step length with guarnteed sufficient decrease [More, Thuente 1994]
    double delta_norm = delta.norm();

    if (delta_norm == 0 || std::isnan(delta_norm)) {
      trans_probability_ = score / static_cast<double>(trans_covs_.size());
      converged_ = delta_norm == 0;
      break;
    }

    delta /= delta_norm;
    delta_norm = computeStepLengthMT(delta,
                                     delta_norm,
                                     step_size_,
                                     transformation_epsilon_ / 2,
                                     score,
                                     score_gradient,
                                     hessian);
    delta *= delta_norm;

    // Convert delta into matrix form
    convertTransform(delta, transformation_);

    // transform += delta;

    // Update Visualizer (untested)
    if (update_visualizer_)
      update_visualizer_(output, std::vector<int>(), *target_, std::vector<int>());

    const double cos_angle =
        0.5 * transformation_.template block<3, 3>(0, 0).trace() - 1;
    const double translation_sqr =
        transformation_.template block<3, 1>(0, 3).squaredNorm();

    nr_iterations_++;

    if (nr_iterations_ >= max_iterations_ ||
        ((transformation_epsilon_ > 0 && translation_sqr <= transformation_epsilon_) &&
         (transformation_rotation_epsilon_ > 0 &&
          cos_angle >= transformation_rotation_epsilon_)) ||
        ((transformation_epsilon_ <= 0) &&
         (transformation_rotation_epsilon_ > 0 &&
          cos_angle >= transformation_rotation_epsilon_)) ||
        ((transformation_epsilon_ > 0 && translation_sqr <= transformation_epsilon_) &&
         (transformation_rotation_epsilon_ <= 0))) {
      converged_ = true;
    }
  }

  transformPointCloud(*input_, output, final_transformation_);

  // Store transformation probability.  The realtive differences within each scan
  // registration are accurate but the normalization constants need to be modified for
  // it to be globally accurate
  trans_probability_ = score / static_cast<double>(trans_covs_.size());
}

template <typename PointSource, typename PointTarget>
double
D2DNDT<PointSource, PointTarget>::computeDerivatives(
    Eigen::Matrix<double, 6, 1>& score_gradient,
    Eigen::Matrix<double, 6, 6>& hessian,
    bool compute_hessian)
{
  score_gradient.setZero();
  hessian.setZero();
  double score = 0;

  for (size_t i = 0; i < trans_covs_.size(); i++) {
    const auto& x_trans_cov = trans_covs_.at(i);
    const auto& x_trans = trans_means_.at(i);
    const auto x_trans_pt = PointSource(x_trans(0), x_trans(1), x_trans(2));

    // Find neighbors (Radius search has been experimentally faster than direct neighbor
    // checking.
    std::vector<TargetGridLeafConstPtr> neighborhood;
    std::vector<float> distances;
    // target_cells_.radiusSearch (x_trans_pt, resolution_, neighborhood, distances);
    target_cells_.nearestKSearch(x_trans_pt, neighbor_points_, neighborhood, distances);

    computeLocalDerivatives(x_trans, x_trans_cov);
    for (const auto& cell : neighborhood) {
      const Eigen::Vector3d x_diff = x_trans - cell->getMean();
      const Eigen::Matrix3d c_inv = (cell->getCov() + x_trans_cov).inverse();
      score +=
          updateDerivatives(score_gradient, hessian, x_diff, c_inv, compute_hessian);
    }
  }
  return score;
}

template <typename PointSource, typename PointTarget>
void
D2DNDT<PointSource, PointTarget>::computeLocalDerivatives(const Eigen::Vector3d& x,
                                                          const Eigen::Matrix3d& cov,
                                                          bool compute_hessian)
{
  // d/dp (Rx + t)|R=I, p is an element of rotation vector
  // = G x  ...  G is the generator corresponding to p
  point_jacobian_.block<3, 3>(0, 3) = computeWedge(x(0), x(1), x(2)) * (-1);
  // d/dp (RCR^T)|R=I, C is cov
  // = (d/dp R)|R=I CI + IC (d/dp R^T)|R=I
  // = GC + (GC)^T  ...  C = C^T
  {
    Eigen::Matrix3d Gcov, GcovT;
    Gcov <<
      0, 0, 0,
      -cov(0,2), -cov(1,2), -cov(2,2),
       cov(0,1),  cov(1,1),  cov(1,2);
    GcovT = Gcov.transpose();
    cov_jacobian_.block<3, 3>(0, 9) = Gcov + GcovT;

    Gcov <<
      cov(0,2), cov(1,2), cov(2,2),
      0, 0, 0,
      -cov(0,0), -cov(0,1), -cov(0,2);
    GcovT = Gcov.transpose();
    cov_jacobian_.block<3, 3>(0, 12) = Gcov + GcovT;

    Gcov <<
      -cov(0,1), -cov(1,1), -cov(1,2),
      cov(0,0), cov(0,1), cov(0,2),
      0, 0, 0;
    GcovT = Gcov.transpose();
    cov_jacobian_.block<3, 3>(0, 15) = Gcov + GcovT;
  }

  if (compute_hessian) {
    // Calculate second derivative of Transformation Equation 6.17 w.r.t. transform
    // vector p. Derivative w.r.t. ith and jth elements of transform vector corresponds
    // to the 3x1 block matrix starting at (3i,j), Equation 6.20 and 6.21 [Magnusson
    // 2009]

    // d^2/dpdq (Rx) = (GpGq + GqGp)/2 x
    Eigen::Vector3d xy, yz, zx, xx, yy, zz;
    xy << x(1), x(0), 0;
    yz << 0, x(2), x(1);
    zx << x(2), 0, x(0);

    xx << 0, -x(1), -x(2);
    yy << -x(0), 0, -x(2);
    zz << -x(0), -x(1), 0;

    point_hessian_.block<3, 1>(9, 3) = xx;
    point_hessian_.block<3, 1>(12, 3) = xy / 2;
    point_hessian_.block<3, 1>(15, 3) = zx / 2;

    point_hessian_.block<3, 1>(9, 4) = xy / 2;
    point_hessian_.block<3, 1>(12, 4) = yy;
    point_hessian_.block<3, 1>(15, 4) = yz / 2;

    point_hessian_.block<3, 1>(9, 5) = zx / 2;
    point_hessian_.block<3, 1>(12, 5) = yz / 2;
    point_hessian_.block<3, 1>(15, 5) = zz;

    // d^2/dpdq (RCR^T)
    // = (d^2/dpdq R) C R^T +
    //   (d/dp R) C (d/dq R^T) + (d/dq R) C (d/dp R^T) +
    //   R C (d^2/dpdq R^T)
    // d^2/dpdq (R C R^T)|R=I
    // = (GpGq + GqGp)/2 C + Gp C Gq^T + Gq C Gp^T + ((GpGq + GqGp)/2 C)^T
    // = (GpGq + GqGp)/2 C + Gp C Gq^T + (Gp C Gq^T)^T + ((GpGq + GqGp)/2 C)^T
    // ... maybe?
    Eigen::Matrix3d Gen[3];
    Gen[0] <<
      0, 0, 0,
      0, 0, -1,
      0, 1, 0;
    Gen[1] <<
      0, 0, 1,
      0, 0, 0,
      -1, 0, 0;
    Gen[2] <<
      0, -1, 0,
      1, 0, 0,
      0, 0, 0;
    for (int i=0; i<3; i++) {
      for (int j=i; j<3; j++) {
        Eigen::Matrix3d block =
          (Gen[i] * Gen[j] + Gen[j] * Gen[i])/2 * cov
          + Gen[i] * cov * Gen[j].transpose();
        cov_hessian_.block<3, 3>(9+3*i, 9+3*j) = block + block.transpose();
      }
    }
    cov_hessian_.block<3, 3>(12, 9) = cov_hessian_.block<3, 3>(9, 12);
    cov_hessian_.block<3, 3>(15, 9) = cov_hessian_.block<3, 3>(9, 15);
    cov_hessian_.block<3, 3>(15, 12) = cov_hessian_.block<3, 3>(12, 15);
  }
}

template <typename PointSource, typename PointTarget>
double
D2DNDT<PointSource, PointTarget>::updateDerivatives(
    Eigen::Matrix<double, 6, 1>& score_gradient,
    Eigen::Matrix<double, 6, 6>& hessian,
    const Eigen::Vector3d& x_trans,
    const Eigen::Matrix3d& c_inv,
    bool compute_hessian) const
{
  // e^(-d_2/2 * mu_{ij}^T (R^T Sigma_i R + Sigma_j)^{-1} mu_{ij})
  // Equation 18 [Stoyanov et al. 2012]
  double e_x_cov_x = std::exp(-gauss_d2_ * x_trans.dot(c_inv * x_trans) / 2);
  // Calculate probability of transtormed points existance,
  // Equation 18 [Stoyanov et al. 2012]
  const double score_inc = -gauss_d1_ * e_x_cov_x;

  e_x_cov_x = gauss_d2_ * e_x_cov_x;

  // Error checking for invalid values.
  if (e_x_cov_x > 1 || e_x_cov_x < 0 || std::isnan(e_x_cov_x))
    return 0;

  // Reusable portion of Equation 20 and 24 [Stoyanov et al. 2012]
  // d_1 d_2 exp(...)
  e_x_cov_x = e_x_cov_x * gauss_d1_;

  for (int i = 0; i < 6; i++) {
    // B j_a, Reusable portion of Equation 20 and 24 [Stoyanov et al. 2012]
    const Eigen::Vector3d BJ = c_inv * point_jacobian_.col(i);
    const double xtBJ = x_trans.dot(BJ);
    const Eigen::Matrix3d BZiB = c_inv * cov_jacobian_.block<3, 3>(0, 3*i) * c_inv;
    const double xtBZiBx = x_trans.dot( BZiB * x_trans );
    const double qi = 2 * xtBJ - xtBZiBx;
    // Update gradient, Equation 20 [Stoyanov et al. 2012]
    score_gradient(i) += qi * e_x_cov_x / 2;

    if (compute_hessian) {
      for (int j = 0; j < hessian.cols(); j++) {
        // Update hessian, Equation 24 [Stoyanov et al. 2012]
        const double xtBJj = x_trans.dot(c_inv * point_jacobian_.col(j));
        const double xtBZjBx = x_trans.dot( c_inv * cov_jacobian_.block<3, 3>(0, 3*j) * c_inv * x_trans );
        const double qj = 2 * xtBJj - xtBZjBx;
        hessian(i, j) +=
            e_x_cov_x * (2 * point_jacobian_.col(j).dot(BJ)
                        - 2 * x_trans.dot( c_inv * cov_jacobian_.block<3, 3>(0, 3*j) * BJ )
                        + 2 * x_trans.dot( c_inv * point_hessian_.block<3, 1>(3*i, j) )
                        - 2 * x_trans.dot( BZiB * point_jacobian_.col(j) )
                        + x_trans.dot( BZiB * cov_jacobian_.block<3, 3>(0, 3*j) * c_inv * x_trans )
                        + x_trans.dot( c_inv * cov_jacobian_.block<3, 3>(0, 3*j) * BZiB * x_trans )
                        - x_trans.dot( c_inv * cov_hessian_.block<3, 3>(3*i, 3*j) * c_inv * x_trans )
                        - gauss_d2_ * qi * qj / 2
                        );
      }
    }
  }

  return score_inc;
}

template <typename PointSource, typename PointTarget>
void
D2DNDT<PointSource, PointTarget>::computeHessian(Eigen::Matrix<double, 6, 6>& hessian)
{
  hessian.setZero();

  // Precompute Angular Derivatives unessisary because only used after regular
  // derivative calculation Update hessian for each point, line 17 in Algorithm 2
  // [Magnusson 2009]
  for (size_t i = 0; i < trans_covs_.size(); i++) {
    const auto& x_trans_cov = trans_covs_.at(i);
    const auto& x_trans = trans_means_.at(i);
    const auto x_trans_pt = PointSource(x_trans(0), x_trans(1), x_trans(2));

    // Find nieghbors (Radius search has been experimentally faster than direct neighbor
    // checking.
    std::vector<TargetGridLeafConstPtr> neighborhood;
    std::vector<float> distances;
    // target_cells_.radiusSearch (x_trans_pt, resolution_, neighborhood, distances);
    target_cells_.nearestKSearch(x_trans_pt, neighbor_points_, neighborhood, distances);

    for (const auto& cell : neighborhood) {
      // Compute derivative of transform function w.r.t. transform vector,
      // j_a, Z_a, H_{ab} and Z_{ab}
      // in Equations 22, 23, 25 and 26 [Stoyanov et al. 2012]
      computeLocalDerivatives(x_trans, x_trans_cov);

      // Denorm point, mu_{ij} in Equations 19 [Stoyanov et al. 2012]
      const Eigen::Vector3d x_diff = x_trans - cell->getMean();
      // Uses precomputed covariance for speed.
      const Eigen::Matrix3d c_inv = (cell->getCov() + x_trans_cov).inverse();

      // Update hessian, according to Equations 24-27,
      // respectively [Stoyanov et al. 2012]
      updateHessian(hessian, x_diff, c_inv);
    }
  }
}
template <typename PointSource, typename PointTarget>
void
D2DNDT<PointSource, PointTarget>::updateHessian(Eigen::Matrix<double, 6, 6>& hessian,
                                                const Eigen::Vector3d& x_trans,
                                                const Eigen::Matrix3d& c_inv) const
{
  // e^(-d_2/2 * mu_{ij}^T (R^T Sigma_i R + Sigma_j)^{-1} mu_{ij})
  // Equation 18 [Stoyanov et al. 2012]
  double e_x_cov_x =
      gauss_d2_ * std::exp(-gauss_d2_ * x_trans.dot(c_inv * x_trans) / 2);

  // Error checking for invalid values.
  if (e_x_cov_x > 1 || e_x_cov_x < 0 || std::isnan(e_x_cov_x))
    return;

  // Reusable portion of Equation 20 and 24 [Stoyanov et al. 2012]
  // d_1 d_2 exp(...)
  e_x_cov_x = e_x_cov_x * gauss_d1_;

  for (int i = 0; i < 6; i++) {
    // B j_a, Reusable portion of Equation 20 and 24 [Stoyanov et al. 2012]
    const Eigen::Vector3d BJ = c_inv * point_jacobian_.col(i);
    const double xtBJ = x_trans.dot(BJ);
    const Eigen::Matrix3d BZiB = c_inv * cov_jacobian_.block<3, 3>(0, 3*i) * c_inv;
    const double xtBZiBx = x_trans.dot( BZiB * x_trans );
    const double qi = 2 * xtBJ - xtBZiBx;

    for (int j = 0; j < hessian.cols(); j++) {
      // Update hessian, Equation 24 [Stoyanov et al. 2012]
      const double xtBJj = x_trans.dot(c_inv * point_jacobian_.col(j));
      const double xtBZjBx = x_trans.dot( c_inv * cov_jacobian_.block<3, 3>(0, 3*j) * c_inv * x_trans );
      const double qj = 2 * xtBJj - xtBZjBx;
      hessian(i, j) +=
          e_x_cov_x * (2 * point_jacobian_.col(j).dot(BJ)
                      - 2 * x_trans.dot( c_inv * cov_jacobian_.block<3, 3>(0, 3*j) * BJ )
                      + 2 * x_trans.dot( c_inv * point_hessian_.block<3, 1>(3*i, j) )
                      - 2 * x_trans.dot( BZiB * point_jacobian_.col(j) )
                      + x_trans.dot( BZiB * cov_jacobian_.block<3, 3>(0, 3*j) * c_inv * x_trans )
                      + x_trans.dot( c_inv * cov_jacobian_.block<3, 3>(0, 3*j) * BZiB * x_trans )
                      - x_trans.dot( c_inv * cov_hessian_.block<3, 3>(3*i, 3*j) * c_inv * x_trans )
                      - gauss_d2_ * qi * qj / 2
                      );
    }
  }
}

template <typename PointSource, typename PointTarget>
bool
D2DNDT<PointSource, PointTarget>::updateIntervalMT(double& a_l,
                                                   double& f_l,
                                                   double& g_l,
                                                   double& a_u,
                                                   double& f_u,
                                                   double& g_u,
                                                   double a_t,
                                                   double f_t,
                                                   double g_t) const
{
  // Case U1 in Update Algorithm and Case a in Modified Update Algorithm [More, Thuente
  // 1994]
  if (f_t > f_l) {
    a_u = a_t;
    f_u = f_t;
    g_u = g_t;
    return false;
  }
  // Case U2 in Update Algorithm and Case b in Modified Update Algorithm [More, Thuente
  // 1994]
  if (g_t * (a_l - a_t) > 0) {
    a_l = a_t;
    f_l = f_t;
    g_l = g_t;
    return false;
  }
  // Case U3 in Update Algorithm and Case c in Modified Update Algorithm [More, Thuente
  // 1994]
  if (g_t * (a_l - a_t) < 0) {
    a_u = a_l;
    f_u = f_l;
    g_u = g_l;

    a_l = a_t;
    f_l = f_t;
    g_l = g_t;
    return false;
  }
  // Interval Converged
  return true;
}

template <typename PointSource, typename PointTarget>
double
D2DNDT<PointSource, PointTarget>::trialValueSelectionMT(double a_l,
                                                        double f_l,
                                                        double g_l,
                                                        double a_u,
                                                        double f_u,
                                                        double g_u,
                                                        double a_t,
                                                        double f_t,
                                                        double g_t) const
{
  // Case 1 in Trial Value Selection [More, Thuente 1994]
  if (f_t > f_l) {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    const double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    const double w = std::sqrt(z * z - g_t * g_l);
    // Equation 2.4.56 [Sun, Yuan 2006]
    const double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates f_l, f_t and g_l
    // Equation 2.4.2 [Sun, Yuan 2006]
    const double a_q =
        a_l - 0.5 * (a_l - a_t) * g_l / (g_l - (f_l - f_t) / (a_l - a_t));

    if (std::fabs(a_c - a_l) < std::fabs(a_q - a_l)) {
      return a_c;
    }
    return 0.5 * (a_q + a_c);
  }
  // Case 2 in Trial Value Selection [More, Thuente 1994]
  if (g_t * g_l < 0) {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    const double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    const double w = std::sqrt(z * z - g_t * g_l);
    // Equation 2.4.56 [Sun, Yuan 2006]
    const double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates f_l, g_l and g_t
    // Equation 2.4.5 [Sun, Yuan 2006]
    const double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

    if (std::fabs(a_c - a_t) >= std::fabs(a_s - a_t)) {
      return a_c;
    }
    return a_s;
  }
  // Case 3 in Trial Value Selection [More, Thuente 1994]
  if (std::fabs(g_t) <= std::fabs(g_l)) {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    const double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    const double w = std::sqrt(z * z - g_t * g_l);
    const double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates g_l and g_t
    // Equation 2.4.5 [Sun, Yuan 2006]
    const double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

    double a_t_next;

    if (std::fabs(a_c - a_t) < std::fabs(a_s - a_t)) {
      a_t_next = a_c;
    }
    else {
      a_t_next = a_s;
    }

    if (a_t > a_l) {
      return std::min(a_t + 0.66 * (a_u - a_t), a_t_next);
    }
    return std::max(a_t + 0.66 * (a_u - a_t), a_t_next);
  }
  // Case 4 in Trial Value Selection [More, Thuente 1994]
  // Calculate the minimizer of the cubic that interpolates f_u, f_t, g_u and g_t
  // Equation 2.4.52 [Sun, Yuan 2006]
  const double z = 3 * (f_t - f_u) / (a_t - a_u) - g_t - g_u;
  const double w = std::sqrt(z * z - g_t * g_u);
  // Equation 2.4.56 [Sun, Yuan 2006]
  return a_u + (a_t - a_u) * (w - g_u - z) / (g_t - g_u + 2 * w);
}

template <typename PointSource, typename PointTarget>
double
D2DNDT<PointSource, PointTarget>::computeStepLengthMT(
    Eigen::Matrix<double, 6, 1>& step_dir,
    double step_init,
    double step_max,
    double step_min,
    double& score,
    Eigen::Matrix<double, 6, 1>& score_gradient,
    Eigen::Matrix<double, 6, 6>& hessian)
{
  // std::cout << "stepdir " << step_dir << std::endl;

  // Set the value of phi(0), Equation 1.3 [More, Thuente 1994]
  const double phi_0 = -score;
  // Set the value of phi'(0), Equation 1.3 [More, Thuente 1994]
  double d_phi_0 = -(score_gradient.dot(step_dir));

  if (d_phi_0 >= 0) {
    // Not a decent direction
    if (d_phi_0 == 0) {
      return 0;
    }
    // Reverse step direction and calculate optimal step.
    d_phi_0 *= -1;
    step_dir *= -1;
  }

  // The Search Algorithm for T(mu) [More, Thuente 1994]

  const int max_step_iterations = 10;
  int step_iterations = 0;

  // Sufficient decreace constant, Equation 1.1 [More, Thuete 1994]
  const double mu = 1.e-4;
  // Curvature condition constant, Equation 1.2 [More, Thuete 1994]
  const double nu = 0.9;

  // Initial endpoints of Interval I,
  double a_l = 0, a_u = 0;

  // Auxiliary function psi is used until I is determined ot be a closed interval,
  // Equation 2.1 [More, Thuente 1994]
  double f_l = auxilaryFunction_PsiMT(a_l, phi_0, phi_0, d_phi_0, mu);
  double g_l = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

  double f_u = auxilaryFunction_PsiMT(a_u, phi_0, phi_0, d_phi_0, mu);
  double g_u = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

  // Check used to allow More-Thuente step length calculation to be skipped by making
  // step_min == step_max
  bool interval_converged = (step_max - step_min) < 0, open_interval = true;

  double a_t = step_init;
  a_t = std::min(a_t, step_max);
  a_t = std::max(a_t, step_min);

  Eigen::Matrix4f saved_final_transformation = final_transformation_;
  updateTransform(step_dir * a_t, final_transformation_);

  // new transformed source
  // std::cout << "first a_t " << a_t << std::endl;
  updateSource(final_transformation_);

  // Updates score, gradient and hessian.  Hessian calculation is unessisary but testing
  // showed that most step calculations use the initial step suggestion and
  // recalculation the reusable portions of the hessian would intail more computation
  // time.
  score = computeDerivatives(score_gradient, hessian, true);

  // Calculate phi(alpha_t)
  double phi_t = -score;
  // Calculate phi'(alpha_t)
  double d_phi_t = -(score_gradient.dot(step_dir));

  // Calculate psi(alpha_t)
  double psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
  // Calculate psi'(alpha_t)
  double d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

  // Iterate until max number of iterations, interval convergance or a value satisfies
  // the sufficient decrease, Equation 1.1, and curvature condition, Equation 1.2 [More,
  // Thuente 1994]
  while (!interval_converged && step_iterations < max_step_iterations &&
         !(psi_t <= 0 /*Sufficient Decrease*/ &&
           d_phi_t <= -nu * d_phi_0 /*Curvature Condition*/)) {
    // Use auxiliary function if interval I is not closed
    if (open_interval) {
      a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
    }
    else {
      a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
    }

    a_t = std::min(a_t, step_max);
    a_t = std::max(a_t, step_min);

    final_transformation_ = saved_final_transformation;
    updateTransform(step_dir * a_t, final_transformation_);

    // new transformed source
    // std::cout << "a_t " << a_t << std::endl;
    updateSource(final_transformation_);

    // Updates score, gradient. Values stored to prevent wasted computation.
    score = computeDerivatives(score_gradient, hessian, false);

    // Calculate phi(alpha_t+)
    phi_t = -score;
    // Calculate phi'(alpha_t+)
    d_phi_t = -(score_gradient.dot(step_dir));

    // Calculate psi(alpha_t+)
    psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
    // Calculate psi'(alpha_t+)
    d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

    // Check if I is now a closed interval
    if (open_interval && (psi_t <= 0 && d_psi_t >= 0)) {
      open_interval = false;

      // Converts f_l and g_l from psi to phi
      f_l += phi_0 - mu * d_phi_0 * a_l;
      g_l += mu * d_phi_0;

      // Converts f_u and g_u from psi to phi
      f_u += phi_0 - mu * d_phi_0 * a_u;
      g_u += mu * d_phi_0;
    }

    if (open_interval) {
      // Update interval end points using Updating Algorithm [More, Thuente 1994]
      interval_converged =
          updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
    }
    else {
      // Update interval end points using Modified Updating Algorithm [More, Thuente
      // 1994]
      interval_converged =
          updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
    }

    step_iterations++;
  }

  // If inner loop was run then hessian needs to be calculated.
  // Hessian is unnessisary for step length determination but gradients are required
  // so derivative and transform data is stored for the next iteration.
  if (step_iterations) {
    computeHessian(hessian);
  }

  return a_t;
}

} // namespace pclex
#endif // PCLEX_D2DNDT_IMPL_H_
