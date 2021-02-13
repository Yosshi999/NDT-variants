#pragma once

#include <pcl/common/utils.h>
#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/registration/registration.h>
#include <pcl/pcl_macros.h>

#include <Eigen/StdVector>

namespace pclex {
using namespace pcl;

template <typename PointSource, typename PointTarget>
class D2DNDT : public Registration<PointSource, PointTarget> {
protected:
  using Parent = Registration<PointSource, PointTarget>;

  using PointCloudSource = typename Parent::PointCloudSource;

  using PointCloudTarget = typename Parent::PointCloudTarget;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

  using SourceGrid = VoxelGridCovariance<PointSource>;
  using SourceGridLeafConstPtr = typename SourceGrid::LeafConstPtr;
  using TargetGrid = VoxelGridCovariance<PointTarget>;
  using TargetGridLeafConstPtr = typename TargetGrid::LeafConstPtr;

public:
  using Ptr = shared_ptr<D2DNDT<PointSource, PointTarget>>;
  using ConstPtr = shared_ptr<const D2DNDT<PointSource, PointTarget>>;

  D2DNDT();
  ~D2DNDT() {}

  inline void
  setInputTarget(const PointCloudTargetConstPtr& cloud) override
  {
    Parent::setInputTarget(cloud);
    init();
  }

  inline float
  getResolution() const
  {
    return resolution_;
  }

  inline void
  setResolution(float resolution)
  {
    resolution_ = resolution;
  }

  inline double
  getStepSize() const
  {
    return step_size_;
  }

  inline void
  setStepSize(double step_size)
  {
    step_size_ = step_size;
  }

  inline double
  getOutlierRatio() const
  {
    return outlier_ratio_;
  }

  inline void
  setOutlierRatio(double outlier_ratio)
  {
    outlier_ratio_ = outlier_ratio;
  }

  inline double
  getTransformationProbability() const
  {
    return trans_probability_;
  }

  inline int
  getFinalNumIteration() const
  {
    return nr_iterations_;
  }

  static inline Eigen::Matrix3d
  computeWedge(double x, double y, double z)
  {
    Eigen::Matrix3d wedge;
    wedge << 0, -z, y, z, 0, -x, -y, x, 0;
    return wedge;
  }

  static inline Eigen::Matrix3d
  computeExpWedge(double x, double y, double z)
  {
    Eigen::Vector3d v(x, y, z);
    double norm = v.norm();
    v.normalize();
    Eigen::Matrix3d wedge = computeWedge(v(0), v(1), v(2));
    auto mat = cos(norm) * Eigen::Matrix3d::Identity() + (1 - cos(norm)) * v * v.transpose() +
               sin(norm) * wedge;
    return mat;
  }

  static void
  convertTransform(const Eigen::Matrix<double, 6, 1>& x, Eigen::Affine3f& trans)
  {
    trans = Eigen::Translation<float, 3>(float(x(0)), float(x(1)), float(x(2))) *
            computeExpWedge(x(3), x(4), x(5)).cast<float>();
  }

  static void
  convertTransform(const Eigen::Matrix<double, 6, 1>& x, Eigen::Matrix4f& trans)
  {
    Eigen::Affine3f _affine;
    convertTransform(x, _affine);
    trans = _affine.matrix();
  }

  static void
  updateTransform(const Eigen::Matrix<double, 6, 1>& x, Eigen::Matrix4f& trans)
  {
    Eigen::Matrix4f delta;
    convertTransform(x, delta);
    trans = delta * trans;
  }

  inline void
  setMinPointPerVoxel(int min_points_per_voxel)
  {
    min_points_per_voxel_ = min_points_per_voxel;
  }

  inline int
  getMinPointPerVoxel()
  {
    return min_points_per_voxel_;
  }

protected:
  using Parent::converged_;
  using Parent::corr_dist_threshold_;
  using Parent::final_transformation_;
  using Parent::getClassName;
  using Parent::indices_;
  using Parent::inlier_threshold_;
  using Parent::input_;
  using Parent::max_iterations_;
  using Parent::nr_iterations_;
  using Parent::previous_transformation_;
  using Parent::reg_name_;
  using Parent::target_;
  using Parent::transformation_;
  using Parent::transformation_epsilon_;
  using Parent::transformation_rotation_epsilon_;

  using Parent::update_visualizer_;

  void
  computeTransformation(PointCloudSource& output, const Eigen::Matrix4f& guess) override;

  void inline init()
  {
    target_cells_.setLeafSize(resolution_, resolution_, resolution_);
    target_cells_.setInputCloud(target_);
    target_cells_.setMinPointPerVoxel(min_points_per_voxel_);
    // Initiate voxel structure.
    target_cells_.filter(true);
  }

  void inline initSource()
  {
    SourceGrid source_cells;
    source_cells.setLeafSize(resolution_, resolution_, resolution_);
    source_cells.setInputCloud(input_);
    source_cells.setMinPointPerVoxel(min_points_per_voxel_);
    source_cells.filter(false);

    source_covs_.clear();
    source_means_.clear();
    trans_covs_.clear();
    trans_means_.clear();
    for (const auto& kv : source_cells.getLeaves()) {
      const auto& value = kv.second;
      if (value.nr_points < min_points_per_voxel_)
        continue;
      source_covs_.push_back(value.getCov());
      trans_covs_.push_back(value.getCov());
      source_means_.push_back(value.getMean());
      trans_means_.push_back(value.getMean());
    }
  }

  void inline updateSource(const Eigen::Matrix4f& mat)
  {
    // std::cout << mat << std::endl;
    Eigen::Matrix3d rot = mat.block<3, 3>(0, 0).template cast<double>();
    Eigen::Vector3d tl = mat.block<3, 1>(0, 3).template cast<double>();
    for (size_t i = 0; i < source_covs_.size(); i++) {
      trans_covs_.at(i) = rot.transpose() * source_covs_.at(i) * rot;
      trans_means_.at(i) = rot * source_means_.at(i) + tl;
    }
  }

  double
  computeDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient,
                     Eigen::Matrix<double, 6, 6>& hessian,
                     bool compute_hessian = true);

  double
  updateDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient,
                    Eigen::Matrix<double, 6, 6>& hessian,
                    const Eigen::Vector3d& x_trans,
                    const Eigen::Matrix3d& c_inv,
                    bool compute_hessian = true) const;

  void
  computePointDerivatives(const Eigen::Vector3d& x, bool compute_hessian = true);

  void
  computeHessian(Eigen::Matrix<double, 6, 6>& hessian);

  void
  updateHessian(Eigen::Matrix<double, 6, 6>& hessian,
                const Eigen::Vector3d& x_trans,
                const Eigen::Matrix3d& c_inv) const;

  double
  computeStepLengthMT(Eigen::Matrix<double, 6, 1>& step_dir,
                      double step_init,
                      double step_max,
                      double step_min,
                      double& score,
                      Eigen::Matrix<double, 6, 1>& score_gradient,
                      Eigen::Matrix<double, 6, 6>& hessian);

    /** \brief Update interval of possible step lengths for More-Thuente method, \f$ I \f$ in More-Thuente (1994)
      * \note Updating Algorithm until some value satisfies \f$ \psi(\alpha_k) \leq 0 \f$ and \f$ \phi'(\alpha_k) \geq 0 \f$
      * and Modified Updating Algorithm from then on [More, Thuente 1994].
      * \param[in,out] a_l first endpoint of interval \f$ I \f$, \f$ \alpha_l \f$ in Moore-Thuente (1994)
      * \param[in,out] f_l value at first endpoint, \f$ f_l \f$ in Moore-Thuente (1994), \f$ \psi(\alpha_l) \f$ for Update Algorithm and \f$ \phi(\alpha_l) \f$ for Modified Update Algorithm
      * \param[in,out] g_l derivative at first endpoint, \f$ g_l \f$ in Moore-Thuente (1994), \f$ \psi'(\alpha_l) \f$ for Update Algorithm and \f$ \phi'(\alpha_l) \f$ for Modified Update Algorithm
      * \param[in,out] a_u second endpoint of interval \f$ I \f$, \f$ \alpha_u \f$ in Moore-Thuente (1994)
      * \param[in,out] f_u value at second endpoint, \f$ f_u \f$ in Moore-Thuente (1994), \f$ \psi(\alpha_u) \f$ for Update Algorithm and \f$ \phi(\alpha_u) \f$ for Modified Update Algorithm
      * \param[in,out] g_u derivative at second endpoint, \f$ g_u \f$ in Moore-Thuente (1994), \f$ \psi'(\alpha_u) \f$ for Update Algorithm and \f$ \phi'(\alpha_u) \f$ for Modified Update Algorithm
      * \param[in] a_t trial value, \f$ \alpha_t \f$ in Moore-Thuente (1994)
      * \param[in] f_t value at trial value, \f$ f_t \f$ in Moore-Thuente (1994), \f$ \psi(\alpha_t) \f$ for Update Algorithm and \f$ \phi(\alpha_t) \f$ for Modified Update Algorithm
      * \param[in] g_t derivative at trial value, \f$ g_t \f$ in Moore-Thuente (1994), \f$ \psi'(\alpha_t) \f$ for Update Algorithm and \f$ \phi'(\alpha_t) \f$ for Modified Update Algorithm
      * \return if interval converges
      */
    bool
    updateIntervalMT (double &a_l, double &f_l, double &g_l,
                      double &a_u, double &f_u, double &g_u,
                      double a_t, double f_t, double g_t);

  /** \brief Select new trial value for More-Thuente method.
    * \note Trial Value Selection [More, Thuente 1994], \f$ \psi(\alpha_k) \f$ is used for \f$ f_k \f$ and \f$ g_k \f$
    * until some value satisfies the test \f$ \psi(\alpha_k) \leq 0 \f$ and \f$ \phi'(\alpha_k) \geq 0 \f$
    * then \f$ \phi(\alpha_k) \f$ is used from then on.
    * \note Interpolation Minimizer equations from Optimization Theory and Methods: Nonlinear Programming By Wenyu Sun, Ya-xiang Yuan (89-100).
    * \param[in] a_l first endpoint of interval \f$ I \f$, \f$ \alpha_l \f$ in Moore-Thuente (1994)
    * \param[in] f_l value at first endpoint, \f$ f_l \f$ in Moore-Thuente (1994)
    * \param[in] g_l derivative at first endpoint, \f$ g_l \f$ in Moore-Thuente (1994)
    * \param[in] a_u second endpoint of interval \f$ I \f$, \f$ \alpha_u \f$ in Moore-Thuente (1994)
    * \param[in] f_u value at second endpoint, \f$ f_u \f$ in Moore-Thuente (1994)
    * \param[in] g_u derivative at second endpoint, \f$ g_u \f$ in Moore-Thuente (1994)
    * \param[in] a_t previous trial value, \f$ \alpha_t \f$ in Moore-Thuente (1994)
    * \param[in] f_t value at previous trial value, \f$ f_t \f$ in Moore-Thuente (1994)
    * \param[in] g_t derivative at previous trial value, \f$ g_t \f$ in Moore-Thuente (1994)
    * \return new trial value
    */
  double
  trialValueSelectionMT (double a_l, double f_l, double g_l,
                          double a_u, double f_u, double g_u,
                          double a_t, double f_t, double g_t);

  /** \brief Auxiliary function used to determine endpoints of More-Thuente interval.
    * \note \f$ \psi(\alpha) \f$ in Equation 1.6 (Moore, Thuente 1994)
    * \param[in] a the step length, \f$ \alpha \f$ in More-Thuente (1994)
    * \param[in] f_a function value at step length a, \f$ \phi(\alpha) \f$ in More-Thuente (1994)
    * \param[in] f_0 initial function value, \f$ \phi(0) \f$ in Moore-Thuente (1994)
    * \param[in] g_0 initial function gradiant, \f$ \phi'(0) \f$ in More-Thuente (1994)
    * \param[in] mu the step length, constant \f$ \mu \f$ in Equation 1.1 [More, Thuente 1994]
    * \return sufficient decrease value
    */
  inline double
  auxilaryFunction_PsiMT(double a, double f_a, double f_0, double g_0, double mu = 1.e-4) const
  {
    return f_a - f_0 - mu * g_0 * a;
  }

  /** \brief Auxiliary function derivative used to determine endpoints of More-Thuente interval.
   * \note \f$ \psi'(\alpha) \f$, derivative of Equation 1.6 (Moore, Thuente 1994)
   * \param[in] g_a function gradient at step length a, \f$ \phi'(\alpha) \f$ in More-Thuente (1994)
   * \param[in] g_0 initial function gradiant, \f$ \phi'(0) \f$ in More-Thuente (1994)
   * \param[in] mu the step length, constant \f$ \mu \f$ in Equation 1.1 [More, Thuente 1994]
   * \return sufficient decrease derivative
   */
  inline double
  auxilaryFunction_dPsiMT(double g_a, double g_0, double mu = 1.e-4) const
  {
    return g_a - mu * g_0;
  }

  TargetGrid target_cells_;

  float resolution_;
  double step_size_;
  double outlier_ratio_;
  int min_points_per_voxel_;

  double gauss_d1_, gauss_d2_;

  double trans_probability_;

  Eigen::Matrix<double, 3, 6> point_jacobian_;
  Eigen::Matrix<double, 18, 6> point_hessian_;

  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> source_covs_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> source_means_;
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> trans_covs_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> trans_means_;

public:
  PCL_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace pclex

#include "impl/d2d_ndt.hpp"
