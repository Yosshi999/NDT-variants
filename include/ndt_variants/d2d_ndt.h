#pragma once

#include <pcl/common/utils.h>
#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/registration/registration.h>
#include <pcl/pcl_macros.h>

#include <Eigen/StdVector>

namespace pclex {
using namespace pcl;
using boost::shared_ptr;
using boost::make_shared;

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

  D2DNDT(float resolution = 1.0f, double step_size = 0.1, double outlier_ratio = 0.55);
  ~D2DNDT() {}

  inline void
  setInputTarget(const PointCloudTargetConstPtr& cloud) override
  {
    Parent::setInputTarget(cloud);
    init();
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
    auto mat = cos(norm) * Eigen::Matrix3d::Identity() +
               (1 - cos(norm)) * v * v.transpose() + sin(norm) * wedge;
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
  computeTransformation(PointCloudSource& output,
                        const Eigen::Matrix4f& guess) override;

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

  bool
  updateIntervalMT(double& a_l,
                   double& f_l,
                   double& g_l,
                   double& a_u,
                   double& f_u,
                   double& g_u,
                   double a_t,
                   double f_t,
                   double g_t) const;

  double
  trialValueSelectionMT(double a_l,
                        double f_l,
                        double g_l,
                        double a_u,
                        double f_u,
                        double g_u,
                        double a_t,
                        double f_t,
                        double g_t) const;

  inline double
  auxilaryFunction_PsiMT(
      double a, double f_a, double f_0, double g_0, double mu = 1.e-4) const
  {
    return f_a - f_0 - mu * g_0 * a;
  }

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
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace pclex

#include "impl/d2d_ndt.hpp"
