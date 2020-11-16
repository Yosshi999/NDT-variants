#include <iostream>
#include <chrono>
#include <string>
#include <thread>
#include <algorithm>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid_covariance.h>

using PC = pcl::PointCloud<pcl::PointXYZ>;
using PPC = pcl::PointCloud<pcl::PointXYZ>::Ptr;
using CPPC = pcl::PointCloud<pcl::PointXYZ>::ConstPtr;
constexpr double SCALE = 20;
constexpr double CAM = 0.5 * SCALE;
constexpr double resolution = 0.02 * SCALE;

int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage: " << argv[0] << " " << "[ply file to visualize]" << std::endl;
    return -1;
  }
  PPC cloud (new PC);
  std::cout << "loading..." << std::endl;
  pcl::io::loadPLYFile(argv[1], *cloud);
  auto aux = Eigen::Affine3f::Identity();
  aux.scale(SCALE);
  pcl::transformPointCloud(*cloud, *cloud, aux);

  std::cout << "filtering..." << std::endl;
  pcl::VoxelGridCovariance<pcl::PointXYZ> cells;
  cells.setLeafSize(resolution, resolution, resolution);
  cells.setInputCloud(cloud);
  cells.filter(true);


  auto* viewer = new pcl::visualization::PCLVisualizer("viewer");
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addCoordinateSystem(resolution);
  viewer->initCameraParameters();
  viewer->setCameraPosition(0, CAM, 0, 0, 0, 0, 0, 0, 1);

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
      green(cloud, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud, green, "cloud");

#if 0
  PPC gm (new PC);
  cells.getDisplayCloud(*gm);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
      red(gm, 255, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ>(gm, red, "gaussian mixture");

#else

  for (const auto& [i, leaf] : cells.getLeaves())
  {
    if (leaf.nr_points < static_cast<long>(cells.getMinimumPointsNumberPerVoxel())) continue;
    Eigen::EigenSolver<Eigen::Matrix3d> es;
    es.compute(leaf.getCov());
    const Eigen::Matrix3f eigenMat = es.eigenvectors().real().cast<float>();
    const Eigen::Vector3f ev0 = eigenMat.col(0);
    const Eigen::Vector3f ev1 = eigenMat.col(1);
    Eigen::Matrix3f rotMat;
    rotMat.block<3,1>(0,0) = ev0;
    rotMat.block<3,1>(0,1) = ev1;
    rotMat.block<3,1>(0,2) = ev0.cross(ev1);
    Eigen::Quaternionf rotQuat(rotMat.cast<float>());
    const Eigen::Vector3d vals = es.eigenvalues().real();
    double v0, v1, v2;
    if (sqrt(abs(vals[2])) * 1 > resolution)
    {
      v2 = resolution;
      v1 = resolution * sqrt(abs(vals[1])) / sqrt(abs(vals[2]));
      v0 = resolution * sqrt(abs(vals[0])) / sqrt(abs(vals[2]));
    }
    else
    {
      v2 = sqrt(abs(vals[2]));
      v1 = sqrt(abs(vals[1]));
      v0 = sqrt(abs(vals[0]));
    }
    std::cout << vals.transpose() << std::endl;
    viewer->addCube(
        leaf.getMean().cast<float>(),
        rotQuat,
	v0,
	v1,
	v2,
        "cube" + std::to_string(i) );
  }
#endif

  while (!viewer->wasStopped())
  {
    viewer->spinOnce();
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
  }

  return 0;
}
