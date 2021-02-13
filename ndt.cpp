#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/registration_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "ndt_variants/d2d_ndt.h"

#include <iostream>
#include <random>

#define REGVIS 0

using PC = pcl::PointCloud<pcl::PointXYZ>;
using PPC = pcl::PointCloud<pcl::PointXYZ>::Ptr;
using CPPC = pcl::PointCloud<pcl::PointXYZ>::ConstPtr;
constexpr int CAM = 1;
constexpr double approxleaf = 0.01;
constexpr double ndtresol = 0.02;
constexpr bool apply_noise = false;
constexpr float noise_scale = 0.01f;

int
main(int argc, char* argv[])
{
  PPC cloud(new PC);
  pcl::io::loadPLYFile(argv[1], *cloud);

  PPC cloud2(new PC);
  pcl::copyPointCloud(*cloud, *cloud2);

  if (apply_noise) {
    std::default_random_engine gen(0);
    std::normal_distribution<float> dist(-noise_scale, noise_scale);
    for (auto& pt : cloud2->points) {
      pt.x += dist(gen);
      pt.y += dist(gen);
      pt.z += dist(gen);
    }
  }

  Eigen::Affine3f affine = Eigen::Translation3f(0.1, 0, 0) *
                           Eigen::AngleAxisf(0.1 * M_PI, Eigen::Vector3f::UnitX());
  pcl::transformPointCloud(*cloud2, *cloud2, affine);

  PPC filtered2(new PC);
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> approx;
  approx.setLeafSize(approxleaf, approxleaf, approxleaf);
  approx.setInputCloud(cloud2);
  approx.filter(*filtered2);

#if 0 // show initial pointclouds
  {
  auto *viewer = new pcl::visualization::PCLVisualizer();
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addCoordinateSystem(0.1);
  viewer->initCameraParameters();
  viewer->setCameraPosition(0, CAM, 0, 0, 0, 0, 0, 0, 1);
  // viewer.showCloud(cloud_filtered);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloud, 255, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud, red, "cloud1");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(cloud2, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud2, green, "cloud2");
  while (!viewer->wasStopped())
  {
    viewer->spinOnce();
    boost::this_thread::sleep(boost::posix_time::milliseconds(100));
  }
  viewer->spinOnce();
  }

#endif
  std::cout << "calculation..." << std::endl;

  PPC aligned(new PC);
  auto init_guess = Eigen::Matrix4f::Identity();
#if 1
  pclex::D2DNDT<pcl::PointXYZ, pcl::PointXYZ> ndt(ndtresol, 0.5);
#else
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  ndt.setResolution(ndtresol);
  ndt.setStepSize(0.5);
#endif
  ndt.setMaximumIterations(40);
  ndt.setTransformationEpsilon(1e-9);
  ndt.setTransformationRotationEpsilon(0);
  ndt.setInputSource(cloud2);
  ndt.setInputTarget(cloud);

#if REGVIS
  pcl::RegistrationVisualizer<pcl::PointXYZ, pcl::PointXYZ> regvis;
  regvis.setRegistration(ndt);
  regvis.startDisplay();
#else
  auto* regvis = new pcl::visualization::PCLVisualizer();
  regvis->setBackgroundColor(0, 0, 0);
  regvis->addCoordinateSystem(0.1);
  regvis->initCameraParameters();
  regvis->setCameraPosition(0, CAM, 0, 0, 0, 0, 0, 0, 1);
  boost::function<void(
      const PC&, const std::vector<int>&, const PC&, const std::vector<int>)>
      callback = [&regvis](const PC& _src,
                           const std::vector<int>& indsrc,
                           const PC& _tgt,
                           const std::vector<int>& indtgt) {
        PPC src(new PC), tgt(new PC);
        pcl::copyPointCloud(_src, *src);
        pcl::copyPointCloud(_tgt, *tgt);
        regvis->removeAllPointClouds();
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(
            src, 255, 0, 0);
        regvis->addPointCloud<pcl::PointXYZ>(src, red, "source");
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(
            tgt, 0, 255, 0);
        regvis->addPointCloud<pcl::PointXYZ>(tgt, green, "target");
        regvis->spinOnce();
        boost::this_thread::sleep(boost::posix_time::milliseconds(100));
      };
  ndt.registerVisualizationCallback(callback);
#endif

  ndt.align(*aligned, init_guess);
  pcl::transformPointCloud(*filtered2, *aligned, ndt.getFinalTransformation());
  std::cout << "steps: " << ndt.getFinalNumIteration() << std::endl;
  std::cout << ndt.getFinalTransformation() << std::endl;

#if REGVIS
  boost::this_thread::sleep(boost::posix_time::milliseconds(2000));
  regvis.stopDisplay();
#endif

  // show results
  {
    auto* viewer = new pcl::visualization::PCLVisualizer();
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(0.1);
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, CAM, 0, 0, 0, 0, 0, 0, 1);
    // viewer.showCloud(cloud_filtered);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(
        cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, red, "cloud1");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(
        aligned, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(aligned, green, "cloud2");
    while (!viewer->wasStopped()) {
      viewer->spinOnce();
      boost::this_thread::sleep(boost::posix_time::milliseconds(100));
    }
  }

  return 0;
}
