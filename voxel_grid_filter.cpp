// https://pcl.readthedocs.io/projects/tutorials/en/latest/voxel_grid.html#voxelgrid
// https://pcl.readthedocs.io/projects/tutorials/en/latest/cloud_viewer.html#cloud-viewer

#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cmath>
#include <iostream>
#define PI 3.14159265

using PC = pcl::PointCloud<pcl::PointXYZ>;
using PPC = pcl::PointCloud<pcl::PointXYZ>::Ptr;
using CPPC = pcl::PointCloud<pcl::PointXYZ>::ConstPtr;

int
main(int argc, char** argv)
{
  constexpr int circle_num = 50;
  PC cloud;
  cloud.width = circle_num * circle_num;
  cloud.height = 1;
  cloud.is_dense = false;
  cloud.points.resize(cloud.width * cloud.height);

  for (size_t i = 0; i < circle_num; i++) {
    float x0 = 2 + cos(2 * PI / circle_num * i);
    float y0 = sin(2 * PI / circle_num * i);
    float z0 = 0;
    for (size_t j = 0; j < circle_num; j++) {
      float x = x0 * cos(2 * PI / circle_num * j) - z0 * sin(2 * PI / circle_num * j);
      float y = y0;
      float z = x0 * sin(2 * PI / circle_num * j) + z0 * cos(2 * PI / circle_num * j);
      cloud.points.at(circle_num * i + j) = pcl::PointXYZ(x, y, z);
    }
  }

  CPPC cloud_ptr(new PC(cloud));
  PPC cloud_filtered(new PC);

  // Create the filtering object
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud_ptr);
  sor.setLeafSize(0.5f, 0.5f, 0.5f);
  sor.filter(*cloud_filtered);

  pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
  viewer.showCloud(cloud_filtered);
  while (!viewer.wasStopped())
    ;

  return (0);
}
