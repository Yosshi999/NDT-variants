cmake_minimum_required(VERSION 3.17)

project(main)

find_package(PCL 1.9 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})
add_compile_options(-Wall -O3)

add_executable(voxel_grid_filter voxel_grid_filter.cpp)
add_executable(ndt ndt.cpp)
set_property(TARGET voxel_grid_filter PROPERTY CXX_STANDARD 14)
set_property(TARGET ndt PROPERTY CXX_STANDARD 14)
target_include_directories(ndt PUBLIC include)
target_precompile_headers(ndt PUBLIC
	include/ndt_variants/d2d_ndt.h include/ndt_variants/impl/d2d_ndt.hpp)
target_link_libraries(voxel_grid_filter ${PCL_LIBRARIES})
target_link_libraries(ndt ${PCL_LIBRARIES})
