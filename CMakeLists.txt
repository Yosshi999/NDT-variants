cmake_minimum_required(VERSION 3.17)

project(main)

find_package(PCL 1.10 REQUIRED)
find_package(MRPT 2.1 REQUIRED COMPONENTS poses)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})
add_compile_options(-Wall -Wextra -O1)

add_executable(voxel_grid_filter sample/voxel_grid_filter.cpp)
set_property(TARGET voxel_grid_filter PROPERTY CXX_STANDARD 17)
target_link_libraries(voxel_grid_filter ${PCL_LIBRARIES})


add_executable(ndt sample/ndt.cpp)
set_property(TARGET ndt PROPERTY CXX_STANDARD 17)
target_include_directories(ndt PUBLIC include)
target_precompile_headers(ndt PUBLIC
	include/ndt_variants/d2d_ndt.h include/ndt_variants/impl/d2d_ndt.hpp)
target_link_libraries(ndt ${PCL_LIBRARIES})


add_executable(vis_vgc sample/visualize_vgc.cpp)
set_property(TARGET vis_vgc PROPERTY CXX_STANDARD 17)
target_link_libraries(vis_vgc ${PCL_LIBRARIES})


add_executable(mrpt sample/mrpt.cpp)
set_property(TARGET mrpt PROPERTY CXX_STANDARD 17)
target_link_libraries(mrpt ${MRPT_LIBRARIES})
