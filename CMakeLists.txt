cmake_minimum_required(VERSION 2.8)
project(CV_MVG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -o3")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_module)

include_directories(${CMAKE_BINARY_DIR}/include)
link_directories(${CMAKE_BINARY_DIR}/lib)

# Opencv
find_package(OpenCV 2.4.13 REQUIRED)
include_directories(${Opencv_INLCUDE_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${Ceres_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")

include_directories(module/Image_Feature/include)
add_subdirectory(module/Image_Feature Image_Feature)

include_directories(module/Epipolar_Geometry/include)
add_subdirectory(module/Epipolar_Geometry Epipolar_Geometry)
add_dependencies(Epipolar_Geometry Image_Feature)

include_directories(module/LKT_Alignment/include)
add_subdirectory(module/LKT_Alignment LKT_Alignment)
add_dependencies(LKT_Alignment Image_Feature Epipolar_Geometry)



