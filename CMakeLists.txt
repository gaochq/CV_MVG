cmake_minimum_required(VERSION 2.8)
project(CV_MVG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -o3")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_module)

# Opencv
find_package(OpenCV 2.4.13 REQUIRED)
include_directories(${Opencv_INLCUDE_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${Ceres_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

set(PROJECT_NAME
    ${OpenCV_LIBS}
    ${Ceres_LIBS}
    ${Sophus_LIBRARIES})

include_directories(
    ${PROJECT_SOURCE_DIR}/module/Epipolar_Geometry/include
        ${PROJECT_SOURCE_DIR}/module/Image_Feature/include
)

add_library(Epipolar_Geometry
    module/Epipolar_Geometry/src/Solve_Fundamental.cpp
    module/Epipolar_Geometry/src/Solve_Homography.cpp
    module/Epipolar_Geometry/src/Epipolar_base.cpp
    module/Image_Feature/src/Feature_base.cpp
)
