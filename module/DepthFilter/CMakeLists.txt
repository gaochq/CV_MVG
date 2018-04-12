cmake_minimum_required(VERSION 2.8)
project(DepthFilter)

add_library(${PROJECT_NAME} STATIC
        src/Depth_Filter.cpp)

target_link_libraries(${PROJECT_NAME}
        PUBLIC DepthFilter)

#Test depth filter
add_executable(test_Depth_Filter test/Test_DepthFilter.cpp)
target_link_libraries(test_Depth_Filter ${PROJECT_NAME})

