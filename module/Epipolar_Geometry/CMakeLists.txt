cmake_minimum_required(VERSION 2.8)
project(Epipolar_Geometry)

add_library(${PROJECT_NAME} STATIC
            src/Epipolar_base.cpp
            src/Solve_Homography.cpp
            src/Solve_Fundamental.cpp)

target_link_libraries(${PROJECT_NAME}
                PUBLIC Image_Feature)

#Test fundamental solver
add_executable(test_Fundamental_solver test/Test_FundamentalSolver.cpp)
target_link_libraries(test_Fundamental_solver ${PROJECT_NAME})

#Test homography solver
add_executable(test_Homography_solver test/Test_HomographySolver.cpp)
target_link_libraries(test_Homography_solver ${PROJECT_NAME})