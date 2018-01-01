cmake_minimum_required(VERSION 2.8)
project(Image_Feature)

add_library(${PROJECT_NAME} STATIC
            src/Feature_base.cpp)

target_link_libraries(${PROJECT_NAME}
        PRIVATE ${OpenCV_LIBS}
        PRIVATE ${Ceres_LIBS}
        PRIVATE ${Sophus_LIBRARIES})

