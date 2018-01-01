cmake_minimum_required(VERSION 2.8)
project(LKT_Alignment)

add_library(${PROJECT_NAME} STATIC
            src/Sprase_ImageAlign.cpp)

target_link_libraries(${PROJECT_NAME}
                PUBLIC Image_Feature)

#Test Srase image align
add_executable(test_Sprase_imageAlign test/Test_SpraseImgALign.cpp)
target_link_libraries(test_Sprase_imageAlign ${PROJECT_NAME})
