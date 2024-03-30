cmake_minimum_required(VERSION 3.0)

project(Example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")

set(SIMPLE_TENSORRT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

#####################example#####################
# opencv
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${SIMPLE_TENSORRT_DIR}/include
)

link_directories(
    ${SIMPLE_TENSORRT_DIR}/lib
)

# resnet50
add_executable(resnet50
    resnet50/resnet50.cpp
)
target_link_libraries(resnet50
    ${OpenCV_LIBS}
    simple_tensorrt
)

add_executable(resnet50_async
    resnet50/resnet50_async.cpp
)
target_link_libraries(resnet50_async
    ${OpenCV_LIBS}
    simple_tensorrt
)

# yolov8_det
add_executable(yolov8_det
    yolov8_det/yolov8_det.cpp
)
target_link_libraries(yolov8_det
    ${OpenCV_LIBS}
    simple_tensorrt
)
