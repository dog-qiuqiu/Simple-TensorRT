cmake_minimum_required(VERSION 3.0)

project(SimpleTensorRT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")

#####################simple_tensorrt#####################
# cuda
find_package(CUDA REQUIRED)
message(STATUS "CUDA version: ${CUDA_VERSION}")

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/interface
)

add_library(simple_tensorrt SHARED
    src/simple_tensorrt.cpp
    src/simple_tensorrt_impl.cpp
    src/tool.cpp
)

target_link_libraries(simple_tensorrt
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvidia-ml
    nvinfer 
    nvinfer_plugin 
    nvparsers
)

# install
install(TARGETS simple_tensorrt DESTINATION lib)
install(DIRECTORY src/interface/ DESTINATION include/ FILES_MATCHING PATTERN "*.h")
