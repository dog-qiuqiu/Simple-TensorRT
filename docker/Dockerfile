# syntax=docker/dockerfile:1

# Version
ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8
ARG OS_VERSION=22.04

# 基础镜像
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${OS_VERSION} 

# 更新apt源
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

# # 安装TensorRT
COPY --link nv-tensorrt-local-repo-ubuntu2204-8.5.1-cuda-11.8_1.0-1_amd64.deb /tmp/nv-tensorrt-local-repo-ubuntu2204-8.5.1-cuda-11.8_1.0-1_amd64.deb
RUN dpkg -i /tmp/nv-tensorrt-local-repo-ubuntu2204-8.5.1-cuda-11.8_1.0-1_amd64.deb && \ 
    cp /var/nv-tensorrt-local-repo-ubuntu2204-8.5.1-cuda-11.8/nv-tensorrt-local-4D5FC10E-keyring.gpg /usr/share/keyrings/ && \
    mv /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list.bak && \
    apt update && apt install -y --no-install-recommends tensorrt cmake pkg-config ffmpeg libavformat-dev libavcodec-dev libswscale-dev 

# 拷贝预编译的OpenCV库
COPY --link ./3rdparty/opencv-4.7.0/usr/local /usr/local

# 添加环境变量
ENV PATH /usr/src/tensorrt/bin:$PATH

ENV NVIDIA_DRIVER_CAPABILITIES video,graphics,compute,utility