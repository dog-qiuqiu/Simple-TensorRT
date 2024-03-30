# Simple-TensorRT
* Secondary encapsulation of NVIDIA TensorRT interface to simplify the calling process

# Compilation dependencies
* Linux
* CUDA
* TensorRT 8.5.1(TensorRT 8.5 GA)
* Opencv 4.7.0

# Docker environment
## Install NVIDIA Container Toolkit  
https://zhuanlan.zhihu.com/p/689473287
## Download TensorRT .deb installation package
https://developer.nvidia.cn/tensorrt/download
* Remember to modify the TensorRT .deb installation package path in line 16 of the dockerfile
## Build Docker image
```
cd docker
docker build -t simgpletrt:0.1 .
docker run --gpus all -it -v /home/qiuqiu/Desktop/simple-tensorrt/:/root simgpletrt:0.1
```

# How to compile
## Linux platform
```
sh build.sh
```
* The compiled libraries and header files and sample programs are saved in the "sdk_out" folder

# Interface description
* Open "doc/index.html" in the browser
  
# Examples
## Taking resnet50 as an example
### 1.Export resnet50 onnx model
```
cd sdk_out/examples/resnet50 && python3 export_onnx.py
```
### 2.Convert .onnx model >> tensorrt .engine，base on “trtexec”
static batch mode:
```
trtexec --onnx=resnet50.onnx --saveEngine=resnet50.engine --fp16
```
dynamic batch mode:
```
trtexec --onnx=resnet50_dynamic.onnx --minShapes=input:1x3x224x224 --optShapes=input:4x3x224x224 --maxShapes=input:8x3x224x224 --saveEngine=resnet50_dynamic.engine --fp16
```
### 3.run example program
```
cd sdk_out/examples 

# sync forward
./build/resnet50 resnet50/resnet50.engine resnet50/cat.jpeg
./build/resnet50 resnet50/resnet50_dynamic.engine resnet50/cat.jpeg 

# async forward
./build/resnet50_async resnet50/resnet50.engine resnet50/cat.jpeg resnet50/airplane.jpeg
./build/resnet50_async resnet50/resnet50_dynamic.engine resnet50/cat.jpeg resnet50/airplane.jpeg
```
## Taking yolov8n detection as an example
### 1.Export yolov8n onnx model  
https://docs.ultralytics.com/modes/export/#key-features-of-export-mode
### 2.Convert .onnx model >> tensorrt .engine，base on “trtexec”
```
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
```
## 3.run example program
```
cd sdk_out/examples
./build/yolov8_det yolov8_det/yolov8n.engine yolov8_det/test.jpg
```


### Example lists
resnet50: Example of image classification based on ResNet50  
yolov8_det: Ultralytics yolov8 detection  
