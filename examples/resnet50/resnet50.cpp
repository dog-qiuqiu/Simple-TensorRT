#include "tool.h"
#include "simple_tensorrt.h"

#include <map>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

//softmax 处理 
std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> result;
    float sum = 0.0;

    // 计算指数和
    for (float val : input) {
        sum += std::exp(val);
    }

    // 计算 softmax 值
    for (float val : input) {
        float softmax_val = std::exp(val) / sum;
        result.push_back(softmax_val);
    }

    return result;
}

//输入预处理
int InputProcess(const cv::Mat &input_img, const std::map<std::string, st::Shape> &inputs_shape,
                 std::map<std::string, st::Tensor> &inputs_tensor) {
    cv::Mat image;
    //resize image to 224x224
    cv::resize(input_img, image, cv::Size(224, 224));
    //convert to float32
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    //mean and std
    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};

    //构建tensor
    for (auto& input_shape : inputs_shape) {
        const std::string name = input_shape.first;
        st::Tensor tensor(input_shape.second);
        //将cv::Mat image数据填充到tensor中
        for (int b = 0; b < input_shape.second.batch_size; b++) {
            for (int c = 0; c < input_shape.second.channel; c++) {
                for (int h = 0; h < input_shape.second.height; h++) {
                    for (int w = 0; w < input_shape.second.width; w++) {
                        int index = b * input_shape.second.channel * input_shape.second.height * input_shape.second.width +
                                    c * input_shape.second.height * input_shape.second.width +
                                    h * input_shape.second.width + w;
                        //HWC to CHW and normalize
                        tensor.data[index] = (image.at<cv::Vec3f>(h, w)[c] - mean[c]) / std[c];
                    }
                }
            }
        }
        inputs_tensor.emplace(name, tensor);
    }

    return 0;
}

//输出后处理
int OutputProcess(std::map<std::string, st::Tensor> &outputs_tensor, 
                  std::vector<std::pair<int, float>> &result) {
    std::vector<std::vector<float>> outputs_data;
    for (auto &output_tensor : outputs_tensor) {
        st::Tensor tensor = output_tensor.second;
        for (int b = 0; b < tensor.shape.batch_size; b++) {
            std::vector<float> output_data;
            for (int c = 0; c < tensor.shape.channel; c++) {
                int index = b * tensor.shape.channel + c;
                output_data.push_back(tensor.data[index]);
            }
            outputs_data.push_back(output_data);
        }
    }
    
    //计算softmax
    for (auto& output_data : outputs_data) {
        std::vector<float> softmax_data = softmax(output_data);
        //打印softmax结果，并找到最大值的索引
        int max_index = 0;
        float max_value = 0.0;
        for (int i = 0; i < softmax_data.size(); i++) {
            if (softmax_data[i] > max_value) {
                max_value = softmax_data[i];
                max_index = i;
            }
        }
        result.push_back(std::make_pair(max_index, max_value));
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "\033[31m" <<"Please specify relevant parameters for execution, as shown below:" << "\033[0m" << std::endl;
        std::cout << "\t" << "Usage: " << argv[0] << " <engine_path>" << " <image_path> " << std::endl;
        return -1;
    }

    int ret = -1;

    //获取CUDA设备信息
    std::vector<st::CUDADeviceInfo> device_info;
    ret = st::GetCudaDeviceCount(device_info);
    if (ret != 0) {
        std::cout << "\033[31m" << "Get CUDA device count failed" << "\033[0m" << std::endl;
        return -1;
    }
    for (auto &info : device_info) {
        info.PrintDeviceInfo();
    }
    
    //Initialize TensorRT
    SimpleTensorrt tensorrt(st::LogLevel::DEBUG);

    //当加载的engine为动态 dynamic batch时，设置max_batch_size参数
    //当加载的engine为固定 static batch时，设置的max_batch_size参数无效
    const int max_batch_size = 8;
    ret = tensorrt.Init(argv[1], max_batch_size);
    if (ret != 0) {
        std::cout << "\033[31m" << "TensorRT initialization failed" << "\033[0m" << std::endl;
        return -1;
    }

    //获取输入输出tensor的shape
    std::map<std::string, st::Shape> inputs_shape;
    std::map<std::string, st::Shape> outputs_shape;
    tensorrt.GetIOShape(inputs_shape, outputs_shape);
    //只支持单输入单输出的模型
    assert(inputs_shape.size() == 1 && outputs_shape.size() == 1);

    //读取图片
    cv::Mat image = cv::imread(argv[2]);
    if (image.empty()) {
        std::cout << "\033[31m" << "Read image failed" << "\033[0m" << std::endl;
        return -1;
    }

    //输入预处理
    std::map<std::string, st::Tensor> inputs_tensor;
    ret = InputProcess(image, inputs_shape, inputs_tensor);
    if (ret != 0) {
        std::cout << "\033[31m" << "Input process failed" << "\033[0m" << std::endl;
        return -1;
    }

    //推理10次,其中第一次推理会比较耗时
    for (int i = 0; i < 10; i++) {
        std::cout << "====================Inference " << i << "====================" << std::endl;

        //推理
        std::map<std::string, st::Tensor> outputs_tensor;
        ret = tensorrt.Forward(inputs_tensor, outputs_tensor);
        if (ret != 0) {
            std::cout << "\033[31m" << "TensorRT forward failed" << "\033[0m" << std::endl;
            return -1;
        }

        //输出后处理
        std::vector<std::pair<int, float>> result;
        ret = OutputProcess(outputs_tensor, result);
        if (ret != 0) {
            std::cout << "\033[31m" << "Output process failed" << "\033[0m" << std::endl;
            return -1;
        }

        //打印结果
        for (int i = 0; i < result.size(); i++) {
            std::cout << "image: " << "batch: " << i << " class: " << result[i].first << " score: " << result[i].second << std::endl;
        }
    }

    //销毁TensorRT资源
    ret = tensorrt.Destroy();
    if (ret != 0) {
        std::cout << "\033[31m" << "TensorRT destroy failed" << "\033[0m" << std::endl;
        return -1;
    }

    return 0;
}