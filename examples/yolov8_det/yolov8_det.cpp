#include "tool.h"
#include "simple_tensorrt.h"

#include <map>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

static const int kCLASS_NUM = 80;
static const int kINPUT_SIZE = 640;

typedef struct {
    cv::Rect rect;
    std::pair<int, float> cls;

    int x1() const { return rect.x; }
    int y1() const { return rect.y; }
    int x2() const { return rect.x + rect.width; }
    int y2() const { return rect.y + rect.height; }

    float area() const {
        return rect.width * rect.height;
    }
} BBox;

float IntersectionArea(const BBox &a, const BBox &b) {
    if (a.x1() > b.x2() || a.x2() < b.x1() || a.y1() > b.y2() || a.y2() < b.y1()) {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2(), b.x2()) - std::max(a.x1(), b.x1());
    float inter_height = std::min(a.y2(), b.y2()) - std::max(a.y1(), b.y1());

    return inter_width * inter_height;
}

bool ScoreSort(const BBox &a, const BBox &b) { 
    return (a.cls.second > b.cls.second); 
}

int NmsHandle(std::vector<BBox> &src_boxes, std::vector<BBox> &dst_boxes)
{
    std::vector<int> picked;
    sort(src_boxes.begin(), src_boxes.end(), ScoreSort);
    for (int i = 0; i < src_boxes.size(); i++) {
        int keep = 1;
        for (int j = 0; j < picked.size(); j++) {
            //交集
            float inter_area = IntersectionArea(src_boxes[i], src_boxes[picked[j]]);
            //并集
            float union_area = src_boxes[i].area() + src_boxes[picked[j]].area() - inter_area;
            float IoU = inter_area / union_area;
            // std::cout << "IoU: " << IoU << std::endl;

            if(IoU > 0.25 && src_boxes[i].cls.first == src_boxes[picked[j]].cls.first) {
                keep = 0;
                break;
            }
        }

        if (keep) {
            picked.push_back(i);
        }
    }

    for (int i = 0; i < picked.size(); i++) {
        dst_boxes.push_back(src_boxes[picked[i]]);
    }

    return 0;
}


int LetterBoxImage(const cv::Mat &src, cv::Mat &dst, const int &target_width, const int &target_height, 
                   int &x_offset, int &y_offset, float &scale) {

    int src_width = src.cols;
    int src_height = src.rows;
    scale = std::min(static_cast<float>(target_width) / src_width, static_cast<float>(target_height) / src_height);
    int width = static_cast<int>(src_width * scale);
    int height = static_cast<int>(src_height * scale);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(width, height));
    dst = cv::Mat::zeros(target_height, target_width, CV_8UC3);
    x_offset = (target_width - width) / 2;
    y_offset = (target_height - height) / 2;
    resized.copyTo(dst(cv::Rect(x_offset, y_offset, width, height)));

    return 0;
}

//输入预处理
int InputProcess(const cv::Mat &input_img, const std::map<std::string, st::Shape> &inputs_shape,
                 std::map<std::string, st::Tensor> &inputs_tensor, int &offset_x, int &offset_y, float &scale) {
    cv::Mat image;
    //resize image to 640x640
    LetterBoxImage(input_img, image, kINPUT_SIZE, kINPUT_SIZE, offset_x, offset_y, scale);
    //to rgb
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    //convert to float32
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);

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
                        tensor.data[index] = image.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
        }
        inputs_tensor.emplace(name, tensor);
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
    float scale;
    int offset_x, offset_y;
    std::map<std::string, st::Tensor> inputs_tensor;
    ret = InputProcess(image, inputs_shape, inputs_tensor, offset_x, offset_y, scale);
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

        //后处理
        std::vector<std::vector<BBox>> bach_boxes;
        for (auto& output_tensor : outputs_tensor) {
            const std::string name = output_tensor.first;
            st::Tensor tensor = output_tensor.second;
            std::cout << "Output tensor name: " << name << std::endl;
            std::cout << "Output tensor shape: " << tensor.shape.batch_size << " " << tensor.shape.channel << " "
                      << tensor.shape.height << " " << tensor.shape.width << std::endl;
            //yolov8检测框解析
            for (int b = 0; b < tensor.shape.batch_size; b++) {
                //chw => hcw: 84x8400x1 => 8400x84x1
                std::vector<float> data;
                data.resize(tensor.shape.size());
                for (int c = 0; c < tensor.shape.channel; c ++) {
                    for (int h = 0; h < tensor.shape.height; h++) {
                        for (int w = 0; w < tensor.shape.width; w++) {
                            int src_index = b * tensor.shape.channel * tensor.shape.height * tensor.shape.width +
                                            c * tensor.shape.height * tensor.shape.width +
                                            h * tensor.shape.width + w;
                            int dst_index = b * tensor.shape.channel * tensor.shape.height * tensor.shape.width +
                                            h * tensor.shape.channel * tensor.shape.width +
                                            c * tensor.shape.width + w;
                            data[dst_index] = tensor.data[src_index];
                        }
                    }
                }

                //解析检测框
                std::vector<BBox> boxes;
                for (int h = 0; h < tensor.shape.height; h++) {
                    //每个检测框的数据格式为：[x, y, w, h, cls1, cls2, cls3, cls4]
                    const float *ptr = data.data() + h * tensor.shape.channel;
                    //解析检测框
                    float bx = (ptr[0] - offset_x) / scale;
                    float by = (ptr[1] - offset_y) / scale;
                    float bw = ptr[2] / scale;
                    float bh = ptr[3] / scale;

                    int x1 = static_cast<int>(bx - bw * 0.5);
                    int y1 = static_cast<int>(by - bh * 0.5);
                    int x2 = static_cast<int>(bx + bw * 0.5);
                    int y2 = static_cast<int>(by + bh * 0.5);

                    //解析类别概率
                    ptr += 4;
                    int class_id = -1;
                    float score = 0.0;
                    for (int k = 0; k < kCLASS_NUM; k++) {
                        if (ptr[k] > score) {
                            class_id = k;
                            score = ptr[k];
                        }
                    }

                    //过滤掉置信度小于0.25的检测框
                    if (score > 0.25) {
                        BBox bbox;
                        bbox.rect = cv::Rect(x1, y1, x2 - x1, y2 - y1);
                        bbox.cls = std::make_pair(class_id, score);
                        boxes.push_back(bbox);
                    }
                }

                //NMS
                std::vector<BBox> nms_boxes;
                NmsHandle(boxes, nms_boxes);
                bach_boxes.push_back(nms_boxes);
            }
        }

        //分batch打印检测结果
        for (int b = 0; b < bach_boxes.size(); b++) {
            std::cout << "Batch " << b << " detection result: " << std::endl;
            for (auto& box : bach_boxes[b]) {
                std::cout << "cls: " << box.cls.first << " score: " << box.cls.second << " rect: " << box.rect << std::endl;
            }
        }

        //绘制检测框 batch_index=0
        for (auto& box : bach_boxes[0]) {
            cv::rectangle(image, box.rect, cv::Scalar(0, 255, 0), 2);
            std::string text = "cls: " + std::to_string(box.cls.first) + " score: " + std::to_string(box.cls.second);
            cv::putText(image, text, cv::Point(box.rect.x, box.rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        //保存结果图片
        cv::imwrite("result.jpg", image);
    }

    //销毁TensorRT资源
    ret = tensorrt.Destroy();
    if (ret != 0) {
        std::cout << "\033[31m" << "TensorRT destroy failed" << "\033[0m" << std::endl;
        return -1;
    }

    return 0;
}