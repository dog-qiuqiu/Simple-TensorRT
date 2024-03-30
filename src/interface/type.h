#ifndef TYPE_H_
#define TYPE_H_

#include <memory>
#include <sstream>
#include <cstring>
#include <iostream>

namespace st {
    /**
     * @brief CUDA设备信息
     * 
     * @param device_id          设备ID
     * @param device_name        设备名称
     * @param total_memory       设备总内存  MB
     * @param free_memory        设备空闲内存 MB
     * @param used_memory        设备已用内存 MB
     * @param utilization_gpu    GPU利用率
     * @param utilization_memory 显存利用率
     * @param temperature        温度
     */
    struct CUDADeviceInfo
    {
        int device_id = -1;
        std::string device_name = "null";

        int total_memory = -1;
        int free_memory = -1;
        int used_memory = -1;

        int utilization_gpu = -1;
        int utilization_memory = -1;

        int temperature = -1;

        //打印设备信息
        void PrintDeviceInfo()
        {
            std::cout << "== Device Info: " << device_id << " ==" << std::endl;
            std::cout << "Device Name: " << device_name << std::endl;
            std::cout << "Total Memory: " << total_memory << "MB, Free Memory: " \
                      << free_memory << "MB, Used Memory: " << used_memory << "MB" << std::endl;
            std::cout << "GPU Utilization: " << utilization_gpu << "%" 
                      << ", Memory Utilization: " << utilization_memory << "%" \
                      << ", Temperature: " << temperature << "°C" << std::endl;
            std::cout << "====================" << std::endl;
        }

    };

    /**
     * @brief 日志级别
     */
    enum LogLevel {
        INFO = 0, /**< 日常信息 */
        DEBUG,    /**< 调试信息 */
        ERROR     /**< 错误信息 */
    };

    /**
     * @brief 数据排列
     */
    enum DataArrange
    {
        UNKNOWN = 0, /**< 未知 */
        NHWC,        /**< NHWC */
        NCHW         /**< NCHW */
    };

    /**
     * @brief 张量形状
     */
    struct Shape{
        int batch_size = 0;  /**< 批大小 */
        int channel = 0;     /**< 通道 */
        int height = 0;      /**< 高度 */
        int width = 0;       /**< 宽度 */

        /**
         * @brief 获取张量大小
         * @return 张量元素个数
         */
        int size() const {
            return batch_size * channel * height * width;
        }

        /**
         * @brief Operator ==
         * @param shape 张量形状
         * @return 是否相等
         */

        bool operator==(const Shape &shape) {
            return batch_size == shape.batch_size &&
                channel == shape.channel &&
                height == shape.height &&
                width == shape.width;
        }

        /**
         * @brief Operator !=
         * @param shape 张量形状
         * @return 是否不相等
         */
        bool operator!=(const Shape &shape) {
            return !(*this == shape);
        }
    };
    
    /**
     * @brief 张量数据 
     */
    class Tensor {
    private:
        bool copy_;
    public:
        Shape shape;
        float *data= nullptr;
        std::string data_str = "nullptr";

        /**
         * @brief 默认构造函数
         */
        Tensor() {}
        
        /**
         * @brief 构造函数
         * @param shape 张量形状
         * @param data 张量数据
         * @param copy 是否拷贝数据
         */
        Tensor(const Shape &shape, float *data, const bool &copy) {
            copy_ = copy;
            this->shape = shape;
            if (copy_) {
                this->data = new float[shape.size()];
                memcpy(this->data, data, this->shape.size() * sizeof(float));
            } else {
                this->data = data;
            }

            data_str = GetDataStr();
        }

        /**
         * @brief 构造函数
         * @param shape 张量形状
         */
        Tensor(const Shape &shape) {
            copy_ = true;
            this->shape = shape;
            this->data = new float[this->shape.size()];

            data_str = GetDataStr();
        }

        /**
         * @brief 拷贝构造函数
         * @param tensor 张量
         */
        Tensor(const Tensor &tensor) {
            copy_ = true;
            this->shape = tensor.shape;
            this->data = new float[this->shape.size()];
            memcpy(this->data, tensor.data, this->shape.size() * sizeof(float));

            data_str = GetDataStr();
        }

        ~Tensor() {
            if (copy_) {
                delete[] this->data;
            }
        }

        /**
         * @brief oeprator= 重载
         * @param tensor 张量
         */
        Tensor& operator=(const Tensor &tensor) {
            if (this == &tensor) {
                return *this;
            }

            copy_ = true;
            this->shape = tensor.shape;
            this->data = new float[this->shape.size()];
            memcpy(this->data, tensor.data, this->shape.size() * sizeof(float));

            data_str = GetDataStr();

            return *this;
        }

        /**
         * @brief 获取张量数据的指针的字符串表示
         * @return 字符串
         */
        std::string GetDataStr() {
            std::stringstream ss;
            ss << (void*)this->data;
            return ss.str();
        }
    };
} // namespace st

#endif // TYPE_H_