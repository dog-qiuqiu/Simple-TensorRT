#ifndef SIMPLE_TENSORRT_H_
#define SIMPLE_TENSORRT_H_

#include <map>
#include <string>
#include <memory>

#include "type.h"

class SimpleTensorrtImpl;
class SimpleTensorrt {
private:
    std::unique_ptr<SimpleTensorrtImpl> impl_;
public:

    /**
     * @brief Construct a new Simple Tensorrt object            
     * 
     * @param log_level  日志级别
     * @param async      是否开启异步推理（开启异步推理后会在内存分配拷贝上增加额外的开销）
     * @param device_id  设备ID
     */
    SimpleTensorrt(const st::LogLevel &log_level, const bool &async = false,
                   const int &device_id = 0);
    ~SimpleTensorrt();

    /**
     * @brief 初始化TensorRT的引擎
     * 
     * @param engine_path    engine文件的路径
     * @param max_batch_size 推理的最大batch size (当TensorRT引擎支持设置的动态batch size时，该参数才有效)
     * @return int 0:成功  -1:失败
     */
    int Init(const std::string &engine_path, const int &max_batch_size);

    /**
     * @brief 初始化TensorRT的引擎 （内存加载）
     * 
     * @param engine_data    engine数据在内存中的地址
     * @param engine_size    engine数据的大小
     * @param max_batch_size 推理的最大batch size (当TensorRT引擎支持设置的动态batch size时，该参数才有效)
     * @return int 0:成功  -1:失败
     */
    int Init(const void *engine_data, const size_t &engine_size, 
             const int &max_batch_size);

    /**
     * @brief 获取枚举到的输入输出tensor的shape
     * 
     * @param inputs_shape   输入tensor的shape
     * @param outputs_shape  输出tensor的shape
     * @return int 0:成功  -1:失败
     */
    int GetIOShape(std::map<std::string, st::Shape> &inputs_shape, 
                   std::map<std::string, st::Shape> &outputs_shape);

    /**
     * @brief 执行推理 同步接口
     * 
     * @param inputs_tensor   输入tensor
     * @param outputs_tensor  输出tensor
     * @return int 0:成功  -1:失败
     */
    int Forward(const std::map<std::string, st::Tensor> &inputs_tensor, 
                std::map<std::string, st::Tensor> &outputs_tensor);
    
    /**
     * @brief 执行推理 异步接口
     * 
     * @param inputs_tensor   输入tensor
     * @param stream_id       流ID
     * @return int 0:成功  -1:失败
     */
    int ForwardAsync(const std::map<std::string, st::Tensor> &inputs_tensor, const int &stream_id);

    /**
     * @brief 获取异步推理的结果
     * 
     * @param outputs_tensor  输出tensor
     * @param stream_id       流ID
     * @return int 0:成功  -1:失败
     */
    int GetForwardAsyncOutput(std::map<std::string, st::Tensor> &outputs_tensor, const int &stream_id);

    /**
     * @brief 释放资源
     * 
     * @return int  0:成功  -1:失败
     */
    int Destroy();
};

#endif // SIMPLE_TENSORRT_H_