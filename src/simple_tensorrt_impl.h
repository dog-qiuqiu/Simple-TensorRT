#ifndef SIMPLE_TENSORRT_IMPL_H_
#define SIMPLE_TENSORRT_IMPL_H_

#include <map>
#include <vector>
#include <string>

#include "type.h"
#include "utils_log.hpp"

#include "NvInfer.h"

class SimpleTensorrtImpl {
private:
    UtilsLog log_;
    bool async_mode_;

    std::vector<std::string> io_tensor_names_;

    std::map<std::string, st::Shape> inputs_shape_;
    std::map<std::string, st::Shape> outputs_shape_;

    std::map<std::string, void*> inputs_gpu_buffer_;
    std::map<std::string, void*> outputs_gpu_buffer_;

    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::map<int, cudaStream_t> streams_;
    std::map<int, std::map<std::string, void*>> inputs_gpu_buffers_async_;
    std::map<int, std::map<std::string, void*>> outputs_gpu_buffers_async_;

    //TesnorRT logger
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char *msg) noexcept override {
            // suppress info-level messages
            if (severity <= Severity::kWARNING)
                std::cout << msg << std::endl;
        }
    } logger_;
    
    void PrintTensorsInfo(const std::map<std::string, st::Tensor> &tensors);
    int GpuBuffersToTensors(const std::map<std::string, void*> &gpu_buffers, const std::map<std::string, st::Shape> &shapes,
                            std::map<std::string, st::Tensor> &tensors);
    int TensorsToGpuBuffers(const std::map<std::string, st::Tensor> &tensors, const std::map<std::string, st::Shape> &shapes,
                            std::map<std::string, void*> &gpu_buffers);
    int LoadEngine(const void *data, const size_t &size, 
                   const int &max_batch_size);
public:
    SimpleTensorrtImpl(const st::LogLevel &log_level, const bool &async, 
                       const int &device_id);
    ~SimpleTensorrtImpl();

    int Init(const std::string &engine_path, const int &max_batch_size);
    int Init(const void *engine_data, const size_t &engine_size, 
             const int &max_batch_size);
    
    int GetIOShape(std::map<std::string, st::Shape> &inputs_shape, 
                   std::map<std::string, st::Shape> &outputs_shape);
    int Forward(const std::map<std::string, st::Tensor> &inputs_tensor, 
                std::map<std::string, st::Tensor> &outputs_tensor);
    int ForwardAsync(const std::map<std::string, st::Tensor> &inputs_tensor,
                     const int &stream_id);
    int GetForwardAsyncOutput(std::map<std::string, st::Tensor> &outputs_tensor, 
                              const int &stream_id);

    int Destroy();
};

#endif // SIMPLE_TENSORRT_H_
