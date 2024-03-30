#include "simple_tensorrt_impl.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <assert.h>

#include "cuda_runtime_api.h"

SimpleTensorrtImpl::SimpleTensorrtImpl(const st::LogLevel &log_level, const bool &async, 
                                       const int &device_id) {
    log_.open(log_level);

    async_mode_ = async;
    log_.Info(__func__, "Set async mode: " + std::to_string(async_mode_));
    
    if (cudaSetDevice(device_id) != cudaSuccess) {
        log_.Error(__func__, "Failed to set device " + std::to_string(device_id));
        throw std::runtime_error("Failed to set device " + std::to_string(device_id));
    }

    log_.Info(__func__, "Create SimpleTensorrtImpl instance, device_id: " + std::to_string(device_id));
}

SimpleTensorrtImpl::~SimpleTensorrtImpl() {
    log_.Info(__func__, "Destroy SimpleTensorrtImpl instance");
}

void SimpleTensorrtImpl::PrintTensorsInfo(const std::map<std::string, st::Tensor> &tensors) {
    for (auto& it : tensors) {
        log_.Debug(__func__, "Tensor name: " + it.first);
        log_.Debug(__func__, "Tensor shape: " + std::to_string(it.second.shape.batch_size) + " " + \
                  std::to_string(it.second.shape.channel) + " " + \
                  std::to_string(it.second.shape.height) + " " + \
                  std::to_string(it.second.shape.width));
        log_.Debug(__func__, "Tensor address: " + it.second.data_str);
    }
}

int SimpleTensorrtImpl::GpuBuffersToTensors(const std::map<std::string, void*> &gpu_buffers, const std::map<std::string, st::Shape> &shapes,
                                            std::map<std::string, st::Tensor> &tensors) {
    for (auto& it : gpu_buffers) {
        //拷贝GPU buffer到tensor
        st::Tensor tensor(shapes.at(it.first));
        if (cudaMemcpy(tensor.data, it.second, 
                       shapes.at(it.first).size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            log_.Error(__func__, "cudaMemcpy output tensor failed:" + it.first);
            return -1;
        }   

        //将输出Tensor保存到map中
        tensors.emplace(it.first, tensor);
    }

    return 0;
}

int SimpleTensorrtImpl::TensorsToGpuBuffers(const std::map<std::string, st::Tensor> &tensors, const std::map<std::string, st::Shape> &shapes,
                                            std::map<std::string, void*> &gpu_buffers) {
    for (auto& it : tensors) {
        //拷贝tensor到GPU buffer
        if (cudaMemcpy(gpu_buffers.at(it.first), it.second.data,
                       shapes.at(it.first).size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            log_.Error(__func__, "cudaMemcpy input tensor failed:" + it.first);
            return -1;
        }
    }

    return 0;
}

int SimpleTensorrtImpl::LoadEngine(const void *data, const size_t &size,
                                   const int &max_batch_size)
{
    //创建runtime
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
    if (runtime == nullptr) {
        log_.Error(__func__, "Create runtime failed");
        return -1;
    }

    //反序列化引擎
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(data, size));
    if (engine_ == nullptr) {
        log_.Error(__func__, "Deserialize engine failed");
        return -1;
    }

    //获取输入输出Tensor的名称和形状
    const int num_io_tensor = engine_->getNbIOTensors();
    for (int i = 0; i < num_io_tensor; i++) {
        //获取Tensor的名称
        const std::string tensor_name = engine_->getIOTensorName(i);
        //记录IO Tensor的顺序，forward时需要用到
        io_tensor_names_.push_back(tensor_name);
        //获取Tensor的类型 kINPUT or kOUTPUT
        auto io_mode = engine_->getTensorIOMode(tensor_name.c_str());
        if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
            st::Shape shape;
            void *gpu_buffer;
            //获取Tensor的形状
            const nvinfer1::Dims dims = engine_->getTensorShape(tensor_name.c_str());
            shape.batch_size = dims.d[0];
            //如果是动态batch，则设置为外部传入的最大batch_size
            if (shape.batch_size == -1) {
                shape.batch_size = max_batch_size;
                log_.Info(__func__, "Engine input tensor is dynamic batch, set max batch size: " + std::to_string(max_batch_size));
            }
            shape.channel = dims.d[1];
            shape.height = dims.d[2];
            shape.width = dims.d[3];
            log_.Info(__func__, "Engine input tensor name: " + tensor_name);
            log_.Info(__func__, "Engine input shape: " + std::to_string(shape.batch_size) + " " + 
                      std::to_string(shape.channel) + " " + std::to_string(shape.height) + " " + 
                      std::to_string(shape.width));
            
            //开辟输入Tensor的内存
            if (cudaMalloc(&gpu_buffer, shape.size() * sizeof(float)) != cudaSuccess) {
                log_.Error(__func__, "Malloc input buffer failed");
                return -1;
            }
            inputs_shape_.emplace(tensor_name, shape);
            inputs_gpu_buffer_.emplace(tensor_name, gpu_buffer);            
        } else {
            st::Shape shape;
            void *gpu_buffer;
            //获取Tensor的形状
            const nvinfer1::Dims dims = engine_->getTensorShape(tensor_name.c_str());
            shape.batch_size = dims.d[0];
            //如果是动态batch，则设置为外部传入的最大batch_size
            if (shape.batch_size == -1) {
                shape.batch_size = max_batch_size;
            }
            shape.channel = dims.d[1] != 0 ? dims.d[1] : 1;
            shape.height = dims.d[2] != 0 ? dims.d[2] : 1;
            shape.width = dims.d[3] != 0 ? dims.d[3] : 1;
            log_.Info(__func__, "Engine output tensor name: " + tensor_name);
            log_.Info(__func__, "Engine output shape: " + std::to_string(shape.batch_size) + " " + 
                      std::to_string(shape.channel) + " " + std::to_string(shape.height) + " " + 
                      std::to_string(shape.width));
            
            //开辟输入Tensor的内存
            if (cudaMalloc(&gpu_buffer, shape.size() * sizeof(float)) != cudaSuccess) {
                log_.Error(__func__, "Malloc output buffer failed");
                return -1;
            }
            outputs_shape_.emplace(tensor_name, shape);
            outputs_gpu_buffer_.emplace(tensor_name, gpu_buffer);
        }
    }

    //要在多个流中同时执行推理，请为每个流使用一个执行上下文
    if (!async_mode_) {
        //创建执行上下文
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (context_ == nullptr) {
            log_.Error(__func__, "Create execution context failed");
            return -1;
        }
    }

    return 0;
}

int SimpleTensorrtImpl::Init(const std::string &engine_path, const int &max_batch_size) {

    int ret = -1;
    //计时开始
    auto start = std::chrono::high_resolution_clock::now();

    //加载TensorRT引擎
    log_.Info(__func__, "Init SimpleTensorrtImpl, engine path: " + engine_path);

    //读取引擎文件
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open()) {
        log_.Error(__func__, "Open engine file failed");
        return -1;
    }
    file.seekg(0, file.end);
    const int length = file.tellg();
    file.seekg(0, file.beg);
    char* data = new char[length];
    file.read(data, length);

    //加载引擎
    ret = LoadEngine(data, length, max_batch_size);
    if (ret != 0) {
        log_.Error(__func__, "Load engine failed");
        return -1;
    }

    //释放引擎文件内存
    file.close();
    delete[] data;

    //计时结束
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    log_.Info(__func__, "Init finished, time: " + std::to_string(duration.count()) + " ms");
    
    return 0;
}

int SimpleTensorrtImpl::Init(const void *engine_data, const size_t &engine_size,
                             const int &max_batch_size) {

    int ret = -1;
    //计时开始
    auto start = std::chrono::high_resolution_clock::now();

    //加载TensorRT引擎
    log_.Info(__func__, "Init SimpleTensorrtImpl, engine size: " + std::to_string(engine_size));

    //加载引擎
    ret = LoadEngine(engine_data, engine_size, max_batch_size);
    if (ret != 0) {
        log_.Error(__func__, "Load engine failed");
        return -1;
    }

    //计时结束
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    log_.Info(__func__, "Init finished, time: " + std::to_string(duration.count()) + " ms");

    return 0;
}

int SimpleTensorrtImpl::GetIOShape(std::map<std::string, st::Shape> &inputs_shape, 
                                   std::map<std::string, st::Shape> &outputs_shape) {
    
    inputs_shape.clear();
    for (auto& it : inputs_shape_) {
        inputs_shape.emplace(it.first, it.second);
    }

    outputs_shape.clear();
    for (auto& it : outputs_shape_) {
        outputs_shape.emplace(it.first, it.second);
    }

    return 0;
}

int SimpleTensorrtImpl::Forward(const std::map<std::string, st::Tensor> &inputs_tensor, 
                                std::map<std::string, st::Tensor> &outputs_tensor) {
    
    int ret = -1;

    //计时开始
    auto start = std::chrono::high_resolution_clock::now();

    //要在多个流中同时执行推理，请为每个流使用一个执行上下文
    if (async_mode_) {
        //创建执行上下文
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (context_ == nullptr) {
            log_.Error(__func__, "Create execution context failed");
            return -1;
        }
    }

    //打印输入tensor的信息
    log_.Debug(__func__, "********** Input tensor **********");
    PrintTensorsInfo(inputs_tensor);
    log_.Debug(__func__, "***********************************");

    //校验输入Tensor的名称和形状是否和枚举Engine时一致
    for (auto& it : inputs_tensor) {
        if (inputs_shape_.find(it.first) == inputs_shape_.end()) {
            log_.Error(__func__, "Input tensor name: " + it.first + " not found in engine");
            return -1;
        }
        if (inputs_shape_[it.first] != it.second.shape) {
            log_.Error(__func__, "Input tensor name: " + it.first + " shape not match");
            return -1;
        }
    }

    //将输入Tensor拷贝到GPU内存
    ret = TensorsToGpuBuffers(inputs_tensor, inputs_shape_, inputs_gpu_buffer_);
    if (ret != 0) {
        log_.Error(__func__, "Copy input tensor to gpu buffer failed");
        return -1;
    }

    //构造输入输出Tensor的指针数组
    void* io_buffers[io_tensor_names_.size()];
    for (int i = 0; i < io_tensor_names_.size(); i++) {
        if (inputs_gpu_buffer_.find(io_tensor_names_[i]) != inputs_gpu_buffer_.end()) {
            io_buffers[i] = inputs_gpu_buffer_[io_tensor_names_[i]];
        } else {
            io_buffers[i] = outputs_gpu_buffer_[io_tensor_names_[i]];
        }
    }

    //设置输入张量的shape
    for (auto& it : inputs_shape_) {
        nvinfer1::Dims dims;
        dims.nbDims = 4;
        dims.d[0] = it.second.batch_size;
        dims.d[1] = it.second.channel;
        dims.d[2] = it.second.height;
        dims.d[3] = it.second.width;
        context_->setInputShape(it.first.c_str(), dims);
    }

    //调用TensorRT的推理接口
    log_.Debug(__func__, "Run Forwarding...");
    //执行推理
    bool status = context_->executeV2(io_buffers);
    if (!status) {
        log_.Error(__func__, "Execute inference failed");
        return -1;
    }

    //将推理结果拷贝到输出Tensor
    ret = GpuBuffersToTensors(outputs_gpu_buffer_, outputs_shape_, outputs_tensor);
    if (ret != 0) {
        log_.Error(__func__, "Copy gpu output buffer to tensor failed");
        return -1;
    }

    //打印输出tensor的信息
    log_.Debug(__func__, "********** Output tensor **********");
    PrintTensorsInfo(outputs_tensor);
    log_.Debug(__func__, "***********************************");

    //计时结束
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    log_.Debug(__func__, "Forwarding finished, time: " + std::to_string(duration.count()) + " ms");

    return 0;
}

int SimpleTensorrtImpl::ForwardAsync(const std::map<std::string, st::Tensor> &inputs_tensor,
                                     const int &stream_id) {
    
    int ret = -1;

    //计时开始
    auto start = std::chrono::high_resolution_clock::now();

    //要在多个流中同时执行推理，请为每个流使用一个执行上下文
    if (async_mode_) {
        //创建执行上下文
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (context_ == nullptr) {
            log_.Error(__func__, "Create execution context failed");
            return -1;
        }
    }

    //打印输入tensor的信息
    log_.Debug(__func__, "********** Input tensor **********");
    PrintTensorsInfo(inputs_tensor);
    log_.Debug(__func__, "***********************************");

    //校验输入Tensor的名称和形状是否和枚举Engine时一致
    for (auto& it : inputs_tensor) {
        if (inputs_shape_.find(it.first) == inputs_shape_.end()) {
            log_.Error(__func__, "Input tensor name: " + it.first + " not found in engine");
            return -1;
        }
        if (inputs_shape_[it.first] != it.second.shape) {
            log_.Error(__func__, "Input tensor name: " + it.first + " shape not match");
            return -1;
        }
    }

    //将输入Tensor拷贝到GPU内存
    std::map<std::string, void*> inputs_gpu_buffer;
    for (auto& it : inputs_tensor) {
        void* gpu_buffer = nullptr;
        if (cudaMalloc(&gpu_buffer, it.second.shape.size() * sizeof(float)) != cudaSuccess) {
            log_.Error(__func__, "Malloc input buffer failed");
            return -1;
        }

        if (cudaMemcpy(gpu_buffer, it.second.data, it.second.shape.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            log_.Error(__func__, "Memcpy input buffer failed");
            return -1;
        }

        inputs_gpu_buffer.emplace(it.first, gpu_buffer);
    }
    inputs_gpu_buffers_async_.emplace(stream_id, inputs_gpu_buffer);
    
    //构造输入输出Tensor的GPU buffer
    std::map<std::string, void*> outputs_gpu_buffer;
    for (auto& it : outputs_shape_) {
        void* gpu_buffer = nullptr;
        if (cudaMalloc(&gpu_buffer, it.second.size() * sizeof(float)) != cudaSuccess) {
            log_.Error(__func__, "Malloc output buffer failed");
            return -1;
        }
        outputs_gpu_buffer.emplace(it.first, gpu_buffer);
    }
    outputs_gpu_buffers_async_.emplace(stream_id, outputs_gpu_buffer);

    //创建CUDA流
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        log_.Error(__func__, "Create cuda stream failed");
        return -1;
    }
    streams_.emplace(stream_id, stream);

    //构造输入输出Tensor的指针数组
    void* io_buffers[io_tensor_names_.size()];
    for (int i = 0; i < io_tensor_names_.size(); i++) {
        if (inputs_gpu_buffer.find(io_tensor_names_[i]) != inputs_gpu_buffer.end()) {
            io_buffers[i] = inputs_gpu_buffer[io_tensor_names_[i]];
        } else {
            io_buffers[i] = outputs_gpu_buffer[io_tensor_names_[i]];
        }
    }

    //设置输入张量的shape
    for (auto& it : inputs_shape_) {
        nvinfer1::Dims dims;
        dims.nbDims = 4;
        dims.d[0] = it.second.batch_size;
        dims.d[1] = it.second.channel;
        dims.d[2] = it.second.height;
        dims.d[3] = it.second.width;
        context_->setInputShape(it.first.c_str(), dims);
    }

    //调用TensorRT的推理接口
    log_.Debug(__func__, "Run Forwarding...");
    //执行推理
    bool status = context_->enqueueV2(io_buffers, stream, nullptr);
    if (!status) {
        log_.Error(__func__, "Execute inference failed");
        return -1;
    }

    //计时结束
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    log_.Debug(__func__, "Async forwarding finished, time: " + std::to_string(duration.count()) + " ms");

    return 0;
}

int SimpleTensorrtImpl::GetForwardAsyncOutput(std::map<std::string, st::Tensor> &outputs_tensor, 
                                              const int &stream_id) {
    
    int ret = -1;

    //计时开始
    auto start = std::chrono::high_resolution_clock::now();

    //同步等待CUDA流
    if (cudaStreamSynchronize(streams_[stream_id]) != cudaSuccess) {
        log_.Error(__func__, "Synchronize cuda stream failed");
        return -1;
    }
    

    //获取输出Tensor的GPU buffer
    std::map<std::string, void*> outputs_gpu_buffer = outputs_gpu_buffers_async_.at(stream_id);

    //将输出Tensor拷贝到CPU内存
    for (auto& it : outputs_gpu_buffer) {
        st::Tensor tensor(outputs_shape_[it.first]);
        if (cudaMemcpy(tensor.data, it.second, outputs_shape_[it.first].size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            log_.Error(__func__, "Memcpy output buffer failed");
            return -1;
        }
        outputs_tensor.emplace(it.first, tensor);

        //释放输出Tensor的GPU buffer
        if (cudaFree(it.second) != cudaSuccess) {
            log_.Error(__func__, "Free output buffer failed");
            return -1;
        }
    }
    outputs_gpu_buffers_async_.erase(stream_id);

    //释放输入Tensor的GPU buffer
    std::map<std::string, void*> inputs_gpu_buffer = inputs_gpu_buffers_async_.at(stream_id);
    for (auto& it : inputs_gpu_buffer) {
        if (cudaFree(it.second) != cudaSuccess) {
            log_.Error(__func__, "Free input buffer failed");
            return -1;
        }
    }
    inputs_gpu_buffers_async_.erase(stream_id);

    //删除创建的CUDA流
    if (cudaStreamDestroy(streams_[stream_id]) != cudaSuccess) {
        log_.Error(__func__, "Destroy cuda stream failed");
        return -1;
    }
    streams_.erase(stream_id);

    //计时结束
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    log_.Debug(__func__, "Get output finished, time: " + std::to_string(duration.count()) + " ms");
    
    return 0;
}

int SimpleTensorrtImpl::Destroy() {
    log_.Info(__func__, "Destroying...");
    //释放输入Tensor的GPU内存
    log_.Info(__func__, "Free input tensor's GPU memory");
    for (auto& it : inputs_gpu_buffer_) {
        if (cudaFree(it.second) != cudaSuccess) {
            log_.Error(__func__, "Free input buffer failed");
            return -1;
        }
    }
    inputs_gpu_buffer_.clear();

    log_.Info(__func__, "Free output tensor's GPU memory");
    //释放输出Tensor的GPU内存
    for (auto& it : outputs_gpu_buffer_) {
        if (cudaFree(it.second) != cudaSuccess) {
            log_.Error(__func__, "Free output buffer failed");
            return -1;
        }
    }
    outputs_gpu_buffer_.clear();

    //删除异步推理的输入的GPU buffer
    log_.Info(__func__, "Free async input tensor's GPU memory");
    for (auto& it : inputs_gpu_buffers_async_) {
        for (auto& it2 : it.second) {
            if (cudaFree(it2.second) != cudaSuccess) {
                log_.Error(__func__, "Free input buffer failed");
                return -1;
            }
        }
    }
    inputs_gpu_buffers_async_.clear();

    //删除异步推理的输出的GPU buffer
    log_.Info(__func__, "Free async output tensor's GPU memory");
    for (auto& it : outputs_gpu_buffers_async_) {
        for (auto& it2 : it.second) {
            if (cudaFree(it2.second) != cudaSuccess) {
                log_.Error(__func__, "Free output buffer failed");
                return -1;
            }
        }
    }
    outputs_gpu_buffers_async_.clear();

    //删除创建的CUDA流
    log_.Info(__func__, "Destroy cuda stream");
    for (auto& it : streams_) {
        if (cudaStreamDestroy(it.second) != cudaSuccess) {
            log_.Error(__func__, "Destroy cuda stream failed");
            return -1;
        }
    }
    streams_.clear();

    log_.Info(__func__, "Destroy finished");

    return 0;
}