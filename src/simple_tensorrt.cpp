#include "simple_tensorrt.h"
#include "simple_tensorrt_impl.h"

SimpleTensorrt::SimpleTensorrt(const st::LogLevel &log_level, const bool &async, const int &device_id) {
    impl_ = std::make_unique<SimpleTensorrtImpl>(log_level, async, device_id);
}

SimpleTensorrt::~SimpleTensorrt() {
}

int SimpleTensorrt::Init(const std::string &engine_path, const int &max_batch_size) {
    return impl_->Init(engine_path, max_batch_size);
}

int SimpleTensorrt::Init(const void *engine_data, const size_t &engine_size, 
                         const int &max_batch_size) {
    return impl_->Init(engine_data, engine_size, max_batch_size);
}

int SimpleTensorrt::GetIOShape(std::map<std::string, st::Shape> &inputs_shape, 
                               std::map<std::string, st::Shape> &outputs_shape) {
    return impl_->GetIOShape(inputs_shape, outputs_shape);
}

int SimpleTensorrt::Forward(const std::map<std::string, st::Tensor> &inputs_tensor, 
                            std::map<std::string, st::Tensor> &outputs_tensor) {
    return impl_->Forward(inputs_tensor, outputs_tensor);
}

int SimpleTensorrt::ForwardAsync(const std::map<std::string, st::Tensor> &inputs_tensor, const int &stream_id) {
    return impl_->ForwardAsync(inputs_tensor, stream_id);
}

int SimpleTensorrt::GetForwardAsyncOutput(std::map<std::string, st::Tensor> &outputs_tensor, const int &stream_id) {
    return impl_->GetForwardAsyncOutput(outputs_tensor, stream_id);
}

int SimpleTensorrt::Destroy() {
    return impl_->Destroy();
}