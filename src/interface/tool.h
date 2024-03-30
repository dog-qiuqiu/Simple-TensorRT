#ifndef TOOL_H_
#define TOOL_H_

#include <string>
#include <vector>
#include <iostream>

#include "type.h"

namespace st
{
    /**
     * @brief 获取CUDA设备数量及名称
     * 
     * @param device_info   获取到的设备信息
     * 
     * @return int 0:成功  -1:失败
     */
    int GetCudaDeviceCount(std::vector<st::CUDADeviceInfo> &device_info);

} // namespace st




#endif  // TOOL_H_  