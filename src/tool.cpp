#include "tool.h"

#include <nvml.h>
#include <iostream>
#include <cuda_runtime.h>

int st::GetCudaDeviceCount(std::vector<st::CUDADeviceInfo> &device_info) {
    device_info.clear();

    nvmlReturn_t result;

    //初始化NVML库
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return -1;
    }

    //获取设备数量
    unsigned int device_count;
    result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get device count: " << nvmlErrorString(result) << std::endl;
        return -1;
    }

    for (int i = 0; i < device_count; i++) {
        //获取设备句柄
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get device handle: " << nvmlErrorString(result) << std::endl;
            nvmlShutdown();
            return 1;
        }

        //获取设备名称
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get device name: " << nvmlErrorString(result) << std::endl;
            nvmlShutdown();
            return 1;
        }

        
        //查询内存信息
        nvmlMemory_t memory;
        result = nvmlDeviceGetMemoryInfo(device, &memory);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get memory info: " << nvmlErrorString(result) << std::endl;
            nvmlShutdown();
            return 1;
        }
        const int total_memory = memory.total / (1024 * 1024);
        const int free_memory = memory.free / (1024 * 1024);
        const int used_memory = memory.used / (1024 * 1024);

        //查询使用率信息
        nvmlUtilization_t utilization;
        result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get utilization rates: " << nvmlErrorString(result) << std::endl;
            nvmlShutdown();
            return 1;
        }

        //查询温度信息
        unsigned int temperature;
        result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get temperature: " << nvmlErrorString(result) << std::endl;
            nvmlShutdown();
            return 1;
        }
        
        //添加设备信息
        device_info.push_back({i, name, total_memory, free_memory, used_memory, 
                               static_cast<int>(utilization.gpu), static_cast<int>(utilization.memory),
                               static_cast<int>(temperature)});
    }

    //关闭NVML库
    result = nvmlShutdown();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to shutdown NVML: " << nvmlErrorString(result) << std::endl;
        return -1;
    }

    return 0;
}