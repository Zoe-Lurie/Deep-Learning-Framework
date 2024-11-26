#ifndef TENSORGPUUTILITYH
#define TENSORGPUUTILITYH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <memory>

namespace TensorGPUUtility{
    void toGPU(double * data, double * d_data, size_t dataLen){
        cudaMalloc(&d_data, sizeof(double) * dataLen);
        cudaMemcpy(d_data, data, sizeof(double) * dataLen, cudaMemcpyHostToDevice);
    }

    void toCPU(double * data, double * d_data, size_t dataLen){
        cudaMemcpy(data, d_data, sizeof(double) * dataLen, cudaMemcpyDeviceToHost);
    }
    std::shared_ptr<double> allocate(size_t dataLen){
        double * tmp;
        cudaMalloc(&tmp, sizeof(double) * dataLen);
        return std::shared_ptr<double>(tmp, cudaFree);
    }
}
#endif

