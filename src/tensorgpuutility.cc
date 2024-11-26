/**
 * @file tensorgpuutility.cc
 * @brief Implements the GPU utility functions.
 * 
 * @author Zoe Lurie
 * @date November 2024
 */

#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "tensorgpuutility.h"

namespace TensorGPUUtility{
    std::shared_ptr<double> toGPU(double * data, size_t dataLen){
        double * d_data;
        cudaMalloc(&d_data, sizeof(double) * dataLen);
        cudaMemcpy(d_data, data, sizeof(double) * dataLen, cudaMemcpyHostToDevice);
        return std::shared_ptr<double>(d_data, cudaFree);
    }

    void toCPU(double * data, double * d_data, size_t dataLen){
        cudaMemcpy(data, d_data, sizeof(double) * dataLen, cudaMemcpyDeviceToHost);
    }

    std::shared_ptr<double> convert(std::shared_ptr<double> p, bool toGPU, size_t dataLen){
        if(toGPU){
            double * d_data;
            cudaMalloc(&d_data, sizeof(double) * dataLen);
            cudaMemcpy(d_data, p.get(), sizeof(double) * dataLen, cudaMemcpyHostToDevice);
            return std::shared_ptr<double>(d_data, cudaFree);
        }
        else{
            double * data = new double[dataLen];
            cudaMemcpy(data, p.get(), sizeof(double) * dataLen, cudaMemcpyDeviceToHost);
            return std::shared_ptr<double>(data, std::default_delete<double[]>());
        }
    }

    std::shared_ptr<double> allocate(size_t dataLen){
        double * tmp;
        cudaMalloc(&tmp, sizeof(double) * dataLen);
        return std::shared_ptr<double>(tmp, cudaFree);
    }
}
