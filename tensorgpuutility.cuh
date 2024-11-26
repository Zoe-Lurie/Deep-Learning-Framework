#ifndef TENSORGPUUTILITYH
#define TENSORGPUUTILITYH

#include "tensorgpufunctions.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <memory>

dim3 numblocks2dDIM(NUMBLOCKS2D, NUMBLOCKS2D);
dim3 numthreads2dDIM(NUMTHREADS2D, NUMTHREADS2D);
dim3 numblocks3dDIM(NUMBLOCKS3D, NUMBLOCKS3D, NUMBLOCKS3D);
dim3 numthreads3dDIM(NUMTHREADS3D, NUMTHREADS3D, NUMTHREADS3D);

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

    void gpuSNeg(double * ret, double * data1, size_t dataLen) {gpuNeg<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, dataLen);}
    void gpuSAdd(double * ret, double * data1, double * data2, size_t dataLen) {gpuAdd<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, data2, dataLen);}
    void gpuSAddScalar(double * ret, double * data1, double n, size_t dataLen) {gpuAddScalar<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, n, dataLen);}
    void gpuSSubtract(double * ret, double * data1, double * data2, size_t dataLen) {gpuSubtract<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, data2, dataLen);}
    void gpuSSubtractScalar(double * ret, double * data1, double n, size_t dataLen) {gpuSubtractScalar<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, n, dataLen);}
    void gpuSScalarSubtract(double * ret, double * data1, double n, size_t dataLen) {gpuScalarSubtract<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, n, dataLen);}
    void gpuSPow(double * ret, double * data1, double n, size_t dataLen) {gpuPow<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, n, dataLen);}
    void gpuSZeroes(double * ret, size_t dataLen) {gpuZeroes<<<NUMBLOCKS, NUMTHREADS>>>(ret, dataLen);}
    void gpuSOnes(double * ret, size_t dataLen) {gpuOnes<<<NUMBLOCKS, NUMTHREADS>>>(ret, dataLen);}
    void gpuSFill(double * ret, double n, size_t dataLen) {gpuFill<<<NUMBLOCKS, NUMTHREADS>>>(ret, n, dataLen);}
    void gpuSElementwiseMult(double * ret, double * data1, double * data2, size_t dataLen) {gpuElementwiseMult<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, data2, dataLen);}
    void gpuSElementwiseMultScalar(double * ret, double * data1, double n, size_t dataLen) {gpuElementwiseMultScalar<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, n, dataLen);}
    void gpuSElementwiseDivision(double * ret, double * data1, double * data2, size_t dataLen) {gpuElementwiseDivision<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, data2, dataLen);}
    void gpuSElementwiseDivisionScalar(double * ret, double * data1, double n, size_t dataLen) {gpuElementwiseDivisionScalar<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, n, dataLen);}
    void gpuSElementwiseDivisionScalar2(double * ret, double * data1, double n, size_t dataLen) {gpuElementwiseDivisionScalar2<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, n, dataLen);}
    void gpuSRelu(double * ret, double * data1, size_t dataLen) {gpuRelu<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, dataLen);}
    void gpuSBinarize(double * ret, double * data1, size_t dataLen) {gpuBinarize<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, dataLen);}
    void gpuSMatmul2d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t data1Dims1, size_t data2Dims1) {gpuMatmul2d<<<numblocks2dDIM, numthreads2dDIM>>>(ret, data1, data2, retDims0, retDims1, data1Dims1, data2Dims1);}
    void gpuSMatmul3d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t retDims2, size_t data1Dims1, size_t data1Dims2, size_t data2Dims1) {gpuMatmul3d<<<numblocks3dDIM, numthreads3dDIM>>>(ret, data1, data2, retDims0, retDims1, retDims2, data1Dims1, data1Dims2, data2Dims1);}
    void gpuSTranspose2d(double * ret, double * data1, size_t retDims0, size_t retDims1) {gpuTranspose2d<<<numblocks2dDIM, numthreads2dDIM>>>(ret, data1, retDims0, retDims1);}
    void gpuSTranspose3d(double * ret, double * data1, size_t retDims0, size_t retDims1, size_t retDims2) {gpuTranspose3d<<<numblocks3dDIM, numthreads3dDIM>>>(ret, data1, retDims0, retDims1, retDims2);}
    void gpuSReduceSum(double * ret, double * data1, size_t dataLen) {gpuReduceSum<<<1, 1>>>(ret, data1, dataLen);}
}
#endif

