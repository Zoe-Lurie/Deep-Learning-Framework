#ifndef TENSORGPUFUNCTIONH
#define TENSORGPUFUNCTIONH

#include <cstdlib>

#define NUMBLOCKS 256
#define NUMTHREADS 256
#define NUMBLOCKS2D 16
#define NUMTHREADS2D 16
#define NUMBLOCKS3D 8
#define NUMTHREADS3D 8

__global__ void gpuNeg(double * ret, double * data1, size_t dataLen);
__global__ void gpuAdd(double * ret, double * data1, double * data2, size_t dataLen);
__global__ void gpuAddScalar(double * ret, double * data1, double n, size_t dataLen);
__global__ void gpuSubtract(double * ret, double * data1, double * data2, size_t dataLen);
__global__ void gpuSubtractScalar(double * ret, double * data1, double n, size_t dataLen);
__global__ void gpuScalarSubtract(double * ret, double * data1, double n, size_t dataLen);
__global__ void gpuPow(double * ret, double * data1, double n, size_t dataLen);
__global__ void gpuZeroes(double * ret, size_t dataLen);
__global__ void gpuOnes(double * ret, size_t dataLen);
__global__ void gpuFill(double * ret, double n, size_t dataLen);
__global__ void gpuElementwiseMult(double * ret, double * data1, double * data2, size_t dataLen);
__global__ void gpuElementwiseMultScalar(double * ret, double * data1, double n, size_t dataLen);
__global__ void gpuElementwiseDivision(double * ret, double * data1, double * data2, size_t dataLen);
__global__ void gpuElementwiseDivisionScalar(double * ret, double * data1, double n, size_t dataLen);
__global__ void gpuElementwiseDivisionScalar2(double * ret, double * data1, double n, size_t dataLen);
__global__ void gpuRelu(double * ret, double * data1, size_t dataLen);
__global__ void gpuBinarize(double * ret, double * data1, size_t dataLen);
__global__ void gpuMatmul2d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t data1Dims1, size_t data2Dims1);
__global__ void gpuMatmul3d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t retDims2, size_t data1Dims1, size_t data1Dims2, size_t data2Dims1);
__global__ void gpuTranspose2d(double * ret, double * data1, size_t retDims0, size_t retDims1);
__global__ void gpuTranspose3d(double * ret, double * data1, size_t retDims0, size_t retDims1, size_t retDims2);
__global__ void gpuReduceSum(double * ret, double * data1, size_t dataLen);

#endif

