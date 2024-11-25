#ifndef TENSORFUNCTIONH
#define TENSORFUNCTIONH

#include <cstdlib>

void cpuNeg(double * ret, double * data1, size_t dataLen);
void cpuAdd(double * ret, double * data1, double * data2, size_t dataLen);
void cpuAddScalar(double * ret, double * data1, double n, size_t dataLen);
void cpuSubtract(double * ret, double * data1, double * data2, size_t dataLen);
void cpuSubtractScalar(double * ret, double * data1, double n, size_t dataLen);
void cpuScalarSubtract(double * ret, double * data1, double n, size_t dataLen);
void cpuPow(double * ret, double * data1, double n, size_t dataLen);
void cpuZeroes(double * ret, size_t dataLen);
void cpuOnes(double * ret, size_t dataLen);
void cpuFill(double * ret, double n, size_t dataLen);
void cpuElementwiseMult(double * ret, double * data1, double * data2, size_t dataLen);
void cpuElementwiseMultScalar(double * ret, double * data1, double n, size_t dataLen);
void cpuRelu(double * ret, double * data1, size_t dataLen);
void cpuBinarize(double * ret, double * data1, size_t dataLen);
void cpuMatmul2d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t data1Dims1, size_t data2Dims1);
void cpuMatmul3d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t retDims2, size_t data1Dims1, size_t data1Dims2, size_t data2Dims1);
void cpuTranspose2d(double * ret, double * data1, size_t retDims0, size_t retDims1);
void cpuTranspose3d(double * ret, double * data1, size_t retDims0, size_t retDims1, size_t retDims2);

#endif

