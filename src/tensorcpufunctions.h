/**
 * @file tensorcpufunctions.h
 * @brief CPU functions for each operation.
 * 
 * @author Zoe Lurie
 * @date November 2024
 */

#ifndef TENSORCPUFUNCTIONH
#define TENSORCPUFUNCTIONH

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
void cpuElementwiseDivision(double * ret, double * data1, double * data2, size_t dataLen);
void cpuElementwiseDivisionScalar(double * ret, double * data1, double n, size_t dataLen);
void cpuElementwiseDivisionScalar2(double * ret, double * data1, double n, size_t dataLen);
void cpuRelu(double * ret, double * data1, size_t dataLen);
void cpuBinarize(double * ret, double * data1, size_t dataLen);
void cpuMatmul2d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t data1Dims1, size_t data2Dims1);
void cpuMatmul3d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t retDims2, size_t data1Dims1, size_t data1Dims2, size_t data2Dims1);
void cpuTranspose2d(double * ret, double * data1, size_t retDims0, size_t retDims1);
void cpuTranspose3d(double * ret, double * data1, size_t retDims0, size_t retDims1, size_t retDims2);
void cpuReduceSum(double * ret, double * data1, size_t dataLen);
void cpuFillRandom(double * ret, double mean, double stddev, size_t dataLen);

#endif

