#ifndef TENSORGPUFUNCTIONH
#define TENSORGPUFUNCTIONH

void gpuSNeg(double * ret, double * data1, size_t dataLen);

void gpuSAdd(double * ret, double * data1, double * data2, size_t dataLen);

void gpuSAddScalar(double * ret, double * data1, double n, size_t dataLen);

void gpuSSubtract(double * ret, double * data1, double * data2, size_t dataLen);

void gpuSSubtractScalar(double * ret, double * data1, double n, size_t dataLen);

void gpuSScalarSubtract(double * ret, double * data1, double n, size_t dataLen);

void gpuSPow(double * ret, double * data1, double n, size_t dataLen);

void gpuSZeroes(double * ret, size_t dataLen);

void gpuSOnes(double * ret, size_t dataLen);

void gpuSFill(double * ret, double n, size_t dataLen);

void gpuSElementwiseMult(double * ret, double * data1, double * data2, size_t dataLen);

void gpuSElementwiseMultScalar(double * ret, double * data1, double n, size_t dataLen);

void gpuSElementwiseDivision(double * ret, double * data1, double * data2, size_t dataLen);

void gpuSElementwiseDivisionScalar(double * ret, double * data1, double n, size_t dataLen);

void gpuSElementwiseDivisionScalar2(double * ret, double * data1, double n, size_t dataLen);

void gpuSRelu(double * ret, double * data1, size_t dataLen);

void gpuSBinarize(double * ret, double * data1, size_t dataLen);

void gpuSMatmul2d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t data1Dims1, size_t data2Dims1);

void gpuSMatmul3d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t retDims2, size_t data1Dims1, size_t data1Dims2, size_t data2Dims1);

void gpuSTranspose2d(double * ret, double * data1, size_t retDims0, size_t retDims1);

void gpuSTranspose3d(double * ret, double * data1, size_t retDims0, size_t retDims1, size_t retDims2);

void gpuSReduceSum(double * ret, double * data1, size_t dataLen);

#endif
