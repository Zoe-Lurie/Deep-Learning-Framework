/**
 * @file tensorgpufunction.cu
 * @brief Implements GPU functions for each operation.
 * 
 * @author Zoe Lurie
 * @date November 2024
 */

#include "tensorgpufunctions.h"

#define NUMBLOCKS 256
#define NUMTHREADS 256
#define NUMBLOCKS2D 16
#define NUMTHREADS2D 16
#define NUMBLOCKS3D 8
#define NUMTHREADS3D 8

dim3 numblocks2dDIM(NUMBLOCKS2D, NUMBLOCKS2D);
dim3 numthreads2dDIM(NUMTHREADS2D, NUMTHREADS2D);
dim3 numblocks3dDIM(NUMBLOCKS3D, NUMBLOCKS3D, NUMBLOCKS3D);
dim3 numthreads3dDIM(NUMTHREADS3D, NUMTHREADS3D, NUMTHREADS3D);


__global__ void gpuNeg(double * ret, double * data1, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = -data1[i];
    }
}

__global__ void gpuAdd(double * ret, double * data1, double * data2, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = data1[i] + data2[i];
    }
}

__global__ void gpuAddScalar(double * ret, double * data1, double n, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = data1[i] + n;
    }
}

__global__ void gpuSubtract(double * ret, double * data1, double * data2, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = data1[i] - data2[i];
    }
}

__global__ void gpuSubtractScalar(double * ret, double * data1, double n, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = data1[i] - n;
    }
}

__global__ void gpuScalarSubtract(double * ret, double * data1, double n, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = n - data1[i];
    }
}

__global__ void gpuPow(double * ret, double * data1, double n, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = pow(data1[i], n);
    }
}

__global__ void gpuZeroes(double * ret, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = 0;
    }
}

__global__ void gpuOnes(double * ret, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = 1;
    }
}

__global__ void gpuFill(double * ret, double n, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = n;
    }
}

__global__ void gpuElementwiseMult(double * ret, double * data1, double * data2, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = data1[i] * data2[i];
    }
}

__global__ void gpuElementwiseMultScalar(double * ret, double * data1, double n, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = data1[i] * n;
    }
}

__global__ void gpuRelu(double * ret, double * data1, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = data1[i] > 0 ? data1[i] : 0;
    }
}

__global__ void gpuElementwiseDivision(double * ret, double * data1, double * data2, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = data1[i] / data2[i];
    }
}

__global__ void gpuElementwiseDivisionScalar(double * ret, double * data1, double n, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = data1[i] / n;
    }
}

__global__ void gpuElementwiseDivisionScalar2(double * ret, double * data1, double n, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = n / data1[i];
    }
}

__global__ void gpuBinarize(double * ret, double * data1, size_t dataLen){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dataLen; i += NUMBLOCKS * NUMTHREADS){
        ret[i] = data1[i] > 0 ? 1 : 0;
    }
}

__global__ void gpuMatmul2d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t data1Dims1, size_t data2Dims1){
    for(size_t i = blockIdx.x  * blockDim.x + threadIdx.x; i < retDims0; i += NUMBLOCKS2D * NUMTHREADS2D){
        for(size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < retDims1; j += NUMBLOCKS2D * NUMTHREADS2D){
            ret[i * retDims0 + j] = 0;
            for(size_t k = 0; k < data1Dims1; ++k){
                ret[i * retDims1 + j] += data1[i * data1Dims1 + k] * data2[k * data2Dims1 + j];
            }
        }
    }
}

__global__ void gpuMatmul3d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t retDims2, size_t data1Dims1, size_t data1Dims2, size_t data2Dims1){
    for(size_t b = blockIdx.x * blockDim.x + threadIdx.x; b < retDims0; b += NUMBLOCKS3D * NUMTHREADS3D){
        for(size_t i = blockIdx.y * blockDim.y + threadIdx.y; i < retDims1; i += NUMBLOCKS3D * NUMTHREADS3D){
            for(size_t j = blockIdx.z * blockDim.z + threadIdx.z; j < retDims2; j += NUMBLOCKS3D * NUMTHREADS3D){
                ret[b * retDims2 * retDims1 + i * retDims1 + j] = 0;
                for(size_t k = 0; k < data1Dims2; ++k){
                    ret[b * retDims2 * retDims1 + i * retDims2 + j] += data1[b * data1Dims2 * data1Dims1 + i * data1Dims2 + k] * data2[k * data2Dims1 + j];
                }
            }
        }
    }
}

__global__ void gpuTranspose2d(double * ret, double * data1, size_t retDims0, size_t retDims1){
    for(size_t i = blockIdx.x  * blockDim.x + threadIdx.x; i < retDims0; i += NUMBLOCKS2D * NUMTHREADS2D){
        for(size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < retDims1; j += NUMBLOCKS2D * NUMTHREADS2D){
            ret[j * retDims0 + i] = data1[i * retDims1 + j];
        }
    }
}

__global__ void gpuTranspose3d(double * ret, double * data1, size_t retDims0, size_t retDims1, size_t retDims2){
    for(size_t b = blockIdx.x * blockDim.x + threadIdx.x; b < retDims0; b += NUMBLOCKS3D * NUMTHREADS3D){
        for(size_t i = blockIdx.y * blockDim.y + threadIdx.y; i < retDims1; i += NUMBLOCKS3D * NUMTHREADS3D){
            for(size_t j = blockIdx.z * blockDim.z + threadIdx.z; j < retDims2; j += NUMBLOCKS3D * NUMTHREADS3D){
                ret[b * retDims1 * retDims2 + j * retDims1 + i] = data1[b * retDims1 * retDims2 + i * retDims2 + j];
            }
        }
    }
}

__global__ void gpuReduceSum(double * ret, double * data1, size_t dataLen){
    // kernel must be started with <<<1,1>>>
    ret[0] = 0;
    for(size_t i = 0; i < dataLen; ++i){
        ret[0] += data1[i];
    }
}

void gpuSNeg(double * ret, double * data1, size_t dataLen)
{gpuNeg<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, dataLen);}

void gpuSAdd(double * ret, double * data1, double * data2, size_t dataLen)
{gpuAdd<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, data2, dataLen);}

void gpuSAddScalar(double * ret, double * data1, double n, size_t dataLen)
{gpuAddScalar<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, n, dataLen);}

void gpuSSubtract(double * ret, double * data1, double * data2, size_t dataLen)
{gpuSubtract<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, data2, dataLen);}

void gpuSSubtractScalar(double * ret, double * data1, double n, size_t dataLen)
{gpuSubtractScalar<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, n, dataLen);}

void gpuSScalarSubtract(double * ret, double * data1, double n, size_t dataLen)
{gpuScalarSubtract<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, n, dataLen);}

void gpuSPow(double * ret, double * data1, double n, size_t dataLen)
{gpuPow<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, n, dataLen);}

void gpuSZeroes(double * ret, size_t dataLen)
{gpuZeroes<<<NUMBLOCKS, NUMTHREADS>>>(ret, dataLen);}

void gpuSOnes(double * ret, size_t dataLen)
{gpuOnes<<<NUMBLOCKS, NUMTHREADS>>>(ret, dataLen);}

void gpuSFill(double * ret, double n, size_t dataLen)
{gpuFill<<<NUMBLOCKS, NUMTHREADS>>>(ret, n, dataLen);}

void gpuSElementwiseMult(double * ret, double * data1, double * data2, size_t dataLen)
{gpuElementwiseMult<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, data2, dataLen);}

void gpuSElementwiseMultScalar(double * ret, double * data1, double n, size_t dataLen)
{gpuElementwiseMultScalar<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, n, dataLen);}

void gpuSElementwiseDivision(double * ret, double * data1, double * data2, size_t dataLen)
{gpuElementwiseDivision<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, data2, dataLen);}

void gpuSElementwiseDivisionScalar(double * ret, double * data1, double n, size_t dataLen)
{gpuElementwiseDivisionScalar<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, n, dataLen);}

void gpuSElementwiseDivisionScalar2(double * ret, double * data1, double n, size_t dataLen)
{gpuElementwiseDivisionScalar2<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, n, dataLen);}

void gpuSRelu(double * ret, double * data1, size_t dataLen)
{gpuRelu<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, dataLen);}

void gpuSBinarize(double * ret, double * data1, size_t dataLen)
{gpuBinarize<<<NUMBLOCKS, NUMTHREADS>>>(ret, data1, dataLen);}

void gpuSMatmul2d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t data1Dims1, size_t data2Dims1)
{gpuMatmul2d<<<numblocks2dDIM, numthreads2dDIM>>>(ret, data1, data2, retDims0, retDims1, data1Dims1, data2Dims1);}

void gpuSMatmul3d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t retDims2, size_t data1Dims1, size_t data1Dims2, size_t data2Dims1)
{gpuMatmul3d<<<numblocks3dDIM, numthreads3dDIM>>>(ret, data1, data2, retDims0, retDims1, retDims2, data1Dims1, data1Dims2, data2Dims1);}

void gpuSTranspose2d(double * ret, double * data1, size_t retDims0, size_t retDims1)
{gpuTranspose2d<<<numblocks2dDIM, numthreads2dDIM>>>(ret, data1, retDims0, retDims1);}

void gpuSTranspose3d(double * ret, double * data1, size_t retDims0, size_t retDims1, size_t retDims2)
{gpuTranspose3d<<<numblocks3dDIM, numthreads3dDIM>>>(ret, data1, retDims0, retDims1, retDims2);}

void gpuSReduceSum(double * ret, double * data1, size_t dataLen)
{gpuReduceSum<<<1, 1>>>(ret, data1, dataLen);}
