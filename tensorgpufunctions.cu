#include "tensorgpufunctions.h"

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

void gpuReduceSum(double * ret, double * data1, size_t dataLen){
    // kernel must be started with <<<1,1>>>
    ret[0] = 0;
    for(size_t i = 0; i < dataLen; ++i){
        ret[0] += data1[i];
    }
}
