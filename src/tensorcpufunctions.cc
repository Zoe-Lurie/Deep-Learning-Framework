/**
 * @file tensorcpufunctions.cc
 * @brief Implements the CPU functions.
 * 
 * @author Zoe Lurie
 * @date November 2024
 */

#include <cstdlib>
#include <cmath>
#include <random>

#include "tensorcpufunctions.h"

std::mt19937 generator(7);

void cpuNeg(double * ret, double * data1, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = -data1[i];
    }
}

void cpuAdd(double * ret, double * data1, double * data2, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] + data2[i];
    }
}

void cpuAddScalar(double * ret, double * data1, double n, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] + n;
    }
}

void cpuSubtract(double * ret, double * data1, double * data2, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] - data2[i];
    }
}

void cpuSubtractScalar(double * ret, double * data1, double n, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] - n;
    }
}

void cpuScalarSubtract(double * ret, double * data1, double n, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = n - data1[i];
    }
}

void cpuPow(double * ret, double * data1, double n, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = std::pow(data1[i], n);
    }
}

void cpuZeroes(double * ret, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = 0;
    }
}

void cpuOnes(double * ret, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = 1;
    }
}

void cpuFill(double * ret, double n, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = n;
    }
}

void cpuElementwiseMult(double * ret, double * data1, double * data2, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] * data2[i];
    }
}

void cpuElementwiseMultScalar(double * ret, double * data1, double n, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] * n;
    }
}

void cpuElementwiseDivision(double * ret, double * data1, double * data2, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] / data2[i];
    }
}

void cpuElementwiseDivisionScalar(double * ret, double * data1, double n, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] / n;
    }
}

void cpuElementwiseDivisionScalar2(double * ret, double * data1, double n, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = n / data1[i];
    }
}

void cpuRelu(double * ret, double * data1, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] > 0 ? data1[i] : 0;
    }
}

void cpuBinarize(double * ret, double * data1, size_t dataLen){
    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] > 0 ? 1 : 0;
    }
}

void cpuMatmul2d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t data1Dims1, size_t data2Dims1){
    #pragma omp parallel for
    for(size_t i = 0; i < retDims0; ++i){
        for(size_t j = 0; j < retDims1; ++j){
            ret[i * retDims0 + j] = 0;
            for(size_t k = 0; k < data1Dims1; ++k){
                ret[i * retDims1 + j] += data1[i * data1Dims1 + k] * data2[k * data2Dims1 + j];
            }
        }
    }
}

void cpuMatmul3d(double * ret, double * data1, double * data2, size_t retDims0, size_t retDims1, size_t retDims2, size_t data1Dims1, size_t data1Dims2, size_t data2Dims1){
    #pragma omp parallel for
    for(size_t b = 0; b < retDims0; ++b){
        for(size_t i = 0; i < retDims1; ++i){
            for(size_t j = 0; j < retDims2; ++j){
                ret[b * retDims2 * retDims1 + i * retDims1 + j] = 0;
                for(size_t k = 0; k < data1Dims2; ++k){
                    ret[b * retDims2 * retDims1 + i * retDims2 + j] += data1[b * data1Dims2 * data1Dims1 + i * data1Dims2 + k] * data2[k * data2Dims1 + j];
                }
            }
        }
    }
}

void cpuTranspose2d(double * ret, double * data1, size_t retDims0, size_t retDims1){
    #pragma omp parallel for
    for(size_t i = 0; i < retDims0; ++i){
        for(size_t j = 0; j < retDims1; ++j){
            ret[j * retDims0 + i] = data1[i * retDims1 + j];
        }
    }
}

void cpuTranspose3d(double * ret, double * data1, size_t retDims0, size_t retDims1, size_t retDims2){
    #pragma omp parallel for
    for(size_t b = 0; b < retDims0; ++b){
        for(size_t i = 0; i < retDims1; ++i){
            for(size_t j = 0; j < retDims2; ++j){
                ret[b * retDims1 * retDims2 + j * retDims1 + i] = data1[b * retDims1 * retDims2 + i * retDims2 + j];
            }
        }
    }
}

void cpuReduceSum(double * ret, double * data1, size_t dataLen){
    ret[0] = 0;
    #pragma omp parallel for reduction(+:ret[0])
    for(size_t i = 0; i < dataLen; ++i){
        ret[0] += data1[i];
    }
}

void cpuFillRandom(double * ret, double mean, double stddev, size_t dataLen){
    std::normal_distribution<double> dist(mean, stddev);

    #pragma omp parallel for
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = dist(generator);
    }
}

