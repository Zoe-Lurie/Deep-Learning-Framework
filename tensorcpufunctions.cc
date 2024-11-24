#include <cstdlib>
#include <cmath>

void cpuNeg(double * ret, double * data1, size_t dataLen){
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = -data1[i];
    }
}

void cpuAdd(double * ret, double * data1, double * data2, size_t dataLen){
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] + data2[i];
    }
}

void cpuAddScalar(double * ret, double * data1, double n, size_t dataLen){
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] + n;
    }
}

void cpuSubtract(double * ret, double * data1, double * data2, size_t dataLen){
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] - data2[i];
    }
}

void cpuSubtractScalar(double * ret, double * data1, double n, size_t dataLen){
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] - n;
    }
}

void cpuScalarSubtract(double * ret, double * data1, double n, size_t dataLen){
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = n - data1[i];
    }
}

void cpuPow(double * ret, double * data1, double n, size_t dataLen){
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = std::pow(data1[i], n);
    }
}

void cpuZeroes(double * ret, size_t dataLen){
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = 0;
    }
}

void cpuOnes(double * ret, size_t dataLen){
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = 1;
    }
}

void cpuFill(double * ret, double n, size_t dataLen){
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = n;
    }
}

void cpuElementwiseMult(double * ret, double * data1, double * data2, size_t dataLen){
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] * data2[i];
    }
}

void cpuElementwiseMultScalar(double * ret, double * data1, double n, size_t dataLen){
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] * n;
    }
}

void cpuRelu(double * ret, double * data1, size_t dataLen){
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] > 0 ? data1[i] : 0;
    }
}

void cpuBinarize(double * ret, double * data1, size_t dataLen){
    for(size_t i = 0; i < dataLen; ++i){
        ret[i] = data1[i] > 0 ? 1 : 0;
    }
}

