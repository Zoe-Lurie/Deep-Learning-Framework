#include <cmath>

#include "tensor.h"

class TensorFunctionNeg : public TensorFunction{
    TensorContentsPtr arg1;
    public:
        TensorFunctionNeg(TensorContentsPtr arg1) : arg1(arg1) { op = NEG; }

        TensorData eval(){
            arg1->eval();
            auto data1 = arg1->getData().getData()->data();

            TensorData retData(arg1->getDims());
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = -data1[i];
            }
            return retData;
        }
};

class TensorFunctionZeroes : public TensorFunction{
    vDims dims;
    public:
        TensorFunctionZeroes(vDims dims) : dims(dims) { op = ZEROES; }

        TensorData eval(){
            TensorData retData(dims);
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = 0;
            }
            return retData;
        }
};

class TensorFunctionOnes : public TensorFunction{
    vDims dims;
    public:
        TensorFunctionOnes(vDims dims) : dims(dims) { op = ONES; }

        TensorData eval(){
            TensorData retData(dims);
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = 1;
            }
            return retData;
        }
};

class TensorFunctionAdd : public TensorFunction{
    TensorContentsPtr arg1, arg2;
    public:
        TensorFunctionAdd(TensorContentsPtr arg1, TensorContentsPtr arg2) : arg1(arg1), arg2(arg2) { op = ADD; }

        TensorData eval(){
            arg1->eval();
            arg2->eval();
            auto data1 = arg1->getData().getData()->data();
            auto data2 = arg2->getData().getData()->data();

            TensorData retData(arg1->getDims());
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = data1[i] + data2[i];
            }
            return retData;
        }
};

class TensorFunctionAddScalar : public TensorFunction{
    TensorContentsPtr arg1;
    double n;
    public:
        TensorFunctionAddScalar(TensorContentsPtr arg1, double n) : arg1(arg1), n(n) { op = ADDSCALAR; }

        TensorData eval(){
            arg1->eval();
            auto data1 = arg1->getData().getData()->data();

            TensorData retData(arg1->getDims());
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = data1[i] + n;
            }
            return retData;
        }
};

class TensorFunctionSoftmax : public TensorFunction{
    TensorContentsPtr arg1;
    public:
        TensorFunctionSoftmax(TensorContentsPtr arg1) : arg1(arg1) { op = SOFTMAX; }

        TensorData eval(){
            arg1->eval();
            auto data1 = arg1->getData().getData()->data();

            TensorData retData(arg1->getDims());
            auto data = retData.getData()->data();

            double sum = 0;
            for(size_t i = 0; i < retData.getDataLen(); ++i){
                sum += data1[i];
            }
            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = data1[i] / sum;
            }
            return retData;
        }
};

class TensorFunctionSubtract : public TensorFunction{
    TensorContentsPtr arg1, arg2;
    public:
        TensorFunctionSubtract(TensorContentsPtr arg1, TensorContentsPtr arg2) : arg1(arg1), arg2(arg2) { op = SUBTRACT; }

        TensorData eval(){
            arg1->eval();
            arg2->eval();
            auto data1 = arg1->getData().getData()->data();
            auto data2 = arg2->getData().getData()->data();

            TensorData retData(arg1->getDims());
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = data1[i] - data2[i];
            }
            return retData;
        }
};

class TensorFunctionElementwiseMult : public TensorFunction{
    TensorContentsPtr arg1, arg2;
    public:
        TensorFunctionElementwiseMult(TensorContentsPtr arg1, TensorContentsPtr arg2) : arg1(arg1), arg2(arg2) { op = ELEMENTWISEMULT; }

        TensorData eval(){
            arg1->eval();
            arg2->eval();
            auto data1 = arg1->getData().getData()->data();
            auto data2 = arg2->getData().getData()->data();

            TensorData retData(arg1->getDims());
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = data1[i] * data2[i];
            }
            return retData;
        }
};

class TensorFunctionElementwiseMultScalar : public TensorFunction{
    TensorContentsPtr arg1;
    double n;
    public:
        TensorFunctionElementwiseMultScalar(TensorContentsPtr arg1, double n) : arg1(arg1), n(n) { op = ELEMENTWISEMULTSCALAR; }

        TensorData eval(){
            arg1->eval();
            auto data1 = arg1->getData().getData()->data();

            TensorData retData(arg1->getDims());
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = data1[i] * n;
            }
            return retData;
        }
};

class TensorFunctionElementwiseDivision : public TensorFunction{
    TensorContentsPtr arg1, arg2;
    public:
        TensorFunctionElementwiseDivision(TensorContentsPtr arg1, TensorContentsPtr arg2) : arg1(arg1), arg2(arg2) { op = ELEMENTWISEDIVISION; }

        TensorData eval(){
            arg1->eval();
            arg2->eval();
            auto data1 = arg1->getData().getData()->data();
            auto data2 = arg2->getData().getData()->data();

            TensorData retData(arg1->getDims());
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = data1[i] / data2[i];
            }
            return retData;
        }
};

class TensorFunctionElementwiseDivisionScalar : public TensorFunction{
    TensorContentsPtr arg1;
    double n;
    public:
        TensorFunctionElementwiseDivisionScalar(TensorContentsPtr arg1, double n) : arg1(arg1), n(n) { op = ELEMENTWISEDIVISIONSCALAR; }

        TensorData eval(){
            arg1->eval();
            auto data1 = arg1->getData().getData()->data();

            TensorData retData(arg1->getDims());
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = data1[i] / n;
            }
            return retData;
        }
};

class TensorFunctionRelu : public TensorFunction{
    TensorContentsPtr arg1;
    public:
        TensorFunctionRelu(TensorContentsPtr arg1) : arg1(arg1) { op = RELU; }

        TensorData eval(){
            arg1->eval();
            auto data1 = arg1->getData().getData()->data();

            TensorData retData(arg1->getDims());
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = data1[i] > 0 ? data1[i] : 0;
            }
            return retData;
        }
};

class TensorFunctionBinarize : public TensorFunction{
    TensorContentsPtr arg1;
    public:
        TensorFunctionBinarize(TensorContentsPtr arg1) : arg1(arg1) { op = BINARIZE; }

        TensorData eval(){
            arg1->eval();
            auto data1 = arg1->getData().getData()->data();

            TensorData retData(arg1->getDims());
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = data1[i] > 0 ? 1 : 0;
            }
            return retData;
        }
};

class TensorFunctionPow : public TensorFunction{
    TensorContentsPtr arg1;
    double n;
    public:
        TensorFunctionPow(TensorContentsPtr arg1, double n) : arg1(arg1), n(n) { op = POW; }

        TensorData eval(){
            arg1->eval();
            auto data1 = arg1->getData().getData()->data();

            TensorData retData(arg1->getDims());
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = std::pow(data1[i], n);
            }
            return retData;
        }
};

class TensorFunctionExp : public TensorFunction{
    TensorContentsPtr arg1;
    public:
        TensorFunctionExp(TensorContentsPtr arg1) : arg1(arg1) { op = EXP; }

        TensorData eval(){
            arg1->eval();
            auto data1 = arg1->getData().getData()->data();

            TensorData retData(arg1->getDims());
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = std::exp(data1[i]);
            }
            return retData;
        }
};

class TensorFunctionReciprocal : public TensorFunction{
    TensorContentsPtr arg1;
    public:
        TensorFunctionReciprocal(TensorContentsPtr arg1) : arg1(arg1) { op = RECIPROCAL; }

        TensorData eval(){
            arg1->eval();
            auto data1 = arg1->getData().getData()->data();

            TensorData retData(arg1->getDims());
            auto data = retData.getData()->data();

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                data[i] = 1 / data1[i];
            }
            return retData;
        }
};

class TensorFunctionMatmul : public TensorFunction{
    TensorContentsPtr arg1, arg2;
    vDims dims;
    public:
        TensorFunctionMatmul(TensorContentsPtr arg1, TensorContentsPtr arg2, vDims dims) : arg1(arg1), arg2(arg2), dims(dims) { op = MATMUL; }

        TensorData eval(){
            arg1->eval();
            arg2->eval();
            auto data1 = arg1->getData().getData()->data();
            auto data2 = arg2->getData().getData()->data();

            TensorData retData(dims);
            auto data = retData.getData()->data();

            if(dims.size() == 2){
                for(size_t i = 0; i < dims[0]; ++i){
                    for(size_t j = 0; j < dims[1]; ++j){
                        data[i * dims[0] + j] = 0;
                        for(size_t k = 0; k < arg1->getDims()[1]; ++k){
                            data[i * dims[1] + j] += data1[i * arg1->getDims()[1] + k] * data2[k * arg2->getDims()[1] + j];
                        }
                    }
                }
            }
            else{
                for(size_t b = 0; b < dims[0]; ++b){
                    for(size_t i = 0; i < dims[1]; ++i){
                        for(size_t j = 0; j < dims[2]; ++j){
                            data[b * dims[2] * dims[1] + i * dims[1] + j] = 0;
                            for(size_t k = 0; k < arg1->getDims()[2]; ++k){
                                data[b * dims[2] * dims[1] + i * dims[2] + j] += data1[b * arg1->getDims()[2] * arg1->getDims()[1] + i * arg1->getDims()[2] + k] * data2[k * arg2->getDims()[1] + j];
                            }
                        }
                    }
                }
            }
            return retData;
        }
};

