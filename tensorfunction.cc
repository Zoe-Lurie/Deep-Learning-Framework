#include "tensor.h"

class TensorFunctionNeg : public TensorFunction{
    TensorContentsPtr arg1;
    public:
        TensorFunctionNeg(TensorContentsPtr arg1) : arg1(arg1) { op = NEG; }

        TensorData eval(){
            arg1->eval();
            auto data1 = arg1->getData().data;

            TensorData retData(arg1->getDims());

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                retData.data[i] = -data1[i];
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

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                retData.data[i] = 0;
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
            auto data1 = arg1->getData().data;
            auto data2 = arg2->getData().data;

            TensorData retData(arg1->getDims());

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                retData.data[i] = data1[i] + data2[i];
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
            auto data1 = arg1->getData().data;

            TensorData retData(arg1->getDims());

            for(size_t i = 0; i < retData.getDataLen(); ++i){
                retData.data[i] = data1[i] + n;
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
            auto data1 = arg1->getData().data;

            TensorData retData(arg1->getDims());

            double sum = 0;
            for(size_t i = 0; i < retData.getDataLen(); ++i){
                sum += data1[i];
            }
            for(size_t i = 0; i < retData.getDataLen(); ++i){
                retData.data[i] = data1[i] / sum;
            }
            return retData;
        }
};

