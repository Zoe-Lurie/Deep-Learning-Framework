#include <memory>

#include "tensor.h"

TensorContents::TensorContents(vDims dims, vDataPtr data, bool saveGradient) : dims(dims), data(data), saveGradient(saveGradient), gradient(Tensor::zeroes(dims)) {
    evaluated = true;

    dataLen = 1;
    for(auto d : dims){
        dataLen *= d;
    }
}

TensorContents::TensorContents(vDims dims, bool saveGradient) : dims(dims), saveGradient(saveGradient), gradient(Tensor::zeroes(dims)) {
    evaluated = false;
    
    dataLen = 1;
    for(auto d : dims){
        dataLen *= d;
    }
}

vDataPtr TensorContents::evalTensor(Tensor t){
    return t.eval();
}

class TensorNeg : public TensorContents{
    Tensor arg1;

    public:
        TensorNeg(vDims dims, bool saveGradient, Tensor arg1)
            : arg1(arg1), TensorContents(dims, saveGradient) {}

        operation getOp() {return NEG;}

        void eval(){
            double * data1 = evalTensor(arg1)->data();
            data->resize(dataLen);
            double * ret = data->data();

            for(size_t i = 0; i < dataLen; ++i){
                ret[i] = -data1[i];
            }
        }

        void backward(Tensor gradient){
            arg1.backward(gradient.neg());
        }
};

class TensorAdd : public TensorContents{
    Tensor arg1, arg2;
    
    public:
        TensorAdd(vDims dims, bool saveGradient, Tensor arg1, Tensor arg2)
            : arg1(arg1), arg2(arg2), TensorContents(dims, saveGradient) {}

        operation getOp() {return ADD;}

        void eval(){
            double * data1 = evalTensor(arg1)->data();
            double * data2 = evalTensor(arg2)->data();
            data->resize(dataLen);
            double * ret = data->data();

            for(size_t i = 0; i < dataLen; ++i){
                ret[i] = data1[i] + data2[i];
            }
        }

        void backward(Tensor gradient){
            arg1.backward(gradient);
            arg2.backward(gradient);
        }
};

