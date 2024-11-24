#include <memory>
#include <cmath>
#include <vector>

#include "tensor.h"

#define MAKEDATA std::make_shared<std::vector<double>>(std::vector<double>(dataLen));

struct TensorContents{
    vDataPtr data;
    size_t dataLen;
    vDims dims;

    bool evaluated;
    bool saveGradient;
    bool foundGradient = false;
    std::shared_ptr<Tensor> gradient;

    virtual ~TensorContents() = default;

    virtual operation getOp() {return DATA;}
    virtual void eval() {};
    virtual void backward(Tensor) {};

    TensorContents(vDims dims, vDataPtr data, bool saveGradient) : dims(dims), data(data), saveGradient(saveGradient) {
        evaluated = true;

        dataLen = 1;
        for(auto d : dims){
            dataLen *= d;
        }
    }

    TensorContents(vDims dims, bool saveGradient) : dims(dims), saveGradient(saveGradient) {
        evaluated = false;
        
        dataLen = 1;
        for(auto d : dims){
            dataLen *= d;
        }
    }

    static vDataPtr evalTensor(Tensor t){
        return t.eval();
    }
};

class TensorNeg : public TensorContents{
    Tensor arg1;

    public:
        TensorNeg(vDims dims, bool saveGradient, Tensor arg1)
            : arg1(arg1), TensorContents(dims, saveGradient) {}

        operation getOp() {return NEG;}

        void eval(){
            double * data1 = evalTensor(arg1)->data();
            data = MAKEDATA;
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
            data = MAKEDATA;
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

class TensorSubtract : public TensorContents{
    Tensor arg1, arg2;
    
    public:
        TensorSubtract(vDims dims, bool saveGradient, Tensor arg1, Tensor arg2)
            : arg1(arg1), arg2(arg2), TensorContents(dims, saveGradient) {}

        operation getOp() {return SUBTRACT;}

        void eval(){
            double * data1 = evalTensor(arg1)->data();
            double * data2 = evalTensor(arg2)->data();
            data = MAKEDATA;
            double * ret = data->data();

            for(size_t i = 0; i < dataLen; ++i){
                ret[i] = data1[i] - data2[i];
            }
        }

        void backward(Tensor gradient){
            arg1.backward(gradient);
            arg2.backward(gradient.neg());
        }
};

class TensorPow : public TensorContents{
    Tensor arg1;
    double n;
    
    public:
        TensorPow(vDims dims, bool saveGradient, Tensor arg1, double n)
            : arg1(arg1), n(n), TensorContents(dims, saveGradient) {}

        operation getOp() {return POW;}

        void eval(){
            double * data1 = evalTensor(arg1)->data();
            data = MAKEDATA;
            double * ret = data->data();

            for(size_t i = 0; i < dataLen; ++i){
                ret[i] = std::pow(data1[i], n);
            }
        }

        void backward(Tensor gradient){
            arg1.backward(gradient * n * (arg1.pow(n - 1)));
        }
};

class TensorReduceSum : public TensorContents{
    Tensor arg1;
    // dimension of reduction
    
    public:
        TensorReduceSum(vDims dims, bool saveGradient, Tensor arg1)
            : arg1(arg1), TensorContents(dims, saveGradient) {}

        operation getOp() {return REDUCESUM;}

        void eval(){
            auto dataV1 = evalTensor(arg1);
            double * data1 = dataV1->data();
            data = MAKEDATA;
            double * ret = data->data();

            ret[0] = 0;
            for(size_t i = 0; i < dataV1->size(); ++i){
                ret[0] += data1[i];
            }
        }

        void backward(Tensor gradient){
            arg1.backward(Tensor::ones(arg1.getDims()) * gradient);
        }
};

class TensorZeroes : public TensorContents{
    public:
        TensorZeroes(vDims dims, bool saveGradient) : TensorContents(dims, saveGradient) {}

        operation getOp() {return ZEROES;}

        void eval(){
            data = MAKEDATA;
            double * ret = data->data();

            for(size_t i = 0; i < dataLen; ++i){
                ret[i] = 0;
            }
        }

        void backward(Tensor gradient){
            (void) gradient;
        }
};

class TensorOnes : public TensorContents{
    public:
        TensorOnes(vDims dims, bool saveGradient) : TensorContents(dims, saveGradient) {}

        operation getOp() {return ONES;}

        void eval(){
            data = MAKEDATA;
            double * ret = data->data();

            for(size_t i = 0; i < dataLen; ++i){
                ret[i] = 1;
            }
        }

        void backward(Tensor gradient){
            (void) gradient;
        }
};

class TensorElementwiseMult : public TensorContents{
    Tensor arg1, arg2;
    
    public:
        TensorElementwiseMult(vDims dims, bool saveGradient, Tensor arg1, Tensor arg2)
            : arg1(arg1), arg2(arg2), TensorContents(dims, saveGradient) {}

        operation getOp() {return ELEMENTWISEMULT;}

        void eval(){
            double * data1 = evalTensor(arg1)->data();
            double * data2 = evalTensor(arg2)->data();
            data = MAKEDATA;
            double * ret = data->data();

            if(arg2.getDims().size() == 1 && arg2.getDims()[0] == 1){
                for(size_t i = 0; i < dataLen; ++i){
                    ret[i] = data1[i] * data2[0];
                }
            }
            else{
                for(size_t i = 0; i < dataLen; ++i){
                    ret[i] = data1[i] * data2[i];
                }
            }
        }

        void backward(Tensor gradient){
            arg1.backward(arg2 * gradient);
            arg2.backward(arg1 * gradient);
        }
};

class TensorElementwiseMultScalar : public TensorContents{
    Tensor arg1;
    double n;
    
    public:
        TensorElementwiseMultScalar(vDims dims, bool saveGradient, Tensor arg1, double n)
            : arg1(arg1), n(n), TensorContents(dims, saveGradient) {}

        operation getOp() {return ELEMENTWISEMULTSCALAR;}

        void eval(){
            double * data1 = evalTensor(arg1)->data();
            data = MAKEDATA;
            double * ret = data->data();

            for(size_t i = 0; i < dataLen; ++i){
                ret[i] = data1[i] * n;
            }
        }

        void backward(Tensor gradient){
            arg1.backward(gradient * n);
        }
};

