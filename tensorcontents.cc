#include <memory>
#include <stdexcept>

#include "tensor.h"

//#define CUDA
#ifdef CUDA
    #include "tensorgpufunctions.cuh"
    #include "tensorgpuutility.cuh"
    
    #define CALLFUNC(NAME, ARGS) if(onGPU) gpu##NAME ARGS; else cpu##NAME ARGS;
    #define MAKEDATA \
            (onGPU) ? TensorGPUUtility::allocate(dataLen) : \
                std::make_shared<double>(new double[dataLen], std::default_delete<double[]>());
#else // no CUDA
    #include "tensorcpufunctions.h"
    #define CALLFUNC(NAME, ARGS) cpu##NAME ARGS;
    #define MAKEDATA std::make_shared<double>(new double[dataLen], std::default_delete<double[]>());
#endif

#define ISSCALAR(TENSOR) ((TENSOR).getDims().size() == 1 && (TENSOR).getDims()[0] == 1)

enum operation {ZEROES, ADD, ADDSCALAR, NEG, SOFTMAX, SUBTRACT, SUBTRACTSCALAR,
    ELEMENTWISEMULT, ELEMENTWISEMULTSCALAR, ELEMENTWISEDIVISION,
    ELEMENTWISEDIVISIONSCALAR, RELU, BINARIZE, POW, 
    ONES, MATMUL, FILL, DATA, REDUCESUM, TRANSPOSE, RESHAPE};


struct TensorContents{
    vDataPtr data;
    bool onGPU = false;

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

    static size_t calculateDataLen(vDims dims){
        size_t dataLen = 1;
        for(auto d : dims){
            dataLen *= d;
        }
        return dataLen;
    }

    TensorContents(vDims dims, vDataPtr data, bool saveGradient)
        : dims(dims), data(data), saveGradient(saveGradient), evaluated(true), dataLen(calculateDataLen(dims)) {}

    TensorContents(vDims dims, bool saveGradient) : dims(dims), saveGradient(saveGradient), evaluated(false), dataLen(calculateDataLen(dims)) {}

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
            double * data1 = evalTensor(arg1).get();
            data = MAKEDATA;
            double * ret = data.get();

            CALLFUNC(Neg, (ret, data1, dataLen));
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
            double * data1 = evalTensor(arg1).get();
            double * data2 = evalTensor(arg2).get();
            data = MAKEDATA;
            double * ret = data.get();

            if(ISSCALAR(arg1)) {CALLFUNC(AddScalar, (ret, data2, data1[0], dataLen));}
            else if(ISSCALAR(arg2)) {CALLFUNC(AddScalar, (ret, data1, data2[0], dataLen));}
            else {CALLFUNC(Add, (ret, data1, data2, dataLen));}
        }

        void backward(Tensor gradient){
            if(ISSCALAR(arg1)) arg1.backward(gradient.reduceSum());
            else arg1.backward(gradient);
            if(ISSCALAR(arg2)) arg2.backward(gradient.reduceSum());
            else arg2.backward(gradient);
        }
};

class TensorAddScalar : public TensorContents{
    Tensor arg1;
    double n;
    
    public:
        TensorAddScalar(vDims dims, bool saveGradient, Tensor arg1, double n)
            : arg1(arg1), n(n), TensorContents(dims, saveGradient) {}

        operation getOp() {return ADDSCALAR;}

        void eval(){
            double * data1 = evalTensor(arg1).get();
            data = MAKEDATA;
            double * ret = data.get();

            CALLFUNC(AddScalar, (ret, data1, n, dataLen));
        }

        void backward(Tensor gradient){
            arg1.backward(gradient);
        }
};

class TensorSubtract : public TensorContents{
    Tensor arg1, arg2;
    
    public:
        TensorSubtract(vDims dims, bool saveGradient, Tensor arg1, Tensor arg2)
            : arg1(arg1), arg2(arg2), TensorContents(dims, saveGradient) {}

        operation getOp() {return SUBTRACT;}

        void eval(){
            double * data1 = evalTensor(arg1).get();
            double * data2 = evalTensor(arg2).get();
            data = MAKEDATA;
            double * ret = data.get();

            if(ISSCALAR(arg1)) {CALLFUNC(ScalarSubtract, (ret, data2, data1[0], dataLen));}
            else if(ISSCALAR(arg2)) {CALLFUNC(SubtractScalar, (ret, data1, data2[0], dataLen));}
            else {CALLFUNC(Subtract, (ret, data1, data2, dataLen));}
        }

        void backward(Tensor gradient){
            if(ISSCALAR(arg1)) arg1.backward(gradient.reduceSum());
            else arg1.backward(gradient);
            if(ISSCALAR(arg2)) arg2.backward(gradient.reduceSum().neg());
            else arg2.backward(gradient.neg());
        }
};

class TensorSubtractScalar : public TensorContents{
    Tensor arg1;
    double n;
    
    public:
        TensorSubtractScalar(vDims dims, bool saveGradient, Tensor arg1, double n)
            : arg1(arg1), n(n), TensorContents(dims, saveGradient) {}

        operation getOp() {return SUBTRACTSCALAR;}

        void eval(){
            double * data1 = evalTensor(arg1).get();
            data = MAKEDATA;
            double * ret = data.get();

            CALLFUNC(SubtractScalar, (ret, data1, n, dataLen));
        }

        void backward(Tensor gradient){
            arg1.backward(gradient);
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
            double * data1 = evalTensor(arg1).get();
            data = MAKEDATA;
            double * ret = data.get();

            CALLFUNC(Pow, (ret, data1, n, dataLen));
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
            double * data1 = dataV1.get();
            data = MAKEDATA;
            double * ret = data.get();

            CALLFUNC(ReduceSum, (ret, data1, arg1.contents->dataLen));
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
            double * ret = data.get();

            CALLFUNC(Zeroes, (ret, dataLen));
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
            double * ret = data.get();

            CALLFUNC(Ones, (ret, dataLen));
        }

        void backward(Tensor gradient){
            (void) gradient;
        }
};

class TensorFill : public TensorContents{
    double n;

    public:
        TensorFill(vDims dims, bool saveGradient, double n) : TensorContents(dims, saveGradient), n(n) {}

        operation getOp() {return FILL;}

        void eval(){
            data = MAKEDATA;
            double * ret = data.get();

            CALLFUNC(Fill, (ret, n, dataLen));
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
            double * data1 = evalTensor(arg1).get();
            double * data2 = evalTensor(arg2).get();
            data = MAKEDATA;
            double * ret = data.get();

            if(ISSCALAR(arg1)) {CALLFUNC(ElementwiseMultScalar, (ret, data2, data1[0], dataLen));}
            else if(ISSCALAR(arg2)) {CALLFUNC(ElementwiseMultScalar, (ret, data1, data2[0], dataLen));}
            else {CALLFUNC(ElementwiseMult, (ret, data1, data2, dataLen));}
        }

        void backward(Tensor gradient){
            if(ISSCALAR(arg1)) arg1.backward((arg2 * gradient).reduceSum());
            else arg1.backward(arg2 * gradient);
            if(ISSCALAR(arg2)) arg2.backward((arg1 * gradient).reduceSum());
            else arg2.backward(arg1 * gradient);
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
            double * data1 = evalTensor(arg1).get();
            data = MAKEDATA;
            double * ret = data.get();

            CALLFUNC(ElementwiseMultScalar, (ret, data1, n, dataLen));
        }

        void backward(Tensor gradient){
            arg1.backward(gradient * n);
        }
};

class TensorRelu : public TensorContents{
    Tensor arg1;
    
    public:
        TensorRelu(vDims dims, bool saveGradient, Tensor arg1)
            : arg1(arg1), TensorContents(dims, saveGradient) {}

        operation getOp() {return RELU;}

        void eval(){
            double * data1 = evalTensor(arg1).get();
            data = MAKEDATA;
            double * ret = data.get();

            CALLFUNC(Relu, (ret, data1, dataLen));
        }

        void backward(Tensor gradient){
            arg1.backward(arg1.binarize() * gradient);
        }
};

class TensorBinarize : public TensorContents{
    Tensor arg1;
    
    public:
        TensorBinarize(vDims dims, bool saveGradient, Tensor arg1)
            : arg1(arg1), TensorContents(dims, saveGradient) {}

        operation getOp() {return BINARIZE;}

        void eval(){
            double * data1 = evalTensor(arg1).get();
            data = MAKEDATA;
            double * ret = data.get();

            CALLFUNC(Binarize, (ret, data1, dataLen));
        }

        void backward(Tensor gradient){
            (void) gradient;
            arg1.backward(Tensor::zeroes(dims));
        }
};

class TensorMatmul : public TensorContents{
    Tensor arg1, arg2;
    
    public:
        TensorMatmul(vDims dims, bool saveGradient, Tensor arg1, Tensor arg2)
            : arg1(arg1), arg2(arg2), TensorContents(dims, saveGradient) {}

        operation getOp() {return MATMUL;}

        void eval(){
            double * data1 = evalTensor(arg1).get();
            double * data2 = evalTensor(arg2).get();
            data = MAKEDATA;
            double * ret = data.get();

            vDims data1Dims = arg1.getDims();
            vDims data2Dims = arg2.getDims();

            if(dims.size() == 2)
                {CALLFUNC(Matmul2d, (ret, data1, data2, dims[0], dims[1], data1Dims[1], data2Dims[1]));}
            else
                {CALLFUNC(Matmul3d, (ret, data1, data2, dims[0], dims[1], dims[2], data1Dims[1], data1Dims[2], data2Dims[1]));}
        }

        void backward(Tensor gradient){
            if(dims.size() == 2){
                arg1.backward(gradient.matmul(arg2.transpose()));
                arg2.backward(arg1.transpose().matmul(gradient));
            }
            else{
                throw std::runtime_error("Backwards not yet implemented for batched tenor multiplication :(");
                //arg1.backward(gradient.matmul(arg2.transpose()));
                //arg2.backward();
            }
        }
};

class TensorTranspose : public TensorContents{
    Tensor arg1;
    
    public:
        TensorTranspose(vDims dims, bool saveGradient, Tensor arg1)
            : arg1(arg1), TensorContents(dims, saveGradient) {}

        operation getOp() {return TRANSPOSE;}

        void eval(){
            double * data1 = evalTensor(arg1).get();
            data = MAKEDATA;
            double * ret = data.get();

            if(dims.size() == 2)
                {CALLFUNC(Transpose2d, (ret, data1, dims[0], dims[1]));}
            else
                {CALLFUNC(Transpose3d, (ret, data1, dims[0], dims[1], dims[2]));}
        }

        void backward(Tensor gradient){
            arg1.backward(gradient.transpose());
        }
};

class TensorReshape : public TensorContents{
    Tensor arg1;
    
    public:
        TensorReshape(vDims dims, bool saveGradient, Tensor arg1)
            : arg1(arg1), TensorContents(dims, saveGradient) {}

        operation getOp() {return RESHAPE;}

        void eval(){
            data = evalTensor(arg1);
        }

        void backward(Tensor gradient){
            arg1.backward(gradient.reshape(arg1.getDims()));
        }
};

class TensorElementwiseDivision : public TensorContents{
    Tensor arg1, arg2;
    
    public:
        TensorElementwiseDivision(vDims dims, bool saveGradient, Tensor arg1, Tensor arg2)
            : arg1(arg1), arg2(arg2), TensorContents(dims, saveGradient) {}

        operation getOp() {return ELEMENTWISEDIVISION;}

        void eval(){
            double * data1 = evalTensor(arg1).get();
            double * data2 = evalTensor(arg2).get();
            data = MAKEDATA;
            double * ret = data.get();

            if(ISSCALAR(arg1)) {CALLFUNC(ElementwiseDivisionScalar2, (ret, data2, data1[0], dataLen));}
            else if(ISSCALAR(arg2)) {CALLFUNC(ElementwiseDivisionScalar, (ret, data1, data2[0], dataLen));}
            else {CALLFUNC(ElementwiseDivision, (ret, data1, data2, dataLen));}
        }

        void backward(Tensor gradient){
            if(ISSCALAR(arg1)) arg1.backward((gradient / arg2).reduceSum());
            else arg1.backward(gradient / arg2);
            if(ISSCALAR(arg2)) arg2.backward((gradient.neg() * arg1 / arg2.pow(2)).reduceSum());
            else arg2.backward(gradient.neg() * arg1 / arg2.pow(2));
        }
};

class TensorElementwiseDivisionScalar : public TensorContents{
    Tensor arg1;
    double n;
    
    public:
        TensorElementwiseDivisionScalar(vDims dims, bool saveGradient, Tensor arg1, double n)
            : arg1(arg1), n(n), TensorContents(dims, saveGradient) {}

        operation getOp() {return ELEMENTWISEDIVISIONSCALAR;}

        void eval(){
            double * data1 = evalTensor(arg1).get();
            data = MAKEDATA;
            double * ret = data.get();

            CALLFUNC(ElementwiseDivisionScalar, (ret, data1, n, dataLen));
        }

        void backward(Tensor gradient){
            arg1.backward(gradient / n);
        }
};

