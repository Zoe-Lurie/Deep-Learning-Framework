#ifndef TENSORH
#define TENSORH

#include <vector>
#include <functional>
#include <variant>
#include <any>
#include <memory>

class Tensor;
class TensorData;
class TensorFunction;
class TensorContents;

typedef std::vector<size_t> vDims;
typedef std::vector<double> vData;
typedef std::shared_ptr<TensorFunction> TensorFunctionPtr;
typedef std::shared_ptr<TensorContents> TensorContentsPtr;

class TensorData{
    size_t dataLen;
    vData data;
    public:
        TensorData(vDims);
        TensorData(vDims, vData);
        size_t getDataLen() {return dataLen;}
        vData& getData() {return data;}
};

class TensorFunction{
    protected:
        enum operation
            {ZEROES, ADD, ADDSCALAR, NEG, SOFTMAX, SUBTRACT, ELEMENTWISEMULT,
                ELEMENTWISEMULTSCALAR, ELEMENTWISEDIVISION,
                ELEMENTWISEDIVISIONSCALAR, RELU, BINARIZE, POW, EXP, RECIPROCAL,
                ONES}
            op;
    public:
        virtual ~TensorFunction() =default;
        operation getOp() {return op;}
        virtual TensorData eval() =0;
};

class TensorContents{
    std::variant<TensorData, TensorFunctionPtr> contents;
    vDims dims;
    size_t forwardArgCount = 0;

    public:
        TensorContents(vDims, TensorFunctionPtr);
        TensorContents(vDims, std::vector<double>);

        TensorData getData();
        TensorFunctionPtr getFunc();
        bool isFunc();
        void eval();

        vDims getDims() {return dims;}
        void addArg() {forwardArgCount++;}
        void delArg() {forwardArgCount--;}
        size_t getArgCount() {return forwardArgCount;}
};

class Tensor{
    private:
        TensorContentsPtr contents;

        Tensor(vDims, TensorFunctionPtr);

    public:
        Tensor(vDims, std::vector<double> data);
        //Tensor(vDims dims, std::vector<std::vector<size_t>> idx, std::vector<double> val);

        void print();
        std::vector<double> getData();
        vDims getDims();

        static Tensor zeroes(vDims);
        static Tensor ones(vDims);

        //Tensor reshape(vDims new_dims);
        //Tensor transpose();

        Tensor add(Tensor);
        Tensor operator + (Tensor x) {return add(x);}
        Tensor add(double);
        Tensor operator + (double x) {return add(x);}
        friend Tensor operator + (double n, Tensor x) {return x.add(n);}

        Tensor subtract( Tensor);
        Tensor operator - (Tensor x) {return subtract(x);}
        Tensor subtract(double);
        Tensor operator - (double x) {return subtract(x);}
        //friend Tensor operator - (double n, Tensor x) {return x.subtract(n);}

        Tensor elementwiseMult(Tensor);
        Tensor operator * (Tensor x) {return elementwiseMult(x);}
        Tensor elementwiseMult(double);
        Tensor operator * (double x) {return elementwiseMult(x);}
        friend Tensor operator * (double n, Tensor x) {return x.elementwiseMult(n);}

        Tensor elementwiseDivision(Tensor);
        Tensor operator / (Tensor x) {return elementwiseDivision(x);}
        Tensor elementwiseDivision(double);
        Tensor operator / (double x) {return elementwiseDivision(x);}
        //friend Tensor operator / (double n, Tensor x) {return x.elementwiseDivision(n);}

        Tensor neg();
        Tensor reciprocal();
        Tensor pow(double);
        Tensor relu();
        Tensor binarize();
        Tensor exp();

        //Tensor matmul(Tensor);

        //Tensor reduceSum(size_t dim = 0);
        //Tensor reduceProd(size_t dim = 0);
        //Tensor reduceMean(size_t dim = 0);
        Tensor softmax();
        //Tensor argmax();
};
#endif

