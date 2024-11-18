#ifndef TENSORH
#define TENSORH

#include <vector>
#include <functional>
#include <variant>
#include <any>
#include <memory>

class Tensor;
class TensorData;

typedef std::vector<size_t> vDims;
typedef Tensor tEvalArg;
typedef std::vector<std::any> evalArgs;
typedef std::function<TensorData(evalArgs)> evalFunc;

class TensorData{
    public:
        std::vector<double> data;
        size_t dataLen;

        TensorData(vDims dims);
        TensorData(vDims dims, std::vector<double> data);
        size_t getDataLen() {return dataLen;}
};

class TensorFunction{
    evalArgs argv;
    evalFunc method;

    public:
        TensorFunction(evalArgs argv, evalFunc func) : argv(argv), method(func) {}
        TensorData callMethod() {return method(argv);}
        evalFunc getMethod() {return method;}
        evalArgs getArgv() {return argv;}
};

class TensorContents{
    std::variant<TensorData, TensorFunction> contents;
    vDims dims;
    size_t forwardArgCount = 0;

    public:
        TensorContents(vDims, evalArgs, evalFunc);
        TensorContents(vDims, std::vector<double>);

        TensorData getData() {return std::get<TensorData>(contents);}
        TensorFunction getFunc() {return std::get<TensorFunction>(contents);}
        bool isFunc() {return std::holds_alternative<TensorFunction>(contents);}

        void changeToData(TensorData& data) {contents = data;}

        vDims getDims() {return dims;}
        void addArg() {forwardArgCount++;}
        void delArg() {forwardArgCount--;}
        size_t getArgCount() {return forwardArgCount;}
};

class Tensor{
    private:
        std::shared_ptr<TensorContents> contents;

    public:
        Tensor(vDims dims, std::vector<double> data);
        //Tensor(vDims dims, std::vector<std::vector<size_t>> idx, std::vector<double> val);

        static Tensor zeroes(vDims dims);
        //static Tensor ones(vDims dims);

        //Tensor reshape(vDims new_dims);
        //Tensor transpose();
        Tensor neg();
        //Tensor reciprocal();

        Tensor add(Tensor);
        Tensor operator + (Tensor x) {return add(x);}
        Tensor add(double);
        Tensor operator + (double x) {return add(x);}
        friend Tensor operator + (double n, Tensor x) {return x.add(n);}

        //Tensor subtract( Tensor);
        //Tensor operator - (Tensor x) {return subtract(x);}
        //Tensor subtract(double);
        //Tensor operator - (double x) {return subtract(x);}
        //friend Tensor operator - (double n, Tensor x) {return x.subtract(n);}

        //Tensor elementwiseMult(Tensor);
        //Tensor operator * (Tensor x) {return elementwiseMult(x);}
        //Tensor elementwiseMult(double);
        //Tensor operator * (double x) {return elementwiseMult(x);}
        //friend Tensor operator * (double n, Tensor x) {return x.elementwiseMult(n);}

        //Tensor elementwiseDivision(Tensor);
        //Tensor operator / (Tensor x) {return elementwiseDivision(x);}
        //Tensor elementwiseDivision(double);
        //Tensor operator / (double x) {return elementwiseDivision(x);}
        //friend Tensor operator / (double n, Tensor x) {return x.elementwiseDivision(n);}

        //Tensor pow(double);
        //Tensor relu();
        //Tensor binarize();
        //Tensor exp();
        //Tensor matmul(Tensor&);

        //Tensor reduceSum(size_t dim = 0);
        //Tensor reduceProd(size_t dim = 0);
        //Tensor reduceMean(size_t dim = 0);
        Tensor softmax();
        //Tensor argmax();

        void print();
        std::vector<double> getData();
        vDims getDims() {return contents->getDims();}
    
    private:
        Tensor(vDims, evalArgs, evalFunc);
        void eval();

        static TensorData evalZeroes(evalArgs);

        static TensorData evalNeg(evalArgs);
        static TensorData evalAdd(evalArgs);
        static TensorData evalAddScalar(evalArgs);
        static TensorData evalSoftmax(evalArgs);
};
#endif

