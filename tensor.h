#ifndef TENSORH
#define TENSORH

#include <vector>
#include <memory>

class Tensor;
struct TensorContents;

typedef std::vector<size_t> vDims;
typedef std::shared_ptr<std::vector<double>> vDataPtr;
typedef std::shared_ptr<TensorContents> TensorContentsPtr;

enum operation {ZEROES, ADD, ADDSCALAR, NEG, SOFTMAX, SUBTRACT, ELEMENTWISEMULT,
    ELEMENTWISEMULTSCALAR, ELEMENTWISEDIVISION,
    ELEMENTWISEDIVISIONSCALAR, RELU, BINARIZE, POW, EXP, RECIPROCAL,
    ONES, MATMUL, FILL, DATA};


class Tensor{
    friend class TensorContents;
    private:
        TensorContentsPtr contents;

        Tensor(TensorContentsPtr);
        vDataPtr eval();

    public:
        Tensor(vDims, std::vector<double> data, bool saveGradient = false);
        //Tensor(vDims dims, std::vector<std::vector<size_t>> idx, std::vector<double> val);

        void print();
        std::vector<double> getData();
        vDims getDims();

        void backward(Tensor grad = Tensor({1}, {1}));
        Tensor getGradient();

        static Tensor zeroes(vDims);
        static Tensor ones(vDims);
        static Tensor fill(vDims, double n);

        //Tensor reshape(vDims);
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

        Tensor matmul(Tensor);

        //Tensor reduceSum(size_t dim = 0);
        //Tensor reduceProd(size_t dim = 0);
        //Tensor reduceMean(size_t dim = 0);
        Tensor softmax();
        //Tensor argmax();
};

struct TensorContents{
    vDataPtr data;
    size_t dataLen;
    vDims dims;

    bool evaluated;
    bool saveGradient;
    bool foundGradient = false;
    Tensor gradient;

    TensorContents(vDims, vDataPtr, bool saveGradient);
    TensorContents(vDims, bool saveGradient);
    virtual ~TensorContents() = default;

    vDataPtr getData();
    virtual operation getOp() {return DATA;}
    virtual void eval() {};
    virtual void backward(Tensor) {};

    static vDataPtr evalTensor(Tensor);
};

#endif

