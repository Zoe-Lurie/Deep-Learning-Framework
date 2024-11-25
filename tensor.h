#ifndef TENSORH
#define TENSORH

#include <vector>
#include <memory>

class Tensor;
struct TensorContents;

typedef std::vector<size_t> vDims;
typedef std::shared_ptr<std::vector<double>> vDataPtr;
typedef std::shared_ptr<TensorContents> TensorContentsPtr;

class Tensor{
    friend struct TensorContents;
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

        static Tensor ones(vDims, bool saveGradient = false);
        static Tensor zeroes(vDims, bool saveGradient = false);
        static Tensor fill(vDims, double n, bool saveGradient = false);

        //Tensor reshape(vDims);
        Tensor transpose();

        Tensor add(Tensor, bool saveGradient = false);
        Tensor operator + (Tensor x) {return add(x);}
        Tensor add(double, bool saveGradient = false);
        Tensor operator + (double x) {return add(x);}
        friend Tensor operator + (double n, Tensor x) {return x.add(n);}

        Tensor subtract( Tensor, bool saveGradient = false);
        Tensor operator - (Tensor x) {return subtract(x);}
        Tensor subtract(double, bool saveGradient = false);
        Tensor operator - (double x) {return subtract(x);}
        friend Tensor operator - (double n, Tensor x) {return x.neg().add(n);}

        Tensor elementwiseMult(Tensor, bool saveGradient = false);
        Tensor operator * (Tensor x) {return elementwiseMult(x);}
        Tensor elementwiseMult(double, bool saveGradient = false);
        Tensor operator * (double x) {return elementwiseMult(x);}
        friend Tensor operator * (double n, Tensor x) {return x.elementwiseMult(n);}

        //Tensor elementwiseDivision(Tensor);
        //Tensor operator / (Tensor x) {return elementwiseDivision(x);}
        //Tensor elementwiseDivision(double);
        //Tensor operator / (double x) {return elementwiseDivision(x);}
        //friend Tensor operator / (double n, Tensor x) {return x.elementwiseDivision(n);}

        Tensor neg(bool saveGradient = false);
        Tensor pow(double, bool saveGradient = false);
        Tensor relu(bool saveGradient = false);
        Tensor binarize(bool saveGradient = false);

        Tensor matmul(Tensor, bool saveGradient = false);

        Tensor reduceSum(bool saveGradient = false);
        //Tensor reduceSum(size_t dim = 0);
        //Tensor reduceProd(size_t dim = 0);
        //Tensor reduceMean(size_t dim = 0);
        //Tensor softmax();
        //Tensor argmax();
};
#endif

