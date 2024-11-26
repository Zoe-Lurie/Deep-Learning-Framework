#ifndef TENSORH
#define TENSORH

#include <vector>
#include <memory>

struct TensorContents;

typedef std::vector<size_t> vDims;
typedef std::shared_ptr<double> vDataPtr;
typedef std::shared_ptr<TensorContents> TensorContentsPtr;

class Tensor{
    friend struct TensorContents;
    friend class TensorReshape;
    friend class TensorReduceSum;
    private:
        TensorContentsPtr contents;

        Tensor(TensorContentsPtr);
        vDataPtr eval();

    public:
        Tensor(vDims, std::vector<double> data, bool saveGradient = false);

        void print();
        std::vector<double> getData();
        vDims getDims();

#ifdef CUDA
      void toGPU();
      void toCPU();
#endif

        static Tensor ones(vDims, bool saveGradient = false);
        static Tensor zeroes(vDims, bool saveGradient = false);
        static Tensor fill(vDims, double n, bool saveGradient = false);

        void backward(Tensor grad = Tensor::ones({1}));
        Tensor getGradient();

        Tensor reshape(vDims, bool saveGradient = false);
        Tensor transpose(bool saveGradient = false);

        Tensor add(Tensor, bool saveGradient = false);
        Tensor operator + (Tensor x) {return add(x);}
        Tensor add(double, bool saveGradient = false);
        Tensor operator + (double x) {return add(x);}
        friend Tensor operator + (double n, Tensor x) {return x.add(n);}

        Tensor subtract( Tensor, bool saveGradient = false);
        Tensor operator - (Tensor x) {return subtract(x);}
        Tensor subtract(double, bool saveGradient = false);
        Tensor operator - (double x) {return subtract(x);}
        friend Tensor operator - (double n, Tensor x) {return fill({1}, n).subtract(x);}

        Tensor elementwiseMult(Tensor, bool saveGradient = false);
        Tensor operator * (Tensor x) {return elementwiseMult(x);}
        Tensor elementwiseMult(double, bool saveGradient = false);
        Tensor operator * (double x) {return elementwiseMult(x);}
        friend Tensor operator * (double n, Tensor x) {return x.elementwiseMult(n);}

        Tensor elementwiseDivision(Tensor, bool saveGradient = false);
        Tensor operator / (Tensor x) {return elementwiseDivision(x);}
        Tensor elementwiseDivision(double, bool saveGradient = false);
        Tensor operator / (double x) {return elementwiseDivision(x);}
        friend Tensor operator / (double n, Tensor x) {return fill({1}, n).elementwiseDivision(x);}

        Tensor neg(bool saveGradient = false);
        Tensor pow(double, bool saveGradient = false);
        Tensor relu(bool saveGradient = false);
        Tensor binarize(bool saveGradient = false);
        Tensor reciprocal(bool saveGradient = false) {return ones({1}).elementwiseDivision(*this, saveGradient);}

        Tensor matmul(Tensor, bool saveGradient = false);

        Tensor reduceSum(bool saveGradient = false);
        Tensor softmax(bool saveGradient = false) {return this->elementwiseDivision(this->reduceSum(saveGradient), saveGradient);}
};
#endif

