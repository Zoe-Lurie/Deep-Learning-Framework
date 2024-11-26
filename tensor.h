#ifndef TENSORH
#define TENSORH

#include <vector>
#include <memory>

struct TensorContents;

typedef std::vector<size_t> vDims;
typedef std::shared_ptr<double> vDataPtr;
typedef std::shared_ptr<TensorContents> TensorContentsPtr;

enum deviceOptions {CPU, GPU, DEFAULTDEVICE};

class Tensor{
    friend struct TensorContents;
    friend class TensorReshape;
    friend class TensorReduceSum;
    private:
        TensorContentsPtr contents;

        Tensor(TensorContentsPtr);
        vDataPtr eval();

    public:
        Tensor(vDims, std::vector<double> data, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        void print();
        std::vector<double> getData();
        vDims getDims();

        static Tensor ones(vDims, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        static Tensor zeroes(vDims, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        static Tensor fill(vDims, double n, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        void backward(Tensor grad = Tensor::ones({1}));
        Tensor getGradient();

        Tensor reshape(vDims, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor transpose(bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        Tensor add(Tensor, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator + (Tensor x) {return add(x);}
        Tensor add(double, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator + (double x) {return add(x);}
        friend Tensor operator + (double n, Tensor x) {return x.add(n);}

        Tensor subtract( Tensor, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator - (Tensor x) {return subtract(x);}
        Tensor subtract(double, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator - (double x) {return subtract(x);}
        friend Tensor operator - (double n, Tensor x) {return fill({1}, n).subtract(x);}

        Tensor elementwiseMult(Tensor, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator * (Tensor x) {return elementwiseMult(x);}
        Tensor elementwiseMult(double, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator * (double x) {return elementwiseMult(x);}
        friend Tensor operator * (double n, Tensor x) {return x.elementwiseMult(n);}

        Tensor elementwiseDivision(Tensor, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator / (Tensor x) {return elementwiseDivision(x);}
        Tensor elementwiseDivision(double, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor operator / (double x) {return elementwiseDivision(x);}
        friend Tensor operator / (double n, Tensor x) {return fill({1}, n).elementwiseDivision(x);}

        Tensor neg(bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor pow(double, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor relu(bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor binarize(bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor reciprocal(bool saveGradient = false, deviceOptions device = DEFAULTDEVICE) {return ones({1}).elementwiseDivision(*this, saveGradient, device);}

        Tensor matmul(Tensor, bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);

        Tensor reduceSum(bool saveGradient = false, deviceOptions device = DEFAULTDEVICE);
        Tensor softmax(bool saveGradient = false, deviceOptions device = DEFAULTDEVICE) {return this->elementwiseDivision(this->reduceSum(saveGradient), saveGradient, device);}
};
#endif

