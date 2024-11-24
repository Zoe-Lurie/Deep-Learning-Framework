#include <memory>
#include <stdexcept>
#include <vector>

#include "tensor.h"
#include "tensorcontents.cc"

#define MAKET(NAME, ARGS) Tensor(std::make_shared<Tensor##NAME>(Tensor##NAME ARGS))
//#define ADDARG(VAR, NUM) (VAR).contents->forwardArgs.push_back(std::pair(ret, NUM))

Tensor::Tensor(vDims dims, std::vector<double> data, bool saveGradient) {
    contents = std::make_shared<TensorContents>(TensorContents(dims, std::make_shared<std::vector<double>>(data), saveGradient));
}

Tensor::Tensor(TensorContentsPtr ptr) : contents(ptr) {}

vDataPtr Tensor::eval(){
    if(!contents->evaluated){
        //contents->optimize();
        contents->eval();
    }
    return contents->data;
}

void Tensor::backward(Tensor grad){
    if(contents->dims != grad.contents->dims) throw std::runtime_error("Dimenions of grad and tensor must match in backward in tensor with operation");
    if(!contents->saveGradient) return;

    if(!contents->gradient) contents->gradient = std::make_shared<Tensor>(grad);
    else contents->gradient = std::make_shared<Tensor>(*(contents->gradient) + grad);
    contents->backward(grad);
    contents->foundGradient = true;
}

Tensor Tensor::getGradient(){
    if(!contents->saveGradient) throw std::runtime_error("saveGradient in Tensor must be true");
    if(!contents->foundGradient) throw std::runtime_error("Backward must have been called on an output to this Tensor");
    return *contents->gradient;
}


vDims Tensor::getDims(){
    return contents->dims;
}

std::vector<double> Tensor::getData(){
    return *(eval());
}

void Tensor::print(){
    auto data = getData();
    for(auto d : data)
        printf("%f\n", d);
}


Tensor Tensor::zeroes(vDims dims, bool saveGradient){
    return MAKET(Zeroes, (dims, saveGradient));
}

Tensor Tensor::ones(vDims dims, bool saveGradient){
    return MAKET(Ones, (dims, saveGradient));
}

/*
Tensor Tensor::fill(vDims dims, double n){
    return MAKET(Fill, dims, (dims, n));
}
*/

Tensor Tensor::neg(bool saveGradient){
    Tensor ret =  MAKET(Neg, (contents->dims, saveGradient, *this));
    //ADDARG(*this, 1);
    return ret;
}

Tensor Tensor::add(Tensor x, bool saveGradient){
    if(getDims() != x.getDims()) throw std::runtime_error("Mismatched dimensions in Tensor::add");
    //ADDARG(*this);
    //ADDARG(x);
    return MAKET(Add, (contents->dims, saveGradient, *this, x));
}

/*
Tensor Tensor::add(double x){
    ADDARG(*this);
    return MAKET(AddScalar, getDims(), (this->contents, x));
}
*/

Tensor Tensor::subtract(Tensor x, bool saveGradient){
    if(getDims() != x.getDims()) throw std::runtime_error("Mismatched dimensions in Tensor::add");
    //ADDARG(*this);
    //ADDARG(x);
    return MAKET(Subtract, (contents->dims, saveGradient, *this, x));
}

/*
Tensor Tensor::subtract(double x){
    ADDARG(*this);
    return MAKET(AddScalar, getDims(), (this->contents, -x));
}
*/

/*
Tensor Tensor::softmax(){
    if(getDims().size() != 1) throw std::runtime_error("Dimensions must be 1d in Tensor::softmax");
    ADDARG(*this);
    return MAKET(Softmax, getDims(), (this->contents));
}
*/

Tensor Tensor::elementwiseMult(Tensor x, bool saveGradient){
    if(getDims() != x.getDims() && !(x.getDims().size() == 1 && x.getDims()[0] == 1)) throw std::runtime_error("Mismatched dimensions in Tensor::elementwiseMult");
    //ADDARG(*this);
    //ADDARG(x);
    return MAKET(ElementwiseMult, (contents->dims, saveGradient, *this, x));
}

Tensor Tensor::elementwiseMult(double x, bool saveGradient){
    Tensor ret =  MAKET(ElementwiseMultScalar, (contents->dims, saveGradient, *this, x));
    //ADDARG(*this, 1);
    return ret;
}

/*
Tensor Tensor::elementwiseDivision(Tensor x){
    if(getDims() != x.getDims()) throw std::runtime_error("Mismatched dimensions in Tensor::elementwiseDivision");
    ADDARG(*this);
    ADDARG(x);
    return MAKET(ElementwiseDivision, getDims(), (this->contents, x.contents));
}

Tensor Tensor::elementwiseDivision(double x){
    ADDARG(*this);
    return MAKET(ElementwiseDivisionScalar, getDims(), (this->contents, x));
}

Tensor Tensor::relu(){
    return MAKET(Relu, getDims(), (this->contents));
}

Tensor Tensor::binarize(){
    return MAKET(Binarize, getDims(), (this->contents));
}
*/

Tensor Tensor::pow(double x, bool saveGradient){
    Tensor ret =  MAKET(Pow, (contents->dims, saveGradient, *this, x));
    //ADDARG(*this, 1);
    return ret;
}

/*
Tensor Tensor::exp(){
    return MAKET(Exp, getDims(), (this->contents));
}

Tensor Tensor::reciprocal(){
    return MAKET(Reciprocal, getDims(), (this->contents));
}

Tensor Tensor::matmul(Tensor x){
    vDims dims = getDims();
    vDims xdims = x.getDims();
    if(xdims.size() != 2){
      throw std::runtime_error("The right operand of matmul must be 2D tensors");
    }
    if(dims.size() != 2 && dims.size() != 3){
      throw std::runtime_error("The left operand of matmul must be 2D tensors or batched 2D tensors");
    }
    if(dims[dims.size() - 1] != xdims[0]){
      throw std::runtime_error("Mismatched matmul matrix dimensions");
    }

    vDims retdims;
    if(dims.size() == 2) retdims = {dims[0], xdims[1]};
    else retdims = {dims[0], dims[1], xdims[1]};

    ADDARG(*this);
    ADDARG(x);
    return MAKET(Matmul, retdims, (this->contents, x.contents, retdims));
}
*/

Tensor Tensor::reduceSum(bool saveGradient){
    Tensor ret =  MAKET(ReduceSum, ({1}, saveGradient, *this));
    //ADDARG(*this, 1);
    return ret;
}

