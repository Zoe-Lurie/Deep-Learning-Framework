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
    if(contents->dims != grad.contents->dims) throw std::runtime_error("Dimenions of grad and tensor must match in backward");
    if(!contents->saveGradient) return;

    contents->gradient = contents->gradient + grad;
    contents->backward(grad);
}

Tensor Tensor::getGradient(){
    if(!contents->saveGradient) throw std::runtime_error("saveGradient in Tensor must be true");
    if(!contents->foundGradient) throw std::runtime_error("Backward must have been called on an output to this Tensor");
    return contents->gradient;
}


vDims Tensor::getDims(){
    return contents->dims;
}

std::vector<double> Tensor::getData(){
    return *(eval().get());
}

void Tensor::print(){
    auto data = getData();
    for(auto d : data)
        printf("%f\n", d);
}


Tensor Tensor::zeroes(vDims dims){
    return MAKET(Zeroes, dims, (dims));
}

Tensor Tensor::ones(vDims dims){
    return MAKET(Ones, dims, (dims));
}

Tensor Tensor::fill(vDims dims, double n){
    return MAKET(Fill, dims, (dims, n));
}

Tensor Tensor::neg(){
    Tensor ret =  MAKET(Neg, (contents->dims, false, *this));
    //ADDARG(*this, 1);
    return ret;
}

Tensor Tensor::add(Tensor x){
    if(getDims() != x.getDims()) throw std::runtime_error("Mismatched dimensions in Tensor::add");
    //ADDARG(*this);
    //ADDARG(x);
    return MAKET(Add, (getDims(), false, this->contents, x.contents));
}

Tensor Tensor::add(double x){
    ADDARG(*this);
    return MAKET(AddScalar, getDims(), (this->contents, x));
}

Tensor Tensor::subtract(Tensor x){
    if(getDims() != x.getDims()) throw std::runtime_error("Mismatched dimensions in Tensor::subtract");
    ADDARG(*this);
    ADDARG(x);
    return MAKET(Subtract, getDims(), (this->contents, x.contents));
}

Tensor Tensor::subtract(double x){
    ADDARG(*this);
    return MAKET(AddScalar, getDims(), (this->contents, -x));
}

Tensor Tensor::softmax(){
    if(getDims().size() != 1) throw std::runtime_error("Dimensions must be 1d in Tensor::softmax");
    ADDARG(*this);
    return MAKET(Softmax, getDims(), (this->contents));
}

Tensor Tensor::elementwiseMult(Tensor x){
    if(getDims() != x.getDims()) throw std::runtime_error("Mismatched dimensions in Tensor::elementwiseMult");
    ADDARG(*this);
    ADDARG(x);
    return MAKET(ElementwiseMult, getDims(), (this->contents, x.contents));
}

Tensor Tensor::elementwiseMult(double x){
    ADDARG(*this);
    return MAKET(ElementwiseMultScalar, getDims(), (this->contents, x));
}

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

Tensor Tensor::pow(double x){
    ADDARG(*this);
    return MAKET(Pow, getDims(), (this->contents, x));
}

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

