#include <memory>
#include <stdexcept>
#include <vector>

#include "tensor.h"
#include "tensorfunction.cc"

#define MAKET(NAME, DIMS, ARGS) Tensor(DIMS, std::make_shared<TensorFunction##NAME>(TensorFunction##NAME ARGS))
#define ADDARG(VAR) (VAR).contents->addArg()

Tensor::Tensor(vDims dims, std::vector<double> data){
    TensorContents cont = TensorContents(dims, std::make_shared<std::vector<double>>(data));
    contents = std::make_shared<TensorContents>(cont);
}

Tensor::Tensor(vDims dims, TensorFunctionPtr ptr){
    TensorContents cont = TensorContents(dims, ptr);
    contents = std::make_shared<TensorContents>(cont);
}


vDims Tensor::getDims(){
    return contents->getDims();
}

std::vector<double> Tensor::getData(){
    this->contents->eval();
    return *(contents->getData().getData());
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

Tensor Tensor::neg(){
    ADDARG(*this);
    return MAKET(Neg, getDims(), (this->contents));
}

Tensor Tensor::add(Tensor x){
    if(getDims() != x.getDims()) throw std::runtime_error("Mismatched dimensions in Tensor::add");
    ADDARG(*this);
    ADDARG(x);
    return MAKET(Add, getDims(), (this->contents, x.contents));
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

