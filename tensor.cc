#include <memory>
#include <stdexcept>
#include <vector>

#include "tensor.h"
#include "tensorfunction.cc"

#define MAKET(NAME, DIMS, ARGS) Tensor(DIMS, std::make_shared<TensorFunction##NAME>(TensorFunction##NAME ARGS))
#define ADDARG(VAR) (VAR).contents->addArg()

Tensor::Tensor(vDims dims, std::vector<double> data){
    TensorContents cont = TensorContents(dims, data);
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
    return (contents->getData().getData());
}

void Tensor::print(){
    auto data = getData();
    for(auto d : data)
        printf("%f\n", d);
}


Tensor Tensor::zeroes(vDims dims){
    return MAKET(Zeroes, dims, (dims));
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

