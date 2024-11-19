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

Tensor Tensor::softmax(){
    if(getDims().size() != 1) throw std::runtime_error("Dimensions must be 1d in Tensor::softmax");
    ADDARG(*this);
    return MAKET(Softmax, getDims(), (this->contents));
}

std::vector<double> Tensor::getData(){
    this->contents->eval();
    return contents->getData().data;
}

void Tensor::print(){
    auto data = getData();
    for(auto d : data)
        printf("%f\n", d);
}

