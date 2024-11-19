#include <memory>
#include <stdexcept>
#include <vector>

#include "tensor.h"
#include "tensorfunction.cc"

Tensor::Tensor(vDims dims, std::vector<double> data){
    TensorContents cont = TensorContents(dims, data);
    contents = std::make_shared<TensorContents>(cont);
}

Tensor::Tensor(vDims dims, TensorFunctionPtr ptr){
    TensorContents cont = TensorContents(dims, ptr);
    contents = std::make_shared<TensorContents>(cont);
}


Tensor Tensor::zeroes(vDims dims){
    return Tensor(dims, std::make_shared<TensorFunctionZeroes>(TensorFunctionZeroes(dims)));
}

Tensor Tensor::neg(){
    this->contents->addArg();
    return Tensor(getDims(), std::make_shared<TensorFunctionNeg>(TensorFunctionNeg(this->contents)));
}

Tensor Tensor::add(Tensor x){
    if(getDims() != x.getDims()) throw std::runtime_error("Mismatched dimensions in Tensor::add");
    this->contents->addArg();
    x.contents->addArg();
    return Tensor(getDims(), std::make_shared<TensorFunctionAdd>(TensorFunctionAdd(this->contents, x.contents)));
}

Tensor Tensor::add(double x){
    this->contents->addArg();
    return Tensor(getDims(), std::make_shared<TensorFunctionAddScalar>(TensorFunctionAddScalar(this->contents, x)));
}

Tensor Tensor::softmax(){
    if(getDims().size() != 1)
        throw std::runtime_error("Dimensions must be 1d in Tensor::softmax");
    this->contents->addArg();
    return Tensor(getDims(), std::make_shared<TensorFunctionSoftmax>(TensorFunctionSoftmax(this->contents)));
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

