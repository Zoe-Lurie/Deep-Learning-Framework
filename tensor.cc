#include <memory>
#include <stdexcept>
#include <vector>

#include "tensor.h"

Tensor::Tensor(vDims dims, std::vector<double> data){
    TensorContents cont = TensorContents(dims, data);
    contents = std::make_shared<TensorContents>(cont);
}

Tensor::Tensor(vDims dims, evalArgs argv, evalFunc method){
    TensorContents cont = TensorContents(dims, argv, method);
    contents = std::make_shared<TensorContents>(cont);
}


Tensor Tensor::zeroes(vDims dims){
    return Tensor(dims, {std::make_any<vDims>(dims)}, evalZeroes);
}

Tensor Tensor::neg(){
    this->contents->addArg();
    return Tensor(getDims(), {std::make_any<tEvalArg>(*this)}, evalNeg);
}

Tensor Tensor::add(Tensor x){
    if(getDims() != x.getDims()) throw std::runtime_error("Mismatched dimensions in Tensor::add");
    this->contents->addArg();
    x.contents->addArg();
    return Tensor(getDims(), {std::make_any<tEvalArg>(*this), std::make_any<tEvalArg>(x)}, evalAdd);
}

Tensor Tensor::add(double x){
    this->contents->addArg();
    return Tensor(getDims(), {std::make_any<tEvalArg>(*this), std::make_any<double>(x)}, evalAddScalar);
}

Tensor Tensor::softmax(){
    if(getDims().size() != 1)
        throw std::runtime_error("Dimensions must be 1d in Tensor::softmax");
    this->contents->addArg();
    return Tensor(getDims(), {std::make_any<tEvalArg>(*this)}, evalSoftmax);
}

std::vector<double> Tensor::getData(){
    this->eval();
    return contents->getData().data;
}

void Tensor::print(){
    auto data = getData();
    for(auto d : data)
        printf("%f\n", d);
}

