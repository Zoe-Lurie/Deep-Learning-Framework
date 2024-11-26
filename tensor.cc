#include <memory>
#include <stdexcept>
#include <vector>

#include "tensor.h"
#include "tensorcontents.cc"

#ifdef CUDA
    #include "tensorgpuutility.cuh"
#endif

#define MAKET(NAME, ARGS) Tensor(std::make_shared<Tensor##NAME>(Tensor##NAME ARGS))

Tensor::Tensor(vDims dims, std::vector<double> data, bool saveGradient) {
    contents = std::make_shared<TensorContents>(
            TensorContents(dims, std::make_shared<double>(new double[data.size()], std::default_delete<double[]>()), saveGradient));
    std::copy(data.begin(), data.end(), contents->data.get());
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
    std::vector<double> ret;

    #ifdef CUDA
        if(contents->onGPU) toCPU();
    #endif

    ret.insert(ret.begin(), contents->data.get(), contents->data.get() + contents->dataLen);
    return ret;
}

void Tensor::print(){
    auto data = getData();
    for(auto d : data)
        printf("%f\n", d);
}

#ifdef CUDA
void Tensor::toGPU(){
    double * tmp;
    TensorGPUUtility::toGPU(contents->data.get(), tmp, contents->dataLen);
    contents->data = std::shared_ptr<double>(tmp, cudaFree);
    contents->onGPU = true;
}

void Tensor::toCPU(){
    double * tmp = new double[contents->dataLen];
    TensorGPUUtility::toCPU(tmp, contents->data.get(), contents->dataLen);
    contents->data = std::shared_ptr<double>(tmp, std::default_delete<double[]>());
    contents->onGPU = false;
}
#endif


bool isBroadcastable(vDims d1, vDims d2){
    return (d1 == d2) ||
        (d1.size() == 1 && d1[0] == 1) ||
        (d2.size() == 1 && d2[0] == 1);
}

vDims getBroadcastDims(vDims d1, vDims d2){
    if(d1 == d2) return d1;
    if(d1.size() == 1) return d2;
    return d1;
}


Tensor Tensor::zeroes(vDims dims, bool saveGradient){
    return MAKET(Zeroes, (dims, saveGradient));
}

Tensor Tensor::ones(vDims dims, bool saveGradient){
    return MAKET(Ones, (dims, saveGradient));
}

Tensor Tensor::fill(vDims dims, double n, bool saveGradient){
    return MAKET(Fill, (dims, saveGradient, n));
}

Tensor Tensor::neg(bool saveGradient){
    saveGradient = saveGradient || contents->saveGradient;
    Tensor ret =  MAKET(Neg, (contents->dims, saveGradient, *this));
    return ret;
}

Tensor Tensor::add(Tensor x, bool saveGradient){
    saveGradient = saveGradient || contents->saveGradient || x.contents->saveGradient;
    if(!isBroadcastable(contents->dims, x.contents->dims)) throw std::runtime_error("Mismatched dimensions in Tensor::add");
    return MAKET(Add, (getBroadcastDims(contents->dims, x.contents->dims), saveGradient, *this, x));
}

Tensor Tensor::add(double x, bool saveGradient){
    saveGradient = saveGradient || contents->saveGradient;
    return MAKET(AddScalar, (contents->dims, saveGradient, *this, x));
}

Tensor Tensor::subtract(Tensor x, bool saveGradient){
    if(!isBroadcastable(contents->dims, x.contents->dims)) throw std::runtime_error("Mismatched dimensions in Tensor::add");
    saveGradient = saveGradient || contents->saveGradient || x.contents->saveGradient;
    return MAKET(Subtract, (getBroadcastDims(contents->dims, x.contents->dims), saveGradient, *this, x));
}

Tensor Tensor::subtract(double x, bool saveGradient){
    saveGradient = saveGradient || contents->saveGradient;
    return MAKET(SubtractScalar, (contents->dims, saveGradient, *this, x));
}

Tensor Tensor::elementwiseMult(Tensor x, bool saveGradient){
    if(!isBroadcastable(contents->dims, x.contents->dims)) throw std::runtime_error("Mismatched dimensions in Tensor::elementwiseMult");
    saveGradient = saveGradient || contents->saveGradient || x.contents->saveGradient;
    return MAKET(ElementwiseMult, (getBroadcastDims(contents->dims, x.contents->dims), saveGradient, *this, x));
}

Tensor Tensor::elementwiseMult(double x, bool saveGradient){
    saveGradient = saveGradient || contents->saveGradient;
    return MAKET(ElementwiseMultScalar, (contents->dims, saveGradient, *this, x));
}

Tensor Tensor::elementwiseDivision(Tensor x, bool saveGradient){
    if(!isBroadcastable(contents->dims, x.contents->dims)) throw std::runtime_error("Mismatched dimensions in Tensor::elementwiseDivision");
    saveGradient = saveGradient || contents->saveGradient || x.contents->saveGradient;
    return MAKET(ElementwiseDivision, (getBroadcastDims(contents->dims, x.contents->dims), saveGradient, *this, x));
}

Tensor Tensor::elementwiseDivision(double x, bool saveGradient){
    saveGradient = saveGradient || contents->saveGradient;
    return MAKET(ElementwiseDivisionScalar, (contents->dims, saveGradient, *this, x));
}

Tensor Tensor::relu(bool saveGradient){
    saveGradient = saveGradient || contents->saveGradient;
    return MAKET(Relu, (contents->dims, saveGradient, *this));
}

Tensor Tensor::binarize(bool saveGradient){
    saveGradient = saveGradient || contents->saveGradient;
    return MAKET(Binarize, (contents->dims, saveGradient, *this));
}

Tensor Tensor::pow(double x, bool saveGradient){
    saveGradient = saveGradient || contents->saveGradient;
    return MAKET(Pow, (contents->dims, saveGradient, *this, x));
}

Tensor Tensor::matmul(Tensor x, bool saveGradient){
    vDims dims = contents->dims;
    vDims xdims = x.contents->dims;
    if(xdims.size() != 2){
      throw std::runtime_error("The right operand of matmul must be 2D tensors");
    }
    if(dims.size() != 2 && dims.size() != 3){
      throw std::runtime_error("The left operand of matmul must be 2D tensors or batched 2D tensors");
    }
    if(dims[dims.size() - 1] != xdims[0]){
      throw std::runtime_error("Mismatched matmul matrix dimensions");
    }

    saveGradient = saveGradient || contents->saveGradient || x.contents->saveGradient;
    vDims retdims;
    if(dims.size() == 2) retdims = {dims[0], xdims[1]};
    else retdims = {dims[0], dims[1], xdims[1]};

    return MAKET(Matmul, (retdims, saveGradient, *this, x));
}

Tensor Tensor::reduceSum(bool saveGradient){
    saveGradient = saveGradient || contents->saveGradient;
    return MAKET(ReduceSum, ({1}, saveGradient, *this));
}

Tensor Tensor::transpose(bool saveGradient){
    vDims retDims;
    if(contents->dims.size() == 2) retDims = {contents->dims[1], contents->dims[0]};
    else if(contents->dims.size() == 3) retDims = {contents->dims[0], contents->dims[2], contents->dims[1]};
    else throw std::runtime_error("The tensor must be 2d or batched 2d tensors in transpose");

    saveGradient = saveGradient || contents->saveGradient;
    return MAKET(Transpose, (retDims, saveGradient, *this));
}

Tensor Tensor::reshape(vDims dims, bool saveGradient){
    size_t newDataLen = TensorContents::calculateDataLen(dims);
    if(newDataLen != contents->dataLen) throw std::runtime_error("Dimensions do not match in reshape");

    saveGradient = saveGradient || contents->saveGradient;
    return MAKET(Reshape, (dims, saveGradient, *this));
}

