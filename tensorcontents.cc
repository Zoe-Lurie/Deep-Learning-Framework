#include "tensor.h"

TensorContents::TensorContents(vDims dims, TensorFunctionPtr ptr)
    : contents(std::in_place_type<TensorFunctionPtr>, ptr), dims(dims) {}

TensorContents::TensorContents(vDims dims, std::vector<double> data)
    : contents(std::in_place_type<TensorData>, dims, data), dims(dims) {}

void TensorContents::eval(){
    if(isFunc()){
        TensorData newData = std::get<TensorFunctionPtr>(contents)->eval();
        contents = newData;
    }
}

