#include <variant>

#include "tensor.h"

TensorContents::TensorContents(vDims dims, TensorFunctionPtr ptr)
    : contents(std::in_place_type<TensorFunctionPtr>, ptr), dims(dims) {}

TensorContents::TensorContents(vDims dims, vData data)
    : contents(std::in_place_type<TensorData>, dims, data), dims(dims) {}

TensorData TensorContents::getData(){
    return std::get<TensorData>(contents);
}

TensorFunctionPtr TensorContents::getFunc(){
    return std::get<TensorFunctionPtr>(contents);
}

bool TensorContents::isFunc(){
    return std::holds_alternative<TensorFunctionPtr>(contents);
}

void TensorContents::eval(){
    if(isFunc()){
        TensorData newData = std::get<TensorFunctionPtr>(contents)->eval();
        contents = newData;
    }
}

