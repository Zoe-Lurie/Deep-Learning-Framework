#include <variant>

#include "tensor.h"

TensorContents::TensorContents(vDims dims, TensorFunctionPtr ptr, bool saveGradient)
    : contents(std::in_place_type<TensorFunctionPtr>, ptr), dims(dims), saveGradient(saveGradient) {}

TensorContents::TensorContents(vDims dims, vData data, bool saveGradient)
    : contents(std::in_place_type<TensorData>, dims, data), dims(dims), saveGradient(saveGradient) {}

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

