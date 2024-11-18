#include <utility>
#include <vector>

#include "tensor.h"

TensorData::TensorData(vDims dims){
    dataLen = 1;
    for(auto d : dims){
        dataLen *= d;
    }

    data.resize(dataLen);
}

TensorData::TensorData(vDims dims, std::vector<double> data) : data(data){
    dataLen = 1;
    for(auto d : dims){
        dataLen *= d;
    }
}


TensorContents::TensorContents(vDims dims, evalArgs argv, evalFunc method)
    : contents(std::in_place_type<TensorFunction>, argv, method), dims(dims){}

TensorContents::TensorContents(vDims dims, std::vector<double> data)
    : contents(std::in_place_type<TensorData>, dims, data), dims(dims){}

