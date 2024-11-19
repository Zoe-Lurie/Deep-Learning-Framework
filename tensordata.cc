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

