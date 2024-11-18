#include <vector>

#include "tensor.h"

void Tensor::eval(){
    if(contents->isFunc()){
        TensorFunction func = contents->getFunc();
        for(auto& arg : func.getArgv()){
            if(arg.type() == typeid(tEvalArg)){
                auto t = std::any_cast<tEvalArg>(arg);
                t.eval();
            }
        }

        TensorData data = func.callMethod();
        contents->changeToData(data);
    }
}

TensorData Tensor::evalNeg(evalArgs argv){
    auto x = std::any_cast<tEvalArg>(argv[0]);
    auto data = x.contents->getData().data;

    TensorData retData = x.contents->getDims();

    for(size_t i = 0; i < retData.getDataLen(); ++i){
        retData.data[i] = -data[i];
    }
    return retData;
}

TensorData Tensor::evalAdd(evalArgs argv){
    auto x = std::any_cast<tEvalArg>(argv[0]);
    auto xData = x.contents->getData().data;

    auto y = std::any_cast<tEvalArg>(argv[1]);
    auto yData = y.contents->getData().data;

    TensorData retData(x.contents->getDims());

    for(size_t i = 0; i < retData.dataLen; ++i){
        retData.data[i] = xData[i] + yData[i];
    }
    return retData;
}

TensorData Tensor::evalZeroes(evalArgs argv){
    vDims dims = std::any_cast<vDims>(argv[0]);
    TensorData retData(dims);

    for(size_t i = 0; i < retData.dataLen; ++i){
        retData.data[i] = 0;
    }
    return retData;
}

TensorData Tensor::evalSoftmax(evalArgs argv){
    auto x = std::any_cast<tEvalArg>(argv[0]);
    auto data = x.contents->getData().data;

    TensorData retData(x.contents->getDims());

    double sum = 0;
    for(size_t i = 0; i < retData.dataLen; ++i){
        sum += data[i];
    }

    for(size_t i = 0; i < retData.dataLen; ++i){
        retData.data[i] = data[i] / sum;
    }
    return retData;
}

TensorData Tensor::evalAddScalar(evalArgs argv){
    auto x = std::any_cast<tEvalArg>(argv[0]);
    auto data = x.contents->getData().data;

    double n = std::any_cast<double>(argv[1]);

    TensorData retData(x.contents->getDims());

    for(size_t i = 0; i < retData.dataLen; ++i){
        retData.data[i] = data[i] + n;
    }
    return retData;
}

