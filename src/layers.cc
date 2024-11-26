/**
 * @file layers.cc
 * @brief Implements layers methods.
 * 
 * @author Zoe Lurie
 * @date November 2024
 */

#include "layers.h"
#include "tensor.h"

Tensor Layers::singleLinearSoftmax(Tensor input, size_t inputSize, size_t outputSize){
    auto weight = Tensor::fillRandom({inputSize, outputSize}, 0, 0.1);
    auto bias = Tensor::fillRandom({outputSize}, 0, 0.1);
    auto probs = (input.matmul(weight) + bias).softmax();
    return probs;
}

Tensor Layers::singleLinearRelu(Tensor input, size_t inputSize, size_t outputSize){
    auto weight = Tensor::fillRandom({inputSize, outputSize}, 0, 0.1);
    auto bias = Tensor::fillRandom({outputSize}, 0, 0.1);
    auto probs = (input.matmul(weight) + bias).relu();
    return probs;
}

Tensor Layers::multiLayer(Tensor input, size_t inputSize, size_t outputSize, std::vector<size_t> intermediateSizes){
    input = singleLinearRelu(input, inputSize, intermediateSizes[0]);

    for(size_t i = 1; i < intermediateSizes.size(); ++i){
        input = singleLinearRelu(input, intermediateSizes[i-1], intermediateSizes[i]);
    }

    auto probs = singleLinearSoftmax(input, intermediateSizes[intermediateSizes.size()-1], outputSize);
    return probs;
}

