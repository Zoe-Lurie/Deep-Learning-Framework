/**
 * @file layers.h
 * @brief Defines layers methods.
 * 
 * @author Zoe Lurie
 * @date November 2024
 */

#include "tensor.h"

namespace Layers{

    Tensor singleLinearSoftmax(Tensor input, size_t inputSize, size_t outputSize);
    Tensor singleLinearRelu(Tensor input, size_t inputSize, size_t outputSize);
    Tensor multiLayer(Tensor input, size_t inputSize, size_t outputSize, std::vector<size_t> intermediateSizes);
}

