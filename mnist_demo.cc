#include <vector>
#include <iostream>

#include "tensor.h"

int main(){
    //Tensor::setOmpNumThreads(8);

    std::string filename = "data/mnist.t";






    size_t num_features = 200;
    size_t num_classes = 10;
    Tensor weights = Tensor::fillRandom({num_classes, num_features}, 0, 0.1, true);


    return 0;
}

