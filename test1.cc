#include <vector>
#include <iostream>

#include "tensor.h"

int main(){

    auto x = Tensor({2,3}, {1,2,3,3,2,1}, true, GPU);
    auto y = Tensor({2,3}, {3,2,1,1,2,3}, true, GPU);

    auto L = (x - y).pow(3).reduceSum();
    L.print();

    L.backward();

    std::cout << "\n";
    x.getGradient().print();
    std::cout << "\n";
    y.getGradient().print();

    return 0;
}

