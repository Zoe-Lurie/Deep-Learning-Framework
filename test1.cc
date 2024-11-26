#include <vector>
#include <iostream>

#include "tensor.h"

int main(){

    auto x = Tensor({2,3}, {1,2,3,3,2,1}, true);
    auto y = Tensor({2,3}, {3,2,1,1,2,3}, true);
    auto z = Tensor({3,2}, {0,1,2,3,4,5}, true);

    auto L = (x - y).pow(3).reduceSum();
    L.print();

    L.backward();

    std::cout << "\n";
    x.getGradient().print();
    std::cout << "\n";
    y.getGradient().print();

    return 0;
}

