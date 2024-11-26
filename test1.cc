#include <vector>
#include <iostream>

#include "tensor.h"


int main(){

    auto x = Tensor({2,3}, {1,2,3,3,2,1}, true);
    auto y = Tensor({2,3}, {3,2,1,1,2,3}, true);
    auto z = Tensor({3,2}, {0,1,2,3,4,5}, true);

    auto L = x.subtract(y).pow(3).relu().subtract(x).elementwiseMult(y).matmul(x.transpose()).reduceSum();

    L.backward();

    L.print();
    std::cout << "\n";
    x.getGradient().print();
    std::cout << "\n";
    y.getGradient().print();

    return 0;
}

