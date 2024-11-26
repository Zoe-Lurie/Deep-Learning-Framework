#include <vector>
#include <iostream>

#include "tensor.h"

#define OP(x, y) (x - y).pow(3).reduceSum();

int main(){

    Tensor::setOmpNumThreads(8);

    std::cout << "CPU output:\n";

    auto cx = Tensor({2,3}, {1,2,3,3,2,1}, true, CPU);
    auto cy = Tensor({2,3}, {3,2,1,1,2,3}, true, CPU);

    auto cL = OP(cx, cy);
    cL.print();

    //cL.backward(Tensor::ones({2,3}, CPU));
    cL.backward();

    std::cout << "\n";
    cx.getGradient().print();
    std::cout << "\n";
    cy.getGradient().print();

/*
    std::cout << "GPU output:\n";

    auto x = Tensor({2,3}, {1,2,3,3,2,1}, true, GPU);
    auto y = Tensor({2,3}, {3,2,1,1,2,3}, true, GPU);

    auto L = OP(x, y);
    L.print();

    L.backward(Tensor::ones({2,3}, GPU));

    std::cout << "\n";
    x.getGradient().print();
    std::cout << "\n";
    //y.getGradient().print();
    */

    return 0;
}

