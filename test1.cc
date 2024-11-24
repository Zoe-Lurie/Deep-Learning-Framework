#include <vector>
#include <iostream>

#include "tensor.h"


int main(){

    auto x = Tensor({2,3}, {1,2,3,3,2,1}, true);
    auto y = Tensor({2,3}, {3,2,1,1,2,3}, true);

    auto L = x.subtract(y, true).pow(3, true).reduceSum(true);

    L.backward();

    x.getGradient().print();
    std::cout << "\n\n";
    y.getGradient().print();

    /*
    std::vector<double> data = {0,.1,.2,.3,.4};
    std::vector<double> data2 = {1,2,3,4,5};

    std::vector<double> data3 = {1,2,3,4,5,6};
    std::vector<double> data4 = {7,8,9,0,1,2};

    auto t1 = Tensor({5}, data2);
    auto t2 = Tensor::zeroes({5});
    auto t3 = t1.neg();
    auto t4 = t3 + t2;
    auto t5 = t4 + 10;
    auto t6 = t5.elementwiseMult(t1);
    t6 = t5 - 0.5;
    t6 = (2 * t6 - 12).reciprocal();
    t6 = t6 / 2;

    auto t7 = Tensor({2,3}, data3);
    auto t8 = Tensor({3,2}, data4);
    auto t9 = t7.matmul(t8);

    t9.print();
    */

    return 0;
}

