#include <vector>

#include "tensor.h"


int main(){

    std::vector<double> data = {0,.1,.2,.3,.4};
    std::vector<double> data2 = {1,2,3,4,5};

    auto t1 = Tensor({5}, data2);
    auto t2 = Tensor::zeroes({5});
    auto t3 = t1.neg();
    auto t4 = t3 + t2;
    auto t5 = t4 + 10;
    auto t6 = t5.elementwiseMult(t1);
    t6 = t5 - 0.5;
    t6 = 2 * t6;

    t6.print();

    return 0;
}

