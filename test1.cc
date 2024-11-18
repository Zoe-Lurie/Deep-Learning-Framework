#include <vector>

#include "tensor.h"


int main(){

    std::vector<double> data = {0,.1,.2,.3,.4};
    std::vector<double> data2 = {1,2,3,4,5};

    Tensor a = Tensor({5}, data);
    Tensor e = Tensor({5}, data2);
    Tensor b = Tensor::zeroes({5});
    Tensor c = a.neg();
    Tensor d = e + e;
    Tensor f = e + c;
    Tensor g = e.softmax();
    Tensor h = 50 + g;
    h = h + h + 3 + h;

    h.print();

    return 0;
}

