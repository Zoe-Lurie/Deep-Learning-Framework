#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(tensor, m){
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<vDims, std::vector<double>, bool>())
        .def("print", &Tensor::print)
        .def("getData", &Tensor::getData)
        .def("getDims", &Tensor::getDims)

        .def("backward", &Tensor::backward)
        .def("getGradient", &Tensor::getGradient)

        .def_static("ones", &Tensor::ones)
        .def_static("zeroes", &Tensor::zeroes)
        .def_static("fill", &Tensor::fill)

        .def("reshape", &Tensor::reshape)
        .def("transpose", &Tensor::transpose)

        .def("add", &Tensor::add)
        .def(py::self + py::self)
        .def(py::self + double())
        .def(double() + py::self)

        .add("subtract", &Tensor::subtract)
        .def(py::self - py::self)
        .def(py::self - double())
        .def(double() - py::self)

        .add("elementwiseMult", &Tensor::elementwiseMult)
        .def(py::self * py::self)
        .def(py::self * double())
        .def(double() * py::self)


        .def("neg", &Tensor::neg)
        .def("pow", &Tensor::pow)
        .def("relu", &Tensor::relu)
        .def("binarize", &Tensor::binarize)

        .def("matmul", &Tensor::matmul)

        .def("reduceSum", &Tensor::reduceSum)
        ;
}

