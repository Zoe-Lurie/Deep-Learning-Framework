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

        .def("add", py::overload_cast<Tensor, bool>(&Tensor::add))
        .def("add", py::overload_cast<double, bool>(&Tensor::add))
        .def(py::self + py::self)
        .def(py::self + double())
        .def(double() + py::self)

        .def("subtract", py::overload_cast<Tensor, bool>(&Tensor::subtract))
        .def("subtract", py::overload_cast<double, bool>(&Tensor::subtract))
        .def(py::self - py::self)
        .def(py::self - double())
        .def(double() - py::self)

        .def("elementwiseMult", py::overload_cast<Tensor, bool>(&Tensor::elementwiseMult))
        .def("elementwiseMult", py::overload_cast<double, bool>(&Tensor::elementwiseMult))
        .def(py::self * py::self)
        .def(py::self * double())
        .def(double() * py::self)


        .def("neg", &Tensor::neg)
        .def("__neg__", [](Tensor a) {return a.neg();}, py::is_operator())
        .def("pow", &Tensor::pow)
        .def("__pow__", [](Tensor a, double b) {return a.pow(b);}, py::is_operator())
        .def("relu", &Tensor::relu)
        .def("binarize", &Tensor::binarize)

        .def("matmul", &Tensor::matmul)
        .def("__matmul__", [](Tensor a, Tensor b) {return a.matmul(b);}, py::is_operator())

        .def("reduceSum", &Tensor::reduceSum)
        ;
}

