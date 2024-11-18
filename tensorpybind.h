#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(tensor, m){
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<vDims, std::vector<double>>())
        .def_static("zeroes", &Tensor::zeroes)
        .def("neg", &Tensor::neg) ;
}

