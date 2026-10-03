#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void registerAdaptive(py::module_& m);
