// Python bindings of photographiq::sampling (submodule photographiq._native.sampling).
//
// Inputs are converted (copied) to complex128 column-major Eigen matrices; the GIL is
// released during evaluation. std::invalid_argument -> ValueError.
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "photographiq/sampling/matrix_functions.hpp"

namespace py = pybind11;
namespace s = photographiq::sampling;

void bind_sampling(py::module_& m) {
  using Release = py::call_guard<py::gil_scoped_release>;
  m.def("permanent", &s::permanent, py::arg("matrix"), Release(),
        "Permanent of a square complex matrix (Gray-code Ryser, O(2^n n)).");
  m.def("permanent_repeated", &s::permanent_repeated, py::arg("matrix"),
        py::arg("row_multiplicity"), py::arg("column_multiplicity"), Release(),
        "Permanent of the matrix with row i repeated r_i and column j repeated c_j times "
        "(multiplicity-aware Ryser, prod(c_j + 1) terms).");
  m.def("hafnian", &s::hafnian, py::arg("matrix"), Release(),
        "Hafnian of a symmetric matrix (Glynn power traces, O(n^4 2^n) for 2n x 2n).");
  m.def("loop_hafnian", &s::loop_hafnian, py::arg("matrix"), Release(),
        "Loop hafnian of a symmetric matrix; the diagonal holds the loop weights.");
  m.def("torontonian", &s::torontonian, py::arg("matrix"), Release(),
        "Torontonian of a 2n x 2n matrix in block (a, a^dag) ordering.");
  m.def("loop_torontonian", &s::loop_torontonian, py::arg("matrix"), py::arg("gamma"), Release(),
        "Loop torontonian of a 2n x 2n matrix and 2n vector in block (a, a^dag) ordering.");
  m.def("threshold_vacuum_sum", &s::threshold_vacuum_sum, py::arg("excess"), py::arg("shift"),
        Release(),
        "Threshold inclusion-exclusion sum over vacuum projections (real deviation form); "
        "returns (sum, sum of |terms|).");
  m.def("max_threads", &s::max_threads, "OpenMP threads used by the sampling kernels.");
  m.def("set_num_threads", &s::set_num_threads, py::arg("threads"),
        "Set the OpenMP thread count (process wide).");
}
