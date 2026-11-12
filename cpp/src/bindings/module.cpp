// Entry point of photographiq._native. Each component registers its own submodule.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "photographiq/core/types.hpp"

namespace py = pybind11;

#if PGQ_WITH_GAUSSIAN
void bind_gaussian(py::module_& m);
#endif
#if PGQ_WITH_FOCK
void bind_fock(py::module_& m);
#endif
#if PGQ_WITH_SAMPLING
void bind_sampling(py::module_& m);
#endif

PYBIND11_MODULE(_native, m) {
  m.doc() = "Native C++20 engines of PhotoGraphiQ (hbar = 2, interleaved quadratures).";
  std::vector<std::string> components;
#if PGQ_WITH_GAUSSIAN
  components.emplace_back("gaussian");
  auto gaussian = m.def_submodule("gaussian", "Covariance-matrix Gaussian engine");
  bind_gaussian(gaussian);
#endif
#if PGQ_WITH_FOCK
  components.emplace_back("fock");
  auto fock = m.def_submodule("fock", "Truncated Fock-space pure and mixed state kernels");
  bind_fock(fock);
#endif
#if PGQ_WITH_SAMPLING
  components.emplace_back("sampling");
  auto sampling = m.def_submodule("sampling", "Permanent, hafnian and torontonian routines");
  bind_sampling(sampling);
#endif
  m.attr("components") = components;
  py::dict info;
  info["compiler"] = PGQ_COMPILER_ID;
  info["build_type"] = PGQ_BUILD_TYPE;
  info["cxx_standard"] = __cplusplus;
#if PGQ_HAVE_OPENMP
  info["openmp"] = true;
#else
  info["openmp"] = false;
#endif
  info["eigen"] = std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) +
                  "." + std::to_string(EIGEN_MINOR_VERSION);
  info["hbar"] = photographiq::kHbar;
  m.attr("build_info") = info;
}
