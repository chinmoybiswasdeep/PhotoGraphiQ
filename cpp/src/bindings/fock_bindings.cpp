// Python bindings of the truncated Fock engine (photographiq._native.fock).
//
// States are NumPy complex128 arrays in FockBasis order: pure states are 1-D vectors,
// density matrices are C-contiguous square arrays. apply_*/phase_* update them in place
// (they must be writeable and contiguous; pybind11 refuses copies for in-place refs).
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "photographiq/fock/basis.hpp"
#include "photographiq/fock/gates.hpp"
#include "photographiq/fock/kernels.hpp"

namespace py = pybind11;
using namespace photographiq;
using namespace photographiq::fock;

namespace {

using Modes = std::vector<int>;
using Release = py::call_guard<py::gil_scoped_release>;

py::array_t<Occupation> basis_table(int modes, int cutoff) {
  const FockBasis basis(modes, cutoff);
  py::array_t<Occupation> out({static_cast<py::ssize_t>(basis.size()), static_cast<py::ssize_t>(modes)});
  std::copy(basis.table().begin(), basis.table().end(), out.mutable_data());
  return out;
}

py::array_t<std::int64_t> rank(const py::array_t<Occupation, py::array::c_style | py::array::forcecast>& occupations,
                               int cutoff) {
  if (occupations.ndim() != 2) throw PhysicsError("Occupations must be a 2-D (states x modes) array");
  const int modes = static_cast<int>(occupations.shape(1));
  const py::ssize_t count = occupations.shape(0);
  const FockBasis basis(modes, cutoff);
  py::array_t<std::int64_t> out(count);
  auto ranks = out.mutable_unchecked<1>();
  const Occupation* data = occupations.data();
  for (py::ssize_t i = 0; i < count; ++i) {
    ranks(i) = basis.rank(std::span<const Occupation>(data + i * modes, static_cast<std::size_t>(modes)));
  }
  return out;
}

}  // namespace

// PhysicsError derives from std::invalid_argument, which pybind11 raises as ValueError.
void bind_fock(py::module_& m) {
  m.def("dimension", &dimension, py::arg("modes"), py::arg("cutoff"),
        "Number of occupations with total photon number < cutoff, C(modes + cutoff - 1, modes).");
  m.def("basis", &basis_table, py::arg("modes"), py::arg("cutoff"),
        "Occupation table (dimension x modes, int32) in Piquasso order: sectors of total photon "
        "number, first mode descending inside a sector.");
  m.def("rank", &rank, py::arg("occupations"), py::arg("cutoff"),
        "Basis indices of occupation rows; ValueError outside the truncated space.");

  m.def("gaussian_unitary", &gaussian_unitary, py::arg("symplectic"), py::arg("displacement"),
        py::arg("cutoff"), Release(),
        "Exact <m|U|n> of the Gaussian unitary with U^dag x U = S x + d (hbar = 2, interleaved), "
        "Bargmann recurrence. Vacuum amplitude of the displacement-free part is real positive.");
  m.def(
      "cubic_phase_projected",
      [](Real gamma, int cutoff, Real tolerance, Index max_points) {
        CubicPhaseMatrix result;
        {
          py::gil_scoped_release release;
          result = cubic_phase_projected(gamma, cutoff, tolerance, max_points);
        }
        return py::make_tuple(result.matrix, result.points, result.change);
      },
      py::arg("gamma"), py::arg("cutoff"), py::arg("tolerance") = 1e-13,
      py::arg("max_points") = Index{1} << 22,
      "(P exp(i gamma q^3/6) P, quadrature samples, last change) by a converged nested "
      "trapezoidal rule on the Hermite-function support.");
  m.def("cubic_phase_truncated_generator", &cubic_phase_truncated_generator, py::arg("gamma"),
        py::arg("cutoff"), Release(), "Piquasso's expm(i gamma Q^3/6) with Q = P q P (cutoff x cutoff).");

  m.def(
      "apply_pure",
      [](StateRef state, int modes, int cutoff, const Modes& local, const OperatorRef& op) {
        apply_local(state, FockBasis(modes, cutoff), local, op);
      },
      py::arg("state").noconvert(), py::arg("modes"), py::arg("cutoff"), py::arg("local"), py::arg("op"),
      Release(), "In place psi -> P U P psi for a local operator on the ordered modes.");
  m.def(
      "apply_mixed",
      [](DensityRef rho, int modes, int cutoff, const Modes& local, const OperatorRef& op) {
        apply_local(rho, FockBasis(modes, cutoff), local, op);
      },
      py::arg("rho").noconvert(), py::arg("modes"), py::arg("cutoff"), py::arg("local"), py::arg("op"),
      Release(), "In place rho -> (P U P) rho (P U P)^dag.");
  m.def(
      "phase_pure",
      [](StateRef state, int modes, int cutoff, const Modes& local, const RealVector& phases) {
        apply_local_phase(state, FockBasis(modes, cutoff), local, phases);
      },
      py::arg("state").noconvert(), py::arg("modes"), py::arg("cutoff"), py::arg("local"), py::arg("phases"),
      Release(), "In place diagonal gate exp(i theta[local rank]).");
  m.def(
      "phase_mixed",
      [](DensityRef rho, int modes, int cutoff, const Modes& local, const RealVector& phases) {
        apply_local_phase(rho, FockBasis(modes, cutoff), local, phases);
      },
      py::arg("rho").noconvert(), py::arg("modes"), py::arg("cutoff"), py::arg("local"), py::arg("phases"),
      Release(), "In place rho -> D rho D^dag for D = exp(i theta[local rank]).");

  m.def(
      "reduced_pure",
      [](ConstStateRef state, int modes, int cutoff, const Modes& keep) {
        return reduced_density(state, FockBasis(modes, cutoff), keep);
      },
      py::arg("state"), py::arg("modes"), py::arg("cutoff"), py::arg("keep"), Release());
  m.def(
      "reduced_mixed",
      [](ConstDensityRef rho, int modes, int cutoff, const Modes& keep) {
        return reduced_density(rho, FockBasis(modes, cutoff), keep);
      },
      py::arg("rho"), py::arg("modes"), py::arg("cutoff"), py::arg("keep"), Release());
  m.def(
      "marginal_pure",
      [](ConstStateRef state, int modes, int cutoff, const Modes& keep) {
        return photon_marginal(state, FockBasis(modes, cutoff), keep);
      },
      py::arg("state"), py::arg("modes"), py::arg("cutoff"), py::arg("keep"), Release());
  m.def(
      "marginal_mixed",
      [](ConstDensityRef rho, int modes, int cutoff, const Modes& keep) {
        return photon_marginal(rho, FockBasis(modes, cutoff), keep);
      },
      py::arg("rho"), py::arg("modes"), py::arg("cutoff"), py::arg("keep"), Release());
  m.def(
      "thermal_loss",
      [](ConstDensityRef rho, int modes, int cutoff, int mode, Real transmissivity, Real thermal_photons) {
        return thermal_loss(rho, FockBasis(modes, cutoff), mode, transmissivity, thermal_photons);
      },
      py::arg("rho"), py::arg("modes"), py::arg("cutoff"), py::arg("mode"), py::arg("transmissivity"),
      py::arg("thermal_photons") = 0.0, Release(),
      "Thermal attenuator as Kraus maps (pure loss then quantum-limited amplifier), returns a new matrix.");
  m.def(
      "physicality",
      [](ConstDensityRef rho) {
        Physicality p;
        {
          py::gil_scoped_release release;
          p = physicality(rho);
        }
        py::dict out;
        out["hermiticity"] = p.hermiticity;
        out["trace"] = p.trace;
        out["min_eigenvalue"] = p.min_eigenvalue;
        out["max_eigenvalue"] = p.max_eigenvalue;
        return out;
      },
      py::arg("rho"), "Hermiticity residual, trace and extreme eigenvalues of a density matrix.");
  m.def("max_threads", &max_threads, "OpenMP threads available to the kernels (1 without OpenMP).");
}
