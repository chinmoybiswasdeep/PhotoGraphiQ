// Python bindings of photographiq::gaussian (submodule photographiq._native.gaussian).
//
// In-place kernels take writable float64 arrays without copying: means must be
// contiguous 1-D, covariances Fortran-contiguous (NumPy order='F'); anything else is
// rejected with TypeError instead of being silently copied. PhysicsError -> ValueError.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "photographiq/gaussian/batch.hpp"
#include "photographiq/gaussian/conditioning.hpp"
#include "photographiq/gaussian/diagnostics.hpp"
#include "photographiq/gaussian/operations.hpp"

namespace py = pybind11;
namespace g = photographiq::gaussian;
using photographiq::Index;
using photographiq::Real;
using photographiq::RealMatrix;
using photographiq::RealVector;

namespace {

py::tuple moments_tuple(g::Moments moments) {
  return py::make_tuple(std::move(moments.mean), std::move(moments.covariance));
}

py::tuple conditioned_tuple(g::Conditioned out) {
  return py::make_tuple(std::move(out.mean), std::move(out.covariance), out.density);
}

}  // namespace

void bind_gaussian(py::module_& m) {
  using Release = py::call_guard<py::gil_scoped_release>;
  m.attr("max_threads") = g::max_threads();

  m.def("apply_symplectic", &g::apply_symplectic, py::arg("mean").noconvert(),
        py::arg("cov").noconvert(), py::arg("modes"), py::arg("matrix"), Release(),
        "In place: V -> S V S^T, mean -> S mean on the listed modes (O(n k)).");
  m.def("apply_channel", &g::apply_channel, py::arg("mean").noconvert(),
        py::arg("cov").noconvert(), py::arg("modes"), py::arg("x"), py::arg("y"),
        py::arg("tolerance") = 1e-10, Release(),
        "In place: V -> X V X^T + Y, mean -> X mean; rejects non-CP channels.");
  m.def("apply_loss", &g::apply_loss, py::arg("mean").noconvert(), py::arg("cov").noconvert(),
        py::arg("mode"), py::arg("transmissivity"), py::arg("thermal_photons") = 0.0, Release(),
        "In place: V -> eta V + (1-eta)(2 n_th + 1) I, mean -> sqrt(eta) mean.");
  m.def("displace", &g::displace, py::arg("mean").noconvert(), py::arg("mode"), py::arg("q"),
        py::arg("p"), "In place: mean[2m] += q, mean[2m+1] += p.");
  m.def(
      "append_mode",
      [](const RealVector& mean, const RealMatrix& cov, const Eigen::Vector2d& mode_mean,
         const Eigen::Matrix2d& mode_cov) {
        return moments_tuple(g::append_mode(mean, cov, mode_mean, mode_cov));
      },
      py::arg("mean"), py::arg("cov"), py::arg("mode_mean"), py::arg("mode_cov"),
      "Block-diagonal extension by one mode; returns (mean, cov).");
  m.def(
      "remove_mode",
      [](const RealVector& mean, const RealMatrix& cov, Index mode) {
        return moments_tuple(g::remove_mode(mean, cov, mode));
      },
      py::arg("mean"), py::arg("cov"), py::arg("mode"), "Partial trace; returns (mean, cov).");

  m.def(
      "homodyne_moments",
      [](g::ConstMeanRef mean, g::ConstCovRef cov, Index mode, Real angle, Real efficiency,
         Real noise) {
        const g::HomodyneMoments out =
            g::homodyne_moments(mean, cov, mode, angle, efficiency, noise);
        return py::make_tuple(out.expected, out.variance);
      },
      py::arg("mean"), py::arg("cov"), py::arg("mode"), py::arg("angle"),
      py::arg("efficiency") = 1.0, py::arg("noise") = 0.0,
      "(expected, variance) of the calibrated homodyne record.");
  m.def(
      "condition_homodyne",
      [](g::ConstMeanRef mean, g::ConstCovRef cov, Index mode, Real angle, Real efficiency,
         Real noise, Real outcome) {
        g::Conditioned out;
        {
          py::gil_scoped_release release;
          out = g::condition_homodyne(mean, cov, mode, angle, efficiency, noise, outcome);
        }
        return conditioned_tuple(std::move(out));
      },
      py::arg("mean"), py::arg("cov"), py::arg("mode"), py::arg("angle"), py::arg("efficiency"),
      py::arg("noise"), py::arg("outcome"),
      "Condition on a given homodyne outcome; returns (mean, cov, density).");
  m.def(
      "generaldyne_moments",
      [](g::ConstMeanRef mean, g::ConstCovRef cov, Index mode, const Eigen::Matrix2d& detector) {
        g::PhaseSpaceMoments out = g::generaldyne_moments(mean, cov, mode, detector);
        return py::make_tuple(RealVector(out.expected), RealMatrix(out.covariance));
      },
      py::arg("mean"), py::arg("cov"), py::arg("mode"), py::arg("detector"),
      "(expected, covariance) of the (q, p) record; detector = I is heterodyne.");
  m.def(
      "condition_generaldyne",
      [](g::ConstMeanRef mean, g::ConstCovRef cov, Index mode, const Eigen::Matrix2d& detector,
         const Eigen::Vector2d& outcome) {
        g::Conditioned out;
        {
          py::gil_scoped_release release;
          out = g::condition_generaldyne(mean, cov, mode, detector, outcome);
        }
        return conditioned_tuple(std::move(out));
      },
      py::arg("mean"), py::arg("cov"), py::arg("mode"), py::arg("detector"), py::arg("outcome"),
      "Condition on a given (q, p) outcome; returns (mean, cov, density).");

  m.def("symplectic_eigenvalues", &g::symplectic_eigenvalues, py::arg("cov"), Release());
  m.def("purity", &g::purity, py::arg("cov"), Release());
  m.def("entropy", &g::entropy, py::arg("cov"), py::arg("tolerance") = 1e-7, Release());
  m.def("log_negativity", &g::log_negativity, py::arg("cov"), py::arg("modes"), Release());
  m.def("overlap", &g::overlap, py::arg("mean1"), py::arg("cov1"), py::arg("mean2"),
        py::arg("cov2"), Release());
  m.def("fidelity", &g::fidelity, py::arg("mean1"), py::arg("cov1"), py::arg("mean2"),
        py::arg("cov2"), py::arg("pure_tolerance") = 1e-9, Release());

  m.def("batch_apply_symplectic", &g::batch_apply_symplectic, py::arg("means").noconvert(),
        py::arg("covariances").noconvert(), py::arg("modes"), py::arg("matrices"), Release(),
        "In place over B states: means (B, 2n), covariances (B, 4n^2), matrices (B, 4k^2).");
  m.def(
      "batch_homodyne_moments",
      [](g::ConstBatchRef means, g::ConstBatchRef covariances, Index mode,
         const RealVector& angles, Real efficiency, Real noise) {
        RealVector expected(means.rows());
        RealVector variance(means.rows());
        {
          py::gil_scoped_release release;
          g::batch_homodyne_moments(means, covariances, mode, angles, efficiency, noise,
                                    expected, variance);
        }
        return py::make_tuple(std::move(expected), std::move(variance));
      },
      py::arg("means"), py::arg("covariances"), py::arg("mode"), py::arg("angles"),
      py::arg("efficiency"), py::arg("noise"));
  m.def(
      "batch_condition_homodyne",
      [](g::ConstBatchRef means, g::ConstBatchRef covariances, Index mode,
         const RealVector& angles, Real efficiency, Real noise, const RealVector& outcomes) {
        const Index batch = means.rows();
        const Index out_dim = std::max<Index>(means.cols() - 2, 0);
        g::RowMatrix out_means(batch, out_dim);
        g::RowMatrix out_covs(batch, out_dim * out_dim);
        RealVector densities(batch);
        {
          py::gil_scoped_release release;
          g::batch_condition_homodyne(means, covariances, mode, angles, efficiency, noise,
                                      outcomes, out_means, out_covs, densities);
        }
        return py::make_tuple(std::move(out_means), std::move(out_covs), std::move(densities));
      },
      py::arg("means"), py::arg("covariances"), py::arg("mode"), py::arg("angles"),
      py::arg("efficiency"), py::arg("noise"), py::arg("outcomes"));
}
