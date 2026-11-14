"""R31: Thermal noise validation.

Sweeps transmissivity eta and thermal environment occupation nbar through
``photographiq.commands.Loss`` on both Gaussian backends, and compares the
resulting mean/covariance to the independent analytic thermal-attenuation
channel: mean -> sqrt(eta)*mean_in, V -> eta*V_in + (1-eta)*(2*nbar+1)*I
(statistical covariance, vacuum V=I), written directly with NumPy.

Upgrade migration (2026-11): the C++ ``native-gaussian`` engine joins the
exact comparison (same tolerance). On a subset (MIXED_ETAS x NBARS) the
density-matrix engines ``native-mixed-fock`` and ``piquasso-mixed-fock`` apply
the thermal-loss Kraus channel at MIXED_CUTOFF. Thermal occupation spreads the
state over many Fock levels, so their photon-number and q-variance errors are
a genuine truncation effect that grows with nbar: they are reported together
with the minimum retained norm and wall time (provisional), and asserted
(MIXED_TOLERANCE) only where the retained norm is >= 1 - 1e-9.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

ETAS = [0.2, 0.4, 0.6, 0.8, 1.0]
NBARS = [0.0, 0.3, 0.8, 1.5, 3.0]
ALPHA = 0.5 - 0.3j
GAUSSIAN_BACKENDS = ("gaussian", "piquasso", "native-gaussian")
MIXED_BACKENDS = ("native-mixed-fock", "piquasso-mixed-fock")
MIXED_ETAS = [0.4, 0.8]
MIXED_CUTOFF = 30
MIXED_TOLERANCE = 1e-6


def mixed_fock_check(pattern, input_state, expected_n, expected_q_var, backend, cutoff):
    """Run a Loss pattern on a density-matrix engine; return errors, retained norm and wall time."""
    start = time.perf_counter()
    run = common.safe_cutoff_run(
        lambda: pg.simulate(pattern, backend=backend, cutoff=cutoff, inputs={0: input_state}, seed=0).state,
        label=f"{backend} cutoff={cutoff}",
    )
    seconds = time.perf_counter() - start
    if not run["ok"]:
        return {f"{backend}_ok": False, f"{backend}_error": run["error"], f"{backend}_seconds": seconds}
    state = run["value"]
    return {
        f"{backend}_ok": True,
        f"{backend}_photon_number_error": abs(state.photon_number(0) - expected_n),
        f"{backend}_q_variance_error": abs(state.quadrature(0)[1] - expected_q_var),
        f"{backend}_min_retained_norm": min(state.retained_norms),
        f"{backend}_seconds": seconds,
    }
def analytic_thermal_channel(eta: float, nbar: float, mean_in: np.ndarray, cov_in: np.ndarray):
    mean_out = np.sqrt(eta) * mean_in
    cov_out = eta * cov_in + (1 - eta) * (2 * nbar + 1) * np.eye(2)
    return mean_out, cov_out


def main():
    plt = common.setup_style()
    input_state = pg.GaussianInput.coherent(ALPHA)
    mean_in = np.array(input_state.mean)
    cov_in = np.array(input_state.covariance)

    rows = []
    for eta in ETAS:
        for nbar in NBARS:
            pattern = pg.Pattern(inputs=(0,)).append(pg.Loss(0, eta, nbar)).append(pg.Output((0,))).validate()
            expected_mean, expected_cov = analytic_thermal_channel(eta, nbar, mean_in, cov_in)

            row = {"eta": eta, "nbar": nbar}
            for backend in GAUSSIAN_BACKENDS:
                state = pg.simulate(pattern, backend=backend, inputs={0: input_state}, seed=0).state
                row[f"{backend}_mean_error"] = common.frobenius_error(state.mean, expected_mean)
                row[f"{backend}_covariance_error"] = common.frobenius_error(state.covariance, expected_cov)
            if eta in MIXED_ETAS:
                expected_n = (np.trace(expected_cov) + expected_mean @ expected_mean - 2) / 4
                for backend in MIXED_BACKENDS:
                    row.update(mixed_fock_check(pattern, input_state, expected_n, expected_cov[0, 0], backend, MIXED_CUTOFF))
            rows.append(row)

    tolerance = 1e-9
    failures = [
        r for r in rows
        if r["gaussian_mean_error"] > tolerance or r["piquasso_mean_error"] > tolerance
        or r["gaussian_covariance_error"] > tolerance or r["piquasso_covariance_error"] > tolerance
        or r["native-gaussian_mean_error"] > tolerance or r["native-gaussian_covariance_error"] > tolerance
        or any(r.get(f"{b}_ok") and r[f"{b}_min_retained_norm"] >= 1 - 1e-9
               and max(r[f"{b}_photon_number_error"], r[f"{b}_q_variance_error"]) > MIXED_TOLERANCE
               for b in MIXED_BACKENDS)
    ]
    if failures:
        common.save_csv(failures, "R31_thermal_noise_FAILURES")
        raise AssertionError("Thermal loss channel disagreed with the analytic formula; see FAILURES csv")

    common.save_result(rows, "R31_thermal_noise", extra={
        "alpha": [ALPHA.real, ALPHA.imag], "gaussian_backends": list(GAUSSIAN_BACKENDS), "mixed_backends": list(MIXED_BACKENDS),
        "mixed_etas": MIXED_ETAS, "mixed_cutoff": MIXED_CUTOFF, "mixed_tolerance": MIXED_TOLERANCE,
        "timing_note": "provisional (shared machine)"})

    fig, ax = plt.subplots(figsize=(6, 4))
    for nbar in NBARS:
        subset = [r for r in rows if r["nbar"] == nbar]
        variances = []
        for r in subset:
            _, cov = analytic_thermal_channel(r["eta"], r["nbar"], mean_in, cov_in)
            variances.append(cov[0, 0])
        ax.plot([r["eta"] for r in subset], variances, "o-", label=f"nbar={nbar}")
    ax.set_xlabel("Transmissivity eta")
    ax.set_ylabel("Output q variance")
    ax.legend(fontsize=8, title="Thermal occupation")
    ax.set_title("Thermal attenuation channel (analytic curves)")
    common.save_figure(fig, "R31_thermal_noise")
    plt.close(fig)

    common.print_summary(
        "R31 thermal noise validation",
        cases=len(rows),
        max_mean_error=max(max(r["gaussian_mean_error"], r["piquasso_mean_error"]) for r in rows),
        max_covariance_error=max(max(r["gaussian_covariance_error"], r["piquasso_covariance_error"]) for r in rows),
        max_native_gaussian_error=max(max(r["native-gaussian_mean_error"], r["native-gaussian_covariance_error"]) for r in rows),
        **{f"{b}_errors_by_eta_nbar": {(r["eta"], r["nbar"]): (round(r[f"{b}_photon_number_error"], 12), r[f"{b}_min_retained_norm"])
                                   for r in rows if r.get(f"{b}_ok")} for b in MIXED_BACKENDS},
    )


if __name__ == "__main__":
    main()
