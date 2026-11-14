"""R30: Loss / attenuation validation.

Sweeps intensity transmissivity eta through ``photographiq.commands.Loss``
on both Gaussian backends. For a coherent input, checks the exact analytic
relations alpha_out = sqrt(eta)*alpha_in and <n>_out = eta*<n>_in (thermal
occupation held at zero), computed independently with NumPy.

Upgrade migration (2026-11): the C++ ``native-gaussian`` engine is added to
the exact comparison (same 1e-9 tolerance). The density-matrix engines
``native-mixed-fock`` and ``piquasso-mixed-fock`` apply the same pure-loss
channel (Kraus decomposition) at MIXED_CUTOFF; their photon number and q
variance are compared with the analytic values, their minimum retained norm
and wall time per run (provisional, shared machine) are recorded to
characterise accuracy and cost. Asserted with MIXED_TOLERANCE when the
retained norm is >= 1 - 1e-9 (coherent |alpha| = 0.72: tail above 20 photons
is negligible).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

ETAS = np.linspace(0.05, 1.0, 20)
ALPHA = 0.6 + 0.4j
GAUSSIAN_BACKENDS = ("gaussian", "piquasso", "native-gaussian")
MIXED_BACKENDS = ("native-mixed-fock", "piquasso-mixed-fock")
MIXED_CUTOFF = 20
MIXED_TOLERANCE = 1e-8


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
def main():
    plt = common.setup_style()
    rows = []
    for eta in ETAS:
        eta = float(eta)
        pattern = pg.Pattern(inputs=(0,)).append(pg.Loss(0, eta, 0.0)).append(pg.Output((0,))).validate()
        expected_alpha = np.sqrt(eta) * ALPHA
        expected_mean = np.array([2 * expected_alpha.real, 2 * expected_alpha.imag])
        expected_n = eta * abs(ALPHA) ** 2

        row = {"eta": eta}
        for backend in GAUSSIAN_BACKENDS:
            state = pg.simulate(pattern, backend=backend, inputs={0: pg.GaussianInput.coherent(ALPHA)}, seed=0).state
            row[f"{backend}_mean_error"] = common.frobenius_error(state.mean, expected_mean)
            row[f"{backend}_photon_number_error"] = abs(state.photon_number(0) - expected_n)
            row[f"{backend}_photon_number"] = state.photon_number(0)
        row["expected_photon_number"] = expected_n
        for backend in MIXED_BACKENDS:
            row.update(mixed_fock_check(pattern, pg.GaussianInput.coherent(ALPHA), expected_n, 1.0, backend, MIXED_CUTOFF))
        rows.append(row)

    tolerance = 1e-9
    failures = [
        r for r in rows
        if r["gaussian_mean_error"] > tolerance or r["piquasso_mean_error"] > tolerance
        or r["gaussian_photon_number_error"] > tolerance or r["piquasso_photon_number_error"] > tolerance
        or r["native-gaussian_mean_error"] > tolerance or r["native-gaussian_photon_number_error"] > tolerance
        or any(r[f"{b}_ok"] and r[f"{b}_min_retained_norm"] >= 1 - 1e-9
               and max(r[f"{b}_photon_number_error"], r[f"{b}_q_variance_error"]) > MIXED_TOLERANCE
               for b in MIXED_BACKENDS)
    ]
    if failures:
        common.save_csv(failures, "R30_loss_validation_FAILURES")
        raise AssertionError("Loss channel disagreed with the analytic attenuation formulas; see FAILURES csv")

    common.save_result(rows, "R30_loss_validation", extra={
        "alpha": [ALPHA.real, ALPHA.imag], "gaussian_backends": list(GAUSSIAN_BACKENDS), "mixed_backends": list(MIXED_BACKENDS),
        "mixed_cutoff": MIXED_CUTOFF, "mixed_tolerance": MIXED_TOLERANCE, "timing_note": "provisional (shared machine)"})

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    etas = [r["eta"] for r in rows]
    ax.plot(etas, [r["expected_photon_number"] for r in rows], "-", label="Analytic: eta*|alpha|^2", color="grey")
    ax.plot(etas, [r["gaussian_photon_number"] for r in rows], "o", markersize=4, label="NumPy Gaussian backend", color="#24677b")
    ax.plot(etas, [r["piquasso_photon_number"] for r in rows], "x", markersize=5, label="Piquasso backend", color="#c66d27")
    ax.plot(etas, [r["native-gaussian_photon_number"] for r in rows], "+", markersize=7, label="native-gaussian", color="#7a3e9d")
    ax.set_xlabel("Transmissivity eta")
    ax.set_ylabel("Output mean photon number")
    ax.legend(fontsize=8)
    common.save_figure(fig, "R30_loss_validation")
    plt.close(fig)

    common.print_summary(
        "R30 loss / attenuation validation",
        eta_points=len(rows),
        max_mean_error=max(max(r["gaussian_mean_error"], r["piquasso_mean_error"]) for r in rows),
        max_photon_error=max(max(r["gaussian_photon_number_error"], r["piquasso_photon_number_error"]) for r in rows),
        max_native_gaussian_error=max(max(r["native-gaussian_mean_error"], r["native-gaussian_photon_number_error"]) for r in rows),
        **{f"{b}_max_photon_error": max(r[f"{b}_photon_number_error"] for r in rows if r[f"{b}_ok"]) for b in MIXED_BACKENDS},
        **{f"{b}_median_seconds_provisional": float(np.median([r[f"{b}_seconds"] for r in rows])) for b in MIXED_BACKENDS},
    )


if __name__ == "__main__":
    main()
