"""R30: Loss / attenuation validation.

Sweeps intensity transmissivity eta through ``photographiq.commands.Loss``
on both Gaussian backends. For a coherent input, checks the exact analytic
relations alpha_out = sqrt(eta)*alpha_in and <n>_out = eta*<n>_in (thermal
occupation held at zero), computed independently with NumPy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

ETAS = np.linspace(0.05, 1.0, 20)
ALPHA = 0.6 + 0.4j


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
        for backend in ("gaussian", "piquasso"):
            state = pg.simulate(pattern, backend=backend, inputs={0: pg.GaussianInput.coherent(ALPHA)}, seed=0).state
            row[f"{backend}_mean_error"] = common.frobenius_error(state.mean, expected_mean)
            row[f"{backend}_photon_number_error"] = abs(state.photon_number(0) - expected_n)
            row[f"{backend}_photon_number"] = state.photon_number(0)
        row["expected_photon_number"] = expected_n
        rows.append(row)

    tolerance = 1e-9
    failures = [
        r for r in rows
        if r["gaussian_mean_error"] > tolerance or r["piquasso_mean_error"] > tolerance
        or r["gaussian_photon_number_error"] > tolerance or r["piquasso_photon_number_error"] > tolerance
    ]
    if failures:
        common.save_csv(failures, "R30_loss_validation_FAILURES")
        raise AssertionError("Loss channel disagreed with the analytic attenuation formulas; see FAILURES csv")

    common.save_result(rows, "R30_loss_validation", extra={"alpha": [ALPHA.real, ALPHA.imag]})

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    etas = [r["eta"] for r in rows]
    ax.plot(etas, [r["expected_photon_number"] for r in rows], "-", label="Analytic: eta*|alpha|^2", color="grey")
    ax.plot(etas, [r["gaussian_photon_number"] for r in rows], "o", markersize=4, label="NumPy Gaussian backend", color="#24677b")
    ax.plot(etas, [r["piquasso_photon_number"] for r in rows], "x", markersize=5, label="Piquasso backend", color="#c66d27")
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
    )


if __name__ == "__main__":
    main()
