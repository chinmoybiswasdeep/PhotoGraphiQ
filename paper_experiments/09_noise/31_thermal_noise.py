"""R31: Thermal noise validation.

Sweeps transmissivity eta and thermal environment occupation nbar through
``photographiq.commands.Loss`` on both Gaussian backends, and compares the
resulting mean/covariance to the independent analytic thermal-attenuation
channel: mean -> sqrt(eta)*mean_in, V -> eta*V_in + (1-eta)*(2*nbar+1)*I
(statistical covariance, vacuum V=I), written directly with NumPy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

ETAS = [0.2, 0.4, 0.6, 0.8, 1.0]
NBARS = [0.0, 0.3, 0.8, 1.5, 3.0]
ALPHA = 0.5 - 0.3j


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
            for backend in ("gaussian", "piquasso"):
                state = pg.simulate(pattern, backend=backend, inputs={0: input_state}, seed=0).state
                row[f"{backend}_mean_error"] = common.frobenius_error(state.mean, expected_mean)
                row[f"{backend}_covariance_error"] = common.frobenius_error(state.covariance, expected_cov)
            rows.append(row)

    tolerance = 1e-9
    failures = [
        r for r in rows
        if r["gaussian_mean_error"] > tolerance or r["piquasso_mean_error"] > tolerance
        or r["gaussian_covariance_error"] > tolerance or r["piquasso_covariance_error"] > tolerance
    ]
    if failures:
        common.save_csv(failures, "R31_thermal_noise_FAILURES")
        raise AssertionError("Thermal loss channel disagreed with the analytic formula; see FAILURES csv")

    common.save_result(rows, "R31_thermal_noise", extra={"alpha": [ALPHA.real, ALPHA.imag]})

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
    )


if __name__ == "__main__":
    main()
