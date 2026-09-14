"""R32: Detector inefficiency / noisy homodyne.

Noisy homodyne (``Homodyne(efficiency, noise)``) is supported on both
Gaussian backends and experimentally on the mixed-Fock backend; the pure
Fock backend explicitly rejects it (feature matrix, docs/validation/
feature-matrix.md). This script verifies the documented added-variance
formula nu = (1-efficiency)/efficiency + noise (docs/theory.md,
"Destructive Gaussian measurements") against sampled outcome variance on the
NumPy Gaussian backend, then separately cross-checks a modest mixed-Fock case
against the exact Gaussian prediction (a finite-cutoff approximation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

SHOTS = 6000
SEED = 20260913
ALPHA = 0.4 + 0.2j
EFFICIENCIES = [1.0, 0.9, 0.7, 0.5]
NOISES = [0.0, 0.2, 0.6]
MIXED_FOCK_CUTOFF = 10
# The mixed-Fock noisy-homodyne path convolves a dense density matrix with a
# Gaussian electronics kernel via scipy.integrate.quad_vec inside a brentq
# root-find, PER SHOT, whenever noise!=0: measured at ~8s/shot at cutoff=10
# (and >13s/shot at cutoff=20) on the reference machine. Shot counts below
# are deliberately small for the noisy cases and are a coarse cross-check,
# not a high-precision statistical estimate; the exact Gaussian-backend sweep
# above is the statistically powered result.
MIXED_FOCK_SHOTS_NOISELESS = 200
MIXED_FOCK_SHOTS_NOISY = 6


def independent_variance(prior_variance: float, efficiency: float, noise: float) -> float:
    """Documented formula: nu=(1-eta)/eta + noise, outcomes calibrated to the incident quadrature."""
    return prior_variance + (1 - efficiency) / efficiency + noise


def main():
    plt = common.setup_style()
    rows = []
    for efficiency in EFFICIENCIES:
        for noise in NOISES:
            pattern = pg.Pattern(inputs=(0,)).append(
                pg.Measure(0, pg.Homodyne(angle=0.0, efficiency=efficiency, noise=noise), key="m")
            )
            shots = pg.run_shots(
                pattern, SHOTS, backend="gaussian", inputs={0: pg.GaussianInput.coherent(ALPHA)}, seed=SEED
            )
            samples = shots.values("m")
            empirical_variance = float(np.var(samples, ddof=1))
            predicted_variance = independent_variance(1.0, efficiency, noise)  # coherent input: prior q variance = 1
            # Sampling error of a variance estimate from N draws scales like
            # variance*sqrt(2/(N-1)); use a 6-sigma band for a robust, non-flaky check.
            sampling_error = predicted_variance * np.sqrt(2 / (SHOTS - 1))
            rows.append(
                {
                    "efficiency": efficiency,
                    "noise": noise,
                    "empirical_variance": empirical_variance,
                    "predicted_variance": predicted_variance,
                    "absolute_error": abs(empirical_variance - predicted_variance),
                    "sampling_error_6sigma": 6 * sampling_error,
                    "within_tolerance": abs(empirical_variance - predicted_variance) <= 6 * sampling_error,
                }
            )

    failures = [r for r in rows if not r["within_tolerance"]]
    if failures:
        common.save_csv(failures, "R32_noisy_measurement_FAILURES")
        raise AssertionError(f"{len(failures)} noisy-homodyne cases exceeded the statistical tolerance; see FAILURES csv")

    # Mixed-Fock cross-check (experimental, finite-cutoff approximation).
    mixed_rows = []
    for efficiency in (1.0, 0.8):
        for noise in (0.0, 0.3):
            pattern = pg.Pattern(inputs=(0,)).append(
                pg.Measure(0, pg.Homodyne(angle=0.0, efficiency=efficiency, noise=noise), key="m")
            )
            shots = MIXED_FOCK_SHOTS_NOISELESS if noise == 0.0 else MIXED_FOCK_SHOTS_NOISY
            failure = common.safe_cutoff_run(
                lambda efficiency=efficiency, noise=noise, shots=shots: pg.run_shots(
                    pattern, shots, backend="piquasso-mixed-fock", cutoff=MIXED_FOCK_CUTOFF,
                    inputs={0: pg.GaussianInput.coherent(ALPHA)}, seed=SEED,
                ),
                label=f"mixed-fock eff={efficiency} noise={noise}",
            )
            if not failure["ok"]:
                mixed_rows.append({"efficiency": efficiency, "noise": noise, "ok": False, "error": failure["error"]})
                continue
            samples = failure["value"].values("m")
            empirical_variance = float(np.var(samples, ddof=1))
            predicted_variance = independent_variance(1.0, efficiency, noise)
            mixed_rows.append(
                {
                    "efficiency": efficiency, "noise": noise, "ok": True,
                    "empirical_variance": empirical_variance, "predicted_variance": predicted_variance,
                    "absolute_error": abs(empirical_variance - predicted_variance),
                    "cutoff": MIXED_FOCK_CUTOFF, "shots": shots,
                }
            )

    common.save_result(
        rows, "R32_noisy_measurement",
        extra={"shots": SHOTS, "alpha": [ALPHA.real, ALPHA.imag], "mixed_fock": mixed_rows},
    )
    common.save_csv(mixed_rows, "R32_noisy_measurement_mixed_fock")

    fig, ax = plt.subplots(figsize=(6, 4))
    for noise in NOISES:
        subset = [r for r in rows if r["noise"] == noise]
        ax.plot([r["efficiency"] for r in subset], [r["predicted_variance"] for r in subset], "-", color="grey")
        ax.plot([r["efficiency"] for r in subset], [r["empirical_variance"] for r in subset], "o", label=f"noise={noise}")
    ax.set_xlabel("Detector efficiency")
    ax.set_ylabel("Outcome variance")
    ax.legend(fontsize=8)
    ax.set_title("Noisy homodyne: sampled vs. documented formula (lines=analytic)")
    common.save_figure(fig, "R32_noisy_measurement")
    plt.close(fig)

    common.print_summary(
        "R32 detector inefficiency / noisy homodyne",
        cases=len(rows),
        max_absolute_error=max(r["absolute_error"] for r in rows),
        mixed_fock_ok=sum(1 for r in mixed_rows if r["ok"]),
    )


if __name__ == "__main__":
    main()
