"""R32: Detector inefficiency / noisy homodyne.

Noisy homodyne (``Homodyne(efficiency, noise)``) is supported on both
Gaussian backends and experimentally on the mixed-Fock backend; the pure
Fock backend explicitly rejects it (feature matrix, docs/validation/
feature-matrix.md). This script verifies the documented added-variance
formula nu = (1-efficiency)/efficiency + noise (docs/theory.md,
"Destructive Gaussian measurements") against sampled outcome variance on the
NumPy Gaussian backend, then separately cross-checks a modest mixed-Fock case
against the exact Gaussian prediction (a finite-cutoff approximation).

Upgrade migration (2026-11):

* the statistically powered sweep also runs on C++ ``native-gaussian`` with the
  same seed; its samples must equal the NumPy samples (documented RNG
  contract) and pass the same 6-sigma check;
* the mixed-Fock cross-check now runs on ``native-mixed-fock`` (main) and
  ``piquasso-mixed-fock`` (historical). NEW deterministic check: the
  postselected outcome DENSITY of the noisy homodyne (``measurement_outcomes``)
  at a grid of outcomes is compared with the analytic Gaussian density
  N(2 Re alpha, 1 + nu) -- exact up to Fock truncation, no shot noise, and
  ~0.04 s per point;
* sampling the noisy branch on a mixed-Fock engine costs ~8-22 s PER SHOT
  (``quad_vec`` inside a root find, shared Python path, so the native engine
  does not remove it; ISSUES_FOUND.md). The default configuration therefore
  samples only the noiseless cases (MIXED_FOCK_SHOTS_NOISELESS shots, 6-sigma
  checked); ``--full`` restores the historical noisy sampling
  (MIXED_FOCK_SHOTS_NOISY shots per noisy case per engine) as the reproducible
  larger configuration.
"""

from __future__ import annotations

import itertools
import sys
import time
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
FULL = "--full" in sys.argv[1:]
MIXED_ENGINES = ("native-mixed-fock", "piquasso-mixed-fock")
DENSITY_OUTCOMES = [-1.5, -0.5, 0.0, 0.5, 0.8, 1.5, 3.0]
DENSITY_TOLERANCE = 1e-6  # |alpha|^2 = 0.2 at cutoff 10: truncated tail < 1e-9
NATIVE_SAMPLE_TOLERANCE = 1e-10
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
            native_samples = pg.run_shots(
                pattern, SHOTS, backend="native-gaussian", inputs={0: pg.GaussianInput.coherent(ALPHA)}, seed=SEED
            ).values("m")
            native_difference = float(np.abs(np.asarray(native_samples) - np.asarray(samples)).max())
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
                    "within_tolerance": abs(empirical_variance - predicted_variance) <= 6 * sampling_error
                    and native_difference <= NATIVE_SAMPLE_TOLERANCE,
                    "native_vs_gaussian_max_sample_difference": native_difference,
                }
            )

    failures = [r for r in rows if not r["within_tolerance"]]
    if failures:
        common.save_csv(failures, "R32_noisy_measurement_FAILURES")
        raise AssertionError(f"{len(failures)} noisy-homodyne cases exceeded the statistical tolerance; see FAILURES csv")

    # Mixed-Fock cross-check (experimental, finite-cutoff approximation).
    from scipy.stats import norm

    mixed_rows, density_rows = [], []
    for efficiency, noise, engine in itertools.product((1.0, 0.8), (0.0, 0.3), MIXED_ENGINES):
        pattern = pg.Pattern(inputs=(0,)).append(
            pg.Measure(0, pg.Homodyne(angle=0.0, efficiency=efficiency, noise=noise), key="m")
        )
        nu = (1 - efficiency) / efficiency + noise
        for outcome in DENSITY_OUTCOMES:
            start = time.perf_counter()
            result = pg.simulate(pattern, backend=engine, cutoff=MIXED_FOCK_CUTOFF,
                                 inputs={0: pg.GaussianInput.coherent(ALPHA)},
                                 measurement_outcomes={"m": outcome}, seed=SEED)
            analytic = float(norm.pdf(outcome, 2 * ALPHA.real, np.sqrt(1.0 + nu)))
            density_rows.append({
                "engine": engine, "efficiency": efficiency, "noise": noise, "outcome": outcome,
                "density": result.measurement_statistics["m"]["value"], "analytic_density": analytic,
                "absolute_error": abs(result.measurement_statistics["m"]["value"] - analytic),
                "seconds": time.perf_counter() - start,
            })
        shots = MIXED_FOCK_SHOTS_NOISELESS if noise == 0.0 else MIXED_FOCK_SHOTS_NOISY
        if noise != 0.0 and not FULL:
            mixed_rows.append({"engine": engine, "efficiency": efficiency, "noise": noise, "ok": None,
                               "error": "noisy sampling skipped in default configuration (~8-22 s/shot); run --full"})
            continue
        start = time.perf_counter()
        failure = common.safe_cutoff_run(
            lambda efficiency=efficiency, noise=noise, shots=shots: pg.run_shots(
                pattern, shots, backend=engine, cutoff=MIXED_FOCK_CUTOFF,
                inputs={0: pg.GaussianInput.coherent(ALPHA)}, seed=SEED,
            ),
            label=f"{engine} eff={efficiency} noise={noise}",
        )
        elapsed = time.perf_counter() - start
        if not failure["ok"]:
            mixed_rows.append({"engine": engine, "efficiency": efficiency, "noise": noise, "ok": False,
                               "error": failure["error"]})
            continue
        samples = failure["value"].values("m")
        empirical_variance = float(np.var(samples, ddof=1))
        predicted_variance = independent_variance(1.0, efficiency, noise)
        mixed_rows.append(
            {
                "engine": engine, "efficiency": efficiency, "noise": noise, "ok": True,
                "empirical_variance": empirical_variance, "predicted_variance": predicted_variance,
                "absolute_error": abs(empirical_variance - predicted_variance),
                "sampling_error_6sigma": 6 * predicted_variance * np.sqrt(2 / (shots - 1)),
                "cutoff": MIXED_FOCK_CUTOFF, "shots": shots, "seconds_per_shot_provisional": elapsed / shots,
            }
        )
    common.save_csv(density_rows, "R32_noisy_measurement_mixed_fock_density")
    mixed_failures = [r for r in density_rows if r["absolute_error"] > DENSITY_TOLERANCE]
    mixed_failures += [r for r in mixed_rows if r["ok"] and r["noise"] == 0.0
                       and r["absolute_error"] > r["sampling_error_6sigma"]]
    if mixed_failures:
        common.save_csv(mixed_failures, "R32_noisy_measurement_FAILURES")
        raise AssertionError("Mixed-Fock noisy-homodyne density or noiseless sampling check failed; see FAILURES csv")

    common.save_result(
        rows, "R32_noisy_measurement",
        extra={"shots": SHOTS, "alpha": [ALPHA.real, ALPHA.imag], "mixed_fock": mixed_rows,
               "mixed_fock_density_check": {"tolerance": DENSITY_TOLERANCE, "points": len(density_rows),
                                            "max_absolute_error": max(r["absolute_error"] for r in density_rows)},
               "configuration": "full" if FULL else "default", "gaussian_engines": ["gaussian", "native-gaussian"]},
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
        max_native_vs_gaussian_sample_difference=max(r["native_vs_gaussian_max_sample_difference"] for r in rows),
        mixed_density_max_error=max(r["absolute_error"] for r in density_rows),
        mixed_density_seconds_per_point=float(np.median([r["seconds"] for r in density_rows])),
        configuration="full" if FULL else "default",
    )


if __name__ == "__main__":
    main()
