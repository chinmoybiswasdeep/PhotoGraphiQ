"""R40: Seed reproducibility / shots.

Verifies that ``photographiq.run_shots`` gives bit-identical outcomes for
repeated runs with the same seed, and that two different seeds draw from
statistically indistinguishable distributions (a two-sample Kolmogorov-
Smirnov test) while producing different raw sample sequences.

Upgrade migration (2026-11): cross-engine reproducibility is tested where the
code DOCUMENTS an RNG contract, and only observed elsewhere:

* documented: ``native-gaussian`` draws outcomes in Python with exactly the
  same ``numpy.random.Generator`` calls as ``gaussian`` (module docstring of
  backends/native_gaussian.py: equal seeds give equal trajectories up to
  floating-point summation order), and ``simulate`` documents that
  ``piquasso`` and ``gaussian`` give identical seeded trajectories. Asserted:
  max |sample difference| <= CONTRACT_TOLERANCE for both pairs;
* observed only: ``native-fock`` and ``piquasso-fock`` share the Python
  sampling path of ``PiquassoFockBackend`` (no written cross-engine contract).
  Photon-number samples of a coherent state are compared and the fraction of
  equal outcomes is REPORTED, not asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

SHOTS = 2000
ALPHA = 0.3 + 0.2j
CONTRACT_TOLERANCE = 1e-12
FOCK_SHOTS = 400
def build_pattern():
    return pg.Pattern(inputs=(0,)).append(pg.Measure(0, pg.Homodyne.q(), key="m"))


def main():
    plt = common.setup_style()
    pattern = build_pattern()
    inputs = {0: pg.GaussianInput.coherent(ALPHA)}

    run_a = pg.run_shots(pattern, SHOTS, backend="gaussian", inputs=inputs, seed=42)
    run_b = pg.run_shots(pattern, SHOTS, backend="gaussian", inputs=inputs, seed=42)
    run_c = pg.run_shots(pattern, SHOTS, backend="gaussian", inputs=inputs, seed=43)

    samples_a, samples_b, samples_c = run_a.values("m"), run_b.values("m"), run_c.values("m")

    identical_same_seed = bool(np.array_equal(samples_a, samples_b))
    identical_cross_seed = bool(np.array_equal(samples_a, samples_c))

    ks_same_distribution = stats.ks_2samp(samples_a, samples_c)
    ks_self_consistency = stats.ks_2samp(samples_a, samples_b)  # trivially identical arrays -> statistic 0

    mean_a, mean_c = float(np.mean(samples_a)), float(np.mean(samples_c))
    theoretical_mean = 2 * ALPHA.real  # q-quadrature mean of a coherent state

    summary = {
        "shots": SHOTS,
        "identical_outcomes_same_seed": identical_same_seed,
        "identical_outcomes_different_seeds": identical_cross_seed,
        "ks_statistic_seed42_vs_seed43": float(ks_same_distribution.statistic),
        "ks_pvalue_seed42_vs_seed43": float(ks_same_distribution.pvalue),
        "ks_statistic_seed42_vs_itself": float(ks_self_consistency.statistic),
        "sample_mean_seed42": mean_a,
        "sample_mean_seed43": mean_c,
        "theoretical_mean": theoretical_mean,
        "mean_error_seed42": abs(mean_a - theoretical_mean),
        "mean_error_seed43": abs(mean_c - theoretical_mean),
    }

    checks = {
        "same_seed_is_bit_identical": identical_same_seed,
        "different_seeds_are_not_identical": not identical_cross_seed,
        "different_seeds_pass_ks_same_distribution": ks_same_distribution.pvalue > 0.01,
        "sample_means_close_to_theory": summary["mean_error_seed42"] < 0.1 and summary["mean_error_seed43"] < 0.1,
    }
    cross = {}
    for backend in ("native-gaussian", "piquasso"):
        other = pg.run_shots(pattern, SHOTS, backend=backend, inputs=inputs, seed=42).values("m")
        cross[backend] = float(np.abs(np.asarray(other) - np.asarray(samples_a)).max())
    summary["max_sample_difference_vs_gaussian_seed42"] = cross
    checks["native_gaussian_follows_documented_rng_contract"] = cross["native-gaussian"] <= CONTRACT_TOLERANCE
    checks["piquasso_follows_documented_rng_contract"] = cross["piquasso"] <= CONTRACT_TOLERANCE
    counting = pg.Pattern(inputs=(0,)).append(pg.Measure(0, pg.PhotonNumber(), key="n"))
    fock_samples = {
        backend: np.asarray(pg.run_shots(counting, FOCK_SHOTS, backend=backend, cutoff=12,
                                         inputs={0: pg.GaussianInput.coherent(0.8)}, seed=42).values("n"))
        for backend in ("native-fock", "piquasso-fock")
    }
    summary["fock_photon_counting_equal_fraction_native_vs_piquasso"] = float(
        np.mean(fock_samples["native-fock"] == fock_samples["piquasso-fock"])
    )
    summary["fock_photon_counting_mean"] = {b: float(v.mean()) for b, v in fock_samples.items()}
    summary["fock_photon_counting_theory_mean"] = 0.64
    summary["fock_contract_note"] = "observed only; no documented cross-engine RNG contract for Fock engines"

    failures = [name for name, ok in checks.items() if not ok]
    if failures:
        common.save_json({"summary": summary, "failures": failures}, "R40_seeded_shots_FAILURES")
        raise AssertionError(f"Seed reproducibility checks failed: {failures}")

    common.save_json({**summary, "checks": checks}, "R40_seeded_shots")
    common.write_metadata("R40_seeded_shots")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(samples_a, bins=40, alpha=0.6, label="seed=42", color="#24677b", density=True)
    ax.hist(samples_c, bins=40, alpha=0.6, label="seed=43", color="#c66d27", density=True)
    ax.axvline(theoretical_mean, color="black", linestyle="--", label="Theoretical mean")
    ax.set_xlabel("Homodyne outcome (q)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_title(f"Seeded shot reproducibility (n={SHOTS} each)")
    common.save_figure(fig, "R40_seeded_shots")
    plt.close(fig)

    common.print_summary("R40 seed reproducibility / shots", **checks)


if __name__ == "__main__":
    main()
