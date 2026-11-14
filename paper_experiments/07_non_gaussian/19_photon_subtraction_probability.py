"""R19: Photon subtraction analytic probability.

Uses the package's physical vacuum-tap subtraction pattern
(``photographiq.non_gaussian.photon_subtraction``): a beamsplitter of angle
theta couples input |n> to a vacuum ancilla, which is then photon-counted.
For input |n>, heralding count k has the exact binomial probability
C(n,k) sin(theta)^(2k) cos(theta)^(2(n-k)) -- this is the analytic reference,
derived independently here with SciPy, and the resulting signal-mode photon
number is deterministically n-k (a number-conserving beamsplitter on a
definite Fock input produces a photon-number-correlated joint state).

Upgrade migration (2026-11): the main engine is now the C++ ``native-fock``
backend (columns ``photographiq_*``, ``conditional_photon_number``). The
historical engine ``piquasso-fock`` and the density-matrix engine
``native-mixed-fock`` run the identical fixed-outcome branches as
independent references; every engine must match the analytic binomial law
and the conditional photon number n-k to the original 1e-9 tolerance. The
minimum retained Fock norm of each run is recorded (no truncation is hidden).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import comb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

THETAS = np.linspace(0.05, np.pi / 2 - 0.05, 15)
PHOTON_NUMBERS = (1, 2, 3)
CUTOFF = 8
MAIN_BACKEND = "native-fock"
REFERENCE_BACKENDS = ("piquasso-fock", "native-mixed-fock")


def analytic_probability(n, k, theta):
    return float(comb(n, k) * np.sin(theta) ** (2 * k) * np.cos(theta) ** (2 * (n - k)))


def main():
    plt = common.setup_style()
    rows = []
    for n in PHOTON_NUMBERS:
        for theta in THETAS:
            pattern = pg.non_gaussian.photon_subtraction(float(theta))
            for k in range(n + 1):
                results = {
                    backend: pg.simulate(
                        pattern,
                        backend=backend,
                        cutoff=CUTOFF,
                        inputs={"in": pg.FockInput.number(n)},
                        measurement_outcomes={"count": k},
                        seed=0,
                    )
                    for backend in (MAIN_BACKEND, *REFERENCE_BACKENDS)
                }
                result = results[MAIN_BACKEND]
                pg_probability = result.measurement_statistics["count"]["value"]
                analytic = analytic_probability(n, k, float(theta))
                conditional_n = result.state.photon_number("in")
                references = {}
                for backend in REFERENCE_BACKENDS:
                    other = results[backend]
                    references[f"{backend}_absolute_error"] = abs(other.measurement_statistics["count"]["value"] - analytic)
                    references[f"{backend}_conditional_photon_number_error"] = abs(other.state.photon_number("in") - (n - k))
                references["min_retained_norm_all_engines"] = min(min(r.state.retained_norms) for r in results.values())
                rows.append(
                    {
                        "n": n,
                        "theta": float(theta),
                        "k": k,
                        "analytic_probability": analytic,
                        "photographiq_probability": pg_probability,
                        "absolute_error": abs(pg_probability - analytic),
                        "relative_error": common.relative_error(pg_probability, analytic, floor=1e-12),
                        "conditional_photon_number": conditional_n,
                        "expected_conditional_photon_number": n - k,
                        "conditional_photon_number_error": abs(conditional_n - (n - k)),
                        **references,
                    }
                )

    tolerance = 1e-9
    photon_tolerance = 1e-9
    failures = [
        r for r in rows
        if r["absolute_error"] > tolerance or r["conditional_photon_number_error"] > photon_tolerance
        or any(r[f"{b}_absolute_error"] > tolerance or r[f"{b}_conditional_photon_number_error"] > photon_tolerance
               for b in REFERENCE_BACKENDS)
    ]
    if failures:
        common.save_csv(failures, "R19_photon_subtraction_probability_FAILURES")
        raise AssertionError(f"{len(failures)} cases disagreed with analytic predictions; see FAILURES csv")

    common.save_result(rows, "R19_photon_subtraction_probability",
                       extra={"cutoff": CUTOFF, "main_backend": MAIN_BACKEND, "reference_backends": list(REFERENCE_BACKENDS),
                              "tolerance": tolerance})

    fig, axes = plt.subplots(1, len(PHOTON_NUMBERS), figsize=(4 * len(PHOTON_NUMBERS), 3.6), sharey=True)
    for ax, n in zip(axes, PHOTON_NUMBERS, strict=True):
        for k in range(n + 1):
            subset = [r for r in rows if r["n"] == n and r["k"] == k]
            ts = [r["theta"] for r in subset]
            ax.plot(ts, [r["analytic_probability"] for r in subset], "-", label=f"k={k} analytic")
            ax.plot(ts, [r["photographiq_probability"] for r in subset], "o", markersize=3, label=f"k={k} native-fock")
        ax.set_xlabel(r"Tap angle $\theta$")
        ax.set_title(f"n={n}")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("Herald probability")
    fig.suptitle("Photon subtraction: analytic binomial vs. simulated probability")
    common.save_figure(fig, "R19_photon_subtraction_probability")
    plt.close(fig)
    worst = {b: max(r[f"{b}_absolute_error"] for r in rows) for b in REFERENCE_BACKENDS}
    common.save_caption(
        "R19_photon_subtraction_probability",
        f"""
**R19: heralded photon subtraction, analytic binomial law.** Probability of heralding k photons when
|n> (n = {', '.join(map(str, PHOTON_NUMBERS))}) passes a beamsplitter of angle theta whose other port
starts in vacuum (`non_gaussian.photon_subtraction`), for {len(THETAS)} angles in
[{THETAS[0]:.2f}, {THETAS[-1]:.2f}]. Lines: C(n, k) sin^(2k)(theta) cos^(2(n-k))(theta); circles: the
postselected branch probability reported by `native-fock` (total-photon cutoff {CUTOFF}). Over
{len(rows)} branches the maximum absolute error is {max(r['absolute_error'] for r in rows):.1e}
(native-fock), {worst['piquasso-fock']:.1e} (piquasso-fock) and {worst['native-mixed-fock']:.1e}
(native-mixed-fock); the conditional signal photon number equals n - k to
{max(r['conditional_photon_number_error'] for r in rows):.1e}. Minimum retained Fock norm over all runs
{min(r['min_retained_norm_all_engines'] for r in rows):.15f} (the input never leaves the truncated space).
Each value is a postselected (conditional) probability, not a deterministic operation. Source:
`paper_experiments/07_non_gaussian/19_photon_subtraction_probability.py`.
""",
    )

    common.print_summary(
        "R19 photon subtraction analytic probability",
        cases=len(rows),
        max_absolute_error=max(r["absolute_error"] for r in rows),
        max_conditional_photon_error=max(r["conditional_photon_number_error"] for r in rows),
        **{f"{b}_max_absolute_error": max(r[f"{b}_absolute_error"] for r in rows) for b in REFERENCE_BACKENDS},
    )


if __name__ == "__main__":
    main()
