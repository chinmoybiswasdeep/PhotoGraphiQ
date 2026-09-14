"""R19: Photon subtraction analytic probability.

Uses the package's physical vacuum-tap subtraction pattern
(``photographiq.non_gaussian.photon_subtraction``): a beamsplitter of angle
theta couples input |n> to a vacuum ancilla, which is then photon-counted.
For input |n>, heralding count k has the exact binomial probability
C(n,k) sin(theta)^(2k) cos(theta)^(2(n-k)) -- this is the analytic reference,
derived independently here with SciPy, and the resulting signal-mode photon
number is deterministically n-k (a number-conserving beamsplitter on a
definite Fock input produces a photon-number-correlated joint state).
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


def analytic_probability(n, k, theta):
    return float(comb(n, k) * np.sin(theta) ** (2 * k) * np.cos(theta) ** (2 * (n - k)))


def main():
    plt = common.setup_style()
    rows = []
    for n in PHOTON_NUMBERS:
        for theta in THETAS:
            pattern = pg.non_gaussian.photon_subtraction(float(theta))
            for k in range(n + 1):
                result = pg.simulate(
                    pattern,
                    backend="piquasso-fock",
                    cutoff=CUTOFF,
                    inputs={"in": pg.FockInput.number(n)},
                    measurement_outcomes={"count": k},
                    seed=0,
                )
                pg_probability = result.measurement_statistics["count"]["value"]
                analytic = analytic_probability(n, k, float(theta))
                conditional_n = result.state.photon_number("in")
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
                    }
                )

    tolerance = 1e-9
    photon_tolerance = 1e-9
    failures = [
        r for r in rows
        if r["absolute_error"] > tolerance or r["conditional_photon_number_error"] > photon_tolerance
    ]
    if failures:
        common.save_csv(failures, "R19_photon_subtraction_probability_FAILURES")
        raise AssertionError(f"{len(failures)} cases disagreed with analytic predictions; see FAILURES csv")

    common.save_result(rows, "R19_photon_subtraction_probability", extra={"cutoff": CUTOFF})

    fig, axes = plt.subplots(1, len(PHOTON_NUMBERS), figsize=(4 * len(PHOTON_NUMBERS), 3.6), sharey=True)
    for ax, n in zip(axes, PHOTON_NUMBERS, strict=True):
        for k in range(n + 1):
            subset = [r for r in rows if r["n"] == n and r["k"] == k]
            ts = [r["theta"] for r in subset]
            ax.plot(ts, [r["analytic_probability"] for r in subset], "-", label=f"k={k} analytic")
            ax.plot(ts, [r["photographiq_probability"] for r in subset], "o", markersize=3, label=f"k={k} PhotoGraphiQ")
        ax.set_xlabel(r"Tap angle $\theta$")
        ax.set_title(f"n={n}")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("Herald probability")
    fig.suptitle("Photon subtraction: analytic binomial vs. simulated probability")
    common.save_figure(fig, "R19_photon_subtraction_probability")
    plt.close(fig)

    common.print_summary(
        "R19 photon subtraction analytic probability",
        cases=len(rows),
        max_absolute_error=max(r["absolute_error"] for r in rows),
        max_conditional_photon_error=max(r["conditional_photon_number_error"] for r in rows),
    )


if __name__ == "__main__":
    main()
