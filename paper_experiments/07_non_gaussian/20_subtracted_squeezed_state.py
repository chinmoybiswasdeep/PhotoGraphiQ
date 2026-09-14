"""R20: Photon subtraction on squeezed vacuum.

Prepares a squeezed-vacuum input (``photographiq.states.GaussianInput.squeezed``,
which the pure-Fock backend accepts directly and converts to a truncated
Fock vector) and performs one-photon heralded subtraction via
``photographiq.non_gaussian.photon_subtraction``. Compares the Fock
distribution, parity and mean photon number before/after, and demonstrates
the expected even-to-odd parity flip (squeezed vacuum has only even photon
numbers; single-photon subtraction produces a cat-like odd-photon-dominant
state). Uses ``cutoff_convergence`` to check the finite-cutoff behavior of
this filtering/attenuating physical subtraction (not an ideal a-operator).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

SQUEEZING_R = 0.6
THETA = 0.15  # small tap angle: closer to ideal (low-attenuation) subtraction
CUTOFF = 24
CONVERGENCE_CUTOFFS = [12, 16, 20, 28, 36]


def main():
    plt = common.setup_style()
    squeezed_input = pg.GaussianInput.squeezed(-SQUEEZING_R)  # q-squeezed: even support, like docs' resource convention

    before_pattern = pg.Pattern(inputs=("in",)).append(pg.Output(("in",))).validate()
    before = pg.simulate(
        before_pattern, backend="piquasso-fock", cutoff=CUTOFF, inputs={"in": squeezed_input}, seed=0
    ).state

    subtraction_pattern = pg.non_gaussian.photon_subtraction(THETA)
    after_result = pg.simulate(
        subtraction_pattern,
        backend="piquasso-fock",
        cutoff=CUTOFF,
        inputs={"in": squeezed_input},
        measurement_outcomes={"count": 1},
        seed=0,
    )
    after = after_result.state

    before_parity = before.parity()
    after_parity = after.parity()
    before_mean_n = before.photon_number("in")
    after_mean_n = after.photon_number("in")

    parity_flipped = before_parity > 0.5 and after_parity < -0.5
    if not parity_flipped:
        raise AssertionError(
            f"Expected an even-to-odd parity flip; got before={before_parity:.4f}, after={after_parity:.4f}"
        )

    before_probs = before.probabilities
    after_probs = after.probabilities
    max_n = max(max(b[0] for b in before_probs), max(b[0] for b in after_probs)) + 1
    distribution_rows = [
        {
            "n": n,
            "before_probability": before_probs.get((n,), 0.0),
            "after_probability": after_probs.get((n,), 0.0),
        }
        for n in range(max_n)
    ]

    summary = {
        "squeezing_r": SQUEEZING_R,
        "tap_theta": THETA,
        "cutoff": CUTOFF,
        "before_parity": before_parity,
        "after_parity": after_parity,
        "before_mean_photon_number": before_mean_n,
        "after_mean_photon_number": after_mean_n,
        "herald_probability": after_result.measurement_statistics["count"]["value"],
        "parity_flipped_even_to_odd": parity_flipped,
    }
    common.save_json(summary, "R20_subtracted_squeezed_state_summary")
    common.save_csv(distribution_rows, "R20_subtracted_squeezed_state_distribution")
    common.write_metadata("R20_subtracted_squeezed_state")

    q_axis = np.linspace(-5, 5, 121)
    p_axis = np.linspace(-5, 5, 121)
    before_wigner = before.wigner(q_axis, p_axis, node="in")
    after_wigner = after.wigner(q_axis, p_axis, node="in")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    width = 0.35
    ns = [r["n"] for r in distribution_rows]
    axes[0].bar(np.array(ns) - width / 2, [r["before_probability"] for r in distribution_rows], width, label="Before", color="#24677b")
    axes[0].bar(np.array(ns) + width / 2, [r["after_probability"] for r in distribution_rows], width, label="After subtraction", color="#c66d27")
    axes[0].set_xlabel("Photon number n")
    axes[0].set_ylabel("Probability")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Fock distribution")

    for ax, grid, title in ((axes[1], before_wigner, "Before (squeezed vacuum)"), (axes[2], after_wigner, "After subtraction")):
        extent = max(abs(grid.values.min()), abs(grid.values.max()))
        mesh = ax.pcolormesh(grid.q, grid.p, grid.values, shading="auto", cmap="RdBu_r", vmin=-extent, vmax=extent)
        ax.set(xlabel="q", ylabel="p", aspect="equal", title=title)
        fig.colorbar(mesh, ax=ax, fraction=0.046)
    fig.suptitle(f"Photon subtraction on squeezed vacuum (r={SQUEEZING_R}, theta={THETA})")
    common.save_figure(fig, "R20_subtracted_squeezed_state")
    plt.close(fig)

    # Cutoff convergence for this fixed conditional branch (measurement_outcomes
    # fixes count=1 so every cutoff compares the same branch).
    study = pg.cutoff_convergence(
        subtraction_pattern,
        CONVERGENCE_CUTOFFS,
        backend="piquasso-fock",
        inputs={"in": squeezed_input},
        measurement_outcomes={"count": 1},
    )
    convergence_rows = [
        {
            "cutoff": row["cutoff"],
            "norm": row["norm"],
            "minimum_retained_norm": row["minimum_retained_norm"],
            "maximum_boundary_population": row["maximum_boundary_population"],
            "fidelity_to_previous": row["fidelity_to_previous"],
            "trace_distance_to_previous": row["trace_distance_to_previous"],
            "comparable_to_previous": row["comparable_to_previous"],
        }
        for row in study.rows
    ]
    common.save_csv(convergence_rows, "R20_subtracted_squeezed_state_convergence")

    common.print_summary(
        "R20 photon subtraction on squeezed vacuum",
        before_parity=before_parity,
        after_parity=after_parity,
        parity_flipped=parity_flipped,
        final_fidelity_to_previous_cutoff=convergence_rows[-1]["fidelity_to_previous"],
    )


if __name__ == "__main__":
    main()
