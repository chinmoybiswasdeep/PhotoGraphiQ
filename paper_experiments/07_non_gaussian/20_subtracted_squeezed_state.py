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

Upgrade migration (2026-11): the main engine is the C++ ``native-fock``
backend. The same conditional branch (count = 1) is recomputed on
``piquasso-fock`` (historical engine) and ``native-mixed-fock`` as independent
references: herald probability, parity, mean photon number and the
total-variation distance of the photon-number distribution must agree within
REFERENCE_TOLERANCE. The cutoff-convergence study runs on ``native-fock`` and
its retained norms are reported; the state is never silently renormalised
without the retained norm being recorded. The minimum of the Wigner function
(negativity) is reported for the subtracted state.
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
MAIN_BACKEND = "native-fock"
REFERENCE_BACKENDS = ("piquasso-fock", "native-mixed-fock")
# Same model, same cutoff: engines may differ only by the Gaussian-input preparation route
# (exact Bargmann column vs Piquasso's P D P R P S|0>, see backends/native_fock.py), which is
# tiny for r = 0.6 at cutoff 24.
REFERENCE_TOLERANCE = 1e-6


def main():
    plt = common.setup_style()
    squeezed_input = pg.GaussianInput.squeezed(-SQUEEZING_R)  # q-squeezed: even support, like docs' resource convention

    before_pattern = pg.Pattern(inputs=("in",)).append(pg.Output(("in",))).validate()
    before = pg.simulate(
        before_pattern, backend=MAIN_BACKEND, cutoff=CUTOFF, inputs={"in": squeezed_input}, seed=0
    ).state

    subtraction_pattern = pg.non_gaussian.photon_subtraction(THETA)
    after_result = pg.simulate(
        subtraction_pattern,
        backend=MAIN_BACKEND,
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

    reference_rows = []
    for backend in (MAIN_BACKEND, *REFERENCE_BACKENDS):
        other = pg.simulate(subtraction_pattern, backend=backend, cutoff=CUTOFF, inputs={"in": squeezed_input},
                            measurement_outcomes={"count": 1}, seed=0)
        dist = other.state.probabilities
        keys = set(dist) | set(after.probabilities)
        reference_rows.append({
            "backend": backend,
            "herald_probability": other.measurement_statistics["count"]["value"],
            "parity": other.state.parity(),
            "mean_photon_number": other.state.photon_number("in"),
            "tvd_vs_main": 0.5 * sum(abs(dist.get(k, 0.0) - after.probabilities.get(k, 0.0)) for k in keys),
            "min_retained_norm": min(other.state.retained_norms),
        })
    main_row = reference_rows[0]
    reference_failures = [
        r for r in reference_rows[1:]
        if max(abs(r["herald_probability"] - main_row["herald_probability"]), abs(r["parity"] - main_row["parity"]),
               abs(r["mean_photon_number"] - main_row["mean_photon_number"]), r["tvd_vs_main"]) > REFERENCE_TOLERANCE
    ]
    common.save_csv(reference_rows, "R20_subtracted_squeezed_state_engines")

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
        "main_backend": MAIN_BACKEND,
        "engine_comparison": reference_rows,
        "reference_tolerance": REFERENCE_TOLERANCE,
        "after_min_retained_norm": min(after.retained_norms),
    }
    common.save_json(summary, "R20_subtracted_squeezed_state_summary")
    common.save_csv(distribution_rows, "R20_subtracted_squeezed_state_distribution")
    common.write_metadata("R20_subtracted_squeezed_state")

    q_axis = np.linspace(-5, 5, 121)
    p_axis = np.linspace(-5, 5, 121)
    before_wigner = before.wigner(q_axis, p_axis, node="in")
    after_wigner = after.wigner(q_axis, p_axis, node="in")
    summary["after_wigner_minimum"] = float(after_wigner.values.min())
    summary["before_wigner_minimum"] = float(before_wigner.values.min())
    common.save_json(summary, "R20_subtracted_squeezed_state_summary")

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
        backend=MAIN_BACKEND,
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
    if reference_failures:
        common.save_csv(reference_failures, "R20_subtracted_squeezed_state_FAILURES")
        raise AssertionError("Reference Fock engine disagreed with native-fock beyond tolerance; see FAILURES csv")

    common.save_caption(
        "R20_subtracted_squeezed_state",
        f"""
**R20: heralded photon subtraction on squeezed vacuum.** Left: photon-number distribution of
q-squeezed vacuum (r = {SQUEEZING_R}) before and after one heralded subtraction (beamsplitter tap
theta = {THETA}, herald count 1; herald probability {summary['herald_probability']:.4e}); centre/right:
Wigner functions (hbar = 2) before and after. Parity changes from {before_parity:+.4f} to
{after_parity:+.4f}; the mean photon number from {before_mean_n:.4f} to {after_mean_n:.4f}; the subtracted
state has Wigner minimum {summary['after_wigner_minimum']:.4f} (negative: non-Gaussian). Main engine
`native-fock` at total-photon cutoff {CUTOFF} (minimum retained norm {summary['after_min_retained_norm']:.12f});
`piquasso-fock` and `native-mixed-fock` reproduce herald probability, parity, mean photon number and the
distribution (total-variation distance <= {max(r['tvd_vs_main'] for r in reference_rows[1:]):.1e}). Cutoff
study on `native-fock` (cutoffs {CONVERGENCE_CUTOFFS}): fidelity between the two highest cutoffs
{convergence_rows[-1]['fidelity_to_previous']}. The branch is postselected (conditional on count = 1) and
the tap is a physical, lossy subtraction, not the ideal annihilation operator. Source:
`paper_experiments/07_non_gaussian/20_subtracted_squeezed_state.py`.
""",
    )

    common.print_summary(
        "R20 photon subtraction on squeezed vacuum",
        before_parity=before_parity,
        after_parity=after_parity,
        parity_flipped=parity_flipped,
        final_fidelity_to_previous_cutoff=convergence_rows[-1]["fidelity_to_previous"],
        max_reference_tvd=max(r["tvd_vs_main"] for r in reference_rows[1:]),
        wigner_minimum_after=summary["after_wigner_minimum"],
    )


if __name__ == "__main__":
    main()
