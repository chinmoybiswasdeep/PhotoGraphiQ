"""R21: Cat state.

Generates even and odd finite-Fock cat resources (``photographiq.resources.
CatResource``, which internally calls ``FockInput.cat``) for several alpha
values, and cross-checks against an independently written NumPy coherent-
state expansion (not reusing ``FockInput.cat``). Reports parity,
photon-number distribution, normalization/retained mass and Wigner functions,
and checks the expected even/odd parity sign.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

ALPHAS = [0.5, 1.0, 1.5, 2.0]
CUTOFF = 32
REPRESENTATIVE_ALPHA = 1.5


def analytic_cat(alpha: float, cutoff: int, parity: int) -> np.ndarray:
    """Independent NumPy |alpha>+parity|-alpha> expansion (not FockInput.cat)."""
    n = np.arange(cutoff)
    log_coherent = n * np.log(alpha) - gammaln(n + 1) / 2 - 0.5 * alpha**2 if alpha > 0 else None
    if alpha == 0:
        vector = np.zeros(cutoff)
        vector[0] = 1 + parity
    else:
        vector = np.exp(log_coherent) * (1 + parity * (-1.0) ** n)
    norm = np.linalg.norm(vector)
    return vector / norm


def main():
    plt = common.setup_style()
    rows = []
    wigner_states = {}
    for alpha in ALPHAS:
        for parity in (1, -1):
            resource = pg.CatResource(alpha, parity=parity)
            pattern = pg.Pattern(inputs=()).append(pg.Prepare(0, state=resource)).append(pg.Output((0,))).validate()
            result = pg.simulate(pattern, backend="piquasso-fock", cutoff=CUTOFF, seed=0)
            state = result.state

            analytic = analytic_cat(alpha, CUTOFF, parity)
            actual_amplitudes = dict(zip(state.basis, state.state_vector, strict=True))
            actual_vector = np.array([actual_amplitudes.get((n,), 0j) for n in range(CUTOFF)])
            overlap = np.vdot(analytic, actual_vector)
            fidelity = float(abs(overlap) ** 2)

            expected_parity = float(parity)
            actual_parity = state.parity()

            rows.append(
                {
                    "alpha": alpha,
                    "parity_label": "even" if parity == 1 else "odd",
                    "fidelity_vs_independent_analytic": fidelity,
                    "expected_ideal_parity": expected_parity,
                    "measured_parity": actual_parity,
                    "mean_photon_number": state.photon_number(0),
                    "norm": state.norm,
                    "retained_norm": state.retained_norms[-1] if state.retained_norms else None,
                    "boundary_population": state.diagnostics[-1]["boundary_population"] if state.diagnostics else None,
                }
            )
            if alpha == REPRESENTATIVE_ALPHA:
                wigner_states[parity] = state

    fidelity_tolerance = 1e-6
    parity_sign_failures = [r for r in rows if np.sign(r["measured_parity"]) != np.sign(r["expected_ideal_parity"])]
    fidelity_failures = [r for r in rows if r["fidelity_vs_independent_analytic"] < 1 - fidelity_tolerance]
    if parity_sign_failures or fidelity_failures:
        common.save_csv(parity_sign_failures + fidelity_failures, "R21_cat_state_FAILURES")
        raise AssertionError("Cat state parity sign or independent-fidelity check failed; see FAILURES csv")

    common.save_result(rows, "R21_cat_state", extra={"cutoff": CUTOFF})

    q_axis = np.linspace(-6, 6, 121)
    p_axis = np.linspace(-6, 6, 121)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (parity, label) in zip(axes, ((1, "Even cat"), (-1, "Odd cat")), strict=True):
        grid = wigner_states[parity].wigner(q_axis, p_axis, node=0)
        extent = max(abs(grid.values.min()), abs(grid.values.max()))
        mesh = ax.pcolormesh(grid.q, grid.p, grid.values, shading="auto", cmap="RdBu_r", vmin=-extent, vmax=extent)
        ax.set(xlabel="q", ylabel="p", aspect="equal", title=f"{label}, alpha={REPRESENTATIVE_ALPHA}")
        fig.colorbar(mesh, ax=ax, fraction=0.046)
    fig.suptitle("Cat-state Wigner functions")
    common.save_figure(fig, "R21_cat_state")
    plt.close(fig)

    common.print_summary(
        "R21 cat state",
        cases=len(rows),
        min_fidelity_to_analytic=min(r["fidelity_vs_independent_analytic"] for r in rows),
        all_parity_signs_correct=not parity_sign_failures,
    )


if __name__ == "__main__":
    main()
