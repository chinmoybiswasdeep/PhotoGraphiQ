"""R30: Two-qubit concurrence against the closed-form Wootters spin-flip
formula.

Scientific question: Does photographiqml.diagnostics.concurrence (which
computes 2|ad-bc| directly from state amplitudes (a,b,c,d)) agree with the
independently coded Wootters spin-flip formula
C = |<psi| (sigma_y (x) sigma_y) |psi*>|, over random states and the known
separable/maximally-entangled boundary cases?

Theory/equations: for a pure two-qubit state, both formulas are proven
equal in closed form; this experiment evaluates them via two different
numerical routes (direct amplitude combination vs. an explicit sigma_y⊗sigma_y
matrix contraction) as a cross-check, not a re-derivation of the theorem.

Functionality tested: photographiqml.diagnostics.concurrence.

Oracle and independence class: A (independent analytic oracle -- the
spin-flip formula is evaluated via fresh matrix construction, not by
reusing the 2|ad-bc| shortcut).

Exact/approximate/statistical status: exact, up to floating-point roundoff.

Primary metric: |concurrence - spin_flip_concurrence| over random states;
absolute values at known separable (concurrence=0) and Bell
(concurrence=1) states.

Declared acceptance condition: max error < tol
(tol = declare_tolerance(scale=1, safety_factor=100)); separable-state
concurrence < tol; Bell-state concurrence within tol of 1.

Expected cost: light.

Manuscript destination: Appendix (diagnostics validation table).

Scientific limitations: Pure-state two-qubit concurrence only (matches the
package's own documented scope; not a general entanglement measure).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml.diagnostics import concurrence
from photographiqml.models import haar_states

EXPERIMENT_ID = "R30"

SIGMA_Y = np.array([[0, -1j], [1j, 0]])
YY = np.kron(SIGMA_Y, SIGMA_Y)


def spin_flip_concurrence(state):
    state = np.asarray(state, dtype=complex)
    return float(abs(state.conj() @ YY @ state.conj()))


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=100.0)
    rows = []
    states = haar_states(2, 40, seed=5)
    for i, state in enumerate(states):
        production = concurrence(state)
        independent = spin_flip_concurrence(state)
        rows.append(
            {
                "case": f"random_{i}",
                "production": production,
                "independent": independent,
                "error": abs(production - independent),
            }
        )

    separable = np.kron([1, 0], [1, 0])
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    for name, state in (("separable_00", separable), ("bell_phi_plus", bell)):
        production = concurrence(state)
        independent = spin_flip_concurrence(state)
        rows.append(
            {
                "case": name,
                "production": production,
                "independent": independent,
                "error": abs(production - independent),
            }
        )

    max_error = max(r["error"] for r in rows)
    separable_value = next(r["production"] for r in rows if r["case"] == "separable_00")
    bell_value = next(r["production"] for r in rows if r["case"] == "bell_phi_plus")
    status = (
        "pass"
        if (max_error < tol and separable_value < tol and abs(bell_value - 1) < tol)
        else "fail"
    )

    common.save_result(
        rows,
        "R30_concurrence",
        extra={
            "protocol": "diagnostics.concurrence vs. independent Wootters spin-flip formula",
            "oracle_class": "A",
            "status_category": "exact",
            "tolerance": tol,
            "acceptance_condition": f"max error < {tol:.3e}; separable concurrence < tol; Bell concurrence within tol of 1",
            "max_error": max_error,
            "separable_value": separable_value,
            "bell_value": bell_value,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].semilogy(
        [r["error"] for r in rows if r["case"].startswith("random")],
        "o",
        color=common.COLORS["photographiqml"],
    )
    axes[0].axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    axes[0].set(
        title="Random-state agreement", xlabel="sample", ylabel="|production - independent|"
    )
    axes[0].legend()
    axes[1].bar(
        ["separable", "Bell"], [separable_value, bell_value], color=common.COLORS["photographiqml"]
    )
    axes[1].axhline(0, color=common.COLORS["acceptance"], linewidth=0.8)
    axes[1].axhline(1, color=common.COLORS["acceptance"], linewidth=0.8)
    axes[1].set(title="Known boundary cases", ylabel="concurrence")
    fig.suptitle(f"R30: concurrence vs. spin-flip formula (status={status})")
    common.save_figure(fig, "R30_concurrence")
    plt.close(fig)

    common.print_summary("R30 concurrence", n_cases=len(rows), max_error=max_error, status=status)
    if status != "pass":
        raise AssertionError(
            f"R30 failed: max_error={max_error} separable={separable_value} bell={bell_value}"
        )


if __name__ == "__main__":
    main()
