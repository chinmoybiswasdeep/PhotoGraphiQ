"""R3: MuTA entangling identity against an independent SciPy matrix exponential.

Scientific question: Does the pivot-wire triangle's cross-coupling (via c1
measurement of the pivot wire, in a two-wire one-layer model) implement the
exact entangling gate exp(i*phi*XX/2) that Appendix B's coupling identity
claims, for arbitrary coupling angle phi?

Theory/equations: For MuTA(2, one_column=True) with only "alpha.w1.c1" bound
(pivot=0, all other angles left at their default 0), U = exp(i*phi*(X⊗X)/2)
(the Ising-XX two-qubit entangling unitary).

Functionality tested: MuTA.unitary for the two-wire, one-layer, one_column
triangle's cross-edge coupling.

Oracle and independence class: A (independent analytic oracle) -- the target
unitary is built via scipy.linalg.expm of a Pauli-XX generator constructed
locally, entirely independent of photographiqml.logical's cz/local_gate
implementation.

Exact/approximate/statistical status: exact, up to floating-point roundoff.

Primary metric: max Frobenius-norm error between MuTA.unitary and
exp(i*phi*XX/2) over a dense phi grid, plus the induced concurrence of the
first output column compared to the closed-form |sin(phi)|.

Declared acceptance condition: max Frobenius error < tol
(tol = declare_tolerance(scale=1)); max concurrence error < tol.

Expected cost: light.

Manuscript destination: Main text (Table I entangling-gate identity, Fig. 1).

Scientific limitations: Restricted to the minimal two-wire, one-layer
triangle; does not test coupling composed with additional trainable local
rotations on other columns (R1/R5 cover broader composition).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np
from scipy.linalg import expm

from photographiqml import MuTA
from photographiqml.diagnostics import concurrence

EXPERIMENT_ID = "R3"

X_REF = np.array([[0, 1], [1, 0]], dtype=complex)


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0)
    phis = np.linspace(-2 * np.pi, 2 * np.pi, 61)
    model = MuTA(2, one_column=True)
    rows = []
    for phi in phis:
        actual = model.unitary({"alpha.w1.c1": float(phi)})
        expected = expm(0.5j * phi * np.kron(X_REF, X_REF))
        error = common.frobenius_error(actual, expected)
        actual_concurrence = concurrence(actual[:, 0])
        expected_concurrence = abs(np.sin(phi))
        rows.append(
            {
                "phi": float(phi),
                "unitary_error": error,
                "actual_concurrence": actual_concurrence,
                "expected_concurrence": expected_concurrence,
                "concurrence_error": abs(actual_concurrence - expected_concurrence),
            }
        )

    max_unitary_error = max(r["unitary_error"] for r in rows)
    max_concurrence_error = max(r["concurrence_error"] for r in rows)
    status = "pass" if max(max_unitary_error, max_concurrence_error) < tol else "fail"

    common.save_result(
        rows,
        "R3_entangling_identity",
        extra={
            "protocol": "Two-wire pivot coupling vs. exp(i*phi*XX/2) (scipy.linalg.expm)",
            "oracle_class": "A",
            "status_category": "exact",
            "tolerance": tol,
            "acceptance_condition": f"max Frobenius error and max concurrence error < {tol:.3e}",
            "max_unitary_error": max_unitary_error,
            "max_concurrence_error": max_concurrence_error,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].semilogy(
        phis,
        [r["unitary_error"] for r in rows],
        color=common.COLORS["photographiqml"],
        marker="o",
        markersize=2,
    )
    axes[0].axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    axes[0].set(title="exp(i*phi*XX/2) Frobenius error", xlabel="phi (rad)", ylabel="error")
    axes[0].legend()
    axes[1].plot(
        phis,
        [r["expected_concurrence"] for r in rows],
        color=common.COLORS["analytic"],
        label="analytic |sin(phi)|",
    )
    axes[1].plot(
        phis,
        [r["actual_concurrence"] for r in rows],
        "--",
        color=common.COLORS["photographiqml"],
        label="MuTA output concurrence",
    )
    axes[1].set(title="Induced concurrence", xlabel="phi (rad)", ylabel="concurrence")
    axes[1].legend()
    fig.suptitle(f"R3: Ising-XX entangling identity (status={status})")
    common.save_figure(fig, "R3_entangling_identity")
    plt.close(fig)

    common.print_summary(
        "R3 entangling identity",
        n_points=len(phis),
        tol=tol,
        max_unitary_error=max_unitary_error,
        max_concurrence_error=max_concurrence_error,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R3 failed: unitary={max_unitary_error} concurrence={max_concurrence_error}"
        )


if __name__ == "__main__":
    main()
