"""R2: Single-wire Table-I identities against hand-written analytic 2x2 unitaries.

Scientific question: Does a single-wire MuTA paper layer implement the exact
Euler-angle single-qubit rotation sequence Table I claims for its three
trainable intermediate measurement columns (c1, c2, c3)?

Theory/equations: each measured column applies gate(a) = H @ Rz(a), with
Rz(a) = exp(i*a*Z/2). Using H Rz(t) H = Rx(t) to push each H rightward through
the chain gate(c3) gate(c2) gate(c1) gate(c0), the four intermediate H's
telescope pairwise to identity, giving the single-wire Euler identity
U = Rx(c3) @ Rz(c2) @ Rx(c1) @ Rz(c0) (alternating X,Z,X,Z from right to
left), an independent derivation of Table I's per-column rotation axes.
Column c0 is the input node itself (also measured, per the model's
"every non-output node is measured" convention) and is swept too, to confirm
its Z-axis sub-identity in isolation alongside c1/c2/c3's X/Z/X axes.

Functionality tested: MuTA.unitary via ansatz/muta.py + logical.execute, for
n_wires=1.

Oracle and independence class: A (independent analytic oracle) -- unitaries
are built directly from scipy.linalg.expm of Pauli matrices defined locally
in this script (not imported from photographiqml.logical), independent of the
package's own H/X/Z arrays and local_gate/cz implementation.

Exact/approximate/statistical status: exact, up to floating-point roundoff.

Primary metric: max Frobenius-norm error between MuTA.unitary and the
Table-I analytic prediction, over a grid of random angle triples.

Declared acceptance condition: max Frobenius error < the single canonical
``common.CANONICAL_R2_TOLERANCE`` (10 times binary64 epsilon), shared by the
script, index, contract, metadata, manifest and consistency tests.

Expected cost: light.

Manuscript destination: Main text (Table I verification, supporting Fig. 1).

Scientific limitations: Single-wire only; does not test entangling identities
(see R3) or multi-wire composition (see R1, R5).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np
from scipy.linalg import expm

from photographiqml import MuTA

EXPERIMENT_ID = "R2"

X_REF = np.array([[0, 1], [1, 0]], dtype=complex)
Z_REF = np.diag([1, -1]).astype(complex)


def main():
    plt = common.setup_style()
    tol = common.CANONICAL_R2_TOLERANCE
    generator = common.rng(0)
    n_trials = 40
    rows = []
    model = MuTA(1, 1)

    for trial in range(n_trials):
        theta, phi, lam = generator.uniform(-2 * np.pi, 2 * np.pi, size=3)
        c0 = generator.uniform(-2 * np.pi, 2 * np.pi)
        angles = {
            "alpha.w0.c0": c0,
            "alpha.w0.c1": theta,
            "alpha.w0.c2": phi,
            "alpha.w0.c3": lam,
        }
        actual = model.unitary(angles)
        # U = Rx(c3) @ Rz(c2) @ Rx(c1) @ Rz(c0), derived by telescoping the
        # H-Rz-H = Rx conjugation identity through the 4-column measurement chain.
        expected = (
            expm(0.5j * lam * X_REF)
            @ expm(0.5j * phi * Z_REF)
            @ expm(0.5j * theta * X_REF)
            @ expm(0.5j * c0 * Z_REF)
        )
        error = common.frobenius_error(actual, expected)
        # Isolated single-column sub-identities (freeze the other three at 0).
        single_errors = {}
        for name, angle, ref in (
            ("c0", c0, Z_REF),
            ("c1", theta, X_REF),
            ("c2", phi, Z_REF),
            ("c3", lam, X_REF),
        ):
            isolated = dict.fromkeys(
                ("alpha.w0.c0", "alpha.w0.c1", "alpha.w0.c2", "alpha.w0.c3"), 0.0
            )
            isolated[f"alpha.w0.{name}"] = angle
            single_errors[f"{name}_error"] = common.frobenius_error(
                model.unitary(isolated), expm(0.5j * angle * ref)
            )
        rows.append(
            {
                "trial": trial,
                "theta": theta,
                "phi": phi,
                "lam": lam,
                "c0": c0,
                "full_sequence_error": error,
                **single_errors,
            }
        )

    max_error = max(r["full_sequence_error"] for r in rows)
    max_single_error = max(
        max(r[k] for k in ("c0_error", "c1_error", "c2_error", "c3_error")) for r in rows
    )
    status = "pass" if max(max_error, max_single_error) < tol else "fail"

    common.save_result(
        rows,
        "R2_table_one_identities",
        extra={
            "protocol": "Single-wire Table-I Euler decomposition vs. scipy.linalg.expm reference",
            "oracle_class": "A",
            "status_category": "exact",
            "tolerance": tol,
            "acceptance_condition": f"max Frobenius error < {tol:.3e}",
            "max_full_sequence_error": max_error,
            "max_single_column_error": max_single_error,
            "status": status,
        },
        meta_extra={
            "experiment_id": EXPERIMENT_ID,
            "oracle_class": "A",
            "status": status,
            "tolerance": common.CANONICAL_R2_TOLERANCE,
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].semilogy(
        [r["full_sequence_error"] for r in rows], "o-", color=common.COLORS["photographiqml"]
    )
    axes[0].axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    axes[0].set(title="Full 3-angle Euler sequence", xlabel="trial", ylabel="Frobenius error")
    axes[0].legend()
    for key in ("c0_error", "c1_error", "c2_error", "c3_error"):
        axes[1].semilogy([r[key] for r in rows], "o", markersize=3, label=key, alpha=0.7)
    axes[1].axhline(tol, color=common.COLORS["acceptance"], linestyle="--")
    axes[1].set(title="Isolated single-column identities", xlabel="trial", ylabel="Frobenius error")
    axes[1].legend(fontsize=6.5)
    fig.suptitle(f"R2: Table-I identities vs. analytic 2x2 unitaries (status={status})")
    common.save_figure(fig, "R2_table_one_identities")
    plt.close(fig)

    common.print_summary(
        "R2 Table-I identities", n_trials=n_trials, tol=tol, max_error=max_error, status=status
    )
    if status != "pass":
        raise AssertionError(f"R2 failed: max_error={max_error} tol={tol}")


if __name__ == "__main__":
    main()
