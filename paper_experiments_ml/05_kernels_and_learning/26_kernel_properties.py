"""R26: Kernel symmetry, unit diagonal, PSD spectrum, and conditioning as
sample size changes.

Scientific question: Does MuTAKernel.gram_matrix always produce a
symmetric, unit-diagonal, positive-semidefinite Gram matrix (as any valid
fidelity-based kernel k(x,y)=|<psi(x)|psi(y)>|^2 must, by construction), and
how does its numerical conditioning behave as sample size grows?

Theory/equations: for a fidelity kernel, K = Phi^dagger Phi where Phi's
columns are pure quantum feature states, so K is Hermitian (real symmetric
here since entries are |.|^2) and PSD by construction (eigenvalues are
squared singular values of Phi); K_ii = |<psi(x_i)|psi(x_i)>|^2 = 1 exactly.
These are analytic invariants of any fidelity kernel, not an empirical
property specific to this implementation.

Functionality tested: MuTAKernel.gram_matrix/diagnostics (photographiqml.kernels).

Oracle and independence class: A (independent analytic invariant -- PSD/
symmetric/unit-diagonal are theorems about fidelity kernels, checked against
the concrete numerical Gram matrix).

Exact/approximate/statistical status: exact (symmetry, diagonal), up to
floating-point roundoff for the PSD spectrum (a numerically tiny negative
eigenvalue at the roundoff floor is expected and not treated as a PSD
violation).

Primary metric: symmetry_error, diagonal_error, minimum_eigenvalue,
condition_number (max/min eigenvalue, min clipped at the roundoff floor for
the ratio only), each vs. sample size N.

Declared acceptance condition: symmetry_error and diagonal_error < tol
(tol = declare_tolerance(scale=1,safety_factor=100)); minimum_eigenvalue >
-tol for every N (allows roundoff-scale negativity, not a real PSD
violation).

Expected cost: light.

Manuscript destination: Main text (Fig. 4, kernel-properties panel).

Scientific limitations: Conditioning is reported descriptively (no claim
that any particular condition number is "good" for downstream SVM use;
that is R27/R28's job).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import MuTAKernel

EXPERIMENT_ID = "R26"


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=100.0)
    kernel = MuTAKernel()
    generator = common.rng(2)
    rows = []
    for n_samples in (2, 4, 8, 16, 32, 64):
        X = generator.uniform(-2 * np.pi, 2 * np.pi, size=(n_samples, 2))
        diagnostics = kernel.diagnostics(X)
        eigenvalues = np.linalg.eigvalsh(kernel.gram_matrix(X))
        min_eig = float(eigenvalues.min())
        max_eig = float(eigenvalues.max())
        condition_number = float(max_eig / max(min_eig, np.finfo(float).eps))
        rows.append(
            {
                "n_samples": n_samples,
                "symmetry_error": diagnostics["symmetry_error"],
                "diagonal_error": diagnostics["diagonal_error"],
                "minimum_eigenvalue": min_eig,
                "maximum_eigenvalue": max_eig,
                "condition_number": condition_number,
            }
        )

    max_symmetry_error = max(r["symmetry_error"] for r in rows)
    max_diagonal_error = max(r["diagonal_error"] for r in rows)
    min_eigenvalue_overall = min(r["minimum_eigenvalue"] for r in rows)
    status = (
        "pass"
        if (max_symmetry_error < tol and max_diagonal_error < tol and min_eigenvalue_overall > -tol)
        else "fail"
    )

    common.save_result(
        rows,
        "R26_kernel_properties",
        extra={
            "protocol": "MuTAKernel Gram-matrix symmetry/diagonal/PSD/conditioning vs. sample size",
            "oracle_class": "A",
            "status_category": "exact",
            "tolerance": tol,
            "acceptance_condition": f"max symmetry/diagonal error < {tol:.3e}; minimum eigenvalue > -{tol:.3e}",
            "max_symmetry_error": max_symmetry_error,
            "max_diagonal_error": max_diagonal_error,
            "min_eigenvalue_overall": min_eigenvalue_overall,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ns = [r["n_samples"] for r in rows]
    axes[0].semilogy(
        ns,
        [max(r["symmetry_error"], 1e-18) for r in rows],
        "o-",
        label="symmetry error",
        color=common.COLORS["photographiqml"],
    )
    axes[0].semilogy(
        ns,
        [max(r["diagonal_error"], 1e-18) for r in rows],
        "s--",
        label="diagonal error",
        color=common.COLORS["mentpy"],
    )
    axes[0].axhline(tol, color=common.COLORS["acceptance"], linestyle=":", label=f"tol={tol:.1e}")
    axes[0].set(title="Symmetry / unit-diagonal error", xlabel="n_samples", ylabel="error")
    axes[0].legend(fontsize=7)
    axes[1].semilogy(
        ns, [r["condition_number"] for r in rows], "o-", color=common.COLORS["photographiqml"]
    )
    axes[1].set(title="Gram-matrix condition number", xlabel="n_samples", ylabel="cond(K)")
    fig.suptitle(f"R26: kernel properties vs. sample size (status={status})")
    common.save_figure(fig, "R26_kernel_properties")
    plt.close(fig)

    common.print_summary(
        "R26 kernel properties",
        n_configs=len(rows),
        max_symmetry_error=max_symmetry_error,
        min_eigenvalue=min_eigenvalue_overall,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R26 failed: symmetry={max_symmetry_error} diagonal={max_diagonal_error} min_eig={min_eigenvalue_overall}"
        )


if __name__ == "__main__":
    main()
