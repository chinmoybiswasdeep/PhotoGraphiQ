"""R31: Pure-state QFI against 4*Var(H), evaluated via an independent
density-matrix trace formula.

Scientific question: Does photographiqml.diagnostics.pure_qfi (computed via
a direct state/image inner-product route) agree with the standard
4*Var(H) = 4*(Tr(rho H^2) - Tr(rho H)^2) formula evaluated via an
independently coded density-matrix trace route, for random states and
generators?

Theory/equations: for a pure state, the quantum Fisher information for
generator H is F = 4*Var(H); pure_qfi's own docstring already states this
is the formula it implements via image=H@state and
4*(<image|image> - <state|image>^2). This experiment evaluates the
identical quantity via rho=|psi><psi|, Tr(rho H), Tr(rho H^2) instead, a
different (if mathematically equivalent) computational route.

Functionality tested: photographiqml.diagnostics.pure_qfi.

Oracle and independence class: A (independent analytic route -- density-
matrix trace formula, not the state/image inner-product shortcut).

Exact/approximate/statistical status: exact, up to floating-point roundoff.

Primary metric: |pure_qfi - 4*Var(H)_via_trace| over random states and
random Hermitian generators, at 1, 2 and 3 qubits.

Declared acceptance condition: max error < tol
(tol = declare_tolerance(scale=4, safety_factor=100), scale=4 since the QFI
formula carries an overall factor of 4).

Expected cost: light.

Manuscript destination: Appendix (diagnostics validation table, companion
to R30).

Scientific limitations: Pure states only (matches pure_qfi's documented
scope).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml.diagnostics import pure_qfi
from photographiqml.models import haar_states

EXPERIMENT_ID = "R31"


def trace_qfi(state, generator):
    rho = np.outer(state, state.conj())
    mean_h = np.trace(rho @ generator).real
    mean_h2 = np.trace(rho @ generator @ generator).real
    return float(4 * (mean_h2 - mean_h**2))


def random_hermitian(n, seed):
    generator = np.random.default_rng(seed)
    m = generator.normal(size=(n, n)) + 1j * generator.normal(size=(n, n))
    return m + m.conj().T


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=4.0, safety_factor=100.0)
    rows = []
    for n_qubits in (1, 2, 3):
        dim = 2**n_qubits
        states = haar_states(n_qubits, 10, seed=n_qubits)
        for i, state in enumerate(states):
            observable = random_hermitian(dim, seed=100 * n_qubits + i)
            production = pure_qfi(state, observable)
            independent = trace_qfi(state, observable)
            rows.append(
                {
                    "n_qubits": n_qubits,
                    "sample": i,
                    "production": production,
                    "independent": independent,
                    "error": abs(production - independent),
                }
            )

    max_error = max(r["error"] for r in rows)
    status = "pass" if max_error < tol else "fail"

    common.save_result(
        rows,
        "R31_qfi",
        extra={
            "protocol": "diagnostics.pure_qfi vs. independent density-matrix trace formula for 4*Var(H)",
            "oracle_class": "A",
            "status_category": "exact",
            "tolerance": tol,
            "acceptance_condition": f"max error < {tol:.3e}",
            "max_error": max_error,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A", "status": status},
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for n_qubits in (1, 2, 3):
        errs = [r["error"] for r in rows if r["n_qubits"] == n_qubits]
        ax.semilogy(errs, "o-", alpha=0.8, label=f"n_qubits={n_qubits}")
    ax.axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    ax.set(
        title=f"R31: pure-state QFI vs. trace formula (status={status})",
        xlabel="sample",
        ylabel="|production - independent|",
    )
    ax.legend(fontsize=7)
    common.save_figure(fig, "R31_qfi")
    plt.close(fig)

    common.print_summary("R31 QFI", n_cases=len(rows), max_error=max_error, status=status)
    if status != "pass":
        raise AssertionError(f"R31 failed: max_error={max_error} tol={tol}")


if __name__ == "__main__":
    main()
