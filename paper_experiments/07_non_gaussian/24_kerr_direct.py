"""R24: Kerr direct execution.

Applies ``photographiq.commands.Kerr(kappa)`` (native Piquasso
``exp(i*kappa*n^2)``) to the explicit custom superposition
``(|0>+|1>+|2>)/sqrt(3)`` (``FockInput`` with arbitrary amplitudes) and
compares each output amplitude against the exact analytic phase rule
``|n> -> exp(i*kappa*n^2)|n>``, computed independently with NumPy.

Upgrade migration (2026-11): main engine ``native-fock`` (Kerr is an exact
diagonal phase there); ``piquasso-fock`` (historical) and the density-matrix
engine ``native-mixed-fock`` are run with the same criteria. For the mixed
engine the fidelity is <e|rho|e> and the "state-vector error" column holds the
Frobenius distance ||rho - |e><e|||.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

KAPPAS = np.linspace(-2.0, 2.0, 17)
CUTOFF = 8
ENGINES = ("native-fock", "piquasso-fock", "native-mixed-fock")
def main():
    plt = common.setup_style()
    amplitudes_in = np.array([1, 1, 1], dtype=complex) / np.sqrt(3)
    fock_input = pg.FockInput(tuple(amplitudes_in))

    rows = []
    for kappa, backend in itertools.product(KAPPAS, ENGINES):
        pattern = pg.Pattern(inputs=(0,)).append(pg.Kerr(0, float(kappa))).append(pg.Output((0,))).validate()
        state = pg.simulate(pattern, backend=backend, cutoff=CUTOFF, inputs={0: fock_input}, seed=0).state
        expected = np.zeros(CUTOFF, dtype=complex)
        for n in range(len(amplitudes_in)):
            expected[n] = amplitudes_in[n] * np.exp(1j * kappa * n**2)
        index = [state.basis.index((n,)) for n in range(CUTOFF)]
        if backend == "native-mixed-fock":
            rho = np.asarray(state.density_matrix)[np.ix_(index, index)]
            fidelity = float(np.clip(np.real(expected.conj() @ rho @ expected), 0, 1))
            vector_error = float(np.linalg.norm(rho - np.outer(expected, expected.conj())))
        else:
            actual = np.asarray(state.state_vector)[index]
            overlap = np.vdot(expected, actual)
            fidelity = float(np.clip(abs(overlap) ** 2, 0, 1))
            vector_error = float(np.linalg.norm(actual - expected))
        rows.append(
            {
                "backend": backend,
                "kappa": float(kappa),
                "fidelity_to_exact_phase_rule": fidelity,
                "state_vector_error": vector_error,
            }
        )

    tolerance_fidelity = 1 - 1e-10
    tolerance_vector = 1e-8
    failures = [r for r in rows if r["fidelity_to_exact_phase_rule"] < tolerance_fidelity or r["state_vector_error"] > tolerance_vector]
    if failures:
        common.save_csv(failures, "R24_kerr_direct_FAILURES")
        raise AssertionError(f"{len(failures)} Kerr cases disagreed with the exact phase rule; see FAILURES csv")

    common.save_result(rows, "R24_kerr_direct", extra={"input_amplitudes": [[a.real, a.imag] for a in amplitudes_in],
                                                        "cutoff": CUTOFF, "engines": list(ENGINES), "main_backend": ENGINES[0]})

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for backend, style in zip(ENGINES, ("o-", "x--", "s:"), strict=True):
        subset = [r for r in rows if r["backend"] == backend]
        ax.semilogy([r["kappa"] for r in subset], [max(1e-17, r["state_vector_error"]) for r in subset], style, label=backend)
    ax.legend(fontsize=7)
    ax.set_xlabel(r"Kerr coefficient $\kappa$")
    ax.set_ylabel("State-vector error vs. exact phase rule")
    ax.set_title(r"Kerr on $(|0\rangle+|1\rangle+|2\rangle)/\sqrt{3}$")
    common.save_figure(fig, "R24_kerr_direct")
    plt.close(fig)

    common.print_summary(
        "R24 Kerr direct execution",
        kappas=len(KAPPAS),
        **{f"{b}_max_state_vector_error": max(r["state_vector_error"] for r in rows if r["backend"] == b) for b in ENGINES},
        min_fidelity=min(r["fidelity_to_exact_phase_rule"] for r in rows),
    )


if __name__ == "__main__":
    main()
