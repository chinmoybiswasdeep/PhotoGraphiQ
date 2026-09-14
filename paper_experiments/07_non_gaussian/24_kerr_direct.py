"""R24: Kerr direct execution.

Applies ``photographiq.commands.Kerr(kappa)`` (native Piquasso
``exp(i*kappa*n^2)``) to the explicit custom superposition
``(|0>+|1>+|2>)/sqrt(3)`` (``FockInput`` with arbitrary amplitudes) and
compares each output amplitude against the exact analytic phase rule
``|n> -> exp(i*kappa*n^2)|n>``, computed independently with NumPy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

KAPPAS = np.linspace(-2.0, 2.0, 17)
CUTOFF = 8


def main():
    plt = common.setup_style()
    amplitudes_in = np.array([1, 1, 1], dtype=complex) / np.sqrt(3)
    fock_input = pg.FockInput(tuple(amplitudes_in))

    rows = []
    for kappa in KAPPAS:
        pattern = pg.Pattern(inputs=(0,)).append(pg.Kerr(0, float(kappa))).append(pg.Output((0,))).validate()
        state = pg.simulate(pattern, backend="piquasso-fock", cutoff=CUTOFF, inputs={0: fock_input}, seed=0).state
        actual_map = dict(zip(state.basis, state.state_vector, strict=True))
        actual = np.array([actual_map.get((n,), 0j) for n in range(CUTOFF)])

        expected = np.zeros(CUTOFF, dtype=complex)
        for n in range(len(amplitudes_in)):
            expected[n] = amplitudes_in[n] * np.exp(1j * kappa * n**2)

        overlap = np.vdot(expected, actual)
        fidelity = float(np.clip(abs(overlap) ** 2, 0, 1))
        vector_error = float(np.linalg.norm(actual - expected))
        rows.append(
            {
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

    common.save_result(rows, "R24_kerr_direct", extra={"input_amplitudes": [[a.real, a.imag] for a in amplitudes_in], "cutoff": CUTOFF})

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.semilogy([r["kappa"] for r in rows], [max(1e-16, r["state_vector_error"]) for r in rows], "o-", color="#24677b")
    ax.set_xlabel(r"Kerr coefficient $\kappa$")
    ax.set_ylabel("State-vector error vs. exact phase rule")
    ax.set_title(r"Kerr on $(|0\rangle+|1\rangle+|2\rangle)/\sqrt{3}$")
    common.save_figure(fig, "R24_kerr_direct")
    plt.close(fig)

    common.print_summary(
        "R24 Kerr direct execution",
        kappas=len(rows),
        max_state_vector_error=max(r["state_vector_error"] for r in rows),
        min_fidelity=min(r["fidelity_to_exact_phase_rule"] for r in rows),
    )


if __name__ == "__main__":
    main()
