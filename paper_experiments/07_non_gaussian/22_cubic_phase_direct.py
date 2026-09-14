"""R22: Cubic phase direct execution.

Applies ``photographiq.commands.CubicPhase(gamma)`` (native Piquasso
``exp(i*gamma*q^3/6)``, hbar=2) to a low-energy coherent input on the pure
Fock backend, and compares the resulting truncated state vector against an
independently built finite-dimensional reference: a dense ladder operator
``a`` constructed directly with NumPy, ``q = a + a^T``, and
``U = scipy.linalg.expm(i*gamma*q^3/6)`` applied to the SAME truncated input
vector. This is a finite-Fock-truncation comparison at matched cutoff, not an
infinite-cutoff convergence claim (see R33/R34 for that).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

GAMMAS = [0.02, 0.05, 0.1, 0.2]
CUTOFFS = [16, 24, 32, 48]
ALPHA = 0.3 + 0.1j


def independent_cubic_unitary(gamma: float, cutoff: int) -> np.ndarray:
    """Independent dense NumPy/SciPy reference: exp(i*gamma*q^3/6) at fixed cutoff."""
    a = np.diag(np.sqrt(np.arange(1, cutoff)), k=1)
    q = a + a.T
    return expm(1j * gamma * (q @ q @ q) / 6)


def truncated_input_vector(cutoff: int) -> np.ndarray:
    pattern = pg.Pattern(inputs=(0,)).append(pg.Output((0,))).validate()
    state = pg.simulate(
        pattern, backend="piquasso-fock", cutoff=cutoff, inputs={0: pg.GaussianInput.coherent(ALPHA)}, seed=0
    ).state
    amplitudes = dict(zip(state.basis, state.state_vector, strict=True))
    return np.array([amplitudes.get((n,), 0j) for n in range(cutoff)])


def main():
    plt = common.setup_style()
    rows = []
    for cutoff in CUTOFFS:
        before = truncated_input_vector(cutoff)
        for gamma in GAMMAS:
            pattern = pg.Pattern(inputs=(0,)).append(pg.CubicPhase(0, gamma)).append(pg.Output((0,))).validate()
            state = pg.simulate(
                pattern, backend="piquasso-fock", cutoff=cutoff, inputs={0: pg.GaussianInput.coherent(ALPHA)}, seed=0
            ).state
            amplitudes = dict(zip(state.basis, state.state_vector, strict=True))
            actual = np.array([amplitudes.get((n,), 0j) for n in range(cutoff)])

            unitary = independent_cubic_unitary(gamma, cutoff)
            predicted = unitary @ before
            predicted /= np.linalg.norm(predicted)
            overlap = np.vdot(predicted, actual)
            fidelity = float(np.clip(abs(overlap) ** 2, 0, 1))
            state_vector_norm_error = float(np.linalg.norm(actual - predicted * (overlap.conjugate() / abs(overlap))))

            rows.append(
                {
                    "gamma": gamma,
                    "cutoff": cutoff,
                    "fidelity_to_independent_reference": fidelity,
                    "state_vector_norm_error_phase_aligned": state_vector_norm_error,
                    "trace_distance_estimate": float(np.sqrt(max(0.0, 1 - fidelity))),
                    "retained_norm": state.retained_norms[-1] if state.retained_norms else None,
                }
            )

    common.save_result(rows, "R22_cubic_phase_direct", extra={"alpha": [ALPHA.real, ALPHA.imag]})

    fig, ax = plt.subplots(figsize=(6, 4))
    for gamma in GAMMAS:
        subset = [r for r in rows if r["gamma"] == gamma]
        ax.semilogy(
            [r["cutoff"] for r in subset], [max(1e-16, 1 - r["fidelity_to_independent_reference"]) for r in subset],
            "o-", label=f"gamma={gamma}",
        )
    ax.set_xlabel("Fock cutoff")
    ax.set_ylabel("1 - fidelity (independent reference)")
    ax.legend(fontsize=8)
    ax.set_title(f"Cubic phase, coherent input alpha={ALPHA}")
    common.save_figure(fig, "R22_cubic_phase_direct")
    plt.close(fig)

    common.print_summary(
        "R22 cubic phase direct execution",
        cases=len(rows),
        min_fidelity=min(r["fidelity_to_independent_reference"] for r in rows),
        max_fidelity=max(r["fidelity_to_independent_reference"] for r in rows),
    )


if __name__ == "__main__":
    main()
