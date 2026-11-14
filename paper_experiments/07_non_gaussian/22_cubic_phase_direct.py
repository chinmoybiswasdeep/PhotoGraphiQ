"""R22: Cubic phase direct execution.

Applies ``photographiq.commands.CubicPhase(gamma)`` (native Piquasso
``exp(i*gamma*q^3/6)``, hbar=2) to a low-energy coherent input on the pure
Fock backend, and compares the resulting truncated state vector against an
independently built finite-dimensional reference: a dense ladder operator
``a`` constructed directly with NumPy, ``q = a + a^T``, and
``U = scipy.linalg.expm(i*gamma*q^3/6)`` applied to the SAME truncated input
vector. This is a finite-Fock-truncation comparison at matched cutoff, not an
infinite-cutoff convergence claim (see R33/R34 for that).

Upgrade migration (2026-11): main engine ``native-fock`` with its default
cubic-phase definition ``truncated_generator`` (expm of the truncated
generator, the same finite-dimensional operator as the reference, so the
fidelity must be 1 to rounding); ``piquasso-fock`` is kept as the historical
reference with the same criterion. The script previously had no acceptance
test; FIDELITY_TOLERANCE is now declared and asserted. The native
``cubic_phase="projected"`` definition (exact projected matrix elements,
reports leakage above the cutoff) is a DIFFERENT finite-cutoff operator: its
fidelity to the truncated-generator reference and its retained norm are
reported, not asserted, and converge as the cutoff grows (see R34).
"""

from __future__ import annotations

import itertools
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
ENGINES = ("native-fock", "piquasso-fock")
FIDELITY_TOLERANCE = 1e-12  # same truncated operator; deviation is rounding only


def engine(name, cutoff):
    if name == "native-fock-projected":
        from photographiq.backends.native_fock import NativeFockBackend

        return NativeFockBackend(cutoff, cubic_phase="projected")
    return name
def independent_cubic_unitary(gamma: float, cutoff: int) -> np.ndarray:
    """Independent dense NumPy/SciPy reference: exp(i*gamma*q^3/6) at fixed cutoff."""
    a = np.diag(np.sqrt(np.arange(1, cutoff)), k=1)
    q = a + a.T
    return expm(1j * gamma * (q @ q @ q) / 6)


def truncated_input_vector(cutoff: int, backend: str = "piquasso-fock") -> np.ndarray:
    pattern = pg.Pattern(inputs=(0,)).append(pg.Output((0,))).validate()
    state = pg.simulate(
        pattern, backend=engine(backend, cutoff), cutoff=cutoff, inputs={0: pg.GaussianInput.coherent(ALPHA)}, seed=0
    ).state
    amplitudes = dict(zip(state.basis, state.state_vector, strict=True))
    return np.array([amplitudes.get((n,), 0j) for n in range(cutoff)])


def main():
    plt = common.setup_style()
    rows = []
    for cutoff, backend in itertools.product(CUTOFFS, (*ENGINES, "native-fock-projected")):
        before = truncated_input_vector(cutoff, backend)
        for gamma in GAMMAS:
            pattern = pg.Pattern(inputs=(0,)).append(pg.CubicPhase(0, gamma)).append(pg.Output((0,))).validate()
            state = pg.simulate(
                pattern, backend=engine(backend, cutoff), cutoff=cutoff, inputs={0: pg.GaussianInput.coherent(ALPHA)}, seed=0
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
                    "backend": backend,
                    "gamma": gamma,
                    "cutoff": cutoff,
                    "fidelity_to_independent_reference": fidelity,
                    "state_vector_norm_error_phase_aligned": state_vector_norm_error,
                    "trace_distance_estimate": float(np.sqrt(max(0.0, 1 - fidelity))),
                    "retained_norm": state.retained_norms[-1] if state.retained_norms else None,
                    "min_retained_norm": min(state.retained_norms) if state.retained_norms else None,
                }
            )

    failures = [r for r in rows if r["backend"] in ENGINES and r["fidelity_to_independent_reference"] < 1 - FIDELITY_TOLERANCE]
    common.save_result(rows, "R22_cubic_phase_direct", extra={
        "alpha": [ALPHA.real, ALPHA.imag], "asserted_engines": list(ENGINES), "fidelity_tolerance": FIDELITY_TOLERANCE,
        "reported_only": "native-fock-projected (different finite-cutoff operator)"})
    if failures:
        common.save_csv(failures, "R22_cubic_phase_direct_FAILURES")
        raise AssertionError("Cubic phase disagreed with the independent truncated-generator reference; see FAILURES csv")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    styles = {"native-fock": ("o-", 1.0), "piquasso-fock": ("x:", 1.0), "native-fock-projected": ("s--", 0.8)}
    colors = dict(zip(GAMMAS, ["#24677b", "#c66d27", "#627a36", "#7a3e9d"], strict=True))
    for backend, (style, alpha) in styles.items():
        for gamma in GAMMAS:
            subset = [r for r in rows if r["gamma"] == gamma and r["backend"] == backend]
            ax.semilogy(
                [r["cutoff"] for r in subset], [max(1e-16, 1 - r["fidelity_to_independent_reference"]) for r in subset],
                style, color=colors[gamma], alpha=alpha,
                label=f"{backend}, gamma={gamma}" if backend != "piquasso-fock" else ("piquasso-fock" if gamma == GAMMAS[0] else None),
            )
    ax.legend(fontsize=6, ncol=2)
    ax.set_ylim(5e-17, 1e-4)
    ax.set_xlabel("Fock cutoff")
    ax.set_ylabel("1 - fidelity (independent reference)")
    ax.set_title(f"Cubic phase, coherent input alpha={ALPHA} (1e-16 = rounding floor)", fontsize=9)
    common.save_figure(fig, "R22_cubic_phase_direct")
    plt.close(fig)

    common.print_summary(
        "R22 cubic phase direct execution",
        cases=len(rows),
        **{f"{b}_min_fidelity": min(r["fidelity_to_independent_reference"] for r in rows if r["backend"] == b)
           for b in (*ENGINES, "native-fock-projected")},
    )


if __name__ == "__main__":
    main()
