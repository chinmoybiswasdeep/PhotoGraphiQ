"""R23: Cubic MBQC injection (finite-resource approximation -- clearly labelled).

Uses ``photographiq.non_gaussian.cubic_injection(gamma, squeezing=r)`` with a
FIXED ancilla homodyne outcome m=0 (so the nonlinear feed-forward corrections
QuadraticPhase(-2*gamma*m) and Displace(p=-gamma*m^2) vanish identically,
isolating the residual finite-resource Gaussian envelope). The result is
compared against an ideal direct CubicPhase(gamma) applied to the same
truncated input, using the same independent NumPy/SciPy matrix-exponential
reference as R22. Per docs/non_gaussian.md this injection is a conditional
filter times a cubic phase, NOT an exact finite-energy unitary cubic gate;
fidelity to the ideal gate is expected to improve as resource squeezing r
grows (the envelope widens toward the infinite-squeezing/no-filter limit).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

GAMMA = 0.15
ALPHA = 0.2 + 0.0j
SQUEEZING_VALUES = [0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
CUTOFFS = [24, 32, 48]
FIXED_OUTCOME = 0.0


def independent_cubic_unitary(gamma: float, cutoff: int) -> np.ndarray:
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


def vector_from_state(state, node, cutoff):
    amplitudes = dict(zip(state.basis, state.state_vector, strict=True))
    return np.array([amplitudes.get((n,), 0j) for n in range(cutoff)])


def main():
    plt = common.setup_style()
    rows = []
    for cutoff in CUTOFFS:
        before = truncated_input_vector(cutoff)
        ideal_after = independent_cubic_unitary(GAMMA, cutoff) @ before
        ideal_after /= np.linalg.norm(ideal_after)

        for r in SQUEEZING_VALUES:
            pattern = pg.non_gaussian.cubic_injection(GAMMA, squeezing=r)
            failure = common.safe_cutoff_run(
                lambda: pg.simulate(
                    pattern, backend="piquasso-fock", cutoff=cutoff,
                    inputs={"in": pg.GaussianInput.coherent(ALPHA)},
                    measurement_outcomes={"m": FIXED_OUTCOME}, seed=0,
                ),
                label=f"cutoff={cutoff} r={r}",
            )
            if not failure["ok"]:
                rows.append(
                    {"gamma": GAMMA, "resource_squeezing": r, "cutoff": cutoff, "ok": False, "error": failure["error"]}
                )
                continue
            result = failure["value"]
            actual = vector_from_state(result.state, "in", cutoff)
            overlap = np.vdot(ideal_after, actual)
            fidelity = float(np.clip(abs(overlap) ** 2, 0, 1))
            rows.append(
                {
                    "gamma": GAMMA,
                    "resource_squeezing": r,
                    "cutoff": cutoff,
                    "ok": True,
                    "fidelity_to_ideal_cubic_phase": fidelity,
                    "trace_distance_estimate": float(np.sqrt(max(0.0, 1 - fidelity))),
                    "outcome_density": result.measurement_statistics["m"]["value"],
                    "outcome_kind": result.measurement_statistics["m"]["kind"],
                    "retained_norm": result.state.retained_norms[-1] if result.state.retained_norms else None,
                }
            )

    common.save_result(rows, "R23_cubic_injection", extra={"gamma": GAMMA, "alpha": [ALPHA.real, ALPHA.imag], "fixed_outcome": FIXED_OUTCOME})

    ok_rows = [r for r in rows if r["ok"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    for cutoff in CUTOFFS:
        subset = [r for r in ok_rows if r["cutoff"] == cutoff]
        ax.plot([r["resource_squeezing"] for r in subset], [r["fidelity_to_ideal_cubic_phase"] for r in subset], "o-", label=f"cutoff={cutoff}")
    ax.set_xlabel("Resource squeezing r")
    ax.set_ylabel("Fidelity to ideal CubicPhase(gamma)")
    ax.legend(fontsize=8)
    ax.set_title(f"Finite-resource cubic injection, gamma={GAMMA} (m fixed at 0)")
    common.save_figure(fig, "R23_cubic_injection")
    plt.close(fig)

    common.print_summary(
        "R23 cubic MBQC injection (finite-resource approximation)",
        cases=len(rows),
        ok=len(ok_rows),
        min_fidelity=min((r["fidelity_to_ideal_cubic_phase"] for r in ok_rows), default=None),
        max_fidelity=max((r["fidelity_to_ideal_cubic_phase"] for r in ok_rows), default=None),
    )


if __name__ == "__main__":
    main()
