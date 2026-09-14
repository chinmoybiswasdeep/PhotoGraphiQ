"""R34: Cubic injection convergence -- 2D scan over cutoff and resource squeezing.

Two-dimensional grid of (Fock cutoff, resource squeezing) for
``photographiq.non_gaussian.cubic_injection`` with a fixed ancilla outcome
m=0 (isolating the residual finite-resource Gaussian envelope from outcome-
dependent feed-forward). Fidelity to an ideal direct CubicPhase(gamma),
computed independently with NumPy/SciPy, is saved as a heatmap-ready CSV.
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
CUTOFFS = [24, 32, 48, 64, 96]
SQUEEZING_VALUES = [0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
FIXED_OUTCOME = 0.0


def independent_cubic_unitary(gamma, cutoff):
    a = np.diag(np.sqrt(np.arange(1, cutoff)), k=1)
    q = a + a.T
    return expm(1j * gamma * (q @ q @ q) / 6)


def truncated_input_vector(cutoff):
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
                rows.append({"cutoff": cutoff, "resource_squeezing": r, "ok": False, "fidelity": None, "error": failure["error"]})
                continue
            result = failure["value"]
            amplitudes = dict(zip(result.state.basis, result.state.state_vector, strict=True))
            actual = np.array([amplitudes.get((n,), 0j) for n in range(cutoff)])
            fidelity = float(np.clip(abs(np.vdot(ideal_after, actual)) ** 2, 0, 1))
            rows.append(
                {
                    "cutoff": cutoff, "resource_squeezing": r, "ok": True, "fidelity": fidelity,
                    "log10_infidelity": common.safe_log10(1 - fidelity),
                    "retained_norm": result.state.retained_norms[-1] if result.state.retained_norms else None,
                }
            )

    common.save_result(rows, "R34_cubic_convergence", extra={"gamma": GAMMA, "alpha": [ALPHA.real, ALPHA.imag]})

    ok_rows = [r for r in rows if r["ok"]]
    grid = np.full((len(SQUEEZING_VALUES), len(CUTOFFS)), np.nan)
    for r in ok_rows:
        i = SQUEEZING_VALUES.index(r["resource_squeezing"])
        j = CUTOFFS.index(r["cutoff"])
        grid[i, j] = r["fidelity"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    mesh = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(CUTOFFS)))
    ax.set_xticklabels(CUTOFFS)
    ax.set_yticks(range(len(SQUEEZING_VALUES)))
    ax.set_yticklabels(SQUEEZING_VALUES)
    ax.set_xlabel("Fock cutoff")
    ax.set_ylabel("Resource squeezing r")
    ax.set_title(f"Cubic injection fidelity to ideal gate, gamma={GAMMA} (m fixed at 0)")
    fig.colorbar(mesh, ax=ax, label="Fidelity")
    common.save_figure(fig, "R34_cubic_convergence")
    plt.close(fig)

    common.print_summary(
        "R34 cubic injection convergence (2D scan)",
        total_cells=len(rows),
        ok_cells=len(ok_rows),
        max_fidelity=max((r["fidelity"] for r in ok_rows), default=None),
        min_fidelity=min((r["fidelity"] for r in ok_rows), default=None),
    )


if __name__ == "__main__":
    main()
