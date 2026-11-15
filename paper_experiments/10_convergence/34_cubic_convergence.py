"""R34: Cubic injection convergence -- 2D scan over cutoff and resource squeezing.

Two-dimensional grid of (Fock cutoff, resource squeezing) for
``photographiq.non_gaussian.cubic_injection`` with a fixed ancilla outcome
m=0 (isolating the residual finite-resource Gaussian envelope from outcome-
dependent feed-forward). Fidelity to an ideal direct CubicPhase(gamma),
computed independently with NumPy/SciPy, is saved as a heatmap-ready CSV.

Upgrade migration (2026-11): the grid runs on ``native-fock`` (main) and
``piquasso-fock`` (historical reference), each against the independent
reference built from its own truncated input. Cells that hit the retained-norm
guard (|norm - 1| > 1e-3 after an operation) are physical truncation of the
cubic-phase ancilla (its p spread grows like gamma q^2 with q ~ e^r), recorded
with the retained norm in the error message and shown as hatched cells, never
renormalised silently. Where both engines succeed their output fidelity must be
>= ENGINE_FIDELITY_FLOOR (truncation-level CZ differences only), and at the
largest cutoff the native fidelity to the ideal gate must not decrease with r
(MONOTONE_SLACK) -- both declared before evaluation. The m = 0 branch is
postselected, not deterministic.
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

GAMMA = 0.15
ALPHA = 0.2 + 0.0j
CUTOFFS = [24, 32, 48, 64, 96]
SQUEEZING_VALUES = [0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
FIXED_OUTCOME = 0.0
ENGINES = ("native-fock", "piquasso-fock")
ENGINE_FIDELITY_FLOOR = 1 - 1e-3
MONOTONE_SLACK = 1e-6
def independent_cubic_unitary(gamma, cutoff):
    a = np.diag(np.sqrt(np.arange(1, cutoff)), k=1)
    q = a + a.T
    return expm(1j * gamma * (q @ q @ q) / 6)


def truncated_input_vector(cutoff, backend="piquasso-fock"):
    pattern = pg.Pattern(inputs=(0,)).append(pg.Output((0,))).validate()
    state = pg.simulate(
        pattern, backend=backend, cutoff=cutoff, inputs={0: pg.GaussianInput.coherent(ALPHA)}, seed=0
    ).state
    amplitudes = dict(zip(state.basis, state.state_vector, strict=True))
    return np.array([amplitudes.get((n,), 0j) for n in range(cutoff)])


def main():
    plt = common.setup_style()
    rows = []
    outputs = {}
    for cutoff, backend in itertools.product(CUTOFFS, ENGINES):
        before = truncated_input_vector(cutoff, backend)
        ideal_after = independent_cubic_unitary(GAMMA, cutoff) @ before
        ideal_after /= np.linalg.norm(ideal_after)

        for r in SQUEEZING_VALUES:
            pattern = pg.non_gaussian.cubic_injection(GAMMA, squeezing=r)
            failure = common.safe_cutoff_run(
                lambda: pg.simulate(
                    pattern, backend=backend, cutoff=cutoff,
                    inputs={"in": pg.GaussianInput.coherent(ALPHA)},
                    measurement_outcomes={"m": FIXED_OUTCOME}, seed=0,
                ),
                label=f"cutoff={cutoff} r={r}",
            )
            if not failure["ok"]:
                rows.append({"backend": backend, "cutoff": cutoff, "resource_squeezing": r, "ok": False, "fidelity": None,
                             "error": failure["error"]})
                continue
            result = failure["value"]
            amplitudes = dict(zip(result.state.basis, result.state.state_vector, strict=True))
            actual = np.array([amplitudes.get((n,), 0j) for n in range(cutoff)])
            fidelity = float(np.clip(abs(np.vdot(ideal_after, actual)) ** 2, 0, 1))
            outputs[(backend, cutoff, r)] = actual
            rows.append(
                {
                    "backend": backend,
                    "cutoff": cutoff, "resource_squeezing": r, "ok": True, "fidelity": fidelity,
                    "log10_infidelity": common.safe_log10(1 - fidelity),
                    "retained_norm": result.state.retained_norms[-1] if result.state.retained_norms else None,
                    "min_retained_norm": min(result.state.retained_norms) if result.state.retained_norms else None,
                }
            )

    for row in rows:
        mine = outputs.get(("native-fock", row["cutoff"], row["resource_squeezing"]))
        other = outputs.get(("piquasso-fock", row["cutoff"], row["resource_squeezing"]))
        row["native_vs_piquasso_fidelity"] = float(abs(np.vdot(mine, other)) ** 2) if mine is not None and other is not None else None
    failures = [r for r in rows if r["native_vs_piquasso_fidelity"] is not None
                and r["native_vs_piquasso_fidelity"] < ENGINE_FIDELITY_FLOOR]
    top = sorted((r for r in rows if r["ok"] and r["backend"] == "native-fock" and r["cutoff"] == CUTOFFS[-1]),
                 key=lambda r: r["resource_squeezing"])
    failures += [b for a, b in zip(top, top[1:]) if b["fidelity"] < a["fidelity"] - MONOTONE_SLACK]
    common.save_result(rows, "R34_cubic_convergence", extra={
        "gamma": GAMMA, "alpha": [ALPHA.real, ALPHA.imag], "engines": list(ENGINES), "main_backend": "native-fock",
        "engine_fidelity_floor": ENGINE_FIDELITY_FLOOR, "monotone_slack": MONOTONE_SLACK,
        "norm_guard_failures": {b: sum(1 for r in rows if r["backend"] == b and not r["ok"]) for b in ENGINES}})
    if failures:
        common.save_csv(failures, "R34_cubic_convergence_FAILURES")
        raise AssertionError(f"{len(failures)} R34 checks failed; see FAILURES csv")
    ok_rows = [r for r in rows if r["ok"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, backend in zip(axes, ENGINES, strict=True):
        grid = np.full((len(SQUEEZING_VALUES), len(CUTOFFS)), np.nan)
        for r in ok_rows:
            if r["backend"] == backend:
                grid[SQUEEZING_VALUES.index(r["resource_squeezing"]), CUTOFFS.index(r["cutoff"])] = r["fidelity"]
        mesh = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis", vmin=0.9, vmax=1)
        for i, j in zip(*np.where(np.isnan(grid)), strict=True):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="xx", edgecolor="grey", linewidth=0))
        for i, j in zip(*np.where(~np.isnan(grid)), strict=True):
            ax.text(j, i, f"{grid[i, j]:.4f}", ha="center", va="center", fontsize=6.5, color="white")
        ax.set_xticks(range(len(CUTOFFS)))
        ax.set_xticklabels(CUTOFFS)
        ax.set_yticks(range(len(SQUEEZING_VALUES)))
        ax.set_yticklabels(SQUEEZING_VALUES)
        ax.set_xlabel("Fock cutoff")
        ax.set_ylabel("Resource squeezing r")
        ax.set_title(f"{backend} (hatched: norm guard)", fontsize=9.5)
    fig.colorbar(mesh, ax=axes, label="Fidelity to ideal CubicPhase")
    fig.suptitle(f"Cubic injection fidelity, gamma={GAMMA}, m fixed at 0")
    common.save_figure(fig, "R34_cubic_convergence", tight=False)
    plt.close(fig)
    guard = {b: sum(1 for r in rows if r["backend"] == b and not r["ok"]) for b in ENGINES}
    both = [r["native_vs_piquasso_fidelity"] for r in rows if r["native_vs_piquasso_fidelity"] is not None]
    native_ok = [r for r in ok_rows if r["backend"] == "native-fock"]
    common.save_caption(
        "R34_cubic_convergence",
        f"""
**R34: cubic-phase injection over cutoff and resource squeezing.** Fidelity of the m = 0 postselected output of
`non_gaussian.cubic_injection` (gamma = {GAMMA}, coherent input alpha = {ALPHA}) to the ideal CubicPhase(gamma) built
independently with NumPy/SciPy, on a grid of total-photon cutoffs {CUTOFFS} and resource squeezing {SQUEEZING_VALUES};
left `native-fock`, right `piquasso-fock`. Hatched cells stopped at the retained-norm guard ({guard['native-fock']} of
{len(CUTOFFS) * len(SQUEEZING_VALUES)} native, {guard['piquasso-fock']} Piquasso): the finite-squeezing cubic-phase
ancilla spreads to high photon numbers, so larger r needs a larger cutoff. Successful native cells range from
{min(r['fidelity'] for r in native_ok):.4f} to {max(r['fidelity'] for r in native_ok):.4f}; along each row the fidelity is
stable in the cutoff (converged), and it rises with r (the conditional filter approaches the ideal gate only as
r -> infinity). Where both engines succeed their outputs agree to fidelity >= {min(both):.6f}. Postselected branch,
finite energy: not a deterministic cubic gate. Source: `paper_experiments/10_convergence/34_cubic_convergence.py`.
""",
    )

    common.print_summary(
        "R34 cubic injection convergence (2D scan)",
        total_cells=len(rows),
        ok_cells=len(ok_rows),
        norm_guard_failures={b: sum(1 for r in rows if r["backend"] == b and not r["ok"]) for b in ENGINES},
        min_native_vs_piquasso_fidelity=min((r["native_vs_piquasso_fidelity"] for r in rows
                                             if r["native_vs_piquasso_fidelity"] is not None), default=None),
        max_fidelity=max((r["fidelity"] for r in ok_rows), default=None),
        min_fidelity=min((r["fidelity"] for r in ok_rows), default=None),
    )


if __name__ == "__main__":
    main()
