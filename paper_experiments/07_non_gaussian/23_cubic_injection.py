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

Upgrade migration (2026-11): main engine ``native-fock``; ``piquasso-fock``
(historical) runs the same grid as reference. The two engines treat the
two-mode CZ differently at a finite cutoff (native: exact projected Gaussian
matrix elements; Piquasso: Euler decomposition truncated after each factor),
so cells that hit the retained-norm guard can differ between engines; every
failed cell is recorded with its error, never dropped. Where both engines
succeed, the fidelity between their output states is reported. Asserted
(declared before evaluation): at the largest cutoff the native-fock fidelity
to the ideal gate does not decrease with resource squeezing among successful
cells (tolerance MONOTONE_SLACK), and native/Piquasso outputs agree to
ENGINE_FIDELITY_FLOOR wherever both succeed. The m = 0 branch is a
postselected conditional result, not deterministic execution.
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
SQUEEZING_VALUES = [0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
CUTOFFS = [24, 32, 48]
FIXED_OUTCOME = 0.0
ENGINES = ("native-fock", "piquasso-fock")
MONOTONE_SLACK = 1e-6
ENGINE_FIDELITY_FLOOR = 1 - 1e-3  # truncation-level differences of the CZ treatment only
def independent_cubic_unitary(gamma: float, cutoff: int) -> np.ndarray:
    a = np.diag(np.sqrt(np.arange(1, cutoff)), k=1)
    q = a + a.T
    return expm(1j * gamma * (q @ q @ q) / 6)


def truncated_input_vector(cutoff: int, backend: str = "piquasso-fock") -> np.ndarray:
    pattern = pg.Pattern(inputs=(0,)).append(pg.Output((0,))).validate()
    state = pg.simulate(
        pattern, backend=backend, cutoff=cutoff, inputs={0: pg.GaussianInput.coherent(ALPHA)}, seed=0
    ).state
    amplitudes = dict(zip(state.basis, state.state_vector, strict=True))
    return np.array([amplitudes.get((n,), 0j) for n in range(cutoff)])


def vector_from_state(state, node, cutoff):
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
                rows.append(
                    {"backend": backend, "gamma": GAMMA, "resource_squeezing": r, "cutoff": cutoff, "ok": False,
                     "error": failure["error"]}
                )
                continue
            result = failure["value"]
            actual = vector_from_state(result.state, "in", cutoff)
            outputs[(backend, cutoff, r)] = actual
            overlap = np.vdot(ideal_after, actual)
            fidelity = float(np.clip(abs(overlap) ** 2, 0, 1))
            rows.append(
                {
                    "backend": backend,
                    "gamma": GAMMA,
                    "resource_squeezing": r,
                    "cutoff": cutoff,
                    "ok": True,
                    "fidelity_to_ideal_cubic_phase": fidelity,
                    "trace_distance_estimate": float(np.sqrt(max(0.0, 1 - fidelity))),
                    "outcome_density": result.measurement_statistics["m"]["value"],
                    "outcome_kind": result.measurement_statistics["m"]["kind"],
                    "retained_norm": result.state.retained_norms[-1] if result.state.retained_norms else None,
                    "min_retained_norm": min(result.state.retained_norms) if result.state.retained_norms else None,
                }
            )

    for row in rows:
        other = outputs.get(("piquasso-fock", row["cutoff"], row["resource_squeezing"]))
        mine = outputs.get(("native-fock", row["cutoff"], row["resource_squeezing"]))
        row["native_vs_piquasso_fidelity"] = (
            float(abs(np.vdot(mine, other)) ** 2) if mine is not None and other is not None else None
        )
    failures = [r for r in rows if r["native_vs_piquasso_fidelity"] is not None
                and r["native_vs_piquasso_fidelity"] < ENGINE_FIDELITY_FLOOR]
    top = sorted((r for r in rows if r["ok"] and r["backend"] == "native-fock" and r["cutoff"] == CUTOFFS[-1]),
                 key=lambda r: r["resource_squeezing"])
    for a, b in zip(top, top[1:]):
        if b["fidelity_to_ideal_cubic_phase"] < a["fidelity_to_ideal_cubic_phase"] - MONOTONE_SLACK:
            failures.append({**b, "check": "fidelity decreased with resource squeezing"})
    common.save_result(rows, "R23_cubic_injection", extra={
        "gamma": GAMMA, "alpha": [ALPHA.real, ALPHA.imag], "fixed_outcome": FIXED_OUTCOME, "engines": list(ENGINES),
        "main_backend": "native-fock", "monotone_slack": MONOTONE_SLACK, "engine_fidelity_floor": ENGINE_FIDELITY_FLOOR,
        "norm_guard_failures": {b: sum(1 for r in rows if r["backend"] == b and not r["ok"]) for b in ENGINES}})
    if failures:
        common.save_csv(failures, "R23_cubic_injection_FAILURES")
        raise AssertionError(f"{len(failures)} R23 checks failed; see FAILURES csv")
    ok_rows = [r for r in rows if r["ok"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = dict(zip(CUTOFFS, ["#24677b", "#c66d27", "#627a36"], strict=False))
    for cutoff in CUTOFFS:
        for backend, style in (("native-fock", "o-"), ("piquasso-fock", "x--")):
            subset = [r for r in ok_rows if r["cutoff"] == cutoff and r["backend"] == backend]
            ax.plot([r["resource_squeezing"] for r in subset], [r["fidelity_to_ideal_cubic_phase"] for r in subset],
                    style, color=colors.get(cutoff), label=f"{backend}, cutoff={cutoff} ({len(subset)}/{len(SQUEEZING_VALUES)} ok)")
    ax.set_xlabel("Resource squeezing r")
    ax.set_ylabel("Fidelity to ideal CubicPhase(gamma)")
    ax.legend(fontsize=6.5)
    ax.set_title(f"Finite-resource cubic injection, gamma={GAMMA} (m fixed at 0)")
    common.save_figure(fig, "R23_cubic_injection")
    plt.close(fig)
    guard = {b: sum(1 for r in rows if r["backend"] == b and not r["ok"]) for b in ENGINES}
    both = [r["native_vs_piquasso_fidelity"] for r in rows if r["native_vs_piquasso_fidelity"] is not None]
    native_ok = [r for r in ok_rows if r["backend"] == "native-fock"]
    common.save_caption(
        "R23_cubic_injection",
        f"""
**R23: finite-resource cubic-phase injection (m = 0 branch).** Fidelity between the output of
`non_gaussian.cubic_injection` (gamma = {GAMMA}, coherent input alpha = {ALPHA}, ancilla homodyne
postselected at m = {FIXED_OUTCOME}, so the nonlinear feed-forward corrections vanish) and the ideal
CubicPhase(gamma) applied to the same truncated input, built independently with NumPy/SciPy, versus
resource squeezing r. Solid: `native-fock` (main engine); dashed: `piquasso-fock` (reference); colours:
total-photon cutoff {CUTOFFS}. The injection is a conditional filter times a cubic phase, not an exact
unitary; its fidelity rises with r from {min(r['fidelity_to_ideal_cubic_phase'] for r in native_ok):.4f} to
{max(r['fidelity_to_ideal_cubic_phase'] for r in native_ok):.4f} on `native-fock`. Cells whose retained Fock
norm left the guard band (norm_tolerance 1e-3) are reported as failures, not plotted: {guard['native-fock']}
of {len(CUTOFFS) * len(SQUEEZING_VALUES)} on `native-fock` and {guard['piquasso-fock']} on `piquasso-fock`;
larger r needs a larger cutoff because the squeezed ancilla's photon number grows as sinh^2 r. Where both
engines succeed their outputs agree to fidelity >= {min(both) if both else float('nan'):.6f}. Postselected
branch, finite energy: no claim about deterministic or fault-tolerant cubic gates. Source:
`paper_experiments/07_non_gaussian/23_cubic_injection.py`.
""",
    )

    common.print_summary(
        "R23 cubic MBQC injection (finite-resource approximation)",
        cases=len(rows),
        ok=len(ok_rows),
        norm_guard_failures={b: sum(1 for r in rows if r["backend"] == b and not r["ok"]) for b in ENGINES},
        min_native_vs_piquasso_fidelity=min((r["native_vs_piquasso_fidelity"] for r in rows
                                             if r["native_vs_piquasso_fidelity"] is not None), default=None),
        min_fidelity=min((r["fidelity_to_ideal_cubic_phase"] for r in ok_rows), default=None),
        max_fidelity=max((r["fidelity_to_ideal_cubic_phase"] for r in ok_rows), default=None),
    )


if __name__ == "__main__":
    main()
