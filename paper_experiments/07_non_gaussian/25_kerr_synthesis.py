"""R25: Kerr synthesis convergence (approximate; synthesis-only, no MBQC noise).

``photographiq.synthesis.synthesize_kerr`` returns an ordinary ``Circuit``
of native rotation/quadratic-phase/cubic-phase gates approximating
exp(i*kappa*n^2) by a Lie product formula. To isolate the product-formula
approximation error from separate finite-resource MBQC injection error
(covered by R23), this script translates the synthesized circuit's gates
directly into PHYSICAL Fock-backend commands (Rotate/QuadraticPhase/
CubicPhase/Displace) and executes them without compiling to an MBQC pattern.
Fidelity to the exact phase rule is compared across synthesis_steps.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.synthesis import synthesize_kerr  # noqa: E402

KAPPA = 0.05
CUTOFF = 16
STEPS = [1, 2, 4, 8, 16, 32, 64]


def physical_command_for(name, mode, params):
    """Translate one synthesized Circuit gate into a direct physical command."""
    if name == "displace":
        return pg.Displace(mode, *params)
    if name == "cubic_phase":
        return pg.CubicPhase(mode, params[0])
    if name == "symplectic":
        matrix = np.asarray(params[0])
        if np.allclose(matrix, [[1.0, 0.0], [matrix[1, 0], 1.0]], atol=1e-12):
            return pg.QuadraticPhase(mode, float(matrix[1, 0]))
        if np.allclose(matrix @ matrix.T, np.eye(2), atol=1e-10):
            angle = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
            return pg.Rotate(mode, angle)
        raise NotImplementedError("Unexpected symplectic gate shape in Kerr synthesis output")
    raise NotImplementedError(f"Unexpected synthesized gate kind: {name}")


def main():
    plt = common.setup_style()
    amplitudes_in = np.array([1, 1, 1], dtype=complex) / np.sqrt(3)
    fock_input = pg.FockInput(tuple(amplitudes_in))
    expected = np.zeros(CUTOFF, dtype=complex)
    for n in range(len(amplitudes_in)):
        expected[n] = amplitudes_in[n] * np.exp(1j * KAPPA * n**2)

    rows = []
    for steps in STEPS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            circuit, report = synthesize_kerr(KAPPA, steps=steps, modes=1, mode=0)
            pattern = pg.Pattern(inputs=(0,))
            for name, modes, params in circuit.gates:
                pattern.append(physical_command_for(name, modes[0], params))
            pattern.append(pg.Output((0,)))
            pattern.validate()

            start = time.perf_counter()
            state = pg.simulate(pattern, backend="piquasso-fock", cutoff=CUTOFF, inputs={0: fock_input}, seed=0).state
            elapsed = time.perf_counter() - start

        actual_map = dict(zip(state.basis, state.state_vector, strict=True))
        actual = np.array([actual_map.get((n,), 0j) for n in range(CUTOFF)])
        overlap = np.vdot(expected, actual)
        fidelity = float(np.clip(abs(overlap) ** 2, 0, 1))
        rows.append(
            {
                "synthesis_steps": steps,
                "fidelity_to_exact_kerr": fidelity,
                "trace_distance_estimate": float(np.sqrt(max(0.0, 1 - fidelity))),
                "primitive_count": report.primitive_count,
                "synthesis_order": report.order,
                "runtime_seconds": elapsed,
                "retained_norm": state.retained_norms[-1] if state.retained_norms else None,
            }
        )
        print(f"  steps={steps}: fidelity={fidelity:.8f}, primitives={report.primitive_count}, {elapsed:.3f}s", flush=True)

    common.save_result(
        rows, "R25_kerr_synthesis",
        extra={"kappa": KAPPA, "cutoff": CUTOFF, "note": "synthesis-only execution; MBQC injection noise excluded (see R23)"},
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.loglog([r["synthesis_steps"] for r in rows], [max(1e-16, 1 - r["fidelity_to_exact_kerr"]) for r in rows], "o-", color="#24677b")
    ax.set_xlabel("Synthesis steps")
    ax.set_ylabel("1 - fidelity to exact Kerr")
    ax.set_title(f"Kerr Lie-product synthesis, kappa={KAPPA} (direct physical execution)")
    common.save_figure(fig, "R25_kerr_synthesis")
    plt.close(fig)

    common.print_summary(
        "R25 Kerr synthesis convergence",
        steps=STEPS,
        min_infidelity=min(1 - r["fidelity_to_exact_kerr"] for r in rows),
        max_infidelity=max(1 - r["fidelity_to_exact_kerr"] for r in rows),
    )


if __name__ == "__main__":
    main()
