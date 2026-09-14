"""R7: Beamsplitter compilation.

Sweeps the beamsplitter mixing angle theta in ``Circuit(2).beamsplitter``,
compiles the (deliberately resource-expensive, per docs/theory.md) SUM-gate
decomposition, and compares the exact unconditional Gaussian channel to an
independently written analytic beamsplitter symplectic matrix (the package's
own convention: c=cos, s=sin on (q0,p0,q1,p1)). Records the MBQC resource
overhead (nodes/measurements/commands), which the docs flag as especially
costly for this gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

THETAS = np.linspace(-np.pi + 0.05, np.pi - 0.05, 17)  # avoid the singular +-pi boundary
SQUEEZING = 1.2


def independent_beamsplitter(theta: float) -> np.ndarray:
    """Independent beamsplitter symplectic matrix (package sign convention)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, 0, -s, 0],
            [0, c, 0, -s],
            [s, 0, c, 0],
            [0, s, 0, c],
        ]
    )


def main():
    plt = common.setup_style()
    rows = []
    for theta in THETAS:
        circuit = pg.Circuit(2).beamsplitter(0, 1, float(theta))
        pattern, trace = circuit.compile(squeezing=SQUEEZING, return_trace=True)
        channel = pg.gaussian_channel(pattern)
        target = independent_beamsplitter(float(theta))
        map_error = common.frobenius_error(channel.matrix, target)
        step = trace.steps[0]
        rows.append(
            {
                "theta": float(theta),
                "squeezing": SQUEEZING,
                "map_frobenius_error": map_error,
                "noise_spectral_norm": float(np.linalg.norm(channel.noise, ord=2)),
                "noise_trace": float(np.trace(channel.noise)),
                "resource_nodes": len(step.resource_nodes),
                "measurements": len(step.measurements),
                "commands": len(pattern.commands),
            }
        )

    tolerance = 1e-6
    failures = [r for r in rows if r["map_frobenius_error"] > tolerance]
    if failures:
        common.save_csv(failures, "R7_beamsplitter_compile_FAILURES")
        raise AssertionError("Compiled beamsplitter map disagreed with the analytic target; see FAILURES csv")

    common.save_result(rows, "R7_beamsplitter_compile", extra={"squeezing": SQUEEZING})

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ts = [r["theta"] for r in rows]
    axes[0].semilogy(ts, [r["map_frobenius_error"] for r in rows], "o-", color="#24677b")
    axes[0].set_xlabel(r"Beamsplitter angle $\theta$ (rad)")
    axes[0].set_ylabel("Compiled map Frobenius error")
    axes[1].plot(ts, [r["resource_nodes"] for r in rows], "o-", color="#c66d27", label="Resource nodes")
    axes[1].plot(ts, [r["measurements"] for r in rows], "s--", color="#627a36", label="Measurements")
    axes[1].set_xlabel(r"Beamsplitter angle $\theta$ (rad)")
    axes[1].set_ylabel("MBQC resource count")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Beamsplitter compilation, resource squeezing r={SQUEEZING}")
    common.save_figure(fig, "R7_beamsplitter_compile")
    plt.close(fig)

    common.print_summary(
        "R7 beamsplitter compilation",
        thetas=len(rows),
        max_map_error=max(r["map_frobenius_error"] for r in rows),
        max_resource_nodes=max(r["resource_nodes"] for r in rows),
        max_commands=max(r["commands"] for r in rows),
    )


if __name__ == "__main__":
    main()
