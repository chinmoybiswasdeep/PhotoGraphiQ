"""R10: Compiler resource scaling.

Builds circuits with an increasing number of repeated gates (rotation,
squeezing, CZ, beamsplitter) and records how compiled resource-node count,
measurement count, entanglement-edge count, command count and compile time
grow. Beamsplitter compilation is a fixed, angle-independent SUM-gate
decomposition that is markedly more resource-expensive per gate (see R7), so
its gate-count sweep is capped lower to keep total runtime practical.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

SQUEEZING = 1.0
GATE_COUNTS = [1, 2, 5, 10, 20, 50, 100]
BEAMSPLITTER_COUNTS = [1, 2, 5, 10, 20]


def build(kind: str, n: int) -> pg.Circuit:
    if kind in ("rotation", "squeeze"):
        circuit = pg.Circuit(1)
        for i in range(n):
            if kind == "rotation":
                circuit.rotate(0, 0.1 + 0.01 * i)
            else:
                circuit.squeeze(0, 0.05 + 0.005 * i)
        return circuit
    circuit = pg.Circuit(2)
    for i in range(n):
        if kind == "cz":
            circuit.cz(0, 1, 0.3 + 0.01 * i)
        else:
            circuit.beamsplitter(0, 1, 0.2 + 0.01 * i)
    return circuit


def measure_compile(circuit):
    repeats = 3 if len(circuit.gates) <= 20 else 1
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = circuit.compile(squeezing=SQUEEZING, return_trace=True)
        times.append(time.perf_counter() - start)
    return result, times


def main():
    plt = common.setup_style()
    all_rows = []
    for kind, counts in (
        ("rotation", GATE_COUNTS),
        ("squeeze", GATE_COUNTS),
        ("cz", GATE_COUNTS),
        ("beamsplitter", BEAMSPLITTER_COUNTS),
    ):
        for n in counts:
            circuit = build(kind, n)
            (pattern, trace), times = measure_compile(circuit)
            entangle_edges = sum(1 for c in pattern.commands if isinstance(c, pg.Entangle))
            measurements = sum(1 for c in pattern.commands if isinstance(c, pg.Measure))
            row = {
                "gate_kind": kind,
                "gate_count": n,
                "resource_nodes": len(pattern.graph.nodes) - circuit.modes,
                "measurements": measurements,
                "edges": entangle_edges,
                "commands": len(pattern.commands),
                "compile_time_median_seconds": float(np.median(times)) if len(times) > 1 else times[0],
                "compile_time_seconds_raw": times,
            }
            all_rows.append(row)
            print(f"  {kind} n={n}: {row['resource_nodes']} nodes, {row['compile_time_median_seconds']:.4f}s", flush=True)

    common.save_result(all_rows, "R10_compiler_scaling")

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
    metrics = [
        ("resource_nodes", "Resource nodes"),
        ("measurements", "Measurements"),
        ("commands", "Commands"),
        ("compile_time_median_seconds", "Compile time (s)"),
    ]
    for ax, (key, label) in zip(axes.flat, metrics, strict=True):
        for kind in ("rotation", "squeeze", "cz", "beamsplitter"):
            rows = [r for r in all_rows if r["gate_kind"] == kind]
            ax.plot([r["gate_count"] for r in rows], [r[key] for r in rows], "o-", label=kind, markersize=4)
        ax.set_xlabel("Number of gates")
        ax.set_ylabel(label)
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Compiler resource scaling")
    common.save_figure(fig, "R10_compiler_scaling")
    plt.close(fig)

    common.print_summary("R10 compiler resource scaling", rows=len(all_rows))


if __name__ == "__main__":
    main()
