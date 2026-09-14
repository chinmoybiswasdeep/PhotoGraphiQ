"""R37: Compiler runtime scaling (with warm-up and repeated timing).

Benchmarks ``Circuit.compile`` alone (not execution) for a growing number of
alternating rotation/CZ logical gates on two modes, using warm-up runs and
repeated ``time.perf_counter`` measurements (median, IQR, std). Records
generated resource nodes, commands and measurements alongside timing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import metadata  # noqa: E402

import photographiq as pg  # noqa: E402

GATE_COUNTS = [1, 2, 5, 10, 20, 50, 100]
SQUEEZING = 1.0


def build_circuit(n_gates: int) -> pg.Circuit:
    circuit = pg.Circuit(2)
    for i in range(n_gates):
        if i % 2 == 0:
            circuit.rotate(i % 2, 0.1 + 0.01 * i)
        else:
            circuit.cz(0, 1, 0.2 + 0.01 * i)
    return circuit


def main():
    plt = common.setup_style()
    rows = []
    for n_gates in GATE_COUNTS:
        circuit = build_circuit(n_gates)
        warmup, repeats = (2, 9) if n_gates <= 20 else (1, 5)
        timing = common.benchmark(
            lambda circuit=circuit: circuit.compile(squeezing=SQUEEZING, return_trace=True),
            warmup=warmup, repeats=repeats,
        )
        pattern, trace = timing["result"]
        measurements = sum(1 for c in pattern.commands if isinstance(c, pg.Measure))
        rows.append(
            {
                "n_gates": n_gates,
                "median_compile_seconds": timing["median_seconds"],
                "std_compile_seconds": timing["std_seconds"],
                "iqr_compile_seconds": timing["iqr_seconds"],
                "repeats": timing["repeats"],
                "warmup": timing["warmup"],
                "resource_nodes": len(pattern.graph.nodes) - circuit.modes,
                "commands": len(pattern.commands),
                "measurements": measurements,
            }
        )
        print(f"  n_gates={n_gates}: median={timing['median_seconds']:.5f}s, nodes={rows[-1]['resource_nodes']}", flush=True)

    common.save_result(rows, "R37_compiler_runtime", extra={"environment": metadata.collect()})

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].errorbar(
        [r["n_gates"] for r in rows], [r["median_compile_seconds"] for r in rows],
        yerr=[r["iqr_compile_seconds"] / 2 for r in rows], fmt="o-", color="#24677b", capsize=3,
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of logical gates")
    axes[0].set_ylabel("Median compile time (s)")

    axes[1].plot([r["n_gates"] for r in rows], [r["resource_nodes"] for r in rows], "o-", label="Resource nodes", color="#c66d27")
    axes[1].plot([r["n_gates"] for r in rows], [r["commands"] for r in rows], "s-", label="Commands", color="#627a36")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Number of logical gates")
    axes[1].legend(fontsize=8)
    fig.suptitle("Compiler runtime scaling (this machine only)")
    common.save_figure(fig, "R37_compiler_runtime")
    plt.close(fig)

    common.print_summary("R37 compiler runtime scaling", gate_counts=GATE_COUNTS)


if __name__ == "__main__":
    main()
