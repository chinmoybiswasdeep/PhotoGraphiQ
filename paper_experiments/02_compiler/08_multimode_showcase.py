"""R8: Multi-mode circuit-to-MBQC compilation showcase (main-text figure candidate).

Builds a five-mode Gaussian circuit mixing rotations, a squeezer, a CZ gate,
a beamsplitter and a displacement, compiles it with an explicit provenance
trace, and renders the package's own gate-to-resource visualization
(``photographiq.visualize_compilation``). Also records the compiled
resource-graph statistics (nodes, measurements, entanglement edges,
corrections, commands) that accompany the figure.

Beamsplitter compilation is deliberately resource-expensive (SUM-gate
decomposition via three CZ/rotation sandwiches per docs/theory.md), so the
circuit here is kept small enough to stay under
``visualize_compilation``'s built-in size guard (<=24 gates, <=150 nodes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

SQUEEZING = 1.0


def build_circuit() -> pg.Circuit:
    circuit = pg.Circuit(5)
    circuit.rotate(0, 0.4)
    circuit.squeeze(1, 0.3)
    circuit.rotate(2, -0.6)
    circuit.cz(0, 1, 0.8)
    circuit.displace(3, q=0.2, p=-0.1)
    circuit.beamsplitter(2, 3, 0.3)
    circuit.rotate(4, 0.5)
    circuit.cz(3, 4, -0.5)
    return circuit


def main():
    common.setup_style()
    circuit = build_circuit()
    pattern, trace = circuit.compile(squeezing=SQUEEZING, return_trace=True)
    graph = pattern.graph

    entangle_edges = [c for c in pattern.commands if isinstance(c, pg.Entangle)]
    measurements = [c for c in pattern.commands if isinstance(c, pg.Measure)]
    corrections = [c for c in pattern.commands if isinstance(c, pg.Displace)]

    summary = {
        "logical_modes": circuit.modes,
        "source_gates": len(circuit.gates),
        "mbqc_nodes": len(graph.nodes),
        "measurements": len(measurements),
        "entanglement_edges": len(entangle_edges),
        "corrections": len(corrections),
        "commands": len(pattern.commands),
    }
    if summary["mbqc_nodes"] > 150 or summary["source_gates"] > 24:
        raise AssertionError("Showcase circuit exceeds visualize_compilation's size guard")

    steps_rows = [
        {
            "gate_index": s.gate_index,
            "source_gate": s.source_gate,
            "source_modes": s.source_modes,
            "resource_nodes": len(s.resource_nodes),
            "measurements": len(s.measurements),
            "corrections": len(s.corrections),
            "commands": len(s.command_indices),
        }
        for s in trace.steps
    ]

    common.save_json(summary, "R8_multimode_showcase_summary")
    common.save_csv(steps_rows, "R8_multimode_showcase_steps")
    common.write_metadata("R8_multimode_showcase")
    common.save_raw(pattern.to_json(), "R8_multimode_showcase_pattern.json")

    fig = pg.visualize_compilation(
        circuit,
        pattern,
        trace=trace,
        output=str(common.FIG_PDF / "R8_multimode_showcase.pdf"),
    )
    fig.savefig(common.FIG_PNG / "R8_multimode_showcase.png", bbox_inches="tight", dpi=300)
    import matplotlib.pyplot as plt

    plt.close(fig)

    common.print_summary("R8 multimode showcase", **summary)


if __name__ == "__main__":
    main()
