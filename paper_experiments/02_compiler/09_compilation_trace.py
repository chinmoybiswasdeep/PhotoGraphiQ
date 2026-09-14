"""R9: Compilation provenance / trace.

Uses the same five-mode showcase circuit as R8 and exports the full
gate-by-gate compilation provenance (``CompilationTrace``/``CompilationStep``)
as a machine-readable CSV plus a human-readable text summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

SQUEEZING = 1.0


def build_circuit() -> pg.Circuit:
    """Identical to 08_multimode_showcase.build_circuit (kept standalone per script)."""
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
    circuit = build_circuit()
    pattern, trace = circuit.compile(squeezing=SQUEEZING, return_trace=True)

    rows = []
    summary_lines = []
    for step in trace.steps:
        rows.append(
            {
                "gate_index": step.gate_index,
                "source_gate": step.source_gate,
                "source_modes": step.source_modes,
                "source_parameters": step.source_parameters,
                "input_nodes": step.input_nodes,
                "output_nodes": step.output_nodes,
                "resource_nodes": step.resource_nodes,
                "n_resource_nodes": len(step.resource_nodes),
                "command_indices": step.command_indices,
                "n_commands": len(step.command_indices),
                "edges": step.edges,
                "n_edges": len(step.edges),
                "measurements": step.measurements,
                "n_measurements": len(step.measurements),
                "corrections": step.corrections,
                "n_corrections": len(step.corrections),
                "synthesis_report": None if step.synthesis_report is None else str(step.synthesis_report),
            }
        )
        summary_lines.append(
            f"Gate {step.gate_index:2d} [{step.source_gate}] modes={step.source_modes} "
            f"params={step.source_parameters} : {len(step.resource_nodes)} resource nodes, "
            f"{len(step.measurements)} measurements, {len(step.corrections)} corrections, "
            f"{len(step.command_indices)} commands "
            f"({step.input_nodes} -> {step.output_nodes})"
        )

    # Consistency check: every command index appears in exactly one step's
    # command_indices list plus the trailing Output command.
    all_indices = sorted(i for step in trace.steps for i in step.command_indices)
    expected = list(range(len(pattern.commands) - 1))  # exclude trailing Output
    if all_indices != expected:
        raise AssertionError("Compilation trace command indices do not partition the command list")

    common.save_csv(rows, "R9_compilation_trace")
    common.save_json({"steps": rows, "pattern_signature": trace.pattern_signature}, "R9_compilation_trace")
    common.write_metadata("R9_compilation_trace")
    common.save_raw("\n".join(summary_lines), "R9_compilation_trace_summary.txt")

    common.print_summary(
        "R9 compilation provenance",
        gates=len(trace.steps),
        total_commands=len(pattern.commands),
        total_resource_nodes=sum(len(s.resource_nodes) for s in trace.steps),
    )


if __name__ == "__main__":
    main()
