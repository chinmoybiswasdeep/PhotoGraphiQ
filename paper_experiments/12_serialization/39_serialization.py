"""R39: Serialization round-trip.

Serializes a nontrivial adaptive pattern (labelled nodes, ``Parameter`` and
``Outcome`` expression dependencies, an outcome-dependent measurement angle)
through the public versioned JSON API (``Pattern.to_json``/``Pattern.from_json``,
backed by ``photographiq.serialization``), and verifies the restored pattern
has structurally equal commands, preserved labels/expression dependencies,
and identical seeded execution behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402


def build_pattern():
    adjacency = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    graph = pg.CVGraph.from_adjacency(adjacency, labels=("alice", "bob", "carol"), squeezing=1.0, inputs=("alice",))
    pattern = pg.Pattern(graph)
    pattern.measure("alice", pg.Homodyne.q(), key="m0")
    pattern.displace("bob", q=-pg.Outcome("m0"))
    pattern.measure("bob", pg.Homodyne(angle=0.1 * pg.Outcome("m0") + pg.Parameter("theta")), key="m1")
    pattern.displace("carol", q=-pg.Outcome("m1"), p=pg.Outcome("m0") + pg.Outcome("m1"))
    pattern.append(pg.Output(("carol",)))
    return pattern.validate()


def main():
    pattern = build_pattern()
    text = pattern.to_json()
    restored = pg.Pattern.from_json(text)

    checks = {
        "inputs_preserved": restored.inputs == pattern.inputs,
        "outputs_preserved": restored.outputs == pattern.outputs,
        "commands_structurally_equal": restored.commands == pattern.commands,
        "parameters_preserved": restored.parameters == pattern.parameters,
        "dependency_graph_edges_preserved": sorted(restored.dependencies().edges()) == sorted(pattern.dependencies().edges()),
    }

    original_result = pg.simulate(
        pattern, backend="gaussian", inputs={"alice": pg.GaussianInput.coherent(0.3 + 0.1j)},
        parameters={"theta": 0.4}, seed=7,
    )
    restored_result = pg.simulate(
        restored, backend="gaussian", inputs={"alice": pg.GaussianInput.coherent(0.3 + 0.1j)},
        parameters={"theta": 0.4}, seed=7,
    )
    checks["identical_seeded_outcomes"] = original_result.outcomes == restored_result.outcomes
    checks["identical_seeded_output_mean"] = common.frobenius_error(
        original_result.state.mean, restored_result.state.mean
    ) < 1e-12
    checks["identical_seeded_output_covariance"] = common.frobenius_error(
        original_result.state.covariance, restored_result.state.covariance
    ) < 1e-12

    failures = [name for name, ok in checks.items() if not ok]
    if failures:
        common.save_json({"failures": failures}, "R39_serialization_FAILURES")
        raise AssertionError(f"Serialization round-trip checks failed: {failures}")

    common.save_json(checks, "R39_serialization")
    common.write_metadata("R39_serialization")
    common.save_raw(text, "R39_serialization_pattern.json")

    common.print_summary("R39 serialization round-trip", **checks)


if __name__ == "__main__":
    main()
