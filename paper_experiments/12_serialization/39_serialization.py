"""R39: Serialization round-trip.

Serializes a nontrivial adaptive pattern (labelled nodes, ``Parameter`` and
``Outcome`` expression dependencies, an outcome-dependent measurement angle)
through the public versioned JSON API (``Pattern.to_json``/``Pattern.from_json``,
backed by ``photographiq.serialization``), and verifies the restored pattern
has structurally equal commands, preserved labels/expression dependencies,
and identical seeded execution behavior.

Upgrade migration (2026-11): the round trip is now checked field by field --
resource-graph edges and weights, node squeezing, ordered inputs/outputs,
measurement metadata (node, key, measurement type, efficiency, noise, angle
expression evaluated on sampled records), classical dependency DAG and the
correction functions (every ``Displace`` q/p expression evaluated on random
record values) -- and the restored pattern must give identical seeded results
on NumPy ``gaussian``, C++ ``native-gaussian`` and ``piquasso`` (backend-
independent semantics). The typed IR ``OpenGraph`` JSON round trip
(``to_dict``/``from_dict``) is checked on the same resource, and a
non-Gaussian pattern with nonlinear feed-forward
(``non_gaussian.cubic_injection``) is serialized, restored and executed on
``native-fock`` on a fixed outcome branch (identical states required).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
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

    from photographiq.expressions import resolve

    generator = common.rng(39)
    keys = [c.result_key for c in pattern.commands if isinstance(c, pg.Measure)]
    displacement_ok = measurement_ok = True
    for _ in range(25):
        records = {k: float(generator.normal(0, 2)) for k in keys}
        params = {"theta": float(generator.uniform(-1, 1))}
        for a, b in zip(pattern.commands, restored.commands, strict=True):
            if isinstance(a, pg.Displace):
                displacement_ok &= all(
                    resolve(getattr(a, f), records, params) == resolve(getattr(b, f), records, params) for f in ("q", "p")
                )
            if isinstance(a, pg.Measure):
                measurement_ok &= (
                    a.node == b.node and a.result_key == b.result_key and type(a.measurement) is type(b.measurement)
                    and a.measurement.efficiency == b.measurement.efficiency and a.measurement.noise == b.measurement.noise
                    and resolve(a.measurement.angle, records, params) == resolve(b.measurement.angle, records, params)
                )
    edges = lambda pat: {frozenset((u, v)): d.get("weight") for u, v, d in pat.graph.network.edges(data=True)}  # noqa: E731
    checks["graph_edges_and_weights_preserved"] = restored.graph is not None and edges(restored) == edges(pattern)
    checks["node_squeezing_preserved"] = (
        dict(restored.graph.network.nodes(data="squeezing")) == dict(pattern.graph.network.nodes(data="squeezing"))
    )
    checks["measurement_metadata_preserved"] = bool(measurement_ok)
    checks["correction_functions_preserved"] = bool(displacement_ok)
    og = pg.OpenGraph.from_cvgraph(pattern.graph)
    checks["open_graph_json_round_trip"] = pg.OpenGraph.from_dict(json.loads(json.dumps(og.to_dict()))).to_dict() == og.to_dict()

    backend_rows = []
    for backend in ("gaussian", "native-gaussian", "piquasso"):
        for seed in (7, 8, 9):
            kwargs = dict(backend=backend, inputs={"alice": pg.GaussianInput.coherent(0.3 + 0.1j)},
                          parameters={"theta": 0.4}, seed=seed)
            a, b = pg.simulate(pattern, **kwargs), pg.simulate(restored, **kwargs)
            ref = pg.simulate(pattern, **{**kwargs, "backend": "gaussian"})
            backend_rows.append({
                "backend": backend, "seed": seed,
                "original_vs_restored": max([abs(a.records[k] - b.records[k]) for k in a.records]
                                            + [float(np.abs(a.state.mean - b.state.mean).max())]),
                "vs_gaussian_original": max([abs(a.records[k] - ref.records[k]) for k in a.records]
                                            + [float(np.abs(a.state.mean - ref.state.mean).max())]),
            })
    checks["identical_on_all_gaussian_engines"] = all(
        r["original_vs_restored"] == 0.0 and r["vs_gaussian_original"] < 1e-10 for r in backend_rows
    )
    common.save_csv(backend_rows, "R39_serialization_backends")

    injection = pg.non_gaussian.cubic_injection(0.1, squeezing=0.5)
    injection_restored = pg.Pattern.from_json(injection.to_json())
    fock = [
        pg.simulate(pat, backend="native-fock", cutoff=24, inputs={"in": pg.GaussianInput.coherent(0.2)},
                    measurement_outcomes={"m": 0.7}, seed=0).state
        for pat in (injection, injection_restored)
    ]
    checks["nonlinear_feedforward_pattern_round_trip"] = injection_restored.commands == injection.commands
    checks["nonlinear_feedforward_identical_native_fock_state"] = bool(
        np.allclose(fock[0].state_vector, fock[1].state_vector, atol=0, rtol=0)
    )

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
