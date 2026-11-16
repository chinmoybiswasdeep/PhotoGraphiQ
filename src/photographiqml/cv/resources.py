"""Physical and computational resource accounting, kept separate from the ML objective."""

from __future__ import annotations

from time import perf_counter

import networkx as nx
import numpy as np

import photographiq as pg

DB_PER_NEPER = 20 / np.log(10)  # squeezing r in dB: 10 log10(e^{2r})


def pattern_resources(pattern):
    """Structural counts of an executable pattern (any backend)."""
    commands = pattern.commands
    measures = [i for i, c in enumerate(commands) if isinstance(c, pg.Measure)]
    feedforward = [
        c
        for c in commands
        if not isinstance(c, (pg.Measure, pg.Output))
        and pg.expressions.dependencies(getattr(c, "q", 0.0))
        | pg.expressions.dependencies(getattr(c, "p", 0.0))
        | pg.expressions.dependencies(getattr(c, "s", 0.0))
    ]
    entangle = [c for c in commands if isinstance(c, pg.Entangle)]
    weights = [float(c.weight) for c in entangle]
    graph = pattern.dependencies()
    depth: dict = {}
    for node in nx.topological_sort(graph):  # longest chain counted in measurements
        here = 1 if node in measures else 0
        depth[node] = here + max((depth[p] for p in graph.predecessors(node)), default=0)
    live = peak = len(pattern.inputs)
    for c in commands:
        if isinstance(c, (pg.Prepare,)):
            live += 1
        elif isinstance(c, pg.Measure):
            live -= 1
        peak = max(peak, live)
    return {
        "resource_modes": sum(isinstance(c, pg.Prepare) for c in commands),
        "entangling_edges": len(entangle),
        "weighted_edges": sum(abs(w - 1) > 1e-12 for w in weights),
        "measurements": len(measures),
        "measurement_depth": max(depth.values(), default=0),
        "feedforward_corrections": len(feedforward),
        "peak_live_modes": peak,
        "trajectory_covariance_bytes": int((2 * peak) ** 2 * 8),
        "non_gaussian_operations": sum(isinstance(c, (pg.Kerr, pg.CubicPhase)) for c in commands),
    }


def resource_report(model, X=None):
    """Resources of a fitted CV model at its trained parameters (first sample if data enters
    gates). Optical energy is the mean input photon number; squeezing in r and dB."""
    from .gaussian import GaussianProgram
    from .mbqc import CVCluster

    program, params = model.program, model.params
    X = None if X is None else model.feature_map.check(X)
    x = X[0] if X is not None and len(X) else np.zeros(program.modes)
    report = {
        "model": type(model).__name__,
        "program": type(program).__name__,
        "optical_modes": program.modes,
        "execution": model.mode,
        "trainable_quantum_parameters": len(model.quantum_parameters),
        "classical_head_parameters": len(model.head),
    }
    if isinstance(program, CVCluster):
        r = program.squeezing if model.squeezing is None else model.squeezing
        pattern = program.pattern(params, squeezing=r)
        report["cluster"] = {
            "wires": program.wires,
            "columns": program.columns,
            "rungs": list(program.rungs),
        }
    elif isinstance(program, GaussianProgram) and not getattr(program, "non_gaussian", False):
        # structure (counts) does not depend on r; an ideal model has no resource squeezing
        r = model.squeezing if model.squeezing is not None else 1.0
        report["structure_compiled_at_r"] = r
        pattern = program.circuit(params, x).compile(squeezing=r)
        gate_r = [float(np.real(v)) for v in program.squeezing_values(params, x)]
        report["gate_squeezing_max_r"] = max(map(abs, gate_r), default=0.0)
        report["gate_squeezing_max_db"] = report["gate_squeezing_max_r"] * DB_PER_NEPER
    else:
        pattern, r = None, None
    if pattern is not None:
        report.update(pattern_resources(pattern))
        used = model.mode == "physical"
        report["resource_squeezing_r"] = r if used else None
        report["resource_squeezing_db"] = r * DB_PER_NEPER if used else None
        if model.mode == "physical":
            Y = program.channel(params, x, mode="physical", squeezing=r)[2]
            report["added_noise_per_quadrature"] = float(np.trace(np.real(Y)) / len(Y))
    if X is not None and len(X):
        report["mean_input_photons"] = float(
            np.mean([model.feature_map.mean_photons(v) for v in X])
        )
        start = perf_counter()
        model.moments(params, X[: min(len(X), 16)])
        report["seconds_per_prediction"] = (perf_counter() - start) / min(len(X), 16)
    report["moment_memory_bytes"] = int(3 * (2 * program.modes) ** 2 * 8)
    if getattr(program, "non_gaussian", False):
        report.update(program.resources(params, x, model.cutoff))
    return report
