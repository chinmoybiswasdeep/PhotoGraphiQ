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

Upgrade migration (2026-11):

* the compiled 10x10 symplectic map and displacement are checked against an
  independently written NumPy product of the eight gates (the added
  finite-squeezing noise N is reported separately, never as map error);
* the graph metadata is validated against the command list and the typed IR
  ``OpenGraph`` (edges/weights, node squeezing, each non-output node measured
  once, feed-forward after the measurement it reads, IR round trip);
* the pattern is EXECUTED on NumPy ``gaussian`` and C++ ``native-gaussian``
  and the sampled ensemble is compared with the analytic channel;
* the main figure is a measurement-order view: x = position of the node's
  homodyne in the command order, y = logical mode, colour = source gate,
  red arrows = classical feed-forward (measurement -> later correction). The
  package's spring-layout overview is kept as ``R8_multimode_showcase_package_view``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

SQUEEZING = 1.0
EXEC_BACKENDS = ("gaussian", "native-gaussian")
EXEC_SHOTS = 200  # ~140 homodynes per trajectory
EXEC_SEED = 20261113
EXEC_ALPHAS = [0.2 + 0.1j, -0.1 + 0.2j, 0.3 + 0.0j, 0.0 - 0.2j, 0.1 + 0.1j]

# Acceptance criteria (declared before evaluation).
MAP_TOLERANCE = 1e-9
SEM_FACTOR = 5.0
NATIVE_TRAJECTORY_TOLERANCE = 1e-9


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


def independent_target():
    """Independent NumPy product of the eight gates (interleaved q,p; hbar=2)."""

    def embed(block, modes):
        m = np.eye(10)
        idx = [i for mode in modes for i in (2 * mode, 2 * mode + 1)]
        m[np.ix_(idx, idx)] = block
        return m

    def rot(t):
        return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    def sq(r):
        return np.diag([np.exp(-r), np.exp(r)])

    def cz(g):
        m = np.eye(4)
        m[1, 2] = m[3, 0] = g
        return m

    def bs(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, 0, -s, 0], [0, c, 0, -s], [s, 0, c, 0], [0, s, 0, c]])

    matrix, shift = np.eye(10), np.zeros(10)
    steps = [
        embed(rot(0.4), [0]), embed(sq(0.3), [1]), embed(rot(-0.6), [2]), embed(cz(0.8), [0, 1]),
        "displace", embed(bs(0.3), [2, 3]), embed(rot(0.5), [4]), embed(cz(-0.5), [3, 4]),
    ]
    for step in steps:
        if isinstance(step, str):
            shift[6:8] += [0.2, -0.1]
            continue
        matrix, shift = step @ matrix, step @ shift
    return matrix, shift


def graph_metadata_checks(pattern) -> dict:
    """Consistency of the compiled pattern's graph metadata with its command list."""
    network = pattern.graph.network
    entangle = {frozenset((c.u, c.v)): float(c.weight) for c in pattern.commands if isinstance(c, pg.Entangle)}
    edges = {frozenset((u, v)): float(d.get("weight", 1.0)) for u, v, d in network.edges(data=True)}
    prepare = {c.node: float(c.squeezing) for c in pattern.commands if isinstance(c, pg.Prepare)}
    node_squeezing = {n: float(d.get("squeezing", 0.0)) for n, d in network.nodes(data=True) if n not in pattern.inputs}
    measured = [c.node for c in pattern.commands if isinstance(c, pg.Measure)]
    dag = pattern.dependencies()
    og = pg.OpenGraph.from_cvgraph(pattern.graph)
    restored = pg.OpenGraph.from_dict(og.to_dict())
    return {
        "edges_match_entangle_commands": entangle == edges,
        "squeezing_matches_prepare_commands": prepare == node_squeezing,
        "each_non_output_measured_once": sorted(measured, key=repr)
        == sorted(set(network.nodes) - set(pattern.outputs), key=repr),
        "outputs_never_measured": not set(measured) & set(pattern.outputs),
        "feedforward_respects_measurement_order": all(u < v for u, v in dag.edges()),
        "open_graph_round_trip": restored.to_dict() == og.to_dict()
        and tuple(og.inputs) == tuple(pattern.inputs)
        and tuple(og.outputs) == tuple(pattern.outputs),
    }


def feedforward_edges(pattern):
    """(measured node, corrected node) pairs read from Pattern.dependencies()."""
    dag = pattern.dependencies()
    pairs = set()
    for u, v in dag.edges():
        source, target = pattern.commands[u], pattern.commands[v]
        if isinstance(source, pg.Measure) and isinstance(target, (pg.Displace, pg.Measure)):
            pairs.add((source.node, target.node))
    return sorted(pairs, key=repr)


def node_rows(pattern, trace, modes):
    """Logical-mode row of every node: the mode whose step input is graph-closest."""
    network = pattern.graph.network
    row = {node: mode for mode, node in enumerate(pattern.inputs)}
    for step in trace.steps:
        anchors = list(zip(step.source_modes, step.input_nodes, strict=False))
        for node in step.resource_nodes:
            distances = []
            for mode, anchor in anchors:
                try:
                    distances.append((nx.shortest_path_length(network, anchor, node), mode))
                except nx.NetworkXNoPath:
                    distances.append((np.inf, mode))
            row[node] = min(distances)[1]
        for mode, node in zip(step.source_modes, step.output_nodes, strict=False):
            row[node] = mode  # a gate's output node carries that gate's mode
    for mode, node in enumerate(pattern.outputs):
        row[node] = mode
    return row


def main():
    plt = common.setup_style()
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

    channel = pg.gaussian_channel(pattern)
    target_matrix, target_shift = independent_target()
    summary["map_frobenius_error"] = common.frobenius_error(channel.matrix, target_matrix)
    summary["displacement_error"] = common.frobenius_error(channel.displacement, target_shift)
    summary["noise_trace"] = float(np.trace(channel.noise))
    summary["noise_spectral_norm"] = float(np.linalg.norm(channel.noise, ord=2))
    checks = graph_metadata_checks(pattern)
    summary["graph_metadata_checks"] = checks

    inputs = {node: pg.GaussianInput.coherent(a) for node, a in zip(pattern.inputs, EXEC_ALPHAS, strict=True)}
    mean = np.concatenate([[2 * a.real, 2 * a.imag] for a in EXEC_ALPHAS])
    ideal = channel.apply(pg.GaussianState(mean, np.eye(10), pattern.inputs))
    conditional = pg.simulate(pattern, backend="gaussian", inputs=inputs, seed=EXEC_SEED).state.covariance
    spread = ideal.covariance - conditional
    d = np.clip(np.diag(spread), 0, None)
    sem_mean, sem_cov = np.sqrt(d / EXEC_SHOTS), np.sqrt((np.outer(d, d) + spread**2) / EXEC_SHOTS)
    runs = {}
    for backend in EXEC_BACKENDS:
        result = pg.run_shots(pattern, EXEC_SHOTS, backend=backend, inputs=inputs, seed=EXEC_SEED)
        runs[backend] = result.trajectories
        ensemble = result.ensemble_state()
        summary[f"{backend}_max_z_mean"] = float((np.abs(ensemble.mean - ideal.mean) / np.maximum(sem_mean, 1e-12)).max())
        summary[f"{backend}_max_z_covariance"] = float(
            (np.abs(ensemble.covariance - ideal.covariance) / np.maximum(sem_cov, 1e-12)).max()
        )
    summary["native_vs_gaussian_max_record_difference"] = max(
        abs(a.outcomes[k] - b.outcomes[k])
        for a, b in zip(runs["gaussian"], runs["native-gaussian"], strict=True)
        for k in a.outcomes
    )
    summary["planner_backend"] = pg.select_backend(pattern, inputs).name
    summary["execution"] = {"backends": list(EXEC_BACKENDS), "shots": EXEC_SHOTS, "seed": EXEC_SEED}
    summary["acceptance"] = {
        "map_tolerance": MAP_TOLERANCE,
        "sem_factor": SEM_FACTOR,
        "native_trajectory_tolerance": NATIVE_TRAJECTORY_TOLERANCE,
    }

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

    failures = []
    if summary["map_frobenius_error"] > MAP_TOLERANCE or summary["displacement_error"] > MAP_TOLERANCE:
        failures.append("compiled map/displacement vs independent target")
    if not all(checks.values()):
        failures.append(f"graph metadata: {[k for k, v in checks.items() if not v]}")
    if any(summary[f"{b}_max_z_{m}"] > SEM_FACTOR for b in EXEC_BACKENDS for m in ("mean", "covariance")):
        failures.append("ensemble vs analytic channel")
    if summary["native_vs_gaussian_max_record_difference"] > NATIVE_TRAJECTORY_TOLERANCE:
        failures.append("native trajectory contract")
    if failures:
        common.save_json({"failures": failures, "summary": summary}, "R8_multimode_showcase_FAILURES")
        raise AssertionError(f"R8 checks failed: {failures}")

    # Package's own overview (spring layout), kept for provenance.
    fig = pg.visualize_compilation(circuit, pattern, trace=trace)
    common.save_figure(fig, "R8_multimode_showcase_package_view", tight=False)
    plt.close(fig)

    # Measurement-order view.
    order = {c.node: i for i, c in enumerate(measurements)}
    x = {n: order.get(n, len(measurements) + 2) for n in graph.nodes}
    rows = node_rows(pattern, trace, circuit.modes)
    by_cell: dict = {}
    for n in sorted(graph.nodes, key=lambda n: x[n]):
        by_cell.setdefault(rows[n], []).append(n)
    y = {n: rows[n] for n in graph.nodes}
    cmap = plt.get_cmap("tab10")
    gate_of = {n: s.gate_index for s in trace.steps for n in s.resource_nodes}
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for u, v in graph.network.edges():
        ax.plot([x[u], x[v]], [y[u], y[v]], color="#9a9a9a", linewidth=0.6, zorder=1)
    for source, target in feedforward_edges(pattern):
        span = max(1.0, abs(x[target] - x[source]))
        ax.annotate(
            "", xy=(x[target], y[target] + 0.18), xytext=(x[source], y[source] + 0.18),
            arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=0.5, alpha=0.55,
                            connectionstyle=f"arc3,rad={-0.35 * min(1.0, 6.0 / span):.3f}", shrinkA=2, shrinkB=2),
            zorder=2,
        )
    for n in graph.nodes:
        if n in pattern.inputs:
            ax.scatter(x[n], y[n], marker="s", s=70, color="#333333", zorder=3)
        elif n in pattern.outputs:
            ax.scatter(x[n], y[n], marker="D", s=70, color="#333333", zorder=3)
        else:
            ax.scatter(x[n], y[n], s=26, color=cmap(gate_of.get(n, 0) % 10), edgecolor="white", linewidth=0.4, zorder=3)
    for step in trace.steps:
        xs = [x[n] for n in step.resource_nodes if n in order]
        if xs:
            ax.text(np.mean(xs), circuit.modes - 0.45 + 0.22 * (step.gate_index % 2), f"{step.gate_index}:{step.source_gate}",
                    ha="center", fontsize=7.5, color=cmap(step.gate_index % 10))
    ax.set_yticks(range(circuit.modes))
    ax.set_yticklabels([f"mode {m}" for m in range(circuit.modes)])
    ax.set_ylim(-0.6, circuit.modes)
    ax.invert_yaxis()
    ax.set_xlabel("Measurement order (index of the node's homodyne in the command list; outputs at right)")
    ax.set_title(
        f"Compiled five-mode circuit: {summary['mbqc_nodes']} nodes, {summary['measurements']} homodynes, "
        f"{summary['corrections']} corrections (squares: inputs, diamonds: outputs, red: feed-forward)",
        fontsize=9.5,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    common.save_figure(fig, "R8_multimode_showcase")
    plt.close(fig)

    max_z = max(summary[f"{b}_max_z_{m}"] for b in EXEC_BACKENDS for m in ("mean", "covariance"))
    common.save_caption(
        "R8_multimode_showcase",
        f"""
**R8: five-mode circuit compiled to a CV-MBQC pattern.** The circuit R(0.4) on mode 0, S(0.3) on
mode 1, R(-0.6) on mode 2, CZ(0.8) on modes 0-1, D(q=0.2, p=-0.1) on mode 3, BS(0.3) on modes 2-3,
R(0.5) on mode 4 and CZ(-0.5) on modes 3-4 is compiled with resource squeezing r = {SQUEEZING} into
{summary['mbqc_nodes']} nodes, {summary['entanglement_edges']} weighted CZ edges (grey),
{summary['measurements']} homodyne measurements and {summary['corrections']} displacement corrections
({summary['commands']} commands). Each node is placed at the position of its homodyne in the command
order (x) and on the row of the logical mode it carries (y; nodes of two-mode gates go to the row of the
graph-closest input); colours give the source gate (labels above), red arrows the classical feed-forward
read from `Pattern.dependencies()` (a measurement outcome used by a later correction or measurement angle).
The compiled channel reproduces the independently multiplied 10x10 symplectic map to
{summary['map_frobenius_error']:.1e} and the displacement to {summary['displacement_error']:.1e}; the
finite-squeezing noise is reported separately (Tr N = {summary['noise_trace']:.3g}, ||N|| =
{summary['noise_spectral_norm']:.3g}). Graph metadata (edges and weights, node squeezing, single
measurement of every non-output node, feed-forward order, `OpenGraph` round trip) is consistent. Executed on
`gaussian` and `native-gaussian` ({EXEC_SHOTS} trajectories each): ensemble moments within {max_z:.2f}
shot-noise standard errors of the analytic channel; native and NumPy records agree to
{summary['native_vs_gaussian_max_record_difference']:.1e}. Source:
`paper_experiments/02_compiler/08_multimode_showcase.py`.
""",
    )
    common.print_summary(
        "R8 multimode showcase",
        **{k: v for k, v in summary.items() if not isinstance(v, dict)},
    )


if __name__ == "__main__":
    main()
