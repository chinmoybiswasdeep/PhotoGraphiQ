"""Optional matplotlib resource, circuit and provenance diagrams."""

import math

import networkx as nx
import numpy as np

from .commands import Measure, command_dependencies
from .expressions import Expr
from .measurements import PhotonNumber


def _text(value):
    if isinstance(value, Expr):
        if not value.parameters and not value.dependencies:
            return f"{value.evaluate({}, {}):.3g}"
        if value.op in ("parameter", "constant"):
            return _text(value.args[0])
        if value.op == "outcome":
            return f"m[{value.args[0]}]"
        return f"{value.op}({', '.join(_text(a) for a in value.args)})"
    return f"{value:.3g}" if isinstance(value, (float, np.floating)) else str(value)


def _axes(ax, size=(9, 5)):
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=size)[1] if ax is None else ax


def _finish(ax, output):
    if output is not None:
        ax.figure.savefig(output, bbox_inches="tight")
    return ax


def _color(role, style):
    if style not in ("default", "monochrome"):
        raise ValueError("style must be default or monochrome")
    return (
        "#dddddd"
        if style == "monochrome"
        else {
            "input": "#008c95",
            "output": "#e69f00",
            "resource": "#a8c7e0",
            "gaussian": "#756bb1",
            "non_gaussian": "#d95f02",
        }[role]
    )


def _positions(graph, layout, positions):
    if positions is not None:
        if set(positions) != set(graph):
            raise ValueError("positions must provide coordinates for every node")
        return positions
    if layout == "spring":
        return nx.spring_layout(graph, seed=7, weight=None)
    if layout == "circular":
        return nx.circular_layout(graph)
    if layout == "grid":
        width = max(1, math.ceil(math.sqrt(len(graph))))
        return {n: (i % width, -(i // width)) for i, n in enumerate(graph)}
    raise ValueError("layout must be spring, circular, or grid; pass positions for manual layout")


def draw_graph(
    graph, *, ax=None, positions=None, layout="spring", style="default", output=None, labels=True
):
    """Draw a CVGraph and return Axes; optionally export PNG, SVG or PDF.

    Args:
        graph (CVGraph): Weighted CVGraph with arbitrary hashable labels.
        ax (Axes): Existing axes or None to create a figure.
        positions (object): Complete node-to-coordinate mapping, overriding layout.
        layout (object): spring, circular, or grid.
        style (object): default or monochrome.
        output (object): Optional output filename.
        labels (tuple): Show squeezing and CZ weights.
    """
    ax = _axes(ax)
    pos = _positions(graph.network, layout, positions)
    nx.draw_networkx_edges(graph.network, pos, ax=ax, edge_color="#64748b")
    for node in graph.nodes:
        role = (
            "input" if node in graph.inputs else "output" if node in graph.outputs else "resource"
        )
        nx.draw_networkx_nodes(
            graph.network,
            pos,
            ax=ax,
            nodelist=[node],
            node_color=_color(role, style),
            node_shape={"input": "s", "output": "D", "resource": "o"}[role],
            node_size=850,
        )
    descriptions = {
        n: str(n) + (f"\nr={_text(graph.network.nodes[n]['squeezing'])}" if labels else "")
        for n in graph.nodes
    }
    nx.draw_networkx_labels(graph.network, pos, descriptions, ax=ax, font_size=8)
    if labels:
        nx.draw_networkx_edge_labels(
            graph.network,
            pos,
            ax=ax,
            edge_labels={(u, v): _text(d["weight"]) for u, v, d in graph.network.edges(data=True)},
            font_size=7,
        )
    ax.set_title("Resource graph · squares: inputs · diamonds: outputs")
    ax.set_axis_off()
    return _finish(ax, output)


def draw_pattern(
    pattern,
    *,
    ax=None,
    positions=None,
    layout="spring",
    style="default",
    output=None,
    labels=True,
    dependencies=True,
    node_colors=None,
):
    """Draw connectivity, measurement order and classical dependencies; return Axes.

    Repeated CZ edges are aggregated by Pattern.graph. Temporal operations are
    available in draw_dependencies, not recoverable from connectivity alone.
    """
    ax = _axes(ax)
    graph = pattern.graph
    pos = _positions(graph.network, layout, positions)
    nx.draw_networkx_edges(graph.network, pos, ax=ax, edge_color="#94a3b8")
    measurements = {
        c.node: (i, c) for i, c in enumerate(pattern.commands) if isinstance(c, Measure)
    }
    descriptions = {}
    for node in graph.nodes:
        if node in measurements:
            index, command = measurements[node]
            role = "non_gaussian" if isinstance(command.measurement, PhotonNumber) else "gaussian"
            shape = "^" if role == "non_gaussian" else "o"
            detail = f"\n#{index} {type(command.measurement).__name__}\nkey={command.result_key}"
            if hasattr(command.measurement, "angle"):
                detail += f" θ={_text(command.measurement.angle)}"
        else:
            role = "output" if node in pattern.outputs else "resource"
            shape, detail = ("D" if role == "output" else "o"), ""
        if node in pattern.inputs:
            shape = "s"
        color = (node_colors or {}).get(node, _color(role, style))
        nx.draw_networkx_nodes(
            graph.network,
            pos,
            nodelist=[node],
            ax=ax,
            node_color=[color],
            node_shape=shape,
            node_size=950,
            edgecolors="#0f172a" if node in pattern.inputs else "white",
            linewidths=2,
        )
        descriptions[node] = str(node) + (
            detail + f"\nr={_text(graph.network.nodes[node]['squeezing'])}" if labels else ""
        )
    nx.draw_networkx_labels(graph.network, pos, descriptions, ax=ax, font_size=7)
    if labels:
        nx.draw_networkx_edge_labels(
            graph.network,
            pos,
            ax=ax,
            edge_labels={(u, v): _text(d["weight"]) for u, v, d in graph.network.edges(data=True)},
            font_size=7,
        )
    if dependencies:
        sources: dict = {}
        for command in pattern.commands:
            source_nodes = set().union(
                *(sources.get(k, set()) for k in command_dependencies(command))
            )
            target = getattr(command, "node", None)
            if target in pos:
                for source in source_nodes:
                    if source != target:
                        ax.annotate(
                            "",
                            xy=pos[target],
                            xytext=pos[source],
                            arrowprops={
                                "arrowstyle": "->",
                                "color": "#c2410c",
                                "linestyle": "--",
                                "connectionstyle": "arc3,rad=0.18",
                                "shrinkA": 22,
                                "shrinkB": 22,
                            },
                        )
            if isinstance(command, Measure):
                sources[command.result_key] = {command.node}
            elif hasattr(command, "key"):
                sources[command.key] = source_nodes
    ax.set_title(
        "MBQC connectivity · squares: inputs · diamonds: outputs\ntriangles: PNR · dashed arrows: classical control",
        fontsize=10,
    )
    ax.set_axis_off()
    return _finish(ax, output)


def draw_circuit(
    circuit, *, ax=None, output=None, style="default", gate_colors=None, wire_labels=None
):
    """Draw optical wires with gates and return customizable matplotlib Axes."""
    from matplotlib.patches import FancyBboxPatch

    ax = _axes(ax, (max(6, min(24, len(circuit.gates) * 1.3)), max(2.5, circuit.modes)))
    for mode in range(circuit.modes):
        ax.plot([0, len(circuit.gates) + 1], [-mode, -mode], color="#64748b")
        ax.text(
            -0.15,
            -mode,
            str(wire_labels[mode]) if wire_labels else f"mode {mode}",
            ha="right",
            va="center",
            fontsize=9,
        )
    labels = {i: (name, value) for i, name, value in circuit._labels}
    names = {
        "symplectic": "G",
        "cubic_phase": "CP",
        "kerr": "Kerr",
        "cz": "CZ",
        "beamsplitter": "BS",
        "displace": "D(q,p)",
        "wire": "I",
    }
    for index, (name, modes, params) in enumerate(circuit.gates):
        x = index + 1
        if len(modes) > 1:
            ax.plot([x, x], [-max(modes), -min(modes)], color="#334155")
        label = names.get(name, name)
        if index in labels:
            label = f"{labels[index][0]}({_text(labels[index][1])})"
        elif name not in ("symplectic", "wire"):
            label += "\n" + ", ".join(_text(p) for p in params)
        for mode in modes:
            ax.add_patch(
                FancyBboxPatch(
                    (x - 0.38, -mode - 0.22),
                    0.76,
                    0.44,
                    boxstyle="round,pad=0.02",
                    facecolor=(gate_colors or {}).get(index, _color("resource", style)),
                    edgecolor="#334155",
                )
            )
            ax.text(x, -mode, label, ha="center", va="center", fontsize=8)
    ax.set_xlim(-0.8, len(circuit.gates) + 1)
    ax.set_ylim(-circuit.modes + 0.4, 0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title("Photonic circuit")
    return _finish(ax, output)


def draw_commands(pattern, **kwargs):
    """Draw a pattern's physical command sequence on labelled optical wires.

    Includes preparations, Gaussian/nonlinear gates and measurements. Wires
    indicate label lanes, not continued quantum survival after measurement.
    Classical signals/corrections remain available in the dependency view.
    """
    from types import SimpleNamespace

    from . import commands as c

    labels = tuple(pattern.graph.nodes)
    gates = []
    params: tuple
    names = {
        c.Rotate: ("R", "angle"),
        c.Squeeze: ("S", "r"),
        c.Kerr: ("Kerr", "kappa"),
        c.CubicPhase: ("CP", "gamma"),
        c.QuadraticPhase: ("Q", "s"),
        c.Loss: ("Loss", "transmissivity"),
    }
    for command in pattern.commands:
        if isinstance(command, (c.Output, c.Signal)):
            continue
        modes = tuple(labels.index(n) for n in c.quantum_nodes(command))
        if isinstance(command, c.Measure):
            name = "PNR" if isinstance(command.measurement, PhotonNumber) else "Measure"
            params = (getattr(command.measurement, "angle", type(command.measurement).__name__),)
        elif isinstance(command, c.Entangle):
            name, params = "cz", (command.weight,)
        elif isinstance(command, c.BeamSplitter):
            name, params = "beamsplitter", (command.theta,)
        elif isinstance(command, c.Displace):
            name, params = "displace", (command.q, command.p)
        elif type(command) in names:
            name, parameter = names[type(command)]
            params = (getattr(command, parameter),)
        else:
            name, params = type(command).__name__, ()
        gates.append((name, modes, params))
    drawing = SimpleNamespace(modes=len(labels), gates=gates, _labels=[])
    output = kwargs.pop("output", None)
    ax = draw_circuit(drawing, wire_labels=labels, **kwargs)
    ax.set_title("Physical command sequence (label lanes)")
    return _finish(ax, output)


def visualize_compilation(
    circuit, compiled, *, trace=None, output=None, layout="spring", style="default"
):
    """Return a Figure linking source gates to their generated MBQC commands.

    Trace must belong to the unchanged circuit/pattern. In-memory compiled
    patterns retain their trace; serialized patterns require the explicit trace.
    """
    import matplotlib.pyplot as plt

    from .compiler import _circuit_signature, _pattern_signature

    trace = trace if trace is not None else getattr(compiled, "_compilation_trace", None)
    if (
        trace is None
        or trace.pattern_signature != _pattern_signature(compiled)
        or trace.source_signature != _circuit_signature(circuit)
    ):
        raise ValueError("Supply the unchanged circuit, compiled pattern and its compilation trace")
    if len(circuit.gates) > 24 or len(compiled.graph.nodes) > 150:
        raise ValueError(
            "Compilation is too large for one overview; draw individual synthesis blocks"
        )
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.5]})
    cmap = plt.get_cmap("tab20")
    gate_colors = {s.gate_index: cmap(s.gate_index % 20) for s in trace.steps}
    node_colors = {n: gate_colors[s.gate_index] for s in trace.steps for n in s.resource_nodes}
    draw_circuit(circuit, ax=axes[0], style=style, gate_colors=gate_colors)
    draw_pattern(
        compiled, ax=axes[1], layout=layout, style=style, node_colors=node_colors, labels=False
    )
    positions = _positions(compiled.graph.network, layout, None)
    for command in compiled.commands:
        if isinstance(command, Measure) and hasattr(command.measurement, "angle"):
            axes[1].annotate(
                f"theta={_text(command.measurement.angle)}",
                positions[command.node],
                xytext=(0, -24),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    fig.text(0.47, 0.91, "Compile →", ha="center", fontsize=14, weight="bold")
    lines = [
        f"Gate {s.gate_index}: {s.source_gate} → nodes {s.resource_nodes}; measurements {s.measurements}; corrections {s.corrections}"
        for s in trace.steps
    ]
    fig.text(0.02, 0.02, "\n".join(lines), fontsize=8, va="bottom", wrap=True)
    fig.subplots_adjust(bottom=min(0.4, 0.1 + 0.025 * len(lines)), top=0.85)
    if output:
        fig.savefig(output, bbox_inches="tight")
    return fig


def draw_dependencies(pattern, *, ax=None, output=None):
    """Draw the causal command DAG; return Axes. This is not a flow certificate."""
    ax = _axes(ax)
    graph = pattern.dependencies()
    for layer, nodes in enumerate(nx.topological_generations(graph)):
        for n in nodes:
            graph.nodes[n]["layer"] = layer
    if graph:
        nx.draw_networkx(
            graph,
            nx.multipartite_layout(graph, subset_key="layer"),
            ax=ax,
            labels={i: f"{i}: {type(c).__name__}" for i, c in enumerate(pattern.commands)},
            node_color="#b9d8db",
            node_size=900,
            font_size=7,
            arrows=True,
        )
    ax.set_title("Causal command dependencies")
    ax.set_axis_off()
    return _finish(ax, output)
