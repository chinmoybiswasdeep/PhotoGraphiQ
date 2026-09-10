"""Static, exportable resource and dependency diagrams (optional matplotlib)."""

import networkx as nx

from .commands import Measure


def draw_graph(graph, *, ax=None, positions=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    positions = nx.spring_layout(graph.network, seed=7) if positions is None else positions
    colors = [
        "#41b3a3" if n in graph.inputs else "#edab4c" if n in graph.outputs else "#a8c7e0"
        for n in graph.nodes
    ]
    labels = {n: f"{n}\nr={graph.network.nodes[n]['squeezing']}" for n in graph.nodes}
    nx.draw_networkx(
        graph.network,
        pos=positions,
        ax=ax,
        node_color=colors,
        labels=labels,
        node_size=1000,
        font_size=8,
    )
    nx.draw_networkx_edge_labels(
        graph.network,
        positions,
        ax=ax,
        edge_labels=nx.get_edge_attributes(graph.network, "weight"),
        font_size=8,
    )
    ax.set_axis_off()
    return ax


def draw_pattern(pattern, *, ax=None):
    ax = draw_graph(pattern.graph, ax=ax)
    info = []
    for i, c in enumerate(pattern.commands):
        if isinstance(c, Measure):
            info.append(
                f"{i}: M({c.node}), angle={getattr(c.measurement, 'angle', type(c.measurement).__name__)}"
            )
    ax.set_title("PhotoGraphiQ resource (green: input; amber: output)")
    if info:
        ax.text(0, -0.05, "\n".join(info), transform=ax.transAxes, fontsize=7, va="top", wrap=True)
    return ax


def draw_dependencies(pattern, *, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    graph = pattern.dependencies()
    labels = {i: f"{i}: {type(c).__name__}" for i, c in enumerate(pattern.commands)}
    for layer, nodes in enumerate(nx.topological_generations(graph)):
        for n in nodes:
            graph.nodes[n]["layer"] = layer
    pos = nx.multipartite_layout(graph, subset_key="layer")
    nx.draw_networkx(
        graph,
        pos,
        ax=ax,
        labels=labels,
        node_color="#b9d8db",
        node_size=1000,
        font_size=7,
        arrows=True,
    )
    ax.set_title("Causal command dependencies (not a flow certificate)")
    ax.set_axis_off()
    return ax
