"""Causal scheduling only: a DAG is not a CV-flow/determinism certificate."""

from __future__ import annotations

import networkx as nx

from .commands import Measure, Signal, command_dependencies, quantum_nodes


def dependency_graph(commands):
    graph = nx.DiGraph()
    producers = {}
    for i, command in enumerate(commands):
        graph.add_node(i)
        if isinstance(command, (Measure, Signal)):
            key = command.result_key if isinstance(command, Measure) else command.key
            if key in producers:
                raise ValueError(f"Duplicate classical key: {key!r}")
            producers[key] = i
    last_touch: dict = {}
    for i, command in enumerate(commands):
        for key in command_dependencies(command):
            if key not in producers:
                raise ValueError(f"No producer for classical key: {key!r}")
            graph.add_edge(producers[key], i)
        for node in quantum_nodes(command):
            if node in last_touch:
                graph.add_edge(last_touch[node], i)
            last_touch[node] = i
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Cyclic command dependencies")
    return graph


def topological_schedule(commands):
    return tuple(nx.lexicographical_topological_sort(dependency_graph(commands)))
