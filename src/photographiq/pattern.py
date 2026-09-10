"""A measurement computation, independent of simulator state and backend."""

from __future__ import annotations

from copy import deepcopy

from .commands import (
    COMMANDS,
    Displace,
    Entangle,
    Measure,
    Output,
    Prepare,
    Signal,
    command_dependencies,
    expressions,
    quantum_nodes,
)
from .expressions import CallableExpression
from .flow import dependency_graph, topological_schedule
from .graph import CVGraph
from .measurements import Generaldyne, Heterodyne, Homodyne, PhotonNumber


class Pattern:
    def __init__(self, graph: CVGraph | None = None, *, inputs=()):
        self.commands: list = []
        self.inputs = tuple(inputs if graph is None else graph.inputs)
        self.resource = None if graph is None else graph.copy()
        if graph is not None:
            graph.validate()
            if graph.ideal:
                raise ValueError(
                    "Ideal graph is symbolic; choose finite squeezing to build a physical pattern"
                )
            for node in graph.nodes:
                if node not in self.inputs:
                    self.append(Prepare(node, graph.network.nodes[node]["squeezing"]))
            self.extend(Entangle(u, v, d["weight"]) for u, v, d in graph.network.edges(data=True))

    def append(self, command):
        if not isinstance(command, COMMANDS):
            raise TypeError(f"Unknown command: {type(command).__name__}")
        self.commands.append(command)
        return self

    def extend(self, commands):
        for command in commands:
            self.append(command)
        return self

    def copy(self):
        return deepcopy(self)

    def measure(self, node, measurement=None, *, key=None):
        return self.append(Measure(node, Homodyne.q() if measurement is None else measurement, key))

    def displace(self, node, *, q=0.0, p=0.0):
        return self.append(Displace(node, q, p))

    @property
    def parameters(self):
        return frozenset().union(*(e.parameters for c in self.commands for e in expressions(c)))

    @property
    def outputs(self):
        explicit = [c.nodes for c in self.commands if isinstance(c, Output)]
        if explicit:
            return tuple(explicit[-1])
        active = list(self.inputs)
        for command in self.commands:
            if isinstance(command, Prepare):
                active.append(command.node)
            elif isinstance(command, Measure) and command.node in active:
                active.remove(command.node)
        return tuple(active)

    @property
    def graph(self):
        graph = CVGraph(inputs=())
        for n in self.inputs:
            graph.add_node(n, squeezing=0.0)
        for c in self.commands:
            if isinstance(c, Prepare):
                graph.add_node(c.node, c.squeezing)
            elif isinstance(c, Entangle):
                if graph.network.has_edge(c.u, c.v):
                    old = graph.network[c.u][c.v]["weight"]
                    graph.network[c.u][c.v]["weight"] = old + c.weight
                else:
                    graph.add_edge(c.u, c.v, c.weight)
        graph.inputs, graph.outputs = self.inputs, self.outputs
        return graph

    def validate(self):
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("Duplicate inputs")
        dependency_graph(self.commands)
        active, ever = set(self.inputs), set(self.inputs)
        records: set = set()
        output_seen = False
        for i, c in enumerate(self.commands):
            if output_seen:
                raise ValueError("Output must be the final command")
            if not command_dependencies(c) <= records:
                raise ValueError(f"Command {i} depends on a future outcome")
            if isinstance(c, Prepare):
                if c.node in ever:
                    raise ValueError(f"Node {c.node!r} prepared twice")
                active.add(c.node)
                ever.add(c.node)
            else:
                if not set(quantum_nodes(c)) <= active:
                    raise ValueError(f"Command {i} uses a missing or measured node")
                if isinstance(c, Entangle) and c.u == c.v:
                    raise ValueError("Self CZ is invalid")
            if isinstance(c, Measure):
                if not isinstance(c.measurement, (Homodyne, Heterodyne, Generaldyne, PhotonNumber)):
                    raise NotImplementedError("Unsupported measurement description")
                active.remove(c.node)
                records.add(c.result_key)
            elif isinstance(c, Signal):
                records.add(c.key)
            elif isinstance(c, Output):
                if len(set(c.nodes)) != len(c.nodes):
                    raise ValueError("Duplicate outputs")
                output_seen = True
        if self.resource and self.resource.outputs and self.outputs != self.resource.outputs:
            raise ValueError(
                "Remaining modes differ from declared graph outputs; add an Output command"
            )
        return self

    def dependencies(self):
        return dependency_graph(self.commands)

    def schedule(self):
        return topological_schedule(self.commands)

    def inspect(self):
        return "\n".join(f"{i}: {command!r}" for i, command in enumerate(self.commands))

    def standardize(self):
        """Move preparations/CZ only across disjoint commands; combine adjacent shifts.

        Does not promise a full N-E-M-X-Z normal form for arbitrary adaptive patterns.
        """
        self.validate()
        result = self.copy()
        seq = result.commands
        for kind in (Prepare, Entangle):
            for i in range(len(seq)):
                j = i
                if not isinstance(seq[j], kind):
                    continue
                while j and not isinstance(seq[j - 1], (Prepare, kind)):
                    a, b = seq[j - 1], seq[j]
                    if set(quantum_nodes(a)) & set(quantum_nodes(b)):
                        break
                    produced = (
                        a.result_key
                        if isinstance(a, Measure)
                        else a.key
                        if isinstance(a, Signal)
                        else None
                    )
                    if produced is not None and produced in command_dependencies(b):
                        break
                    seq[j - 1], seq[j] = seq[j], seq[j - 1]
                    j -= 1
        combined: list = []
        for c in seq:
            if (
                isinstance(c, Displace)
                and combined
                and isinstance(combined[-1], Displace)
                and combined[-1].node == c.node
                and not any(
                    isinstance(v, CallableExpression)
                    for v in (combined[-1].q, combined[-1].p, c.q, c.p)
                )
            ):
                prev = combined.pop()
                c = Displace(c.node, prev.q + c.q, prev.p + c.p)
            if (
                isinstance(c, Displace)
                and isinstance(c.q, (int, float))
                and isinstance(c.p, (int, float))
                and c.q == c.p == 0
            ):
                continue
            combined.append(c)
        result.commands = combined
        return result.validate()

    def to_json(self, path=None):
        from .serialization import dumps

        text = dumps(self)
        if path is not None:
            from pathlib import Path

            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, text):
        from .serialization import loads

        return loads(text)

    def draw(self, **kwargs):
        from .visualization import draw_pattern

        return draw_pattern(self, **kwargs)
