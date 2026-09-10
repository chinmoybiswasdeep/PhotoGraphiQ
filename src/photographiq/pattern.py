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
    PrepareResource,
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
    """Causal measurement computation with ordered input labels and backend-neutral commands.

    Args:
        graph (CVGraph): Labelled weighted resource graph.
        inputs (tuple): Ordered input labels supplied externally.

    Raises:
        ValueError: Duplicate inputs.
        ValueError: Remaining modes differ from declared graph outputs; add an Output command.
        ValueError: Ideal graph is symbolic; choose finite squeezing to build a physical pattern.
        ValueError: Output must be the final command.
        ValueError: Empty, duplicate or previously prepared resource nodes.
    """

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
        """Append one supported command and return this pattern.

        Args:
            command (object): Backend-neutral command.

        Returns:
            result (Pattern): This pattern for fluent construction.
        """
        if not isinstance(command, COMMANDS):
            raise TypeError(f"Unknown command: {type(command).__name__}")
        self.commands.append(command)
        return self

    def __repr__(self):
        return f"Pattern(inputs={len(self.inputs)}, outputs={len(self.outputs)}, commands={len(self.commands)})"

    def extend(self, commands):
        """Append an iterable of commands and return this pattern.

        Args:
            commands (iterable): Commands in intended execution order.

        Returns:
            result (Pattern): This pattern for fluent construction.
        """
        for command in commands:
            self.append(command)
        return self

    def copy(self):
        """Return an independent copy of this object.

        Returns:
            result (object): Independent object copy.
        """
        return deepcopy(self)

    def measure(self, node, measurement=None, *, key=None):
        """Append a destructive measurement; its key becomes a later classical dependency.

        Args:
            node (object): Hashable mode label.
            measurement (object): Homodyne, Heterodyne, Generaldyne or PhotonNumber description.
            key (object): Unique classical record key; None uses the measured node.

        Returns:
            result (object): Pattern when constructing; sampled outcome when executing.
        """
        return self.append(Measure(node, Homodyne.q() if measurement is None else measurement, key))

    def displace(self, node, *, q=0.0, p=0.0):
        """Apply or append quadrature translations q and p in hbar=2 coordinates.

        Args:
            node (object): Hashable mode label.
            q (float): Position translation; hbar=2 quadrature units.
            p (float): Momentum translation; hbar=2 quadrature units.
        """
        return self.append(Displace(node, q, p))

    @property
    def parameters(self):
        """Names of external scalar parameters required by this object.

        Returns:
            result (frozenset): External parameter names.
        """
        return frozenset().union(*(e.parameters for c in self.commands for e in expressions(c)))

    @property
    def outputs(self):
        """Ordered labels of surviving modes selected by the final Output command."""
        explicit = [c.nodes for c in self.commands if isinstance(c, Output)]
        if explicit:
            return tuple(explicit[-1])
        active = list(self.inputs)
        for command in self.commands:
            if isinstance(command, Prepare):
                active.append(command.node)
            elif isinstance(command, PrepareResource):
                active.extend(command.nodes)
            elif isinstance(command, Measure) and command.node in active:
                active.remove(command.node)
        return tuple(active)

    @property
    def graph(self):
        """Construct entangling connectivity; repeated CZ edges are aggregated, not temporally simulated."""
        graph = CVGraph(inputs=())
        for n in self.inputs:
            graph.add_node(n, squeezing=0.0)
        for c in self.commands:
            if isinstance(c, Prepare):
                graph.add_node(c.node, c.squeezing)
            elif isinstance(c, PrepareResource):
                for node in c.nodes:
                    graph.add_node(node, squeezing=0.0)
            elif isinstance(c, Entangle):
                if graph.network.has_edge(c.u, c.v):
                    old = graph.network[c.u][c.v]["weight"]
                    graph.network[c.u][c.v]["weight"] = old + c.weight
                else:
                    graph.add_edge(c.u, c.v, c.weight)
        graph.inputs, graph.outputs = self.inputs, self.outputs
        return graph

    def validate(self):
        """Validate dimensions, labels, causality or physicality for this object; return self.

        Returns:
            result (object): This validated object.

        Raises:
            ValueError: Duplicate inputs.
            ValueError: Remaining modes differ from declared graph outputs; add an Output command.
            ValueError: Output must be the final command.
            ValueError: Empty, duplicate or previously prepared resource nodes.
            ValueError: Self CZ is invalid.
        """
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
            if isinstance(c, (Prepare, PrepareResource)):
                nodes = quantum_nodes(c)
                if not nodes or len(set(nodes)) != len(nodes) or set(nodes) & ever:
                    raise ValueError("Empty, duplicate or previously prepared resource nodes")
                active.update(nodes)
                ever.update(nodes)
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
        """Classical/quantum dependencies required before executing this object.

        Returns:
            result (object): Dependency keys or command DAG, according to the owning object.
        """
        return dependency_graph(self.commands)

    def schedule(self):
        """Return a causally valid ordering of command indices."""
        return topological_schedule(self.commands)

    def inspect(self):
        """Return a compact description of inputs, outputs, commands and parameters."""
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
        """Serialize the pattern using the versioned allowlisted JSON schema; optionally write a file.

        Args:
            path (str): Optional destination path.

        Returns:
            result (str): Versioned JSON text.
        """
        from .serialization import dumps

        text = dumps(self)
        if path is not None:
            from pathlib import Path

            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, text):
        """Load a pattern from JSON text, rejecting unknown types and invalid causal structure.

        Args:
            text (str): Serialized JSON text, not a filename.

        Returns:
            result (Pattern): Validated reconstructed pattern.
        """
        from .serialization import loads

        return loads(text)

    def draw(self, **kwargs):
        """Draw this computation with optional matplotlib; return Axes for customization/export.

        Returns:
            result (Axes): Customizable matplotlib axes.
        """
        from .visualization import draw_pattern

        return draw_pattern(self, **kwargs)
