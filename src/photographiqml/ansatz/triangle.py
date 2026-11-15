"""Paper-layer geometry, independently expressed in semantic coordinates."""

from dataclasses import dataclass

import networkx as nx


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TriangleNeuron:
    """One paper (n,pivot) layer; nodes are (wire, local column).

    A triangle comprises its tip and three base sites. ``connections=None``
    connects every non-pivot wire; an empty tuple gives disconnected wires.
    """

    n_wires: int = 2
    pivot: int = 0
    connections: tuple | None = None

    def __post_init__(self):
        positive_integer(self.n_wires, "n_wires")
        if (
            isinstance(self.pivot, bool)
            or not isinstance(self.pivot, int)
            or not 0 <= self.pivot < self.n_wires
        ):
            raise ValueError("pivot must index a wire")
        connected = (
            tuple(w for w in range(self.n_wires) if w != self.pivot)
            if self.connections is None
            else tuple(self.connections)
        )
        if len(set(connected)) != len(connected) or any(
            isinstance(w, bool)
            or not isinstance(w, int)
            or w == self.pivot
            or not 0 <= w < self.n_wires
            for w in connected
        ):
            raise ValueError("connections must be distinct non-pivot wire indices")
        object.__setattr__(self, "connections", connected)

    @property
    def graph(self):
        graph = nx.Graph()
        graph.add_nodes_from((w, c) for w in range(self.n_wires) for c in range(5))
        graph.add_edges_from(((w, c), (w, c + 1)) for w in range(self.n_wires) for c in range(4))
        for w in self.connections or ():
            graph.add_edges_from([((self.pivot, 1), (w, 0)), ((self.pivot, 1), (w, 2))])
        return graph

    @property
    def input_nodes(self):
        return tuple((w, 0) for w in range(self.n_wires))

    @property
    def output_nodes(self):
        return tuple((w, 4) for w in range(self.n_wires))

    @property
    def trainable_nodes(self):
        return tuple((w, c) for w in range(self.n_wires) for c in range(4))

    def draw(self, ax=None):
        """Draw the logical graph; this is not an optical layout."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        nx.draw(self.graph, pos={v: (v[1], -v[0]) for v in self.graph}, ax=ax, with_labels=True)
        return ax
