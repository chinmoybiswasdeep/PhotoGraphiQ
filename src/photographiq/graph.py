"""Weighted, labelled canonical CV resource graphs."""

from __future__ import annotations

import networkx as nx
import numpy as np

from .expressions import Expr, resolve


class CVGraph:
    def __init__(self, graph=None, *, squeezing=1.0, inputs=(), outputs=(), ideal=False):
        if graph is not None and (graph.is_directed() or graph.is_multigraph()):
            raise ValueError("Resources require an undirected simple graph")
        self.network = nx.Graph() if graph is None else nx.Graph(graph)
        self.inputs, self.outputs = tuple(inputs), tuple(outputs)
        self.ideal = bool(ideal)
        for node in self.network:
            value = squeezing[node] if isinstance(squeezing, dict) else squeezing
            self.network.nodes[node].setdefault("squeezing", value)
        for _, _, data in self.network.edges(data=True):
            data.setdefault("weight", 1.0)
        self.validate()

    @property
    def nodes(self):
        return tuple(self.network.nodes)

    @property
    def ancillas(self):
        return tuple(n for n in self.nodes if n not in self.inputs)

    def validate(self):
        if nx.number_of_selfloops(self.network):
            raise ValueError("Self edges are local quadratic phases, not resource edges")
        if len(set(self.inputs)) != len(self.inputs) or len(set(self.outputs)) != len(self.outputs):
            raise ValueError("Duplicate input/output labels")
        if not set(self.inputs + self.outputs) <= set(self.nodes):
            raise ValueError("Input/output nodes must belong to the graph")
        for _, data in self.network.nodes(data=True):
            r = data["squeezing"]
            if not isinstance(r, Expr) and (not np.isfinite(r) or r < 0):
                raise ValueError(
                    "Resource squeezing must be finite and nonnegative; use ideal=True symbolically"
                )
        for _, _, data in self.network.edges(data=True):
            g = data["weight"]
            if not isinstance(g, Expr) and (not np.isfinite(g) or g == 0):
                raise ValueError("Edge weights must be finite, real and nonzero")
        return self

    def copy(self):
        return CVGraph(self.network, inputs=self.inputs, outputs=self.outputs, ideal=self.ideal)

    def add_node(self, node, squeezing=1.0):
        if node in self.network:
            raise ValueError("Node already exists")
        trial = self.copy()
        trial.network.add_node(node, squeezing=squeezing)
        trial.validate()
        self.network = trial.network
        return self

    def add_edge(self, u, v, weight=1.0):
        if u not in self.network or v not in self.network:
            raise ValueError("Create edge endpoints first")
        trial = self.copy()
        trial.network.add_edge(u, v, weight=weight)
        trial.validate()
        self.network = trial.network
        return self

    def adjacency(self, parameters=None):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for u, v, d in self.network.edges(data=True):
            i, j = self.nodes.index(u), self.nodes.index(v)
            matrix[i, j] = matrix[j, i] = resolve(d["weight"], {}, parameters or {})
        return matrix

    def nullifiers(self, parameters=None):
        """Rows representing delta_i=p_i-sum_j A_ij q_j, including ideal graphs."""
        result = np.zeros((len(self.nodes), 2 * len(self.nodes)))
        result[:, ::2] = -self.adjacency(parameters)
        result[:, 1::2] = np.eye(len(self.nodes))
        return result

    @classmethod
    def from_adjacency(cls, adjacency, labels=None, **kwargs):
        a = np.asarray(adjacency, dtype=float)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
            raise ValueError("Adjacency must be finite and square")
        if not np.allclose(a, a.T, atol=1e-12, rtol=0) or np.any(np.diag(a) != 0):
            raise ValueError("Adjacency must be symmetric with zero diagonal")
        labels = tuple(range(len(a))) if labels is None else tuple(labels)
        if len(labels) != len(a) or len(set(labels)) != len(labels):
            raise ValueError("Invalid labels")
        graph = nx.relabel_nodes(nx.from_numpy_array(a), dict(enumerate(labels)))
        return cls(graph, **kwargs)

    @classmethod
    def line(cls, modes, **kwargs):
        if not isinstance(modes, int) or modes < 1:
            raise ValueError("modes must be positive")
        return cls(nx.path_graph(modes), **kwargs)

    @classmethod
    def ring(cls, modes, **kwargs):
        if modes < 3:
            raise ValueError("A ring needs at least three modes")
        return cls(nx.cycle_graph(modes), **kwargs)

    @classmethod
    def star(cls, leaves, **kwargs):
        if leaves < 1:
            raise ValueError("Need at least one leaf")
        return cls(nx.star_graph(leaves), **kwargs)

    @classmethod
    def rectangular(cls, rows, columns, **kwargs):
        if rows < 1 or columns < 1:
            raise ValueError("Dimensions must be positive")
        return cls(nx.grid_2d_graph(rows, columns), **kwargs)

    @classmethod
    def square(cls, size, **kwargs):
        return cls.rectangular(size, size, **kwargs)

    @classmethod
    def tree(cls, branching, height, **kwargs):
        if branching < 1 or height < 0:
            raise ValueError("Invalid tree dimensions")
        return cls(nx.balanced_tree(branching, height), **kwargs)

    @classmethod
    def temporal(cls, bins, **kwargs):
        """A time-labelled line graph; not an optical hardware scheduling claim."""
        return cls.line(bins, **kwargs)

    @classmethod
    def dual_rail(cls, bins, **kwargs):
        """Two-rail ladder topology, not a calibrated experimental optical circuit."""
        return cls.rectangular(2, bins, **kwargs)


ClusterState = CVGraph
