"""Faithful logical MuTA geometry and flow with explicit representation boundaries."""

import json
import warnings
from pathlib import Path

import networkx as nx
import numpy as np

from ..logical import execute
from ..parameters import ParameterStore
from .triangle import TriangleNeuron, positive_integer


class MuTA:
    """MuTA with MentPy's block-count convention and persistent fixed measurements.

    ``one_column=False`` gives n_wires paper layers per requested layer.
    ``restrict_trainable=False`` is the full paper model (the default).
    Fixed column-3 measurements are retained across composition when enabled.
    ``connections=()`` removes triangles; None uses fully connected blocks.
    """

    def __init__(
        self,
        n_wires=2,
        n_layers=1,
        *,
        one_column=False,
        restrict_trainable=False,
        representation="logical",
        connections=None,
    ):
        positive_integer(n_wires, "n_wires")
        positive_integer(n_layers, "n_layers")
        if not isinstance(one_column, bool) or not isinstance(restrict_trainable, bool):
            raise ValueError("one_column and restrict_trainable must be booleans")
        if representation not in ("logical", "gkp", "gkp-resource"):
            raise ValueError(
                "Choose logical or gkp-resource; use PhysicalMuTA for gkp-physical. CVMuTA is a separate proposal"
            )
        if representation == "gkp":
            warnings.warn(
                "Legacy 'gkp' remains resource-only. Use 'gkp-resource' or explicitly construct PhysicalMuTA for the supported physical subset.",
                DeprecationWarning,
                stacklevel=2,
            )
        if connections is not None and not one_column:
            raise ValueError("Custom connections require one_column=True")
        self.n_wires, self.n_layers = n_wires, n_layers
        self.one_column, self.restrict_trainable = one_column, restrict_trainable
        self.representation = representation
        self.connections = None if connections is None else tuple(connections)
        self.pivots = tuple([0] if one_column else range(n_wires)) * n_layers
        self.paper_depth = len(self.pivots)
        graph = nx.Graph()
        for block, pivot in enumerate(self.pivots):
            cell = TriangleNeuron(n_wires, pivot, self.connections)
            mapping = {v: (v[0], 4 * block + v[1]) for v in cell.graph}
            graph.update(nx.relabel_nodes(cell.graph, mapping))
        self.graph = nx.freeze(graph)
        self.input_nodes = tuple((w, 0) for w in range(n_wires))
        self.output_nodes = tuple((w, 4 * self.paper_depth) for w in range(n_wires))
        self.flow = {v: (v[0], v[1] + 1) for v in graph if v not in self.output_nodes}
        dag = nx.DiGraph()
        dag.add_nodes_from(graph)
        self.corrections = {}
        for node, successor in self.flow.items():
            z = frozenset(graph.neighbors(successor)) - {node}
            self.corrections[node] = (successor, z)
            dag.add_edges_from((node, target) for target in z | {successor})
        self.dependency_graph = nx.freeze(dag)
        self.measurement_order = tuple(
            v
            for v in nx.lexicographical_topological_sort(dag, key=lambda v: (v[1], v[0]))
            if v in self.flow
        )
        self.measured_nodes = tuple(sorted(self.flow, key=lambda v: (v[1] // 4, v[0], v[1] % 4)))
        names = [self.parameter_name(v) for v in self.measured_nodes]
        fixed = [
            self.parameter_name(v)
            for v in self.measured_nodes
            if restrict_trainable and v[1] % 4 == 3
        ]
        self._parameters = ParameterStore(names, frozen=fixed)

    @staticmethod
    def parameter_name(node):
        return f"alpha.w{node[0]}.c{node[1]}"

    @property
    def trainable_nodes(self):
        return tuple(
            v for v in self.measured_nodes if self.parameter_name(v) not in self._parameters.frozen
        )

    @property
    def n_parameters(self):
        return len(self.trainable_nodes)

    @property
    def expected_parameter_count(self):
        """Construction-time analytical count, before user freeze/unfreeze."""
        return (3 if self.restrict_trainable else 4) * self.n_wires * self.paper_depth

    def parameters(self):
        return dict(self._parameters.symbols)

    def trainable_parameters(self):
        return {n: p for n, p in self.parameters().items() if n not in self._parameters.frozen}

    def freeze(self, names, value=None):
        self._parameters.freeze(names, value)
        return self

    def unfreeze(self, names):
        self._parameters.unfreeze(names)
        return self

    def initialize(self, seed=None, scale=0.1):
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("Initialization scale must be finite and nonnegative")
        return dict(
            zip(
                self.trainable_parameters(),
                np.random.default_rng(seed).normal(0, scale, self.n_parameters),
            )
        )

    def run(self, input_state, parameters=None):
        if self.representation in ("gkp", "gkp-resource"):
            raise NotImplementedError(
                "Physical GKP MuTA requires a validated logical XY measurement/injection instrument and decoder; use GKPBridge.logical_target explicitly for ideal targets"
            )
        return execute(self, input_state, self._parameters.bind(parameters))

    def unitary(self, parameters=None):
        if self.n_wires > 10:
            raise ValueError("Dense unitary is limited to 10 wires; use run for statevectors")
        return np.column_stack([self.run(v, parameters).state for v in np.eye(2**self.n_wires)])

    def run_batch(self, states, parameters=None):
        states = np.asarray(states, dtype=complex)
        if states.ndim != 2 or states.shape[1] != 2**self.n_wires:
            raise ValueError(f"Batch must have shape (N,{2**self.n_wires})")
        if not len(states):
            return np.empty_like(states)
        return np.array([self.run(s, parameters).state for s in states])

    def state_dict(self):
        return {
            "schema": 1,
            "config": {
                "n_wires": self.n_wires,
                "n_layers": self.n_layers,
                "one_column": self.one_column,
                "restrict_trainable": self.restrict_trainable,
                "representation": self.representation,
                "connections": self.connections,
            },
            "values": dict(self._parameters.values),
            "frozen": sorted(self._parameters.frozen),
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["schema"] not in (1, 2):
            raise ValueError("Unsupported model schema")
        model: MuTA
        if data["schema"] == 2 and data["config"]["representation"] == "gkp-physical":
            from ..lowering import GKPPhysicalConfig
            from ..physical import PhysicalMuTA

            model = PhysicalMuTA(
                **data["config"],
                physical_config=GKPPhysicalConfig(**data["physical_config"]),
                measurement_family=data["measurement_family"],
            )
        else:
            # Loading through a subclass must not reinterpret schema-1 resources.
            model = MuTA(**data["config"])
        if set(data["values"]) != set(model.parameters()):
            raise ValueError("Serialized parameter names do not match topology")
        model._parameters.frozen.clear()
        model._parameters.values = model._parameters.bind(data["values"])
        model.freeze(data["frozen"])
        return model

    def physical_capabilities(self, parameters=None, *, config=None):
        from ..lowering import physical_capabilities

        return physical_capabilities(
            self, parameters, config=config or getattr(self, "physical_config", None)
        )

    def draw_physical(self, parameters=None, *, config=None, ax=None):
        from ..lowering import visualize_lowering

        return visualize_lowering(
            self, parameters, config=config or getattr(self, "physical_config", None), ax=ax
        )

    def summary(self):
        return f"MuTA ({self.representation})\nWires: {self.n_wires}\nRequested layers: {self.n_layers}\nPaper layers: {self.paper_depth}\nTrainable parameters: {self.n_parameters}\nResource nodes: {len(self.graph)}\nMeasurements: {len(self.flow)}\nCZ edges: {self.graph.number_of_edges()}\nCausal dependencies: {self.dependency_graph.number_of_edges()}"

    def draw(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(max(5, self.paper_depth * 3), self.n_wires + 1))
        colors = [
            "skyblue"
            if v in self.output_nodes
            else "gold"
            if v in self.trainable_nodes
            else "lightgray"
            for v in self.graph
        ]
        nx.draw(
            self.graph,
            {v: (v[1], -v[0]) for v in self.graph},
            ax=ax,
            node_color=colors,
            with_labels=True,
        )
        return ax
