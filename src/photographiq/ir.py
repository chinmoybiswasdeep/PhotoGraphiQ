"""Backend-independent graph-level intermediate representation.

``OpenGraph`` is the shared, typed description of a measurement-based computation
used by flow analysis, compilation, measurement calculus and visualization. It
fixes *which computational model* a graph belongs to, because flow criteria,
measurement semantics and corrections differ between models:

* ``Model.CV_GAUSSIAN``: weighted CV cluster, homodyne measurements, continuous
  outcomes, displacement feed-forward (finite squeezing is physical noise).
* ``Model.CV_NONGAUSSIAN``: as above plus non-Gaussian resources or gates.
* ``Model.QUBIT``: unweighted qubit graph state, plane/angle measurements, binary
  outcomes, Pauli X/Z corrections.
* ``Model.GKP_LOGICAL``: logical qubits encoded in finite GKP resources.

The command-level IR for CV execution remains :class:`photographiq.Pattern`; qubit
command patterns live in :mod:`photographiq.calculus`. Layout coordinates are never
stored here: they belong to visualization metadata.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .graph import CVGraph


class Model(str, enum.Enum):
    """Computational model of an open graph; models are never interchangeable."""

    CV_GAUSSIAN = "cv-gaussian"
    CV_NONGAUSSIAN = "cv-nongaussian"
    QUBIT = "qubit"
    GKP_LOGICAL = "gkp-logical"

    @property
    def continuous(self) -> bool:
        return self in (Model.CV_GAUSSIAN, Model.CV_NONGAUSSIAN)


class Plane(str, enum.Enum):
    """Qubit measurement plane of the Bloch sphere."""

    XY = "XY"
    YZ = "YZ"
    XZ = "XZ"


@dataclass(frozen=True)
class QubitMeasurement:
    """Qubit measurement in ``plane`` at ``angle`` radians.

    ``pauli`` optionally marks a Pauli measurement ("X", "Y" or "Z"), which Pauli flow
    may exploit. Measured observables (outcome 0 is the +1 eigenstate), as in Backens
    et al. and Graphix 0.4:

    * XY: cos(a) X + sin(a) Y
    * YZ: cos(a) Z + sin(a) Y
    * XZ: cos(a) Z + sin(a) X
    """

    plane: Plane = Plane.XY
    angle: float = 0.0
    pauli: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "plane", Plane(self.plane))
        if not math.isfinite(float(self.angle)):
            raise ValueError("Measurement angle must be finite")
        if self.pauli not in (None, "X", "Y", "Z"):
            raise ValueError("pauli must be None, 'X', 'Y' or 'Z'")


@dataclass(frozen=True)
class HomodyneSpec:
    """CV wire-step measurement: homodyne at angle atan2(1, shear).

    ``shear`` is the coefficient k of the teleported shear D(k) (see
    :func:`photographiq.gaussian.teleportation_matrix`); ``shear=0`` is a p measurement.
    """

    shear: float = 0.0

    @property
    def angle(self) -> float:
        return math.atan2(1.0, float(self.shear))


@dataclass
class OpenGraph:
    """Typed open graph: vertices, weighted edges, inputs, outputs and measurements.

    Args:
        model: Computational model.
        graph: Undirected simple graph; edge attribute ``weight`` (default 1).
        inputs: Ordered input labels.
        outputs: Ordered output labels (never measured).
        measurements: Measurement specification of every non-output node.
        squeezing: CV resource squeezing per node (CV models only).
    """

    model: Model
    graph: nx.Graph
    inputs: tuple = ()
    outputs: tuple = ()
    measurements: dict = field(default_factory=dict)
    squeezing: dict = field(default_factory=dict)

    def __post_init__(self):
        self.model = Model(self.model)
        self.graph = nx.Graph(self.graph)
        for _, _, data in self.graph.edges(data=True):
            data.setdefault("weight", 1.0)
        self.inputs, self.outputs = tuple(self.inputs), tuple(self.outputs)
        self.validate()

    @property
    def nodes(self) -> tuple:
        return tuple(self.graph.nodes)

    @property
    def measured(self) -> tuple:
        """Non-output nodes in graph order."""
        return tuple(n for n in self.graph.nodes if n not in self.outputs)

    def weight(self, u, v) -> float:
        return float(self.graph.edges[u, v]["weight"])

    def adjacency(self, order=None) -> np.ndarray:
        """Weighted adjacency matrix in ``order`` (default graph order)."""
        order = self.nodes if order is None else tuple(order)
        return nx.to_numpy_array(self.graph, nodelist=order, weight="weight")

    def validate(self) -> OpenGraph:
        """Check labels and model-specific semantics; return self."""
        nodes = set(self.graph.nodes)
        if nx.number_of_selfloops(self.graph):
            raise ValueError("Open graphs have no self edges")
        if len(set(self.inputs)) != len(self.inputs) or len(set(self.outputs)) != len(self.outputs):
            raise ValueError("Duplicate input/output labels")
        if not set(self.inputs) | set(self.outputs) <= nodes:
            raise ValueError("Inputs and outputs must be graph nodes")
        if not set(self.measurements) <= nodes:
            raise ValueError("Measurement specified for an unknown node")
        if set(self.measurements) & set(self.outputs):
            raise ValueError("Output nodes are not measured")
        for u, v, data in self.graph.edges(data=True):
            w = float(data["weight"])
            if not math.isfinite(w) or w == 0:
                raise ValueError(f"Edge ({u!r}, {v!r}) needs a finite nonzero weight")
            if not self.model.continuous and w != 1:
                raise ValueError(f"{self.model.value} graphs are unweighted; edge ({u!r}, {v!r})")
        spec = HomodyneSpec if self.model.continuous else QubitMeasurement
        wrong = [n for n, m in self.measurements.items() if not isinstance(m, spec)]
        if wrong:
            raise TypeError(f"{self.model.value} measurements must be {spec.__name__}: {wrong}")
        if not self.model.continuous and self.squeezing:
            raise ValueError("Squeezing is a CV resource property")
        return self

    def copy(self) -> OpenGraph:
        return OpenGraph(
            self.model,
            self.graph.copy(),
            self.inputs,
            self.outputs,
            dict(self.measurements),
            dict(self.squeezing),
        )

    @classmethod
    def from_cvgraph(cls, graph: CVGraph, shears=None, *, model=Model.CV_GAUSSIAN) -> OpenGraph:
        """Typed view of a :class:`CVGraph`; ``shears`` maps measured nodes to k."""
        model = Model(model)
        if not model.continuous:
            raise ValueError("A CVGraph converts only to a CV model")
        shears = dict(shears or {})
        measured = [n for n in graph.nodes if n not in graph.outputs]
        network = nx.Graph()
        network.add_nodes_from(graph.network.nodes)
        for u, v, data in graph.network.edges(data=True):
            network.add_edge(u, v, weight=float(data["weight"]))
        return cls(
            model,
            network,
            graph.inputs,
            graph.outputs,
            {n: HomodyneSpec(float(shears.get(n, 0.0))) for n in measured},
            {n: float(graph.network.nodes[n]["squeezing"]) for n in graph.nodes},
        )

    def to_cvgraph(self) -> CVGraph:
        """Executable CV resource graph (CV models only)."""
        if not self.model.continuous:
            raise ValueError(f"{self.model.value} graphs are not CV resources")
        network = nx.Graph()
        for node in self.graph.nodes:
            network.add_node(node, squeezing=self.squeezing.get(node, 1.0))
        network.add_weighted_edges_from(
            (u, v, float(d["weight"])) for u, v, d in self.graph.edges(data=True)
        )
        return CVGraph(network, inputs=self.inputs, outputs=self.outputs)

    def to_dict(self) -> dict:
        """JSON-compatible semantic description (no layout)."""

        def spec(m):
            if isinstance(m, HomodyneSpec):
                return {"shear": m.shear}
            return {"plane": m.plane.value, "angle": m.angle, "pauli": m.pauli}

        return {
            "model": self.model.value,
            "nodes": [
                {"id": n, **({"squeezing": self.squeezing[n]} if n in self.squeezing else {})}
                for n in self.graph.nodes
            ],
            "edges": [
                {"source": u, "target": v, "weight": float(d["weight"])}
                for u, v, d in self.graph.edges(data=True)
            ],
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "measurements": {str(n): spec(m) for n, m in self.measurements.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> OpenGraph:
        """Inverse of :meth:`to_dict`.

        JSON turns tuple labels into lists, so lists are converted back to tuples
        (recursively); measurement keys, which JSON stringifies, are matched by ``str``.
        """

        def label(value):
            return tuple(label(v) for v in value) if isinstance(value, list) else value

        model = Model(data["model"])
        network = nx.Graph()
        squeezing = {}
        for node in data["nodes"]:
            network.add_node(label(node["id"]))
            if "squeezing" in node:
                squeezing[label(node["id"])] = float(node["squeezing"])
        for edge in data["edges"]:
            network.add_edge(
                label(edge["source"]), label(edge["target"]), weight=float(edge["weight"])
            )
        by_name = {str(n): n for n in network.nodes}
        measurements: dict = {}
        for key, m in data.get("measurements", {}).items():
            node = by_name[key]
            if model.continuous:
                measurements[node] = HomodyneSpec(float(m["shear"]))
            else:
                measurements[node] = QubitMeasurement(m["plane"], float(m["angle"]), m.get("pauli"))
        return cls(
            model,
            network,
            tuple(label(n) for n in data.get("inputs", ())),
            tuple(label(n) for n in data.get("outputs", ())),
            measurements,
            squeezing,
        )
