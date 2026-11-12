"""Typed open-graph IR: model semantics, CVGraph conversion and JSON round trips."""

import json

import networkx as nx
import numpy as np
import pytest

import photographiq as pg
from photographiq.ir import HomodyneSpec, Model, OpenGraph, Plane, QubitMeasurement


def _json(graph):
    return OpenGraph.from_dict(json.loads(json.dumps(graph.to_dict())))


def test_cvgraph_conversion_preserves_weights_squeezing_and_roles():
    graph = pg.CVGraph.line(3, squeezing=0.7, inputs=(0,), outputs=(2,))
    graph.network.edges[0, 1]["weight"] = -0.4
    view = OpenGraph.from_cvgraph(graph, {0: 0.3})
    assert view.model is Model.CV_GAUSSIAN and view.measured == (0, 1)
    assert view.measurements[0] == HomodyneSpec(0.3) and view.measurements[1].shear == 0.0
    np.testing.assert_array_equal(view.adjacency(), [[0, -0.4, 0], [-0.4, 0, 1], [0, 1, 0]])
    back = view.to_cvgraph()
    assert back.inputs == (0,) and back.outputs == (2,)
    assert nx.utils.graphs_equal(back.network, graph.network)


def test_json_round_trip_restores_tuple_labels():
    view = OpenGraph.from_cvgraph(pg.CVGraph.rectangular(2, 3), {(0, 0): -1.5})
    restored = _json(view)
    assert restored.nodes == view.nodes and restored.to_dict() == view.to_dict()
    assert restored.measurements[(0, 0)] == HomodyneSpec(-1.5)


def test_qubit_model_semantics_are_enforced():
    graph = nx.path_graph(3)
    qubit = OpenGraph(Model.QUBIT, graph, (0,), (2,), {0: QubitMeasurement("YZ", 0.2)})
    assert _json(qubit).measurements[0] == QubitMeasurement(Plane.YZ, 0.2)
    with pytest.raises(TypeError):
        OpenGraph(Model.QUBIT, graph, (0,), (2,), {0: HomodyneSpec()})
    weighted = nx.Graph([(0, 1, {"weight": 0.5})])
    with pytest.raises(ValueError, match="unweighted"):
        OpenGraph(Model.QUBIT, weighted, (0,), (1,))
    with pytest.raises(ValueError, match="Squeezing"):
        OpenGraph(Model.QUBIT, graph, squeezing={0: 1.0})
    with pytest.raises(ValueError, match="not measured"):
        OpenGraph(Model.CV_GAUSSIAN, graph, (), (2,), {2: HomodyneSpec()})
    with pytest.raises(ValueError):
        QubitMeasurement(pauli="W")
