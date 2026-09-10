import json

import networkx as nx
import numpy as np
import pytest

import photographiq as pg


@pytest.mark.parametrize(
    "a", [np.ones((2, 3)), [[0, 1], [0, 0]], [[1]], [[0, np.nan], [np.nan, 0]]]
)
def test_invalid_adjacency(a):
    with pytest.raises(ValueError):
        pg.CVGraph.from_adjacency(a)


def test_graph_labels_and_families():
    for graph in [
        pg.CVGraph.line(4),
        pg.CVGraph.ring(4),
        pg.CVGraph.star(3),
        pg.CVGraph.tree(2, 2),
        pg.CVGraph.rectangular(2, 3),
        pg.CVGraph.dual_rail(3),
    ]:
        graph.validate()
    with pytest.raises(ValueError):
        pg.CVGraph(nx.DiGraph())
    with pytest.raises(ValueError):
        pg.CVGraph.line(2, inputs=(4,))
    graph = pg.CVGraph.from_adjacency([[0, -0.4], [-0.4, 0]], labels=["in", (1, 2)])
    assert graph.nodes == ("in", (1, 2))
    with pytest.raises(ValueError):
        graph.add_edge("missing", "in")
    with pytest.raises(ValueError):
        graph.add_edge("in", "in")


def test_future_and_cycle_and_missing_modes():
    pattern = pg.Pattern(pg.CVGraph.line(2))
    pattern.measure(0, pg.Homodyne(pg.Outcome(1))).measure(1)
    with pytest.raises(ValueError, match="future"):
        pattern.validate()
    pattern = pg.Pattern().extend(
        [pg.Signal("a", pg.Outcome("b")), pg.Signal("b", pg.Outcome("a"))]
    )
    with pytest.raises(ValueError, match="Cyclic"):
        pattern.validate()
    with pytest.raises(ValueError):
        pg.Pattern().measure(9).validate()


def test_parameters_and_callable_dependency():
    p = pg.protocols.wire([pg.Parameter("k")], squeezing=pg.Parameter("r"))
    with pytest.raises(ValueError, match="Unbound"):
        pg.simulate(p)
    pg.simulate(p, parameters={"k": 0.2, "r": 0.5})
    expression = pg.CallableExpression(lambda r: r[2], frozenset({1}))
    with pytest.raises(KeyError):
        expression.evaluate({1: 1, 2: 2}, {})


def test_json_roundtrip_and_security():
    graph = pg.CVGraph.rectangular(2, 2, squeezing=pg.Parameter("r"))
    p = pg.Pattern(graph).measure((0, 0), pg.Homodyne(pg.Parameter("theta")))
    q = pg.Pattern.from_json(p.to_json())
    assert q.inputs == p.inputs and q.outputs == p.outputs
    a = pg.simulate(p, parameters={"r": 0.3, "theta": 0.7}, seed=8)
    b = pg.simulate(q, parameters={"r": 0.3, "theta": 0.7}, seed=8)
    np.testing.assert_allclose(a.state.covariance, b.state.covariance)
    assert a.outcomes == b.outcomes
    data = json.loads(p.to_json())
    data["commands"][0]["type"] = "__import__"
    with pytest.raises(ValueError):
        pg.Pattern.from_json(json.dumps(data))
    p = (
        pg.Pattern(pg.CVGraph.line(2))
        .measure(0)
        .measure(1, pg.Homodyne(pg.CallableExpression(lambda r: r[0], {0})))
    )
    with pytest.raises(TypeError):
        p.to_json()


def test_invalid_state_and_requests():
    with pytest.raises(ValueError):
        pg.GaussianState([0, 0], 0.5 * np.eye(2), (0,))
    with pytest.raises(ValueError):
        pg.GaussianState([0, 0], [[1, 0.2], [0, 1]], (0,))
    with pytest.raises(ValueError):
        pg.simulate(pg.Pattern(), backend="missing")
    with pytest.raises(ValueError):
        pg.run_shots(pg.Pattern(), 0)
    with pytest.raises(ValueError):
        pg.Pattern(pg.CVGraph.line(2, ideal=True))
    with pytest.raises(NotImplementedError):
        pg.simulate(pg.Pattern(pg.CVGraph.line(1)).measure(0, pg.PhotonNumber()))
