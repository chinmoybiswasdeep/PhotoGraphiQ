"""Structural comparisons only. Qubit states are never a CV numerical oracle."""

import networkx as nx
import pytest

graphix = pytest.importorskip("graphix")
from graphix import command  # noqa: E402

import photographiq as pg  # noqa: E402


def test_wire_connectivity_io_measurement_order_and_domains():
    dv = graphix.Pattern(input_nodes=[0])
    dv.add(command.N(1))
    dv.add(command.N(2))
    dv.add(command.E((0, 1)))
    dv.add(command.E((1, 2)))
    dv.add(command.M(0))
    dv.add(command.M(1, s_domain={0}))
    dv.add(command.X(2, domain={1}))
    dv.add(command.Z(2, domain={0}))
    graph = pg.CVGraph.line(3, inputs=(0,))
    cv = pg.Pattern(graph).measure(0, pg.Homodyne.p())
    cv.measure(1, pg.Homodyne(0.2 + pg.Outcome(0)))
    cv.displace(2, q=pg.Outcome(1), p=pg.Outcome(0))
    og = dv.to_opengraph()
    assert nx.is_isomorphic(og.graph, cv.graph.network)
    assert tuple(dv.input_nodes) == cv.inputs
    assert tuple(dv.output_nodes) == cv.outputs
    assert [c.node for c in dv if isinstance(c, command.M)] == [
        c.node for c in cv.commands if isinstance(c, pg.Measure)
    ]
    cv_measurements = [(i, c) for i, c in enumerate(cv.commands) if isinstance(c, pg.Measure)]
    assert cv.dependencies().has_edge(cv_measurements[0][0], cv_measurements[1][0])
    assert next(c for c in dv if isinstance(c, command.M) and c.node == 1).s_domain == {0}


def test_safe_preparation_standardization_matches_graphix():
    dv = graphix.Pattern()
    dv.add(command.N(0))
    dv.add(command.N(1))
    dv.add(command.E((0, 1)))
    dv.add(command.N(2))
    dv.standardize()
    cv = (
        pg.Pattern()
        .extend([pg.Prepare(0), pg.Prepare(1), pg.Entangle(0, 1), pg.Prepare(2)])
        .standardize()
    )
    assert dv.is_standard()
    assert [type(c).__name__ for c in cv.commands] == ["Prepare", "Prepare", "Prepare", "Entangle"]
    assert [type(c).__name__ for c in dv] == ["N", "N", "N", "E"]
