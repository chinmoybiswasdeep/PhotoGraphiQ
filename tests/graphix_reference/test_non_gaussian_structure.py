"""Only resource connectivity, measurement order and correction domains are compared."""

import networkx as nx
import pytest

graphix = pytest.importorskip("graphix")
from graphix import command  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.commands import command_dependencies  # noqa: E402


def test_cubic_injection_has_corresponding_causal_skeleton():
    cv = pg.non_gaussian.cubic_injection(0.3, 0.2, input_node=0, ancilla=1, key=1)
    dv = graphix.Pattern(input_nodes=[0])
    dv.add(command.N(1))
    dv.add(command.E((0, 1)))
    dv.add(command.M(1))
    dv.add(command.X(0, domain={1}))
    dv.add(command.Z(0, domain={1}))
    assert nx.is_isomorphic(cv.graph.network, dv.to_opengraph().graph)
    assert cv.inputs == tuple(dv.input_nodes)
    assert cv.outputs == tuple(dv.output_nodes)
    corrections = [c for c in cv.commands if isinstance(c, (pg.QuadraticPhase, pg.Displace))]
    assert all(command_dependencies(c) == {1} for c in corrections)
    assert [c.node for c in cv.commands if isinstance(c, pg.Measure)] == [1]
    # No Graphix amplitudes, probabilities, angles or correction matrices are compared.
