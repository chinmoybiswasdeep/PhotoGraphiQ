import numpy as np

import photographiq as pg
from photographiq.cvflow import certify_cv_flow, flow_pattern


def test_weighted_real_flow_certificate():
    graph = pg.CVGraph.from_adjacency([[0, 0.7], [0.7, 0]], inputs=(0,), outputs=(1,))
    flow = certify_cv_flow(graph, [0])
    assert flow is not None
    assert np.isclose(flow.corrections[0][1], 1 / 0.7)
    channel = pg.gaussian_channel(flow_pattern(graph, [0]))
    np.testing.assert_allclose(channel.matrix, [[0, -1 / 0.7], [0.7, 0]], atol=1e-12)


def test_flow_pattern_and_streaming_wire_have_same_channel():
    graph = pg.CVGraph.line(4, squeezing=0.6, inputs=(0,), outputs=(3,))
    a = pg.gaussian_channel(flow_pattern(graph, [0, 1, 2], shears={0: 0.2, 1: -0.4, 2: 0.7}))
    b = pg.gaussian_channel(pg.protocols.wire([0.2, -0.4, 0.7], squeezing=0.6))
    np.testing.assert_allclose(a.matrix, b.matrix, atol=1e-12)
    np.testing.assert_allclose(a.noise, b.noise, atol=1e-12)
    assert certify_cv_flow(graph, [2, 1, 0]) is None


def test_no_flow_for_isolated_measured_node():
    graph = pg.CVGraph.from_adjacency(np.zeros((2, 2)), inputs=(0,), outputs=(1,))
    assert certify_cv_flow(graph, [0]) is None
