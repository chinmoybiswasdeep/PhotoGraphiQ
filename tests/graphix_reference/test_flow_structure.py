"""Common causal-flow correction support, without equating real and binary signals."""

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("graphix")
from graphix.fundamentals import Plane  # noqa: E402
from graphix.opengraph import OpenGraph  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.cvflow import certify_cv_flow  # noqa: E402


def test_line_correction_support_agrees():
    dv = OpenGraph(nx.path_graph(3), [0], [2], {0: Plane.XY, 1: Plane.XY})
    dv_flow = dv.to_causalflow()
    corrections = dv_flow.to_xzcorrections()
    cv_flow = certify_cv_flow(pg.CVGraph.line(3, inputs=(0,), outputs=(2,)), [0, 1])
    assert cv_flow is not None
    for node in (0, 1):
        support = {n for n, value in cv_flow.corrections[node].items() if not np.isclose(value, 0)}
        assert support == set(corrections.x_corrections[node])
