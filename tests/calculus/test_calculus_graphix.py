"""Cross-check the qubit calculus against Graphix 0.4 where the semantics coincide.

Graphix angles are in units of pi and its planes are XY: cos X + sin Y, YZ: cos Z + sin Y,
XZ: cos Z + sin X, i.e. the same as photographiq.calculus.qubit. Graphix orders corrections
Z before X in standard form, so standard forms are compared per output as XOR-ed domains and
as linear maps per branch (up to the branch phase).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from calculus_random import random_pattern

from photographiq.calculus import qubit as q

gx = pytest.importorskip("graphix", minversion="0.4")
from graphix import command as gc  # noqa: E402
from graphix.branch_selector import FixedBranchSelector  # noqa: E402
from graphix.fundamentals import Plane as GPlane  # noqa: E402
from graphix.measurements import Measurement  # noqa: E402
from graphix.pattern import Pattern as GPattern  # noqa: E402
from graphix.sim.statevec import Statevector  # noqa: E402

_PLANES = {q.Plane.XY: Measurement.XY, q.Plane.YZ: Measurement.YZ, q.Plane.XZ: Measurement.XZ}


def to_graphix(pattern: q.QubitPattern) -> GPattern:
    g = GPattern(input_nodes=list(pattern.inputs))
    for c in pattern.commands:
        if isinstance(c, q.N):
            g.add(gc.N(c.node))
        elif isinstance(c, q.E):
            g.add(gc.E((c.u, c.v)))
        elif isinstance(c, q.M):
            m = _PLANES[c.plane](c.angle / math.pi)
            g.add(gc.M(c.node, m, set(c.s_domain), set(c.t_domain)))
        elif isinstance(c, q.X):
            g.add(gc.X(c.node, set(c.domain)))
        elif isinstance(c, q.Z):
            g.add(gc.Z(c.node, set(c.domain)))
        else:
            raise TypeError(c)
    g.reorder_output_nodes(list(pattern.outputs))
    return g


def summary(commands, outputs):
    """Semantic standard-form summary: N set, CZ parity set, measurements, output frames."""
    preps, edges, meas = set(), set(), {}
    frames = {o: [frozenset(), frozenset()] for o in outputs}
    for c in commands:
        name = type(c).__name__
        if name == "N":
            preps.add(c.node)
        elif name == "E":
            u, v = c.nodes
            edges ^= {frozenset((u, v))}
        elif name == "M":
            if isinstance(c, q.M):
                plane, angle = c.plane.value, c.angle
            else:
                bloch = c.measurement.to_bloch()
                plane, angle = bloch.plane.name, float(bloch.angle) * math.pi
            meas[c.node] = (
                plane,
                angle % (2 * math.pi),
                frozenset(c.s_domain),
                frozenset(c.t_domain),
            )
        elif name in ("X", "Z"):
            frames[c.node][name == "Z"] ^= frozenset(c.domain)
    return preps, edges, meas, {o: tuple(f) for o, f in frames.items()}


def assert_same_summary(a, b):
    pa, ea, ma, fa = a
    pb, eb, mb, fb = b
    assert pa == pb and ea == eb and fa == fb
    assert ma.keys() == mb.keys()
    for node in ma:
        assert ma[node][0] == mb[node][0] and ma[node][2:] == mb[node][2:]
        diff = (ma[node][1] - mb[node][1] + math.pi) % (2 * math.pi) - math.pi
        assert abs(diff) < 1e-12


def test_branch_states_match_graphix_statevector():
    rng = np.random.default_rng(99)
    checked = 0
    for _ in range(25):
        pattern = random_pattern(rng)
        g = to_graphix(pattern)
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        psi /= np.linalg.norm(psi)
        for branch in q.enumerate_branches(pattern)[:: max(1, 2 ** len(pattern.measured) // 4)]:
            ours = branch.operator @ psi
            if np.linalg.norm(ours) < 1e-6:
                continue
            ours /= np.linalg.norm(ours)
            selector = FixedBranchSelector({n: bool(r) for n, r in branch.outcomes.items()})
            theirs = g.simulate(input_state=Statevector(data=psi), branch_selector=selector)
            theirs = np.asarray(theirs.flatten())
            assert abs(abs(np.vdot(theirs, ours)) - 1) < 1e-10
            checked += 1
    assert checked >= 50


def test_standardization_matches_graphix():
    rng = np.random.default_rng(4)
    for _ in range(40):
        pattern = random_pattern(rng)
        ours = q.standardize(pattern)
        g = to_graphix(pattern)
        g.standardize()
        assert g.is_standard()
        assert_same_summary(summary(ours.commands, ours.outputs), summary(list(g), ours.outputs))


def test_signal_shifting_matches_graphix():
    rng = np.random.default_rng(8)
    for _ in range(40):
        pattern = q.standardize(random_pattern(rng))
        ours = q.signal_shift(pattern)
        g = to_graphix(pattern)
        shifted = g.shift_signals(method="direct")
        quantum = [c for c in ours.commands if not isinstance(c, q.S)]
        assert_same_summary(summary(quantum, ours.outputs), summary(list(g), ours.outputs))
        ours_signals = {
            c.node: frozenset(c.domain) | {c.node} for c in ours.commands if isinstance(c, q.S)
        }
        assert ours_signals == {n: frozenset(d) | {n} for n, d in shifted.items() if d}


def test_graphix_plane_convention_is_ours():
    for plane, gplane in (
        (q.Plane.XY, GPlane.XY),
        (q.Plane.YZ, GPlane.YZ),
        (q.Plane.XZ, GPlane.XZ),
    ):
        for a in (0.3, 1.1):
            vec = gplane.polar(a / math.pi)
            sigma = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
            theirs = sum(v * s for v, s in zip(vec, sigma))
            assert np.allclose(theirs, q.observable(plane, a), atol=1e-12)
