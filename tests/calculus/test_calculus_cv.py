"""CV command-level view and ideal-limit identity checks of Gaussian patterns."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from photographiq.calculus.cv import (
    AffineForm,
    CommandRole,
    affine_form,
    command_view,
    ideal_limit_check,
    wire_step_pattern,
)
from photographiq.commands import Displace, Prepare
from photographiq.compiler import Circuit, compile_circuit
from photographiq.cvflow import flow_pattern
from photographiq.expressions import CallableExpression, Outcome, Parameter, resolve
from photographiq.gaussian import teleportation_matrix, wire_channel
from photographiq.graph import CVGraph
from photographiq.measurements import Heterodyne, Homodyne, PhotonNumber
from photographiq.pattern import Pattern

SQUEEZINGS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


@pytest.mark.parametrize("k", [0.0, 0.7, -2.5])
def test_wire_step_ideal_map_is_exact_and_noise_is_analytic(k):
    report = ideal_limit_check(
        lambda r: wire_step_pattern(k, r),
        teleportation_matrix(k),
        SQUEEZINGS,
        expected_noise=lambda r: np.diag([0.0, math.exp(-2 * r)]),
    )
    # T(k) is reproduced at every finite r (not only in the limit); r only adds noise.
    assert report.max_matrix_error < 1e-14
    assert all(p.noise_error < 1e-15 for p in report.points)
    assert report.noise_decreasing
    assert report.points[-1].noise_norm == pytest.approx(math.exp(-16), rel=1e-12)
    assert all(np.allclose(p.displacement, 0) for p in report.points)


def test_compiled_identity_matches_closed_form_wire_channel():
    def build(r):
        return compile_circuit(Circuit(1).identity(0), squeezing=r)

    report = ideal_limit_check(
        build, np.eye(2), SQUEEZINGS, expected_noise=lambda r: wire_channel([0.0] * 4, r)[1]
    )
    assert report.max_matrix_error < 1e-14
    assert all(p.noise_error < 1e-14 for p in report.points)
    assert report.noise_decreasing and report.points[0].noise_norm > 0.5


def test_ideal_limit_check_rejects_shape_mismatch_and_adaptive_patterns():
    with pytest.raises(ValueError, match="target"):
        ideal_limit_check(lambda r: wire_step_pattern(0, r), np.eye(4), [1.0])
    adaptive = wire_step_pattern(0.0, 1.0)
    adaptive.measure(1, Homodyne(Outcome(0)), key="m1")
    with pytest.raises(NotImplementedError):
        ideal_limit_check(lambda r: adaptive, np.eye(2), [1.0])


def test_command_view_roles_outcomes_and_corrections():
    view = command_view(wire_step_pattern(0.3, 1.0))
    roles = [c.role for c in view.commands]
    assert roles == [
        CommandRole.QUANTUM,
        CommandRole.QUANTUM,
        CommandRole.MEASUREMENT,
        CommandRole.FEEDFORWARD,
    ]
    (outcome,) = view.outcomes
    assert outcome.continuous and outcome.components == 1
    assert outcome.angle == pytest.approx(math.atan2(1, 0.3))
    (correction,) = view.corrections
    assert correction.affine and correction.node == 1
    assert correction.q.coefficients == pytest.approx({(0, None): -math.hypot(1, 0.3)})
    assert correction.p.coefficients == {} and correction.p.constant == 0


def test_cv_flow_corrections_are_exact_affine_forms():
    graph = CVGraph(nx.path_graph(4), squeezing=1.0, inputs=(0,), outputs=(3,))
    pattern = flow_pattern(graph, (0, 1, 2), shears={0: 0.4, 1: -1.0, 2: 0.0})
    view = command_view(pattern)
    rng = np.random.default_rng(0)
    assert len(view.corrections) >= 3 and all(c.affine for c in view.corrections)
    for c in view.corrections:
        records = {o.key: float(rng.normal()) for o in view.outcomes}
        cmd = pattern.commands[c.index]
        assert isinstance(cmd, Displace)
        assert c.q.evaluate(records) == pytest.approx(resolve(cmd.q, records, {}), abs=1e-12)
        assert c.p.evaluate(records) == pytest.approx(resolve(cmd.p, records, {}), abs=1e-12)


def test_nonaffine_and_unbound_expressions_are_kept_verbatim():
    m = Outcome("m")
    assert affine_form(m * m) is None
    assert affine_form(Parameter("g") * m) is None
    assert affine_form(Parameter("g") * m, {"g": 2.0}) == AffineForm({("m", None): 2.0}, 0.0)
    assert affine_form(CallableExpression(lambda r: r["m"], {"m"})) is None
    assert affine_form(Outcome("h", 1) / 2 + 3) == AffineForm({("h", 1): 0.5}, 3.0)
    pattern = Pattern(inputs=(0,))
    pattern.extend([Prepare(1, 1.0)])
    pattern.measure(0, Heterodyne(), key="h")
    pattern.displace(1, q=Outcome("h", 0) ** 2, p=Outcome("h", 1))
    view = command_view(pattern)
    (corr,) = view.corrections
    assert not corr.affine and corr.q is pattern.commands[-1].q
    assert corr.p == AffineForm({("h", 1): 1.0}, 0.0)
    assert view.outcomes[0].components == 2 and view.outcomes[0].angle is None


def test_photon_counting_outcomes_are_discrete():
    pattern = Pattern(inputs=(0,))
    pattern.measure(0, PhotonNumber())
    (outcome,) = command_view(pattern).outcomes
    assert not outcome.continuous and outcome.measurement == "PhotonNumber"
