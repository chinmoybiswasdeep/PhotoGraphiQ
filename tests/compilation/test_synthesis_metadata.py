import pytest

import photographiq as pg


def test_synthesis_reports_approximation_order():
    _, exact = pg.synthesis.quadrature_polynomial([(0.02, 0.0, 3)], steps=2)
    assert exact.approximate is False
    assert exact.order is None
    _, quartic = pg.synthesis.quadrature_polynomial([(0.002, 0.0, 4)], steps=8)
    assert quartic.approximate is True
    assert quartic.order == 0.5
    assert quartic.steps == 8


def test_compilation_preserves_report_and_warns():
    with pytest.warns(UserWarning, match="Kerr.*approximate"):
        _, trace = pg.Circuit(1).kerr(0, 0.002).compile(return_trace=True, synthesis_steps=2)
    report = trace.steps[0].synthesis_report
    assert report.target_gate == "Kerr"
    assert report.target_parameter == 0.002
    assert report.steps == 2
    assert report.order == 0.5
    assert report.approximate


def test_cubic_trace_exposes_finite_resource_approximation():
    _, trace = pg.Circuit(1).cubic_phase(0, 0.02).compile(return_trace=True)
    assert trace.steps[0].synthesis_report.approximate
    assert "resource" in trace.steps[0].synthesis_report.method
