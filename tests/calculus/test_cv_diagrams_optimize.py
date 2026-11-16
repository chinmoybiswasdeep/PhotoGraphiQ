"""CV-MBQC optimisation, resource costs, flow checks, non-Gaussian and GKP diagram semantics."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import photographiq as pg
from photographiq.analysis import gaussian_channel
from photographiq.calculus.cv_diagrams import (
    BeamSplitter,
    CVDiagram,
    Displacement,
    FockPrepare,
    GKPPrepare,
    Homodyne,
    Kerr,
    ModularDecode,
    PhasePolynomial,
    PhotonCount,
    Postselect,
    Prepare,
    apply_rewrite,
    compare,
    cutoff_study,
    from_pattern,
    optimize_cv_pattern,
    pareto_front,
    pattern_flow_report,
    resource_cost,
    run_fock,
    teleport_steps,
    to_pattern,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from cv_fock_reference import FockReference  # noqa: E402

pytest.importorskip("photographiq._native")


def circuit():
    return (
        pg.Circuit(2)
        .rotate(0, 0.3)
        .rotate(0, 0.4)
        .squeeze(1, 0.2)
        .cz(0, 1, 0.5)
        .rotate(1, 0.2)
        .displace(0, 0.3, 0.1)
    )


# ----------------------------------------------------------------------------- optimisation
def test_physical_mode_never_changes_the_channel_and_logical_mode_shortens_wires():
    pattern = circuit().compile(squeezing=1.0)
    physical = optimize_cv_pattern(pattern, equivalence="physical")
    assert physical.report.physical_equivalent and not physical.improved  # nothing exact applies
    logical = optimize_cv_pattern(pattern, equivalence="logical")
    assert logical.improved and logical.report.logical_equivalent
    assert not logical.report.physical_equivalent  # finite-squeezing noise changed, reported
    assert logical.best_cost["resource_modes"] < logical.original_cost["resource_modes"]
    executed = gaussian_channel(to_pattern(logical.best))
    np.testing.assert_allclose(executed.matrix, gaussian_channel(pattern).matrix, atol=1e-12)
    again = optimize_cv_pattern(pattern, equivalence="logical")
    assert again.best.fingerprint() == logical.best.fingerprint()  # deterministic


def test_approximate_mode_respects_its_noise_budget():
    pattern = circuit().compile(squeezing=1.0)
    logical = optimize_cv_pattern(pattern, equivalence="logical")
    tight = optimize_cv_pattern(pattern, equivalence="approximate", epsilon=1e-6)
    loose = optimize_cv_pattern(pattern, equivalence="approximate", epsilon=10.0)
    assert tight.report.channel_Y_residual <= 1e-6
    assert loose.best.fingerprint() == logical.best.fingerprint()


def test_flow_is_certified_before_and_after_optimisation():
    pattern = pg.Circuit(1).rotate(0, 0.3).rotate(0, 0.5).squeeze(0, 0.2).compile(squeezing=1.0)
    result = optimize_cv_pattern(pattern, equivalence="logical")
    for d in (result.original, result.best):
        report = pattern_flow_report(d)
        assert report["status"] == "certified", report
    unsupported = pattern_flow_report(
        CVDiagram(
            (0,),
            (BeamSplitter(0, "a", 0.3),)
            if False
            else (Prepare.squeezed("a", 0.3), BeamSplitter(0, "a", 0.3)),
            (0, "a"),
        )
    )
    assert unsupported["status"] == "unsupported"


def test_identity_wires_are_removed_and_costs_are_counted():
    pattern = pg.Circuit(1).identity(0).identity(0).compile(squeezing=0.5)
    diagram = from_pattern(pattern)
    assert len(teleport_steps(diagram)) == 8
    cost = resource_cost(diagram)
    assert cost["resource_modes"] == cost["measurement_count"] == 8
    assert cost["squeezing_energy"] == pytest.approx(8 * np.sinh(0.5) ** 2)
    best = optimize_cv_pattern(pattern, equivalence="logical").best
    assert resource_cost(best)["resource_modes"] == 0 and not best.boxes
    assert pareto_front([{"a": 1, "b": 2}, {"a": 2, "b": 1}, {"a": 2, "b": 2}], ("a", "b")) == [
        0,
        1,
    ]


def test_review_fixes_gaussian_phase_polynomial_sources_and_strategy():
    from photographiq.calculus.cv_diagrams import ClassicalSource, Feedforward, evaluate

    d = CVDiagram((0,), (PhasePolynomial(0, (0.0, 0.3, 0.05)),), (0,))
    S, t = evaluate(d).unitary()
    np.testing.assert_allclose(S, [[1, 0], [0.2, 1]])  # shear 4 c2
    np.testing.assert_allclose(t, [0, 0.6])  # p displacement 2 c1
    with pytest.raises(NotImplementedError):
        to_pattern(CVDiagram((), (ClassicalSource("k", 1.0, 0.0),), ()))
    two = CVDiagram(
        (0,),
        (
            ClassicalSource("k1", 0.0, 1.0),
            ClassicalSource("k2", 0.0, 1.0),
            Feedforward(0, (("k1", (1.0, 0.0)),)),
        ),
        (0,),
    )
    run = run_fock(two, 6, seed=3)
    assert run.records["k1"] != run.records["k2"]  # independent classical sources
    with pytest.raises(ValueError):
        optimize_cv_pattern(circuit().compile(squeezing=1.0), strategy="annealing")


# ----------------------------------------------------------------------------- non-Gaussian
def test_phase_polynomial_rules_converge_in_fock_to_the_matrix_exponential_reference():
    d = CVDiagram(
        (0,),
        (
            Displacement(0, 0.4, 0.0),
            PhasePolynomial(0, (0.0, 0.1, 0.05, 0.03)),
            PhasePolynomial(0, (0.0, 0.0, -0.02, 0.01)),
        ),
        (0,),
    )
    fused = apply_rewrite(d, "phase-polynomial-fusion")
    moved = apply_rewrite(fused, "displacement-through-phase-polynomial")
    assert fused.boxes[-1].coefficients == pytest.approx((0.0, 0.1, 0.03, 0.04))
    ref = FockReference(140)
    psi = ref.displace(0.5, 0.2) @ ref.vacuum()
    from scipy.linalg import expm

    total = expm(
        1j * (0.1 * ref.q + 0.03 * ref.q @ ref.q + 0.04 * np.linalg.matrix_power(ref.q, 3))
    ) @ ref.displace(0.4, 0.0)
    target = total @ psi
    errors = []
    for cutoff in (30, 45, 60):
        for diagram in (d, fused, moved):
            state = run_fock(diagram, cutoff, inputs={0: pg.GaussianInput((0.5, 0.2))}).state
            vec = np.asarray(state.state_vector)
            n = len(vec)
            errors.append((cutoff, 1 - abs(np.vdot(target[:n], vec)) ** 2))
    worst = {c: max(e for cc, e in errors if cc == c) for c in (30, 45, 60)}
    assert worst[30] > worst[45] > worst[60] and worst[60] < 1e-9  # convergence, not proof


def test_heralded_photon_subtraction_probability_matches_poisson():
    alpha, theta = 0.8, 0.3
    d = CVDiagram(
        (0,),
        (
            FockPrepare("a", 0),
            BeamSplitter(0, "a", theta),
            PhotonCount("a", "n"),
            Postselect("n", 1),
        ),
        (0,),
    )
    rows = cutoff_study(
        d,
        (12, 20),
        lambda run: run.probability,
        inputs={0: pg.GaussianInput.coherent(alpha)},
        outcomes={"n": 1},
    )
    mu = (alpha * np.sin(theta)) ** 2
    assert rows[-1]["value"] == pytest.approx(mu * np.exp(-mu), rel=1e-10)
    with pytest.raises(ValueError, match="Postselect"):
        run_fock(d, 12, inputs={0: pg.GaussianInput.coherent(alpha)})
    with pytest.raises(NotImplementedError):
        compare(d, d)  # non-Gaussian diagrams have no Gaussian semantics


def test_kerr_diagram_matches_reference_and_rejects_gaussian_rewrites():
    d = CVDiagram((0,), (Kerr(0, 0.2), PhasePolynomial(0, (0, 0, 0, 0.02))), (0,))
    ref = FockReference(80)
    psi = ref.cubic(0.12) @ ref.kerr(0.2) @ ref.displace(0.7, 0.0) @ ref.vacuum()
    q_ref = ref.moments(psi)[0]
    state = run_fock(d, 50, inputs={0: pg.GaussianInput((0.7, 0.0))}).state
    from photographiq.fock_analysis import quadrature_moment

    assert quadrature_moment(state, 0, 1) == pytest.approx(q_ref, abs=1e-8)
    assert not d.is_gaussian


def test_gkp_z_readout_diagram_matches_existing_protocol_and_grid_reference():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
    from gkp_xy_reference import GridModel

    code = pg.GKPCode(cutoff=48, peak_width=0.6, envelope=0.6, peaks=6, grid_points=2049)
    logical = (np.cos(0.4), np.sin(0.4))
    d = CVDiagram(
        (),
        (GKPPrepare("g", code, *logical), Homodyne("g", 0.0, "m"), ModularDecode("m", "bit")),
        (),
    )
    spacing = np.sqrt(2 * np.pi)  # cell-aligned midpoints: each sample strictly inside a cell
    dx = spacing / 40
    xs = -6.5 * spacing + (np.arange(13 * 40) + 0.5) * dx
    p0 = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x in xs:
            run = run_fock(d, 48, outcomes={"m": float(x)})
            if run.records["bit"] == 0:
                p0 += run.density * dx
    state = run_fock(CVDiagram((), (GKPPrepare("g", code, *logical),), ("g",)), 48).state
    existing = pg.multimode_readout(state, {"g": "Z"})["joint_probabilities"][(0,)]
    grid = GridModel(0.6, 0.6, 6)
    reference = grid.z_readout(grid.operator(grid.encode(*logical)))[0].real
    assert p0 == pytest.approx(existing, abs=2e-3)
    assert p0 == pytest.approx(reference, abs=2e-3)
