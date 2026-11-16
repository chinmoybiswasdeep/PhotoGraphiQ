"""Typed CV diagrams: typing, Gaussian semantics, rewrites, translations, GaussianGraph interop.

References are independent of the module under test wherever possible: photographiq.gaussian
matrices, ``analysis.gaussian_channel`` (source-vector propagation of a Pattern), trajectory
execution with postselection on the Gaussian backend (sequential Schur complements), the
MFvL ``GaussianGraph`` rules, and hand-derived formulas.
"""

import json

import numpy as np
import pytest

import photographiq as pg
from photographiq import gaussian as gm
from photographiq.analysis import gaussian_channel
from photographiq.calculus import GaussianGraph
from photographiq.calculus.cv_diagrams import (
    CZ,
    RULES,
    BeamSplitter,
    ClassicalSource,
    CVDiagram,
    DiagramTypeError,
    Displacement,
    Feedforward,
    GaussianChannelBox,
    GaussianLoss,
    Homodyne,
    Prepare,
    Rotation,
    Shear,
    Squeezer,
    Symplectic,
    apply_rewrite,
    compare,
    compose_channels,
    conditional_semantics,
    draw,
    evaluate,
    from_circuit,
    from_gaussian_graph,
    from_pattern,
    is_completely_positive,
    to_gaussian_graph,
    to_pattern,
    trace_json,
    unitary_semantics,
)
from photographiq.calculus.cv_diagrams.rules import GAUSSIAN_RULES, Commute


def random_gaussian_diagram(rng, modes=2, depth=6):
    boxes = []
    for _ in range(depth):
        kind = rng.integers(6)
        m = int(rng.integers(modes))
        o = (m + 1) % modes
        boxes.append(
            [
                Rotation(m, rng.normal()),
                Squeezer(m, 0.4 * rng.normal()),
                Shear(m, rng.normal()),
                Displacement(m, rng.normal(), rng.normal()),
                BeamSplitter(m, o, rng.normal()),
                CZ(m, o, rng.normal()),
            ][kind]
        )
    return CVDiagram(tuple(range(modes)), tuple(boxes), tuple(range(modes)))


# ----------------------------------------------------------------------------- typing
def test_typing_rejects_malformed_diagrams():
    with pytest.raises(DiagramTypeError, match="reads record"):
        CVDiagram((0,), (Feedforward(0, (("m", (1.0, 0.0)),)),), (0,))
    with pytest.raises(DiagramTypeError, match="not alive"):
        CVDiagram((0,), (Homodyne(0, 0.3, "m"), Rotation(0, 0.1)), ())
    with pytest.raises(DiagramTypeError, match="classical record"):
        CVDiagram((0,), (Homodyne(0, 0.3, "m"), Rotation("m", 0.1)), ())
    with pytest.raises(DiagramTypeError, match="Outputs"):
        CVDiagram((0,), (Homodyne(0, 0.3, "m"),), (0,))
    with pytest.raises(DiagramTypeError, match="recreates"):
        CVDiagram((0,), (Prepare.vacuum(0),), (0,))
    with pytest.raises(DiagramTypeError, match="uncertainty"):
        Prepare(0, covariance=((0.1, 0.0), (0.0, 0.1)))
    with pytest.raises(DiagramTypeError, match="completely positive"):
        GaussianChannelBox((0,), 2 * np.eye(2), np.zeros((2, 2)))
    with pytest.raises(DiagramTypeError, match="Symplectic"):
        Symplectic((0,), [[2.0, 0.0], [0.0, 1.0]])


def test_rewrites_never_mutate_and_record_a_trace():
    d = CVDiagram((0,), (Rotation(0, 0.2), Squeezer(0, 0.1), Rotation(0, 0.3)), (0,))
    before = d.to_dict()
    with pytest.raises(ValueError):
        apply_rewrite(d, "rotation-fusion")  # not wire-adjacent: a squeezer sits between
    d2 = CVDiagram((0,), (Rotation(0, 0.2), Rotation(0, 0.3)), (0,))
    out = apply_rewrite(d2, "rotation-fusion")
    assert d.to_dict() == before and len(d2.boxes) == 2 and len(out.boxes) == 1
    step = out.trace[-1]
    assert step.rule == "rotation-fusion" and step.equivalence == "exact-instrument"
    json.loads(trace_json(out))


# ----------------------------------------------------------------------------- semantics
def test_generator_semantics_match_photographiq_matrices():
    S, d = unitary_semantics(
        CVDiagram(
            (0, 1),
            (
                Rotation(0, 0.4),
                Squeezer(1, 0.3),
                Shear(0, 0.2),
                BeamSplitter(0, 1, 0.7),
                CZ(1, 0, -0.5),
                Displacement(1, 0.2, -0.1),
            ),
            (0, 1),
        )
    )
    expected = (
        gm.embed(gm.cz(-0.5), [2, 3, 0, 1], 4)
        @ gm.beamsplitter(0.7)
        @ gm.embed(np.array([[1, 0], [0.2, 1]]), [0, 1], 4)
        @ gm.embed(gm.squeezing(0.3), [2, 3], 4)
        @ gm.embed(gm.rotation(0.4), [0, 1], 4)
    )
    np.testing.assert_allclose(S, expected, atol=1e-14)
    np.testing.assert_allclose(d, [0, 0, 0.2, -0.1], atol=1e-15)
    np.testing.assert_allclose(S @ gm.omega(2) @ S.T, gm.omega(2), atol=1e-12)


def test_composition_identity_associativity_tensor_and_permutation():
    rng = np.random.default_rng(1)
    a, b, c = (random_gaussian_diagram(rng) for _ in range(3))
    sem = lambda d: evaluate(d).channel()  # noqa: E731
    for x, y in zip(sem(a.then(b).then(c)), sem(a.then(b.then(c)))):
        np.testing.assert_allclose(x, y, atol=1e-11)
    for x, y in zip(sem(a.then(b)), compose_channels(sem(a), sem(b))):
        np.testing.assert_allclose(x, y, atol=1e-11)
    ident = CVDiagram((0, 1), (), (0, 1))
    for x, y in zip(sem(ident.then(a)), sem(a)):
        np.testing.assert_allclose(x, y, atol=1e-14)
    relabel = CVDiagram((2,), (Squeezer(2, 0.3),), (2,))
    X, _, _ = sem(a.tensor(relabel))
    np.testing.assert_allclose(X[:4, :4], sem(a)[0], atol=1e-14)
    np.testing.assert_allclose(X[4:, 4:], gm.squeezing(0.3), atol=1e-14)
    assert not X[:4, 4:].any()
    swapped = a.permute_outputs((1, 0))
    np.testing.assert_allclose(sem(swapped)[0], sem(a)[0][[2, 3, 0, 1]], atol=1e-14)


@pytest.mark.parametrize("efficiency,noise", [(1.0, 0.0), (0.8, 0.1)])
def test_channel_semantics_match_gaussian_channel_of_patterns(efficiency, noise):
    pattern = (
        pg.Circuit(2)
        .rotate(0, 0.3)
        .squeeze(1, 0.2)
        .beamsplitter(0, 1, 0.7)
        .cz(0, 1, 0.5)
        .displace(1, 0.3, -0.2)
        .compile(squeezing=0.7)
    )
    pattern = pattern.copy()
    pattern.commands.insert(len(pattern.commands) - 1, pg.Loss(pattern.outputs[0], 0.9, 0.05))
    if efficiency != 1.0:
        for k, cmd in enumerate(pattern.commands):
            if isinstance(cmd, pg.Measure):
                pattern.commands[k] = pg.Measure(
                    cmd.node, pg.Homodyne(cmd.measurement.angle, efficiency, noise), cmd.key
                )
                break
    diagram = from_pattern(pattern.validate())
    X, Y, d = evaluate(diagram).channel()
    ref = gaussian_channel(pattern)
    np.testing.assert_allclose(X, ref.matrix, atol=1e-11)
    np.testing.assert_allclose(Y, ref.noise, atol=1e-10)
    np.testing.assert_allclose(d, ref.displacement, atol=1e-12)
    assert is_completely_positive(X, Y)


def test_conditional_semantics_match_postselected_trajectories():
    pattern = pg.protocols.teleportation(squeezing=0.6) if hasattr(pg, "protocols") else None
    pattern = pg.Circuit(1).rotate(0, 0.5).squeeze(0, 0.2).compile(squeezing=0.6)
    diagram = from_pattern(pattern)
    rng = np.random.default_rng(3)
    outcomes = {k: rng.normal() for k in evaluate(diagram).records}
    mean_in = np.array([0.4, -0.2])
    cov_in = np.array([[1.3, 0.2], [0.2, 0.9]])
    mean, cov, density = conditional_semantics(diagram, outcomes, mean_in, cov_in)
    result = pg.simulate(
        pattern,
        backend="gaussian",
        measurement_outcomes=outcomes,
        initial_state=pg.GaussianState(mean_in, cov_in, pattern.inputs),
    )
    np.testing.assert_allclose(result.state.mean, mean, atol=1e-9)
    np.testing.assert_allclose(result.state.covariance, cov, atol=1e-9)
    assert np.exp(result.log_likelihood) == pytest.approx(density, rel=1e-8)


def test_gaussian_graph_interoperability_and_measurement_rules():
    graph = GaussianGraph.cluster([[0, 1.0, 0], [1.0, 0, 0.5], [0, 0.5, 0]], 0.7)
    diagram = from_gaussian_graph(graph)
    back = to_gaussian_graph(diagram)
    np.testing.assert_allclose(back.Z, graph.Z, atol=1e-12)
    # cluster built from generators equals GaussianGraph.cluster (MFvL Eq. 2.29)
    built = CVDiagram(
        (),
        (
            Prepare.squeezed(0, 0.7),
            Prepare.squeezed(1, 0.7),
            Prepare.squeezed(2, 0.7),
            CZ(0, 1, 1.0),
            CZ(1, 2, 0.5),
        ),
        (0, 1, 2),
    )
    np.testing.assert_allclose(to_gaussian_graph(built).Z, graph.Z, atol=1e-12)
    # q and p homodyne graph rules (MFvL Sec. III) vs conditional diagram semantics
    for theta, rule in ((0.0, graph.measure_q), (np.pi / 2, graph.measure_p)):
        measured = CVDiagram((), built.boxes + (Homodyne(1, theta, "m"),), (0, 2))
        expected = rule(1, 0.37)
        got = to_gaussian_graph(measured, {"m": 0.37})
        np.testing.assert_allclose(got.Z, expected.Z, atol=1e-10)
        np.testing.assert_allclose(got.mean, expected.mean, atol=1e-10)
    with pytest.raises(ValueError):  # outcome-averaged state is mixed: no graph
        to_gaussian_graph(CVDiagram((), built.boxes + (Homodyne(1, 0.0, "m"),), (0, 2)))
    with pytest.raises(ValueError):
        to_gaussian_graph(CVDiagram((), (Prepare.squeezed(0, 0.3), GaussianLoss(0, 0.5)), (0,)))


# ----------------------------------------------------------------------------- rewrites
@pytest.mark.parametrize("seed", range(6))
def test_random_exact_rewrites_preserve_the_full_instrument(seed):
    rng = np.random.default_rng(seed)
    d = random_gaussian_diagram(rng, modes=2, depth=10)
    d = CVDiagram(
        d.inputs,
        d.boxes
        + (
            Rotation(0, 0.3),
            Rotation(0, -0.3),
            Displacement(1, 0.2, 0.1),
            Displacement(1, -0.1, 0.4),
            Squeezer(1, 0.0),
        ),
        d.outputs,
    )
    applied = 0
    for rule in GAUSSIAN_RULES:
        for match in rule.find(d)[:3]:
            out = rule.apply(d, match)
            report = compare(d, out)
            assert report.physical_equivalent and report.conditional_equivalent, rule.name
            applied += 1
    assert applied >= 4


def test_measurement_rewrites_are_exact_instruments():
    base = (
        Prepare.squeezed("a", 0.4),
        CZ(0, "a", 0.8),
        Homodyne("a", 0.0, "m"),
        Feedforward(0, (("m", (0.3, 0.0)),)),
    )
    d = CVDiagram((0,), base, (0,))
    step1 = apply_rewrite(d, "cz-q-measurement-to-feedforward")
    step2 = apply_rewrite(step1, "prepare-measure-to-source")
    assert any(isinstance(b, ClassicalSource) for b in step2.boxes)
    for out in (step1, step2):
        report = compare(d, out)
        assert report.conditional_equivalent and report.physical_equivalent
    assert sum(isinstance(b, Prepare) for b in step2.boxes) == 0  # the ancilla is eliminated
    # the recompiled pattern reintroduces one mode per classical source, exactly
    ref = gaussian_channel(to_pattern(step2))
    X, Y, dd = evaluate(d).channel()
    np.testing.assert_allclose(ref.matrix, X, atol=1e-12)
    np.testing.assert_allclose(ref.noise, Y, atol=1e-12)
    with pytest.raises(ValueError):  # an inefficient detector breaks the CZ rule
        apply_rewrite(
            CVDiagram(
                (0,),
                (
                    Prepare.squeezed("a", 0.4),
                    CZ(0, "a", 0.8),
                    Homodyne("a", 0.0, "m", efficiency=0.9),
                ),
                (0,),
            ),
            "cz-q-measurement-to-feedforward",
        )


def test_channel_fusion_and_commutation():
    d = CVDiagram((0, 1), (GaussianLoss(0, 0.7, 0.1), Rotation(1, 0.2), Squeezer(0, 0.3)), (0, 1))
    fused = apply_rewrite(d, "channel-fusion")
    assert fused.trace[-1].equivalence == "exact-channel"
    assert compare(d, fused).physical_equivalent
    swapped = Commute().apply(d, (0, 1))
    assert compare(d, swapped).conditional_equivalent


# ----------------------------------------------------------------------------- translation
def test_pattern_round_trip_and_unsupported_inputs():
    pattern = pg.Circuit(2).rotate(0, 0.3).cz(0, 1, 0.4).compile(squeezing=0.9)
    diagram = from_pattern(pattern)
    again = to_pattern(diagram)
    a, b = gaussian_channel(pattern), gaussian_channel(again)
    np.testing.assert_allclose(a.matrix, b.matrix, atol=1e-13)
    np.testing.assert_allclose(a.noise, b.noise, atol=1e-13)
    adaptive = pg.Pattern(inputs=(0,)).extend([pg.Prepare(1, 0.5), pg.Entangle(0, 1)])
    adaptive.measure(0, pg.Homodyne(0.4))
    adaptive.measure(1, pg.Homodyne(pg.Outcome(0)))
    adaptive.append(pg.Output(()))
    with pytest.raises(NotImplementedError):
        from_pattern(adaptive.validate())
    circuit = pg.Circuit(2).rotate(0, 0.3).squeeze(1, -0.2).beamsplitter(0, 1, 0.5).cz(0, 1, 0.2)
    S, _ = unitary_semantics(from_circuit(circuit))
    np.testing.assert_allclose(
        S, gaussian_channel(circuit.compile(squeezing=0.5)).matrix, atol=1e-12
    )


def test_serialisation_and_fingerprint_round_trip():
    d = CVDiagram(
        (0,),
        (
            Prepare.squeezed("a", 0.4),
            CZ(0, "a", 0.8),
            Homodyne("a", 0.3, "m"),
            Feedforward(0, (("m", (0.3, -0.2)),)),
            Symplectic((0,), gm.squeezing(0.2)),
        ),
        (0,),
    )
    again = CVDiagram.from_dict(json.loads(d.to_json()))
    assert again.fingerprint() == d.fingerprint()
    for x, y in zip(evaluate(d).channel(), evaluate(again).channel()):
        np.testing.assert_allclose(x, y)


def test_drawing_runs():
    import matplotlib

    matplotlib.use("Agg")
    d = from_pattern(pg.Circuit(1).rotate(0, 0.3).compile(squeezing=0.5))
    ax = draw(d, highlight=(0, 1))
    assert ax.get_yticks().size >= 2
    assert set(RULES) >= {
        "rotation-fusion",
        "teleport-chain-resynthesis",
        "phase-polynomial-fusion",
    }


def test_every_generator_translates_and_round_trips():
    from photographiq.calculus.cv_diagrams import (
        FockPrepare,
        Kerr,
        PhasePolynomial,
        PhotonCount,
        to_circuit,
    )

    gaussian = CVDiagram(
        (0, 1),
        (
            Rotation(0, 0.3),
            Squeezer(1, -0.2),
            Shear(0, 0.4),
            Displacement(1, 0.1, -0.3),
            BeamSplitter(0, 1, 0.6),
            CZ(1, 0, 0.5),
            Symplectic((0,), gm.squeezing(0.2) @ gm.rotation(0.7)),
            PhasePolynomial(1, (0.0, 0.2, 0.05)),
            GaussianLoss(0, 0.8, 0.1),
        ),
        (0, 1),
    )
    pattern = to_pattern(gaussian)
    ref = gaussian_channel(pattern)
    X, Y, d = evaluate(gaussian).channel()
    np.testing.assert_allclose(ref.matrix, X, atol=1e-12)
    np.testing.assert_allclose(ref.noise, Y, atol=1e-12)
    np.testing.assert_allclose(ref.displacement, d, atol=1e-12)
    np.testing.assert_allclose(evaluate(from_pattern(pattern)).channel()[0], X, atol=1e-12)
    unitary = CVDiagram((0, 1), gaussian.boxes[:-1], (0, 1))
    circuit = to_circuit(unitary)
    S, t = evaluate(unitary).unitary()
    np.testing.assert_allclose(evaluate(from_circuit(circuit)).unitary()[0], S, atol=1e-12)
    np.testing.assert_allclose(
        gaussian_channel(circuit.compile(squeezing=0.5)).matrix, S, atol=1e-10
    )
    fock = CVDiagram(
        (0,),
        (
            FockPrepare("a", 1),
            BeamSplitter(0, "a", 0.4),
            Kerr(0, 0.1),
            PhasePolynomial(0, (0.0, 0.0, 0.0, 0.02)),
            PhotonCount("a", "n"),
        ),
        (0,),
    )
    back = from_pattern(to_pattern(fock))
    assert [type(b).__name__ for b in back.boxes] == [
        "FockPrepare",
        "BeamSplitter",
        "Kerr",
        "PhasePolynomial",
        "PhotonCount",
    ]
    assert back.boxes[3].coefficients == pytest.approx((0.0, 0.0, 0.0, 0.02))
    with pytest.raises(NotImplementedError):
        to_circuit(CVDiagram((0,), (Homodyne(0, 0.2, "m"),), ()))
