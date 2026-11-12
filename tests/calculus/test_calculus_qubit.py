"""Qubit measurement calculus: conventions, gate patterns, rewrite soundness, flows."""

from __future__ import annotations

import math
from collections import Counter

import networkx as nx
import numpy as np
import pytest
from calculus_random import PLANES, brute_force_gflow, random_pattern

from photographiq.calculus.qubit import (
    E,
    M,
    N,
    QubitPattern,
    S,
    X,
    Z,
    _rewrite,
    effective_angle,
    enumerate_branches,
    equivalent,
    is_deterministic,
    measurement_basis,
    observable,
    pattern_from_flow,
    rewrite_at,
    run_branch,
    signal_shift,
    standardize,
)
from photographiq.ir import Model, OpenGraph, Plane, QubitMeasurement

ATOL = 1e-12  # exact linear algebra on <= 2**8 dimensional objects; observed errors ~1e-15
PX = np.array([[0, 1], [1, 0]], complex)
PY = np.array([[0, -1j], [1j, 0]])
PZ = np.diag([1.0 + 0j, -1.0])
H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)


def up_to_phase(a: np.ndarray, b: np.ndarray) -> float:
    phase = np.vdot(b, a)
    phase = phase / abs(phase) if abs(phase) > 1e-300 else 1
    return float(np.abs(a - phase * b).max())


def J(alpha: float) -> np.ndarray:
    """J(alpha) = H Rz(alpha), Rz(alpha) = diag(1, e^{i alpha})."""
    return H @ np.diag([1, np.exp(1j * alpha)])


@pytest.mark.parametrize("plane", PLANES)
def test_basis_is_eigenbasis_of_documented_observable(plane):
    for a in np.linspace(-4, 4, 9):
        o = observable(plane, a)
        for r in (0, 1):
            v = measurement_basis(plane, a, r)
            assert np.allclose(o @ v, (-1) ** r * v, atol=ATOL)
            assert np.isclose(np.vdot(v, v), 1, atol=ATOL)
    assert np.allclose(observable(Plane.XY, 0), PX) and np.allclose(
        observable(Plane.XY, 1.5708), PY, atol=1e-5
    )
    assert np.allclose(observable(Plane.YZ, 0), PZ) and np.allclose(
        observable(Plane.XZ, 1.5708), PX, atol=1e-5
    )


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("s,t", [(0, 1), (1, 0), (1, 1)])
def test_domain_table_equals_pauli_conjugation(plane, s, t):
    p = np.linalg.matrix_power(PX, s) @ np.linalg.matrix_power(PZ, t)
    for a in np.linspace(-3, 3, 7):
        conj = p.conj().T @ observable(plane, a) @ p
        assert np.allclose(conj, observable(plane, effective_angle(plane, a, s, t)), atol=ATOL)


@pytest.mark.parametrize("alpha", [0.0, 0.4, -1.3, math.pi / 2, 2.9])
def test_j_pattern_implements_h_rz(alpha):
    # Measuring qubit 1 at XY angle -alpha implements J(alpha); outcome 1 is fixed by X^{s1}.
    pattern = QubitPattern((1,), [N(2), E(1, 2), M(1, Plane.XY, -alpha), X(2, {1})])
    report = is_deterministic(pattern)
    assert report.deterministic and report.max_deviation < ATOL
    assert up_to_phase(report.isometry, J(alpha)) < ATOL
    # Without the correction the branches differ by X: not deterministic.
    bare = QubitPattern((1,), [N(2), E(1, 2), M(1, Plane.XY, -alpha)])
    assert not is_deterministic(bare).deterministic


def test_dkp_cnot_pattern():
    # DKP (2007) CNOT: X_4^{s3} Z_4^{s2} Z_1^{s2} M_3 M_2 E_13 E_23 E_34 N_3 N_4 (control 1).
    pattern = QubitPattern(
        (1, 2),
        [N(3), N(4), E(1, 3), E(2, 3), E(3, 4), M(2), M(3), X(4, {3}), Z(4, {2}), Z(1, {2})],
        outputs=(1, 4),
    )
    cnot = np.eye(4)[[0, 1, 3, 2]]
    report = is_deterministic(pattern)
    assert report.deterministic and up_to_phase(report.isometry, cnot) < ATOL
    for branch in enumerate_branches(pattern):
        assert np.isclose(branch.probability(), 0.25, atol=ATOL)


def test_validation_rejects_indefinite_patterns():
    with pytest.raises(ValueError, match="unprepared or measured"):
        QubitPattern((0,), [M(0), X(0, {0})])
    with pytest.raises(ValueError, match="unmeasured"):
        QubitPattern((0,), [N(1), M(0, s_domain={1}), M(1)])
    with pytest.raises(ValueError, match="re-prepares"):
        QubitPattern((0,), [N(0)], outputs=(0,))
    with pytest.raises(ValueError, match="one qubit"):
        QubitPattern((0,), [E(0, 0)])
    with pytest.raises(ValueError, match="Outputs"):
        QubitPattern((0,), [N(1), M(0)], outputs=(0, 1))
    with pytest.raises(ValueError, match="other signals"):
        QubitPattern((0, 1), [M(0), S(0, {0})], outputs=(1,))


def test_dependencies_and_schedule():
    p = QubitPattern(
        (0,),
        [
            N(1),
            N(2),
            N(3),
            E(0, 1),
            E(1, 2),
            E(2, 3),
            M(0),
            M(1, s_domain={0}),
            M(2, t_domain={0, 1}),
        ],
    )
    assert p.dependencies() == {0: frozenset(), 1: frozenset({0}), 2: frozenset({0, 1})}
    assert p.schedule() == (frozenset({0}), frozenset({1}), frozenset({2}))
    assert p.dependency_graph().edges[0, 2]["kind"] == "t"
    with pytest.raises(ValueError, match="unmeasured"):
        p.reorder_measurements([1, 0, 2])


def test_each_rewrite_rule_preserves_the_map():
    rng = np.random.default_rng(11)
    used: Counter = Counter()
    for _ in range(60):
        pattern = random_pattern(rng, signals=True)
        for i in range(len(pattern.commands) - 1):
            if _rewrite(pattern.commands[i], pattern.commands[i + 1]) is None:
                continue
            rule, rewritten = rewrite_at(pattern, i)
            used[rule] += 1
            report = equivalent(pattern, rewritten, atol=ATOL)
            assert report.equivalent, (rule, report.max_deviation)
            # EX/EZ/commutations are exact operator identities. MX/MZ hold up to a phase of the
            # branch, e.g. <+_a| X = exp(-ia) <+_{-a}|.
            if rule not in ("MX", "MZ"):
                assert all(abs(p) < 1e-9 for p in report.phases.values()), rule
    assert {"EX", "EZ", "MX", "MZ", "commute", "S-commute"} <= used.keys(), used


def test_standardization_preserves_map_and_reaches_nemc():
    rng = np.random.default_rng(5)
    for _ in range(40):
        pattern = random_pattern(rng, signals=True)
        trace: list = []
        standard = standardize(pattern, trace=trace)
        assert standard.is_standard()
        assert all(
            isinstance(c, (X, Z)) == (c.node in standard.outputs)
            for c in standard.commands
            if isinstance(c, (M, X, Z))
        )
        assert equivalent(pattern, standard, atol=ATOL).equivalent
        assert standardize(standard).commands == standard.commands  # idempotent


def test_signal_shifting_preserves_map_and_empties_flipping_domains():
    rng = np.random.default_rng(7)
    for _ in range(40):
        pattern = random_pattern(rng)
        shifted = signal_shift(pattern)
        assert equivalent(pattern, shifted, atol=ATOL).equivalent
        for c in shifted.commands:
            if isinstance(c, M):
                assert not (c.t_domain if c.plane is Plane.XY else c.s_domain)
        both = standardize(signal_shift(standardize(pattern)))
        assert both.is_standard() and equivalent(pattern, both, atol=ATOL).equivalent


def test_reordering_measurements_along_dependencies():
    rng = np.random.default_rng(3)
    for _ in range(20):
        standard = standardize(random_pattern(rng))
        graph = standard.dependency_graph()
        order = list(nx.lexicographical_topological_sort(graph, key=lambda n: -n))
        assert equivalent(standard, standard.reorder_measurements(order), atol=ATOL)


def test_run_branch_probabilities_sum_to_one():
    rng = np.random.default_rng(1)
    pattern = random_pattern(rng)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    assert np.isclose(sum(b.probability(psi) for b in enumerate_branches(pattern)), 1, atol=ATOL)
    with pytest.raises(ValueError, match="Missing outcomes"):
        run_branch(pattern, {})


def _open_graph(graph, inputs, outputs, planes, angles):
    measurements = {v: QubitMeasurement(planes[v], angles[v]) for v in graph if v not in outputs}
    return OpenGraph(Model.QUBIT, graph, inputs, outputs, measurements)


def test_causal_flow_line_pattern_is_composition_of_j_gates():
    graph = nx.path_graph(4)
    angles = {0: 0.3, 1: -1.1, 2: 2.0}
    og = _open_graph(graph, (0,), (3,), {v: Plane.XY for v in graph}, angles)
    corrections = {i: {"x": {i + 1}, "z": set(graph[i + 1]) - {i}} for i in range(3)}
    pattern = pattern_from_flow(og, corrections, order=(0, 1, 2))
    report = is_deterministic(pattern)
    expected = J(-angles[2]) @ J(-angles[1]) @ J(-angles[0])
    assert report.deterministic and up_to_phase(report.isometry, expected) < ATOL
    transposed = {
        "x": {i: c["x"] for i, c in corrections.items()},
        "z": {i: c["z"] for i, c in corrections.items()},
    }
    assert equivalent(pattern, pattern_from_flow(og, transposed, order=(0, 1, 2)))
    with pytest.raises(ValueError, match="earlier"):
        pattern_from_flow(og, corrections, order=(1, 0, 2))


def test_gflow_patterns_are_deterministic_for_all_planes():
    rng = np.random.default_rng(2024)
    checked = Counter()
    for _ in range(80):
        n = int(rng.integers(3, 6))
        graph = nx.gnp_random_graph(n, 0.6, seed=int(rng.integers(1 << 30)))
        if not nx.is_connected(graph):
            continue
        inputs, outputs = (0,), (n - 1,) if rng.random() < 0.5 else (n - 2, n - 1)
        planes = {v: PLANES[rng.integers(3)] if v not in inputs else Plane.XY for v in graph}
        angles = {v: float(rng.uniform(-3, 3)) for v in graph}
        flow = brute_force_gflow(graph, inputs, outputs, planes)
        if flow is None:
            continue
        order, corrections = flow
        og = _open_graph(graph, inputs, outputs, planes, angles)
        report = is_deterministic(pattern_from_flow(og, corrections, order))
        assert report.deterministic, report.max_deviation
        checked.update(planes[v] for v in order)
        # Dropping the corrections of one measurement breaks determinism (negative control).
        broken = dict(corrections)
        victim = next((i for i in order if corrections[i]["x"] | corrections[i]["z"]), None)
        if victim is not None:
            broken[victim] = {"x": set(), "z": set()}
            assert not is_deterministic(pattern_from_flow(og, broken, order)).deterministic
    assert all(checked[p] >= 3 for p in PLANES), checked


def test_long_line_runs_with_two_live_qubits():
    # 31 qubits in total but at most 2 live (+1 input batch axis): within MAX_QUBITS.
    n = 31
    graph = nx.path_graph(n)
    angles = {v: 0.1 * v - 1.0 for v in graph}
    og = _open_graph(graph, (0,), (n - 1,), {v: Plane.XY for v in graph}, angles)
    corrections = {i: {"x": {i + 1}, "z": set(graph[i + 1]) - {i}} for i in range(n - 1)}
    standard = pattern_from_flow(og, corrections, order=range(n - 1))
    # Interleave N/E/M so the register stays small (standard form would keep all 31 live).
    commands = []
    for i in range(n - 1):
        commands += [N(i + 1), E(i, i + 1)]
        commands += [c for c in standard.commands if isinstance(c, M) and c.node == i]
    commands += [c for c in standard.commands if isinstance(c, (X, Z))]
    line = QubitPattern((0,), commands, (n - 1,))
    expected = np.eye(2)
    for i in range(n - 1):
        expected = J(-angles[i]) @ expected
    rng = np.random.default_rng(0)
    branch = run_branch(line, {i: int(rng.integers(2)) for i in range(n - 1)})
    assert up_to_phase(branch.operator * 2 ** ((n - 1) / 2), expected) < ATOL


def test_correction_table_named_x_z_is_rejected_as_ambiguous():
    graph = nx.path_graph(["x", "z", "o"])
    og = _open_graph(graph, ("x",), ("o",), {v: Plane.XY for v in graph}, {"x": 0.1, "z": 0.2})
    plain = {"x": {"x": {"z"}, "z": {"o"}}, "z": {"x": {"o"}}}
    with pytest.raises(ValueError, match="ambiguous"):
        pattern_from_flow(og, plain, order=("x", "z"))


def test_pattern_from_flow_rejects_cv_graphs():
    og = OpenGraph(Model.CV_GAUSSIAN, nx.path_graph(2), (0,), (1,), {})
    with pytest.raises(ValueError, match="QUBIT"):
        pattern_from_flow(og, {}, ())


def test_pattern_from_flow_accepts_flow_analysis_results():
    flows = pytest.importorskip("photographiq.flows")
    rng = np.random.default_rng(31)
    certified = rejected = 0
    for _ in range(150):
        n = int(rng.integers(3, 6))
        graph = nx.gnp_random_graph(n, 0.6, seed=int(rng.integers(1 << 30)))
        if not nx.is_connected(graph):
            continue
        planes = {v: PLANES[rng.integers(3)] if v else Plane.XY for v in graph}
        og = _open_graph(graph, (0,), (n - 1,), planes, {v: rng.uniform(-3, 3) for v in graph})
        result = flows.analyze(og)
        if not result.certified:
            assert brute_force_gflow(graph, (0,), (n - 1,), planes) is None
            with pytest.raises(ValueError, match="not certified"):
                pattern_from_flow(og, result)
            rejected += 1
            continue
        assert is_deterministic(pattern_from_flow(og, result)).deterministic
        certified += 1
    assert certified >= 10 and rejected >= 10


def test_pattern_from_pauli_flow_results_is_deterministic():
    # Pauli flow lets X_{p(u)} / Z_{Odd(p(u))} act on earlier Pauli-measured qubits where
    # they commute with that measurement. flows.find_pauli_flow drops those components
    # from correction_map, so pattern_from_flow accepts the result (a raw table keeping
    # them raises ValueError) and the pattern must be strongly deterministic.
    flows = pytest.importorskip("photographiq.flows")
    from photographiq.flows._common import odd_neighbourhood

    paulis = {"X": (Plane.XY, 0.0), "Y": (Plane.XY, math.pi / 2), "Z": (Plane.XZ, 0.0)}
    rng = np.random.default_rng(7)
    counts: Counter = Counter()
    for _ in range(200):
        n = int(rng.integers(3, 7))
        graph = nx.gnp_random_graph(n, 0.55, seed=int(rng.integers(1 << 30)))
        if not nx.is_connected(graph):
            continue
        outputs = (n - 1,) if rng.random() < 0.5 else (n - 2, n - 1)
        measurements = {}
        for v in graph:
            if v in outputs:
                continue
            if v and rng.random() < 0.5:
                label = "XYZ"[rng.integers(3)]
                measurements[v] = QubitMeasurement(*paulis[label], label)
            else:
                measurements[v] = QubitMeasurement(Plane.XY, float(rng.uniform(-3, 3)))
        og = OpenGraph(Model.QUBIT, graph, (0,), outputs, measurements)
        result = flows.find_pauli_flow(og)
        if not result.certified:
            continue
        order = result.measurement_order
        position = {u: k for k, u in enumerate(order)}
        sets = result.certificate["correction_sets"]
        raw = {u: {"x": set(sets[u]), "z": set(odd_neighbourhood(og, sets[u]))} for u in order}
        past = [
            t
            for u in order
            for t in (raw[u]["x"] | raw[u]["z"]) - {u}
            if t in position and position[t] < position[u]
        ]
        if past:
            counts["past"] += 1
            assert all(measurements[t].pauli for t in past)
            with pytest.raises(ValueError, match="earlier"):
                pattern_from_flow(og, raw, order)
        pattern = pattern_from_flow(og, result)
        report = is_deterministic(pattern)
        assert report.deterministic, report.max_deviation
        gflow = flows.find_gflow(og)
        if gflow.certified:
            assert equivalent(pattern, pattern_from_flow(og, gflow))
        else:
            counts["pauli only"] += 1
        # Negative control: dropping the output corrections of one outcome breaks it.
        victim = next(u for u in order if result.correction_map[u]["x"] & set(outputs))
        broken = {**result.correction_map, victim: {"x": set(), "z": set()}}
        assert not is_deterministic(pattern_from_flow(og, broken, order)).deterministic
        counts["certified"] += 1
    assert counts["past"] >= 10 and counts["pauli only"] >= 10 and counts["certified"] >= 40
