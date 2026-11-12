"""CV (Model.CV_GAUSSIAN) flow search, weighted conditions and determinism verification."""

import itertools
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

import photographiq as pg
from photographiq.commands import Displace
from photographiq.cvflow import certify_cv_flow, flow_pattern
from photographiq.flows import analyze, find_gflow, resource_graph, verify_cv_determinism
from photographiq.gaussian import wire_channel
from photographiq.ir import HomodyneSpec, Model, OpenGraph, QubitMeasurement


def cv_graph(edges, inputs, outputs, shears=None, squeezing=0.8):
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    shears = shears or {}
    return OpenGraph(
        Model.CV_GAUSSIAN,
        graph,
        inputs,
        outputs,
        {n: HomodyneSpec(shears.get(n, 0.0)) for n in graph if n not in outputs},
        {n: squeezing for n in graph},
    )


def assert_ideal_limit(report):
    assert report["ideal_limit_deterministic"]
    # Outcome dependence of the conditional output decays as exp(-2r): kappa = 1.
    for kappa in report["decay_exponents"].values():
        assert kappa == np.inf or abs(kappa - 1) < 0.05


def test_wire_order_search_matches_teleportation_closed_form():
    shears = {0: 0.2, 1: -0.4, 2: 0.7}
    og = cv_graph([(i, i + 1, 1.0) for i in range(3)], (0,), (3,), shears, squeezing=0.6)
    result = analyze(og)
    assert result.certified and result.layers == ((0,), (1,), (2,))
    # m = record/sin(theta) = p + k q; unit-weight wire corrects q of the next node.
    assert result.correction_map[0] == {1: {"q": -1.0, "p": 0.0}, 2: {"q": 0.0, "p": -1.0}}
    report = result.certificate["determinism"]
    matrix, noise = wire_channel(shears.values(), 0.6)  # independent closed form
    np.testing.assert_allclose(report["matrix"], matrix, atol=1e-12)
    np.testing.assert_allclose(report["noise"], noise, atol=1e-12)
    assert report["symplectic_residual"] < 1e-12
    assert_ideal_limit(report)
    # Finite squeezing is reported, not hidden: records still shift the conditional mean.
    assert report["finite_squeezing_gain"] > 0.1


def test_weighted_two_node_correction_closed_form():
    og = cv_graph([(0, 1, 0.7)], (0,), (1,))
    result = analyze(og)
    assert result.correction_map[0][1]["q"] == pytest.approx(-1 / 0.7)
    report = result.certificate["determinism"]
    np.testing.assert_allclose(report["matrix"], [[0, -1 / 0.7], [0.7, 0]], atol=1e-12)
    assert_ideal_limit(report)


def paper_unitary(adjacency, shears):
    """U = C_Z(A) (+)_i F D(s_i), F = [[0,-1],[1,0]], D(s) = [[1,0],[s,1]]."""
    n = len(adjacency)
    local = np.zeros((2 * n, 2 * n))
    for i, s in enumerate(shears):
        local[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = [[-s, -1], [1, 0]]
    cz = np.eye(2 * n)
    cz[1::2, 0::2] = adjacency
    return cz @ local


def test_multimode_graph_teleportation_matches_closed_form():
    n = 4
    rng = np.random.default_rng(4)
    base = nx.cycle_graph(n)
    weights = {e: rng.uniform(0.3, 1.2) for e in base.edges}
    shears = rng.uniform(-1, 1, n)
    edges = [(("C", u), ("C", v), w) for (u, v), w in weights.items()]
    edges += [(("R", i), ("C", i), 1.0) for i in range(n)]
    inputs, outputs = tuple(("R", i) for i in range(n)), tuple(("C", i) for i in range(n))
    og = cv_graph(edges, inputs, outputs, dict(zip(inputs, shears)), squeezing=1.2)
    result = analyze(og)
    assert result.certified and result.layers == (inputs,)  # one parallel layer
    adjacency = nx.to_numpy_array(base, nodelist=range(n))
    for (u, v), w in weights.items():
        adjacency[u, v] = adjacency[v, u] = w
    report = result.certificate["determinism"]
    np.testing.assert_allclose(report["matrix"], paper_unitary(adjacency, shears), atol=1e-12)
    np.testing.assert_allclose(
        report["noise"], np.exp(-2 * 1.2) * np.diag(np.tile([0.0, 1.0], n)), atol=1e-12
    )
    assert_ideal_limit(report)


def test_weights_decide_cv_flow_on_the_same_skeleton():
    unit = [("u", "a", 1.0), ("u", "b", 1.0), ("v", "a", 1.0), ("v", "b", 1.0)]
    signed = unit[:3] + [("v", "b", -1.0)]
    result = analyze(cv_graph(unit, ("u", "v"), ("a", "b")))
    assert result.status == "not_certified" and result.diagnostics["complete"]
    assert "not a proof of nondeterminism" in result.diagnostics["reason"]
    signed_result = analyze(cv_graph(signed, ("u", "v"), ("a", "b")))
    assert signed_result.certified
    assert_ideal_limit(signed_result.certificate["determinism"])


def qubit_skeleton(og):
    graph = nx.Graph(og.graph.edges)
    measured = {n: QubitMeasurement() for n in og.measured}
    return OpenGraph(Model.QUBIT, graph, og.inputs, og.outputs, measured)


def test_cv_flow_and_qubit_gflow_are_different_notions():
    hexagon = [(u, o, 1.0) for u, o in [(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "a"), (3, "c")]]
    og = cv_graph(hexagon, (1, 2, 3), ("a", "b", "c"))
    # GF(2): the 0/1 matrix has even determinant; over the reals it is invertible.
    assert find_gflow(qubit_skeleton(og)).status == "not_certified"
    result = analyze(og)
    assert result.certified
    assert_ideal_limit(result.certificate["determinism"])
    weighted = hexagon + [(3, "b", 2.0)]
    og = cv_graph(weighted, (1, 2, 3), ("a", "b", "c"))
    assert find_gflow(qubit_skeleton(og)).certified
    assert analyze(og).status == "not_certified"  # weighted determinant is zero


def test_broken_feed_forward_fails_ideal_limit_check():
    og = cv_graph([(i, i + 1, 1.0) for i in range(3)], (0,), (3,), {0: 0.3})
    result = analyze(og)
    pattern = flow_pattern(
        resource_graph(og), result.measurement_order, shears=result.certificate["shears"]
    )
    # Any missing or wrong feed-forward leaves antisqueezed resource noise (variance
    # ~e^{2r}) in some output quadrature beyond the |O|-|I| ideal squeezed modes.
    variants = {
        "none": lambda c: None,
        "q_only": lambda c: replace(c, p=0.0),
        "p_only": lambda c: replace(c, q=0.0),
        "half_q": lambda c: replace(c, q=0.5 * c.q),
    }
    for name, change in variants.items():
        broken = pattern.copy()
        broken.commands = [change(c) if isinstance(c, Displace) else c for c in broken.commands]
        broken.commands = [c for c in broken.commands if c is not None]
        report = verify_cv_determinism(broken)
        assert not report["ideal_limit_deterministic"], name
        assert report["ideal_singular_directions"] > report["expected_singular_directions"], name


def test_found_order_is_admissible_where_supplied_reverse_order_is_not():
    og = cv_graph([(i, i + 1, 1.0) for i in range(4)], (0,), (4,))
    graph = resource_graph(og)
    assert certify_cv_flow(graph, (3, 2, 1, 0)) is None
    result = analyze(og, verify=False)
    assert result.measurement_order == (0, 1, 2, 3)
    assert "numerical determinism verification skipped" in result.assumptions


def random_cv_graph(rng, n):
    graph = nx.gnp_random_graph(n, rng.uniform(0.3, 0.7), seed=int(rng.integers(1 << 30)))
    for u, v in graph.edges:
        graph.edges[u, v]["weight"] = float(rng.choice([-1, 1]) * rng.uniform(0.3, 1.5))
    nodes = [int(v) for v in rng.permutation(n)]
    k = int(rng.integers(1, n // 2 + 2))
    outputs = tuple(nodes[:k])
    inputs = tuple(int(v) for v in rng.choice(nodes, int(rng.integers(0, k + 1)), replace=False))
    specs = {v: HomodyneSpec(float(rng.uniform(-1, 1))) for v in nodes if v not in outputs}
    squeezing = {v: float(rng.uniform(0.3, 1.5)) for v in nodes}
    return OpenGraph(Model.CV_GAUSSIAN, graph, inputs, outputs, specs, squeezing)


def test_random_weighted_graphs_existence_matches_brute_force_over_orders():
    rng = np.random.default_rng(5)
    certified = 0
    for _ in range(40):
        og = random_cv_graph(rng, int(rng.integers(2, 7)))
        graph = resource_graph(og)
        brute = any(
            certify_cv_flow(graph, order) is not None
            for order in itertools.permutations(og.measured)
        )
        result = analyze(og, verify=False)
        assert result.certified == brute, og.to_dict()
        if result.certified:
            certified += 1
            assert certify_cv_flow(graph, result.measurement_order) is not None
    assert certified >= 8


def test_random_certified_graphs_pass_ideal_limit_verification():
    rng = np.random.default_rng(9)
    checked = 0
    while checked < 6:
        og = random_cv_graph(rng, int(rng.integers(3, 6)))
        result = analyze(og)
        if result.status == "not_certified":
            continue
        assert result.certified, result.diagnostics
        assert_ideal_limit(result.certificate["determinism"])
        checked += 1


def test_conditional_simulation_agrees_with_unconditional_channel_on_average():
    # Law of total covariance, independent of the verifier: sample trajectories and
    # compare the outcome-averaged output moments with gaussian_channel(pattern).
    og = cv_graph([(0, 1, 0.9), (1, 2, -0.6)], (0,), (2,), {0: 0.3, 1: -0.2}, squeezing=0.7)
    result = analyze(og, verify=False)
    pattern = flow_pattern(resource_graph(og), result.measurement_order, shears={0: 0.3, 1: -0.2})
    state = pg.GaussianState(np.array([0.5, -0.4]), np.diag([1.5, 0.8]), (0,))
    shots = pg.run_shots(pattern, 4000, seed=3, backend="gaussian", initial_state=state)
    means = np.array([r.state.mean for r in shots.trajectories])
    expected = pg.gaussian_channel(pattern).apply(state)
    covariance = np.cov(means.T) + shots.trajectories[0].state.covariance
    # Monte Carlo: 4000 shots give ~2% relative standard error on second moments.
    np.testing.assert_allclose(means.mean(axis=0), expected.mean, atol=0.1)
    np.testing.assert_allclose(covariance, expected.covariance, rtol=0.1, atol=0.05)


def test_corrections_respect_layers_and_large_squeezing_is_undecided_not_an_error():
    # Review regression: total-order minimum-norm vectors corrected nodes of the same
    # (declared parallel) layer; e.g. the hexagon's single input layer.
    hexagon = [(u, o, 1.0) for u, o in [(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "a"), (3, "c")]]
    result = analyze(cv_graph(hexagon, (1, 2, 3), ("a", "b", "c")))
    assert result.certified and result.layers == ((1, 2, 3),)
    assert all(set(t) <= {"a", "b", "c"} for t in result.correction_map.values())
    # The check raises the squeezing to r + 6. For r >= 6, e^{-2(r+6)} < eps e^{2(r+6)}:
    # the squeezed quadratures are lost to float64 rounding of the anti-squeezed ones
    # and GaussianState validation (absolute 1e-7) fails, so the check cannot run. The
    # result must be undecided ("unknown"), never an exception and never certified.
    for r in (6.0, 6.0 + 1e-12, 7.3, 8.0, 10.0):
        late = analyze(cv_graph([(i, i + 1, 1.0) for i in range(3)], (0,), (3,), squeezing=r))
        assert late.status == "unknown" and "could not run" in late.diagnostics["reason"], r
        assert "ValueError" in late.certificate["determinism"]["error"], r
    assert analyze(cv_graph([], (), ())).certified  # empty graph: trivially deterministic


def test_large_squeezing_is_certified_or_undecided_never_failed():
    # Regression: an absolute 1e-9 bound on the law-of-total-covariance spread rejected
    # r = 3.1 and 3.3 (spread_min ~ -1e-8 from rounding of e^{2(r+6)} ~ 1e8 entries).
    # Between r ~ 3.5 and 4.5 float64 validation may or may not fail at the raised
    # levels depending on rounding; then the check is undecided, but a check that runs
    # on this deterministic wire must pass.
    values = [3.0 + 0.1 * k for k in range(16)] + [3.3 + 1e-12, 4.0, 4.000000000000002]
    for r in values:
        result = analyze(cv_graph([(i, i + 1, 1.0) for i in range(3)], (0,), (3,), squeezing=r))
        report = result.certificate["determinism"]
        if "error" in report:
            assert result.status == "unknown" and "could not run" in result.diagnostics["reason"]
        else:
            assert result.certified, (r, result.diagnostics)
            assert_ideal_limit(report)
        if r < 3.45:
            assert result.certified, r


def test_wrong_correction_is_still_rejected_at_large_squeezing():
    # The relative rounding allowance must not hide nondeterminism: wrong feed-forward
    # leaves anti-squeezed noise in an output (an extra ideal singular direction).
    for r in (3.1, 3.3):
        og = cv_graph([(i, i + 1, 1.0) for i in range(3)], (0,), (3,), {0: 0.3}, squeezing=r)
        result = analyze(og)
        assert result.certified, r
        pattern = flow_pattern(
            resource_graph(og), result.measurement_order, shears=result.certificate["shears"]
        )
        for scale in (0.0, 0.5, 1 + 1e-3):
            broken = pattern.copy()
            broken.commands = [
                replace(c, q=scale * c.q) if isinstance(c, Displace) else c for c in broken.commands
            ]
            report = verify_cv_determinism(broken)
            assert not report["ideal_limit_deterministic"], (r, scale)
            assert report["ideal_singular_directions"] > report["expected_singular_directions"]
