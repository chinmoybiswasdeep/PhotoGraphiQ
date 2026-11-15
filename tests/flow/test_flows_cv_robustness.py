"""CV-flow robustness: rank margins, exact re-checks, regimes and the ``shift`` option.

See docs/validation/cv-flow-robustness.md for the graph set and the classification table.
"""

from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from photographiq.flows import analyze, exact_ideal_limit, find_cv_flow
from photographiq.flows.cv import _search
from photographiq.gaussian import wire_channel
from photographiq.ir import HomodyneSpec, Model, OpenGraph


def cv_graph(edges, inputs, outputs, shears=None, squeezing=0.8, nodes=()):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
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


def bridge(d):
    """Inputs u, v; outputs a, b; A[{u,v},{a,b}] = [[1, 1], [1, 1 + d]], det = d."""
    edges = [("u", "a", 1.0), ("u", "b", 1.0), ("v", "a", 1.0), ("v", "b", 1.0 + d)]
    return cv_graph(edges, ("u", "v"), ("a", "b"))


def wire(n, w=1.0, squeezing=0.8, shears=None):
    return cv_graph([(i, i + 1, w) for i in range(n)], (0,), (n,), shears, squeezing)


def grid(m, n):
    g = nx.grid_2d_graph(m, n)
    inputs = tuple((i, 0) for i in range(m))
    return cv_graph([(u, v, 1.0) for u, v in g.edges], inputs, tuple((i, n - 1) for i in range(m)))


def assert_both_regimes(result):
    assert result.certified and result.diagnostics["decision"] == "certified"
    assert set(result.certificate["regimes"]) == {"ideal_limit", "finite_squeezing"}
    report = result.certificate["determinism"]
    assert report["failed_checks"] == ()
    for kappa in report["decay_exponents"].values():
        assert kappa == np.inf or abs(kappa - 1) < 0.05


# ------------------------------------------------------------------ provably impossible


def check_witnesses(og, result):
    """Independently verify each rational witness: y^T A[W, C] = 0 with y_u != 0."""
    graph = og.graph
    for u, y in result.diagnostics["witnesses"].items():
        assert y.get(u, 0) != 0
        columns = [n for n in graph if n in og.outputs and n not in og.inputs]
        for c in columns:
            total = sum(
                Fraction(w) * Fraction(graph.edges[n, c]["weight"])
                for n, w in y.items()
                if graph.has_edge(n, c)
            )
            assert total == 0


@pytest.mark.parametrize(
    "og",
    [
        bridge(0.0),  # rank-1 correction matrix: e_u, e_v not both in its range
        cv_graph([(0, 2, 1.0), (1, 2, 1.0)], (0, 1), (2,)),  # |O| < |I|
        cv_graph(
            [("u", "a", 0.5), ("u", "b", 2.0), ("v", "a", 0.25), ("v", "b", 1.0)],
            ("u", "v"),
            ("a", "b"),
        ),
    ],
    ids=["unit-bridge", "more-inputs-than-outputs", "scaled-bridge"],
)
def test_impossible_graphs_carry_exact_witnesses(og):
    for exact in (None, True):
        result = analyze(og, exact=exact)
        assert result.status == "not_certified" and result.diagnostics["decision"] == "impossible"
        assert result.diagnostics["proof"].startswith("exact rank test")
        assert "only be postselected" in result.diagnostics["reason"]
        check_witnesses(og, result)
    # float64 only: a backward-error statement with a large margin, not an exact proof
    result = analyze(og, exact=False)
    assert result.diagnostics["decision"] == "impossible"
    assert result.diagnostics["proof"].startswith("float64 rank test (backward-error")
    assert result.diagnostics["rank_decisions"]["margin_decades"] > 10


def test_combinatorial_obstruction():
    result = analyze(cv_graph([(0, 1, 1.0)], (0,), (1,), nodes=[5]))
    assert result.diagnostics["decision"] == "impossible"
    assert result.diagnostics["proof"].startswith("combinatorial obstruction: nodes [5]")
    result = analyze(cv_graph([(0, 1, 1.0)], (0, 1), (1,)))  # only inputs are solved
    assert result.diagnostics["proof"].startswith("combinatorial obstruction: no measured")


# ------------------------------------------------------------------ near the threshold


@pytest.mark.parametrize("d", [1e-10, 1e-13])
def test_near_singular_weights_are_uncertain_or_resolved_exactly(d):
    og = bridge(d)
    floating = analyze(og, exact=False)
    assert (
        floating.status == "unknown" and floating.diagnostics["decision"] == "numerically_uncertain"
    )
    assert floating.diagnostics["rank_decisions"]["uncertain_float64_decisions"] >= 1
    assert "uncertain band" in floating.diagnostics["reason"]
    # Default: the exact re-check runs automatically and finds the flow (det = d != 0),
    # but corrections ~1/d cannot be re-certified in float64: still undecided.
    default = analyze(og)
    assert default.diagnostics["rank_decisions"]["decided_by"] == "exact (automatic)"
    assert default.status == "unknown" and "exact=True" in default.diagnostics["reason"]
    # exact=True: exact ideal-limit proof; finite squeezing is not claimed.
    exact = analyze(og, exact=True)
    assert exact.certified and set(exact.certificate["regimes"]) == {"ideal_limit"}
    assert "exact rational check" in exact.certificate["regimes"]["ideal_limit"]
    assert exact.diagnostics["correction_gain_max"] == pytest.approx(1 / d, rel=1e-3)
    assert "warning" in exact.diagnostics


def test_rounding_level_determinant_float_and_exact_answers_differ_and_are_labelled():
    # 1 + eps: the float64 data are invertible (det = eps), but the decision sits at the
    # rounding level. float64 says "impossible up to max(dims)*eps perturbations",
    # exact arithmetic finds a flow with gains ~1e16. Neither is reported as a float
    # certificate.
    og = bridge(2.0**-52)
    assert analyze(og, exact=False).diagnostics["proof"].startswith("float64 rank test")
    assert analyze(og).status == "unknown"
    exact = analyze(og, exact=True)
    assert exact.certified and set(exact.certificate["regimes"]) == {"ideal_limit"}
    assert exact.diagnostics["correction_gain_max"] > 1e15


# ------------------------------------------------------------------ scales and sizes


@pytest.mark.parametrize(
    "og",
    [
        wire(3, 1e-4),
        wire(3, 1e4),
        cv_graph([(0, 1, 1e-4), (1, 2, 1e4), (2, 3, 1e-4), (3, 4, 1e4)], (0,), (4,)),
        cv_graph(
            [
                (1, "a", 1e-4),
                (1, "b", 1e4),
                (2, "b", 1.0),
                (2, "c", 1e-4),
                (3, "a", 1e4),
                (3, "c", 1.0),
            ],
            (1, 2, 3),
            ("a", "b", "c"),
        ),
    ],
    ids=["wire-1e-4", "wire-1e4", "wire-alternating", "hexagon-mixed"],
)
def test_extreme_weights_float_check_cannot_run_exact_check_certifies(og):
    default = analyze(og)
    assert (
        default.status == "unknown" and default.diagnostics["decision"] == "numerically_uncertain"
    )
    assert default.diagnostics["rank_decisions"]["margin_decades"] > 10  # rank is clear
    exact = analyze(og, exact=True)
    assert exact.certified and set(exact.certificate["regimes"]) == {"ideal_limit"}


@pytest.mark.parametrize(
    "og", [wire(30), grid(3, 4), grid(4, 6)], ids=["wire-30", "grid-3x4", "grid-4x6"]
)
def test_long_wires_and_grids_certified_in_both_regimes(og):
    result = analyze(og)
    assert_both_regimes(result)
    assert result.diagnostics["rank_decisions"]["margin_decades"] > 10
    exact = analyze(og, exact=True)
    assert exact.certificate["exact_ideal_limit"]["ideal_limit_deterministic"]
    np.testing.assert_allclose(
        exact.certificate["exact_ideal_limit"]["matrix"],
        result.certificate["determinism"]["matrix"],
        atol=1e-9,
    )


# ------------------------------------------------------------------ squeezing and shift


@pytest.mark.parametrize("r", [3.5, 3.8, 4.0, 4.25, 4.5])
def test_large_squeezing_certified_with_smaller_shift(r):
    og = wire(3, squeezing=r)
    assert "could not run" in analyze(og).diagnostics["reason"]  # r + 6 exceeds float64
    result = analyze(og, shift=1.5)
    assert_both_regimes(result)
    assert result.certificate["determinism"]["shift"] == 1.5


def test_very_large_squeezing_ideal_limit_only_by_exact_check():
    og = wire(3, squeezing=8.0)
    result = analyze(og, exact=True)
    assert result.certified and set(result.certificate["regimes"]) == {"ideal_limit"}
    assert "error" in result.certificate["determinism"]
    with pytest.raises(ValueError, match="shift"):
        find_cv_flow(og, shift=0.0)


def test_exact_check_matches_closed_form_and_rejects_wrong_corrections():
    shears = {0: 0.2, 1: -0.4, 2: 0.7}
    og = wire(3, shears=shears, squeezing=0.6)
    search = _search(og, True)
    check = exact_ideal_limit(og, search["layers"], search["exact_vectors"])
    assert check["ideal_limit_deterministic"] and check["symplectic"]
    np.testing.assert_allclose(check["matrix"], wire_channel(shears.values(), 0.6)[0], atol=1e-12)
    for u in (0, 1, 2):
        for factor in (Fraction(0), Fraction(1, 2), Fraction(1) + Fraction(1, 2**60)):
            vectors = dict(search["exact_vectors"])
            vectors[u] = {n: factor * v for n, v in vectors[u].items()}
            assert not exact_ideal_limit(og, search["layers"], vectors)["ideal_limit_deterministic"]


def test_random_graphs_exact_and_float_agree():
    rng = np.random.default_rng(13)
    agree = 0
    for _ in range(30):
        n = int(rng.integers(3, 7))
        graph = nx.gnp_random_graph(n, rng.uniform(0.3, 0.7), seed=int(rng.integers(1 << 30)))
        edges = [(u, v, float(rng.choice([-1, 1]) * rng.uniform(0.3, 1.5))) for u, v in graph.edges]
        nodes = [int(v) for v in rng.permutation(n)]
        k = int(rng.integers(1, n // 2 + 2))
        inputs = tuple(
            int(v) for v in rng.choice(nodes, int(rng.integers(0, k + 1)), replace=False)
        )
        og = cv_graph(
            edges,
            inputs,
            tuple(nodes[:k]),
            {v: float(rng.uniform(-1, 1)) for v in nodes},
            nodes=nodes,
        )
        floating, exact = analyze(og, exact=False), analyze(og, exact=True)
        assert floating.status == exact.status, og.to_dict()
        assert floating.layers == exact.layers
        if exact.certified:
            assert "float64 decay check" in exact.certificate["regimes"]["ideal_limit"]
            agree += 1
    assert agree >= 5


# ------------------------------------------------------------------ float64 false negatives

# From the 600-graph adversarial set (docs/validation/cv-flow-robustness.md).
SLOW_DECAY = {  # weights 0.02..0.03 make the gain pre-asymptotic: kappa ~ 0.13
    "model": "cv-gaussian",
    "nodes": [{"id": i, "squeezing": r} for i, r in enumerate([3.182, 1.184, 2.047, 2.654, 2.070])],
    "edges": [
        {"source": 0, "target": 2, "weight": -0.7797811596989167},
        {"source": 0, "target": 3, "weight": -0.0235180637679479},
        {"source": 1, "target": 4, "weight": 1.1293195685548851},
        {"source": 2, "target": 3, "weight": 0.09390210961516596},
        {"source": 2, "target": 4, "weight": 0.02738323844115178},
    ],
    "inputs": [3, 2],
    "outputs": [1, 0, 4],
    "measurements": {"2": {"shear": -3.5207042731722558}, "3": {"shear": 1.582118143436272}},
}
DYNAMIC_RANGE = {  # N_plus eigenvalues 1.5e13 and 1e2: float64 counts one direction
    "model": "cv-gaussian",
    "nodes": [
        {"id": i, "squeezing": r}
        for i, r in enumerate([2.51, 2.146, 2.569, 3.34, 0.937, 2.989, 1.253])
    ],
    "edges": [
        {"source": 0, "target": 1, "weight": 1.654225976832471},
        {"source": 0, "target": 2, "weight": -4.819635722373409},
        {"source": 0, "target": 4, "weight": 2.0205602932168807},
        {"source": 1, "target": 2, "weight": 0.46364740899398377},
        {"source": 1, "target": 5, "weight": -0.05937474798273504},
        {"source": 2, "target": 3, "weight": -0.2608471700548828},
        {"source": 2, "target": 6, "weight": -0.028158556470908768},
        {"source": 3, "target": 5, "weight": -0.08142393880006119},
    ],
    "inputs": [],
    "outputs": [2, 3],
    "measurements": {
        "4": {"shear": 1.6290018098556436},
        "6": {"shear": -3.5006182525615994},
        "0": {"shear": 0.43629972335317646},
        "5": {"shear": 4.8725215078054305},
        "1": {"shear": 2.732325699365208},
    },
}


@pytest.mark.parametrize(
    "data,failed",
    [(SLOW_DECAY, ("decay",)), (DYNAMIC_RANGE, ("singular_directions",))],
    ids=["slow-decay", "dynamic-range"],
)
def test_float64_false_negatives_resolved_by_exact_check(data, failed):
    og = OpenGraph.from_dict(data)
    default = analyze(og)
    assert default.status == "unknown"
    assert default.certificate["determinism"]["failed_checks"] == failed
    result = analyze(og, exact=True)
    assert result.certified, result.diagnostics
    assert set(result.certificate["regimes"]) == {"ideal_limit", "finite_squeezing"}
    assert "exact rational check" in result.certificate["regimes"]["ideal_limit"]


def test_failed_float_checks_are_named_and_never_overruled_for_extra_directions():
    from dataclasses import replace

    from photographiq.commands import Displace
    from photographiq.cvflow import flow_pattern
    from photographiq.flows import resource_graph, verify_cv_determinism

    og = wire(3, shears={0: 0.3})
    result = analyze(og)
    pattern = flow_pattern(
        resource_graph(og), result.measurement_order, shears=result.certificate["shears"]
    )
    pattern.commands = [
        replace(c, q=0.5 * c.q) if isinstance(c, Displace) else c for c in pattern.commands
    ]
    report = verify_cv_determinism(pattern)
    assert "singular_directions" in report["failed_checks"]
    assert report["ideal_singular_directions"] > report["expected_singular_directions"]
