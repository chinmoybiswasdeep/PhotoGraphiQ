"""Qubit flows: textbook cases, independent verifiers, statevector determinism, Graphix.

The strongest check is ``assert_deterministic``: an independent NumPy statevector
simulation of every outcome branch of the corrected pattern, with each input qubit
maximally entangled with a reference qubit, so equality of all branch states (up to
global phase) is equality of the implemented linear maps (Choi states).
"""

import itertools
import math

import networkx as nx
import numpy as np
import pytest

from photographiq.flows import (
    analyze,
    find_causal_flow,
    find_gflow,
    find_pauli_flow,
    verify_causal_flow,
    verify_gflow,
    verify_pauli_flow,
)
from photographiq.ir import HomodyneSpec, Model, OpenGraph, Plane, QubitMeasurement

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0 + 0j, -1.0])
CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
CNOT = np.eye(4, dtype=complex)[[0, 1, 3, 2]]
PAULI_ANGLE = {"X": {"XY": 0.0, "XZ": math.pi / 2}, "Y": {"XY": math.pi / 2, "YZ": math.pi / 2}}
PAULI_ANGLE["Z"] = {"XZ": 0.0, "YZ": 0.0}


def observable(m: QubitMeasurement) -> np.ndarray:
    """XY: cos X + sin Y; XZ: cos Z + sin X; YZ: cos Z + sin Y."""
    a, b = {"XY": (X, Y), "XZ": (Z, X), "YZ": (Z, Y)}[m.plane.value]
    return math.cos(m.angle) * a + math.sin(m.angle) * b


def apply(state, op, axes):
    k = len(axes)
    out = np.tensordot(op.reshape((2,) * 2 * k), state, axes=(list(range(k, 2 * k)), axes))
    return np.moveaxis(out, list(range(k)), axes)


def branch_states(og, order, corrections, outcomes, *, correct=True):
    """Unnormalised output Choi vector of one outcome branch."""
    nodes = list(og.nodes)
    alive = nodes + [("ref", n) for n in og.inputs]
    state = np.zeros((2,) * len(alive), dtype=complex)
    state[(0,) * len(alive)] = 1
    hadamard = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    for i in range(len(nodes)):
        state = apply(state, hadamard, [i])
    for n in og.inputs:  # |+>|0> -> Bell pair with the reference qubit
        state = apply(state, CNOT, [alive.index(n), alive.index(("ref", n))])
    for u, v in og.graph.edges:
        state = apply(state, CZ, [alive.index(u), alive.index(v)])
    for u, s in zip(order, outcomes):
        values, vectors = np.linalg.eigh(observable(og.measurements[u]))
        phi = vectors[:, int(np.argmin(values))] if s else vectors[:, int(np.argmax(values))]
        state = np.tensordot(phi.conj(), state, axes=([0], [alive.index(u)]))
        alive.remove(u)
        if s and correct:
            for gate, key in ((X, "x"), (Z, "z")):
                for t in corrections[u][key]:
                    assert t in alive, "correction on an already measured node"
                    state = apply(state, gate, [alive.index(t)])
    final = list(og.outputs) + [("ref", n) for n in og.inputs]
    return np.transpose(state, [alive.index(n) for n in final]).reshape(-1)


def branch_spread(og, order, corrections, *, correct=True):
    """Max over branches of (1 - |<ref|branch>|, |p_branch - 2^-m|)."""
    m = len(order)
    reference = branch_states(og, order, corrections, (0,) * m, correct=correct)
    reference = reference / np.linalg.norm(reference)
    overlap_err = prob_err = 0.0
    for outcomes in itertools.product((0, 1), repeat=m):
        vec = branch_states(og, order, corrections, outcomes, correct=correct)
        prob = np.linalg.norm(vec) ** 2
        prob_err = max(prob_err, abs(prob - 2.0**-m))
        overlap_err = max(overlap_err, 1 - abs(np.vdot(reference, vec)) / math.sqrt(prob))
    return overlap_err, prob_err


def assert_deterministic(og, result):
    assert result.certified
    overlap_err, prob_err = branch_spread(og, result.measurement_order, result.correction_map)
    # Exact arithmetic gives 0; observed float64 residuals are ~1e-15.
    assert overlap_err < 1e-12 and prob_err < 1e-12


def qubit_graph(edges, inputs, outputs, specs=None, nodes=()):
    graph = nx.Graph(list(edges))
    graph.add_nodes_from(nodes)
    specs = specs or {}
    measurements = {
        n: specs.get(n, QubitMeasurement(Plane.XY, 0.3 + 0.1 * i))
        for i, n in enumerate(graph.nodes)
        if n not in outputs
    }
    return OpenGraph(Model.QUBIT, graph, inputs, outputs, measurements)


def pauli(label, plane):
    return QubitMeasurement(Plane(plane), PAULI_ANGLE[label][plane], label)


# ----------------------------------------------------------------------------- textbook


def test_line_has_causal_flow_with_textbook_flow_function_and_is_deterministic():
    og = qubit_graph(nx.path_graph(5).edges, (0,), (4,))
    result = analyze(og)
    assert result.algorithm.startswith("causal")
    assert result.certificate["flow"] == {0: 1, 1: 2, 2: 3, 3: 4}
    assert result.layers == ((0,), (1,), (2,), (3,))
    assert result.correction_map[1] == {"x": frozenset({2}), "z": frozenset({3})}
    assert verify_causal_flow(og, result.certificate["flow"], result.layers) == []
    assert_deterministic(og, result)


def test_rectangular_cluster_has_causal_flow():
    grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, 3), ordering="sorted")
    og = qubit_graph(grid.edges, (0, 3), (2, 5))  # rows (0,1,2), (3,4,5)
    result = find_causal_flow(og)
    assert result.certified and result.diagnostics["complete"]
    assert_deterministic(og, result)


def test_dropping_corrections_breaks_determinism():
    og = qubit_graph(nx.path_graph(4).edges, (0,), (3,))
    result = find_causal_flow(og)
    overlap_err, _ = branch_spread(
        og, result.measurement_order, result.correction_map, correct=False
    )
    assert overlap_err > 0.1


def test_gflow_without_causal_flow_xy_only():
    edges = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 4), (3, 4)]
    og = qubit_graph(edges, (3, 2), (0, 4))
    causal = find_causal_flow(og)
    assert causal.status == "not_certified" and causal.diagnostics["complete"]
    result = analyze(og)
    assert result.algorithm.startswith("gflow")
    assert verify_gflow(og, result.certificate["correction_sets"], result.layers) == []
    assert_deterministic(og, result)


def test_yz_and_xz_planes_need_gflow():
    og = qubit_graph([(0, 1)], (), (1,), {0: QubitMeasurement(Plane.YZ, 0.7)})
    assert find_causal_flow(og).status == "not_certified"
    result = find_gflow(og)
    assert result.certificate["correction_sets"] == {0: frozenset({0})}
    assert_deterministic(og, result)
    # An XZ node inside an input line has no gflow (it would need its input neighbour
    # later); as the first node of a state-preparation line it is fine.
    line = nx.path_graph(4).edges
    assert analyze(qubit_graph(line, (0,), (3,), {1: QubitMeasurement(Plane.XZ, 0.4)})).status == (
        "not_certified"
    )
    og = qubit_graph(line, (), (3,), {0: QubitMeasurement(Plane.XZ, 0.4)})
    result = analyze(og)
    assert result.algorithm.startswith("gflow")
    assert result.certificate["correction_sets"][0] == frozenset({0, 1})
    assert_deterministic(og, result)


def test_pauli_flow_needed_for_x_measurements_in_xz_plane():
    # Graphix documentation example: XZ-labelled nodes have no gflow; as X Paulis they do.
    og = qubit_graph([(0, 1), (1, 2)], (0,), (2,), {0: pauli("X", "XZ"), 1: pauli("X", "XZ")})
    assert find_gflow(og).status == "not_certified"
    result = analyze(og)
    assert result.algorithm.startswith("pauli")
    assert result.certificate["correction_sets"] == {0: frozenset({1}), 1: frozenset({2})}
    assert result.layers == ((0, 1),)  # X nodes may correct each other without order
    assert verify_pauli_flow(og, result.certificate["correction_sets"], result.layers) == []
    assert_deterministic(og, result)


def test_pauli_flow_with_y_and_z_measurements_is_deterministic():
    specs = {1: pauli("Y", "YZ"), 2: pauli("Z", "XZ"), 3: QubitMeasurement(Plane.XY, 0.9)}
    og = qubit_graph([(0, 1), (1, 2), (2, 3), (1, 3), (3, 4), (2, 5)], (0,), (4, 5), specs)
    result = find_pauli_flow(og)
    assert result.certified
    assert_deterministic(og, result)


def test_verifiers_reject_corrupted_certificates():
    og = qubit_graph(nx.path_graph(4).edges, (0,), (3,))
    result = find_gflow(og)
    sets, layers = result.certificate["correction_sets"], result.layers
    assert verify_gflow(og, sets, layers) == []
    assert verify_gflow(og, sets, layers[::-1])  # reversed order breaks G1/G2
    assert verify_gflow(og, {**sets, 0: frozenset({2})}, layers)  # plane condition
    assert verify_gflow(og, {**sets, 1: frozenset({0, 2})}, layers)  # input in g
    assert verify_pauli_flow(og, sets, layers[:-1])  # not a partition
    flow = find_causal_flow(og).certificate["flow"]
    assert verify_causal_flow(og, {**flow, 1: 3}, layers)  # not a neighbour


def test_no_flow_is_not_reported_as_nondeterminism():
    og = qubit_graph([(0, 1), (0, 2), (1, 2)], (0, 1), (2,))  # |I| > |O|
    result = analyze(og)
    assert result.status == "not_certified"
    assert "does not prove nondeterminism" in result.diagnostics["reason"]
    assert set(result.diagnostics["attempts"]) == {"causal", "gflow"}


def test_model_and_input_validation():
    cv = OpenGraph(Model.CV_GAUSSIAN, nx.path_graph(2), (0,), (1,), {0: HomodyneSpec()})
    for name in ("causal", "gflow", "pauli"):
        assert analyze(cv, name).status == "unsupported"
    og = qubit_graph([(0, 1)], (0,), (1,))
    assert analyze(og, "cv").status == "unsupported"
    # Regression: CV options with "cv" on a qubit graph raised TypeError.
    assert analyze(og, "cv", verify=False, tolerance=1e-8).status == "unsupported"
    with pytest.raises(TypeError, match="Unexpected options"):
        analyze(og, "auto", verify=False)
    for model in (Model.CV_NONGAUSSIAN, Model.GKP_LOGICAL):
        specs = {0: HomodyneSpec() if model.continuous else QubitMeasurement()}
        other = OpenGraph(model, nx.path_graph(2), (0,), (1,), specs)
        assert analyze(other).status == "unsupported"
    with pytest.raises(ValueError, match="Missing measurement"):
        find_gflow(OpenGraph(Model.QUBIT, nx.path_graph(2), (0,), (1,), {}))
    bad = qubit_graph([(0, 1)], (0,), (1,), {0: QubitMeasurement(Plane.XY, 0.3, "X")})
    with pytest.raises(ValueError, match="disagrees"):
        find_pauli_flow(bad)
    with pytest.raises(ValueError):
        analyze(og, "nonsense")


# ----------------------------------------------------------------------------- random


def random_open_graph(rng, n, pauli_fraction, planes=("XY", "XZ", "YZ")):
    graph = nx.gnp_random_graph(n, rng.uniform(0.25, 0.6), seed=int(rng.integers(1 << 30)))
    nodes = [int(v) for v in rng.permutation(n)]
    k = int(rng.integers(max(1, n // 3), n // 2 + 2))
    outputs = tuple(nodes[:k])
    inputs = tuple(int(v) for v in rng.choice(nodes, int(rng.integers(0, k + 1)), replace=False))
    specs = {}
    for v in graph.nodes:
        if v in outputs:
            continue
        if rng.random() < pauli_fraction:
            label = str(rng.choice(["X", "Y", "Z"]))
            specs[v] = pauli(label, str(rng.choice(sorted(PAULI_ANGLE[label]))))
        else:
            specs[v] = QubitMeasurement(Plane(str(rng.choice(planes))), rng.uniform(0.1, 1.4))
    return OpenGraph(Model.QUBIT, graph, inputs, outputs, specs)


CASES = [  # (finder, verifier, planes, pauli fraction)
    (find_causal_flow, None, ("XY",), 0.0),
    (find_gflow, verify_gflow, ("XY", "XZ", "YZ"), 0.0),
    (find_pauli_flow, verify_pauli_flow, ("XY", "XZ", "YZ"), 0.4),
]


@pytest.mark.parametrize("finder,verifier,planes,fraction", CASES)
def test_random_flows_verify_and_are_deterministic_by_statevector(
    finder, verifier, planes, fraction
):
    rng = np.random.default_rng(7)
    certified = 0
    for _ in range(60):
        og = random_open_graph(rng, int(rng.integers(3, 7)), fraction, planes)
        result = finder(og)
        if not result.certified:
            assert result.status == "not_certified" and result.diagnostics["complete"]
            continue
        certified += 1
        if verifier is None:
            assert verify_causal_flow(og, result.certificate["flow"], result.layers) == []
        else:
            assert verifier(og, result.certificate["correction_sets"], result.layers) == []
        assert_deterministic(og, result)
    assert certified >= 10


def graphix_open_graph(og, *, pauli_axes):
    from graphix.fundamentals import Axis
    from graphix.fundamentals import Plane as GPlane
    from graphix.opengraph import OpenGraph as GOpenGraph

    def label(m):
        if pauli_axes and m.pauli:
            return getattr(Axis, m.pauli)
        return getattr(GPlane, m.plane.value)

    measurements = {n: label(m) for n, m in og.measurements.items()}
    return GOpenGraph(og.graph, list(og.inputs), list(og.outputs), measurements)


@pytest.mark.parametrize("finder,verifier,planes,fraction", CASES)
def test_existence_agrees_with_graphix(finder, verifier, planes, fraction):
    pytest.importorskip("graphix")
    rng = np.random.default_rng(11)
    agree = positive = 0
    for _ in range(150):
        og = random_open_graph(rng, int(rng.integers(3, 9)), fraction, planes)
        g = graphix_open_graph(og, pauli_axes=finder is find_pauli_flow)
        theirs = {
            find_causal_flow: g.to_causalflow_or_none,
            find_gflow: g.to_gflow_or_none,
            find_pauli_flow: g.to_pauliflow_or_none,
        }[finder]()
        mine = finder(og)
        assert mine.certified == (theirs is not None), og.to_dict()
        agree += 1
        if theirs is None or verifier is None:
            continue
        positive += 1
        # Our independent verifier accepts Graphix's (Mitosek-Backens) certificate ...
        layers = [set(layer) for layer in theirs.partial_order_layers]
        layers = tuple(tuple(layer) for layer in reversed(layers[1:] if og.outputs else layers))
        sets = {u: set(p) for u, p in theirs.correction_function.items()}
        assert verifier(og, sets, layers) == []
        # ... and ours is maximally delayed, so never deeper.
        assert len(mine.layers) <= len(layers)
    assert agree == 150 and (verifier is None or positive >= 20)
