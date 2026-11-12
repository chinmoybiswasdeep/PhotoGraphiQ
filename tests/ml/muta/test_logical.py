import itertools

import networkx as nx
import numpy as np
import pytest
from scipy.linalg import expm

from photographiqml import MuTA, TriangleNeuron
from photographiqml.diagnostics import concurrence
from photographiqml.logical import X, Z
from photographiqml.models import haar_states
from photographiqml.validation import contract_branch


@pytest.mark.parametrize("wires,layers", [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2)])
@pytest.mark.parametrize("one_column", [False, True])
@pytest.mark.parametrize("restricted", [False, True])
def test_topology(wires, layers, one_column, restricted):
    m = MuTA(wires, layers, one_column=one_column, restrict_trainable=restricted)
    d = layers * (1 if one_column else wires)
    assert len(m.graph) == wires * (4 * d + 1)
    assert m.graph.number_of_edges() == 4 * wires * d + 2 * d * (wires - 1)
    assert m.n_parameters == m.expected_parameter_count == (3 if restricted else 4) * wires * d
    assert nx.is_bipartite(m.graph)
    assert nx.is_directed_acyclic_graph(m.dependency_graph)
    positions = {v: i for i, v in enumerate(m.measurement_order + m.output_nodes)}
    assert all(positions[u] < positions[v] for u, v in m.dependency_graph.edges)
    assert np.allclose(m.unitary(), np.eye(2**wires))


def test_triangle_and_table_one():
    triangle = TriangleNeuron()
    assert len(triangle.graph) == 10
    assert len(triangle.trainable_nodes) == 8
    assert len(triangle.graph.edges) == 10
    assert triangle.input_nodes == ((0, 0), (1, 0))
    assert triangle.output_nodes == ((0, 4), (1, 4))
    m = MuTA(2, one_column=True)
    phi = 0.731
    u = m.unitary({"alpha.w1.c1": phi})
    assert np.allclose(u, expm(0.5j * phi * np.kron(X, X)))
    assert concurrence(u[:, 0]) == pytest.approx(abs(np.sin(phi)))
    theta, phi, lam = 0.21, 0.45, -0.72
    u = m.unitary({"alpha.w0.c1": theta, "alpha.w0.c2": phi, "alpha.w0.c3": lam})
    expected = expm(0.5j * lam * X) @ expm(0.5j * phi * Z) @ expm(0.5j * theta * X)
    assert np.allclose(u, np.kron(expected, np.eye(2)))


def test_all_adaptive_branches():
    model = MuTA(2, one_column=True)
    state = haar_states(2, 1, 71)[0]
    parameters = model.initialize(14, scale=1)
    target = model.run(state, parameters).density_matrix
    total = 0.0
    for outcomes in itertools.product((0, 1), repeat=8):
        output, probability = contract_branch(model, state, parameters, outcomes)
        assert np.allclose(np.outer(output, output.conj()), target, atol=1e-11)
        assert probability == pytest.approx(1 / 256)
        total += probability
    assert total == pytest.approx(1)


@pytest.mark.parametrize("wires", [1, 2, 3])
def test_depth_inclusion_and_composition(wires):
    small = MuTA(wires, 1)
    larger = MuTA(wires, 2)
    values = small.initialize(42, scale=1)
    assert np.allclose(small.unitary(values), larger.unitary(values))
    assert np.allclose(small.unitary(values).conj().T @ small.unitary(values), np.eye(2**wires))


def test_disconnected_wires():
    m = MuTA(2, one_column=True, connections=())
    a = m.initialize(42)
    one = MuTA(1, one_column=True)
    left = {one.parameter_name((0, c)): a[m.parameter_name((0, c))] for c in range(4)}
    right = {one.parameter_name((0, c)): a[m.parameter_name((1, c))] for c in range(4)}
    assert np.allclose(m.unitary(a), np.kron(one.unitary(left), one.unitary(right)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_wires": 0},
        {"n_layers": True},
        {"representation": "cv"},
        {"one_column": 1},
        {"connections": ()},
    ],
)
def test_bad_model(kwargs):
    with pytest.raises(ValueError):
        MuTA(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"pivot": 2}, {"connections": (0,)}, {"connections": (1, 1)}, {"connections": (False,)}],
)
def test_bad_triangle(kwargs):
    with pytest.raises(ValueError):
        TriangleNeuron(**kwargs)
