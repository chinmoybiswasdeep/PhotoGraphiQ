import pytest

pytest.importorskip("mentpy")

from photographiqml import MuTA
from photographiqml.models import haar_states
from photographiqml.validation import compare_mentpy, mentpy_reference


@pytest.mark.parametrize("wires,layers", [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2)])
@pytest.mark.parametrize("one_column", [False, True])
@pytest.mark.parametrize("restricted", [False, True])
def test_exact_semantic_reference(wires, layers, one_column, restricted):
    model = MuTA(wires, layers, one_column=one_column, restrict_trainable=restricted)
    ref, mapping = mentpy_reference(model)
    edges = {frozenset((mapping[u], mapping[v])) for u, v in ref.graph.edges}
    assert edges == {frozenset(e) for e in model.graph.edges}
    assert tuple(mapping[v] for v in ref.input_nodes) == model.input_nodes
    assert tuple(mapping[v] for v in ref.output_nodes) == model.output_nodes
    assert {mapping[v] for v in ref.graph} == set(model.graph)
    actual = {mapping[v] for v in ref.trainable_nodes}
    expected = set(model.trainable_nodes)
    if restricted and (model.paper_depth > 1 or wires > 1):
        # Both edge insertion and stacking rebuild the trainable-node list.
        assert actual == set(model.measured_nodes)
        assert actual - expected == {v for v in model.measured_nodes if v[1] % 4 == 3}
    else:
        assert actual == expected
    order = {mapping[v]: i for i, v in enumerate(ref.measurement_order)}
    assert all(order[u] < order[v] for u, v in model.dependency_graph.edges)


@pytest.mark.parametrize(
    "wires,layers,one_column",
    [(1, 1, True), (2, 1, True), (3, 1, True), (2, 1, False), (2, 2, False), (3, 2, False)],
)
@pytest.mark.parametrize("restricted", [False, True])
def test_numerical_reference(wires, layers, one_column, restricted):
    model = MuTA(wires, layers, one_column=one_column, restrict_trainable=restricted)
    state = haar_states(wires, 1, 11)[0]
    discrepancy = compare_mentpy(model, state, model.initialize(12, 1))
    assert discrepancy < 1e-10
