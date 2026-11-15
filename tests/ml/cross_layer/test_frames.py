from itertools import product

import numpy as np

import photographiq as pg
from photographiqml import MuTA
from photographiqml.logical import X, Z, cz, local_gate
from photographiqml.lowering import node_frame
from photographiqml.models import haar_states


def test_all_signed_x_graph_branches_with_virtual_frames():
    model = MuTA(2, one_column=True)
    angles = {"alpha.w1.c1": np.pi, "alpha.w0.c2": np.pi}
    values = model._parameters.bind(angles)
    source = haar_states(2, 1, 13)[0]
    target = model.run(source, angles).density_matrix
    nodes0 = list(model.input_nodes) + [v for v in model.graph if v not in model.input_nodes]
    initial = source
    for _ in range(8):
        initial = np.kron(initial, np.ones(2) / np.sqrt(2))
    for u, v in model.graph.edges:
        initial = cz(initial, nodes0.index(u), nodes0.index(v), 10)
    for branch in product((0, 1), repeat=8):
        state, nodes, records = initial.copy(), list(nodes0), {}
        for node, raw_bit in zip(model.measurement_order, branch, strict=True):
            # Raw signed-X measurement, without physical corrections on survivors.
            alpha = values[model.parameter_name(node)]
            bra = np.array([1, (-1) ** raw_bit * np.exp(-1j * alpha)]) / np.sqrt(2)
            tensor = np.moveaxis(state.reshape([2] * len(nodes)), nodes.index(node), 0)
            state = bra @ tensor.reshape(2, -1)
            state /= np.linalg.norm(state)
            nodes.remove(node)
            frame = node_frame(model, node, records)
            records[("bit", node)] = raw_bit ^ frame.correction("X")
        for node in model.output_nodes:
            frame = node_frame(model, node, records)
            if frame.x_bit:
                state = local_gate(state, X, nodes.index(node), 2)
            if frame.z_bit:
                state = local_gate(state, Z, nodes.index(node), 2)
        state = (
            state.reshape(2, 2).transpose([nodes.index(v) for v in model.output_nodes]).reshape(-1)
        )
        assert np.allclose(np.outer(state, state.conj()), target)
    # Gate-conjugation helpers exist, but are not also applied to graph corrections.
    left, right = pg.LogicalPauliFrame(1, 0).cz(pg.LogicalPauliFrame())
    assert left == pg.LogicalPauliFrame(1, 0) and right == pg.LogicalPauliFrame(0, 1)
