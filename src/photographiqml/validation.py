"""Independent, deliberately small logical validation and optional MentPy comparison."""

import numpy as np

from .logical import X, Z, cz, local_gate, statevector

MENTPY_COMMIT = "63c3d83e495696b4491c9d376dab7e3e6cf6c863"


def contract_branch(model, input_state, parameters=None, outcomes=None):
    """Full graph-state branch contraction with physical Pauli corrections.

    Returns (normalized logical state, branch probability). This exponential
    validator is capped at 16 graph nodes; it is not a production CV backend.
    Outcomes are bits in model.measurement_order; None chooses all zero.
    """
    if len(model.graph) > 16:
        raise ValueError("Independent contraction is limited to 16 graph nodes")
    measured = model.measurement_order
    outcomes = [0] * len(measured) if outcomes is None else list(outcomes)
    if len(outcomes) != len(measured) or any(s not in (0, 1) for s in outcomes):
        raise ValueError("Supply one binary outcome per measured node")
    angles = model._parameters.bind(parameters)
    nodes = list(model.input_nodes) + [v for v in model.graph if v not in model.input_nodes]
    state = statevector(input_state, model.n_wires)
    for _ in range(len(nodes) - model.n_wires):
        state = np.kron(state, np.ones(2) / np.sqrt(2))
    for u, v in model.graph.edges:
        state = cz(state, nodes.index(u), nodes.index(v), len(nodes))
    probability = 1.0
    for node, s in zip(measured, outcomes):
        angle = angles[model.parameter_name(node)]
        bra = np.array([1, (-1) ** s * np.exp(-1j * angle)]) / np.sqrt(2)
        tensor = np.moveaxis(state.reshape([2] * len(nodes)), nodes.index(node), 0)
        state = (bra @ tensor.reshape(2, -1)).reshape(-1)
        mass = float(np.vdot(state, state).real)
        probability *= mass
        state /= np.sqrt(mass)
        nodes.remove(node)
        if s:
            successor, z_targets = model.corrections[node]
            state = local_gate(state, X, nodes.index(successor), len(nodes))
            for target in z_targets:
                state = local_gate(state, Z, nodes.index(target), len(nodes))
    order = [nodes.index(v) for v in model.output_nodes]
    state = state.reshape([2] * model.n_wires).transpose(order).reshape(-1)
    return state, probability


def mentpy_reference(model, *, fix_measurements=False):
    """Return reference and its semantic label map; no isomorphism guessing.

    With fix_measurements=True, explicitly fix all model-frozen nodes before
    numerical comparison. The unmodified reference remains the topology and
    upstream trainability audit target.
    """
    try:
        import mentpy as mp
    except ImportError as exc:
        raise ImportError("MentPy validation requires photographiq[ml-validation]") from exc
    if model.connections is not None:
        raise ValueError("MentPy template reference supports full connectivity only")
    reference = mp.templates.muta(
        model.n_wires,
        model.n_layers,
        one_column=model.one_column,
        restrict_trainable=model.restrict_trainable,
    )
    mapping = {}
    for w, start in enumerate(reference.input_nodes):
        node = start
        for c in range(4 * model.paper_depth + 1):
            mapping[node] = (w, c)
            if node not in reference.output_nodes:
                node = reference.flow(node)
        if node != reference.output_nodes[w]:
            raise ValueError("Reference flow does not preserve expected wire semantics")
    if fix_measurements:
        for node, semantic in mapping.items():
            name = model.parameter_name(semantic)
            if name in model._parameters.frozen:
                reference[node] = mp.Ment(model._parameters.values[name], "XY")
    return reference, mapping


def compare_mentpy(model, input_state, parameters=None):
    # mentpy_reference raises the install hint when MentPy is missing.
    reference, mapping = mentpy_reference(model, fix_measurements=True)
    import mentpy as mp

    angles = model._parameters.bind(parameters)
    values = [angles[model.parameter_name(mapping[v])] for v in reference.trainable_nodes]
    # A full graph window is only appropriate for small references. Use the
    # flow schedule window otherwise; enforce all required incident resources.
    reverse = {v: k for k, v in mapping.items()}
    schedule = [reverse[v] for v in model.measurement_order + model.output_nodes]
    positions = {v: i for i, v in enumerate(schedule)}
    window = max(abs(positions[u] - positions[v]) + 1 for u, v in reference.graph.edges)
    window = max(window, model.n_wires + 1)
    simulator = mp.PatternSimulator(
        reference,
        input_state=np.asarray(input_state, complex),
        backend="numpy-sv",
        schedule=schedule,
        window_size=window,
    )
    actual = simulator.run(values, output_form="dm")
    expected = model.run(input_state, parameters).density_matrix
    return float(np.max(abs(actual - expected)))
