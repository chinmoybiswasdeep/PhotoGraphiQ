"""R37: Exhaustive virtual-Pauli-frame validation for all signed-X branches
of the smallest nontrivial two-wire model.

Scientific question: For every one of the 2**8 raw measurement-outcome
branches of a two-wire, one-layer, fully-connected signed-X MuTA graph, does
tracking photographiqml.lowering.node_frame's virtual LogicalPauliFrame
(without ever applying a physical correction operator) and applying the
resulting logical Pauli correction at the very end reproduce exactly the
same target density matrix that MuTA.run computes deterministically?

Theory/equations: the fully-entangled open-graph correction convention
(docs/physical/pauli-frames.md): frame contributions compose by XOR from
interpreted earlier outcomes; at the end, the accumulated frame's X/Z
components are applied as physical Pauli gates to the surviving output
qubits.

Functionality tested: photographiqml.lowering.node_frame (the same public
function the physical execution layer uses to track virtual frames), tested
here purely in the qubit Hilbert space (no Fock simulation), for a random
signed-X angle assignment across every measured column (broader than the
single fixed 2-angle case in tests/cross_layer/test_frames.py).

Oracle and independence class: D (independent code path -- this experiment
performs its own raw-outcome graph-state contraction, a different algorithm
from logical.execute's frontier translation, while reusing the production
node_frame function itself as the object under test, matching how the
physical execution layer actually uses it).

Exact/approximate/statistical status: exact, up to floating-point roundoff.

Primary metric: max density-matrix Frobenius error across all 256 branches.

Declared acceptance condition: max error < tol
(tol = declare_tolerance(scale=1, safety_factor=100)).

Expected cost: light (256 branches, qubit-space contraction only, no Fock
simulation).

Manuscript destination: Main text (Fig. 9, Pauli-frame validation panel).

Scientific limitations: Establishes the discrete flow/frame convention in
the qubit Hilbert space only; it does not by itself certify finite-GKP
physical execution accuracy (separate finite-resource errors are R43-R48's
job, per docs/physical/pauli-frames.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from itertools import product

import common
import numpy as np

from photographiqml import MuTA
from photographiqml.logical import X, Z, cz, local_gate
from photographiqml.lowering import node_frame
from photographiqml.models import haar_states

EXPERIMENT_ID = "R37"


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=100.0)
    model = MuTA(2, one_column=True)
    n_measured = len(model.measurement_order)  # 8
    angles = dict(
        zip(model.trainable_parameters(), common.rng(0).choice([0.0, np.pi], model.n_parameters))
    )
    values = model._parameters.bind(angles)
    source_state = haar_states(2, 1, seed=17)[0]
    target = model.run(source_state, angles).density_matrix

    nodes0 = list(model.input_nodes) + [v for v in model.graph.nodes if v not in model.input_nodes]
    initial = source_state
    for _ in range(len(nodes0) - model.n_wires):
        initial = np.kron(initial, np.ones(2) / np.sqrt(2))
    for u, v in model.graph.edges:
        initial = cz(initial, nodes0.index(u), nodes0.index(v), len(nodes0))

    max_error = 0.0
    worst_branch = None
    for branch in product((0, 1), repeat=n_measured):
        state, nodes, records = initial.copy(), list(nodes0), {}
        for node, raw_bit in zip(model.measurement_order, branch, strict=True):
            alpha = values[model.parameter_name(node)]
            bra = np.array([1, (-1) ** raw_bit * np.exp(-1j * alpha)]) / np.sqrt(2)
            tensor = np.moveaxis(state.reshape([2] * len(nodes)), nodes.index(node), 0)
            state = bra @ tensor.reshape(2, -1)
            state = state / np.linalg.norm(state)
            nodes.remove(node)
            frame = node_frame(model, node, records)
            records[("bit", node)] = raw_bit ^ frame.correction("X")
        for node in model.output_nodes:
            frame = node_frame(model, node, records)
            if frame.x_bit:
                state = local_gate(state, X, nodes.index(node), model.n_wires)
            if frame.z_bit:
                state = local_gate(state, Z, nodes.index(node), model.n_wires)
        state = (
            state.reshape([2] * model.n_wires)
            .transpose([nodes.index(v) for v in model.output_nodes])
            .reshape(-1)
        )
        error = common.frobenius_error(np.outer(state, state.conj()), target)
        if error > max_error:
            max_error, worst_branch = error, branch

    status = "pass" if max_error < tol else "fail"

    common.save_result(
        [{"n_branches": 2**n_measured, "max_error": max_error, "worst_branch": worst_branch}],
        "R37_pauli_frame_validation",
        extra={
            "protocol": "Exhaustive raw-branch contraction + node_frame correction vs. MuTA.run target, 2-wire signed-X",
            "oracle_class": "D",
            "status_category": "exact",
            "tolerance": tol,
            "acceptance_condition": f"max error over all {2**n_measured} branches < {tol:.3e}",
            "max_error": max_error,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "D", "status": status},
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["max branch error"],
        [max_error],
        color=common.COLORS["photographiqml"] if status == "pass" else common.COLORS["piquasso"],
    )
    ax.axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    ax.set_yscale("log")
    ax.set(
        title=f"R37: exhaustive Pauli-frame validation, {2**n_measured} branches (status={status})",
        ylabel="max density-matrix error",
    )
    ax.legend()
    common.save_figure(fig, "R37_pauli_frame_validation")
    plt.close(fig)

    common.print_summary(
        "R37 Pauli frame validation", n_branches=2**n_measured, max_error=max_error, status=status
    )
    if status != "pass":
        raise AssertionError(
            f"R37 failed: max_error={max_error} tol={tol} worst_branch={worst_branch}"
        )


if __name__ == "__main__":
    main()
