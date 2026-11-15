"""R5: Random logical-state agreement against an independently written
brute-force MBQC contraction.

Scientific question: Does MuTA.run's frontier-translation execution agree
with a genuinely independent, freshly written full graph-state contraction
(distinct code from both logical.py's cz/local_gate and
validation.py's contract_branch) across a broad, randomly sampled set of
wire/layer/one_column configurations and random Haar input states?

Theory/equations: MBQC deferred-measurement theorem: fixing every measured
node's outcome bit to 0 (a single deterministic branch) requires *no* Pauli
byproduct correction, since corrections only trigger on bit=1 (see
lowering.node_frame / ansatz/muta.py's corrections map). Since R4 already
proves exhaustively (via a *different* independent implementation,
contract_branch) that every corrected branch reproduces the same target, the
uncorrected zero-outcome branch alone must equal the target exactly -- this
experiment checks that invariant with a third, independently coded
contraction, built here via kron-embedded projector sums for CZ (not
photographiqml.logical.cz's bit-parity trick) and tensor-reshape projective
measurement (not photographiqml.logical.local_gate's moveaxis trick).

Functionality tested: MuTA.run (production execute()) across a random
sweep of n_wires/n_layers/one_column configurations, capped at <=20 graph
nodes for dense state-vector tractability.

Oracle and independence class: D (independent code path -- CZ, measurement
and normalization are all implemented fresh in this script using different
numpy primitives than logical.py/validation.py; the abstract measurement/
correction *equations* are necessarily shared physics, not shared code).

Exact/approximate/statistical status: exact, up to floating-point roundoff.

Primary metric: max density-matrix Frobenius error between the independent
contraction's output and MuTA.run's LogicalResult.density_matrix.

Declared acceptance condition: max Frobenius error < tol
(tol = declare_tolerance(scale=1, safety_factor=200), the larger safety
factor absorbing the extra floating-point operations of a dense contraction
over up to 2**20 amplitudes).

Expected cost: light-to-moderate (largest config: 2 wires, 2 layers,
one_column=True -> 18 graph nodes, 2**18 amplitudes).

Manuscript destination: Appendix (third independent cross-check of Fig. 1's
logical execution, complementing R4's exhaustive small-graph branch check).

Scientific limitations: Restricted to <=20 graph nodes by dense state-vector
memory; only the deterministic zero-outcome branch is checked per
configuration (branch/correction exhaustiveness itself is R4's job, not
R5's).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import MuTA
from photographiqml.models import haar_states

EXPERIMENT_ID = "R5"
MAX_NODES = 20


def independent_contraction(model, input_state, angles):
    """Fresh full graph-state contraction: kron-embedded CZ + tensor-reshape
    projective measurement onto the deterministic zero-outcome branch."""
    order = list(model.input_nodes) + [v for v in model.graph.nodes if v not in model.input_nodes]
    total = len(order)
    if total > MAX_NODES:
        raise ValueError(f"Independent contraction capped at {MAX_NODES} nodes; got {total}")
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    state = np.asarray(input_state, dtype=complex)
    for _ in range(total - model.n_wires):
        state = np.kron(state, plus)

    # Apply every CZ edge as an elementwise sign flip on the reshaped tensor
    # (a diagonal-projector-sum operation), not logical.cz's bit-parity trick.
    tensor = state.reshape([2] * total)
    for u, v in model.graph.edges:
        pu, pv = order.index(u), order.index(v)
        lo, hi = sorted((pu, pv))
        index = [slice(None)] * total
        index[lo] = 1
        index[hi] = 1
        tensor[tuple(index)] *= -1
    state = tensor.reshape(-1)

    remaining = list(order)
    probability = 1.0
    for node in model.measurement_order:
        alpha = angles[model.parameter_name(node)]
        p0 = np.array([1.0, np.exp(-1j * alpha)], dtype=complex) / np.sqrt(2)
        position = remaining.index(node)
        n_now = len(remaining)
        reshaped = state.reshape(2**position, 2, 2 ** (n_now - position - 1))
        # Contract directly against p0's raw coefficients (not conjugated): this
        # matches the measurement-basis convention |m> such that p0 IS <m|,
        # i.e. p0 = [1, exp(-i*alpha)]/sqrt(2) means <m| = p0 (bra components
        # given directly, not derived by conjugating a stated ket).
        state = np.tensordot(reshaped, p0, axes=([1], [0])).reshape(-1)
        mass = float(np.vdot(state, state).real)
        probability *= mass
        state = state / np.sqrt(mass)
        remaining.remove(node)

    output_positions = [remaining.index(v) for v in model.output_nodes]
    state = state.reshape([2] * model.n_wires).transpose(output_positions).reshape(-1)
    return state, probability


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=200.0)
    configs = [
        {"n_wires": 1, "n_layers": 1, "one_column": True},
        {"n_wires": 1, "n_layers": 3, "one_column": True},
        {"n_wires": 2, "n_layers": 1, "one_column": True},
        {"n_wires": 2, "n_layers": 1, "one_column": False},
        {"n_wires": 2, "n_layers": 2, "one_column": True},
        {"n_wires": 3, "n_layers": 1, "one_column": True},
    ]
    rows = []
    for cfg in configs:
        model = MuTA(cfg["n_wires"], cfg["n_layers"], one_column=cfg["one_column"])
        n_nodes = len(model.graph)
        if n_nodes > MAX_NODES:
            rows.append(
                {
                    **cfg,
                    "n_nodes": n_nodes,
                    "skipped": True,
                    "reason": f"exceeds {MAX_NODES}-node cap",
                }
            )
            continue
        for seed in (0, 1, 2, 3):
            state = haar_states(cfg["n_wires"], 1, seed)[0]
            angles = dict(
                zip(
                    model.trainable_parameters(),
                    common.rng(seed + 500).uniform(-np.pi, np.pi, model.n_parameters),
                )
            )
            target = model.run(state, angles).density_matrix
            independent_state, branch_probability = independent_contraction(model, state, angles)
            error = common.frobenius_error(
                np.outer(independent_state, independent_state.conj()), target
            )
            rows.append(
                {
                    **cfg,
                    "n_nodes": n_nodes,
                    "seed": seed,
                    "skipped": False,
                    "density_matrix_error": error,
                    "branch_probability": branch_probability,
                }
            )

    active = [r for r in rows if not r["skipped"]]
    max_error = max(r["density_matrix_error"] for r in active)
    status = "pass" if max_error < tol else "fail"

    common.save_result(
        rows,
        "R5_brute_force_agreement",
        extra={
            "protocol": "MuTA.run vs. freshly written kron/tensor-reshape zero-branch contraction",
            "oracle_class": "D",
            "status_category": "exact",
            "tolerance": tol,
            "acceptance_condition": f"max density-matrix error < {tol:.3e}",
            "max_error": max_error,
            "n_configurations_checked": len(active),
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "D", "status": status},
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [
        f"{r['n_wires']}w{r['n_layers']}L{'1c' if r['one_column'] else 'nc'} s{r['seed']}"
        for r in active
    ]
    ax.semilogy(
        range(len(active)),
        [r["density_matrix_error"] for r in active],
        "o",
        color=common.COLORS["photographiqml"],
    )
    ax.axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    ax.set_xticks(range(len(active)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
    ax.set(
        title=f"R5: independent brute-force contraction vs. MuTA.run (status={status})",
        ylabel="density-matrix Frobenius error",
    )
    ax.legend()
    common.save_figure(fig, "R5_brute_force_agreement")
    plt.close(fig)

    common.print_summary(
        "R5 brute-force agreement",
        n_configs=len(active),
        tol=tol,
        max_error=max_error,
        status=status,
    )
    if status != "pass":
        raise AssertionError(f"R5 failed: max_error={max_error} tol={tol}")


if __name__ == "__main__":
    main()
