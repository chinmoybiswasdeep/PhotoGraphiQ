"""R4: Exhaustive adaptive-branch and byproduct-correction verification.

Scientific question: For every possible measurement outcome branch of a small
MuTA graph, does the full graph-state contraction with explicit Pauli
byproduct correction reproduce the same target output state that the
production frontier-translation algorithm (MuTA.run) computes deterministically
without ever branching, and do the branch probabilities form a valid
distribution?

Theory/equations: MBQC's deferred-measurement / adaptive-correction theorem
guarantees that for every one of the 2^m measurement branches (m = number of
measured nodes), applying the corresponding Pauli byproduct correction to the
post-measurement state reproduces the same logical output density matrix, and
the branch probabilities sum to 1 with none exceeding 1/2^m by more than
floating-point roundoff when angles are algebraically generic (each two-outcome
measurement bisects probability mass evenly is NOT generally true for nonzero
angles; only structural properties -- exact target agreement, sum to 1, all
nonnegative -- are the checked invariants here).

Functionality tested: photographiqml.validation.contract_branch (an
independent full graph-state contraction with explicit correction), compared
against MuTA.run's production frontier-translation execution.

Oracle and independence class: D (independent code path within the package --
contract_branch performs full Kronecker graph-state construction and
sequential projective measurement/correction, an algorithmically distinct
route from logical.execute's frontier-translation trick that never branches).

Exact/approximate/statistical status: exact, up to floating-point roundoff
(graphs capped at 16 nodes per contract_branch's own guard).

Primary metric: max density-matrix Frobenius error between every branch's
corrected output and MuTA.run's target, and |sum(branch probabilities) - 1|.

Declared acceptance condition: max branch error < tol
(tol = declare_tolerance(scale=1, safety_factor=100) to absorb the extra
floating-point operations of full graph-state contraction); probability-sum
error < tol; all branch probabilities >= -tol.

Expected cost: light (graphs capped at 10-16 nodes; at most 256 branches).

Manuscript destination: Appendix (independent-contraction cross-check
supporting Fig. 1 / Table II adaptive flow).

Scientific limitations: Exhaustive verification is only tractable up to
contract_branch's 16-node cap (one- and two-wire, one-layer models here);
larger models are not exhaustively checked this way (see R5 for a
statistically sampled, independently written contraction on larger models).
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import MuTA
from photographiqml.models import haar_states
from photographiqml.validation import contract_branch

EXPERIMENT_ID = "R4"


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=100.0)
    rows = []
    configs = [
        {"n_wires": 1, "one_column": True, "seeds": (0, 1, 2)},
        {"n_wires": 2, "one_column": True, "seeds": (0, 1, 2)},
    ]
    for cfg in configs:
        model = MuTA(cfg["n_wires"], 1, one_column=cfg["one_column"])
        n_measured = len(model.measurement_order)
        assert len(model.graph) <= 16, "contract_branch cap exceeded"
        for seed in cfg["seeds"]:
            state = haar_states(cfg["n_wires"], 1, seed)[0]
            parameters = model.initialize(seed + 1000, scale=1.5)
            target = model.run(state, parameters).density_matrix
            total_probability = 0.0
            max_branch_error = 0.0
            min_probability = np.inf
            for outcomes in itertools.product((0, 1), repeat=n_measured):
                output, probability = contract_branch(model, state, parameters, outcomes)
                branch_error = common.frobenius_error(np.outer(output, output.conj()), target)
                max_branch_error = max(max_branch_error, branch_error)
                total_probability += probability
                min_probability = min(min_probability, probability)
            rows.append(
                {
                    "n_wires": cfg["n_wires"],
                    "seed": seed,
                    "n_measured": n_measured,
                    "n_branches": 2**n_measured,
                    "max_branch_error": max_branch_error,
                    "probability_sum_error": abs(total_probability - 1.0),
                    "min_branch_probability": min_probability,
                }
            )

    max_branch_error = max(r["max_branch_error"] for r in rows)
    max_prob_error = max(r["probability_sum_error"] for r in rows)
    min_probability = min(r["min_branch_probability"] for r in rows)
    status = (
        "pass"
        if (max_branch_error < tol and max_prob_error < tol and min_probability > -tol)
        else "fail"
    )

    common.save_result(
        rows,
        "R4_adaptive_branches",
        extra={
            "protocol": "Exhaustive branch contraction (contract_branch) vs. MuTA.run target, 1- and 2-wire models",
            "oracle_class": "D",
            "status_category": "exact",
            "tolerance": tol,
            "acceptance_condition": f"max branch error and |sum(p)-1| < {tol:.3e}; min probability > -tol",
            "max_branch_error": max_branch_error,
            "max_probability_sum_error": max_prob_error,
            "min_branch_probability": min_probability,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "D", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    xs = range(len(rows))
    axes[0].semilogy(
        xs, [r["max_branch_error"] for r in rows], "o", color=common.COLORS["photographiqml"]
    )
    axes[0].axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    axes[0].set(
        title="Max corrected-branch error per config",
        xlabel="config index",
        ylabel="Frobenius error",
    )
    axes[0].legend()
    axes[1].bar(
        xs, [r["probability_sum_error"] for r in rows], color=common.COLORS["photographiqml"]
    )
    axes[1].axhline(tol, color=common.COLORS["acceptance"], linestyle="--")
    axes[1].set(title="|sum(branch probabilities) - 1|", xlabel="config index", ylabel="error")
    fig.suptitle(f"R4: Exhaustive adaptive-branch verification (status={status})")
    common.save_figure(fig, "R4_adaptive_branches")
    plt.close(fig)

    common.print_summary(
        "R4 adaptive branches",
        n_configs=len(rows),
        tol=tol,
        max_branch_error=max_branch_error,
        max_probability_sum_error=max_prob_error,
        status=status,
    )
    if status != "pass":
        raise AssertionError(f"R4 failed: branch={max_branch_error} prob={max_prob_error}")


if __name__ == "__main__":
    main()
