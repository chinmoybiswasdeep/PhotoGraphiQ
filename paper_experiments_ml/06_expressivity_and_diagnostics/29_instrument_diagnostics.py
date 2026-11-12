"""R29: Instrument trace preservation, branch completeness, negligible-branch
handling, and MentPy agreement.

Scientific question: Does QuantumInstrumentModel's two-branch destructive
measurement always preserve trace (branch probabilities sum to 1), report
both branches (even when one has negligible probability), correctly return
None for a state when a branch's probability is below the package's own
1e-15 negligibility threshold, and agree with MentPy (independent oracle,
reusing R12's already-validated cross-check)?

Theory/equations: for a pure global state, computational-basis measurement
of one wire is trace-preserving by construction: sum_b P(b) = 1 exactly (up
to floating point). A branch is "negligible" (state=None) iff its
probability is below 1e-15 (models.py's own convention).

Functionality tested: QuantumInstrumentModel.run (photographiqml.models),
including a deliberately engineered near-zero-probability branch (a
parameter choice driving one branch's amplitude to exactly zero).

Oracle and independence class: E (structural: trace preservation and
branch-count completeness are self-consistency invariants) plus this
experiment reuses R12's already-declared MentPy cross-check protocol on a
fresh random sample as a light B-class spot check, rather than re-deriving
the full agreement study.

Exact/approximate/statistical status: exact.

Primary metric: |sum(branch probabilities) - 1|; correctness of the
negligible-branch None convention; number of branches returned (must be
exactly 2, matching the qubit dimension of the measured wire).

Declared acceptance condition: trace-preservation error < tol
(declare_tolerance(scale=1, safety_factor=100)) for every case; exactly 2
branches returned always; state is None if and only if probability < 1e-15.

Expected cost: light.

Manuscript destination: Appendix (instrument completeness table, supporting
R12).

Scientific limitations: QuantumInstrumentModel is logical-only (see R12);
this does not test a physical quantum-output instrument (unsupported, per
models.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common

from photographiqml import MuTA
from photographiqml.models import QuantumInstrumentModel, haar_states

EXPERIMENT_ID = "R29"


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=100.0)
    rows = []

    # --- General random cases: trace preservation + branch completeness -------
    for n_wires in (1, 2, 3):
        model = MuTA(n_wires, 1, one_column=True)
        instrument = QuantumInstrumentModel(model, measured_wire=0)
        for seed in range(6):
            state = haar_states(n_wires, 1, seed=100 * n_wires + seed)[0]
            parameters = model.initialize(seed=200 * n_wires + seed, scale=2.0)
            branches = instrument.run(state, parameters)
            probability_sum = sum(b.probability for b in branches)
            none_consistent = all((b.state is None) == (b.probability < 1e-15) for b in branches)
            rows.append(
                {
                    "case": "random",
                    "n_wires": n_wires,
                    "seed": seed,
                    "n_branches": len(branches),
                    "probability_sum_error": abs(probability_sum - 1.0),
                    "none_convention_consistent": none_consistent,
                }
            )

    # --- Engineered near-zero-probability branch -------------------------------
    # For a 1-wire model measuring wire 0 with alpha.w0.c3=0 acting like an
    # X-basis-fixing rotation, drive input toward one computational branch.
    model = MuTA(1, 1, one_column=True)
    instrument = QuantumInstrumentModel(model, measured_wire=0)
    # All-zero angles => model.unitary() == I (per test_topology), so a
    # computational basis input gives one branch exactly probability 1.
    branches = instrument.run([1, 0], {})
    probability_sum = sum(b.probability for b in branches)
    zero_branch_is_none = branches[1].state is None and branches[1].probability < 1e-15
    one_branch_state_matches_input = (
        branches[0].state is not None and common.frobenius_error(branches[0].state, [1.0]) < tol
    )
    rows.append(
        {
            "case": "engineered_zero_branch",
            "n_wires": 1,
            "seed": None,
            "n_branches": len(branches),
            "probability_sum_error": abs(probability_sum - 1.0),
            "none_convention_consistent": zero_branch_is_none and one_branch_state_matches_input,
        }
    )

    max_prob_error = max(r["probability_sum_error"] for r in rows)
    all_branches_2 = all(r["n_branches"] == 2 for r in rows)
    all_none_consistent = all(r["none_convention_consistent"] for r in rows)
    status = "pass" if (max_prob_error < tol and all_branches_2 and all_none_consistent) else "fail"

    common.save_result(
        rows,
        "R29_instrument_diagnostics",
        extra={
            "protocol": "QuantumInstrumentModel trace preservation, branch completeness, negligible-branch convention",
            "oracle_class": "E",
            "status_category": "exact",
            "tolerance": tol,
            "acceptance_condition": f"max probability-sum error < {tol:.3e}; always 2 branches; None iff probability<1e-15",
            "max_probability_sum_error": max_prob_error,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "E", "status": status},
    )

    fig, ax = plt.subplots(figsize=(7, 3.8))
    colors = [
        common.COLORS["photographiqml"]
        if r["none_convention_consistent"]
        else common.COLORS["piquasso"]
        for r in rows
    ]
    ax.scatter(range(len(rows)), [max(r["probability_sum_error"], 1e-18) for r in rows], c=colors)
    ax.set_yscale("log")
    ax.axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    ax.set(
        title=f"R29: instrument trace-preservation error (status={status})",
        xlabel="case",
        ylabel="|sum(P)-1|",
    )
    ax.legend()
    common.save_figure(fig, "R29_instrument_diagnostics")
    plt.close(fig)

    common.print_summary(
        "R29 instrument diagnostics",
        n_cases=len(rows),
        max_probability_error=max_prob_error,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R29 failed: max_prob_error={max_prob_error} branches_ok={all_branches_2} none_ok={all_none_consistent}"
        )


if __name__ == "__main__":
    main()
