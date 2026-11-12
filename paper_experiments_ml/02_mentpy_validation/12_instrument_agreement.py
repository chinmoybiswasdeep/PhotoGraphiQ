"""R12: Quantum-instrument branch probabilities and conditional states against
MentPy.

Scientific question: Does QuantumInstrumentModel.run's destructive
computational-basis measurement of one wire (branch probabilities and
conditional post-measurement states) agree with what one gets by
independently projecting/reducing an *independently executed MentPy* output
density matrix, rather than PhotoGraphiQML's own state?

Theory/equations: for a pure global state rho = |psi><psi| on n wires,
measuring wire w in the computational basis gives P(b) = Tr(Pi_w^b rho) and a
pure conditional state on the remaining n-1 wires, obtained here by directly
slicing the reshaped density-matrix tensor at ket/bra index b on axis w (a
projection-and-read, mathematically identical to a partial trace of the
projected operator since the sliced axes are fixed scalars, not summed).

Functionality tested: QuantumInstrumentModel.run (photographiqml.models) vs.
an independent projection of the MentPy-executed density matrix (built via
the same small mentpy_density_matrix helper used in R11, since
validation.compare_mentpy exposes only a scalar error).

Oracle and independence class: B (independent external implementation) for
the underlying state; the branch projection itself is E (structural,
self-consistent quantum mechanics applied fresh to that independent state).

Exact/approximate/statistical status: exact, up to floating-point roundoff.

Primary metric: max branch-probability error; max conditional-state
density-matrix error (skipped for branches below the 1e-15 negligible-branch
threshold, matching QuantumInstrumentModel's own convention).

Declared acceptance condition: both max errors < tol
(tol = declare_tolerance(scale=1, safety_factor=1000)); trace-preservation
(sum of branch probabilities == 1) holds for both PhotoGraphiQML and the
independent MentPy-based computation.

Expected cost: light.

Manuscript destination: Appendix (instrument/branch agreement supporting
Fig. 4/Table II).

Scientific limitations: QuantumInstrumentModel requires representation=logical
only; no physical quantum-output instrument exists (models.py), so this
validates the ideal logical instrument alone.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import MuTA
from photographiqml.models import QuantumInstrumentModel, haar_states
from photographiqml.validation import MENTPY_COMMIT, mentpy_reference

EXPERIMENT_ID = "R12"
NEGLIGIBLE = 1e-15


def mentpy_density_matrix(model, input_state, parameters):
    import mentpy as mp

    reference, mapping = mentpy_reference(model, fix_measurements=True)
    angles = model._parameters.bind(parameters)
    values = [angles[model.parameter_name(mapping[v])] for v in reference.trainable_nodes]
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
    return simulator.run(values, output_form="dm")


def project_wire(rho, wire, bit, n_wires):
    full = rho.reshape([2] * n_wires + [2] * n_wires)
    index = [slice(None)] * (2 * n_wires)
    index[wire], index[n_wires + wire] = bit, bit
    block = full[tuple(index)].reshape(2 ** (n_wires - 1), 2 ** (n_wires - 1))
    probability = float(np.trace(block).real)
    return probability, block


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=1000.0)
    rows = []
    for n_wires in (1, 2, 3):
        model = MuTA(n_wires, 1, one_column=True)
        instrument = QuantumInstrumentModel(model, measured_wire=0)
        for seed in range(4):
            state = haar_states(n_wires, 1, seed=n_wires * 10 + seed)[0]
            parameters = model.initialize(seed=n_wires * 20 + seed, scale=2.0)
            branches = instrument.run(state, parameters)
            mentpy_rho = mentpy_density_matrix(model, state, parameters)
            prob_sum_pqml, prob_sum_mentpy = 0.0, 0.0
            max_prob_error, max_state_error = 0.0, 0.0
            for branch in branches:
                mp_probability, mp_block = project_wire(mentpy_rho, 0, branch.outcome, n_wires)
                prob_sum_pqml += branch.probability
                prob_sum_mentpy += mp_probability
                max_prob_error = max(max_prob_error, abs(branch.probability - mp_probability))
                if branch.state is not None and mp_probability > NEGLIGIBLE:
                    pqml_conditional = np.outer(branch.state, branch.state.conj())
                    mp_conditional = mp_block / mp_probability
                    max_state_error = max(
                        max_state_error, common.frobenius_error(pqml_conditional, mp_conditional)
                    )
            rows.append(
                {
                    "n_wires": n_wires,
                    "seed": seed,
                    "max_probability_error": max_prob_error,
                    "max_conditional_state_error": max_state_error,
                    "photographiqml_probability_sum_error": abs(prob_sum_pqml - 1.0),
                    "mentpy_probability_sum_error": abs(prob_sum_mentpy - 1.0),
                }
            )

    max_prob_error = max(r["max_probability_error"] for r in rows)
    max_state_error = max(r["max_conditional_state_error"] for r in rows)
    max_trace_error = max(
        max(r["photographiqml_probability_sum_error"], r["mentpy_probability_sum_error"])
        for r in rows
    )
    status = "pass" if max(max_prob_error, max_state_error, max_trace_error) < tol else "fail"

    common.save_result(
        rows,
        "R12_instrument_agreement",
        extra={
            "protocol": "QuantumInstrumentModel vs. independent projection of MentPy-executed density matrix",
            "oracle_class": "B",
            "status_category": "exact",
            "mentpy_commit": MENTPY_COMMIT,
            "tolerance": tol,
            "acceptance_condition": f"max branch-probability error, max conditional-state error and trace-preservation error all < {tol:.3e}",
            "max_probability_error": max_prob_error,
            "max_conditional_state_error": max_state_error,
            "max_trace_preservation_error": max_trace_error,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "B", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].semilogy(
        [r["max_probability_error"] for r in rows],
        "o-",
        color=common.COLORS["photographiqml"],
        label="probability",
    )
    axes[0].semilogy(
        [r["max_conditional_state_error"] for r in rows],
        "s--",
        color=common.COLORS["mentpy"],
        label="conditional state",
    )
    axes[0].axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    axes[0].set(
        title="Branch agreement vs. MentPy projection", xlabel="config x seed", ylabel="error"
    )
    axes[0].legend(fontsize=7)
    axes[1].semilogy(
        [r["photographiqml_probability_sum_error"] for r in rows],
        "o",
        color=common.COLORS["photographiqml"],
        label="PhotoGraphiQML",
    )
    axes[1].semilogy(
        [r["mentpy_probability_sum_error"] for r in rows],
        "x",
        color=common.COLORS["mentpy"],
        label="MentPy-derived",
    )
    axes[1].axhline(tol, color=common.COLORS["acceptance"], linestyle="--")
    axes[1].set(title="Trace preservation |sum(P)-1|", xlabel="config x seed", ylabel="error")
    axes[1].legend(fontsize=7)
    fig.suptitle(f"R12: instrument branch agreement (status={status})")
    common.save_figure(fig, "R12_instrument_agreement")
    plt.close(fig)

    common.print_summary(
        "R12 instrument agreement",
        n_cases=len(rows),
        tol=tol,
        max_probability_error=max_prob_error,
        max_conditional_state_error=max_state_error,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R12 failed: prob={max_prob_error} state={max_state_error} trace={max_trace_error}"
        )


if __name__ == "__main__":
    main()
