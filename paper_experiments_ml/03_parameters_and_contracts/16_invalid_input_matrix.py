"""R16: Invalid-input and fail-fast matrix.

Scientific question: Does every documented invalid input (non-normalized or
malformed logical states, nonfinite parameters, unsupported representations,
unsupported physical angles, bad seeds/modes/shots, infeasible resource
budgets, unsupported backends) fail fast with the correct exception type,
rather than silently producing a wrong answer?

Theory/equations: none (fail-fast contract verification across
logical.py/parameters.py/ansatz/muta.py/lowering.py/physical.py).

Functionality tested: input validation across the public API surface.

Oracle and independence class: E (structural/self-consistency -- each case's
"correct" behavior is that a declared invalid input raises, per the
package's own documented contract, not an external physics oracle).

Exact/approximate/statistical status: exact (discrete pass/fail per case).

Primary metric: number of cases where the expected exception was NOT raised.

Declared acceptance condition: 0 such cases.

Expected cost: light (every case is a single cheap call that is expected to
raise before any expensive computation).

Manuscript destination: Appendix (fail-fast capability matrix, Table III).

Scientific limitations: This enumerates a representative, not exhaustive,
set of invalid-input cases; it is not a formal verification of total input
coverage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import GKPPhysicalConfig, MuTA, PhysicalMuTA

EXPERIMENT_ID = "R16"


def case(name, fn, expected_exceptions):
    try:
        fn()
        return {
            "case": name,
            "expected": expected_exceptions.__name__
            if isinstance(expected_exceptions, type)
            else str(expected_exceptions),
            "raised": None,
            "passed": False,
        }
    except expected_exceptions as exc:
        return {
            "case": name,
            "expected": expected_exceptions.__name__
            if isinstance(expected_exceptions, type)
            else str(expected_exceptions),
            "raised": type(exc).__name__,
            "passed": True,
        }
    except Exception as exc:  # noqa: BLE001 -- deliberately catching to report a wrong exception type as a failed case
        return {
            "case": name,
            "expected": expected_exceptions.__name__
            if isinstance(expected_exceptions, type)
            else str(expected_exceptions),
            "raised": type(exc).__name__,
            "passed": False,
        }


def main():
    plt = common.setup_style()
    model = MuTA(2, 1, one_column=True)
    physical_config = GKPPhysicalConfig(
        cutoff=16, peak_width=0.9, envelope=0.9, peaks=3, grid_points=513
    )
    physical_model = PhysicalMuTA(1, physical_config=physical_config)

    cases = [
        ("non_normalized_state", lambda: model.run([1, 1, 0, 0]), ValueError),
        ("wrong_shape_state", lambda: model.run([1, 0, 0]), ValueError),
        ("nonfinite_state", lambda: model.run([np.nan, 0, 0, 0]), ValueError),
        (
            "nonfinite_parameter",
            lambda: model.run([1, 0, 0, 0], {"alpha.w0.c0": np.inf}),
            ValueError,
        ),
        (
            "unknown_parameter_name",
            lambda: model.run([1, 0, 0, 0], {"alpha.w9.c9": 0.1}),
            ValueError,
        ),
        (
            "wrong_length_vector_parameters",
            lambda: model.run([1, 0, 0, 0], np.zeros(model.n_parameters + 1)),
            ValueError,
        ),
        ("unsupported_representation", lambda: MuTA(1, representation="cv"), ValueError),
        ("bad_n_wires_zero", lambda: MuTA(0), ValueError),
        ("bad_n_wires_bool", lambda: MuTA(True), ValueError),
        ("custom_connections_without_one_column", lambda: MuTA(2, connections=(1,)), ValueError),
        (
            "legacy_gkp_execution_rejected",
            lambda: MuTA(1, representation="gkp-resource").run([1, 0]),
            NotImplementedError,
        ),
        (
            "unsupported_physical_angle",
            lambda: physical_model.run([1, 0], parameters={"alpha.w0.c0": 0.37}),
            NotImplementedError,
        ),
        (
            "unsupported_output_basis",
            lambda: physical_model.run([1, 0], output_basis="Y"),
            NotImplementedError,
        ),
        ("bad_shots_zero", lambda: physical_model.run([1, 0], shots=0), ValueError),
        (
            "bad_seed_negative",
            lambda: physical_model.run(
                [1, 0],
                shots=1,
                seed=-1,
                mode="physical-conditional",
                analog_outcomes=dict.fromkeys(physical_model.measurement_order, 0.0),
            ),
            ValueError,
        ),
        ("bad_mode", lambda: physical_model.run([1, 0], mode="logical-exact"), ValueError),
        (
            "conditional_requires_all_outcomes",
            lambda: physical_model.run([1, 0], mode="physical-conditional", analog_outcomes={}),
            ValueError,
        ),
        (
            "conditional_requires_shots_one",
            lambda: physical_model.run(
                [1, 0],
                mode="physical-conditional",
                shots=2,
                analog_outcomes=dict.fromkeys(physical_model.measurement_order, 0.0),
            ),
            ValueError,
        ),
        ("unsupported_backend", lambda: GKPPhysicalConfig(backend="strawberryfields"), ValueError),
        (
            "unsupported_decoder",
            lambda: GKPPhysicalConfig(decoder="maximum-likelihood"),
            ValueError,
        ),
        (
            "infeasible_dimension_budget",
            lambda: physical_model.run(
                [1, 0],
                config=GKPPhysicalConfig(
                    cutoff=16,
                    peak_width=0.9,
                    envelope=0.9,
                    peaks=3,
                    grid_points=513,
                    max_dimension=1,
                ),
            ),
            (ValueError, Exception),
        ),
        (
            "physical_measurement_family_unsupported",
            lambda: PhysicalMuTA(1, measurement_family="Z"),
            ValueError,
        ),
        (
            "physical_continuous_init_rejected",
            lambda: physical_model.initialize(seed=0, scale=0.1),
            ValueError,
        ),
        (
            "physical_run_batch_unsupported",
            lambda: physical_model.run_batch([[1, 0]]),
            NotImplementedError,
        ),
        ("physical_unitary_unsupported", lambda: physical_model.unitary(), NotImplementedError),
        ("dense_unitary_too_many_wires", lambda: MuTA(11).unitary(), ValueError),
        ("initialize_bad_scale", lambda: model.initialize(scale=-1.0), ValueError),
        ("freeze_unknown_name", lambda: model.freeze("alpha.w9.c9"), ValueError),
        (
            "freeze_nonfinite_value",
            lambda: model.freeze(list(model.parameters())[0], np.inf),
            ValueError,
        ),
    ]

    rows = [case(name, fn, expected) for name, fn, expected in cases]
    n_failed = sum(1 for r in rows if not r["passed"])
    status = "pass" if n_failed == 0 else "fail"

    common.save_result(
        rows,
        "R16_invalid_input_matrix",
        extra={
            "protocol": "Fail-fast matrix across public API validation paths",
            "oracle_class": "E",
            "status_category": "exact",
            "acceptance_condition": "0 cases where the expected exception was not raised",
            "n_cases": len(rows),
            "n_failed": n_failed,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "E", "status": status},
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = [
        common.COLORS["photographiqml"] if r["passed"] else common.COLORS["piquasso"] for r in rows
    ]
    ax.barh(range(len(rows)), [1] * len(rows), color=colors)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["case"] for r in rows], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_title(f"R16: fail-fast invalid-input matrix (status={status}, {n_failed} failed)")
    common.save_figure(fig, "R16_invalid_input_matrix")
    plt.close(fig)

    common.print_summary(
        "R16 invalid-input matrix", n_cases=len(rows), n_failed=n_failed, status=status
    )
    if status != "pass":
        raise AssertionError(f"R16 failed cases: {[r['case'] for r in rows if not r['passed']]}")


if __name__ == "__main__":
    main()
