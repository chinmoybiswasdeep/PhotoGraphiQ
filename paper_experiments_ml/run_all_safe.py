"""Run the light, deterministic subset of the experiment suite.

Each script runs as a fresh subprocess (isolated state, matches how a
reviewer would run it standalone). Heavy/physical-simulation experiments
are explicitly skipped with a reason (never silently omitted); run them via
`run_all_full.py`. Exits nonzero if any *executed* script fails; a skip is
never a failure.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (relative path, skip reason or None). Order matches R1..R48, R_PERF, R50..R59.
SCRIPTS = [
    ("01_logical_foundations/01_triangle_anatomy.py", None),
    ("01_logical_foundations/02_table_one_identities.py", None),
    ("01_logical_foundations/03_entangling_identity.py", None),
    ("01_logical_foundations/04_adaptive_branches.py", None),
    ("01_logical_foundations/05_brute_force_agreement.py", None),
    ("01_logical_foundations/06_batch_consistency.py", None),
    ("02_mentpy_validation/07_semantic_topology.py", None),
    ("02_mentpy_validation/08_flow_dependency_order.py", None),
    ("02_mentpy_validation/09_state_density_agreement.py", None),
    ("02_mentpy_validation/10_trainability_audit.py", None),
    ("02_mentpy_validation/11_kernel_gram_agreement.py", None),
    ("02_mentpy_validation/12_instrument_agreement.py", None),
    ("02_mentpy_validation/13_runtime_scaling.py", None),
    ("03_parameters_and_contracts/14_parameter_contracts.py", None),
    ("03_parameters_and_contracts/15_persistence_roundtrip.py", None),
    ("03_parameters_and_contracts/16_invalid_input_matrix.py", None),
    (
        "03_parameters_and_contracts/17_seeded_reproducibility.py",
        "moderate (includes one 32-shot physical run at cutoff=40)",
    ),
    ("04_gradients_and_training/18_parameter_shift_vs_fd.py", None),
    ("04_gradients_and_training/19_optimizer_convergence.py", None),
    ("04_gradients_and_training/20_haar_gate_learning.py", None),
    ("04_gradients_and_training/21_ising_gate_learning.py", None),
    (
        "04_gradients_and_training/22_gate_learning_sensitivity.py",
        "moderate (108 short training runs, ~1-2 min)",
    ),
    ("04_gradients_and_training/23_classifier_verification.py", None),
    ("04_gradients_and_training/24_regressor_verification.py", None),
    ("05_kernels_and_learning/25_kernel_feature_state.py", None),
    ("05_kernels_and_learning/26_kernel_properties.py", None),
    (
        "05_kernels_and_learning/27_kernel_classification.py",
        "moderate (24 SVM fits across 3 datasets, ~1 min)",
    ),
    ("05_kernels_and_learning/28_kernel_robustness.py", "moderate (~1 min)"),
    ("06_expressivity_and_diagnostics/29_instrument_diagnostics.py", None),
    ("06_expressivity_and_diagnostics/30_concurrence.py", None),
    ("06_expressivity_and_diagnostics/31_qfi.py", None),
    ("06_expressivity_and_diagnostics/32_lie_closure.py", None),
    ("06_expressivity_and_diagnostics/33_fisher_spectra.py", None),
    ("07_gkp_resources/34_gkp_codeword_projection.py", None),
    ("08_physical_lowering/35_capability_map.py", None),
    ("08_physical_lowering/36_lowering_preservation.py", None),
    ("08_physical_lowering/37_pauli_frame_validation.py", None),
    (
        "08_physical_lowering/38_public_pattern_comparison.py",
        "heavy (Fock simulation, independent Pattern construction)",
    ),
    ("09_piquasso_validation/39_raw_piquasso_state_prep.py", None),
    (
        "09_piquasso_validation/40_raw_piquasso_cz.py",
        "heavy (2-mode Fock simulation, cutoff up to 80)",
    ),
    (
        "09_piquasso_validation/41_conditional_execution_comparison.py",
        "heavy (4 independent Fock simulations)",
    ),
    (
        "09_piquasso_validation/42_abstraction_overhead.py",
        "heavy (repeated Fock simulations for timing)",
    ),
    (
        "10_physical_statistics/43_decoded_statistics.py",
        "heavy (1- and 2-wire physical-conditional Fock simulation)",
    ),
    (
        "10_physical_statistics/44_shot_convergence.py",
        "heavy (many physical-shots Fock simulations, several minutes)",
    ),
    (
        "10_physical_statistics/45_joint_readout_correlations.py",
        "heavy (2-wire physical-conditional Fock simulation)",
    ),
    ("10_physical_statistics/46_hard_vs_soft_decoding.py", None),
    (
        "11_convergence/47_resource_axis_convergence.py",
        "heavy (5 axes x up to 4 physical-conditional Fock simulations)",
    ),
    (
        "10_physical_statistics/48_physical_training_stability.py",
        "heavy (6 seeds x DiscreteSearch x physical shots, several minutes)",
    ),
    ("12_performance/49_aggregate_performance_figure.py", None),
    ("13_extended/50_canonical_reproduction.py", None),
    ("13_extended/51_expanded_gate_learning.py", "moderate (32 optimization runs)"),
    ("13_extended/52_comprehensive_classification.py", "moderate (repeated model fitting)"),
    ("13_extended/53_comprehensive_regression.py", "moderate (repeated model fitting)"),
    ("13_extended/54_kernel_ablations.py", "moderate (108 kernel/SVC fits)"),
    ("13_extended/55_expressivity_trainability.py", "moderate (QFI and optimization sweeps)"),
    ("13_extended/56_logical_physical_gap.py", "heavy (physical Fock simulations)"),
    ("13_extended/57_physical_training_evaluation.py", "heavy (fresh-shot physical training)"),
    ("13_extended/58_physical_resource_boundary.py", "heavy (physical resource sweep)"),
    ("13_extended/59_end_to_end_scaling.py", "moderate (end-to-end benchmarks)"),
]


def main():
    results = []
    start_all = time.perf_counter()
    for relative_path, skip_reason in SCRIPTS:
        script = ROOT / relative_path
        if skip_reason is not None:
            print(f"SKIP  {relative_path}: {skip_reason}")
            results.append(
                {
                    "script": relative_path,
                    "status": "skipped",
                    "reason": skip_reason,
                    "seconds": 0.0,
                }
            )
            continue
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(script)], cwd=str(script.parent), capture_output=True, text=True
        )
        seconds = time.perf_counter() - start
        if proc.returncode == 0:
            print(f"PASS  {relative_path} ({seconds:.1f}s)")
            results.append(
                {"script": relative_path, "status": "passed", "seconds": seconds, "returncode": 0}
            )
        else:
            print(f"FAIL  {relative_path} ({seconds:.1f}s)")
            print((proc.stdout + proc.stderr)[-2000:])
            results.append(
                {
                    "script": relative_path,
                    "status": "FAILED",
                    "seconds": seconds,
                    "returncode": proc.returncode,
                }
            )

    total_seconds = time.perf_counter() - start_all
    ran = sum(1 for r in results if r["status"] in ("passed", "FAILED"))
    passed = sum(1 for r in results if r["status"] == "passed")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(
        f"\n=== run_all_safe summary ===\nran={ran} passed={passed} failed={failed} skipped={skipped} total_wall_seconds={total_seconds:.1f}"
    )

    summary_path = ROOT / "results" / "json" / "run_all_safe_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "results": results,
                "ran": ran,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "total_wall_seconds": total_seconds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
