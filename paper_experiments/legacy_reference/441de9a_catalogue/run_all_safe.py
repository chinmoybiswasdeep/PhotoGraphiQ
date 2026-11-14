"""Run every lightweight experiment; skip expensive/optional-dependency/large-Fock/long-benchmark ones.

Each experiment script is run as a fresh subprocess with the current Python
interpreter, so a crash or warning in one script cannot affect another.
A skip reason is recorded, not silently omitted, for every excluded script.
See paper_experiments/EXPERIMENT_INDEX.md for the full R1-R42 catalogue and
paper_experiments/STATUS.md for the most recent full-suite run results.

Usage:
    python run_all_safe.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Every script, with an explicit skip reason where applicable (None = run it).
# Categories used for skipping, per the suite's stated policy:
#   "optional dependency" - requires graphix or jax
#   "large Fock"           - non-Gaussian workloads at cutoff >~20-96
#   "long benchmark"       - repeated-timing performance/benchmark scripts
SCRIPTS = [
    ("01_core_mbqc/01_wire_teleportation.py", None),
    ("01_core_mbqc/02_identity_finite_squeezing.py", None),
    ("02_compiler/03_rotation_compile.py", None),
    ("02_compiler/04_squeezing_compile.py", None),
    ("02_compiler/05_random_symplectic_compile.py", None),
    ("02_compiler/06_cz_compile.py", None),
    ("02_compiler/07_beamsplitter_compile.py", None),
    ("02_compiler/08_multimode_showcase.py", None),
    ("02_compiler/09_compilation_trace.py", None),
    ("02_compiler/10_compiler_scaling.py", None),
    ("03_adaptive_flow/11_adaptive_feedforward.py", None),
    ("03_adaptive_flow/12_dependency_dag.py", None),
    ("03_adaptive_flow/13_invalid_causality.py", None),
    ("03_adaptive_flow/14_cvflow.py", None),
    ("04_piquasso_validation/15_raw_piquasso_gaussian.py", None),
    ("04_piquasso_validation/16_numpy_analytic_reference.py", None),
    ("05_graphix_validation/17_graphix_structure.py", "optional dependency (graphix)"),
    ("06_backend_crosschecks/18_gaussian_backends.py", "large Fock (piquasso-mixed-fock, cutoff=14)"),
    ("07_non_gaussian/19_photon_subtraction_probability.py", None),
    ("07_non_gaussian/20_subtracted_squeezed_state.py", "large Fock (cutoff up to 36)"),
    ("07_non_gaussian/21_cat_state.py", "large Fock (cutoff=32)"),
    ("07_non_gaussian/22_cubic_phase_direct.py", "large Fock (cutoff up to 48)"),
    ("07_non_gaussian/23_cubic_injection.py", "large Fock (cutoff up to 48)"),
    ("07_non_gaussian/24_kerr_direct.py", None),
    ("07_non_gaussian/25_kerr_synthesis.py", "long benchmark (up to 64 synthesis steps)"),
    ("08_gkp/26_gkp_resource.py", "large Fock (cutoff up to 96)"),
    ("08_gkp/27_gkp_stabilizers.py", "large Fock (cutoff=48)"),
    ("08_gkp/28_gkp_syndrome.py", "large Fock + long benchmark (cutoff=80, ~9 minutes)"),
    ("09_noise/29_finite_squeezing.py", None),
    ("09_noise/30_loss_validation.py", None),
    ("09_noise/31_thermal_noise.py", None),
    ("09_noise/32_noisy_measurement.py", "long benchmark + large Fock (mixed-fock shots)"),
    ("10_convergence/33_fock_cutoff.py", "large Fock (cutoff up to 96)"),
    ("10_convergence/34_cubic_convergence.py", "large Fock (cutoff up to 96) + long benchmark (2D scan)"),
    ("11_performance/35_fock_dimension_scaling.py", None),
    ("11_performance/36_gaussian_runtime.py", "long benchmark (up to 200 modes, repeated timing)"),
    ("11_performance/37_compiler_runtime.py", "long benchmark (repeated timing)"),
    ("11_performance/38_piquasso_overhead.py", "long benchmark (repeated timing)"),
    ("12_serialization/39_serialization.py", None),
    ("12_serialization/40_seeded_shots.py", None),
    ("13_autodiff/41_jax_gradients.py", "optional dependency (jax)"),
    ("13_autodiff/42_stochastic_gradients.py", "optional dependency (jax) + long benchmark"),
]


def main():
    results = []
    for relative_path, skip_reason in SCRIPTS:
        script = ROOT / relative_path
        if skip_reason is not None:
            print(f"SKIP  {relative_path}  ({skip_reason})")
            results.append({"script": relative_path, "status": "skipped", "reason": skip_reason, "seconds": 0.0})
            continue
        print(f"RUN   {relative_path}", flush=True)
        start = time.perf_counter()
        completed = subprocess.run([sys.executable, str(script)], cwd=str(script.parent), capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        status = "passed" if completed.returncode == 0 else "FAILED"
        print(f"  -> {status} in {elapsed:.1f}s")
        if completed.returncode != 0:
            print(completed.stdout[-2000:])
            print(completed.stderr[-2000:])
        results.append({"script": relative_path, "status": status.lower(), "seconds": elapsed, "returncode": completed.returncode})

    n_run = sum(1 for r in results if r["status"] != "skipped")
    n_passed = sum(1 for r in results if r["status"] == "passed")
    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    total_seconds = sum(r["seconds"] for r in results)

    print("\n=== run_all_safe summary ===")
    print(f"  ran: {n_run}, passed: {n_passed}, failed: {n_failed}, skipped: {n_skipped}")
    print(f"  total wall time: {total_seconds:.1f}s")
    if n_failed:
        print("  FAILED scripts:")
        for r in results:
            if r["status"] == "failed":
                print(f"    - {r['script']}")

    import json

    (ROOT / "results" / "json" / "run_all_safe_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
