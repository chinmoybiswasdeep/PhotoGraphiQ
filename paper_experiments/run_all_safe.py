"""Run the core manuscript experiments R1-R60 with explicit skip reasons.

Each experiment script is run as a fresh subprocess with the current Python
interpreter, so a crash or warning in one script cannot affect another.
A skip reason is recorded, never silently omitted, for every excluded script.
See EXPERIMENT_INDEX.md for the catalogue and STATUS.md for the latest results.

Usage:
    python run_all_safe.py             # light experiments (default configuration)
    python run_all_safe.py --smoke     # the quick subset used by pull-request CI
    python run_all_safe.py --all       # every experiment, publication configuration (--full)
    python run_all_safe.py --only R43 R51

Tiers:
    smoke  - seconds to a minute, no optional dependency beyond the dev extra
    safe   - light enough for the default run
    heavy  - large Fock cutoffs, long benchmarks or timing studies; run with --all
An experiment whose optional dependency (``needs``) is missing is recorded as
skipped with that reason; it is never reported as passed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

NATIVE = "photographiq._native"  # compiled engines; absent in a pure-Python install
PQ = ("piquasso", NATIVE)

# (script, tier, needs, reason for the heavy tier, accepts --full)
SCRIPTS = [
    ("01_core_mbqc/01_wire_teleportation.py", "safe", PQ, None, False),
    ("01_core_mbqc/02_identity_finite_squeezing.py", "smoke", (NATIVE,), None, False),
    ("02_compiler/03_rotation_compile.py", "smoke", (NATIVE,), None, False),
    ("02_compiler/04_squeezing_compile.py", "smoke", (NATIVE,), None, False),
    ("02_compiler/05_random_symplectic_compile.py", "safe", (NATIVE,), None, False),
    ("02_compiler/06_cz_compile.py", "smoke", (NATIVE,), None, False),
    ("02_compiler/07_beamsplitter_compile.py", "safe", (NATIVE,), None, False),
    ("02_compiler/08_multimode_showcase.py", "safe", (NATIVE,), None, False),
    ("02_compiler/09_compilation_trace.py", "smoke", (), None, False),
    ("02_compiler/10_compiler_scaling.py", "safe", (NATIVE,), None, False),
    ("03_adaptive_flow/11_adaptive_feedforward.py", "smoke", (NATIVE,), None, False),
    ("03_adaptive_flow/12_dependency_dag.py", "smoke", (NATIVE,), None, False),
    ("03_adaptive_flow/13_invalid_causality.py", "smoke", (NATIVE,), None, False),
    ("03_adaptive_flow/14_cvflow.py", "smoke", (NATIVE,), None, False),
    ("04_piquasso_validation/15_raw_piquasso_gaussian.py", "safe", PQ, None, False),
    ("04_piquasso_validation/16_numpy_analytic_reference.py", "safe", PQ, None, False),
    ("05_graphix_validation/17_graphix_structure.py", "safe", ("graphix", NATIVE), None, False),
    (
        "06_backend_crosschecks/18_gaussian_backends.py",
        "heavy",
        PQ,
        "four Fock engines incl. piquasso-mixed-fock, cutoff 14 (~3 min idle, ~15 min loaded)",
        False,
    ),
    ("07_non_gaussian/19_photon_subtraction_probability.py", "safe", PQ, None, False),
    ("07_non_gaussian/20_subtracted_squeezed_state.py", "safe", PQ, None, False),
    ("07_non_gaussian/21_cat_state.py", "safe", PQ, None, False),
    ("07_non_gaussian/22_cubic_phase_direct.py", "safe", PQ, None, False),
    (
        "07_non_gaussian/23_cubic_injection.py",
        "heavy",
        PQ,
        "36-cell cutoff/squeezing grid (~2.5 min)",
        False,
    ),
    ("07_non_gaussian/24_kerr_direct.py", "smoke", PQ, None, False),
    (
        "07_non_gaussian/25_kerr_synthesis.py",
        "heavy",
        PQ,
        "long benchmark (up to 64 synthesis steps)",
        False,
    ),
    ("08_gkp/26_gkp_resource.py", "safe", PQ, None, False),
    (
        "08_gkp/27_gkp_stabilizers.py",
        "heavy",
        PQ,
        "GKP readout integrals at cutoff 48 (~3 min)",
        False,
    ),
    (
        "08_gkp/28_gkp_syndrome.py",
        "heavy",
        PQ,
        "native-fock cutoff 80, ~10 min incl. 48 piquasso reference runs",
        False,
    ),
    ("09_noise/29_finite_squeezing.py", "safe", (NATIVE,), None, False),
    ("09_noise/30_loss_validation.py", "safe", PQ, None, False),
    ("09_noise/31_thermal_noise.py", "safe", PQ, None, False),
    (
        "09_noise/32_noisy_measurement.py",
        "heavy",
        PQ,
        "mixed-Fock noisy homodyne sampling (seconds per shot)",
        True,
    ),
    ("10_convergence/33_fock_cutoff.py", "safe", PQ, None, False),
    ("10_convergence/34_cubic_convergence.py", "safe", PQ, None, False),
    ("11_performance/35_fock_dimension_scaling.py", "safe", PQ, None, False),
    (
        "11_performance/36_gaussian_runtime.py",
        "heavy",
        PQ,
        "timing study (repeated measurements)",
        False,
    ),
    (
        "11_performance/37_compiler_runtime.py",
        "heavy",
        (NATIVE,),
        "timing study (repeated measurements)",
        False,
    ),
    (
        "11_performance/38_piquasso_overhead.py",
        "heavy",
        PQ,
        "timing study (repeated measurements)",
        False,
    ),
    ("12_serialization/39_serialization.py", "smoke", PQ, None, False),
    ("12_serialization/40_seeded_shots.py", "smoke", PQ, None, False),
    ("13_autodiff/41_jax_gradients.py", "safe", ("jax",) + PQ, None, False),
    (
        "13_autodiff/42_stochastic_gradients.py",
        "heavy",
        ("jax", NATIVE),
        "stochastic estimators (many samples)",
        False,
    ),
    ("14_native_gaussian/43_native_gaussian_validation.py", "safe", (NATIVE,), None, True),
    (
        "14_native_gaussian/44_native_gaussian_performance.py",
        "heavy",
        (),
        "long benchmark (timing, subprocess per thread count)",
        True,
    ),
    ("15_native_fock/45_native_fock_accuracy.py", "safe", ("piquasso", "mpmath"), None, True),
    ("15_native_fock/46_fock_space_optimization.py", "heavy", (), "long benchmark (timing)", True),
    (
        "15_native_fock/47_native_mixed_fock_validation.py",
        "heavy",
        ("piquasso",),
        "large Fock (mixed states, cutoff up to 48)",
        True,
    ),
    ("16_sampling/48_boson_sampling.py", "safe", ("piquasso",), None, True),
    ("16_sampling/49_gaussian_boson_sampling.py", "safe", ("piquasso",), None, True),
    (
        "16_sampling/50_threshold_gbs.py",
        "heavy",
        ("mpmath", "piquasso"),
        "long benchmark (multiprecision references)",
        True,
    ),
    ("17_flow_calculus/51_cv_flow_finding.py", "smoke", (), None, True),
    ("17_flow_calculus/52_cv_mbqc_determinism.py", "smoke", (), None, False),
    ("17_flow_calculus/53_qubit_flows_graphix.py", "safe", ("graphix",), None, True),
    ("17_flow_calculus/54_measurement_calculus.py", "safe", ("graphix",), None, True),
    ("17_flow_calculus/55_local_clifford.py", "smoke", (), None, False),
    ("17_flow_calculus/56_zx_calculus.py", "smoke", (), None, True),
    ("18_visualization/57_interactive_visualization.py", "safe", (), None, False),
    (
        "19_gpu/58_gpu_gaussian.py",
        "heavy",
        ("cupy",),
        "GPU timing study (needs a CUDA device)",
        True,
    ),
    (
        "20_photographiqml/59_photographiqml_physical.py",
        "heavy",
        ("piquasso",),
        "large Fock (physical GKP)",
        True,
    ),
    ("21_showcase/60_end_to_end_showcase.py", "safe", (), None, True),
]

assert all(len(entry) == 5 and entry[1] in {"smoke", "safe", "heavy"} for entry in SCRIPTS)


def experiment_id(path: str) -> str:
    return "R" + str(int(re.match(r"(\d+)_", Path(path).name).group(1)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="run only the smoke tier")
    group.add_argument(
        "--all", action="store_true", help="run every tier with --full where supported"
    )
    parser.add_argument("--only", nargs="+", metavar="ID", help="run only these experiment IDs")
    args = parser.parse_args(argv)
    tiers = (
        {"smoke"} if args.smoke else {"smoke", "safe", "heavy"} if args.all else {"smoke", "safe"}
    )

    results = []
    for relative_path, tier, needs, heavy_reason, accepts_full in SCRIPTS:
        rid = experiment_id(relative_path)
        record = {"id": rid, "script": relative_path, "tier": tier, "seconds": 0.0}
        missing = [m for m in needs if importlib.util.find_spec(m) is None]
        if args.only and rid not in args.only:
            continue
        if tier not in tiers and not args.only:
            reason = heavy_reason if tier == "heavy" else "not in the smoke tier"
            print(f"SKIP  {relative_path}  ({reason})")
            results.append({**record, "status": "skipped", "reason": reason})
            continue
        if missing:
            reason = f"optional dependency missing: {', '.join(missing)}"
            print(f"SKIP  {relative_path}  ({reason})")
            results.append({**record, "status": "skipped", "reason": reason})
            continue
        command = [sys.executable, str(ROOT / relative_path)]
        if args.all and accepts_full:
            command.append("--full")
        print(f"RUN   {relative_path} {' '.join(command[2:])}", flush=True)
        start = time.perf_counter()
        completed = subprocess.run(
            command, cwd=str((ROOT / relative_path).parent), capture_output=True, text=True
        )
        elapsed = time.perf_counter() - start
        status = "passed" if completed.returncode == 0 else "failed"
        print(f"  -> {status} in {elapsed:.1f}s")
        if completed.returncode != 0:
            print(completed.stdout[-2000:])
            print(completed.stderr[-2000:])
        results.append(
            {
                **record,
                "status": status,
                "seconds": round(elapsed, 1),
                "returncode": completed.returncode,
                "args": command[2:],
            }
        )

    n_passed = sum(r["status"] == "passed" for r in results)
    n_failed = sum(r["status"] == "failed" for r in results)
    n_skipped = sum(r["status"] == "skipped" for r in results)
    print("\n=== run_all_safe summary ===")
    print(
        f"  ran: {n_passed + n_failed}, passed: {n_passed}, failed: {n_failed}, skipped: {n_skipped}"
    )
    print(f"  total wall time: {sum(r['seconds'] for r in results):.1f}s")
    for r in results:
        if r["status"] == "failed":
            print(f"    FAILED {r['script']}")
    mode = "only" if args.only else "smoke" if args.smoke else "all" if args.all else "safe"
    summary = ROOT / "results" / "json" / f"run_all_{mode}_summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
