"""Aggregate individual experiment results into manuscript-ready summary tables.

Reads the JSON results already written by the R1-R42 scripts under
results/json/ and writes summary CSV + Markdown tables under tables/.
Run this AFTER running the experiments you want summarized; missing source
files are reported and skipped rather than silently producing an empty row.

Usage:
    python generate_tables.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JSON_DIR = ROOT / "results" / "json"
TABLES_DIR = ROOT / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load(name: str):
    path = JSON_DIR / f"{name}.json"
    if not path.exists():
        print(f"  (missing, skipped: {name}.json -- run the corresponding experiment first)")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(name: str, fieldnames: list[str], rows: list[dict]):
    csv_path = TABLES_DIR / f"{name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["| " + " | ".join(fieldnames) + " |", "|" + "|".join(["---"] * len(fieldnames)) + "|"]
    for row in rows:
        md_lines.append("| " + " | ".join(str(row.get(f, "")) for f in fieldnames) + " |")
    (TABLES_DIR / f"{name}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"  wrote {name}.csv / {name}.md ({len(rows)} rows)")


def compiler_accuracy():
    rows = []
    for exp_id, name, error_key in [
        ("R3", "R3_rotation_compile_theta_sweep", "map_frobenius_error"),
        ("R4", "R4_squeezing_compile", "map_frobenius_error"),
        ("R5", "R5_random_symplectic_compile", "frobenius_error"),
        ("R6", "R6_cz_compile", "map_frobenius_error"),
        ("R7", "R7_beamsplitter_compile", "map_frobenius_error"),
    ]:
        data = load(name)
        if data is None:
            continue
        values = [r[error_key] for r in data["rows"] if r.get(error_key) is not None]
        if not values:
            continue
        rows.append(
            {
                "experiment": exp_id,
                "gate": name.replace("_compile", "").replace(f"{exp_id}_", ""),
                "n_cases": len(values),
                "max_error": max(values),
                "median_error": sorted(values)[len(values) // 2],
            }
        )
    if rows:
        write_table("compiler_accuracy", ["experiment", "gate", "n_cases", "max_error", "median_error"], rows)


def backend_agreement():
    rows = []
    data = load("R18_gaussian_backends")
    if data is not None:
        vals = data["rows"]
        rows.append(
            {
                "comparison": "NumPy Gaussian vs. Piquasso Gaussian (exact)",
                "n_cases": len(vals),
                "max_mean_error": max(r["gaussian_vs_piquasso_mean_error"] for r in vals),
                "max_covariance_error": max(r["gaussian_vs_piquasso_covariance_error"] for r in vals),
            }
        )
        ok = [r for r in vals if r.get("mixed_fock_ok")]
        if ok:
            rows.append(
                {
                    "comparison": f"Gaussian vs. mixed-Fock (cutoff={data.get('fock_cutoff')}, approximation)",
                    "n_cases": len(ok),
                    "max_mean_error": None,
                    "max_covariance_error": None,
                }
            )
    if rows:
        write_table("backend_agreement", ["comparison", "n_cases", "max_mean_error", "max_covariance_error"], rows)


def piquasso_validation():
    rows = []
    data = load("R15_raw_piquasso_gaussian")
    if data is not None:
        by_gate: dict[str, list] = {}
        for r in data["rows"]:
            by_gate.setdefault(r["gate"], []).append(r)
        for gate, items in by_gate.items():
            rows.append(
                {
                    "source": "R15 (raw Piquasso)",
                    "gate": gate,
                    "n_cases": len(items),
                    "max_mean_error": max(i["mean_vector_norm_error"] for i in items),
                    "max_covariance_error": max(i["covariance_frobenius_error"] for i in items),
                }
            )
    data16 = load("R16_numpy_analytic_reference")
    if data16 is not None:
        for r in data16["rows"]:
            rows.append(
                {
                    "source": "R16 (independent NumPy analytic)",
                    "gate": r["gate"],
                    "n_cases": 1,
                    "max_mean_error": r["raw_piquasso_mean_error"],
                    "max_covariance_error": r["raw_piquasso_covariance_error"],
                }
            )
    if rows:
        write_table("piquasso_validation", ["source", "gate", "n_cases", "max_mean_error", "max_covariance_error"], rows)


def graphix_structure():
    data = load("R17_graphix_structure")
    if data is None:
        return
    rows = [
        {
            "topology": r["case"],
            "nodes": r["nodes"],
            "edges": r["edges"],
            "graph_isomorphic": r["graph_isomorphic"],
            "measurement_order_match": r["measurement_order_match"],
            "dependency_domains_match": r["dependency_domains_match"],
        }
        for r in data["rows"]
    ]
    write_table("graphix_structure", list(rows[0]), rows)


def non_gaussian_validation():
    rows = []
    data19 = load("R19_photon_subtraction_probability")
    if data19 is not None:
        vals = data19["rows"]
        rows.append({"experiment": "R19", "workload": "photon subtraction (binomial)", "n_cases": len(vals), "max_error": max(r["absolute_error"] for r in vals)})
    data21 = load("R21_cat_state")
    if data21 is not None:
        vals = data21["rows"]
        rows.append({"experiment": "R21", "workload": "cat state (independent expansion)", "n_cases": len(vals), "max_error": max(1 - r["fidelity_vs_independent_analytic"] for r in vals)})
    data22 = load("R22_cubic_phase_direct")
    if data22 is not None:
        vals = data22["rows"]
        rows.append({"experiment": "R22", "workload": "cubic phase (dense reference)", "n_cases": len(vals), "max_error": max(1 - r["fidelity_to_independent_reference"] for r in vals)})
    data24 = load("R24_kerr_direct")
    if data24 is not None:
        vals = data24["rows"]
        rows.append({"experiment": "R24", "workload": "Kerr (exact phase rule)", "n_cases": len(vals), "max_error": max(r["state_vector_error"] for r in vals)})
    if rows:
        write_table("non_gaussian_validation", ["experiment", "workload", "n_cases", "max_error"], rows)


def convergence_summary():
    rows = []
    data33 = load("R33_fock_cutoff")
    if data33 is not None:
        by_workload: dict[str, list] = {}
        for r in data33["rows"]:
            by_workload.setdefault(r["workload"], []).append(r)
        for workload, items in by_workload.items():
            items = sorted(items, key=lambda r: r["cutoff"])
            rows.append(
                {
                    "experiment": "R33", "workload": workload, "cutoffs_tested": len(items),
                    "smallest_cutoff": items[0]["cutoff"], "largest_cutoff": items[-1]["cutoff"],
                    "fidelity_at_second_largest_cutoff": items[-2]["fidelity_to_highest_cutoff"] if len(items) > 1 else None,
                }
            )
    data25 = load("R25_kerr_synthesis")
    if data25 is not None:
        vals = data25["rows"]
        rows.append(
            {
                "experiment": "R25", "workload": "Kerr synthesis (steps)", "cutoffs_tested": len(vals),
                "smallest_cutoff": vals[0]["synthesis_steps"], "largest_cutoff": vals[-1]["synthesis_steps"],
                "fidelity_at_second_largest_cutoff": vals[-2]["fidelity_to_exact_kerr"],
            }
        )
    if rows:
        write_table(
            "convergence_summary",
            ["experiment", "workload", "cutoffs_tested", "smallest_cutoff", "largest_cutoff", "fidelity_at_second_largest_cutoff"],
            rows,
        )


def performance_summary():
    rows = []
    data35 = load("R35_fock_dimension_scaling")
    if data35 is not None:
        rows.append({"experiment": "R35", "metric": "default max_dimension guard", "value": data35["default_max_dimension"]})
    data36 = load("R36_gaussian_runtime")
    if data36 is not None:
        for r in data36["rows"]:
            rows.append({"experiment": "R36", "metric": f"{r['backend']} @ {r['modes']} modes median runtime (s)", "value": r["median_seconds"]})
    data38 = load("R38_piquasso_overhead")
    if data38 is not None:
        for r in data38["rows"]:
            rows.append({"experiment": "R38", "metric": f"overhead ratio @ {r['modes']} modes", "value": r["overhead_ratio"]})
    if rows:
        write_table("performance_summary", ["experiment", "metric", "value"], rows)


def feature_status():
    """Transcribed from docs/validation/feature-matrix.md, cross-referenced with this suite's evidence."""
    rows = [
        {"feature": "Homodyne", "numpy_gaussian": "S", "piquasso_gaussian": "S", "pure_fock": "E", "mixed_fock": "E", "suite_evidence": "R1,R2,R11-R14,R19,R20"},
        {"feature": "Noisy homodyne", "numpy_gaussian": "S", "piquasso_gaussian": "S", "pure_fock": "-", "mixed_fock": "E", "suite_evidence": "R32"},
        {"feature": "CZ/rotation/squeezing", "numpy_gaussian": "S", "piquasso_gaussian": "S", "pure_fock": "E", "mixed_fock": "E", "suite_evidence": "R3,R4,R5,R6,R15,R16,R18"},
        {"feature": "Cubic/Kerr execution", "numpy_gaussian": "-", "piquasso_gaussian": "-", "pure_fock": "E", "mixed_fock": "E", "suite_evidence": "R22,R23,R24,R25"},
        {"feature": "Cat resources", "numpy_gaussian": "-", "piquasso_gaussian": "-", "pure_fock": "E", "mixed_fock": "E", "suite_evidence": "R20,R21,R33"},
        {"feature": "Photon subtraction", "numpy_gaussian": "-", "piquasso_gaussian": "-", "pure_fock": "E", "mixed_fock": "E", "suite_evidence": "R19,R20,R33"},
        {"feature": "Loss/thermal noise", "numpy_gaussian": "S", "piquasso_gaussian": "S", "pure_fock": "-", "mixed_fock": "E", "suite_evidence": "R30,R31"},
        {"feature": "GKP resources", "numpy_gaussian": "-", "piquasso_gaussian": "-", "pure_fock": "E", "mixed_fock": "E", "suite_evidence": "R26,R27,R28"},
        {"feature": "Gaussian compilation", "numpy_gaussian": "S", "piquasso_gaussian": "S", "pure_fock": "E", "mixed_fock": "E", "suite_evidence": "R3-R10,R29"},
        {"feature": "Cubic injection/Kerr synthesis", "numpy_gaussian": "-", "piquasso_gaussian": "-", "pure_fock": "E", "mixed_fock": "E", "suite_evidence": "R23,R25,R34"},
        {"feature": "Graphix structural comparison", "numpy_gaussian": "n/a", "piquasso_gaussian": "n/a", "pure_fock": "n/a", "mixed_fock": "n/a", "suite_evidence": "R17"},
        {"feature": "JAX autodiff/estimators", "numpy_gaussian": "n/a", "piquasso_gaussian": "n/a", "pure_fock": "n/a", "mixed_fock": "n/a", "suite_evidence": "R41,R42"},
    ]
    write_table("feature_status", list(rows[0]), rows)


def main():
    print("Generating manuscript summary tables...")
    compiler_accuracy()
    backend_agreement()
    piquasso_validation()
    graphix_structure()
    non_gaussian_validation()
    convergence_summary()
    performance_summary()
    feature_status()
    print(f"Done. See {TABLES_DIR}")


if __name__ == "__main__":
    sys.exit(main())
