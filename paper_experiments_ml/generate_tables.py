"""Generate CSV + Markdown summary tables from already-saved experiment JSON.

Reads only `results/json/<name>.json` files that experiment scripts already
wrote; never recomputes or fabricates a result. If a source file is
missing, that table's rows are skipped with a printed note, not silently
invented. Run after `run_all_safe.py` / `run_all_full.py`, or any time the
saved results change.
"""

import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JSON_DIR, TABLES  # noqa: E402
from publication import atomic_write_text  # noqa: E402


def load(name):
    path = JSON_DIR / f"{name}.json"
    if not path.exists():
        print(f"  skip: {name}.json not found")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(name, fieldnames, rows):
    csv_path = TABLES / f"{name}.csv"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    atomic_write_text(csv_path, buffer.getvalue())
    md_lines = [f"| {' | '.join(fieldnames)} |", f"|{'|'.join(['---'] * len(fieldnames))}|"]
    for row in rows:
        md_lines.append(f"| {' | '.join(str(row.get(f, '')) for f in fieldnames)} |")
    atomic_write_text(TABLES / f"{name}.md", "\n".join(md_lines) + "\n")
    print(f"  wrote {name}.csv / {name}.md ({len(rows)} rows)")


def logical_mentpy_agreement():
    rows = []
    for name, metric_key in [
        ("R7_semantic_topology", "max_symmetric_difference"),
        ("R8_flow_dependency_order", "max_order_violations"),
        ("R9_state_density_agreement", "max_error"),
        ("R11_kernel_gram_agreement", "max_state_error"),
        ("R12_instrument_agreement", "max_probability_error"),
    ]:
        data = load(name)
        if data is None:
            continue
        rows.append(
            {
                "experiment": name,
                "metric": metric_key,
                "value": data.get(metric_key),
                "status": data.get("status"),
            }
        )
    write_table("logical_mentpy_agreement", ["experiment", "metric", "value", "status"], rows)


def piquasso_physical_agreement():
    rows = []
    for name in [
        "R38_public_pattern_comparison",
        "R39_raw_piquasso_state_prep",
        "R40_raw_piquasso_cz",
        "R41_conditional_execution_comparison",
    ]:
        data = load(name)
        if data is None:
            continue
        rows.append(
            {
                "experiment": name,
                "oracle_class": data.get("oracle_class"),
                "max_error": data.get("max_error") or data.get("max_infidelity"),
                "status": data.get("status"),
            }
        )
    write_table(
        "piquasso_physical_agreement", ["experiment", "oracle_class", "max_error", "status"], rows
    )


def kernel_and_learning():
    rows = []
    for name in [
        "R20_haar_gate_learning",
        "R21_ising_gate_learning",
        "R23_classifier_verification",
        "R24_regressor_verification",
        "R26_kernel_properties",
        "R27_kernel_classification",
    ]:
        data = load(name)
        if data is None:
            continue
        rows.append(
            {
                "experiment": name,
                "status_category": data.get("status_category"),
                "status": data.get("status"),
            }
        )
    write_table("kernel_and_learning", ["experiment", "status_category", "status"], rows)


def gradient_and_training():
    rows = []
    for name in [
        "R18_parameter_shift_vs_fd",
        "R19_optimizer_convergence",
        "R22_gate_learning_sensitivity",
    ]:
        data = load(name)
        if data is None:
            continue
        rows.append({"experiment": name, "status": data.get("status")})
    write_table("gradient_and_training", ["experiment", "status"], rows)


def gkp_convergence():
    rows = []
    data = load("R47_resource_axis_convergence")
    if data is not None:
        for row in data.get("rows", []):
            rows.append(
                {
                    "axis": row["axis"],
                    "axis_type": row["axis_type"],
                    "value": row["value"],
                    "max_probability_delta": row["max_probability_delta"],
                }
            )
    write_table("gkp_convergence", ["axis", "axis_type", "value", "max_probability_delta"], rows)


def capability_matrix():
    data = load("R35_capability_map")
    rows = []
    if data is not None:
        for row in data.get("rows", []):
            rows.append(
                {
                    "case": row["center"],
                    "angle": row["angle"],
                    "supported": row["actual_supported"],
                    "correct": row["correct"],
                }
            )
    write_table("capability_matrix", ["case", "angle", "supported", "correct"], rows)


def performance_summary():
    rows = []
    r13 = load("R13_runtime_scaling")
    if r13:
        for row in r13.get("rows", []):
            rows.append(
                {
                    "source": "R13",
                    "config": f"n_wires={row['n_wires']}",
                    "metric": "photographiqml_median_seconds",
                    "value": row["photographiqml_median_seconds"],
                }
            )
    r42 = load("R42_abstraction_overhead")
    if r42:
        for row in r42.get("rows", []):
            rows.append(
                {
                    "source": "R42",
                    "config": row["pipeline"],
                    "metric": "median_seconds",
                    "value": row["median_seconds"],
                }
            )
    write_table("performance_summary", ["source", "config", "metric", "value"], rows)


def feature_status():
    """Hand-authored capability matrix transcribed from docs/physical/supported-measurements.md."""
    rows = [
        {"feature": "Logical X/XY(0)", "logical": "Yes", "physical": "Yes"},
        {"feature": "Logical XY(pi)", "logical": "Yes", "physical": "Yes (flipped bit)"},
        {"feature": "Logical Z (output/instrument)", "logical": "Yes", "physical": "Output only"},
        {"feature": "Logical Y / XY(pi/2)", "logical": "Yes", "physical": "Unsupported"},
        {"feature": "XY(pi/4)", "logical": "Yes", "physical": "Unsupported"},
        {"feature": "Arbitrary XY", "logical": "Yes", "physical": "Unsupported"},
        {
            "feature": "Continuous logical training",
            "logical": "Yes (Adam/SGD/L-BFGS)",
            "physical": "No (categorical DiscreteSearch only)",
        },
        {"feature": "Eq. 5 kernel", "logical": "Yes", "physical": "Unsupported"},
        {"feature": "QuantumInstrumentModel", "logical": "Yes", "physical": "Unsupported"},
        {
            "feature": "Soft decoder for flow-based execution",
            "logical": "n/a",
            "physical": "Rejected before allocation",
        },
    ]
    write_table("feature_status", ["feature", "logical", "physical"], rows)


def main():
    print("Generating tables from saved results...")
    logical_mentpy_agreement()
    piquasso_physical_agreement()
    kernel_and_learning()
    gradient_and_training()
    gkp_convergence()
    capability_matrix()
    performance_summary()
    feature_status()
    print("Done. See tables/.")


if __name__ == "__main__":
    main()
