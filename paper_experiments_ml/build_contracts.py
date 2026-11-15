"""Build the canonical machine-readable contracts from index declarations.

This is a source-definition generator, not a result generator.  Its output is
checked in before a publication run and consequently participates in the
source-tree fingerprint.
"""

from __future__ import annotations

import ast
from pathlib import Path

from publication import EXPECTED_IDS, ROOT, atomic_write_json, index_rows, required_artifacts

OUT = ROOT / "EXPERIMENT_CONTRACTS.json"

EXTENDED = {
    "R20": {"seed_set": list(range(20)), "repetitions": 20},
    "R21": {"seed_set": list(range(20)), "repetitions": 20},
    "R22": {"seed_set": list(range(6)), "repetitions": 6},
    "R23": {"seed_set": list(range(5)), "repetitions": 5},
    "R24": {"seed_set": list(range(5)), "repetitions": 5},
    "R27": {"seed_set": list(range(8)), "repetitions": 8},
    "R28": {"seed_set": list(range(6)), "repetitions": 6},
    "R44": {
        "seed_set": [2026, 999, *range(1000, 1008)],
        "repetitions": 10,
        "uncertainty_method": "Rao-Blackwell/empirical standard errors and eight-seed coverage",
    },
    "R48": {
        "seed_set": [7, 19, 23, 31, 42, 101],
        "repetitions": 6,
        "uncertainty_method": "six-seed empirical instability rate",
    },
    "R50": {
        "seed_set": [3, 11, 29, 47],
        "repetitions": 4,
        "preprocessing": "none",
        "hyperparameter_selection_protocol": "none",
        "uncertainty_method": "percentile bootstrap, 2000 resamples",
    },
    "R51": {
        "seed_set": [2, 7, 19, 31],
        "repetitions": 4,
        "preprocessing": "none",
        "hyperparameter_selection_protocol": "predeclared optimizer/depth grid; no test selection",
        "uncertainty_method": "percentile bootstrap, 2000 resamples",
    },
    "R52": {
        "seed_set": [5, 13, 23, 41, 59],
        "repetitions": 5,
        "preprocessing": "StandardScaler fit on training fold only",
        "hyperparameter_selection_protocol": "kernel C selected on validation fold only; test fold isolated",
        "uncertainty_method": "percentile bootstrap, 2000 resamples",
    },
    "R53": {
        "seed_set": [3, 17, 37, 67],
        "repetitions": 4,
        "preprocessing": "StandardScaler fit on training fold only",
        "hyperparameter_selection_protocol": "predeclared model settings; test fold/domain isolated",
        "uncertainty_method": "percentile bootstrap, 2000 resamples",
    },
    "R54": {
        "seed_set": [7, 19, 43, 71],
        "repetitions": 4,
        "preprocessing": "none; input angle scale is an explicit ablation axis",
        "hyperparameter_selection_protocol": "all C values reported; none selected on test results",
        "uncertainty_method": "percentile bootstrap, 2000 resamples",
    },
    "R55": {
        "seed_set": [2, 11, 29, 53, 83],
        "repetitions": 5,
        "preprocessing": "none",
        "hyperparameter_selection_protocol": "predeclared configurations",
        "uncertainty_method": "percentile bootstrap, 2000 resamples",
    },
    "R56": {
        "seed_set": [5602, 5608],
        "repetitions": 2,
        "preprocessing": "none",
        "hyperparameter_selection_protocol": "one resource axis at a time",
        "uncertainty_method": "shot standard errors plus deterministic resource sweep",
    },
    "R57": {
        "seed_set": [7, 23, 47, 89],
        "repetitions": 4,
        "preprocessing": "none",
        "hyperparameter_selection_protocol": "training shots select; validation and final test shots are disjoint",
        "uncertainty_method": "percentile bootstrap, 2000 resamples",
    },
    "R58": {
        "seed_set": [],
        "repetitions": 1,
        "preprocessing": "none",
        "hyperparameter_selection_protocol": "predeclared grid and boundary criterion",
        "uncertainty_method": "none (deterministic conditional sweep)",
    },
    "R59": {
        "seed_set": [],
        "repetitions": 3,
        "preprocessing": "none",
        "hyperparameter_selection_protocol": "predeclared measured ranges; no extrapolation",
        "uncertainty_method": "median and interquartile range",
    },
}


def _literal_constant(script: Path, name: str):
    """Read a simple top-level literal without importing an experiment."""
    try:
        tree = ast.parse(script.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError):
                return None
    return None


def main() -> None:
    contracts = []
    rows = index_rows()
    if tuple(row["experiment_id"] for row in rows) != EXPECTED_IDS:
        raise ValueError("index does not contain the exact canonical 59-experiment sequence")
    for row in rows:
        experiment_id = row["experiment_id"]
        script = ROOT / row["script"]
        override = EXTENDED.get(experiment_id, {})
        seeds = override.get("seed_set")
        if seeds is None:
            literal = _literal_constant(script, "SEEDS")
            seeds = list(literal) if literal is not None else []
        repetitions = override.get("repetitions", len(seeds) if seeds else 1)
        descriptive = "N/A" in row["oracle_class"]
        contracts.append(
            {
                "experiment_id": experiment_id,
                "script": row["script"],
                "scientific_question": row["scientific_question"],
                "hypothesis": "descriptive measurement; no directional performance hypothesis"
                if descriptive
                else "the implementation agrees with the declared independent oracle or invariant",
                "implementation_api": row["api"],
                "independent_oracle_or_baseline": row["oracle"],
                "oracle_independence_class": row["oracle_class"],
                "dataset_or_generated_data_protocol": "defined completely by the numbered script and its immutable constants",
                "preprocessing": override.get(
                    "preprocessing", "none unless explicitly declared in the numbered script"
                ),
                "seed_set": seeds,
                "number_of_repetitions": repetitions,
                "hyperparameter_selection_protocol": override.get(
                    "hyperparameter_selection_protocol",
                    "predeclared in the numbered script; no test-set tuning",
                ),
                "metric_definitions": row["metric"],
                "metric_units": "dimensionless unless the metric name explicitly states seconds or bytes",
                "uncertainty_method": override.get(
                    "uncertainty_method",
                    "recorded by result when stochastic; none for deterministic exact checks",
                ),
                "exact_acceptance_condition": row["acceptance_condition"],
                "expected_artifacts": [
                    str(path.relative_to(ROOT)).replace("\\", "/")
                    for path in required_artifacts(experiment_id, row["script"])
                ],
                "structural_status": "pass required",
                "scientific_outcome": "computed by the experiment; one of positive/negative/descriptive/inconclusive",
                "supported_claim": row["scientific_question"],
                "unsupported_claims": "no claim beyond the declared scope; no quantum advantage",
                "limitations": f"scope is limited to {row['result_kind']} evidence and the declared API/configuration",
                "manuscript_destination": row["manuscript_destination"],
            }
        )
    atomic_write_json(
        OUT,
        {
            "schema_version": 3,
            "expected_experiment_ids": list(EXPECTED_IDS),
            "contracts": contracts,
        },
    )
    print(f"Wrote {OUT} with {len(contracts)} contracts")


if __name__ == "__main__":
    main()
