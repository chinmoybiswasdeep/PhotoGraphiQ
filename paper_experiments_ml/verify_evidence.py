"""Independently verify manuscript evidence and return nonzero on any defect."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from publication import (  # noqa: E402
    ALLOWED_OUTCOMES,
    EXPECTED_IDS,
    ROOT,
    SCHEMA_VERSION,
    index_rows,
    required_artifacts,
    result_stem,
    sha256,
    source_tree_fingerprint,
)


def _load(path: Path, errors: list[str]):
    if not path.is_file():
        errors.append(f"missing file: {path.relative_to(ROOT)}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid JSON {path.relative_to(ROOT)}: {error}")
        return {}


def _unique_exact(label: str, actual: list[str], errors: list[str]) -> None:
    duplicates = [item for item, count in Counter(actual).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate {label}: {duplicates}")
    if tuple(actual) != EXPECTED_IDS:
        errors.append(f"{label} is not exact canonical sequence: {actual}")


def verify(*, pre_status: bool = False) -> dict:
    errors: list[str] = []
    source = source_tree_fingerprint()["sha256"]
    rows = index_rows()
    _unique_exact("index IDs", [row["experiment_id"] for row in rows], errors)
    scripts = [row["script"] for row in rows]
    if len(scripts) != len(set(scripts)):
        errors.append("duplicate indexed scripts")
    discovered = sorted(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in ROOT.glob("[0-9][0-9]_*/*.py")
        if re.match(r"\d{2}_", path.name)
    )
    if set(discovered) != set(scripts):
        errors.append(
            f"indexed/discovered script mismatch; unindexed={sorted(set(discovered) - set(scripts))}, "
            f"missing={sorted(set(scripts) - set(discovered))}"
        )

    contract_payload = _load(ROOT / "EXPERIMENT_CONTRACTS.json", errors)
    if contract_payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("contract schema is incompatible")
    contracts_list = contract_payload.get("contracts", [])
    _unique_exact("contract IDs", [item.get("experiment_id") for item in contracts_list], errors)
    contracts = {item.get("experiment_id"): item for item in contracts_list}
    required_contract_keys = {
        "experiment_id",
        "scientific_question",
        "hypothesis",
        "implementation_api",
        "independent_oracle_or_baseline",
        "oracle_independence_class",
        "dataset_or_generated_data_protocol",
        "preprocessing",
        "seed_set",
        "number_of_repetitions",
        "hyperparameter_selection_protocol",
        "metric_definitions",
        "uncertainty_method",
        "exact_acceptance_condition",
        "expected_artifacts",
        "structural_status",
        "scientific_outcome",
        "supported_claim",
        "unsupported_claims",
        "limitations",
        "manuscript_destination",
    }
    for contract in contracts_list:
        missing = required_contract_keys - contract.keys()
        if missing:
            errors.append(f"{contract.get('experiment_id')} incomplete contract: {sorted(missing)}")

    manifest = _load(ROOT / "EXPERIMENT_MANIFEST.json", errors)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("manifest schema is incompatible")
    if manifest.get("source_tree_fingerprint") != source:
        errors.append("manifest source-tree fingerprint is stale")
    manifest_list = manifest.get("experiments", [])
    _unique_exact("manifest IDs", [item.get("experiment_id") for item in manifest_list], errors)
    manifest_by_id = {item.get("experiment_id"): item for item in manifest_list}

    artifact_count = 0
    for row in rows:
        experiment_id = row["experiment_id"]
        script = ROOT / row["script"]
        if not script.is_file():
            errors.append(f"missing script: {row['script']}")
            continue
        stem = result_stem(experiment_id, row["script"])
        result = _load(ROOT / "results" / "json" / f"{stem}.json", errors)
        metadata = _load(ROOT / "results" / "json" / f"{stem}.metadata.json", errors)
        if result.get("execution_status") != "completed":
            errors.append(f"{experiment_id} execution status is not completed")
        if result.get("structural_status") != "pass" or result.get("status") != "pass":
            errors.append(f"{experiment_id} structural status is not pass")
        if result.get("scientific_outcome") not in ALLOWED_OUTCOMES - {"not_run"}:
            errors.append(f"{experiment_id} has invalid scientific outcome")
        for field in (
            "claim_supported",
            "claim_not_supported",
            "acceptance_condition",
            "oracle_class",
        ):
            if not result.get(field):
                errors.append(f"{experiment_id} result missing {field}")
        if metadata.get("experiment_id") != experiment_id:
            errors.append(f"{experiment_id} metadata ID mismatch")
        if metadata.get("source_dirty") or metadata.get("git_dirty"):
            errors.append(f"{experiment_id} metadata reports dirty scientific source")
        if metadata.get("source_tree_fingerprint") != source:
            errors.append(f"{experiment_id} metadata source fingerprint mismatch")
        if metadata.get("provenance_schema_version") != SCHEMA_VERSION:
            errors.append(f"{experiment_id} metadata schema mismatch")
        contract = contracts.get(experiment_id, {})
        contract_seeds = contract.get("seed_set", [])
        result_seeds = result.get("seeds", metadata.get("seed_set", []))
        if contract_seeds and list(result_seeds) != list(contract_seeds):
            errors.append(f"{experiment_id} seed set differs between contract and result")
        entry = manifest_by_id.get(experiment_id, {})
        for key in ("execution_status", "structural_status", "scientific_outcome"):
            if entry.get(key) != result.get(key):
                errors.append(f"{experiment_id} manifest/result {key} mismatch")
        if entry.get("source_tree_fingerprint") != source:
            errors.append(f"{experiment_id} manifest fingerprint mismatch")
        records = {item.get("path"): item for item in entry.get("artifacts", [])}
        expected_paths = required_artifacts(experiment_id, row["script"])
        if set(records) != {
            str(path.relative_to(ROOT)).replace("\\", "/") for path in expected_paths
        }:
            errors.append(f"{experiment_id} manifest artifact set mismatch")
        for path in expected_paths:
            relative = str(path.relative_to(ROOT)).replace("\\", "/")
            if not path.is_file():
                errors.append(f"{experiment_id} missing artifact {relative}")
                continue
            artifact_count += 1
            record = records.get(relative, {})
            if record.get("sha256") != sha256(path) or record.get("size") != path.stat().st_size:
                errors.append(f"{experiment_id} artifact hash/size mismatch: {relative}")
        if contract.get("expected_artifacts") != [
            str(path.relative_to(ROOT)).replace("\\", "/") for path in expected_paths
        ]:
            errors.append(f"{experiment_id} contract artifact set mismatch")

    for record in manifest.get("derived_artifacts", []):
        path = ROOT / record.get("path", "")
        if not path.is_file() or record.get("sha256") != sha256(path):
            errors.append(f"derived artifact missing/hash mismatch: {record.get('path')}")

    r2 = next((row for row in rows if row["experiment_id"] == "R2"), None)
    if r2:
        stem = result_stem("R2", r2["script"])
        result = _load(ROOT / "results" / "json" / f"{stem}.json", errors)
        metadata = _load(ROOT / "results" / "json" / f"{stem}.metadata.json", errors)
        exact = common.CANONICAL_R2_TOLERANCE
        declarations = [result.get("tolerance"), metadata.get("tolerance")]
        match = re.search(r"([0-9.]+e-[0-9]+)", r2["acceptance_condition"])
        declarations.append(float(match.group(1)) if match else None)
        if any(value != exact for value in declarations):
            errors.append(f"R2 tolerance declarations are not exactly {exact!r}: {declarations}")
    r48 = next((row for row in rows if row["experiment_id"] == "R48"), None)
    if r48:
        result = _load(
            ROOT / "results" / "json" / f"{result_stem('R48', r48['script'])}.json", errors
        )
        if (
            result.get("structural_status") != "pass"
            or result.get("scientific_outcome") != "negative"
        ):
            errors.append("R48 must be structural pass / scientific negative")

    if not pre_status:
        summary_path = ROOT / "results" / "json" / "run_all_full_summary.json"
        summary = _load(summary_path, errors)
        if summary.get("schema_version") != SCHEMA_VERSION:
            errors.append("summary schema is incompatible")
        if not summary.get("publication_mode"):
            errors.append("summary is not a publication run")
        if summary.get("source_tree_fingerprint") != source:
            errors.append("summary source-tree fingerprint is stale")
        if summary.get("dirty_source_paths"):
            errors.append("summary reports dirty source at run start")
        for key in ("expected_experiment_ids", "completed_experiment_ids"):
            if tuple(summary.get(key, [])) != EXPECTED_IDS:
                errors.append(f"summary {key} is incomplete")
        if summary.get("failed_experiment_ids") or summary.get("skipped_experiment_ids"):
            errors.append("summary contains failed or skipped experiments")
        results = summary.get("experiments", [])
        _unique_exact(
            "summary experiment IDs", [item.get("experiment_id") for item in results], errors
        )
        for item in results:
            if item.get("execution_status") not in {"completed", "resumed"}:
                errors.append(f"summary invalid execution status for {item.get('experiment_id')}")
            if item.get("fingerprint", {}).get("source_tree_fingerprint") != source:
                errors.append(
                    f"summary stale experiment fingerprint for {item.get('experiment_id')}"
                )
            for artifact in item.get("artifacts", []):
                path = ROOT / artifact.get("path", "")
                if not path.is_file() or artifact.get("sha256") != sha256(path):
                    errors.append(f"summary artifact mismatch: {artifact.get('path')}")
        gates = summary.get("finalizer_and_quality_gates", [])
        if not gates or any(gate.get("returncode") != 0 for gate in gates):
            errors.append("summary finalizer/quality gates are missing or failed")
        if summary.get("manifest_sha256") != sha256(ROOT / "EXPERIMENT_MANIFEST.json"):
            errors.append("summary/manifest hash mismatch")
        if not summary.get("overall_acceptance") or summary.get("rejection_reasons"):
            errors.append("summary is not accepted")
        status_path = ROOT / "STATUS.md"
        if not status_path.is_file():
            errors.append("STATUS.md is missing")
        else:
            status = status_path.read_text(encoding="utf-8")
            if (
                summary.get("run_identifier", "<missing>") not in status
                or "Accepted: `True`" not in status
            ):
                errors.append("STATUS.md does not describe the accepted summary")

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "pre-status" if pre_status else "complete",
        "source_tree_fingerprint": source,
        "experiments_checked": len(rows),
        "artifacts_checked": artifact_count,
        "errors": errors,
        "valid": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pre-status",
        action="store_true",
        help="verify artifacts/manifest before final summary and status exist",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = verify(pre_status=args.pre_status)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"evidence valid={report['valid']} experiments={report['experiments_checked']} artifacts={report['artifacts_checked']}"
        )
        for error in report["errors"]:
            print(f"ERROR: {error}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
