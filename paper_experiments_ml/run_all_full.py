"""Schema-v3, hash-resumable runner for accepted publication evidence."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from publication import (
    EXPECTED_IDS,
    ROOT,
    SCHEMA_VERSION,
    artifact_records,
    atomic_write_json,
    atomic_write_text,
    environment_record,
    fingerprint,
    fingerprint_hash,
    git,
    index_rows,
    photographiq_commit,
    sha256,
    source_dirty_paths,
    source_tree_fingerprint,
)

LOG_DIR = ROOT / "results" / "logs"
STATE_PATH = ROOT / "results" / "json" / "run_all_full_state.json"
SUMMARY_PATH = ROOT / "results" / "json" / "run_all_full_summary.json"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _artifacts_match(records: list[dict]) -> bool:
    if len(records) != 6:
        return False
    for record in records:
        path = ROOT / record.get("path", "")
        if (
            not path.is_file()
            or record.get("size") != path.stat().st_size
            or record.get("sha256") != sha256(path)
        ):
            return False
    return True


def _write_log(path: Path, command: list[str], process: subprocess.CompletedProcess) -> None:
    atomic_write_text(
        path,
        f"$ {' '.join(command)}\n\n--- stdout ---\n{process.stdout}\n\n--- stderr ---\n{process.stderr}\n",
    )


def _step(name: str, command: list[str], *, cwd: Path, env: dict | None = None) -> dict:
    started = time.perf_counter()
    process = subprocess.run(command, cwd=cwd, capture_output=True, text=True, env=env)
    log_path = LOG_DIR / f"publication_{name}.log"
    _write_log(log_path, command, process)
    record = {
        "name": name,
        "command": command,
        "returncode": process.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
    }
    print(
        f"GATE {'PASS' if process.returncode == 0 else 'FAIL'} {name} ({record['runtime_seconds']:.1f}s)"
    )
    return record


def _rejection_reasons(summary: dict) -> list[str]:
    reasons = []
    if not summary["publication_mode"]:
        reasons.append("publication mode is false")
    if summary["dirty_source_paths"]:
        reasons.append("scientifically relevant source was dirty at run start")
    if tuple(summary["expected_experiment_ids"]) != EXPECTED_IDS:
        reasons.append("expected experiment sequence is not canonical")
    if tuple(summary["completed_experiment_ids"]) != EXPECTED_IDS:
        reasons.append("not every expected experiment completed or resumed")
    if summary["failed_experiment_ids"]:
        reasons.append("one or more experiments failed")
    if summary["skipped_experiment_ids"]:
        reasons.append("one or more experiments were skipped")
    if any(
        item["fingerprint"].get("source_tree_fingerprint") != summary["source_tree_fingerprint"]
        for item in summary["experiments"]
    ):
        reasons.append("per-experiment source fingerprint disagreement")
    if any(not _artifacts_match(item["artifacts"]) for item in summary["experiments"]):
        reasons.append("missing or hash-invalid experiment artifacts")
    if any(gate["returncode"] != 0 for gate in summary["finalizer_and_quality_gates"]):
        reasons.append("a finalizer or quality gate failed")
    manifest = ROOT / "EXPERIMENT_MANIFEST.json"
    if not manifest.is_file() or summary.get("manifest_sha256") != sha256(manifest):
        reasons.append("manifest is missing or disagrees with summary")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="ignore all saved experiment state")
    parser.add_argument(
        "--publication", action="store_true", help="enforce clean-source publication acceptance"
    )
    parser.add_argument(
        "--filter", help="run only one ID or a script substring (never accepted as a full run)"
    )
    args = parser.parse_args()

    rows = index_rows()
    if tuple(row["experiment_id"] for row in rows) != EXPECTED_IDS:
        print("PREFLIGHT FAILED: experiment index is not the canonical 59-entry sequence")
        return 2
    selected = [
        row
        for row in rows
        if not args.filter
        or args.filter.casefold() in row["script"].casefold()
        or args.filter.casefold() == row["experiment_id"].casefold()
    ]
    if not selected:
        print(f"PREFLIGHT FAILED: no experiment matched {args.filter!r}")
        return 2
    dirty_at_start = source_dirty_paths()
    if args.publication and dirty_at_start:
        print("PUBLICATION PREFLIGHT FAILED: scientifically relevant source files are dirty:")
        print("\n".join(dirty_at_start))
        return 2
    if args.publication and photographiq_commit() is None:
        print(
            "PUBLICATION PREFLIGHT FAILED: the unified PhotoGraphiQ repository at "
            f"{ROOT.parent} is not a Git checkout, so the core commit cannot be recorded"
        )
        return 2

    source_commit = git("rev-parse", "HEAD")
    source_branch = git("branch", "--show-current")
    source = source_tree_fingerprint()
    started_utc = datetime.now(timezone.utc).isoformat()
    run_identifier = f"{source_commit[:12]}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    prior_state = {} if args.restart else _load_json(STATE_PATH)
    state_experiments = (
        prior_state.get("experiments", {})
        if prior_state.get("schema_version") == SCHEMA_VERSION
        else {}
    )
    experiment_records = []

    for row in selected:
        experiment_id, relative = row["experiment_id"], row["script"]
        script = ROOT / relative
        current_fingerprint = fingerprint(script)
        current_hash = fingerprint_hash(current_fingerprint)
        previous = state_experiments.get(relative, {})
        resumable = (
            not args.restart
            and previous.get("fingerprint") == current_fingerprint
            and previous.get("fingerprint_sha256") == current_hash
            and previous.get("execution_status") in {"completed", "resumed"}
            and _artifacts_match(previous.get("artifacts", []))
        )
        if resumable:
            record = {
                "experiment_id": experiment_id,
                "script": relative,
                "execution_status": "resumed",
                "runtime_seconds": 0.0,
                "returncode": 0,
                "reason": "complete fingerprint and every artifact hash matched",
                "fingerprint": current_fingerprint,
                "fingerprint_sha256": current_hash,
                "artifacts": previous["artifacts"],
                "log": previous.get("log"),
            }
            print(f"RESUMED {experiment_id} {relative} (fingerprint and hashes matched)")
        else:
            started = time.perf_counter()
            process = subprocess.run(
                [sys.executable, str(script)], cwd=script.parent, capture_output=True, text=True
            )
            runtime = time.perf_counter() - started
            log_path = LOG_DIR / f"{Path(relative).stem}.log"
            _write_log(log_path, [sys.executable, str(script)], process)
            artifacts = artifact_records(experiment_id, relative)
            execution_status = (
                "completed" if process.returncode == 0 and _artifacts_match(artifacts) else "failed"
            )
            record = {
                "experiment_id": experiment_id,
                "script": relative,
                "execution_status": execution_status,
                "runtime_seconds": runtime,
                "returncode": process.returncode,
                "reason": "executed because restart was requested or fingerprint/artifact hashes differed",
                "fingerprint": current_fingerprint,
                "fingerprint_sha256": current_hash,
                "artifacts": artifacts,
                "log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
            }
            print(f"{execution_status.upper()} {experiment_id} {relative} ({runtime:.1f}s)")
        experiment_records.append(record)
        state_experiments[relative] = record
        atomic_write_json(
            STATE_PATH,
            {
                "schema_version": SCHEMA_VERSION,
                "source_commit": source_commit,
                "source_tree_fingerprint": source["sha256"],
                "experiments": state_experiments,
            },
        )

    # A filtered invocation is diagnostic by definition: it never runs finalizers
    # and cannot replace the canonical full-run summary.
    if args.filter:
        return int(any(item["execution_status"] == "failed" for item in experiment_records))

    completed_ids = [
        item["experiment_id"]
        for item in experiment_records
        if item["execution_status"] in {"completed", "resumed"}
    ]
    failed_ids = [
        item["experiment_id"] for item in experiment_records if item["execution_status"] == "failed"
    ]
    resumed_ids = [
        item["experiment_id"]
        for item in experiment_records
        if item["execution_status"] == "resumed"
    ]
    gates = []
    if not failed_ids:
        gates.append(
            _step("generate_tables", [sys.executable, str(ROOT / "generate_tables.py")], cwd=ROOT)
        )
        gates.append(
            _step("build_notebook", [sys.executable, str(ROOT / "build_notebook.py")], cwd=ROOT)
        )
        gates.append(
            _step("build_manifest", [sys.executable, str(ROOT / "build_manifest.py")], cwd=ROOT)
        )
        gates.append(
            _step(
                "verify_evidence_pre_status",
                [sys.executable, str(ROOT / "verify_evidence.py"), "--pre-status"],
                cwd=ROOT,
            )
        )
        validation_dir = ROOT / "results" / "validation" / run_identifier
        gates.append(
            _step(
                "repository_validation",
                [
                    sys.executable,
                    "scripts/ml/validate_release.py",
                    "--output-dir",
                    str(validation_dir),
                ],
                cwd=ROOT.parent,
            )
        )
        gates.append(
            _step(
                "notebook_validation",
                [
                    sys.executable,
                    "-c",
                    "import nbformat; nbformat.validate(nbformat.read('paper_experiments_ml/PhotoGraphiQML_Manuscript_Experiments.ipynb', 4))",
                ],
                cwd=ROOT.parent,
            )
        )
        with tempfile.TemporaryDirectory(prefix="photographiqml-install-") as temporary:
            target = Path(temporary) / "site"
            install = _step(
                "clean_install",
                [sys.executable, "-m", "pip", "install", ".", "--no-deps", "--target", str(target)],
                cwd=ROOT.parent,
            )
            gates.append(install)
            smoke_env = dict(os.environ, PYTHONPATH=str(target))
            gates.append(
                _step(
                    "installed_import_smoke",
                    [
                        sys.executable,
                        "-c",
                        "import photographiq, photographiqml; print(photographiqml.__version__)",
                    ],
                    cwd=ROOT.parent,
                    env=smoke_env,
                )
            )

    manifest_path = ROOT / "EXPERIMENT_MANIFEST.json"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "publication_mode": args.publication,
        "run_identifier": run_identifier,
        "source_commit": source_commit,
        "source_tree_fingerprint": source["sha256"],
        "evidence_commit": None,
        "branch": source_branch,
        "dirty_source_paths": dirty_at_start,
        "environment": environment_record(),
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "expected_experiment_ids": list(EXPECTED_IDS),
        "completed_experiment_ids": completed_ids,
        "resumed_experiment_ids": resumed_ids,
        "failed_experiment_ids": failed_ids,
        "skipped_experiment_ids": [],
        "counts": {
            "expected": len(EXPECTED_IDS),
            "completed_or_resumed": len(completed_ids),
            "executed": len(completed_ids) - len(resumed_ids),
            "resumed": len(resumed_ids),
            "failed": len(failed_ids),
            "skipped": 0,
        },
        "experiments": experiment_records,
        "finalizer_and_quality_gates": gates,
        "manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else None,
        "total_wall_seconds": sum(item["runtime_seconds"] for item in experiment_records)
        + sum(gate["runtime_seconds"] for gate in gates),
        "overall_acceptance": False,
        "rejection_reasons": [],
    }
    summary["rejection_reasons"] = _rejection_reasons(summary)
    summary["overall_acceptance"] = not summary["rejection_reasons"]
    atomic_write_json(SUMMARY_PATH, summary)

    if summary["overall_acceptance"]:
        status_gate = _step(
            "generate_status", [sys.executable, str(ROOT / "generate_status.py")], cwd=ROOT
        )
        if status_gate["returncode"]:
            print("FINAL STATUS GENERATION FAILED")
            return 1
        final_verify = _step(
            "verify_evidence_final", [sys.executable, str(ROOT / "verify_evidence.py")], cwd=ROOT
        )
        if final_verify["returncode"]:
            print("FINAL EVIDENCE VERIFICATION FAILED")
            return 1
    print(
        f"PUBLICATION ACCEPTED={summary['overall_acceptance']} reasons={summary['rejection_reasons']}"
    )
    return 0 if summary["overall_acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
