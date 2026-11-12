"""Generate STATUS.md only from a fully accepted schema-v3 summary."""

from __future__ import annotations

import json

from publication import (
    EXPECTED_IDS,
    ROOT,
    SCHEMA_VERSION,
    atomic_write_text,
    git,
    sha256,
    source_tree_fingerprint,
)
from verify_evidence import verify


def _validate(summary: dict) -> None:
    reasons = []
    if summary.get("schema_version") != SCHEMA_VERSION:
        reasons.append("incompatible summary schema")
    if not summary.get("publication_mode"):
        reasons.append("summary is not from publication mode")
    if summary.get("source_tree_fingerprint") != source_tree_fingerprint()["sha256"]:
        reasons.append("summary source fingerprint is stale")
    if summary.get("dirty_source_paths"):
        reasons.append("source was dirty at run start")
    if tuple(summary.get("expected_experiment_ids", [])) != EXPECTED_IDS:
        reasons.append("expected IDs are incomplete")
    if tuple(summary.get("completed_experiment_ids", [])) != EXPECTED_IDS:
        reasons.append("completed IDs are incomplete")
    if summary.get("failed_experiment_ids") or summary.get("skipped_experiment_ids"):
        reasons.append("run has failures or skips")
    if any(gate.get("returncode") != 0 for gate in summary.get("finalizer_and_quality_gates", [])):
        reasons.append("a finalizer or quality gate failed")
    manifest = ROOT / "EXPERIMENT_MANIFEST.json"
    if not manifest.is_file() or summary.get("manifest_sha256") != sha256(manifest):
        reasons.append("manifest hash disagrees with summary")
    if not summary.get("overall_acceptance") or summary.get("rejection_reasons"):
        reasons.append("summary does not assert clean acceptance")
    evidence_report = verify(pre_status=True)
    reasons.extend(evidence_report["errors"])
    if reasons:
        raise ValueError("STATUS.md generation rejected:\n- " + "\n- ".join(reasons))


def main() -> None:
    summary_path = ROOT / "results" / "json" / "run_all_full_summary.json"
    if not summary_path.is_file():
        raise ValueError("STATUS.md generation rejected: final run summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate(summary)
    manifest = json.loads((ROOT / "EXPERIMENT_MANIFEST.json").read_text(encoding="utf-8"))
    outcomes: dict[str, int] = {}
    for experiment in manifest["experiments"]:
        outcome = experiment["scientific_outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    current_commit = git("rev-parse", "HEAD")
    evidence_commit = summary.get("evidence_commit") or (
        current_commit if current_commit != summary["source_commit"] else "pending evidence commit"
    )
    lines = [
        "# Suite execution status",
        "",
        "## Accepted publication evidence",
        "",
        f"- Accepted: `{summary['overall_acceptance']}`",
        f"- Run identifier: `{summary['run_identifier']}`",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Source-tree fingerprint: `{summary['source_tree_fingerprint']}`",
        f"- Evidence commit: `{evidence_commit}`",
        f"- Branch: `{summary['branch']}`",
        f"- Started: `{summary['started_utc']}`",
        f"- Completed: `{summary['completed_utc']}`",
        f"- Experiments completed or hash-resumed: `{summary['counts']['completed_or_resumed']}` / `{summary['counts']['expected']}`",
        f"- Executed in this run: `{summary['counts']['executed']}`",
        f"- Hash-resumed: `{summary['counts']['resumed']}`",
        f"- Failed: `{summary['counts']['failed']}`",
        f"- Skipped: `{summary['counts']['skipped']}`",
        f"- Manifest SHA-256: `{summary['manifest_sha256']}`",
        "",
        "## Scientific outcomes",
        "",
    ]
    lines.extend(f"- {name}: `{count}`" for name, count in sorted(outcomes.items()))
    lines.extend(
        [
            "",
            "Structural pass means execution and declared checks passed; it is not a positive scientific finding.",
            "R48 retains the observed physical-training instability as a negative result.",
            "",
            "## Quality gates",
            "",
        ]
    )
    lines.extend(
        f"- `{gate['name']}`: `pass` ({gate['runtime_seconds']:.1f} s; log `{gate['log']}`)"
        for gate in summary["finalizer_and_quality_gates"]
    )
    lines.extend(
        [
            "",
            "## Capability boundary",
            "",
            "Arbitrary-angle physical XY measurements, general physical MuTA, native CVMuTA, loss/detector-noise models, soft adaptive flow decoding, and physical MuTA kernels remain unsupported. No experiment is described as quantum advantage.",
            "",
        ]
    )
    atomic_write_text(ROOT / "STATUS.md", "\n".join(lines))
    print("Wrote paper_experiments_ml/STATUS.md from accepted final summary")


if __name__ == "__main__":
    main()
