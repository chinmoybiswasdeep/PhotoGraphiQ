"""Build the schema-v3 evidence manifest after experiments and derived outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from publication import (
    EXPECTED_IDS,
    ROOT,
    SCHEMA_VERSION,
    artifact_records,
    atomic_write_json,
    git,
    index_rows,
    result_stem,
    sha256,
    source_tree_fingerprint,
)

OUT = ROOT / "EXPERIMENT_MANIFEST.json"


def main() -> None:
    contracts_payload = json.loads((ROOT / "EXPERIMENT_CONTRACTS.json").read_text(encoding="utf-8"))
    contracts = {item["experiment_id"]: item for item in contracts_payload["contracts"]}
    rows = index_rows()
    if tuple(row["experiment_id"] for row in rows) != EXPECTED_IDS:
        raise ValueError("index is not the canonical 59-experiment sequence")
    experiments = []
    for row in rows:
        experiment_id = row["experiment_id"]
        stem = result_stem(experiment_id, row["script"])
        result = json.loads(
            (ROOT / "results" / "json" / f"{stem}.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (ROOT / "results" / "json" / f"{stem}.metadata.json").read_text(encoding="utf-8")
        )
        experiments.append(
            {
                "experiment_id": experiment_id,
                "script": row["script"],
                "contract": contracts[experiment_id],
                "execution_status": result.get("execution_status"),
                "structural_status": result.get("structural_status"),
                "scientific_outcome": result.get("scientific_outcome"),
                "claim_supported": result.get("claim_supported"),
                "claim_not_supported": result.get("claim_not_supported"),
                "seed_set": result.get("seeds", metadata.get("seed_set", [])),
                "source_tree_fingerprint": metadata.get("source_tree_fingerprint"),
                "artifacts": artifact_records(experiment_id, row["script"]),
            }
        )
    derived_paths = [ROOT / "PhotoGraphiQML_Manuscript_Experiments.ipynb"]
    derived_paths.extend(sorted((ROOT / "tables").glob("*")))
    derived = [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in derived_paths
        if path.is_file()
    ]
    source = source_tree_fingerprint()
    atomic_write_json(
        OUT,
        {
            "schema_version": SCHEMA_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": git("rev-parse", "HEAD"),
            "source_branch": git("branch", "--show-current"),
            "source_tree_fingerprint": source["sha256"],
            "evidence_commit": None,
            "expected_experiment_ids": list(EXPECTED_IDS),
            "experiments": experiments,
            "derived_artifacts": derived,
        },
    )
    print(f"Wrote {OUT} with {len(experiments)} experiments and {len(derived)} derived artifacts")


if __name__ == "__main__":
    main()
