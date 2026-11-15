"""Source-contract and publication-evidence consistency checks."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "paper_experiments_ml"
sys.path.insert(0, str(EXPERIMENTS))

import common  # noqa: E402
import generate_status  # noqa: E402
import publication  # noqa: E402
import run_all_full  # noqa: E402


def rows():
    return publication.index_rows()


def stem(row):
    return publication.result_stem(row["experiment_id"], row["script"])


def load_result(row):
    return json.loads(
        (EXPERIMENTS / "results" / "json" / f"{stem(row)}.json").read_text(encoding="utf-8")
    )


def test_exact_59_indexed_scripts_and_contracts():
    indexed = rows()
    assert tuple(row["experiment_id"] for row in indexed) == publication.EXPECTED_IDS
    assert len(indexed) == 59
    assert len({row["script"] for row in indexed}) == 59
    contracts = json.loads((EXPERIMENTS / "EXPERIMENT_CONTRACTS.json").read_text(encoding="utf-8"))
    assert contracts["schema_version"] == publication.SCHEMA_VERSION
    assert (
        tuple(item["experiment_id"] for item in contracts["contracts"]) == publication.EXPECTED_IDS
    )
    for row, contract in zip(indexed, contracts["contracts"], strict=True):
        assert (EXPERIMENTS / row["script"]).is_file()
        assert contract["script"] == row["script"]
        assert contract["seed_set"] is not None
        assert contract["number_of_repetitions"] >= 1
        assert (
            "test" not in contract["preprocessing"].casefold()
            or "training" in contract["preprocessing"].casefold()
        )
        assert contract["uncertainty_method"]
        assert contract["supported_claim"] and contract["unsupported_claims"]


def test_colab_notebook_has_complete_experiment_and_plot_structure():
    notebook_path = EXPERIMENTS / "PhotoGraphiQML_Manuscript_Experiments.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)

    code_cells = [
        (index, cell) for index, cell in enumerate(notebook.cells) if cell.cell_type == "code"
    ]
    experiment_cells = [
        cell for _, cell in code_cells if "experiment" in cell.metadata.get("tags", [])
    ]
    plot_cells = [
        cell for _, cell in code_cells if "data-complete-plot" in cell.metadata.get("tags", [])
    ]

    def experiment_id(cell):
        return next(tag for tag in cell.metadata["tags"] if tag.startswith("R"))

    assert tuple(map(experiment_id, experiment_cells)) == publication.EXPECTED_IDS
    assert tuple(map(experiment_id, plot_cells)) == publication.EXPECTED_IDS
    assert all(
        index > 0 and notebook.cells[index - 1].cell_type == "markdown" for index, _ in code_cells
    )
    assert all("#" in cell.source for _, cell in code_cells)
    assert all(
        "Mathematical model and declared contract" in notebook.cells[index - 1].source
        for index, cell in code_cells
        if "experiment" in cell.metadata.get("tags", [])
    )
    for index, cell in code_cells:
        compile(cell.source, f"{notebook_path.name}:cell-{index}", "exec")


def test_generated_evidence_agrees_when_schema_v3_manifest_exists():
    manifest = json.loads((EXPERIMENTS / "EXPERIMENT_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != publication.SCHEMA_VERSION:
        pytest.skip("legacy evidence is replaced by the pending publication run")
    assert (
        tuple(item["experiment_id"] for item in manifest["experiments"]) == publication.EXPECTED_IDS
    )
    for row, entry in zip(rows(), manifest["experiments"], strict=True):
        result = load_result(row)
        metadata = json.loads(
            (EXPERIMENTS / "results" / "json" / f"{stem(row)}.metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert entry["experiment_id"] == row["experiment_id"]
        assert result["oracle_class"] == row["oracle_class"]
        assert metadata["oracle_class"] == row["oracle_class"]
        assert result["structural_status"] == "pass"
        assert result["scientific_outcome"] in publication.ALLOWED_OUTCOMES - {"not_run"}
        assert entry["structural_status"] == result["structural_status"]
        assert entry["scientific_outcome"] == result["scientific_outcome"]
        assert entry["claim_supported"] == result["claim_supported"]
        assert entry["claim_not_supported"] == result["claim_not_supported"]
        for artifact in entry["artifacts"]:
            path = EXPERIMENTS / artifact["path"]
            assert path.is_file()
            assert publication.sha256(path) == artifact["sha256"]


def test_r2_uses_one_exact_programmatic_tolerance():
    row = next(row for row in rows() if row["experiment_id"] == "R2")
    exact = common.CANONICAL_R2_TOLERANCE
    assert exact == 2.220446049250313e-15
    assert float(re.search(r"([0-9.]+e-[0-9]+)", row["acceptance_condition"]).group(1)) == exact
    source = (EXPERIMENTS / row["script"]).read_text(encoding="utf-8")
    assert "common.CANONICAL_R2_TOLERANCE" in source
    result_path = EXPERIMENTS / "results" / "json" / f"{stem(row)}.json"
    if result_path.is_file() and load_result(row).get("tolerance") == exact:
        metadata = json.loads(
            (EXPERIMENTS / "results" / "json" / f"{stem(row)}.metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["tolerance"] == exact


def test_r48_interpretation_and_known_scientific_labels():
    r48_source = (
        EXPERIMENTS / next(row for row in rows() if row["experiment_id"] == "R48")["script"]
    ).read_text(encoding="utf-8")
    assert '"scientific_outcome": "negative"' in r48_source
    assert "robust physical classifier performance" in r48_source.casefold()
    r23 = next(row for row in rows() if row["experiment_id"] == "R23")
    assert "one-dimensional threshold" in r23["scientific_question"].casefold()
    r32_path = EXPERIMENTS / "results" / "json" / "R32_lie_closure.json"
    if r32_path.is_file():
        dimensions = {
            item["case"]: item["production_dimension"]
            for item in json.loads(r32_path.read_text(encoding="utf-8"))["rows"]
        }
        assert dimensions["two_qubit_local_su2_direct_sum_dim6"] == 6
        assert dimensions["two_qubit_full_su4_dim15"] == 15


def test_bootstrap_calls_name_the_statistic():
    for path in EXPERIMENTS.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in {"bootstrap_ci", "paired_difference_ci"}
            ):
                assert any(keyword.arg == "statistic" for keyword in call.keywords), path


def _accepted_summary(source: str):
    return {
        "schema_version": publication.SCHEMA_VERSION,
        "publication_mode": True,
        "source_tree_fingerprint": source,
        "dirty_source_paths": [],
        "expected_experiment_ids": list(publication.EXPECTED_IDS),
        "completed_experiment_ids": list(publication.EXPECTED_IDS),
        "failed_experiment_ids": [],
        "skipped_experiment_ids": [],
        "finalizer_and_quality_gates": [{"returncode": 0}],
        "manifest_sha256": "manifest",
        "overall_acceptance": True,
        "rejection_reasons": [],
    }


@pytest.fixture
def isolated_status_validation(tmp_path, monkeypatch):
    (tmp_path / "EXPERIMENT_MANIFEST.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(generate_status, "ROOT", tmp_path)
    monkeypatch.setattr(generate_status, "source_tree_fingerprint", lambda: {"sha256": "source"})
    monkeypatch.setattr(generate_status, "sha256", lambda path: "manifest")
    monkeypatch.setattr(generate_status, "verify", lambda pre_status: {"errors": []})
    return _accepted_summary("source")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version=1), "schema"),
        (lambda value: value.update(publication_mode=False), "publication"),
        (lambda value: value.update(source_tree_fingerprint="stale"), "fingerprint"),
        (lambda value: value.update(dirty_source_paths=["src/x.py"]), "dirty"),
        (lambda value: value.update(skipped_experiment_ids=["R4"]), "skips"),
        (lambda value: value.update(failed_experiment_ids=["R4"]), "failures"),
        (lambda value: value.update(finalizer_and_quality_gates=[{"returncode": 1}]), "gate"),
    ],
)
def test_status_rejects_invalid_evidence(isolated_status_validation, mutation, message):
    summary = copy.deepcopy(isolated_status_validation)
    mutation(summary)
    with pytest.raises(ValueError, match=message):
        generate_status._validate(summary)


def test_resume_rejects_missing_or_modified_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all_full, "ROOT", tmp_path)
    path = tmp_path / "artifact.json"
    path.write_text("original", encoding="utf-8")
    records = [
        {"path": "artifact.json", "size": path.stat().st_size, "sha256": publication.sha256(path)}
    ] * 6
    # Duplicate records also fail the exact expected cardinality semantics once a byte changes.
    assert run_all_full._artifacts_match(records)
    path.write_text("modified", encoding="utf-8")
    assert not run_all_full._artifacts_match(records)
    path.unlink()
    assert not run_all_full._artifacts_match(records)


def test_git_attributes_define_cross_platform_text_and_binary_policy():
    expected = {
        "sample.py": ("text", "set", "eol", "lf"),
        "sample.json": ("text", "set", "eol", "lf"),
        "sample.csv": ("text", "set", "eol", "lf"),
        "sample.md": ("text", "set", "eol", "lf"),
        "sample.svg": ("text", "set", "eol", "lf"),
        "sample.ipynb": ("text", "set", "eol", "lf"),
        "sample.pdf": ("text", "unset", "eol", "unspecified"),
        "sample.png": ("text", "unset", "eol", "unspecified"),
    }
    for name, attributes in expected.items():
        output = subprocess.run(
            ["git", "check-attr", "text", "eol", "--", name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert f"text: {attributes[1]}" in output
        assert f"eol: {attributes[3]}" in output


def test_source_hash_normalizes_declared_text_line_endings(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_bytes(b"first\r\nsecond\r\n")
    monkeypatch.setattr(publication, "REPO", tmp_path)
    monkeypatch.setattr(
        publication.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args, returncode=0, stdout=b"source.py\0text\0set\0"
        ),
    )
    expected = hashlib.sha256(b"first\nsecond\n").hexdigest()
    assert publication.source_sha256(source) == expected
