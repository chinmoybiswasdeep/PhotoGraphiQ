"""scripts/compare_recorded_results.py: regenerated-only, volatile keys ignored, diffs found."""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_recorded_results.py"


def run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True
    )


def test_compare_counts_only_regenerated_files_and_detects_differences(tmp_path):
    rerun, recorded = tmp_path / "copy" / "results" / "json", tmp_path / "recorded"
    rerun.mkdir(parents=True)
    recorded.mkdir()
    payload = {"error": 1.25e-15, "status": "pass", "seconds": 3.0, "rows": [1, 2, float("nan")]}
    for directory in (rerun, recorded):
        (directory / "R1_same.json").write_text(json.dumps(payload))
        (directory / "R2_stale.json").write_text(json.dumps(payload))
    (tmp_path / "copy" / ".copied_at").write_text("1000")
    os.utime(rerun / "R2_stale.json", (500, 500))  # copied, never regenerated
    timing_only = dict(payload, seconds=99.0)  # volatile key differs: still agrees
    (rerun / "R1_same.json").write_text(json.dumps(timing_only))
    report = tmp_path / "report.json"
    assert run(rerun, recorded, "--output", report).returncode == 0
    files = json.loads(report.read_text())["files"]
    assert set(files) == {"R1_same.json"} and files["R1_same.json"]["status"] == "agrees"

    (rerun / "R1_same.json").write_text(json.dumps(dict(payload, error=2e-15)))
    completed = run(rerun, recorded, "--output", report)
    assert completed.returncode == 1
    assert json.loads(report.read_text())["files"]["R1_same.json"]["different"] == 1
