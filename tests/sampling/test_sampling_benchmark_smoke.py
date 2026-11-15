"""Smoke test: the BS / GBS benchmark scripts run in their tiny configuration and emit valid JSON."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"


def _load(relative: str):
    path = BENCH / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "script, tasks",
    [
        (
            "boson_sampling/bench_boson_sampling.py",
            {"permanent", "probability_api", "distribution", "sampling"},
        ),
        (
            "gaussian_boson_sampling/bench_gbs.py",
            {"hafnian", "torontonian", "gbs_pnr", "gbs_threshold"},
        ),
    ],
)
def test_benchmark_smoke(tmp_path, script, tasks):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    out = tmp_path / "result.json"
    _load(script).run("smoke", output=out, figures=True)
    payload = json.loads(out.read_text(encoding="utf-8"))
    env = payload["environment"]
    assert {"cpu", "logical_cpus", "python", "packages", "git", "seed"} <= set(env)
    assert tasks <= {r["task"] for r in payload["results"]}
    for row in payload["results"]:
        if "median_s" in row:
            assert row["median_s"] > 0 and row["min_s"] <= row["median_s"]
        for key in ("rel_error", "rel_error_vs_piquasso", "rel_error_vs_python"):
            if row.get(key) is not None:
                assert row[key] < 1e-8, row
    assert payload["figures"] and all(Path(p).exists() for p in payload["figures"])


def test_time_call_excludes_cold_first_call():
    import time

    timer = _load("bench_common.py").time_call
    calls = []

    def fn():  # cold first call (like JIT compilation), then cheap
        calls.append(None)
        if len(calls) == 1:
            time.sleep(0.3)

    stats = timer(fn, repeats=3, budget_s=0.5)
    assert stats["first_call_s"] >= 0.3 and stats["median_s"] < 0.05 and stats["warmup"] == 2
