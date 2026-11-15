"""Tiny end-to-end run of benchmarks/graphix/bench_graphix_mbqc.py (correctness checks only)."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("graphix", minversion="0.4")
pytest.importorskip("numba")

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "graphix" / "bench_graphix_mbqc.py"


def test_graphix_benchmark_tiny_run(tmp_path):
    spec = importlib.util.spec_from_file_location("bench_graphix_mbqc", BENCH)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    cfg = {
        "flow_sizes": {"linear": [6], "brickwork4": [9], "lc_random": [4]},
        "pattern_sizes": {"linear": [4], "lc_random": [3]},
        "sv_sizes": {"grid3": [3]},
        "agreement_instances": 10,
        "cap_s": 1.0,
        "warmup": 1,
        "repeats": 2,
    }
    res = bench.run(cfg, out_dir=tmp_path, plots=False, log=lambda *_: None)
    assert all(
        t["existence_agree"] == t["all_checks_ok"] == 10
        for t in res["agreement_random_small"].values()
    )
    assert len(res["flows"]) == 9 and all(r["all_checks_ok"] for r in res["flows"])
    assert all(
        r["pg_exists"] for r in res["flows"] if r["family"] != "lc_random" or r["task"] != "causal"
    )
    for row in res["patterns"]:
        assert row["both_standard"] and row["std_same_semantic_form"]
        assert row["shift_same_quantum_commands"] and row["shift_same_signal_map"]
        assert all(v["ok"] for v in row["map_equivalence"].values())
    assert res["statevector"][0]["abs_overlap_error"] < 1e-10
    assert list(tmp_path.glob("graphix_mbqc_*.json"))
