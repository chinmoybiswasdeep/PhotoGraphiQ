"""Smoke test: the native Gaussian benchmark runs its tiny config and emits valid, checked JSON."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from photographiq import native

pytestmark = pytest.mark.skipif(
    "gaussian" not in native.components(), reason="native gaussian component not built"
)

SCRIPT = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "gaussian" / "bench_native_gaussian.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("bench_native_gaussian", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_gaussian_benchmark_smoke(tmp_path):
    bench = _load()
    out = tmp_path / "result.json"
    bench.run("smoke", output=out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    env = payload["environment"]
    assert {"cpu", "logical_cpus", "git", "env_threads", "native_gaussian_max_threads"} <= set(env)
    assert {"numpy_blas", "threadpoolctl"} <= set(env)
    assert payload["config"] == "smoke" and payload["comparisons"]
    rows = payload["results"]
    modes = set(bench.CONFIGS["smoke"]["modes"])
    for task in ("batch_gate", "batch_homodyne"):
        for n in modes:
            got = {r["kind"] for r in rows if r["task"] == task and r["modes"] == n}
            assert got == {"native-batch", "numpy-vectorized", "python-loop"}
    for row in rows:
        assert row["kind"] == bench.KINDS[row["implementation"]]
        assert 0 < row["min_s"] <= row["median_s"] and row["iqr_s"] >= 0
        if row["task"].startswith("batch_"):
            assert row["max_scaled_diff_vs_native"] <= row["agreement_tol"] == 1e-12
    assert {s["comparison"] for s in payload["speedups"]} >= {
        "numpy-vectorized / native-batch",
        "python-loop / native-batch",
    }
    assert all(s["ratio_of_medians"] > 0 for s in payload["speedups"])


def test_agreement_check_rejects_wrong_output():
    bench = _load()
    state = bench.chain_state(4)
    data = bench.batch_inputs(state, 3, seed=1)
    idx = bench.np.array([2, 3, 4, 5])
    reference = bench.ng.batch_apply_symplectic(data["means"], data["covs"], [1, 2], data["mats"])
    wrong = bench.numpy_batch_gate(
        data["means"], data["covs"], idx, data["mats"].transpose(0, 2, 1)
    )
    with pytest.raises(RuntimeError, match="disagrees with native"):
        bench.check("batch_gate", "numpy_vectorized", 4, wrong, reference)
    # A wrong shape must not broadcast into a pass, e.g. densities (B, 1) vs (B,).
    ones = bench.np.ones(3)
    with pytest.raises(RuntimeError, match="disagrees with native"):
        bench.check("batch_homodyne", "numpy_vectorized", 4, [ones[:, None]], [ones])


def test_chain_state_matches_dense_reference():
    """The native-built benchmark input equals the dense GaussianBackend construction."""
    bench = _load()
    for n in (2, 5, 9):
        got = bench.chain_state(n, seed=3)
        ref = bench.GaussianBackend()
        rng = bench.np.random.default_rng(3)
        for node in range(n):
            ref.prepare(node, float(rng.uniform(0.3, 1.0)))
        for node in range(n - 1):
            ref.entangle(node, node + 1, 1.0)
        for node in range(n):
            ref.loss(node, 0.9, 0.05)
        want = ref.get_state()
        assert got.nodes == want.nodes
        assert bench.np.allclose(got.mean, want.mean, rtol=0, atol=1e-12)
        assert bench.np.allclose(got.covariance, want.covariance, rtol=0, atol=1e-12)
