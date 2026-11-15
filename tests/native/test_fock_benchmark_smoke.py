"""Smoke test: the native Fock benchmark runs its tiny config and emits checked JSON."""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pytest

from photographiq import native

pytestmark = pytest.mark.skipif(
    "fock" not in native.components(), reason="native fock component not built"
)

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bench():
    pytest.importorskip("piquasso")
    return _load(BENCHMARKS / "fock" / "bench_fock.py")


def test_fock_benchmark_smoke(tmp_path):
    bench = _bench()
    out = tmp_path / "result.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # retained-norm warnings at cutoff 6
        bench.run("smoke", output=out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["config"] == "smoke"
    env = payload["environment"]
    assert {"cpu", "git", "packages", "native_fock_max_threads", "physical_cpus"} <= set(env)
    cfg = bench.CONFIGS["smoke"]
    gates = payload["gates"]
    expected = len(bench.GATES) * len(cfg["pure"]) + (len(bench.GATES) + 1) * len(cfg["mixed"])
    assert len(gates) == 2 * expected
    for row in gates:
        assert row["native_vs_piquasso_trace_distance"] <= row["agreement_tol"]
        assert row["agreement_tol"] == bench.ROUNDING_TOL or row["gate"] == "cz"
        assert row["median_s"] > 0 and row["iqr_s"] >= 0
    assert {r["engine"] for r in payload["patterns"]} == {
        "native",
        "piquasso",
        "native-mixed",
        "piquasso-mixed",
    }
    for row in payload["patterns"]:
        assert row["native_vs_piquasso_trace_distance"] <= row["agreement_tol"]
        assert row["native_vs_piquasso_statistics_rel_diff"] <= row["agreement_tol"]
        assert row["median_s"] > 0 and "peak_memory_bytes" in row
    assert {r["threads"] for r in payload["threads"]} == set(cfg["threads"])
    for row in payload["threads"]:
        assert row["checksum_rel_diff_vs_first"] <= row["agreement_tol"] == bench.ROUNDING_TOL
    conv = payload["convergence"]
    assert {(r["engine"], r["cutoff"]) for r in conv} == {
        (e, c) for e in ("native", "piquasso") for c in cfg["convergence"]
    }
    for engine in ("native", "piquasso"):  # truncation error shrinks with the cutoff
        errors = [r["max_abs_photon_number_error"] for r in conv if r["engine"] == engine]
        assert all(b < a for a, b in zip(errors, errors[1:], strict=False))
    by_cutoff = {}
    for r in conv:  # both engines truncate the same Gaussian circuit identically
        by_cutoff.setdefault(r["cutoff"], []).append(r["max_abs_photon_number_error"])
    for a, b in by_cutoff.values():
        assert abs(a - b) <= 1e-12


def test_agreement_check_rejects_wrong_output():
    bench = _bench()
    from photographiq.backends.fock import PiquassoFockBackend

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        backends = {
            "native": bench.prepared(bench.NativeFockBackend(cutoff=8), 2),
            "piquasso": bench.prepared(PiquassoFockBackend(cutoff=8), 2),
        }
        ours, reference = bench.gate_outputs(backends, bench.GATES["kerr"])
        bench.agreement("kerr", ours, reference, False)
        slightly_wrong, _ = bench.gate_outputs(backends, lambda b: b.kerr(0, 0.2 + 1e-6))
        wrong, _ = bench.gate_outputs(backends, lambda b: b.kerr(0, 0.3))
    with pytest.raises(RuntimeError, match="disagrees with Piquasso"):  # trace distance 3e-7
        bench.agreement("kerr", slightly_wrong, reference, False)
    # The truncation-scheme tolerance, sqrt(boundary population) = 1.1e-3 here, is no free
    # pass: a kappa error of 0.1 (trace distance 3e-2) is rejected.
    with pytest.raises(RuntimeError, match="disagrees with Piquasso"):
        bench.agreement("kerr", wrong, reference, True)
    stats = {"n": {"value": 0.5}}
    with pytest.raises(RuntimeError, match="disagrees with Piquasso"):
        bench.agreement("kerr", ours, reference, False, (stats, {"n": {"value": 0.6}}))
    # A zero reference statistic is a mismatch, not a ZeroDivisionError.
    with pytest.raises(RuntimeError, match="disagrees with Piquasso"):
        bench.agreement("kerr", ours, reference, False, (stats, {"n": {"value": 0.0}}))


def test_memory_fields_are_none_without_psutil(monkeypatch, capsys):
    bench = _bench()
    monkeypatch.setattr(bench, "psutil", None)
    env = bench.environment_metadata()
    assert env["physical_cpus"] is env["memory_bytes"] is None
    assert env["system_cpu_percent_at_start"] is None
    bench.child("network", 3, 6, "native", 1)
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["peak_memory_bytes"] is result["baseline_rss_bytes"] is None
    assert result["median_s"] > 0


def test_threshold_mp_reference_output_option(tmp_path):
    pytest.importorskip("mpmath")
    ref = _load(BENCHMARKS / "gaussian_boson_sampling" / "threshold_mp_reference.py")
    args = ref.parse_args(["24,12,0.9", "--output", str(tmp_path / "r.json"), "28,14,0.9"])
    assert args.cases == [(24.0, 12.0, 0.9), (28.0, 14.0, 0.9)]
    assert args.output == tmp_path / "r.json"
    assert ref.parse_args(["24,12,0.9"]).output is None  # default: results/<stamp>.json
    for bad in ("24,12", "24,12,0.9,1", "a,b,c"):  # usage error, not a traceback
        with pytest.raises(SystemExit):
            ref.parse_args([bad])
