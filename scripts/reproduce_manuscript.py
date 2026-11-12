"""Reproduce the evidence behind the PhotoGraphiQ manuscript.

    python scripts/reproduce_manuscript.py                 # fast validation (minutes)
    python scripts/reproduce_manuscript.py --mode full     # benchmarks + experiment suites (hours)
    python scripts/reproduce_manuscript.py --list          # show the steps, run nothing

Every step runs in a subprocess with a fixed seed where the step takes one. Its log goes to
``<output>/logs/``, its artifacts to ``<output>/artifacts/``, and ``<output>/summary.json``
records the environment, the git revision, each command, its exit code, runtime and status.
A step whose optional dependency is missing is recorded as ``skipped`` with the reason, never
as passed. The exit code is non-zero if any executed step failed.

Nothing is written into the repository: benchmark outputs go to ``<output>`` (default
``build/reproduction/<timestamp>``, git-ignored), and the manuscript experiment suites run
from copies under ``<output>`` so the recorded results in ``paper_experiments*/results`` stay
byte-for-byte unchanged. The copies' own git-provenance fields refer to the build
directory, not the source tree; ``summary.json`` records the real revision and dirty state.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Step:
    name: str
    command: list[str]
    needs: tuple[str, ...] = ()  # importable modules the step requires
    claim: str = ""  # manuscript claim the step supports
    modes: tuple[str, ...] = ("fast", "full")
    timeout: int = 3600
    cwd: Path | None = None
    setup: object = field(default=None, repr=False)  # callable(out) run before the step


def _py(*args: str) -> list[str]:
    return [sys.executable, *args]


def _copy_suite(name: str):
    """Copy an experiment suite under ``out`` so its recorded results are not modified."""

    def setup(out: Path) -> None:
        target = out / "suites" / name
        if not target.exists():
            shutil.copytree(ROOT / name, target, ignore=shutil.ignore_patterns("__pycache__"))
            time.sleep(1.0)  # file mtimes after this marker identify regenerated results
            (target / ".copied_at").write_text(f"{time.time():.3f}", encoding="utf-8")

    return setup


def steps(out: Path) -> list[Step]:
    art = out / "artifacts"
    tests = [
        ("tests-validation", "tests/validation", "independent Gaussian/Fock validation matrix"),
        ("tests-native", "tests/native", "native engines agree with references"),
        ("tests-sampling", "tests/sampling", "BS/GBS probabilities and samplers"),
        ("tests-flow-calculus", "tests/flow tests/calculus", "flows, measurement calculus, ZX"),
        ("tests-cudaq-adapter", "tests/cudaq_adapter", "CUDA-Q translation (NumPy reference)"),
        ("tests-gpu", "tests/gpu", "GPU kernels equal CPU (skips without CUDA)"),
        ("tests-ml", "tests/ml", "PhotoGraphiQML contract and physical lowering"),
    ]
    result = [
        Step(
            name,
            _py("-m", "pytest", "-q", "-p", "no:cacheprovider", *paths.split()),
            needs=("pytest",),
            claim=claim,
        )
        for name, paths, claim in tests
    ]
    result.append(
        Step(
            "tests-full",
            _py("-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"),
            needs=("pytest",),
            claim="complete regression suite",
            modes=("full",),
            timeout=7200,
        )
    )
    result.append(
        Step(
            "figures-check",
            _py("scripts/generate_architecture_figures.py", "--check"),
            needs=("matplotlib",),
            claim="architecture figures cite existing code",
        )
    )
    bench = [
        ("bench-boson-sampling", "benchmarks/boson_sampling/bench_boson_sampling.py", "--output"),
        ("bench-gbs", "benchmarks/gaussian_boson_sampling/bench_gbs.py", "--output"),
        ("bench-native-gaussian", "benchmarks/gaussian/bench_native_gaussian.py", "--output"),
        ("bench-fock", "benchmarks/fock/bench_fock.py", "--out"),
        ("bench-gpu-gaussian", "benchmarks/gpu/bench_gpu_gaussian.py", "--output"),
        ("bench-gpu-fock", "benchmarks/gpu/bench_gpu_fock.py", "--output"),
    ]
    needs = {
        "bench-boson-sampling": ("matplotlib",),
        "bench-gbs": ("matplotlib",),
        "bench-gpu-gaussian": ("cupy",),
        "bench-gpu-fock": ("cupy",),
        "bench-fock": ("piquasso",),
    }
    for name, script, flag in bench:
        smoke = [script, "--smoke", flag, str(art / f"{name}-smoke.json")]
        full = [script, flag, str(art / f"{name}.json")]
        if name == "bench-gpu-gaussian":
            full += ["--dtypes", "float64", "float32"]
        result.append(
            Step(name + "-smoke", _py(*smoke), needs=needs.get(name, ()), claim="benchmark runs")
        )
        result.append(
            Step(
                name,
                _py(*full),
                needs=needs.get(name, ()),
                claim="performance tables",
                modes=("full",),
                timeout=7200,
            )
        )
    result.append(
        Step(
            "bench-graphix",
            _py("benchmarks/graphix/bench_graphix_mbqc.py", "--out-dir", str(art / "graphix")),
            needs=("graphix", "matplotlib"),
            claim="qubit-MBQC comparison with Graphix",
            modes=("full",),
        )
    )
    result.append(
        Step(
            "threshold-mp-reference",
            _py(
                "benchmarks/gaussian_boson_sampling/threshold_mp_reference.py",
                "24,12,0.9",
                "28,14,0.9",
                "32,16,0.9",
                "--output",
                str(art / "threshold_mp_reference.json"),
            ),
            needs=("mpmath", "piquasso"),
            claim="threshold-probability accuracy vs 50-digit reference",
            modes=("full",),
        )
    )
    for suite in ("paper_experiments", "paper_experiments_ml"):
        result.append(
            Step(
                f"{suite}-safe",
                _py("run_all_safe.py"),
                claim=f"{suite} light experiments",
                modes=("full",),
                timeout=7200,
                cwd=out / "suites" / suite,
                setup=_copy_suite(suite),
            )
        )
        result.append(
            Step(
                f"{suite}-compare",
                _py(
                    "scripts/compare_recorded_results.py",
                    str(out / "suites" / suite / "results" / "json"),
                    str(ROOT / suite / "results" / "json"),
                    "--atol",
                    "1e-12",
                    "--output",
                    str(art / f"{suite}-comparison.json"),
                ),
                claim=f"{suite}: regenerated values equal the recorded ones",
                modes=("full",),
            )
        )
    return result


def environment() -> dict:
    def version(mod: str) -> str | None:
        try:
            from importlib import metadata

            return metadata.version(mod)
        except Exception:
            return None

    info: dict = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {
            p: version(p)
            for p in (
                "photographiq",
                "numpy",
                "scipy",
                "networkx",
                "piquasso",
                "graphix",
                "mpmath",
                "jax",
                "cupy-cuda12x",
                "cupy-cuda13x",
                "mentpy",
                "matplotlib",
            )
        },
    }
    try:
        info["git_head"] = subprocess.run(
            ["git", "--no-optional-locks", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "--no-optional-locks", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        info["git_dirty"] = bool(dirty.strip())
    except (OSError, subprocess.CalledProcessError):
        info["git_head"], info["git_dirty"] = None, None
    try:
        from photographiq import native

        info["native"] = {"components": list(native.components()), **native.build_info()}
    except Exception as exc:  # report, never fail the record
        info["native"] = f"unavailable: {type(exc).__name__}: {exc}"
    return info


def missing(modules) -> list[str]:
    return [m for m in modules if importlib.util.find_spec(m) is None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("fast", "full"), default="fast")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only", nargs="+", help="run only these step names")
    parser.add_argument("--list", action="store_true", help="list steps and exit")
    args = parser.parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = (args.output or ROOT / "build" / "reproduction" / f"{args.mode}-{stamp}").resolve()
    plan = [s for s in steps(out) if args.mode in s.modes]
    if args.only:
        unknown = set(args.only) - {s.name for s in plan}
        if unknown:
            parser.error(f"unknown or out-of-mode steps: {sorted(unknown)}")
        plan = [s for s in plan if s.name in args.only]
    if args.list:
        for s in plan:
            print(f"{s.name:32s} needs={','.join(s.needs) or '-':22s} {s.claim}")
        return 0
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)
    summary = {"mode": args.mode, "output": str(out), "environment": environment(), "steps": []}
    for s in plan:
        record = {"name": s.name, "claim": s.claim, "command": s.command}
        absent = missing(s.needs)
        if absent:
            record |= {"status": "skipped", "reason": f"missing optional modules: {absent}"}
        else:
            if s.setup is not None:
                s.setup(out)
            log = out / "logs" / f"{s.name}.log"
            start = time.perf_counter()
            try:
                with log.open("w", encoding="utf-8") as stream:
                    code = subprocess.run(
                        s.command,
                        cwd=s.cwd or ROOT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=s.timeout,
                        env={**__import__("os").environ, "MPLBACKEND": "Agg"},
                    ).returncode
                status = "passed" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                code, status = None, "failed"
                record["reason"] = f"timeout after {s.timeout} s"
            record |= {
                "status": status,
                "exit_code": code,
                "seconds": round(time.perf_counter() - start, 1),
                "log": str(log),
            }
        summary["steps"].append(record)
        print(f"{record['status'].upper():8s} {s.name} {record.get('reason', '')}", flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failed = [r["name"] for r in summary["steps"] if r["status"] == "failed"]
    skipped = [r["name"] for r in summary["steps"] if r["status"] == "skipped"]
    print(
        f"\n{len(summary['steps'])} steps; failed: {failed or 'none'}; skipped: {skipped or 'none'}"
    )
    print(f"summary: {out / 'summary.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
