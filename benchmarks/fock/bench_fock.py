"""Native C++ vs Piquasso-backed Fock backends: per-gate and per-pattern runtime.

    python benchmarks/fock/bench_fock.py [--smoke | --quick] [--out PATH]

Timing protocol: the backend state is prepared once; each repetition restores a copy of
it OUTSIDE the timed region, then times one backend call (including the shared Python
renormalisation and diagnostics), and the prepared state is restored afterwards, so every
gate is timed on it (before 2026-11-11 each gate started from the previous gate's output).
``warmup`` untimed calls absorb JIT compilation (Piquasso uses numba) and caches.
Reported: median and interquartile range (s).
Memory of whole patterns: each runs in a fresh subprocess; ``peak_memory_bytes`` is that
process's peak working set (psutil ``peak_wset``: Windows only) and
``baseline_rss_bytes`` its RSS after imports and one warm-up run, so the difference bounds
the transient C++ and NumPy allocations of one run (interpreter + imports are included in
both numbers). psutil is optional: without it both memory fields and the psutil
environment fields are None.
Thread scaling sets OMP_NUM_THREADS in fresh subprocesses (OpenMP reads it at start-up).

Correctness, checked before anything is timed (a mismatch raises RuntimeError):
every gate and pattern case is run once, untimed, by the native engine and by its
Piquasso counterpart, and the trace distance of the two output states (plus, for
patterns, the relative difference of every recorded outcome probability or density) is
stored in each row as ``native_vs_piquasso_*`` next to its ``agreement_tol``:

* ``ROUNDING_TOL`` (1e-12) where both engines apply the same truncated operator: every
  single-mode gate, beam splitter (it conserves total photon number), thermal loss (same
  Kraus decomposition). Observed <= 1.3e-15 at every benchmark size, and <= 2.2e-15 for
  every single-mode gate and the beam splitter on number-state inputs up to cutoff 30.
* ``max(ROUNDING_TOL, sqrt(p_boundary))`` where the truncation schemes differ: the CZ
  (native: exact P U P; Piquasso: Euler factors, truncated after each) and whole patterns
  (CZ and conditioning chains). ``p_boundary`` is the largest probability with total
  photon number >= cutoff - 2 that either engine recorded in its diagnostics for the case.
  The engines then differ only in amplitude that crosses the cutoff, whose size scales as
  sqrt(p_boundary). This is an empirical scale, NOT a proven bound. Deviation /
  sqrt(p_boundary) for a single CZ (2026-11-11, pure and mixed engines): <= 0.047 over 358
  number-state inputs (gains 0.05-3, cutoffs 8-16) at the backends' default
  ``norm_tolerance`` (1e-3), which the benchmark uses; <= 0.098 over 240 squeezed-coherent
  inputs (gains 0.1-1, cutoffs 6-12) even at norm_tolerance 0.5; but 1.4 for a
  number-state input at norm_tolerance 0.9 (cutoff 16, (0.3|0> + |13>)|1>, g = 1, 10 %
  norm loss), so the tolerance is not valid for heavily truncated states. Trace distance bounds every absolute outcome-probability difference; the
  statistics are compared as relative differences, which is stricter for values below 1.
  Observed at every benchmark size (2026-11-11): deviation / tolerance <= 0.0098 (CZ),
  <= 0.015 (pattern states) and <= 0.024 (pattern statistics).
* Piquasso-backed loss above ``PIQUASSO_LOSS_MAX_DIMENSION`` is neither timed nor
  checked; those rows carry ``agreement_skipped``.
* Thread rows time the raw kernel on a random state (no Piquasso counterpart; the same CZ
  kernel is checked in the gate rows): each child returns a checksum of its output, and
  ``checksum_rel_diff_vs_first`` against the first thread count must be <= ROUNDING_TOL.

Environment metadata, the time stamp and JSON output come from ``benchmarks/bench_common``
(``environment``, ``rel_err``, ``stamp``, ``write_json``); ``bench_common.time_call`` is NOT used, because
its back-to-back inner loop cannot restore the state outside the timed region.
Results are JSON in benchmarks/fock/results/.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_common import environment, rel_err, stamp, write_json  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq import native  # noqa: E402
from photographiq.backends.native_fock import (  # noqa: E402
    NativeFockBackend,
    NativeMixedFockBackend,
)
from photographiq.fock_analysis import trace_distance  # noqa: E402

try:
    import psutil
except ImportError:  # memory and CPU-load fields become None
    psutil = None

HERE = Path(__file__).resolve().parent

CONFIGS = {
    "smoke": {
        "pure": [(2, 6)],
        "mixed": [(2, 6)],
        "gate_repeats": 3,
        "patterns": [("network", 3, 6)],
        "pattern_repeats": 2,
        "threads": (1, 2),
        "kernels": (("pure", 3, 8), ("mixed", 3, 6)),
        "convergence": (4, 6, 8),
    },
    "quick": {
        "pure": [(2, 10), (4, 10)],
        "mixed": [(2, 10)],
        "gate_repeats": 5,
        "patterns": [("wire", 4, 16)],
        "pattern_repeats": 5,
        "threads": (1, 2),
        "kernels": (("pure", 5, 20), ("mixed", 3, 16)),
        "convergence": (4, 6, 8, 10, 12, 16),
    },
    "full": {
        "pure": [(2, 10), (2, 30), (4, 10), (4, 20), (6, 10)],
        "mixed": [(1, 30), (2, 10), (2, 14), (3, 10)],
        "gate_repeats": 9,
        "patterns": [
            ("wire", 4, 16),
            ("wire", 8, 16),
            ("wire", 4, 30),
            ("network", 3, 10),
            ("network", 4, 10),
        ],
        "pattern_repeats": 5,
        "threads": (1, 2, 4, 8),
        "kernels": (("pure", 5, 20), ("mixed", 3, 16)),
        "convergence": (4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32),
    },
}

ROUNDING_TOL = 1e-12
# Gates whose truncated operator differs between the engines (see the module docstring).
TRUNCATION_SCHEME_GATES = frozenset({"cz"})


def environment_metadata() -> dict:
    return environment(
        native_fock_max_threads=native.require("fock").max_threads(),
        native_path=os.environ.get("PHOTOGRAPHIQ_NATIVE_PATH"),
        physical_cpus=None if psutil is None else psutil.cpu_count(logical=False),
        memory_bytes=None if psutil is None else psutil.virtual_memory().total,
        # Other processes share the machine; the system-wide load is recorded, not controlled.
        system_cpu_percent_at_start=None if psutil is None else psutil.cpu_percent(interval=1.0),
        timing="see benchmarks/fock/bench_fock.py docstring (restore outside timed region)",
    )


def stats(samples: list[float]) -> dict:
    q1, median, q3 = np.percentile(samples, [25, 50, 75])
    return {"median_s": float(median), "iqr_s": float(q3 - q1), "repeats": len(samples)}


def prepared(backend, modes: int):
    for m in range(modes):
        backend.prepare(m, 0.1)
        backend.displace(m, 0.15, -0.05)
    return backend


GATES = {
    "displace": lambda b: b.displace(0, 0.1, 0.05),
    "squeeze": lambda b: b.squeeze(0, 0.05),
    "rotate": lambda b: b.rotate(0, 0.3),
    "kerr": lambda b: b.kerr(0, 0.2),
    "cubic_phase": lambda b: b.cubic_phase(0, 0.01),
    "beamsplitter": lambda b: b.beamsplitter(0, 1, 0.4),
    "cz": lambda b: b.entangle(0, 1, 0.1),
    "quadratic_phase": lambda b: b.quadratic_phase(0, 0.1),
}


# MixedFockBackend.loss rebuilds its result through one Piquasso DensityMatrix instruction
# per nonzero element (dim^2 of them), so it is not timed above this dimension.
PIQUASSO_LOSS_MAX_DIMENSION = 105
SLOW_PIQUASSO_LOSS = "Piquasso-backed loss is too slow to run at this dimension"


def boundary_population(*states) -> float:
    return max(
        (d.get("boundary_population", 0.0) for s in states for d in s.diagnostics), default=0.0
    )


def agreement(label: str, ours, reference, scheme_differs: bool, statistics=None) -> dict:
    """Compare native vs Piquasso output states (and outcome statistics); raise on mismatch."""
    tol = ROUNDING_TOL
    if scheme_differs:
        tol = max(tol, float(np.sqrt(boundary_population(ours, reference))))
    result = {"native_vs_piquasso_trace_distance": trace_distance(ours, reference)}
    worst = result["native_vs_piquasso_trace_distance"]
    if statistics is not None:
        mine, theirs = statistics
        if mine.keys() != theirs.keys():
            raise RuntimeError(f"{label}: native and Piquasso record different outcomes")
        result["native_vs_piquasso_statistics_rel_diff"] = max(
            (rel_err(mine[k]["value"], theirs[k]["value"]) for k in mine),
            default=0.0,
        )
        worst = max(worst, result["native_vs_piquasso_statistics_rel_diff"])
    result["agreement_tol"] = tol
    if not worst <= tol:  # also rejects NaN
        raise RuntimeError(f"{label}: native disagrees with Piquasso ({worst:.3g} > {tol:.3g})")
    return result


def gate_outputs(backends: dict, gate) -> list:
    """Apply ``gate`` once to each backend's prepared state; return snapshots, restore state."""
    states = []
    for backend in backends.values():
        base = backend.native.copy()
        backend.retained_norms.clear()
        backend.diagnostics.clear()
        gate(backend)
        states.append(backend.get_state())
        backend.native = base
    return states


def time_gate(backend, gate, warmup: int, repeats: int) -> dict:
    base = backend.native.copy()
    samples = []
    for i in range(warmup + repeats):
        backend.native = base.copy()
        backend.retained_norms.clear()
        backend.diagnostics.clear()
        start = time.perf_counter()
        gate(backend)
        elapsed = time.perf_counter() - start
        if i >= warmup:
            samples.append(elapsed)
    backend.native = base  # every gate starts from the prepared state, not the last output
    return stats(samples)


def gate_benchmarks(cfg: dict) -> list[dict]:
    from photographiq.backends.fock import PiquassoFockBackend
    from photographiq.backends.mixed_fock import MixedFockBackend

    rows = []
    for kind, engines in (
        ("pure", (("native", NativeFockBackend), ("piquasso", PiquassoFockBackend))),
        ("mixed", (("native", NativeMixedFockBackend), ("piquasso", MixedFockBackend))),
    ):
        for modes, cutoff in cfg[kind]:
            dimension = comb(modes + cutoff - 1, modes)
            backends = {
                engine: prepared(cls(cutoff=cutoff, max_dimension=10**7), modes)
                for engine, cls in engines
            }
            gates = {
                n: g for n, g in GATES.items() if modes >= 2 or n not in ("beamsplitter", "cz")
            }
            if kind == "mixed":
                gates["thermal_loss"] = lambda b: b.loss(0, 0.8, 0.1)
            for name, gate in gates.items():
                case = {"kind": kind, "gate": name, "modes": modes, "cutoff": cutoff}
                slow = name == "thermal_loss" and dimension > PIQUASSO_LOSS_MAX_DIMENSION
                if slow:
                    check = {"agreement_skipped": SLOW_PIQUASSO_LOSS}
                else:
                    label = json.dumps(case)
                    scheme_differs = name in TRUNCATION_SCHEME_GATES
                    ours, reference = gate_outputs(backends, gate)
                    check = agreement(label, ours, reference, scheme_differs)
                for engine, backend in backends.items():
                    row = {"kind": kind, "engine": engine, "gate": name, "modes": modes}
                    row.update(cutoff=cutoff, dimension=dimension, **check)
                    if name == "thermal_loss":
                        if engine == "piquasso" and slow:
                            row["skipped"] = SLOW_PIQUASSO_LOSS
                        else:
                            row.update(time_gate(backend, gate, 1, 3))
                    else:
                        mixed_pq = engine == "piquasso" and kind == "mixed"
                        repeats = 3 if mixed_pq else cfg["gate_repeats"]
                        row.update(time_gate(backend, gate, warmup=2, repeats=repeats))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    return rows


def wire_pattern(length: int) -> pg.Pattern:
    """MBQC-style CZ wire: cat input teleported through `length` homodyne steps."""
    pattern = pg.Pattern(inputs=(0,))
    for i in range(1, length + 1):
        pattern.extend(
            [
                pg.Prepare(i, 0.3),
                pg.Entangle(i - 1, i, 0.6),
                pg.Measure(i - 1, pg.Homodyne(0.2), i - 1),
            ]
        )
    pattern.extend([pg.CubicPhase(length, 0.02), pg.Output((length,))])
    return pattern


def network_pattern(modes: int) -> pg.Pattern:
    """All modes alive: squeezers, beam-splitter ladder, CZ ladder, Kerr, photon counting."""
    pattern = pg.Pattern(inputs=tuple(range(modes)))
    pattern.extend([pg.Squeeze(m, 0.1) for m in range(modes)])
    pattern.extend([pg.BeamSplitter(m, m + 1, 0.5) for m in range(modes - 1)])
    pattern.extend([pg.Entangle(m, m + 1, 0.2) for m in range(modes - 1)])
    pattern.extend(
        [pg.Kerr(0, 0.1), pg.Measure(0, pg.PhotonNumber(), "n"), pg.Output(tuple(range(1, modes)))]
    )
    return pattern


PATTERNS = {"wire": wire_pattern, "network": network_pattern}
ENGINES = {
    "native": NativeFockBackend,
    "native-mixed": NativeMixedFockBackend,
}
REFERENCE = {"native": "piquasso", "native-mixed": "piquasso-mixed"}


def pattern_call(name: str, size: int, cutoff: int, engine: str):
    """Untimed set-up of one pattern run; returns the call that is timed."""
    cls: type
    if engine.startswith("piquasso"):
        from photographiq.backends.fock import PiquassoFockBackend
        from photographiq.backends.mixed_fock import MixedFockBackend

        cls = {"piquasso": PiquassoFockBackend, "piquasso-mixed": MixedFockBackend}[engine]
    else:
        cls = ENGINES[engine]
    pattern = PATTERNS[name](size)
    key = {"wire": {i: 0.0 for i in range(size)}, "network": {"n": 0}}[name]
    inputs = {0: pg.FockInput.cat(0.8, cutoff, 1)} if name == "wire" else None

    def call():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return pg.simulate(
                pattern,
                backend=cls(cutoff=cutoff, max_dimension=10**7),
                inputs=inputs,
                measurement_outcomes=key,
            )

    return call


def run_pattern(name: str, size: int, cutoff: int, engine: str) -> float:
    call = pattern_call(name, size, cutoff, engine)
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


def subprocess_measure(args: list[str], env: dict | None = None) -> dict:
    out = subprocess.run(
        [sys.executable, __file__, *args],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def child(name: str, size: int, cutoff: int, engine: str, repeats: int) -> None:
    run_pattern(name, size, cutoff, engine)  # warm-up (JIT, caches)
    baseline = None if psutil is None else int(psutil.Process().memory_info().rss)
    samples = [run_pattern(name, size, cutoff, engine) for _ in range(repeats)]
    peak = None if psutil is None else getattr(psutil.Process().memory_info(), "peak_wset", None)
    result = stats(samples)
    result["peak_memory_bytes"] = None if peak is None else int(peak)
    result["baseline_rss_bytes"] = baseline
    print(json.dumps(result))


def pattern_benchmarks(cfg: dict) -> list[dict]:
    rows = []
    for name, size, cutoff in cfg["patterns"]:
        modes = 2 if name == "wire" else size
        engines = ["native", "piquasso"]
        if comb(modes + cutoff - 1, modes) <= 300:
            engines += ["native-mixed", "piquasso-mixed"]
        checks = {}
        for ours in engines[::2]:
            a, b = (pattern_call(name, size, cutoff, e)() for e in (ours, REFERENCE[ours]))
            label = f"pattern {name} size={size} cutoff={cutoff} {ours}"
            stats_pair = (a.measurement_statistics, b.measurement_statistics)
            checks[ours] = checks[REFERENCE[ours]] = agreement(
                label, a.state, b.state, True, stats_pair
            )
        for engine in engines:
            repeats = 2 if engine.startswith("piquasso") else cfg["pattern_repeats"]
            row = {"pattern": name, "size": size, "cutoff": cutoff, "engine": engine}
            row.update(checks[engine])
            row.update(
                subprocess_measure(["--child", name, str(size), str(cutoff), engine, str(repeats)])
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def thread_benchmarks(cfg: dict) -> list[dict]:
    """Large-kernel scaling: CZ on a pure state and a density matrix."""
    rows: list[dict] = []
    first: dict[str, complex] = {}
    for threads in cfg["threads"]:
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))
        for kind, modes, cutoff in cfg["kernels"]:
            row = {
                "threads": threads,
                "kind": kind,
                "modes": modes,
                "cutoff": cutoff,
                "dimension": comb(modes + cutoff - 1, modes),
            }
            row.update(subprocess_measure(["--kernel", kind, str(modes), str(cutoff)], env))
            checksum = complex(*row.pop("checksum"))
            reference = first.setdefault(kind, checksum)
            row["checksum_rel_diff_vs_first"] = abs(checksum - reference) / abs(reference)
            row["agreement_tol"] = ROUNDING_TOL
            if not row["checksum_rel_diff_vs_first"] <= ROUNDING_TOL:
                raise RuntimeError(f"kernel {kind} output depends on the thread count: {row}")
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def kernel_child(kind: str, modes: int, cutoff: int) -> None:
    F = native.require("fock")
    rng = np.random.default_rng(0)
    dim = F.dimension(modes, cutoff)
    symplectic = np.eye(4)
    symplectic[1, 2] = symplectic[3, 0] = 0.3
    op = F.gaussian_unitary(symplectic, np.zeros(4), cutoff)
    if kind == "pure":
        state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        apply = F.apply_pure
    else:
        x = rng.normal(size=(dim, 2)) + 1j * rng.normal(size=(dim, 2))
        state = x @ x.conj().T
        apply = F.apply_mixed
    samples = []
    for i in range(7):
        start = time.perf_counter()
        apply(state, modes, cutoff, [1, 3 if modes > 3 else 2], op)
        if i >= 2:
            samples.append(time.perf_counter() - start)
    result = stats(samples)
    result["max_threads"] = F.max_threads()
    # The kernel works in place: a fixed projection of the final state, compared across
    # thread counts by the parent.
    probe = np.random.default_rng(1).normal(size=state.size)
    checksum = complex(np.vdot(probe, state.ravel()))
    result["checksum"] = [checksum.real, checksum.imag]
    print(json.dumps(result))


def convergence_pattern() -> pg.Pattern:
    """Two-mode Gaussian circuit with an exact phase-space reference."""
    return pg.Pattern(inputs=("a", "b")).extend(
        [
            pg.Squeeze("a", 0.4),
            pg.Displace("a", 0.5, 0.2),
            pg.Squeeze("b", -0.3),
            pg.BeamSplitter("a", "b", 0.7),
            pg.Output(("a", "b")),
        ]
    )


def convergence_benchmarks(cfg: dict) -> list[dict]:
    """Accuracy vs cutoff: Fock photon numbers against the exact Gaussian simulation.

    Not timed. The reference is ``GaussianBackend`` (no truncation), so the error is the
    truncation error of each engine: max over modes of |<n>_Fock - <n>_exact|, plus the
    product of the retained norms (1 - product is the population cut off) and the largest
    boundary population.
    """
    from photographiq.backends.fock import PiquassoFockBackend

    exact = pg.simulate(convergence_pattern(), backend="gaussian").state
    reference = {node: exact.photon_number(node) for node in ("a", "b")}
    rows = []
    for cutoff in cfg["convergence"]:
        for engine, cls in (("native", NativeFockBackend), ("piquasso", PiquassoFockBackend)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = pg.simulate(
                    convergence_pattern(), backend=cls(cutoff=cutoff, norm_tolerance=0.5)
                )
            error = max(abs(result.state.photon_number(n) - reference[n]) for n in reference)
            row = {
                "engine": engine,
                "cutoff": cutoff,
                "max_abs_photon_number_error": error,
                # product over operations of the norm kept before each renormalisation
                "retained_norm_product": float(np.prod(result.state.retained_norms)),
                "max_boundary_population": boundary_population(result.state),
                "reference": "GaussianBackend (exact, untruncated)",
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def run(config: str = "full", output: Path | None = None) -> dict:
    cfg = CONFIGS[config]
    report = {"config": config, "environment": environment_metadata()}
    report["gates"] = gate_benchmarks(cfg)
    report["patterns"] = pattern_benchmarks(cfg)
    report["threads"] = thread_benchmarks(cfg)
    report["convergence"] = convergence_benchmarks(cfg)
    out = output or HERE / "results" / f"fock-{platform.node()}-{stamp()}.json"
    write_json(report, out)
    print(f"wrote {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="tiny sizes (seconds)")
    group.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--child", nargs=5)
    parser.add_argument("--kernel", nargs=3)
    args = parser.parse_args()
    if args.child:
        name, size, cutoff, engine, repeats = args.child
        child(name, int(size), int(cutoff), engine, int(repeats))
        return
    if args.kernel:
        kind, modes, cutoff = args.kernel
        kernel_child(kind, int(modes), int(cutoff))
        return
    run("smoke" if args.smoke else "quick" if args.quick else "full", args.out)


if __name__ == "__main__":
    main()
