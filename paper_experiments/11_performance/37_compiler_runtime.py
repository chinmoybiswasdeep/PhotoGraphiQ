"""R37: Compiler runtime scaling (with warm-up and repeated timing).

Benchmarks ``Circuit.compile`` alone (not execution) for a growing number of
alternating rotation/CZ logical gates on two modes, using warm-up runs and
repeated ``time.perf_counter`` measurements (median, IQR, std). Records
generated resource nodes, commands and measurements alongside timing.

Upgrade migration (2026-11): the cost of the pipeline is separated into
compilation (``Circuit.compile``), analytic channel extraction
(``gaussian_channel``) and execution of one trajectory on NumPy ``gaussian``
and C++ ``native-gaussian`` (same seed; records must agree, a correctness gate
before any timing is reported). The compiled map is checked against the
independently composed target (rotations about mode 0 and CZs, which do not
commute, are multiplied in order). ``--quick`` runs a short sweep to prove the
script; timings are provisional unless PGQ_IDLE_MACHINE=1.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import numpy as np  # noqa: E402
import metadata  # noqa: E402

import photographiq as pg  # noqa: E402

GATE_COUNTS = [1, 2, 5, 10, 20, 50, 100]
SQUEEZING = 1.0


QUICK = "--quick" in sys.argv[1:]


def timing_environment() -> dict:
    """Thread/hardware context for timings (the metadata companion adds CPU/GPU/packages)."""
    import os

    from photographiq import native

    return {
        "configuration": "quick" if QUICK else "full",
        # Timings are provisional unless the run was declared to be on an idle machine.
        "timing_status": "final (idle machine declared via PGQ_IDLE_MACHINE=1)"
        if os.environ.get("PGQ_IDLE_MACHINE") == "1"
        else "provisional (shared machine)",
        "os_cpu_count": os.cpu_count(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "native_build_info": native.build_info(),
        "native_components": list(native.components()),
    }

if QUICK:
    GATE_COUNTS = [1, 2, 5, 10, 20]
MAP_TOLERANCE = 1e-9  # relative
NATIVE_TOLERANCE = 1e-8


def independent_target(n_gates: int) -> np.ndarray:
    target = np.eye(4)
    for i in range(n_gates):
        if i % 2 == 0:
            t = 0.1 + 0.01 * i
            step = np.eye(4)
            step[:2, :2] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
        else:
            step = np.eye(4)
            step[1, 2] = step[3, 0] = 0.2 + 0.01 * i
        target = step @ target
    return target
def build_circuit(n_gates: int) -> pg.Circuit:
    circuit = pg.Circuit(2)
    for i in range(n_gates):
        if i % 2 == 0:
            circuit.rotate(i % 2, 0.1 + 0.01 * i)
        else:
            circuit.cz(0, 1, 0.2 + 0.01 * i)
    return circuit


def main():
    plt = common.setup_style()
    rows = []
    for n_gates in GATE_COUNTS:
        circuit = build_circuit(n_gates)
        warmup, repeats = (2, 9) if n_gates <= 20 else (1, 5)
        timing = common.benchmark(
            lambda circuit=circuit: circuit.compile(squeezing=SQUEEZING, return_trace=True),
            warmup=warmup, repeats=repeats,
        )
        pattern, trace = timing["result"]
        measurements = sum(1 for c in pattern.commands if isinstance(c, pg.Measure))
        quick_warm, quick_rep = (1, 3) if QUICK else (warmup, repeats)
        channel_timing = common.benchmark(lambda pattern=pattern: pg.gaussian_channel(pattern), warmup=quick_warm, repeats=quick_rep)
        map_error = common.relative_error(channel_timing["result"].matrix, independent_target(n_gates))
        runs = {b: pg.simulate(pattern, backend=b, seed=3) for b in ("gaussian", "native-gaussian")}
        record_difference = max(abs(runs["gaussian"].records[k] - runs["native-gaussian"].records[k]) for k in runs["gaussian"].records)
        if map_error > MAP_TOLERANCE or record_difference > NATIVE_TOLERANCE:
            raise AssertionError(f"n_gates={n_gates}: map error {map_error}, native record difference {record_difference}")
        execution = {
            b: common.benchmark(lambda pattern=pattern, b=b: pg.simulate(pattern, backend=b, seed=3), warmup=quick_warm, repeats=quick_rep)
            for b in ("gaussian", "native-gaussian")
        }
        rows.append(
            {
                "n_gates": n_gates,
                "median_compile_seconds": timing["median_seconds"],
                "std_compile_seconds": timing["std_seconds"],
                "iqr_compile_seconds": timing["iqr_seconds"],
                "repeats": timing["repeats"],
                "warmup": timing["warmup"],
                "resource_nodes": len(pattern.graph.nodes) - circuit.modes,
                "commands": len(pattern.commands),
                "measurements": measurements,
                "median_channel_seconds": channel_timing["median_seconds"],
                "median_gaussian_execution_seconds": execution["gaussian"]["median_seconds"],
                "median_native_gaussian_execution_seconds": execution["native-gaussian"]["median_seconds"],
                "relative_map_error": map_error,
                "native_vs_gaussian_record_difference": record_difference,
            }
        )
        print(f"  n_gates={n_gates}: median={timing['median_seconds']:.5f}s, nodes={rows[-1]['resource_nodes']}", flush=True)

    common.save_result(rows, "R37_compiler_runtime", extra={
        "environment": metadata.collect(), "timing_environment": timing_environment(),
        "map_tolerance": MAP_TOLERANCE, "native_tolerance": NATIVE_TOLERANCE})

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].errorbar(
        [r["n_gates"] for r in rows], [r["median_compile_seconds"] for r in rows],
        yerr=[r["iqr_compile_seconds"] / 2 for r in rows], fmt="o-", color="#24677b", capsize=3,
    )
    for key, label, color in (("median_channel_seconds", "gaussian_channel", "#627a36"),
                              ("median_gaussian_execution_seconds", "execute: gaussian", "#c66d27"),
                              ("median_native_gaussian_execution_seconds", "execute: native-gaussian", "#7a3e9d")):
        axes[0].plot([r["n_gates"] for r in rows], [r[key] for r in rows], "s--", color=color, label=label)
    axes[0].plot([], [], "o-", color="#24677b", label="compile")
    axes[0].legend(fontsize=7)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of logical gates")
    axes[0].set_ylabel("Median time (s)")

    axes[1].plot([r["n_gates"] for r in rows], [r["resource_nodes"] for r in rows], "o-", label="Resource nodes", color="#c66d27")
    axes[1].plot([r["n_gates"] for r in rows], [r["commands"] for r in rows], "s-", label="Commands", color="#627a36")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Number of logical gates")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Compile / analyse / execute cost -- timings {timing_environment()['timing_status']}")
    common.save_figure(fig, "R37_compiler_runtime")
    plt.close(fig)

    common.print_summary(
        "R37 compiler runtime scaling",
        gate_counts=GATE_COUNTS,
        max_relative_map_error=max(r["relative_map_error"] for r in rows),
        max_native_record_difference=max(r["native_vs_gaussian_record_difference"] for r in rows),
        timing_status=timing_environment()["timing_status"],
    )


if __name__ == "__main__":
    main()
