"""R36: Gaussian runtime scaling.

Benchmarks preparing and entangling an N-mode line resource
(``CVGraph.line(N)``) through the NumPy Gaussian backend and the Piquasso
Gaussian backend, with warm-up runs and repeated timing measurements
(median, IQR, std). Timing is a local observation of this machine, not a
comparative performance claim (docs/performance.md, docs/validation.md).

Upgrade migration (2026-11): the C++ ``native-gaussian`` engine is timed next
to NumPy ``gaussian`` and Piquasso ``piquasso``. Two workloads separate the
costs: ``prepare_entangle`` (state preparation + CZ entangling, the original
workload) and ``measure_feedforward`` (the same resource with every node but
the last q-homodyne measured and the next node displaced by the outcome);
their difference estimates the measurement + feed-forward cost. Before timing,
every engine's output is checked against NumPy ``gaussian`` (same seed: equal
records and moments, CORRECTNESS_TOLERANCE). ``--quick`` runs a short sweep
to prove the script; the default is the publication sweep. Timings carry
thread/build information and are provisional unless PGQ_IDLE_MACHINE=1.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import numpy as np  # noqa: E402
import metadata  # noqa: E402

import photographiq as pg  # noqa: E402

MODE_COUNTS = [10, 20, 50, 100, 200]


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
    MODE_COUNTS = [10, 20, 50]
BACKENDS = ("gaussian", "native-gaussian", "piquasso")
CORRECTNESS_TOLERANCE = 1e-8


def build_measured_pattern(n_modes: int) -> pg.Pattern:
    graph = pg.CVGraph.line(n_modes, squeezing=1.0, inputs=(0,))
    pattern = pg.Pattern(graph)
    for i in range(n_modes - 1):
        pattern.measure(i, pg.Homodyne.q(), key=f"m{i}")
        pattern.displace(i + 1, p=pg.Outcome(f"m{i}"))
    return pattern.validate()
def build_pattern(n_modes: int) -> pg.Pattern:
    graph = pg.CVGraph.line(n_modes, squeezing=1.0, inputs=(0,))
    return pg.Pattern(graph)


def main():
    plt = common.setup_style()
    rows = []
    for n_modes, workload in [(n, w) for n in MODE_COUNTS for w in ("prepare_entangle", "measure_feedforward")]:
        pattern = build_pattern(n_modes) if workload == "prepare_entangle" else build_measured_pattern(n_modes)
        reference = pg.simulate(pattern, backend="gaussian", seed=0)
        for backend in BACKENDS:
            check = pg.simulate(pattern, backend=backend, seed=0)
            difference = max(
                [abs(check.records[k] - reference.records[k]) for k in reference.records]
                + [float(np.abs(check.state.mean - reference.state.mean).max()),
                   float(np.abs(check.state.covariance - reference.state.covariance).max())]
            )
            if difference > CORRECTNESS_TOLERANCE:
                raise AssertionError(f"{backend} disagrees with gaussian on {workload} n={n_modes}: {difference}")
            warmup, repeats = (2, 9) if n_modes <= 50 else (1, 5)
            if QUICK:
                warmup, repeats = 1, 3
            timing = common.benchmark(
                lambda pattern=pattern, backend=backend: pg.simulate(pattern, backend=backend, seed=0),
                warmup=warmup, repeats=repeats,
            )
            rows.append(
                {
                    "workload": workload,
                    "correctness_difference_vs_gaussian": difference,
                    "modes": n_modes,
                    "commands": len(pattern.commands),
                    "backend": backend,
                    "median_seconds": timing["median_seconds"],
                    "std_seconds": timing["std_seconds"],
                    "iqr_seconds": timing["iqr_seconds"],
                    "min_seconds": timing["min_seconds"],
                    "max_seconds": timing["max_seconds"],
                    "repeats": timing["repeats"],
                    "warmup": timing["warmup"],
                }
            )
            print(f"  {workload} modes={n_modes} backend={backend}: median={timing['median_seconds']:.5f}s", flush=True)

    common.save_result(rows, "R36_gaussian_runtime", extra={
        "environment": metadata.collect(), "timing_environment": timing_environment(),
        "correctness_tolerance": CORRECTNESS_TOLERANCE})

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for backend, color in (("gaussian", "#24677b"), ("native-gaussian", "#7a3e9d"), ("piquasso", "#c66d27")):
        for workload, style in (("prepare_entangle", "o-"), ("measure_feedforward", "s--")):
            subset = [r for r in rows if r["backend"] == backend and r["workload"] == workload]
            ax.errorbar(
                [r["modes"] for r in subset], [r["median_seconds"] for r in subset],
                yerr=[r["iqr_seconds"] / 2 for r in subset], fmt=style, label=f"{backend}, {workload}", color=color, capsize=3,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of modes (line resource)")
    ax.set_ylabel("Median runtime (s)")
    ax.legend(fontsize=6.5)
    env = timing_environment()
    ax.set_title(f"Gaussian engine runtime -- timings {env['timing_status']}", fontsize=9.5)
    common.save_figure(fig, "R36_gaussian_runtime")
    plt.close(fig)

    def median(backend, workload, n):
        return next(r["median_seconds"] for r in rows if r["backend"] == backend and r["workload"] == workload and r["modes"] == n)

    big = MODE_COUNTS[-1]
    common.save_caption(
        "R36_gaussian_runtime",
        f"""
**R36: Gaussian engine runtime.** Median wall time (error bars: half the IQR) of one `simulate` call on an N-mode line
resource (r = 1) for N = {MODE_COUNTS}: preparation + CZ entangling only (circles) and with N - 1 q-homodynes and
outcome-dependent displacements (squares; the difference estimates measurement + feed-forward cost), on NumPy
`gaussian`, C++ `native-gaussian` and Piquasso `piquasso`. Before timing, each engine reproduced the NumPy records and
output moments for the same seed (max difference {max(r['correctness_difference_vs_gaussian'] for r in rows):.1e}). At
N = {big}: preparation {median('gaussian', 'prepare_entangle', big):.3g} s (gaussian), {median('native-gaussian', 'prepare_entangle', big):.3g} s
(native-gaussian), {median('piquasso', 'prepare_entangle', big):.3g} s (piquasso); with measurements
{median('gaussian', 'measure_feedforward', big):.3g} s, {median('native-gaussian', 'measure_feedforward', big):.3g} s,
{median('piquasso', 'measure_feedforward', big):.3g} s. Timings are {env['timing_status']} ({env['configuration']} configuration,
{env['os_cpu_count']} logical CPUs, OMP_NUM_THREADS={env['OMP_NUM_THREADS']}, native build
{env['native_build_info'].get('compiler')}, OpenMP {env['native_build_info'].get('openmp')}); local observations only. Source:
`paper_experiments/11_performance/36_gaussian_runtime.py`.
""",
    )

    common.print_summary("R36 Gaussian runtime scaling", mode_counts=MODE_COUNTS)


if __name__ == "__main__":
    main()
