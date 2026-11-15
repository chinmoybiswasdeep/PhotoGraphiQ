"""R35: Fock-space dimension and memory model (analytical, cheap).

Uses the actual PhotoGraphiQ total-photon Fock dimension formula
D = binom(modes+cutoff-1, modes) (``PiquassoFockBackend.dimension``, cross-
checked against ``common.fock_dimension``) over a modes x cutoff grid, and
estimates pure-vector and dense-density-matrix byte costs. The default
``max_dimension`` allocation guard is read directly from the backend class.
Dense-density-matrix bytes are a rough dense upper-style estimate; the actual
mixed-Fock implementation may differ (see docs/performance.md).

Upgrade migration (2026-11): the dimension formula of the C++ engine
(``NativeFockBackend.dimension``) is checked against the same independent
formula, and a MEASURED part was added: wall time of one two-mode workload
(two coherent inputs, BeamSplitter, Kerr, CubicPhase) on ``native-fock``,
``piquasso-fock`` and ``native-mixed-fock`` over a cutoff sweep, separated into
state preparation (inputs only) and gate execution (difference), with the
retained norm recorded and the engines' final states compared (fidelity).
``--quick`` uses a short sweep and few repeats to prove the script works;
the default is the publication sweep. Timings carry
``timing_status`` (provisional unless PGQ_IDLE_MACHINE=1 declares an idle
machine) and thread/build information.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import photographiq as pg  # noqa: E402
from photographiq.backends.fock import PiquassoFockBackend  # noqa: E402

MODES = list(range(1, 21))
CUTOFFS = list(range(2, 41))


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

TIMING_CUTOFFS = [8, 16, 24] if QUICK else [8, 16, 24, 32, 48, 64]
TIMING_ENGINES = ("native-fock", "piquasso-fock", "native-mixed-fock")
MIXED_MAX_CUTOFF = 32  # dense density matrix: dimension^2 grows like cutoff^4 for two modes
TIMING_REPEATS = 2 if QUICK else 5
ENGINE_FIDELITY_FLOOR = 1 - 1e-6


def timing_rows():
    prepare = pg.Pattern(inputs=(0, 1)).append(pg.Output((0, 1))).validate()
    full = (
        pg.Pattern(inputs=(0, 1))
        .append(pg.BeamSplitter(0, 1, 0.4))
        .append(pg.Kerr(0, 0.2))
        .append(pg.CubicPhase(1, 0.05))
        .append(pg.Output((0, 1)))
        .validate()
    )
    inputs = {0: pg.GaussianInput.coherent(0.8 + 0.2j), 1: pg.GaussianInput.coherent(-0.5 + 0.4j)}
    rows = []
    for cutoff in TIMING_CUTOFFS:
        states = {}
        for engine in TIMING_ENGINES:
            if engine == "native-mixed-fock" and cutoff > MIXED_MAX_CUTOFF:
                continue
            prep = common.benchmark(lambda engine=engine: pg.simulate(prepare, backend=engine, cutoff=cutoff, inputs=inputs, seed=0),
                                    warmup=1, repeats=TIMING_REPEATS)
            run = common.benchmark(lambda engine=engine: pg.simulate(full, backend=engine, cutoff=cutoff, inputs=inputs, seed=0),
                                   warmup=1, repeats=TIMING_REPEATS)
            states[engine] = run["result"].state
            rows.append({
                "cutoff": cutoff, "backend": engine, "dimension": common.fock_dimension(2, cutoff),
                "preparation_median_seconds": prep["median_seconds"],
                "total_median_seconds": run["median_seconds"],
                "gate_execution_seconds_estimate": run["median_seconds"] - prep["median_seconds"],
                "total_iqr_seconds": run["iqr_seconds"], "repeats": TIMING_REPEATS,
                "min_retained_norm": min(run["result"].state.retained_norms),
            })
        for row in rows[-len(states):]:
            row["fidelity_to_native_fock"] = float(pg.fidelity(states[row["backend"]], states["native-fock"]))
    return rows
def default_max_dimension() -> int:
    return inspect.signature(PiquassoFockBackend.__init__).parameters["max_dimension"].default


def main():
    plt = common.setup_style()
    max_dimension = default_max_dimension()

    backend = PiquassoFockBackend(cutoff=CUTOFFS[0])
    rows = []
    for modes in MODES:
        for cutoff in CUTOFFS:
            backend.cutoff = cutoff
            dimension_native = backend.dimension(modes)
            dimension_independent = common.fock_dimension(modes, cutoff)
            if dimension_native != dimension_independent:
                raise AssertionError(f"Dimension formulas disagree at modes={modes}, cutoff={cutoff}")
            rows.append(
                {
                    "modes": modes,
                    "cutoff": cutoff,
                    "dimension": dimension_native,
                    "pure_vector_bytes": common.vector_bytes(dimension_native),
                    "dense_density_matrix_bytes_rough_estimate": common.density_bytes(dimension_native),
                    "exceeds_default_max_dimension": dimension_native > max_dimension,
                }
            )

    from photographiq.backends.native_fock import NativeFockBackend

    native_backend = NativeFockBackend(cutoff=CUTOFFS[0])
    native_mismatch = []
    for row in rows:
        native_backend.cutoff = row["cutoff"]
        if native_backend.dimension(row["modes"]) != row["dimension"]:
            native_mismatch.append((row["modes"], row["cutoff"]))
    if native_mismatch:
        raise AssertionError(f"Native dimension formula disagrees at {native_mismatch[:5]}")
    timing = timing_rows()
    common.save_csv(timing, "R35_fock_dimension_scaling_timing")
    timing_failures = [r for r in timing if r["fidelity_to_native_fock"] < ENGINE_FIDELITY_FLOOR]
    common.save_result(rows, "R35_fock_dimension_scaling", extra={
        "default_max_dimension": max_dimension, "native_dimension_formula_checked": True,
        "timing": timing, "timing_environment": timing_environment(), "engine_fidelity_floor": ENGINE_FIDELITY_FLOOR})
    if timing_failures:
        common.save_csv(timing_failures, "R35_fock_dimension_scaling_FAILURES")
        raise AssertionError("Fock engines disagree on the timed workload; see FAILURES csv")

    dims = np.array([[common.fock_dimension(m, c) for c in CUTOFFS] for m in MODES])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    mesh = axes[0].imshow(
        np.log10(dims), aspect="auto", origin="lower", cmap="viridis",
        extent=[CUTOFFS[0], CUTOFFS[-1], MODES[0], MODES[-1]],
    )
    axes[0].set_xlabel("Cutoff")
    axes[0].set_ylabel("Modes")
    axes[0].set_title(r"$\log_{10}$(Fock dimension)")
    fig.colorbar(mesh, ax=axes[0])

    for modes in (1, 2, 4, 8, 16, 20):
        axes[1].semilogy(CUTOFFS, [common.vector_bytes(common.fock_dimension(modes, c)) for c in CUTOFFS], "o-", label=f"{modes} modes", markersize=3)
    axes[1].axhline(common.vector_bytes(max_dimension), color="red", linestyle="--", label="default max_dimension")
    axes[1].set_xlabel("Cutoff")
    axes[1].set_ylabel("Pure state-vector bytes")
    axes[1].legend(fontsize=6)
    for engine, style, color in zip(TIMING_ENGINES, ("o-", "x--", "s:"), ("#24677b", "#c66d27", "#7a3e9d"), strict=True):
        subset = [r for r in timing if r["backend"] == engine]
        axes[2].loglog([r["dimension"] for r in subset], [r["total_median_seconds"] for r in subset], style, color=color,
                       label=f"{engine} total")
        axes[2].loglog([r["dimension"] for r in subset], [r["preparation_median_seconds"] for r in subset], style,
                       color=color, alpha=0.35, label=f"{engine} preparation")
    axes[2].set_xlabel("Two-mode Fock dimension")
    axes[2].set_ylabel("Median wall time (s)")
    axes[2].set_title(f"Measured -- timings {timing_environment()['timing_status']}", fontsize=9)
    axes[2].legend(fontsize=6)
    fig.suptitle("Fock-space dimension and memory scaling (D = binom(modes+cutoff-1, modes))")
    common.save_figure(fig, "R35_fock_dimension_scaling")
    plt.close(fig)
    largest = max(r["cutoff"] for r in timing)
    at = {r["backend"]: r for r in timing if r["cutoff"] == largest}
    common.save_caption(
        "R35_fock_dimension_scaling",
        f"""
**R35: Fock-space dimension, memory and measured cost.** Left: log10 of the total-photon Fock dimension
D = binom(modes + cutoff - 1, modes) for 1-{MODES[-1]} modes and cutoffs {CUTOFFS[0]}-{CUTOFFS[-1]} (the formula of both
`piquasso-fock` and `native-fock` equals the independent binomial at every grid point); centre: bytes of one complex128
state vector, with the default allocation guard max_dimension = {max_dimension} (red). Right: measured median wall
time of a two-mode workload (two coherent inputs, BeamSplitter, Kerr, CubicPhase) versus dimension, total (solid) and
state preparation only (faint), on `native-fock`, `piquasso-fock` and `native-mixed-fock` (density matrix, cutoff <=
{MIXED_MAX_CUTOFF}); at cutoff {largest}: {', '.join(f"{b} {r['total_median_seconds']:.3g} s" for b, r in at.items())}.
The engines' final states agree to fidelity >= {min(r['fidelity_to_native_fock'] for r in timing):.10f}. Timings are
{timing_environment()['timing_status']} ({timing_environment()['configuration']} configuration, {TIMING_REPEATS} repeats,
{timing_environment()['os_cpu_count']} logical CPUs, native build {timing_environment()['native_build_info'].get('compiler')},
OpenMP {timing_environment()['native_build_info'].get('openmp')}); they are local observations, not cross-machine claims.
Source: `paper_experiments/11_performance/35_fock_dimension_scaling.py`.
""",
    )

    common.print_summary(
        "R35 Fock-space dimension and memory model",
        grid_points=len(rows),
        default_max_dimension=max_dimension,
        max_dimension_in_grid=max(r["dimension"] for r in rows),
        n_exceeding_default_guard=sum(r["exceeds_default_max_dimension"] for r in rows),
        timing={f"{r['backend']}@{r['cutoff']}": round(r["total_median_seconds"], 6) for r in timing},
        timing_status=timing_environment()["timing_status"],
    )


if __name__ == "__main__":
    main()
