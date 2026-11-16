"""HYB-12: CPU runtime and memory of the hybrid executor (numpy-reference) vs native-fock.

Not a GPU or CUDA-Q measurement (HYB-11 is blocked). Workloads: the three-mode cubic-phase
pattern (pure) over per-mode dims 8..32 and the lossy Kerr step (mixed, two modes) over dims
8..24; native-fock / native-mixed-fock over total cutoffs with a comparable converged accuracy
are timed alongside. Peak traced allocation (tracemalloc) is compared with the planner's state
memory estimate. Timings: common.benchmark (1 warm-up, 3 repeats, median), this machine only.

Acceptance (methodological): the planner's estimate never exceeds the measured peak (the state
is the minimum live allocation), and every timing is finite. No speed-up is claimed.
"""

import platform
import tracemalloc
import warnings

from hyb_support import COLORS, common, kerr_wire, pg, plt, save, three_mode_cubic

from photographiq.hybrid.executor import run
from photographiq.hybrid.planner import plan

INPUT = {0: pg.GaussianInput((0.3, 0.1))}


def measure(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bench = common.benchmark(fn, warmup=1, repeats=3)
        tracemalloc.start()
        fn()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return bench["median_seconds"], bench["iqr_seconds"], peak


def main():
    rows = []
    cubic, out = three_mode_cubic(0.15), {0: 0.2, "m": -0.3}
    for d in (8, 12, 16, 20, 24, 28, 32):
        s, iqr, peak = measure(
            lambda d=d: run(cubic, dims=d, inputs=INPUT, measurement_outcomes=out)
        )
        rows.append(
            {
                "workload": "3-mode cubic (pure)",
                "engine": "hybrid numpy-reference",
                "size": d,
                "hilbert_dimension": d**3,
                "median_seconds": s,
                "iqr_seconds": iqr,
                "peak_bytes": peak,
                "planned_state_bytes": plan(cubic, d).peak_bytes,
            }
        )
    for c in (20, 26, 32, 40, 48):
        s, iqr, peak = measure(
            lambda c=c: pg.simulate(
                cubic, backend="native-fock", cutoff=c, inputs=INPUT, measurement_outcomes=out
            )
        )
        rows.append(
            {
                "workload": "3-mode cubic (pure)",
                "engine": "native-fock",
                "size": c,
                "hilbert_dimension": common.fock_dimension(3, c),
                "median_seconds": s,
                "iqr_seconds": iqr,
                "peak_bytes": peak,
            }
        )
    lossy = kerr_wire(0.1, losses=((0, 0.85), (1, 0.85)))
    for d in (8, 12, 16, 20, 24):
        s, iqr, peak = measure(
            lambda d=d: run(lossy, dims=d, inputs=INPUT, measurement_outcomes={0: 0.2})
        )
        rows.append(
            {
                "workload": "2-mode lossy Kerr (mixed)",
                "engine": "hybrid numpy-reference",
                "size": d,
                "hilbert_dimension": d**2,
                "median_seconds": s,
                "iqr_seconds": iqr,
                "peak_bytes": peak,
                "planned_state_bytes": plan(lossy, d).peak_bytes,
            }
        )
    for c in (18, 22, 26, 30):
        s, iqr, peak = measure(
            lambda c=c: pg.simulate(
                lossy,
                backend="native-mixed-fock",
                cutoff=c,
                inputs=INPUT,
                measurement_outcomes={0: 0.2},
            )
        )
        rows.append(
            {
                "workload": "2-mode lossy Kerr (mixed)",
                "engine": "native-mixed-fock",
                "size": c,
                "hilbert_dimension": common.fock_dimension(2, c),
                "median_seconds": s,
                "iqr_seconds": iqr,
                "peak_bytes": peak,
            }
        )
    for r in rows:
        print(
            r["workload"],
            r["engine"],
            r["size"],
            f"{r['median_seconds']:.3f}s {r['peak_bytes'] / 2**20:.1f} MiB",
            flush=True,
        )
    hyb = [r for r in rows if "planned_state_bytes" in r]
    checks = {
        "plan_is_lower_bound": all(r["planned_state_bytes"] <= r["peak_bytes"] for r in hyb),
        "finite_timings": all(r["median_seconds"] > 0 for r in rows),
    }
    status = "pass" if all(checks.values()) else "fail"
    pyplot = plt()
    fig, axes = pyplot.subplots(1, 2, figsize=(7.0, 2.5))
    for (workload, engine), colour, style in (
        (("3-mode cubic (pure)", "hybrid numpy-reference"), COLORS["hybrid"], "o-"),
        (("3-mode cubic (pure)", "native-fock"), COLORS["native"], "s-"),
        (("2-mode lossy Kerr (mixed)", "hybrid numpy-reference"), COLORS["hybrid"], "o--"),
        (("2-mode lossy Kerr (mixed)", "native-mixed-fock"), COLORS["native"], "s--"),
    ):
        sel = [r for r in rows if r["workload"] == workload and r["engine"] == engine]
        axes[0].loglog(
            [r["hilbert_dimension"] for r in sel],
            [r["median_seconds"] for r in sel],
            style,
            color=colour,
            label=f"{engine}, {workload}",
        )
        axes[1].loglog(
            [r["hilbert_dimension"] for r in sel],
            [r["peak_bytes"] / 2**20 for r in sel],
            style,
            color=colour,
        )
    axes[0].set(
        xlabel="Hilbert dimension", ylabel="Median seconds (CPU)", title="Runtime per trajectory"
    )
    axes[0].legend(fontsize=5)
    axes[1].set(xlabel="Hilbert dimension", ylabel="Peak traced MiB", title="Memory")
    save(
        rows,
        "HYB12_cpu_scaling",
        configuration={
            "machine": platform.processor() or platform.machine(),
            "engine": "numpy-reference (CPU)",
            "note": "CPU only; not CUDA-Q, not GPU",
        },
        validation={"status": status, "checks": checks},
        figure=fig,
        caption="CPU runtime and memory: hybrid executor (NumPy reference) vs native Fock engines. No GPU measurement.",
    )
    print(f"HYB12 status={status} {checks}")


if __name__ == "__main__":
    main()
