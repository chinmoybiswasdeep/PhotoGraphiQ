"""R44: Native Gaussian backend performance, measured together with accuracy.

Scientific question: what does the compiled ``native-gaussian`` backend (in-place O(n k)
local updates) buy over the NumPy ``gaussian`` backend (dense O(n^3) S V S^T per gate) and
over independent implementations (a dense NumPy symplectic propagation written here, and
Piquasso 8.0.1's GaussianSimulator executing the whole circuit as one program), as a function
of mode count, gate count, number of conditional measurements, batch size and OpenMP thread
count, and at what accuracy?

Model: seeded random circuits of rotations, squeezers, beam splitters, CZ, shears,
displacements and thermal loss (generator of R43) on a random mixed n-mode Gaussian state;
sequences of postselected homodyne measurements; batched symplectic updates and batched
homodyne conditioning (``photographiq.backends.native_gaussian.batch_*``) against a vectorised
NumPy dense update / Schur complement written here.

Measurement method: every configuration runs in a fresh subprocess (so peak memory is per
configuration and OMP_NUM_THREADS takes effect). Runtime: ``common.benchmark`` (warm-up
excluded, median and IQR of repeats). Memory: peak process working set from
``psutil.Process().memory_info().peak_wset`` (Windows; ``ru_maxrss`` elsewhere), reported as
absolute peak and as the increase over the peak just before the timed section (0 means the
run stayed below the import-time peak). tracemalloc is NOT used (it misses C++ allocations).
Gate timings exclude the O(n^3) physicality validation of state construction (an unvalidated
copy of a validated start state is loaded each repeat); the NumPy backend's own per-gate
validation is part of its cost and is included. Accuracy: max relative covariance error of
the final state against the dense first-principles reference of R43.

Threads: the native Gaussian single-trajectory kernels are serial; only the batched kernels
use OpenMP (cpp/src/gaussian/batch.cpp). OMP_NUM_THREADS is honoured (checked:
``_native.gaussian.max_threads`` follows it); it also limits OpenBLAS/MKL threads of NumPy in
the same process, which is stated with the results.

Timings are labelled final only when the run declares an idle machine (``PGQ_IDLE_MACHINE=1``);
otherwise they are marked provisional. The manuscript configuration is ``--full``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
import common  # noqa: E402

SEED = 20261144
# ----- acceptance criteria (declared before evaluation) -----------------------------------
TOL_ACCURACY_REL = 1e-10  # every implementation vs the dense reference (R43 measured <= 2e-12)
TOL_BATCH_REL = 1e-12  # native batch kernels vs the vectorised NumPy formulas, per entry relative
MAX_CONFIG_SECONDS = 600  # a single configuration may not hang the run


def _load_r43():
    spec = importlib.util.spec_from_file_location("r43", HERE.parent / "43_native_gaussian_validation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def peak_memory():
    """(peak bytes, source) of this process."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        if hasattr(info, "peak_wset"):
            return int(info.peak_wset), "psutil peak_wset (Windows peak working set)"
    except ImportError:
        pass
    try:
        import resource  # POSIX; ru_maxrss is in KiB on Linux

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, "ru_maxrss"
    except ImportError:
        return -1, "unavailable"


def unvalidated(state_cls, mean, cov, nodes):
    st = object.__new__(state_cls)
    st.mean, st.covariance, st.nodes = mean.copy(), cov.copy(), nodes
    return st


# =============================== worker (one configuration) ================================
def worker(cfg):
    import photographiq as pg
    from photographiq import registry
    from photographiq.backends import native_gaussian as ng
    from photographiq.states import GaussianState

    r43 = _load_r43()
    rng = np.random.default_rng([SEED, cfg["n"], cfg.get("gates", 0), cfg.get("batch", 0)])
    n, impl = cfg["n"], cfg["impl"]
    out = {}
    if cfg["kind"] == "circuit":
        mean0, cov0, _ = r43.random_state(rng, n)
        ops = r43.random_circuit(rng, n, cfg["gates"], r43.GATES_NOISY)
        start = GaussianState(mean0, cov0, tuple(range(n)))
        if impl in ("native", "numpy-backend"):
            engine = registry.create_backend("native-gaussian" if impl == "native" else "gaussian")

            def run():
                engine.reset(0)
                engine.state = unvalidated(GaussianState, mean0, cov0, start.nodes)
                r43.apply_backend(engine, ops)
                return engine

            final = lambda e: (e.get_state().mean, e.get_state().covariance)  # noqa: E731
        elif impl == "dense":
            ref = r43.ref
            mats = []
            for kind, modes, value in ops:
                if kind == "displace":
                    mats.append(("d", modes, np.asarray(value)))
                elif kind == "loss":
                    x, y = ref.loss_xy(*value)
                    mats.append(("l", modes, (x, y)))
                else:
                    gen = {"rotate": ref.gen_rotation, "squeeze": ref.gen_squeeze,
                           "beamsplitter": ref.gen_beamsplitter, "cz": ref.gen_cz,
                           "quadratic": ref.gen_quadratic}[kind](value)
                    mats.append(("s", modes, ref.from_generator(gen)))

            def run():
                mean, cov = mean0.copy(), cov0.copy()
                for tag, modes, m in mats:
                    if tag == "d":
                        mean[2 * modes[0]:2 * modes[0] + 2] += m
                    elif tag == "l":
                        x = ref.embed(m[0], modes, n)
                        cov = x @ cov @ x.T
                        cov[2 * modes[0]:2 * modes[0] + 2, 2 * modes[0]:2 * modes[0] + 2] += m[1]
                        mean = x @ mean
                    else:
                        s = ref.embed(m, modes, n)
                        mean, cov = s @ mean, s @ cov @ s.T
                return mean, cov

            final = lambda r: r  # noqa: E731
        elif impl == "piquasso":
            import piquasso as pq

            def program():
                with pq.Program() as prog:
                    for kind, modes, value in ops:
                        q = pq.Q(*modes)
                        if kind == "rotate":
                            q | pq.Phaseshifter(phi=value)
                        elif kind == "squeeze":
                            q | pq.Squeezing(r=value)
                        elif kind == "beamsplitter":
                            q | pq.Beamsplitter(theta=value, phi=0.0)
                        elif kind == "cz":
                            q | pq.ControlledZ(s=value)
                        elif kind == "quadratic":
                            q | pq.QuadraticPhase(s=value)
                        elif kind == "displace":
                            alpha = complex(*value) / 2
                            q | pq.Displacement(r=abs(alpha), phi=np.angle(alpha))
                        else:
                            q | pq.Attenuator(theta=np.arccos(np.sqrt(value[0])),
                                              mean_thermal_excitation=value[1])
                return prog

            prog = program()
            sim = pq.GaussianSimulator(d=n, config=pq.Config(hbar=2.0))

            def run():
                st = pq.GaussianState(d=n, connector=pq.NumpyConnector(), config=pq.Config(hbar=2.0))
                st.xpxp_mean_vector = mean0.copy()
                st.xpxp_covariance_matrix = 2 * cov0
                return sim.execute(prog, initial_state=st).state

            final = lambda s: (np.asarray(s.xpxp_mean_vector), np.asarray(s.xpxp_covariance_matrix) / 2)  # noqa: E731
        else:
            raise ValueError(impl)
        before, source = peak_memory()
        timing = common.benchmark(run, warmup=cfg["warmup"], repeats=cfg["repeats"])
        after, _ = peak_memory()
        m_fin, v_fin = final(timing["result"])
        m_ref, v_ref, _ = r43.reference_evolution(mean0.copy(), cov0.copy(), ops, n)
        out["accuracy_rel"] = float(max(np.abs(v_fin - v_ref).max() / np.abs(v_ref).max(),
                                        np.abs(m_fin - m_ref).max() / max(1.0, np.abs(m_ref).max())))
    elif cfg["kind"] == "measure":
        mean0, cov0, _ = r43.random_state(rng, n)
        start = GaussianState(mean0, cov0, tuple(range(n)))
        name = "native-gaussian" if impl == "native" else "gaussian"
        engine = registry.create_backend(name)
        theta, outcome = 0.3, 0.2

        def run():
            engine.reset(0)
            engine.state = unvalidated(GaussianState, mean0, cov0, start.nodes)
            for k in range(cfg["count"]):
                engine.measure(k, pg.Homodyne(theta), theta, outcome=outcome)
            return engine

        before, source = peak_memory()
        timing = common.benchmark(run, warmup=cfg["warmup"], repeats=cfg["repeats"])
        after, _ = peak_memory()
        st = timing["result"].get_state()
        m_ref, v_ref = mean0.copy(), cov0.copy()
        for _ in range(cfg["count"]):  # mode 0 of the shrinking state, precision-matrix reference
            m_ref, v_ref, _ = r43.ref._homodyne_reference(m_ref, v_ref, 0, theta, 1.0, 0.0, outcome)
        out["accuracy_rel"] = float(max(np.abs(st.covariance - v_ref).max() / np.abs(v_ref).max(),
                                        np.abs(st.mean - m_ref).max() / max(1.0, np.abs(m_ref).max())))
    elif cfg["kind"] == "batch":
        batch = cfg["batch"]
        dim = 2 * n
        means = rng.normal(size=(batch, dim))
        covs = np.empty((batch, dim, dim))
        for b in range(batch):
            _, covs[b], _ = r43.random_state(rng, n)
        u, v = 2, n - 3
        s_local = r43.ref.from_generator(r43.ref.gen_beamsplitter(0.7))
        s_full = r43.ref.embed(s_local, (u, v), n)
        angles = rng.uniform(0, np.pi, batch)
        outcomes = rng.normal(size=batch)

        if impl == "native":
            def run():
                m, c = ng.batch_apply_symplectic(means, covs, [u, v], s_local)
                return ng.batch_condition_homodyne(m, c, 0, angles, outcomes)
        else:
            def run():
                m = means @ s_full.T
                c = s_full @ covs @ s_full.T
                h = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # record = h . x_0
                var = np.einsum("bi,bij,bj->b", h, c[:, :2, :2], h)
                cross = np.einsum("bij,bj->bi", c[:, 2:, :2], h)  # Cov(rest, record)
                shift = outcomes - np.einsum("bi,bi->b", h, m[:, :2])
                m_out = m[:, 2:] + cross * (shift / var)[:, None]
                c_out = c[:, 2:, 2:] - np.einsum("bi,bj->bij", cross, cross) / var[:, None, None]
                dens = np.exp(-shift**2 / (2 * var)) / np.sqrt(2 * np.pi * var)
                return m_out, c_out, dens

        before, source = peak_memory()
        timing = common.benchmark(run, warmup=cfg["warmup"], repeats=cfg["repeats"])
        after, _ = peak_memory()
        np.save(cfg["save"], np.concatenate([timing["result"][0].ravel(), timing["result"][1].ravel(),
                                             timing["result"][2].ravel()]))
        out["accuracy_rel"] = float("nan")  # compared across implementations by the parent
    else:
        raise ValueError(cfg["kind"])
    out.update({
        "median_seconds": timing["median_seconds"], "iqr_seconds": timing["iqr_seconds"],
        "min_seconds": timing["min_seconds"], "repeats": timing["repeats"], "warmup": timing["warmup"],
        "peak_working_set_bytes": after, "peak_increase_bytes": max(0, after - before),
        "memory_source": source, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "native_max_threads": __import__("photographiq.native", fromlist=["x"]).require("gaussian").max_threads,
    })
    print("RESULT " + json.dumps(out))


# ===================================== parent =============================================
def run_config(cfg, threads=None):
    env = dict(os.environ)
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, str(HERE), "--worker", json.dumps(cfg)], env=env,
                          capture_output=True, text=True, timeout=MAX_CONFIG_SECONDS)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    if proc.returncode or not lines:
        raise RuntimeError(f"worker failed for {cfg}: {proc.stderr[-2000:]}")
    row = {**{k: v for k, v in cfg.items() if k != "save"}, **json.loads(lines[-1][7:]),
           "threads_requested": threads, "wall_seconds_incl_startup": time.perf_counter() - start}
    print(f"  {cfg['kind']:7s} {cfg['impl']:13s} n={cfg['n']:3d} "
          + " ".join(f"{k}={cfg[k]}" for k in ("gates", "count", "batch") if k in cfg)
          + (f" threads={threads}" if threads else "")
          + f": median {row['median_seconds']:.3e}s  acc {row['accuracy_rel']:.1e}"
          + f"  peak+{row['peak_increase_bytes'] / 2**20:.1f} MiB", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--full", action="store_true", help="publication sweep")
    parser.add_argument("--repeats", type=int, default=None)
    args = parser.parse_args()
    if args.worker:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            worker(json.loads(args.worker))
        return

    full = args.full
    repeats = args.repeats or (11 if full else 5)
    warmup = 2
    modes = [4, 8, 16, 32, 64, 128, 256] if full else [4, 16, 64]
    gate_counts = [50, 200, 800, 3200] if full else [50, 400]
    measure_counts = [1, 4, 16, 31] if full else [1, 8]
    batches = [1, 16, 256, 4096] if full else [16, 1024]
    threads = [1, 2, 4, 8, 12] if full else [1, 4]
    impls = ["native", "numpy-backend", "dense", "piquasso"]
    scratch = Path(tempfile.mkdtemp(prefix="pgq_r44_"))  # worker outputs compared by the parent

    rows = []
    print("mode sweep (200 gates)")
    for n in modes:
        for impl in impls:
            if impl == "piquasso" and n > 128:
                continue  # ponytail: Piquasso program at 256 modes exceeds the per-config time budget here
            rows.append(run_config({"kind": "circuit", "sweep": "modes", "impl": impl, "n": n, "gates": 200,
                                    "warmup": warmup, "repeats": repeats}))
    print("gate-count sweep (n = 32)")
    for g in gate_counts:
        for impl in impls:
            rows.append(run_config({"kind": "circuit", "sweep": "gates", "impl": impl, "n": 32, "gates": g,
                                    "warmup": warmup, "repeats": repeats}))
    print("measurement sweep (n = 32)")
    for m in measure_counts:
        for impl in ("native", "numpy-backend"):
            rows.append(run_config({"kind": "measure", "sweep": "measurements", "impl": impl, "n": 32,
                                    "count": m, "warmup": warmup, "repeats": repeats}))
    print("batch sweep (n = 16, BS + homodyne conditioning per state)")
    batch_pairs = []
    for b in batches:
        pair = {}
        for impl in ("native", "numpy-vectorised"):
            save = str(scratch / f"batch_{b}_{impl}.npy")
            rows.append(run_config({"kind": "batch", "sweep": "batch", "impl": impl, "n": 16, "batch": b,
                                    "warmup": warmup, "repeats": repeats, "save": save}, threads=1))
            pair[impl] = np.load(save)
        diff = np.abs(pair["native"] - pair["numpy-vectorised"]) / np.maximum(1.0, np.abs(pair["numpy-vectorised"]))
        batch_pairs.append({"batch": b, "native_vs_numpy_rel": float(diff.max())})
        rows[-1]["accuracy_rel"] = rows[-2]["accuracy_rel"] = float(diff.max())
    print("thread sweep (batch 4096 if full else 1024, n = 16)")
    big = batches[-1]
    for t in threads:
        for impl in ("native", "numpy-vectorised"):
            save = str(scratch / f"threads_{t}_{impl}.npy")
            row = run_config({"kind": "batch", "sweep": "threads", "impl": impl, "n": 16, "batch": big,
                              "warmup": warmup, "repeats": repeats, "save": save}, threads=t)
            rows.append(row)
        a, b = np.load(str(scratch / f"threads_{t}_native.npy")), np.load(str(scratch / f"threads_{t}_numpy-vectorised.npy"))
        rows[-1]["accuracy_rel"] = rows[-2]["accuracy_rel"] = float((np.abs(a - b) / np.maximum(1.0, np.abs(b))).max())

    failures = [r for r in rows if r["kind"] != "batch" and not r["accuracy_rel"] <= TOL_ACCURACY_REL]
    failures += [r for r in rows if r["kind"] == "batch" and not r["accuracy_rel"] <= TOL_BATCH_REL]
    from photographiq import native

    extra = {
        "backend": "native-gaussian vs NumPy 'gaussian' backend vs dense NumPy symplectic (this script) vs "
                   "Piquasso GaussianSimulator (one program per circuit)",
        "precision": "float64",
        "cutoff": None,
        "seed": SEED,
        "full": full,
        "timing_status": common.timing_label(),
        "parameters": {"modes": modes, "gate_counts": gate_counts, "measure_counts": measure_counts,
                       "batches": batches, "threads": threads, "repeats": repeats, "warmup": warmup},
        "native_build_info": native.build_info(),
        "memory_method": "per-configuration subprocess; psutil peak_wset (Windows) or ru_maxrss",
        "thread_note": "OMP_NUM_THREADS set for the whole worker process (native OpenMP and NumPy BLAS)",
        "batch_cross_check": batch_pairs,
        "acceptance": {"TOL_ACCURACY_REL": TOL_ACCURACY_REL, "TOL_BATCH_REL": TOL_BATCH_REL},
    }
    common.save_result(rows, "R44_native_gaussian_performance", extra=extra)
    if failures:
        common.save_csv(failures, "R44_native_gaussian_performance_FAILURES")

    plt = common.setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    colors = {"native": "#24677b", "numpy-backend": "#c66d27", "dense": "#627a36", "piquasso": "#8a3b8f",
              "numpy-vectorised": "#c66d27"}

    def series(ax, sweep, xkey, ykey, impl_list, kind="circuit", err=True):
        for impl in impl_list:
            sub = [r for r in rows if r["sweep"] == sweep and r["impl"] == impl]
            if not sub:
                continue
            y = [max(r[ykey], 1e-17) for r in sub]
            ax.errorbar([r[xkey] for r in sub], y, yerr=[r["iqr_seconds"] / 2 for r in sub] if err else None,
                        fmt="o-", color=colors[impl], label=impl, capsize=2)

    series(axes[0, 0], "modes", "n", "median_seconds", impls)
    axes[0, 0].set(xscale="log", yscale="log", xlabel="modes n (200 gates)", ylabel="median runtime (s)",
                   title="(a) runtime vs modes")
    series(axes[0, 1], "gates", "gates", "median_seconds", impls)
    axes[0, 1].set(xscale="log", yscale="log", xlabel="gates (n = 32)", ylabel="median runtime (s)",
                   title="(b) runtime vs gate count")
    series(axes[0, 2], "measurements", "count", "median_seconds", ["native", "numpy-backend"])
    axes[0, 2].set(xscale="log", yscale="log", xlabel="sequential homodyne measurements (n = 32)",
                   ylabel="median runtime (s)", title="(c) runtime vs measurements")
    series(axes[1, 0], "modes", "n", "accuracy_rel", impls, err=False)
    axes[1, 0].axhline(TOL_ACCURACY_REL, color="k", ls="--", lw=1, label="acceptance")
    axes[1, 0].set(xscale="log", yscale="log", xlabel="modes n", ylabel="max rel. error vs dense ref.",
                   title="(d) accuracy alongside (a)")
    series(axes[1, 1], "batch", "batch", "median_seconds", ["native", "numpy-vectorised"])
    axes[1, 1].set(xscale="log", yscale="log", xlabel="batch size B (n = 16, 1 thread)",
                   ylabel="median runtime (s)", title="(e) batched BS + homodyne")
    for impl in ("native", "numpy-vectorised"):
        sub = [r for r in rows if r["sweep"] == "threads" and r["impl"] == impl]
        axes[1, 2].errorbar([r["threads_requested"] for r in sub], [r["median_seconds"] for r in sub],
                            yerr=[r["iqr_seconds"] / 2 for r in sub], fmt="o-", color=colors[impl], label=impl,
                            capsize=2)
    axes[1, 2].set(xlabel="OMP_NUM_THREADS", ylabel="median runtime (s)", yscale="log",
                   title=f"(f) threads, batch {big}")
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    for ax, ticks in ((axes[0, 0], modes), (axes[0, 1], gate_counts), (axes[0, 2], measure_counts),
                      (axes[1, 0], modes), (axes[1, 1], batches)):
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[1, 2].set_xticks(threads)
    for ax in axes.ravel():
        ax.legend(fontsize=7)
    fig.suptitle("Native Gaussian performance with accuracy" + ("" if common.timing_final() else " (PROVISIONAL timings)"), fontsize=11)
    common.save_figure(fig, "R44_native_gaussian_performance")
    plt.close(fig)

    def med(sweep, impl, key, value):
        return next(r["median_seconds"] for r in rows if r["sweep"] == sweep and r["impl"] == impl and r[key] == value)

    nmax = max(n for n in modes if any(r["impl"] == "piquasso" and r["n"] == n for r in rows))
    speed_np = med("modes", "numpy-backend", "n", modes[-1]) / med("modes", "native", "n", modes[-1])
    speed_pq = med("modes", "piquasso", "n", nmax) / med("modes", "native", "n", nmax)
    t1 = med("threads", "native", "threads_requested", threads[0])
    tk = med("threads", "native", "threads_requested", threads[-1])
    worst_acc = max(r["accuracy_rel"] for r in rows if r["kind"] != "batch")
    worst_batch = max(r["accuracy_rel"] for r in rows if r["kind"] == "batch")
    common.save_caption(
        "R44_native_gaussian_performance",
        f"""
**R44 - native Gaussian performance with accuracy.** Timings: {common.timing_label()}.
Median runtime (error bars: IQR/2; {warmup} warm-up runs excluded, {repeats} repeats) of seeded random
noisy Gaussian circuits (R43 gate set incl. thermal loss) on a random mixed n-mode state for the
compiled `native-gaussian` backend, the NumPy `gaussian` backend (dense S V S^T and per-gate
validation), a dense NumPy symplectic propagation written in the script, and Piquasso 8.0.1's
GaussianSimulator executing the whole circuit as one program. (a) n = {modes[0]}..{modes[-1]} at 200 gates:
native is {speed_np:.0f}x faster than the NumPy backend at n = {modes[-1]} and {speed_pq:.0f}x faster than
Piquasso at n = {nmax}. (b) gate count at n = 32. (c) sequential postselected homodyne measurements at
n = 32. (d) accuracy of the same runs against the first-principles reference (worst {worst_acc:.1e};
acceptance {TOL_ACCURACY_REL:g}; the script's dense propagation uses the same gate matrices as the
reference, so its ~1e-16 is a consistency floor, not independent evidence). (e) batched beam splitter + homodyne conditioning on n = 16 states
(native OpenMP kernels vs vectorised NumPy; agreement {worst_batch:.1e}), one thread. (f) thread count
(OMP_NUM_THREADS, applied to the whole process incl. NumPy BLAS): native batch {big} takes {t1:.2e} s on
{threads[0]} and {tk:.2e} s on {threads[-1]} threads. Single-trajectory native kernels are serial.
Peak memory per configuration (psutil peak working set in a fresh subprocess) is in the CSV.
Seed {SEED}. Source: R44 (14_native_gaussian/44_native_gaussian_performance.py). Limitation: timings
are single-machine observations{'' if common.timing_final() else '; this run was not declared idle, rerun with PGQ_IDLE_MACHINE=1'}.
""",
    )
    common.print_summary(f"R44 native Gaussian performance ({common.timing_label()})", configs=len(rows),
                         speedup_vs_numpy_backend=f"{speed_np:.1f} (n={modes[-1]})",
                         speedup_vs_piquasso=f"{speed_pq:.1f} (n={nmax})", worst_accuracy=f"{worst_acc:.2e}",
                         worst_batch_agreement=f"{worst_batch:.2e}", thread_times=f"{t1:.3e}s -> {tk:.3e}s",
                         failures=len(failures))
    if failures:
        raise AssertionError(f"{len(failures)} R44 accuracy checks failed; see FAILURES csv")


if __name__ == "__main__":
    main()
