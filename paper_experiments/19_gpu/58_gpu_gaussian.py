"""R58: Batched Gaussian kernels: NumPy CPU vs native C++ CPU vs CuPy GPU (accuracy and stage timing).

Scientific question: for batches of B independent n-mode Gaussian states, how do the NumPy
(vectorised CPU), native C++/OpenMP batch kernels and the CuPy GPU path of
``photographiq.gpu`` compare in (i) accuracy against the float64 NumPy reference and (ii) time,
with the GPU time SEPARATED into host data preparation, host->device transfer, kernel
execution and device->host transfer? Did the GPU actually execute (no CPU fallback)?

Operations (identical inputs for all engines; hbar = 2, interleaved, vacuum V = I):

* ``local_symplectic``: beam splitter (theta = 0.37) on modes (n/3, 2n/3), O(B n) work;
* ``dense_symplectic``: a different random 2n x 2n symplectic per state on all modes (batched
  GEMM, O(B n^3)); S_b = O(U_b) D(r_b) with U_b Haar-random unitary (QR of complex Gaussian) and
  single-mode squeezers |r| <= 0.3;
* ``homodyne``: conditioning on a given outcome 0.1 at angle 0.4, efficiency 0.9, mode n/3
  (Schur complement, O(B n^2)); returns conditioned moments and densities (postselected
  branches, not sampled trajectories);
* ``purity``: det(V)^(-1/2) by batched slogdet (no native batch kernel: NumPy vs GPU only).

Input: B copies of a physical state (squeezed CZ chain with thermal loss, ``GaussianBackend``),
so every engine sees the same data. Timing: ``time.perf_counter`` on the host for CPU engines and
GPU end-to-end; CUDA events (``cupy.cuda.Event``, synchronised) for the GPU stages H2D, kernel
(device-resident input, ``on_device=True``; includes the library's validity-check syncs) and
D2H. Median of ``--repeats`` after ``--warmup`` calls; first calls (CuPy compilation, handle
creation) are never samples.

Device verification: every GPU result must report ``execution.device == 'cuda:<id>'``,
backend 'cupy', and device-resident outputs must be device memory
(``pointerGetAttributes(ptr).type == memoryTypeDevice``); otherwise the script fails. No CPU
number is ever labelled GPU. Precision: float64 everywhere; float32 GPU rows only as an
explicit, separately labelled accuracy/speed trade-off. CUDA-Q is not used.

Timings are final only when the run declares an idle machine (``PGQ_IDLE_MACHINE=1``);
the manuscript configuration is ``--full``. Seed 20261158.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from photographiq import gpu, native  # noqa: E402
from photographiq.backends.gaussian import GaussianBackend  # noqa: E402
from photographiq.gaussian import beamsplitter, omega  # noqa: E402

SEED = 20261158
ANGLE, OUTCOME, EFFICIENCY = 0.4, 0.1, 0.9

# ----- acceptance criteria (declared before any evaluation) ---------------------------------
# float64 engines vs NumPy float64, max |diff| / max(1, |ref|): same algorithm (GPU) or the same
# arithmetic in C++ (native); docs/gpu.md observed <= 3e-15 over seeds 0-19 -> 1e-12 bound.
TOL_FLOAT64 = 1e-12
TOL_FLOAT64_PURITY = 1e-11  # slogdet of 2n x 2n via different LAPACK/cuSOLVER paths
# float32 GPU: ~ 2^-24 relative per op times cond/accumulation; docs report <= 1e-4 at these
# squeezings; fail above 1e-3 (a precision bug, not rounding).
TOL_FLOAT32 = 1e-3


def chain_state(n: int):
    backend = GaussianBackend()
    rng = np.random.default_rng(SEED)
    for node in range(n):
        backend.prepare(node, float(rng.uniform(0.3, 1.0)))
    for node in range(n - 1):
        backend.entangle(node, node + 1, 1.0)
    for node in range(n):
        backend.loss(node, 0.9, 0.05)
    return backend.get_state()


def random_symplectics(rng, batch: int, n: int) -> np.ndarray:
    """S_b = O(U_b) D(r_b): Haar unitary (batched QR) times squeezers, interleaved ordering."""
    z = (rng.normal(size=(batch, n, n)) + 1j * rng.normal(size=(batch, n, n))) / np.sqrt(2)
    qm, rm = np.linalg.qr(z)
    d = np.diagonal(rm, axis1=1, axis2=2)
    u = qm * (d / np.abs(d))[:, None, :]
    block = np.zeros((batch, 2 * n, 2 * n))
    block[:, :n, :n], block[:, :n, n:] = u.real, -u.imag
    block[:, n:, :n], block[:, n:, n:] = u.imag, u.real
    perm = np.ravel(np.column_stack([np.arange(n), np.arange(n) + n]))  # (q1,p1,...) <- (q...,p...)
    orth = block[:, perm][:, :, perm]
    r = rng.uniform(-0.3, 0.3, (batch, n))
    sq = np.zeros((batch, 2 * n))
    sq[:, 0::2], sq[:, 1::2] = np.exp(-r), np.exp(r)
    return orth * sq[:, None, :]


def rel(a, ref):
    a, ref = np.asarray(a, float), np.asarray(ref, float)
    return float(np.abs(a - ref).max() / max(1.0, np.abs(ref).max()))


def host_time(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(repeats):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    return float(np.median(t)), float(np.percentile(t, 75) - np.percentile(t, 25))


def event_time(cp, fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    cp.cuda.Device().synchronize()
    t = []
    for _ in range(repeats):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        fn()
        end.record()
        end.synchronize()
        t.append(cp.cuda.get_elapsed_time(start, end) / 1e3)
    return float(np.median(t)), float(np.percentile(t, 75) - np.percentile(t, 25))


def verify_device(cp, result, on_device):
    ex = result.execution
    ok = ex.device.startswith("cuda:") and ex.backend == "cupy" and ex.returned_on_device == on_device
    if on_device:
        arrays = [v for v in vars(result).values() if isinstance(v, cp.ndarray)]
        ok &= bool(arrays) and all(
            cp.cuda.runtime.pointerGetAttributes(a.data.ptr).type == cp.cuda.runtime.memoryTypeDevice
            for a in arrays)
    return bool(ok), ex.device, ex.device_name, ex.dtype


def run_config(n, batch, dtypes, warmup, repeats, has_native, has_gpu):
    rng = np.random.default_rng(SEED + 7919 * n + batch)
    t0 = time.perf_counter()
    state = chain_state(n)
    means = np.ascontiguousarray(np.repeat(state.mean[None], batch, axis=0))
    covs = np.ascontiguousarray(np.repeat(state.covariance[None], batch, axis=0))
    prep_state = time.perf_counter() - t0
    t0 = time.perf_counter()
    dense = random_symplectics(rng, batch, n)
    prep_dense = time.perf_counter() - t0
    sym_err = float(np.abs(dense @ omega(n) @ np.swapaxes(dense, 1, 2) - omega(n)).max())
    u, v = n // 3, (2 * n) // 3 if n > 2 else n - 1
    bs = beamsplitter(0.37)
    allm = list(range(n))
    from photographiq.backends import native_gaussian as ng

    ops = {
        "local_symplectic": dict(
            ours=lambda m, c, **k: gpu.apply_symplectic(m, c, [u, v], bs, **k),
            pick=lambda r: r.covariances,
            native=lambda m, c: ng.batch_apply_symplectic(m, c, [u, v], bs)[1],
            extra=None, prep=prep_state, out_bytes=covs.nbytes + means.nbytes),
        "dense_symplectic": dict(
            ours=lambda m, c, mats=dense, **k: gpu.apply_symplectic(m, c, allm, mats, **k),
            pick=lambda r: r.covariances,
            native=lambda m, c: ng.batch_apply_symplectic(m, c, allm, dense)[1],
            extra=dense, prep=prep_state + prep_dense, out_bytes=covs.nbytes + means.nbytes),
        "homodyne": dict(
            ours=lambda m, c, **k: gpu.condition_homodyne(m, c, u, ANGLE, OUTCOME, EFFICIENCY, **k),
            pick=lambda r: r.covariances,
            native=lambda m, c: ng.batch_condition_homodyne(m, c, u, ANGLE, OUTCOME, EFFICIENCY)[1],
            extra=None, prep=prep_state, out_bytes=batch * (2 * n - 2) * (2 * n - 1) * 8),
        "purity": dict(
            ours=lambda m, c, **k: gpu.purity(c, **k),
            pick=lambda r: r.values, native=None, extra=None, prep=prep_state, out_bytes=batch * 8),
    }
    rows = []
    for op, spec in ops.items():
        reference = spec["pick"](spec["ours"](means, covs, device="cpu"))
        tol64 = TOL_FLOAT64_PURITY if op == "purity" else TOL_FLOAT64
        base = {"op": op, "modes": n, "batch": batch, "prep_s": spec["prep"],
                "input_bytes": means.nbytes + covs.nbytes + (spec["extra"].nbytes if spec["extra"] is not None else 0)}
        med, iqr = host_time(lambda: spec["ours"](means, covs, device="cpu"), warmup, repeats)
        rows.append({**base, "engine": "numpy_cpu", "dtype": "float64", "total_s": med, "iqr_s": iqr,
                     "rel_err_vs_numpy64": 0.0, "accuracy_ok": True, "device": "cpu"})
        if has_native and spec["native"] is not None:
            err = rel(spec["native"](means, covs), reference)
            med, iqr = host_time(lambda: spec["native"](means, covs), warmup, repeats)
            rows.append({**base, "engine": "native_cpp", "dtype": "float64", "total_s": med, "iqr_s": iqr,
                         "rel_err_vs_numpy64": err, "accuracy_ok": err <= tol64, "device": "cpu (OpenMP)"})
        if not has_gpu:
            continue
        import cupy as cp
        for dtype in dtypes:
            tol = tol64 if dtype == "float64" else TOL_FLOAT32
            res_e2e = spec["ours"](means, covs, device="gpu", dtype=dtype)
            ok_e2e, dev, name, dt = verify_device(cp, res_e2e, False)
            err = rel(spec["pick"](res_e2e), reference)
            e2e, e2e_iqr = host_time(lambda: spec["ours"](means, covs, device="gpu", dtype=dtype), warmup, repeats)
            holder = {}

            def h2d():
                holder["m"], holder["c"] = cp.asarray(means, dtype=dtype), cp.asarray(covs, dtype=dtype)
                if spec["extra"] is not None:
                    holder["x"] = cp.asarray(spec["extra"])

            h2d_t, h2d_iqr = event_time(cp, h2d, warmup, repeats)
            kw = {"device": "gpu", "dtype": dtype, "on_device": True}
            if spec["extra"] is not None:
                kw["mats"] = holder["x"]
            res_dev = spec["ours"](holder["m"], holder["c"], **kw)
            ok_dev, *_ = verify_device(cp, res_dev, True)
            err_dev = rel(cp.asnumpy(spec["pick"](res_dev)), reference)
            kern_t, kern_iqr = event_time(cp, lambda: spec["ours"](holder["m"], holder["c"], **kw), warmup, repeats)
            kern_host, _ = host_time(lambda: (spec["ours"](holder["m"], holder["c"], **kw), cp.cuda.Device().synchronize()),
                                     warmup, repeats)
            out = spec["pick"](res_dev)
            d2h_t, d2h_iqr = event_time(cp, lambda: cp.asnumpy(out), warmup, repeats)
            rows.append({**base, "engine": "cupy_gpu", "dtype": dtype, "total_s": e2e, "iqr_s": e2e_iqr,
                         "h2d_s": h2d_t, "kernel_s": kern_t, "kernel_host_sync_s": kern_host, "d2h_s": d2h_t,
                         "h2d_iqr_s": h2d_iqr, "kernel_iqr_s": kern_iqr, "d2h_iqr_s": d2h_iqr,
                         "output_bytes": int(out.nbytes),
                         "rel_err_vs_numpy64": max(err, err_dev), "accuracy_ok": max(err, err_dev) <= tol,
                         "gpu_executed_verified": ok_e2e and ok_dev, "device": dev, "device_name": name,
                         "execution_dtype": dt})
            holder.clear()  # release device buffers before the next dtype
            res_dev = out = res_e2e = None
            cp.get_default_memory_pool().free_all_blocks()
    return rows, sym_err


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="publication grid (modes 4..64, B up to 16384)")
    parser.add_argument("--quick", action="store_true", help="default small grid (explicit flag accepted)")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-bytes", type=float, default=512e6)
    args = parser.parse_args()
    if args.full:
        modes, batches, repeats = [4, 16, 32, 64], [64, 1024, 16384], args.repeats or 21
    else:
        modes, batches, repeats = [4, 16], [64, 1024, 8192], args.repeats or 5
    dtypes = ["float64", "float32"]
    plt = common.setup_style()
    has_native = "gaussian" in native.components()
    has_gpu = gpu.available()
    if not has_gpu:
        print("CuPy/CUDA device unavailable: GPU rows are NOT run (no CPU fallback is ever counted)")
    rows, skipped, sym = [], [], 0.0
    for n in modes:
        for b in batches:
            if b * (2 * n) ** 2 * 8 > args.max_bytes:
                skipped.append({"modes": n, "batch": b})
                continue
            r, s = run_config(n, b, dtypes, args.warmup, repeats, has_native, has_gpu)
            rows += r
            sym = max(sym, s)
            print(f"done n={n} B={b}")

    failures = [r for r in rows if not r["accuracy_ok"] or (r["engine"] == "cupy_gpu" and not r["gpu_executed_verified"])]
    info = gpu.device_info() if has_gpu else None
    common.save_result(rows, "R58_gpu_gaussian", extra={
        "seed": SEED,
        "timing_status": common.timing_label(),
        "parameters": {"modes": modes, "batches": batches, "repeats": repeats, "warmup": args.warmup,
                       "dtypes": dtypes, "angle": ANGLE, "outcome": OUTCOME, "efficiency": EFFICIENCY,
                       "full": args.full},
        "gpu_device": info,
        "native_gaussian_max_threads": native.require("gaussian").max_threads if has_native else None,
        "precision": "float64 reference; float32 GPU rows explicit",
        "acceptance": {"TOL_FLOAT64": TOL_FLOAT64, "TOL_FLOAT64_PURITY": TOL_FLOAT64_PURITY, "TOL_FLOAT32": TOL_FLOAT32},
        "dense_symplectic_input_check_max": sym,
        "skipped": skipped,
        "cudaq": "not used; not part of positive results",
    })

    gpu_rows = [r for r in rows if r["engine"] == "cupy_gpu" and r["dtype"] == "float64"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    big = [r for r in rows if r["batch"] == max(batches) and r["modes"] == max(modes)]
    ops = list(dict.fromkeys(r["op"] for r in big))
    x = np.arange(len(ops))

    def pick(o, eng, dt, key):
        r = next((r for r in big if r["op"] == o and r["engine"] == eng and r["dtype"] == dt), None)
        return np.nan if r is None or key not in r else r[key] * 1e3

    series = [
        ("host data prep (all engines)", "numpy_cpu", "float64", "prep_s", "#bdbdbd"),
        ("NumPy CPU", "numpy_cpu", "float64", "total_s", "#78909c"),
        ("native C++ CPU", "native_cpp", "float64", "total_s", "#8d6e63"),
        ("GPU H2D", "cupy_gpu", "float64", "h2d_s", "#90caf9"),
        ("GPU kernel f64", "cupy_gpu", "float64", "kernel_s", "#1e88e5"),
        ("GPU D2H", "cupy_gpu", "float64", "d2h_s", "#0d47a1"),
        ("GPU end-to-end f64", "cupy_gpu", "float64", "total_s", "#5e35b1"),
        ("GPU kernel f32", "cupy_gpu", "float32", "kernel_s", "#43a047"),
    ]
    w = 0.85 / len(series)
    for k, (label, eng, dt, key, color) in enumerate(series):
        axes[0].bar(x + (k - (len(series) - 1) / 2) * w, [pick(o, eng, dt, key) for o in ops], w, color=color, label=label)
    axes[0].set_xticks(x, ops, fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("time per call [ms] (log)")
    axes[0].set_title(f"(a) n = {max(modes)}, B = {max(batches)}: separated stages" + ("" if common.timing_final() else " (PROVISIONAL)"))
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2, loc="upper right")
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 30)
    for o, mk in zip(ops, "osD^"):
        for n, ls in zip(modes, ("--", "-", ":", "-.")):
            sub = sorted([r for r in gpu_rows if r["op"] == o and r["modes"] == n], key=lambda r: r["batch"])
            npy = {r["batch"]: r["total_s"] for r in rows if r["engine"] == "numpy_cpu" and r["op"] == o and r["modes"] == n}
            if sub:
                axes[1].plot([r["batch"] for r in sub], [npy[r["batch"]] / r["kernel_s"] for r in sub], ls, marker=mk,
                             label=f"{o}, n={n}")
    axes[1].axhline(1, color="k", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("batch size B")
    axes[1].set_ylabel("NumPy time / GPU kernel time (float64)")
    axes[1].set_title("(b) GPU kernel speed-up vs NumPy" + ("" if common.timing_final() else " (PROVISIONAL)"))
    axes[1].legend(frameon=False, fontsize=6.5, ncol=2)
    common.save_figure(fig, "R58_gpu_gaussian")
    plt.close("all")
    worst64 = max((r["rel_err_vs_numpy64"] for r in rows if r["dtype"] == "float64"), default=0.0)
    worst32 = max((r["rel_err_vs_numpy64"] for r in rows if r["dtype"] == "float32"), default=0.0)
    dev = f"{info['device_name']} (CC {info['compute_capability']}, CuPy {info['cupy_version']}, CUDA runtime {info['cuda_runtime_version']})" if info else "no GPU"
    common.save_caption("R58_gpu_gaussian", f"""
**R58 - batched Gaussian kernels on CPU (NumPy, native C++/OpenMP) and GPU (CuPy).** Timings:
{common.timing_label()} (median of {repeats} calls after {args.warmup} warm-ups). Device: {dev}; GPU execution verified for every GPU row (execution record
'cuda:0' and device-memory pointer check; no CPU fallback counted). Inputs: B copies of an n-mode
squeezed CZ chain with thermal loss (modes {modes}, B {batches}); ops: beam splitter on two modes,
a different random symplectic per state on all modes, homodyne conditioning on a given outcome
(angle {ANGLE}, efficiency {EFFICIENCY}), purity. (a) Largest configuration: host data preparation (shared by
all engines, not included in any engine time), NumPy and native host times, and the GPU stages host->device copy, kernel (CUDA events,
device-resident data, including validity-check syncs) and device->host copy, GPU end-to-end (NumPy in/out), and the explicit float32 kernel. (b) NumPy time over GPU kernel time versus batch size (float64); end-to-end
ratios including transfers are in the CSV. Accuracy vs the NumPy float64 reference: float64
engines <= {worst64:.1e}, float32 GPU <= {worst32:.1e} (relative to max |entry|). Source: R58.
""")
    if failures:
        common.save_csv(failures, "R58_gpu_gaussian_FAILURES")
        raise AssertionError(f"{len(failures)} R58 row(s) failed accuracy or GPU-execution checks; see FAILURES csv")
    common.print_summary(f"R58 GPU Gaussian ({common.timing_label()})", rows=len(rows), device=dev,
                         worst_float64=f"{worst64:.2e}", worst_float32=f"{worst32:.2e}", skipped=skipped)


if __name__ == "__main__":
    main()
