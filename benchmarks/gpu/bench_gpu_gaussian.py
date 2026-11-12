"""Batched Gaussian kernels: NumPy (vectorised CPU) vs native C++/OpenMP batch vs CuPy GPU.

Every implementation runs the IDENTICAL task on the same input batch (B states of n modes, a
physical squeezed CZ chain with loss, see ``chain_state``):

* ``local_symplectic``: a beam splitter on modes (n/3, 2n/3) of every state (O(B n) work);
* ``dense_symplectic``: a different random 2n x 2n symplectic matrix per state on all modes
  (O(B n^3), batched GEMM, the case a GPU is built for);
* ``homodyne``: destructive homodyne on mode n/3 with angle 0.4, efficiency 0.9 and outcome
  0.1, returning conditioned moments and densities (O(B n^2)).

Implementations (rows ``implementation``):

* ``numpy``: ``photographiq.gpu`` with ``device="cpu"`` (same algorithm, NumPy/OpenBLAS);
* ``native``: ``photographiq.backends.native_gaussian.batch_*`` (C++, OpenMP over the batch);
* ``gpu_end_to_end``: ``device="gpu"`` with NumPy input and output (H2D + kernels + D2H);
* ``gpu_kernel``: device-resident input, ``on_device=True``, then ``Device().synchronize()``
  inside the timed region, so queued work is included;
* ``gpu_h2d`` / ``gpu_d2h``: the input upload and the result download alone (synchronised).

Timing is ``bench_common.time_call`` (first call reported separately and never a sample, so
CuPy's one-off kernel compilation is excluded; warm-up; median and IQR). Before timing, each
implementation's output is compared with the float64 ``numpy`` result (max deviation
relative to the largest entry; for float32 rows this is the precision loss) and recorded. Combinations whose input exceeds ``--max-bytes`` are skipped and listed.
``--smoke`` runs a tiny grid only to check the script; its timings mean nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_common import environment, stamp, time_call, write_json  # noqa: E402

from photographiq import gpu, native  # noqa: E402
from photographiq.backends.gaussian import GaussianBackend  # noqa: E402
from photographiq.gaussian import beamsplitter, embed, squeezing  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
ANGLE, OUTCOME, EFFICIENCY = 0.4, 0.1, 0.9


def chain_state(n: int, seed: int = 0):
    backend = GaussianBackend()
    rng = np.random.default_rng(seed)
    for node in range(n):
        backend.prepare(node, float(rng.uniform(0.3, 1.0)))
    for node in range(n - 1):
        backend.entangle(node, node + 1, 1.0)
    for node in range(n):
        backend.loss(node, 0.9, 0.05)
    return backend.get_state()


def random_symplectics(rng, batch: int, n: int) -> np.ndarray:
    """Per state: beam splitters on random pairs interleaved with single-mode squeezers."""
    out = np.empty((batch, 2 * n, 2 * n))
    for b in range(batch):
        s = np.eye(2 * n)
        for _ in range(2 * n):
            u, v = rng.choice(n, 2, replace=False) if n > 1 else (0, 0)
            g = embed(squeezing(rng.uniform(-0.3, 0.3)), [2 * u, 2 * u + 1], 2 * n)
            if n > 1:
                idx = [2 * u, 2 * u + 1, 2 * v, 2 * v + 1]
                g = embed(beamsplitter(rng.uniform(0, np.pi)), idx, 2 * n) @ g
            s = g @ s
        out[b] = s
    return out


def tasks(n: int, batch: int, rng):
    u, v = n // 3, (2 * n) // 3 if n > 2 else n - 1
    bs = beamsplitter(0.37)
    dense = random_symplectics(rng, batch, n)
    all_modes = list(range(n))
    return {
        "local_symplectic": {
            "ours": lambda m, c, **kw: gpu.apply_symplectic(m, c, [u, v], bs, **kw).covariances,
            "native": lambda m, c: native_batch().batch_apply_symplectic(m, c, [u, v], bs)[1],
        },
        "dense_symplectic": {
            # ``mats`` lets the gpu_kernel row pass device-resident matrices (no H2D in it)
            "matrices": dense,
            "ours": lambda m, c, mats=dense, **kw: (
                gpu.apply_symplectic(m, c, all_modes, mats, **kw).covariances
            ),
            "native": lambda m, c: native_batch().batch_apply_symplectic(m, c, all_modes, dense)[1],
        },
        "homodyne": {
            "ours": lambda m, c, **kw: (
                gpu.condition_homodyne(m, c, u, ANGLE, OUTCOME, EFFICIENCY, **kw).covariances
            ),
            "native": lambda m, c: native_batch().batch_condition_homodyne(
                m, c, u, ANGLE, OUTCOME, EFFICIENCY
            )[1],
        },
    }


def native_batch():
    from photographiq.backends import native_gaussian

    return native_gaussian


def deviation(value, reference) -> float:
    reference = np.asarray(reference)
    return float(np.abs(np.asarray(value) - reference).max() / max(np.abs(reference).max(), 1.0))


def bench(n: int, batch: int, dtype: str, budget: float, has_native: bool, has_gpu: bool):
    rng = np.random.default_rng(n * 1000 + batch)
    state = chain_state(n)
    means = np.ascontiguousarray(np.repeat(state.mean[None], batch, axis=0))
    covs = np.ascontiguousarray(np.repeat(state.covariance[None], batch, axis=0))
    rows = []
    common = {"modes": n, "batch": batch, "dtype": dtype}
    for task, impl in tasks(n, batch, rng).items():
        reference = impl["ours"](means, covs, device="cpu")  # float64 for every dtype

        def record(name, fn, check=None):
            row = {"task": task, "implementation": name, **common}
            if check is not None:
                row["max_rel_deviation_vs_numpy_float64"] = deviation(check, reference)
            row.update(time_call(fn, budget_s=budget))
            rows.append(row)
            print(
                f"n={n:4d} B={batch:6d} {dtype:7s} {task:17s} {name:15s} "
                f"median {row['median_s'] * 1e3:10.3f} ms"
            )

        cpu_call = lambda: impl["ours"](means, covs, device="cpu", dtype=dtype)  # noqa: E731
        record("numpy", cpu_call, cpu_call())
        if has_native and dtype == "float64":  # native kernels are float64 only
            record("native", lambda: impl["native"](means, covs), impl["native"](means, covs))
        if has_gpu:
            import cupy as cp

            sync = cp.cuda.Device().synchronize
            d_means, d_covs = cp.asarray(means, dtype=dtype), cp.asarray(covs, dtype=dtype)
            # per-state matrices also live on the device, so gpu_kernel contains no transfer
            # (kept float64 as given: the symplecticity check runs before the cast to dtype)
            resident = {"mats": cp.asarray(impl["matrices"])} if "matrices" in impl else {}
            gpu_kw = {"device": "gpu", "dtype": dtype, "on_device": True, **resident}
            on = impl["ours"](d_means, d_covs, **gpu_kw)

            def kernel():
                impl["ours"](d_means, d_covs, **gpu_kw)
                sync()

            def h2d():
                cp.asarray(means, dtype=dtype), cp.asarray(covs, dtype=dtype)
                if "matrices" in impl:
                    cp.asarray(impl["matrices"])
                sync()

            record(
                "gpu_end_to_end",
                lambda: impl["ours"](means, covs, device="gpu", dtype=dtype),
                impl["ours"](means, covs, device="gpu", dtype=dtype),
            )
            record("gpu_kernel", kernel, on.get())
            record("gpu_h2d", h2d)
            record("gpu_d2h", on.get)
            del d_means, d_covs, on, resident, gpu_kw
            cp.get_default_memory_pool().free_all_blocks()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 256, 4096])
    parser.add_argument("--dtypes", nargs="+", default=["float64"], choices=["float64", "float32"])
    parser.add_argument("--budget", type=float, default=5.0, help="seconds per timed row")
    parser.add_argument("--max-bytes", type=float, default=512e6, help="cap on one input array")
    parser.add_argument("--smoke", action="store_true", help="tiny grid, checks the script only")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.smoke:
        args.modes, args.batches, args.budget = [3], [4], 0.05
    has_native = "gaussian" in native.components()
    has_gpu = gpu.available()
    if not has_gpu:
        print("CuPy/CUDA device unavailable: GPU rows are NOT run (no CPU fallback)")
    rows, skipped = [], []
    for dtype in args.dtypes:
        for n in args.modes:
            for batch in args.batches:
                size = batch * (2 * n) ** 2 * np.dtype(dtype).itemsize
                if size > args.max_bytes:
                    skipped.append({"modes": n, "batch": batch, "dtype": dtype, "bytes": size})
                    continue
                rows.extend(bench(n, batch, dtype, args.budget, has_native, has_gpu))
    extra = {"gpu": gpu.device_info() if has_gpu else None}
    if has_native:
        extra["native_gaussian_max_threads"] = native.require("gaussian").max_threads
    name = f"gpu_gaussian_{'smoke_' if args.smoke else ''}{stamp()}.json"
    path = args.output or RESULTS / name
    payload = {
        "benchmark": "gpu_gaussian",
        "smoke": args.smoke,
        "environment": environment(seed=None, **extra),
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "skipped": skipped,
        "results": rows,
    }
    print(f"wrote {write_json(payload, path)}")


if __name__ == "__main__":
    main()
