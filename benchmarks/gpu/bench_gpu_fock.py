"""Batched Fock local gates: native C++ loop vs NumPy/scipy CPU path vs CuPy/cuSPARSE GPU.

Every implementation runs the IDENTICAL task: one local gate on B states of ``modes`` modes in
the total-photon truncated basis (sum n < cutoff, D = C(modes + cutoff - 1, modes)):

* ``displace``: displacement (0.1, 0.05) on mode 1 (a 1-mode matrix from the native
  ``gaussian_unitary``);
* ``cz``: CZ with weight 0.3 on modes (0, 1) (2-mode matrix);
* ``kerr``: Kerr 0.2 n^2 on mode 1 (diagonal phase gate; pure states only).

Pure batches are ``(B, D)``, mixed ``(B, D, D)`` (rho -> U rho U^dag).

Implementations (rows ``implementation``):

* ``native_loop``: ``photographiq._native.fock.apply_pure`` / ``apply_mixed`` / ``phase_*``
  state by state on a fresh copy of the batch (the copy is timed; the GPU kernels are
  out-of-place too);
* ``numpy``: ``photographiq.gpu.fock`` with ``device="cpu"`` (same algorithm, scipy CSR);
* ``gpu_end_to_end``: ``device="gpu"``, NumPy in and out (H2D + kernels + D2H);
* ``gpu_kernel``: device-resident input, ``on_device=True``, ``Device().synchronize()``
  inside the timed region;
* ``gpu_h2d`` / ``gpu_d2h``: upload of the batch / download of the result (synchronised).

Timing is ``bench_common.time_call`` (the first call, which builds and caches the index maps
and the cuSPARSE handle, is reported as ``first_call_s`` and is never a sample; warm-up;
median and IQR). Before timing, every implementation's output is compared with the
complex128 ``native_loop`` result (max deviation relative to the largest entry) and the run
stops if it exceeds 1e-12 (complex128) or 1e-5 (complex64). ``peak_device_pool_bytes`` is
the CuPy memory-pool footprint after one ``gpu_kernel`` call started from an emptied pool
(input batch, cached index maps, cuSPARSE workspace and output; the pool keeps freed blocks,
so this bounds the call's peak). Combinations whose input exceeds ``--max-bytes`` are skipped
and listed. System CPU load is recorded at start and end (the machine may be shared).
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
from photographiq.gpu import fock as gf  # noqa: E402

try:
    import psutil
except ImportError:
    psutil = None

RESULTS = Path(__file__).resolve().parent / "results"
PURE = [(2, 30), (4, 20), (6, 10), (5, 20)]  # (modes, cutoff)
MIXED = [(2, 20), (3, 10), (3, 16), (4, 12)]
TOL = {"complex128": 1e-12, "complex64": 1e-5}


def engine():
    return native.require("fock")


def gate_table(cutoff: int) -> dict:
    """name -> (local modes, op matrix or ("phase", phases))."""
    f = engine()
    cz = np.eye(4)
    cz[1, 2] = cz[3, 0] = 0.3
    return {
        "displace": ((1,), f.gaussian_unitary(np.eye(2), np.array([0.1, 0.05]), cutoff)),
        "cz": ((0, 1), f.gaussian_unitary(cz, np.zeros(4), cutoff)),
        "kerr": ((1,), ("phase", 0.2 * np.arange(cutoff) ** 2.0)),
    }


def ours(states, modes, cutoff, local, gate, **kw):
    if isinstance(gate, tuple):
        return gf.apply_phase(states, modes, cutoff, local, gate[1], **kw).states
    return gf.apply_operator(states, modes, cutoff, local, gate, **kw).states


def native_loop(states, modes, cutoff, local, gate):
    f, out = engine(), states.copy()
    kind = "mixed" if out.ndim == 3 else "pure"
    call = getattr(f, f"{'phase' if isinstance(gate, tuple) else 'apply'}_{kind}")
    arg = gate[1] if isinstance(gate, tuple) else gate
    for state in out:
        call(state, modes, cutoff, list(local), arg)
    return out


def batch_states(kind: str, batch: int, dim: int, rng) -> np.ndarray:
    if kind == "pure":
        x = rng.normal(size=(batch, dim)) + 1j * rng.normal(size=(batch, dim))
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    a = rng.normal(size=(batch, dim, 4)) + 1j * rng.normal(size=(batch, dim, 4))
    rho = a @ np.conj(np.swapaxes(a, 1, 2))
    return rho / np.trace(rho, axis1=1, axis2=2)[:, None, None]


def deviation(value, reference) -> float:
    reference = np.asarray(reference)
    return float(np.abs(np.asarray(value) - reference).max() / np.abs(reference).max())


def bench(kind, modes, cutoff, batch, dtype, budget, has_gpu):
    dim = engine().dimension(modes, cutoff)
    states = batch_states(kind, batch, dim, np.random.default_rng(modes * 1000 + cutoff))
    rows = []
    common = {"kind": kind, "modes": modes, "cutoff": cutoff, "dimension": dim, "batch": batch}
    common["dtype"] = dtype
    gates = gate_table(cutoff)
    if kind == "mixed":
        gates.pop("kerr")  # an element-wise phase: nothing to learn beyond the pure rows
    for task, (local, gate) in gates.items():
        reference = native_loop(states, modes, cutoff, local, gate)

        def record(name, fn, check=None, **extra):
            row = {"task": task, "implementation": name, **common, **extra}
            if check is not None:
                row["max_rel_deviation_vs_native_complex128"] = deviation(check, reference)
                if not row["max_rel_deviation_vs_native_complex128"] <= TOL[dtype]:
                    raise RuntimeError(f"{name} disagrees with native: {row}")
            row.update(time_call(fn, budget_s=budget))
            rows.append(row)
            print(
                f"{kind:5s} m={modes} c={cutoff:2d} D={dim:6d} B={batch:4d} {dtype:10s} {task:8s} "
                f"{name:15s} median {row['median_s'] * 1e3:10.3f} ms",
                flush=True,
            )

        if dtype == "complex128":  # the native engine is complex128 only
            record(
                "native_loop", lambda: native_loop(states, modes, cutoff, local, gate), reference
            )
        cpu = lambda: ours(states, modes, cutoff, local, gate, device="cpu", dtype=dtype)  # noqa: E731
        record("numpy", cpu, cpu())
        if not has_gpu:
            continue
        import cupy as cp

        sync, pool = cp.cuda.Device().synchronize, cp.get_default_memory_pool()
        kw = {"device": "gpu", "dtype": dtype}
        end_to_end = lambda: ours(states, modes, cutoff, local, gate, **kw)  # noqa: E731
        record("gpu_end_to_end", end_to_end, end_to_end())
        resident = cp.asarray(states, dtype=dtype)
        pool.free_all_blocks()
        out = ours(resident, modes, cutoff, local, gate, on_device=True, **kw)
        sync()
        peak = pool.total_bytes()

        def kernel():
            ours(resident, modes, cutoff, local, gate, on_device=True, **kw)
            sync()

        def h2d():
            cp.asarray(states, dtype=dtype)
            sync()

        record("gpu_kernel", kernel, out.get(), peak_device_pool_bytes=int(peak))
        record("gpu_h2d", h2d)
        record("gpu_d2h", out.get)
        resident = out = None  # release before the next task
        pool.free_all_blocks()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 256])
    parser.add_argument("--mixed-batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--dtypes", nargs="+", default=["complex128", "complex64"], choices=TOL)
    parser.add_argument("--budget", type=float, default=1.0, help="seconds per timed row")
    parser.add_argument("--max-bytes", type=float, default=256e6, help="cap on the input batch")
    parser.add_argument("--smoke", action="store_true", help="tiny grid, checks the script only")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    grids = {"pure": (PURE, args.batches), "mixed": (MIXED, args.mixed_batches)}
    if args.smoke:
        grids = {"pure": ([(3, 6)], [1, 4]), "mixed": ([(2, 5)], [2])}
        args.budget = 0.05
    has_gpu = gpu.available()
    if not has_gpu:
        print("CuPy/CUDA device unavailable: GPU rows are NOT run (no CPU fallback)")
    load_start = None if psutil is None else psutil.cpu_percent(interval=1.0)
    rows, skipped = [], []
    for dtype in args.dtypes:
        for kind, (shapes, batches) in grids.items():
            for modes, cutoff in shapes:
                dim = engine().dimension(modes, cutoff)
                for batch in batches:
                    size = batch * dim ** (1 if kind == "pure" else 2) * np.dtype(dtype).itemsize
                    case = {"kind": kind, "modes": modes, "cutoff": cutoff, "batch": batch}
                    if size > args.max_bytes:
                        skipped.append({**case, "dtype": dtype, "bytes": size})
                        continue
                    rows.extend(bench(kind, modes, cutoff, batch, dtype, args.budget, has_gpu))
    extra = {
        "gpu": gpu.device_info() if has_gpu else None,
        "native_fock_max_threads": engine().max_threads(),
        "system_cpu_percent_at_start": load_start,
        "system_cpu_percent_at_end": None if psutil is None else psutil.cpu_percent(interval=1.0),
    }
    name = f"gpu_fock_{'smoke_' if args.smoke else ''}{stamp()}.json"
    path = args.output or RESULTS / name
    payload = {
        "benchmark": "gpu_fock",
        "smoke": args.smoke,
        "environment": {**environment(seed=None, **extra), "precision": "see rows (dtype)"},
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "skipped": skipped,
        "results": rows,
    }
    print(f"wrote {write_json(payload, path)}")


if __name__ == "__main__":
    main()
