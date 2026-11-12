"""Benchmark the native Gaussian engine against NumPy, with every comparison labelled.

Tasks, each versus the number of modes n (``kind`` in the JSON says what a row is):

* ``gate`` (one state, one two-mode beam splitter, updated in place):
  ``numpy_backend`` = ``GaussianBackend.transform`` (dense embedded S V S^T, O(n^3), the
  reference implementation); ``numpy_local`` = NumPy O(n k) local update of the touched
  rows/columns (isolates the algorithmic gain from the C++ gain); ``native`` =
  ``NativeGaussianBackend.transform`` (C++ O(n k) local update).
* ``homodyne`` (one state, one destructive homodyne with a fixed outcome, Schur complement):
  ``GaussianBackend.measure`` vs ``NativeGaussianBackend.measure``. The fresh backend for each
  sample is built outside the timed region.
* ``batch_gate`` / ``batch_homodyne`` (B distinct states, distinct beam-splitter angles,
  homodyne angles and outcomes):

  - ``native_batch``: ``batch_apply_symplectic`` / ``batch_condition_homodyne`` (OpenMP over B);
  - ``numpy_vectorized``: the same algorithm as the C++ kernel written with batched NumPy
    (matmul / einsum / fancy indexing over a leading batch axis), no Python loop over B;
  - ``python_loop``: a Python loop over B calling the per-state NumPy path (the NumPy local
    update for gates, ``GaussianBackend.measure`` for homodyne).

  All three receive identical inputs and must return identical outputs: before any timing
  the outputs are compared with native and the run aborts if
  max|x - x_native| > ``AGREEMENT_TOL`` * max(1, max|x_native|) for any output array.

Timing is ``bench_common.time_call`` (median / IQR / min / mean, cold first call excluded).
Ratios in ``speedups`` are median / median from the same run and name both sides. The
environment block records CPU, BLAS build, thread variables, threadpoolctl info (if
installed), the native OpenMP max thread count and the git revision.

Run from the repository root: ``python benchmarks/gaussian/bench_native_gaussian.py``
[``--smoke`` | ``--full``]. Set ``OMP_NUM_THREADS`` (native) / ``OPENBLAS_NUM_THREADS``
(NumPy BLAS) before starting Python to change thread counts; both are recorded.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_common import _stats, environment, stamp, time_call, write_json  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq import native  # noqa: E402
from photographiq.backends import native_gaussian as ng  # noqa: E402
from photographiq.backends.gaussian import GaussianBackend  # noqa: E402
from photographiq.gaussian import beamsplitter  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
AGREEMENT_TOL = 1e-12
EFFICIENCY, NOISE = 0.9, 0.0

CONFIGS = {
    "smoke": {"modes": [4, 6], "batch": 4, "repeats": 3, "budget_s": 0.2, "batch_mb": 8},
    "default": {
        "modes": [8, 16, 32, 64, 128, 256],
        "batch": 64,
        "repeats": 11,
        "budget_s": 2.0,
        "batch_mb": 32,
    },
    "full": {
        "modes": [8, 16, 32, 64, 128, 256, 512],
        "batch": 256,
        "repeats": 15,
        "budget_s": 5.0,
        "batch_mb": 128,
    },
}

KINDS = {
    "numpy_backend": "numpy-dense-backend",
    "numpy_local": "numpy-local-single",
    "native": "native-single",
    "native_batch": "native-batch",
    "numpy_vectorized": "numpy-vectorized",
    "python_loop": "python-loop",
}

COMPARISONS = {
    "gate native vs numpy_local": "same O(n k) algorithm, one state: C++ vs NumPy",
    "gate native vs numpy_backend": "algorithm + language: O(n k) C++ vs dense O(n^3) reference",
    "homodyne native vs numpy_backend": "backend API, one state; reference also re-validates "
    "the new GaussianState (O(n^3) eigen-check), native does not",
    "batch_* native_batch vs numpy_vectorized": "identical algorithm and outputs: OpenMP C++ "
    "vs batched NumPy (no Python loop); the honest batched speedup",
    "batch_* native_batch vs python_loop": "native batch vs per-state Python loop over the NumPy "
    "path; includes Python-loop overhead and, for batch_homodyne, GaussianBackend.measure's "
    "O(n^3) re-validation of every output state (neither batch implementation does this); "
    "an upper bound on the C++ gain, not a like-for-like comparison",
    "gate native vs numpy_backend (note)": "GaussianBackend.transform also re-validates the new "
    "state (O(n^3) complex eigvalsh) on every call",
}


# ----- inputs (setup, never timed) ------------------------------------------------------
def chain_state(n: int, seed: int = 0) -> pg.GaussianState:
    """Physical, fully correlated n-mode state: squeezed CZ chain with loss on every mode.

    Built with the native O(n k) updates (the dense reference re-validates after every one of
    the ~3n operations, O(n^4) total: ~1 h of setup at n = 512) and validated once on return.
    """
    backend = ng.NativeGaussianBackend()
    rng = np.random.default_rng(seed)
    for node in range(n):
        backend.prepare(node, float(rng.uniform(0.3, 1.0)))
    for node in range(n - 1):
        backend.entangle(node, node + 1, 1.0)
    for node in range(n):
        backend.loss(node, 0.9, 0.05)
    return backend.get_state()


def batch_inputs(state: pg.GaussianState, batch: int, seed: int) -> dict:
    """B distinct physical states: random displacements and added classical noise V + l_b I."""
    rng = np.random.default_rng(seed)
    dim = state.mean.size
    cov = (state.covariance + state.covariance.T) / 2
    means = state.mean[None] + rng.normal(0.0, 1.0, (batch, dim))
    covs = cov[None] + rng.uniform(0.0, 0.1, batch)[:, None, None] * np.eye(dim)[None]
    return {
        "means": means,
        "covs": covs,
        "mats": np.stack([beamsplitter(t) for t in rng.uniform(0, np.pi, batch)]),
        "angles": rng.uniform(0, np.pi, batch),
        "outcomes": rng.normal(0.0, 1.0, batch),
    }


# ----- NumPy implementations ------------------------------------------------------------
def local_numpy_update(mean: np.ndarray, cov: np.ndarray, idx: np.ndarray, s: np.ndarray):
    """O(n k) per-state NumPy update in place: rows then columns of the touched quadratures."""
    mean[idx] = s @ mean[idx]
    cov[idx, :] = s @ cov[idx, :]
    cov[:, idx] = cov[:, idx] @ s.T
    block = cov[np.ix_(idx, idx)]
    cov[np.ix_(idx, idx)] = (block + block.T) / 2


def numpy_batch_gate(means, covs, idx, mats):
    """Batched NumPy of the C++ transform_local: W = V[:, P] X^T, block X W[P], symmetrised."""
    m = means.copy()
    c = 0.5 * (covs + covs.transpose(0, 2, 1))  # the kernel's symmetrisation; also the copy
    m[:, idx] = np.einsum("bij,bj->bi", mats, m[:, idx])
    w = c[:, :, idx] @ mats.transpose(0, 2, 1)
    inner = mats @ w[:, idx, :]
    c[:, :, idx] = w
    c[:, idx, :] = w.transpose(0, 2, 1)
    c[:, idx[:, None], idx[None, :]] = 0.5 * (inner + inner.transpose(0, 2, 1))
    return m, c


def numpy_loop_gate(means, covs, idx, mats):
    m, c = means.copy(), 0.5 * (covs + covs.transpose(0, 2, 1))
    for b in range(len(m)):
        local_numpy_update(m[b], c[b], idx, mats[b])
    return m, c


def numpy_batch_homodyne(means, covs, mode, angles, outcomes, efficiency, noise):
    """Batched NumPy of the C++ condition_homodyne_into (Schur complement on one mode)."""
    cov = 0.5 * (covs + covs.transpose(0, 2, 1))
    q = 2 * mode
    keep = np.r_[0:q, q + 2 : means.shape[1]]
    u = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (B, 2)
    block = cov[:, q : q + 2, q : q + 2]
    variance = np.einsum("bi,bij,bj->b", u, block, u) + (1 - efficiency) / efficiency + noise
    expected = np.einsum("bi,bi->b", u, means[:, q : q + 2])
    cross = np.einsum("bij,bj->bi", cov[:, keep, q : q + 2], u)
    out_mean = means[:, keep] + cross * ((outcomes - expected) / variance)[:, None]
    out_cov = cov[:, keep[:, None], keep[None, :]] - (
        cross[:, :, None] * cross[:, None, :] / variance[:, None, None]
    )
    sigma = np.sqrt(variance)
    z = (outcomes - expected) / sigma
    density = np.exp(-z * z / 2) / math.sqrt(2 * math.pi) / sigma
    return out_mean, out_cov, density


def numpy_loop_homodyne(states, node, angles, outcomes):
    backend = GaussianBackend()
    out_means, out_covs, densities = [], [], []
    for state, angle, outcome in zip(states, angles, outcomes, strict=True):
        backend.state = state  # plain attribute; measure() never mutates the input state
        backend.measure(
            node, pg.Homodyne(efficiency=EFFICIENCY, noise=NOISE), float(angle), outcome=outcome
        )
        out_means.append(backend.state.mean)
        out_covs.append(backend.state.covariance)
        densities.append(backend.last_measurement["value"])
    return np.array(out_means), np.array(out_covs), np.array(densities)


# ----- checks and timing ----------------------------------------------------------------
def max_scaled_diff(outputs, reference) -> float:
    """max over output arrays of max|x - ref| / max(1, max|ref|); inf on a shape mismatch."""
    if any(np.shape(x) != np.shape(r) for x, r in zip(outputs, reference, strict=True)):
        return math.inf  # never let broadcasting hide a wrong output shape
    return max(
        float(np.max(np.abs(np.asarray(x) - r), initial=0.0) / max(1.0, np.max(np.abs(r))))
        for x, r in zip(outputs, reference, strict=True)
    )


def check(task: str, impl: str, n: int, outputs, reference) -> float:
    diff = max_scaled_diff(outputs, reference)
    if not diff <= AGREEMENT_TOL:
        raise RuntimeError(f"{task} {impl} n={n}: disagrees with native by {diff:.3e}")
    return diff


def time_with_setup(
    setup: Callable[[], object], call: Callable[[object], None], repeats: int, warmup: int = 2
) -> dict:
    """For destructive calls: one ``setup()`` per sample, outside the timed region."""
    start = time.perf_counter()
    call(setup())
    first = time.perf_counter() - start
    for _ in range(warmup):
        call(setup())
    samples = []
    for _ in range(repeats):
        target = setup()
        start = time.perf_counter()
        call(target)
        samples.append(time.perf_counter() - start)
    return _stats(samples, first, warmup, 1)


def bench_size(n: int, cfg: dict, seed: int) -> list[dict]:
    rows = []
    timer = {"repeats": cfg["repeats"], "budget_s": cfg["budget_s"]}

    def record(task, impl, stats, **extra):
        rows.append(
            {"task": task, "implementation": impl, "kind": KINDS[impl], "modes": n} | extra | stats
        )

    state = chain_state(n, seed)
    u, v = n // 3, (2 * n) // 3
    idx = np.array([2 * u, 2 * u + 1, 2 * v, 2 * v + 1])
    s = beamsplitter(0.37)

    # --- single state, gate (in place; the state stays physical under repeated BS) ---
    ref = GaussianBackend()
    ref.set_state(state)
    nat = ng.NativeGaussianBackend()
    nat.set_state(state)
    mean, cov = state.mean.copy(), (state.covariance + state.covariance.T) / 2
    record("gate", "numpy_backend", time_call(lambda: ref.transform((u, v), s), **timer))
    record("gate", "numpy_local", time_call(lambda: local_numpy_update(mean, cov, idx, s), **timer))
    record("gate", "native", time_call(lambda: nat.transform((u, v), s), **timer))

    # --- single state, destructive homodyne ---
    def fresh(cls):
        def setup():
            backend = cls()
            backend.set_state(state)
            return backend

        return setup

    def measure(backend):
        backend.measure(u, pg.Homodyne(efficiency=EFFICIENCY, noise=NOISE), 0.4, outcome=0.1)

    for impl, cls in (("numpy_backend", GaussianBackend), ("native", ng.NativeGaussianBackend)):
        record("homodyne", impl, time_with_setup(fresh(cls), measure, cfg["repeats"]))

    # --- batches: identical inputs for all implementations ---
    batch = min(cfg["batch"], max(4, cfg["batch_mb"] * 2**20 // (8 * 4 * n * n)))
    data = batch_inputs(state, batch, seed + 1)
    means, covs, mats = data["means"], data["covs"], data["mats"]
    angles, outcomes = data["angles"], data["outcomes"]
    states = [pg.GaussianState(m, c, state.nodes) for m, c in zip(means, covs, strict=True)]
    extra = {"batch": batch, "agreement_tol": AGREEMENT_TOL}

    gate_impls = {
        "native_batch": lambda: ng.batch_apply_symplectic(means, covs, [u, v], mats),
        "numpy_vectorized": lambda: numpy_batch_gate(means, covs, idx, mats),
        "python_loop": lambda: numpy_loop_gate(means, covs, idx, mats),
    }
    hom_impls = {
        "native_batch": lambda: ng.batch_condition_homodyne(
            means, covs, u, angles, outcomes, EFFICIENCY, NOISE
        ),
        "numpy_vectorized": lambda: numpy_batch_homodyne(
            means, covs, u, angles, outcomes, EFFICIENCY, NOISE
        ),
        "python_loop": lambda: numpy_loop_homodyne(states, u, angles, outcomes),
    }
    for task, impls in (("batch_gate", gate_impls), ("batch_homodyne", hom_impls)):
        reference = impls["native_batch"]()
        diffs = {impl: check(task, impl, n, fn(), reference) for impl, fn in impls.items()}
        for impl, fn in impls.items():  # every agreement check passes before any timing
            record(
                task, impl, time_call(fn, **timer), **extra, max_scaled_diff_vs_native=diffs[impl]
            )
    return rows


def speedups(rows: list[dict]) -> list[dict]:
    pairs = [
        ("gate", "numpy_local", "native"),
        ("gate", "numpy_backend", "native"),
        ("homodyne", "numpy_backend", "native"),
        ("batch_gate", "numpy_vectorized", "native_batch"),
        ("batch_gate", "python_loop", "native_batch"),
        ("batch_homodyne", "numpy_vectorized", "native_batch"),
        ("batch_homodyne", "python_loop", "native_batch"),
    ]
    med = {(r["task"], r["implementation"], r["modes"]): r["median_s"] for r in rows}
    out = []
    for n in sorted({r["modes"] for r in rows}):
        for task, base, fast in pairs:
            out.append(
                {
                    "task": task,
                    "modes": n,
                    "comparison": f"{KINDS[base]} / {KINDS[fast]}",
                    "ratio_of_medians": med[task, base, n] / med[task, fast, n],
                }
            )
    return out


def thread_info() -> dict:
    info = {"native_gaussian_max_threads": native.require("gaussian").max_threads}
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        info["numpy_blas"] = {k: blas.get(k) for k in ("name", "version", "openblas configuration")}
    except Exception as exc:  # numpy build without dict config
        info["numpy_blas"] = f"unavailable: {exc}"
    try:
        import threadpoolctl

        info["threadpoolctl"] = threadpoolctl.threadpool_info()
    except ImportError:
        info["threadpoolctl"] = "not installed"
    return info


def run(config: str = "default", seed: int = 0, output: Path | None = None) -> dict:
    cfg = CONFIGS[config]
    rows = [row for n in cfg["modes"] for row in bench_size(n, cfg, seed)]
    payload = {
        "benchmark": "native_gaussian",
        "config": config,
        "parameters": cfg
        | {"agreement_tol": AGREEMENT_TOL, "efficiency": EFFICIENCY, "noise": NOISE},
        "environment": environment(
            seed=seed, timing="see benchmarks/bench_common.py: time_call", **thread_info()
        ),
        "comparisons": COMPARISONS,
        "results": rows,
        "speedups": speedups(rows),
    }
    path = output or RESULTS / f"native_gaussian_{config}_{stamp()}.json"
    write_json(payload, path)
    payload["path"] = str(path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="tiny sizes (seconds)")
    group.add_argument("--full", action="store_true", help="larger sizes and batches (long)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = "smoke" if args.smoke else "full" if args.full else "default"
    payload = run(config, args.seed, args.output)
    for r in payload["results"]:
        print(
            f"n={r['modes']:4d} {r['task']:15s} {r['kind']:20s} B={r.get('batch', 1):4d} "
            f"median {r['median_s'] * 1e6:12.1f} us  IQR {r['iqr_s'] * 1e6:10.1f} us"
        )
    for r in payload["speedups"]:
        print(
            f"n={r['modes']:4d} {r['task']:15s} {r['comparison']:42s} {r['ratio_of_medians']:8.2f}x"
        )
    print(f"wrote {payload['path']}")


if __name__ == "__main__":
    main()
