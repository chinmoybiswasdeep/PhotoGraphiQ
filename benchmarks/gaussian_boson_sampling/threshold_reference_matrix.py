"""Accuracy matrix of GBS threshold (click) probabilities against a 40-digit mpmath reference.

Grid: modes {4, 8, 12, 16, 20, 24}; clicks k = 1..min(modes, 16); for every (modes, k) all
five squeezings r in {0.05, 0.2, 0.5, 1.0, 1.5}, each with a seeded random transmissivity in
{1.0, 0.9, 0.5}, displacement on or off (alpha_j = 0.4 u_j e^{2 pi i v_j}, u, v uniform), a
seeded Haar interferometer and a random k-click pattern (360 cases). Zero-probability cases add
a vacuum input mode that the interferometer leaves alone and that clicks (P = 0 up to the
rounding of the stored covariance), for k in {1, 3, min(m, 8), min(m, 12)}.

Per case: PhotoGraphiQ ``threshold_probability`` on the native and Python backends (value,
returned relative bound, the absolute bound ``B`` of ``_threshold_double``, whether a
``ThresholdPrecisionWarning`` fired, runtime), ``precision="auto"`` (path, digits, value; run afterwards in the worker pool),
Piquasso ``get_threshold_detection_probability`` on the identical program (``bench_gbs``), and
the reference. The reference is evaluated from each library's own stored covariance and mean
(Piquasso's ``xpxp`` covariance / 2 at hbar = 2; they differ by ~1e-16), exactly as stored:
``P = P_Z sum_T (-1)^|T| 2^|T| exp(-e_T^T C_T^-1 e_T / 2) / sqrt(det C_T)`` with the Schur
complement ``C`` and shift ``e`` of the dark modes (module doc of ``photographiq.sampling.gbs``),
summed over the 2^k subsets by a depth-first search that extends one Cholesky factor of ``C_T``
per added mode (``O(k^2)`` per subset). It shares no code with PhotoGraphiQ's
``precision=`` path. ``ref_rel_uncertainty = 10^(3 - DPS) sum|terms| / |P|``.

Usage: ``python threshold_reference_matrix.py [--smoke] [--procs N] [--output PATH]``.
Writes ``--output``, by default ``results/threshold_reference_matrix_<stamp>.json`` next to this
script. Full run: ~25 min on 12 processes (382 cases; the k >= 13 references and auto passes dominate).
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import environment, stamp, write_json  # noqa: E402
from bench_gbs import _pq_state  # noqa: E402

from photographiq.sampling import (  # noqa: E402
    haar_unitary,
    squeezed_interferometer_state,
    threshold_probability,
)
from photographiq.sampling.gbs import (  # noqa: E402
    ThresholdPrecisionWarning,
    _threshold_double,
    gbs_matrices,
)
from photographiq.sampling.linalg import _kernels  # noqa: E402

DPS = 40
PREFIX = 6  # 2^6 tasks per reference
MODES = (4, 8, 12, 16, 20, 24)
SQUEEZING = (0.05, 0.2, 0.5, 1.0, 1.5)
ETAS = (1.0, 0.9, 0.5)
MAX_CLICKS = 16


# --------------------------------------------------------------------------- reference


def _branch(task) -> tuple[mp.mpf, mp.mpf]:
    """Signed sum and absolute sum of the subset terms whose first ``p`` bits are ``prefix``."""
    c, e, k, p, prefix = task
    mp.mp.dps = DPS
    idx: list[int] = []
    rows: list[list] = []
    y: list = []
    det_sqrt = mp.mpf(1)

    def extend(idx, rows, y, det_sqrt, mode):
        idx, rows, y = list(idx), list(rows), list(y)
        for j in (2 * mode, 2 * mode + 1):
            col = []
            for a, row in enumerate(rows):
                col.append((c[idx[a]][j] - mp.fdot(row[:a], col)) / row[a])
            pivot = c[j][j] - mp.fdot(col, col)
            if pivot <= 0:
                raise ValueError("conditioned block is not positive definite")
            diag = mp.sqrt(pivot)
            y.append((e[j] - mp.fdot(col, y)) / diag)
            rows.append(col + [diag])
            idx.append(j)
            det_sqrt *= diag
        return idx, rows, y, det_sqrt

    size = 0
    for mode in range(p):
        if prefix >> mode & 1:
            idx, rows, y, det_sqrt = extend(idx, rows, y, det_sqrt, mode)
            size += 1
    total, magnitude = [mp.mpf(0)], [mp.mpf(0)]

    def visit(idx, rows, y, det_sqrt, size, start):
        term = mp.mpf(2) ** size * mp.exp(-mp.fdot(y, y) / 2) / det_sqrt
        total[0] += -term if size % 2 else term
        magnitude[0] += term
        for mode in range(start, k):
            visit(*extend(idx, rows, y, det_sqrt, mode), size + 1, mode + 1)

    visit(idx, rows, y, det_sqrt, size, p)
    return total[0], magnitude[0]


def reference(pool, cov: np.ndarray, mean: np.ndarray, clicks) -> dict:
    """``DPS``-digit click probability of the stored ``(cov, mean)`` (module doc)."""
    mp.mp.dps = DPS
    m = len(clicks)
    s = [i for i in range(m) if clicks[i]]
    z = [i for i in range(m) if not clicks[i]]
    qs = [q for i in s for q in (2 * i, 2 * i + 1)]
    qz = [q for i in z for q in (2 * i, 2 * i + 1)]
    big = mp.matrix(cov.tolist()) + mp.eye(2 * m)
    d = mp.matrix(mean.tolist())

    def sub(r, cc):
        return mp.matrix([[big[a, b] for b in cc] for a in r])

    def vec(r):
        return mp.matrix([d[a] for a in r])

    if qz:
        mzz_inv = mp.inverse(sub(qz, qz))
        schur = sub(qs, qs) - sub(qs, qz) * mzz_inv * sub(qz, qs)
        shift = vec(qs) - sub(qs, qz) * mzz_inv * vec(qz)
        quad = (vec(qz).T * mzz_inv * vec(qz))[0]
        pz = mp.mpf(2) ** len(z) * mp.exp(-quad / 2) / mp.sqrt(mp.det(sub(qz, qz)))
    else:
        schur, shift, pz = sub(qs, qs), vec(qs), mp.mpf(1)
    k = len(s)
    c = [[schur[a, b] for b in range(2 * k)] for a in range(2 * k)]
    e = [shift[a] for a in range(2 * k)]
    p = min(k, PREFIX)
    parts = pool.map(_branch, [(c, e, k, p, prefix) for prefix in range(1 << p)])
    total = pz * mp.fsum(t for t, _ in parts)
    magnitude = pz * mp.fsum(a for _, a in parts)
    return {
        "value": total,
        "magnitude": float(magnitude),
        "uncertainty": float(magnitude * mp.mpf(10) ** (3 - DPS)),
    }


# --------------------------------------------------------------------------------- cases


def cases(seed: int, smoke: bool) -> list[dict]:
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    modes = (4, 8) if smoke else MODES
    for m in modes:
        for k in range(1, min(m, MAX_CLICKS) + 1):
            for r in SQUEEZING[::2] if smoke else SQUEEZING:
                out.append(
                    {
                        "kind": "random",
                        "modes": m,
                        "clicks": k,
                        "squeezing": r,
                        "transmissivity": float(rng.choice(ETAS)),
                        "displaced": bool(rng.random() < 0.5),
                        "vacuum_mode": None,
                    }
                )
        for k in sorted({1, 3, min(m, 8), min(m, 12)}):
            out.append(
                {
                    "kind": "zero",
                    "modes": m,
                    "clicks": k,
                    "squeezing": float(rng.choice(SQUEEZING)),
                    "transmissivity": float(rng.choice(ETAS)),
                    "displaced": bool(rng.random() < 0.5),
                    "vacuum_mode": int(rng.integers(m)),
                }
            )
    for i, case in enumerate(out):
        case["case_id"] = i
        case["unitary_seed"] = seed * 1000 + i
    return out


def build(case: dict, rng: np.random.Generator):
    m, k, vac = case["modes"], case["clicks"], case["vacuum_mode"]
    squeezing = np.full(m, case["squeezing"])
    alpha = np.zeros(m, dtype=complex)
    if case["displaced"]:
        alpha = 0.4 * rng.random(m) * np.exp(2j * np.pi * rng.random(m))
    pattern = np.zeros(m, dtype=int)
    if vac is None:
        unitary = haar_unitary(m, rng=case["unitary_seed"])
        pattern[rng.choice(m, k, replace=False)] = 1
    else:
        rest = [i for i in range(m) if i != vac]
        unitary = np.eye(m, dtype=complex)
        if rest:
            unitary[np.ix_(rest, rest)] = haar_unitary(m - 1, rng=case["unitary_seed"])
        squeezing[vac], alpha[vac] = 0.0, 0.0
        pattern[vac] = 1
        pattern[rng.choice(rest, k - 1, replace=False)] = 1
    return squeezing, unitary, alpha, pattern


def _timed(fn):
    """Value of ``fn()``, warnings it raised, seconds (min of up to 3 calls within ~1 s)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ThresholdPrecisionWarning)
        start = time.perf_counter()
        value = fn()
        best = time.perf_counter() - start
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ThresholdPrecisionWarning)
        for _ in range(2 if best < 0.3 else 0):
            start = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - start)
    messages = [str(w.message) for w in caught if issubclass(w.category, ThresholdPrecisionWarning)]
    return value, messages, best


def _warning_kind(messages: list[str]) -> str | None:
    if not messages:
        return None
    text = messages[0]
    if "negative beyond" in text:
        return "negative"
    return "zero" if "consistent with zero" in text else "relative"


def _errors(value: float, ref: float) -> dict:
    abs_error = abs(value - ref)
    return {
        "value": value,
        "abs_error": abs_error,
        "rel_error": abs_error / abs(ref) if ref else None,
    }


def run_case(pool, case: dict, seed: int) -> dict:
    import piquasso as pq

    rng = np.random.default_rng([seed, case["case_id"]])
    squeezing, unitary, alpha, pattern = build(case, rng)
    state = squeezed_interferometer_state(squeezing, unitary, case["transmissivity"], alpha)
    g = gbs_matrices(state)
    start = time.perf_counter()
    ref = reference(pool, state.covariance, state.mean, pattern)
    ref_seconds = time.perf_counter() - start
    p_ref = float(ref["value"])
    row = {
        **case,
        "pattern": pattern.tolist(),
        "p_ref": p_ref,
        "p_ref_digits": mp.nstr(ref["value"], 25),
        "ref_magnitude": ref["magnitude"],
        "ref_cancellation": ref["magnitude"] / abs(p_ref) if p_ref else None,
        "ref_abs_uncertainty": ref["uncertainty"],
        "ref_rel_uncertainty": ref["uncertainty"] / abs(p_ref) if p_ref else None,
        "ref_seconds": ref_seconds,
    }
    for impl in ("native", "python"):
        (value, rel_bound), messages, seconds = _timed(
            lambda impl=impl: threshold_probability(g, pattern, backend=impl, return_error=True)
        )
        same, bound, _ = _threshold_double(g, pattern, _kernels(impl))
        assert same == value
        err = _errors(value, p_ref)
        row[impl] = {
            **err,
            "bound_rel": rel_bound,
            "bound_abs": bound,
            # The reference is itself uncertain by ref_abs_uncertainty (~1e-37 absolute).
            "bound_held": err["abs_error"] <= bound + ref["uncertainty"],
            "bound_over_actual": bound / err["abs_error"] if err["abs_error"] else None,
            "warned": bool(messages),
            "warning_kind": _warning_kind(messages),
            "seconds": seconds,
        }
    program, sim = _pq_state(pq, squeezing, unitary, case["transmissivity"], alpha)
    pq_state = sim.execute(program).state
    pq_cov = np.asarray(pq_state.xpxp_covariance_matrix, dtype=float) / 2
    pq_mean = np.asarray(pq_state.xpxp_mean_vector, dtype=float)
    same_state = np.array_equal(pq_cov, state.covariance) and np.array_equal(pq_mean, state.mean)
    ref_pq = ref if same_state else reference(pool, pq_cov, pq_mean, pattern)
    p_ref_pq = float(ref_pq["value"])
    value, _, seconds = _timed(lambda: float(pq_state.get_threshold_detection_probability(pattern)))
    row["piquasso"] = {
        **_errors(value, p_ref_pq),
        "p_ref_own_state": p_ref_pq,
        "state_max_abs_diff": float(
            max(np.abs(pq_cov - state.covariance).max(), np.abs(pq_mean - state.mean).max())
        ),
        "ref_state_sensitivity": abs(p_ref_pq - p_ref) / abs(p_ref) if p_ref else None,
        "seconds": seconds,
    }
    return row, (g, pattern)


def _auto(task) -> tuple[float, object, list[str], float]:
    """``precision="auto"`` in a worker (its mpmath pass is single-threaded and slow)."""
    g, pattern = task
    (value, info), messages, seconds = _timed(
        lambda: threshold_probability(g, pattern, precision="auto", return_info=True)
    )
    return value, info, messages, seconds


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--procs", type=int, default=12)
    parser.add_argument("--smoke", action="store_true", help="modes 4, 8; three squeezings")
    parser.add_argument("--output", type=Path, help="JSON path (default: results/<stamp>.json)")
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    todo = cases(args.seed, args.smoke)
    rows, tasks = [], []
    start = time.perf_counter()
    with Pool(args.procs) as pool:
        for case in todo:
            row, task = run_case(pool, case, args.seed)
            rows.append(row)
            tasks.append(task)
            print(
                f"[{len(rows)}/{len(todo)} {time.perf_counter() - start:7.1f}s] "
                f"m={row['modes']} k={row['clicks']} r={row['squeezing']} "
                f"eta={row['transmissivity']} disp={int(row['displaced'])} {row['kind']}: "
                f"P={row['p_ref']:.3e} native={row['native']['rel_error']} "
                f"held={row['native']['bound_held']}/{row['python']['bound_held']}",
                flush=True,
            )
        for row, (value, info, messages, seconds) in zip(
            rows, pool.imap(_auto, tasks), strict=True
        ):
            row["auto"] = {
                **_errors(value, row["p_ref"]),
                "path": info.path,
                "digits": info.digits,
                "bound_rel": info.error,
                "double_bound_rel": info.double_error,
                "warned": bool(messages),
                "seconds": seconds,
            }
        print(f"auto done {time.perf_counter() - start:7.1f}s", flush=True)
    out = args.output or (
        Path(__file__).resolve().parent / "results" / f"threshold_reference_matrix_{stamp()}.json"
    )
    write_json(
        {
            "benchmark": "threshold_reference_matrix",
            "mp_dps": DPS,
            "grid": {
                "modes": list(MODES),
                "squeezing": list(SQUEEZING),
                "transmissivity": list(ETAS),
                "max_clicks": MAX_CLICKS,
                "smoke": args.smoke,
            },
            "environment": environment(
                seed=args.seed, precision_reference=f"mpmath {mp.__version__}, dps={DPS}"
            ),
            "total_seconds": time.perf_counter() - start,
            "results": rows,
        },
        out,
    )
    print("wrote", out)
    return out


if __name__ == "__main__":
    main()
