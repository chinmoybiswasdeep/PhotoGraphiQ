"""High-precision (mpmath, 50 digits) reference for GBS threshold probabilities of bench_gbs cases.

Usage: python threshold_mp_reference.py 24,12,0.9 28,14,0.9 ... [--output PATH]  (modes,clicks,eta;
r=0.5, seed=11, the exact states/patterns of bench_gbs.bench_case). Writes ``--output``, by default
results/threshold_mp_reference_<stamp>.json next to this script.
The reference is computed from each library's own double-precision covariance (the two differ by
~1e-15; ``ref_cov_sensitivity`` shows the effect on P). Cost: 2^clicks mp determinants (16 clicks:
~6-7 min on 12 processes).


P(click exactly on S) = sum_{T subset S} (-1)^{|T|} P(vacuum on Z u T), Z = modes outside S,
P(vac on R) = 2^{|R|} / sqrt(det(V_R + I))  (hbar=2, zero mean).
det(M_{Z u T}) = det(M_Z) det(C_T) with C = M_S - M_SZ M_Z^{-1} M_ZS (Schur complement, mp).
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import mpmath as mp
import numpy as np

DPS = 50
ROOT = Path(__file__).resolve().parents[2]
_C = None


def _init(c_rows):
    global _C
    mp.mp.dps = DPS
    _C = mp.matrix([[mp.mpf(x) for x in row] for row in c_rows])


def _chunk(args):
    k, masks = args
    acc = mp.mpf(0)
    for mask in masks:
        t = [i for i in range(k) if mask >> i & 1]
        idx = [q for i in t for q in (2 * i, 2 * i + 1)]
        d = mp.mpf(1) if not idx else mp.det(mp.matrix([[_C[a, b] for b in idx] for a in idx]))
        acc += (-1) ** len(t) * mp.mpf(2) ** len(t) / mp.sqrt(d)
    return str(acc)


def reference(cov, clicks, procs=12):
    mp.mp.dps = DPS
    m = len(clicks)
    s = [i for i in range(m) if clicks[i]]
    z = [i for i in range(m) if not clicks[i]]
    qs = [q for i in s for q in (2 * i, 2 * i + 1)]
    qz = [q for i in z for q in (2 * i, 2 * i + 1)]
    M = mp.matrix(
        [
            [mp.mpf(float(cov[a, b])) + (1 if a == b else 0) for b in range(2 * m)]
            for a in range(2 * m)
        ]
    )
    sub = lambda r, c: mp.matrix([[M[a, b] for b in c] for a in r])  # noqa: E731
    if qz:
        mz = sub(qz, qz)
        c = sub(qs, qs) - sub(qs, qz) * mp.inverse(mz) * sub(qz, qs)
        pref = mp.mpf(2) ** len(z) / mp.sqrt(mp.det(mz))
    else:
        c, pref = sub(qs, qs), mp.mpf(1)
    k = len(s)
    rows = [[str(c[a, b]) for b in range(2 * k)] for a in range(2 * k)]
    masks = list(range(1 << k))
    chunks = [(k, masks[i :: procs * 4]) for i in range(procs * 4)]
    with Pool(procs, initializer=_init, initargs=(rows,)) as pool:
        parts = pool.map(_chunk, chunks)
    return pref * mp.fsum(mp.mpf(p) for p in parts)


def case(m, photons, eta, r=0.5, seed=11):
    sys.path.insert(0, str(ROOT / "benchmarks" / "gaussian_boson_sampling"))
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import bench_gbs as b
    import piquasso as pq

    from photographiq.sampling import (
        haar_unitary,
        squeezed_interferometer_state,
        threshold_probability,
    )

    rng = np.random.default_rng(seed + 31 * m + photons)
    u = haar_unitary(m, rng=seed + m)
    alpha = np.zeros(m, dtype=complex)
    state = squeezed_interferometer_state(np.full(m, r), u, eta, alpha)
    clicks = b._pattern(m, photons, rng)
    program, sim = b._pq_state(pq, np.full(m, r), u, eta, alpha)
    pq_state = sim.execute(program).state
    p_pg, est_pg = threshold_probability(state, clicks, backend="native", return_error=True)
    p_py, est_py = (
        threshold_probability(state, clicks, backend="python", return_error=True)
        if photons <= 16
        else (None, None)
    )
    p_pq = float(pq_state.get_threshold_detection_probability(clicks))
    t0 = time.perf_counter()
    ref = reference(state.covariance, clicks)
    ref_pq = reference(np.asarray(pq_state.xpxp_covariance_matrix) / 2, clicks)
    dt = time.perf_counter() - t0
    out = {
        "modes": m,
        "clicks": photons,
        "eta": eta,
        "p_ref_mp_pgcov": mp.nstr(ref, 20),
        "p_ref_mp_pqcov": mp.nstr(ref_pq, 20),
        "ref_cov_sensitivity": float(abs(ref - ref_pq) / abs(ref)),
        "p_photographiq_native": p_pg,
        "p_photographiq_python": p_py,
        "p_piquasso": p_pq,
        "relerr_photographiq_native": float(abs(p_pg - ref) / ref),
        "relerr_photographiq_python": None if p_py is None else float(abs(p_py - ref) / ref),
        "relerr_piquasso": float(abs(p_pq - ref_pq) / ref_pq),
        "estimated_relerr_photographiq_native": est_pg,
        "estimated_relerr_photographiq_python": est_py,
        "seconds": dt,
    }
    print(out, flush=True)
    return out


def _case(text):
    # argparse turns the ValueError (wrong count or non-numeric field) into a usage error.
    case = tuple(map(float, text.split(",")))
    if len(case) != 3:
        raise ValueError(text)
    return case


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="+", type=_case, help="modes,clicks,eta (e.g. 24,12,0.9)")
    parser.add_argument("--output", type=Path, help="JSON path (default: results/<stamp>.json)")
    return parser.parse_intermixed_args(argv)


if __name__ == "__main__":
    args = parse_args()
    res = [case(int(m), int(k), e) for m, k, e in args.cases]
    from bench_common import environment, stamp, write_json

    out = args.output or (
        Path(__file__).resolve().parent / "results" / f"threshold_mp_reference_{stamp()}.json"
    )
    write_json(
        {
            "benchmark": "threshold_mp_reference",
            "mp_dps": DPS,
            "environment": environment(
                seed=11, precision_reference=f"mpmath {mp.__version__}, dps={DPS}"
            ),
            "results": res,
        },
        out,
    )
    print("wrote", out)
