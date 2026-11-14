"""R50: Threshold-detector GBS: torontonian click probabilities, cancellation and precision.

Scientific question: how accurate are PhotoGraphiQ's threshold (click) probabilities
``photographiq.sampling.threshold_probability`` in double precision (native C++ and Python
sums), is the reported rounding bound B (``return_error`` / ``ThresholdInfo``) reliable, where
does the alternating inclusion-exclusion sum lose accuracy by cancellation (low probabilities,
many clicks, weak squeezing), and does the multiprecision path (``precision=<digits>`` and the
opt-in ``precision="auto"``) recover it? How does Piquasso 8.0.1 compare on the same states?

Method (REUSED, rerun here - no recorded numbers are presented as new):
``benchmarks/gaussian_boson_sampling/threshold_reference_matrix.py`` supplies the case builder
(``build``: Haar interferometer, squeezing r, transmissivity eta, optional displacement
alpha_j = 0.4 u e^{2 pi i v}, random k-click pattern; "zero" cases with an untouched vacuum mode
that clicks), the per-case evaluation ``run_case`` (native / Python double values with their
bound B, warnings, Piquasso on the identical program compared with a reference for ITS OWN stored
state) and the 40-digit mpmath reference ``reference`` (Schur complement on the dark modes, then a
depth-first Cholesky-extension walk over the 2^k subsets; separate code from PhotoGraphiQ's own
``precision=`` path; uncertainty 10^(3-40) sum|terms|). This script adds: the package's
``precision=50`` path and ``precision="auto"`` (path, digits, ThresholdInfo) per case, a
precision sweep (digits 16..60) on the hardest cases, the normalisation of full click
distributions (sum over 2^m patterns), and the acceptance tests below. The earlier recorded matrix
(docs/validation/gbs-threshold-accuracy.md, results/threshold_reference_matrix_20261111-233720.json)
is cited for comparison only.

Do NOT read double-precision results as "full double precision": the relative error grows with the
cancellation ratio sum|terms| / |P| (reported per case) and exceeds 1 for many-click, weak-squeezing
cases; the bound B is what the function promises, and ``precision="auto"`` is the remedy.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO / "benchmarks" / "gaussian_boson_sampling"))
import threshold_reference_matrix as trm  # noqa: E402

from photographiq import sampling as smp  # noqa: E402
from photographiq.sampling.gbs import ThresholdPrecisionWarning, gbs_matrices  # noqa: E402

SEED = 20261150
SQUEEZING = trm.SQUEEZING  # (0.05, 0.2, 0.5, 1.0, 1.5)
ETAS = trm.ETAS
# ----- acceptance criteria (declared before evaluation) ---------------------------------
AUTO_RTOL = 1e-6  # precision="auto" default rtol; its result must meet it against the reference
MP_DIGITS = 50  # explicit multiprecision path checked against the independent 40-digit reference
MP_AGREEMENT_FACTOR = 10  # |P_50 - P_ref| <= 10 (ref abs uncertainty + P_50 * reported mp error)
WARN_THRESHOLD = 1e-6  # the function must warn whenever its own relative bound exceeds this
TOL_NORMALISATION = 1e-10  # |sum of all 2^m click probabilities - 1| (double, small m)
RELIABLE_REFERENCE = 1e-8  # cases whose reference relative uncertainty exceeds this are not scored
ZERO_SCALE = 1e-14  # zero-probability cases: |P_ref(stored state)| at the rounding scale of V entries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="modes up to 20, clicks up to 12")
    parser.add_argument("--procs", type=int, default=8)
    return parser.parse_args()


def case_list(full, rng):
    out = []
    modes = (4, 8, 12, 16, 20) if full else (4, 8, 12)
    clicks = (1, 2, 3, 4, 6, 8, 10, 12) if full else (1, 2, 3, 4, 6, 8, 10)
    for m in modes:
        for k in [k for k in clicks if k <= m]:
            for r in SQUEEZING:
                out.append({"kind": "random", "modes": m, "clicks": k, "squeezing": r,
                            "transmissivity": float(rng.choice(ETAS)), "displaced": bool(rng.random() < 0.5),
                            "vacuum_mode": None})
        for k in sorted({1, 3, min(m, 8)}):
            out.append({"kind": "zero", "modes": m, "clicks": k, "squeezing": float(rng.choice(SQUEEZING)),
                        "transmissivity": float(rng.choice(ETAS)), "displaced": bool(rng.random() < 0.5),
                        "vacuum_mode": int(rng.integers(m))})
    for i, case in enumerate(out):
        case["case_id"], case["unitary_seed"] = i, SEED * 1000 + i
    return out


def extra_paths(row, g, pattern):
    """Package mpmath (50 digits) and precision='auto' on the same matrices."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ThresholdPrecisionWarning)
        t0 = time.perf_counter()
        p50, info50 = smp.threshold_probability(g, pattern, precision=MP_DIGITS, return_info=True)
        t50 = time.perf_counter() - t0
        t0 = time.perf_counter()
        pa, info = smp.threshold_probability(g, pattern, precision="auto", return_info=True)
        ta = time.perf_counter() - t0
    ref = row["p_ref"]
    row["mp50"] = {"value": p50, "rel_error": abs(p50 - ref) / abs(ref) if ref else None, "abs_error": abs(p50 - ref),
                   "reported_error": info50.error, "seconds": t50}
    row["auto"] = {"value": pa, "rel_error": abs(pa - ref) / abs(ref) if ref else None, "abs_error": abs(pa - ref),
                   "path": info.path, "digits": info.digits, "reported_error": info.error,
                   "double_error": info.double_error, "seconds": ta,
                   "warned": any(issubclass(w.category, ThresholdPrecisionWarning) for w in caught)}


def flatten(row):
    flat = {k: row[k] for k in ("case_id", "kind", "modes", "clicks", "squeezing", "transmissivity", "displaced",
                                "p_ref", "ref_cancellation", "ref_rel_uncertainty", "ref_abs_uncertainty", "ref_seconds")}
    for impl in ("native", "python", "piquasso", "mp50", "auto"):
        for key, value in row[impl].items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                flat[f"{impl}_{key}"] = value
    return flat


def precision_sweep(pool, rows, n_cases):
    hard = sorted((r for r in rows if r["kind"] == "random" and r["ref_rel_uncertainty"] is not None
                   and r["ref_rel_uncertainty"] < RELIABLE_REFERENCE),
                  key=lambda r: -(r["ref_cancellation"] or 0))[:n_cases]
    out = []
    for r in hard:
        squeezing, unitary, alpha, pattern = trm.build(r, np.random.default_rng([SEED, r["case_id"]]))
        state = smp.squeezed_interferometer_state(squeezing, unitary, r["transmissivity"], alpha)
        g = gbs_matrices(state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ThresholdPrecisionWarning)
            for digits in (None, 16, 20, 24, 30, 40, 50, 60):
                value, info = smp.threshold_probability(g, pattern, precision=digits, backend="native",
                                                        return_info=True)
                out.append({"case_id": r["case_id"], "modes": r["modes"], "clicks": r["clicks"],
                            "squeezing": r["squeezing"], "cancellation": r["ref_cancellation"],
                            "digits": 15.95 if digits is None else digits, "path": info.path,
                            "rel_error": abs(value - r["p_ref"]) / abs(r["p_ref"]), "reported_error": info.error})
    return out


def normalisation_rows(rng):
    out = []
    for m in (2, 4, 6, 8):
        for r, disp in ((0.5, False), (1.0, True)):
            u = smp.haar_unitary(m, rng=int(rng.integers(2**31)))
            alpha = 0.4 * np.exp(2j * np.pi * rng.random(m)) if disp else None
            state = smp.squeezed_interferometer_state(np.full(m, r), u, 0.9, alpha)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ThresholdPrecisionWarning)
                dist = smp.threshold_distribution(state)
            out.append({"modes": m, "squeezing": r, "displaced": disp, "patterns": len(dist),
                        "normalisation_error": abs(math.fsum(dist.values()) - 1),
                        "most_negative": min(min(dist.values()), 0.0)})
    return out


def main():
    args = parse_args()
    rng = common.rng(SEED)
    todo = case_list(args.full, rng)
    rows = []
    start = time.perf_counter()
    with Pool(args.procs) as pool:
        for case in todo:
            row, (g, pattern) = trm.run_case(pool, case, SEED)
            extra_paths(row, g, pattern)
            rows.append(row)
            print(f"[{len(rows)}/{len(todo)} {time.perf_counter() - start:6.1f}s] m={row['modes']} k={row['clicks']} "
                  f"r={row['squeezing']} {row['kind']}: P={row['p_ref']:.2e} native rel={row['native']['rel_error']} "
                  f"auto={row['auto']['path']}/{row['auto']['digits']}", flush=True)
        sweep = precision_sweep(pool, rows, 4 if not args.full else 8)
    norms = normalisation_rows(rng)

    flat = [flatten(r) for r in rows]
    failures = []
    for r in rows:
        reliable = r["ref_rel_uncertainty"] is not None and r["ref_rel_uncertainty"] < RELIABLE_REFERENCE
        for impl in ("native", "python"):
            d = r[impl]
            if not d["bound_held"]:
                failures.append({"check": f"double bound violated ({impl})", **flatten(r)})
            if d["bound_rel"] > WARN_THRESHOLD and not d["warned"]:
                failures.append({"check": f"missing ThresholdPrecisionWarning ({impl})", **flatten(r)})
        if r["kind"] == "zero":
            # The exact P of the STORED state is not exactly 0 (its covariance is physical only to
            # rounding, so the reference itself can be ~ -1e-17); the double value is scored against
            # that reference by bound_held above. Here: the stored state's P is at rounding scale,
            # and the user is warned (negative, zero-consistent or imprecise result).
            if abs(r["p_ref"]) > ZERO_SCALE or not (r["native"]["warned"] and r["python"]["warned"]):
                failures.append({"check": "zero case not at rounding scale or not warned", **flatten(r)})
            continue
        if reliable and r["auto"]["rel_error"] > AUTO_RTOL:
            failures.append({"check": "precision='auto' misses rtol", **flatten(r)})
        limit = MP_AGREEMENT_FACTOR * (r["ref_abs_uncertainty"] + abs(r["mp50"]["value"]) * r["mp50"]["reported_error"])
        if r["mp50"]["abs_error"] > limit:
            failures.append({"check": "precision=50 vs 40-digit reference", "limit": limit, **flatten(r)})
    failures += [{"check": "normalisation", **n} for n in norms if n["normalisation_error"] > TOL_NORMALISATION]

    random_rows = [r for r in rows if r["kind"] == "random"]
    over = {impl: sum(1 for r in random_rows if (r[impl]["rel_error"] or 0) > 1e-6) for impl in ("native", "python", "piquasso")}
    extra = {"backend": "photographiq.sampling.threshold_probability: native C++ and Python double sums, mpmath "
                        "precision=50, precision='auto'; Piquasso 8.0.1 get_threshold_detection_probability",
             "reference": "40-digit mpmath reference of benchmarks/gaussian_boson_sampling/threshold_reference_matrix.py "
                          "(rerun here)", "precision": "float64 / mpmath", "cutoff": None, "seed": SEED,
             "full": args.full, "reference_dps": trm.DPS,
             "parameters": {"cases": len(todo), "squeezing": SQUEEZING, "transmissivity": ETAS,
                            "displacement": "alpha_j = 0.4 u e^{2 pi i v} on half the cases"},
             "double_rel_error_above_1e-6": over,
             "cited_previous_record": "docs/validation/gbs-threshold-accuracy.md (382 cases, 2026-11-11), not reused as data",
             "acceptance": {"AUTO_RTOL": AUTO_RTOL, "MP_DIGITS": MP_DIGITS, "MP_AGREEMENT_FACTOR": MP_AGREEMENT_FACTOR,
                            "WARN_THRESHOLD": WARN_THRESHOLD, "TOL_NORMALISATION": TOL_NORMALISATION,
                            "RELIABLE_REFERENCE": RELIABLE_REFERENCE, "ZERO_SCALE": ZERO_SCALE,
                            "bound": "|double - ref| <= B + ref uncertainty for every case"}}
    common.save_result(flat, "R50_threshold_gbs", extra=extra)
    common.save_result(sweep, "R50_threshold_precision_sweep", extra=extra)
    common.save_result(norms, "R50_threshold_normalisation", extra=extra)
    if failures:
        common.save_csv(failures, "R50_threshold_gbs_FAILURES")

    plt = common.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    ok = [r for r in random_rows if r["ref_rel_uncertainty"] is not None and r["ref_rel_uncertainty"] < RELIABLE_REFERENCE]
    for impl, marker, color in (("native", "o", "#24677b"), ("python", "s", "#c66d27"), ("piquasso", "^", "#8a3b8f"),
                                ("auto", "x", "#000000")):
        axes[0, 0].loglog([abs(r["p_ref"]) for r in ok], [max(r[impl]["rel_error"], 1e-18) for r in ok], marker,
                          color=color, ms=3.5, alpha=0.75, mfc="none" if marker != "x" else None,
                          label=impl if impl != "auto" else 'precision="auto"')
    axes[0, 0].axhline(1e-6, color="grey", ls=":", lw=1)
    axes[0, 0].set(xlabel="reference probability P", ylabel="relative error (1e-18 = exact in float64)",
                   title="(a) error vs probability magnitude")
    axes[0, 0].legend(fontsize=7)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(SQUEEZING)))
    for r_sq, color in zip(SQUEEZING, cmap):
        sub = [r for r in ok if r["squeezing"] == r_sq]
        axes[0, 1].loglog([r["ref_cancellation"] for r in sub], [max(r["native"]["rel_error"], 1e-18) for r in sub], "o",
                          color=color, ms=3.5, label=f"r = {r_sq}")
    c = np.logspace(0, 20, 50)
    axes[0, 1].loglog(c, c * np.finfo(float).eps, "k--", lw=1, label="eps x cancellation")
    axes[0, 1].set(xlabel="cancellation  sum|terms| / |P|  (reference)", ylabel="native double relative error",
                   title="(b) cancellation controls the double error")
    axes[0, 1].legend(fontsize=7)
    ratio = [r["native"]["bound_over_actual"] for r in random_rows if r["native"]["bound_over_actual"]]
    ratio_py = [r["python"]["bound_over_actual"] for r in random_rows if r["python"]["bound_over_actual"]]
    bins = np.logspace(-1, 8, 37)
    axes[1, 0].hist(ratio, bins=bins, alpha=0.6, color="#24677b", label="native")
    axes[1, 0].hist(ratio_py, bins=bins, alpha=0.6, color="#c66d27", label="Python")
    axes[1, 0].axvline(1, color="k", ls="--", lw=1, label="bound = actual (violation left of line)")
    axes[1, 0].set(xscale="log", xlabel="reported bound B / actual |error|", ylabel="cases",
                   title="(c) rounding bound is conservative")
    axes[1, 0].legend(fontsize=7)
    for cid, color in zip(sorted({s["case_id"] for s in sweep}), plt.cm.tab10(np.arange(10))):
        sub = [s for s in sweep if s["case_id"] == cid]
        axes[1, 1].semilogy([s["digits"] for s in sub], [max(s["rel_error"], 1e-17) for s in sub], "o-", color=color,
                            ms=3.5, label=f"m={sub[0]['modes']}, k={sub[0]['clicks']}, r={sub[0]['squeezing']}, "
                                          f"canc. {sub[0]['cancellation']:.0e}")
    axes[1, 1].axvspan(39, 61, color="grey", alpha=0.12, lw=0)
    axes[1, 1].text(50, 1e-14, chr(10).join(["beyond the 40-digit", "reference: agreement", "to float64 output"]),
                    fontsize=6.5, ha="center")
    axes[1, 1].set(xlabel="decimal digits (15.95 = double)", ylabel="relative error vs 40-digit reference (1e-17 = 0)",
                   title="(d) error vs working precision (hardest cases)")
    axes[1, 1].legend(fontsize=6)
    common.save_figure(fig, "R50_threshold_gbs")
    plt.close(fig)

    bound_held = sum(r["native"]["bound_held"] and r["python"]["bound_held"] for r in rows)
    auto_scored = [r for r in ok]
    auto_mp = sum(1 for r in auto_scored if r["auto"]["path"] == "mpmath")
    worst_auto = max(r["auto"]["rel_error"] for r in auto_scored)
    min_ratio = min(ratio + ratio_py)
    worst_double = max(r["native"]["rel_error"] for r in ok)
    common.save_caption(
        "R50_threshold_gbs",
        f"""
**R50 - threshold-detector GBS: cancellation and precision.** {len(random_rows)} random cases (modes
{sorted({r['modes'] for r in rows})}, clicks k <= {max(r['clicks'] for r in rows)}, squeezing r in {list(SQUEEZING)}, transmissivity in
{list(ETAS)}, half displaced) plus {len(rows) - len(random_rows)} zero-probability cases, all scored against a 40-digit mpmath reference
of the stored state (benchmarks/gaussian_boson_sampling/threshold_reference_matrix.py, rerun; cases with
reference uncertainty > {RELIABLE_REFERENCE:g} excluded from scoring: {len(random_rows) - len(ok)}). (a) Relative error vs
probability: double precision (native, Python, Piquasso 8.0.1) degrades for small P - native errors above 1e-6 in
{over['native']} / Python {over['python']} / Piquasso {over['piquasso']} of {len(random_rows)} cases (worst native {worst_double:.1e}); `precision="auto"` (mpmath
in {auto_mp} cases) stays within {worst_auto:.1e} of the reference everywhere (rtol {AUTO_RTOL:g}). (b) The double error tracks eps x
cancellation sum|terms|/|P|, largest for weak squeezing and many clicks: double precision is NOT full precision
here. (c) The reported rounding bound B held in {bound_held}/{len(rows)} cases (native and Python); smallest B/actual
{min_ratio:.1f}. (d) Error vs mpmath working digits for the {len({s['case_id'] for s in sweep})} highest-cancellation cases. Full click
distributions (m <= 8) sum to 1 within {max(n['normalisation_error'] for n in norms):.1e}. Seed {SEED}. Source: R50
(16_sampling/50_threshold_gbs.py).
""",
    )
    common.print_summary("R50 threshold GBS", cases=len(rows), bound_held=f"{bound_held}/{len(rows)}",
                         double_over_1e6=over, worst_double_native=f"{worst_double:.2e}", auto_mpmath=auto_mp,
                         worst_auto=f"{worst_auto:.2e}", min_bound_ratio=f"{min_ratio:.2f}",
                         normalisation=f"{max(n['normalisation_error'] for n in norms):.1e}", failures=len(failures))
    if failures:
        raise AssertionError(f"{len(failures)} R50 checks failed; see R50_threshold_gbs_FAILURES.csv")


if __name__ == "__main__":
    main()
