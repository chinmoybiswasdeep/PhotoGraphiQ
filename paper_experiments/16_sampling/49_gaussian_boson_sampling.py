"""R49: Gaussian boson sampling with photon-number-resolving (PNR) detectors.

Scientific question: are PhotoGraphiQ's PNR probabilities P(n) = p_0 lhaf(A_{n+n}; gamma)/prod n!
(``photographiq.sampling.pnr_probability``: hafnians for undisplaced states, loop hafnians for
displaced ones, pure |lhaf(B_n)|^2 or mixed formula) correct for squeezed states through Haar
interferometers with loss and displacement, across modes, squeezing, photon number and cutoff?

References independent of the hafnian route:

* hafnian / loop hafnian: the defining sums over perfect matchings (with loops), written here,
  for 2N <= 12;
* Fock-space amplitudes: the product of single-mode squeezed vacua (native closed-form column
  ``gaussian_unitary(S)[:, 0]``) restricted to total photon number < c, then the m-mode passive
  interferometer's Fock matrix (native sector recurrence). For undisplaced, lossless inputs this is
  EXACT for every pattern with total < c (the truncated product is P_tot of the exact state and the
  passive unitary preserves sectors), so P(n) must agree to rounding at every cutoff. With
  displacement, P D P acts after the truncation, so agreement is a convergence in the cutoff;
* Piquasso 8.0.1 ``GaussianState.get_particle_detection_probability`` on the identical program
  (``benchmarks/gaussian_boson_sampling/bench_gbs._pq_state``), incl. loss and displacement;
* sum rule: the hafnian probability mass below the cutoff equals the exact retained mass of the
  Fock reference (undisplaced, lossless).

Haar unitaries from scipy.stats.unitary_group with fixed seeds.
"""

from __future__ import annotations

import argparse
import functools
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO / "benchmarks" / "gaussian_boson_sampling"))
from bench_gbs import _pq_state  # noqa: E402

import piquasso as pq  # noqa: E402

from photographiq import native  # noqa: E402
from photographiq import sampling as smp  # noqa: E402

SEED = 20261149
ENGINE = native.require("fock")
# ----- acceptance criteria (declared before evaluation) ---------------------------------
TOL_HAF_REL = 1e-12  # (loop) hafnian vs matching-sum definition, relative to sum |terms| scale
TOL_EXACT_FOCK = 1e-13  # undisplaced lossless P(n) vs exact-in-truncation Fock amplitudes (absolute)
TOL_PIQUASSO_REL = 1e-9  # relative to P for P > 1e-12 (Piquasso uses its own hafnian / covariance)
TOL_MASS = 1e-12  # hafnian mass below the cutoff vs the exact retained Fock mass
TOL_DISPLACED_TOP = 1e-10  # displaced P(n) vs Fock reference at the largest cutoff (convergence)
PHOTON_CAP = 12  # patterns with more photons are not evaluated (hafnian cost 2^(N/2) N^4)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="more modes and photons")
    return parser.parse_args()


def haar(m, seed):
    return stats.unitary_group.rvs(m, random_state=seed)


# ----- definitions (perfect matchings) ------------------------------------------------------------
def haf_definition(a, loops=False):
    @functools.lru_cache(maxsize=None)
    def rec(rest):
        if not rest:
            return 1.0 + 0j, 1.0
        i, others = rest[0], rest[1:]
        total, mag = 0j, 0.0
        if loops:
            v, w = rec(others)
            total += a[i, i] * v
            mag += abs(a[i, i]) * w
        for k, j in enumerate(others):
            v, w = rec(others[:k] + others[k + 1:])
            total += a[i, j] * v
            mag += abs(a[i, j]) * w
        return total, mag

    return rec(tuple(range(len(a))))


def hafnian_rows(rng, sizes):
    rows = []
    for size in sizes:
        x = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        a = (x + x.T) / 2
        for loops in (False, True):
            if not loops and size % 2:
                continue
            ref, scale = haf_definition(a, loops)
            for backend in ("native", "python"):
                val = smp.loop_hafnian(a, backend=backend) if loops else smp.hafnian(a, backend=backend)
                rows.append({"size": size, "loop": loops, "backend": backend, "abs_ref": abs(ref),
                             "scaled_error": abs(val - ref) / scale})
    return rows


# ----- Fock-space reference -----------------------------------------------------------------------
def fock_reference(r, u, alpha, cutoff):
    """|amplitudes|^2 (dict pattern -> P) of the truncated Fock evolution, and the input mass kept."""
    m = len(r)
    basis = [tuple(int(x) for x in b) for b in ENGINE.basis(m, cutoff)]
    cols = [ENGINE.gaussian_unitary(np.diag([np.exp(-x), np.exp(x)]), np.zeros(2), cutoff)[:, 0] for x in r]
    psi = np.array([math.prod(cols[i][b[i]] for i in range(m)) for b in basis])
    kept = float(np.vdot(psi, psi).real)
    psi = ENGINE.gaussian_unitary(smp.passive_symplectic(u), np.zeros(2 * m), cutoff) @ psi
    if alpha is not None:
        for i, a in enumerate(alpha):
            op = ENGINE.gaussian_unitary(np.eye(2), np.array([2 * a.real, 2 * a.imag]), cutoff)
            ENGINE.apply_pure(psi, m, cutoff, [i], op)
    return {b: abs(psi[k]) ** 2 for k, b in enumerate(basis)}, kept


def exact_fock_rows(rng, configs):
    rows = []
    for m, r, cutoff, displaced in configs:
        # Same unitary and displacement for every cutoff of one (m, r, displaced) family, so the
        # cutoff sweep is a convergence study of ONE state.
        local = np.random.default_rng([SEED, m, int(round(1000 * r)), int(displaced)])
        u = haar(m, int(local.integers(2**31)))
        rs = np.full(m, r)
        alpha = (0.3 * np.exp(2j * np.pi * local.random(m))) if displaced else None
        state = smp.squeezed_interferometer_state(rs, u, 1.0, alpha)
        g = smp.gbs_matrices(state)
        ref, kept = fock_reference(rs, u, alpha, cutoff)
        captured = []
        for t, p_ref in ref.items():
            if sum(t) > PHOTON_CAP:
                continue  # hafnian cost ~2^(N/2) N^4: displaced cutoffs up to 40 are compared for |n| <= PHOTON_CAP
            p = smp.pnr_probability(g, t)
            captured.append(p)
            rows.append({"modes": m, "squeezing": r, "cutoff": cutoff, "displaced": displaced,
                         "pattern": str(t), "photons": sum(t), "p_hafnian": p, "p_fock": p_ref,
                         "abs_error": abs(p - p_ref), "rel_error": abs(p - p_ref) / p_ref if p_ref > 0 else np.nan,
                         "p_mixed_formula": smp.pnr_probability(g, t, method="mixed") if sum(t) <= 6 else np.nan})
        if not displaced and cutoff - 1 <= PHOTON_CAP:
            # Sum rule: hafnian mass below the cutoff equals the exact retained Fock mass (1 - tail).
            rows[-1]["_summary"] = {"modes": m, "squeezing": r, "cutoff": cutoff,
                                    "hafnian_captured_mass": math.fsum(captured),
                                    "fock_captured_mass": math.fsum(ref.values()),
                                    "mass_difference": abs(math.fsum(captured) - math.fsum(ref.values())),
                                    "truncated_mass": 1 - math.fsum(captured), "fock_input_mass_kept": kept}
        print(f"  Fock ref m={m} r={r} c={cutoff} disp={displaced}: max abs err "
              f"{max(x['abs_error'] for x in rows if x['modes'] == m and x['cutoff'] == cutoff and x['displaced'] == displaced and x['squeezing'] == r):.1e}",
              flush=True)
    return rows


def piquasso_rows(rng, configs):
    rows = []
    for m, r, eta, displaced, max_photons in configs:
        u = haar(m, int(rng.integers(2**31)))
        rs = np.full(m, r)
        alpha = 0.4 * rng.random(m) * np.exp(2j * np.pi * rng.random(m)) if displaced else np.zeros(m, complex)
        state = smp.squeezed_interferometer_state(rs, u, eta, alpha if displaced else None)
        program, sim = _pq_state(pq, rs, u, eta, alpha)
        pq_state = sim.execute(program).state
        g = smp.gbs_matrices(state)
        for total in range(0, max_photons + 1, 2 if not displaced and eta == 1 else 1):
            for _ in range(4):
                t = np.zeros(m, dtype=int)
                for k in rng.choice(m, total, replace=True):
                    t[k] += 1
                p = smp.pnr_probability(g, t)
                p_py = smp.pnr_probability(g, t, backend="python")
                p_pq = float(pq_state.get_particle_detection_probability(tuple(int(x) for x in t)))
                rows.append({"modes": m, "squeezing": r, "transmissivity": eta, "displaced": displaced,
                             "pattern": str(tuple(t)), "photons": total, "pure": g.pure, "p_native": p,
                             "p_python": p_py, "p_piquasso": p_pq,
                             "rel_native_vs_python": abs(p - p_py) / abs(p) if p else 0.0,
                             "rel_native_vs_piquasso": abs(p - p_pq) / abs(p) if abs(p) > 1e-300 else np.nan})
        print(f"  Piquasso m={m} r={r} eta={eta} disp={displaced}: done", flush=True)
    return rows


def main():
    args = parse_args()
    # Accuracy study: pin the shared OpenMP runtime of photographiq._native to one thread. On this
    # loaded machine the multi-threaded kernels spent most of their time spin-waiting (an aborted
    # --full run accumulated 11,000 CPU-seconds); results do not depend on the thread count.
    native.require("sampling").set_num_threads(1)
    rng = common.rng(SEED)
    haf = hafnian_rows(rng, list(range(1, 13)))
    fock_configs = [(2, 0.5, c, False) for c in (6, 10, 16)] + [(3, 0.5, 10, False), (3, 1.0, 12, False),
                                                               (4, 0.3, 9, False), (2, 0.5, 8, True),
                                                               (2, 0.5, 14, True), (2, 0.5, 22, True),
                                                               (2, 0.5, 30, True)]
    if args.full:
        fock_configs += [(4, 0.6, 12, False), (5, 0.4, 9, False), (2, 0.5, 40, True), (3, 0.4, 12, True),
                         (3, 0.4, 20, True)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fock = exact_fock_rows(rng, fock_configs)
    summaries = [r.pop("_summary") for r in fock if "_summary" in r]
    pq_configs = [(4, 0.5, 1.0, False, 8), (6, 0.5, 0.9, False, 8), (6, 0.8, 1.0, True, 6),
                  (8, 0.6, 0.7, True, 8)]
    if args.full:
        pq_configs += [(10, 0.7, 0.9, False, 12), (12, 0.5, 1.0, False, 12), (10, 0.5, 0.8, True, 10)]
    pqr = piquasso_rows(rng, pq_configs)

    failures = [{"check": "hafnian", **r} for r in haf if r["scaled_error"] > TOL_HAF_REL]
    top_disp = {}
    for r in fock:
        if r["displaced"]:
            key = (r["modes"], r["squeezing"])
            top_disp[key] = max(top_disp.get(key, 0), r["cutoff"])
    for r in fock:
        if not r["displaced"] and r["abs_error"] > TOL_EXACT_FOCK:
            failures.append({"check": "undisplaced exact Fock", **r})
        if not r["displaced"] and not np.isnan(r["p_mixed_formula"]) and abs(r["p_mixed_formula"] - r["p_hafnian"]) > TOL_EXACT_FOCK:
            failures.append({"check": "pure vs mixed formula", **r})
        if r["displaced"] and r["cutoff"] == top_disp[(r["modes"], r["squeezing"])] and r["photons"] <= 4 \
                and r["abs_error"] > TOL_DISPLACED_TOP:
            failures.append({"check": "displaced convergence at top cutoff", **r})
    for s in summaries:
        if s["mass_difference"] > TOL_MASS:
            failures.append({"check": "mass sum rule", **s})
    for r in pqr:
        if r["p_native"] > 1e-12 and (r["rel_native_vs_piquasso"] > TOL_PIQUASSO_REL or r["rel_native_vs_python"] > TOL_PIQUASSO_REL):
            failures.append({"check": "Piquasso / python agreement", **r})

    extra = {"backend": "photographiq.sampling (native and NumPy hafnian / loop hafnian)",
             "reference": "perfect-matching definitions (this script); native Fock-space evolution (exact in truncation "
                          "for undisplaced lossless states); Piquasso 8.0.1 get_particle_detection_probability",
             "precision": "complex128", "cutoff": "see rows", "seed": SEED, "full": args.full,
             "parameters": {"openmp_threads": 1, "fock_configs": fock_configs, "piquasso_configs": pq_configs},
             "acceptance": {"TOL_HAF_REL": TOL_HAF_REL, "TOL_EXACT_FOCK": TOL_EXACT_FOCK,
                            "TOL_PIQUASSO_REL": TOL_PIQUASSO_REL, "TOL_MASS": TOL_MASS,
                            "TOL_DISPLACED_TOP": TOL_DISPLACED_TOP}}
    common.save_result(fock, "R49_gaussian_boson_sampling", extra=extra)
    common.save_result(summaries, "R49_gbs_mass_sum_rules", extra=extra)
    common.save_result(haf, "R49_gbs_hafnians", extra=extra)
    common.save_result(pqr, "R49_gbs_vs_piquasso", extra=extra)
    if failures:
        common.save_csv(failures, "R49_gaussian_boson_sampling_FAILURES")

    plt = common.setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.1))
    und = [r for r in fock if not r["displaced"] and r["p_fock"] > 0]
    axes[0].loglog([r["p_fock"] for r in und], [r["p_hafnian"] for r in und], "o", ms=3, color="#24677b", alpha=0.7)
    lim = [min(r["p_fock"] for r in und) / 3, 1]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set(xlabel="P(n), Fock-space evolution (exact in truncation)", ylabel="P(n), hafnian",
                title="(a) undisplaced, lossless")
    axes[1].loglog([r["p_fock"] for r in und], [max(r["abs_error"], 1e-19) for r in und], "o", ms=3, color="#24677b",
                   alpha=0.7, label="hafnian vs Fock (undisplaced)")
    pv = [r for r in pqr if r["p_native"] > 0]
    axes[1].loglog([r["p_native"] for r in pv], [max(r["rel_native_vs_piquasso"] * r["p_native"], 1e-19) for r in pv],
                   "^", ms=3, color="#c66d27", alpha=0.7, label="native vs Piquasso (loss, displacement)")
    axes[1].axhline(TOL_EXACT_FOCK, color="k", ls="--", lw=1)
    axes[1].set(xlabel="probability P(n)", ylabel="absolute error (1e-19 = exact)", title="(b) error vs probability magnitude")
    axes[1].legend(fontsize=6.5)
    for (m, r_), color in zip(sorted({(r["modes"], r["squeezing"]) for r in fock if r["displaced"]}),
                              ("#8a3b8f", "#627a36", "#c66d27")):
        cut = sorted({r["cutoff"] for r in fock if r["displaced"] and r["modes"] == m})
        worst = [max(r["abs_error"] for r in fock if r["displaced"] and r["modes"] == m and r["cutoff"] == c
                     and r["photons"] <= 4) for c in cut]
        axes[2].semilogy(cut, [max(w, 1e-19) for w in worst], "o-", color=color, label=f"m={m}, r={r_}, |alpha|<=0.3")
    axes[2].set(xlabel="Fock cutoff of the reference", ylabel="max |P_lhaf - P_Fock| (|n| <= 4)",
                title="(c) displaced: loop hafnian vs Fock")
    axes[2].legend(fontsize=7)
    for backend, color in (("native", "#24677b"), ("python", "#c66d27")):
        for loops, marker in ((False, "o"), (True, "s")):
            sub = [r for r in haf if r["backend"] == backend and r["loop"] == loops]
            axes[3].semilogy([r["size"] for r in sub], [max(r["scaled_error"], 1e-19) for r in sub], marker,
                             color=color, ms=4, label=f"{backend} {'loop ' if loops else ''}hafnian")
    axes[3].axhline(TOL_HAF_REL, color="k", ls="--", lw=1)
    axes[3].set(xlabel="matrix size (odd: loop hafnian only)", ylabel="|haf - def| / sum|terms| (1e-19 = 0)", title="(d) hafnian vs matching sum")
    axes[3].legend(fontsize=6.5)
    common.save_figure(fig, "R49_gaussian_boson_sampling")
    plt.close(fig)

    worst_und = max(r["abs_error"] for r in fock if not r["displaced"])
    worst_pq = max(r["rel_native_vs_piquasso"] for r in pqr if r["p_native"] > 1e-12)
    disp_top = max(r["abs_error"] for r in fock if r["displaced"] and r["photons"] <= 4
                   and r["cutoff"] == top_disp[(r["modes"], r["squeezing"])])
    common.save_caption(
        "R49_gaussian_boson_sampling",
        f"""
**R49 - Gaussian boson sampling with PNR detection.** Squeezed vacua (q-squeezed, r per mode) through
scipy Haar interferometers, optional loss and coherent displacement (`squeezed_interferometer_state`).
(a, b) Undisplaced lossless states (m = 2..{max(c[0] for c in fock_configs)} modes, r = 0.3..1.0): hafnian probabilities agree with an
independent Fock-space evolution (product of single-mode squeezed columns, total-photon cutoff, m-mode passive
Fock matrix), which is exact for every pattern below the cutoff: worst absolute error {worst_und:.1e}; the pure
|haf(B)|^2 and mixed haf(A) formulas agree. (b) also: lossy and displaced states (m = 4..{max(c[0] for c in pq_configs)}, eta =
0.7..1) against Piquasso 8.0.1 `get_particle_detection_probability`, worst relative difference {worst_pq:.1e} for
P > 1e-12. (c) Displaced states: the loop-hafnian probabilities are approached by the Fock reference as its
cutoff grows (displacement applied after truncation; {disp_top:.1e} at the largest cutoff, |n| <= 4). (d) Hafnian
and loop hafnian vs the perfect-matching definition. Sum rule: mass difference
{max(s['mass_difference'] for s in summaries):.1e} (hafnian mass below the cutoff vs the exact retained Fock mass,
undisplaced cases with cutoff <= {PHOTON_CAP + 1}); patterns with more than {PHOTON_CAP} photons are not evaluated.
Seed {SEED}. Source: R49 (16_sampling/49_gaussian_boson_sampling.py).
""",
    )
    common.print_summary("R49 GBS (PNR)", fock_rows=len(fock), worst_undisplaced=f"{worst_und:.2e}",
                         displaced_top=f"{disp_top:.2e}", worst_vs_piquasso=f"{worst_pq:.2e}",
                         worst_haf=f"{max(r['scaled_error'] for r in haf):.2e}",
                         mass=f"{max(s['mass_difference'] for s in summaries):.1e}",
                         failures=len(failures))
    if failures:
        raise AssertionError(f"{len(failures)} R49 checks failed; see R49_gaussian_boson_sampling_FAILURES.csv")


if __name__ == "__main__":
    main()
