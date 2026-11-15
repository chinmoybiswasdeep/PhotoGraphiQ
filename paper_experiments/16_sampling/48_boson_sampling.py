"""R48: Standard (Fock-state) boson sampling: permanents, distributions, samplers, scaling.

Scientific question: are PhotoGraphiQ's boson-sampling probabilities P(t|s) =
|Perm(U_{t,s})|^2 / (prod s! prod t!) (``photographiq.sampling``; native Gray-code Ryser kernel
and NumPy multiplicity-aware Ryser) correct for collision-free and collision (bunched) inputs and
outputs, normalised, consistent with a Fock-space simulation of the same interferometer, and are
the exact samplers distributed correctly? How do the permanent kernels scale?

References written or chosen to be independent of the code under test:

* permanents: the defining sum over permutations (brute force, n <= 8) and Glynn's formula
  (2^(n-1) Gray-free sum over sign vectors, n <= 16, numpy) implemented here;
* bunched permanents: brute force on the explicitly repeated matrix;
* distributions: an independent Fock-space evolution - the m-mode passive Gaussian unitary's Fock
  matrix from the native Bargmann/sector recurrence (``_native.fock.gaussian_unitary``; a different
  algorithm and code path from the permanent kernels) acting on |s> with cutoff n + 1; Piquasso
  8.0.1 PureFockSimulator with ``NumberState`` + ``Interferometer`` (same convention, checked);
* an exact physical law: the Fourier suppression law (Tichy et al. PRL 104, 220405): for the m x m
  DFT and the input with one photon in every mode, P(t) = 0 unless sum_k k t_k = 0 mod m;
* samplers: Pearson chi-square of 20000 seeded samples against the exact distribution.

Haar unitaries come from scipy.stats.unitary_group with fixed seeds (independent of
``photographiq.sampling.haar_unitary``). Timings are PROVISIONAL (shared machine).
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import piquasso as pq  # noqa: E402

from photographiq import native  # noqa: E402
from photographiq import sampling as smp  # noqa: E402

SEED = 20261148
ENGINE = native.require("fock")
DEFAULT_THREADS = native.require("sampling").max_threads()
# ----- acceptance criteria (declared before evaluation) ---------------------------------
TOL_PERM_SCALED = 1e-13  # |perm - ref| / prod_i sum_j |a_ij| (the magnitude scale of the Ryser terms)
TOL_NORMALISATION = 1e-12  # |sum_t P(t|s) - 1|
TOL_FOCK_REFERENCE = 1e-12  # max |P_perm - P_fock|
TOL_PIQUASSO = 1e-12  # max |P_perm - P_piquasso|
TOL_SUPPRESSION = 1e-14  # total probability of suppressed DFT outputs
MIN_CHI2_PVALUE = 1e-4  # seeded sampler goodness of fit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="larger permanents and distributions")
    return parser.parse_args()


def haar(m, seed):
    return stats.unitary_group.rvs(m, random_state=seed)


def perm_definition(a):
    n = len(a)
    return complex(sum(math.prod(a[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n))))


def perm_glynn(a):
    n = len(a)
    if n == 0:
        return 1.0 + 0j
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n - 1)))
    delta = np.hstack([np.ones((len(signs), 1)), signs])  # first sign fixed to +1
    total = 0j
    for chunk in range(0, len(delta), 1 << 14):
        d = delta[chunk:chunk + (1 << 14)]
        total += np.sum(np.prod(d, axis=1) * np.prod(d @ a, axis=1))
    return total / 2 ** (n - 1)


def permanent_rows(sizes, rng):
    rows = []
    for n in sizes:
        for trial in range(3):
            u = haar(max(2 * n, 2), int(rng.integers(2**31)))
            a = u[:n, :n] * np.sqrt(2 * n)  # O(1) entries
            scale = float(np.prod(np.abs(a).sum(axis=1)))
            row = {"kind": "square", "n": n, "trial": trial, "scale": scale}
            kern = native.require("sampling")
            for label, threads in (("native_seconds", DEFAULT_THREADS), ("native_1thread_seconds", 1)):
                kern.set_num_threads(threads)
                timing = common.benchmark(lambda a=a: smp.permanent(a, backend="native"), warmup=1, repeats=3)
                row[label] = timing["median_seconds"]
            kern.set_num_threads(DEFAULT_THREADS)
            nat = timing["result"]
            if n <= 18:
                timing = common.benchmark(lambda a=a: smp.permanent(a, backend="python"), warmup=0 if n > 14 else 1,
                                          repeats=1 if n > 14 else 3)
                py, row["python_seconds"] = timing["result"], timing["median_seconds"]
            else:
                py, row["python_seconds"] = np.nan, np.nan
            ref = perm_definition(a) if n <= 8 else perm_glynn(a)
            row["reference"] = "definition" if n <= 8 else "Glynn"
            row["abs_perm"] = abs(ref)
            row["native_scaled_error"] = abs(nat - ref) / scale
            row["python_scaled_error"] = abs(py - ref) / scale if n <= 18 else np.nan
            row["native_rel_error"] = abs(nat - ref) / abs(ref)
            rows.append(row)
    # bunched (repeated rows/columns) against brute force on the explicitly repeated matrix
    for reps_r, reps_c in (((2, 1, 1, 0), (1, 1, 1, 1)), ((3, 0, 1, 0), (2, 2, 0, 0)), ((2, 2, 1, 1), (3, 1, 1, 1))):
        u = haar(4, int(rng.integers(2**31))) * 2
        big = u[np.ix_(np.repeat(np.arange(4), reps_r), np.repeat(np.arange(4), reps_c))]
        ref = perm_definition(big)
        scale = float(np.prod(np.abs(big).sum(axis=1)))
        nat = smp.permanent(u, rows=list(reps_r), columns=list(reps_c), backend="native")
        py = smp.permanent(u, rows=list(reps_r), columns=list(reps_c), backend="python")
        rows.append({"kind": f"repeated rows {reps_r} cols {reps_c}", "n": sum(reps_r), "trial": 0, "scale": scale,
                     "reference": "definition (repeated matrix)", "abs_perm": abs(ref),
                     "native_scaled_error": abs(nat - ref) / scale, "python_scaled_error": abs(py - ref) / scale,
                     "native_rel_error": abs(nat - ref) / abs(ref), "native_seconds": np.nan,
                     "native_1thread_seconds": np.nan, "python_seconds": np.nan})
    return rows


def distribution_rows(cases, rng):
    rows = []
    for m, inputs in cases:
        u = haar(m, int(rng.integers(2**31)))
        n = int(sum(inputs))
        cutoff = n + 1
        t0 = time.perf_counter()
        dist = smp.output_distribution(u, inputs)
        t_perm = time.perf_counter() - t0
        basis = [tuple(int(x) for x in b) for b in ENGINE.basis(m, cutoff)]
        op = ENGINE.gaussian_unitary(smp.passive_symplectic(u), np.zeros(2 * m), cutoff)
        psi = op[:, basis.index(tuple(inputs))]
        fock = {b: abs(psi[i]) ** 2 for i, b in enumerate(basis) if sum(b) == n}
        with pq.Program() as prog:
            pq.Q(*range(m)) | pq.NumberState(tuple(inputs))
            pq.Q(*range(m)) | pq.Interferometer(u)
        pq_probs = pq.PureFockSimulator(d=m, config=pq.Config(cutoff=cutoff)).execute(prog).state.fock_probabilities_map
        split = smp.collision_probabilities(dist)
        rows.append({
            "modes": m, "photons": n, "inputs": str(tuple(inputs)), "patterns": len(dist),
            "collision_free_input": smp.is_collision_free(inputs),
            "normalisation_error": abs(math.fsum(dist.values()) - 1),
            "max_diff_vs_fock_space": max(abs(dist[t] - fock[t]) for t in dist),
            "max_diff_vs_piquasso": max(abs(dist[t] - pq_probs.get(t, 0.0)) for t in dist),
            "fock_leakage_outside_sector": float(1 - sum(fock.values())),
            "collision_free_mass": split["collision_free"], "collision_mass": split["collision"],
            "max_probability": max(dist.values()), "permanent_route_seconds": t_perm,
        })
        print(f"  m={m} n={n} {inputs}: norm err {rows[-1]['normalisation_error']:.1e}, vs Fock "
              f"{rows[-1]['max_diff_vs_fock_space']:.1e}, vs Piquasso {rows[-1]['max_diff_vs_piquasso']:.1e}", flush=True)
    return rows


def suppression_rows(sizes):
    rows = []
    for m in sizes:
        f = np.exp(2j * np.pi * np.outer(np.arange(m), np.arange(m)) / m) / np.sqrt(m)
        dist = smp.output_distribution(f, [1] * m)
        bad = [t for t in dist if sum(k * x for k, x in enumerate(t)) % m != 0]
        rows.append({"modes": m, "photons": m, "patterns": len(dist), "suppressed_patterns": len(bad),
                     "suppressed_mass": math.fsum(dist[t] for t in bad),
                     "allowed_mass": math.fsum(dist[t] for t in dist if t not in set(bad))})
    return rows


def sampler_rows(rng, shots):
    rows = []
    for m, inputs, method in ((6, (1, 1, 1, 0, 0, 0), "clifford"), (5, (2, 1, 0, 0, 0), "enumerate")):
        u = haar(m, int(rng.integers(2**31)))
        dist = smp.output_distribution(u, inputs)
        samples = smp.sample(u, inputs, shots, rng=int(rng.integers(2**31)), method=method)
        counts = {}
        for s in map(tuple, samples):
            counts[s] = counts.get(s, 0) + 1
        keys = list(dist)
        expected = np.array([dist[k] * shots for k in keys])
        observed = np.array([counts.get(k, 0) for k in keys])
        mask = expected >= 5  # Pearson validity; pool the rest
        obs = np.r_[observed[mask], observed[~mask].sum()]
        exp = np.r_[expected[mask], expected[~mask].sum()]
        if exp[-1] < 5:
            obs, exp = obs[:-1], exp[:-1]
            obs[-1] += observed[~mask].sum()
            exp[-1] += expected[~mask].sum()
        chi2 = stats.chisquare(obs, exp * obs.sum() / exp.sum())
        tvd = 0.5 * np.abs(observed / shots - expected / shots).sum()
        rows.append({"modes": m, "inputs": str(inputs), "method": method, "shots": shots,
                     "chi2": float(chi2.statistic), "dof": len(obs) - 1, "p_value": float(chi2.pvalue),
                     "total_variation": float(tvd), "outside_support": int(sum(c for s, c in counts.items() if s not in dist))})
    return rows


def main():
    args = parse_args()
    rng = common.rng(SEED)
    sizes = list(range(1, 15)) + ([16, 18, 20] if args.full else [])
    perms = permanent_rows(sizes, rng)
    cases = [(4, (1, 1, 0, 0)), (4, (2, 0, 0, 0)), (6, (1, 1, 1, 0, 0, 0)), (6, (2, 1, 0, 0, 0, 0)),
             (8, (1, 1, 1, 1, 0, 0, 0, 0)), (8, (2, 2, 0, 0, 0, 0, 0, 0)), (10, (1, 1, 1, 1, 0, 0, 0, 0, 0, 0))]
    if args.full:
        cases += [(12, (1,) * 4 + (0,) * 8), (10, (1,) * 5 + (0,) * 5), (8, (3, 2, 0, 0, 0, 0, 0, 0))]
    dists = distribution_rows(cases, rng)
    supp = suppression_rows([3, 4, 5, 6] + ([7] if args.full else []))
    samp = sampler_rows(rng, 20000)

    failures = [{"check": "permanent", **r} for r in perms
                if r["native_scaled_error"] > TOL_PERM_SCALED
                or (not np.isnan(r["python_scaled_error"]) and r["python_scaled_error"] > TOL_PERM_SCALED)]
    failures += [{"check": "distribution", **r} for r in dists
                 if r["normalisation_error"] > TOL_NORMALISATION or r["max_diff_vs_fock_space"] > TOL_FOCK_REFERENCE
                 or r["max_diff_vs_piquasso"] > TOL_PIQUASSO]
    failures += [{"check": "suppression law", **r} for r in supp if r["suppressed_mass"] > TOL_SUPPRESSION]
    failures += [{"check": "sampler", **r} for r in samp if r["p_value"] < MIN_CHI2_PVALUE or r["outside_support"]]

    extra = {"backend": "photographiq.sampling (native Gray-code Ryser 'native', NumPy multiplicity Ryser 'python')",
             "reference": "permutation-sum definition, Glynn formula (this script), native Fock-space evolution of the "
                          "passive unitary, Piquasso 8.0.1 PureFockSimulator, DFT suppression law",
             "precision": "complex128", "cutoff": "photons + 1 for the Fock-space references", "seed": SEED,
             "full": args.full, "timing_status": "PROVISIONAL (shared machine)",
             "parameters": {"permanent_sizes": sizes, "distribution_cases": [list(map(int, c[1])) for c in cases],
                            "sampler_shots": 20000, "haar_source": "scipy.stats.unitary_group"},
             "native_threads": DEFAULT_THREADS,
             "acceptance": {"TOL_PERM_SCALED": TOL_PERM_SCALED, "TOL_NORMALISATION": TOL_NORMALISATION,
                            "TOL_FOCK_REFERENCE": TOL_FOCK_REFERENCE, "TOL_PIQUASSO": TOL_PIQUASSO,
                            "TOL_SUPPRESSION": TOL_SUPPRESSION, "MIN_CHI2_PVALUE": MIN_CHI2_PVALUE}}
    common.save_result(perms, "R48_boson_sampling_permanents", extra=extra)
    common.save_result(dists, "R48_boson_sampling", extra=extra)
    common.save_result(supp, "R48_boson_sampling_suppression", extra=extra)
    common.save_result(samp, "R48_boson_sampling_samplers", extra=extra)
    if failures:
        common.save_csv(failures, "R48_boson_sampling_FAILURES")

    plt = common.setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    sq = [r for r in perms if r["kind"] == "square"]
    ns = sorted({r["n"] for r in sq})
    for key, color, label in (("native", "#24677b", "native (Gray-code Ryser)"),
                              ("python", "#c66d27", "NumPy (multiplicity Ryser)")):
        pts = [r for r in sq if not np.isnan(r[f"{key}_scaled_error"])]
        axes[0].semilogy([r["n"] for r in pts], [max(r[f"{key}_scaled_error"] * r["scale"] / r["abs_perm"], 1e-18)
                                                 for r in pts], "o", color=color, label=label, alpha=0.8, ms=4)
    axes[0].axvline(8.5, color="grey", ls=":", lw=1)
    axes[0].text(4.3, 1e-9, "vs definition", fontsize=7, ha="center")
    axes[0].text(14, 1e-9, "vs Glynn (float64)", fontsize=7, ha="center")
    axes[0].set(xlabel="matrix size n", ylabel="relative error |Perm - ref| / |ref|",
                title="(a) permanent accuracy", ylim=(1e-18, 1e-8))
    axes[0].set_xticks(range(0, max(ns) + 1, 2))
    axes[0].legend(fontsize=7, loc="lower right")
    med = lambda n, k: np.median([r[k] for r in sq if r["n"] == n])  # noqa: E731
    axes[1].semilogy(ns, [med(n, "native_seconds") for n in ns], "o-", color="#24677b",
                     label=f"native, {DEFAULT_THREADS} OpenMP threads")
    axes[1].semilogy(ns, [med(n, "native_1thread_seconds") for n in ns], "^--", color="#24677b",
                     label="native, 1 thread")
    py_ns = [n for n in ns if not np.isnan(med(n, "python_seconds"))]
    axes[1].semilogy(py_ns, [med(n, "python_seconds") for n in py_ns], "s-", color="#c66d27", label="NumPy")
    axes[1].set(xlabel="matrix size n", ylabel="median time per call (s)", title="(b) permanent runtime (provisional)")
    axes[1].legend(fontsize=7)
    x = np.arange(len(dists))
    axes[2].semilogy(x, [max(r["normalisation_error"], 1e-18) for r in dists], "o", label="|sum P - 1|")
    axes[2].semilogy(x, [max(r["max_diff_vs_fock_space"], 1e-18) for r in dists], "s",
                     label="max |P - P_Fock-space evolution|")
    axes[2].semilogy(x, [max(r["max_diff_vs_piquasso"], 1e-18) for r in dists], "^", label="max |P - P_Piquasso|")
    axes[2].axhline(TOL_FOCK_REFERENCE, color="k", ls="--", lw=1)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"m={r['modes']}\n{r['inputs'].replace(' ', '')}" for r in dists], fontsize=5.5,
                            rotation=45, ha="right")
    axes[2].set(ylabel="probability error", title="(c) output distributions", ylim=(1e-18, 1e-10))
    axes[2].legend(fontsize=6.5, loc="upper left")
    common.save_figure(fig, "R48_boson_sampling")
    plt.close(fig)

    biggest = max(ns)
    worst_perm = max(max(r["native_scaled_error"], np.nan_to_num(r["python_scaled_error"])) for r in perms)
    common.save_caption(
        "R48_boson_sampling",
        f"""
**R48 - Fock-state boson sampling.** (a) Permanent accuracy of the native Gray-code Ryser kernel and
the NumPy multiplicity-aware Ryser (3 seeded submatrices of scipy Haar unitaries per size, scaled to
O(1) entries) against the permutation-sum definition (n <= 8) and Glynn's formula (n = 9..{biggest}),
shown as relative error (worst {max(r['native_rel_error'] for r in perms):.1e}; the Glynn reference is itself float64);
acceptance is on the error normalised by the Ryser term scale prod_i sum_j |a_ij| (worst {worst_perm:.1e});
bunched permanents (repeated rows/columns) are checked against the definition on the repeated matrix. (b) Single-call runtime
(median of 3 after warm-up; provisional, shared machine): native at n = {biggest}: {med(biggest, 'native_seconds'):.2e} s with
{DEFAULT_THREADS} OpenMP threads, {med(biggest, 'native_1thread_seconds'):.2e} s with 1 thread; the multi-threaded kernel was slower
than 1 thread for n in {[n for n in ns if med(n, 'native_seconds') > med(n, 'native_1thread_seconds')]} on this loaded machine
(OpenMP fork/join and spin-wait overhead; see R48 notes). (c) Full output
distributions for {len(dists)} (modes, input) cases incl. collision inputs: normalisation (worst
{max(r['normalisation_error'] for r in dists):.1e}), agreement with an independent Fock-space evolution of the same
interferometer (native sector recurrence, cutoff n + 1; worst {max(r['max_diff_vs_fock_space'] for r in dists):.1e})
and with Piquasso 8.0.1 (worst {max(r['max_diff_vs_piquasso'] for r in dists):.1e}). Not shown: the DFT suppression law
holds (suppressed mass <= {max(r['suppressed_mass'] for r in supp):.1e} for m = {supp[0]['modes']}..{supp[-1]['modes']});
seeded samplers pass Pearson chi-square (p = {', '.join(f"{r['p_value']:.2f}" for r in samp)}, 20000 shots).
Seed {SEED}. Source: R48 (16_sampling/48_boson_sampling.py).
""",
    )
    common.print_summary("R48 boson sampling", permanents=len(perms), worst_perm_scaled=f"{worst_perm:.2e}",
                         worst_norm=f"{max(r['normalisation_error'] for r in dists):.2e}",
                         worst_vs_fock=f"{max(r['max_diff_vs_fock_space'] for r in dists):.2e}",
                         worst_vs_piquasso=f"{max(r['max_diff_vs_piquasso'] for r in dists):.2e}",
                         suppression=f"{max(r['suppressed_mass'] for r in supp):.2e}",
                         sampler_p=[round(r["p_value"], 3) for r in samp], failures=len(failures))
    if failures:
        raise AssertionError(f"{len(failures)} R48 checks failed; see R48_boson_sampling_FAILURES.csv")


if __name__ == "__main__":
    main()
