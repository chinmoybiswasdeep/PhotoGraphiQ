"""R45: Native pure-state Fock backend accuracy, truncation and cutoff convergence.

Scientific question: how accurate is ``native-fock`` (compiled C++ engine, total-photon
truncation n_1 + ... + n_m < cutoff, gates applied as P U P with renormalisation and a
recorded retained norm) for Fock inputs, displacement, squeezing, beam splitters, Kerr,
cubic phase and conditional photon counting, and how do state, observable and norm errors
converge with the cutoff? Where does Piquasso 8.0.1 (piquasso-fock) agree or not?

Two kinds of reference are used and kept apart:

* exact-in-truncation references (what the engine SHOULD produce at this cutoff):
  P U |psi> / ||P U |psi>|| from closed forms (coherent and squeezed amplitudes with 30-digit
  mpmath, ``tests/validation/test_val_fock.py``), exact phases (Kerr), expm of the generator on
  an enlarged per-mode product space (beam splitter; ``unitary`` of the same test module), and the
  exact projected cubic-phase elements <m|exp(i gamma q^3/6)|n> by Hermite-function quadrature
  (``cubic_projected``). Their retained norm ||P U psi||^2 is also exact. Deviations here are
  rounding of the engine.
* untruncated physics (what the infinite-dimensional state gives): <n> = |alpha|^2, sinh^2 r;
  cubic phase on vacuum p -> p + gamma q^2 (hbar = 2, [q^3, p] = 6i q^2) gives <p> = gamma,
  <p^2> = 1 + 3 gamma^2;
  Kerr <a> from a 400-level sum; two-mode squeezed vacuum (two squeezers + 50:50 splitter)
  photon counting P(k) = (1 - l^2) l^(2k), l = tanh r, conditional state |k>. Deviations here
  are TRUNCATION, reported as convergence with the cutoff, never hidden.

Cubic phase: ``cubic_phase="projected"`` (exact P V P, leaks norm) and the default
``"truncated_generator"`` (expm of the truncated q^3, unitary on the truncated space, never
reports leakage) are both studied; only the former has an exact-in-truncation reference.

Assumptions/limitations: double precision; one- and two-mode systems (multimode scaling is
R46); the photon-counting check uses the exact factorisation P_tot (BS (S x S)) |00>, whose
conditional state is exactly |k> for 2k < cutoff.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "validation"))
import test_val_fock as tvf  # noqa: E402  (first-principles Fock references)

import photographiq as pg  # noqa: E402
from photographiq import registry  # noqa: E402

SEED = 20261145
ALPHA, ALPHA_BIG = 1.2 + 0.5j, 4.0
R_SQ, KAPPA, GAMMAS, R_TMSV = 0.8, 0.35, (0.1, 0.3), 0.6
BS_INPUTS = ((1, 1), (2, 1), (3, 3), (4, 2))

# ----- acceptance criteria (declared before evaluation) ---------------------------------
TOL_EXACT = 1e-12  # engine vs exact-in-truncation reference (state and retained norm); float64
#                    rounding of the Laguerre/Jacobi recurrences is <= 1e-14 at these sizes
TOL_HOM = 1e-15  # Hong-Ou-Mandel coincidence P(1,1) after a 50:50 splitter (exact zero)
TOL_CUBIC_PROJECTED = 1e-10  # projected cubic vs quadrature reference (quadrature floor ~2e-12)
K_TRUNC = 10  # untruncated observable error must be explained by the recorded leaked norm:
#              |err| <= K * cutoff * (1 - retained) + 1e-12 (an observable <= cutoff in size, times the
#              omitted mass; Gaussian-tailed cases at the top cutoff of the sweep)
TOL_PHYSICAL = 1e-12  # |norm - 1| of the returned (renormalised) states


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="cutoffs up to 96")
    return parser.parse_args()


def engine_for(name, cutoff, **kw):
    cls = registry.BACKENDS[name].load()
    eng = cls(cutoff, norm_tolerance=1 - 1e-12, max_dimension=10**7, **kw)
    eng.reset(SEED)
    return eng


def aligned_error(vec, ref):
    """max |vec - e^{i phi} ref| with the global phase fixed at the largest component of ref."""
    k = int(np.argmax(np.abs(ref)))
    phase = vec[k] / ref[k] / abs(vec[k] / ref[k]) if abs(vec[k]) > 0 else 1.0
    return float(np.abs(vec - phase * ref).max())


def as_vector(state):
    return np.asarray(state.state_vector), [tuple(int(x) for x in b) for b in state.basis]


def normalised(v):
    norm = float(np.vdot(v, v).real)
    return v / np.sqrt(norm), norm


def displaced_fock_mp(alpha, n, cutoff):
    """<m|D(alpha)|n>, m < cutoff, from the associated-Laguerre closed form (50 digits;
    L_n^(k)(x) = sum_j (-1)^j C(n+k, n-j) x^j / j! summed exactly, cancellation <= e^x)."""
    mp.mp.dps = 50

    def laguerre(order, k, x):
        return mp.fsum((-1) ** j * mp.binomial(order + k, order - j) * x**j / mp.factorial(j)
                       for j in range(order + 1))

    a, x = mp.mpf(alpha), mp.mpf(alpha) ** 2
    out = []
    for m in range(cutoff):
        if m >= n:
            v = mp.sqrt(mp.factorial(n) / mp.factorial(m)) * a ** (m - n) * laguerre(n, m - n, x)
        else:
            v = mp.sqrt(mp.factorial(m) / mp.factorial(n)) * (-a) ** (n - m) * laguerre(m, n - m, x)
        out.append(complex(v * mp.exp(-x / 2)))
    return np.array(out)


def single_mode_cases(cutoff, name):
    """Rows for one cutoff and one engine: exact-in-truncation and untruncated comparisons."""
    rows = []
    mp.mp.dps = 30
    # --- displacement of vacuum (coherent) -----------------------------------------------
    for alpha in (ALPHA, ALPHA_BIG):
        eng = engine_for(name, cutoff)
        eng.prepare(0, state=pg.FockInput.number(0))
        eng.displace(0, 2 * alpha.real if isinstance(alpha, complex) else 2 * alpha,
                     2 * alpha.imag if isinstance(alpha, complex) else 0.0)
        st = eng.get_state()
        vec, _ = as_vector(st)
        exact = np.array([complex(x) for x in tvf._coherent_mp(complex(alpha), cutoff)])
        ref, retained = normalised(exact)
        rows.append({"case": f"coherent D(alpha)|0>, |alpha|={abs(alpha):.2f}", "engine": name, "cutoff": cutoff,
                     "exact_error": aligned_error(vec, ref),
                     "retained_norm": eng.retained_norms[-1], "retained_norm_error": abs(eng.retained_norms[-1] - retained),
                     "observable": "<n>", "observable_value": st.photon_number(0),
                     "observable_exact": abs(alpha) ** 2,
                     "observable_error": abs(st.photon_number(0) - abs(alpha) ** 2), "norm_error": abs(st.norm - 1)})
    # --- displaced Fock state D(alpha)|n0>, n0 = cutoff // 2 (interior columns of D) ----------
    n0 = cutoff // 2
    eng = engine_for(name, cutoff)
    eng.prepare(0, state=pg.FockInput.number(n0))
    eng.displace(0, 2 * ALPHA_BIG, 0.0)
    vec, _ = as_vector(eng.get_state())
    ref, retained = normalised(displaced_fock_mp(ALPHA_BIG, n0, cutoff))
    rows.append({"case": f"displaced Fock D(alpha)|cutoff/2>, |alpha|={ALPHA_BIG:.2f}", "engine": name,
                 "cutoff": cutoff, "exact_error": aligned_error(vec, ref), "retained_norm": eng.retained_norms[-1],
                 "retained_norm_error": abs(eng.retained_norms[-1] - retained), "observable": "-",
                 "observable_value": np.nan, "observable_exact": np.nan, "observable_error": np.nan,
                 "norm_error": abs(eng.get_state().norm - 1)})
    # --- squeezing of vacuum ---------------------------------------------------------------
    eng = engine_for(name, cutoff)
    eng.prepare(0, state=pg.FockInput.number(0))
    eng.squeeze(0, R_SQ)
    st = eng.get_state()
    vec, _ = as_vector(st)
    ref, retained = normalised(np.array([complex(x) for x in tvf._squeezed_mp(R_SQ, 0.0, cutoff)]))
    n_exact = np.sinh(R_SQ) ** 2
    rows.append({"case": f"squeezed S(r)|0>, r={R_SQ}", "engine": name, "cutoff": cutoff,
                 "exact_error": aligned_error(vec, ref),
                 "retained_norm": eng.retained_norms[-1], "retained_norm_error": abs(eng.retained_norms[-1] - retained),
                 "observable": "<n>", "observable_value": st.photon_number(0), "observable_exact": n_exact,
                 "observable_error": abs(st.photon_number(0) - n_exact), "norm_error": abs(st.norm - 1)})
    # --- Kerr on a coherent input --------------------------------------------------------------
    eng = engine_for(name, cutoff)
    eng.prepare(0, state=pg.GaussianInput.coherent(ALPHA))
    eng.kerr(0, KAPPA)
    st = eng.get_state()
    vec, _ = as_vector(st)
    n = np.arange(cutoff)
    exact = np.array([complex(x) for x in tvf._coherent_mp(ALPHA, cutoff)]) * np.exp(1j * KAPPA * n**2)
    ref, retained = normalised(exact)
    big = np.arange(400)
    from scipy.special import gammaln

    amp = np.exp(-abs(ALPHA) ** 2 / 2 + big * np.log(abs(ALPHA)) - 0.5 * gammaln(big + 1)) \
        * np.exp(1j * (big * np.angle(ALPHA) + KAPPA * big**2))
    a_exact = np.sum(np.conj(amp[:-1]) * amp[1:] * np.sqrt(big[1:]))
    a_val = np.sum(np.conj(vec[:-1]) * vec[1:] * np.sqrt(n[1:]))
    rows.append({"case": f"Kerr on coherent, kappa={KAPPA}", "engine": name, "cutoff": cutoff,
                 "exact_error": aligned_error(vec, ref), "retained_norm": eng.retained_norms[0],
                 "retained_norm_error": abs(eng.retained_norms[0] - retained),
                 "observable": "<a>", "observable_value": abs(a_val), "observable_exact": abs(a_exact),
                 "observable_error": abs(a_val - a_exact), "norm_error": abs(st.norm - 1)})
    # --- cubic phase on vacuum, both definitions ---------------------------------------------
    if name == "native-fock" or cutoff <= 48:
        for gamma in GAMMAS:
            methods = ("projected", "truncated_generator") if name == "native-fock" else ("truncated_generator",)
            for method in methods:
                kw = {"cubic_phase": method} if name == "native-fock" else {}
                eng = engine_for(name, cutoff, **kw)
                eng.prepare(0, state=pg.FockInput.number(0))
                eng.cubic_phase(0, gamma)
                st = eng.get_state()
                vec, _ = as_vector(st)
                if method == "projected":
                    ref, retained = normalised(tvf.cubic_projected(gamma, cutoff, 8001)[:, 0])
                    exact_err = aligned_error(vec, ref)
                    norm_err = abs(eng.retained_norms[-1] - retained)
                else:
                    exact_err, norm_err = np.nan, np.nan
                p_mean, p_var = st.quadrature(0, np.pi / 2)
                p2 = p_var + p_mean**2
                p2_exact = 1 + 3 * gamma**2
                rows.append({"case": f"cubic phase ({method}) on |0>, gamma={gamma}", "engine": name,
                             "cutoff": cutoff, "exact_error": exact_err, "retained_norm": eng.retained_norms[-1],
                             "retained_norm_error": norm_err, "observable": "<p^2> (and <p>)",
                             "observable_value": p2, "observable_exact": p2_exact,
                             "observable_error": max(abs(p2 - p2_exact), abs(p_mean - gamma)),
                             "norm_error": abs(st.norm - 1)})
    return rows


def two_mode_cases(cutoff, name):
    rows = []
    # --- beam splitter on Fock inputs vs enlarged expm (number conserving: exact per sector) ---
    for n1, n2 in BS_INPUTS:
        if n1 + n2 >= cutoff:
            continue
        eng = engine_for(name, cutoff)
        eng.prepare(0, state=pg.FockInput.number(n1))
        eng.prepare(1, state=pg.FockInput.number(n2))
        eng.beamsplitter(0, 1, np.pi / 4)
        st = eng.get_state()
        vec, basis = as_vector(st)
        dim = n1 + n2 + 1
        full = tvf.unitary("beamsplitter", float(np.pi / 4), dim) @ tvf.dense_vector({(n1, n2): 1.0}, 2, dim)
        ref = np.array([full[tvf.index(b, dim)] if max(b) < dim else 0.0 for b in basis])
        row = {"case": f"50:50 BS on |{n1},{n2}>", "engine": name, "cutoff": cutoff,
               "exact_error": aligned_error(vec, ref), "retained_norm": eng.retained_norms[-1],
               "retained_norm_error": abs(eng.retained_norms[-1] - 1.0), "observable": "P(1,1)" if (n1, n2) == (1, 1) else "-",
               "observable_value": np.nan, "observable_exact": np.nan, "observable_error": np.nan,
               "norm_error": abs(st.norm - 1)}
        if (n1, n2) == (1, 1):
            p11 = st.probabilities[(1, 1)]
            row.update({"observable_value": p11, "observable_exact": 0.0, "observable_error": p11})
        rows.append(row)
    # --- TMSV + photon counting -------------------------------------------------------------
    lam = np.tanh(R_TMSV)
    kept = [k for k in range(cutoff) if 2 * k < cutoff]
    trunc_norm = sum((1 - lam**2) * lam ** (2 * k) for k in kept)
    for k in (0, 1, 2, 3):
        if 2 * k >= cutoff:
            continue
        eng = engine_for(name, cutoff)
        eng.prepare(0, state=pg.GaussianInput.squeezed(R_TMSV))
        eng.prepare(1, state=pg.GaussianInput.squeezed(-R_TMSV))
        eng.beamsplitter(0, 1, np.pi / 4)
        eng.measure(0, pg.PhotonNumber(), outcome=k)
        prob = eng.last_measurement["value"]
        st = eng.get_state()
        vec, basis = as_vector(st)
        target = np.array([1.0 if b == (k,) else 0.0 for b in basis])
        fid = abs(np.vdot(target, vec)) ** 2
        p_exact_trunc = (1 - lam**2) * lam ** (2 * k) / trunc_norm
        p_exact = (1 - lam**2) * lam ** (2 * k)
        rows.append({"case": f"TMSV r={R_TMSV}: count k={k} on mode 0", "engine": name, "cutoff": cutoff,
                     "exact_error": max(abs(prob - p_exact_trunc), 1 - fid),
                     "retained_norm": float(np.prod(eng.retained_norms)), "retained_norm_error": np.nan,
                     "observable": "P(k)", "observable_value": prob, "observable_exact": p_exact,
                     "observable_error": abs(prob - p_exact), "norm_error": abs(st.norm - 1),
                     "conditional_infidelity": 1 - fid})
    return rows


def main():
    args = parse_args()
    plt = common.setup_style()
    cutoffs = [6, 8, 12, 16, 24, 32, 48] + ([64, 96] if args.full else [])
    pq_cutoffs = [c for c in cutoffs if c <= 48]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for cutoff in cutoffs:
            for name in ("native-fock", "piquasso-fock"):
                if name == "piquasso-fock" and cutoff not in pq_cutoffs:
                    continue
                rows += single_mode_cases(cutoff, name)
                rows += two_mode_cases(cutoff, name)
            print(f"  cutoff {cutoff}: done", flush=True)

    # native vs piquasso on identical sequences
    for r in rows:
        r["is_native"] = r["engine"] == "native-fock"
    native = [r for r in rows if r["is_native"]]
    top = cutoffs[-1]
    failures = []
    for r in native:
        exactish = not np.isnan(r["exact_error"])
        tol = TOL_CUBIC_PROJECTED if r["case"].startswith("cubic") else TOL_EXACT
        if exactish and r["exact_error"] > tol:
            failures.append({"check": "exact-in-truncation", "tolerance": tol, **r})
        if not np.isnan(r["retained_norm_error"]) and r["retained_norm_error"] > tol:
            failures.append({"check": "retained norm", "tolerance": tol, **r})
        if r["norm_error"] > TOL_PHYSICAL:
            failures.append({"check": "normalisation", **r})
        if r["case"].startswith("50:50 BS on |1,1>") and r["observable_error"] > TOL_HOM:
            failures.append({"check": "HOM", **r})
        if r["cutoff"] == top and not r["case"].startswith(("50:50", "cubic")):
            limit = K_TRUNC * top * (1 - r["retained_norm"]) + 1e-12
            if r["observable_error"] > limit:
                failures.append({"check": "observable error not explained by leaked norm", "tolerance": limit, **r})
    # Cubic phase has a heavy (non-Gaussian) Fock tail: only monotone convergence over the last three
    # cutoffs is required (docs: <p^2> error 9e-3/3e-4/1e-6 at cutoffs 80/160/320 for gamma = 0.8).
    for case in {r["case"] for r in native if r["case"].startswith("cubic")}:
        errs = [r["observable_error"] for r in sorted((r for r in native if r["case"] == case),
                                                       key=lambda r: r["cutoff"])][-3:]
        if not all(b <= a + 1e-14 for a, b in zip(errs, errs[1:])):
            failures.append({"check": "cubic convergence not monotone", "case": case, "errors": str(errs)})

    comparison = []
    keyed = {(r["case"], r["cutoff"]): r for r in native}
    for r in rows:
        if r["engine"] != "piquasso-fock":
            continue
        n_row = keyed.get((r["case"].replace("(truncated_generator)", "(truncated_generator)"), r["cutoff"]))
        if n_row is None:
            continue
        comparison.append({"case": r["case"], "cutoff": r["cutoff"],
                           "native_exact_error": n_row["exact_error"], "piquasso_exact_error": r["exact_error"],
                           "native_observable_error": n_row["observable_error"],
                           "piquasso_observable_error": r["observable_error"]})

    extra = {"backend": "native-fock (compiled) with piquasso-fock 8.0.1 as secondary reference",
             "reference": "closed forms (mpmath 30 digits), enlarged-space expm, Hermite quadrature of the "
                          "projected cubic phase (tests/validation/test_val_fock.py helpers), untruncated analytic "
                          "observables",
             "precision": "complex128", "cutoff": cutoffs, "piquasso_cutoffs": pq_cutoffs, "seed": SEED,
             "full": args.full,
             "parameters": {"alpha": [ALPHA.real, ALPHA.imag], "alpha_big": ALPHA_BIG, "squeezing": R_SQ,
                            "kerr": KAPPA, "cubic_gammas": GAMMAS, "tmsv_r": R_TMSV, "bs_inputs": BS_INPUTS,
                            "norm_tolerance": "1 - 1e-12 (truncation studied, never rejected)"},
             "acceptance": {"TOL_EXACT": TOL_EXACT, "TOL_HOM": TOL_HOM, "TOL_CUBIC_PROJECTED": TOL_CUBIC_PROJECTED,
                            "K_TRUNC": K_TRUNC, "TOL_PHYSICAL": TOL_PHYSICAL}}
    for r in rows:
        r.pop("is_native", None)
    common.save_result(rows, "R45_native_fock_accuracy", extra=extra)
    common.save_result(comparison, "R45_native_fock_vs_piquasso", extra=extra)
    if failures:
        common.save_csv(failures, "R45_native_fock_accuracy_FAILURES")

    # ----- figure --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.6))
    cases = sorted({r["case"] for r in native})
    palette = plt.cm.tab20(np.linspace(0, 1, 20))
    markers = "osD^v<>p*hX"
    style = {case: dict(color=palette[i % 20], marker=markers[i % len(markers)], ms=3.5, lw=1.2)
             for i, case in enumerate(cases)}
    for case in cases:
        sub = sorted((r for r in native if r["case"] == case), key=lambda r: r["cutoff"])
        c = [r["cutoff"] for r in sub]
        if not np.isnan(sub[0]["exact_error"]):
            axes[0, 0].semilogy(c, [max(r["exact_error"], 1e-17) for r in sub], **style[case], label=case)
        if not np.isnan(sub[0]["observable_error"]) and not case.startswith("50:50"):
            axes[0, 1].semilogy(c, [max(r["observable_error"], 1e-17) for r in sub], **style[case], label=case)
        if "TMSV" not in case and not case.startswith("50:50"):
            axes[1, 0].semilogy(c, [max(1 - r["retained_norm"], 1e-17) for r in sub], **style[case], label=case)
    axes[0, 0].axhline(TOL_EXACT, color="k", ls="--", lw=1)
    axes[0, 0].set(xlabel="cutoff (total photons <)", ylabel="max |psi - psi_exact(cutoff)|",
                   title="(a) engine vs exact-in-truncation reference")
    axes[0, 1].set(xlabel="cutoff", ylabel="|observable - untruncated value|",
                   title="(b) truncation convergence of observables")
    axes[1, 0].set(xlabel="cutoff", ylabel="1 - retained norm ||P U psi||^2",
                   title="(c) leaked norm (recorded, never hidden)")
    for case_key, marker in (("displaced Fock D(alpha)|cutoff/2>, |alpha|=4.00", "o"), (f"squeezed S(r)|0>, r={R_SQ}", "s"),
                             ("cubic phase (truncated_generator) on |0>, gamma=0.3", "^")):
        sub = sorted((r for r in comparison if r["case"] == case_key), key=lambda r: r["cutoff"])
        if not sub:
            continue
        key = "observable_error" if case_key.startswith("cubic") else "exact_error"
        axes[1, 1].semilogy([r["cutoff"] for r in sub], [max(r[f"native_{key}"], 1e-17) for r in sub],
                            marker + "-", color="#24677b", label=f"native: {case_key}", ms=4)
        axes[1, 1].semilogy([r["cutoff"] for r in sub], [max(r[f"piquasso_{key}"], 1e-17) for r in sub],
                            marker + "--", color="#c66d27", label=f"Piquasso: {case_key}", ms=4)
    axes[1, 1].set(xlabel="cutoff", ylabel="error (state, or <p^2> for cubic)",
                   title="(d) native vs Piquasso 8.0.1, same sequence")
    handles = [plt.Line2D([], [], **style[c]) for c in cases]
    fig.legend(handles, cases, loc="lower center", ncol=3, fontsize=7, frameon=False,
               title="panels (a)-(c), native-fock (floor 1e-17 = exact zero)", title_fontsize=7.5)
    axes[1, 1].legend(fontsize=5.8, loc="upper right")
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    common.save_figure(fig, "R45_native_fock_accuracy", tight=False)
    plt.close(fig)

    def at(case, cutoff, key, engine="native-fock"):
        return next(r[key] for r in rows if r["case"] == case and r["cutoff"] == cutoff and r["engine"] == engine)

    worst_exact = max(r["exact_error"] for r in native if not np.isnan(r["exact_error"]))
    big_case = "displaced Fock D(alpha)|cutoff/2>, |alpha|=4.00"
    pq_big = max((r["piquasso_exact_error"] for r in comparison if r["case"] == big_case), default=np.nan)
    nat_big = max(r["native_exact_error"] for r in comparison if r["case"] == big_case)
    cubic_case = "cubic phase (projected) on |0>, gamma=0.3"
    common.save_caption(
        "R45_native_fock_accuracy",
        f"""
**R45 - native Fock (pure-state) accuracy and cutoff convergence.** Total-photon truncation
n_1 + n_2 < cutoff, cutoffs {cutoffs[0]}..{cutoffs[-1]}, gates applied as P U P and renormalised (retained norm
recorded). (a) Distance (global phase aligned) between the engine's state and the exact state of
the SAME truncated model, P U psi / ||P U psi||, from 30-digit closed forms (coherent |alpha| = {abs(ALPHA):.2f}
and 4, squeezed r = {R_SQ}), exact phases (Kerr kappa = {KAPPA}), enlarged-space expm (50:50 beam splitter on
|1,1>, |2,1>, |3,3>, |4,2>), Hermite quadrature of the projected cubic phase (gamma = {GAMMAS}), and the exact
two-mode-squeezed-vacuum photon-count probability (r = {R_TMSV}): worst {worst_exact:.1e} over all cutoffs.
(b) Convergence of untruncated observables (<n> = |alpha|^2, sinh^2 r; <a> under Kerr; cubic
<p> = gamma, <p^2> = 1 + 3 gamma^2; P(k) = (1 - l^2) l^(2k)); at cutoff {top} the projected cubic
(gamma = 0.3) <p^2> error is {at(cubic_case, top, 'observable_error'):.1e}. (c) Leaked norm 1 - ||P U psi||^2
(number-conserving gates leak nothing; the truncated-generator cubic phase is unitary on the truncated
space and reports no leakage by construction). (d) Same sequences on Piquasso 8.0.1: for |alpha| = 4 the
worst error of D(4)|cutoff/2> (interior columns of the displacement matrix) is {pq_big:.1e} (Piquasso) vs
{nat_big:.1e} (native) over cutoffs <= {pq_cutoffs[-1]}; on D(4)|0> both reach rounding. HOM coincidence P(1,1) =
{at('50:50 BS on |1,1>', cutoffs[0], 'observable_value'):.1e}. Source: R45 (15_native_fock/45_native_fock_accuracy.py).
Limitations: one- and two-mode systems; float64.
""",
    )
    common.print_summary("R45 native Fock accuracy", rows=len(rows), worst_exact_in_truncation=f"{worst_exact:.2e}",
                         alpha4_native=f"{nat_big:.2e}", alpha4_piquasso=f"{pq_big:.2e}",
                         cubic_projected_p2_err_top=f"{at(cubic_case, top, 'observable_error'):.2e}",
                         failures=len(failures))
    if failures:
        raise AssertionError(f"{len(failures)} R45 checks failed; see R45_native_fock_accuracy_FAILURES.csv")


if __name__ == "__main__":
    main()
