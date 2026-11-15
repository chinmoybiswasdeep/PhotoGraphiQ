"""R47: Native mixed-state (density-matrix) Fock backend validation.

Scientific question: does ``native-mixed-fock`` (compiled density-matrix engine, total-photon
truncation, P U P gates, Kraus thermal-loss channel, renormalisation with recorded retained
norm) produce physical density matrices that reproduce the exact mixed-state physics of
thermal/mixed Gaussian inputs, loss, partial traces and conditional measurements, and how do
the results converge with the cutoff?

References (independent of the engine):

* exact-in-truncation closed forms: thermal input rho_kk = nbar^k / (1 + nbar)^(k+1) restricted to
  k < c and renormalised; the reduced state of a two-mode squeezed vacuum built from two
  squeezers and a 50:50 splitter is diagonal with rho_kk ~ l^(2k), l = tanh r, for 2k < c (the
  truncated product state is P_tot of the exact state, which the passive splitter preserves);
  a displaced squeezed thermal input from dense expm on a 90-level space
  (``tests/validation/test_val_fock.py`` construction);
* channel invariants: pure loss on |alpha> gives the pure coherent state |sqrt(eta) alpha>
  (fidelity, purity), thermal loss gives <n> = eta |alpha|^2 + (1 - eta) n_th;
* Gaussian moments: the same Gaussian sequence (squeezers, splitter, loss, homodyne conditioning
  on a mixed two-mode state) on the NumPy ``gaussian`` backend; the Fock conditional state's
  quadrature mean and covariance (from exact ladder moments) and the outcome density must
  converge to the Gaussian ones;
* consistency: on a non-Gaussian sequence without loss, a pure input evolved by the mixed engine
  equals |psi><psi| of the pure ``native-fock`` engine;
* secondary: Piquasso 8.0.1 ``piquasso-mixed-fock`` on the same sequences at cutoff <= 8 only (its
  program validation dominated: 378 s for one cutoff-16 pass on this machine). CZ there is an
  Euler decomposition truncated after every factor, so it differs at truncation level.

Physicality after EVERY operation of the long sequence: Hermiticity residual, |tr - 1|, minimum
eigenvalue (``NativeMixedFockState.physicality``).

Assumptions/limitations: double precision; at most 3 modes (density matrices scale as
C(m + c - 1, m)^2); the photon-count probability check of the lossy TMSV uses the untruncated
reduced thermal state of the counted mode (analytic), so its deviation is truncation.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "validation"))
import test_val_fock as tvf  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq import registry  # noqa: E402

SEED = 20261147
NBARS = (0.3, 1.0)
ALPHA, ETAS, NTH = 1.5 + 0.0j, (0.3, 0.7), 0.4
R_TMSV, ETA_TMSV, HOMODYNE_X = 0.6, 0.8, 0.7
PIQUASSO_MAX_CUTOFF = 8  # secondary comparison only; Piquasso mixed Fock is very slow beyond this
# ----- acceptance criteria (declared before evaluation) ---------------------------------
TOL_EXACT = 1e-12  # engine vs exact-in-truncation density matrix (float64 rounding of Kraus/BS sums)
TOL_HERMITIAN = 1e-13  # max |rho - rho^dag| after every operation
TOL_TRACE = 1e-12  # |tr rho - 1| after renormalisation
TOL_MIN_EIG = -1e-12  # min eigenvalue of (rho + rho^dag)/2
TOL_CONSISTENCY = 1e-12  # mixed engine on a pure input vs outer product of the pure engine
K_TRUNC = 10  # untruncated observables: |err| <= K * cutoff * (1 - retained) + 1e-12 (as R45)
TOL_GAUSSIAN_TOP = 1e-6  # Fock conditional moments vs Gaussian moments at the largest cutoff


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="cutoffs up to 48")
    return parser.parse_args()


def engine(name, cutoff, **kw):
    eng = registry.BACKENDS[name].load()(cutoff, norm_tolerance=1 - 1e-12, **kw)
    eng.reset(SEED)
    return eng


def physical(eng):
    if not hasattr(eng.native, "physicality"):  # Piquasso state: same quantities from rho
        rho = eng.get_state().density_matrix
        return {"hermiticity": float(np.abs(rho - rho.conj().T).max()), "trace_error": abs(np.trace(rho) - 1),
                "min_eigenvalue": float(np.linalg.eigvalsh((rho + rho.conj().T) / 2).min())}
    p = eng.native.physicality()
    return {"hermiticity": p["hermiticity"], "trace_error": abs(complex(p["trace"]) - 1),
            "min_eigenvalue": p["min_eigenvalue"]}


def retained(eng):
    return float(np.prod(eng.retained_norms))


def thermal_cov(nbar):
    return tuple(map(tuple, (2 * nbar + 1) * np.eye(2)))


def gaussian_moments_from_fock(state, node):
    m0, v0 = state.quadrature(node, 0.0)
    m1, v1 = state.quadrature(node, np.pi / 2)
    _, v2 = state.quadrature(node, np.pi / 4)
    return np.array([m0, m1]), np.array([[v0, v2 - (v0 + v1) / 2], [v2 - (v0 + v1) / 2, v1]])


def rows_for(cutoff, name):
    rows = []
    # 1. thermal inputs ---------------------------------------------------------------------
    for nbar in NBARS:
        eng = engine(name, cutoff)
        eng.prepare(0, state=pg.GaussianInput(covariance=thermal_cov(nbar)))
        st = eng.get_state()
        p = nbar ** np.arange(cutoff) / (1 + nbar) ** np.arange(1, cutoff + 1)
        exact = np.diag(p / p.sum())
        rows.append({"case": f"thermal input nbar={nbar}", "engine": name, "cutoff": cutoff,
                     "exact_error": float(np.abs(st.density_matrix - exact).max()),
                     "retained_norm": retained(eng), "retained_norm_error": abs(retained(eng) - p.sum()),
                     "observable": "<n>", "observable_error": abs(st.photon_number(0) - nbar), **physical(eng)})
    # 2. displaced squeezed thermal input vs dense expm construction -------------------------
    r, theta, beta, nth = 0.4, 0.8, 0.6 + 0.3j, 0.3
    big = 90
    a = tvf.ladders(1, big)[0]
    u = expm(beta * a.conj().T - np.conj(beta) * a) @ np.diag(np.exp(1j * theta * np.arange(big))) \
        @ expm(r * (a @ a - a.T @ a.T) / 2)
    weights = nth ** np.arange(60) / (1 + nth) ** np.arange(1, 61)
    rho_big = sum(w * np.outer(u[:, k], u[:, k].conj()) for k, w in enumerate(weights))
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cov = (2 * nth + 1) * rot @ np.diag(np.exp([-2 * r, 2 * r])) @ rot.T
    mean = np.array([2 * beta.real, 2 * beta.imag])
    eng = engine(name, cutoff)
    eng.prepare(0, state=pg.GaussianInput(tuple(mean), tuple(map(tuple, cov))))
    st = eng.get_state()
    exact = rho_big[:cutoff, :cutoff]
    keep = np.trace(exact).real
    m_f, v_f = gaussian_moments_from_fock(st, 0)
    rows.append({"case": "displaced squeezed thermal input", "engine": name, "cutoff": cutoff,
                 "exact_error": float(np.abs(st.density_matrix - exact / keep).max()),
                 "retained_norm": retained(eng), "retained_norm_error": abs(retained(eng) - keep),
                 "observable": "Gaussian mean & covariance",
                 "observable_error": float(max(np.abs(m_f - mean).max(), np.abs(v_f - cov).max())), **physical(eng)})
    # 3. pure loss on a coherent state -> coherent sqrt(eta) alpha ---------------------------
    for eta in ETAS:
        eng = engine(name, cutoff)
        eng.prepare(0, state=pg.GaussianInput.coherent(ALPHA))
        eng.loss(0, eta, 0.0)
        st = eng.get_state()
        rho = st.density_matrix
        target = np.array([complex(x) for x in tvf._coherent_mp(np.sqrt(eta) * ALPHA, cutoff)])
        fid = float(np.vdot(target, rho @ target).real)
        rows.append({"case": f"pure loss eta={eta} on |alpha={abs(ALPHA)}>", "engine": name, "cutoff": cutoff,
                     "exact_error": np.nan, "retained_norm": retained(eng), "retained_norm_error": np.nan,
                     "observable": "1 - F(|sqrt(eta) alpha>), 1 - purity, <n>",
                     "observable_error": float(max(1 - fid, 1 - np.trace(rho @ rho).real,
                                                   abs(st.photon_number(0) - eta * abs(ALPHA) ** 2))),
                     **physical(eng)})
    # 4. thermal loss: <n> = eta |alpha|^2 + (1 - eta) n_th; Gaussian moments ----------------
    for eta in ETAS:
        eng = engine(name, cutoff)
        eng.prepare(0, state=pg.GaussianInput.coherent(ALPHA))
        eng.loss(0, eta, NTH)
        st = eng.get_state()
        m_f, v_f = gaussian_moments_from_fock(st, 0)
        m_g = np.sqrt(eta) * np.array([2 * ALPHA.real, 2 * ALPHA.imag])
        v_g = (eta + (1 - eta) * (2 * NTH + 1)) * np.eye(2)
        rows.append({"case": f"thermal loss eta={eta}, n_th={NTH}", "engine": name, "cutoff": cutoff,
                     "exact_error": np.nan, "retained_norm": retained(eng), "retained_norm_error": np.nan,
                     "observable": "<n>, mean, covariance",
                     "observable_error": float(max(abs(st.photon_number(0) - (eta * abs(ALPHA) ** 2 + (1 - eta) * NTH)),
                                                   np.abs(m_f - m_g).max(), np.abs(v_f - v_g).max())),
                     **physical(eng)})
    # 5. partial trace of a two-mode squeezed vacuum -----------------------------------------
    eng = engine(name, cutoff)
    eng.prepare(0, state=pg.GaussianInput.squeezed(R_TMSV))
    eng.prepare(1, state=pg.GaussianInput.squeezed(-R_TMSV))
    eng.beamsplitter(0, 1, np.pi / 4)
    red = eng.get_state(nodes=(1,))
    lam = np.tanh(R_TMSV)
    p = np.array([lam ** (2 * k) if 2 * k < cutoff else 0.0 for k in range(cutoff)])
    order = [b[0] for b in red.basis]
    exact = np.diag((p / p.sum())[order])
    rows.append({"case": f"partial trace of TMSV r={R_TMSV}", "engine": name, "cutoff": cutoff,
                 "exact_error": float(np.abs(red.density_matrix - exact).max()),
                 "retained_norm": retained(eng), "retained_norm_error": np.nan,
                 "observable": "<n> = sinh^2 r", "observable_error": abs(red.photon_number(1) - np.sinh(R_TMSV) ** 2),
                 **physical(eng)})
    # 6. lossy TMSV: photon counting and homodyne conditioning vs Gaussian moments -------------
    def lossy_tmsv(e):
        e.prepare(0, state=pg.GaussianInput.squeezed(R_TMSV))
        e.prepare(1, state=pg.GaussianInput.squeezed(-R_TMSV))
        e.beamsplitter(0, 1, np.pi / 4)
        e.loss(0, ETA_TMSV, 0.0)

    nbar_counted = ETA_TMSV * np.sinh(R_TMSV) ** 2
    for k in (0, 1, 2):
        eng = engine(name, cutoff)
        lossy_tmsv(eng)
        eng.measure(0, pg.PhotonNumber(), outcome=k)
        prob = eng.last_measurement["value"]
        rows.append({"case": f"lossy TMSV: count k={k}", "engine": name, "cutoff": cutoff, "exact_error": np.nan,
                     "retained_norm": retained(eng), "retained_norm_error": np.nan, "observable": "P(k)",
                     "observable_error": abs(prob - nbar_counted**k / (1 + nbar_counted) ** (k + 1)),
                     **physical(eng)})
    eng = engine(name, cutoff)
    lossy_tmsv(eng)
    eng.measure(0, pg.Homodyne(0.0), 0.0, outcome=HOMODYNE_X)
    density_f = eng.last_measurement["value"]
    m_f, v_f = gaussian_moments_from_fock(eng.get_state(), 1)
    g = registry.create_backend("gaussian")
    g.reset(0)
    lossy_tmsv(g)
    g.measure(0, pg.Homodyne(0.0), 0.0, outcome=HOMODYNE_X)
    gs = g.get_state()
    rows.append({"case": "lossy TMSV: homodyne-conditioned mode 1 vs Gaussian", "engine": name, "cutoff": cutoff,
                 "exact_error": np.nan, "retained_norm": retained(eng), "retained_norm_error": np.nan,
                 "observable": "conditional mean, covariance, outcome density",
                 "observable_error": float(max(np.abs(m_f - gs.mean).max(), np.abs(v_f - gs.covariance).max(),
                                               abs(density_f / g.last_measurement["value"] - 1))),
                 **physical(eng)})
    return rows


def long_sequence(cutoff):
    """Physicality after every op on 3 modes, and mixed-vs-pure consistency without loss."""
    steps = []
    ops = [("bs", (0, 1), 0.7), ("cz", (1, 2), 0.3), ("kerr", (0,), 0.5), ("cubic", (2,), 0.1),
           ("squeeze", (1,), 0.1), ("displace", (2,), (0.1, 0.2)), ("loss", (1,), (0.9, 0.05))] * 3

    def run(eng, with_loss, record):
        eng.prepare(0, state=pg.GaussianInput.coherent(0.4))
        eng.prepare(1, state=pg.FockInput.number(1))
        eng.prepare(2, 0.2)
        for i, (kind, modes, v) in enumerate(ops):
            if kind == "bs":
                eng.beamsplitter(*modes, v)
            elif kind == "cz":
                eng.entangle(*modes, v)
            elif kind == "kerr":
                eng.kerr(modes[0], v)
            elif kind == "cubic":
                eng.cubic_phase(modes[0], v)
            elif kind == "squeeze":
                eng.squeeze(modes[0], v)
            elif kind == "displace":
                eng.displace(modes[0], *v)
            elif with_loss:
                eng.loss(modes[0], *v)
            if record is not None:
                record.append({"cutoff": cutoff, "step": i, "operation": kind, **physical(eng),
                               "retained_norm_step": eng.retained_norms[-1]})
        return eng

    mixed = run(engine("native-mixed-fock", cutoff), True, steps)
    pure_m = run(engine("native-mixed-fock", cutoff), False, None).get_state().density_matrix
    pure_v = run(engine("native-fock", cutoff), False, None).get_state().state_vector
    consistency = float(np.abs(pure_m - np.outer(pure_v, pure_v.conj())).max())
    if cutoff > PIQUASSO_MAX_CUTOFF:
        return steps, consistency, "not run (cutoff above PIQUASSO_MAX_CUTOFF)"
    try:
        pq = run(engine("piquasso-mixed-fock", cutoff), True, None).get_state().density_matrix
        pq_diff = float(np.abs(pq - mixed.get_state().density_matrix).max())
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        pq_diff = f"piquasso failed: {type(exc).__name__}: {exc}"
    return steps, consistency, pq_diff


def main():
    args = parse_args()
    # Accuracy study, not a timing: pin the shared OpenMP runtime of photographiq._native to one
    # thread. With the default thread count, apply_mixed on small density matrices (serial outer
    # path) runs Eigen's internally OpenMP-parallel GEMM and was measured 30-300x slower on this
    # loaded machine (0.4 s vs 1.4 ms at 2 modes, cutoff 12); see the R47 notes.
    from photographiq import native

    native.require("sampling").set_num_threads(1)
    cutoffs = [8, 12, 16, 24, 32] + ([48] if args.full else [])
    seq_cutoffs = [6, 8, 10] + ([12] if args.full else [])
    rows, steps, seq = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for c in cutoffs:
            for name in ("native-mixed-fock", "piquasso-mixed-fock"):
                if name.startswith("piquasso") and c > PIQUASSO_MAX_CUTOFF:
                    continue  # ponytail: Piquasso's mixed-state program validation took 378 s per cutoff-16 pass here
                rows += rows_for(c, name)
            print(f"  cutoff {c}: done", flush=True)
        for c in seq_cutoffs:
            s, consistency, pq_diff = long_sequence(c)
            steps += s
            seq.append({"cutoff": c, "steps": len(s), "mixed_vs_pure_consistency": consistency,
                        "native_vs_piquasso_mixed": pq_diff,
                        "worst_hermiticity": max(x["hermiticity"] for x in s),
                        "worst_trace_error": max(x["trace_error"] for x in s),
                        "worst_min_eigenvalue": min(x["min_eigenvalue"] for x in s),
                        "min_retained_norm_step": min(x["retained_norm_step"] for x in s)})
            print(f"  sequence cutoff {c}: consistency {consistency:.1e}, piquasso diff {pq_diff}", flush=True)

    native = [r for r in rows if r["engine"] == "native-mixed-fock"]
    top = cutoffs[-1]
    failures = []
    for r in native:
        if not np.isnan(r["exact_error"]) and r["exact_error"] > TOL_EXACT:
            failures.append({"check": "exact-in-truncation", **r})
        if not np.isnan(r["retained_norm_error"]) and r["retained_norm_error"] > TOL_EXACT:
            failures.append({"check": "retained norm", **r})
        if r["hermiticity"] > TOL_HERMITIAN or r["trace_error"] > TOL_TRACE or r["min_eigenvalue"] < TOL_MIN_EIG:
            failures.append({"check": "physicality", **r})
        if r["cutoff"] == top:
            limit = TOL_GAUSSIAN_TOP if "Gaussian" in r["case"] else K_TRUNC * top * (1 - r["retained_norm"]) + 1e-12
            if r["observable_error"] > limit:
                failures.append({"check": "observable at top cutoff", "tolerance": limit, **r})
    for s in steps:
        if s["hermiticity"] > TOL_HERMITIAN or s["trace_error"] > TOL_TRACE or s["min_eigenvalue"] < TOL_MIN_EIG:
            failures.append({"check": "physicality during sequence", **s})
    for q in seq:
        if q["mixed_vs_pure_consistency"] > TOL_CONSISTENCY:
            failures.append({"check": "mixed vs pure", **q})

    extra = {"backend": "native-mixed-fock (compiled) with piquasso-mixed-fock 8.0.1 secondary; NumPy gaussian "
                        "backend for Gaussian moments",
             "reference": "closed forms, dense expm (90 levels), channel invariants, Gaussian conditioning",
             "precision": "complex128", "cutoff": cutoffs, "sequence_cutoffs": seq_cutoffs, "seed": SEED,
             "full": args.full,
             "parameters": {"piquasso_max_cutoff": PIQUASSO_MAX_CUTOFF, "openmp_threads": 1, "nbars": NBARS, "alpha": abs(ALPHA), "etas": ETAS, "n_th": NTH, "tmsv_r": R_TMSV,
                            "tmsv_eta": ETA_TMSV, "homodyne_outcome": HOMODYNE_X,
                            "norm_tolerance": "1 - 1e-12 (truncation studied, never rejected)"},
             "acceptance": {"TOL_EXACT": TOL_EXACT, "TOL_HERMITIAN": TOL_HERMITIAN, "TOL_TRACE": TOL_TRACE,
                            "TOL_MIN_EIG": TOL_MIN_EIG, "TOL_CONSISTENCY": TOL_CONSISTENCY, "K_TRUNC": K_TRUNC,
                            "TOL_GAUSSIAN_TOP": TOL_GAUSSIAN_TOP}}
    common.save_result(rows, "R47_native_mixed_fock_validation", extra=extra)
    common.save_result(steps, "R47_native_mixed_fock_sequence_physicality", extra=extra)
    common.save_result(seq, "R47_native_mixed_fock_sequence_summary", extra=extra)
    if failures:
        common.save_csv(failures, "R47_native_mixed_fock_validation_FAILURES")

    plt = common.setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    cases = sorted({r["case"] for r in native})
    palette = plt.cm.tab20(np.linspace(0, 1, 20))
    markers = "osD^v<>p*hX"
    for i, case in enumerate(cases):
        sub = sorted((r for r in native if r["case"] == case), key=lambda r: r["cutoff"])
        style = dict(color=palette[i % 20], marker=markers[i % len(markers)], ms=3.5, lw=1.2, label=case)
        if not np.isnan(sub[0]["exact_error"]):
            axes[0].semilogy([r["cutoff"] for r in sub], [max(r["exact_error"], 1e-17) for r in sub], **style)
        axes[1].semilogy([r["cutoff"] for r in sub], [max(r["observable_error"], 1e-17) for r in sub], **style)
    axes[0].axhline(TOL_EXACT, color="k", ls="--", lw=1)
    axes[0].set(xlabel="cutoff", ylabel="max |rho - rho_exact(cutoff)|", title="(a) exact-in-truncation")
    axes[0].legend(fontsize=6)
    axes[1].set(xlabel="cutoff", ylabel="|observable - untruncated / Gaussian value|",
                title="(b) convergence of physical invariants")
    axes[1].legend(fontsize=5.5, loc="lower left")
    for c, color in zip(seq_cutoffs, plt.cm.viridis(np.linspace(0, 0.85, len(seq_cutoffs)))):
        sub = [s for s in steps if s["cutoff"] == c]
        axes[2].semilogy([s["step"] for s in sub], [max(s["hermiticity"], s["trace_error"], 1e-17) for s in sub],
                         "o-", color=color, ms=3, label=f"c={c}: max(Herm., |tr-1|)")
        axes[2].semilogy([s["step"] for s in sub], [max(-s["min_eigenvalue"], 1e-17) for s in sub], "x:",
                         color=color, ms=4, label=f"c={c}: max(-lambda_min, 0)")
    axes[2].set_xticks(range(0, 21, 2))
    axes[2].set(xlabel="operation index (BS, CZ, Kerr, cubic, squeeze, displace, loss) x 3",
                ylabel="physicality residual", title="(c) physicality after every operation (floor 1e-17 = 0)")
    axes[2].legend(fontsize=6, ncol=2)
    common.save_figure(fig, "R47_native_mixed_fock_validation")
    plt.close(fig)

    worst_exact = max(r["exact_error"] for r in native if not np.isnan(r["exact_error"]))
    hom = next(r for r in native if r["cutoff"] == top and r["case"].startswith("lossy TMSV: homodyne"))
    loss_top = max(r["observable_error"] for r in native if r["cutoff"] == top and r["case"].startswith("pure loss"))
    worst_herm = max(max(s["hermiticity"], s["trace_error"]) for s in steps)
    worst_eig = min(s["min_eigenvalue"] for s in steps)
    common.save_caption(
        "R47_native_mixed_fock_validation",
        f"""
**R47 - native mixed-state Fock validation.** (a) Density matrices of `native-mixed-fock` vs exact
density matrices of the same truncated model (total-photon cutoff c = {cutoffs[0]}..{cutoffs[-1]}): thermal inputs
(nbar = {NBARS}), a displaced squeezed thermal input (r = 0.4, nbar = 0.3, from dense expm), and the reduced state of
a two-mode squeezed vacuum (r = {R_TMSV}, partial trace); worst {worst_exact:.1e} (acceptance {TOL_EXACT:g}).
(b) Convergence of physical invariants with the cutoff: pure loss (eta = {ETAS}) on |alpha = {abs(ALPHA)}> must give the
pure coherent state |sqrt(eta) alpha> (1 - fidelity, 1 - purity and <n> error at c = {top}: {loss_top:.1e}); thermal loss
(n_th = {NTH}) moments; lossy (eta = {ETA_TMSV}) TMSV photon-count probabilities against the analytic thermal marginal;
homodyne conditioning (x = {HOMODYNE_X}) of the lossy TMSV against Gaussian conditioning on the NumPy Gaussian backend
(conditional mean, covariance and outcome density; {hom['observable_error']:.1e} at c = {top}). (c) Physicality after each of
21 operations (incl. Kerr, cubic phase, CZ, thermal loss) on 3 modes: worst Hermiticity/trace residual {worst_herm:.1e},
smallest eigenvalue {worst_eig:.1e}. Without loss, the mixed engine on a pure input equals |psi><psi| of `native-fock`
to {max(q['mixed_vs_pure_consistency'] for q in seq):.1e}. Source: R47 (15_native_fock/47_native_mixed_fock_validation.py).
Limitations: <= 3 modes; float64.
""",
    )
    common.print_summary("R47 native mixed Fock validation", rows=len(rows), worst_exact=f"{worst_exact:.2e}",
                         homodyne_vs_gaussian_top=f"{hom['observable_error']:.2e}", pure_loss_top=f"{loss_top:.2e}",
                         worst_physicality=f"{worst_herm:.2e}", min_eig=f"{worst_eig:.2e}",
                         sequence=[(q["cutoff"], q["mixed_vs_pure_consistency"], q["native_vs_piquasso_mixed"]) for q in seq],
                         failures=len(failures))
    if failures:
        raise AssertionError(f"{len(failures)} R47 checks failed; see R47_native_mixed_fock_validation_FAILURES.csv")


if __name__ == "__main__":
    main()
