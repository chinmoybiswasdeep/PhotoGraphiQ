"""R52: CV-MBQC determinism: outcome independence in the ideal limit, finite-squeezing noise apart.

Scientific question: for CV-flow corrected Gaussian patterns, (i) does the outcome-averaged
channel (S, N) equal the intended map S at every finite squeezing r, with the added noise N
reported separately and vanishing as r -> oo; (ii) is the corrected conditional output
mean independent of the homodyne records in the ideal limit (residual dependence G ~ e^{-2r}),
with outcome-independent conditional covariance; and (iii) do the library's channel and
postselected trajectories agree with an independent model?

Cases: single-mode teleportation (one wire step), a weighted 4-node wire, a 4-mode graph
teleportation (inputs R_i, cycle cluster C_i), a weighted 2x3 cluster, and a branched graph
(1 input -> 2 outputs, an ideal isometry with one infinitely squeezed output quadrature).
Corrections come from ``photographiq.flows.find_cv_flow``; the executed pattern is
``photographiq.flows.cv.corrected_pattern`` built from exactly that correction map.

Independent reference (this script, no library simulation code): linear Wigner-variable
model. Variables x = (input quadratures, (q_j, p_j) per resource node) with covariance
diag(V_in, e^{2r}, e^{-2r}, ...); CZ(w): p_u += w q_v; records m_u = k_u q_u + p_u
(= record / sin(theta), theta = atan2(1, k_u)); corrections q_t += c_q m_u, p_t += c_p m_u (correction_map convention).
Output rows L, record rows R give S = L_in, N = L_res Sigma L_res^T = e^{2r} N_+ + e^{-2r} N_-
(exact split by the anti-squeezed / squeezed resource columns), and Gaussian conditioning
E[out | m] = L mu + G (m - R mu), G = L Sigma R^T (R Sigma R^T)^{-1}. Closed forms for S:
T(k) = [[-k, -1], [1, 0]] per unit step, [[-k/w, -1/w], [w, 0]] per weighted step, and
C_Z(A) (+)_i [[-s_i, -1], [1, 0]] for graph teleportation.

Library quantities compared: ``gaussian_channel`` (S, N) and ``simulate(backend='gaussian',
measurement_outcomes=...)`` (exact postselection: each run is ONE conditional branch, never a
deterministic execution) plus seeded sampled trajectories, and ``verify_cv_determinism``.

Finite squeezing: S is r-independent, but N != 0 and G != 0 at every finite r; no finite-r
channel is called an ideal unitary. Outcome dependence is measured on the finite subspace
F = null(N_+) (the ideal-limit squeezed output directions of an isometry are excluded).
Seed 20261152. Unit-efficiency homodyne, lossless resources.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.commands import Measure  # noqa: E402
from photographiq.flows import find_cv_flow, resource_graph, verify_cv_determinism  # noqa: E402
from photographiq.flows.cv import corrected_pattern  # noqa: E402
from photographiq.ir import HomodyneSpec, Model, OpenGraph  # noqa: E402
from photographiq.states import GaussianState  # noqa: E402

SEED = 20261152
SQUEEZINGS = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
N_OUTCOME_SETS = 6
N_SEEDED = 40

# ----- acceptance criteria (declared before any evaluation) ---------------------------------
TOL_S = 1e-10  # S vs closed form / model, relative to max(1, |S|): O(10) flops per entry
TOL_N = 1e-9  # N vs model, relative to max(1, |N|) (entries up to e^{2r} ~ 1e3 for isometries)
# Postselected conditional moments: Schur complements of covariances with entries up to
# e^{2 r_max} ~ 1.1e3; float64 rounding ~ 1e2 eps relative per entry -> 1e-9 relative bound.
TOL_CONDITIONAL = 1e-9
SLOPE_N = (-2.05, -1.95)  # d log||P_F N P_F|| / dr, fit over r >= 1 (exact e^{-2r} expected)
# Sampled (seeded) trajectories: std of the corrected output mean over N_SEEDED runs vs the
# model's sqrt(diag G K G^T); with 40 samples the std estimate has ~11% relative error, so a
# 40% band is ~3.5 sigma (statistical check, not exact).
TOL_SEEDED_RATIO = 0.4
SLOPE_G = (-2.15, -1.85)  # d log||P_F G|| / dr over r >= 2 (asymptotic e^{-2r}; pre-asymptotic below)


def make_og(edges, inputs, outputs, shears, nodes=()):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    return OpenGraph(
        Model.CV_GAUSSIAN, graph, inputs, outputs,
        {n: HomodyneSpec(float(shears.get(n, 0.0))) for n in graph if n not in outputs},
        {n: 1.0 for n in graph},
    )


def step(k, w=1.0):
    return np.array([[-k / w, -1.0 / w], [w, 0.0]])


def cases(rng):
    out = {}
    k = 0.35
    out["teleport_1step"] = (make_og([(0, 1, 1.0)], (0,), (1,), {0: k}), step(k))
    ws, ks = (0.7, 1.3, -0.9), (0.2, -0.4, 0.7)
    s = np.eye(2)
    for w, kk in zip(ws, ks):
        s = step(kk, w) @ s
    out["weighted_wire_4"] = (
        make_og([(i, i + 1, w) for i, w in enumerate(ws)], (0,), (3,), dict(enumerate(ks))), s
    )
    n = 4
    weights = {(i, (i + 1) % n): float(rng.uniform(0.3, 1.2)) for i in range(n)}
    shears = rng.uniform(-1, 1, n)
    edges = [(("C", u), ("C", v), w) for (u, v), w in weights.items()]
    edges += [(("R", i), ("C", i), 1.0) for i in range(n)]
    inputs, outputs = tuple(("R", i) for i in range(n)), tuple(("C", i) for i in range(n))
    adjacency = np.zeros((n, n))
    for (u, v), w in weights.items():
        adjacency[u, v] = adjacency[v, u] = w
    local = np.zeros((2 * n, 2 * n))
    for i, si in enumerate(shears):
        local[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = step(si)
    cz = np.eye(2 * n)
    cz[1::2, 0::2] = adjacency  # p_i += sum_j A_ij q_j
    out["graph_teleport_4mode"] = (
        make_og(edges, inputs, outputs, dict(zip(inputs, shears.tolist()))), cz @ local
    )
    g = nx.grid_2d_graph(2, 3)
    grid_edges = [(u, v, round(float(rng.uniform(0.6, 1.4)), 2)) for u, v in g.edges]
    grid_out = ((0, 2), (1, 2))
    out["weighted_cluster_2x3"] = (
        make_og(grid_edges, ((0, 0), (1, 0)), grid_out,
                {v: round(float(rng.uniform(-0.5, 0.5)), 2) for v in g if v not in grid_out}),
        None,
    )
    out["branched_isometry"] = (
        make_og([(0, 1, 1.0), (1, 2, 0.8), (1, 3, 1.3)], (0,), (2, 3), {0: 0.25, 1: -0.3}), None
    )
    return out


def with_squeezing(og, r):
    og = og.copy()
    og.squeezing = {n: float(r) for n in og.nodes}
    return og


# ----- independent linear Wigner-variable model ------------------------------------------
def linear_model(og, order, corrections, shears, outputs, r, v_in):
    inputs = list(og.inputs)
    resource = [n for n in og.nodes if n not in og.inputs]
    dim = 2 * len(inputs) + 2 * len(resource)
    eye = np.eye(dim)
    quad = {}
    for i, n in enumerate(inputs):
        quad[n] = [eye[2 * i].copy(), eye[2 * i + 1].copy()]
    anti, squeezed = [], []
    for j, n in enumerate(resource):
        kq = 2 * len(inputs) + 2 * j
        quad[n] = [eye[kq].copy(), eye[kq + 1].copy()]
        anti.append(kq)
        squeezed.append(kq + 1)
    q0 = {n: quad[n][0].copy() for n in quad}
    for u, v, d in og.graph.edges(data=True):
        w = float(d["weight"])
        quad[u][1] = quad[u][1] + w * q0[v]
        quad[v][1] = quad[v][1] + w * q0[u]
    records = []
    for u in order:
        q, p = quad.pop(u)
        m = shears[u] * q + p
        records.append(m)
        for t, c in corrections[u].items():
            if t in quad:
                quad[t][0] = quad[t][0] + c["q"] * m
                quad[t][1] = quad[t][1] + c["p"] * m
    L = np.array([row for n in outputs for row in quad[n]])
    R = np.array(records)
    sigma = np.zeros((dim, dim))
    sigma[: 2 * len(inputs), : 2 * len(inputs)] = v_in
    sigma[anti, anti] = math.exp(2 * r)
    sigma[squeezed, squeezed] = math.exp(-2 * r)
    nin = 2 * len(inputs)
    n_plus = L[:, anti] @ L[:, anti].T
    n_minus = L[:, squeezed] @ L[:, squeezed].T
    return {"L": L, "R": R, "sigma": sigma, "S": L[:, :nin], "N": math.exp(2 * r) * n_plus + math.exp(-2 * r) * n_minus,
            "N_plus": n_plus, "N_minus": n_minus}


def conditional(model, mu, m):
    L, R, sig = model["L"], model["R"], model["sigma"]
    K = R @ sig @ R.T
    G = np.linalg.solve(K, R @ sig @ L.T).T
    mean = L @ mu + G @ (m - R @ mu)
    cov = L @ sig @ L.T - G @ R @ sig @ L.T
    return mean, cov, G, K


def dist(a, b):
    return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).max(initial=0.0))


def run_case(name, og0, closed, rng):
    flow = find_cv_flow(og0, seed=0)
    assert flow.certified, f"{name}: expected a CV flow certificate, got {flow.status}"
    order, corr, shears = flow.measurement_order, flow.correction_map, flow.certificate["shears"]
    nin = 2 * len(og0.inputs)
    mu_in = rng.normal(0, 1.0, nin)
    v_in = np.eye(nin)  # coherent inputs (vacuum covariance), displaced mean mu_in
    rows, curves = [], []
    for r in SQUEEZINGS:
        og = with_squeezing(og0, r)
        graph = resource_graph(og)
        pattern = corrected_pattern(graph, order, corr, shears)
        channel = pg.gaussian_channel(pattern)
        outputs = tuple(channel.outputs)
        model = linear_model(og, order, corr, shears, outputs, r, v_in)
        eig, vec = np.linalg.eigh(model["N_plus"])
        finite = vec[:, np.abs(eig) <= 1e-9 * max(1.0, np.abs(eig).max(initial=0.0))]
        project = finite @ finite.T
        keys = [(c.result_key, c.node) for c in pattern.commands if isinstance(c, Measure)]
        sin_theta = np.array([1 / math.hypot(1.0, shears[node]) for _, node in keys])
        mu = np.r_[mu_in, np.zeros(model["sigma"].shape[0] - nin)]
        _, _, G, K = conditional(model, mu, model["R"] @ mu)
        record_std = np.sqrt(np.diag(K)) * sin_theta  # std of the calibrated records
        state_in = GaussianState(mu_in, v_in, og.inputs)
        base_mean = base_cov = None
        cond_err = cov_dep = spread = 0.0
        sweep = []
        for j in range(N_OUTCOME_SETS):
            records = np.zeros(len(keys)) if j == 0 else rng.normal(0, 1, len(keys)) * record_std * (1 + j % 3)
            m = records / sin_theta + 0.0
            # model conditional at the SAME records (records in calibrated units, m = record/sin)
            ref_mean, ref_cov, _, _ = conditional(model, mu, m)
            res = pg.simulate(pattern, backend="gaussian", initial_state=state_in,
                              measurement_outcomes={k: float(x) for (k, _), x in zip(keys, records)})
            st = res.state
            idx = [st.nodes.index(o) for o in outputs]
            sel = [i for o in idx for i in (2 * o, 2 * o + 1)]
            mean, cov = st.mean[sel], st.covariance[np.ix_(sel, sel)]
            scale = max(1.0, np.abs(ref_cov).max(), np.abs(model["sigma"]).max())
            cond_err = max(cond_err, dist(mean, ref_mean) / scale, dist(cov, ref_cov) / scale)
            if base_mean is None:
                base_mean, base_cov = mean, cov
            cov_dep = max(cov_dep, dist(cov, base_cov) / scale)
            spread = max(spread, dist(project @ (mean - base_mean), 0))
        if name == "teleport_1step":  # deterministic sweep of the record in units of its std
            for z in np.linspace(-3, 3, 13):
                rec = {keys[0][0]: float(z * record_std[0])}
                st = pg.simulate(pattern, backend="gaussian", initial_state=state_in, measurement_outcomes=rec).state
                sweep.append((float(z), float(st.mean[2 * st.nodes.index(outputs[0])])))
        seeded = np.array([
            (lambda s: s.mean[[i for o in [s.nodes.index(x) for x in outputs] for i in (2 * o, 2 * o + 1)]])(
                pg.simulate(pattern, backend="gaussian", initial_state=state_in, seed=SEED + 31 * k).state)
            for k in range(N_SEEDED)
        ])
        seeded_std = float((project @ (seeded - seeded.mean(0)).T).std(axis=1, ddof=1).max())
        predicted_std = float(np.sqrt(np.clip(np.diag(project @ G @ K @ G.T @ project), 0, None)).max())
        n_lib, s_lib = channel.noise, channel.matrix
        row = {
            "case": name,
            "squeezing_r": r,
            "inputs": len(og.inputs),
            "outputs": len(outputs),
            "measured": len(order),
            "S_lib_vs_model": dist(s_lib, model["S"]) / max(1.0, np.abs(model["S"]).max()),
            "S_lib_vs_closed_form": (dist(s_lib, closed) / max(1.0, np.abs(closed).max())) if closed is not None else None,
            "N_lib_vs_model": dist(n_lib, model["N"]) / max(1.0, np.abs(model["N"]).max()),
            "N_finite_max": float(np.abs(project @ n_lib @ project).max()),
            "N_plus_rank": int(len(eig) - finite.shape[1]),
            "expected_N_plus_rank": len(outputs) - len(og.inputs),
            "gain_finite_max": float(np.abs(project @ G).max()),
            "conditional_map_error_finite": float(np.abs(project @ (model["S"] - G @ model["R"][:, :nin])
                                                         - project @ model["S"]).max()),
            "conditional_lib_vs_model": cond_err,
            "conditional_cov_outcome_dependence": cov_dep,
            "postselected_mean_spread_finite": spread,
            "seeded_output_mean_std": seeded_std,
            "model_predicted_mean_std": predicted_std,
        }
        rows.append(row)
        if sweep:
            curves.append((r, sweep, float((s_lib @ mu_in)[0])))
    og1 = with_squeezing(og0, 1.0)
    report = verify_cv_determinism(corrected_pattern(resource_graph(og1), order, corr, shears), seed=0)
    meta = {
        "case": name,
        "order": [str(n) for n in order],
        "layers": [[str(n) for n in layer] for layer in flow.layers],
        "ideal_map_S": pg.gaussian_channel(corrected_pattern(resource_graph(og1), order, corr, shears)).matrix,
        "verify_ideal_limit_deterministic": report["ideal_limit_deterministic"],
        "verify_failed_checks": report["failed_checks"],
        "verify_decay_exponents": report["decay_exponents"],
        "verify_symplectic_residual": report["symplectic_residual"],
        "input_mean": mu_in,
    }
    return rows, curves, meta


def slope(rows, key, rmin):
    pts = [(r["squeezing_r"], r[key]) for r in rows if r["squeezing_r"] >= rmin and r[key] > 0]
    x, y = np.array(pts).T
    return float(np.polyfit(x, np.log(y), 1)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    plt = common.setup_style()
    rng = common.rng(SEED)
    all_rows, metas, curves, slopes = [], [], [], []
    for name, (og, closed) in cases(rng).items():
        rows, cv, meta = run_case(name, og, closed, rng)
        all_rows += rows
        metas.append(meta)
        curves += cv
        slopes.append({"case": name, "slope_N_finite": slope(rows, "N_finite_max", 1.0),
                       "slope_gain": slope(rows, "gain_finite_max", 2.0)})

    failures = []
    for r in all_rows:
        p = []
        if r["S_lib_vs_model"] > TOL_S or (r["S_lib_vs_closed_form"] is not None and r["S_lib_vs_closed_form"] > TOL_S):
            p.append("S mismatch")
        if r["N_lib_vs_model"] > TOL_N:
            p.append("N mismatch")
        if r["conditional_lib_vs_model"] > TOL_CONDITIONAL:
            p.append("postselected moments differ from model")
        if r["conditional_cov_outcome_dependence"] > TOL_CONDITIONAL:
            p.append("conditional covariance depends on outcomes")
        if r["N_plus_rank"] != r["expected_N_plus_rank"]:
            p.append("ideal-squeezed direction count wrong")
        if abs(r["seeded_output_mean_std"] / r["model_predicted_mean_std"] - 1) > TOL_SEEDED_RATIO:
            p.append("seeded trajectory spread disagrees with model")
        if not r["N_finite_max"] > 0:
            p.append("finite-squeezing noise reported as zero")
        if p:
            failures.append({**r, "problems": "; ".join(p)})
    for s in slopes:
        if not (SLOPE_N[0] <= s["slope_N_finite"] <= SLOPE_N[1]) or not (SLOPE_G[0] <= s["slope_gain"] <= SLOPE_G[1]):
            failures.append({**s, "problems": "decay slope outside acceptance"})
    for m in metas:
        if not m["verify_ideal_limit_deterministic"]:
            failures.append({"case": m["case"], "problems": f"verify_cv_determinism failed {m['verify_failed_checks']}"})

    common.save_result(
        all_rows,
        "R52_cv_mbqc_determinism",
        extra={
            "seed": SEED,
            "backend": "gaussian (NumPy) for simulate/gaussian_channel; independent linear Wigner model (NumPy)",
            "precision": "float64",
            "parameters": {"squeezings": SQUEEZINGS, "outcome_sets": N_OUTCOME_SETS, "seeded_trajectories": N_SEEDED,
                           "input": "coherent (vacuum covariance) with random mean, seeded"},
            "acceptance": {"TOL_S": TOL_S, "TOL_N": TOL_N, "TOL_CONDITIONAL": TOL_CONDITIONAL,
                           "SLOPE_N": SLOPE_N, "SLOPE_G": SLOPE_G, "TOL_SEEDED_RATIO": TOL_SEEDED_RATIO},
            "decay_slopes": slopes,
            "cases": metas,
            "note": "Postselected runs are single conditional branches; S is the ideal map, N the finite-"
                    "squeezing added noise (nonzero at every finite r); never an ideal unitary at finite r.",
        },
    )

    colors = dict(zip([m["case"] for m in metas], plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    rs = np.array(SQUEEZINGS)
    for case, c in colors.items():
        rr = [r for r in all_rows if r["case"] == case]
        axes[0].semilogy([r["squeezing_r"] for r in rr], [r["N_finite_max"] for r in rr], "o-", color=c, label=case, ms=4)
        axes[1].semilogy([r["squeezing_r"] for r in rr], [r["gain_finite_max"] for r in rr], "o-", color=c, ms=4)
    for ax in axes[:2]:
        ax.semilogy(rs, np.exp(-2 * rs), "k--", lw=1, label=r"$e^{-2r}$")
        ax.set_xlabel("resource squeezing r")
    axes[0].set_ylabel(r"$\max|P_F N P_F|$ (added noise)")
    axes[0].set_title("(a) finite-squeezing noise N, ideal map S apart")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_ylabel(r"$\max|P_F G|$ (output shift per record unit)")
    axes[1].set_title("(b) residual outcome dependence")
    for r, sweep, ideal in curves:
        if r in (0.5, 1.0, 2.0, 3.0):
            x, y = np.array(sorted(sweep)).T
            axes[2].plot(x, y, "o-", ms=3, label=f"r = {r:g}")
    ideal_q = curves[0][2]
    axes[2].axhline(ideal_q, color="k", ls="--", lw=1, label=r"ideal $(S\mu)_q$")
    axes[2].set_xlabel("postselected record / its standard deviation")
    axes[2].set_ylabel("corrected output mean q")
    axes[2].set_title("(c) teleportation: E[q_out | record]")
    axes[2].legend(frameon=False, fontsize=7)
    common.save_figure(fig, "R52_cv_mbqc_determinism")
    plt.close("all")
    worst_s = max(max(r["S_lib_vs_model"], r["S_lib_vs_closed_form"] or 0) for r in all_rows)
    worst_n = max(r["N_lib_vs_model"] for r in all_rows)
    worst_c = max(r["conditional_lib_vs_model"] for r in all_rows)
    sl = ", ".join(f"{s['case']}: {s['slope_N_finite']:.3f} / {s['slope_gain']:.3f}" for s in slopes)
    tel = [r for r in all_rows if r["case"] == "teleport_1step"]
    common.save_caption(
        "R52_cv_mbqc_determinism",
        f"""
**R52 - CV-MBQC determinism: ideal map, finite-squeezing noise and outcome dependence kept
apart.** Five CV-flow corrected Gaussian patterns (corrections from `find_cv_flow`, executed
as `corrected_pattern`): one-step teleportation (shear 0.35), weighted 4-node wire, 4-mode graph
teleportation, weighted 2x3 cluster and a branched 1 -> 2 isometry, at resource squeezing
r = {SQUEEZINGS[0]}..{SQUEEZINGS[-1]}. (a) Added noise N of the outcome-averaged channel
V -> S V S^T + N, projected on F = null(N_+) (the isometry's ideal infinitely squeezed output
quadrature excluded); S equals the closed form / independent linear Wigner model to {worst_s:.1e} at
every r, N matches the model to {worst_n:.1e}. (b) Residual dependence of the corrected conditional
mean on the records, G = L Sigma R^T (R Sigma R^T)^-1. Fitted slopes d ln/dr (N over r >= 1 /
G over r >= 2): {sl}; dashed: e^{{-2r}}. (c) Postselected conditional output mean of the
teleported q quadrature versus the record (exact Gaussian postselection; each point is one
branch, not a deterministic run): the slope flattens towards the ideal (S mu)_q as r grows
(teleport N_finite = {tel[0]['N_finite_max']:.3g} at r = {tel[0]['squeezing_r']:g},
{tel[-1]['N_finite_max']:.3g} at r = {tel[-1]['squeezing_r']:g}). Library postselected moments agree with
the independent model to {worst_c:.1e} (relative). At finite r the channel is noisy: never an
ideal unitary. Source: R52 (52_cv_mbqc_determinism.py).
""",
    )
    if failures:
        common.save_csv(failures, "R52_cv_mbqc_determinism_FAILURES")
        raise AssertionError(f"{len(failures)} R52 check(s) failed; see FAILURES csv")
    common.print_summary(
        "R52 CV-MBQC determinism",
        cases=len(metas),
        rows=len(all_rows),
        worst_S=f"{worst_s:.2e}",
        worst_N=f"{worst_n:.2e}",
        worst_conditional=f"{worst_c:.2e}",
        slopes=sl,
        verify_all=all(m["verify_ideal_limit_deterministic"] for m in metas),
    )


if __name__ == "__main__":
    main()
