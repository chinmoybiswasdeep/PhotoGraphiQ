"""R60: End-to-end showcase, from a target Gaussian operation to validated output states.

Pipeline (every number below comes from a real library call or an independent oracle):

1. target: S = R_1(-0.6) CZ(0.8) [R_0(0.4) (+) S_1(0.3)] on two modes (hbar = 2,
   interleaved (q0, p0, q1, p1), R(t) = [[cos t, -sin t], [sin t, cos t]],
   S(r) = diag(e^-r, e^r), CZ(g): p0 += g q1, p1 += g q0);
2. circuit ``pg.Circuit`` -> MBQC pattern ``Circuit.compile(squeezing=r)``;
3. complete weighted cluster graph ``GraphView.from_pattern`` (roles, weights, squeezing,
   homodyne shears) before any flow analysis;
4. flow/determinism analysis ``flows.analyze`` (certified CV flow, layers, correction map,
   ideal-limit determinism map);
5. measurement schedules: the flow's measurement order respects every dependency of its
   correction map, and the compiled command order respects every dependency of the
   feed-forward it executes (the two schedules differ; both are reported);
6. classical correction map: the flow's map (standard form) and the compiled feed-forward
   (``Displace`` commands) are both drawn; a pattern executing exactly the flow's map
   (``flows.cv.corrected_pattern``) is shown to give the same channel as the compiled one;
7. ``native-gaussian`` simulation of K trajectories (distinct seeds, coherent inputs);
8. corrected output-state analysis: per-trajectory conditional covariance (outcome
   independent), conditional means scatter, ensemble mean/covariance vs S mu, S V S^T + N;
9. finite-squeezing noise: intended map S (r independent) and added noise N(r); N(r) e^{2r}
   is checked to be constant (every resource contributes variance e^{-2r});
10. interactive page (all four stages) and static exports;
11. independent validation: analytic symplectic product (NumPy, written here) and the
    NumPy ``gaussian`` backend (same seeds, same trajectories);
12. runtime and reproducibility metadata (timings are provisional on a shared machine).

Usage: python 60_end_to_end_showcase.py [--quick | --full] [--repeats N]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.commands import Measure, command_dependencies  # noqa: E402
from photographiq.flows import analyze  # noqa: E402
from photographiq.flows.cv import corrected_pattern, resource_graph  # noqa: E402
from photographiq.interactive import GraphView, draw_static, execution_trace  # noqa: E402
from photographiq.interactive.static import COLORS, legend_handles  # noqa: E402

NAME = "R60_end_to_end_showcase"
RESOURCE_SQUEEZING = 1.5
SQUEEZING_SWEEP = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
INPUT_ALPHAS = (0.6 + 0.2j, -0.3 + 0.5j)
SEED = 60
GATES = {"rotate0": 0.4, "squeeze1": 0.3, "cz": 0.8, "rotate1": -0.6}

# Acceptance criteria (declared before evaluation).
MAP_TOL = 1e-9  # compiled channel S vs analytic product; shear decompositions are
#                 verified to 1e-8 by the compiler, observed errors are ~1e-12
FLOW_CHANNEL_TOL = 1e-9  # channel of the flow-corrected pattern vs the compiled pattern
NOISE_SCALING_TOL = 1e-9  # relative spread of N(r) e^{2r} over the sweep
BACKEND_TOL = 1e-10  # native-gaussian vs NumPy gaussian, same seed (float rounding only)
CONDITIONAL_COV_TOL = 1e-10  # conditional covariance is outcome independent
STAT_Z = 5.0  # ensemble mean / covariance vs channel prediction, in standard errors
EXPECTED_STATUS = "certified"


def build_circuit() -> pg.Circuit:
    circuit = pg.Circuit(2)
    circuit.rotate(0, GATES["rotate0"])
    circuit.squeeze(1, GATES["squeeze1"])
    circuit.cz(0, 1, GATES["cz"])
    circuit.rotate(1, GATES["rotate1"])
    return circuit


def analytic_target() -> np.ndarray:
    """Independent closed form of the target symplectic map (written out here)."""

    def rot(t):
        return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])

    def local(a, b):
        out = np.zeros((4, 4))
        out[:2, :2], out[2:, 2:] = a, b
        return out

    squeeze = np.diag([math.exp(-GATES["squeeze1"]), math.exp(GATES["squeeze1"])])
    cz = np.eye(4)
    cz[1, 2] = cz[3, 0] = GATES["cz"]  # p0 += g q1, p1 += g q0
    return local(np.eye(2), rot(GATES["rotate1"])) @ cz @ local(rot(GATES["rotate0"]), squeeze)


def stats_check(means: np.ndarray, covariance: np.ndarray, mean_ref, cov_ref) -> dict:
    """z-scores of the ensemble mean and covariance (Gaussian standard errors)."""
    k = len(means)
    sample_mean = means.mean(axis=0)
    spread = np.cov(means.T, ddof=1)
    ensemble = covariance + spread  # law of total covariance (conditional cov is constant)
    z_mean = np.abs(sample_mean - mean_ref) / np.sqrt(np.diag(spread) / k)
    # var of a sample covariance element of the means: (S_ii S_jj + S_ij^2) / (k - 1)
    se = np.sqrt((np.outer(np.diag(spread), np.diag(spread)) + spread**2) / (k - 1))
    z_cov = np.abs(ensemble - cov_ref) / np.where(se > 0, se, np.inf)
    return {
        "sample_mean": sample_mean,
        "ensemble_cov": ensemble,
        "spread": spread,
        "z_mean_max": float(z_mean.max()),
        "z_cov_max": float(z_cov.max()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="100 trajectories, 1 timing repeat")
    mode.add_argument("--full", action="store_true", help="4000 trajectories")
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats")
    args = parser.parse_args(argv)
    trajectories = 100 if args.quick else 4000 if args.full else 600
    repeats = 1 if args.quick else args.repeats
    plt = common.setup_style()
    checks: list[dict] = []

    def check(name, ok, measured, criterion):
        checks.append(
            {
                "check": name,
                "status": "pass" if ok else "FAIL",
                "measured": measured,
                "criterion": criterion,
            }
        )

    timings = {}
    t0 = time.perf_counter()
    # 1-2. target and compilation
    circuit = build_circuit()
    target = analytic_target()
    pattern, ctrace = circuit.compile(squeezing=RESOURCE_SQUEEZING, return_trace=True)
    timings["compile_s"] = time.perf_counter() - t0
    # 3. complete graph before flow analysis
    view = GraphView.from_pattern(pattern, title="Two-mode target, compiled")
    graph = view.graph
    # 4. flow / determinism
    t1 = time.perf_counter()
    flow = analyze(graph)
    timings["flow_analysis_s"] = time.perf_counter() - t1
    check("flow certified", flow.status == EXPECTED_STATUS, flow.status, EXPECTED_STATUS)
    ideal = np.asarray(flow.certificate["determinism"]["matrix"], dtype=float)
    err_ideal = float(np.abs(ideal - target).max())
    check(
        "ideal-limit determinism map == analytic S",
        err_ideal <= MAP_TOL,
        err_ideal,
        f"<= {MAP_TOL}",
    )
    # 5. schedules: each order respects the dependencies of its own correction map
    executed = [c.node for c in pattern.commands if isinstance(c, Measure)]
    position = {n: i for i, n in enumerate(executed)}
    flow_position = {n: i for i, n in enumerate(flow.measurement_order)}
    deps = {(s, t) for s, targets in flow.correction_map.items() for t in targets}
    late = [(s, t) for s, t in deps if t in flow_position and flow_position[s] > flow_position[t]]
    check("flow order respects flow map", not late, len(late), "0 violations")
    executed_deps = set()
    for index, command in enumerate(pattern.commands):
        for key in command_dependencies(command):
            executed_deps.add((key, index))
    measured_at = {
        c.result_key: i for i, c in enumerate(pattern.commands) if isinstance(c, Measure)
    }
    acausal = [(k, i) for k, i in executed_deps if measured_at.get(k, -1) >= i]
    check(
        "compiled order respects executed feed-forward", not acausal, len(acausal), "0 violations"
    )
    reordered = sum(1 for s, t in deps if t in position and position[s] > position[t])
    # 6. correction maps and their equivalence
    flow_view = view.with_flow(flow)
    executed_view = flow_view.with_corrections(pattern)
    flow_map_view = flow_view.with_corrections()
    shears = {u: graph.measurements[u].shear for u in flow.measurement_order}
    flow_pattern = corrected_pattern(
        resource_graph(graph), flow.measurement_order, flow.correction_map, shears
    )
    channel = pg.gaussian_channel(pattern)
    channel_flow = pg.gaussian_channel(flow_pattern)
    eq = max(
        float(np.abs(channel.matrix - channel_flow.matrix).max()),
        float(np.abs(channel.noise - channel_flow.noise).max()),
        float(np.abs(channel.displacement - channel_flow.displacement).max()),
    )
    check(
        "flow-map pattern channel == compiled channel",
        eq <= FLOW_CHANNEL_TOL,
        eq,
        f"<= {FLOW_CHANNEL_TOL}",
    )
    err_map = float(np.abs(channel.matrix - target).max())
    check("compiled S == analytic S", err_map <= MAP_TOL, err_map, f"<= {MAP_TOL}")
    symplectic = float(
        np.abs(
            channel.matrix @ np.kron(np.eye(2), [[0, 1], [-1, 0]]) @ channel.matrix.T
            - np.kron(np.eye(2), [[0, 1], [-1, 0]])
        ).max()
    )
    check("compiled S symplectic", symplectic <= MAP_TOL, symplectic, f"<= {MAP_TOL}")
    output_formulas = {
        str(t): c.formula() for t, c in executed_view.corrections.items() if t in graph.outputs
    }

    # 7-8. trajectories
    inputs = {i: pg.GaussianInput.coherent(a) for i, a in enumerate(INPUT_ALPHAS)}
    mean_in = np.array([v for a in INPUT_ALPHAS for v in (2 * a.real, 2 * a.imag)])
    cov_in = np.eye(4)
    mean_ref = target @ mean_in + channel.displacement
    cov_ref = target @ cov_in @ target.T + channel.noise
    means, covs, records = [], [], []
    t2 = time.perf_counter()
    for k in range(trajectories):
        result = pg.simulate(pattern, backend="native-gaussian", seed=SEED + k, inputs=inputs)
        assert tuple(result.state.nodes) == tuple(channel.outputs)
        means.append(result.state.mean)
        covs.append(result.state.covariance)
        records.append(result)
    timings["native_trajectories_s"] = time.perf_counter() - t2
    means, covs = np.array(means), np.array(covs)
    cond_spread = float(np.abs(covs - covs[0]).max())
    check(
        "conditional covariance outcome independent",
        cond_spread <= CONDITIONAL_COV_TOL,
        cond_spread,
        f"<= {CONDITIONAL_COV_TOL}",
    )
    stats = stats_check(means, covs[0], mean_ref, cov_ref)
    check(
        "ensemble mean == S mu (z)",
        stats["z_mean_max"] <= STAT_Z,
        stats["z_mean_max"],
        f"<= {STAT_Z}",
    )
    check(
        "ensemble cov == S V S^T + N (z)",
        stats["z_cov_max"] <= STAT_Z,
        stats["z_cov_max"],
        f"<= {STAT_Z}",
    )
    target_state = pg.GaussianState(mean_ref, target @ cov_in @ target.T, channel.outputs)
    ensemble_state = pg.GaussianState(mean_ref, cov_ref, channel.outputs)
    fidelity_r = float(target_state.fidelity(ensemble_state))

    # 11. NumPy gaussian backend: identical seeded trajectories
    backend_err = 0.0
    compared = min(trajectories, 60)
    for k in range(compared):
        ref = records[k]
        alt = pg.simulate(pattern, backend="gaussian", seed=SEED + k, inputs=inputs)
        backend_err = max(
            backend_err,
            max(abs(ref.outcomes[key] - alt.outcomes[key]) for key in ref.outcomes),
            float(np.abs(ref.state.mean - alt.state.mean).max()),
            float(np.abs(ref.state.covariance - alt.state.covariance).max()),
        )
    check(
        "native-gaussian == NumPy gaussian (same seeds)",
        backend_err <= BACKEND_TOL,
        backend_err,
        f"<= {BACKEND_TOL}",
    )

    # 9. finite-squeezing sweep: intended map vs added noise
    sweep = []
    for r in SQUEEZING_SWEEP:
        ch = pg.gaussian_channel(circuit.compile(squeezing=r))
        out_state = pg.GaussianState(
            target @ mean_in, target @ cov_in @ target.T + ch.noise, ch.outputs
        )
        sweep.append(
            {
                "squeezing_r": r,
                "squeezing_dB": 20 * math.log10(math.e) * r,
                "map_error_vs_analytic": float(np.abs(ch.matrix - target).max()),
                "trace_noise": float(np.trace(ch.noise)),
                "max_eig_noise": float(np.linalg.eigvalsh(ch.noise).max()),
                "fidelity_to_ideal_output": float(target_state.fidelity(out_state)),
                "noise_times_e2r": ch.noise * math.exp(2 * r),
            }
        )
    scaled = np.array([s.pop("noise_times_e2r") for s in sweep])
    scaling = float(np.abs(scaled - scaled[0]).max() / np.abs(scaled[0]).max())
    check(
        "N(r) = e^{-2r} N0 over sweep",
        scaling <= NOISE_SCALING_TOL,
        scaling,
        f"<= {NOISE_SCALING_TOL} (rel)",
    )
    worst_map = max(s["map_error_vs_analytic"] for s in sweep)
    check("S independent of r", worst_map <= MAP_TOL, worst_map, f"<= {MAP_TOL}")
    slope = float(
        np.polyfit([s["squeezing_r"] for s in sweep], np.log([s["trace_noise"] for s in sweep]), 1)[
            0
        ]
    )

    # 10. interactive + static exports
    trace = execution_trace(pattern, records[0])
    exec_view = executed_view.with_execution(trace)
    html_path = common.RAW_DIR / f"{NAME}.interactive.html"
    exec_view.save_html(html_path, "execution")
    graph_svg = exec_view.save(
        common.RAW_DIR / f"{NAME}_graph_corrections.svg", stage="corrections"
    )

    # 12. timings (provisional; shared machine)
    bench_native = common.benchmark(
        lambda: pg.simulate(pattern, backend="native-gaussian", seed=SEED, inputs=inputs),
        warmup=1,
        repeats=repeats,
    )
    bench_numpy = common.benchmark(
        lambda: pg.simulate(pattern, backend="gaussian", seed=SEED, inputs=inputs),
        warmup=1,
        repeats=repeats,
    )
    timings["native_single_run"] = {k: v for k, v in bench_native.items() if k != "result"}
    timings["numpy_single_run"] = {k: v for k, v in bench_numpy.items() if k != "result"}

    failed = [c for c in checks if c["status"] == "FAIL"]
    rows = checks + [{"check": f"sweep r={s['squeezing_r']}", "status": "data", **s} for s in sweep]
    extra = {
        "circuit": "Circuit(2).rotate(0, 0.4).squeeze(1, 0.3).cz(0, 1, 0.8).rotate(1, -0.6)",
        "conventions": "hbar=2, interleaved (q0,p0,q1,p1), vacuum covariance I",
        "resource_squeezing_r": RESOURCE_SQUEEZING,
        "backend": "native-gaussian (validated against NumPy gaussian)",
        "seeds": [SEED, SEED + trajectories - 1],
        "trajectories": trajectories,
        "input_alphas": [[a.real, a.imag] for a in INPUT_ALPHAS],
        "graph": {
            "nodes": len(graph.nodes),
            "edges": graph.graph.number_of_edges(),
            "inputs": list(graph.inputs),
            "outputs": list(graph.outputs),
            "compiled_gate_steps": [
                (
                    s.source_gate if isinstance(s.source_gate, str) else str(s.source_gate),
                    len(s.measurements),
                )
                for s in ctrace.steps
            ],
        },
        "flow": {
            "status": flow.status,
            "algorithm": flow.algorithm,
            "layers": [list(layer) for layer in flow.layers],
            "measurement_order": list(flow.measurement_order),
            "dependencies": len(deps),
            "assumptions": list(flow.assumptions),
        },
        "executed_measurement_order": executed,
        "flow_pairs_reordered_by_compiled_schedule": reordered,
        "executed_corrections": len(executed_view.correction_arrows()),
        "flow_corrections": len(flow_map_view.correction_arrows()),
        "output_corrections": output_formulas,
        "target_S": target,
        "compiled_S": channel.matrix,
        "added_noise_N": channel.noise,
        "displacement_d": channel.displacement,
        "ensemble_mean": stats["sample_mean"],
        "predicted_mean": mean_ref,
        "ensemble_cov": stats["ensemble_cov"],
        "predicted_cov": cov_ref,
        "conditional_cov": covs[0],
        "fidelity_ensemble_vs_ideal_at_r": fidelity_r,
        "log_trace_noise_slope": slope,
        "timings_provisional_shared_machine": timings,
        "interactive_html": html_path.relative_to(common.ROOT).as_posix(),
        "graph_svg": graph_svg.relative_to(common.ROOT).as_posix(),
    }
    if failed:
        common.save_csv(failed, f"{NAME}_FAILURES")
    common.save_result(rows, NAME, extra=extra)
    figure(plt, circuit, view, flow_view, executed_view, flow, means, mean_ref, stats, sweep, extra)
    caption(checks, extra, sweep, stats)
    common.print_summary(
        "R60 end-to-end showcase",
        checks=len(checks),
        failed=len(failed),
        compiled_map_error=err_map,
        trace_noise_at_r=float(np.trace(channel.noise)),
        fidelity_at_r=fidelity_r,
        slope=slope,
        trajectories=trajectories,
    )
    if failed:
        raise AssertionError(f"R60 checks failed: {[c['check'] for c in failed]}")


def ellipse(ax, mean, cov, color, label=None, nsig=2.0, ls="--"):
    values, vectors = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(t), np.sin(t)])
    pts = vectors @ (nsig * np.sqrt(np.maximum(values, 0))[:, None] * circle)
    ax.plot(mean[0] + pts[0], mean[1] + pts[1], ls=ls, color=color, lw=1.1, label=label)


def figure(
    plt, circuit, view, flow_view, executed_view, flow, means, mean_ref, stats, sweep, extra
):
    from matplotlib.gridspec import GridSpec

    from photographiq.visualization import draw_circuit

    fig = plt.figure(figsize=(7.2, 9.0))
    grid = GridSpec(
        5, 2, figure=fig, height_ratios=[0.8, 1.05, 1.25, 1.05, 2.3], hspace=0.38, wspace=0.3
    )
    small = {"node_size": 170, "fontsize": 6.0, "formulas": False, "legend": False}
    # A: target circuit
    ax = fig.add_subplot(grid[0, :])
    draw_circuit(circuit, ax=ax)
    ax.set_title("")  # draw_circuit's own centred title
    for text in ax.texts:
        text.set_fontsize(5.5)
    ax.set_title(
        "A  target circuit: R(0.4) on mode 0, S(0.3) on mode 1, CZ(0.8), R(-0.6) on mode 1",
        fontsize=8,
        loc="left",
    )
    # B: complete weighted cluster graph
    ax = fig.add_subplot(grid[1, :])
    draw_static(view, stage="construct", ax=ax, title="", **small)
    ax.set_title(
        f"B  complete weighted cluster graph: {extra['graph']['nodes']} modes, {extra['graph']['edges']} CZ edges, resource r = {RESOURCE_SQUEEZING}",
        fontsize=8,
        loc="left",
    )
    # C: flow dependencies with each node's measurement layer Lk (same layout as B)
    ax = fig.add_subplot(grid[2, :])
    draw_static(
        flow_view, stage="flow", ax=ax, title="", weights=False, order_label="layer", **small
    )
    ax.set_title(
        f"C  CV flow (status: {flow.status}): {len(flow.layers)} measurement layers L0..L{len(flow.layers) - 1}, {extra['flow']['dependencies']} dependencies",
        fontsize=8,
        loc="left",
    )
    # D: executed correction functions
    ax = fig.add_subplot(grid[3, :])
    draw_static(executed_view, stage="corrections", ax=ax, title="", weights=False, **small)
    outs = "; ".join(f"{t}: {f}" for t, f in extra["output_corrections"].items())
    ax.set_title(
        f"D  executed corrections (raw records m[k]); outputs {outs}", fontsize=8, loc="left"
    )
    for panel in fig.axes:  # equal-aspect panels: align left so titles line up
        panel.set_anchor("W")
    # E: output-state observables
    ax = fig.add_subplot(grid[4, 0])
    colors = (COLORS["input_edge"], COLORS["output_edge"])
    spread = extra["predicted_cov"] - extra["conditional_cov"]  # predicted spread of the means
    for mode, color in enumerate(colors):
        sl = slice(2 * mode, 2 * mode + 2)
        ax.scatter(
            means[:, sl][:, 0],
            means[:, sl][:, 1],
            s=3,
            alpha=0.35,
            color=color,
            lw=0,
            rasterized=True,
        )
        ellipse(ax, mean_ref[sl], spread[sl, sl], color, label=f"mode {mode}: predicted 2$\\sigma$")
        ax.plot(*mean_ref[sl], marker="+", ms=10, mew=1.6, color="black")
        ax.plot(*stats["sample_mean"][sl], marker="x", ms=6, mew=1.2, color=color)
    ax.set_xlabel("conditional output mean  $\\langle q\\rangle$")
    ax.set_ylabel("$\\langle p\\rangle$")
    ax.set_title(
        f"E  {extra['trajectories']} native-gaussian runs (+: S$\\mu$, x: sample)",
        fontsize=8,
        loc="left",
    )
    ax.legend(fontsize=6, frameon=False, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    # F: accuracy and finite-squeezing convergence
    ax = fig.add_subplot(grid[4, 1])
    r = np.array([s["squeezing_r"] for s in sweep])
    tr = np.array([s["trace_noise"] for s in sweep])
    ax.semilogy(r, tr, "o", color=COLORS["correction"], label="Tr N(r) (gaussian_channel)")
    ax.semilogy(
        r, tr[0] * np.exp(-2 * (r - r[0])), "-", color="#475569", lw=1, label="$\\propto e^{-2r}$"
    )
    ax.semilogy(
        r,
        [1 - s["fidelity_to_ideal_output"] for s in sweep],
        "s",
        ms=4,
        color=COLORS["flow"],
        label="1 - F(ideal, noisy output)",
    )
    ax.semilogy(
        r,
        [max(s["map_error_vs_analytic"], 1e-17) for s in sweep],
        "^",
        ms=4,
        color="black",
        label="max|S(r) - S$_{analytic}$|",
    )
    ax.set_xlabel("resource squeezing r")
    ax.set_ylim(1e-17, 1e2)
    ax.set_title("F  intended map vs added noise", fontsize=8, loc="left")
    ax.legend(fontsize=5.6, frameon=False, loc="center right")
    handles = legend_handles("flow") + legend_handles("correction")[-1:]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=6,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.07)
    common.save_figure(fig, NAME, tight=False)
    plt.close(fig)


def caption(checks, extra, sweep, stats):
    passed = sum(c["status"] == "pass" for c in checks)
    at_r = next(s for s in sweep if s["squeezing_r"] == RESOURCE_SQUEEZING)
    timing = extra["timings_provisional_shared_machine"]
    common.save_caption(
        NAME,
        f"""
**R60: end-to-end showcase.** (A) Target two-mode Gaussian circuit R(0.4) on mode 0,
S(0.3) on mode 1, CZ(0.8), R(-0.6) on mode 1 (hbar = 2, interleaved quadratures). (B) The
complete weighted cluster graph produced by `Circuit.compile(squeezing={RESOURCE_SQUEEZING})`
before any flow analysis: {extra["graph"]["nodes"]} modes, {extra["graph"]["edges"]} CZ edges
(inputs blue squares, outputs green diamonds, thick outlines = homodyne-measured). (C) The
certified CV flow found by `flows.analyze` (Booth-Markham real-linear condition), drawn in
its {len(extra["flow"]["layers"])} measurement layers (each node labelled with its layer Lk), with all
{extra["flow"]["dependencies"]} classical dependencies of its correction map (the compiled
pattern uses its own causal schedule, which orders {extra["flow_pairs_reordered_by_compiled_schedule"]} of
these pairs differently). (D) The {extra["executed_corrections"]}
outcome-dependent displacements the compiled pattern executes (coefficients of raw homodyne
records); the flow's standard-form map has {extra["flow_corrections"]} entries and a pattern
executing exactly that map gives the same channel to {next(c for c in checks if c["check"].startswith("flow-map"))["measured"]:.1e}.
(E) Conditional output means of {extra["trajectories"]} `native-gaussian` trajectories
(seeds {extra["seeds"][0]}-{extra["seeds"][1]}, coherent inputs alpha = 0.6+0.2i, -0.3+0.5i):
they scatter around S mu (+) because finite squeezing leaves an outcome-dependent shift; the
dashed ellipses are the predicted 2-sigma spread (S V S^T + N minus the conditional
covariance); largest z-scores {stats["z_mean_max"]:.2f} (mean) and {stats["z_cov_max"]:.2f}
(covariance), criterion <= {STAT_Z}. (F) The intended map S is independent of r
(max |S(r) - S_analytic| <= {max(s["map_error_vs_analytic"] for s in sweep):.1e}); the added
noise N(r) is exactly e^(-2r) N(0) (fitted log-slope {extra["log_trace_noise_slope"]:.4f}), Tr N
= {at_r["trace_noise"]:.4f} at r = {RESOURCE_SQUEEZING} ({at_r["squeezing_dB"]:.1f} dB), where
the ensemble output has fidelity {extra["fidelity_ensemble_vs_ideal_at_r"]:.4f} with the ideal
output. The finite-squeezing map is a noisy channel, not the ideal unitary. Validation:
{passed}/{len(checks)} checks pass, including the analytic symplectic product (NumPy, written
independently) and identical trajectories on the NumPy `gaussian` backend (max difference
{next(c for c in checks if c["check"].startswith("native"))["measured"]:.1e}). Provisional timings
(shared machine): one native-gaussian run {timing["native_single_run"]["median_seconds"] * 1e3:.1f} ms,
NumPy gaussian {timing["numpy_single_run"]["median_seconds"] * 1e3:.1f} ms, flow analysis
{timing["flow_analysis_s"]:.2f} s. Every panel is regenerated by
`paper_experiments/21_showcase/60_end_to_end_showcase.py`; the interactive page with all four
stages is `{extra["interactive_html"]}`. Limitations: ideal homodyne detection, no loss;
the flow certificate is an infinite-squeezing statement.
""",
    )


if __name__ == "__main__":
    main()
