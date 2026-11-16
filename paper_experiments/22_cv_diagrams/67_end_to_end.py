"""R67 (O8): end-to-end executable optimisation -- circuit -> CV-MBQC -> optimise -> execute.

Hypothesis: a circuit compiled to CV-MBQC (compile_circuit, r = 1), translated to a typed
diagram, optimised in logical mode and recompiled with ``to_pattern`` executes on the Gaussian
trajectory backend with (i) the same ensemble mean as the original (the logical map and the
displacement are exactly preserved), (ii) the ensemble covariance predicted by *its own*
finite-squeezing channel, which differs from the original's (noise is not preserved by a logical
rewrite), (iii) a certified CV flow, and (iv) fewer resource modes and lower measurement depth.

Method: two-mode circuit R, S, R, CZ, S, R, D, R; coherent input (0.6, -0.3) x (-0.2, 0.4);
2000 trajectories of each pattern (run_shots, Gaussian backend, seed 67), outputs reduced;
ensemble mean/covariance (conditional covariance + spread of conditional means) vs channel
predictions; standard errors from the predicted covariance; find_cv_flow on both clusters.

Acceptance: all |z| <= 4.5 for each pattern against its own channel; mean predictions of the
two channels equal to 1e-9; flow certified for both; resource modes reduced.
"""

import numpy as np
from cvzx_support import COLORS, Timer, plt, save

import photographiq as pg
from photographiq.calculus.cv_diagrams import (
    draw,
    evaluate,
    from_pattern,
    optimize_cv_pattern,
    pattern_flow_report,
    resource_cost,
    to_pattern,
)

NAME = "R67_cvzx_end_to_end"
SHOTS = 2000


def ensemble(pattern, mean_in):
    inputs = {m: pg.GaussianInput(tuple(mean_in[2 * i: 2 * i + 2])) for i, m in enumerate(pattern.inputs)}
    with Timer() as t:
        runs = pg.run_shots(pattern, SHOTS, seed=67, inputs=inputs).trajectories
    states = [r.state.reduced(pattern.outputs) for r in runs]
    means = np.array([s.mean for s in states])
    return means.mean(0), states[0].covariance + np.cov(means.T), t.seconds / SHOTS


def main():
    circuit = (pg.Circuit(2).rotate(0, 0.4).squeeze(0, 0.25).rotate(0, -0.9).cz(0, 1, 0.6)
               .squeeze(1, -0.2).rotate(1, 0.7).displace(0, 0.3, -0.2).rotate(0, 0.5))
    pattern = circuit.compile(squeezing=1.0)
    with Timer() as t_opt:
        result = optimize_cv_pattern(pattern, equivalence="logical")
    optimised = to_pattern(result.best)
    mean_in = np.array([0.6, -0.3, -0.2, 0.4])
    rows, zs = [], {}
    for label, pat in (("compiled", pattern), ("optimised", optimised)):
        X, Y, d = evaluate(from_pattern(pat)).channel()
        mean_pred, cov_pred = X @ mean_in + d, X @ X.T + Y
        mean_meas, cov_meas, per_shot = ensemble(pat, mean_in)
        diag = np.diag(cov_pred)
        z_mean = (mean_meas - mean_pred) / np.sqrt(diag / SHOTS)
        cov_se = np.sqrt((cov_pred**2 + np.outer(diag, diag)) / SHOTS)
        z_cov = (cov_meas - cov_pred) / cov_se
        flow = pattern_flow_report(from_pattern(pat))
        zs[label] = float(max(np.abs(z_mean).max(), np.abs(z_cov).max()))
        cost = resource_cost(from_pattern(pat))
        rows.append({"pattern": label, "resource_modes": cost["resource_modes"], "measurements": cost["measurement_count"],
                     "measurement_depth": cost["measurement_depth"], "entangling_edges": cost["entangling_edges"],
                     "added_noise": cost["added_noise"], "flow_status": flow["status"], "flow_layers": flow.get("layers"),
                     "max_abs_z": zs[label], "seconds_per_shot": per_shot,
                     "mean_pred": mean_pred.tolist(), "mean_meas": mean_meas.tolist(),
                     "cov_pred": cov_pred.tolist(), "cov_meas": cov_meas.tolist()})
        print(f"{label}: modes {cost['resource_modes']}, depth {cost['measurement_depth']}, noise {cost['added_noise']:.3f}, "
              f"flow {flow['status']}, max|z| {zs[label]:.2f}", flush=True)
    c, o = rows
    checks = {"own_channel_compiled": zs["compiled"] <= 4.5, "own_channel_optimised": zs["optimised"] <= 4.5,
              "same_logical_mean": float(np.abs(np.array(c["mean_pred"]) - np.array(o["mean_pred"])).max()) <= 1e-9,
              "flow": c["flow_status"] == o["flow_status"] == "certified",
              "fewer_modes": o["resource_modes"] < c["resource_modes"]}
    status = "pass" if all(checks.values()) else "fail"

    pyplot = plt()
    fig = pyplot.figure(figsize=(7.0, 5.0))
    ax = fig.add_axes([0.08, 0.58, 0.4, 0.34])
    keys = ("resource_modes", "measurements", "measurement_depth", "entangling_edges")
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [c[k] for k in keys], 0.4, color=COLORS["original"], label="compiled")
    ax.bar(x + 0.2, [o[k] for k in keys], 0.4, color=COLORS["optimised"], label="optimised")
    ax.set_xticks(x, ["modes", "measurements", "depth", "CZ edges"], fontsize=6)
    ax.set(ylabel="Count", title="Executable resources")
    ax.legend(fontsize=6)
    ax = fig.add_axes([0.58, 0.58, 0.38, 0.34])
    for row, colour, marker in ((c, COLORS["original"], "o"), (o, COLORS["optimised"], "s")):
        pred, meas = np.array(row["cov_pred"]).ravel(), np.array(row["cov_meas"]).ravel()
        ax.plot(pred, meas, marker, ms=3, color=colour, label=f"{row['pattern']} (max|z| {row['max_abs_z']:.1f})")
    lim = [min(np.min(c["cov_pred"]), np.min(o["cov_pred"])) - 0.2, max(np.max(c["cov_pred"]), np.max(o["cov_pred"])) + 0.2]
    ax.plot(lim, lim, "k-", lw=0.5)
    ax.set(xlabel="Channel covariance prediction", ylabel=f"Trajectory ensemble ({SHOTS} shots)", title="Each pattern vs its own channel")
    ax.legend(fontsize=6)
    ax = fig.add_axes([0.08, 0.07, 0.88, 0.36])
    small = from_pattern(pg.Circuit(1).rotate(0, 0.4).squeeze(0, 0.25).rotate(0, -0.9).compile(squeezing=1.0))
    best = optimize_cv_pattern(small, equivalence="logical").best
    draw(best, ax, title=f"Mode-0 gate chain after optimisation ({resource_cost(small)['resource_modes']} -> "
                         f"{resource_cost(best)['resource_modes']} resource modes)")
    save(rows, NAME, configuration={"circuit": "R,S,R,CZ,S,R,D,R", "squeezing": 1.0, "shots": SHOTS, "seed": 67,
                                    "input_mean": mean_in.tolist()},
         validation={"status": status, "checks": checks, "optimisation_seconds": t_opt.seconds,
                     "equivalence_report": result.report.to_dict()},
         figure=fig, caption="End-to-end: compiled vs optimised executable CV-MBQC patterns, each validated by trajectories.")
    print(f"R67 status={status} {checks}")
    if status != "pass":
        raise AssertionError("R67 acceptance failed")


if __name__ == "__main__":
    main()
