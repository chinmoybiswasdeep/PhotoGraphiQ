"""R74 (Figure CVML-7): physical and computational resources, and accuracy-resource trade-offs.

Hypothesis: resources follow from structure. A compiled layered program needs one resource
mode and one homodyne per teleportation step (4-5 per single-mode gate, 8 per CZ, ~100 per
beam splitter), while a w x c CVCluster cluster needs w (c - 1) measurements and w c - w resource
modes; the per-sample cost of the exact moment channel grows polynomially with that size.
Prediction quality under finite shots improves with resource squeezing; a larger cluster is
not automatically better.

Method: resource_report / pattern_resources of executable patterns for compiled programs
(1-4 modes, 1-4 layers) and clusters (1-4 wires, 3-8 columns); timings of the exact channel
(photographiqml.cv), gaussian_channel of the full Pattern (small sizes) and one Gaussian-backend
trajectory. Trade-off: shot-trained (S = 100) 2 x c clusters, c in {3, 4, 5, 6}, r in
{0.5, 1, 1.5}, on the linear-regression split; expected test MSE (S = 100) from exact moments.

Acceptance (structural): cluster measurements = w (c - 1) and resource modes = w (c - 1);
compiled resource modes = measurements = teleportation steps; measurement depth <= measurements.
Timings are provisional (single machine, not idle); the trade-off is descriptive.
"""

import numpy as np
from cv_support import PALETTE, Timer, linear_regression, plt, save

import photographiq as pg
from photographiq.analysis import gaussian_channel
from photographiqml.cv import CVCluster, CVRegressor, layered_program, pattern_resources

NAME = "R74_resource_scaling"


def timed(fn, repeats=3):
    best = np.inf
    for _ in range(repeats):
        with Timer() as t:
            fn()
        best = min(best, t.seconds)
    return best


def main():
    rows = []
    for modes in (1, 2, 3, 4):
        for layers in (1, 2, 3, 4):
            prog = layered_program(modes, layers)
            params = {n: 0.3 for n in prog.parameters}
            pattern = prog.circuit(params).compile(squeezing=1.0)
            res = pattern_resources(pattern)
            steps = prog.channel(params, mode="physical", squeezing=1.0)[3]
            Y = prog.channel(params, mode="physical", squeezing=1.0)[2]
            row = {"family": "compiled", "modes": modes, "size": layers, "teleportation_steps": steps, **res,
                   "added_noise": float(np.trace(Y) / len(Y)),
                   "channel_seconds": timed(lambda: prog.channel(params, mode="physical", squeezing=1.0)),
                   "trajectory_seconds": timed(lambda: pg.simulate(pattern, seed=0), 1)}
            if res["measurements"] <= 600:
                row["pattern_channel_seconds"] = timed(lambda: gaussian_channel(pattern), 1)
            rows.append(row)
        print(f"compiled modes={modes} done", flush=True)
    for wires in (1, 2, 3, 4):
        for columns in range(3, 9):
            model = CVCluster(wires, columns, squeezing=1.0)
            params = {n: 0.2 for n in model.parameters}
            pattern = model.pattern(params)
            res = pattern_resources(pattern)
            Y = model.channel(params, mode="physical")[2]
            rows.append({"family": "cluster", "modes": wires, "size": columns, **res,
                         "added_noise": float(np.trace(Y) / len(Y)),
                         "channel_seconds": timed(lambda: model.channel(params, mode="physical")),
                         "trajectory_seconds": timed(lambda: pg.simulate(pattern, seed=0), 1),
                         "pattern_channel_seconds": timed(lambda: gaussian_channel(pattern), 1)})
    (Xtr, ytr), _, (Xte, yte) = linear_regression()
    for columns in (3, 4, 5, 6):
        for r in (0.5, 1.0, 1.5):
            model = CVRegressor(CVCluster(2, columns, squeezing=r), mode="physical", shots=100, epochs=80,
                                seed=74).fit(Xtr, ytr)
            means, covs = model.moments(model.params, Xte)
            w, b = np.real(model.head[:-1]), np.real(model.head[-1])
            mu = np.real(means[:, 0::2]) @ w + b
            var = np.einsum("j,njk,k->n", w, np.real(covs[:, 0::2, 0::2]), w) / 100
            report = model.resource_report(Xte)
            rows.append({"family": "tradeoff", "modes": 2, "size": columns, "r": r, "db": 8.6859 * r,
                         "expected_mse": float(np.mean((mu - yte) ** 2 + var)),
                         "resource_modes": report["resource_modes"], "measurements": report["measurements"],
                         "total_squeezing_db": float(8.6859 * r * report["resource_modes"])})
        print(f"trade-off columns={columns} done", flush=True)

    cl = [q for q in rows if q["family"] == "cluster"]
    co = [q for q in rows if q["family"] == "compiled"]
    checks = {
        "cluster_counts": all(q["measurements"] == q["modes"] * (q["size"] - 1) == q["resource_modes"] for q in cl),
        "compiled_counts": all(q["resource_modes"] == q["measurements"] == q["teleportation_steps"] for q in co),
        "depth_bound": all(q["measurement_depth"] <= q["measurements"] for q in cl + co),
    }
    status = "pass" if all(checks.values()) else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(2, 2, figsize=(7.0, 5.4))
    colours = [PALETTE["ideal"], PALETTE["physical"], PALETTE["cluster"], PALETTE["nongaussian"]]
    ax = axes[0, 0]
    for modes, colour in zip((1, 2, 3, 4), colours):
        sub = [q for q in co if q["modes"] == modes]
        ax.semilogy([q["size"] for q in sub], [q["measurements"] for q in sub], "o-", color=colour, label=f"compiled, {modes} mode{'s' if modes > 1 else ''}")
        ax.semilogy([q["size"] for q in sub], [q["measurement_depth"] for q in sub], "o:", color=colour, mfc="none")
        sub = [q for q in cl if q["modes"] == modes]
        ax.semilogy([q["size"] - 1 for q in sub], [q["measurements"] for q in sub], "s--", color=colour, label=f"cluster, {modes} wire{'s' if modes > 1 else ''}")
    ax.set(xlabel="Layers (compiled) / cluster columns - 1", ylabel="Homodyne measurements (dotted: depth)",
           title="Measurement resources")
    ax.legend(fontsize=5, ncol=2, loc="upper right")
    ax = axes[0, 1]
    for family, marker, colour in (("compiled", "o", PALETTE["physical"]), ("cluster", "s", PALETTE["cluster"])):
        sub = [q for q in rows if q["family"] == family]
        ax.loglog([q["measurements"] for q in sub], [q["channel_seconds"] for q in sub], marker, color=colour, label=f"{family}: exact channel")
        ax.loglog([q["measurements"] for q in sub], [q["trajectory_seconds"] for q in sub], marker, color=colour, mfc="none",
                  label=f"{family}: one trajectory")
        both = [q for q in sub if "pattern_channel_seconds" in q]
        ax.loglog([q["measurements"] for q in both], [q["pattern_channel_seconds"] for q in both], "x", color=colour,
                  label=f"{family}: gaussian_channel(Pattern)")
    ax.set(xlabel="Homodyne measurements", ylabel="Seconds (provisional)", title="Simulation cost")
    ax.legend(fontsize=5)
    ax = axes[1, 0]
    for modes, colour in zip((1, 2, 3, 4), colours):
        sub = [q for q in co if q["modes"] == modes]
        ax.semilogy([q["measurements"] for q in sub], [q["added_noise"] for q in sub], "o-", color=colour, label=f"compiled, {modes} mode{'s' if modes > 1 else ''}")
        sub = [q for q in cl if q["modes"] == modes]
        ax.semilogy([q["measurements"] for q in sub], [q["added_noise"] for q in sub], "s--", color=colour, label=f"cluster, {modes} wire{'s' if modes > 1 else ''}")
    ax.set_xscale("log")
    ax.set(xlabel="Homodyne measurements", ylabel="Added noise per quadrature at r = 1 (8.7 dB)",
           title="Finite-squeezing noise vs size")
    ax.legend(fontsize=5, ncol=2)
    ax = axes[1, 1]
    tr = [q for q in rows if q["family"] == "tradeoff"]
    for r, colour in zip((0.5, 1.0, 1.5), (PALETTE["physical"], PALETTE["cluster"], PALETTE["ideal"])):
        sub = [q for q in tr if q["r"] == r]
        ax.plot([q["size"] for q in sub], [q["expected_mse"] for q in sub], "o-", color=colour, label=f"{8.6859 * r:.1f} dB")
    ax.set_yscale("log")
    ax.set_xticks([3, 4, 5, 6])
    ax.set(xlabel="Cluster columns (2 wires)", ylabel="Expected test MSE (S = 100)", title="Accuracy vs size and squeezing")
    ax.legend(fontsize=6)
    save(rows, NAME, {"oracle_class": "A", "status": status, "checks": checks, "timing_status": "PROVISIONAL"}, fig)
    print(f"R74 status={status} checks={checks}")
    if status != "pass":
        raise AssertionError("R74 acceptance failed")


if __name__ == "__main__":
    main()
