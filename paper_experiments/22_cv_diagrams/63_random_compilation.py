"""R63 (O3): random small Gaussian circuits, compiled to CV-MBQC and optimised.

Hypothesis: compile_circuit spends 4-5 teleportation steps per single-mode gate, 8 extra per
CZ and ~100 per beam splitter, so compiled patterns of small random circuits contain teleport
chains whose products can be re-synthesised with fewer steps. A verified logical-mode search
should reduce resource modes and measurement depth while preserving the logical map exactly
(X, d), keep a certified CV flow and remain executable; the finite-squeezing noise changes and
is reported (not assumed to decrease). Physical-mode search should find nothing to change in
compiled patterns (no exact rule applies) -- a negative result reported as such.

Method: 3 x 3 x 6 random circuits (modes 1-3, depth 2/4/6, 6 seeds; cvzx_support.random_circuit),
compile at r = 1; optimize_cv_pattern in physical and logical mode (greedy, objective
resource_modes); find_cv_flow on original and optimised cluster (graphs <= 120 nodes);
executed with gaussian_channel(to_pattern(best)) vs the original.

Acceptance: logical residual <= 1e-9 and executed residual <= 1e-9 for all; flow certified
for every checked optimised pattern; physical mode leaves every channel unchanged.
"""

import numpy as np
from cvzx_support import COLORS, Timer, plt, random_circuit, save

from photographiq.analysis import gaussian_channel
from photographiq.calculus.cv_diagrams import from_pattern, optimize_cv_pattern, pattern_flow_report, to_pattern

NAME = "R63_cvzx_random_compilation"


def main():
    rows, example = [], None
    for modes in (1, 2, 3):
        for depth in (2, 4, 6):
            for seed in range(6):
                rng = np.random.default_rng(1000 * modes + 10 * depth + seed)
                circuit = random_circuit(rng, modes, depth)
                pattern = circuit.compile(squeezing=1.0)
                with Timer() as t_phys:
                    phys = optimize_cv_pattern(pattern, equivalence="physical")
                with Timer() as t_log:
                    log = optimize_cv_pattern(pattern, equivalence="logical")
                executed = gaussian_channel(to_pattern(log.best))
                original = gaussian_channel(pattern)
                row = {"modes": modes, "depth": depth, "seed": seed,
                       **{f"orig_{k}": log.original_cost[k] for k in ("resource_modes", "measurement_depth", "entangling_edges", "added_noise", "squeezing_energy")},
                       **{f"opt_{k}": log.best_cost[k] for k in ("resource_modes", "measurement_depth", "entangling_edges", "added_noise", "squeezing_energy")},
                       "physical_changed": not phys.report.physical_equivalent or phys.improved,
                       "logical_residual": max(log.report.channel_X_residual, log.report.displacement_residual),
                       "executed_residual": float(max(np.abs(executed.matrix - original.matrix).max(), np.abs(executed.displacement - original.displacement).max())),
                       "physical_seconds": t_phys.seconds, "logical_seconds": t_log.seconds, "explored": log.explored}
                if log.best_cost["resource_modes"] + len(pattern.inputs) <= 120:
                    flow = pattern_flow_report(log.best)
                    row.update(flow_status=flow["status"], flow_layers=flow.get("layers"))
                rows.append(row)
                if modes == 2 and depth == 4 and example is None and log.improved:
                    example = (log.original, log.best)
            print(f"modes={modes}: modes {sum(r['orig_resource_modes'] for r in rows if r['modes'] == modes)} -> "
                  f"{sum(r['opt_resource_modes'] for r in rows if r['modes'] == modes)}", flush=True)
    flows = [r["flow_status"] for r in rows if "flow_status" in r]
    checks = {"logical": max(r["logical_residual"] for r in rows) <= 1e-9,
              "executed": max(r["executed_residual"] for r in rows) <= 1e-9,
              "flow": bool(flows) and all(s == "certified" for s in flows),
              "physical_unchanged": not any(r["physical_changed"] for r in rows)}
    status = "pass" if all(checks.values()) else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(2, 2, figsize=(7.0, 5.4))
    ax = axes[0, 0]
    for modes, marker in ((1, "o"), (2, "s"), (3, "^")):
        sub = [r for r in rows if r["modes"] == modes]
        ax.scatter([r["orig_resource_modes"] for r in sub], [r["opt_resource_modes"] for r in sub], s=10, marker=marker,
                   color=COLORS["optimised"], label=f"{modes} mode{'s' if modes > 1 else ''}")
    top = max(r["orig_resource_modes"] for r in rows)
    ax.plot([0, top], [0, top], "k-", lw=0.5)
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set(xlabel="Resource modes, compiled", ylabel="Resource modes, optimised", title="Logical-mode optimisation")
    ax.legend(fontsize=6)
    ax = axes[0, 1]
    ax.scatter([r["orig_measurement_depth"] for r in rows], [r["opt_measurement_depth"] for r in rows], s=10, color=COLORS["logical"])
    top = max(r["orig_measurement_depth"] for r in rows)
    ax.plot([0, top], [0, top], "k-", lw=0.5)
    ax.set(xlabel="Measurement depth, compiled", ylabel="Measurement depth, optimised", title="Measurement depth")
    ax = axes[1, 0]
    ratio = [r["opt_added_noise"] / r["orig_added_noise"] for r in rows if r["orig_added_noise"] > 0]
    ax.hist(ratio, bins=20, color=COLORS["physical"])
    ax.axvline(1, color="k", ls=":", lw=0.7)
    ax.set(xlabel="Added noise ratio (optimised / compiled), r = 1", ylabel="Programs", title="Finite-squeezing noise changes")
    ax = axes[1, 1]
    from photographiq.calculus.cv_diagrams.optimize import resource_cost

    if example:
        for d, colour, label in ((example[0], COLORS["original"], "compiled"), (example[1], COLORS["optimised"], "optimised")):
            c = resource_cost(d)
            keys = ("resource_modes", "entangling_edges", "measurement_count", "measurement_depth", "feedforward_dependencies")
            ax.plot(range(len(keys)), [c[k] for k in keys], "o-", color=colour, label=label)
        ax.set_xticks(range(5), ["modes", "edges", "measurements", "depth", "feed-forward"], fontsize=6)
        ax.set_yscale("log")
        ax.set(title="Example 2-mode, depth-4 circuit", ylabel="Count")
        ax.legend(fontsize=6)
    save(rows, NAME, configuration={"modes": [1, 2, 3], "depths": [2, 4, 6], "seeds": 6, "squeezing": 1.0,
                                    "search": "greedy, objective resource_modes", "generator": "cvzx_support.random_circuit"},
         validation={"status": status, "checks": checks, "flow_checked": len(flows)},
         figure=fig, caption="Random Gaussian circuits: compiled vs optimised CV-MBQC resources, depth and noise.")
    print(f"R63 status={status} {checks}")
    if status != "pass":
        raise AssertionError("R63 acceptance failed")


if __name__ == "__main__":
    main()
