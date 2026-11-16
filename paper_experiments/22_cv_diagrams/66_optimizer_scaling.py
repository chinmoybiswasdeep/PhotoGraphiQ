"""R66 (O7): growth of candidates, verification cost and runtime of the rewrite search.

Hypothesis: the search cost is dominated by semantic verification (one evaluation of the
candidate's hybrid Gaussian semantics per candidate, O(boxes x live coordinates)) and grows
polynomially with program size; beam search explores more candidates than greedy and finds
the same or better resource counts on these programs. Timings are provisional (one machine).

Method: single-mode random programs of 2..32 gates (rotations, squeezers, shears; seed 66) and
two-mode programs of 2..12 gates with CZ, compiled at r = 1; logical-mode search, greedy and
beam (width 3), objective resource_modes. Also timed: diagram semantics vs
``analysis.gaussian_channel`` on the same pattern, and ``find_cv_flow`` on the cluster (<= 80 nodes;
larger clusters trigger the exact rational fallback, whose cost is outside this study).

Acceptance (structural): every result logically equivalent; beam never worse than greedy.
"""

import numpy as np
from cvzx_support import COLORS, Timer, plt, random_circuit, save

from photographiq.analysis import gaussian_channel
from photographiq.calculus.cv_diagrams import evaluate, from_pattern, optimize_cv_pattern, pattern_flow_report

NAME = "R66_cvzx_optimizer_scaling"


def main():
    rows = []
    rng = np.random.default_rng(66)
    cases = [(1, g) for g in (2, 4, 8, 16, 32)] + [(2, g) for g in (2, 4, 8, 12)]
    for modes, gates in cases:
        for seed in range(3):
            circuit = random_circuit(np.random.default_rng(1000 * modes + 10 * gates + seed), modes, gates,
                                     allow=("rotate", "squeeze", "cz") if modes > 1 else ("rotate", "squeeze"))
            pattern = circuit.compile(squeezing=1.0)
            diagram = from_pattern(pattern)
            with Timer() as ts:
                evaluate(diagram).channel()
            with Timer() as tg:
                gaussian_channel(pattern)
            row = {"modes": modes, "gates": gates, "seed": seed, "boxes": len(diagram.boxes),
                   "semantics_seconds": ts.seconds, "gaussian_channel_seconds": tg.seconds}
            for strategy in ("greedy", "beam"):
                res = optimize_cv_pattern(pattern, equivalence="logical", strategy=strategy, beam_width=3)
                row.update({f"{strategy}_explored": res.explored, f"{strategy}_rejected": res.rejected,
                            f"{strategy}_seconds": res.seconds, f"{strategy}_modes": res.best_cost["resource_modes"],
                            f"{strategy}_logical": bool(res.report.logical_equivalent),
                            f"{strategy}_seconds_per_candidate": res.seconds / max(res.explored, 1)})
            row["original_modes"] = res.original_cost["resource_modes"]
            if row["original_modes"] + modes <= 80:
                flow = pattern_flow_report(diagram)
                row["flow_seconds"], row["flow_status"] = flow["seconds"], flow["status"]
            rows.append(row)
        print(f"modes={modes} gates={gates}: boxes {row['boxes']}, greedy {row['greedy_seconds']:.2f}s, "
              f"beam {row['beam_seconds']:.2f}s, modes {row['original_modes']} -> {row['greedy_modes']}/{row['beam_modes']}", flush=True)
    checks = {"logical": all(r["greedy_logical"] and r["beam_logical"] for r in rows),
              "beam_not_worse": all(r["beam_modes"] <= r["greedy_modes"] for r in rows)}
    status = "pass" if all(checks.values()) else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 3, figsize=(7.0, 2.7))
    boxes = np.array([r["boxes"] for r in rows])
    ax = axes[0]
    ax.loglog(boxes, [r["greedy_explored"] for r in rows], "o", color=COLORS["optimised"], ms=3, label="greedy")
    ax.loglog(boxes, [r["beam_explored"] for r in rows], "s", color=COLORS["logical"], ms=3, mfc="none", label="beam (3)")
    ax.set(xlabel="Diagram boxes", ylabel="Candidates explored", title="Search size")
    ax.legend(fontsize=6)
    ax = axes[1]
    ax.loglog(boxes, [r["semantics_seconds"] for r in rows], "o", ms=3, color=COLORS["optimised"], label="diagram semantics")
    ax.loglog(boxes, [r["gaussian_channel_seconds"] for r in rows], "x", ms=3, color=COLORS["original"], label="gaussian_channel")
    fl = [r for r in rows if "flow_seconds" in r]
    ax.loglog([r["boxes"] for r in fl], [r["flow_seconds"] for r in fl], "^", ms=3, color=COLORS["reference"], label="find_cv_flow")
    ax.set(xlabel="Diagram boxes", ylabel="Seconds (provisional)", title="Verification primitives")
    ax.legend(fontsize=6)
    ax = axes[2]
    ax.loglog(boxes, [r["greedy_seconds"] for r in rows], "o", ms=3, color=COLORS["optimised"], label="greedy total")
    ax.loglog(boxes, [r["beam_seconds"] for r in rows], "s", ms=3, mfc="none", color=COLORS["logical"], label="beam total")
    ax.set(xlabel="Diagram boxes", ylabel="Search seconds (provisional)", title="Optimiser runtime")
    ax.legend(fontsize=6)
    from matplotlib.ticker import NullFormatter

    for ax in axes:
        ax.xaxis.set_minor_formatter(NullFormatter())
    save(rows, NAME, configuration={"cases": cases, "seeds": 3, "squeezing": 1.0, "beam_width": 3},
         validation={"status": status, "checks": checks, "timing_status": "PROVISIONAL"},
         figure=fig, caption="Scaling of the verified rewrite search: candidates, verification cost and runtime.")
    print(f"R66 status={status} {checks}")
    if status != "pass":
        raise AssertionError("R66 acceptance failed")


if __name__ == "__main__":
    main()
