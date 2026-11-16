"""Tutorial 5: verified rewrite search on a compiled CV-MBQC pattern.

``optimize_cv_pattern`` runs a bounded, deterministic search (greedy or beam) in one of three
modes, and compares every candidate with the *original*:
* ``physical``: exact rules only; candidates must preserve (X, Y, d) of the averaged channel;
* ``logical``: adds teleport-chain resynthesis; candidates must preserve (X, d) only;
* ``approximate``: logical plus max|Y_A - Y_B| <= epsilon (an entrywise residual, not a
  diamond-norm bound).

Key claims checked: on a compile_circuit pattern, physical mode finds nothing to change (no
exact rule applies there); logical mode removes resource modes, keeps the logical map, changes
the noise and keeps a certified CV flow; approximate mode with a tiny epsilon accepts no noise
change; beam search is never worse than greedy here; the Pareto front is computed exactly.
"""

from time import perf_counter

import photographiq as pg
from photographiq.calculus.cv_diagrams import (
    apply_rewrite,
    optimize_cv_pattern,
    pareto_front,
    pattern_flow_report,
    resource_cost,
    weighted_cost,
)

start = perf_counter()
circuit = (
    pg.Circuit(2)
    .rotate(0, 0.3)
    .rotate(0, 0.4)
    .squeeze(1, 0.2)
    .cz(0, 1, 0.5)
    .rotate(1, 0.2)
    .displace(0, 0.3, 0.1)
)
pattern = circuit.compile(squeezing=1.0)

physical = optimize_cv_pattern(pattern, equivalence="physical")
logical = optimize_cv_pattern(pattern, equivalence="logical")
beam = optimize_cv_pattern(pattern, equivalence="logical", strategy="beam", beam_width=3)
tight = optimize_cv_pattern(pattern, equivalence="approximate", epsilon=1e-6)
depth = optimize_cv_pattern(pattern, equivalence="logical", objective="measurement_depth")
one_chain = apply_rewrite(logical.original, "teleport-chain-resynthesis")  # first chain only

keys = ("resource_modes", "measurement_depth", "entangling_edges", "added_noise")
print("original:", {k: round(logical.original_cost[k], 4) for k in keys})
for name, res in (
    ("physical", physical),
    ("logical greedy", logical),
    ("logical beam", beam),
    ("approx eps=1e-6", tight),
    ("logical, depth objective", depth),
):
    rep = res.report
    print(
        f"{name:>24}: improved {res.improved!s:5} explored {res.explored:4} "
        f"rejected {res.rejected:3} | "
        + ", ".join(f"{k} {res.best_cost[k]:.4g}" for k in keys)
        + f" | logical {rep.logical_equivalent} physical {rep.physical_equivalent} "
        f"Y residual {rep.channel_Y_residual:.3g}"
    )
print("rules applied (logical greedy):", [h["rule"] for h in logical.history])

flows = [pattern_flow_report(d) for d in (logical.original, logical.best)]
print("CV flow:", [(f["status"], f["layers"], f["nodes"]) for f in flows])

# Costs have different units; weights are explicit choices. The Pareto front avoids them.
costs = [logical.original_cost, resource_cost(one_chain), logical.best_cost]
print("weighted costs (default weights):", [round(weighted_cost(c), 3) for c in costs])
front = pareto_front(costs, ("resource_modes", "added_noise"))
print("(original, one chain, all chains) Pareto front over (modes, noise):", front)
print(f"runtime: {perf_counter() - start:.1f} s")

assert physical.report.physical_equivalent and not physical.improved
assert logical.improved and logical.report.logical_equivalent
assert not logical.report.physical_equivalent  # finite-squeezing noise changed, and reported
assert logical.best_cost["resource_modes"] < logical.original_cost["resource_modes"]
assert tight.report.channel_Y_residual <= 1e-6 and not tight.improved
assert beam.best_cost["resource_modes"] <= logical.best_cost["resource_modes"]
assert all(f["status"] == "certified" for f in flows)
again = optimize_cv_pattern(pattern, equivalence="logical")
assert again.best.fingerprint() == logical.best.fingerprint()  # deterministic
