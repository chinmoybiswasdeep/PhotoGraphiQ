"""R65 (O6): optical-resource reductions versus finite-squeezing added noise.

Hypothesis. Logical rewrites (teleport-chain resynthesis) remove resource modes but change
the added noise Y; since every step adds diag(0, e^{-2r}) propagated through later steps,
Y(r) = e^{-2r} Y_1 for both versions, so the noise ratio optimised/original is r-independent
while the absolute difference scales as e^{-2r}. Removing modes does not always lower noise:
re-synthesised shears can amplify the remaining step noise. The exact (modes, noise) Pareto
front over all subsets of re-synthesisable chains makes the trade-off explicit.

Method. 24 random 1-2 mode circuits (depth 3-6, seed 65) compiled at r = 1. For each, every
subset of its re-synthesisable chains (<= 5 chains, so <= 32 variants) is applied; each variant
is verified logically equivalent and its resource modes and added noise (tr Y / 2n) recorded at
r in {0.5, 1, 1.5, 2} (recompiled patterns at each r). Exact physical-mode ancilla elimination
(CZ + q homodyne -> classical source) is included as a control: modes drop, channel unchanged.

Acceptance: every variant logically equivalent (<= 1e-9); Y ratio r-independent (<= 1e-9
relative); the control has identical averaged channel and instrument. The Pareto fronts and the
count of programs where the fewest-mode variant is not the lowest-noise one are reported.
"""

from itertools import combinations

import numpy as np
from cvzx_support import COLORS, plt, random_circuit, save

from photographiq.calculus.cv_diagrams import (
    CZ,
    CVDiagram,
    Feedforward,
    Homodyne,
    Prepare,
    Rotation,
    apply_rewrite,
    compare,
    from_pattern,
    pareto_front,
    resource_cost,
)
from photographiq.calculus.cv_diagrams.rules import TeleportResynthesis

NAME = "R65_cvzx_resource_vs_noise"
SQUEEZINGS = (0.5, 1.0, 1.5, 2.0)


def variants(diagram):
    """All subsets of re-synthesised chains (applied right-to-left so indices stay valid)."""
    rule = TeleportResynthesis()
    matches = rule.find(diagram)[:5]
    out = []
    for k in range(len(matches) + 1):
        for subset in combinations(range(len(matches)), k):
            d = diagram
            for idx in sorted(subset, key=lambda i: -min(min(b) for b in matches[i][0])):
                current = {tuple(sorted(i for b in m[0] for i in b)): m for m in rule.find(d)}
                target = sorted(i for b in matches[idx][0] for i in b)
                # find the same chain in the current diagram (same boxes, indices unchanged to its left)
                m = next((m for key, m in current.items() if key == tuple(target)), None)
                if m is None:
                    break
                d = rule.apply(d, m)
            else:
                out.append((subset, d))
    return out


def main():
    rng = np.random.default_rng(65)
    rows = []
    fronts = []
    for t in range(24):
        modes = 1 + t % 2
        circuit = random_circuit(rng, modes, int(rng.integers(3, 7)), allow=("rotate", "squeeze", "displace", "cz"))
        per_r = {r: from_pattern(circuit.compile(squeezing=r)) for r in SQUEEZINGS}
        sets = {r: variants(d) for r, d in per_r.items()}
        base = sets[1.0]
        items = []
        for subset, d in base:
            rep = compare(per_r[1.0], d)
            noise = {r: resource_cost(next(v for s, v in sets[r] if s == subset))["added_noise"] for r in SQUEEZINGS}
            orig = {r: resource_cost(per_r[r])["added_noise"] for r in SQUEEZINGS}
            ratio = [noise[r] / orig[r] for r in SQUEEZINGS]
            item = {"program": t, "modes": modes, "chains_applied": len(subset), "resource_modes": resource_cost(d)["resource_modes"],
                    "noise_r1": noise[1.0], "noise_ratio_spread": float(np.ptp(ratio) / max(ratio)),
                    "logical_residual": max(rep.channel_X_residual, rep.displacement_residual),
                    **{f"noise_r{r}": noise[r] for r in SQUEEZINGS}, **{f"orig_noise_r{r}": orig[r] for r in SQUEEZINGS}}
            items.append(item)
        front = pareto_front(items, ("resource_modes", "noise_r1"))
        for i, item in enumerate(items):
            item["pareto"] = i in front
        fewest = min(items, key=lambda x: (x["resource_modes"], x["noise_r1"]))
        quietest = min(items, key=lambda x: x["noise_r1"])
        fronts.append({"program": t, "variants": len(items), "front": len(front),
                       "fewest_is_quietest": fewest is quietest})
        rows += items
    # control: exact physical ancilla elimination
    control = CVDiagram((0,), (Prepare.squeezed("a", 0.7), Rotation(0, 0.3), CZ(0, "a", 0.6), Homodyne("a", 0.0, "m"),
                               Feedforward(0, (("m", (0.1, 0.0)),))), (0,))
    eliminated = apply_rewrite(apply_rewrite(control, "cz-q-measurement-to-feedforward"), "prepare-measure-to-source")
    rc = compare(control, eliminated)
    rows.append({"program": "control", "resource_modes_before": resource_cost(control)["resource_modes"],
                 "resource_modes_after": resource_cost(eliminated)["resource_modes"], "physical": rc.physical_equivalent,
                 "conditional": rc.conditional_equivalent})
    var = [r for r in rows if r["program"] != "control"]
    checks = {"logical": max(r["logical_residual"] for r in var) <= 1e-9,
              "ratio_r_independent": max(r["noise_ratio_spread"] for r in var) <= 1e-9,
              "control_exact": bool(rc.physical_equivalent and rc.conditional_equivalent)}
    status = "pass" if all(checks.values()) else "fail"
    not_quietest = sum(not f["fewest_is_quietest"] for f in fronts)

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 3, figsize=(7.0, 2.7))
    ax = axes[0]
    for t in range(4):
        sub = [r for r in var if r["program"] == t]
        pts = sorted((r["resource_modes"], r["noise_r1"], r["pareto"]) for r in sub)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=8, color=f"C{t}", alpha=0.5)
        fr = [p for p in pts if p[2]]
        ax.plot([p[0] for p in fr], [p[1] for p in fr], "-", color=f"C{t}", lw=1)
    ax.set(xlabel="Resource modes", ylabel="Added noise per quadrature, r = 1", title="Variants and Pareto fronts")
    ax = axes[1]
    sub = [r for r in var if r["program"] == 0]
    best = min(sub, key=lambda r: r["resource_modes"])
    orig = max(sub, key=lambda r: r["resource_modes"])
    db = [8.6859 * r for r in SQUEEZINGS]
    ax.semilogy(db, [orig[f"noise_r{r}"] for r in SQUEEZINGS], "o-", color=COLORS["original"], label="compiled")
    ax.semilogy(db, [best[f"noise_r{r}"] for r in SQUEEZINGS], "s-", color=COLORS["optimised"], label="fewest modes")
    ax.set(xlabel="Resource squeezing (dB)", ylabel="Added noise", title="Noise scales as $e^{-2r}$ for both")
    ax.legend(fontsize=6)
    ax = axes[2]
    ratios = [r["noise_r1"] / r["orig_noise_r1.0"] for r in var if r["chains_applied"] == max(x["chains_applied"] for x in var if x["program"] == r["program"])]
    ax.hist(ratios, bins=15, color=COLORS["logical"])
    ax.axvline(1, color="k", ls=":", lw=0.7)
    ax.set(xlabel="Noise ratio, all chains re-synthesised", ylabel="Programs", title=f"Fewest modes not quietest: {not_quietest}/24")
    save(rows, NAME, configuration={"programs": 24, "seed": 65, "squeezings": list(SQUEEZINGS), "max_chains": 5},
         validation={"status": status, "checks": checks, "fronts": fronts, "fewest_not_quietest": not_quietest},
         figure=fig, caption="Resource modes versus finite-squeezing noise for logically equivalent CV-MBQC variants.")
    print(f"R65 status={status} {checks}; fewest-mode variant not quietest in {not_quietest}/24")
    if status != "pass":
        raise AssertionError("R65 acceptance failed")


if __name__ == "__main__":
    main()
