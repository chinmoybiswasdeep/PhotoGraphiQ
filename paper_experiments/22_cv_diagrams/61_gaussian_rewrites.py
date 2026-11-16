"""R61 (O1): semantically verified Gaussian rewrites of typed CV diagrams.

Hypothesis: each implemented Gaussian rule (rotation/squeezing/shear/displacement fusion,
identity removal, symplectic fusion, channel fusion, feed-forward merge, CZ + q-homodyne ->
feed-forward, preparation + homodyne -> classical source) is an exact identity of the stated
equivalence class, so the rewritten diagram has the original's semantics to rounding.

Method: 40 random instances per rule (seeded generators below), each rewrite applied once.
The rewritten diagram is evaluated with the diagram semantics and compared with an
*independent* reference computed from the original: ``analysis.gaussian_channel`` of the
original's executable Pattern (averaged channel), and for measurement rules the conditional
state and record density of a postselected trajectory of the original pattern on the Gaussian
backend (sequential Schur complements). The displacement-fusion phase is compared with a
dense matrix-exponential Fock computation (tests/validation/cv_fock_reference.py, cutoff 120).

Acceptance: every residual <= 1e-9 (relative to max(1, |matrix|)); phase residual <= 1e-10.
Demonstrated: soundness of these rules on the tested instances, with proofs in the theory note.
Not demonstrated: completeness, or any rule outside this list.
"""

import numpy as np
from cvzx_support import COLORS, plt, save
from cv_fock_reference import FockReference

import photographiq as pg
from photographiq.analysis import gaussian_channel
from photographiq.calculus.cv_diagrams import (
    CZ,
    RULES,
    CVDiagram,
    Displacement,
    Feedforward,
    GaussianLoss,
    Homodyne,
    Prepare,
    Rotation,
    Shear,
    Squeezer,
    draw,
    evaluate,
    to_pattern,
)

NAME = "R61_cvzx_gaussian_rewrites"
N = 40


def instance(rule, rng):
    """A random diagram containing at least one match of ``rule``."""
    a, b = rng.normal(size=2)
    filler = [Rotation(1, rng.normal()), CZ(0, 1, rng.normal())]
    if rule == "rotation-fusion":
        core = [Rotation(0, a), Rotation(0, b)]
    elif rule == "squeezing-fusion":
        core = [Squeezer(0, 0.3 * a), Squeezer(0, 0.3 * b)]
    elif rule == "shear-fusion":
        core = [Shear(0, a), Shear(0, b)]
    elif rule == "displacement-fusion":
        core = [Displacement(0, a, b), Displacement(0, *rng.normal(size=2))]
    elif rule == "identity-removal":
        core = [Rotation(0, 2 * np.pi * rng.integers(-2, 3)), Squeezer(0, 0.0)]
    elif rule == "symplectic-fusion":
        core = [Squeezer(0, 0.3 * a), Rotation(0, b)]
    elif rule == "channel-fusion":
        core = [GaussianLoss(0, rng.uniform(0.3, 1), rng.uniform(0, 0.5)), Rotation(0, a)]
    elif rule == "feedforward-merge":
        core = [Prepare.squeezed("x", 0.5), CZ(0, "x", 1.0), Homodyne("x", rng.uniform(0.2, 2.9), "m"),
                Feedforward(1, (("m", tuple(rng.normal(size=2))),)), Feedforward(1, (("m", tuple(rng.normal(size=2))),))]
    elif rule == "cz-q-measurement-to-feedforward":
        core = [Prepare.squeezed("x", rng.uniform(0, 1.5)), CZ(0, "x", a), Homodyne("x", 0.0, "m")]
    else:  # prepare-measure-to-source
        core = [Prepare.squeezed("x", rng.uniform(0, 1.5)), Homodyne("x", rng.uniform(0, np.pi), "m"),
                Feedforward(0, (("m", tuple(rng.normal(size=2))),))]
    return CVDiagram((0, 1), tuple(filler[:1] + core + filler[1:]), (0, 1))


def residuals(original, rewritten, rng):
    pattern = to_pattern(original)
    ref = gaussian_channel(pattern)
    X, Y, d = evaluate(rewritten).channel()
    scale = max(1.0, np.abs(ref.matrix).max(), np.abs(ref.noise).max())
    out = {"channel": max(np.abs(X - ref.matrix).max(), np.abs(Y - ref.noise).max(), np.abs(d - ref.displacement).max()) / scale}
    records = evaluate(original).records
    if records:
        outcomes = {k: float(rng.normal()) for k in records}
        mean_in, cov_in = rng.normal(size=4), np.eye(4) * 1.5
        result = pg.simulate(pattern, backend="gaussian", measurement_outcomes=outcomes,
                             initial_state=pg.GaussianState(mean_in, cov_in, pattern.inputs))
        mean, cov, density = evaluate(rewritten).conditional(outcomes, mean_in, cov_in)
        out["conditional"] = max(np.abs(mean - result.state.mean).max(), np.abs(cov - result.state.covariance).max()) / scale
        out["density"] = abs(density - np.exp(result.log_likelihood)) / max(density, 1e-300)
    return out


def main():
    rng = np.random.default_rng(61)
    names = ["rotation-fusion", "squeezing-fusion", "shear-fusion", "displacement-fusion", "identity-removal",
             "symplectic-fusion", "channel-fusion", "feedforward-merge", "cz-q-measurement-to-feedforward",
             "prepare-measure-to-source"]
    rows = []
    for name in names:
        rule = RULES[name]
        for t in range(N):
            original = instance(name, rng)
            match = rule.find(original)[0]
            rewritten = rule.apply(original, match)
            res = residuals(original, rewritten, rng)
            rows.append({"study": "rule", "rule": name, "instance": t, "equivalence": rule.equivalence,
                         "boxes_before": len(original.boxes), "boxes_after": len(rewritten.boxes), **res})
        print(f"{name}: worst {max(max(v for k, v in r.items() if k in ('channel', 'conditional', 'density')) for r in rows if r['rule'] == name):.1e}", flush=True)
    ref = FockReference(120)
    for t in range(N):
        a, b = rng.normal(scale=0.6, size=2), rng.normal(scale=0.6, size=2)
        out = RULES["displacement-fusion"].apply(CVDiagram((0,), (Displacement(0, *a), Displacement(0, *b)), (0,)), (0, 1))
        predicted = float(out.trace[-1].scalar.split("*")[1].rstrip(")"))
        psi = ref.vacuum()
        measured = np.angle(np.vdot(ref.displace(*(a + b)) @ psi, ref.displace(*b) @ ref.displace(*a) @ psi))
        rows.append({"study": "phase", "instance": t, "predicted": predicted, "measured": float(measured),
                     "residual": abs(predicted - measured)})
    worst_rule = max(max(r.get(k, 0) for k in ("channel", "conditional")) for r in rows if r["study"] == "rule")
    worst_density = max(r.get("density", 0) for r in rows if r["study"] == "rule")
    worst_phase = max(r["residual"] for r in rows if r["study"] == "phase")
    checks = {"semantics": worst_rule <= 1e-9, "density": worst_density <= 1e-9, "phase": worst_phase <= 1e-10}
    status = "pass" if all(checks.values()) else "fail"

    pyplot = plt()
    fig = pyplot.figure(figsize=(7.0, 5.6))
    example = CVDiagram((0,), (Prepare.squeezed("a", 0.6), Rotation(0, 0.4), CZ(0, "a", 0.8),
                               Homodyne("a", 0.0, "m"), Feedforward(0, (("m", (0.2, 0.0)),))), (0,))
    s1 = RULES["cz-q-measurement-to-feedforward"].apply(example, RULES["cz-q-measurement-to-feedforward"].find(example)[0])
    s2 = RULES["prepare-measure-to-source"].apply(s1, RULES["prepare-measure-to-source"].find(s1)[0])
    s3 = RULES["feedforward-merge"].apply(s2, RULES["feedforward-merge"].find(s2)[0])
    for k, (d, title) in enumerate(((example, "Before: ancilla, CZ, q homodyne"), (s3, "After 3 exact rewrites"))):
        ax = fig.add_axes([0.06 + 0.48 * k, 0.62, 0.42, 0.3])
        draw(d, ax, title=title)
    ax = fig.add_axes([0.08, 0.08, 0.5, 0.42])
    for i, name in enumerate(names):
        vals = [max(r.get("channel", 0), r.get("conditional", 0), 1e-18) for r in rows if r.get("rule") == name]
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.2, 0.2, len(vals)), vals, s=4,
                   color=COLORS["optimised" if RULES[name].equivalence == "exact-instrument" else "logical"])
    ax.axhline(1e-9, color="k", ls=":", lw=0.7)
    ax.set_yscale("log")
    short = ["rotation", "squeezing", "shear", "displacement", "identity", "symplectic", "channel",
             "feed-forward merge", "CZ + q homodyne", "prepare + measure"]
    ax.set_xticks(range(len(names)), short, rotation=35, ha="right", fontsize=6)
    ax.set(ylabel="Residual vs independent reference", title=f"Rule soundness ({N} random instances each)")
    ax = fig.add_axes([0.68, 0.12, 0.3, 0.38])
    ph = [r for r in rows if r["study"] == "phase"]
    ax.plot([r["predicted"] for r in ph], [r["measured"] for r in ph], "o", ms=3, color=COLORS["reference"])
    lim = [min(r["predicted"] for r in ph), max(r["predicted"] for r in ph)]
    ax.plot(lim, lim, "k-", lw=0.6)
    ax.set(xlabel="Predicted phase $(a_q b_p - a_p b_q)/4$", ylabel="Phase from Fock expm", title="Displacement-fusion phase")
    save(rows, NAME, configuration={"instances_per_rule": N, "seed": 61, "fock_reference_cutoff": 120},
         validation={"status": status, "checks": checks, "worst_semantic_residual": worst_rule,
                     "worst_density_residual": worst_density, "worst_phase_residual": worst_phase,
                     "references": ["gaussian_channel(Pattern)", "postselected Gaussian-backend trajectory", "dense expm"]},
         figure=fig, caption="Typed CV diagram rewrites: (a,b) an ancilla eliminated by exact rewrites; "
                             "(c) residuals of each rule against independent references; (d) displacement-fusion phase.")
    print(f"R61 status={status} {checks}")
    if status != "pass":
        raise AssertionError("R61 acceptance failed")


if __name__ == "__main__":
    main()
