"""R68 (Figure CVML-1): architecture of CV-native measurement-based learning, from real models.

Hypothesis: the CV-native pipeline (classical data -> CV encoding -> trainable Gaussian program
or weighted cluster -> homodyne readout -> classical head -> optimiser) is realised by
executable PhotoGraphiQ patterns whose resource counts follow from the model structure.

Method: train a compiled two-mode layered Gaussian regressor (resource squeezing r = 1) and a
directly trainable 2 x 4 CVCluster cluster on the deterministic linear-regression split; draw the
pipeline annotated with their measured resource reports and the trained cluster (homodyne
angles, weights). Reference checks: cluster measurements = wires (columns - 1); resource
modes of the compiled pattern = teleportation steps reported by the channel propagation;
logical map of the compiled channel = ideal circuit (machine precision).

Demonstrated: structure and resources of executable models. Not demonstrated: any
performance claim (see R70-R75).
"""

import numpy as np
from cv_support import PALETTE, linear_regression, plt, save
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from photographiqml.cv import CVCluster, CVRegressor, layered_program

NAME = "R68_cv_architecture"


def main():
    (Xtr, ytr), _, _ = linear_regression()
    compiled = CVRegressor(layered_program(2, 1), mode="physical", squeezing=1.0, epochs=40, seed=68).fit(Xtr, ytr)
    cluster = CVRegressor(CVCluster(2, 4, squeezing=1.0), mode="physical", epochs=40, seed=68).fit(Xtr, ytr)
    reports = {"compiled": compiled.resource_report(Xtr), "cluster": cluster.resource_report(Xtr)}
    steps = compiled.program.channel(compiled.params, mode="physical", squeezing=1.0)[3]
    logical_error = float(np.abs(compiled.program.channel(compiled.params, mode="physical", squeezing=1.0)[0]
                                 - compiled.program.channel(compiled.params)[0]).max())
    checks = {
        "cluster_measurements": reports["cluster"]["measurements"] == 2 * 3,
        "compiled_resources_equal_steps": reports["compiled"]["resource_modes"] == steps,
        "compiled_logical_map": logical_error < 1e-10,
    }
    rows = [{"model": k, **{a: (float(b) if isinstance(b, (np.floating, float)) else b)
                            for a, b in v.items() if not isinstance(b, dict)}} for k, v in reports.items()]

    pyplot = plt()
    fig = pyplot.figure(figsize=(7.0, 5.0))
    ax = fig.add_axes([0.0, 0.52, 1.0, 0.46])
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    c, k = reports["compiled"], reports["cluster"]
    boxes = [
        ("Data $x$", "classical", "splits 60/20/40"),
        ("CV encoding\n$D(sx)|0\\rangle$", "quantum", f"{c['mean_input_photons']:.2f} photons"),
        ("Gaussian program\nor cluster", "quantum",
         f"{c['resource_modes']} / {k['resource_modes']}\nresource modes"),
        ("Homodyne\nreadout", "quantum", f"{c['measurements']} / {k['measurements']}\nmeasurements"),
        ("Classical\nhead", "classical", f"{c['classical_head_parameters']} weights"),
        ("Prediction\n$\\hat y(x)$", "classical", ""),
    ]
    width, gap = 1.62, 0.36
    for i, (label, kind, note) in enumerate(boxes):
        x0 = 0.1 + i * (width + gap)
        colour = PALETTE["physical"] if kind == "quantum" else PALETTE["baseline"]
        ax.add_patch(FancyBboxPatch((x0, 2.3), width, 1.2, boxstyle="round,pad=0.05", fc="white",
                                    ec=colour, lw=1.3))
        ax.text(x0 + width / 2, 2.9, label, ha="center", va="center", fontsize=7)
        ax.text(x0 + width / 2, 2.1, note, ha="center", va="top", fontsize=6, color="0.25")
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x0 + width + 0.06, 2.9), (x0 + width + gap - 0.06, 2.9),
                                         arrowstyle="-|>", mutation_scale=8, lw=0.9, color="0.2"))
    ax.add_patch(FancyArrowPatch((11.0, 1.15), (5.1, 1.15), connectionstyle="arc3,rad=0.12",
                                 arrowstyle="-|>", mutation_scale=8, lw=0.9, color=PALETTE["ideal"], ls="--"))
    ax.text(8.0, 0.2, "optimiser (Adam / SGD / L-BFGS) with complex-step gradients of the loss",
            ha="center", fontsize=6.5, color=PALETTE["ideal"])
    ax.text(0.1, 3.95, "orange: quantum (physically executed)    grey: classical", fontsize=6.5)
    ax.text(0.1, 1.45, "numbers: compiled layered program / 2x4 cluster", fontsize=6, color="0.25")

    gx = fig.add_axes([0.03, 0.04, 0.45, 0.42])
    model = cluster.program
    angles = model.angles(cluster.params)
    theta = dict(zip(model.order, angles))
    pos = {n: (n[1], -n[0]) for n in model.nodes}
    for u, v, _ in model.edges:
        gx.plot(*zip(pos[u], pos[v]), color="0.35", lw=1.0, zorder=1)
    for (u, v) in model.rung_edges:
        gx.plot(*zip(pos[u], pos[v]), color=PALETTE["cluster"], lw=2.0, zorder=1)
    for n in model.nodes:
        role = "input" if n in model.inputs else "output" if n in model.outputs else "resource"
        fc = {"input": "white", "output": PALETTE["ideal"], "resource": PALETTE["physical"]}[role]
        gx.scatter(*pos[n], s=150, fc=fc, ec="0.1", lw=0.8, zorder=2)
        if n in theta:
            gx.text(pos[n][0] + 0.12, pos[n][1] + 0.22, f"{theta[n]:.2f}", fontsize=6)
    gx.set_xlim(-0.4, model.columns - 0.6)
    gx.set_ylim(-model.wires - 0.1, 0.55)
    gx.set_axis_off()
    gx.set_title(f"Trained cluster, homodyne angles (rad), r = 1 ({k['resource_squeezing_db']:.1f} dB)")
    gx.text(-0.35, -model.wires + 0.05, "white input, orange measured, teal output, purple CZ rung",
            fontsize=6)

    bx = fig.add_axes([0.6, 0.08, 0.38, 0.34])
    keys = [("resource_modes", "resource modes"), ("entangling_edges", "CZ edges"),
            ("measurements", "measurements"), ("measurement_depth", "measurement depth")]
    y = np.arange(len(keys))
    bx.barh(y + 0.2, [c[a] for a, _ in keys], 0.38, color=PALETTE["physical"], label="compiled 2-mode layer")
    bx.barh(y - 0.2, [k[a] for a, _ in keys], 0.38, color=PALETTE["cluster"], label="2x4 cluster")
    bx.set_yticks(y, [b for _, b in keys])
    bx.set_xscale("log")
    bx.set_xlim(3, 3000)
    bx.set_xlabel("count (from the executable pattern)")
    bx.set_title("Measured resources")
    bx.legend(loc="lower right", fontsize=6)
    status = "pass" if all(checks.values()) else "fail"
    save(rows, NAME, {"oracle_class": "A", "status": status, "checks": checks,
                      "logical_map_error": logical_error, "reports": reports,
                      "cluster_angles": [float(a) for a in angles]}, fig)
    print(f"R68 status={status} checks={checks}")
    if status != "pass":
        raise AssertionError("R68 acceptance failed")


if __name__ == "__main__":
    main()
