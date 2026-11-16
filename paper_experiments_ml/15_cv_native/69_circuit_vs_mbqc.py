"""R69 (Figure CVML-2): trainable direct Gaussian circuit vs its compiled CV-MBQC realisation.

Hypothesis: compiling a trainable Gaussian circuit with photographiq.compile_circuit gives an
unconditional channel d -> X d + t, V -> X V X^T + Y whose logical part X, t equals the intended
symplectic map S of the direct circuit for every parameter value, while the finite resource
squeezing r adds the noise Y = exp(-2r) Y_1 (Y_1 fixed by the program), i.e. intended map and
physical channel are separate objects.

Method: random parameters of layered programs (1-3 modes, 1-4 layers, 3 seeds each). Three
independent evaluations: (i) the ideal circuit (direct gate matrices), (ii) the channel
mirrored inside photographiqml.cv, (iii) photographiq.analysis.gaussian_channel of the actual
compiled Pattern; and (iv) trajectory execution of the compiled Pattern (Gaussian backend,
exact Schur-complement conditioning) whose ensemble moments are compared with the channel.

Acceptance: max |X - S|, |t_phys - t_ideal| <= 1e-9; |Y - Y_pattern| <= 1e-9 (1 + |Y|);
Y exp(2r) constant across r to 1e-9; trajectory ensemble mean and covariance within 4 SE
of the channel for every entry. Demonstrated: map/channel correctness. Not demonstrated:
learning performance (R70-R72).
"""

import numpy as np
from cv_support import PALETTE, Timer, plt, save

import photographiq as pg
from photographiq.analysis import gaussian_channel
from photographiqml.cv import CVCluster, layered_program

NAME = "R69_circuit_vs_mbqc"
SHOTS = 1000


def map_rows():
    rows = []
    for modes in (1, 2, 3):
        for layers in (1, 2, 3, 4):
            for seed in range(3):
                prog = layered_program(modes, layers)
                rng = np.random.default_rng(1000 * modes + 10 * layers + seed)
                params = {n: rng.normal(scale=0.7) for n in prog.parameters}
                S, t0, _, _ = prog.channel(params)
                with Timer() as fast:
                    X, t, Y, steps = prog.channel(params, mode="physical", squeezing=0.8)
                with Timer() as slow:
                    pattern = prog.circuit(params).compile(squeezing=0.8)
                    reference = gaussian_channel(pattern)
                rows.append({"study": "map", "modes": modes, "layers": layers, "seed": seed,
                             "logical_error": float(max(np.abs(X - S).max(), np.abs(t - t0).max())),
                             "pattern_matrix_error": float(np.abs(X - reference.matrix).max()),
                             "pattern_noise_error": float(np.abs(Y - reference.noise).max()
                                                          / (1 + np.abs(reference.noise).max())),
                             "teleportation_steps": steps, "commands": len(pattern.commands),
                             "mirror_seconds": fast.seconds, "pattern_seconds": slow.seconds})
            print(f"map modes={modes} layers={layers} worst logical "
                  f"{max(r['logical_error'] for r in rows[-3:]):.1e}", flush=True)
    return rows


def noise_rows():
    rows = []
    for layers in (1, 2, 3):
        prog = layered_program(2, layers)
        params = {n: 0.4 * np.sin(i + 1) for i, n in enumerate(prog.parameters)}
        base = None
        for r in np.linspace(0.25, 2.0, 8):
            Y = prog.channel(params, mode="physical", squeezing=r)[2]
            scaled = Y * np.exp(2 * r)
            base = scaled if base is None else base
            rows.append({"study": "noise", "layers": layers, "r": float(r), "db": float(8.6859 * r),
                         "added_noise": float(np.trace(Y) / 4),
                         "scaling_error": float(np.abs(scaled - base).max() / np.abs(base).max())})
    return rows


def trajectory_rows():
    rows = []
    cases = {"compiled 1-mode layer": None, "2x3 CVCluster cluster": CVCluster(2, 3, squeezing=0.6)}
    for label, cluster in cases.items():
        if cluster is None:
            prog = layered_program(1, 1)
            params = {n: 0.3 + 0.2 * i for i, n in enumerate(prog.parameters)}
            pattern = prog.circuit(params).compile(squeezing=0.6)
            X, t, Y, _ = prog.channel(params, mode="physical", squeezing=0.6)
            d = np.array([0.5, -0.2])
            inputs = {0: pg.GaussianInput(tuple(d))}
            outputs = pattern.outputs
        else:
            params = {n: 0.3 * (i - 1.5) for i, n in enumerate(cluster.parameters)}
            pattern = cluster.pattern(params)
            X, t, Y, _ = cluster.channel(params, mode="physical")
            d = np.array([0.4, 0.1, -0.3, 0.2])
            inputs = {n: pg.GaussianInput(tuple(d[2 * i: 2 * i + 2])) for i, n in enumerate(cluster.inputs)}
            outputs = cluster.outputs
        with Timer() as timer:
            runs = pg.run_shots(pattern, SHOTS, seed=69, inputs=inputs).trajectories
        states = [s.state.reduced(outputs) for s in runs]
        means = np.array([s.mean for s in states])
        conditional = states[0].covariance
        ensemble = conditional + np.cov(means.T)
        predicted_cov, predicted_mean = X @ X.T + Y, X @ d + t
        diag = np.diag(predicted_cov)
        cov_se = np.sqrt((predicted_cov**2 + np.outer(diag, diag)) / SHOTS)
        mean_se = np.sqrt(diag / SHOTS)
        n = len(diag)
        for i in range(n):
            rows.append({"study": "trajectory", "case": label, "kind": "mean", "i": i, "j": -1,
                         "predicted": float(predicted_mean[i]), "measured": float(means[:, i].mean()),
                         "se": float(mean_se[i]), "seconds_per_shot": timer.seconds / SHOTS})
            for j in range(i, n):
                rows.append({"study": "trajectory", "case": label, "kind": "covariance", "i": i, "j": j,
                             "predicted": float(predicted_cov[i, j]), "measured": float(ensemble[i, j]),
                             "conditional": float(conditional[i, j]), "se": float(cov_se[i, j]),
                             "seconds_per_shot": timer.seconds / SHOTS})
        print(f"trajectories {label}: {timer.seconds:.1f}s", flush=True)
    return rows


def main():
    rows = map_rows() + noise_rows() + trajectory_rows()
    m = [r for r in rows if r["study"] == "map"]
    nz = [r for r in rows if r["study"] == "noise"]
    tr = [r for r in rows if r["study"] == "trajectory"]
    z = [abs(r["measured"] - r["predicted"]) / r["se"] for r in tr]
    checks = {
        "logical_map": max(r["logical_error"] for r in m) <= 1e-9,
        "pattern_channel": max(max(r["pattern_matrix_error"], r["pattern_noise_error"]) for r in m) <= 1e-9,
        "noise_scaling": max(r["scaling_error"] for r in nz) <= 1e-9,
        "trajectories": max(z) <= 4,
    }
    status = "pass" if all(checks.values()) else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(2, 2, figsize=(7.0, 5.4))
    ax = axes[0, 0]
    for modes, marker, colour in ((1, "o", PALETTE["ideal"]), (2, "s", PALETTE["physical"]),
                                  (3, "^", PALETTE["cluster"])):
        sub = [r for r in m if r["modes"] == modes]
        layers = sorted({r["layers"] for r in sub})
        worst = [max(max(r["logical_error"], r["pattern_matrix_error"], r["pattern_noise_error"], 1e-17)
                     for r in sub if r["layers"] == L) for L in layers]
        ax.semilogy(layers, worst, marker + "-", color=colour, label=f"{modes} mode{'s' if modes > 1 else ''}")
    ax.set_xticks([1, 2, 3, 4])
    ax.axhline(1e-9, color="k", ls=":", lw=0.8, label="acceptance")
    ax.set(xlabel="Layers", ylabel="Worst error (map, channel)", title="Logical map and channel agreement")
    ax.legend(fontsize=6)
    ax = axes[0, 1]
    for layers, colour in zip((1, 2, 3), (PALETTE["ideal"], PALETTE["physical"], PALETTE["cluster"])):
        sub = [r for r in nz if r["layers"] == layers]
        ax.semilogy([r["db"] for r in sub], [r["added_noise"] for r in sub], "o-", color=colour,
                    label=f"{layers} layer{'s' if layers > 1 else ''}")
    ax.axhline(1, color="0.5", ls="--", lw=0.8, label="vacuum noise")
    ax.set(xlabel="Resource squeezing (dB)", ylabel="Added noise per quadrature (vacuum units)",
           title="Finite-squeezing channel noise")
    ax.legend(fontsize=6)
    ax = axes[1, 0]
    for label, colour in zip(("compiled 1-mode layer", "2x3 CVCluster cluster"), (PALETTE["physical"], PALETTE["cluster"])):
        sub = [r for r in tr if r["case"] == label]
        ax.errorbar([r["predicted"] for r in sub], [r["measured"] for r in sub], yerr=[2 * r["se"] for r in sub],
                    fmt="o", color=colour, ms=3, label=label)
    lim = [min(r["predicted"] for r in tr) - 0.3, max(r["predicted"] for r in tr) + 0.3]
    ax.plot(lim, lim, "k-", lw=0.6)
    ax.set(xlabel="Channel prediction (mean, covariance entries)", ylabel=f"Trajectory ensemble ({SHOTS} shots)",
           title="Executed patterns vs channel (2 SE bars)")
    ax.legend(fontsize=6)
    ax = axes[1, 1]
    steps = sorted({(r["modes"], r["layers"]) for r in m})
    for modes, marker in ((1, "o"), (2, "s"), (3, "^")):
        sub = [r for r in m if r["modes"] == modes]
        ax.loglog([r["teleportation_steps"] for r in sub], [r["pattern_seconds"] for r in sub], marker,
                  color=PALETTE["baseline"], mfc="none")
        ax.loglog([r["teleportation_steps"] for r in sub], [r["mirror_seconds"] for r in sub], marker,
                  color=PALETTE["physical"])
    ax.plot([], [], "o", color=PALETTE["baseline"], mfc="none", label="gaussian_channel(Pattern)")
    ax.plot([], [], "o", color=PALETTE["physical"], label="photographiqml.cv channel")
    ax.set(xlabel="Teleportation steps (measurements)", ylabel="Seconds per channel evaluation",
           title="Cost of the two channel evaluations")
    ax.legend(fontsize=6)
    save(rows, NAME, {"oracle_class": "B", "status": status, "checks": checks, "shots": SHOTS,
                      "max_trajectory_z": float(max(z)), "configurations": len(steps)}, fig)
    print(f"R69 status={status} checks={checks} max z={max(z):.2f}")
    if status != "pass":
        raise AssertionError("R69 acceptance failed")


if __name__ == "__main__":
    main()
