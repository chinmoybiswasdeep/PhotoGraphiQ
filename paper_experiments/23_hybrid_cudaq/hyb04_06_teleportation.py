"""HYB-04 (feed-forward teleportation) and HYB-06 (noisy teleportation with loss) vs Gaussian.

Pattern: one CZ teleportation step, resource squeezing r = 0.6, homodyne angle 1.1, correction
q -> -m / sin(theta), coherent input (0.4, -0.2). The Gaussian backend's conditional state is
exact, so it is the reference for every trajectory.

HYB-04. (a) Fixed outcome 0.3, dims 12..28: conditional mean/covariance and outcome density.
(b) 200 sampled trajectories (seed 4): each shot's conditional state vs the Gaussian backend
postselected on the same sampled outcome; KS test of the outcomes against the exact Gaussian
marginal of the measured quadrature. Acceptance: at dim 28 moment errors < 1e-4; KS below the
1% critical value; per-shot moment errors < 1e-3.

Recorded deviation: the first run declared the per-shot criterion at dims 20 and FAILED (max
1.1e-2, at a 3.7-sigma outcome m = -5.13; 99th percentile 3.3e-3): conditioning on a rare outcome
divides the absolute truncation error of the high levels by p(m), while every boundary population
stayed < 4e-5. The executor now reports ``conditional_boundary`` (share of p(m) carried by the
measured mode's top two levels). The study keeps dims 20 (expected to fail the per-shot bound,
reported) and adds dims 32 (acceptance), and checks that the diagnostic ranks the bad shots
(Spearman correlation with the per-shot error >= 0.7 at dims 20).

HYB-06. Losses eta in {0.9, 0.7, 0.5} on both modes after the CZ (pure loss) and one thermal
case (eta 0.8, n_th 0.1), mixed-state execution, outcome 0.3, dims 12..24. Acceptance: at dim 24
mean error < 1e-4, covariance error < 1e-3, density error < 1e-6.
"""

import numpy as np
from hyb_support import COLORS, Timer, pg, plt, save, simulate, wire
from scipy.stats import norm, spearmanr

from photographiq.hybrid.executor import run, run_shots

INPUT = {0: pg.GaussianInput((0.4, -0.2))}
THETA = 1.1


def compare(p, x, d):
    g = simulate(p, backend="gaussian", inputs=INPUT, measurement_outcomes={0: x})
    with Timer() as t:
        h = run(p, dims=d, inputs=INPUT, measurement_outcomes={0: x})
    m, c = h.state.moments()
    return {
        "dim": d,
        "outcome": float(x),
        "mean_error": float(np.abs(m - g.state.mean).max()),
        "cov_error": float(np.abs(c - g.state.covariance).max()),
        "density_error": abs(h.densities[0] - float(np.exp(g.log_likelihood))),
        "representation": h.state.representation,
        "lost_weight": 1 - h.retained_weight,
        "seconds": t.seconds,
        "boundary_population": h.truncation["max_boundary_population"],
        "conditional_boundary": h.truncation["max_conditional_boundary"],
    }


def main():
    p = wire(THETA, 0.6)
    fixed = [compare(p, 0.3, d) | {"study": "fixed"} for d in (12, 16, 20, 24, 28)]
    with Timer() as t:
        shots = run_shots(p, 200, seed=4, dims=20, inputs=INPUT)
    per_shot = [
        compare(p, s.records[0], d) | {"study": f"sampled-dims{d}"} for d in (20, 32) for s in shots
    ]
    # exact marginal of x_theta on mode 0 just before the measurement
    pre = (
        pg.Pattern(inputs=(0,))
        .extend([pg.Prepare(1, 0.6), pg.Entangle(0, 1)])
        .append(pg.Output((0, 1)))
        .validate()
    )
    s0 = simulate(pre, backend="gaussian", inputs=INPUT).state
    h = np.array([np.cos(THETA), np.sin(THETA), 0, 0])
    mu, sd = h @ s0.mean, np.sqrt(h @ s0.covariance @ h)
    xs = np.sort([s.records[0] for s in shots])
    F = norm.cdf(xs, mu, sd)
    n = len(xs)
    ks = float(max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n)))
    err = {
        d: np.array([max(r["mean_error"], r["cov_error"]) for r in per_shot if r["dim"] == d])
        for d in (20, 32)
    }
    diag20 = np.array([r["conditional_boundary"] for r in per_shot if r["dim"] == 20])
    rho = float(spearmanr(err[20], diag20).statistic)
    checks04 = {
        "fixed_dim28": fixed[-1]["mean_error"] < 1e-4 and fixed[-1]["cov_error"] < 1e-4,
        "per_shot_dims32": float(err[32].max()) < 1e-3,
        "ks": ks < 1.63 / np.sqrt(n),
        "diagnostic_ranks_errors_dims20": rho >= 0.7,
    }
    first_run = {
        "per_shot_dims20": float(err[20].max()) < 1e-3,
        "max_error_dims20": float(err[20].max()),
        "p99_error_dims20": float(np.percentile(err[20], 99)),
        "spearman_dims20": rho,
    }
    st04 = "pass" if all(checks04.values()) else "fail"

    noisy = []
    cases = [(eta, 0.0) for eta in (0.9, 0.7, 0.5)] + [(0.8, 0.1)]
    for eta, nth in cases:
        q = pg.Pattern(inputs=(0,)).extend(
            [pg.Prepare(1, 0.6), pg.Entangle(0, 1), pg.Loss(0, eta, nth), pg.Loss(1, eta, nth)]
        )
        q.measure(0, pg.Homodyne(THETA))
        q.displace(1, q=-pg.Outcome(0) / np.sin(THETA))
        q = q.append(pg.Output((1,))).validate()
        for d in (12, 16, 20, 24):
            noisy.append(compare(q, 0.3, d) | {"eta": eta, "n_th": nth})
            print(
                "HYB06",
                eta,
                nth,
                d,
                {k: noisy[-1][k] for k in ("mean_error", "cov_error", "density_error")},
                flush=True,
            )
    final = [r for r in noisy if r["dim"] == 24]
    checks06 = {
        f"eta={r['eta']},nth={r['n_th']}": r["mean_error"] < 1e-4
        and r["cov_error"] < 1e-3
        and r["density_error"] < 1e-6
        and r["representation"] == "mixed-fock"
        for r in final
    }
    st06 = "pass" if all(checks06.values()) else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 3, figsize=(7.0, 2.4))
    axes[0].semilogy(
        [r["dim"] for r in fixed],
        [r["cov_error"] for r in fixed],
        "o-",
        color=COLORS["hybrid"],
        label="covariance",
    )
    axes[0].semilogy(
        [r["dim"] for r in fixed],
        [r["mean_error"] for r in fixed],
        "s-",
        color=COLORS["native"],
        label="mean",
    )
    axes[0].set(
        xlabel="Per-mode dimension",
        ylabel="Max error vs Gaussian",
        title="Teleportation, fixed outcome",
    )
    axes[0].legend(fontsize=6)
    axes[1].hist(
        xs,
        bins=25,
        density=True,
        color=COLORS["reference"],
        alpha=0.6,
        label="200 sampled outcomes",
    )
    grid = np.linspace(mu - 4 * sd, mu + 4 * sd, 200)
    axes[1].plot(
        grid,
        norm.pdf(grid, mu, sd),
        color=COLORS["gaussian"],
        label=f"exact marginal (KS {ks:.3f})",
    )
    axes[1].set(xlabel="Outcome", ylabel="Density", title="Sampled trajectories")
    axes[1].legend(fontsize=5.5)
    for (eta, nth), colour in zip(
        cases,
        [COLORS["hybrid"], COLORS["native"], COLORS["gaussian"], COLORS["accent"]],
        strict=True,
    ):
        sel = [r for r in noisy if r["eta"] == eta and r["n_th"] == nth]
        axes[2].semilogy(
            [r["dim"] for r in sel],
            [r["cov_error"] for r in sel],
            "o-",
            color=colour,
            label=f"eta {eta}" + (f", n_th {nth}" if nth else ""),
        )
    axes[2].set(
        xlabel="Per-mode dimension",
        ylabel="Max covariance error",
        title="Lossy teleportation (mixed)",
    )
    axes[2].legend(fontsize=5.5)
    config = {
        "theta": THETA,
        "r": 0.6,
        "input": [0.4, -0.2],
        "fixed_outcome": 0.3,
        "shots": 200,
        "seed": 4,
        "sampled_dims": 20,
        "sampling_seconds": t.seconds,
    }
    save(
        fixed + per_shot,
        "HYB04_feedforward_teleportation",
        configuration=config,
        validation={
            "status": st04,
            "checks": checks04,
            "ks": ks,
            "marginal": [float(mu), float(sd)],
            "per_shot_dims20_reported": first_run,
        },
        figure=fig,
        caption="Hybrid teleportation vs the exact Gaussian backend: cutoff sweep, sampled trajectories, loss.",
    )
    save(
        noisy,
        "HYB06_noisy_teleportation",
        configuration=config | {"cases": cases},
        validation={"status": st06, "checks": checks06},
    )
    print(f"HYB04 {st04} {checks04} ks={ks:.3f}\nHYB06 {st06} {checks06}")


if __name__ == "__main__":
    main()
