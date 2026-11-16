"""HYB-02: Gaussian-to-Fock bridge accuracy vs per-mode dimension.

Hypothesis: gaussian_to_fock reproduces the mean and covariance of pure and mixed one- and
two-mode Gaussian states, with errors and lost weight (1 - retained_weight) decreasing with the
local dimension, and mixed outputs positive semidefinite.

Reference: the Gaussian moments themselves (exact); the two-mode states are produced by the
Gaussian backend. Acceptance: at the largest dimension every max moment error < 1e-4,
1 - retained_weight < 1e-6, min eigenvalue > -1e-10.
"""

import numpy as np
from hyb_support import COLORS, Timer, pg, plt, save, simulate

from photographiq.hybrid.states import gaussian_to_fock


def states():
    sq = np.diag([np.exp(-1.0), np.exp(1.0)])
    rot = np.array([[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]])
    two = pg.Pattern(inputs=(0, 1)).extend(
        [pg.Squeeze(0, 0.4), pg.Entangle(0, 1, 0.8), pg.Rotate(1, 0.3)]
    )
    two = two.append(pg.Output((0, 1))).validate()
    g = simulate(two, backend="gaussian", inputs={0: pg.GaussianInput((0.3, -0.2))}).state
    noisy = g.covariance + 0.6 * np.eye(4)  # adds thermal noise: mixed, still physical
    return {
        "1-mode squeezed coherent (pure, r=0.5)": (
            np.array([0.8, -0.4]),
            rot @ sq @ rot.T,
            (8, 12, 16, 20, 24, 28),
        ),
        "1-mode displaced squeezed thermal (mixed)": (
            np.array([0.4, -0.3]),
            np.array([[1.8, 0.3], [0.3, 1.1]]),
            (8, 12, 16, 20, 24, 28),
        ),
        "2-mode squeezed CZ (pure)": (g.mean, g.covariance, (6, 8, 12, 16, 20, 24)),
        "2-mode squeezed CZ + thermal noise (mixed)": (g.mean, noisy, (6, 8, 12, 16, 20, 24)),
    }


def main():
    rows = []
    for label, (mean, cov, dims) in states().items():
        n = len(mean) // 2
        for d in dims:
            with Timer() as t:
                rep = gaussian_to_fock(mean, cov, (d,) * n)
            m, c = rep.moments()
            psd = (
                float(np.linalg.eigvalsh(rep.density_matrix()).min()) if rep.data.ndim == 2 else 0.0
            )
            rows.append(
                {
                    "state": label,
                    "dim": d,
                    "representation": rep.representation,
                    "mean_error": float(np.abs(m - mean).max()),
                    "cov_error": float(np.abs(c - cov).max()),
                    "lost_weight": 1 - rep.retained_weight,
                    "min_eigenvalue": psd,
                    "seconds": t.seconds,
                }
            )
            print(
                label,
                d,
                f"{rows[-1]['mean_error']:.1e} {rows[-1]['cov_error']:.1e} {rows[-1]['lost_weight']:.1e}",
                flush=True,
            )
    last = {}
    for r in rows:
        last[r["state"]] = r
    checks = {
        k: v["mean_error"] < 1e-4
        and v["cov_error"] < 1e-4
        and v["lost_weight"] < 1e-6
        and v["min_eigenvalue"] > -1e-10
        for k, v in last.items()
    }
    status = "pass" if all(checks.values()) else "fail"
    pyplot = plt()
    fig, axes = pyplot.subplots(1, 2, figsize=(7.0, 2.6))
    palette = [COLORS["hybrid"], COLORS["native"], COLORS["gaussian"], COLORS["piquasso"]]
    for colour, label in zip(palette, last, strict=True):
        sel = [r for r in rows if r["state"] == label]
        dims = [r["dim"] for r in sel]
        axes[0].semilogy(
            dims, [max(r["cov_error"], 1e-16) for r in sel], "o-", color=colour, label=label
        )
        axes[1].semilogy(dims, [max(r["lost_weight"], 1e-17) for r in sel], "o-", color=colour)
    axes[0].set(xlabel="Per-mode dimension", ylabel="Max covariance error", title="Bridge moments")
    axes[1].set(
        xlabel="Per-mode dimension", ylabel="1 - retained weight", title="Truncated probability"
    )
    axes[0].legend(fontsize=5.5)
    save(
        rows,
        "HYB02_gaussian_to_fock_bridge",
        configuration={"states": list(last), "pad": "max(dims)"},
        validation={"status": status, "checks": checks},
        figure=fig,
        caption="Gaussian-to-Fock bridge: moment errors and truncated weight vs per-mode dimension.",
    )
    print(f"HYB02 status={status}")


if __name__ == "__main__":
    main()
