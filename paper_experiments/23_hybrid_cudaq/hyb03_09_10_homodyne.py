"""HYB-03 (homodyne instrument), HYB-09 (finite-resolution bins), HYB-10 (sampling).

HYB-03. Outcome densities of the Fock homodyne instrument vs closed forms (vacuum, coherent at
three angles, squeezed variance vs angle, number states via scipy Hermite polynomials), vs the
Gaussian backend's conditional state (two-mode entangled, pure and mixed, outcome grid), and vs
native-fock postselection on a non-Gaussian two-mode state (cubic phase + beam splitter).
Acceptance: closed forms to 1e-9; Gaussian-conditional moments to 1e-4 and densities to 1e-6
(dim 24); native-fock densities to 1e-5 and conditional-state infidelity < 1e-6 (dim 30 vs cutoff 40).

HYB-09. Bin convention: fine measurement then coarse graining. As the resolution w -> 0,
P(bin)/w -> p(centre) and the binned conditional state -> the ideal one, both O(w^2).
Acceptance: errors decrease monotonically and the fitted log-log slope is 2 +- 0.2.

HYB-10. 5000 sampled outcomes (seed 10) of a cubic-phase (non-Gaussian) state: KS statistic vs
the exact CDF below the 1% critical value 1.63/sqrt(N); identical seeds give identical records;
reported tail mass < 1e-10.
"""

import numpy as np
from hyb_support import COLORS, infidelity, pg, plt, save, simulate, trace_distance, wire
from scipy.linalg import expm
from scipy.special import eval_hermite, factorial

from photographiq import cudaq_dynamics as cq
from photographiq.hybrid import homodyne as hd
from photographiq.hybrid.executor import run
from photographiq.hybrid.states import FockRep, fock_input, gaussian_to_fock

X = np.linspace(-10, 10, 2001)


def closed_forms():
    rows = []
    vac = gaussian_to_fock([0, 0], np.eye(2), (30,))
    g = np.exp(-(X**2) / 2) / np.sqrt(2 * np.pi)
    rows.append(
        {"case": "vacuum", "max_error": float(np.abs(hd.density(vac, 0, 0.7, X) - g).max())}
    )
    coh = gaussian_to_fock([0.8, -0.4], np.eye(2), (30,))
    for th in (0.0, 0.7, 2.0):
        mu = 0.8 * np.cos(th) - 0.4 * np.sin(th)
        want = np.exp(-((X - mu) ** 2) / 2) / np.sqrt(2 * np.pi)
        rows.append(
            {
                "case": f"coherent theta={th}",
                "max_error": float(np.abs(hd.density(coh, 0, th, X) - want).max()),
            }
        )
    r = 0.4
    sq = gaussian_to_fock([0, 0], np.diag([np.exp(-2 * r), np.exp(2 * r)]), (40,))
    for th in (0.0, 0.5, 1.2):
        var = np.exp(-2 * r) * np.cos(th) ** 2 + np.exp(2 * r) * np.sin(th) ** 2
        want = np.exp(-(X**2) / (2 * var)) / np.sqrt(2 * np.pi * var)
        rows.append(
            {
                "case": f"squeezed r=0.4 theta={th}",
                "max_error": float(np.abs(hd.density(sq, 0, th, X) - want).max()),
            }
        )
    for n in (1, 2, 3, 4):
        psi = (
            (2 * np.pi) ** -0.25
            * np.exp(-(X**2) / 4)
            * eval_hermite(n, X / np.sqrt(2))
            / np.sqrt(2.0**n * factorial(n))
        )
        rows.append(
            {
                "case": f"number n={n}",
                "max_error": float(
                    np.abs(
                        hd.density(fock_input(np.eye(n + 1)[n], 0, 12), 0, 1.1, X) - psi**2
                    ).max()
                ),
            }
        )
    return rows


def gaussian_conditional(mixed):
    p = wire(1.1, 0.6, losses=((1, 0.85),) if mixed else ())
    rows = []
    for x in np.linspace(-1.5, 1.5, 7):
        g = simulate(
            p,
            backend="gaussian",
            inputs={0: pg.GaussianInput((0.4, -0.2))},
            measurement_outcomes={0: x},
        )
        h = run(p, dims=24, inputs={0: pg.GaussianInput((0.4, -0.2))}, measurement_outcomes={0: x})
        m, c = h.state.moments()
        rows.append(
            {
                "case": f"gaussian-conditional {'mixed' if mixed else 'pure'}",
                "outcome": float(x),
                "mean_error": float(np.abs(m - g.state.mean).max()),
                "cov_error": float(np.abs(c - g.state.covariance).max()),
                "density_error": abs(h.densities[0] - float(np.exp(g.log_likelihood))),
            }
        )
    return rows


def native_conditional():
    p = pg.Pattern(inputs=(0,)).extend(
        [pg.Prepare(1, 0.3), pg.CubicPhase(0, 0.2), pg.BeamSplitter(0, 1, 0.5)]
    )
    p.measure(0, pg.Homodyne(0.6))
    p = p.append(pg.Output((1,))).validate()
    inputs = {0: pg.GaussianInput((0.5, 0.2))}
    rows = []
    for x in np.linspace(-2, 2, 5):
        n = simulate(
            p, backend="native-fock", cutoff=40, inputs=inputs, measurement_outcomes={0: x}
        )
        h = run(p, dims=30, inputs=inputs, measurement_outcomes={0: x})
        rows.append(
            {
                "case": "native-fock non-Gaussian",
                "outcome": float(x),
                "density_error": abs(h.densities[0] - float(np.exp(n.log_likelihood))),
                "infidelity": infidelity(h.state.data, n.state.state_vector),
            }
        )
    return rows


def binned():
    rep = gaussian_to_fock([0.3, 0.0, 0.0, 0.0], np.eye(4), (12, 12))
    rep = FockRep(
        expm(-1j * cq.to_matrix(cq.controlled_z(0, 1, 0.8), {0: 12, 1: 12})) @ rep.data,
        rep.modes,
        rep.dims,
    )
    ideal = hd.measure(rep, 0, 0.4, outcome=0.2)
    rows = []
    for w in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        b = hd.measure(rep, 0, 0.4, outcome=0.2, resolution=w)
        rows.append(
            {
                "resolution": w,
                "probability_error": abs(b.density / w - ideal.density),
                "state_error": trace_distance(b.state.data, ideal.state.density_matrix()),
            }
        )
    return rows


def cubic_state(dim=30):
    sq = gaussian_to_fock([0.3, 0.0], np.diag([np.exp(-0.6), np.exp(0.6)]), (dim,))
    return FockRep(
        expm(-1j * cq.to_matrix(cq.cubic_phase(0, 0.25), {0: dim})) @ sq.data, (0,), (dim,)
    )


def sampling(N=5000):
    rep = cubic_state()
    rng = np.random.default_rng(10)
    outs = [hd.measure(rep, 0, 0.3, rng=rng, grid=4001) for _ in range(N)]
    xs = np.sort([o.outcome for o in outs])
    lo, hi = hd.window(rep, 0, 0.3)
    grid = np.linspace(lo, hi, 20001)
    p = hd.density(rep, 0, 0.3, grid)
    cdf = np.concatenate([[0], np.cumsum((p[1:] + p[:-1]) / 2 * np.diff(grid))])
    F = np.interp(xs, grid, cdf)
    ks = float(max(np.max(np.arange(1, N + 1) / N - F), np.max(F - np.arange(N) / N)))
    again = [hd.measure(rep, 0, 0.3, rng=np.random.default_rng(10)).outcome for _ in range(1)]
    return {
        "samples": N,
        "ks": ks,
        "ks_critical_1pct": 1.63 / np.sqrt(N),
        "max_tail_mass": max(o.tail_mass for o in outs),
        "reproducible": again[0] == outs[0].outcome,
        "outcomes": xs.tolist(),
    }, (grid, p)


def main():
    cf = closed_forms()
    gc = gaussian_conditional(False) + gaussian_conditional(True)
    nc = native_conditional()
    checks03 = {
        "closed_forms": max(r["max_error"] for r in cf) < 1e-9,
        "gaussian_moments": max(max(r["mean_error"], r["cov_error"]) for r in gc) < 1e-4,
        "gaussian_density": max(r["density_error"] for r in gc) < 1e-6,
        "native_density": max(r["density_error"] for r in nc) < 1e-5,
        "native_state": max(r["infidelity"] for r in nc) < 1e-6,
    }
    for r in cf + gc + nc:
        print(r, flush=True)
    st03 = "pass" if all(checks03.values()) else "fail"
    bn = binned()
    w = np.log([r["resolution"] for r in bn])
    slopes = {
        k: float(np.polyfit(w, np.log([r[k] for r in bn]), 1)[0])
        for k in ("probability_error", "state_error")
    }
    checks09 = {
        "monotone": all(
            bn[i + 1][k] < bn[i][k]
            for i in range(len(bn) - 1)
            for k in ("probability_error", "state_error")
        ),
        "second_order": all(abs(s - 2) < 0.2 for s in slopes.values()),
    }
    st09 = "pass" if all(checks09.values()) else "fail"
    sm, (grid, dens) = sampling()
    checks10 = {
        "ks": sm["ks"] < sm["ks_critical_1pct"],
        "reproducible": sm["reproducible"],
        "tail": sm["max_tail_mass"] < 1e-10,
    }
    st10 = "pass" if all(checks10.values()) else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 3, figsize=(7.0, 2.4))
    axes[0].semilogy(
        range(len(cf)),
        [max(r["max_error"], 1e-17) for r in cf],
        "o",
        color=COLORS["hybrid"],
        label="closed forms",
    )
    axes[0].semilogy(
        range(len(cf), len(cf) + len(gc)),
        [max(r["density_error"], 1e-17) for r in gc],
        "s",
        color=COLORS["gaussian"],
        label="Gaussian backend",
    )
    axes[0].semilogy(
        range(len(cf) + len(gc), len(cf) + len(gc) + len(nc)),
        [max(r["density_error"], 1e-17) for r in nc],
        "^",
        color=COLORS["native"],
        label="native-fock",
    )
    axes[0].set(xlabel="Case", ylabel="Max density error", title="Homodyne densities")
    axes[0].legend(fontsize=5.5)
    res = [r["resolution"] for r in bn]
    axes[1].loglog(
        res,
        [r["probability_error"] for r in bn],
        "o-",
        color=COLORS["hybrid"],
        label=f"P(bin)/w (slope {slopes['probability_error']:.2f})",
    )
    axes[1].loglog(
        res,
        [r["state_error"] for r in bn],
        "s-",
        color=COLORS["native"],
        label=f"state (slope {slopes['state_error']:.2f})",
    )
    axes[1].set(xlabel="Bin width w", ylabel="Error vs ideal homodyne", title="Finite resolution")
    axes[1].legend(fontsize=5.5)
    axes[2].hist(
        sm["outcomes"],
        bins=60,
        density=True,
        color=COLORS["reference"],
        alpha=0.6,
        label=f"{sm['samples']} samples",
    )
    axes[2].plot(grid, dens, color=COLORS["hybrid"], label=f"exact (KS {sm['ks']:.3f})")
    axes[2].set(
        xlabel="Outcome x", ylabel="Density", title="Sampling a cubic-phase state", xlim=(-6, 6)
    )
    axes[2].legend(fontsize=5.5)
    save(
        cf + gc + nc,
        "HYB03_homodyne_instrument",
        configuration={
            "dims": {"closed": "12-40", "gaussian": 24, "native_hybrid": 30, "native_cutoff": 40}
        },
        validation={"status": st03, "checks": checks03},
        figure=fig,
        caption="Fock homodyne instrument: closed forms, Gaussian conditional states, native-fock, bins and sampling.",
    )
    save(
        bn,
        "HYB09_binned_homodyne",
        configuration={
            "convention": "fine measurement then coarse graining, 64-point midpoint rule",
            "state": "CZ(0.8)-entangled coherent x vacuum, dim 12, theta 0.4, centre 0.2",
        },
        validation={"status": st09, "checks": checks09, "slopes": slopes},
    )
    save(
        [{k: v for k, v in sm.items() if k != "outcomes"}],
        "HYB10_homodyne_sampling",
        configuration={
            "state": "CP(0.25) S(0.3) displaced, dim 30",
            "theta": 0.3,
            "seed": 10,
            "grid": 4001,
        },
        validation={"status": st10, "checks": checks10},
    )
    print(
        f"HYB03 {st03} {checks03}\nHYB09 {st09} {checks09} {slopes}\nHYB10 {st10} {checks10} ks={sm['ks']:.4f}"
    )


if __name__ == "__main__":
    main()
