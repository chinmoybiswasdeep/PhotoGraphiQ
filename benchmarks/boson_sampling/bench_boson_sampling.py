"""Standard (Fock-state) boson sampling benchmark: PhotoGraphiQ vs Piquasso 8.

Four task families, each compared only between implementations that compute the SAME thing:

* ``permanent``: permanent of one ``n x n`` complex matrix (submatrix of a Haar unitary of
  size ``2n``). PhotoGraphiQ native (C++/OpenMP Ryser), PhotoGraphiQ Python (NumPy Ryser,
  small ``n``), ``piquasso._math.permanent.permanent`` (C++, Eq. (8) of arXiv:2309.07027).
  Reference: the permutation sum for ``n <= 8``, else PhotoGraphiQ Python where run, else
  Piquasso.
* ``probability``: one output probability ``P(t|s)`` of ``n`` photons in the first ``n`` of
  ``m`` modes through a Haar unitary, for a collision-free and a bunched output ``t``.
  ``api`` level: ``photographiq.sampling.output_probability`` vs Piquasso
  ``PassiveSimulator`` state ``get_particle_detection_probability`` (program execution is
  timed separately as setup). ``kernel`` level: the rectangular permanent with row/column
  multiplicities, ``permanent(U, rows=t, columns=s)`` vs Piquasso ``permanent(U, t, s)``.
* ``distribution``: all ``C(m+n-1, n)`` output probabilities. PhotoGraphiQ
  ``output_distribution`` (one permanent per pattern) vs Piquasso ``fock_probabilities_map``
  with ``cutoff = n + 1`` (Piquasso's basis also holds every pattern with fewer photons,
  which are exactly zero here; each Piquasso call re-executes the program, ~0.3 ms).
* ``sampling``: ``shots`` exact samples. PhotoGraphiQ Clifford & Clifford algorithm B
  (collision-free input) vs Piquasso ``ParticleNumberMeasurement`` (generalised Clifford &
  Clifford of Brod & Oszmaniec with Laplace-expansion permanents). Both sample the exact
  ideal output distribution of the same input/unitary; Piquasso's time includes program
  execution. Validation: photon-number conservation, and total-variation distance to the
  exact distribution where that distribution is enumerated.

Run ``python benchmarks/boson_sampling/bench_boson_sampling.py`` (default, ~1-2 min),
``--smoke`` (seconds) or ``--full`` (larger sizes, much longer). Results:
``benchmarks/boson_sampling/results/boson_sampling_<config>_<stamp>.json`` and figures.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_common import (  # noqa: E402
    environment,
    new_axes,
    peak_python_bytes,
    rel_err,
    save_figure,
    series_style,
    stamp,
    time_call,
    write_json,
)

from photographiq import native  # noqa: E402
from photographiq.sampling import (  # noqa: E402
    haar_unitary,
    output_distribution,
    output_probability,
    permanent,
    permanent_by_definition,
    sample,
)

RESULTS = Path(__file__).resolve().parent / "results"

CONFIGS = {
    "smoke": {
        "perm_sizes": [2, 4, 6],
        "perm_python_max": 6,
        "prob_workloads": [(4, 2), (8, 3)],
        "prob_python_max": 3,
        "dist_workloads": [(4, 2)],
        "sample_workloads": [(4, 2)],
        "shots": 20,
        "repeats": 3,
        "budget_s": 0.5,
    },
    "notebook": {
        "perm_sizes": list(range(2, 17, 2)),
        "perm_python_max": 12,
        "prob_workloads": [(4, 2), (8, 3), (12, 4), (16, 6), (24, 10)],
        "prob_python_max": 6,
        "dist_workloads": [(4, 2), (8, 3), (12, 4)],
        "sample_workloads": [(4, 2), (8, 3), (12, 4)],
        "shots": 100,
        "repeats": 5,
        "budget_s": 1.0,
    },
    "default": {
        "perm_sizes": list(range(2, 21, 2)),
        "perm_python_max": 14,
        "prob_workloads": [(4, 2), (8, 3), (12, 4), (16, 6), (20, 8), (24, 10), (32, 14), (40, 18)],
        "prob_python_max": 10,
        "dist_workloads": [(4, 2), (8, 3), (12, 4), (16, 6)],
        "sample_workloads": [(4, 2), (8, 3), (12, 4), (16, 6)],
        "shots": 200,
        "repeats": 7,
        "budget_s": 4.0,
    },
    "full": {
        "perm_sizes": list(range(2, 27, 2)),
        "perm_python_max": 18,
        "prob_workloads": [
            (4, 2),
            (8, 3),
            (12, 4),
            (16, 6),
            (20, 8),
            (24, 10),
            (32, 14),
            (40, 18),
            (48, 22),
            (60, 25),
        ],
        "prob_python_max": 14,
        "dist_workloads": [(4, 2), (8, 3), (12, 4), (16, 6), (20, 7)],
        "sample_workloads": [(4, 2), (8, 3), (12, 4), (16, 6), (20, 8), (24, 10)],
        "shots": 1000,
        "repeats": 9,
        "budget_s": 20.0,
    },
}


def _piquasso():
    try:
        import piquasso as pq
        from piquasso._math.permanent import permanent as pq_permanent
    except ImportError:
        return None, None
    return pq, pq_permanent


def _native() -> bool:
    return "sampling" in native.components()


def _row(task: str, impl: str, timing: dict | None, **fields) -> dict:
    return {"task": task, "implementation": impl, **fields, **(timing or {})}


# --------------------------------------------------------------------------- permanent


def bench_permanent(cfg: dict, seed: int) -> list[dict]:
    pq, pq_perm = _piquasso()
    rows = []
    for n in cfg["perm_sizes"]:
        m = haar_unitary(2 * n, rng=seed + n)[:n, :n]
        ones = np.ones(n, dtype=np.int32)
        impls = {}
        if _native():
            impls["photographiq_native"] = lambda: permanent(m, backend="native")
        if n <= cfg["perm_python_max"]:
            impls["photographiq_python"] = lambda: permanent(m, backend="python")
        if pq is not None:
            impls["piquasso"] = lambda: complex(pq_perm(m, ones, ones))
        values = {k: complex(f()) for k, f in impls.items()}
        if n <= 8:
            ref_name, ref = "definition", permanent_by_definition(m)
        elif "photographiq_python" in values:
            ref_name, ref = "photographiq_python", values["photographiq_python"]
        else:
            ref_name, ref = ("piquasso", values.get("piquasso")) if pq else (None, None)
        for impl, f in impls.items():
            rows.append(
                _row(
                    "permanent",
                    impl,
                    time_call(f, repeats=cfg["repeats"], budget_s=cfg["budget_s"]),
                    n=n,
                    value=values[impl],
                    reference=ref_name,
                    rel_error=None
                    if ref is None or impl == ref_name
                    else rel_err(values[impl], ref),
                    peak_python_bytes=peak_python_bytes(f),
                )
            )
    return rows


# ------------------------------------------------------------------------- probability


def _workload(m: int, n: int, seed: int):
    u = haar_unitary(m, rng=seed + 1000 * m + n)
    s = np.array([1] * n + [0] * (m - n))
    rng = np.random.default_rng(seed + m + n)
    free = np.zeros(m, dtype=int)
    free[rng.choice(m, n, replace=False)] = 1
    outputs = {"collision_free": free}
    if n >= 2:
        bunched = np.zeros(m, dtype=int)
        modes = rng.choice(m, n - 1, replace=False)
        bunched[modes] = 1
        bunched[modes[0]] = 2
        outputs["bunched"] = bunched
    return u, s, outputs


def _pq_passive_state(pq, u, s, cutoff=None, measure=False, seed=0):
    with pq.Program() as program:
        pq.Q() | pq.NumberState(list(map(int, s)))
        pq.Q() | pq.Interferometer(u)
        if measure:
            pq.Q(all) | pq.ParticleNumberMeasurement()
    config = pq.Config(hbar=2.0, cutoff=cutoff or int(sum(s)) + 1, seed_sequence=seed)
    return program, pq.PassiveSimulator(d=len(u), config=config)


def bench_probability(cfg: dict, seed: int) -> list[dict]:
    pq, pq_perm = _piquasso()
    rows = []
    for m, n in cfg["prob_workloads"]:
        u, s, outputs = _workload(m, n, seed)
        state = None
        if pq is not None:
            program, sim = _pq_passive_state(pq, u, s)
            setup = time_call(
                lambda: sim.execute(program), repeats=cfg["repeats"], budget_s=cfg["budget_s"]
            )
            rows.append(
                _row("probability_setup", "piquasso_execute_program", setup, modes=m, photons=n)
            )
            state = sim.execute(program).state
        for kind, t in outputs.items():
            s32, t32 = s.astype(np.int32), t.astype(np.int32)
            api, kernel = {}, {}
            if _native():
                api["photographiq_native"] = lambda: output_probability(u, s, t, backend="native")
                kernel["photographiq_native"] = lambda: permanent(
                    u, rows=t, columns=s, backend="native"
                )
            if n <= cfg["prob_python_max"]:
                api["photographiq_python"] = lambda: output_probability(u, s, t, backend="python")
                kernel["photographiq_python"] = lambda: permanent(
                    u, rows=t, columns=s, backend="python"
                )
            if state is not None:
                api["piquasso"] = lambda: float(state.get_particle_detection_probability(t))
                kernel["piquasso"] = lambda: complex(pq_perm(u, t32, s32))
            norm = math.prod(math.factorial(int(x)) for x in np.r_[s, t])
            for level, impls in (("api", api), ("kernel", kernel)):
                values = {k: f() for k, f in impls.items()}
                probs = {
                    k: float(v) if level == "api" else abs(v) ** 2 / norm for k, v in values.items()
                }
                ref_name = "photographiq_python" if "photographiq_python" in probs else "piquasso"
                ref = probs.get(ref_name)
                for impl, f in impls.items():
                    rows.append(
                        _row(
                            f"probability_{level}",
                            impl,
                            time_call(f, repeats=cfg["repeats"], budget_s=cfg["budget_s"]),
                            modes=m,
                            photons=n,
                            output_kind=kind,
                            output=t.tolist(),
                            probability=probs[impl],
                            reference=ref_name if ref is not None else None,
                            rel_error=None
                            if ref is None or impl == ref_name
                            else rel_err(probs[impl], ref),
                            peak_python_bytes=peak_python_bytes(f),
                        )
                    )
    return rows


# ------------------------------------------------------------------------ distribution


def bench_distribution(cfg: dict, seed: int) -> tuple[list[dict], dict]:
    pq, _ = _piquasso()
    rows, exact = [], {}
    for m, n in cfg["dist_workloads"]:
        u, s, _ = _workload(m, n, seed)
        backend = "native" if _native() else "python"
        dist = output_distribution(u, s, backend=backend)
        exact[(m, n)] = (u, s, dist)
        total = math.fsum(dist.values())
        timing = time_call(
            lambda: output_distribution(u, s, backend=backend),
            repeats=3,
            warmup=1,
            budget_s=cfg["budget_s"] * 3,
        )
        rows.append(
            _row(
                "distribution",
                f"photographiq_{backend}",
                timing,
                modes=m,
                photons=n,
                patterns=len(dist),
                normalization_error=abs(total - 1.0),
            )
        )
        if pq is None:
            continue
        program, sim = _pq_passive_state(pq, u, s)

        def pq_dist():
            return sim.execute(program).state.fock_probabilities_map

        ref = pq_dist()
        sector = {k: v for k, v in ref.items() if sum(k) == n}
        other = math.fsum(abs(v) for k, v in ref.items() if sum(k) != n)
        rows.append(
            _row(
                "distribution",
                "piquasso",
                time_call(pq_dist, repeats=3, warmup=1, budget_s=cfg["budget_s"] * 3),
                modes=m,
                photons=n,
                patterns=len(sector),
                piquasso_basis_size=len(ref),
                normalization_error=abs(math.fsum(sector.values()) - 1.0),
                mass_outside_n_photon_sector=other,
                max_abs_diff_vs_photographiq=max(abs(dist[k] - sector.get(k, 0.0)) for k in dist),
                missing_patterns=len(set(dist) ^ set(sector)),
            )
        )
    return rows, exact


# ---------------------------------------------------------------------------- sampling


def _tvd(samples: np.ndarray, dist: dict) -> float:
    counts = Counter(map(tuple, samples))
    shots = len(samples)
    keys = set(dist) | set(counts)
    return 0.5 * math.fsum(abs(counts.get(k, 0) / shots - dist.get(k, 0.0)) for k in keys)


def bench_sampling(cfg: dict, seed: int, exact: dict) -> list[dict]:
    pq, _ = _piquasso()
    rows, shots = [], cfg["shots"]
    backend = "native" if _native() else "python"
    for m, n in cfg["sample_workloads"]:
        u, s, _ = _workload(m, n, seed)
        dist = exact.get((m, n), (None, None, None))[2]
        impls = {
            f"photographiq_{backend}_clifford": lambda: sample(
                u, s, shots, rng=seed, method="clifford", backend=backend
            )
        }
        if pq is not None:
            program, sim = _pq_passive_state(pq, u, s, measure=True, seed=seed)
            impls["piquasso_generalized_clifford"] = lambda: np.array(
                sim.execute(program, shots=shots).samples
            )
        for impl, f in impls.items():
            out = np.asarray(f())
            timing = time_call(
                f, repeats=min(cfg["repeats"], 5), warmup=1, budget_s=cfg["budget_s"]
            )
            rows.append(
                _row(
                    "sampling",
                    impl,
                    timing,
                    modes=m,
                    photons=n,
                    shots=shots,
                    shots_per_s=shots / timing["median_s"],
                    photon_number_conserved=bool((out.sum(axis=1) == n).all()),
                    tvd_vs_exact=_tvd(out, dist) if dist is not None else None,
                )
            )
    return rows


# ------------------------------------------------------------------------------- plots


def _series(rows, task, x, impl, **match):
    pts = [
        r
        for r in rows
        if r["task"] == task
        and r["implementation"] == impl
        and all(r.get(k) == v for k, v in match.items())
    ]
    return [r[x] for r in pts], pts


def plot(payload: dict, outdir: Path, save: bool = True) -> list:
    rows, figs = payload["results"], []
    impls = sorted({r["implementation"] for r in rows if r["task"] == "permanent"})
    order = {k: i for i, k in enumerate(["photographiq_native", "photographiq_python", "piquasso"])}

    fig, (ax1, ax2) = new_axes(2)
    for impl in impls:
        xs, pts = _series(rows, "permanent", "n", impl)
        st = series_style(order.get(impl, 3))
        ax1.errorbar(
            xs, [p["median_s"] for p in pts], yerr=[p["iqr_s"] / 2 for p in pts], label=impl, **st
        )
        err = [(x, p["rel_error"]) for x, p in zip(xs, pts) if p["rel_error"] is not None]
        if err:
            ax2.plot(*zip(*err), label=impl, **st)
    ax1.set(
        yscale="log",
        xlabel="matrix size n",
        ylabel="time per call [s] (median, IQR bar)",
        title="Permanent runtime",
    )
    ax2.set(
        yscale="log",
        xlabel="matrix size n",
        ylabel="relative error vs reference",
        title="Permanent agreement",
    )
    ax1.legend(frameon=False, fontsize=8)
    ax2.legend(frameon=False, fontsize=8)
    figs.append(("bs_permanent", fig))

    fig, axes = new_axes(2)
    for ax, level in zip(axes, ("api", "kernel")):
        for impl in ["photographiq_native", "photographiq_python", "piquasso"]:
            xs, pts = _series(
                rows, f"probability_{level}", "photons", impl, output_kind="collision_free"
            )
            if pts:
                ax.plot(xs, [p["median_s"] for p in pts], label=impl, **series_style(order[impl]))
        ax.set(
            yscale="log",
            xlabel="photons n (modes m in JSON)",
            ylabel="time per call [s]",
            title=f"Single BS probability ({level} level)",
        )
        ax.legend(frameon=False, fontsize=8)
    figs.append(("bs_probability_runtime", fig))

    fig, (ax1, ax2) = new_axes(2)
    for i, impl in enumerate(
        sorted({r["implementation"] for r in rows if r["task"] == "distribution"})
    ):
        xs, pts = _series(rows, "distribution", "patterns", impl)
        st = series_style(order.get(impl, i))
        ax1.plot(xs, [p["median_s"] for p in pts], label=impl, **st)
        ax2.plot(xs, [max(p["normalization_error"], 1e-17) for p in pts], label=impl, **st)
    ax1.set(
        xscale="log",
        yscale="log",
        xlabel="output patterns C(m+n-1, n)",
        ylabel="time [s]",
        title="Full output distribution",
    )
    ax2.set(
        xscale="log",
        yscale="log",
        xlabel="output patterns",
        ylabel="|sum p - 1| (floored 1e-17)",
        title="Normalization",
    )
    ax1.legend(frameon=False, fontsize=8)
    figs.append(("bs_distribution", fig))

    fig, (ax1, ax2) = new_axes(2)
    for i, impl in enumerate(
        sorted({r["implementation"] for r in rows if r["task"] == "sampling"})
    ):
        xs, pts = _series(rows, "sampling", "photons", impl)
        st = series_style(0 if impl.startswith("photographiq") else 2)
        ax1.plot(xs, [p["shots_per_s"] for p in pts], label=impl, **st)
        tv = [(x, p["tvd_vs_exact"]) for x, p in zip(xs, pts) if p["tvd_vs_exact"] is not None]
        if tv:
            ax2.plot(*zip(*tv), label=impl, **st)
    ax1.set(
        yscale="log",
        xlabel="photons n",
        ylabel="samples per second",
        title="Exact sampler throughput",
    )
    ax2.set(
        xlabel="photons n", ylabel="TVD (empirical vs exact)", title="Sampler check (finite shots)"
    )
    ax1.legend(frameon=False, fontsize=8)
    figs.append(("bs_sampling", fig))

    fig, (ax1, ax2) = new_axes(2)
    for impl in impls:
        xs, pts = _series(rows, "permanent", "n", impl)
        ax1.plot(
            xs,
            [p["peak_python_bytes"] / 1024 for p in pts],
            label=impl,
            **series_style(order.get(impl, 3)),
        )
    for impl in ["photographiq_native", "photographiq_python", "piquasso"]:
        xs, pts = _series(rows, "probability_api", "photons", impl, output_kind="collision_free")
        if pts:
            ax2.plot(
                xs,
                [p["peak_python_bytes"] / 1024 for p in pts],
                label=impl,
                **series_style(order[impl]),
            )
    for ax, title in ((ax1, "Permanent"), (ax2, "Single probability (api)")):
        ax.set(yscale="log", ylabel="tracemalloc peak [KiB]", title=f"{title}: Python-heap peak")
        ax.legend(frameon=False, fontsize=8)
    ax1.set_xlabel("matrix size n")
    ax2.set_xlabel("photons n")
    figs.append(("bs_memory", fig))

    if save:
        payload.setdefault("figures", [])
        for name, f in figs:
            payload["figures"] += save_figure(f, outdir / name)
    return figs


def run(
    config: str = "default", seed: int = 7, output: Path | None = None, figures: bool = True
) -> dict:
    cfg = CONFIGS[config]
    rows = bench_permanent(cfg, seed)
    rows += bench_probability(cfg, seed)
    dist_rows, exact = bench_distribution(cfg, seed)
    rows += dist_rows
    rows += bench_sampling(cfg, seed, exact)
    payload = {
        "benchmark": "boson_sampling",
        "config": config,
        "parameters": cfg,
        "environment": environment(
            seed=seed,
            timing="see benchmarks/bench_common.py: time_call",
            memory="tracemalloc peak; excludes C/C++/numba allocations",
        ),
        "comparisons": {
            "permanent": "equivalent: same matrix, same function",
            "probability_api": "equivalent: single P(t|s) from (U, s); Piquasso program execution timed as setup",
            "probability_kernel": "equivalent: same rectangular permanent with row/column multiplicities",
            "distribution": "equivalent n-photon distribution; Piquasso basis also includes <n photon patterns (zero)",
            "sampling": "both exact samplers of the same distribution; different algorithms (C&C B vs generalised C&C)",
            "single probability vs full simulation": "not applicable (not equivalent tasks)",
        },
        "results": rows,
    }
    path = output or RESULTS / f"boson_sampling_{config}_{stamp()}.json"
    if figures:
        plot(payload, path.parent / f"figures_{config}")
    write_json(payload, path)
    payload["path"] = str(path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="tiny sizes (seconds)")
    group.add_argument("--full", action="store_true", help="larger sizes (long)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    config = "smoke" if args.smoke else "full" if args.full else "default"
    payload = run(config, args.seed, args.output, not args.no_figures)
    for r in payload["results"]:
        size = r.get("n") or (r.get("modes"), r.get("photons"))
        print(
            f"{r['task']:20s} {r['implementation']:34s} {size!s:10s} "
            f"median {r['median_s']:.3e} s  err {r.get('rel_error')}"
        )
    print(f"wrote {payload['path']}")


if __name__ == "__main__":
    main()
