"""Gaussian boson sampling benchmark: PhotoGraphiQ vs Piquasso 8 on identical inputs.

Matrix functions (same matrix, same function):

* ``hafnian`` / ``loop_hafnian`` of a random complex symmetric ``d x d`` matrix (``d`` even):
  PhotoGraphiQ native (C++/OpenMP power trace), PhotoGraphiQ Python (NumPy power trace,
  small ``d``), Piquasso ``hafnian_with_reduction`` / ``loop_hafnian_with_reduction``
  (numba power trace; occupation numbers all 1). Reference: the matching sum for
  ``d <= 10``, else PhotoGraphiQ Python where run, else Piquasso.
* ``torontonian`` / ``loop_torontonian`` of ``O = I - Sigma^{-1}`` for a random real
  ``Sigma >= I`` (``2n x 2n``). Piquasso's C++ kernel takes real matrices with mode pairs
  ``(2i, 2i+1)``; PhotoGraphiQ uses pairs ``(i, i+n)``, so the identical matrix is passed
  after the index permutation ``(0,2,..,1,3,..)`` (a relabelling, Tor is unchanged).
  Reference: PhotoGraphiQ Python where run, else Piquasso.

GBS probabilities (single pattern, ``state -> probability``): the state is squeezing ``r``
on every mode, a Haar interferometer, uniform loss ``eta`` and optional displacement
``alpha``, built by ``squeezed_interferometer_state`` and by an equivalent Piquasso
``GaussianSimulator`` program (state construction is timed separately as setup; the two
covariances are checked equal). Implementations per pattern:

* PNR: ``pnr_probability`` with ``method="auto"`` (pure-state ``|haf(B_n)|^2`` of size
  ``N`` when lossless, else the mixed ``haf(A_{n+n})`` of size ``2N``), with
  ``method="mixed"`` (same algorithm class as Piquasso: always size ``2N``), Python
  backend for small sizes, and native with precomputed ``GBSMatrices`` (excludes the
  ``O(m^3)`` setup; not compared to Piquasso, which has no such entry point) vs Piquasso
  ``GaussianState.get_particle_detection_probability``.
* threshold: ``threshold_probability`` (inclusion-exclusion over vacuum projections of the
  dark-mode-conditioned state, the real form of the (loop) torontonian; see
  ``photographiq.sampling.gbs``) vs Piquasso ``get_threshold_detection_probability``. Each
  PhotoGraphiQ row also records the library's own ``estimated_rel_error``;
  ``ThresholdPrecisionWarning`` is silenced here because that estimate is recorded instead.

Sweeps: modes (pattern with m/2 photons/clicks), photon number at fixed modes, squeezing,
loss, displacement. Agreement errors are relative to Piquasso (and PhotoGraphiQ Python where
run). Run ``python benchmarks/gaussian_boson_sampling/bench_gbs.py`` (default ~1-2 min),
``--smoke`` or ``--full``.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
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
    gbs_matrices,
    haar_unitary,
    hafnian,
    hafnian_by_definition,
    loop_hafnian,
    loop_hafnian_by_definition,
    loop_torontonian,
    pnr_probability,
    squeezed_interferometer_state,
    threshold_distribution,
    threshold_probability,
    torontonian,
)
from photographiq.sampling.gbs import ThresholdPrecisionWarning  # noqa: E402

warnings.simplefilter("ignore", ThresholdPrecisionWarning)

RESULTS = Path(__file__).resolve().parent / "results"

CONFIGS = {
    "smoke": {
        "haf_sizes": [2, 4, 6],
        "haf_python_max": 6,
        "tor_sizes": [1, 2, 3],
        "tor_python_max": 3,
        "modes_sweep": [4],
        "photon_sweep": {"modes": 4, "photons": [2]},
        "squeezing_sweep": [0.5],
        "loss_sweep": [0.9],
        "sweep_modes": 4,
        "normalization_modes": [3],
        "pnr_python_max_photons": 3,
        "threshold_python_max_clicks": 3,
        "repeats": 3,
        "budget_s": 0.5,
    },
    "notebook": {
        "haf_sizes": list(range(2, 21, 2)),
        "haf_python_max": 12,
        "tor_sizes": list(range(1, 13)),
        "tor_python_max": 8,
        "modes_sweep": [4, 8, 12, 16],
        "photon_sweep": {"modes": 12, "photons": [2, 4, 6, 8, 10]},
        "squeezing_sweep": [0.1, 0.5, 1.0],
        "loss_sweep": [1.0, 0.5, 0.1],
        "sweep_modes": 6,
        "normalization_modes": [4, 6],
        "pnr_python_max_photons": 4,
        "threshold_python_max_clicks": 6,
        "repeats": 5,
        "budget_s": 0.5,
    },
    "default": {
        "haf_sizes": list(range(2, 25, 2)),
        "haf_python_max": 16,
        "tor_sizes": list(range(1, 15)),
        "tor_python_max": 10,
        "modes_sweep": [4, 8, 12, 16, 20, 24],
        "photon_sweep": {"modes": 16, "photons": [2, 4, 6, 8, 10, 12, 14]},
        "squeezing_sweep": [0.1, 0.5, 1.0, 1.5],
        "loss_sweep": [1.0, 0.9, 0.5, 0.1],
        "sweep_modes": 8,
        "normalization_modes": [4, 8],
        "pnr_python_max_photons": 6,
        "threshold_python_max_clicks": 8,
        "repeats": 7,
        "budget_s": 2.0,
    },
    "full": {
        "haf_sizes": list(range(2, 33, 2)),
        "haf_python_max": 20,
        "tor_sizes": list(range(1, 19)),
        "tor_python_max": 12,
        "modes_sweep": [4, 8, 12, 16, 20, 24, 28, 32],
        "photon_sweep": {"modes": 20, "photons": [2, 4, 6, 8, 10, 12, 14, 16]},
        "squeezing_sweep": [0.1, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "loss_sweep": [1.0, 0.95, 0.9, 0.7, 0.5, 0.3, 0.1],
        "sweep_modes": 12,
        "normalization_modes": [4, 8, 12],
        "pnr_python_max_photons": 8,
        "threshold_python_max_clicks": 10,
        "repeats": 9,
        "budget_s": 10.0,
    },
}


def _native() -> bool:
    return "sampling" in native.components()


def _piquasso():
    try:
        import piquasso as pq
    except ImportError:
        return None
    return pq


def _row(task: str, impl: str, timing: dict | None, **fields) -> dict:
    return {"task": task, "implementation": impl, **fields, **(timing or {})}


# --------------------------------------------------------------------- matrix functions


def _complex_symmetric(rng, d):
    x = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    return (x + x.T) / np.sqrt(4 * d)


def _bench_function(task, n, impls, values_ref, cfg, rows):
    values = {k: complex(f()) for k, f in impls.items()}
    ref_name, ref = values_ref(values)
    for impl, f in impls.items():
        rows.append(
            _row(
                task,
                impl,
                time_call(f, repeats=cfg["repeats"], budget_s=cfg["budget_s"]),
                n=n,
                value=values[impl],
                reference=ref_name,
                rel_error=None if ref is None or impl == ref_name else rel_err(values[impl], ref),
                peak_python_bytes=peak_python_bytes(f),
            )
        )


def _pick_ref(definition=None):
    def choose(values):
        if definition is not None:
            return "definition", definition()
        for name in ("photographiq_python", "piquasso"):
            if name in values:
                return name, values[name]
        return None, None

    return choose


def bench_matrix_functions(cfg: dict, seed: int) -> list[dict]:
    pq = _piquasso()
    rows = []
    rng = np.random.default_rng(seed)
    for d in cfg["haf_sizes"]:
        a = _complex_symmetric(rng, d)
        diag = rng.normal(size=d) + 1j * rng.normal(size=d)
        a_loop = a.copy()
        np.fill_diagonal(a_loop, diag)
        ones = np.ones(d, dtype=int)
        for task, pg_fn, defn in (
            ("hafnian", lambda b: hafnian(a, backend=b), lambda: hafnian_by_definition(a)),
            (
                "loop_hafnian",
                lambda b: loop_hafnian(a, diag, backend=b),
                lambda: loop_hafnian_by_definition(a_loop),
            ),
        ):
            impls = {}
            if _native():
                impls["photographiq_native"] = lambda f=pg_fn: f("native")
            if d <= cfg["haf_python_max"]:
                impls["photographiq_python"] = lambda f=pg_fn: f("python")
            if pq is not None:
                from piquasso._math.hafnian import (
                    hafnian_with_reduction,
                    loop_hafnian_with_reduction,
                )

                impls["piquasso"] = (
                    (lambda: hafnian_with_reduction(a, ones))
                    if task == "hafnian"
                    else (lambda: loop_hafnian_with_reduction(a, diag, ones))
                )
            _bench_function(task, d, impls, _pick_ref(defn if d <= 10 else None), cfg, rows)
    for n in cfg["tor_sizes"]:
        x = rng.normal(size=(2 * n, 2 * n))
        sigma = np.eye(2 * n) + 0.3 * x @ x.T / n
        o = np.eye(2 * n) - np.linalg.inv(sigma)  # xpxp pairs (2i, 2i+1)
        gamma = rng.normal(size=2 * n)
        perm = np.r_[np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)]
        ob, gb = o[np.ix_(perm, perm)], gamma[perm]
        for task in ("torontonian", "loop_torontonian"):
            if task == "torontonian":

                def pg_fn(b):
                    return torontonian(ob, backend=b)
            else:

                def pg_fn(b):
                    return loop_torontonian(ob, gb, backend=b)

            impls = {}
            if _native():
                impls["photographiq_native"] = lambda f=pg_fn: f("native")
            if n <= cfg["tor_python_max"]:
                impls["photographiq_python"] = lambda f=pg_fn: f("python")
            if pq is not None:
                from piquasso._math import torontonian as pqt

                impls["piquasso"] = (
                    (lambda: pqt.torontonian(o))
                    if task == "torontonian"
                    else (lambda: pqt.loop_torontonian(o, gamma))
                )
            _bench_function(task, n, impls, _pick_ref(), cfg, rows)
    return rows


# ------------------------------------------------------------------------ GBS states


def _pq_state(pq, squeezing, unitary, eta, alpha):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        for i, r in enumerate(squeezing):
            pq.Q(i) | pq.Squeezing(r=r)
        pq.Q(*range(len(squeezing))) | pq.Interferometer(unitary)
        if eta != 1:
            for i in range(len(squeezing)):
                pq.Q(i) | pq.Attenuator(theta=np.arccos(np.sqrt(eta)))
        for i, a in enumerate(alpha):
            if a != 0:
                pq.Q(i) | pq.Displacement(r=abs(a), phi=np.angle(a))
    sim = pq.GaussianSimulator(d=len(squeezing), config=pq.Config(hbar=2.0))
    return program, sim


def _pattern(m: int, k: int, rng) -> np.ndarray:
    p = np.zeros(m, dtype=int)
    p[rng.choice(m, k, replace=False)] = 1
    return p


def bench_case(sweep, m, photons, r, eta, alpha, cfg, seed, rows):
    pq = _piquasso()
    rng = np.random.default_rng(seed + 31 * m + photons)
    u = haar_unitary(m, rng=seed + m)
    alpha_vec = np.full(m, alpha, dtype=complex)
    case = {
        "sweep": sweep,
        "modes": m,
        "photons": photons,
        "squeezing": r,
        "transmissivity": eta,
        "displacement": alpha,
    }

    def build():
        return squeezed_interferometer_state(np.full(m, r), u, eta, alpha_vec)

    state = build()
    rows.append(
        _row(
            "gbs_setup",
            "photographiq_state",
            time_call(build, repeats=cfg["repeats"], budget_s=cfg["budget_s"]),
            **case,
        )
    )
    g = gbs_matrices(state)
    rows.append(
        _row(
            "gbs_setup",
            "photographiq_gbs_matrices",
            time_call(
                lambda: gbs_matrices(state), repeats=cfg["repeats"], budget_s=cfg["budget_s"]
            ),
            **case,
        )
    )
    ref_state = None
    if pq is not None:
        program, sim = _pq_state(pq, np.full(m, r), u, eta, alpha_vec)
        rows.append(
            _row(
                "gbs_setup",
                "piquasso_execute_program",
                time_call(
                    lambda: sim.execute(program), repeats=cfg["repeats"], budget_s=cfg["budget_s"]
                ),
                **case,
            )
        )
        ref_state = sim.execute(program).state
        case["state_cov_max_abs_diff"] = float(
            np.abs(ref_state.xpxp_covariance_matrix / 2 - state.covariance).max()
        )
        case["state_mean_max_abs_diff"] = float(
            np.abs(ref_state.xpxp_mean_vector - state.mean).max()
        )
    n = _pattern(m, photons, rng)
    clicks = n  # the same mode subset, read as click pattern
    pnr = {}
    if _native():
        pnr["photographiq_native_auto"] = lambda: pnr_probability(state, n, backend="native")
        pnr["photographiq_native_mixed"] = lambda: pnr_probability(
            state, n, method="mixed", backend="native"
        )
        pnr["photographiq_native_precomputed"] = lambda: pnr_probability(g, n, backend="native")
    if photons <= cfg["pnr_python_max_photons"]:
        pnr["photographiq_python_auto"] = lambda: pnr_probability(state, n, backend="python")
    if ref_state is not None:
        pnr["piquasso"] = lambda: float(ref_state.get_particle_detection_probability(n))
    thr = {}
    if _native():
        thr["photographiq_native"] = lambda: threshold_probability(state, clicks, backend="native")
        thr["photographiq_native_precomputed"] = lambda: threshold_probability(
            g, clicks, backend="native"
        )
    if photons <= cfg["threshold_python_max_clicks"]:
        thr["photographiq_python"] = lambda: threshold_probability(state, clicks, backend="python")
    if ref_state is not None:
        thr["piquasso"] = lambda: float(ref_state.get_threshold_detection_probability(clicks))
    # The library's own rounding-error estimate for each PhotoGraphiQ threshold value
    # (untimed; the timed calls are identical without return_error).
    estimates = {
        "photographiq_native": ("native", state),
        "photographiq_native_precomputed": ("native", g),
        "photographiq_python": ("python", state),
    }
    for task, impls in (("gbs_pnr", pnr), ("gbs_threshold", thr)):
        values = {k: float(f()) for k, f in impls.items()}
        py = next((v for k, v in values.items() if k.startswith("photographiq_python")), None)
        for impl, f in impls.items():
            v = values[impl]
            rows.append(
                _row(
                    task,
                    impl,
                    time_call(f, repeats=cfg["repeats"], budget_s=cfg["budget_s"]),
                    **case,
                    pattern=n.tolist(),
                    state_pure=g.pure,
                    probability=v,
                    rel_error_vs_piquasso=None
                    if "piquasso" not in values or impl == "piquasso"
                    else rel_err(v, values["piquasso"]),
                    abs_error_vs_piquasso=None
                    if "piquasso" not in values or impl == "piquasso"
                    else abs(v - values["piquasso"]),
                    rel_error_vs_python=None
                    if py is None or impl.startswith("photographiq_python")
                    else rel_err(v, py),
                    estimated_rel_error=threshold_probability(
                        estimates[impl][1], clicks, backend=estimates[impl][0], return_error=True
                    )[1]
                    if task == "gbs_threshold" and impl in estimates
                    else None,
                    peak_python_bytes=peak_python_bytes(f),
                )
            )


def bench_gbs(cfg: dict, seed: int) -> list[dict]:
    rows = []
    for m in cfg["modes_sweep"]:
        for eta in (1.0, 0.9):
            bench_case("modes", m, m // 2, 0.5, eta, 0.0, cfg, seed, rows)
    ps = cfg["photon_sweep"]
    for k in ps["photons"]:
        bench_case("photons", ps["modes"], k, 0.5, 0.9, 0.0, cfg, seed, rows)
    mm = cfg["sweep_modes"]
    for r in cfg["squeezing_sweep"]:
        bench_case("squeezing", mm, mm // 2, r, 0.9, 0.0, cfg, seed, rows)
    for eta in cfg["loss_sweep"]:
        bench_case("loss", mm, mm // 2, 0.5, eta, 0.0, cfg, seed, rows)
    for eta in (1.0, 0.9):
        bench_case("displacement", mm, mm // 2, 0.5, eta, 0.3 + 0.2j, cfg, seed, rows)
    for m in cfg["normalization_modes"]:
        state = squeezed_interferometer_state(np.full(m, 0.5), haar_unitary(m, rng=seed), 0.9)
        dist = threshold_distribution(state)
        rows.append(
            _row(
                "threshold_normalization",
                "photographiq_native" if _native() else "photographiq_python",
                None,
                modes=m,
                patterns=len(dist),
                normalization_error=abs(math.fsum(dist.values()) - 1.0),
                min_probability=min(dist.values()),
            )
        )
    return rows


# ------------------------------------------------------------------------------- plots

_COLORS = {
    "photographiq_native": 0,
    "photographiq_native_auto": 0,
    "photographiq_python": 1,
    "photographiq_python_auto": 1,
    "piquasso": 2,
    "photographiq_native_mixed": 3,
    "photographiq_native_precomputed": 4,
}


def _pts(rows, task, impl, **match):
    return [
        r
        for r in rows
        if r["task"] == task
        and r["implementation"] == impl
        and all(r.get(k) == v for k, v in match.items())
    ]


def plot(payload: dict, outdir: Path, save: bool = True) -> list:
    rows, figs = payload["results"], []
    tasks = ["hafnian", "loop_hafnian", "torontonian", "loop_torontonian"]
    fig, axes = new_axes(4, width=3.6)
    fig2, axes2 = new_axes(4, width=3.6)
    for ax, ax2, task in zip(axes, axes2, tasks):
        for impl in ("photographiq_native", "photographiq_python", "piquasso"):
            pts = _pts(rows, task, impl)
            if not pts:
                continue
            st = series_style(_COLORS[impl])
            ax.errorbar(
                [p["n"] for p in pts],
                [p["median_s"] for p in pts],
                yerr=[p["iqr_s"] / 2 for p in pts],
                label=impl,
                **st,
            )
            err = [(p["n"], max(p["rel_error"], 1e-17)) for p in pts if p["rel_error"] is not None]
            if err:
                ax2.plot(*zip(*err), label=impl, **st)
        xl = "matrix size 2n" if "hafnian" in task else "modes n (matrix 2n x 2n)"
        ax.set(yscale="log", xlabel=xl, ylabel="time per call [s]", title=task)
        ax2.set(
            yscale="log", xlabel=xl, ylabel="rel. error vs reference (floored 1e-17)", title=task
        )
    axes[0].legend(frameon=False, fontsize=7)
    axes2[0].legend(frameon=False, fontsize=7)
    figs += [("gbs_matrix_functions_runtime", fig), ("gbs_matrix_functions_correctness", fig2)]

    pnr_impls = [
        "photographiq_native_auto",
        "photographiq_native_mixed",
        "photographiq_python_auto",
        "piquasso",
    ]
    thr_impls = ["photographiq_native", "photographiq_python", "piquasso"]
    for sweep, x, name in (
        ("modes", "modes", "gbs_runtime_vs_modes"),
        ("photons", "photons", "gbs_runtime_vs_photons"),
    ):
        fig, axes = new_axes(2)
        for ax, task, impls, title in (
            (axes[0], "gbs_pnr", pnr_impls, "PNR (hafnian)"),
            (axes[1], "gbs_threshold", thr_impls, "threshold (torontonian)"),
        ):
            for impl in impls:
                for eta, ls in ((0.9, "-"), (1.0, ":")):
                    pts = _pts(rows, task, impl, sweep=sweep, transmissivity=eta)
                    if pts:
                        ax.plot(
                            [p[x] for p in pts],
                            [p["median_s"] for p in pts],
                            ls=ls,
                            label=f"{impl} (eta={eta})",
                            **{k: v for k, v in series_style(_COLORS[impl]).items()},
                        )
            ax.set(
                yscale="log",
                xlabel="modes m (pattern with m/2 photons)"
                if x == "modes"
                else "photons / clicks in pattern",
                ylabel="time per probability [s]",
                title=title,
            )
            ax.legend(frameon=False, fontsize=6)
        figs.append((name, fig))

    fig, axes = new_axes(2)
    for ax, task, impls in (
        (axes[0], "gbs_pnr", pnr_impls[:3]),
        (axes[1], "gbs_threshold", thr_impls[:2]),
    ):
        for impl in impls:
            pts = [
                p
                for p in rows
                if p["task"] == task
                and p["implementation"] == impl
                and p.get("rel_error_vs_piquasso") is not None
            ]
            if pts:
                ax.plot(
                    range(len(pts)),
                    [max(p["rel_error_vs_piquasso"], 1e-17) for p in pts],
                    ls="none",
                    label=impl,
                    **series_style(_COLORS[impl]),
                )
        ax.set(
            yscale="log",
            xlabel="case index (all sweeps, JSON order)",
            ylabel="|p - p_piquasso| / p_piquasso (floored 1e-17)",
            title=f"{'PNR' if task == 'gbs_pnr' else 'threshold'} agreement with Piquasso",
        )
        ax.legend(frameon=False, fontsize=7)
    figs.append(("gbs_correctness", fig))

    fig, axes = new_axes(2)
    for ax, task, impls in ((axes[0], "gbs_pnr", pnr_impls), (axes[1], "gbs_threshold", thr_impls)):
        for impl in impls:
            pts = _pts(rows, task, impl, sweep="photons")
            if pts:
                ax.plot(
                    [p["photons"] for p in pts],
                    [p["peak_python_bytes"] / 1024 for p in pts],
                    label=impl,
                    **series_style(_COLORS[impl]),
                )
        ax.set(
            yscale="log",
            xlabel="photons / clicks",
            ylabel="tracemalloc peak [KiB]",
            title=f"{task}: Python-heap peak (photon sweep)",
        )
        ax.legend(frameon=False, fontsize=7)
    figs.append(("gbs_memory", fig))

    if save:
        payload.setdefault("figures", [])
        for name, f in figs:
            payload["figures"] += save_figure(f, outdir / name)
    return figs


def run(
    config: str = "default", seed: int = 11, output: Path | None = None, figures: bool = True
) -> dict:
    cfg = CONFIGS[config]
    rows = bench_matrix_functions(cfg, seed) + bench_gbs(cfg, seed)
    payload = {
        "benchmark": "gaussian_boson_sampling",
        "config": config,
        "parameters": cfg,
        "environment": environment(
            seed=seed,
            timing="see benchmarks/bench_common.py: time_call",
            memory="tracemalloc peak; excludes C/C++/numba allocations",
        ),
        "comparisons": {
            "matrix functions": "equivalent: same matrix, same function (torontonian: index relabelling only)",
            "gbs_pnr photographiq_native_auto/mixed/python vs piquasso": "equivalent: same state -> one PNR probability; "
            "auto uses the N-dim pure formula for lossless states, mixed and Piquasso use 2N-dim hafnians",
            "gbs_threshold vs piquasso": "equivalent: same state -> one click probability",
            "photographiq_native_precomputed": "not applicable for Piquasso comparison (excludes O(m^3) setup)",
            "gbs_setup": "reported separately; PhotoGraphiQ state construction vs Piquasso program execution",
        },
        "results": rows,
    }
    path = output or RESULTS / f"gbs_{config}_{stamp()}.json"
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
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    config = "smoke" if args.smoke else "full" if args.full else "default"
    payload = run(config, args.seed, args.output, not args.no_figures)
    for r in payload["results"]:
        if "median_s" not in r:
            continue
        size = r.get("n") or (
            r.get("sweep"),
            r.get("modes"),
            r.get("photons"),
            r.get("transmissivity"),
        )
        err = r.get("rel_error", r.get("rel_error_vs_piquasso"))
        print(
            f"{r['task']:18s} {r['implementation']:32s} {size!s:28s} median {r['median_s']:.3e} s  err {err}"
        )
    print(f"wrote {payload['path']}")


if __name__ == "__main__":
    main()
