"""Figures for the native Gaussian, Fock and GPU benchmark JSON files.

    python benchmarks/plot_results.py                      # newest non-smoke JSON of each
    python benchmarks/plot_results.py --gaussian A.json --fock B.json --gpu C.json

Only plots what the JSON contains (no re-computation). Figures (SVG + PDF) go to
``<benchmark>/results/figures_<json stem>/``. The sampling and Graphix benchmarks draw their
own figures.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_common import new_axes, save_figure, series_style  # noqa: E402


def newest(directory: Path, pattern: str) -> Path | None:
    files = [p for p in directory.glob(pattern) if "smoke" not in p.name]
    return max(files, key=lambda p: p.stat().st_mtime, default=None)


def _line(ax, xs, rows, index, label):
    """Median with an IQR band; ``rows`` aligned with ``xs``."""
    med = [r["median_s"] for r in rows]
    iqr = [r.get("iqr_s", 0.0) for r in rows]
    style = series_style(index)
    ax.plot(xs, med, label=label, **style)
    lo = [max(m - i / 2, m * 1e-3) for m, i in zip(med, iqr, strict=True)]
    ax.fill_between(
        xs,
        lo,
        [m + i / 2 for m, i in zip(med, iqr, strict=True)],
        color=style["color"],
        alpha=0.15,
        lw=0,
    )


def _finish(ax, xlabel, title, logx=True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("median runtime (s)")
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_title(title, fontsize=10)
    # below the axes, so it never hides data
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)


def plot_gaussian(data: dict, out: Path) -> list[str]:
    rows = data["results"]
    tasks = ["gate", "homodyne", "batch_gate", "batch_homodyne"]
    fig, axes = new_axes(ncols=4, width=3.6)
    for ax, task in zip(axes, tasks, strict=True):
        kinds = sorted({r["kind"] for r in rows if r["task"] == task})
        for i, kind in enumerate(kinds):
            sel = sorted(
                (r for r in rows if r["task"] == task and r["kind"] == kind),
                key=lambda r: r["modes"],
            )
            _line(ax, [r["modes"] for r in sel], sel, i, kind)
        batch = {r.get("batch") for r in rows if r["task"] == task} - {None}
        note = f" (B = {min(batch)}..{max(batch)})" if batch else ""
        _finish(ax, "modes n", task + note)
    fig.suptitle("Native Gaussian engine vs NumPy: runtime vs modes", fontsize=11)
    return save_figure(fig, out / "gaussian_runtime_vs_modes")


def plot_fock(data: dict, out: Path) -> list[str]:
    paths = []
    # 1. per-gate runtime vs Hilbert-space dimension (geometric mean over gates)
    fig, axes = new_axes(ncols=2)
    for ax, kind in zip(axes, ("pure", "mixed"), strict=True):
        groups = defaultdict(list)
        for r in data["gates"]:
            if r["kind"] == kind and r.get("median_s"):
                groups[r["engine"], r["dimension"]].append(r["median_s"])
        for i, engine in enumerate(sorted({e for e, _ in groups})):
            dims = sorted(d for e, d in groups if e == engine)
            geo = [
                math.exp(sum(map(math.log, groups[engine, d])) / len(groups[engine, d]))
                for d in dims
            ]
            ax.plot(dims, geo, label=engine, **series_style(i))
        _finish(ax, "Fock-space dimension", f"{kind} state: per-gate runtime (geo. mean)")
    paths += save_figure(fig, out / "fock_gate_runtime_vs_dimension")

    # 2. whole patterns: runtime and peak memory per case
    cases = sorted({(r["pattern"], r["size"], r["cutoff"]) for r in data["patterns"]})
    engines = sorted({r["engine"] for r in data["patterns"]})
    fig, axes = new_axes(ncols=2, width=5.2)
    width = 0.8 / max(len(engines), 1)
    for i, engine in enumerate(engines):
        rows = {
            (r["pattern"], r["size"], r["cutoff"]): r
            for r in data["patterns"]
            if r["engine"] == engine
        }
        xs = [k + i * width for k in range(len(cases))]
        color = series_style(i)["color"]
        axes[0].bar(
            xs,
            [rows[c]["median_s"] if c in rows else 0 for c in cases],
            width,
            label=engine,
            color=color,
        )
        mem = [rows[c].get("peak_memory_bytes") if c in rows else None for c in cases]
        axes[1].bar(xs, [(m or 0) / 2**20 for m in mem], width, label=engine, color=color)
    labels = [f"{p} {s}\ncutoff {c}" for p, s, c in cases]
    for ax, ylabel in zip(axes, ("median runtime (s)", "peak working set (MiB)"), strict=True):
        ax.set_xticks([k + 0.4 - width / 2 for k in range(len(cases))], labels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_yscale("log")
    axes[0].set_title("whole patterns: runtime", fontsize=10)
    axes[1].set_title("whole patterns: process peak memory", fontsize=10)
    paths += save_figure(fig, out / "fock_patterns_runtime_memory")

    # 3. thread scaling and 4. accuracy vs cutoff
    fig, axes = new_axes(ncols=2)
    for i, kind in enumerate(sorted({r["kind"] for r in data["threads"]})):
        sel = sorted((r for r in data["threads"] if r["kind"] == kind), key=lambda r: r["threads"])
        _line(
            axes[0],
            [r["threads"] for r in sel],
            sel,
            i,
            f"{kind} ({sel[0]['modes']} modes, cutoff {sel[0]['cutoff']})",
        )
    _finish(axes[0], "OpenMP threads", "native kernel thread scaling")
    for i, engine in enumerate(sorted({r["engine"] for r in data.get("convergence", [])})):
        sel = sorted(
            (r for r in data["convergence"] if r["engine"] == engine), key=lambda r: r["cutoff"]
        )
        axes[1].plot(
            [r["cutoff"] for r in sel],
            [max(r["max_abs_photon_number_error"], 1e-17) for r in sel],
            label=f"{engine}: |<n> - exact|",
            **series_style(i),
        )
        axes[1].plot(
            [r["cutoff"] for r in sel],
            [max(1 - r["retained_norm_product"], 1e-17) for r in sel],
            label=f"{engine}: 1 - retained norm",
            ls="--",
            **{k: v for k, v in series_style(i).items() if k != "lw"},
            lw=1,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("total-photon cutoff")
    axes[1].set_ylabel("error vs exact Gaussian")
    axes[1].set_title("accuracy vs cutoff (2-mode Gaussian circuit)", fontsize=10)
    axes[1].legend(fontsize=7, frameon=False)
    paths += save_figure(fig, out / "fock_threads_and_convergence")
    return paths


def plot_gpu(data: dict, out: Path) -> list[str]:
    rows = data["results"]
    tasks = sorted({r["task"] for r in rows})
    n = max(r["modes"] for r in rows)
    impls = ["numpy", "native", "gpu_kernel", "gpu_end_to_end"]
    fig, axes = new_axes(ncols=len(tasks), width=3.8)
    for ax, task in zip(axes, tasks, strict=True):
        index = 0
        for dtype in ("float64", "float32"):
            for impl in impls:
                sel = sorted(
                    (
                        r
                        for r in rows
                        if r["task"] == task
                        and r["implementation"] == impl
                        and r["dtype"] == dtype
                        and r["modes"] == n
                        and r.get("median_s")
                    ),
                    key=lambda r: r["batch"],
                )
                if sel:
                    _line(ax, [r["batch"] for r in sel], sel, index, f"{impl} {dtype}")
                index += 1
        _finish(ax, "batch size B", f"{task} (n = {n})")
    fig.suptitle("CPU (NumPy, native C++) vs GPU (CuPy): runtime vs batch", fontsize=11)
    return save_figure(fig, out / "gpu_runtime_vs_batch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gaussian", type=Path)
    parser.add_argument("--fock", type=Path)
    parser.add_argument("--gpu", type=Path)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    jobs = (
        (
            args.gaussian or newest(HERE / "gaussian" / "results", "native_gaussian_*.json"),
            plot_gaussian,
        ),
        (args.fock or newest(HERE / "fock" / "results", "fock-*.json"), plot_fock),
        (args.gpu or newest(HERE / "gpu" / "results", "gpu_gaussian_*.json"), plot_gpu),
    )
    for path, plot in jobs:
        if path is None:
            print(f"{plot.__name__}: no results file, skipped")
            continue
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for figure in plot(data, Path(path).parent / f"figures_{Path(path).stem}"):
            print("wrote", figure)


if __name__ == "__main__":
    main()
