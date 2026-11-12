"""benchmarks/plot_results.py renders every figure from minimal synthetic benchmark JSON."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
ROOT = Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location(
        "plot_results", ROOT / "benchmarks" / "plot_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timing(median):
    return {"median_s": median, "iqr_s": median / 10}


def test_every_plot_renders(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    plots = load()
    gaussian = {
        "results": [
            {"task": t, "kind": k, "modes": n, "batch": 4} | timing(1e-5 * n)
            for t in ("gate", "homodyne", "batch_gate", "batch_homodyne")
            for k in ("native-single", "numpy")
            for n in (4, 8)
        ]
    }
    fock = {
        "gates": [
            {"kind": kind, "engine": e, "gate": "rotate", "dimension": d} | timing(1e-5 * d)
            for kind in ("pure", "mixed")
            for e in ("native", "piquasso")
            for d in (21, 66)
        ],
        "patterns": [
            {"pattern": "wire", "size": 4, "cutoff": 16, "engine": e, "peak_memory_bytes": None}
            | timing(1e-3)
            for e in ("native", "piquasso")
        ],
        "threads": [
            {"kind": "pure", "threads": t, "modes": 3, "cutoff": 8} | timing(1e-4 / t)
            for t in (1, 2)
        ],
        "convergence": [
            {
                "engine": e,
                "cutoff": c,
                "max_abs_photon_number_error": 10.0**-c,
                "retained_norm_product": 1 - 10.0**-c,
            }
            for e in ("native", "piquasso")
            for c in (4, 8)
        ],
    }
    gpu = {
        "results": [
            {"task": t, "implementation": i, "dtype": "float64", "modes": 8, "batch": b}
            | timing(1e-4 * b)
            for t in ("homodyne", "local_symplectic")
            for i in ("numpy", "native", "gpu_kernel", "gpu_end_to_end")
            for b in (4, 64)
        ]
    }
    written = (
        plots.plot_gaussian(gaussian, tmp_path)
        + plots.plot_fock(fock, tmp_path)
        + plots.plot_gpu(gpu, tmp_path)
    )
    assert len(written) == 10 and all(Path(p).stat().st_size > 0 for p in written)
