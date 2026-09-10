"""Reproducible v0.3 sweeps; independent oracles live only in tests/.

Run: python -m experiments.v03_scientific_hardening --output DIR
Retains underresolved rows. Accuracy acceptance belongs to regression tests;
this artifact records actual measurements, not just a passing label.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np

import photographiq as pg
from tests.compilation.test_compiled_cubic_validation import cubic_observation
from tests.gkp.test_independent_convergence import overlap
from tests.v03_reference import synthesis_observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    for path in sorted([*root.glob("src/**/*.py"), *root.glob("tests/**/*.py"), Path(__file__)]):
        source.update(path.read_bytes().replace(b"\r\n", b"\n"))
    data = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": source.hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "piquasso", "jax")
        },
        "synthesis": [],
        "gkp": [],
        "gkp_refinement": [],
        "cubic": [],
        "mixed_scaling": [],
        "thermal": [],
        "gradients": [],
        "estimators": [],
    }

    def checkpoint():
        (output / "measurements.json").write_text(
            json.dumps(data, indent=2, allow_nan=False), encoding="utf-8"
        )

    for kind in ("quartic", "kerr"):
        for strength in (0.001, 0.005):
            for cutoff in (32, 64):
                for state in ("vacuum", "one", "superposition", "coherent", "cat"):
                    for steps in (4, 16, 64, 256):
                        data["synthesis"].append(
                            synthesis_observation(kind, strength, steps, cutoff, state)
                        )
        print(f"Measured {kind} synthesis", flush=True)
        checkpoint()
    data["orders"] = {}
    for kind in ("quartic", "kerr"):
        rows = [
            r
            for r in data["synthesis"]
            if r["kind"] == kind
            and r["strength"] == 0.001
            and r["cutoff"] == 64
            and r["source"] == "superposition"
        ]
        data["orders"][kind] = float(
            -np.polyfit(
                np.log([r["steps"] for r in rows]), np.log([r["amplitude_error"] for r in rows]), 1
            )[0]
        )
    for width, envelope in ((0.4, 0.4), (0.6, 0.5), (0.8, 0.4)):
        for cutoff in (16, 32, 64, 80):
            zero, one = [pg.GKPResource(bit, width, envelope, 6, 4097) for bit in (0, 1)]
            a, mass = zero.project(cutoff)
            b, _ = one.project(cutoff)
            state = pg.simulate(
                pg.Pattern(inputs=(0,)), inputs={0: a}, backend="piquasso-fock", cutoff=cutoff
            ).state
            stabilizers = pg.gkp.stabilizers(state)
            data["gkp"].append(
                {
                    "peak_width": width,
                    "envelope": envelope,
                    "cutoff": cutoff,
                    "captured_weight": mass,
                    "logical_overlap": float(abs(np.vdot(a.amplitudes, b.amplitudes))),
                    "quadrature_overlap": float(abs(overlap(zero, one))),
                    "photon_number": state.photon_number(0),
                    "stabilizers": {k: [v.real, v.imag] for k, v in stabilizers.items()},
                }
            )
    for peaks, grid in ((2, 4097), (4, 4097), (6, 4097), (6, 2049), (6, 8193)):
        r = pg.GKPResource(0, 0.4, 0.4, peaks, grid)
        _, mass = r.project(64)
        reference = pg.GKPResource(0, 0.4, 0.4, 8, 8193)
        data["gkp_refinement"].append(
            {
                "peaks": peaks,
                "grid_points": grid,
                "cutoff": 64,
                "captured_weight": mass,
                "analytic_peak_infidelity": max(0.0, float(1 - abs(overlap(r, reference)) ** 2)),
            }
        )
    print("Measured GKP controls independently", flush=True)
    checkpoint()
    for state in ("vacuum", "coherent", "cat"):
        for r in (0.2, 0.4, 0.7):
            for cutoff in (32, 64, 96):
                data["cubic"].append(cubic_observation(0.03, r, cutoff, state))
    for nbar in (0.2, 1.0):
        for cutoff in (10, 18, 28):
            result = pg.simulate(
                pg.Pattern(inputs=(0,)).append(pg.Loss(0, 0.2, nbar)),
                backend="piquasso-mixed-fock",
                cutoff=cutoff,
            )
            rho = result.state.density_matrix
            data["thermal"].append(
                {
                    "environment_photons": nbar,
                    "cutoff": cutoff,
                    "photon_number": result.state.photon_number(0),
                    "expected_photons": 0.8 * nbar,
                    "retained_trace": result.state.retained_norms[-1],
                    "normalized_trace": float(np.trace(rho).real),
                    "minimum_eigenvalue": float(np.linalg.eigvalsh(rho).min()),
                }
            )
    print("Measured injection and thermal tails", flush=True)
    checkpoint()
    for modes, cutoff in ((1, 16), (1, 32), (1, 64), (2, 8), (2, 12), (2, 18), (3, 5), (3, 7)):
        pattern = pg.Pattern(inputs=tuple(range(modes))).append(pg.Rotate(0, 0.2))
        pg.simulate(
            pattern, backend="piquasso-mixed-fock", cutoff=cutoff
        )  # warm native compilation
        tracemalloc.start()
        start = time.perf_counter()
        state = pg.simulate(pattern, backend="piquasso-mixed-fock", cutoff=cutoff).state
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        dimension = len(state.basis)
        data["mixed_scaling"].append(
            {
                "modes": modes,
                "cutoff": cutoff,
                "dimension": dimension,
                "density_matrix_bytes": 16 * dimension**2,
                "runtime_seconds": elapsed,
                "tracemalloc_peak_bytes": peak,
                "allocation_scope": "Python/registered NumPy allocations; not total RSS or all native workspaces",
                "workload": "vacuum preparation and rotation; warmed; not dense-state worst case",
            }
        )
    import jax

    from photographiq import autodiff as ad

    jax.config.update("jax_enable_x64", True)
    p = pg.Pattern(inputs=(0,)).append(pg.CubicPhase(0, pg.Parameter("g")))
    for gamma in (0.05, 0.15, 0.3):
        for cutoff in (8, 16, 28, 40):

            def objective(g):
                return ad.expectation(p, {"g": g}, cutoff=cutoff, observable="p")

            result = ad.fock_state(p, {"g": gamma}, cutoff=cutoff)
            data["gradients"].append(
                {
                    "gamma": gamma,
                    "cutoff": cutoff,
                    "observable": float(objective(gamma)),
                    "gradient": float(jax.grad(objective)(gamma)),
                    "analytic_observable": gamma,
                    "analytic_gradient": 1.0,
                    "retained_norm": float(min(result.retained_norms)),
                    "boundary_population": float(result.boundary_population),
                }
            )
    for size in (1000, 10000, 100000):
        samples = np.random.default_rng(37).binomial(1, 0.3, size)
        contributions = (samples - 0.2) * (samples / 0.3 - (1 - samples) / 0.7)
        data["estimators"].append(
            {
                "samples": size,
                "seed": 37,
                "score_gradient": float(contributions.mean()),
                "analytic_gradient": 1.0,
                "sample_variance": float(contributions.var(ddof=1)),
                "standard_error": float(contributions.std(ddof=1) / np.sqrt(size)),
            }
        )
    checkpoint()
    print("Measured gradients, estimator variance and allocation scaling", flush=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for kind in ("quartic", "kerr"):
        rows = [
            r
            for r in data["synthesis"]
            if r["kind"] == kind
            and r["strength"] == 0.001
            and r["cutoff"] == 64
            and r["source"] == "superposition"
        ]
        axes[0].loglog(
            [r["steps"] for r in rows], [r["amplitude_error"] for r in rows], "o-", label=kind
        )
    axes[0].set(xlabel="Synthesis slices", ylabel="Phase-aligned amplitude error")
    axes[0].legend()
    rows = [r for r in data["gkp"] if r["peak_width"] == 0.4]
    axes[1].semilogy(
        [r["cutoff"] for r in rows], [max(1e-16, 1 - r["captured_weight"]) for r in rows], "o-"
    )
    axes[1].set(xlabel="GKP Fock cutoff", ylabel="Uncaptured projection weight")
    rows = [r for r in data["gradients"] if r["gamma"] == 0.15]
    axes[2].semilogy(
        [r["cutoff"] for r in rows], [max(1e-16, abs(r["gradient"] - 1)) for r in rows], "o-"
    )
    axes[2].set(xlabel="JAX Fock cutoff", ylabel="Cubic gradient absolute error")
    fig.tight_layout()
    fig.savefig(output / "convergence.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
