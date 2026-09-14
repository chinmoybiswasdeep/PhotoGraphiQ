"""R5: Random single-mode symplectic compilation (major validation result).

Draws many random valid single-mode Gaussian transformations
S = R(phi1) S(r) R(phi2) with a deterministic RNG, compiles each through
``Circuit(1).gaussian(0, S).compile(...)``, and compares the compiled affine
map against the exact target S using the independent Gaussian-channel
analyzer. ``decompose_symplectic`` can raise for numerically unstable near-
singular requests (docs/theory.md); those cases are recorded as decomposition
failures rather than silently skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

N_SAMPLES = 400
SEED = 20260913
SQUEEZING = 1.0
R_RANGE = (-1.6, 1.6)


def independent_rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def independent_squeezing(r):
    return np.array([[np.exp(-r), 0.0], [0.0, np.exp(r)]])


def random_symplectic(generator: np.random.Generator):
    phi1 = generator.uniform(-np.pi, np.pi)
    phi2 = generator.uniform(-np.pi, np.pi)
    r = generator.uniform(*R_RANGE)
    matrix = independent_rotation(phi1) @ independent_squeezing(r) @ independent_rotation(phi2)
    return matrix, phi1, r, phi2


def main():
    plt = common.setup_style()
    generator = common.rng(SEED)
    rows = []
    for i in range(N_SAMPLES):
        target, phi1, r, phi2 = random_symplectic(generator)
        row = {
            "sample": i,
            "phi1": phi1,
            "r": r,
            "phi2": phi2,
            "determinant": float(np.linalg.det(target)),
            "target_matrix": target.tolist(),
        }
        try:
            circuit = pg.Circuit(1).gaussian(0, target)
            pattern, trace = circuit.compile(squeezing=SQUEEZING, return_trace=True)
            channel = pg.gaussian_channel(pattern)
            error = common.frobenius_error(channel.matrix, target)
            step = trace.steps[0]
            row.update(
                {
                    "decomposition_failed": False,
                    "compiled_matrix": channel.matrix.tolist(),
                    "frobenius_error": error,
                    "log10_error": common.safe_log10(error),
                    "resource_nodes": len(step.resource_nodes),
                    "measurements": len(step.measurements),
                    "commands": len(pattern.commands),
                }
            )
        except ValueError as exc:
            row.update(
                {
                    "decomposition_failed": True,
                    "error_message": str(exc),
                    "compiled_matrix": None,
                    "frobenius_error": None,
                    "log10_error": None,
                    "resource_nodes": None,
                    "measurements": None,
                    "commands": None,
                }
            )
        rows.append(row)

    successes = [r for r in rows if not r["decomposition_failed"]]
    failed = [r for r in rows if r["decomposition_failed"]]
    tolerance = 1e-7
    bad = [r for r in successes if r["frobenius_error"] > tolerance]
    if bad:
        common.save_csv(bad, "R5_random_symplectic_compile_FAILURES")
        raise AssertionError(f"{len(bad)} compiled maps exceeded tolerance {tolerance}; see FAILURES csv")

    common.save_result(
        rows,
        "R5_random_symplectic_compile",
        extra={
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "squeezing": SQUEEZING,
            "r_range": list(R_RANGE),
            "n_successes": len(successes),
            "n_decomposition_failures": len(failed),
        },
    )
    if failed:
        common.save_csv(failed, "R5_random_symplectic_compile_decomposition_failures")

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    errors = [r["log10_error"] for r in successes]
    ax.hist(errors, bins=30, color="#24677b", edgecolor="white")
    ax.set_xlabel(r"$\log_{10}$(Frobenius error)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Random single-mode symplectic compilation (n={len(successes)}, "
        f"{len(failed)} decomposition failures)"
    )
    common.save_figure(fig, "R5_random_symplectic_compile")
    plt.close(fig)

    common.print_summary(
        "R5 random symplectic compilation",
        n_samples=N_SAMPLES,
        n_successes=len(successes),
        n_decomposition_failures=len(failed),
        max_error=max((r["frobenius_error"] for r in successes), default=None),
        median_log10_error=float(np.median(errors)) if errors else None,
    )


if __name__ == "__main__":
    main()
