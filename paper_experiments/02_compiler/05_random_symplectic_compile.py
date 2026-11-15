"""R5: Random single-mode symplectic compilation (major validation result).

Draws many random valid single-mode Gaussian transformations
S = R(phi1) S(r) R(phi2) with a deterministic RNG, compiles each through
``Circuit(1).gaussian(0, S).compile(...)``, and compares the compiled affine
map against the exact target S using the independent Gaussian-channel
analyzer. ``decompose_symplectic`` can raise for numerically unstable near-
singular requests (docs/theory.md); those cases are recorded as decomposition
failures rather than silently skipped.

Upgrade migration (2026-11): every compiled pattern's graph metadata is
validated against its command list and the typed IR ``OpenGraph``
(``graph_metadata_checks``); the first EXEC_SAMPLES compiled patterns are
EXECUTED on NumPy ``gaussian`` and C++ ``native-gaussian`` (``execute_compiled``)
and compared with the analytic channel (map and added noise) within 5
shot-noise standard errors. The added noise N is reported in its own column
(noise_trace) and is never counted as compilation error.
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
EXEC_SAMPLES = 8
EXEC_ALPHA = 0.25 + 0.15j


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


def graph_metadata_checks(pattern) -> dict:
    """Consistency of the compiled pattern's graph metadata with its command list.

    Checks: Entangle commands reproduce the resource-graph edges and weights;
    Prepare commands reproduce the node squeezing of every non-input node; every
    non-output node is measured exactly once and no output is measured; every
    outcome-dependent command (feed-forward) comes after the measurement it reads
    (``Pattern.dependencies`` edges point forward); the typed IR
    ``OpenGraph.from_cvgraph`` keeps nodes, weights, inputs and outputs through a
    JSON-style ``to_dict``/``from_dict`` round trip.
    """
    network = pattern.graph.network
    entangle = {frozenset((c.u, c.v)): float(c.weight) for c in pattern.commands if isinstance(c, pg.Entangle)}
    edges = {frozenset((u, v)): float(d.get("weight", 1.0)) for u, v, d in network.edges(data=True)}
    prepare = {c.node: float(c.squeezing) for c in pattern.commands if isinstance(c, pg.Prepare)}
    node_squeezing = {n: float(d.get("squeezing", 0.0)) for n, d in network.nodes(data=True) if n not in pattern.inputs}
    measured = [c.node for c in pattern.commands if isinstance(c, pg.Measure)]
    expected_measured = set(network.nodes) - set(pattern.outputs)
    dag = pattern.dependencies()
    og = pg.OpenGraph.from_cvgraph(pattern.graph)
    restored = pg.OpenGraph.from_dict(og.to_dict())
    return {
        "edges_match_entangle_commands": entangle == edges,
        "squeezing_matches_prepare_commands": prepare == node_squeezing,
        "each_non_output_measured_once": sorted(measured, key=repr) == sorted(expected_measured, key=repr),
        "outputs_never_measured": not set(measured) & set(pattern.outputs),
        "feedforward_respects_measurement_order": all(u < v for u, v in dag.edges()),
        "open_graph_round_trip": restored.to_dict() == og.to_dict()
        and tuple(og.inputs) == tuple(pattern.inputs)
        and tuple(og.outputs) == tuple(pattern.outputs),
        "measurement_order": measured,
    }


# --------------------------------------------------------------------------- #
# Upgrade migration: execution of the compiled pattern on the trajectory engines
# --------------------------------------------------------------------------- #
EXEC_BACKENDS = ("gaussian", "native-gaussian")
EXEC_SHOTS = 600
EXEC_SEED = 20261113
SEM_FACTOR = 5.0  # ensemble moments within 5 shot-noise standard errors of the channel
NATIVE_TRAJECTORY_TOLERANCE = 1e-9  # native-gaussian vs gaussian records, same seed


def execute_compiled(pattern, channel, alphas, *, shots=EXEC_SHOTS, seed=EXEC_SEED):
    """Run ``pattern`` on both Gaussian engines and compare with ``channel``.

    Returns the worst z-scores of the ensemble mean/covariance against
    ``channel.apply`` (shot-noise standard errors from the analytic spread of
    the conditional means), the native-vs-NumPy trajectory difference, and the
    planner's backend choice. The compilation map error and the added noise N
    are reported separately by the caller; N is never counted as map error.
    """
    inputs = {node: pg.GaussianInput.coherent(a) for node, a in zip(pattern.inputs, alphas, strict=True)}
    mean = np.concatenate([[2 * a.real, 2 * a.imag] for a in alphas])
    ideal = channel.apply(pg.GaussianState(mean, np.eye(2 * len(alphas)), pattern.inputs))
    conditional = pg.simulate(pattern, backend="gaussian", inputs=inputs, seed=seed).state.covariance
    spread = ideal.covariance - conditional
    d = np.clip(np.diag(spread), 0, None)
    sem_mean = np.sqrt(d / shots)
    sem_cov = np.sqrt((np.outer(d, d) + spread**2) / shots)
    out, runs = {}, {}
    for backend in EXEC_BACKENDS:
        result = pg.run_shots(pattern, shots, backend=backend, inputs=inputs, seed=seed)
        runs[backend] = result.trajectories
        ensemble = result.ensemble_state()
        out[f"{backend}_max_z_mean"] = float((np.abs(ensemble.mean - ideal.mean) / np.maximum(sem_mean, 1e-12)).max())
        out[f"{backend}_max_z_covariance"] = float(
            (np.abs(ensemble.covariance - ideal.covariance) / np.maximum(sem_cov, 1e-12)).max()
        )
    out["native_vs_gaussian_max_record_difference"] = max(
        abs(a.outcomes[k] - b.outcomes[k])
        for a, b in zip(runs["gaussian"], runs["native-gaussian"], strict=True)
        for k in a.outcomes
    )
    out["planner_backend"] = pg.select_backend(pattern, inputs).name
    out["shots"] = shots
    out["measurements"] = sum(isinstance(c, pg.Measure) for c in pattern.commands)
    return out


def execution_failed(row) -> bool:
    return (
        any(row[f"{b}_max_z_{m}"] > SEM_FACTOR for b in EXEC_BACKENDS for m in ("mean", "covariance"))
        or row["native_vs_gaussian_max_record_difference"] > NATIVE_TRAJECTORY_TOLERANCE
    )


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
            metadata_checks = graph_metadata_checks(pattern)
            row["graph_metadata_ok"] = all(v for k, v in metadata_checks.items() if k != "measurement_order")
            row["noise_trace"] = float(np.trace(channel.noise))
            if i < EXEC_SAMPLES:
                row.update(execute_compiled(pattern, channel, [EXEC_ALPHA]))
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
    bad += [r for r in successes if not r["graph_metadata_ok"]]
    executed = [r for r in successes if "planner_backend" in r]
    bad += [r for r in executed if execution_failed(r)]
    if bad:
        common.save_csv(bad, "R5_random_symplectic_compile_FAILURES")
        raise AssertionError(f"{len(bad)} compiled patterns failed a map/metadata/execution check; see FAILURES csv")

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
            "n_executed": len(executed),
            "execution_backends": list(EXEC_BACKENDS),
            "execution_alpha": [EXEC_ALPHA.real, EXEC_ALPHA.imag],
            "acceptance": {"map_tolerance": tolerance, "sem_factor": SEM_FACTOR,
                           "native_trajectory_tolerance": NATIVE_TRAJECTORY_TOLERANCE},
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

    max_z = max(max(r[f"{b}_max_z_mean"], r[f"{b}_max_z_covariance"]) for r in executed for b in EXEC_BACKENDS)
    noise = [r["noise_trace"] for r in successes]
    common.save_caption(
        "R5_random_symplectic_compile",
        f"""
**R5: random single-mode symplectic compilation.** Histogram of the Frobenius error between the
unconditional channel matrix S of the compiled MBQC pattern (`Circuit(1).gaussian(0, S).compile`,
resource squeezing r = {SQUEEZING}) and the target S = R(phi1) S(r) R(phi2) for {len(successes)} random
targets (seed {SEED}, phi uniform in (-pi, pi), logical r uniform in {R_RANGE}); {len(failed)}
decomposition failures. Maximum map error {max(r['frobenius_error'] for r in successes):.1e} (acceptance
<= {tolerance:g}); median log10 error {float(np.median(errors)):.2f}. The added finite-squeezing noise is
reported separately (Tr N between {min(noise):.3g} and {max(noise):.3g}) and is not part of this error.
All {len(successes)} compiled patterns pass the graph-metadata checks (edges/weights equal to the
Entangle commands, node squeezing equal to the Prepare commands, each non-output node measured once,
feed-forward after the measurement it reads, `OpenGraph` round trip). The first {len(executed)} patterns
were executed on `gaussian` and `native-gaussian` ({EXEC_SHOTS} shots each): ensemble moments within
{max_z:.2f} shot-noise standard errors of the analytic channel (acceptance <= {SEM_FACTOR:g}); the
planner selects `{executed[0]['planner_backend']}`. Source:
`paper_experiments/02_compiler/05_random_symplectic_compile.py`.
""",
    )

    common.print_summary(
        "R5 random symplectic compilation",
        n_samples=N_SAMPLES,
        n_successes=len(successes),
        n_decomposition_failures=len(failed),
        max_error=max((r["frobenius_error"] for r in successes), default=None),
        median_log10_error=float(np.median(errors)) if errors else None,
        graph_metadata_ok=all(r["graph_metadata_ok"] for r in successes),
        executed=len(executed),
        execution_max_z=max_z,
        execution_native_record_difference=max(r["native_vs_gaussian_max_record_difference"] for r in executed),
    )


if __name__ == "__main__":
    main()
