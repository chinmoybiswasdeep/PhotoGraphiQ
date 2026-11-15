"""R10: Compiler resource scaling.

Builds circuits with an increasing number of repeated gates (rotation,
squeezing, CZ, beamsplitter) and records how compiled resource-node count,
measurement count, entanglement-edge count, command count and compile time
grow. Beamsplitter compilation is a fixed, angle-independent SUM-gate
decomposition that is markedly more resource-expensive per gate (see R7), so
its gate-count sweep is capped lower to keep total runtime practical.

Upgrade migration (2026-11): the LARGEST compiled pattern of every gate kind
is additionally (a) checked against the independently composed ideal map
(rotations/squeezers/CZs/beamsplitters with summed parameters commute within
each kind), with the added noise reported separately, and (b) executed once
on NumPy ``gaussian`` and C++ ``native-gaussian`` with the same seed, which
must give the same homodyne records (documented RNG contract). Compile times
are wall-clock observations on a shared machine (provisional).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

SQUEEZING = 1.0
GATE_COUNTS = [1, 2, 5, 10, 20, 50, 100]
BEAMSPLITTER_COUNTS = [1, 2, 5, 10, 20]
# RELATIVE tolerances: 100 chained squeezers reach total r = 29.75, so the map entries are
# ~e^29.75 = 8e12 and the records/means scale likewise; absolute float64 errors are not
# meaningful there (conditioning), relative ones are.
MAP_TOLERANCE = 1e-9  # ||S - S_target|| / ||S_target||
NATIVE_TRAJECTORY_TOLERANCE = 1e-9  # |record_native - record_numpy| / max(1, |record_numpy|)


def ideal_map(kind: str, n: int) -> np.ndarray:
    """Independent composition: parameters of one gate kind add (all gates commute)."""
    if kind == "rotation":
        t = sum(0.1 + 0.01 * i for i in range(n))
        return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    if kind == "squeeze":
        r = sum(0.05 + 0.005 * i for i in range(n))
        return np.diag([np.exp(-r), np.exp(r)])
    if kind == "cz":
        g = sum(0.3 + 0.01 * i for i in range(n))
        m = np.eye(4)
        m[1, 2] = m[3, 0] = g
        return m
    t = sum(0.2 + 0.01 * i for i in range(n))
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, -s, 0], [0, c, 0, -s], [s, 0, c, 0], [0, s, 0, c]])


def build(kind: str, n: int) -> pg.Circuit:
    if kind in ("rotation", "squeeze"):
        circuit = pg.Circuit(1)
        for i in range(n):
            if kind == "rotation":
                circuit.rotate(0, 0.1 + 0.01 * i)
            else:
                circuit.squeeze(0, 0.05 + 0.005 * i)
        return circuit
    circuit = pg.Circuit(2)
    for i in range(n):
        if kind == "cz":
            circuit.cz(0, 1, 0.3 + 0.01 * i)
        else:
            circuit.beamsplitter(0, 1, 0.2 + 0.01 * i)
    return circuit


def measure_compile(circuit):
    repeats = 3 if len(circuit.gates) <= 20 else 1
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = circuit.compile(squeezing=SQUEEZING, return_trace=True)
        times.append(time.perf_counter() - start)
    return result, times


def main():
    plt = common.setup_style()
    all_rows = []
    for kind, counts in (
        ("rotation", GATE_COUNTS),
        ("squeeze", GATE_COUNTS),
        ("cz", GATE_COUNTS),
        ("beamsplitter", BEAMSPLITTER_COUNTS),
    ):
        for n in counts:
            circuit = build(kind, n)
            (pattern, trace), times = measure_compile(circuit)
            entangle_edges = sum(1 for c in pattern.commands if isinstance(c, pg.Entangle))
            measurements = sum(1 for c in pattern.commands if isinstance(c, pg.Measure))
            row = {
                "gate_kind": kind,
                "gate_count": n,
                "resource_nodes": len(pattern.graph.nodes) - circuit.modes,
                "measurements": measurements,
                "edges": entangle_edges,
                "commands": len(pattern.commands),
                "compile_time_median_seconds": float(np.median(times)) if len(times) > 1 else times[0],
                "compile_time_seconds_raw": times,
            }
            all_rows.append(row)
            print(f"  {kind} n={n}: {row['resource_nodes']} nodes, {row['compile_time_median_seconds']:.4f}s", flush=True)

    largest_rows = []
    # The 100-squeezer chain has total r = 29.75 (cond(S) = e^59.5 ~ 7e25, ~258 dB): float64
    # cannot make two engines agree trajectory by trajectory there (record differences of
    # ~eps * cond are expected), so its record agreement is RECORDED but not asserted; the
    # trajectory contract is asserted on the 20-squeezer chain (r = 1.95) instead.
    for kind, n in (("rotation", 100), ("squeeze", 20), ("squeeze", 100), ("cz", 100), ("beamsplitter", 20)):
        pattern = build(kind, n).compile(squeezing=SQUEEZING)
        channel = pg.gaussian_channel(pattern)
        runs = {b: pg.simulate(pattern, backend=b, seed=11) for b in ("gaussian", "native-gaussian")}
        largest_rows.append(
            {
                "gate_kind": kind,
                "gate_count": n,
                "map_frobenius_error": common.frobenius_error(channel.matrix, ideal_map(kind, n)),
                "map_relative_error": common.relative_error(channel.matrix, ideal_map(kind, n)),
                "target_condition_number": float(np.linalg.cond(ideal_map(kind, n))),
                "noise_trace": float(np.trace(channel.noise)),
                "measurements": sum(isinstance(c, pg.Measure) for c in pattern.commands),
                "native_vs_gaussian_max_record_difference": max(
                    abs(runs["gaussian"].outcomes[k] - runs["native-gaussian"].outcomes[k]) for k in runs["gaussian"].outcomes
                ),
                "native_vs_gaussian_max_relative_record_difference": max(
                    abs(runs["gaussian"].outcomes[k] - runs["native-gaussian"].outcomes[k])
                    / max(1.0, abs(runs["gaussian"].outcomes[k]))
                    for k in runs["gaussian"].outcomes
                ),
                "native_vs_gaussian_output_mean_difference": float(
                    np.abs(runs["gaussian"].state.mean - runs["native-gaussian"].state.mean).max()
                ),
            }
        )
    common.save_csv(largest_rows, "R10_compiler_scaling_largest_validation")
    common.save_result(all_rows, "R10_compiler_scaling",
                       extra={"largest_pattern_validation": largest_rows, "timing_note": "provisional (shared machine)",
                              "map_tolerance": MAP_TOLERANCE, "native_trajectory_tolerance": NATIVE_TRAJECTORY_TOLERANCE})
    for r in largest_rows:
        r["trajectory_contract_asserted"] = r["target_condition_number"] <= 1e8
    bad = [r for r in largest_rows if r["map_relative_error"] > MAP_TOLERANCE
           or (r["trajectory_contract_asserted"]
               and r["native_vs_gaussian_max_relative_record_difference"] > NATIVE_TRAJECTORY_TOLERANCE)]
    if bad:
        common.save_csv(bad, "R10_compiler_scaling_FAILURES")
        raise AssertionError("Largest compiled pattern failed map or native-execution check; see FAILURES csv")

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
    metrics = [
        ("resource_nodes", "Resource nodes"),
        ("measurements", "Measurements"),
        ("commands", "Commands"),
        ("compile_time_median_seconds", "Compile time (s)"),
    ]
    for ax, (key, label) in zip(axes.flat, metrics, strict=True):
        for kind in ("rotation", "squeeze", "cz", "beamsplitter"):
            rows = [r for r in all_rows if r["gate_kind"] == kind]
            ax.plot([r["gate_count"] for r in rows], [r[key] for r in rows], "o-", label=kind, markersize=4)
        ax.set_xlabel("Number of gates")
        ax.set_ylabel(label)
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Compiler resource scaling")
    common.save_figure(fig, "R10_compiler_scaling")
    plt.close(fig)

    common.print_summary(
        "R10 compiler resource scaling",
        rows=len(all_rows),
        largest_max_relative_map_error=max(r["map_relative_error"] for r in largest_rows),
        largest_max_relative_native_record_difference=max(
            r["native_vs_gaussian_max_relative_record_difference"] for r in largest_rows if r["trajectory_contract_asserted"]
        ),
        unasserted_ill_conditioned=[(r["gate_kind"], r["gate_count"], r["native_vs_gaussian_max_relative_record_difference"])
                                    for r in largest_rows if not r["trajectory_contract_asserted"]],
    )


if __name__ == "__main__":
    main()
