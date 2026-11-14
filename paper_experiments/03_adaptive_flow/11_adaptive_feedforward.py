"""R11: Adaptive feed-forward.

Builds a three-node line resource where the *measurement angle* of the
second homodyne detector depends on the first measurement's outcome
(``pg.Outcome("m0")``), in addition to the standard outcome-dependent
displacement corrections. This is genuinely adaptive: the affine
``gaussian_channel`` analyzer explicitly rejects outcome-dependent gate
angles (docs/theory.md), which is used here as a structural check that the
pattern is not secretly representable as a fixed-angle Gaussian channel.
Reproducibility is achieved by fixing the trajectory seed.

Upgrade migration (2026-11): the historical docstring said Gaussian homodyne
had no explicit-outcome API; ``simulate(..., measurement_outcomes=...)`` now
postselects homodyne records on the Gaussian engines. The script therefore
also (1) runs every seed on NumPy ``gaussian`` and C++ ``native-gaussian``
(records must agree, documented RNG contract), (2) evaluates a grid of
FIXED outcomes (m0, m1) on both engines and compares each conditional output
with a non-adaptive control pattern whose second angle is set by hand to
0.05*m0 + theta -- equality proves the angle was resolved from the earlier
outcome, (3) checks that different m0 give different conditional outputs (the
feed-forward is not inert) and (4) checks the measurement order in
``Pattern.schedule()``. Postselected runs are conditional branches, not
deterministic execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import numpy as np  # noqa: E402

import photographiq as pg  # noqa: E402

SQUEEZING = 1.0
THETA_PARAMETER = 0.3
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
EXEC_BACKENDS = ("gaussian", "native-gaussian")
FIXED_M0 = [-1.5, -0.5, 0.0, 0.7, 2.0]
FIXED_M1 = [-0.8, 0.4]
CONTROL_TOLERANCE = 1e-12  # adaptive branch == hand-built fixed-angle branch
NATIVE_TOLERANCE = 1e-10


def build_control(angle: float):
    """Same pattern with the second homodyne angle fixed by hand (no adaptivity)."""
    graph = pg.CVGraph.line(3, squeezing=SQUEEZING, inputs=(0,))
    pattern = pg.Pattern(graph)
    pattern.measure(0, pg.Homodyne.q(), key="m0")
    pattern.displace(1, q=-pg.Outcome("m0"))
    pattern.measure(1, pg.Homodyne(angle=angle), key="m1")
    pattern.displace(2, q=-pg.Outcome("m1"))
    pattern.append(pg.Output((2,)))
    return pattern.validate()


def build_pattern():
    graph = pg.CVGraph.line(3, squeezing=SQUEEZING, inputs=(0,))
    pattern = pg.Pattern(graph)
    pattern.measure(0, pg.Homodyne.q(), key="m0")
    pattern.displace(1, q=-pg.Outcome("m0"))
    # Adaptive: the homodyne angle itself is a function of the earlier outcome.
    pattern.measure(1, pg.Homodyne(angle=0.05 * pg.Outcome("m0") + pg.Parameter("theta")), key="m1")
    pattern.displace(2, q=-pg.Outcome("m1"))
    pattern.append(pg.Output((2,)))
    return pattern.validate()


def main():
    common.setup_style()
    pattern = build_pattern()

    dependency_graph = pattern.dependencies()
    m0_index = next(i for i, c in enumerate(pattern.commands) if isinstance(c, pg.Measure) and c.result_key == "m0")
    m1_index = next(i for i, c in enumerate(pattern.commands) if isinstance(c, pg.Measure) and c.result_key == "m1")
    has_dependency_edge = dependency_graph.has_edge(m0_index, m1_index)
    if not has_dependency_edge:
        raise AssertionError("Expected a causal dependency edge from m0's measurement to m1's")

    # Structural proof of genuine adaptivity: the affine analyzer must reject
    # this pattern because the second homodyne angle depends on an outcome.
    rejected_by_affine_analyzer = False
    try:
        pg.gaussian_channel(pattern)
    except NotImplementedError:
        rejected_by_affine_analyzer = True
    if not rejected_by_affine_analyzer:
        raise AssertionError("Expected the affine Gaussian-channel analyzer to reject this adaptive pattern")

    schedule = list(pattern.schedule())
    measured_in_order = schedule.index(m0_index) < schedule.index(m1_index)
    if not measured_in_order:
        raise AssertionError("Pattern.schedule() measures m1 before m0")

    rows = []
    input_state = pg.GaussianInput.coherent(0.3 + 0.1j)
    for seed in SEEDS:
        result = pg.simulate(
            pattern, backend="gaussian", inputs={0: input_state}, parameters={"theta": THETA_PARAMETER}, seed=seed
        )
        native = pg.simulate(
            pattern, backend="native-gaussian", inputs={0: input_state}, parameters={"theta": THETA_PARAMETER}, seed=seed
        )
        native_difference = max(
            max(abs(result.records[k] - native.records[k]) for k in ("m0", "m1")),
            float(np.abs(result.state.mean - native.state.mean).max()),
        )
        expected_angle = 0.05 * result.records["m0"] + THETA_PARAMETER
        rows.append(
            {
                "seed": seed,
                "m0_outcome": result.records["m0"],
                "m1_outcome": result.records["m1"],
                "adaptive_angle_used": expected_angle,
                "output_q_mean": result.state.quadrature(2)[0],
                "output_p_mean": result.state.quadrature(2, expected_angle)[0],
                "native_vs_gaussian_max_difference": native_difference,
            }
        )

    branch_rows = []
    for m0 in FIXED_M0:
        angle = 0.05 * m0 + THETA_PARAMETER
        control = build_control(angle)
        for m1 in FIXED_M1:
            outcomes = {"m0": m0, "m1": m1}
            row = {"m0": m0, "m1": m1, "adaptive_angle": angle}
            for backend in EXEC_BACKENDS:
                adaptive = pg.simulate(
                    pattern, backend=backend, inputs={0: input_state}, parameters={"theta": THETA_PARAMETER},
                    measurement_outcomes=outcomes, seed=0,
                )
                fixed = pg.simulate(control, backend=backend, inputs={0: input_state}, measurement_outcomes=outcomes, seed=0)
                row[f"{backend}_output_q_mean"] = float(adaptive.state.mean[0])
                row[f"{backend}_output_p_mean"] = float(adaptive.state.mean[1])
                row[f"{backend}_m1_density"] = adaptive.measurement_statistics["m1"]["value"]
                row[f"{backend}_vs_fixed_angle_control"] = max(
                    common.frobenius_error(adaptive.state.mean, fixed.state.mean),
                    common.frobenius_error(adaptive.state.covariance, fixed.state.covariance),
                    abs(adaptive.measurement_statistics["m1"]["value"] - fixed.measurement_statistics["m1"]["value"]),
                )
            row["native_vs_gaussian"] = max(
                abs(row[f"gaussian_{k}"] - row[f"native-gaussian_{k}"])
                for k in ("output_q_mean", "output_p_mean", "m1_density")
            )
            branch_rows.append(row)
    common.save_csv(branch_rows, "R11_adaptive_feedforward_branches")
    distinct_outputs = len(
        {(round(r["gaussian_output_q_mean"], 9), round(r["gaussian_output_p_mean"], 9)) for r in branch_rows}
    )
    failures = [r for r in rows if r["native_vs_gaussian_max_difference"] > NATIVE_TOLERANCE]
    failures += [
        r for r in branch_rows
        if max(r[f"{b}_vs_fixed_angle_control"] for b in EXEC_BACKENDS) > CONTROL_TOLERANCE
        or r["native_vs_gaussian"] > NATIVE_TOLERANCE
    ]
    if distinct_outputs < len(branch_rows):
        failures.append({"check": "feed-forward inert", "distinct_outputs": distinct_outputs})
    if failures:
        common.save_csv(failures, "R11_adaptive_feedforward_FAILURES")
        raise AssertionError(f"{len(failures)} adaptive feed-forward checks failed; see FAILURES csv")

    common.save_result(
        rows,
        "R11_adaptive_feedforward",
        extra={
            "pattern_json": pattern.to_json(),
            "dependency_edges": sorted(dependency_graph.edges()),
            "m0_command_index": m0_index,
            "m1_command_index": m1_index,
            "affine_analyzer_rejected_pattern": rejected_by_affine_analyzer,
            "theta_parameter": THETA_PARAMETER,
            "schedule_measures_m0_before_m1": measured_in_order,
            "execution_backends": list(EXEC_BACKENDS),
            "fixed_outcome_branches": branch_rows,
            "acceptance": {"control_tolerance": CONTROL_TOLERANCE, "native_tolerance": NATIVE_TOLERANCE},
        },
    )
    common.save_raw(pattern.to_json(), "R11_adaptive_feedforward_pattern.json")

    from photographiq.visualization import draw_dependencies, draw_pattern

    ax = draw_pattern(pattern)
    common.save_figure(ax.figure, "R11_adaptive_feedforward_pattern")
    import matplotlib.pyplot as plt

    plt.close(ax.figure)

    ax = draw_dependencies(pattern)
    common.save_figure(ax.figure, "R11_adaptive_feedforward_dependencies")
    plt.close(ax.figure)

    common.print_summary(
        "R11 adaptive feed-forward",
        seeds=len(rows),
        dependency_edge_m0_to_m1=has_dependency_edge,
        affine_analyzer_rejected_pattern=rejected_by_affine_analyzer,
        schedule_measures_m0_before_m1=measured_in_order,
        fixed_outcome_branches=len(branch_rows),
        distinct_conditional_outputs=distinct_outputs,
        max_adaptive_vs_control=max(max(r[f"{b}_vs_fixed_angle_control"] for b in EXEC_BACKENDS) for r in branch_rows),
        max_native_vs_gaussian=max(
            max(r["native_vs_gaussian_max_difference"] for r in rows), max(r["native_vs_gaussian"] for r in branch_rows)
        ),
    )


if __name__ == "__main__":
    main()
