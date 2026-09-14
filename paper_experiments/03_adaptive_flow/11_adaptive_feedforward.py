"""R11: Adaptive feed-forward.

Builds a three-node line resource where the *measurement angle* of the
second homodyne detector depends on the first measurement's outcome
(``pg.Outcome("m0")``), in addition to the standard outcome-dependent
displacement corrections. This is genuinely adaptive: the affine
``gaussian_channel`` analyzer explicitly rejects outcome-dependent gate
angles (docs/theory.md), which is used here as a structural check that the
pattern is not secretly representable as a fixed-angle Gaussian channel.
Reproducibility is achieved by fixing the trajectory seed (Gaussian homodyne
has no explicit-outcome postselection API, unlike the Fock backends).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

SQUEEZING = 1.0
THETA_PARAMETER = 0.3
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]


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

    rows = []
    input_state = pg.GaussianInput.coherent(0.3 + 0.1j)
    for seed in SEEDS:
        result = pg.simulate(
            pattern, backend="gaussian", inputs={0: input_state}, parameters={"theta": THETA_PARAMETER}, seed=seed
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
            }
        )

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
    )


if __name__ == "__main__":
    main()
