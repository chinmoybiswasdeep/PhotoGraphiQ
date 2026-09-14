"""R12: Dependency DAG / causal schedule.

Builds a four-node adaptive pattern with multiple classical dependencies
(two homodyne outcomes feeding three downstream corrections/measurements),
extracts ``Pattern.dependencies()`` (a command DAG) and
``Pattern.schedule()`` (a topological command ordering), and checks that
every dependency edge points strictly backward-to-forward in that schedule.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

SQUEEZING = 1.0


def build_pattern():
    graph = pg.CVGraph.line(4, squeezing=SQUEEZING, inputs=(0,))
    pattern = pg.Pattern(graph)
    pattern.measure(0, pg.Homodyne.q(), key="m0")
    pattern.displace(1, q=-pg.Outcome("m0"))
    pattern.measure(1, pg.Homodyne(angle=0.1 * pg.Outcome("m0")), key="m1")
    # Node 2's correction depends on BOTH earlier outcomes.
    pattern.displace(2, q=-pg.Outcome("m1"), p=0.2 * pg.Outcome("m0"))
    pattern.measure(2, pg.Homodyne.p(), key="m2")
    pattern.displace(3, q=-pg.Outcome("m2"), p=-pg.Outcome("m0") + pg.Outcome("m1"))
    pattern.append(pg.Output((3,)))
    return pattern.validate()


def main():
    common.setup_style()
    pattern = build_pattern()
    dag = pattern.dependencies()
    schedule = pattern.schedule()

    position = {command_index: position for position, command_index in enumerate(schedule)}
    edges = sorted(dag.edges())
    violations = [(u, v) for u, v in edges if position[u] >= position[v]]

    rows = [
        {
            "command_index": i,
            "command_type": type(c).__name__,
            "schedule_position": position[i],
            "dependencies": sorted(dag.predecessors(i)),
        }
        for i, c in enumerate(pattern.commands)
    ]

    common.save_result(
        rows,
        "R12_dependency_dag",
        extra={
            "dependency_edges": edges,
            "topological_schedule": list(schedule),
            "backward_violations": violations,
            "pattern_json": pattern.to_json(),
        },
    )

    if violations:
        common.save_json(violations, "R12_dependency_dag_VIOLATIONS")
        raise AssertionError(f"{len(violations)} dependency edges violate the topological schedule")

    from photographiq.visualization import draw_dependencies

    ax = draw_dependencies(pattern)
    common.save_figure(ax.figure, "R12_dependency_dag")
    import matplotlib.pyplot as plt

    plt.close(ax.figure)

    common.print_summary(
        "R12 dependency DAG / causal schedule",
        commands=len(pattern.commands),
        dependency_edges=len(edges),
        backward_violations=len(violations),
    )


if __name__ == "__main__":
    main()
