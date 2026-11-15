"""R12: Dependency DAG / causal schedule.

Builds a four-node adaptive pattern with multiple classical dependencies
(two homodyne outcomes feeding three downstream corrections/measurements),
extracts ``Pattern.dependencies()`` (a command DAG) and
``Pattern.schedule()`` (a topological command ordering), and checks that
every dependency edge points strictly backward-to-forward in that schedule.

Upgrade migration (2026-11): the schedule is also EXECUTED. The commands are
re-assembled in ``Pattern.schedule()`` order into a new pattern, which must
validate and must reproduce the original pattern's seeded records and output
moments on NumPy ``gaussian`` and C++ ``native-gaussian`` for several seeds
(different outcome branches). Records of the two engines must agree
(documented RNG contract).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import numpy as np  # noqa: E402

import photographiq as pg  # noqa: E402

SQUEEZING = 1.0
SEEDS = [3, 5, 8, 13, 21]
EXEC_BACKENDS = ("gaussian", "native-gaussian")
EXECUTION_TOLERANCE = 1e-10


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

    scheduled = pg.Pattern(inputs=pattern.inputs)
    scheduled.extend([pattern.commands[i] for i in schedule])
    scheduled.validate()
    inputs = {0: pg.GaussianInput.coherent(0.2 - 0.3j)}
    execution_rows = []
    for seed in SEEDS:
        runs = {}
        for backend in EXEC_BACKENDS:
            runs[(backend, "listed")] = pg.simulate(pattern, backend=backend, inputs=inputs, seed=seed)
            runs[(backend, "scheduled")] = pg.simulate(scheduled, backend=backend, inputs=inputs, seed=seed)
        reference = runs[("gaussian", "listed")]
        worst = 0.0
        for result in runs.values():
            worst = max(
                worst,
                max(abs(result.records[k] - reference.records[k]) for k in reference.records),
                float(np.abs(result.state.mean - reference.state.mean).max()),
                float(np.abs(result.state.covariance - reference.state.covariance).max()),
            )
        execution_rows.append({"seed": seed, **{k: float(v) for k, v in reference.records.items()},
                               "max_difference_over_order_and_backend": worst})
    # Pattern.schedule() returns the listed order for this pattern, so additionally run a
    # genuinely different topological order of the same DAG (latest-index-first tie
    # breaking) on fixed outcome branches (postselection makes the comparison independent
    # of the order in which random numbers are drawn).
    import networkx as nx

    alternative = list(nx.lexicographical_topological_sort(dag, key=lambda i: -i))
    reordered = pg.Pattern(inputs=pattern.inputs)
    reordered.extend([pattern.commands[i] for i in alternative])
    reordered.validate()
    for branch in ({"m0": 0.4, "m1": -1.1, "m2": 0.3}, {"m0": -2.0, "m1": 0.5, "m2": 1.7}):
        results = [
            pg.simulate(p, backend=b, inputs=inputs, measurement_outcomes=branch, seed=0)
            for p in (pattern, reordered) for b in EXEC_BACKENDS
        ]
        worst = max(
            max(float(np.abs(r.state.mean - results[0].state.mean).max()),
                float(np.abs(r.state.covariance - results[0].state.covariance).max()))
            for r in results
        )
        execution_rows.append({"seed": None, "branch": branch, "alternative_order": alternative,
                               "max_difference_over_order_and_backend": worst})
    common.save_csv(execution_rows, "R12_dependency_dag_execution")
    bad = [r for r in execution_rows if r["max_difference_over_order_and_backend"] > EXECUTION_TOLERANCE]
    bad += [] if alternative != list(range(len(pattern.commands))) else [{"check": "alternative order is trivial"}]
    if bad:
        common.save_csv(bad, "R12_dependency_dag_FAILURES")
        raise AssertionError("Scheduled reordering or native engine changed the seeded execution; see FAILURES csv")

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
        schedule_differs_from_listed_order=list(schedule) != list(range(len(pattern.commands))),
        scheduled_execution_seeds=len(execution_rows),
        max_difference_over_order_and_backend=max(r["max_difference_over_order_and_backend"] for r in execution_rows),
    )


if __name__ == "__main__":
    main()
