"""R17: Graphix structural comparison (structure only, never CV amplitudes).

For a line and a star resource topology, builds a hand-matched Graphix
qubit-MBQC ``Pattern`` (N/E/M/X/Z commands) alongside the equivalent
PhotoGraphiQ CV pattern, and compares only *structural* invariants: resource
graph isomorphism/edges, input/output node sets, measurement order, and
classical-dependency domains. Continuous-variable amplitudes, quadrature
statistics or any numerical physics are never compared against Graphix,
per the package's documented validation policy (docs/validation/index.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

if not common.HAS_GRAPHIX:
    print("SKIPPED: graphix is not installed; see requirements_notes.md")
    sys.exit(0)

import graphix  # noqa: E402
from graphix import command  # noqa: E402

import photographiq as pg  # noqa: E402


def build_line_case(n: int):
    graph = pg.CVGraph.line(n, inputs=(0,))
    cv = pg.Pattern(graph)
    cv.measure(0, pg.Homodyne.q())
    for i in range(1, n - 1):
        cv.measure(i, pg.Homodyne(0.2 + 0.1 * i + pg.Outcome(i - 1)))
    cv.displace(n - 1, q=pg.Outcome(n - 2), p=(pg.Outcome(0) if n > 2 else 0.0))
    cv.append(pg.Output((n - 1,)))
    cv.validate()

    dv = graphix.Pattern(input_nodes=[0])
    for i in range(1, n):
        dv.add(command.N(i))
    for i in range(n - 1):
        dv.add(command.E((i, i + 1)))
    dv.add(command.M(0))
    for i in range(1, n - 1):
        dv.add(command.M(i, s_domain={i - 1}))
    dv.add(command.X(n - 1, domain={n - 2}))
    if n > 2:
        dv.add(command.Z(n - 1, domain={0}))
    return cv, dv


def build_star_case(leaves: int):
    graph = pg.CVGraph.star(leaves, inputs=(0,))
    leaf_nodes = tuple(range(1, leaves + 1))
    cv = pg.Pattern(graph)
    cv.measure(0, pg.Homodyne.q())
    for leaf in leaf_nodes:
        cv.displace(leaf, q=pg.Outcome(0))
    cv.append(pg.Output(leaf_nodes))
    cv.validate()

    dv = graphix.Pattern(input_nodes=[0])
    for leaf in leaf_nodes:
        dv.add(command.N(leaf))
    for leaf in leaf_nodes:
        dv.add(command.E((0, leaf)))
    dv.add(command.M(0))
    for leaf in leaf_nodes:
        dv.add(command.X(leaf, domain={0}))
    return cv, dv


def compare(name, cv, dv):
    og = dv.to_opengraph()
    isomorphic = nx.is_isomorphic(og.graph, cv.graph.network)
    edges_match = common.canonical_edges(og.graph) == common.canonical_edges(cv.graph.network)
    io_match = tuple(dv.input_nodes) == cv.inputs and tuple(dv.output_nodes) == cv.outputs
    graphix_measure_order = [c.node for c in dv if isinstance(c, command.M)]
    cv_measure_order = [c.node for c in cv.commands if isinstance(c, pg.Measure)]
    order_match = graphix_measure_order == cv_measure_order

    cv_measurements = [(i, c) for i, c in enumerate(cv.commands) if isinstance(c, pg.Measure)]
    dependency_graph = cv.dependencies()
    domain_checks = []
    for gx_command in dv:
        if isinstance(gx_command, command.M) and gx_command.s_domain:
            target_index = next(i for i, c in cv_measurements if c.node == gx_command.node)
            for source_node in gx_command.s_domain:
                source_index = next(i for i, c in cv_measurements if c.node == source_node)
                domain_checks.append(dependency_graph.has_edge(source_index, target_index))
        if isinstance(gx_command, command.X) and gx_command.domain:
            for source_node in gx_command.domain:
                source_index = next(i for i, c in cv_measurements if c.node == source_node)
                # The corresponding correction is the Displace on gx_command.node.
                target_index = next(
                    i for i, c in enumerate(cv.commands)
                    if isinstance(c, pg.Displace) and c.node == gx_command.node
                )
                domain_checks.append(dependency_graph.has_edge(source_index, target_index))
    domains_match = all(domain_checks) and len(domain_checks) > 0

    return {
        "case": name,
        "nodes": len(cv.graph.nodes),
        "edges": cv.graph.network.number_of_edges(),
        "graph_isomorphic": isomorphic,
        "canonical_edges_match": edges_match,
        "input_output_match": io_match,
        "measurement_order_match": order_match,
        "dependency_domain_checks": len(domain_checks),
        "dependency_domains_match": domains_match,
    }


def main():
    rows = [
        compare("line_n3", *build_line_case(3)),
        compare("line_n5", *build_line_case(5)),
        compare("star_leaves_3", *build_star_case(3)),
        compare("star_leaves_5", *build_star_case(5)),
    ]

    failures = [
        r for r in rows
        if not (r["graph_isomorphic"] and r["canonical_edges_match"] and r["input_output_match"]
                and r["measurement_order_match"] and r["dependency_domains_match"])
    ]
    if failures:
        common.save_csv(failures, "R17_graphix_structure_FAILURES")
        raise AssertionError("Structural comparison against Graphix failed; see FAILURES csv")

    common.save_result(
        rows,
        "R17_graphix_structure",
        extra={
            "note": "Structural comparison only: node/edge/I-O/order/domain equivalence. "
            "Graphix never validates CV amplitudes or quadrature statistics."
        },
    )

    common.print_summary("R17 Graphix structural comparison", cases=len(rows), all_passed=not failures)


if __name__ == "__main__":
    main()
