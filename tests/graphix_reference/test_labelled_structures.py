"""Label/domain support only. Graphix is never a numerical CV oracle."""

import networkx as nx
import pytest

graphix = pytest.importorskip("graphix")
from graphix import command  # noqa: E402
from graphix.fundamentals import Plane  # noqa: E402
from graphix.opengraph import OpenGraph  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.commands import command_dependencies  # noqa: E402
from photographiq.cvflow import certify_cv_flow  # noqa: E402
from tests.graphix_reference.helpers import canonical_edges  # noqa: E402


@pytest.mark.parametrize(
    "graph",
    [
        nx.path_graph(5),
        nx.cycle_graph(5),
        nx.star_graph(4),
        nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, 3)),
        nx.relabel_nodes(nx.path_graph(4), {0: 17, 1: 3, 2: 41, 3: 8}),
    ],
)
def test_label_aware_graph_families_and_measurement_order(graph):
    nodes = tuple(graph)
    inputs = (nodes[0],)
    measured = nodes[:-1]
    cv = pg.Pattern(pg.CVGraph(graph, inputs=inputs))
    dv = graphix.Pattern(input_nodes=inputs)
    for node in nodes[1:]:
        dv.add(command.N(node))
    for edge in graph.edges():
        dv.add(command.E(edge))
    for node in measured:
        cv.measure(node)
        dv.add(command.M(node))
    assert canonical_edges(cv.graph.network) == canonical_edges(dv.to_opengraph().graph)
    assert set(cv.graph.nodes) == set(dv.to_opengraph().graph)
    assert cv.inputs == tuple(dv.input_nodes)
    assert cv.outputs == tuple(dv.output_nodes)
    assert [c.node for c in cv.commands if isinstance(c, pg.Measure)] == [
        c.node for c in dv if isinstance(c, command.M)
    ]
    cv.validate()


def test_arbitrary_cv_labels_with_explicit_graphix_bijection():
    labels = ("input", ("ancilla", 1), 97, "output")
    mapping = dict(enumerate(labels))
    graph = nx.relabel_nodes(nx.path_graph(4), mapping)
    cv = pg.Pattern(pg.CVGraph(graph, inputs=(labels[0],))).measure(labels[0]).measure(labels[1])
    dv = graphix.Pattern(input_nodes=[0])
    for n in (1, 2, 3):
        dv.add(command.N(n))
    for e in ((0, 1), (1, 2), (2, 3)):
        dv.add(command.E(e))
    for n in (0, 1):
        dv.add(command.M(n))
    labelled = nx.relabel_nodes(dv.to_opengraph().graph, mapping)
    assert canonical_edges(cv.graph.network) == canonical_edges(labelled)
    assert cv.outputs == tuple(mapping[n] for n in dv.output_nodes)
    assert cv.inputs == tuple(mapping[n] for n in dv.input_nodes)
    # A symmetric graph can remain isomorphic under an incorrect semantic map.
    wrong = nx.relabel_nodes(labelled, {labels[1]: labels[3], labels[3]: labels[1]})
    assert nx.is_isomorphic(wrong, cv.graph.network)
    assert canonical_edges(wrong) != canonical_edges(cv.graph.network)


def test_multiple_adaptive_domains_corrections_and_causal_order():
    cv = pg.Pattern(pg.CVGraph.line(4, inputs=(0,))).measure(0)
    cv.measure(1, pg.Homodyne(0.2 + pg.Outcome(0)))
    cv.displace(2, q=pg.Outcome(0)).displace(3, p=pg.Outcome(1))
    cv.append(pg.QuadraticPhase(2, pg.Outcome(0) + pg.Outcome(1) ** 2))
    dv = graphix.Pattern(input_nodes=[0])
    for n in (1, 2, 3):
        dv.add(command.N(n))
    for e in ((0, 1), (1, 2), (2, 3)):
        dv.add(command.E(e))
    dv.add(command.M(0))
    dv.add(command.M(1, s_domain={0}))
    dv.add(command.X(2, domain={0}))
    dv.add(command.Z(3, domain={1}))
    dv.add(command.X(2, domain={0, 1}))
    cv.validate()
    dv_commands = list(dv)
    assert command_dependencies(cv.commands[7]) == dv_commands[7].s_domain
    for c, d in zip(cv.commands[8:], dv_commands[8:], strict=True):
        assert c.node == d.node
        assert command_dependencies(c) == d.domain
    graph = cv.dependencies()
    order = cv.schedule()
    assert all(order.index(a) < order.index(b) for a, b in graph.edges())
    assert graph.has_edge(6, 7)
    assert graph.has_edge(6, 10) and graph.has_edge(7, 10)
    bad = cv.copy()
    bad.commands[6], bad.commands[7] = bad.commands[7], bad.commands[6]
    with pytest.raises(ValueError, match="future|Cyclic"):
        bad.validate()
    # Graphix can store domains before validating/scheduling; do not assume
    # its command constructor enforces PhotoGraphiQ's execution-time causality.


def test_s_t_domains_and_transitive_classical_signal_support():
    cv = pg.Pattern(pg.CVGraph.line(4, inputs=(0,))).measure(0)
    cv.measure(1, pg.Homodyne(pg.Outcome(0)))
    cv.measure(2, pg.Homodyne(pg.Outcome(0) + pg.Outcome(1)))
    cv.append(pg.Signal("combined", pg.Outcome(0) + pg.Outcome(1)))
    cv.displace(3, p=pg.Outcome("combined") + pg.Outcome(2))
    measurement = command.M(2, s_domain={0}, t_domain={1})
    correction = command.Z(3, domain={0, 1, 2})
    assert command_dependencies(cv.commands[8]) == measurement.s_domain | measurement.t_domain
    support = set(command_dependencies(cv.commands[-1]))
    support.remove("combined")
    support.update(command_dependencies(cv.commands[-2]))
    assert support == correction.domain
    assert cv.commands[-1].node == correction.node
    cv.validate()
    # Compare supports only; real-valued Signal addition is not Graphix XOR.


@pytest.mark.parametrize("length", [3, 4, 5])
def test_line_flow_support_and_standardization(length):
    cvgraph = pg.CVGraph.line(length, inputs=(0,), outputs=(length - 1,))
    flow = certify_cv_flow(cvgraph, range(length - 1))
    dvflow = (
        OpenGraph(
            nx.path_graph(length), [0], [length - 1], {n: Plane.XY for n in range(length - 1)}
        )
        .to_causalflow()
        .to_xzcorrections()
    )
    assert flow is not None
    for node, coefficients in flow.corrections.items():
        assert {n for n, value in coefficients.items() if abs(value) > 1e-10} == set(
            dvflow.x_corrections[node]
        )
    assert certify_cv_flow(cvgraph, reversed(range(length - 1))) is None
    cv = pg.Pattern().extend([pg.Prepare(0), pg.Prepare(1), pg.Entangle(0, 1), pg.Prepare(2)])
    dv = graphix.Pattern(cmds=[command.N(0), command.N(1), command.E((0, 1)), command.N(2)])
    dv.standardize()
    standard = cv.standardize()
    assert canonical_edges(standard.graph.network) == canonical_edges(dv.to_opengraph().graph)
    assert standard.outputs == tuple(dv.output_nodes)
    assert [c.node for c in standard.commands if isinstance(c, pg.Prepare)] == [
        c.node for c in dv if isinstance(c, command.N)
    ]
