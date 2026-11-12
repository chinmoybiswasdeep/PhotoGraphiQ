"""Staged interactive graph view: semantics, overlays, traces, editing and exports."""

import json
import math
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

import photographiq as pg
from photographiq.cvflow import certify_cv_flow, flow_pattern
from photographiq.expressions import CallableExpression, Outcome
from photographiq.interactive import (
    LAYOUTS,
    GraphView,
    Layout,
    apply_edits,
    compute_layout,
    execution_trace,
    export_layout,
    import_layout,
    normalize_corrections,
    pattern_corrections,
)
from photographiq.ir import HomodyneSpec, Model, OpenGraph, QubitMeasurement

NODE = shutil.which("node") or "C:/Program Files/nodejs/node.exe"
ORDER = (0, 1)
SHEARS = {0: 0.3, 1: -0.5}


@pytest.fixture(scope="module")
def cv_case():
    graph = pg.CVGraph.line(3, squeezing=1.5, inputs=(0,), outputs=(2,))
    pattern = flow_pattern(graph, ORDER, shears=SHEARS)
    result = pg.simulate(pattern, backend="gaussian", seed=11)
    return graph, pattern, result


def qubit_graph() -> OpenGraph:
    measurements = {0: QubitMeasurement("XY", 0.3), 1: QubitMeasurement("XY", 0.0, "X")}
    return OpenGraph(Model.QUBIT, nx.path_graph(3), (0,), (2,), measurements)


# Flow of the line 0-1-2 (f(0) = 1, f(1) = 2) in the FlowAnalysisResult convention:
# X on f(u), Z on Odd(f(u)) minus u, restricted to later nodes; written out as data.
QUBIT_FLOW = SimpleNamespace(
    status="certified",
    model=Model.QUBIT,
    algorithm="gflow",
    measurement_order=(0, 1),
    layers=((0,), (1,)),
    correction_map={0: {"x": frozenset({1}), "z": frozenset({2})}, 1: {"x": {2}, "z": set()}},
    certificate={"g": {0: {1}, 1: {2}}},
    assumptions=("ideal graph state", "XY-plane measurements"),
    diagnostics=None,
)


def page_data(html: str) -> dict:
    return json.loads(re.search(r"const DATA = (.*);\n", html).group(1))


def scripts(html: str) -> list[str]:
    return re.findall(r"<script>(.*?)</script>", html, flags=re.S)


# ---- layouts never change semantics --------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(LAYOUTS) - {"manual", "layered"}))
def test_layouts_are_deterministic_and_semantics_free(cv_case, name):
    open_graph = OpenGraph.from_cvgraph(cv_case[0], SHEARS)
    first, second = compute_layout(open_graph, name), compute_layout(open_graph, name)
    assert first == second
    view = GraphView(open_graph, layout=first)
    moved = view.with_layout(Layout({n: (10 * x, -y + 3) for n, (x, y) in first.positions.items()}))
    for v in (view, moved):
        assert v.graph.to_dict() == open_graph.to_dict()
        np.testing.assert_array_equal(v.graph.adjacency(), open_graph.adjacency())
    physical = [e["data"] for e in view.elements() if e["group"] == "edges"]
    moved_physical = [e["data"] for e in moved.elements() if e["group"] == "edges"]
    assert physical == moved_physical


def test_force_layout_seed_and_layered_columns():
    open_graph = OpenGraph.from_cvgraph(pg.CVGraph.square(3), {})
    a, b = (compute_layout(open_graph, "force", seed=4) for _ in range(2))
    assert a == b and a != compute_layout(open_graph, "force", seed=5)
    graph = qubit_graph()
    layout = compute_layout(graph, "layered", layers=[[0], [1]])
    assert [layout[n][0] for n in (0, 1, 2)] == [0, 1, 2]
    with pytest.raises(ValueError):
        compute_layout(graph, "layered", layers=[[0], [0]])
    with pytest.raises(ValueError):
        compute_layout(graph, "manual", positions={0: (0, 0)})


# ---- stage A ------------------------------------------------------------------------
def test_construct_stage_shows_graph_only(cv_case):
    _, pattern, _ = cv_case
    view = GraphView.from_pattern(pattern).construct()
    data = page_data(view.to_html())
    assert data == json.loads(json.dumps(view.payload()))
    nodes = {e["data"]["id"]: e for e in data["elements"] if e["group"] == "nodes"}
    edges = [e for e in data["elements"] if e["group"] == "edges"]
    assert {e["classes"] for e in edges} == {"physical"}  # no flow arrows in stage A
    assert sorted(
        (e["data"]["source"], e["data"]["target"], e["data"]["label"]) for e in edges
    ) == [
        ("0", "1", "1"),
        ("1", "2", "1"),
    ]
    assert nodes["0"]["classes"] == "input measured"
    assert nodes["1"]["classes"] == "measured ancilla"
    assert nodes["2"]["classes"] == "output ancilla"
    assert nodes["1"]["data"]["squeezing"] == 1.5 and nodes["0"]["data"]["squeezing"] is None
    assert nodes["0"]["data"]["meas"]["shear"] == pytest.approx(0.3, abs=1e-12)
    assert nodes["0"]["data"]["annotation"] == "external input"
    assert data["flow"] is None and data["trace"] is None
    assert view.graph.model is Model.CV_GAUSSIAN
    with pytest.raises(ValueError):
        view.elements("flow")


def test_pattern_conversion_reports_what_it_cannot_draw():
    pattern = pg.Pattern(inputs=(0, 1))
    pattern.append(pg.BeamSplitter(0, 1, 0.4)).append(pg.Rotate(1, 0.2))
    pattern.measure(0, pg.Homodyne.q()).measure(1, pg.PhotonNumber())
    pattern.append(pg.Output(()))
    view = GraphView.from_pattern(pattern)
    assert view.graph.measurements == {}  # q homodyne and photon counting are not wire steps
    assert view.graph.model is Model.CV_NONGAUSSIAN
    assert any("BeamSplitter" in n for n in view.notes)
    assert "homodyne angle 0" in view.annotations[0]
    assert "PhotonNumber" in view.annotations[1]


# ---- stage B ------------------------------------------------------------------------
def test_flow_arrows_equal_reported_dependencies():
    view = GraphView(qubit_graph()).with_flow(QUBIT_FLOW)
    expected = {(0, 1), (1, 2), (0, 2)}
    assert view.flow_arrows() == view.flow.dependencies() == expected
    data = page_data(view.to_html())
    flow_edges = {
        (e["data"]["source"], e["data"]["target"])
        for e in data["elements"]
        if e["classes"] == "flow"
    }
    assert flow_edges == {(str(s), str(t)) for s, t in expected}
    assert data["flow"]["status"] == "certified" and data["flow"]["algorithm"] == "gflow"
    assert data["flow"]["assumptions"] == list(QUBIT_FLOW.assumptions)
    orders = {
        e["data"]["id"]: e["data"]["order"] for e in data["elements"] if e["group"] == "nodes"
    }
    assert orders == {"0": 1, "1": 2, "2": None}
    as_dict = GraphView(qubit_graph()).with_flow(vars(QUBIT_FLOW))
    assert as_dict.flow_arrows() == expected


def test_flow_result_is_checked_against_graph():
    graph = qubit_graph()
    with pytest.raises(ValueError):
        GraphView(graph).with_flow({**vars(QUBIT_FLOW), "measurement_order": (0, 9)})
    with pytest.raises(ValueError):
        GraphView(graph).with_flow({**vars(QUBIT_FLOW), "model": "cv-gaussian"})


def test_correction_map_normalization():
    graph = qubit_graph()
    target = normalize_corrections(QUBIT_FLOW.correction_map, graph)
    assert {t: (c.x, c.z) for t, c in target.items()} == {1: ((0,), ()), 2: ((1,), (0,))}
    assert target[2].formula() == "X^(m[1]) Z^(m[0])"
    assert target[2].evaluate({0: 1, 1: 1}) == (1, 1)
    assert normalize_corrections({0: {"x": {2}}, 1: {"x": {2}}}, graph)[2].x == (0, 1)
    cv = OpenGraph.from_cvgraph(pg.CVGraph.line(3, outputs=(2,)), {0: 0.75, 1: 0.0})
    scaled = normalize_corrections({0: {1: {"q": -1.0, "p": 0.0}}}, cv)[1]
    assert scaled.q == ((0, -1.25),) and scaled.p == ()  # m = record/sin(theta): sqrt(1+k^2)
    with pytest.raises(TypeError):
        normalize_corrections({0: {1: 0.5}}, cv)  # ambiguous bare number
    with pytest.raises(TypeError):
        normalize_corrections({0: {1: "X"}}, graph)
    with pytest.raises(ValueError):
        normalize_corrections({2: {1: {"q": 1.0}}}, cv)  # output node has no HomodyneSpec


def test_real_flow_analysis_overlays(cv_case):
    flows = pytest.importorskip("photographiq.flows")
    qubit = flows.analyze(qubit_graph())
    assert qubit.certified
    view = GraphView(qubit_graph()).with_flow(qubit)
    expected = {(s, t) for s, c in qubit.correction_map.items() for a in ("x", "z") for t in c[a]}
    assert view.flow_arrows() == expected
    assert view.payload()["flow"]["status"] == "certified"
    graph = cv_case[0]
    open_graph = OpenGraph.from_cvgraph(graph, SHEARS)
    result = flows.analyze(open_graph)
    assert result.certified
    shown = GraphView(open_graph).with_flow(result).with_corrections().corrections
    # Independent check: the displacements actually executed by the flow pattern.
    executed = pattern_corrections(flow_pattern(graph, result.measurement_order, shears=SHEARS))
    assert shown.keys() == executed.keys()
    for target, entry in executed.items():
        for axis in ("q", "p"):
            got, want = dict(getattr(shown[target], axis)), dict(getattr(entry, axis))
            assert got.keys() == want.keys()
            for key in want:
                assert got[key] == pytest.approx(want[key], rel=1e-12)


# ---- stage C ------------------------------------------------------------------------
def certified_corrections(graph, order, shears):
    """Independent reference: Booth-Markham correction vectors of cvflow.certify_cv_flow."""
    flow = certify_cv_flow(graph, order)
    a, index = graph.adjacency(), {n: i for i, n in enumerate(graph.nodes)}
    expected: dict = {}
    for i, node in enumerate(order):
        scale = 1 / math.sin(math.atan2(1, shears[node]))
        c = flow.corrections[node]
        for target in graph.nodes:
            if target in order[: i + 1]:
                continue
            q = -c.get(target, 0.0) * scale
            p = -sum(a[index[target], index[n]] * v for n, v in c.items()) * scale
            for axis, value in (("q", q), ("p", p)):
                if value:
                    expected.setdefault(target, {}).setdefault(axis, {})[node] = value
    return expected


def test_cv_corrections_equal_certified_flow(cv_case):
    graph, pattern, _ = cv_case
    extracted = pattern_corrections(pattern)
    expected = certified_corrections(graph, ORDER, SHEARS)
    assert set(extracted) == set(expected)
    for target, entry in extracted.items():
        for axis in ("q", "p"):
            got = dict(getattr(entry, axis))
            assert got.keys() == expected[target].get(axis, {}).keys()
            for key, value in got.items():
                assert value == pytest.approx(expected[target][axis][key], rel=1e-12)
    view = GraphView.from_pattern(pattern).with_corrections(pattern)
    arrows = view.correction_arrows()
    assert set(arrows) == {(s, t) for t, c in extracted.items() for s in c.sources}
    nodes = {e["data"]["id"]: e["data"] for e in view.elements() if e["group"] == "nodes"}
    assert nodes["2"]["formula"] == extracted[2].formula()
    assert "m[0]" in nodes["2"]["formula"] and "m[1]" in nodes["2"]["formula"]
    rescaled = {  # FlowAnalysisResult convention: coefficients of m = record/sin(theta)
        source: {} for source in ORDER
    }
    for target, axes in expected.items():
        for axis, terms in axes.items():
            for source, value in terms.items():
                scale = math.sin(math.atan2(1, SHEARS[source]))
                rescaled[source].setdefault(target, {})[axis] = value * scale
    flow_like = {
        "status": "certified",
        "model": "cv-gaussian",
        "algorithm": "cv-flow",
        "measurement_order": ORDER,
        "layers": None,
        "correction_map": rescaled,
        "certificate": None,
        "assumptions": ["infinite squeezing limit"],
        "diagnostics": None,
    }
    from_flow = GraphView.from_pattern(pattern).with_flow(flow_like).with_corrections()
    assert from_flow.correction_arrows() == arrows


def test_nonlinear_feed_forward_is_rejected():
    pattern = pg.Pattern(inputs=(0, 1))
    pattern.measure(0, pg.Homodyne.p()).displace(1, q=Outcome(0) * Outcome(0))
    with pytest.raises(ValueError, match="not affine"):
        pattern_corrections(pattern)
    # max(m, -5) equals m at most sampled points: opaque callables are never accepted.
    relu = CallableExpression(lambda r: max(r[0], -5.0), {0})
    pattern = pg.Pattern(inputs=(0, 1)).measure(0, pg.Homodyne.p()).displace(1, q=relu)
    with pytest.raises(ValueError, match="opaque callable"):
        pattern_corrections(pattern)


def test_repeated_cz_conversion_is_never_silent():
    cancel = pg.Pattern(inputs=(0, 1))
    cancel.append(pg.Entangle(0, 1, 0.5)).append(pg.Entangle(0, 1, -0.5))
    view = GraphView.from_pattern(cancel)
    assert view.graph.graph.number_of_edges() == 0 and any("cancel" in n for n in view.notes)
    split = pg.Pattern(inputs=(0, 1))
    split.append(pg.Entangle(0, 1, 0.5)).append(pg.Rotate(0, 1.0)).append(pg.Entangle(0, 1, 0.5))
    assert any("need not equal" in n for n in GraphView.from_pattern(split).notes)
    plain = pg.Pattern(inputs=(0, 1))
    plain.append(pg.Entangle(0, 1, 0.5)).append(pg.Entangle(0, 1, 0.5))
    view = GraphView.from_pattern(plain)
    assert view.graph.weight(0, 1) == 1.0 and not any("need not" in n for n in view.notes)


# ---- stage D ------------------------------------------------------------------------
def test_execution_trace_follows_simulator(cv_case):
    _, pattern, result = cv_case
    trace = execution_trace(pattern, result)
    steps = trace.steps
    assert [s.index for s in steps] == list(range(len(pattern.commands) + 1))
    assert steps[-1].kind == "output" and steps[-1].nodes == pattern.outputs
    measured = {s.outcome_key: s.outcome for s in steps if s.kind == "measure"}
    assert measured == result.outcomes
    corrections = pattern_corrections(pattern)
    total: dict = {}
    for step, command in zip(steps, pattern.commands):
        assert step.nodes == pg.commands.quantum_nodes(command)
        if step.kind == "feed-forward":
            old = total.get(command.node, np.zeros(2))
            total[command.node] = old + [step.values["q"], step.values["p"]]
    for node, value in total.items():
        np.testing.assert_allclose(value, corrections[node].evaluate(result.records), rtol=1e-12)
    # The recorded outcomes define the branch: replaying them reproduces the output state.
    replay = pg.simulate(
        pattern, backend="gaussian", seed=999, measurement_outcomes=dict(result.outcomes)
    )
    np.testing.assert_allclose(replay.state.mean, result.state.mean, rtol=0, atol=1e-10)
    view = GraphView.from_pattern(pattern).with_corrections(pattern).with_execution(trace)
    data = page_data(view.to_html())
    assert [s["label"] for s in data["trace"]["steps"]] == [s.label for s in steps]
    assert trace.postselected == ()
    with pytest.raises(ValueError):
        execution_trace(pg.Pattern(inputs=(0,)).measure(0), result)


def test_postselected_branch_is_labelled(cv_case):
    _, pattern, _ = cv_case
    result = pg.simulate(pattern, backend="gaussian", measurement_outcomes={0: 0.25, 1: -0.5})
    trace = execution_trace(pattern, result)
    assert trace.postselected == (0, 1)
    assert all("conditional branch" in s.label for s in trace.steps if s.kind == "measure")
    view = GraphView.from_pattern(pattern).with_execution(trace)
    assert any("not a deterministic run" in n for n in view.payload()["notes"])


# ---- editing ------------------------------------------------------------------------
def browser_export(view: GraphView) -> dict:
    """What the page's Export button produces for an unedited view (mirrors the JS)."""
    data = view.payload()
    nodes = [e["data"] for e in data["elements"] if e["group"] == "nodes"]
    raw = {n["id"]: n["raw"] for n in nodes}
    return {
        "graph": {
            "model": data["model"],
            "nodes": [
                {
                    "id": n["raw"],
                    **({} if n["squeezing"] is None else {"squeezing": n["squeezing"]}),
                }
                for n in nodes
            ],
            "edges": [
                {
                    "source": raw[e["data"]["source"]],
                    "target": raw[e["data"]["target"]],
                    "weight": e["data"]["weight"],
                }
                for e in data["elements"]
                if e.get("classes") == "physical"
            ],
            "inputs": [raw[i] for i in data["inputs"]],
            "outputs": [raw[o] for o in data["outputs"]],
            "measurements": {n["id"]: n["meas"] for n in nodes if n["meas"]},
        },
        "layout": view.layout.to_dict(),
    }


def test_apply_edits_round_trip_and_revalidation(cv_case):
    open_graph = OpenGraph.from_cvgraph(cv_case[0], SHEARS)
    view = GraphView(open_graph)
    exported = json.loads(json.dumps(browser_export(view)))
    graph, layout = apply_edits(json.dumps(exported))
    assert graph.to_dict() == open_graph.to_dict() and layout == view.layout
    exported["graph"]["edges"].append({"source": 0, "target": 2, "weight": -0.5})
    exported["layout"]["0"] = [5.0, 7.0]
    graph, layout = apply_edits(exported)
    assert graph.weight(0, 2) == -0.5 and graph.adjacency()[0, 2] == -0.5
    assert layout[0] == (5.0, 7.0)
    bad = json.loads(json.dumps(exported))
    bad["graph"]["edges"][-1]["weight"] = 0
    with pytest.raises(ValueError, match="nonzero"):
        apply_edits(bad)
    bad = json.loads(json.dumps(exported))
    bad["graph"]["measurements"]["2"] = {"shear": 0.0}
    with pytest.raises(ValueError, match="Output"):
        apply_edits(bad)
    bad = json.loads(json.dumps(exported))
    del bad["layout"]["1"]
    with pytest.raises(ValueError, match="lacks"):
        apply_edits(bad)
    qubit = browser_export(GraphView(qubit_graph()))
    qubit["graph"]["edges"][0]["weight"] = 2.0
    with pytest.raises(ValueError, match="unweighted"):
        apply_edits(qubit)


def test_tuple_labels_and_layout_files(tmp_path):
    open_graph = OpenGraph.from_cvgraph(pg.CVGraph.rectangular(2, 2, inputs=((0, 0),)), {})
    view = GraphView(open_graph)
    graph, layout = apply_edits(json.loads(json.dumps(browser_export(view))))
    assert graph.to_dict() == open_graph.to_dict()
    assert set(layout.positions) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    path = tmp_path / "layout.json"
    export_layout(view.layout, path)
    assert import_layout(path, open_graph) == view.layout


# ---- exports -------------------------------------------------------------------------
@pytest.mark.parametrize("stage", ["construct", "corrections", "execution"])
@pytest.mark.parametrize("suffix", ["svg", "pdf", "png"])
def test_static_exports_are_valid_files(cv_case, tmp_path, stage, suffix):
    pytest.importorskip("matplotlib")
    _, pattern, result = cv_case
    view = (
        GraphView.from_pattern(pattern)
        .with_corrections(pattern)
        .with_execution(execution_trace(pattern, result))
    )
    path = view.save(tmp_path / f"fig.{suffix}", stage=stage)
    content = path.read_bytes()
    if suffix == "svg":
        root = ET.fromstring(content)
        assert root.tag.endswith("svg")
        text = content.decode()
        assert "CZ edge" in text and ("correction" in text) == (stage != "construct")
    elif suffix == "pdf":
        assert content.startswith(b"%PDF") and content.rstrip().endswith(b"%%EOF")
    else:
        assert content.startswith(b"\x89PNG\r\n\x1a\n")


def test_static_flow_export(tmp_path):
    pytest.importorskip("matplotlib")
    path = GraphView(qubit_graph()).with_flow(QUBIT_FLOW).save(tmp_path / "flow.svg")
    assert "dependency" in path.read_text(encoding="utf-8")


@pytest.mark.skipif(
    not shutil.which(NODE) and not __import__("os").path.exists(NODE),
    reason="Node.js not installed",
)
def test_generated_javascript_parses(cv_case, tmp_path):
    _, pattern, result = cv_case
    views = [
        GraphView(qubit_graph()).with_flow(QUBIT_FLOW),
        GraphView.from_pattern(pattern)
        .with_corrections(pattern)
        .with_execution(execution_trace(pattern, result)),
    ]
    for i, view in enumerate(views):
        for stage in ("construct", view.stage):
            for j, code in enumerate(scripts(view.to_html(stage))):
                path = tmp_path / f"page{i}_{stage}_{j}.js"
                path.write_text(code, encoding="utf-8")
                check = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True)
                assert check.returncode == 0, check.stderr


def test_html_is_injection_safe_and_offline_capable(tmp_path):
    graph = OpenGraph(Model.CV_GAUSSIAN, nx.Graph([("</script><b>", "x")]))
    view = GraphView(graph, title="<i>t</i>")
    page = view.to_html()
    assert "</script><b>" not in page and "<title>&lt;i&gt;t&lt;/i&gt;</title>" in page
    assert page_data(page)["elements"][0]["data"]["id"] == "</script><b>"
    library = tmp_path / "cytoscape.min.js"
    library.write_text("window.cytoscape = function () {};", encoding="utf-8")
    offline = view.to_html(cytoscape_js=library)
    assert "cdn.jsdelivr" not in offline and "window.cytoscape" in offline
    assert 'srcdoc="' in view._repr_html_()
    assert HomodyneSpec(0.0).angle == pytest.approx(math.pi / 2)


def test_hierarchical_layout_accepts_mixed_label_types():
    graph = nx.Graph()
    graph.add_nodes_from(["a", 0, 1])  # two components, labels not mutually comparable
    graph.add_edge(0, 1)
    layout = compute_layout(graph, "hierarchical")
    assert set(layout.positions) == {"a", 0, 1}
