"""Real-browser checks of the interactive graph pages (Playwright + installed Edge/Chrome).

Unlike ``tests/visualization/test_interactive_browser.py`` (a JS harness), this drives
the pages with real mouse and keyboard events, takes screenshots and compares what is
drawn with the Python model, the flow result, the pattern and the simulation trace.

Usage::

    python scripts/check_interactive_browser.py OUT_DIR [--channel msedge|chrome]

Needs ``playwright`` (no browser download: it uses the installed Edge or Chrome) and
network access to the pinned CDN, or ``PHOTOGRAPHIQ_CYTOSCAPE_JS`` naming a local copy.
Writes ``*.html``, ``*.png`` and ``report.json`` to OUT_DIR; exits 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import networkx as nx
from playwright.sync_api import sync_playwright

import photographiq as pg
from photographiq.commands import Displace
from photographiq.cvflow import flow_pattern
from photographiq.flows import analyze
from photographiq.interactive import GraphView, apply_edits, execution_trace, pattern_corrections
from photographiq.ir import Model, OpenGraph, QubitMeasurement

CYTOSCAPE_JS = os.environ.get("PHOTOGRAPHIQ_CYTOSCAPE_JS")
VIEWPORT = {"width": 1280, "height": 800}
SCALE = 110.0


class Checker:
    def __init__(self, browser, out: Path):
        self.browser, self.out, self.results = browser, out, []

    def page(self, view: GraphView, name: str, stage: str | None = None, viewport=VIEWPORT):
        path = self.out / f"{name}.html"
        view.save_html(path, stage, cytoscape_js=CYTOSCAPE_JS)
        page = self.browser.new_page(viewport=viewport, accept_downloads=True)
        page.errors = []
        page.on("pageerror", lambda error: page.errors.append(str(error)))
        page.goto(path.as_uri())
        page.wait_for_load_state("load")
        if not page.evaluate("typeof cytoscape !== 'undefined'"):
            raise RuntimeError("Cytoscape.js did not load; set PHOTOGRAPHIQ_CYTOSCAPE_JS")
        return page

    def shot(self, page, name: str) -> str:
        page.screenshot(path=str(self.out / f"{name}.png"))
        return f"{name}.png"

    def run(self, name: str, check) -> None:
        try:
            evidence = check()
            self.results.append({"check": name, "ok": True, "evidence": evidence})
        except Exception:  # record every failure, keep checking
            self.results.append({"check": name, "ok": False, "error": traceback.format_exc()})
        print(("PASS " if self.results[-1]["ok"] else "FAIL ") + name, flush=True)


# ---- page geometry -------------------------------------------------------------------
def node_xy(page, node_id: str) -> tuple[float, float]:
    """Client (viewport) coordinates of a node centre."""
    return tuple(
        page.evaluate(
            """(id) => { const r = cy.getElementById(id).renderedPosition();
            const b = cy.container().getBoundingClientRect();
            return [b.left + r.x, b.top + r.y]; }""",
            node_id,
        )
    )


def edge_xy(page, source: str, target: str) -> tuple[float, float]:
    """Client coordinates of the midpoint of the CZ edge source-target."""
    return tuple(
        page.evaluate(
            """([s, t]) => { const e = cy.getElementById(s).edgesWith(cy.getElementById(t))
              .filter('.physical')[0]; const r = e.renderedMidpoint();
            const b = cy.container().getBoundingClientRect();
            return [b.left + r.x, b.top + r.y]; }""",
            [source, target],
        )
    )


def selected_edges(page) -> list:
    return page.evaluate("cy.edges(':selected').map((e) => [e.source().id(), e.target().id()])")


def model_positions(page) -> dict:
    return page.evaluate(
        "() => Object.fromEntries(cy.nodes().map((n) => [n.id(), [n.position('x'), "
        "n.position('y')]]))"
    )


def viewport_state(page) -> dict:
    return page.evaluate("() => ({ zoom: cy.zoom(), pan: cy.pan() })")


def drag(page, start, delta, steps: int = 12) -> None:
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move(start[0] + delta[0], start[1] + delta[1], steps=steps)
    page.mouse.up()


def export(page) -> str:
    """Click Export JSON and return the downloaded file (the documented round trip)."""
    with page.expect_download() as info:
        page.click("#export")
    text = Path(info.value.path()).read_text(encoding="utf-8")
    assert text == page.input_value("#export-text"), "download differs from the text box"
    return text


def edge_set(graph: OpenGraph) -> set:
    return {(frozenset((u, v)), float(d["weight"])) for u, v, d in graph.graph.edges(data=True)}


def overlaps(page) -> list:
    """Pairs of nodes whose rendered boxes (shape + label) intersect."""
    return page.evaluate(
        """() => { const boxes = cy.nodes().map((n) => [n.id(),
          n.renderedBoundingBox({ includeLabels: true, includeOverlays: false })]);
        const out = [];
        for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
          const [a, p] = boxes[i], [b, q] = boxes[j];
          if (p.x1 < q.x2 && q.x1 < p.x2 && p.y1 < q.y2 && q.y1 < p.y2) out.push([a, b]); }
        return out; }"""
    )


def drawn(page, selector: str) -> list:
    return page.evaluate(
        """(sel) => cy.edges(sel).map((e) => [e.source().id(), e.target().id(), e.data('label'),
           e.style('line-style'), e.style('line-color'), e.style('target-arrow-shape')])""",
        selector,
    )


# ---- cases -----------------------------------------------------------------------------
def cv_wire():
    graph = pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,))
    shears = {0: 0.3, 1: -0.5, 2: 0.0}
    open_graph = OpenGraph.from_cvgraph(graph, shears)
    flow = analyze(open_graph)
    pattern = flow_pattern(graph, flow.measurement_order, shears=shears)
    result = pg.simulate(pattern, backend="gaussian", seed=7)
    trace = execution_trace(pattern, result)
    view = GraphView(open_graph, title="Four-node CV wire")
    return open_graph, flow, pattern, result, trace, view


def qubit_six() -> OpenGraph:
    network = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (1, 5)])
    measurements = {n: QubitMeasurement("XY", 0.1 * n) for n in (0, 1, 2, 3)}
    return OpenGraph(Model.QUBIT, network, (0,), (4, 5), measurements)


def run_checks(out: Path, channel: str = "msedge") -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    open_graph, flow, pattern, result, trace, view = cv_wire()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        c = Checker(browser, out)

        def dragging():
            page = c.page(view, "drag")
            before, shots = model_positions(page), [c.shot(page, "01_drag_before")]
            zoom = viewport_state(page)["zoom"]
            drag(page, node_xy(page, "1"), (90, 120))
            after = model_positions(page)
            shots.append(c.shot(page, "02_drag_after"))
            moved = (after["1"][0] - before["1"][0], after["1"][1] - before["1"][1])
            assert math.isclose(moved[0], 90 / zoom, abs_tol=1.5), moved
            assert math.isclose(moved[1], 120 / zoom, abs_tol=1.5), moved
            assert all(after[n] == before[n] for n in before if n != "1"), "other nodes moved"
            graph, layout = apply_edits(export(page))
            want = (after["1"][0] / SCALE, -after["1"][1] / SCALE)
            assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(layout[1], want))
            assert layout[0] == view.layout[0] and layout[2] == view.layout[2]
            assert edge_set(graph) == edge_set(open_graph), "drag changed the edges"
            assert graph.to_dict() == open_graph.to_dict(), "drag changed the semantics"
            assert page.errors == [], page.errors
            return {
                "screenshots": shots,
                "moved_px_model": moved,
                "zoom": zoom,
                "layout_1": layout[1],
            }

        def pan_zoom():
            page = c.page(view, "panzoom")
            before, state = model_positions(page), viewport_state(page)
            box = page.evaluate("() => cy.container().getBoundingClientRect().toJSON()")
            start = (box["left"] + 20, box["top"] + 20)
            hit = page.evaluate(
                """([x, y]) => cy.elements().filter((e) => { const b = e.renderedBoundingBox();
                  return b.x1 <= x && x <= b.x2 && b.y1 <= y && y <= b.y2; }).length""",
                [20, 20],
            )
            assert hit == 0, "pan start point is not background"
            drag(page, start, (150, 90))
            panned = viewport_state(page)
            shots = [c.shot(page, "03_pan")]
            assert math.isclose(panned["pan"]["x"] - state["pan"]["x"], 150, abs_tol=1)
            assert math.isclose(panned["pan"]["y"] - state["pan"]["y"], 90, abs_tol=1)
            assert panned["zoom"] == state["zoom"]
            assert model_positions(page) == before, "panning moved nodes"
            assert page.evaluate("cy.nodes(':selected').length") == 0
            anchor = node_xy(page, "1")
            page.mouse.move(*anchor)
            page.mouse.wheel(0, -100)  # one wheel notch, zooming about node 1
            page.wait_for_timeout(300)
            zoomed = viewport_state(page)
            assert all(math.isclose(a, b, abs_tol=1) for a, b in zip(node_xy(page, "1"), anchor))
            shots.append(c.shot(page, "04_zoom_in"))
            assert zoomed["zoom"] > panned["zoom"] * 1.1, zoomed
            assert model_positions(page) == before, "zooming moved nodes"
            graph, layout = apply_edits(export(page))
            assert layout == view.layout and graph.to_dict() == open_graph.to_dict()
            return {"screenshots": shots, "before": state, "panned": panned, "zoomed": zoomed}

        def edits():
            page = c.page(view, "edits")
            # On the straight wire a new 0-2 edge would run through node 1 and over edge
            # 1-2 (straight edges), so move node 2 off the line first (a real drag).
            drag(page, node_xy(page, "2"), (0, 110))
            page.mouse.click(*node_xy(page, "0"))
            page.keyboard.down("Shift")
            page.mouse.click(*node_xy(page, "2"))
            page.keyboard.up("Shift")
            selected = page.evaluate("cy.nodes(':selected').map((n) => n.id())")
            assert sorted(selected) == ["0", "2"], selected
            page.fill("#weight", "-0.5")
            page.click("#add-edge")
            page.mouse.click(*edge_xy(page, "1", "2"))
            assert selected_edges(page) == [["1", "2"]], selected_edges(page)
            page.fill("#weight", "2.5")
            page.click("#set-weight")
            page.mouse.click(*edge_xy(page, "0", "1"))
            assert selected_edges(page) == [["0", "1"]], selected_edges(page)
            page.click("#remove")
            page.mouse.click(*node_xy(page, "1"))
            page.fill("#shear", "0.75")
            page.click("#set-meas")
            assert "shear k=0.75" in page.inner_text("#info"), "selection panel is stale"
            shots = [c.shot(page, "05_edits")]
            graph, layout = apply_edits(export(page))
            weights = {frozenset(e): w for e, w in edge_set(graph)}
            assert weights == {
                frozenset((0, 2)): -0.5,
                frozenset((1, 2)): 2.5,
                frozenset((2, 3)): 1.0,
            }, weights
            assert graph.measurements[1].shear == 0.75
            assert graph.squeezing == open_graph.squeezing and graph.outputs == (3,)
            assert layout[0] == view.layout[0] and layout[2][1] < view.layout[2][1] - 0.5
            reanalysed = analyze(graph)  # the edited model is a usable OpenGraph
            # An invalid edit made in the page is rejected by Python on import.
            page.mouse.click(*edge_xy(page, "2", "3"))
            assert selected_edges(page) == [["2", "3"]], selected_edges(page)
            page.fill("#weight", "0")
            page.click("#set-weight")
            try:
                apply_edits(export(page))
            except ValueError as error:
                rejected = str(error)
            else:
                raise AssertionError("zero weight was accepted")
            assert page.errors == [], page.errors
            return {
                "screenshots": shots,
                "edges": sorted(map(str, weights.items())),
                "reanalysed": str(reanalysed.status),
                "zero_weight_rejected": rejected,
            }

        def construct_stage():
            page = c.page(view, "stage_construct", "construct")
            physical = drawn(page, ".physical")
            assert len(physical) == open_graph.graph.number_of_edges()
            assert page.evaluate("cy.edges().not('.physical').length") == 0
            assert page.evaluate("cy.nodes().filter((n) => n.data('order') != null).length") == 0
            return {"screenshots": [c.shot(page, "06_stage_construct")], "edges": physical}

        def flow_stage():
            flow_view = view.with_flow(flow)
            page = c.page(flow_view, "stage_flow", "flow")
            arrows = drawn(page, ".flow")
            physical = drawn(page, ".physical")
            assert {(int(s), int(t)) for s, t, *_ in arrows} == flow_view.flow_arrows()
            assert all(tuple(a[3:]) == ("dashed", "rgb(37,99,235)", "triangle") for a in arrows), (
                arrows
            )
            assert all(tuple(e[3:]) == ("solid", "rgb(100,116,139)", "none") for e in physical)
            layers = page.evaluate(
                "Object.fromEntries(cy.nodes().map((n) => [n.id(), n.data('layer')]))"
            )
            want = {str(n): i for i, layer in enumerate(flow.layers) for n in layer}
            assert {k: v for k, v in layers.items() if v is not None} == want, layers
            text = page.inner_text("#flow")
            assert (
                "layers: "
                + " ".join("[" + ", ".join(map(str, layer)) + "]" for layer in flow.layers)
                in text
            ), text
            order = page.evaluate("cy.nodes().map((n) => [n.id(), n.data('display')])")
            for i, node in enumerate(flow.measurement_order):
                assert f"#{i + 1}" in dict(order)[str(node)]
            return {
                "screenshots": [c.shot(page, "07_stage_flow")],
                "arrows": arrows,
                "layers": layers,
                "flow_panel": text,
            }

        def corrections_stage():
            corr_view = view.with_flow(flow).with_corrections(pattern)
            page = c.page(corr_view, "stage_corrections", "corrections")
            arrows = drawn(page, ".correction")
            got = {(int(s), int(t)): label for s, t, label, *_ in arrows}
            extracted = pattern_corrections(pattern)
            want = {(s, t): lab for t, cor in extracted.items() for s, lab in cor.terms()}
            assert got == want, (got, want)
            # Independent of pattern_corrections: the Displace commands with outcome deps.
            raw = set()
            for command in pattern.commands:
                if isinstance(command, Displace):
                    for value in (command.q, command.p):
                        raw |= {(k, command.node) for k in getattr(value, "dependencies", ())}
            assert set(got) == raw, (set(got), raw)
            # Shorter arcs are drawn above longer ones so their labels are not crossed.
            z = page.evaluate(
                """cy.edges('.correction').map((e) => [e.source().id(), e.target().id(),
                   Number(e.style('z-index'))])"""
            )
            for s1, t1, z1 in z:
                for s2, t2, z2 in z:
                    if abs(int(t1) - int(s1)) < abs(int(t2) - int(s2)):
                        assert z1 > z2, z
            assert all(tuple(a[3:]) == ("dotted", "rgb(194,65,12)", "triangle-tee") for a in arrows)
            # The flow analysis predicts the same feed-forward as the pattern executes.
            from_flow = view.with_flow(flow).with_corrections().correction_arrows()
            assert set(from_flow) == set(got)
            notes = page.inner_text("#notes")
            for target, cor in extracted.items():
                assert f"{target}: {cor.formula()}" in notes, notes
            page.mouse.click(*node_xy(page, "2"))
            info = page.inner_text("#info")
            assert "correction: " + extracted[2].formula() in info, info
            return {
                "screenshots": [c.shot(page, "08_stage_corrections")],
                "arrows": sorted(map(str, got.items())),
                "info_node_2": info,
                "notes": notes,
            }

        def execution_stage():
            exec_view = view.with_flow(flow).with_corrections(pattern).with_execution(trace)
            page = c.page(exec_view, "stage_execution", "execution")
            shown, shots = [], []
            for i, step in enumerate(trace.steps):
                page.click("#step")
                label = page.inner_text("#step-label")
                assert label == f"Step {i + 1}/{len(trace.steps)}: {step.label}", label
                if step.kind == "measure":
                    node = str(step.nodes[0])
                    display = page.evaluate("(id) => cy.getElementById(id).data('display')", node)
                    value = float(display.split("m=")[1])
                    recorded = result.outcomes[step.outcome_key]
                    assert step.outcome == recorded
                    assert math.isclose(value, recorded, rel_tol=5e-4, abs_tol=1e-9), (
                        value,
                        recorded,
                    )
                    shown.append((node, value, recorded))
                    assert page.evaluate(
                        "cy.nodes('.done').every((n) => "
                        "n.style('text-opacity') == 1 && n.style('opacity') == 1)"
                    )
                    if len(shown) == 2:
                        shots.append(c.shot(page, "09_execution_mid"))
            shots.append(c.shot(page, "10_execution_end"))
            assert len(shown) == len(result.outcomes)
            page.click("#reset")
            page.click("#play")
            page.wait_for_function(
                f"document.getElementById('step-label').textContent.startsWith("
                f"'Step {len(trace.steps)}/')",
                timeout=60_000,
            )
            assert page.errors == [], page.errors
            return {"screenshots": shots, "outcomes": shown}

        def layout_scale():
            cases = {
                "11_small_cv4": GraphView(open_graph, title="CV wire, 4 nodes"),
                "12_small_qubit6": GraphView(qubit_six(), title="Qubit graph, 6 nodes"),
                "13_qubit6_flow": GraphView(qubit_six(), title="Qubit 6, flow").with_flow(
                    analyze(qubit_six())
                ),
                "14_grid6x6": GraphView(
                    OpenGraph.from_cvgraph(
                        pg.CVGraph.rectangular(
                            6, 6, squeezing=1.0, inputs=((0, 0),), outputs=((5, 5),)
                        ),
                        {},
                    ),
                    title="6x6 CV cluster",
                ),
            }
            evidence = {}
            for name, case in cases.items():
                page = c.page(case, name)
                clash = overlaps(page)
                font = page.evaluate("cy.zoom() * 11")
                evidence[name] = {
                    "overlaps": clash,
                    "font_px": round(font, 1),
                    "screenshot": c.shot(page, name),
                }
                assert page.errors == [], page.errors
                if "grid" not in name:
                    assert clash == [], (name, clash)
                assert font >= 8, (name, font)
            return evidence

        def static_exports():
            exec_view = view.with_flow(flow).with_corrections(pattern).with_execution(trace)
            grid_view = GraphView(qubit_six(), title="Qubit 6").with_flow(analyze(qubit_six()))
            files = []
            for stage in ("construct", "flow", "corrections", "execution"):
                files.append(exec_view.save(out / f"static_{stage}.svg", stage=stage))
            files.append(exec_view.save(out / "static_corrections.pdf", stage="corrections"))
            files.append(grid_view.save(out / "static_qubit6_flow.png", stage="flow", dpi=150))
            shots = []
            for path in files:
                if path.suffix == ".svg":
                    page = c.browser.new_page(viewport={"width": 1400, "height": 700})
                    page.goto(path.as_uri())
                    size = page.evaluate(
                        "(() => { const b = document.querySelector('svg')"
                        ".getBoundingClientRect(); return [b.width, b.height]; })()"
                    )
                    page.set_viewport_size(
                        {"width": math.ceil(size[0]) + 16, "height": math.ceil(size[1]) + 16}
                    )
                    shots.append(c.shot(page, path.stem + "_svg"))
                    page.close()
                elif path.suffix == ".pdf" and shutil.which("pdftoppm"):
                    stem = out / (path.stem + "_pdf")
                    subprocess.run(
                        ["pdftoppm", "-png", "-r", "150", "-singlefile", str(path), str(stem)],
                        check=True,
                    )
                    shots.append(stem.name + ".png")
                elif path.suffix == ".png":
                    shots.append(path.name)
            for path in files:
                text = path.read_text(encoding="latin-1") if path.suffix == ".svg" else ""
                if path.suffix == ".svg":
                    assert "<text" in text, "SVG labels are not text"
                    assert "sans-serif" in text, "SVG fonts have no generic fallback"
            return {"files": [f.name for f in files], "screenshots": shots}

        for name, check in (
            ("1 node dragging", dragging),
            ("2 panning and zooming", pan_zoom),
            ("3 graph edits round trip", edits),
            ("4a construct stage", construct_stage),
            ("4b flow stage and layers", flow_stage),
            ("4c corrections stage", corrections_stage),
            ("5 execution animation", execution_stage),
            ("6 layout scale", layout_scale),
            ("7 static exports", static_exports),
        ):
            c.run(name, check)
        version = browser.version
        browser.close()
    report = {"browser": version, "results": c.results}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return c.results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="output directory for pages and screenshots")
    parser.add_argument("--channel", default="msedge", help="installed browser: msedge or chrome")
    args = parser.parse_args(argv)
    results = run_checks(args.out, args.channel)
    failed = [r for r in results if not r["ok"]]
    for r in failed:
        print(f"--- {r['check']}\n{r['error']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
