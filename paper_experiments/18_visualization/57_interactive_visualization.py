"""R57: Interactive visualization validation (one validated graph, four views).

Builds the complete-graph, flow, correction and execution views of ONE validated
graph (the compiled two-mode Gaussian circuit R(0.4) (+) S(0.3), then CZ(0.8), at
resource squeezing r = 1.5) and checks programmatically that

1. changing visual positions (every offered layout plus random manual layouts) leaves the
   OpenGraph, the pattern, the drawn flow/correction data and a seeded simulation
   bit-for-bit unchanged;
2. the drawn flow arrows equal ``flows.analyze(...).correction_map`` and the drawn
   correction arrows/labels equal the outcome-dependent ``Displace`` commands of the
   pattern, read here directly from the command expressions (independent of
   ``pattern_corrections``);
3. exported static SVG files match the model (node/edge/arrow groups and labels parsed
   from the SVG), and every routed edge keeps the documented clearance from other nodes;
4. the browser-format export (graph + layout JSON), layout files and OpenGraph JSON round
   trip restore positions and metadata exactly;
5. in a real browser (Playwright, installed Edge; real mouse/select/click events) a node
   drag, a stage switch and a layout switch keep the exported graph identical and the
   drawn arrows equal to the model; the full ``scripts/check_interactive_browser.py``
   suite runs as well;
6. the view renders as a notebook output (nbclient executes a notebook; the iframe
   ``srcdoc`` it produced is then opened in the browser).

Steps 5-6 need Playwright (``PYTHONPATH`` pointing at a venv that has it) and a local
Cytoscape.js copy in ``PHOTOGRAPHIQ_CYTOSCAPE_JS`` (or network access to the CDN). When
they cannot run they are recorded as ``not_run`` (never as passing); ``--require-browser``
turns that into a failure.

Usage: python 57_interactive_visualization.py [--require-browser]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.commands import Displace  # noqa: E402
from photographiq.expressions import resolve  # noqa: E402
from photographiq.flows import analyze  # noqa: E402
from photographiq.flows.cv import corrected_pattern, resource_graph  # noqa: E402
from photographiq.interactive import (  # noqa: E402
    GraphView,
    Layout,
    apply_edits,
    execution_trace,
    export_layout,
    import_layout,
)
from photographiq.interactive.graph_view import ARC  # noqa: E402
from photographiq.interactive.layouts import CLEARANCE, _clearance, route  # noqa: E402
from photographiq.ir import OpenGraph  # noqa: E402

NAME = "R57_interactive_visualization"
RESOURCE_SQUEEZING = 1.5
SEED = 57
MANUAL_LAYOUTS = 5
REPO = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = REPO / "scripts" / "check_interactive_browser.py"
SVG_NS = "{http://www.w3.org/2000/svg}"

# Acceptance criteria (declared before evaluation).
COEFFICIENT_TOL = 1e-12  # displayed correction coefficients vs. command expressions
POSITION_TOL = 1e-12  # layout round trip (exact float JSON round trip expected)
CHANNEL_TOL = 1e-9  # flow-map pattern vs compiled pattern channel (same tolerance as R60)
EXPECTED_FLOW_STATUS = "certified"


def build_circuit() -> pg.Circuit:
    circuit = pg.Circuit(2)
    circuit.rotate(0, 0.4)
    circuit.squeeze(1, 0.3)
    circuit.cz(0, 1, 0.8)
    return circuit


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def semantic_payload(view: GraphView, stage: str) -> dict:
    """Payload with every visual-only field (positions, bends, z-order) removed."""
    elements = []
    for element in view.elements(stage):
        data = {k: v for k, v in element["data"].items() if k not in ("bend", "z")}
        elements.append({"group": element["group"], "data": data, "classes": element["classes"]})
    return {"elements": sorted(elements, key=lambda e: json.dumps(e, sort_keys=True))}


def command_corrections(pattern) -> dict:
    """{(outcome key, node): {"q": c, "p": c}} straight from the Displace expressions.

    Coefficients are evaluated at unit records; affinity is checked at a random point.
    """
    rng = common.rng(SEED)
    table: dict = {}
    for command in pattern.commands:
        if not isinstance(command, Displace):
            continue
        keys = set()
        for value in (command.q, command.p):
            keys |= set(getattr(value, "dependencies", ()))
        if not keys:
            continue
        zero = {k: 0.0 for k in keys}
        for axis, value in (("q", command.q), ("p", command.p)):
            constant = resolve(value, zero, {})
            coefficients = {k: resolve(value, {**zero, k: 1.0}, {}) - constant for k in keys}
            point = {k: float(rng.normal()) for k in keys}
            affine = constant + sum(c * point[k] for k, c in coefficients.items())
            assert abs(resolve(value, point, {}) - affine) < 1e-9, "non-affine feed-forward"
            for key, c in coefficients.items():
                if c != 0:
                    entry = table.setdefault((key, command.node), {})
                    entry[axis] = entry.get(axis, 0.0) + c
    return table


def label_of(coefficients: dict) -> str:
    return " ".join(f"{a}:{coefficients[a]:.4g}" for a in ("q", "p") if a in coefficients)


def svg_groups(path: Path) -> dict:
    root = ET.parse(path).getroot()
    groups: dict = {"node": [], "edge": [], "flow": [], "correction": [], "label": {}}
    for g in root.iter(f"{SVG_NS}g"):
        gid = g.get("id") or ""
        kind, _, rest = gid.partition(":")
        if kind == "label":
            groups["label"][rest] = "".join(g.itertext()).strip()
        elif kind in groups:
            groups[kind].append(rest)
    return groups


def load_browser_checks():
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError as error:
        return None, f"playwright not importable: {error}"
    spec = importlib.util.spec_from_file_location("check_interactive_browser", CHECK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--require-browser", action="store_true")
    args = parser.parse_args(argv)
    plt = common.setup_style()
    rows: list[dict] = []

    def record(check, passed, measured, criterion):
        rows.append(
            {"check": check, "status": passed, "measured": str(measured), "criterion": criterion}
        )

    # ---- one validated graph -------------------------------------------------------
    circuit = build_circuit()
    pattern = circuit.compile(squeezing=RESOURCE_SQUEEZING)
    pattern_json = pattern.to_json()
    base = GraphView.from_pattern(pattern, title="Compiled R(0.4)+S(0.3), CZ(0.8)")
    graph = base.graph
    OpenGraph.from_dict(graph.to_dict())  # re-validates every invariant
    started = time.perf_counter()
    flow = analyze(graph)
    flow_seconds = time.perf_counter() - started
    record("flow status", flow.status == EXPECTED_FLOW_STATUS, flow.status, "certified")
    result = pg.simulate(pattern, backend="native-gaussian", seed=SEED)
    trace = execution_trace(pattern, result)
    views = {
        "construct": base.construct(),
        "flow": base.with_flow(flow),
        "corrections": base.with_flow(flow).with_corrections(pattern),
        "execution": base.with_flow(flow).with_corrections(pattern).with_execution(trace),
    }
    flow_corrections = base.with_flow(flow).with_corrections()  # the flow's own map

    # ---- 2. arrows equal validated data --------------------------------------------
    reported = {(s, t) for s, targets in flow.correction_map.items() for t in targets}
    drawn_flow = views["flow"].flow_arrows()
    record("flow arrows == flow.correction_map", drawn_flow == reported, len(drawn_flow), "equal")
    from_commands = command_corrections(pattern)
    drawn = views["corrections"].correction_arrows()
    want = {pair: label_of(c) for pair, c in from_commands.items()}
    record(
        "correction arrows == Displace commands", drawn == want, len(drawn), "equal pairs+labels"
    )
    worst = 0.0
    for target, correction in views["corrections"].corrections.items():
        for axis, terms in (("q", correction.q), ("p", correction.p)):
            for key, c in terms:
                worst = max(worst, abs(c - from_commands[(key, target)][axis]))
    record("correction coefficients", worst <= COEFFICIENT_TOL, worst, f"<= {COEFFICIENT_TOL}")
    executed_order = [c.node for c in pattern.commands if type(c).__name__ == "Measure"]
    layer = {n: i for i, lay in enumerate(flow.layers) for n in lay}
    respects = all(layer[s] < layer.get(t, math.inf) for s, t in reported)
    record("flow layers respect dependencies", respects, len(flow.layers), "every s before t")
    flow_vs_pattern = sorted(map(str, set(flow_corrections.correction_arrows()) ^ set(drawn)))
    shears = {u: graph.measurements[u].shear for u in flow.measurement_order}
    flow_only = corrected_pattern(
        resource_graph(graph), flow.measurement_order, flow.correction_map, shears
    )
    a, b = pg.gaussian_channel(pattern), pg.gaussian_channel(flow_only)
    channel_gap = max(
        float(np.abs(a.matrix - b.matrix).max()), float(np.abs(a.noise - b.noise).max())
    )
    record(
        "flow-map pattern channel == compiled channel",
        channel_gap <= CHANNEL_TOL,
        channel_gap,
        f"<= {CHANNEL_TOL}",
    )

    # ---- 1. visual changes never change semantics ----------------------------------
    reference = {stage: digest(semantic_payload(v, stage)) for stage, v in views.items()}
    graph_digest, outcomes = digest(graph.to_dict()), dict(result.outcomes)
    layouts = dict(views["execution"].layouts())
    rng = common.rng(SEED)
    for i in range(MANUAL_LAYOUTS):
        layouts[f"manual{i}"] = Layout({n: rng.normal(size=2) * 3 for n in graph.nodes})
    invariant = 0
    for name, layout in layouts.items():
        for stage, view in views.items():
            moved = view.with_layout(layout)
            same = (
                digest(moved.graph.to_dict()) == graph_digest
                and digest(semantic_payload(moved, stage)) == reference[stage]
                and moved.flow_arrows() == view.flow_arrows()
                and moved.correction_arrows() == view.correction_arrows()
            )
            invariant += same
    rerun = pg.simulate(pattern, backend="native-gaussian", seed=SEED)
    same_run = dict(rerun.outcomes) == outcomes and pattern.to_json() == pattern_json
    total = len(layouts) * len(views)
    record("semantics invariant under layouts", invariant == total, f"{invariant}/{total}", "all")
    record("pattern and seeded run unchanged", same_run, len(outcomes), "identical")

    # ---- 3. static SVG export matches the model; routing clearance -----------------
    svg_dir = common.RAW_DIR / "R57_exports"
    svg_dir.mkdir(parents=True, exist_ok=True)
    svg_ok = True
    for stage, view in views.items():
        groups = svg_groups(view.save(svg_dir / f"{stage}.svg", stage=stage))
        level_arrows = {
            "flow": {f"{s}->{t}" for s, t in view.flow_arrows()} if stage == "flow" else set(),
            "correction": {f"{s}->{t}" for s, t in view.correction_arrows()}
            if stage in ("corrections", "execution")
            else set(),
        }
        ok = (
            sorted(groups["node"]) == sorted(map(str, graph.nodes))
            and len(groups["edge"]) == graph.graph.number_of_edges()
            and set(groups["flow"]) == level_arrows["flow"]
            and set(groups["correction"]) == level_arrows["correction"]
            and groups["label"] == {str(n): str(n) for n in graph.nodes}
        )
        svg_ok &= ok
        record(
            f"static SVG {stage}",
            ok,
            f"{len(groups['node'])} nodes, {len(groups['edge'])} edges, "
            f"{len(groups['flow']) + len(groups['correction'])} arrows",
            "equal to model",
        )
    pos = views["execution"].layout
    gaps = []
    for u, v in graph.graph.edges:
        others = [xy for n, xy in pos.positions.items() if n not in (u, v)]
        gaps.append(_clearance(pos[u], pos[v], route(pos, u, v), others))
    for s, t in drawn_flow:
        others = [xy for n, xy in pos.positions.items() if n not in (s, t)]
        offset = route(pos, s, t, ARC * math.dist(pos[s], pos[t]))
        gaps.append(_clearance(pos[s], pos[t], offset, others))
    for s, t in drawn:
        others = [xy for n, xy in pos.positions.items() if n not in (s, t)]
        offset = route(pos, s, t, -ARC * math.dist(pos[s], pos[t]))
        gaps.append(_clearance(pos[s], pos[t], offset, others))
    unrouted = sum(g < CLEARANCE for g in gaps)
    record(
        "routed curves clear other nodes", unrouted == 0, f"min {min(gaps):.3f}", f">= {CLEARANCE}"
    )

    # ---- 4. serialization round trips ----------------------------------------------
    exported = json.dumps({"graph": graph.to_dict(), "layout": pos.to_dict()})
    edited, layout_back = apply_edits(exported)
    drift = max(max(abs(a - b) for a, b in zip(layout_back[n], pos[n])) for n in graph.nodes)
    round_trip = edited.to_dict() == graph.to_dict() and drift <= POSITION_TOL
    record(
        "editor JSON round trip", round_trip, f"position drift {drift:.1e}", f"<= {POSITION_TOL}"
    )
    layout_file = svg_dir / "layout.json"
    export_layout(pos, layout_file)
    record("layout file round trip", import_layout(layout_file, graph) == pos, "exact", "exact")
    reloaded = OpenGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    meta_ok = (
        reloaded.measurements == graph.measurements
        and reloaded.squeezing == graph.squeezing
        and reloaded.inputs == graph.inputs
        and reloaded.outputs == graph.outputs
    )
    record("OpenGraph metadata round trip", meta_ok, "measurements, squeezing, I/O", "exact")

    # ---- 5-6. browser and notebook ----------------------------------------------------
    module, why = load_browser_checks()
    browser = {"status": "not_run", "reason": why}
    if module is not None:
        out = common.RAW_DIR / "R57_browser"
        try:
            suite = module.run_checks(out, "msedge")
            browser = {
                "status": "ran",
                "suite": {r["check"]: r["ok"] for r in suite},
                "failures": {r["check"]: r["error"][-600:] for r in suite if not r["ok"]},
            }
            browser.update(r57_browser(module, views["execution"], graph, out))
        except Exception as error:  # record and fail below, never silently pass
            browser = {"status": "error", "reason": repr(error)}
    if browser["status"] == "not_run":
        record("browser real-input checks", "not_run", why, "Playwright + Edge")
    else:
        suite_ok = browser["status"] == "ran" and not browser.get("failures")
        record(
            "browser suite (check_interactive_browser)", suite_ok, browser.get("suite"), "all pass"
        )
        r57_ok = browser.get("r57", {}).get("ok", False)
        record("browser drag/stage/layout on R57 graph", r57_ok, browser.get("r57"), "all pass")
    notebook = notebook_check(views["execution"], module)
    record("notebook execution", notebook["executed"], notebook.get("reason", ""), "nbclient")
    if notebook.get("browser") is not None:
        record(
            "notebook iframe renders in browser",
            notebook["browser"],
            notebook.get("nodes"),
            "== model",
        )
    else:
        record(
            "notebook iframe renders in browser",
            "not_run",
            notebook.get("reason", ""),
            "Playwright",
        )

    # ---- outputs ---------------------------------------------------------------------
    failed = [r for r in rows if r["status"] is False]
    if args.require_browser:
        failed += [r for r in rows if r["status"] == "not_run"]
    for row in rows:
        row["status"] = {True: "pass", False: "FAIL"}.get(row["status"], row["status"])
    views["execution"].save_html(common.RAW_DIR / "R57_interactive.html", "execution")
    extra = {
        "circuit": "Circuit(2).rotate(0, 0.4).squeeze(1, 0.3).cz(0, 1, 0.8)",
        "resource_squeezing_r": RESOURCE_SQUEEZING,
        "backend": "native-gaussian",
        "seed": SEED,
        "graph_nodes": len(graph.nodes),
        "graph_edges": graph.graph.number_of_edges(),
        "flow_status": flow.status,
        "flow_algorithm": flow.algorithm,
        "flow_layers": [list(layer) for layer in flow.layers],
        "flow_seconds_provisional_shared_machine": flow_seconds,
        "executed_measurement_order": executed_order,
        "flow_map_vs_pattern_feedforward_symmetric_difference": flow_vs_pattern,
        "flow_map_channel_difference": channel_gap,
        "layouts_checked": sorted(layouts),
        "min_route_clearance": min(gaps),
        "browser": browser,
        "notebook": notebook,
        "outcome_keys": len(outcomes),
    }
    if failed:
        common.save_csv(failed, f"{NAME}_FAILURES")
    common.save_result(rows, NAME, extra=extra)
    figure(plt, views)
    caption(views, flow, rows, extra)
    common.print_summary(
        "R57 interactive visualization",
        checks=len(rows),
        passed=sum(r["status"] == "pass" for r in rows),
        not_run=sum(r["status"] == "not_run" for r in rows),
        failed=len(failed),
    )
    if failed:
        raise AssertionError(f"R57 checks failed: {[r['check'] for r in failed]}")


def r57_browser(module, view, graph, out: Path) -> dict:
    """Real input on the R57 graph: drag, stage switch, layout switch, export, re-validate."""
    from playwright.sync_api import sync_playwright

    evidence: dict = {"ok": False}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        checker = module.Checker(browser, out)
        page = checker.page(view, "R57_graph", "construct")
        node = str(graph.nodes[len(graph.nodes) // 2])
        before = module.model_positions(page)
        module.drag(page, module.node_xy(page, node), (70, 90))
        after = module.model_positions(page)
        moved_only = [n for n in after if after[n] != before[n]] == [node]
        stage_arrows = {}
        for stage in ("flow", "corrections", "execution"):
            page.select_option("#stage-select", stage)
            selector = ".flow" if stage == "flow" else ".correction"
            got = {(a, b) for a, b, *_ in module.drawn(page, selector)}
            want = {
                (str(s), str(t))
                for s, t in (view.flow_arrows() if stage == "flow" else view.correction_arrows())
            }
            stage_arrows[stage] = got == want
        page.select_option("#layout-select", "layered")
        layered = module.model_positions(page)
        want = view.layouts()["layered"]
        layout_ok = all(
            abs(layered[str(n)][0] - x * module.SCALE) < 1e-6
            and abs(layered[str(n)][1] + y * module.SCALE) < 1e-6
            for n, (x, y) in want.positions.items()
        )
        exported, layout = apply_edits(module.export(page))
        semantics = exported.to_dict() == graph.to_dict()
        page.screenshot(path=str(out / "R57_graph_after_input.png"))
        evidence.update(
            {
                "dragged_node": node,
                "only_dragged_node_moved": moved_only,
                "stage_arrows_equal_model": stage_arrows,
                "layered_layout_positions_equal": layout_ok,
                "exported_graph_equal": semantics,
                "page_errors": page.errors,
            }
        )
        evidence["ok"] = (
            moved_only
            and all(stage_arrows.values())
            and layout_ok
            and semantics
            and not page.errors
        )
        browser.close()
    return {"r57": evidence}


def notebook_check(view, module) -> dict:
    try:
        import nbclient
        import nbformat
    except ImportError as error:
        return {"executed": "not_run", "browser": None, "reason": str(error)}
    library = os.environ.get("PHOTOGRAPHIQ_CYTOSCAPE_JS")
    source = (
        "import photographiq as pg\n"
        "from photographiq.interactive import GraphView\n"
        f"view = GraphView.from_pattern(pg.Pattern.from_json({view_pattern_json()!r}))\n"
    )
    if library:  # offline: the notebook page inlines the local Cytoscape.js copy
        source += (
            "_html = view.to_html\n"
            "object.__setattr__(view, 'to_html', "
            f"lambda *a, **k: _html(*a, cytoscape_js={library!r}, **k))\n"
        )
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(source), nbformat.v4.new_code_cell("view")]
    try:
        nbclient.NotebookClient(notebook, timeout=300, kernel_name="python3").execute()
    except Exception as error:  # executed but failed: a real failure
        return {"executed": False, "browser": None, "reason": repr(error)[:400]}
    html = next(
        o["data"]["text/html"]
        for o in notebook.cells[1].outputs
        if "text/html" in o.get("data", {})
    )
    result: dict = {"executed": "<iframe srcdoc=" in html, "browser": None}
    if module is None:
        result["reason"] = "iframe produced; browser rendering not checked (no Playwright)"
        return result
    import html as htmllib
    import re

    page_source = htmllib.unescape(re.search(r'srcdoc="(.*?)" style', html, flags=re.S).group(1))
    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "notebook_output.html"
        path.write_text(page_source, encoding="utf-8")
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page()
            errors: list = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(path.as_uri())
            nodes = page.evaluate("typeof cy === 'undefined' ? -1 : cy.nodes().length")
            edges = page.evaluate("typeof cy === 'undefined' ? -1 : cy.edges('.physical').length")
            browser.close()
    graph = view.graph
    result["nodes"] = {"nodes": nodes, "edges": edges, "errors": errors}
    result["browser"] = (
        nodes == len(graph.nodes) and edges == graph.graph.number_of_edges() and not errors
    )
    return result


def view_pattern_json() -> str:
    return build_circuit().compile(squeezing=RESOURCE_SQUEEZING).to_json()


def figure(plt, views) -> None:
    from photographiq.interactive import draw_static
    from photographiq.interactive.static import legend_handles

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 7.6))
    titles = {
        "construct": "(a) complete weighted graph (CZ weights, resource squeezing r)",
        "flow": "(b) certified CV flow: dependencies and measurement order #k",
        "corrections": "(c) executed corrections (coefficients of the raw record m)",
        "execution": "(d) one recorded native-gaussian run (outcomes m)",
    }
    for ax, (stage, view) in zip(axes.flat, views.items()):
        draw_static(
            view,
            stage=stage,
            ax=ax,
            formulas=False,
            legend=False,
            title=titles[stage],
            node_size=260,
            fontsize=7.0,
            weights=stage == "construct",
        )
        ax.title.set_fontsize(8)
    handles = legend_handles("flow") + legend_handles("correction")[-1:]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.4)
    common.save_figure(fig, NAME, tight=False)
    plt.close(fig)


def caption(views, flow, rows, extra) -> None:
    passed = sum(r["status"] == "pass" for r in rows)
    not_run = [r["check"] for r in rows if r["status"] == "not_run"]
    view = views["corrections"]
    common.save_caption(
        NAME,
        f"""
**R57: one validated graph, four views.** (a) Complete weighted CV cluster graph of the
circuit R(0.4) on mode 0, S(0.3) on mode 1, then CZ(0.8), compiled by
`Circuit.compile(squeezing={RESOURCE_SQUEEZING})`: {extra["graph_nodes"]} nodes,
{extra["graph_edges"]} CZ edges (weights printed on the edges), blue squares are the inputs,
green diamonds the outputs, thick outlines mark measured nodes. (b) Flow found by
`flows.analyze` (status `{flow.status}`, {len(flow.layers)} measurement layers, `#k` gives the
measurement order); blue arrows are exactly the reported `correction_map`
({len(view.flow_arrows())} dependencies). (c) Classical corrections the compiled pattern
executes ({len(view.correction_arrows())} outcome-dependent displacements, orange dashed,
coefficients of the raw homodyne record), read from its `Displace` commands. (d) One
trajectory simulated with `native-gaussian` (seed {SEED}); `m=` values are the recorded
outcomes. The flow's correction map is the standard-form map (corrections after all CZs);
the compiled pattern applies each correction before the next CZ, so the two arrow sets
differ by {len(extra["flow_map_vs_pattern_feedforward_symmetric_difference"])} arrows; a
pattern executing exactly the flow map has the same channel as the compiled pattern (largest
difference {extra["flow_map_channel_difference"]:.1e}).
Validation: {passed}/{len(rows)} checks pass (layout invariance over
{len(extra["layouts_checked"])} layouts, SVG export parsed against the model, JSON round
trips, routed-curve clearance >= {CLEARANCE} layout units, minimum
{extra["min_route_clearance"]:.3f}); not run: {not_run or "none"}. Positions are display
metadata only. Limitations: browser checks use headless Edge only (no touch, Firefox or
WebKit); finite squeezing r = {RESOURCE_SQUEEZING} adds noise that this figure does not show.
Source: `paper_experiments/18_visualization/57_interactive_visualization.py`.
""",
    )


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
