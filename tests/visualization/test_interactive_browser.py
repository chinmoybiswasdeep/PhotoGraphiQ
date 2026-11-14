"""End-to-end check of the generated page in a real headless browser (Edge or Chrome).

The page runs with Cytoscape.js (pinned CDN, or a local copy named by the environment
variable PHOTOGRAPHIQ_CYTOSCAPE_JS), a harness script drives the editor and the
execution controls, and the exported JSON is re-validated by ``apply_edits``.
Skips when no browser is installed or Cytoscape.js cannot be loaded (offline).
"""

import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import photographiq as pg
from photographiq.cvflow import flow_pattern
from photographiq.interactive import (
    GraphView,
    apply_edits,
    execution_trace,
    squeezing_ellipses,
    wigner_plot,
)

BROWSERS = [
    shutil.which("msedge"),
    shutil.which("chrome"),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    shutil.which("chromium"),
    shutil.which("google-chrome"),
]
BROWSER = next((b for b in BROWSERS if b and Path(b).exists()), None)

ERRORS = (
    '<script>window.__errors = []; window.addEventListener("error", '
    "(e) => window.__errors.push(String(e.message)));</script>"
)
HARNESS = """<script>
(function () {
  const out = document.createElement("pre"); out.id = "harness"; document.body.appendChild(out);
  try {
    if (typeof cytoscape === "undefined") { out.textContent = "NO-CYTOSCAPE"; return; }
    const click = (id) => document.getElementById(id).click();
    const report = { errors: window.__errors.slice(), elements: cy.elements().length };
    report.ready = document.getElementById("step-label").textContent;
    click("step"); click("step"); click("step"); click("step"); click("step");
    report.fifth = document.getElementById("step-label").textContent;
    report.active = cy.nodes(".active").map((n) => n.id());
    report.display0 = cy.getElementById("0").data("display");
    click("reset");
    report.overlays = cy.edges(".correction").length;
    cy.getElementById("0").position({ x: 550, y: 330 });
    cy.elements().unselect(); cy.getElementById("0").select(); cy.getElementById("2").select();
    document.getElementById("weight").value = "-0.5"; click("add-edge");
    cy.elements().unselect(); cy.getElementById("1").select();
    document.getElementById("shear").value = "0.75"; click("set-meas");
    click("add-node");
    report.overlaysAfterEdit = cy.edges(".flow, .correction").length;
    HTMLAnchorElement.prototype.click = function () {};  // no real download in headless
    click("export");
    report.exported = JSON.parse(document.getElementById("export-text").value);
    report.errors = window.__errors.slice();
    out.textContent = JSON.stringify(report);
  } catch (error) { out.textContent = "ERROR " + error.stack; }
})();
</script>"""


def run_page(page: str, tmp_path: Path) -> str:
    path = tmp_path / "page.html"
    path.write_text(page, encoding="utf-8")
    command = [
        BROWSER,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={tmp_path / 'profile'}",
        "--dump-dom",
        path.as_uri(),
    ]
    if sys.platform.startswith("linux"):
        # Ubuntu 24.04+ (GitHub ubuntu-latest) restricts unprivileged user namespaces via
        # AppArmor, so headless Chrome cannot start its sandbox. The page is our own
        # generated file, so running it unsandboxed is safe.
        command.insert(1, "--no-sandbox")
    done = subprocess.run(command, capture_output=True, text=True, timeout=120, encoding="utf-8")
    match = re.search(r'<pre id="harness">(.*?)</pre>', done.stdout, flags=re.S)
    assert match, f"harness did not run: {done.stderr[-2000:]}"
    return html.unescape(match.group(1))


@pytest.mark.skipif(BROWSER is None, reason="No Chromium-based browser installed")
def test_editor_and_animation_in_browser(tmp_path):
    graph = pg.CVGraph.line(3, squeezing=1.5, inputs=(0,), outputs=(2,))
    pattern = flow_pattern(graph, (0, 1), shears={0: 0.3, 1: -0.5})
    result = pg.simulate(pattern, backend="gaussian", seed=11)
    trace = execution_trace(pattern, result)
    view = GraphView.from_pattern(pattern).with_corrections(pattern).with_execution(trace)
    page = view.to_html(cytoscape_js=os.environ.get("PHOTOGRAPHIQ_CYTOSCAPE_JS"))
    page = page.replace("<head>", "<head>" + ERRORS, 1).replace("</body>", HARNESS + "</body>")
    text = run_page(page, tmp_path)
    if text == "NO-CYTOSCAPE":
        pytest.skip("Cytoscape.js could not be loaded (offline); set PHOTOGRAPHIQ_CYTOSCAPE_JS")
    assert not text.startswith("ERROR"), text
    report = json.loads(text)
    assert report["errors"] == []
    assert report["ready"] == f"Ready: {len(trace.steps)} steps"
    assert report["fifth"] == f"Step 5/{len(trace.steps)}: {trace.steps[4].label}"
    assert report["active"] == ["0"] and trace.steps[4].kind == "measure"
    assert report["display0"] == f"0\nm={float(f'{result.outcomes[0]:.4g}')}"
    assert report["overlays"] == len(view.correction_arrows()) and report["overlaysAfterEdit"] == 0
    edited, layout = apply_edits(report["exported"])
    assert edited.weight(0, 2) == -0.5 and edited.measurements[1].shear == 0.75
    assert set(edited.nodes) == {0, 1, 2, 3} and edited.outputs == (2,)
    assert layout[0] == pytest.approx((550 / 110, -330 / 110))
    original = view.graph
    assert edited.weight(0, 1) == original.weight(0, 1) and edited.squeezing == original.squeezing


PLOT_HARNESS = """<script>
(function () {
  const out = document.createElement("pre"); out.id = "harness"; document.body.appendChild(out);
  if (typeof Plotly === "undefined") { out.textContent = "NO-PLOTLY"; return; }
  const plot = document.getElementById("plot");
  out.textContent = JSON.stringify({ errors: window.__errors.slice(),
    svg: plot.querySelectorAll(".main-svg").length, traces: plot.data.length });
})();
</script>"""


@pytest.mark.skipif(BROWSER is None, reason="No Chromium-based browser installed")
def test_plots_render_in_browser(tmp_path):
    state = pg.GaussianState([0.4, -0.2], [[0.5, 0.1], [0.1, 2.1]], ("m",))
    for plot in (wigner_plot(state, points=41), squeezing_ellipses(state)):
        page = plot.to_html(plotly_js=os.environ.get("PHOTOGRAPHIQ_PLOTLY_JS"))
        page = page.replace("<head>", "<head>" + ERRORS, 1)
        text = run_page(page.replace("</body>", PLOT_HARNESS + "</body>"), tmp_path)
        if text == "NO-PLOTLY":
            pytest.skip("Plotly.js could not be loaded (offline); set PHOTOGRAPHIQ_PLOTLY_JS")
        report = json.loads(text)
        assert report["errors"] == [] and report["svg"] >= 1
        assert report["traces"] == len(plot.traces)
