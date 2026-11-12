"""Inline HTML/JS template of the interactive graph view (Cytoscape.js).

The browser only displays and edits: every number shown was computed in Python and
embedded as JSON. Edits are exported as ``{graph: OpenGraph.to_dict(), layout}`` and
must be re-validated with :func:`photographiq.interactive.apply_edits`.
Placeholders: ``__TITLE__`` (escaped text), ``__CYTOSCAPE__`` (script tag), ``__DATA__``.
"""

GRAPH_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --ink:#0f172a; --muted:#475569; --line:#cbd5e1; --panel:#f8fafc; --accent:#2563eb; }
* { box-sizing:border-box; }
body { margin:0; font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:#fff; }
header { padding:8px 12px; border-bottom:1px solid var(--line); display:flex; gap:12px; flex-wrap:wrap; align-items:baseline; }
header h1 { font-size:15px; margin:0; }
.badge { font-size:11px; padding:1px 7px; border-radius:9px; background:#e2e8f0; }
.badge.ok { background:#dcfce7; color:#166534; } .badge.warn { background:#fef3c7; color:#92400e; }
main { display:grid; grid-template-columns:300px 1fr; height:calc(100vh - 42px); min-height:420px; }
aside { overflow:auto; padding:8px 10px; border-right:1px solid var(--line); background:var(--panel); }
#cy { width:100%; height:100%; min-height:360px; }
section { margin-bottom:10px; } h2 { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:6px 0; }
button, select, input { font:inherit; margin:1px 0; } button { cursor:pointer; padding:2px 7px; }
input[type=number] { width:76px; }
pre, #info { white-space:pre-wrap; margin:0; font:12px/1.35 ui-monospace,Consolas,monospace; }
#steps { padding-left:20px; margin:4px 0; max-height:220px; overflow:auto; font-size:12px; }
#steps li.current { font-weight:600; color:#b91c1c; } #steps li { cursor:pointer; }
textarea { width:100%; height:90px; font:11px ui-monospace,Consolas,monospace; }
.legend span { display:inline-block; margin-right:8px; } .hidden { display:none; }
#status { color:#92400e; min-height:1em; }
@media (max-width:700px) { main { grid-template-columns:1fr; height:auto; } #cy { height:70vh; } }
</style></head>
<body>
<header><h1 id="title"></h1><span class="badge" id="stage"></span><span class="badge" id="model"></span></header>
<main>
<aside>
<section class="legend"><span>&#9632; input</span><span>&#9670; output</span><span>&#9679; measured</span><br>
<span>&#9472; CZ edge</span><span style="color:#2563eb">- - &#9654; dependency</span><span style="color:#c2410c">&middot;&middot; &#9654; correction</span></section>
<section id="flow-panel" class="hidden"><h2>Flow analysis</h2><pre id="flow"></pre></section>
<section id="trace-panel" class="hidden"><h2>Execution</h2>
<button id="play">Play</button> <button id="pause">Pause</button> <button id="step">Step</button> <button id="reset">Reset</button>
<div id="step-label"></div><ol id="steps"></ol></section>
<section id="edit-panel" class="hidden"><h2>Edit</h2>
<button id="add-node">Add node</button> <button id="remove">Remove selected</button><br>
weight <input id="weight" type="number" value="1" step="0.1"> <button id="add-edge">Add edge (2 selected)</button> <button id="set-weight">Set weight</button><br>
<button id="toggle-input">Toggle input</button> <button id="toggle-output">Toggle output</button><br>
<span id="cv-edit">shear k <input id="shear" type="number" value="0" step="0.1"> squeezing r <input id="squeeze" type="number" value="1" step="0.1" min="0"><br><button id="set-squeeze">Set squeezing</button> </span>
<span id="qubit-edit">plane <select id="plane"><option>XY</option><option>YZ</option><option>XZ</option></select> angle <input id="angle" type="number" value="0" step="0.1">
pauli <select id="pauli"><option value="">none</option><option>X</option><option>Y</option><option>Z</option></select><br></span>
<button id="set-meas">Set measurement</button> <button id="clear-meas">Clear measurement</button><br>
<button id="export">Export JSON</button>
<textarea id="export-text" class="hidden" readonly></textarea></section>
<section><div id="status"></div><h2>Selection</h2><div id="info">Click or hover a node or edge.</div></section>
<section id="notes-panel" class="hidden"><h2>Notes</h2><pre id="notes"></pre></section>
</aside>
<div id="cy"></div>
</main>
__CYTOSCAPE__
<script>
"use strict";
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
const fmt = (x) => (typeof x === "number" ? Number(x.toPrecision(4)).toString() : String(x));
function say(message) { $("status").textContent = message; }
$("title").textContent = DATA.title;
$("stage").textContent = "stage: " + DATA.stage;
$("model").textContent = DATA.model;

const STYLE = [
  { selector: "node", style: { label: "data(display)", "text-wrap": "wrap", "text-valign": "center",
    "text-halign": "center", "font-size": 11, width: 42, height: 42, "background-color": "#a8c7e0",
    "border-width": 1.5, "border-color": "#334155", color: "#0f172a", shape: "ellipse" } },
  { selector: "node.input", style: { shape: "round-rectangle", "background-color": "#7fd1d6" } },
  { selector: "node.output", style: { shape: "diamond", "background-color": "#f3c766", width: 48, height: 48 } },
  { selector: "edge.physical", style: { width: 2.5, "line-color": "#64748b", label: "data(label)",
    "font-size": 10, "text-background-color": "#fff", "text-background-opacity": 0.85,
    "text-background-padding": 2, "curve-style": "straight" } },
  { selector: "edge.flow", style: { width: 2, "line-color": "#2563eb", "line-style": "dashed",
    "target-arrow-shape": "triangle", "target-arrow-color": "#2563eb", "curve-style": "unbundled-bezier",
    "control-point-distances": "data(bend)", "control-point-weights": [0.5], "arrow-scale": 1.2,
    "z-index": "data(z)" } },
  { selector: "edge.correction", style: { width: 2, "line-color": "#c2410c", "line-style": "dotted",
    "target-arrow-shape": "triangle-tee", "target-arrow-color": "#c2410c", "curve-style": "unbundled-bezier",
    "control-point-distances": "data(bend)", "control-point-weights": [0.5], label: "data(label)",
    "font-size": 9, color: "#c2410c", "text-background-color": "#fff", "text-background-opacity": 1,
    "text-background-padding": 1, "z-index": "data(z)" } },
  { selector: ".active", style: { "border-width": 5, "border-color": "#dc2626", "line-color": "#dc2626" } },
  { selector: "node.source", style: { "border-width": 4, "border-color": "#2563eb" } },
  { selector: "node.done", style: { "background-opacity": 0.45, "border-opacity": 0.45 } },  // text stays legible
  { selector: ":selected", style: { "overlay-color": "#7c3aed", "overlay-opacity": 0.25 } },
];

const cy = cytoscape({ container: $("cy"), elements: DATA.elements, layout: { name: "preset" },
  style: STYLE, wheelSensitivity: 0.25 });
const state = { inputs: DATA.inputs.slice(), outputs: DATA.outputs.slice(), outcomes: {}, overlays: true };

function roleClasses(node) {
  const isInput = state.inputs.includes(node.id()), isOutput = state.outputs.includes(node.id());
  node.toggleClass("input", isInput).toggleClass("output", isOutput)
    .toggleClass("measured", !isOutput).toggleClass("ancilla", !isInput);
}
function refresh(node) {
  roleClasses(node);
  let text = node.data("label");
  if (node.data("order") !== null && node.data("order") !== undefined) text += "\n#" + node.data("order");
  if (state.outcomes[node.id()] !== undefined) text += "\nm=" + fmt(state.outcomes[node.id()]);
  node.data("display", text);
}
function refreshAll() { cy.nodes().forEach(refresh); }
refreshAll();
cy.fit(undefined, 30);
if (cy.zoom() > 1.3) { cy.zoom(1.3); cy.center(); }

function measText(node) {
  const m = node.data("meas");
  if (!m) return node.data("measText") || (state.outputs.includes(node.id()) ? "output (not measured)" : "unspecified");
  if ("shear" in m) return "homodyne angle " + fmt(Math.atan2(1, m.shear)) + " rad (shear k=" + fmt(m.shear) + ")";
  return (m.pauli ? "Pauli " + m.pauli + ", " : "") + m.plane + " plane, angle " + fmt(m.angle);
}
function describe(ele) {
  const lines = [];
  if (ele.isNode()) {
    const roles = ["input", "output", "measured", "ancilla"].filter((r) => ele.hasClass(r));
    lines.push("node " + ele.data("label"), "roles: " + roles.join(", "));
    if (ele.data("squeezing") !== null && ele.data("squeezing") !== undefined) lines.push("squeezing r = " + fmt(ele.data("squeezing")));
    lines.push("measurement: " + measText(ele));
    if (ele.data("order") !== null && ele.data("order") !== undefined) lines.push("measurement order #" + ele.data("order"));
    if (ele.data("layer") !== null && ele.data("layer") !== undefined) lines.push("layer " + ele.data("layer"));
    if (ele.data("formula")) lines.push("correction: " + ele.data("formula"));
    if (state.outcomes[ele.id()] !== undefined) lines.push("recorded outcome: " + fmt(state.outcomes[ele.id()]));
    if (ele.data("annotation")) lines.push("note: " + ele.data("annotation"));
  } else if (ele.hasClass("physical")) {
    lines.push("CZ edge " + ele.source().data("label") + " - " + ele.target().data("label"), "weight g = " + fmt(ele.data("weight")));
  } else {
    lines.push((ele.hasClass("flow") ? "classical dependency: " : "correction: ") + ele.data("description"));
  }
  $("info").textContent = lines.join("\n");
}
cy.on("tap mouseover", "node, edge", (event) => describe(event.target));
cy.on("data", "node, edge", (event) => { if (event.target.selected()) describe(event.target); });

if (DATA.flow) {
  const f = DATA.flow;
  const lines = ["status: " + f.status, "algorithm: " + f.algorithm];
  if (f.order.length) lines.push("measurement order: " + f.order.join(", "));
  if (f.layers.length) lines.push("layers: " + f.layers.map((l) => "[" + l.join(", ") + "]").join(" "));
  lines.push("assumptions:" + (f.assumptions.length ? "" : " none reported"));
  f.assumptions.forEach((a) => lines.push("  - " + a));
  if (f.certificate) lines.push("certificate: " + f.certificate);
  if (f.diagnostics) lines.push("diagnostics: " + f.diagnostics);
  $("flow").textContent = lines.join("\n");
  $("flow-panel").classList.remove("hidden");
}
if (DATA.notes.length) { $("notes").textContent = DATA.notes.join("\n"); $("notes-panel").classList.remove("hidden"); }

// ---- Stage D: execution trace -------------------------------------------------
let current = -1, timer = null;
function showStep(index) {
  const steps = DATA.trace.steps;
  current = Math.max(-1, Math.min(index, steps.length - 1));
  cy.elements().removeClass("active source done");
  state.outcomes = {};
  for (let j = 0; j <= current; j++) {
    const s = steps[j];
    if (s.kind === "measure") {
      s.nodes.forEach((n) => { state.outcomes[n] = s.outcome; if (j < current) cy.getElementById(n).addClass("done"); });
    }
  }
  if (current >= 0) {
    const s = steps[current];
    s.nodes.forEach((n) => cy.getElementById(n).addClass("active"));
    if (s.kind === "entangle" && s.nodes.length === 2) {
      cy.getElementById(s.nodes[0]).edgesWith(cy.getElementById(s.nodes[1])).filter(".physical").addClass("active");
    }
    s.sources.forEach((n) => cy.getElementById(n).addClass("source"));
  }
  refreshAll();
  $("step-label").textContent = current < 0 ? "Ready: " + steps.length + " steps" :
    "Step " + (current + 1) + "/" + steps.length + ": " + steps[current].label;
  Array.from($("steps").children).forEach((li, j) => li.classList.toggle("current", j === current));
}
function pause() { if (timer) { clearInterval(timer); timer = null; } }
if (DATA.trace) {
  DATA.trace.steps.forEach((s, j) => {
    const li = document.createElement("li");
    li.textContent = s.label;
    li.addEventListener("click", () => { pause(); showStep(j); });
    $("steps").appendChild(li);
  });
  $("play").onclick = () => {
    pause();
    if (current >= DATA.trace.steps.length - 1) showStep(-1);
    timer = setInterval(() => { if (current >= DATA.trace.steps.length - 1) pause(); else showStep(current + 1); }, 900);
  };
  $("pause").onclick = pause;
  $("step").onclick = () => { pause(); showStep(current + 1); };
  $("reset").onclick = () => { pause(); showStep(-1); };
  $("trace-panel").classList.remove("hidden");
  showStep(-1);
}

// ---- Editing (semantics are re-validated in Python by apply_edits) -------------
function clearOverlays() {
  if (!state.overlays) return;
  state.overlays = false;
  pause();
  cy.remove("edge.flow, edge.correction");
  cy.nodes().forEach((n) => n.data({ order: null, layer: null, formula: null }));
  state.outcomes = {};
  $("trace-panel").classList.add("hidden");
  $("flow-panel").classList.add("hidden");
  say("Semantic edit: flow, correction and execution overlays were removed. Re-run the analysis in Python.");
}
function selectedNodes() { return cy.nodes(":selected"); }
function newRaw() {
  const raws = cy.nodes().map((n) => n.data("raw"));
  if (raws.every(Number.isInteger)) return raws.length ? Math.max(...raws) + 1 : 0;
  let k = raws.length;
  while (cy.getElementById("n" + k).nonempty()) k++;
  return "n" + k;
}
let edgeCounter = 0;
function numberFrom(id) {
  const x = Number($(id).value);
  if (!Number.isFinite(x)) { say(id + " must be a finite number"); return null; }
  return x;
}
if (DATA.editable) {
  $("edit-panel").classList.remove("hidden");
  $(DATA.continuous ? "qubit-edit" : "cv-edit").classList.add("hidden");
  $("add-node").onclick = () => {
    const raw = newRaw(), box = cy.extent();
    const node = cy.add({ group: "nodes", data: { id: String(raw), raw: raw, label: String(raw), squeezing: null,
      meas: null, order: null, layer: null, formula: null }, position: { x: (box.x1 + box.x2) / 2, y: (box.y1 + box.y2) / 2 } });
    clearOverlays(); refresh(node); say("Added node " + raw + " (drag to place).");
  };
  $("remove").onclick = () => {
    const doomed = cy.$("node:selected, edge.physical:selected");
    if (!doomed.length) { say("Select nodes or CZ edges to remove."); return; }
    doomed.nodes().forEach((n) => {
      state.inputs = state.inputs.filter((i) => i !== n.id());
      state.outputs = state.outputs.filter((o) => o !== n.id());
    });
    clearOverlays(); doomed.remove(); refreshAll();
  };
  $("add-edge").onclick = () => {
    const pick = selectedNodes(), w = numberFrom("weight");
    if (pick.length !== 2 || w === null) { say("Select exactly two nodes and give a weight."); return; }
    const [a, b] = [pick[0], pick[1]];
    if (a.edgesWith(b).filter(".physical").length) { say("Edge already exists; use Set weight."); return; }
    cy.add({ group: "edges", classes: "physical", data: { id: "p:new" + (edgeCounter++), source: a.id(),
      target: b.id(), weight: w, label: DATA.continuous ? fmt(w) : "" } });
    clearOverlays();
  };
  $("set-weight").onclick = () => {
    const w = numberFrom("weight"), edges = cy.edges(".physical:selected");
    if (w === null || !edges.length) { say("Select CZ edges and give a weight."); return; }
    edges.forEach((e) => e.data({ weight: w, label: DATA.continuous ? fmt(w) : "" }));
    clearOverlays();
  };
  const toggle = (list) => () => {
    const pick = selectedNodes();
    if (!pick.length) { say("Select nodes first."); return; }
    pick.forEach((n) => {
      const i = state[list].indexOf(n.id());
      if (i >= 0) state[list].splice(i, 1); else state[list].push(n.id());
      if (list === "outputs" && i < 0 && n.data("meas")) { n.data("meas", null); say("Outputs are not measured: cleared the measurement of " + n.id()); }
    });
    clearOverlays(); refreshAll();
  };
  $("toggle-input").onclick = toggle("inputs");
  $("toggle-output").onclick = toggle("outputs");
  $("set-squeeze").onclick = () => {
    const r = numberFrom("squeeze"), pick = selectedNodes();
    if (r === null || !pick.length) { say("Select nodes and give a squeezing value."); return; }
    pick.forEach((n) => n.data("squeezing", r)); clearOverlays();
  };
  $("set-meas").onclick = () => {
    const pick = selectedNodes().filter((n) => !state.outputs.includes(n.id()));
    if (!pick.length) { say("Select measured (non-output) nodes."); return; }
    let spec;
    if (DATA.continuous) { const k = numberFrom("shear"); if (k === null) return; spec = { shear: k }; }
    else { const a = numberFrom("angle"); if (a === null) return; spec = { plane: $("plane").value, angle: a, pauli: $("pauli").value || null }; }
    pick.forEach((n) => n.data("meas", Object.assign({}, spec))); clearOverlays();
  };
  $("clear-meas").onclick = () => { selectedNodes().forEach((n) => n.data("meas", null)); clearOverlays(); };
  $("export").onclick = () => {
    const raw = (id) => cy.getElementById(id).data("raw");
    const measurements = {}, layout = {};
    cy.nodes().forEach((n) => {
      if (n.data("meas")) measurements[n.id()] = n.data("meas");
      const p = n.position();
      layout[n.id()] = [p.x / DATA.scale, -p.y / DATA.scale];
    });
    const graph = {
      model: DATA.model,
      nodes: cy.nodes().map((n) => (n.data("squeezing") === null || n.data("squeezing") === undefined)
        ? { id: n.data("raw") } : { id: n.data("raw"), squeezing: n.data("squeezing") }),
      edges: cy.edges(".physical").map((e) => ({ source: e.source().data("raw"), target: e.target().data("raw"), weight: e.data("weight") })),
      inputs: state.inputs.map(raw), outputs: state.outputs.map(raw), measurements: measurements,
    };
    const text = JSON.stringify({ graph: graph, layout: layout }, null, 1);
    $("export-text").value = text; $("export-text").classList.remove("hidden");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    link.download = "photographiq-graph.json";
    document.body.appendChild(link); link.click(); link.remove();
    say("Exported; load with photographiq.interactive.apply_edits(json_text).");
  };
}
</script>
</body></html>
"""
