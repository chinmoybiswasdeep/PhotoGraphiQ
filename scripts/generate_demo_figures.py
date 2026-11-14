"""Generate the demonstration figures in docs/figures/demos/ from real library calls.

Every panel is computed by PhotoGraphiQ (graphs, flows, corrections, simulations,
sampling, training) or by an independent closed form stated in the caption; nothing is
drawn by hand. Each demo writes ``<name>.svg``, ``<name>.pdf``, ``<name>.png`` (300 dpi)
and ``<name>.caption.md``; graph demos also write ``<name>.interactive.html`` (all
available stages, Cytoscape.js from the pinned CDN). Captions quote numbers computed here.

Visual language (shared with ``photographiq.interactive``): CZ edges neutral solid grey;
flow dependencies solid blue arrows; classical corrections dashed vermillion arrows;
inputs blue squares, outputs green diamonds, ancillas neutral circles, measured nodes
thick dark outline; white background, vector output, Okabe-Ito colours.

Usage::

    python scripts/generate_demo_figures.py [--only NAME[,NAME...]] [--quick] [--list]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.cvflow import flow_pattern  # noqa: E402
from photographiq.flows import analyze  # noqa: E402
from photographiq.interactive import (  # noqa: E402
    GraphView,
    covariance_heatmap,
    draw_static,
    execution_trace,
    optimization_curves,
    wigner_plot,
)
from photographiq.interactive.static import COLORS, RC, legend_handles  # noqa: E402
from photographiq.ir import Model, OpenGraph, QubitMeasurement  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "demos"
DEMOS: dict[str, Callable[[argparse.Namespace], None]] = {}
SOURCE = "scripts/generate_demo_figures.py"
STYLE = {
    **RC,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
SMALL = {"node_size": 260, "fontsize": 7.0, "formulas": False, "legend": False}


def demo(name: str):
    def register(fn):
        DEMOS[name] = fn
        return fn

    return register


def save(name: str, fig, caption: str, view: GraphView | None = None) -> None:
    """Write the figure in three formats, the caption and (optionally) the HTML page."""
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    text = caption.strip() + f"\n\nSource: `{SOURCE}` (`{name}`).\n"
    (OUT / f"{name}.caption.md").write_text(text, encoding="utf-8")
    if view is not None:
        view.save_html(OUT / f"{name}.interactive.html")
    print(f"wrote {name}", flush=True)


def legend_below(fig, kinds=("flow", "correction"), y=0.0) -> None:
    handles = legend_handles(kinds[0]) + [h for k in kinds[1:] for h in legend_handles(k)[4:]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=3, frameon=False)


def panel(ax, letter: str, text: str) -> None:
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=8)


def ellipse(ax, mean, cov, *, nsig=1.0, **style) -> None:
    values, vectors = np.linalg.eigh(np.asarray(cov, dtype=float))
    t = np.linspace(0, 2 * np.pi, 200)
    pts = vectors @ (
        nsig * np.sqrt(np.maximum(values, 0))[:, None] * np.stack([np.cos(t), np.sin(t)])
    )
    ax.plot(mean[0] + pts[0], mean[1] + pts[1], **style)


def qubit_six() -> OpenGraph:
    network = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (1, 5)])
    measurements = {n: QubitMeasurement("XY", 0.1 * n) for n in (0, 1, 2, 3)}
    return OpenGraph(Model.QUBIT, network, (0,), (4, 5), measurements)


def cv_wire():
    graph = pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,))
    shears = {0: 0.3, 1: -0.5, 2: 0.0}
    open_graph = OpenGraph.from_cvgraph(graph, shears)
    flow = analyze(open_graph)
    return graph, shears, open_graph, flow


# ---------------------------------------------------------------------------------- 1
@demo("01_single_mode_teleportation")
def teleportation(args):
    r = 1.5
    pattern = pg.protocols.wire((0.0,), squeezing=r)
    alpha = 0.8 + 0.4j
    inputs = {0: pg.GaussianInput.coherent(alpha)}
    result = pg.simulate(pattern, backend="native-gaussian", seed=1, inputs=inputs)
    base = GraphView.from_pattern(pattern, title="One-step CV teleportation")
    flow = analyze(base.graph)
    view = (
        base.with_flow(flow)
        .with_corrections(pattern)
        .with_execution(execution_trace(pattern, result))
    )
    channel = pg.gaussian_channel(pattern)
    ideal = np.array([[0.0, -1.0], [1.0, 0.0]])  # Fourier transform F: q -> -p, p -> q
    mean_in = np.array([2 * alpha.real, 2 * alpha.imag])
    fig = plt.figure(figsize=(7.0, 2.8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], wspace=0.25, hspace=0.5)
    ax = fig.add_subplot(grid[0, 0])
    draw_static(view, stage="corrections", ax=ax, title="", **SMALL)
    panel(ax, "a", "graph and certified correction")
    ax = fig.add_subplot(grid[1, 0])
    draw_static(view, stage="execution", ax=ax, title="", **SMALL)
    panel(ax, "b", "one recorded run")
    ax = fig.add_subplot(grid[:, 1])
    ellipse(ax, mean_in, np.eye(2), color="#94a3b8", ls=":", label="input (coherent)")
    ellipse(ax, ideal @ mean_in, np.eye(2), color=COLORS["flow"], label="ideal F(input)")
    ellipse(
        ax,
        channel.matrix @ mean_in,
        np.eye(2) + channel.noise,
        color=COLORS["correction"],
        ls="--",
        label="ensemble output",
    )
    ellipse(
        ax,
        result.state.mean,
        result.state.covariance,
        color=COLORS["outcome"],
        label="this run (conditional)",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    panel(ax, "c", "1-sigma ellipses")
    save(
        "01_single_mode_teleportation",
        fig,
        f"""
**Single-mode CV teleportation (one cluster wire step).** (a) The resource graph of
`protocols.wire((0.0,), squeezing={r})`: input 0 is entangled with a p-squeezed mode 1 by
CZ(1) and measured in p (homodyne angle pi/2, shear k = 0); `flows.analyze` certifies the
CV flow (`{flow.status}`) and the dashed arrow is the feed-forward the pattern executes,
{view.corrections[1].formula()}. (b) One `native-gaussian` trajectory (seed 1) for a coherent
input alpha = {alpha}; the recorded outcome is m = {result.outcomes[0]:.4g}. (c) Phase-space
1-sigma ellipses (hbar = 2, vacuum = unit circle): ideal map F (Fourier transform), the
unconditional channel from `gaussian_channel` (S = F to
{np.abs(channel.matrix - ideal).max():.1e}, added noise Tr N = {np.trace(channel.noise):.4f},
i.e. finite squeezing r = {r} adds noise; it is not the ideal unitary) and this run's
conditional output state, whose mean is shifted by an outcome-dependent amount at finite
squeezing. Limitations: ideal homodyne detection; no loss.
""",
        view,
    )


# ---------------------------------------------------------------------------------- 2
@demo("02_two_mode_cv_cluster")
def two_mode(args):
    rs = np.linspace(0.0, 2.0, 9)
    measured = []
    for r in rs:
        state = pg.simulate(
            pg.Pattern(pg.CVGraph.line(2, squeezing=float(r))), backend="gaussian", seed=0
        ).state
        v = state.covariance
        n0 = np.array([0, 1, -1, 0])  # p0 - q1
        n1 = np.array([-1, 0, 0, 1])  # p1 - q0
        measured.append((float(n0 @ v @ n0), float(n1 @ v @ n1)))
    measured = np.array(measured)
    r0 = 1.0
    cluster = pg.CVGraph.line(2, squeezing=r0)
    state = pg.simulate(pg.Pattern(cluster), backend="gaussian", seed=0).state
    view = GraphView(OpenGraph.from_cvgraph(cluster, {}), title="Two-mode CV cluster")
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.0, 2.3),
        gridspec_kw={"width_ratios": [0.7, 1.1, 1.1]},
        layout="constrained",
    )
    draw_static(view, ax=axes[0], title="", **SMALL)
    panel(axes[0], "a", f"graph (r = {r0})")
    covariance_heatmap(state).draw(axes[1])
    axes[1].set_title("")
    panel(axes[1], "b", "covariance (q0, p0, q1, p1)")
    ax = axes[2]
    ax.semilogy(rs, measured[:, 0], "o", color=COLORS["flow"], label="Var(p0 - q1), simulated")
    ax.semilogy(
        rs, measured[:, 1], "x", color=COLORS["correction"], label="Var(p1 - q0), simulated"
    )
    ax.semilogy(rs, np.exp(-2 * rs), "-", color="#475569", lw=1, label="e^(-2r) (closed form)")
    ax.set_xlabel("resource squeezing r")
    ax.set_ylabel("nullifier variance")
    ax.legend(frameon=False)
    panel(ax, "c", "nullifiers")
    err = float(np.abs(measured - np.exp(-2 * rs)[:, None]).max())
    save(
        "02_two_mode_cv_cluster",
        fig,
        f"""
**Two-mode CV cluster state.** (a) `CVGraph.line(2, squeezing={r0})`: two p-squeezed modes
joined by CZ(1) (both are outputs; nothing is measured). (b) Covariance matrix of the state
prepared by `simulate(Pattern(graph), backend="gaussian")` (hbar = 2, vacuum = I). (c) The
nullifier variances Var(p0 - q1) and Var(p1 - q0) of the simulated states for r = 0...2
against the closed form e^(-2r) (largest deviation {err:.1e}); the state is an ideal cluster
only as r -> infinity. Limitations: pure Gaussian resource, no loss.
""",
        view,
    )


# ---------------------------------------------------------------------------------- 3
@demo("03_multimode_cv_cluster")
def multimode(args):
    rows, cols = 3, 4
    cluster = pg.CVGraph.rectangular(
        rows,
        cols,
        squeezing=1.0,
        inputs=tuple((i, 0) for i in range(rows)),
        outputs=tuple((i, cols - 1) for i in range(rows)),
    )
    graph = OpenGraph.from_cvgraph(
        cluster, {n: 0.0 for n in cluster.nodes if n not in cluster.outputs}
    )
    flow = analyze(graph)
    view = GraphView(graph, title="3 x 4 CV cluster").with_flow(flow)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    tuples = {**SMALL, "fontsize": 5.2, "node_size": 420}
    draw_static(view, stage="construct", ax=axes[0], title="", **tuples)
    panel(axes[0], "a", "complete graph (3 inputs, 3 outputs)")
    draw_static(
        view,
        stage="flow",
        ax=axes[1],
        title="",
        order_label="layer",
        weights=False,
        **tuples,
    )
    panel(axes[1], "b", f"CV flow ({flow.status}), layers L0..L{len(flow.layers) - 1}")
    legend_below(fig, ("flow",), y=0.12)
    save(
        "03_multimode_cv_cluster",
        fig,
        f"""
**Multimode CV cluster.** (a) `CVGraph.rectangular({rows}, {cols}, squeezing=1)` with the first
column as inputs and the last as outputs; all other nodes are measured with homodyne shear
k = 0 (p measurements). (b) `flows.analyze` on this open graph: status `{flow.status}`,
{len(flow.layers)} measurement layers {[list(map(str, layer)) for layer in flow.layers]},
{len(view.flow_arrows())} classical dependencies from the reported correction map (blue
arrows). Flow certification is an infinite-squeezing determinism statement; at finite
squeezing every measured node adds noise. Limitations: square-lattice nearest-neighbour
graph with unit weights only.
""",
        view,
    )


# ---------------------------------------------------------------------------------- 4
@demo("04_circuit_to_mbqc_compilation")
def compilation(args):
    circuit = pg.Circuit(2).rotate(0, 0.4).squeeze(1, 0.3).cz(0, 1, 0.8)
    pattern, trace = circuit.compile(squeezing=1.5, return_trace=True)
    view = GraphView.from_pattern(pattern, title="Compiled two-mode circuit")
    figure = plt.figure(figsize=(7.0, 3.3))
    grid = figure.add_gridspec(2, 1, height_ratios=[0.8, 1.3], hspace=0.35)
    ax = figure.add_subplot(grid[0])
    from photographiq.visualization import draw_circuit

    draw_circuit(circuit, ax=ax)
    ax.set_title("")
    for text in ax.texts:
        text.set_fontsize(6.5)
    panel(ax, "a", "source circuit")
    ax = figure.add_subplot(grid[1])
    draw_static(view, ax=ax, title="", weights=True, **SMALL)
    palette = ("#56b4e9", "#e69f00", "#cc79a7", "#009e73")
    from matplotlib.patches import FancyBboxPatch

    for step, color in zip(trace.steps, palette):
        xy = np.array([view.layout[n] for n in step.resource_nodes])
        (x0, y0), (x1, y1) = xy.min(axis=0) - 0.42, xy.max(axis=0) + 0.42
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                boxstyle="round,pad=0.02",
                fc=color,
                alpha=0.18,
                ec=color,
                lw=0.8,
                zorder=0,
            )
        )
        ax.text(
            x0 + 0.05,
            y1 + 0.03,
            f"gate {step.gate_index}: {step.source_gate}",
            fontsize=6,
            color="#1f2937",
            va="bottom",
        )
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.25)
    panel(ax, "b", "compiled weighted graph; shaded: resource nodes generated per gate")
    counts = {
        "nodes": len(view.graph.nodes),
        "edges": view.graph.graph.number_of_edges(),
        "measurements": sum(isinstance(c, pg.Measure) for c in pattern.commands),
        "corrections": len(view.with_corrections(pattern).corrections),
    }
    save(
        "04_circuit_to_mbqc_compilation",
        figure,
        f"""
**Circuit-to-MBQC compilation.** (a) `Circuit(2).rotate(0, 0.4).squeeze(1, 0.3).cz(0, 1,
0.8)` drawn by `draw_circuit`. (b) The weighted graph of `compile(squeezing=1.5,
return_trace=True)` (`GraphView.from_pattern`); each shaded box encloses the resource nodes
that the `CompilationTrace` attributes to one source gate
({"; ".join(f"gate {s.gate_index} {s.source_gate}: {list(s.resource_nodes)}" for s in trace.steps)}).
`pg.visualize_compilation` draws the same provenance with a spring layout. The compiled pattern has
{counts["nodes"]} modes, {counts["edges"]} CZ edges, {counts["measurements"]} homodyne
measurements and {counts["corrections"]} outcome-dependent corrections. Single-mode gates are
decomposed into four or five shear (wire) steps by `decompose_symplectic`; CZ adds four
identity steps per mode. Finite squeezing makes every step a noisy channel. The interactive
page shows the same graph with stage selection.
""",
        view,
    )


# ---------------------------------------------------------------------------------- 5
@demo("05_adaptive_feedforward")
def adaptive(args):
    k = 0.3
    pattern = pg.protocols.adaptive(squeezing=1.5)
    runs = 60 if args.quick else 300
    m0, angles = [], []
    first = None
    for seed in range(runs):
        result = pg.simulate(pattern, backend="native-gaussian", seed=seed, parameters={"k": k})
        trace = execution_trace(pattern, result, parameters={"k": k})
        measures = [s for s in trace.steps if s.kind == "measure"]
        m0.append(measures[0].outcome)
        angles.append(measures[1].values["angle"])
        first = first or (result, trace)
    m0, angles = np.array(m0), np.array(angles)
    predicted = np.arctan2(1.0, k + 0.1 * m0)
    err = float(np.abs(angles - predicted).max())
    view = GraphView.from_pattern(
        pattern, parameters={"k": k}, title="Adaptive wire"
    ).with_execution(first[1])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.1, 1]})
    draw_static(view, stage="execution", ax=axes[0], title="", **SMALL)
    panel(axes[0], "a", "one recorded run (outcomes m)")
    ax = axes[1]
    order = np.argsort(m0)
    ax.plot(m0[order], predicted[order], "-", color="#475569", lw=1, label="atan2(1, k + 0.1 m0)")
    ax.plot(m0, angles, "o", ms=2.5, color=COLORS["flow"], label=f"angle used ({runs} runs)")
    ax.set_xlabel("first outcome m0")
    ax.set_ylabel("second homodyne angle (rad)")
    ax.legend(frameon=False)
    panel(ax, "b", "adaptive measurement angle")
    save(
        "05_adaptive_feedforward",
        fig,
        f"""
**Adaptive feed-forward.** `protocols.adaptive(squeezing=1.5)` is a two-step wire whose
second homodyne angle is theta_2 = atan2(1, k + 0.1 m0), set by the first outcome m0
(k = {k}). (a) One `native-gaussian` trajectory with its recorded outcomes. No correction
arrows are drawn: the second correction q -= m1 / sin(theta_2(m0)) is not affine in the
outcomes, and `pattern_corrections` rejects it instead of linearising it; the adaptive angle
is listed as a node annotation in the interactive page. (b) The angle actually
evaluated in {runs} trajectories (from `execution_trace`, which re-evaluates the pattern's
expressions on the recorded outcomes) against the closed form (largest difference
{err:.1e} rad). Limitations: Gaussian feed-forward only; flow analysis is not applicable
because the measurement angle itself depends on an outcome.
""",
        view,
    )


# ---------------------------------------------------------------------------------- 6
@demo("06_flow_finding_and_layering")
def flow_finding(args):
    qubit = qubit_six()
    qflow = analyze(qubit)
    qview = GraphView(qubit, title="Qubit graph").with_flow(qflow)
    rows, cols = 2, 4
    cluster = pg.CVGraph.rectangular(
        rows, cols, squeezing=1.0, inputs=((0, 0), (1, 0)), outputs=((0, 3), (1, 3))
    )
    cgraph = OpenGraph.from_cvgraph(
        cluster,
        {
            n: 0.3 * (i + 1)
            for i, n in enumerate(n for n in cluster.nodes if n not in cluster.outputs)
        },
    )
    cflow = analyze(cgraph)
    cview = GraphView(cgraph, title="CV ladder").with_flow(cflow).with_layout("layered")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [0.8, 1, 1.45]})
    draw_static(
        qview, stage="construct", ax=axes[0], title="", **{**SMALL, "node_size": 170, "fontsize": 6}
    )
    panel(axes[0], "a", "qubit open graph")
    draw_static(
        qview.with_layout("layered"),
        stage="flow",
        ax=axes[1],
        title="",
        order_label="both",
        **{**SMALL, "node_size": 200, "fontsize": 6},
    )
    panel(axes[1], "b", f"{qflow.algorithm.split(' ')[0]}: layers")
    draw_static(
        cview,
        stage="flow",
        ax=axes[2],
        title="",
        order_label="layer",
        weights=False,
        **{**SMALL, "fontsize": 5.0, "node_size": 300},
    )
    panel(axes[2], "c", f"CV flow ({cflow.status}), layered")
    legend_below(fig, ("flow",), y=0.2)
    save(
        "06_flow_finding_and_layering",
        fig,
        f"""
**Flow finding and measurement layering.** (a) A six-node qubit open graph (input 0, outputs
4 and 5, XY-plane measurements at angles 0.1 n). (b) `flows.analyze` finds a
`{qflow.algorithm}` with status `{qflow.status}`; the layered layout places each measured node
in its layer column (`Lk`, measurement order `#k`): layers
{[list(layer) for layer in qflow.layers]}. Arrows are exactly the dependencies of the
reported correction map (corrections act on later layers and outputs only). (c) A 2 x 4 CV
ladder (`CVGraph.rectangular`, homodyne shears 0.3 ... 1.8) analysed with the real-linear CV
flow: status `{cflow.status}`, layers {[list(map(str, layer)) for layer in cflow.layers]}.
Qubit flow certificates are not CV certificates and vice versa; failing to certify is not a
proof of nondeterminism.
""",
        qview,
    )


# ---------------------------------------------------------------------------------- 7
@demo("07_correction_map_generation")
def correction_maps(args):
    graph, shears, open_graph, flow = cv_wire()
    pattern = flow_pattern(graph, flow.measurement_order, shears=shears)
    view = GraphView(open_graph, title="Four-node CV wire").with_flow(flow)
    cv = view.with_corrections()
    executed = view.with_corrections(pattern)
    qubit = qubit_six()
    qview = GraphView(qubit, title="Qubit graph").with_flow(analyze(qubit)).with_corrections()
    fig = plt.figure(figsize=(7.0, 4.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.3)
    ax = fig.add_subplot(grid[0, 0])
    draw_static(view, stage="flow", ax=ax, title="", **SMALL)
    panel(ax, "a", "CV flow dependencies")
    ax = fig.add_subplot(grid[0, 1])
    draw_static(qview, stage="corrections", ax=ax, title="", **SMALL)
    panel(ax, "c", "qubit X/Z corrections")
    ax = fig.add_subplot(grid[1, :])
    draw_static(cv, stage="corrections", ax=ax, title="", **{**SMALL, "fontsize": 8})
    panel(ax, "b", "CV corrections (coefficients of the raw record m)")
    legend_below(fig, ("flow", "correction"), y=0.08)
    formulas = "; ".join(f"node {t}: {c.formula()}" for t, c in cv.corrections.items())
    qformulas = "; ".join(f"node {t}: {c.formula()}" for t, c in qview.corrections.items())
    same = cv.correction_arrows() == executed.correction_arrows()
    save(
        "07_correction_map_generation",
        fig,
        f"""
**Correction-map generation.** (a) `flows.analyze` on a four-node CV wire
(`CVGraph.line(4, squeezing=1.5)`, homodyne shears k = 0.3, -0.5, 0): certified CV flow, one
blue arrow per (outcome, corrected node) pair of its correction map. (b) The same map as
displacement corrections in terms of the raw homodyne records m[k] (flow coefficients are
per rescaled record m / sin(theta); the overlay multiplies by sqrt(1 + k^2)): {formulas}.
These equal the `Displace` commands of `cvflow.flow_pattern` for this order: {same}.
(c) Binary corrections of the qubit graph of demo 06 from its causal flow (X on the flow
successor, Z on its other neighbours; exponents are XORs of outcome bits): {qformulas}.
Formulas are shown only where they are read from a correction map or from executed commands.
""",
        cv,
    )


# ---------------------------------------------------------------------------------- 8
@demo("08_non_gaussian_injection")
def injection(args):
    gamma, r, cutoff, seed = 0.25, 0.5, 56, 3
    pattern = pg.non_gaussian.cubic_injection(gamma, squeezing=r)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pg.simulate(
            pattern,
            backend="native-fock",
            cutoff=cutoff,
            inputs={"in": pg.GaussianInput.coherent(0.0)},
            seed=seed,
        )
    norm = min(result.state.retained_norms)
    view = GraphView.from_pattern(pattern, title="Cubic-phase injection")
    status = analyze(view.graph)
    plot = wigner_plot(result.state, points=101, extent=4.5)
    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1, 1.1]}, layout="constrained"
    )
    draw_static(view, ax=axes[0], title="", **{**SMALL, "node_size": 520, "fontsize": 6})
    import textwrap

    notes = "\n".join(
        textwrap.fill(f"{n}: {t}", 62, subsequent_indent="   ") for n, t in view.annotations.items()
    )
    axes[0].text(0.0, 0.05, notes, transform=axes[0].transAxes, fontsize=5.5, va="top")
    panel(axes[0], "a", "injection graph (non-Gaussian model)")
    plot.draw(axes[1])
    title = plot.layout["title"]["text"]
    axes[1].set_title("")
    panel(axes[1], "b", f"output Wigner function, m = {result.outcomes['m']:.3g}")
    save(
        "08_non_gaussian_injection",
        fig,
        f"""
**Non-Gaussian (cubic-phase) injection.** (a) `non_gaussian.cubic_injection({gamma},
squeezing={r})`: the input `in` is coupled to a finite-energy cubic-phase resource by CZ and
the resource is measured in q (key `m`); the feed-forward is nonlinear (QuadraticPhase(-2
gamma m) and Displace(p = -gamma m^2)), so it is listed as an annotation, not drawn as a
linear correction, and `flows.analyze` returns `{status.status}` for this model. (b) Signed
Wigner function of the output of one `native-fock` trajectory (vacuum input, seed {seed},
total-photon cutoff {cutoff}, smallest retained norm over all operations {norm:.6f},
{len(caught)} truncation warnings); {title.lower()}. The finite resource implements CP(gamma)
times a Gaussian filter exp(-(q + m)^2 / (4 e^(2r))), a conditional operation, not an exact
unitary cubic gate. Limitations: one trajectory; Fock truncation as stated.
""",
        view,
    )


# ---------------------------------------------------------------------------------- 9
@demo("09_gkp_encoded_mbqc")
def gkp(args):
    width, cutoff, delta = 0.45, 60, 0.25
    runs = 2 if args.quick else 4
    code = pg.GKPResource(0, peak_width=width, envelope=width)
    zero, _ = code.project(cutoff)
    plus = pg.gkp.superposition(1, 1, cutoff=cutoff, peak_width=width, envelope=width)
    pattern = pg.gkp.correction_pattern(quadrature="q", resource=plus)
    shift = pg.Pattern(inputs=(0,)).append(pg.Displace(0, q=delta)).append(pg.Output((0,)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shifted = pg.simulate(shift, backend="native-fock", cutoff=cutoff, inputs={0: zero}).state
        vector = np.array(
            [dict(zip(shifted.basis, shifted.state_vector)).get((n,), 0j) for n in range(cutoff)]
        )
        shifted_input = pg.FockInput(tuple(vector / np.linalg.norm(vector)))
        residuals = []
        for seed in range(runs):
            res = pg.simulate(
                pattern,
                backend="native-fock",
                cutoff=cutoff,
                inputs={"in": shifted_input},
                seed=100 + seed,
            )
            residuals.append(pg.gkp.decode_shift(res.records["syndrome"])[0])
    view = GraphView.from_pattern(pattern, title="GKP syndrome extraction")
    status = analyze(view.graph)
    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [0.8, 1.3, 1]}, layout="constrained"
    )
    draw_static(view, ax=axes[0], title="", **SMALL)
    panel(axes[0], "a", "SUM syndrome gadget")
    plot = wigner_plot(shifted, points=121, extent=6.0)
    plot.draw(axes[1])
    axes[1].set_title("")
    panel(axes[1], "b", f"|0_L> shifted by q = {delta}")
    ax = axes[2]
    spacing = pg.gkp.SPACING
    ax.axhline(delta, color="#475569", lw=1, label=f"imposed shift {delta}")
    ax.plot(
        range(1, runs + 1), residuals, "o", color=COLORS["correction"], label="decoded residual"
    )
    ax.set_ylim(-spacing / 2, spacing / 2)
    ax.set_xlabel("trajectory")
    ax.set_ylabel("decoded residual (q units)")
    ax.legend(frameon=False, loc="lower right")
    panel(ax, "c", "nearest-cell decoder")
    save(
        "09_gkp_encoded_mbqc",
        fig,
        f"""
**GKP-encoded MBQC step (finite energy).** (a) `gkp.correction_pattern(quadrature="q")`: the
data mode `in` is coupled by CZ (between Fourier rotations, i.e. a SUM gate) to a
finite-energy logical-plus GKP ancilla that is measured in q; the modular feed-forward
q -= residual(m) is an opaque nearest-cell decoder (`decode_shift`), so it is an annotation,
not a drawn linear correction. (b) Wigner function of the finite-energy |0_L> (peak width =
envelope = {width}, total-photon cutoff {cutoff}) after a known shift q = {delta}. (c) Decoded
residuals of {runs} `native-fock` trajectories: {", ".join(f"{x:.3f}" for x in residuals)}
(lattice spacing {spacing:.4f}); single shots scatter because the finite ancilla has its own
peak width. This is a finite-ancilla demonstration, not a fault-tolerance claim; GKP logical
flows are not analysed: `flows.analyze` on this {view.graph.model.value} graph returns `{status.status}`.
""",
        view,
    )


# --------------------------------------------------------------------------------- 10
@demo("10_native_backend_execution")
def native(args):
    circuit = pg.Circuit(2).rotate(0, 0.4).squeeze(1, 0.3).cz(0, 1, 0.8)
    pattern = circuit.compile(squeezing=1.5)
    inputs = {0: pg.GaussianInput.coherent(0.5), 1: pg.GaussianInput.coherent(-0.2 + 0.3j)}
    runs = 5 if args.quick else 20
    diffs, times = [], {"native-gaussian": [], "gaussian": []}
    first = None
    for seed in range(runs):
        out = {}
        for backend in times:
            t0 = time.perf_counter()
            out[backend] = pg.simulate(pattern, backend=backend, seed=seed, inputs=inputs)
            times[backend].append(time.perf_counter() - t0)
        a, b = out["native-gaussian"], out["gaussian"]
        diffs.append([abs(a.outcomes[k] - b.outcomes[k]) for k in a.outcomes])
        first = first or a
    diffs = np.array(diffs)
    view = (
        GraphView.from_pattern(pattern, title="Native execution")
        .with_corrections(pattern)
        .with_execution(execution_trace(pattern, first))
    )
    fig = plt.figure(figsize=(7.0, 3.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    ax = fig.add_subplot(grid[0, :])
    draw_static(
        view, stage="execution", ax=ax, title="", **{**SMALL, "node_size": 200, "fontsize": 6}
    )
    panel(ax, "a", "one native-gaussian trajectory (seed 0)")
    ax = fig.add_subplot(grid[1, 0])
    ax.semilogy(
        range(diffs.shape[1]), np.maximum(diffs.max(axis=0), 1e-18), "o", ms=3, color=COLORS["flow"]
    )
    ax.set_ylim(1e-18, 1e-10)
    ax.set_xlabel("measurement index")
    ax.set_ylabel("max |m_native - m_numpy|")
    panel(ax, "b", f"same seeds, {runs} runs")
    ax = fig.add_subplot(grid[1, 1])
    medians = {k: 1e3 * float(np.median(v)) for k, v in times.items()}
    ax.bar(list(medians), list(medians.values()), color=[COLORS["flow"], "#94a3b8"])
    ax.set_ylabel("median ms / run")
    panel(ax, "c", "provisional timing (shared machine)")
    fig.tight_layout()
    save(
        "10_native_backend_execution",
        fig,
        f"""
**Native backend execution.** The compiled two-mode pattern of demo 04 executed by the C++
`native-gaussian` backend and by the NumPy `gaussian` backend with identical seeds and
coherent inputs. (a) The execution stage of one native trajectory (recorded outcomes `m`).
(b) Largest per-measurement outcome difference over {runs} seeded runs: {diffs.max():.1e}
(identical trajectories up to float rounding; points at 1e-18 are exact zeros). (c) Median
wall time per `simulate` call: native {medians["native-gaussian"]:.1f} ms, NumPy
{medians["gaussian"]:.1f} ms. Timings are provisional (machine shared with other jobs,
{runs} runs, includes Python dispatch). Native components available:
{", ".join(pg.native.components())}.
""",
        view,
    )


# --------------------------------------------------------------------------------- 11
@demo("11_boson_sampling")
def boson_sampling(args):
    from photographiq import sampling

    modes, inputs = 4, (1, 1, 1, 0)
    shots = 4000 if args.quick else 40000
    unitary = sampling.haar_unitary(modes, rng=7)
    dist = sampling.output_distribution(unitary, inputs)
    samples = sampling.sample(unitary, inputs, shots, rng=1, method="enumerate")
    counts: dict = {}
    for row in samples:
        counts[tuple(int(x) for x in row)] = counts.get(tuple(int(x) for x in row), 0) + 1
    top = sorted(dist, key=dist.get, reverse=True)[:14]
    exact = np.array([dist[t] for t in top])
    emp = np.array([counts.get(t, 0) / shots for t in top])
    tvd = 0.5 * sum(abs(dist[t] - counts.get(t, 0) / shots) for t in dist)
    collisions = sampling.collision_probabilities(dist)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [0.7, 1.6]})
    image = axes[0].imshow(np.abs(unitary) ** 2, cmap="Blues", vmin=0)
    axes[0].set_xlabel("input mode")
    axes[0].set_ylabel("output mode")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    panel(axes[0], "a", "|U_ij|^2 (Haar, seed 7)")
    x = np.arange(len(top))
    axes[1].bar(x - 0.2, exact, 0.4, color=COLORS["flow"], label="exact (permanents)")
    axes[1].bar(x + 0.2, emp, 0.4, color=COLORS["correction"], label=f"{shots} samples")
    axes[1].errorbar(
        x + 0.2,
        emp,
        yerr=np.sqrt(exact * (1 - exact) / shots),
        fmt="none",
        ecolor="#1f2937",
        lw=0.6,
    )
    axes[1].set_xticks(x, ["".join(map(str, t)) for t in top], rotation=60, fontsize=6)
    axes[1].set_ylabel("probability")
    axes[1].legend(frameon=False)
    panel(axes[1], "b", "14 most likely output patterns")
    fig.tight_layout()
    save(
        "11_boson_sampling",
        fig,
        f"""
**Boson sampling.** (a) Transition probabilities |U_ij|^2 of a 4-mode Haar-random
interferometer (`sampling.haar_unitary(4, rng=7)`). (b) For the input |1,1,1,0>, exact
output probabilities from permanents (`sampling.output_distribution`, all
{len(dist)} patterns, collision-free mass {collisions["collision_free"]:.4f}) against
{shots} exact samples from `sampling.sample(..., method="enumerate")` (rng 1; this draws from the same enumerated distribution, so it checks the sampler, not the permanents); error bars are binomial 1-sigma.
Total variation distance over all patterns: {tvd:.4f}. Lossless, perfectly indistinguishable
photons and ideal photon-number-resolving detectors.
""",
    )


# --------------------------------------------------------------------------------- 12
@demo("12_gaussian_boson_sampling")
def gbs(args):
    from photographiq import sampling

    modes, r, cutoff = 4, 0.5, 7
    shots = 4000 if args.quick else 40000
    unitary = sampling.haar_unitary(modes, rng=11)
    state = sampling.squeezed_interferometer_state([r] * modes, unitary)
    dist = sampling.pnr_distribution(state, cutoff)
    draws = sampling.sample_pnr(state, shots, cutoff, rng=2)
    totals = np.arange(cutoff)
    exact_total = np.array(
        [sum(p for t, p in dist.probabilities.items() if sum(t) == n) for n in totals]
    )
    kept = draws.samples[draws.samples[:, 0] >= 0]
    emp_total = np.array([(kept.sum(axis=1) == n).sum() / shots for n in totals])
    top = sorted(dist.probabilities, key=dist.probabilities.get, reverse=True)[1:13]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1, 1.6]})
    axes[0].bar(totals - 0.2, exact_total, 0.4, color=COLORS["flow"], label="exact (hafnians)")
    axes[0].bar(totals + 0.2, emp_total, 0.4, color=COLORS["correction"], label=f"{shots} samples")
    axes[0].set_xlabel("total photon number")
    axes[0].set_ylabel("probability")
    axes[0].legend(frameon=False)
    panel(axes[0], "a", "photon-number distribution")
    x = np.arange(len(top))
    exact = np.array([dist.probabilities[t] for t in top])
    counts = {}
    for row in kept:
        counts[tuple(row)] = counts.get(tuple(row), 0) + 1
    axes[1].bar(x - 0.2, exact, 0.4, color=COLORS["flow"])
    axes[1].bar(x + 0.2, [counts.get(t, 0) / shots for t in top], 0.4, color=COLORS["correction"])
    axes[1].set_xticks(x, ["".join(map(str, t)) for t in top], rotation=60, fontsize=6)
    panel(axes[1], "b", "12 most likely non-vacuum patterns")
    fig.tight_layout()
    save(
        "12_gaussian_boson_sampling",
        fig,
        f"""
**Gaussian boson sampling.** Four single-mode squeezed vacua (r = {r}) enter a Haar-random
interferometer (`sampling.haar_unitary(4, rng=11)`, `squeezed_interferometer_state`). (a)
Total photon-number distribution from exact PNR probabilities (`pnr_distribution`,
hafnians, all patterns with fewer than {cutoff} photons; truncated mass
{dist.truncated_mass:.2e}, shown explicitly, never renormalised) against {shots} exact samples
from `sample_pnr` (rng 2; {draws.overflow} shots fell in the truncated mass). Odd totals have
zero probability for pure squeezed vacuum. (b) The most likely non-vacuum PNR patterns.
Lossless interferometer, ideal PNR detectors.
""",
    )


# --------------------------------------------------------------------------------- 13
@demo("13_photographiqml")
def ml(args):
    from scipy.linalg import expm

    import photographiqml as pqml
    from photographiqml.logical import X
    from photographiqml.models import haar_states, infidelity

    model = pqml.MuTA(1)
    states = haar_states(1, 7, 42)
    targets = states @ expm(0.4j * X).T

    def loss(p):
        return infidelity(model.run_batch(states, p), targets)

    epochs = 40 if args.quick else 100
    parameters, history = pqml.Trainer(epochs=epochs).fit(loss, np.zeros(model.n_parameters))
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    optimization_curves(history, log=True).draw(ax)
    ax.set_title("")
    ax.set_xlabel("epoch")
    ax.set_title("MuTA(1) learns exp(0.4 i X)", loc="left", fontsize=8)
    save(
        "13_photographiqml",
        fig,
        f"""
**PhotoGraphiQML training.** A one-wire logical MuTA model (`photographiqml.MuTA(1)`,
{model.n_parameters} parameters) trained with `Trainer(epochs={epochs})` to reproduce the
rotation exp(0.4 i X) on 7 Haar-random input states (`haar_states(1, 7, 42)`); the loss is
the mean infidelity, final value {history.losses[-1]:.2e}. The curve is drawn by
`interactive.optimization_curves`. This is a logical (qubit-level) model; lowering continuous
angles to a physical GKP execution is a separate, explicitly bounded step
(`MuTA.physical_capabilities`) and is not shown here.
""",
    )


# --------------------------------------------------------------------------------- 14
@demo("14_end_to_end_execution")
def end_to_end(args):
    r, theta, s = 1.5, 0.5, 0.3
    circuit = pg.Circuit(1).rotate(0, theta).squeeze(0, s)
    pattern = circuit.compile(squeezing=r)
    channel = pg.gaussian_channel(pattern)
    c, si = math.cos(theta), math.sin(theta)
    target = np.diag([math.exp(-s), math.exp(s)]) @ np.array([[c, -si], [si, c]])
    alpha = 0.6 - 0.4j
    mean_in = np.array([2 * alpha.real, 2 * alpha.imag])
    runs = 40 if args.quick else 200
    means, first = [], None
    for seed in range(runs):
        result = pg.simulate(
            pattern,
            backend="native-gaussian",
            seed=seed,
            inputs={0: pg.GaussianInput.coherent(alpha)},
        )
        means.append(result.state.mean)
        first = first or result
    means = np.array(means)
    base = GraphView.from_pattern(pattern, title="R(0.5) then S(0.3), compiled")
    flow = analyze(base.graph)
    view = (
        base.with_flow(flow)
        .with_corrections(pattern)
        .with_execution(execution_trace(pattern, first))
    )
    fig = plt.figure(figsize=(7.0, 3.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.4], width_ratios=[1, 0.8])
    ax = fig.add_subplot(grid[0, :])
    draw_static(view, stage="execution", ax=ax, title="", **SMALL)
    panel(ax, "a", f"compiled graph, flow {flow.status}, one recorded run")
    ax = fig.add_subplot(grid[1, 0])
    ax.plot(
        means[:, 0],
        means[:, 1],
        "o",
        ms=2,
        alpha=0.5,
        color=COLORS["outcome"],
        label=f"conditional means ({runs} runs)",
    )
    ellipse(
        ax,
        target @ mean_in,
        target @ target.T,
        color=COLORS["flow"],
        label="ideal output (1 sigma)",
    )
    ellipse(
        ax,
        channel.matrix @ mean_in,
        channel.matrix @ channel.matrix.T + channel.noise,
        color=COLORS["correction"],
        ls="--",
        label="ensemble S V S^T + N",
    )
    ellipse(
        ax,
        first.state.mean,
        first.state.covariance,
        color="#1f2937",
        lw=0.8,
        label="one conditional state",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left", bbox_to_anchor=(0.0, 1.08), ncol=2)
    panel(ax, "b", "")
    ax = fig.add_subplot(grid[1, 1])
    ideal_cov = target @ target.T
    predicted = channel.matrix @ channel.matrix.T + channel.noise
    empirical = first.state.covariance + np.cov(means.T, ddof=1)
    labels = ["Var q", "Cov qp", "Var p"]
    pick = [(0, 0), (0, 1), (1, 1)]
    x = np.arange(3)
    for shift, (matrix, color, name) in zip(
        (-0.27, 0.0, 0.27),
        (
            (ideal_cov, COLORS["flow"], "ideal S S^T"),
            (predicted, COLORS["correction"], "S S^T + N"),
            (empirical, COLORS["outcome"], f"sampled ({runs} runs)"),
        ),
    ):
        ax.bar(x + shift, [matrix[i, j] for i, j in pick], 0.27, color=color, label=name)
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, fontsize=5.5)
    panel(ax, "c", "output covariance")
    fig.tight_layout()
    err = float(np.abs(channel.matrix - target).max())
    save(
        "14_end_to_end_execution",
        fig,
        f"""
**End-to-end execution (compact version of manuscript experiment R60).** `Circuit(1).rotate(0,
{theta}).squeeze(0, {s}).compile(squeezing={r})` gives a {len(base.graph.nodes)}-mode wire;
`flows.analyze` returns `{flow.status}`. (a) Execution stage of one `native-gaussian`
trajectory (outcomes `m`, executed corrections dashed). (b) Output phase space for a coherent
input alpha = {alpha}: ideal target S(0.3) R(0.5) (the compiled channel matrix equals it to
{err:.1e}), the unconditional finite-squeezing channel (added noise Tr N =
{np.trace(channel.noise):.4f}) and the conditional output means of {runs} trajectories, which
scatter because finite squeezing leaves an outcome-dependent shift. (c) Output covariance
entries: ideal S S^T, the channel prediction S S^T + N and the sampled ensemble (one conditional
covariance plus the sample covariance of the conditional means; statistical, not exact). The full pipeline with
independent validation is `paper_experiments/21_showcase/60_end_to_end_showcase.py`.
""",
        view,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="comma-separated demo names (or name prefixes such as 01)")
    parser.add_argument("--quick", action="store_true", help="fewer runs/shots")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(DEMOS))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = [w.strip() for w in args.only.split(",")] if args.only else list(DEMOS)
    chosen = [n for n in DEMOS if any(n == w or n.startswith(w) for w in wanted)]
    if not chosen:
        print(f"no demo matches {wanted}", file=sys.stderr)
        return 1
    with plt.rc_context(STYLE):
        for name in chosen:
            DEMOS[name](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
