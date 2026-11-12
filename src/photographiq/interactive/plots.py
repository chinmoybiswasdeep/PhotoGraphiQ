"""Scientific plots: data computed by PhotoGraphiQ, rendered by Plotly.js or Matplotlib.

Each function returns a :class:`Plot`: a restricted Plotly figure description (trace
types ``scatter``, ``bar``, ``heatmap``) that renders either as self-contained HTML
with Plotly.js from a pinned CDN (the Python ``plotly`` package is not needed) or as
a static Matplotlib figure (SVG/PDF/PNG). No physics is evaluated in JavaScript.

Conventions: hbar=2, q=a+a^dag, statistical covariance with vacuum V=I, homodyne at
angle theta measures q cos(theta) + p sin(theta).
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..fock_analysis import WignerGrid, fidelity, wigner
from ..fock_measurements import wavefunctions
from ..states import GaussianState
from . import _html

_DIVERGING = [[0.0, "#2166ac"], [0.5, "#f7f7f7"], [1.0, "#b2182b"]]


@dataclass(frozen=True)
class Plot:
    """Plotly-compatible figure description with HTML and Matplotlib renderers."""

    traces: list
    layout: dict = field(default_factory=dict)

    def to_html(self, *, plotly_js: str | Path | None = None, height: int = 460) -> str:
        """Self-contained page; ``plotly_js`` inlines a local Plotly.js copy (offline)."""
        title = html.escape(_text(self.layout.get("title")) or "PhotoGraphiQ plot")
        figure = _html.script_json({"data": self.traces, "layout": self.layout})
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{title}</title><style>body{{margin:0;background:#fff}}"
            f"#plot{{width:100%;height:{int(height)}px}}</style></head><body>"
            f'<div id="plot"></div>{_html.script_tag(_html.PLOTLY_URL, plotly_js)}'
            f"<script>const FIGURE = {figure};\n"
            'Plotly.newPlot("plot", FIGURE.data, FIGURE.layout, {responsive: true});</script>'
            "</body></html>"
        )

    def save_html(self, path: str | Path, **options) -> Path:
        return _html.write(self.to_html(**options), path)

    def _repr_html_(self) -> str:
        return _html.iframe(self.to_html(), 480)

    def show(self, **options) -> Path:
        return _html.open_in_browser(self.to_html(**options), "photographiq-plot-")

    def draw(self, ax=None):
        """Render with Matplotlib on ``ax`` (new figure if None); return the Axes."""
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.subplots(figsize=(6.4, 4.2))[1]
        bars = [t for t in self.traces if t["type"] == "bar"]
        for trace in self.traces:
            kind = trace["type"]
            if kind == "scatter":
                mode = trace.get("mode", "lines")
                dash = {"dash": "--", "dot": ":"}.get(trace.get("line", {}).get("dash"), "-")
                ax.plot(
                    trace["x"],
                    trace["y"],
                    ls=dash if "lines" in mode else "",
                    marker="o" if "markers" in mode else None,
                    ms=4,
                    label=trace.get("name"),
                    color=trace.get("line", {}).get("color"),
                )
            elif kind == "bar":
                width = 0.8 / len(bars)
                offset = (bars.index(trace) - (len(bars) - 1) / 2) * width
                x = np.arange(len(trace["x"]))
                ax.bar(x + offset, trace["y"], width, label=trace.get("name"))
                ax.set_xticks(x, [str(v) for v in trace["x"]])
            elif kind == "heatmap":
                z = np.asarray(trace["z"], dtype=float)
                extra = {}
                if trace.get("zmid") == 0:
                    bound = float(np.abs(z).max()) or 1.0
                    extra = {"cmap": "RdBu_r", "vmin": -bound, "vmax": bound}
                xs, ys = trace.get("x"), trace.get("y")
                numeric = xs is not None and np.issubdtype(np.asarray(xs).dtype, np.number)
                if numeric:
                    mesh = ax.pcolormesh(xs, ys, z, shading="auto", **extra)
                else:
                    mesh = ax.imshow(z, **extra)
                    if xs is not None:
                        ax.set_xticks(range(len(xs)), xs, rotation=90)
                        ax.set_yticks(range(len(ys)), ys)
                ax.figure.colorbar(mesh, ax=ax, label=trace.get("colorbar", {}).get("title"))
            else:
                raise NotImplementedError(f"Trace type {kind!r}")
        layout = self.layout
        ax.set_title(_text(layout.get("title")) or "", fontsize=11)
        for axis, setter in (("xaxis", ax.set_xlabel), ("yaxis", ax.set_ylabel)):
            setter(_text(layout.get(axis, {}).get("title")) or "")
            if layout.get(axis, {}).get("type") == "log":
                (ax.set_xscale if axis == "xaxis" else ax.set_yscale)("log")
        if layout.get("yaxis", {}).get("scaleanchor") == "x":
            ax.set_aspect("equal")
        if any(t.get("name") for t in self.traces if t["type"] != "heatmap"):
            ax.legend(fontsize=8, frameon=False)
        return ax

    def save(self, path: str | Path, *, dpi: int = 300) -> Path:
        """Static SVG/PDF/PNG export via a standalone Matplotlib ``Figure`` (no GUI)."""
        import matplotlib
        from matplotlib.figure import Figure

        with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            ax = self.draw(Figure(figsize=(6.4, 4.2)).subplots())
            ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
        return Path(path)


def _text(value: Any) -> str | None:
    return value.get("text") if isinstance(value, dict) else value


def _layout(title: str, x: str, y: str, **extra) -> dict:
    layout: dict[str, Any] = {
        "title": {"text": title},
        "xaxis": {"title": {"text": x}},
        "yaxis": {"title": {"text": y}},
        "margin": {"t": 50},
    }
    for key, options in extra.items():
        if isinstance(options, dict):
            layout.setdefault(key, {}).update(options)
        else:
            layout[key] = options
    return layout


# ---- state diagnostics -------------------------------------------------------------
def photon_number_marginal(state, node) -> dict[int, float]:
    """P(n) of one mode of a Fock-backend state (marginal of ``state.probabilities``)."""
    if not hasattr(state, "probabilities"):
        raise NotImplementedError(
            "Photon-number distributions need a Fock state (native-fock, native-mixed-fock, "
            "piquasso-fock or piquasso-mixed-fock backend)"
        )
    i = state.nodes.index(node)
    marginal: dict[int, float] = {}
    for basis, p in state.probabilities.items():
        n = int(basis[i])
        marginal[n] = marginal.get(n, 0.0) + float(p)
    return dict(sorted(marginal.items()))


def photon_number_distribution(state, node) -> Plot:
    """Bar chart of P(n); the title reports the retained (truncated) probability mass."""
    marginal = photon_number_marginal(state, node)
    total = sum(marginal.values())
    return Plot(
        [
            {
                "type": "bar",
                "x": list(marginal),
                "y": list(marginal.values()),
                "name": f"mode {node}",
            }
        ],
        _layout(
            f"Photon-number distribution, mode {node} (retained mass {total:.6f})", "n", "P(n)"
        ),
    )


def quadrature_density(state, node, angle: float, x: np.ndarray) -> np.ndarray:
    """Ideal homodyne density of q cos(angle) + p sin(angle) on ``node`` at points ``x``.

    Gaussian states: exact normal density from :meth:`GaussianState.quadrature`. Fock
    states: p(x) = v(x)^T rho v(x)^*, v_n = exp(-i n angle) psi_n(x), with the hbar=2
    Hermite functions of :func:`photographiq.fock_measurements.wavefunctions`; exact
    for the truncated state. Detector inefficiency is not included.
    """
    x = np.asarray(x, dtype=float)
    if isinstance(state, GaussianState):
        mean, variance = state.quadrature(node, angle)
        return np.exp(-((x - mean) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)
    reduced = state.reduced((node,))
    rho, levels = reduced.density_matrix, np.array([b[0] for b in reduced.basis])
    size = int(levels.max()) + 1
    phase = np.exp(-1j * angle * levels)
    density = np.empty_like(x)
    for i, point in enumerate(x):
        v = phase * wavefunctions(float(point), size)[levels]
        density[i] = float(np.real(v @ rho @ v.conj()))
    return density


def quadrature_distribution(
    state=None,
    node=None,
    angle: float = 0.0,
    *,
    samples: Sequence[float] | None = None,
    x: Sequence[float] | None = None,
    bins: int = 40,
) -> Plot:
    """Ideal homodyne density of a state and/or a normalized histogram of outcomes.

    ``samples`` are recorded homodyne outcomes, e.g. ``ShotResult.values(key)``; their
    histogram includes any detector noise, unlike the ideal state density.
    """
    if state is None and samples is None:
        raise ValueError("Give a state, samples, or both")
    traces = []
    if samples is not None:
        sample_array = np.asarray(samples, dtype=float)
        counts, edges = np.histogram(sample_array, bins=bins, density=True)
        traces.append(
            {
                "type": "bar",
                "x": ((edges[:-1] + edges[1:]) / 2).tolist(),
                "y": counts.tolist(),
                "name": f"{len(sample_array)} recorded outcomes",
                "width": float(edges[1] - edges[0]),
            }
        )
    if state is not None:
        if x is None:
            mean, variance = state.quadrature(node, angle)
            half = 5 * np.sqrt(variance)
            x_points = np.linspace(mean - half, mean + half, 301)
        else:
            x_points = np.asarray(x, dtype=float)
        density = quadrature_density(state, node, angle, x_points)
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": x_points.tolist(),
                "y": density.tolist(),
                "name": "ideal density",
            }
        )
    plot = Plot(traces, _layout(f"Homodyne angle {angle:.4g} rad", "x", "density"))
    if samples is not None and state is not None:
        return _histogram_as_line(plot)
    return plot


def _histogram_as_line(plot: Plot) -> Plot:
    """Draw the histogram as a step line so the Matplotlib renderer keeps numeric x."""
    hist, curve = plot.traces
    x, y, w = np.asarray(hist["x"]), np.asarray(hist["y"]), hist["width"]
    step_x = np.repeat(np.concatenate([x - w / 2, [x[-1] + w / 2]]), 2)[1:-1]
    step = {
        "type": "scatter",
        "mode": "lines",
        "x": step_x.tolist(),
        "y": np.repeat(y, 2).tolist(),
        "name": hist["name"],
        "line": {"dash": "dot"},
    }
    return Plot([step, curve], plot.layout)


def gaussian_wigner(state: GaussianState, q, p, node=None) -> WignerGrid:
    """Exact single-mode Gaussian Wigner function.

    W(x) = exp(-(x-mu)^T V^-1 (x-mu) / 2) / (2 pi sqrt(det V)), hbar=2, vacuum V=I, so
    W integrates to one over dq dp (same normalization as :func:`fock_analysis.wigner`).
    """
    if node is not None:
        state = state.reduced((node,))
    if len(state.nodes) != 1:
        raise ValueError("Choose one mode for the Wigner marginal")
    q, p = (np.asarray(v, dtype=float) for v in (q, p))
    inverse = np.linalg.inv(state.covariance)
    dq, dp = q[None, :] - state.mean[0], p[:, None] - state.mean[1]
    exponent = inverse[0, 0] * dq**2 + 2 * inverse[0, 1] * dq * dp + inverse[1, 1] * dp**2
    norm = 2 * np.pi * np.sqrt(np.linalg.det(state.covariance))
    return WignerGrid(q, p, np.exp(-exponent / 2) / norm)


def wigner_plot(
    state, node=None, q=None, p=None, *, points: int = 121, extent: float = 5.0
) -> Plot:
    """Signed Wigner heatmap (Gaussian closed form or ``fock_analysis.wigner``)."""
    q = np.linspace(-extent, extent, points) if q is None else np.asarray(q, dtype=float)
    p = np.linspace(-extent, extent, points) if p is None else np.asarray(p, dtype=float)
    grid = (
        gaussian_wigner(state, q, p, node)
        if isinstance(state, GaussianState)
        else wigner(state, q, p, node)
    )
    label = "" if node is None else f", mode {node}"
    title = (
        f"Wigner function{label} (grid mass {grid.captured_mass:.4f}, "
        f"negative volume {grid.negative_volume:.3g})"
    )
    return Plot(
        [
            {
                "type": "heatmap",
                "x": q.tolist(),
                "y": p.tolist(),
                "z": grid.values.tolist(),
                "zmid": 0,
                "colorscale": _DIVERGING,
                "colorbar": {"title": "W(q,p)"},
            }
        ],
        _layout(title, "q", "p", yaxis={"scaleanchor": "x"}),
    )


def _labels(state: GaussianState) -> list[str]:
    return [f"{axis}[{n}]" for n in state.nodes for axis in ("q", "p")]


def covariance_heatmap(state: GaussianState) -> Plot:
    """Statistical covariance matrix (vacuum = I), interleaved (q0, p0, q1, p1, ...)."""
    labels = _labels(state)
    bound = float(np.abs(state.covariance).max())
    return Plot(
        [
            {
                "type": "heatmap",
                "x": labels,
                "y": labels,
                "z": state.covariance.tolist(),
                "zmid": 0,
                "zmin": -bound,
                "zmax": bound,
                "colorscale": _DIVERGING,
                "colorbar": {"title": "V"},
            }
        ],
        _layout("Covariance matrix (vacuum V = I)", "", "", yaxis={"autorange": "reversed"}),
    )


def covariance_ellipse(state: GaussianState, node, points: int = 181) -> np.ndarray:
    """Points x with (x-mu)^T V^-1 (x-mu) = 1 for the reduced 2x2 covariance of ``node``."""
    reduced = state.reduced((node,))
    values, vectors = np.linalg.eigh(reduced.covariance)
    t = np.linspace(0, 2 * np.pi, points)
    circle = np.stack([np.cos(t), np.sin(t)])
    return (reduced.mean[:, None] + vectors @ (np.sqrt(values)[:, None] * circle)).T


def squeezing_ellipses(state: GaussianState, nodes: Sequence | None = None) -> Plot:
    """One-sigma covariance ellipses in phase space with dashed vacuum circles."""
    traces = []
    for node in state.nodes if nodes is None else nodes:
        ellipse = covariance_ellipse(state, node)
        mean = state.reduced((node,)).mean
        t = np.linspace(0, 2 * np.pi, 181)
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": ellipse[:, 0].tolist(),
                "y": ellipse[:, 1].tolist(),
                "name": f"mode {node}",
            }
        )
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": (mean[0] + np.cos(t)).tolist(),
                "y": (mean[1] + np.sin(t)).tolist(),
                "name": f"vacuum @ mode {node}",
                "line": {"dash": "dash", "color": "#94a3b8"},
            }
        )
    return Plot(
        traces, _layout("Covariance ellipses (1 sigma)", "q", "p", yaxis={"scaleanchor": "x"})
    )


def entanglement_diagnostics(state: GaussianState) -> Plot:
    """Per mode: log2 negativity of (mode | rest) and reduced entropy (nats)."""
    nodes = list(state.nodes)
    negativity = [state.log_negativity((n,)) for n in nodes]
    entropy = [state.reduced((n,)).entropy() for n in nodes]
    return Plot(
        [
            {
                "type": "bar",
                "x": [str(n) for n in nodes],
                "y": negativity,
                "name": "log negativity E_N (mode | rest), bits",
            },
            {
                "type": "bar",
                "x": [str(n) for n in nodes],
                "y": entropy,
                "name": "reduced entropy S, nats",
            },
        ],
        _layout(
            f"Entanglement diagnostics (global purity {state.purity():.4f})",
            "mode",
            "value",
            barmode="group",
        ),
    )


# ---- runs, optimization and numerics -------------------------------------------------
def measurement_probabilities(source: Any) -> Plot:
    """Recorded measurement likelihoods of a ``Result``, or a mapping outcome -> P.

    For a ``Result``, bars show ``measurement_statistics`` values: probabilities for
    discrete outcomes and densities (calibrated units, not probabilities) for
    continuous ones. Postselected keys are labelled as conditional branches.
    """
    if hasattr(source, "measurement_statistics"):
        traces = []
        for kind in ("probability", "density"):
            items = [(k, s) for k, s in source.measurement_statistics.items() if s["kind"] == kind]
            if items:
                traces.append(
                    {
                        "type": "bar",
                        "x": [
                            f"{k}{' (postselected)' if s.get('postselected') else ''}"
                            for k, s in items
                        ],
                        "y": [float(s["value"]) for _, s in items],
                        "name": kind,
                    }
                )
        return Plot(traces, _layout("Recorded measurement likelihoods", "outcome key", "value"))
    outcomes = dict(source)
    return Plot(
        [
            {
                "type": "bar",
                "x": [str(k) for k in outcomes],
                "y": [float(v) for v in outcomes.values()],
                "name": "probability",
            }
        ],
        _layout("Outcome probabilities", "outcome", "probability"),
    )


def optimization_curves(history: Any, *, log: bool = False) -> Plot:
    """Loss curves of a ``photographiqml`` ``History`` (or any object with list fields)."""
    traces = []
    for name in ("losses", "validation_losses", "gradient_norms"):
        values = list(getattr(history, name, None) or [])
        if values:
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": list(range(len(values))),
                    "y": [float(v) for v in values],
                    "name": name.replace("_", " "),
                }
            )
    if not traces:
        raise ValueError("History has no losses to plot")
    return Plot(
        traces,
        _layout("Optimization history", "iteration", "value", yaxis={"type": "log"} if log else {}),
    )


def fidelity_vs_cutoff(study: Any, reference=None) -> Plot:
    """Infidelity 1 - F versus Fock cutoff for a ``CutoffStudy``.

    With ``reference`` (a state with the same node order) F = fidelity(state, reference)
    per cutoff; otherwise the study's ``fidelity_to_previous`` of comparable rows
    (branches with differing outcomes are skipped, never compared). With a reference,
    every row must record the same outcomes (one conditional branch), else ValueError.
    Infidelities are floored at 1e-17 for the log axis.
    """
    if reference is not None and any(
        r.outcomes != study.results[0].outcomes for r in study.results
    ):
        raise ValueError(
            "Cutoff rows are different conditional branches; rerun cutoff_convergence with "
            "measurement_outcomes before comparing them with one reference state"
        )
    cutoffs, infidelity = [], []
    for row, result in zip(study.rows, study.results):
        if reference is not None:
            value = fidelity(result.state, reference)
        elif row.get("comparable_to_previous") and row.get("fidelity_to_previous") is not None:
            value = row["fidelity_to_previous"]
        else:
            continue
        cutoffs.append(row["cutoff"])
        infidelity.append(max(1.0 - float(value), 1e-17))
    label = "1 - F(state, reference)" if reference is not None else "1 - F(cutoff, previous)"
    return Plot(
        [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": cutoffs,
                "y": infidelity,
                "name": label,
            }
        ],
        _layout(
            "Cutoff convergence", "cutoff (total photons)", "infidelity", yaxis={"type": "log"}
        ),
    )


def runtime_scaling(
    sizes: Sequence[float], timings: Mapping[str, Sequence[float]], *, xlabel: str = "modes"
) -> Plot:
    """Log-log measured runtimes (seconds) versus problem size per method."""
    traces = []
    for name, seconds in timings.items():
        if len(seconds) != len(sizes):
            raise ValueError(f"{name!r} has {len(seconds)} timings for {len(sizes)} sizes")
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": list(sizes),
                "y": [float(s) for s in seconds],
                "name": name,
            }
        )
    return Plot(
        traces,
        _layout("Runtime scaling", xlabel, "seconds", xaxis={"type": "log"}, yaxis={"type": "log"}),
    )
