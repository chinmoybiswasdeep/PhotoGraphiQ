"""Interactive and publication-quality visualization of measurement-based computations.

Python is the authoritative model: graph semantics live in
:class:`photographiq.ir.OpenGraph`, numbers are computed by PhotoGraphiQ, and the
browser (Cytoscape.js / Plotly.js from pinned CDNs, or inlined local copies) only
displays and edits. Static figures use Matplotlib and need no browser.

Staged workflow::

    view = GraphView(open_graph).construct()        # A: graph, weights, roles
    view = view.with_flow(flow_result)              # B: dependency arrows, order, status
    view = view.with_corrections()                  # C: X/Z domains or CV displacements
    view = view.with_execution(execution_trace(pattern, result))  # D: animation
    view.save("figure.pdf", stage="corrections")    # static export
    graph, layout = apply_edits(exported_json)      # browser edits, re-validated
"""

from .editor import apply_edits, export_layout, import_layout
from .graph_view import STAGES, GraphView, measurement_text
from .layouts import (
    LAYOUTS,
    Layout,
    circular,
    compute_layout,
    force_directed,
    grid,
    hierarchical,
    layered,
    manual,
    publication,
)
from .overlays import Correction, FlowOverlay, normalize_corrections, pattern_corrections
from .plots import (
    Plot,
    covariance_ellipse,
    covariance_heatmap,
    entanglement_diagnostics,
    fidelity_vs_cutoff,
    gaussian_wigner,
    measurement_probabilities,
    optimization_curves,
    photon_number_distribution,
    photon_number_marginal,
    quadrature_density,
    quadrature_distribution,
    runtime_scaling,
    squeezing_ellipses,
    wigner_plot,
)
from .static import draw_static
from .trace import ExecutionTrace, TraceStep, execution_trace

__all__ = [
    "GraphView",
    "STAGES",
    "measurement_text",
    "Layout",
    "LAYOUTS",
    "compute_layout",
    "grid",
    "circular",
    "hierarchical",
    "layered",
    "force_directed",
    "publication",
    "manual",
    "Correction",
    "FlowOverlay",
    "normalize_corrections",
    "pattern_corrections",
    "ExecutionTrace",
    "TraceStep",
    "execution_trace",
    "apply_edits",
    "export_layout",
    "import_layout",
    "draw_static",
    "Plot",
    "photon_number_marginal",
    "photon_number_distribution",
    "quadrature_density",
    "quadrature_distribution",
    "gaussian_wigner",
    "wigner_plot",
    "covariance_heatmap",
    "covariance_ellipse",
    "squeezing_ellipses",
    "entanglement_diagnostics",
    "measurement_probabilities",
    "optimization_curves",
    "fidelity_vs_cutoff",
    "runtime_scaling",
]
