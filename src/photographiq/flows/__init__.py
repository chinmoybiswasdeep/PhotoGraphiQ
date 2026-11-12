"""Flow and determinism analysis of typed open graphs (:class:`photographiq.ir.OpenGraph`).

Each computational model gets its own notion; they are never interchanged:

* ``Model.QUBIT``: causal flow, plane-aware gflow and Pauli flow (GF(2), unweighted).
* ``Model.CV_GAUSSIAN``: real-linear CV flow on the weighted graph (Booth-Markham
  2023), order search, and numerical determinism verification by Gaussian
  postselection. Qubit flows are never used as a CV determinism certificate.
* ``Model.CV_NONGAUSSIAN`` and ``Model.GKP_LOGICAL``: ``unsupported``.

``photographiq.flow`` (command DAG scheduling) and ``photographiq.cvflow``
(supplied-order CV certificate) are separate, lower-level modules.
"""

from __future__ import annotations

from ..ir import Model, OpenGraph, QubitMeasurement
from .causal import find_causal_flow, verify_causal_flow
from .cv import find_cv_flow, resource_graph, search_cv_order, verify_cv_determinism
from .gflow import find_gflow, verify_gflow
from .pauli import find_pauli_flow, pauli_label, verify_pauli_flow
from .result import FlowAnalysisResult, unsupported

ALGORITHMS = ("auto", "causal", "gflow", "pauli", "cv")

_QUBIT = {"causal": find_causal_flow, "gflow": find_gflow, "pauli": find_pauli_flow}


def analyze(open_graph: OpenGraph, algorithm: str = "auto", **options) -> FlowAnalysisResult:
    """Analyse determinism of ``open_graph`` with the flow notion of its model.

    Args:
        open_graph: Typed open graph; every measured node needs a measurement spec.
        algorithm: ``auto`` (qubit: causal, then gflow, then Pauli flow when Pauli
            labels are present; CV Gaussian: ``cv``), or one of causal/gflow/pauli/cv.
        **options: Forwarded to :func:`find_cv_flow` (tolerance, verify, seed,
            verify_tolerance); rejected for qubit algorithms.

    Returns:
        The first certified result in ``auto`` mode, otherwise the last attempt with the
        earlier attempts recorded in ``diagnostics['attempts']``.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    model = open_graph.model
    if model in (Model.CV_NONGAUSSIAN, Model.GKP_LOGICAL):
        return unsupported(
            model,
            algorithm,
            f"no determinism analysis is implemented for {model.value}: displacement "
            "feed-forward does not commute with non-Gaussian elements, and GKP logical "
            "flows would need a finite-GKP error model",
        )
    if model is Model.CV_GAUSSIAN:
        if algorithm not in ("auto", "cv"):
            return unsupported(model, algorithm, "qubit flow rules do not apply to CV graphs")
        return find_cv_flow(open_graph, **options)
    if algorithm == "cv":  # before the options check: CV options are valid for "cv"
        return unsupported(model, algorithm, "CV flow applies to CV models only")
    if options:
        raise TypeError(f"Unexpected options for qubit flow analysis: {sorted(options)}")
    if algorithm != "auto":
        return _QUBIT[algorithm](open_graph)
    names = ["causal", "gflow"]
    if any(isinstance(m, QubitMeasurement) and m.pauli for m in open_graph.measurements.values()):
        names.append("pauli")
    attempts = {}
    for name in names:
        result = _QUBIT[name](open_graph)
        if result.certified:
            return result
        attempts[name] = result.diagnostics.get("reason")
    diagnostics = dict(result.diagnostics, attempts=attempts)
    diagnostics["reason"] = (
        "no certificate from " + ", ".join(names) + "; this does not prove nondeterminism"
    )
    return FlowAnalysisResult(
        "not_certified", model, "auto", assumptions=result.assumptions, diagnostics=diagnostics
    )


__all__ = [
    "ALGORITHMS",
    "FlowAnalysisResult",
    "analyze",
    "find_causal_flow",
    "find_cv_flow",
    "find_gflow",
    "find_pauli_flow",
    "pauli_label",
    "resource_graph",
    "search_cv_order",
    "verify_causal_flow",
    "verify_cv_determinism",
    "verify_gflow",
    "verify_pauli_flow",
]
