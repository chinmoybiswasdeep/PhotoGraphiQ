"""Typed CV process diagrams: Gaussian semantics, sound rewrites and CV-MBQC optimisation.

A separate calculus from the qubit ZX module (:mod:`photographiq.calculus.zx`) and from the
pure-state Gaussian graphical calculus (:class:`~photographiq.calculus.GaussianGraph`), with
explicit converters to both PhotoGraphiQ objects. It is *not* a complete CV-ZX calculus:

* Gaussian layer: typed generators, hybrid classical-quantum Gaussian semantics
  (unitary / averaged channel / conditional instrument), rewrite rules each with a proof or a
  cited theorem and a declared equivalence class (``exact-instrument``, ``exact-channel``,
  ``logical``).
* MBQC layer: Pattern <-> diagram translation, resource costs, CV-flow checks, verified
  bounded search in ``physical``, ``logical`` and ``approximate`` modes.
* Non-Gaussian layer (restricted): 1 -> 1 q-spiders exp(i f(q)) of Nagayoshi et al.
  (PRR 7, 033141 (2025)) with their fusion and displacement rules as operator identities,
  Kerr, Fock preparation, photon counting, post-selection and finite-energy GKP
  preparation/readout, evaluated on the native Fock engines (truncated, convergence reported).

Theory and rule proofs: docs/theory/CV_ZX_FOUNDATIONS.md.
"""

from .fock import FockRun, cutoff_study, run_fock
from .ir import (
    CZ,
    BeamSplitter,
    Box,
    ClassicalSource,
    CVDiagram,
    DiagramTypeError,
    Discard,
    Displacement,
    Feedforward,
    FockPrepare,
    GaussianChannelBox,
    GaussianLoss,
    GKPPrepare,
    Homodyne,
    Kerr,
    ModularDecode,
    PhasePolynomial,
    PhotonCount,
    Postselect,
    Prepare,
    PrepareState,
    Rotation,
    Shear,
    Squeezer,
    Symplectic,
)
from .optimize import (
    EquivalenceReport,
    OptimizationResult,
    compare,
    optimize_cv_pattern,
    pareto_front,
    pattern_flow_report,
    resource_cost,
    weighted_cost,
)
from .rules import RULES, RewriteStep, apply_rewrite, teleport_steps
from .semantics import (
    HybridGaussian,
    channel_semantics,
    compose_channels,
    conditional_semantics,
    evaluate,
    is_completely_positive,
    unitary_semantics,
)
from .translate import (
    from_circuit,
    from_gaussian_graph,
    from_pattern,
    to_circuit,
    to_gaussian_graph,
    to_pattern,
)
from .visualization import draw, trace_json

__all__ = [
    "BeamSplitter",
    "Box",
    "CVDiagram",
    "CZ",
    "ClassicalSource",
    "DiagramTypeError",
    "Discard",
    "Displacement",
    "EquivalenceReport",
    "Feedforward",
    "FockPrepare",
    "FockRun",
    "GKPPrepare",
    "GaussianChannelBox",
    "GaussianLoss",
    "Homodyne",
    "HybridGaussian",
    "Kerr",
    "ModularDecode",
    "OptimizationResult",
    "PhasePolynomial",
    "PhotonCount",
    "Postselect",
    "Prepare",
    "PrepareState",
    "RULES",
    "RewriteStep",
    "Rotation",
    "Shear",
    "Squeezer",
    "Symplectic",
    "apply_rewrite",
    "channel_semantics",
    "compare",
    "compose_channels",
    "conditional_semantics",
    "cutoff_study",
    "draw",
    "evaluate",
    "from_circuit",
    "from_gaussian_graph",
    "from_pattern",
    "is_completely_positive",
    "optimize_cv_pattern",
    "pareto_front",
    "pattern_flow_report",
    "resource_cost",
    "run_fock",
    "teleport_steps",
    "to_circuit",
    "to_gaussian_graph",
    "to_pattern",
    "trace_json",
    "unitary_semantics",
    "weighted_cost",
]
