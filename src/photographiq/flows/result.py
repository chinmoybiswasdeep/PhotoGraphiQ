"""Common result type of every flow/determinism analysis.

Status semantics (never conflate them):

* ``certified``: a certificate was found *and* independently verified.
* ``not_certified``: the algorithm found no certificate. This is a statement about
  the flow notion named in ``algorithm`` only; it is never a proof that the
  measurement pattern is nondeterministic. ``diagnostics['complete']`` says whether
  the algorithm is complete for that notion (then "no such flow exists").
* ``unknown``: the analysis could not decide (e.g. a numerical check disagreed with
  an algebraic certificate, or a heuristic gave up).
* ``unsupported``: the model/algorithm combination is outside the implemented scope.

CV flow results also carry ``diagnostics['decision']``: ``certified``, ``impossible``
(status ``not_certified`` with a proof in ``diagnostics['proof']``; the pattern can only
be postselected) or ``numerically_uncertain`` (status ``unknown``); certified CV results
state the covered regimes in ``certificate['regimes']`` (``ideal_limit``,
``finite_squeezing``). See docs/validation/cv-flow-robustness.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ir import Model

STATUSES = frozenset({"certified", "not_certified", "unknown", "unsupported"})


@dataclass(frozen=True)
class FlowAnalysisResult:
    """Outcome of a flow search on an :class:`~photographiq.ir.OpenGraph`.

    Args:
        status: One of ``certified``, ``not_certified``, ``unknown``, ``unsupported``.
        model: Computational model of the analysed graph.
        algorithm: Flow notion/algorithm that produced this result.
        measurement_order: Total measurement order of non-output nodes, earliest first
            (a linear extension of ``layers``).
        layers: Partial order as layers of measured nodes, earliest first; nodes in a
            layer are mutually unordered. Outputs follow every layer and are not listed.
        correction_map: Feed-forward per measured node. Qubit models:
            ``{'x': frozenset, 'z': frozenset}`` of Pauli corrections conditioned on
            the binary outcome, restricted to nodes measured later and outputs. CV:
            ``{target: {'q': cq, 'p': cp}}``, displacements ``q += cq*m``, ``p += cp*m``
            with ``m = record / sin(theta)`` the rescaled homodyne record.
        certificate: Flow function or correction sets and the checks that passed.
        assumptions: Conditions under which the certificate is meaningful.
        diagnostics: Failure reasons, residuals and other numbers.
    """

    status: str
    model: Model
    algorithm: str
    measurement_order: tuple = ()
    layers: tuple = ()
    correction_map: dict = field(default_factory=dict)
    certificate: dict = field(default_factory=dict)
    assumptions: tuple = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}")

    @property
    def certified(self) -> bool:
        return self.status == "certified"


def unsupported(model: Model, algorithm: str, reason: str) -> FlowAnalysisResult:
    return FlowAnalysisResult("unsupported", model, algorithm, diagnostics={"reason": reason})
