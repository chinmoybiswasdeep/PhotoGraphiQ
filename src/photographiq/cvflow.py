"""Real-linear CV-flow certificates for a supplied total measurement order.

Implements the correction-matrix condition of Booth and Markham, Quantum 7,
1146 (2023), Definitions 5--6 and Lemma 7. Does not search over orders or claim
maximally parallel flow. Certificates are numerical, with reported residuals.
"""

from dataclasses import dataclass

import numpy as np

from .expressions import Outcome, atan2, sin
from .measurements import Homodyne
from .pattern import Pattern


@dataclass(frozen=True)
class CVFlow:
    order: tuple
    corrections: dict
    residuals: tuple[float, ...]


def certify_cv_flow(graph, order, *, parameters=None, tolerance=1e-10):
    """Return a supplied-order CV-flow certificate, or None if no solution exists."""
    graph.validate()
    order = tuple(order)
    if len(set(order)) != len(order) or set(order) != set(graph.nodes) - set(graph.outputs):
        raise ValueError("Order must contain each non-output node exactly once")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be positive and finite")
    a = graph.adjacency(parameters)
    indices = {n: i for i, n in enumerate(graph.nodes)}
    corrections, residuals = {}, []
    for i, node in enumerate(order):
        past = order[: i + 1]
        future = tuple(n for n in graph.nodes if n not in past and n not in graph.inputs)
        matrix = a[np.ix_([indices[n] for n in past], [indices[n] for n in future])]
        target = np.zeros(len(past))
        target[-1] = 1
        vector = np.linalg.lstsq(matrix, target, rcond=None)[0]
        residual = float(np.linalg.norm(matrix @ vector - target, ord=np.inf))
        if residual > tolerance:
            return None
        corrections[node] = {n: float(v) for n, v in zip(future, vector)}
        residuals.append(residual)
    return CVFlow(order, corrections, tuple(residuals))


def flow_pattern(graph, order, *, shears=None, parameters=None):
    """Build the finite-resource protocol associated with a certified total order.

    All graph entanglement is prepared first. Corrections apply on unmeasured
    nodes. The certificate concerns the ideal-limit model, not finite-r unitarity.
    """
    flow = certify_cv_flow(graph, order, parameters=parameters)
    if flow is None:
        raise ValueError("The supplied order has no CV-flow certificate")
    adjacency = graph.adjacency(parameters)
    # Freeze the certified numerical couplings: rebinding them later would
    # invalidate the correction coefficients derived from this certificate.
    resource = graph.copy()
    for u, v, data in resource.network.edges(data=True):
        data["weight"] = float(adjacency[graph.nodes.index(u), graph.nodes.index(v)])
    pattern = Pattern(resource)
    shears = dict(shears or {})
    for i, node in enumerate(flow.order):
        angle = atan2(1, shears.get(node, 0.0))
        pattern.measure(node, Homodyne(angle))
        m = Outcome(node) / sin(angle)
        correction = flow.corrections[node]
        for target in graph.nodes:
            if target in flow.order[: i + 1]:
                continue
            q = correction.get(target, 0.0)
            p = sum(
                adjacency[graph.nodes.index(target), graph.nodes.index(n)] * v
                for n, v in correction.items()
            )
            if q != 0 or p != 0:
                pattern.displace(target, q=-q * m, p=-p * m)
    return pattern.validate()
