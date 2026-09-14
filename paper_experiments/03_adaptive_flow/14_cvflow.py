"""R14: CV-flow validation.

Uses the current, supplied-total-order CV-flow certificate
(``photographiq.cvflow.certify_cv_flow``/``flow_pattern``), which implements
the real-linear condition of Booth and Markham (Quantum 7, 1146, 2023) for
one given measurement order. It does NOT search over orders, so this script
only demonstrates certification/rejection for two concrete orders on the
same three-node line resource -- not a general optimal-flow finder.

Valid example: measuring the line graph 0->1->2 (input 0, output 2) in the
natural order (0, 1) yields a zero-residual certificate. Invalid example:
measuring the same graph in the reversed order (1, 0) has no exact real-linear
correction (node 1 alone cannot supply the needed influence on the surviving
output once node 0, its only other neighbor, is excluded as an input), so
certification returns None with reported residuals in the diagnostic case.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.cvflow import certify_cv_flow, flow_pattern  # noqa: E402

SQUEEZING = 1.0


def main():
    graph = pg.CVGraph.line(3, squeezing=SQUEEZING, inputs=(0,), outputs=(2,))

    valid_order = (0, 1)
    valid_certificate = certify_cv_flow(graph, valid_order)
    if valid_certificate is None:
        raise AssertionError("Expected order (0,1) to admit a CV-flow certificate")

    invalid_order = (1, 0)
    invalid_certificate = certify_cv_flow(graph, invalid_order)
    if invalid_certificate is not None:
        raise AssertionError("Expected order (1,0) to have no CV-flow certificate")

    # Diagnose *why* the invalid order fails: recompute the same linear system
    # certify_cv_flow solves internally, to report an explicit residual.
    import numpy as np

    a = graph.adjacency()
    indices = {n: i for i, n in enumerate(graph.nodes)}
    past = invalid_order[:2]
    future = tuple(n for n in graph.nodes if n not in past and n not in graph.inputs)
    matrix = a[np.ix_([indices[n] for n in past], [indices[n] for n in future])]
    target = np.zeros(len(past))
    target[-1] = 1
    solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = float(np.linalg.norm(matrix @ solution - target, ord=np.inf))

    pattern = flow_pattern(graph, valid_order)
    input_state = pg.GaussianInput.coherent(0.3 + 0.1j)
    result = pg.simulate(pattern, backend="gaussian", inputs={0: input_state}, seed=1)
    ideal_channel = pg.gaussian_channel(pattern)
    ideal = ideal_channel.apply(input_state.state(0))

    summary = {
        "graph": "line(3), inputs=(0,), outputs=(2,)",
        "valid_order": valid_order,
        "valid_certificate_corrections": {str(k): v for k, v in valid_certificate.corrections.items()},
        "valid_certificate_residuals": valid_certificate.residuals,
        "invalid_order": invalid_order,
        "invalid_certificate_result": None,
        "invalid_order_diagnostic_residual": residual,
        "invalid_order_diagnostic_solution": solution.tolist(),
        "flow_pattern_commands": len(pattern.commands),
        "flow_pattern_output_mean": result.state.mean.tolist(),
        "flow_pattern_output_covariance": result.state.covariance.tolist(),
        "affine_channel_mean": ideal.mean.tolist(),
        "affine_channel_covariance": ideal.covariance.tolist(),
    }
    common.save_json(summary, "R14_cvflow")
    common.write_metadata("R14_cvflow")
    common.save_raw(pattern.to_json(), "R14_cvflow_pattern.json")

    common.print_summary(
        "R14 CV-flow validation",
        valid_certificate_found=True,
        invalid_certificate_found=False,
        invalid_order_residual=residual,
    )


if __name__ == "__main__":
    main()
