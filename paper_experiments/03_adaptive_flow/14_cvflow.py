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

Upgrade migration (2026-11): the historical script predates the CV-flow
FINDER in ``photographiq.flows``. ``flows_study`` now adds:

* ``flows.analyze`` on the same line graph (typed IR ``OpenGraph``,
  ``Model.CV_GAUSSIAN``): status, decision, regimes, measurement order and the
  correction map, which must equal the supplied-order certificate above;
* the corrected pattern built from exactly that correction map
  (``flows.cv.corrected_pattern``), executed on NumPy ``gaussian`` and C++
  ``native-gaussian`` over several sampled outcomes (records must agree), with
  the ensemble compared with ``gaussian_channel``;
* a deliberately WRONG correction map (q-correction scaled by 0.5), which
  ``verify_cv_determinism`` must reject;
* cases where the algorithm must DECLINE certification: a singular bridge
  (``impossible`` with exact rational witnesses), a near-singular bridge
  (det 1e-10: ``unknown``/``numerically_uncertain`` by default, ideal-limit
  certificate only with ``exact=True``), a unit wire at r = 4 (``unknown`` with
  the default float64 check shift 3.0, certified with shift 1.5), a qubit
  algorithm on a CV graph and a non-Gaussian model (``unsupported``).

A qubit flow certificate is never used for a CV graph, and ``not_certified``
or ``unknown`` is never read as proof of nondeterminism except where an
exact impossibility witness is returned.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.cvflow import certify_cv_flow, flow_pattern  # noqa: E402

SQUEEZING = 1.0
FLOW_SHOTS = 400
FLOW_SEED = 20261113
SEM_FACTOR = 5.0
NATIVE_TOLERANCE = 1e-10


def _cv_graph(edges, inputs, outputs, squeezing):
    import networkx as nx

    from photographiq.ir import HomodyneSpec, Model, OpenGraph

    g = nx.Graph()
    g.add_weighted_edges_from(edges)
    return OpenGraph(Model.CV_GAUSSIAN, g, inputs, outputs,
                     {n: HomodyneSpec() for n in g if n not in outputs}, {n: squeezing for n in g})


def flows_study(valid_certificate) -> dict:
    import networkx as nx
    import numpy as np

    from photographiq import flows
    from photographiq.ir import HomodyneSpec, Model, OpenGraph

    report: dict = {}
    failures: list = []
    line = _cv_graph([(0, 1, 1.0), (1, 2, 1.0)], (0,), (2,), SQUEEZING)
    found = flows.analyze(line)
    report["line_status"] = found.status
    report["line_decision"] = found.diagnostics.get("decision")
    report["line_regimes"] = sorted(found.certificate.get("regimes", {}))
    report["line_measurement_order"] = list(found.measurement_order)
    report["line_layers"] = [list(layer) for layer in found.layers]
    report["line_correction_map"] = {str(k): {str(t): v for t, v in c.items()} for k, c in found.correction_map.items()}
    report["line_finite_squeezing_gain"] = found.certificate["determinism"]["finite_squeezing_gain"]
    report["line_decay_exponents"] = found.certificate["determinism"]["decay_exponents"]
    supplied = {str(k): {str(t): float(x) for t, x in v.items()} for k, v in valid_certificate.corrections.items()}
    finder = {str(k): {str(t): float(x) for t, x in v.items()} for k, v in found.certificate["layer_corrections"].items()}
    report["supplied_order_certificate_corrections"] = supplied
    report["finder_layer_corrections"] = finder
    if supplied != finder:
        failures.append("finder correction vectors differ from the supplied-order certificate")
    if not (found.certified and tuple(found.measurement_order) == (0, 1)
            and set(report["line_regimes"]) == {"ideal_limit", "finite_squeezing"}):
        failures.append("line graph not certified in both regimes with order (0, 1)")

    # Execute exactly the reported correction map on both Gaussian engines.
    graph = flows.resource_graph(line)
    shears = found.certificate["shears"]
    pattern = flows.cv.corrected_pattern(graph, found.measurement_order, found.correction_map, shears)
    channel = pg.gaussian_channel(pattern)
    inputs = {0: pg.GaussianInput.coherent(0.3 + 0.1j)}
    ideal = channel.apply(inputs[0].state(0))
    conditional = pg.simulate(pattern, backend="gaussian", inputs=inputs, seed=FLOW_SEED).state.covariance
    spread = ideal.covariance - conditional
    d = np.clip(np.diag(spread), 0, None)
    sem_mean, sem_cov = np.sqrt(d / FLOW_SHOTS), np.sqrt((np.outer(d, d) + spread**2) / FLOW_SHOTS)
    runs = {}
    for backend in ("gaussian", "native-gaussian"):
        result = pg.run_shots(pattern, FLOW_SHOTS, backend=backend, inputs=inputs, seed=FLOW_SEED)
        runs[backend] = result.trajectories
        ens = result.ensemble_state()
        report[f"{backend}_max_z"] = float(max(
            (np.abs(ens.mean - ideal.mean) / np.maximum(sem_mean, 1e-12)).max(),
            (np.abs(ens.covariance - ideal.covariance) / np.maximum(sem_cov, 1e-12)).max(),
        ))
        if report[f"{backend}_max_z"] > SEM_FACTOR:
            failures.append(f"{backend} ensemble of flow-corrected pattern")
    report["native_vs_gaussian_max_record_difference"] = max(
        abs(a.outcomes[k] - b.outcomes[k]) for a, b in zip(runs["gaussian"], runs["native-gaussian"], strict=True)
        for k in a.outcomes
    )
    report["corrected_channel_matrix"] = channel.matrix.tolist()
    report["corrected_channel_noise"] = channel.noise.tolist()
    if report["native_vs_gaussian_max_record_difference"] > NATIVE_TOLERANCE:
        failures.append("native trajectory contract")

    wrong = {u: {t: dict(c) for t, c in targets.items()} for u, targets in found.correction_map.items()}
    for targets in wrong.values():
        for c in targets.values():
            c["q"] *= 0.5
    wrong_report = flows.verify_cv_determinism(flows.cv.corrected_pattern(graph, found.measurement_order, wrong, shears))
    report["wrong_corrections_ideal_limit_deterministic"] = bool(wrong_report["ideal_limit_deterministic"])
    if wrong_report["ideal_limit_deterministic"]:
        failures.append("wrong correction map was not rejected")

    declines = []
    bridge0 = _cv_graph([("u", "a", 1.0), ("u", "b", 1.0), ("v", "a", 1.0), ("v", "b", 1.0)], ("u", "v"), ("a", "b"), 0.8)
    r = flows.analyze(bridge0)
    declines.append({"case": "singular bridge", "status": r.status, "decision": r.diagnostics.get("decision"),
                     "regimes": [], "witnesses": str(r.diagnostics.get("witnesses")),
                     "expected": "impossible with witnesses",
                     "ok": r.diagnostics.get("decision") == "impossible" and bool(r.diagnostics.get("witnesses"))})
    bridge = _cv_graph([("u", "a", 1.0), ("u", "b", 1.0), ("v", "a", 1.0), ("v", "b", 1.0 + 1e-10)], ("u", "v"), ("a", "b"), 0.8)
    for exact in (None, True):
        r = flows.analyze(bridge, exact=exact)
        regimes = sorted(r.certificate.get("regimes", {}))
        ok = (r.status == "unknown" and r.diagnostics.get("decision") == "numerically_uncertain") if exact is None else (
            r.certified and regimes == ["ideal_limit"])
        declines.append({"case": f"near-singular bridge det=1e-10, exact={exact}", "status": r.status,
                         "decision": r.diagnostics.get("decision"), "regimes": regimes,
                         "witnesses": None, "correction_gain_max": r.diagnostics.get("correction_gain_max"),
                         "expected": "unknown (declines)" if exact is None else "certified ideal_limit only", "ok": ok})
    wire = _cv_graph([(i, i + 1, 1.0) for i in range(4)], (0,), (4,), 4.0)
    for shift in (3.0, 1.5):
        r = flows.find_cv_flow(wire, shift=shift)
        regimes = sorted(r.certificate.get("regimes", {}))
        ok = r.status == "unknown" if shift == 3.0 else (r.certified and regimes == ["finite_squeezing", "ideal_limit"])
        declines.append({"case": f"unit wire r=4, shift={shift}", "status": r.status, "decision": r.diagnostics.get("decision"),
                         "regimes": regimes, "witnesses": None,
                         "expected": "unknown (float64 check cannot run)" if shift == 3.0 else "certified both regimes",
                         "ok": ok})
    r = flows.analyze(line, "gflow")
    declines.append({"case": "qubit gflow on CV graph", "status": r.status, "decision": None, "regimes": [],
                     "witnesses": None, "expected": "unsupported", "ok": r.status == "unsupported"})
    nongaussian = OpenGraph(Model.CV_NONGAUSSIAN, nx.path_graph(2), (0,), (1,), {0: HomodyneSpec()})
    r = flows.analyze(nongaussian)
    declines.append({"case": "CV_NONGAUSSIAN model", "status": r.status, "decision": None, "regimes": [],
                     "witnesses": None, "expected": "unsupported", "ok": r.status == "unsupported"})
    report["declines_and_rejections"] = declines
    failures += [f"decline case: {c['case']}" for c in declines if not c["ok"]]
    report["failures"] = failures
    return report


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
    study = flows_study(valid_certificate)
    summary["flows_study"] = study
    common.save_csv(study["declines_and_rejections"], "R14_cvflow_declines")
    common.save_json(summary, "R14_cvflow")
    common.write_metadata("R14_cvflow")
    common.save_raw(pattern.to_json(), "R14_cvflow_pattern.json")

    common.print_summary(
        "R14 CV-flow validation",
        valid_certificate_found=True,
        invalid_certificate_found=False,
        invalid_order_residual=residual,
        flows_line_status=study["line_status"],
        flows_line_regimes=study["line_regimes"],
        flows_measurement_order=study["line_measurement_order"],
        native_vs_gaussian=study["native_vs_gaussian_max_record_difference"],
        wrong_corrections_rejected=not study["wrong_corrections_ideal_limit_deterministic"],
        declines={c["case"]: (c["status"], c["ok"]) for c in study["declines_and_rejections"]},
    )
    if study["failures"]:
        common.save_json(study, "R14_cvflow_FAILURES")
        raise AssertionError(f"R14 flow checks failed: {study['failures']}")


if __name__ == "__main__":
    main()
