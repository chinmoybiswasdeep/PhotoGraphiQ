"""R13: Invalid causality / software-safety test (appendix result).

Constructs several intentionally invalid patterns and confirms that
``Pattern.validate()`` (or construction itself) rejects each one with the
expected, actual validator error — not a fabricated exception. This audits
the software's causal/structural safety checks rather than its physics.

Upgrade migration (2026-11): every invalid pattern is additionally handed
straight to ``simulate`` on the C++ ``native-gaussian`` engine without a prior
``validate()`` call; it must be rejected with the same exception type and
message before any state is prepared (``simulate`` validates first). A second
table uses ``photographiq.flows`` on CV open graphs: a graph with no CV flow
(rank-deficient correction matrix) must be classified ``impossible`` with
exact rational witnesses, and a qubit flow algorithm applied to a CV graph
must return ``unsupported`` -- never a qubit certificate for a CV graph.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.commands import Displace, Entangle, Measure, Output, Prepare  # noqa: E402
from photographiq.expressions import Outcome  # noqa: E402
from photographiq.measurements import Homodyne  # noqa: E402


MODE = {"finish": "validate"}


def finish(pattern):
    """Validate, or hand the unvalidated pattern straight to the native engine."""
    if MODE["finish"] == "validate":
        return pattern.validate()
    return pg.simulate(pattern, backend="native-gaussian", seed=0)


def case_duplicate_inputs():
    return finish(pg.Pattern(inputs=(0, 0)))


def case_duplicate_resource_node():
    graph = pg.CVGraph.line(2, inputs=(0,))
    pattern = pg.Pattern(graph)
    pattern.append(Prepare(1, 1.0))  # node 1 already prepared by the graph
    return finish(pattern)


def case_self_cz():
    pattern = pg.Pattern(inputs=(0,))
    pattern.append(Entangle(0, 0, 1.0))
    return finish(pattern)


def case_output_not_final():
    pattern = pg.Pattern(inputs=(0,))
    pattern.append(Output((0,)))
    pattern.append(Displace(0, q=1.0))
    return finish(pattern)


def case_duplicate_outputs():
    pattern = pg.Pattern(inputs=(0,))
    pattern.append(Output((0, 0)))
    return finish(pattern)


def case_missing_or_measured_node():
    pattern = pg.Pattern(inputs=(0,))
    pattern.measure(0, Homodyne.q(), key="m0")
    pattern.append(Displace(0, q=1.0))  # 0 was destructively measured
    return finish(pattern)


def case_future_outcome_dependency():
    pattern = pg.Pattern(inputs=(0, 1))
    pattern.append(Displace(0, q=Outcome("future")))  # "future" not yet produced
    pattern.append(Measure(1, Homodyne.q(), "future"))
    return finish(pattern)


def case_cyclic_dependency():
    pattern = pg.Pattern(inputs=(0, 1))
    pattern.append(Measure(0, Homodyne(angle=Outcome("m1")), "m0"))
    pattern.append(Measure(1, Homodyne(angle=Outcome("m0")), "m1"))
    return finish(pattern)


def case_undefined_classical_key():
    pattern = pg.Pattern(inputs=(0,))
    pattern.append(Displace(0, q=Outcome("nonexistent")))
    pattern.append(Output((0,)))
    return finish(pattern)


def case_duplicate_classical_key():
    pattern = pg.Pattern(inputs=(0, 1))
    pattern.append(Measure(0, Homodyne.q(), "m"))
    pattern.append(Measure(1, Homodyne.q(), "m"))  # same key twice
    return finish(pattern)


def case_unsupported_measurement_description():
    pattern = pg.Pattern(inputs=(0,))
    pattern.append(Measure(0, "not a measurement", "m"))
    return finish(pattern)


def case_ideal_graph_pattern():
    graph = pg.CVGraph.line(2, inputs=(0,), ideal=True)
    return pg.Pattern(graph)


CASES = [
    ("duplicate_inputs", case_duplicate_inputs),
    ("duplicate_resource_node", case_duplicate_resource_node),
    ("self_cz", case_self_cz),
    ("output_not_final", case_output_not_final),
    ("duplicate_outputs", case_duplicate_outputs),
    ("missing_or_measured_node", case_missing_or_measured_node),
    ("future_outcome_dependency", case_future_outcome_dependency),
    ("cyclic_dependency", case_cyclic_dependency),
    ("undefined_classical_key", case_undefined_classical_key),
    ("duplicate_classical_key", case_duplicate_classical_key),
    ("unsupported_measurement_description", case_unsupported_measurement_description),
    ("ideal_graph_pattern", case_ideal_graph_pattern),
]


def flow_rejection_rows():
    """CV open graphs whose determinism must be rejected (or declared out of scope)."""
    import networkx as nx

    from photographiq import flows
    from photographiq.ir import HomodyneSpec, Model, OpenGraph

    def cv_graph(edges, inputs, outputs):
        g = nx.Graph()
        g.add_weighted_edges_from(edges)
        return OpenGraph(Model.CV_GAUSSIAN, g, inputs, outputs,
                         {n: HomodyneSpec() for n in g if n not in outputs}, {n: 0.8 for n in g})

    singular_bridge = cv_graph([("u", "a", 1.0), ("u", "b", 1.0), ("v", "a", 1.0), ("v", "b", 1.0)], ("u", "v"), ("a", "b"))
    rows = []
    result = flows.analyze(singular_bridge)
    rows.append({
        "case": "singular_bridge_cv_flow", "status": result.status, "decision": result.diagnostics.get("decision"),
        "has_witnesses": bool(result.diagnostics.get("witnesses")),
        "expected": "not_certified/impossible with rational witnesses",
        "ok": result.status == "not_certified" and result.diagnostics.get("decision") == "impossible"
        and bool(result.diagnostics.get("witnesses")),
    })
    result = flows.analyze(singular_bridge, "gflow")
    rows.append({"case": "qubit_gflow_on_cv_graph", "status": result.status, "decision": None, "has_witnesses": False,
                 "expected": "unsupported", "ok": result.status == "unsupported"})
    return rows


def main():
    rows = []
    unexpected = []
    for name, builder in CASES:
        try:
            builder()
            rows.append({"case": name, "raised": False, "exception_type": None, "message": None})
            unexpected.append(name)
        except (ValueError, NotImplementedError, TypeError) as exc:
            rows.append(
                {"case": name, "raised": True, "exception_type": type(exc).__name__, "message": str(exc)}
            )

    MODE["finish"] = "simulate"
    native_mismatch = []
    for row, (name, builder) in zip(rows, CASES, strict=True):
        try:
            builder()
            row.update(native_raised=False, native_exception_type=None, native_message=None)
        except (ValueError, NotImplementedError, TypeError) as exc:
            row.update(native_raised=True, native_exception_type=type(exc).__name__, native_message=str(exc))
        row["native_matches_validate"] = (
            row["native_raised"] and row["native_exception_type"] == row["exception_type"]
            and row["native_message"] == row["message"]
        )
        if not row["native_matches_validate"]:
            native_mismatch.append(name)
    MODE["finish"] = "validate"
    flow_rows = flow_rejection_rows()
    common.save_csv(flow_rows, "R13_invalid_causality_flows")
    common.save_result(rows, "R13_invalid_causality", extra={"flow_rejections": flow_rows})
    if native_mismatch or not all(r["ok"] for r in flow_rows):
        raise AssertionError(f"native-gaussian rejection mismatch {native_mismatch} or flow rejection failure {flow_rows}")
    if unexpected:
        raise AssertionError(f"Expected these invalid patterns to be rejected but they were not: {unexpected}")

    common.print_summary(
        "R13 invalid causality / software safety",
        cases=len(rows),
        all_rejected=len(unexpected) == 0,
        native_gaussian_rejects_identically=not native_mismatch,
        flow_rejections={r["case"]: r["status"] for r in flow_rows},
    )


if __name__ == "__main__":
    main()
