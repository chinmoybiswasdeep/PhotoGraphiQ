"""R13: Invalid causality / software-safety test (appendix result).

Constructs several intentionally invalid patterns and confirms that
``Pattern.validate()`` (or construction itself) rejects each one with the
expected, actual validator error — not a fabricated exception. This audits
the software's causal/structural safety checks rather than its physics.
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


def case_duplicate_inputs():
    return pg.Pattern(inputs=(0, 0)).validate()


def case_duplicate_resource_node():
    graph = pg.CVGraph.line(2, inputs=(0,))
    pattern = pg.Pattern(graph)
    pattern.append(Prepare(1, 1.0))  # node 1 already prepared by the graph
    return pattern.validate()


def case_self_cz():
    pattern = pg.Pattern(inputs=(0,))
    pattern.append(Entangle(0, 0, 1.0))
    return pattern.validate()


def case_output_not_final():
    pattern = pg.Pattern(inputs=(0,))
    pattern.append(Output((0,)))
    pattern.append(Displace(0, q=1.0))
    return pattern.validate()


def case_duplicate_outputs():
    pattern = pg.Pattern(inputs=(0,))
    pattern.append(Output((0, 0)))
    return pattern.validate()


def case_missing_or_measured_node():
    pattern = pg.Pattern(inputs=(0,))
    pattern.measure(0, Homodyne.q(), key="m0")
    pattern.append(Displace(0, q=1.0))  # 0 was destructively measured
    return pattern.validate()


def case_future_outcome_dependency():
    pattern = pg.Pattern(inputs=(0, 1))
    pattern.append(Displace(0, q=Outcome("future")))  # "future" not yet produced
    pattern.append(Measure(1, Homodyne.q(), "future"))
    return pattern.validate()


def case_cyclic_dependency():
    pattern = pg.Pattern(inputs=(0, 1))
    pattern.append(Measure(0, Homodyne(angle=Outcome("m1")), "m0"))
    pattern.append(Measure(1, Homodyne(angle=Outcome("m0")), "m1"))
    return pattern.validate()


def case_undefined_classical_key():
    pattern = pg.Pattern(inputs=(0,))
    pattern.append(Displace(0, q=Outcome("nonexistent")))
    pattern.append(Output((0,)))
    return pattern.validate()


def case_duplicate_classical_key():
    pattern = pg.Pattern(inputs=(0, 1))
    pattern.append(Measure(0, Homodyne.q(), "m"))
    pattern.append(Measure(1, Homodyne.q(), "m"))  # same key twice
    return pattern.validate()


def case_unsupported_measurement_description():
    pattern = pg.Pattern(inputs=(0,))
    pattern.append(Measure(0, "not a measurement", "m"))
    return pattern.validate()


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

    common.save_result(rows, "R13_invalid_causality")

    if unexpected:
        raise AssertionError(f"Expected these invalid patterns to be rejected but they were not: {unexpected}")

    common.print_summary(
        "R13 invalid causality / software safety",
        cases=len(rows),
        all_rejected=len(unexpected) == 0,
    )


if __name__ == "__main__":
    main()
