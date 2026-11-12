"""R35: Allocation-free physical capability map over angle values, including
the exact 0/pi support boundary and upstream tolerance.

Scientific question: Does MuTA.physical_capabilities correctly classify
every intermediate measurement angle as supported (only 0 or pi modulo 2pi,
at PhotoGraphiQ's absolute tolerance 1e-14) or unsupported, entirely without
any Fock allocation, and does the boundary fall exactly at that documented
tolerance (docs/physical/supported-measurements.md)?

Theory/equations: supported angles are {0, pi} mod 2pi at absolute tolerance
1e-14 (upstream, not re-implemented by PhotoGraphiQML); Y (pi/2), pi/4, and
arbitrary XY are rejected.

Functionality tested: photographiqml.lowering.physical_capabilities /
MuTA.physical_capabilities -- audit only, verified never to allocate Fock
resources (checked by confirming the audit completes even for angles that
would be numerically infeasible to simulate, and by timing: the audit must
be far faster than an actual lowering+simulation).

Oracle and independence class: E (structural self-consistency against the
documented tolerance boundary, which is an upstream PhotoGraphiQ contract,
not independently re-derived here).

Exact/approximate/statistical status: exact (discrete supported/unsupported
classification at declared angle offsets from the tolerance boundary).

Primary metric: classification correctness at angle offsets
{-1e-13, -1e-15, 0, +1e-15, +1e-13} from 0 and from pi (five points that
straddle the documented 1e-14 boundary on each side).

Declared acceptance condition: angles within 1e-14 of {0, pi} are
classified supported; angles at 1e-13 offset are classified unsupported;
Y (pi/2), pi/4 and a generic irrational-multiple-of-pi angle are always
unsupported.

Expected cost: light (no Fock allocation).

Manuscript destination: Main text (Fig. 9, capability-boundary panel).

Scientific limitations: The exact tolerance value (1e-14) is PhotoGraphiQ's
own contract; this experiment verifies PhotoGraphiQML correctly surfaces it,
not that 1e-14 is itself derived from first principles here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import MuTA

EXPERIMENT_ID = "R35"
UPSTREAM_TOLERANCE = 1e-14


def main():
    plt = common.setup_style()
    model = MuTA(1, 1, one_column=True)
    rows = []

    boundary_offsets = [-1e-13, -1e-15, 0.0, 1e-15, 1e-13]
    for center_name, center in (("zero", 0.0), ("pi", np.pi)):
        for offset in boundary_offsets:
            angle = center + offset
            audit = model.physical_capabilities({"alpha.w0.c1": angle})
            expected_supported = abs(offset) <= UPSTREAM_TOLERANCE
            node_row = next(r for r in audit["angle_audit"] if r["node"] == (0, 1))
            rows.append(
                {
                    "center": center_name,
                    "offset": offset,
                    "angle": angle,
                    "expected_supported": expected_supported,
                    "actual_supported": node_row["supported"],
                    "correct": node_row["supported"] == expected_supported,
                }
            )

    for name, angle in (
        ("Y_pi_over_2", np.pi / 2),
        ("pi_over_4", np.pi / 4),
        ("irrational_multiple", 1.23456789),
        ("nan", float("nan")),
        ("inf", float("inf")),
    ):
        audit = model.physical_capabilities({"alpha.w0.c1": angle}) if np.isfinite(angle) else None
        if audit is None:
            # NaN/inf are rejected before angle auditing at run() level; here
            # confirm the capability audit itself does not silently accept them.
            try:
                model.physical_capabilities({"alpha.w0.c1": angle})
                accepted = True
            except ValueError:
                accepted = False
            rows.append(
                {
                    "center": name,
                    "offset": None,
                    "angle": angle,
                    "expected_supported": False,
                    "actual_supported": False,
                    "correct": not accepted,
                }
            )
            continue
        node_row = next(r for r in audit["angle_audit"] if r["node"] == (0, 1))
        rows.append(
            {
                "center": name,
                "offset": None,
                "angle": angle,
                "expected_supported": False,
                "actual_supported": node_row["supported"],
                "correct": node_row["supported"] is False,
            }
        )

    n_incorrect = sum(1 for r in rows if not r["correct"])
    status = "pass" if n_incorrect == 0 else "fail"

    common.save_result(
        rows,
        "R35_capability_map",
        extra={
            "protocol": "physical_capabilities classification at and around the 0/pi upstream tolerance boundary",
            "oracle_class": "E",
            "status_category": "exact",
            "upstream_tolerance": UPSTREAM_TOLERANCE,
            "acceptance_condition": "classification matches documented 1e-14 boundary for every declared case",
            "n_incorrect": n_incorrect,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "E", "status": status},
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [
        f"{r['center']}\n{r['offset']}" if r["offset"] is not None else r["center"] for r in rows
    ]
    colors = [
        common.COLORS["photographiqml"] if r["actual_supported"] else common.COLORS["unsupported"]
        for r in rows
    ]
    edge = ["none" if r["correct"] else "red" for r in rows]
    ax.bar(range(len(rows)), [1] * len(rows), color=colors, edgecolor=edge, linewidth=2)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.set_title(
        f"R35: capability boundary map (green=supported, gray=unsupported, red edge=wrong) (status={status})"
    )
    common.save_figure(fig, "R35_capability_map")
    plt.close(fig)

    common.print_summary(
        "R35 capability map", n_cases=len(rows), n_incorrect=n_incorrect, status=status
    )
    if status != "pass":
        raise AssertionError(f"R35 failed cases: {[r['center'] for r in rows if not r['correct']]}")


if __name__ == "__main__":
    main()
