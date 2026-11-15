"""R47: Separate one-axis-at-a-time convergence studies for cutoff, grid
points, peak count, peak width, and envelope.

Scientific question: For a one-wire zero-angle physical-conditional
baseline, how do decoded probabilities change as each of the five resource/
numerical axes (cutoff, grid_points, peaks, peak_width, envelope) is
independently varied while holding the others fixed, and does the package
correctly refuse to silently certify convergence from this alone?

Theory/equations: docs/physical/convergence.md distinguishes numerical
refinement axes (cutoff, grid_points, peaks -- should stabilize decoded
probabilities as resolution increases, for a *fixed* physical resource) from
physical resource-change axes (peak_width, envelope -- these change the
finite-energy resource itself, so their "convergence" reflects a genuine
physical modification, not just numerical error shrinking).

Functionality tested: PhysicalMuTA.physical_convergence /
photographiqml.physical.validate_physical_model, for all five declared
axes.

Oracle and independence class: E (self-consistency -- successive-point
probability deltas are checked against a declared threshold for the
numerical axes; this is not compared to an external convergence oracle,
since no independent finite-resource GKP simulator is available at this
resolution).

Exact/approximate/statistical status: statistical/descriptive (finite-
resource numerical study; no claimed convergence rate).

Primary metric: max_probability_delta between successive points on each
axis; certified flag (must always be False).

Declared acceptance condition: certified is False for every axis (the
package must never silently claim certification); for the two numerical
axes (cutoff, grid_points), the max_probability_delta at the finest pair of
points is smaller than at the coarsest pair (a genuine refinement trend,
not merely "some decrease somewhere").

Expected cost: moderate-to-heavy (5 axes x up to 4 points each, all
physical-conditional Fock simulations).

Manuscript destination: Main text (Fig. 12, convergence panel).

Scientific limitations: One-wire, zero-angle case only (matching the
tractable resource baseline in docs/physical/evidence.json); numerical
grid/cutoff stability does not by itself remove finite-resource physical
error (peak_width/envelope axes), per docs/physical/convergence.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common

from photographiqml import GKPPhysicalConfig, PhysicalMuTA

EXPERIMENT_ID = "R47"


def main():
    plt = common.setup_style()
    baseline = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
    model = PhysicalMuTA(1, physical_config=baseline)
    outcomes = dict.fromkeys(model.measurement_order, 0.0)

    axes_values = {
        "cutoff": [20, 24, 28],
        "grid_points": [513, 1025, 2049],
        "peaks": [3, 4, 5],
        "peak_width": [0.85, 0.9, 0.95],
        "envelope": [0.85, 0.9, 0.95],
    }

    studies = {}
    rows = []
    for axis, values in axes_values.items():
        study = model.physical_convergence(
            [1, 0], values, axis=axis, mode="physical-conditional", analog_outcomes=outcomes
        )
        studies[axis] = study
        for row in study["rows"]:
            rows.append(
                {
                    "axis": axis,
                    "axis_type": study["axis_type"],
                    "value": row["value"],
                    "max_probability_delta": row["max_probability_delta"],
                    "prediction": row["prediction"],
                }
            )

    all_uncertified = all(not s["certified"] for s in studies.values())
    numerical_axes_ok = True
    for axis in ("cutoff", "grid_points"):
        deltas = [
            r["max_probability_delta"]
            for r in rows
            if r["axis"] == axis and r["max_probability_delta"] is not None
        ]
        if len(deltas) < 2 or not (deltas[-1] < deltas[0]):
            numerical_axes_ok = False

    status = "pass" if (all_uncertified and numerical_axes_ok) else "fail"

    common.save_result(
        rows,
        "R47_resource_axis_convergence",
        extra={
            "protocol": "One-axis-at-a-time physical_convergence study, 5 axes, one-wire zero-angle baseline",
            "oracle_class": "E",
            "status_category": "statistical",
            "baseline_config": baseline.to_dict(),
            "acceptance_condition": "certified False for every axis; numerical axes (cutoff, grid_points) show decreasing successive-point delta",
            "all_uncertified": all_uncertified,
            "numerical_axes_ok": numerical_axes_ok,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "E", "status": status},
    )

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.4))
    for ax, (axis, values) in zip(axes, axes_values.items()):
        axis_rows = [r for r in rows if r["axis"] == axis]
        deltas = [
            r["max_probability_delta"] if r["max_probability_delta"] is not None else 0
            for r in axis_rows
        ]
        color = (
            common.COLORS["photographiqml"]
            if axis in ("cutoff", "grid_points", "peaks")
            else common.COLORS["piquasso"]
        )
        ax.plot([r["value"] for r in axis_rows], deltas, "o-", color=color)
        ax.set(
            title=f"{axis}\n({'numerical' if axis in ('cutoff', 'grid_points', 'peaks') else 'physical resource'})",
            xlabel=axis,
            ylabel="max prob delta",
        )
    fig.suptitle(f"R47: one-axis-at-a-time convergence (status={status}, never certified)")
    common.save_figure(fig, "R47_resource_axis_convergence")
    plt.close(fig)

    common.print_summary(
        "R47 resource axis convergence",
        all_uncertified=all_uncertified,
        numerical_axes_ok=numerical_axes_ok,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R47 failed: all_uncertified={all_uncertified} numerical_axes_ok={numerical_axes_ok}"
        )


if __name__ == "__main__":
    main()
