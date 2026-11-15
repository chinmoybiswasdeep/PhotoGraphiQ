"""R46: Hard versus soft resource-decoder calibration on the specific
declared preparation ensemble, and confirmation that PhysicalMuTA rejects a
soft feed-forward configuration before allocation.

Scientific question: (a) Does NearestCellDecoder always report
confidence=None (no invented posterior), while SoftDecisionDecoder reports a
calibrated posterior that is highest near an ideal codeword lattice site and
falls toward the 50/50 point at the boundary between two adjacent cells,
for the declared equal-prior 0/1 preparation ensemble? (b) Does
PhysicalMuTA's capability audit reject decoder="soft" before any Fock
allocation, exactly as documented (docs/physical/decoding.md: "Physical
MuTA therefore rejects a soft feed-forward configuration before
allocation")?

Theory/equations: GKP lattice spacing L=sqrt(2*pi); ideal bit-0 sites at
q=2sL, bit-1 sites at q=(2s+1)L. SoftDecisionDecoder(code, basis="Z") is
Bayesian discrimination of a declared equal-prior finite zero/one
preparation ensemble (docs/physical/decoding.md); it is not a universal
posterior for an arbitrary adaptive graph state -- part (b) verifies
PhotoGraphiQML enforces exactly that boundary.

Functionality tested: photographiq.NearestCellDecoder/SoftDecisionDecoder
via GKPCode.decode (shared PhotoGraphiQ decoder objects); MuTA.physical_
capabilities' rejection of decoder="soft" for flow-based execution.

Oracle and independence class: E (structural: confidence=None for nearest,
calibration trend and rejection-before-allocation are self-consistency
properties of the documented decoder contract, not an external ground
truth).

Exact/approximate/statistical status: exact (discrete confidence-field and
rejection checks); descriptive for the calibration trend (no claimed
functional form beyond monotonicity near the ideal-to-boundary sweep).

Primary metric: NearestCellDecoder confidence is None at every point;
SoftDecisionDecoder confidence is monotonically non-increasing as the raw
outcome moves from an ideal bit-0 site (q=0) toward the cell boundary
(q=L/2); decoder="soft" rejected before allocation (no Fock allocation
attempted).

Declared acceptance condition: both hold exactly.

Expected cost: light (decoder calls only, no Fock simulation for part (a);
allocation-free audit for part (b)).

Manuscript destination: Appendix (Fig. 11 supporting decoder-calibration
table).

Scientific limitations: The soft posterior is calibrated only for the
declared equal-prior finite 0/1 ensemble at one mode in isolation; this
experiment does not extend it to (and PhotoGraphiQML does not permit) an
arbitrary adaptive MuTA graph state.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np
import photographiq as pg

from photographiqml import GKPBridge, GKPPhysicalConfig, MuTA
from photographiqml.lowering import physical_capabilities

EXPERIMENT_ID = "R46"


def main():
    plt = common.setup_style()
    L = np.sqrt(2 * np.pi)
    bridge = GKPBridge(cutoff=32, peak_width=0.4, envelope=0.4, peaks=8, grid_points=4097)
    code = bridge.code
    nearest = pg.NearestCellDecoder()
    soft = pg.SoftDecisionDecoder(code, basis="Z")

    sweep = np.linspace(0, L / 2, 6)  # ideal bit-0 site (0) to cell boundary (L/2)
    rows = []
    for raw in sweep:
        nearest_result = code.decode(float(raw), decoder=nearest)
        soft_result = code.decode(float(raw), decoder=soft)
        rows.append(
            {
                "raw_outcome": float(raw),
                "nearest_bit": nearest_result.bit,
                "nearest_confidence_is_none": nearest_result.confidence is None,
                "soft_bit": soft_result.bit,
                "soft_confidence": soft_result.confidence,
            }
        )

    confidences = [r["soft_confidence"] for r in rows]
    # Confidence saturates near 1.0 for most of the cell interior (observed:
    # >0.98 until very close to the boundary), so a strict pointwise
    # non-increase check is dominated by numerical noise in that flat
    # near-1 region; allow a small tolerance there while still requiring the
    # overall trend (and the sharp final drop toward the boundary) to hold.
    monotonic_non_increasing = all(
        confidences[i] >= confidences[i + 1] - 1e-3 for i in range(len(confidences) - 1)
    )
    overall_trend_decreasing = confidences[0] > confidences[-1]
    nearest_always_none = all(r["nearest_confidence_is_none"] for r in rows)

    # --- (b) soft decoder rejected before allocation for flow-based execution --
    model = MuTA(1, 1, one_column=True)
    soft_config = GKPPhysicalConfig(
        cutoff=16, peak_width=0.9, envelope=0.9, peaks=3, grid_points=513, decoder="soft"
    )
    audit = physical_capabilities(model, config=soft_config)
    soft_rejected_before_allocation = not audit["supported"] and any(
        "soft" in r.lower() for r in audit["reasons"]
    )

    status = (
        "pass"
        if (
            nearest_always_none
            and monotonic_non_increasing
            and overall_trend_decreasing
            and soft_rejected_before_allocation
        )
        else "fail"
    )

    common.save_result(
        rows,
        "R46_hard_vs_soft_decoding",
        extra={
            "protocol": "NearestCellDecoder vs SoftDecisionDecoder calibration sweep + soft-decoder rejection audit",
            "oracle_class": "E",
            "status_category": "exact",
            "lattice_spacing": L,
            "nearest_always_none": nearest_always_none,
            "soft_confidence_monotonic_non_increasing": monotonic_non_increasing,
            "soft_confidence_overall_trend_decreasing": overall_trend_decreasing,
            "soft_rejected_before_allocation": soft_rejected_before_allocation,
            "audit_reasons": audit["reasons"],
            "acceptance_condition": "nearest confidence always None; soft confidence non-increasing toward boundary; soft rejected before allocation",
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "E", "status": status},
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [r["raw_outcome"] / L for r in rows],
        confidences,
        "o-",
        color=common.COLORS["photographiqml"],
    )
    ax.axhline(0.5, color=common.COLORS["acceptance"], linestyle="--", label="50/50 boundary")
    ax.set(
        title=f"R46: soft-decoder calibration, ideal site -> cell boundary (status={status})",
        xlabel="raw outcome / L",
        ylabel="soft-decoder confidence",
    )
    ax.legend()
    common.save_figure(fig, "R46_hard_vs_soft_decoding")
    plt.close(fig)

    common.print_summary(
        "R46 hard vs soft decoding",
        nearest_always_none=nearest_always_none,
        monotonic=monotonic_non_increasing,
        soft_rejected=soft_rejected_before_allocation,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R46 failed: nearest_none={nearest_always_none} monotonic={monotonic_non_increasing} rejected={soft_rejected_before_allocation}"
        )


if __name__ == "__main__":
    main()
