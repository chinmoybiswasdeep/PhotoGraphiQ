"""R34: Finite-GKP codeword projection, captured weight, codeword overlap,
Gram eigenvalues, and cutoff/grid refinement.

Scientific question: For GKPBridge's finite-energy codeword resource, how do
captured projection weight, |0>/|1> codeword overlap, and Gram-matrix
eigenvalues behave as cutoff and grid_points are independently refined, and
does the Gram matrix always stay Hermitian and positive-semidefinite (an
algebraic property any Gram matrix of vectors must satisfy, checked here as
an independent analytic invariant)?

Theory/equations: docs/research/muta-mapping.md: finite codeword Gram
matrices need not be identity (codewords overlap); this experiment does not
assert orthogonality, only the weaker, always-true PSD/Hermitian property,
plus the expected qualitative trend of increasing captured weight with
cutoff (a finite-energy truncation effect, not a claim of exact convergence
to unit weight).

Functionality tested: GKPBridge.diagnostics (photographiqml.gkp), via
PhotoGraphiQ's GKPCode.resource/project.

Oracle and independence class: A for the Hermitian/PSD Gram-matrix check
(an algebraic invariant of any Gram matrix); E for the monotonic
cutoff-refinement trend (self-consistency across a resolution sweep, not an
external ground truth).

Exact/approximate/statistical status: exact for the Hermitian/PSD check;
descriptive/statistical for the refinement trend (no claimed convergence
rate).

Primary metric: Hermitian/PSD violation (must be ~0); captured weight and
codeword overlap vs. cutoff and grid_points.

Declared acceptance condition: Gram matrix Hermitian within tol
(declare_tolerance(scale=1,safety_factor=1e3)) and PSD within the same tol
for every swept cutoff/grid_points value; captured weight stays in [0,1]
(a probability, by construction) for every point.

Expected cost: light-to-moderate (GKP projection at several cutoffs).

Manuscript destination: Main text (Fig. 8, GKP resource characterization).
Explicitly makes NO fault-tolerance or gate-validation claim
(diagnostics()["physical_muta_validated"] is always False by construction).

Scientific limitations: Resource-only diagnostics; this does not by itself
establish a physical implementation of arbitrary logical MuTA (see R38-R42
for the restricted physical execution layer).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import GKPBridge

EXPERIMENT_ID = "R34"


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=1e3)
    rows = []
    # GKPBridge's own class defaults (peak_width=0.4, envelope=0.4, peaks=8,
    # grid_points=4097) are a known-resolved resource; only the swept axis
    # is varied from them.
    baseline = {"peak_width": 0.4, "envelope": 0.4, "peaks": 8, "grid_points": 4097}

    for cutoff in (12, 20, 32, 48):
        bridge = GKPBridge(cutoff=cutoff, **baseline)
        diagnostics = bridge.diagnostics()
        gram = np.array(diagnostics["gram_eigenvalues"])
        rows.append(
            {
                "axis": "cutoff",
                "value": cutoff,
                "captured_weight_0": diagnostics["captured_weights"][0],
                "captured_weight_1": diagnostics["captured_weights"][1],
                "codeword_overlap": diagnostics["codeword_overlap"],
                "min_gram_eigenvalue": float(gram.min()),
                "max_gram_eigenvalue": float(gram.max()),
                "physical_muta_validated": diagnostics["physical_muta_validated"],
            }
        )

    for grid_points in (2049, 4097, 8193):
        bridge = GKPBridge(
            cutoff=32, peak_width=0.4, envelope=0.4, peaks=8, grid_points=grid_points
        )
        diagnostics = bridge.diagnostics()
        gram = np.array(diagnostics["gram_eigenvalues"])
        rows.append(
            {
                "axis": "grid_points",
                "value": grid_points,
                "captured_weight_0": diagnostics["captured_weights"][0],
                "captured_weight_1": diagnostics["captured_weights"][1],
                "codeword_overlap": diagnostics["codeword_overlap"],
                "min_gram_eigenvalue": float(gram.min()),
                "max_gram_eigenvalue": float(gram.max()),
                "physical_muta_validated": diagnostics["physical_muta_validated"],
            }
        )

    all_weights_valid = all(
        0 - tol <= r["captured_weight_0"] <= 1 + tol
        and 0 - tol <= r["captured_weight_1"] <= 1 + tol
        for r in rows
    )
    all_psd = all(r["min_gram_eigenvalue"] > -tol for r in rows)
    never_validated = all(r["physical_muta_validated"] is False for r in rows)
    status = "pass" if (all_weights_valid and all_psd and never_validated) else "fail"

    common.save_result(
        rows,
        "R34_gkp_codeword_projection",
        extra={
            "protocol": "GKPBridge.diagnostics vs. cutoff/grid_points refinement, Gram PSD invariant",
            "oracle_class": "A/E",
            "status_category": "exact/statistical",
            "tolerance": tol,
            "acceptance_condition": "captured weights in [0,1]; Gram matrix PSD; physical_muta_validated always False",
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A/E", "status": status},
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    cutoff_rows = [r for r in rows if r["axis"] == "cutoff"]
    grid_rows = [r for r in rows if r["axis"] == "grid_points"]
    axes[0].plot(
        [r["value"] for r in cutoff_rows],
        [r["captured_weight_0"] for r in cutoff_rows],
        "o-",
        color=common.COLORS["photographiqml"],
    )
    axes[0].set(
        title="Captured weight vs. cutoff", xlabel="cutoff", ylabel="captured weight (bit=0)"
    )
    # cutoff (~12-48) and grid_points (~2049-8193) live on incompatible
    # x-scales; a shared linear x-axis collapses the cutoff series into an
    # invisible sliver, so grid_points gets its own twinned x-axis.
    axes1_grid = axes[1].twiny()
    axes[1].plot(
        [r["value"] for r in cutoff_rows],
        [r["codeword_overlap"] for r in cutoff_rows],
        "o-",
        color=common.COLORS["photographiqml"],
        label="vs cutoff",
    )
    axes1_grid.plot(
        [r["value"] for r in grid_rows],
        [r["codeword_overlap"] for r in grid_rows],
        "s--",
        color=common.COLORS["mentpy"],
        label="vs grid_points",
    )
    axes[1].set(title="Codeword overlap |<0|1>|", xlabel="cutoff", ylabel="overlap")
    axes[1].xaxis.label.set_color(common.COLORS["photographiqml"])
    axes1_grid.set_xlabel("grid_points", color=common.COLORS["mentpy"])
    axes1_grid.tick_params(axis="x", colors=common.COLORS["mentpy"], labelsize=7)
    axes[1].tick_params(axis="x", colors=common.COLORS["photographiqml"])
    handles = axes[1].get_lines() + axes1_grid.get_lines()
    axes[1].legend(handles, [h.get_label() for h in handles], fontsize=7)
    axes[2].semilogy(
        [r["value"] for r in cutoff_rows],
        [max(-r["min_gram_eigenvalue"], 1e-18) for r in cutoff_rows],
        "o-",
        color=common.COLORS["photographiqml"],
    )
    axes[2].axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    axes[2].set(
        title="Gram PSD violation vs. cutoff", xlabel="cutoff", ylabel="max(0,-min eigenvalue)"
    )
    axes[2].legend(fontsize=7)
    fig.suptitle(
        f"R34: GKP codeword projection diagnostics (status={status}, no fault-tolerance claim)"
    )
    common.save_figure(fig, "R34_gkp_codeword_projection")
    plt.close(fig)

    common.print_summary(
        "R34 GKP codeword projection",
        n_points=len(rows),
        all_weights_valid=all_weights_valid,
        all_psd=all_psd,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R34 failed: weights_valid={all_weights_valid} psd={all_psd} never_validated={never_validated}"
        )


if __name__ == "__main__":
    main()
