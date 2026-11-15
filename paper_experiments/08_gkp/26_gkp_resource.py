"""R26: GKP resource characterization.

Generates finite-energy square-lattice GKP resources
(``photographiq.gkp.GKPResource``), characterizes their q-quadrature
probability structure (comparing the intended smooth target wavefunction to
its finite-Fock projection, reconstructed from the Hermite basis functions
the package itself uses -- ``photographiq.fock_measurements.wavefunctions``),
stabilizer expectations (``photographiq.gkp.stabilizers``), logical-codeword
overlap (``GKPCode.gram``), and normalization/cutoff sensitivity via
``GKPResource.project``. This makes NO fault-tolerance or threshold claim.

Upgrade migration (2026-11): the resources are loaded on ``native-fock``
(main) and ``piquasso-fock`` (historical reference); both engines must give
the same stabilizer expectations and mean photon number to
ENGINE_TOLERANCE. The cutoff-sensitivity sweep now also evaluates the
stabilizers on ``native-fock`` at every cutoff (cutoff requirement of the
stabilizer estimate itself), next to the captured mass of the projection.
The projection's captured mass is always reported: the Fock input is the
normalised projection, so 1 - captured_mass is the discarded weight.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.fock_measurements import wavefunctions  # noqa: E402

PARAMETER_SETS = [
    {"peak_width": 0.3, "envelope": 0.3, "label": "sharp"},
    {"peak_width": 0.4, "envelope": 0.4, "label": "default"},
    {"peak_width": 0.6, "envelope": 0.6, "label": "fuzzy"},
]
DEFAULT_CUTOFF = 48
CUTOFF_SENSITIVITY = [24, 32, 48, 64, 96]
MAIN_BACKEND = "native-fock"
REFERENCE_BACKEND = "piquasso-fock"
ENGINE_TOLERANCE = 1e-9  # same normalised input vector, identity evolution
def reconstructed_probability(amplitudes: np.ndarray, q_grid: np.ndarray) -> np.ndarray:
    """|sum_n c_n phi_n(q)|^2 using the package's own Hermite wavefunctions."""
    cutoff = len(amplitudes)
    basis = np.array([wavefunctions(x, cutoff) for x in q_grid])  # (len(q), cutoff)
    values = basis @ amplitudes
    return np.abs(values) ** 2


def main():
    plt = common.setup_style()
    rows = []
    representative_data = None

    for params in PARAMETER_SETS:
        for logical in (0, 1):
            resource = pg.GKPResource(logical, peak_width=params["peak_width"], envelope=params["envelope"])
            fock_input, captured_mass = resource.project(DEFAULT_CUTOFF)

            pattern = pg.Pattern(inputs=(0,)).append(pg.Output((0,))).validate()
            state = pg.simulate(pattern, backend=MAIN_BACKEND, cutoff=DEFAULT_CUTOFF, inputs={0: fock_input}, seed=0).state
            stabilizer_values = pg.gkp.stabilizers(state)
            reference = pg.simulate(pattern, backend=REFERENCE_BACKEND, cutoff=DEFAULT_CUTOFF, inputs={0: fock_input}, seed=0).state
            reference_stabilizers = pg.gkp.stabilizers(reference)
            engine_difference = max(
                abs(stabilizer_values[k] - reference_stabilizers[k]) for k in ("q_translation", "p_translation")
            )
            engine_difference = max(engine_difference, abs(state.photon_number(0) - reference.photon_number(0)))

            q_target, psi_target = resource.wavefunction()
            q_grid = np.linspace(q_target.min(), q_target.max(), 2000)
            target_probability = np.interp(q_grid, q_target, psi_target**2)
            fock_probability = reconstructed_probability(np.array(fock_input.amplitudes), q_grid)
            l1_distance = float(np.trapezoid(np.abs(target_probability - fock_probability), q_grid))

            rows.append(
                {
                    "label": params["label"],
                    "peak_width": params["peak_width"],
                    "envelope": params["envelope"],
                    "logical": logical,
                    "cutoff": DEFAULT_CUTOFF,
                    "captured_mass": captured_mass,
                    "q_translation_stabilizer_abs": abs(stabilizer_values["q_translation"]),
                    "p_translation_stabilizer_abs": abs(stabilizer_values["p_translation"]),
                    "target_vs_fock_probability_l1": l1_distance,
                    "mean_photon_number": state.photon_number(0),
                    "native_vs_piquasso_max_difference": engine_difference,
                    "min_retained_norm": min(state.retained_norms),
                }
            )
            if params["label"] == "default" and logical == 0:
                representative_data = (q_grid, target_probability, fock_probability, params)

        code = pg.GKPCode(peak_width=params["peak_width"], envelope=params["envelope"], cutoff=DEFAULT_CUTOFF)
        rows.append(
            {
                "label": params["label"], "peak_width": params["peak_width"], "envelope": params["envelope"],
                "logical": "overlap|0><1|", "cutoff": DEFAULT_CUTOFF, "captured_mass": None,
                "q_translation_stabilizer_abs": None, "p_translation_stabilizer_abs": None,
                "target_vs_fock_probability_l1": None,
                "mean_photon_number": None,
                "native_vs_piquasso_max_difference": None,
                "min_retained_norm": None,
                "logical_overlap_abs": abs(code.gram[0, 1]),
            }
        )

    cutoff_rows = []
    for cutoff in CUTOFF_SENSITIVITY:
        resource = pg.GKPResource(0, peak_width=0.4, envelope=0.4)
        fock_input, mass = resource.project(cutoff)
        identity = pg.Pattern(inputs=(0,)).append(pg.Output((0,))).validate()
        st = pg.simulate(identity, backend=MAIN_BACKEND, cutoff=cutoff, inputs={0: fock_input}, seed=0).state
        stab = pg.gkp.stabilizers(st)
        cutoff_rows.append({"cutoff": cutoff, "captured_mass": mass,
                            "q_translation_stabilizer_abs": abs(stab["q_translation"]),
                            "p_translation_stabilizer_abs": abs(stab["p_translation"]),
                            "mean_photon_number": st.photon_number(0), "backend": MAIN_BACKEND})

    failures = [r for r in rows if r["native_vs_piquasso_max_difference"] is not None
                and r["native_vs_piquasso_max_difference"] > ENGINE_TOLERANCE]
    common.save_result(rows, "R26_gkp_resource", extra={
        "main_backend": MAIN_BACKEND, "reference_backend": REFERENCE_BACKEND, "default_cutoff": DEFAULT_CUTOFF,
        "engine_tolerance": ENGINE_TOLERANCE, "no_fault_tolerance_claim": True})
    if failures:
        common.save_csv(failures, "R26_gkp_resource_FAILURES")
        raise AssertionError("native-fock and piquasso-fock disagree on GKP stabilizers; see FAILURES csv")
    common.save_csv(cutoff_rows, "R26_gkp_resource_cutoff_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    q_grid, target_probability, fock_probability, params = representative_data
    axes[0].plot(q_grid, target_probability, label="Target wavefunction |psi(q)|^2", color="#24677b")
    axes[0].plot(q_grid, fock_probability, "--", label=f"Fock reconstruction (c={DEFAULT_CUTOFF})", color="#c66d27")
    axes[0].set_xlabel("q")
    axes[0].set_ylabel("Probability density")
    axes[0].legend(fontsize=7)
    axes[0].set_title(f"GKP |0>_L, {params['label']} (width={params['peak_width']})")

    axes[1].semilogy(
        [r["cutoff"] for r in cutoff_rows], [max(1e-16, 1 - r["captured_mass"]) for r in cutoff_rows], "o-", color="#627a36"
    )
    axes[1].set_xlabel("Fock cutoff")
    axes[1].set_ylabel("1 - captured mass (circles)")
    twin = axes[1].twinx()
    twin.plot([r["cutoff"] for r in cutoff_rows], [r["q_translation_stabilizer_abs"] for r in cutoff_rows], "s--",
              color="#7a3e9d", label="|<S_q>| (native-fock)")
    twin.set_ylabel("|<S_q>| stabilizer (squares)")
    twin.legend(fontsize=7, loc="center right")
    twin.spines["right"].set_visible(True)
    axes[0].set_xlim(-12, 12)
    axes[1].set_title("Cutoff sensitivity, peak_width=envelope=0.4")
    fig.suptitle("GKP resource characterization (no fault-tolerance claim)")
    common.save_figure(fig, "R26_gkp_resource")
    plt.close(fig)
    numeric = [r for r in rows if r["captured_mass"] is not None]
    common.save_caption(
        "R26_gkp_resource",
        f"""
**R26: finite-energy GKP resources.** Left: q-quadrature probability density of the finite-energy square-lattice
GKP |0>_L (peak width = envelope = {params['peak_width']}): the target wavefunction of `GKPResource` versus its
projection onto Fock states below {DEFAULT_CUTOFF} photons, reconstructed from Hermite functions (L1 distance
{[r for r in numeric if r['label'] == 'default' and r['logical'] == 0][0]['target_vs_fock_probability_l1']:.2e}).
Right: discarded weight 1 - captured mass of the projection (circles) and the q-translation stabilizer
|<exp(-i sqrt(2 pi) p)>| on `native-fock` (squares) versus cutoff {CUTOFF_SENSITIVITY}. Across widths 0.3/0.4/0.6
and both codewords, |<S_q>| ranges {min(r['q_translation_stabilizer_abs'] for r in numeric):.4f} to
{max(r['q_translation_stabilizer_abs'] for r in numeric):.4f} and the captured mass is >=
{min(r['captured_mass'] for r in numeric):.6f}; `native-fock` and `piquasso-fock` agree on stabilizers and photon
number to {max(r['native_vs_piquasso_max_difference'] for r in numeric):.1e}. Finite-energy resources only: no
fault-tolerance, threshold or logical-error-rate claim follows from these numbers. Model: hbar = 2, lattice spacing
sqrt(2 pi). Source: `paper_experiments/08_gkp/26_gkp_resource.py`.
""",
    )

    numeric_rows = [r for r in rows if r["captured_mass"] is not None]
    common.print_summary(
        "R26 GKP resource characterization",
        cases=len(rows),
        min_captured_mass=min(r["captured_mass"] for r in numeric_rows),
        max_stabilizer_deviation_from_unity=max(1 - r["q_translation_stabilizer_abs"] for r in numeric_rows),
        max_native_vs_piquasso=max(r["native_vs_piquasso_max_difference"] for r in numeric_rows),
    )


if __name__ == "__main__":
    main()
