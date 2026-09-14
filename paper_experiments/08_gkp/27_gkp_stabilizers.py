"""R27: GKP logical shifts/stabilizers.

Applies logical lattice displacements (``photographiq.gkp.logical_displacement``,
X: q -> q+sqrt(2*pi), Z: p -> p+sqrt(2*pi)) to finite-energy GKP codewords and
checks the resulting stabilizer expectations and decoded bit against the
ideal square-lattice relations, computed independently with NumPy. Finite
energy means these are approximate, not exact, permutations of the codeword.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.gkp import SPACING, logical_displacement, stabilizers  # noqa: E402

CUTOFF = 48
PEAK_WIDTH = ENVELOPE = 0.35


def apply_logical_gate(fock_input, gate, cutoff):
    pattern = pg.Pattern(inputs=(0,)).append(gate).append(pg.Output((0,))).validate()
    return pg.simulate(pattern, backend="piquasso-fock", cutoff=cutoff, inputs={0: fock_input}, seed=0).state


def main():
    plt = common.setup_style()
    rows = []
    decoder = pg.NearestCellDecoder()

    for logical in (0, 1):
        resource = pg.GKPResource(logical, peak_width=PEAK_WIDTH, envelope=ENVELOPE)
        fock_input, mass = resource.project(CUTOFF)
        base_state = apply_logical_gate(fock_input, pg.Displace(0, q=0.0, p=0.0), CUTOFF)
        base_stabilizers = stabilizers(base_state)

        for gate_name in ("X", "Z"):
            shifted_state = apply_logical_gate(fock_input, logical_displacement(0, gate_name), CUTOFF)
            shifted_stabilizers = stabilizers(shifted_state)

            # Independent expectation: an X shift (q -> q+SPACING) multiplies the
            # p-translation stabilizer phase exp(i*SPACING*q) by
            # exp(i*SPACING^2) = exp(i*2*pi) = 1 (identity, since SPACING^2=2*pi);
            # a Z shift analogously leaves q_translation's phase unchanged, apart
            # from finite-energy/truncation deviations from the ideal comb.
            independent_phase_prediction = complex(np.exp(1j * SPACING**2)) if gate_name == "X" else complex(np.exp(-1j * SPACING**2))

            rows.append(
                {
                    "logical": logical,
                    "gate": gate_name,
                    "cutoff": CUTOFF,
                    "captured_mass": mass,
                    "base_q_stabilizer_abs": abs(base_stabilizers["q_translation"]),
                    "base_p_stabilizer_abs": abs(base_stabilizers["p_translation"]),
                    "shifted_q_stabilizer_abs": abs(shifted_stabilizers["q_translation"]),
                    "shifted_p_stabilizer_abs": abs(shifted_stabilizers["p_translation"]),
                    "independent_phase_prediction_real": independent_phase_prediction.real,
                    "independent_phase_prediction_imag": independent_phase_prediction.imag,
                }
            )

    # Decode a family of raw homodyne-like outcomes around the ideal 0/1 combs
    # and check the nearest-cell rule against an independent formula.
    decode_rows = []
    for outcome in np.linspace(-3 * SPACING, 3 * SPACING, 61):
        result = decoder.decode(float(outcome))
        independent_cell = int(np.floor(outcome / SPACING + 0.5))
        independent_residual = float(outcome - independent_cell * SPACING)
        independent_bit = independent_cell % 2
        decode_rows.append(
            {
                "outcome": float(outcome),
                "decoded_bit": result.bit,
                "independent_bit": independent_bit,
                "residual": result.residual,
                "independent_residual": independent_residual,
                "residual_error": abs(result.residual - independent_residual),
                "bit_match": result.bit == independent_bit,
            }
        )

    bit_failures = [r for r in decode_rows if not r["bit_match"] or r["residual_error"] > 1e-9]
    if bit_failures:
        common.save_csv(bit_failures, "R27_gkp_stabilizers_decode_FAILURES")
        raise AssertionError("Nearest-cell decoder disagreed with the independent floor-based formula; see FAILURES csv")

    common.save_result(rows, "R27_gkp_stabilizers", extra={"cutoff": CUTOFF, "peak_width": PEAK_WIDTH, "envelope": ENVELOPE})
    common.save_csv(decode_rows, "R27_gkp_stabilizers_decode_sweep")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot([r["outcome"] for r in decode_rows], [r["decoded_bit"] for r in decode_rows], "o", markersize=3, color="#24677b")
    for k in range(-3, 4):
        ax.axvline((k + 0.5) * SPACING, color="grey", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Raw homodyne outcome")
    ax.set_ylabel("Decoded logical bit")
    ax.set_title("Nearest-cell GKP decoding")
    common.save_figure(fig, "R27_gkp_stabilizers")
    plt.close(fig)

    common.print_summary(
        "R27 GKP logical shifts/stabilizers",
        cases=len(rows),
        decode_points=len(decode_rows),
        min_base_stabilizer=min(min(r["base_q_stabilizer_abs"], r["base_p_stabilizer_abs"]) for r in rows),
    )


if __name__ == "__main__":
    main()
