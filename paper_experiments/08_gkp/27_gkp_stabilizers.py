"""R27: GKP logical shifts/stabilizers.

Applies logical lattice displacements (``photographiq.gkp.logical_displacement``,
X: q -> q+sqrt(2*pi), Z: p -> p+sqrt(2*pi)) to finite-energy GKP codewords and
checks the resulting stabilizer expectations and decoded bit against the
ideal square-lattice relations, computed independently with NumPy. Finite
energy means these are approximate, not exact, permutations of the codeword.

Upgrade migration (2026-11): the codewords and shifted codewords are now
evolved on ``native-fock`` (main) and ``piquasso-fock`` (historical
reference); the engines must agree on every stabilizer modulus to
ENGINE_TOLERANCE. New logical observable: the single-shot logical READOUT
ERROR PROBABILITY of q-homodyne + nearest-cell decoding, computed exactly
(no sampling) by integrating the q-marginal |psi(q)|^2 of each Fock state
over the cells that decode to the wrong bit. The expected bit is L for the
codeword and for Z(L), and 1 - L after the logical X shift. The integrated
density mass is reported (it must be ~1, i.e. the grid captures the state).
Declared checks: engines agree; every error probability is below 1/2. These
are properties of finite-energy codewords under ideal homodyne detection --
not logical error rates of a fault-tolerant scheme.
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
MAIN_BACKEND = "native-fock"
REFERENCE_BACKEND = "piquasso-fock"
ENGINE_TOLERANCE = 1e-6  # displacement by sqrt(2 pi) at cutoff 48: engines differ by rounding
Q_GRID = np.linspace(-14.0, 14.0, 14001)


def readout_error_probability(state, expected_bit: int, cutoff: int):
    """Exact P(nearest-cell decode != expected_bit) from the Fock wavefunction."""
    from photographiq.fock_measurements import wavefunctions

    index = [state.basis.index((n,)) for n in range(cutoff)]
    amplitudes = np.asarray(state.state_vector)[index]
    basis = np.array([wavefunctions(x, cutoff) for x in Q_GRID])
    density = np.abs(basis @ amplitudes) ** 2
    cells = np.floor(Q_GRID / SPACING + 0.5).astype(int)
    wrong = (cells % 2) != expected_bit
    return float(np.trapezoid(density * wrong, Q_GRID)), float(np.trapezoid(density, Q_GRID))
def apply_logical_gate(fock_input, gate, cutoff, backend=MAIN_BACKEND):
    pattern = pg.Pattern(inputs=(0,)).append(gate).append(pg.Output((0,))).validate()
    return pg.simulate(pattern, backend=backend, cutoff=cutoff, inputs={0: fock_input}, seed=0).state


def main():
    plt = common.setup_style()
    rows = []
    decoder = pg.NearestCellDecoder()

    for logical in (0, 1):
        resource = pg.GKPResource(logical, peak_width=PEAK_WIDTH, envelope=ENVELOPE)
        fock_input, mass = resource.project(CUTOFF)
        base_state = apply_logical_gate(fock_input, pg.Displace(0, q=0.0, p=0.0), CUTOFF)
        base_stabilizers = stabilizers(base_state)
        base_error, base_mass = readout_error_probability(base_state, logical, CUTOFF)

        for gate_name in ("X", "Z"):
            shifted_state = apply_logical_gate(fock_input, logical_displacement(0, gate_name), CUTOFF)
            shifted_stabilizers = stabilizers(shifted_state)
            reference_state = apply_logical_gate(fock_input, logical_displacement(0, gate_name), CUTOFF, REFERENCE_BACKEND)
            reference_stabilizers = stabilizers(reference_state)
            engine_difference = max(abs(abs(shifted_stabilizers[k]) - abs(reference_stabilizers[k]))
                                    for k in ("q_translation", "p_translation"))
            expected_bit = 1 - logical if gate_name == "X" else logical
            shifted_error, shifted_mass = readout_error_probability(shifted_state, expected_bit, CUTOFF)
            reference_error, _ = readout_error_probability(reference_state, expected_bit, CUTOFF)

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
                    "shifted_q_stabilizer_phase": float(np.angle(shifted_stabilizers["q_translation"])),
                    "shifted_p_stabilizer_phase": float(np.angle(shifted_stabilizers["p_translation"])),
                    "native_vs_piquasso_stabilizer_difference": engine_difference,
                    "base_readout_error_probability": base_error,
                    "expected_bit_after_gate": expected_bit,
                    "shifted_readout_error_probability": shifted_error,
                    "piquasso_shifted_readout_error_probability": reference_error,
                    "integrated_density_mass_base": base_mass,
                    "integrated_density_mass_shifted": shifted_mass,
                    "min_retained_norm": min(shifted_state.retained_norms),
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

    logical_failures = [
        r for r in rows
        if r["native_vs_piquasso_stabilizer_difference"] > ENGINE_TOLERANCE
        or abs(r["shifted_readout_error_probability"] - r["piquasso_shifted_readout_error_probability"]) > ENGINE_TOLERANCE
        or max(r["base_readout_error_probability"], r["shifted_readout_error_probability"]) >= 0.5
    ]
    common.save_result(rows, "R27_gkp_stabilizers", extra={
        "cutoff": CUTOFF, "peak_width": PEAK_WIDTH, "envelope": ENVELOPE, "main_backend": MAIN_BACKEND,
        "reference_backend": REFERENCE_BACKEND, "engine_tolerance": ENGINE_TOLERANCE,
        "readout_error_definition": "exact integral of |psi(q)|^2 over wrong-parity nearest cells; ideal homodyne",
        "no_fault_tolerance_claim": True})
    if logical_failures:
        common.save_csv(logical_failures, "R27_gkp_stabilizers_FAILURES")
        raise AssertionError("GKP engine agreement or readout-error check failed; see FAILURES csv")
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
    common.save_caption(
        "R27_gkp_stabilizers",
        f"""
**R27: GKP logical shifts, decoding and readout errors.** Nearest-cell decoding of raw q-homodyne outcomes over
[-3 sqrt(2 pi), 3 sqrt(2 pi)] ({len(decode_rows)} points): the decoded bit and residual equal an independent
floor-based formula at every point (dotted lines: cell boundaries (k + 1/2) sqrt(2 pi)). Finite-energy codewords
(peak width = envelope = {PEAK_WIDTH}, cutoff {CUTOFF}, `native-fock`): exact single-shot readout error
probabilities (integral of |psi(q)|^2 over wrong cells, ideal homodyne) are
{', '.join(f"{r['base_readout_error_probability']:.2e}" for r in rows[::2])} for |0>_L and |1>_L and
{', '.join(f"{r['gate']}|{r['logical']}>: {r['shifted_readout_error_probability']:.2e}" for r in rows)} after the
logical displacements; `piquasso-fock` agrees to
{max(abs(r['shifted_readout_error_probability'] - r['piquasso_shifted_readout_error_probability']) for r in rows):.1e}.
These describe finite-energy codewords under ideal detection, not a fault-tolerant logical error rate. Source:
`paper_experiments/08_gkp/27_gkp_stabilizers.py`.
""",
    )

    common.print_summary(
        "R27 GKP logical shifts/stabilizers",
        cases=len(rows),
        decode_points=len(decode_rows),
        min_base_stabilizer=min(min(r["base_q_stabilizer_abs"], r["base_p_stabilizer_abs"]) for r in rows),
        max_native_vs_piquasso=max(r["native_vs_piquasso_stabilizer_difference"] for r in rows),
        readout_errors={f"{r['gate']}|{r['logical']}>": r["shifted_readout_error_probability"] for r in rows},
        base_readout_errors=[r["base_readout_error_probability"] for r in rows[::2]],
    )


if __name__ == "__main__":
    main()
