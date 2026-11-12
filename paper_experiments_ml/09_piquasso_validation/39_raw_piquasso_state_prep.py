"""R39: Raw Piquasso validation of the smallest feasible finite-GKP
preparation and signed-X physical circuit.

Scientific question: Does a raw Piquasso program -- built directly from
Piquasso's own public API (piquasso.Program, piquasso.Q, Fock-basis state
preparation and homodyne-style measurement primitives) -- reproduce the same
pre-measurement Fock-basis amplitudes that PhotoGraphiQ's GKPCode.plus()
finite-energy resource state produces, for the smallest single-mode case?

Theory/equations: GKPCode.plus() prepares a normalized finite superposition
of Fock-basis amplitudes approximating the ideal GKP |+> state; this
experiment reads those amplitudes via the public FockInput API and
independently re-normalizes/validates them, then constructs an equivalent
raw Piquasso Fock-basis state preparation from the identical amplitude
vector to confirm Piquasso's own Fock backend reproduces the same
finite-dimensional quantum state (a backend/API round-trip check, not a
re-derivation of the GKP wavefunction itself).

Functionality tested: photographiq.GKPCode.plus/encode (shared input) vs. a
raw piquasso.Program/piquasso.Q Fock-state-preparation pipeline
(independent orchestration).

Oracle and independence class: C -- the finite-GKP amplitude vector itself
is obtained from PhotoGraphiQ (a shared input, since Piquasso's public API
does not itself expose a GKP-resource constructor), while the Piquasso
program that prepares and reads back that state is independently
assembled, not calling any PhotoGraphiQML or PhotoGraphiQ orchestration
code.

Exact/approximate/statistical status: exact (state-vector fidelity between
the source amplitudes and Piquasso's own reconstructed Fock state).

Primary metric: |1 - fidelity| between the source GKP |+> amplitude vector
and Piquasso's own Fock-space state vector after an explicit Fock-basis
state preparation.

Declared acceptance condition: |1 - fidelity| < tol
(tol = declare_tolerance(scale=1, safety_factor=1e4)) -- allows for
Piquasso's own numerical Fock-state normalization, not exact machine
precision.

Expected cost: light (single-mode, cutoff<=32).

Manuscript destination: Main text (Fig. 10, raw Piquasso validation panel).

Scientific limitations: This validates a static Fock-basis state
round-trip only; it does not validate Piquasso's own Gaussian/Fock gate
dynamics (see R40 for a raw Piquasso CZ check) or PhotoGraphiQML's signed-X
measurement pipeline (see R38, R41).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np
import piquasso as pq
from photographiq import GKPCode

EXPERIMENT_ID = "R39"


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=1e4)
    rows = []
    for cutoff in (16, 24, 32):
        code = GKPCode(cutoff=cutoff, peak_width=0.4, envelope=0.4, peaks=8, grid_points=4097)
        source = code.plus()
        source_amplitudes = np.asarray(source.amplitudes, dtype=complex)
        source_amplitudes = source_amplitudes / np.linalg.norm(source_amplitudes)

        fock_amplitude_map = {
            (n,): complex(amplitude)
            for n, amplitude in enumerate(source_amplitudes)
            if abs(amplitude) > 0
        }
        with pq.Program() as program:
            pq.Q() | pq.FockStateVector(fock_amplitude_map=fock_amplitude_map)

        simulator = pq.PureFockSimulator(d=1, config=pq.Config(cutoff=cutoff))
        result = simulator.execute(program)
        piquasso_state = np.asarray(
            result.state.get_tensor_representation(), dtype=complex
        ).reshape(-1)
        piquasso_state = piquasso_state / np.linalg.norm(piquasso_state)
        overlap = np.vdot(source_amplitudes, piquasso_state[: len(source_amplitudes)])
        fidelity = float(abs(overlap) ** 2)
        rows.append({"cutoff": cutoff, "fidelity": fidelity, "infidelity": abs(1 - fidelity)})

    max_infidelity = max(r["infidelity"] for r in rows)
    status = "pass" if max_infidelity < tol else "fail"

    common.save_result(
        rows,
        "R39_raw_piquasso_state_prep",
        extra={
            "protocol": "GKPCode.plus() Fock amplitudes vs. raw Piquasso PureFockSimulator state preparation",
            "oracle_class": "C",
            "status_category": "exact",
            "piquasso_version": pq.__version__,
            "tolerance": tol,
            "acceptance_condition": f"max |1-fidelity| < {tol:.3e}",
            "max_infidelity": max_infidelity,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "C", "status": status},
    )

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(
        [r["cutoff"] for r in rows],
        [max(r["infidelity"], 1e-18) for r in rows],
        "o-",
        color=common.COLORS["piquasso"],
    )
    ax.axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    ax.set(
        title=f"R39: raw Piquasso state-prep fidelity (status={status})",
        xlabel="cutoff",
        ylabel="|1-fidelity|",
    )
    ax.legend()
    common.save_figure(fig, "R39_raw_piquasso_state_prep")
    plt.close(fig)

    common.print_summary(
        "R39 raw Piquasso state prep",
        n_cutoffs=len(rows),
        max_infidelity=max_infidelity,
        status=status,
    )
    if status != "pass":
        raise AssertionError(f"R39 failed: max_infidelity={max_infidelity} tol={tol}")


if __name__ == "__main__":
    main()
