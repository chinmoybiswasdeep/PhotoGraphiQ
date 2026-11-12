"""R40: Raw Piquasso validation of the finite physical CZ action on selected
finite-GKP basis preparations, including cutoff dependence.

Scientific question: Does directly calling Piquasso's own
pq.GaussianTransform gate (with the exact passive/active matrices
PhotoGraphiQ's Fock backend uses for a unit-weight controlled-Z,
photographiq/backends/fock.py's entangle method) on a hand-prepared joint
two-mode finite-GKP basis state reproduce the same output amplitudes as
going through PhotoGraphiQ's own GKPCode.logical_cz command inside a full
Pattern/pg.simulate call, for all four |mu>|nu> basis combinations and
several cutoffs?

Theory/equations: docs/research/muta-mapping.md: physical CZ of unit weight
exp(i*q1*q2/2) gives (-1)^(mu*nu) on ideal lattice sites; the exact finite-
codeword Fock-basis matrices are
passive=[[1, i/2],[i/2, 1]], active=[[0, i/2],[i/2, 0]] (weight=1).

Functionality tested: photographiq.backends.fock.PiquassoFockBackend.entangle
(exercised indirectly via GKPCode.logical_cz + Pattern + pg.simulate) vs. a
raw piquasso.GaussianTransform call assembled directly in this script.

Oracle and independence class: C -- the finite-GKP codeword amplitude
vectors are a shared input (from PhotoGraphiQ's GKPCode, since raw Piquasso
has no native GKP-resource constructor), while the CZ *gate application* is
independently invoked twice: once through PhotoGraphiQ's full Pattern/
backend-dispatch stack, once by calling piquasso.GaussianTransform directly
with hand-matched parameters, bypassing that stack entirely.

Exact/approximate/statistical status: exact (both paths simulate the
identical finite-dimensional Fock-space gate on the same input state; any
difference indicates a stack-dispatch bug, not physical/sampling noise).

Primary metric: max amplitude-vector Frobenius error between the two paths,
over 4 basis combinations x 3 cutoffs.

Declared acceptance condition: max error < tol
(tol = declare_tolerance(scale=1, safety_factor=1e4)).

Expected cost: light-to-moderate (two-mode Fock simulations, cutoff<=32).

Manuscript destination: Main text (Fig. 10, raw Piquasso CZ validation).

Scientific limitations: Validates the CZ gate-dispatch pipeline on finite
GKP basis states only; it does not itself certify the (-1)^(mu*nu) ideal-
lattice sign pattern for the finite (non-orthogonal, non-ideal) codewords
used here (see docs/research/muta-mapping.md's own caveat that this
identity does not extend exactly to finite peaks/envelopes).
"""

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np
import photographiq as pg_
import piquasso as pq
from photographiq import GKPCode

EXPERIMENT_ID = "R40"


def raw_piquasso_cz(amp_mu, amp_nu, cutoff):
    fock_amplitude_map = {
        (n, m): complex(amp_mu[n] * amp_nu[m])
        for n in range(len(amp_mu))
        for m in range(len(amp_nu))
        if n + m < cutoff and abs(amp_mu[n] * amp_nu[m]) > 0
    }
    passive = np.array([[1, 0.5j], [0.5j, 1]])
    active = np.array([[0, 0.5j], [0.5j, 0]])
    with pq.Program() as program:
        pq.Q() | pq.FockStateVector(fock_amplitude_map=fock_amplitude_map)
        pq.Q(0, 1) | pq.GaussianTransform(passive=passive, active=active)
    result = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=cutoff)).execute(program)
    vector = np.asarray(result.state.get_tensor_representation(), dtype=complex).reshape(
        cutoff, cutoff
    )
    return vector / np.linalg.norm(vector)


def production_cz(code, amp_mu_input, amp_nu_input, cutoff):
    pattern = pg_.Pattern(inputs=[0, 1])
    pattern.append(code.logical_cz(0, 1))
    pattern.append(pg_.Output((0, 1)))
    pattern.validate()
    result = pg_.simulate(
        pattern,
        seed=0,
        inputs={0: amp_mu_input, 1: amp_nu_input},
        backend="piquasso-fock",
        cutoff=cutoff,
    )
    # .native is the raw underlying piquasso.PureFockState (photographiq's own
    # .state_vector uses a compact total-photon-truncated basis instead).
    vector = np.asarray(result.state.native.get_tensor_representation(), dtype=complex).reshape(
        cutoff, cutoff
    )
    return vector / np.linalg.norm(vector)


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=1e4)
    rows = []
    # Two-mode CZ on pure |mu>|nu> basis codewords needs a larger cutoff
    # margin than single-mode prep (R39) or superposition inputs; cutoff=24
    # still tripped PhotoGraphiQ's retained-norm guard (0.9989 < 0.999).
    for cutoff in (48, 64, 80):
        code = GKPCode(cutoff=cutoff, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
        basis = {0: code.zero(), 1: code.one()}
        amps = {
            0: np.asarray(basis[0].amplitudes, dtype=complex),
            1: np.asarray(basis[1].amplitudes, dtype=complex),
        }
        for mu, nu in product((0, 1), repeat=2):
            raw_vector = raw_piquasso_cz(amps[mu], amps[nu], cutoff)
            production_vector = production_cz(code, basis[mu], basis[nu], cutoff)
            overlap = np.vdot(raw_vector.ravel(), production_vector.ravel())
            fidelity = float(abs(overlap) ** 2)
            error = common.frobenius_error(raw_vector, production_vector)
            rows.append(
                {
                    "cutoff": cutoff,
                    "mu": mu,
                    "nu": nu,
                    "fidelity": fidelity,
                    "amplitude_error": error,
                }
            )

    max_error = max(r["amplitude_error"] for r in rows)
    status = "pass" if max_error < tol else "fail"

    common.save_result(
        rows,
        "R40_raw_piquasso_cz",
        extra={
            "protocol": "Raw piquasso.GaussianTransform CZ vs. GKPCode.logical_cz through full Pattern/pg.simulate",
            "oracle_class": "C",
            "status_category": "exact",
            "piquasso_version": pq.__version__,
            "tolerance": tol,
            "acceptance_condition": f"max amplitude-vector error < {tol:.3e}",
            "max_error": max_error,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "C", "status": status},
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"c{r['cutoff']}({r['mu']},{r['nu']})" for r in rows]
    colors = [
        common.COLORS["piquasso"] if r["amplitude_error"] < tol else common.COLORS["photographiqml"]
        for r in rows
    ]
    ax.scatter(range(len(rows)), [max(r["amplitude_error"], 1e-18) for r in rows], c=colors)
    ax.set_yscale("log")
    ax.axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.set(
        title=f"R40: raw Piquasso CZ vs. production dispatch (status={status})",
        ylabel="amplitude error",
    )
    ax.legend()
    common.save_figure(fig, "R40_raw_piquasso_cz")
    plt.close(fig)

    common.print_summary(
        "R40 raw Piquasso CZ", n_cases=len(rows), max_error=max_error, status=status
    )
    if status != "pass":
        raise AssertionError(f"R40 failed: max_error={max_error} tol={tol}")


if __name__ == "__main__":
    main()
