"""Reproducible finite GKP metrics; independent oracle lives in the test suite."""

import argparse
import json
from pathlib import Path

import numpy as np

import photographiq as pg
from photographiq.backends.fock import PiquassoFockBackend
from tests.gkp.multimode.test_cz_reference import run_case


def evidence():
    report = {
        "model": "finite-energy-gkp",
        "physical_muta_validated": False,
        "supported_readout_bases": ["X", "Z"],
        "single_mode": [],
        "two_mode": [],
        "studies": {},
    }
    for cutoff in (24, 48, 80):
        code = pg.GKPCode(0.55, 0.5, cutoff, 6, 2049)
        for coefficients in ((1, 0), (0, 1), (1, 1), (1, -1)):
            source = code.encode(*coefficients)
            for gate in ("X", "Z", "H", "S"):
                logical = {
                    "X": np.array([[0, 1], [1, 0]]),
                    "Z": np.diag([1, -1]),
                    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
                    "S": np.diag([1, 1j]),
                }[gate]
                target = code.encode(*(logical @ coefficients))
                result = pg.simulate(
                    pg.Pattern(inputs=(0,)).append(code.logical_gate(0, gate)),
                    inputs={0: source},
                    backend=PiquassoFockBackend(cutoff, norm_tolerance=0.1),
                )
                diagnostics = code.diagnostics(result.state)
                report["single_mode"].append(
                    {
                        "cutoff": cutoff,
                        "input": coefficients,
                        "gate": gate,
                        "target_fidelity": float(
                            abs(np.vdot(target.amplitudes, result.state.state_vector)) ** 2
                        ),
                        "code_subspace_leakage": diagnostics["code_subspace_leakage"],
                        "stabilizers": {
                            k: [z.real, z.imag] for k, z in diagnostics["stabilizers"].items()
                        },
                    }
                )
    for cutoff in (24, 40, 64):
        code, result, reference, mass = run_case(cutoff)
        v = result.state.state_vector
        B = np.array(
            [
                [code._basis[i, a] * code._basis[j, b] for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
                for i, j in result.state.basis
            ]
        )
        gram = B.conj().T @ B
        overlap = B.conj().T @ v
        weight = float(np.vdot(overlap, np.linalg.solve(gram, overlap)).real)
        target = B @ np.array([1, 1, 1, -1])
        target /= np.linalg.norm(target)
        readout = pg.multimode_readout(result.state, {0: "X", 1: "Z"})
        report["two_mode"].append(
            {
                "cutoff": cutoff,
                "preparation_mass": mass,
                "independent_reference_infidelity": float(1 - abs(np.vdot(reference, v)) ** 2),
                "target_fidelity": float(abs(np.vdot(target, v)) ** 2),
                "code_subspace_leakage": 1 - weight,
                "joint_probabilities": {
                    str(k): p for k, p in readout["joint_probabilities"].items()
                },
                "marginal_probabilities": readout["marginal_probabilities"],
                "stabilizers": {
                    str(n): {
                        k: [z.real, z.imag] for k, z in pg.gkp.stabilizers(result.state, n).items()
                    }
                    for n in (0, 1)
                },
            }
        )
    code = pg.GKPCode(0.55, 0.5, 48, 6, 2049)
    for axis, values in (
        ("cutoff", (24, 48, 80)),
        ("grid_points", (2049, 4097)),
        ("peaks", (4, 6)),
        ("peak_width", (0.4, 0.55, 0.7)),
        ("envelope", (0.35, 0.5, 0.6)),
    ):
        report["studies"][axis] = pg.measurement_convergence(
            code, values, axis=axis, basis="X", coefficients=(1, -1)
        ).rows
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="validation/gkp-logical/evidence.json")
    args = parser.parse_args()
    report = evidence()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)
