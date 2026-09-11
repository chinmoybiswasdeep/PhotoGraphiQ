import numpy as np

import photographiq as pg
from tests.gkp.multimode.test_cz_reference import run_case


def test_joint_fidelity_and_leakage_convergence_are_distinct():
    rows = []
    for cutoff in (24, 40, 64):
        code, result, _, _ = run_case(cutoff)
        B = np.array(
            [
                [code._basis[i, a] * code._basis[j, b] for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
                for i, j in result.state.basis
            ]
        )
        v = result.state.state_vector
        overlap = B.conj().T @ v
        leakage = 1 - np.vdot(overlap, np.linalg.solve(B.conj().T @ B, overlap)).real
        target = B @ np.array([1, 1, 1, -1])
        fidelity = abs(np.vdot(target / np.linalg.norm(target), v)) ** 2
        probabilities = pg.multimode_readout(result.state, {0: "X", 1: "Z"})
        joint = np.array(list(probabilities["joint_probabilities"].values()))
        marginals = np.array(list(probabilities["marginal_probabilities"].values()))
        rows.append(np.r_[joint, marginals.ravel(), fidelity, leakage])
    assert max(abs(rows[-1] - rows[-2])) < 1e-4
    assert max(abs(rows[-1] - rows[-2])) < max(abs(rows[1] - rows[0]))
    assert 0.4 < rows[-1][-1] < 0.5  # converged physical distortion, not numerical leakage
    assert 0.5 < rows[-1][-2] < 0.6
    # Entanglement-induced correlation is visible despite nearly balanced marginals.
    assert rows[-1][0] > 0.4
    assert abs(rows[-1][4] * rows[-1][6] - 0.25) < 1e-4
