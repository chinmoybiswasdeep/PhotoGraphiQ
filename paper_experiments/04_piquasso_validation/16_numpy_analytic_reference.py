"""R16: Independent NumPy analytic Gaussian reference.

Builds small symplectic target matrices directly with NumPy (not
``photographiq.gaussian``), applies them analytically to a coherent-state
input, and compares the result against three execution paths: PhotoGraphiQ's
NumPy Gaussian backend, PhotoGraphiQ's Piquasso Gaussian backend, and a raw
Piquasso program built independently of PhotoGraphiQ's backend adapters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import piquasso as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

HBAR = 2.0


# --------------------------------------------------------------------------- #
# Independent NumPy target matrices (not photographiq.gaussian)
# --------------------------------------------------------------------------- #


def target_rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def target_squeezing(r):
    return np.array([[np.exp(-r), 0.0], [0.0, np.exp(r)]])


def target_displacement_delta(q, p):
    return np.array([q, p])


def target_beamsplitter(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, -s, 0], [0, c, 0, -s], [s, 0, c, 0], [0, s, 0, c]])


def target_cz(weight):
    matrix = np.eye(4)
    matrix[1, 2] = weight
    matrix[3, 0] = weight
    return matrix


# --------------------------------------------------------------------------- #
# Raw Piquasso helper (independent of photographiq.backends.piquasso)
# --------------------------------------------------------------------------- #


def raw_piquasso(alphas, gate_builder, modes):
    with pq.Program() as program:
        pq.Q(*range(modes)) | pq.Vacuum()
        for i, alpha in enumerate(alphas):
            pq.Q(i) | pq.Displacement(r=abs(alpha), phi=np.angle(alpha))
        gate_builder()
    state = pq.GaussianSimulator(d=modes, config=pq.Config(hbar=HBAR)).execute(program).state
    return np.array(state.xpxp_mean_vector), np.array(state.xpxp_covariance_matrix) / 2


def coherent_mean_cov(alpha: complex):
    return np.array([2 * alpha.real, 2 * alpha.imag]), np.eye(2)


def run_case(name, params, modes, alphas, target_matrix, target_delta, pg_commands, raw_builder):
    means = [2 * a.real for a in alphas], [2 * a.imag for a in alphas]
    mean_in = np.empty(2 * modes)
    mean_in[0::2] = [2 * a.real for a in alphas]
    mean_in[1::2] = [2 * a.imag for a in alphas]
    cov_in = np.eye(2 * modes)

    analytic_mean = target_matrix @ mean_in + target_delta
    analytic_cov = target_matrix @ cov_in @ target_matrix.T

    inputs = {i: pg.GaussianInput.coherent(a) for i, a in enumerate(alphas)}
    rows = {}
    for backend in ("gaussian", "piquasso"):
        pattern = pg.Pattern(inputs=tuple(range(modes)))
        pattern.extend(pg_commands)
        pattern.append(pg.Output(tuple(range(modes))))
        result = pg.simulate(pattern, backend=backend, inputs=inputs, seed=0)
        rows[f"{backend}_mean_error"] = common.frobenius_error(result.state.mean, analytic_mean)
        rows[f"{backend}_covariance_error"] = common.frobenius_error(result.state.covariance, analytic_cov)

    raw_mean, raw_cov = raw_piquasso(alphas, raw_builder, modes)
    rows["raw_piquasso_mean_error"] = common.frobenius_error(raw_mean, analytic_mean)
    rows["raw_piquasso_covariance_error"] = common.frobenius_error(raw_cov, analytic_cov)

    return {"gate": name, "params": params, **rows}


def main():
    plt = common.setup_style()
    rows = []

    rows.append(
        run_case(
            "rotation", {"theta": 0.6}, 1, [0.4 + 0.2j],
            target_rotation(0.6), np.zeros(2), [pg.Rotate(0, 0.6)],
            lambda: (pq.Q(0) | pq.Phaseshifter(phi=0.6)),
        )
    )
    rows.append(
        run_case(
            "squeezing", {"r": 0.5}, 1, [0.3 - 0.1j],
            target_squeezing(0.5), np.zeros(2), [pg.Squeeze(0, 0.5)],
            lambda: (pq.Q(0) | pq.Squeezing(r=0.5)),
        )
    )
    rows.append(
        run_case(
            "displacement", {"q": 0.4, "p": -0.3}, 1, [0.1 + 0.1j],
            np.eye(2), target_displacement_delta(0.4, -0.3), [pg.Displace(0, q=0.4, p=-0.3)],
            lambda: (pq.Q(0) | pq.Displacement(r=abs(complex(0.4, -0.3) / 2), phi=np.angle(complex(0.4, -0.3) / 2))),
        )
    )
    rows.append(
        run_case(
            "beamsplitter", {"theta": 0.35}, 2, [0.3 + 0.1j, -0.2 + 0.2j],
            target_beamsplitter(0.35), np.zeros(4), [pg.BeamSplitter(0, 1, 0.35)],
            lambda: (pq.Q(0, 1) | pq.Beamsplitter(theta=0.35, phi=0.0)),
        )
    )
    rows.append(
        run_case(
            "cz", {"weight": 0.7}, 2, [0.2 + 0.3j, 0.1 - 0.4j],
            target_cz(0.7), np.zeros(4), [pg.Entangle(0, 1, 0.7)],
            lambda: (pq.Q(0, 1) | pq.ControlledZ(s=0.7)),
        )
    )
    compound_matrix = target_cz(0.4) @ np.block(
        [[target_rotation(0.3), np.zeros((2, 2))], [np.zeros((2, 2)), np.eye(2)]]
    ) @ np.block([[target_squeezing(0.2), np.zeros((2, 2))], [np.zeros((2, 2)), np.eye(2)]])
    rows.append(
        run_case(
            "compound(squeeze+rotate+cz)", {"r": 0.2, "angle": 0.3, "weight": 0.4}, 2,
            [0.2 + 0.0j, -0.1 + 0.2j], compound_matrix, np.zeros(4),
            [pg.Squeeze(0, 0.2), pg.Rotate(0, 0.3), pg.Entangle(0, 1, 0.4)],
            lambda: (
                pq.Q(0) | pq.Squeezing(r=0.2),
                pq.Q(0) | pq.Phaseshifter(phi=0.3),
                pq.Q(0, 1) | pq.ControlledZ(s=0.4),
            ),
        )
    )

    tolerance = 1e-9
    error_keys = [k for k in rows[0] if k.endswith("_error")]
    failures = [r for r in rows if any(r[k] > tolerance for k in error_keys)]
    if failures:
        common.save_csv(failures, "R16_numpy_analytic_reference_FAILURES")
        raise AssertionError("A backend disagreed with the independent analytic target; see FAILURES csv")

    common.save_result(rows, "R16_numpy_analytic_reference")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    gates = [r["gate"] for r in rows]
    x = np.arange(len(gates))
    width = 0.25
    ax.bar(x - width, [common.safe_log10(r["gaussian_covariance_error"]) for r in rows], width, label="NumPy Gaussian backend")
    ax.bar(x, [common.safe_log10(r["piquasso_covariance_error"]) for r in rows], width, label="Piquasso Gaussian backend")
    ax.bar(x + width, [common.safe_log10(r["raw_piquasso_covariance_error"]) for r in rows], width, label="Raw Piquasso")
    ax.set_xticks(x)
    ax.set_xticklabels(gates, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel(r"$\log_{10}$(covariance error vs. NumPy analytic target)")
    ax.legend(fontsize=7)
    common.save_figure(fig, "R16_numpy_analytic_reference")
    plt.close(fig)

    common.print_summary(
        "R16 independent NumPy analytic reference",
        cases=len(rows),
        max_error=max(max(r[k] for k in error_keys) for r in rows),
    )


if __name__ == "__main__":
    main()
