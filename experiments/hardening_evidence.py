"""Independent validation evidence for the release report (requires source tests)."""

import argparse
import json
from pathlib import Path

import numpy as np

import photographiq as pg
from tests.non_gaussian.test_injection import decomposition_observation, reference_injection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    density = np.exp(-(0.4**2) / (2 * (1 + np.exp(0.4)))) / np.sqrt(2 * np.pi * (1 + np.exp(0.4)))
    for cutoff in (36, 48, 64, 96):
        result = pg.simulate(
            pg.non_gaussian.cubic_injection(0.3, 0.2),
            backend="piquasso-fock",
            cutoff=cutoff,
            measurement_outcomes={"m": 0.4},
        )
        oracle = reference_injection(cutoff, 0.4, gamma=0.3, r=0.2)
        rows.append(
            {
                "cutoff": cutoff,
                "density_error": abs(result.measurement_statistics["m"]["value"] - density),
                "infidelity": float(
                    max(0, 1 - abs(np.vdot(oracle, result.state.state_vector)) ** 2)
                ),
                "minimum_retained_norm": min(result.state.retained_norms),
            }
        )
    p = pg.Pattern().extend([pg.Prepare(0, 0), pg.CubicPhase(0, 0.65)])
    moments = pg.cutoff_convergence(p, [24, 48, 96], high_order_moments=True)
    report = {
        "decompositions": [decomposition_observation(c) for c in (12, 18, 24, 32, 48)],
        "injection": rows,
        "strong_cubic_moments": moments.rows,
        "oracle_quadrature_vector_change": float(
            np.linalg.norm(
                reference_injection(96, 0.4, gamma=0.3, r=0.2, eps=1e-9, bound=12)
                - reference_injection(96, 0.4, gamma=0.3, r=0.2, eps=1e-12, bound=16)
            )
        ),
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
