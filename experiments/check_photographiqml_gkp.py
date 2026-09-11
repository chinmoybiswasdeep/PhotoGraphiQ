"""Optional sibling-repository contract check; no ML import in the core package."""

import argparse
import sys
from pathlib import Path

import numpy as np

import photographiq as pg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--photographiqml",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "PhotoGraphiQML",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.photographiqml / "src"))
    from photographiqml.gkp import GKPBridge

    bridge = GKPBridge(cutoff=48)
    code = pg.GKPCode(cutoff=48)
    source = bridge.encode(np.array([1, 1]) / np.sqrt(2))
    np.testing.assert_allclose(source.amplitudes, code.plus().amplitudes, atol=1e-12)
    p = pg.Pattern(inputs=(0,)).measure(0, code.logical_measurement("XY", alpha=0), key="x")
    result = pg.simulate(
        p, inputs={0: source}, backend="piquasso-fock", cutoff=48, measurement_outcomes={"x": 0.2}
    )
    assert result.outcomes["x"].bit == 0
    try:
        bridge.run()
    except NotImplementedError:
        print(
            "Finite bridge resource -> physical signed-X primitive: passed. Full MuTA remains blocked."
        )
    else:
        raise AssertionError("Re-audit downstream physical execution: bridge behavior changed")


if __name__ == "__main__":
    main()
