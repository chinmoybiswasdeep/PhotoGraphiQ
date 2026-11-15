import numpy as np
import pytest

from photographiq import Pattern, simulate
from photographiq.commands import Output
from photographiqml import GKPBridge, MuTA


def test_resource_projection_and_execution():
    bridge = GKPBridge(cutoff=24, grid_points=2049)
    diagnostics = bridge.diagnostics()
    assert all(0 < w <= 1.000001 for w in diagnostics["captured_weights"])
    assert not diagnostics["physical_muta_validated"]
    encoded = bridge.encode([1, 0])
    assert np.linalg.norm(encoded.amplitudes) == pytest.approx(1)
    pattern = Pattern(inputs=(0,)).append(Output((0,)))
    result = simulate(pattern, inputs={0: encoded}, backend="piquasso-fock", cutoff=24, seed=0)
    assert result.state is not None


def test_projection_cutoff_and_grid_refinement():
    small = GKPBridge(cutoff=12, grid_points=2049).diagnostics()
    large = GKPBridge(cutoff=32, grid_points=2049).diagnostics()
    refined = GKPBridge(cutoff=32, grid_points=4097).diagnostics()
    assert np.all(np.array(large["captured_weights"]) >= small["captured_weights"])
    assert np.allclose(large["captured_weights"], refined["captured_weights"], atol=1e-9)


def test_representation_boundary():
    model = MuTA(1, representation="gkp")
    with pytest.raises(NotImplementedError, match="logical XY"):
        model.run([1, 0])
    bridge = GKPBridge()
    target = bridge.logical_target(model, [1, 0])
    assert target.representation == "logical"
    with pytest.raises(NotImplementedError, match="protocol"):
        bridge.run(model)
