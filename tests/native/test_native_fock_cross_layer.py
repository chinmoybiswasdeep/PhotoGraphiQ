"""PhotoGraphiQML physical GKP lowering executed on the native Fock engine.

The engine is swapped through the simulator's backend factory, so this test keeps
config.backend="piquasso-fock". Selecting backend="native-fock" in GKPPhysicalConfig is
tested in tests/ml/cross_layer/test_native_backend.py. Every gate of the lowered pattern (GKP resources, CZ
wires, homodyne readout) then runs natively; results must match Piquasso up to the
truncation difference of the CZ implementation (exact P U P vs Euler factors).
"""

import pytest

from photographiq import native

try:
    native.require("fock")
except ImportError as exc:  # pragma: no cover - exercised only without the extension
    pytest.skip(f"native fock component unavailable: {exc}", allow_module_level=True)

pytest.importorskip("piquasso")

import photographiq.simulator as simulator  # noqa: E402
from photographiq.backends.native_fock import NativeFockBackend  # noqa: E402
from photographiq.fock_analysis import fidelity  # noqa: E402
from photographiqml import GKPPhysicalConfig, PhysicalMuTA  # noqa: E402


def _run(monkeypatch, wires, cutoff, use_native):
    config = GKPPhysicalConfig(
        cutoff=cutoff, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025
    )
    model = PhysicalMuTA(wires, physical_config=config)
    original = simulator._backend
    if use_native:
        monkeypatch.setattr(
            simulator,
            "_backend",
            lambda backend, c: (
                NativeFockBackend(cutoff=c) if backend == "piquasso-fock" else original(backend, c)
            ),
        )
    bits = [1] + [0] * (2**wires - 1)
    result = model.run(
        bits,
        mode="physical-conditional",
        analog_outcomes=dict.fromkeys(model.measurement_order, 0.0),
    )
    monkeypatch.setattr(simulator, "_backend", original)
    return result


def _differences(reference, result):
    infidelity = 1 - fidelity(result.physical_result.state, reference.physical_result.state)
    probability = max(
        abs(result.decoded_joint_probabilities[k] - v)
        for k, v in reference.decoded_joint_probabilities.items()
    )
    return infidelity, probability


# Observed (1 - F, max |dP|): 1 wire c=24 (1.0e-7, 4.6e-5), c=32 (3.9e-9, 3.6e-6),
# c=40 (9.5e-11, 7.6e-7); 2 wires c=24 (1.9e-7, 7.3e-5), c=32 (7.8e-9, 1.0e-5). Both engines
# converge to the same limit at the same rate (vs native c=48 at c=40: Piquasso 6.5e-10,
# native 6.2e-10). Tolerances are ten times the observed differences.
@pytest.mark.parametrize(
    "wires,cutoff,infidelity_tol,probability_tol", [(1, 32, 4e-8, 4e-5), (2, 24, 2e-6, 7e-4)]
)
def test_physical_muta_conditional_run_matches_piquasso(
    monkeypatch, wires, cutoff, infidelity_tol, probability_tol
):
    reference = _run(monkeypatch, wires, cutoff, use_native=False)
    result = _run(monkeypatch, wires, cutoff, use_native=True)
    assert result.physical_result.backend == "NativeFockBackend"
    assert reference.physical_result.backend == "PiquassoFockBackend"
    infidelity, probability = _differences(reference, result)
    assert infidelity < infidelity_tol and probability < probability_tol
    assert sum(result.decoded_joint_probabilities.values()) == pytest.approx(1, abs=1e-12)


def test_physical_muta_difference_vanishes_with_cutoff(monkeypatch):
    low = _differences(_run(monkeypatch, 1, 24, False), _run(monkeypatch, 1, 24, True))
    high = _differences(_run(monkeypatch, 1, 32, False), _run(monkeypatch, 1, 32, True))
    assert high[0] < low[0] / 10 and high[1] < low[1] / 5
