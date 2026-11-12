"""Backend registry, name resolution, planner and the CUDA-Q probe (adapter, not backend)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import photographiq as pg
from photographiq import registry
from photographiq.backends.cudaq_dynamics import cudaq_capabilities, cudaq_probe
from photographiq.backends.fock import PiquassoFockBackend
from photographiq.backends.gaussian import GaussianBackend
from photographiq.backends.mixed_fock import MixedFockBackend
from photographiq.backends.piquasso import PiquassoBackend
from photographiq.capabilities import preflight, required_features
from photographiq.simulator import _backend

AVAILABLE = set(registry.available_backends())


def _cheapest(*names):
    return next(name for name in names if name in AVAILABLE)


# --- registry entries ---------------------------------------------------------------


@pytest.mark.parametrize("name", list(registry.BACKENDS))
def test_entry_resolves_or_reports_reason(name):
    describe = registry.describe_backend(name)
    assert describe["name"] == name
    if describe["available"]:
        assert describe["unavailable_reason"] is None
        cls = registry.BACKENDS[name].load()
        assert describe["capabilities"] == sorted(cls.capabilities)
        engine = registry.create_backend(name, cutoff=4)
        assert type(engine) is cls
    else:
        assert isinstance(describe["unavailable_reason"], str) and describe["unavailable_reason"]
        with pytest.raises((ImportError, NotImplementedError)):
            registry.create_backend(name, cutoff=4)


@pytest.mark.parametrize(
    "name, cls",
    [
        ("gaussian", GaussianBackend),
        ("piquasso", PiquassoBackend),
        ("piquasso-fock", PiquassoFockBackend),
        ("piquasso-mixed-fock", MixedFockBackend),
    ],
)
def test_capabilities_are_read_from_class(name, cls):
    assert registry.BACKENDS[name].capabilities == cls.capabilities
    assert registry.describe_backend(name)["capabilities"] == sorted(cls.capabilities)


def test_fock_entries_declare_total_photon_truncation():
    for name, info in registry.BACKENDS.items():
        if info.representation in ("pure-fock", "mixed-fock"):
            assert info.uses_cutoff
            assert any("total photon number" in a for a in info.approximations), name
    assert not registry.BACKENDS["gaussian"].approximations


def test_registry_import_is_lazy():
    code = (
        "import sys, photographiq.registry as r; r.BACKENDS['piquasso'].unavailable_reason();"
        "print('piquasso' in sys.modules, 'cudaq' in sys.modules, 'jax' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False", "False"]


# --- simulator name resolution keeps existing behaviour ------------------------------


def test_backend_names_resolve_as_before():
    assert type(_backend("gaussian", None)) is GaussianBackend
    assert type(_backend("piquasso", 7)) is PiquassoBackend
    fock = _backend("piquasso-fock", 7)
    mixed = _backend("piquasso-mixed-fock", 5)
    assert type(fock) is PiquassoFockBackend and fock.cutoff == 7
    assert type(mixed) is MixedFockBackend and mixed.cutoff == 5
    engine = GaussianBackend()
    assert _backend(engine, None) is engine


@pytest.mark.parametrize("name", ["piquasso-fock", "piquasso-mixed-fock"])
def test_fock_names_still_require_cutoff(name):
    with pytest.raises(ValueError, match="explicit total-photon cutoff"):
        _backend(name, None)


@pytest.mark.parametrize("name", ["nope", ["gaussian"], None])
def test_unknown_backend_message_unchanged(name):
    with pytest.raises(ValueError, match=r"^Unknown backend: "):
        _backend(name, None)


def test_simulate_unknown_backend_message_unchanged():
    with pytest.raises(ValueError, match=r"^Unknown backend: 'nope'$"):
        pg.simulate(pg.Pattern(inputs=(0,)), backend="nope")


@pytest.mark.parametrize("name", ["native-gaussian", "native-fock", "native-mixed-fock"])
def test_native_names_resolve_or_raise_import_error(name):
    if name in AVAILABLE:
        assert type(_backend(name, 4)) is registry.BACKENDS[name].load()
    else:
        with pytest.raises(ImportError):
            _backend(name, 4)


# --- required features -------------------------------------------------------------


def _gaussian_pattern():
    return pg.Pattern(inputs=(0,)).extend(
        [pg.Prepare(1, 0.3), pg.Entangle(0, 1), pg.Measure(0, pg.Homodyne(), key="m")]
    )


def test_required_features_basic():
    pattern = _gaussian_pattern()
    assert required_features(pattern) == {"homodyne"}
    assert required_features(pattern, measurement_outcomes={"m": 0.0}) == {
        "homodyne",
        "postselection",
    }
    with pytest.raises(ValueError, match="Postselection key"):
        required_features(pattern, measurement_outcomes={"zz": 0.0})
    thermal = pg.GaussianInput(covariance=((2.0, 0.0), (0.0, 2.0)))
    assert "mixed_gaussian_input" in required_features(pattern, {0: thermal})
    state = pg.GaussianInput().state(0)
    assert "correlated_gaussian_state" in required_features(pattern, initial_state=state)


PATTERNS = {
    "gaussian": (_gaussian_pattern(), {}),
    "cat": (
        pg.Pattern(inputs=()).extend(
            [pg.Prepare("a", 0.0, pg.CatResource(0.5)), pg.Measure("a", pg.PhotonNumber())]
        ),
        {},
    ),
    "lossy_fock": (
        pg.Pattern(inputs=(0,)).extend([pg.Loss(0, 0.5), pg.Measure(0, pg.PhotonNumber())]),
        {0: pg.FockInput.number(1)},
    ),
    "cubic_heterodyne": (
        pg.Pattern(inputs=(0,)).extend([pg.CubicPhase(0, 0.1), pg.Measure(0, pg.Heterodyne())]),
        {},
    ),
}


@pytest.mark.parametrize("name", sorted(AVAILABLE))
@pytest.mark.parametrize("case", sorted(PATTERNS))
def test_planner_feature_check_agrees_with_preflight(case, name):
    pattern, inputs = PATTERNS[case]
    missing = registry.missing_features(name, required_features(pattern, inputs))
    engine = registry.create_backend(name, cutoff=8)
    if missing:
        with pytest.raises(NotImplementedError):
            preflight(engine, pattern, inputs)
    else:
        preflight(engine, pattern, inputs)


def test_preflight_mixed_gaussian_rejection_preserved():
    thermal = pg.GaussianInput(covariance=((2.0, 0.0), (0.0, 2.0)))
    with pytest.raises(NotImplementedError, match="Mixed Gaussian inputs require"):
        preflight(PiquassoFockBackend(cutoff=6), _gaussian_pattern(), {0: thermal})
    preflight(MixedFockBackend(cutoff=6), _gaussian_pattern(), {0: thermal})
    preflight(GaussianBackend(), _gaussian_pattern(), {0: thermal})


# --- select_backend ----------------------------------------------------------------


def test_select_gaussian_pattern():
    selection = registry.select_backend(_gaussian_pattern(), measurement_outcomes={"m": 0.1})
    assert selection.name == _cheapest("native-gaussian", "gaussian")
    assert selection.representation == "gaussian"
    assert selection.features == {"homodyne", "postselection"}
    assert selection.name in selection.explanation
    if "piquasso" in AVAILABLE:
        chosen = registry.select_backend(_gaussian_pattern(), prefer=("piquasso",)).name
        assert chosen == "piquasso"


def test_select_correlated_gaussian_state_needs_gaussian_backend_class():
    pattern = pg.Pattern(inputs=(0, 1)).extend([pg.Measure(0, pg.Homodyne())])
    state = pg.GaussianState(np.zeros(4), np.eye(4), (0, 1))
    selection = registry.select_backend(pattern, initial_state=state, cutoff=6)
    assert issubclass(registry.BACKENDS[selection.name].load(), GaussianBackend)


@pytest.mark.skipif("piquasso-fock" not in AVAILABLE, reason="needs piquasso")
def test_select_non_gaussian_pattern():
    pattern, inputs = PATTERNS["cat"]
    with pytest.raises(NotImplementedError, match="needs an explicit cutoff"):
        registry.select_backend(pattern, inputs)
    selection = registry.select_backend(pattern, inputs, cutoff=10)
    assert selection.name == _cheapest("native-fock", "piquasso-fock")
    assert selection.rejected["gaussian"].startswith("missing")
    assert "cat_state" in selection.rejected["gaussian"]


@pytest.mark.skipif("piquasso-mixed-fock" not in AVAILABLE, reason="needs piquasso")
def test_select_mixed_pattern():
    pattern, inputs = PATTERNS["lossy_fock"]
    selection = registry.select_backend(pattern, inputs, cutoff=6)
    assert selection.name == _cheapest("native-mixed-fock", "piquasso-mixed-fock")
    assert "loss" in selection.rejected["piquasso-fock"]
    thermal = pg.GaussianInput(covariance=((1.5, 0.0), (0.0, 1.5)))
    counting = pg.Pattern(inputs=(0,)).extend([pg.Measure(0, pg.PhotonNumber())])
    chosen = registry.select_backend(counting, {0: thermal}, cutoff=6)
    assert chosen.representation == "mixed-fock"
    assert "mixed_gaussian_input" in chosen.rejected["piquasso-fock"]


def test_select_raises_for_unsupported_combination():
    pattern, inputs = PATTERNS["cubic_heterodyne"]
    with pytest.raises(NotImplementedError, match="No available backend") as info:
        registry.select_backend(pattern, inputs, cutoff=8)
    assert "cubic_phase" in str(info.value) and "heterodyne" in str(info.value)


@pytest.mark.skipif("piquasso-fock" not in AVAILABLE, reason="needs piquasso")
def test_select_rejects_cutoff_too_small_via_preflight():
    pattern = pg.Pattern(inputs=(0,)).extend([pg.Measure(0, pg.PhotonNumber())])
    with pytest.raises(NotImplementedError, match="preflight"):
        registry.select_backend(pattern, {0: pg.FockInput.number(5)}, cutoff=4)
    assert registry.select_backend(pattern, {0: pg.FockInput.number(5)}, cutoff=8)


def test_select_bad_postselection_key():
    with pytest.raises(ValueError, match="Postselection key"):
        registry.select_backend(_gaussian_pattern(), measurement_outcomes={"zz": 0.0})


# --- CUDA-Q probe ------------------------------------------------------------------


def test_cudaq_reported_without_claiming_execution():
    capabilities = cudaq_capabilities()
    assert capabilities["verified"] is False
    assert {"bosonic_operators", "truncation", "open_system_dynamics"} <= set(
        capabilities["documented"]
    )
    assert "pattern execution" in capabilities["not_provided"]


def test_cudaq_is_an_adapter_not_a_backend():
    assert "cudaq-dynamics" not in registry.BACKENDS
    assert all(info.implementation != "cudaq" for info in registry.BACKENDS.values())
    assert all(info.device == "cpu" for info in registry.BACKENDS.values())
    with pytest.raises(ValueError, match="photographiq.cudaq_dynamics"):
        pg.simulate(pg.Pattern(inputs=(0,)), backend="cudaq-dynamics", cutoff=4)
    with pytest.raises(ValueError, match="adapter"):
        registry.describe_backend("cudaq-dynamics")
    probe = cudaq_probe()
    if not probe["available"]:
        assert "cudaq" in probe["error"]
