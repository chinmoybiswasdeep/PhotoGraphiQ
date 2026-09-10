"""Allocation-free rejection regressions: invalid input must not run earlier gates."""

import numpy as np
import pytest

import photographiq as pg
from photographiq.backends import GaussianBackend
from photographiq.backends.fock import PiquassoFockBackend


@pytest.mark.parametrize(
    "source,error",
    [
        (pg.FockInput.number(8), ValueError),
        (pg.FockSuperposition.number((0, 1)), ValueError),
        (pg.GaussianInput(covariance=((1.0000001, 0), (0, 1.0000001))), NotImplementedError),
        (pg.GaussianInput(covariance=((-1, 0), (0, -1))), ValueError),
    ],
)
def test_invalid_late_preparation_rejected_before_allocation(monkeypatch, source, error):
    engine = PiquassoFockBackend(6)
    pattern = pg.Pattern().extend([pg.Prepare(0, 0), pg.Prepare(1, state=source)])

    def forbidden(*args, **kwargs):
        pytest.fail("Preflight allowed quantum state allocation")

    monkeypatch.setattr(engine, "prepare", forbidden)
    with pytest.raises(error):
        pg.simulate(pattern, backend=engine)


@pytest.mark.parametrize(
    "command",
    [
        pg.Loss(0, 0.9),
        pg.Loss(0, 0.9, 1),
        pg.Measure(0, pg.Homodyne(0, efficiency=0.9)),
        pg.Measure(0, pg.Homodyne(0, noise=0.1)),
    ],
)
def test_mixed_channels_rejected_before_preparation(monkeypatch, command):
    engine = PiquassoFockBackend(6)
    monkeypatch.setattr(
        engine, "prepare", lambda *a, **k: pytest.fail("Allocated before rejection")
    )
    with pytest.raises(NotImplementedError):
        pg.simulate(pg.Pattern().extend([pg.Prepare(0, 0), command]), backend=engine)


def test_gaussian_backend_and_postselection_guards():
    for pattern in [
        pg.Pattern().append(pg.Prepare(0, state=pg.FockInput.number(1))),
        pg.Pattern().append(pg.Prepare(0, 0)).measure(0, pg.PhotonNumber()),
        pg.Pattern().extend([pg.Prepare(0, 0), pg.CubicPhase(0, 0.3)]),
    ]:
        engine = GaussianBackend()
        with pytest.raises(NotImplementedError):
            pg.simulate(pattern, backend=engine)
        assert engine.state.nodes == ()
    with pytest.raises(ValueError, match="key"):
        pg.simulate(
            pg.Pattern().append(pg.Prepare(0, 0)),
            backend="piquasso-fock",
            cutoff=8,
            measurement_outcomes={"missing": 0},
        )


@pytest.mark.parametrize("modes,cutoff,expected", [(0, 4, 1), (1, 6, 6), (2, 4, 10), (3, 5, 35)])
def test_total_photon_dimension(modes, cutoff, expected):
    import itertools

    engine = PiquassoFockBackend(cutoff)
    assert engine.dimension(modes) == expected
    assert sum(sum(b) < cutoff for b in itertools.product(range(cutoff), repeat=modes)) == expected


def test_dimension_warning_does_not_allocate():
    engine = PiquassoFockBackend(100)
    with pytest.warns(RuntimeWarning, match="vector alone"):
        engine._guard(3)  # combinatorial guard only; no 171700-amplitude allocation
    with pytest.raises(MemoryError):
        engine._guard(4)
    with pytest.raises(ValueError):
        engine.dimension(True)


def test_mixed_reduced_input_is_never_implicitly_purified():
    p = pg.Pattern().append(
        pg.PrepareResource(
            (0, 1),
            pg.FockSuperposition.from_mapping({(0, 0): 1 / np.sqrt(2), (1, 1): 1 / np.sqrt(2)}),
        )
    )
    mixed = pg.simulate(p, backend="piquasso-fock", cutoff=5).state.reduced((0,))
    with pytest.raises((ValueError, NotImplementedError, TypeError)):
        pg.simulate(pg.Pattern(inputs=(0,)), backend="piquasso-fock", cutoff=5, initial_state=mixed)
