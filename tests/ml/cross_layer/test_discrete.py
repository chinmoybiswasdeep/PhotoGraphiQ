import numpy as np
import pytest

from photographiqml import GKPPhysicalConfig, MuTA, PhysicalMuTA
from photographiqml.models import MuTAClassifier, MuTARegressor, QuantumInstrumentModel
from photographiqml.physical_training import DiscreteSearch
from photographiqml.training import Trainer


def test_discrete_coordinate_search():
    model = PhysicalMuTA()
    search = DiscreteSearch(sweeps=2)
    result = search.fit(model, lambda p: np.sum((p - np.pi) ** 2), initial=np.zeros(4))
    assert np.all(result.parameters == np.pi) and result.loss == 0
    assert len(result.evaluations) == 9
    assert DiscreteSearch(sweeps=0).fit(model, lambda p: 1).loss == 1
    with pytest.raises(ValueError):
        search.fit(MuTA(), lambda p: 0)
    with pytest.raises(ValueError):
        search.fit(model, lambda p: 0, initial=[0.1] * 4)
    with pytest.raises(ValueError):
        search.fit(model, lambda p: np.nan)
    with pytest.raises(ValueError):
        DiscreteSearch(sweeps=-1)


def test_supervised_physical_guards():
    model = PhysicalMuTA()
    for options in [dict(trainer=Trainer()), {}, dict(physical_options={"mode": "physical-shots"})]:
        with pytest.raises(ValueError):
            MuTAClassifier(model, **options)
    with pytest.raises(NotImplementedError, match="post-measurement"):
        QuantumInstrumentModel(model)


@pytest.mark.parametrize("wrapper,labels", [(MuTAClassifier, [0, 1]), (MuTARegressor, [-1, 1])])
def test_physical_supervised_head_real_shots(wrapper, labels):
    model = PhysicalMuTA(
        physical_config=GKPPhysicalConfig(
            cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025
        )
    )
    # A zero-sweep baseline exercises actual Fock features and classical-head fitting.
    fitted = wrapper(
        model,
        trainer=DiscreteSearch(sweeps=0),
        physical_options={"mode": "physical-shots", "shots": 1, "seed": 14},
    )
    fitted.fit([[0], [np.pi]], labels)
    prediction = fitted.predict([[0], [np.pi]])
    assert prediction.shape == (2,) and np.isfinite(prediction).all()
    assert len(fitted.physical_diagnostics) == 2
    assert all(not c["certified"] for c in fitted.convergence)
    assert set(fitted.weights[:-2]) <= {0, np.pi}
