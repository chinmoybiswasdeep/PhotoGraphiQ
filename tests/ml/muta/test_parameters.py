import numpy as np
import pytest

from photographiq.expressions import Expr
from photographiqml import MuTA


def test_binding_freezing_serialization(tmp_path):
    model = MuTA(2, 2, restrict_trainable=True)
    assert all(isinstance(x, Expr) for x in model.parameters().values())
    name = next(iter(model.trainable_parameters()))
    model.freeze(name, 0.42)
    assert name not in model.trainable_parameters()
    assert model._parameters.bind()[name] == 0.42
    with pytest.raises(ValueError, match="frozen"):
        model.run([1, 0, 0, 0], {name: 0.3})
    model.unfreeze(name)
    model._parameters.values = model._parameters.bind(model.initialize(7))
    model.freeze(name)
    path = tmp_path / "model.json"
    model.save(path)
    loaded = MuTA.load(path)
    assert loaded.state_dict() == model.state_dict()
    assert np.allclose(loaded.unitary(), model.unitary())
    assert "Paper layers: 4" in loaded.summary()


def test_seed_and_shapes():
    m = MuTA(1, one_column=True)
    assert m.initialize(0) == m.initialize(0)
    assert np.allclose(m.run_batch(np.eye(2)), np.eye(2))
    assert m.run_batch(np.empty((0, 2))).shape == (0, 2)
    result = m.run([1, 0])
    assert result.expectation(np.diag([1, -1])) == pytest.approx(1)
    assert result.probabilities.sum() == pytest.approx(1)
    for params in ({"missing": 0}, [1], {"alpha.w0.c0": np.nan}):
        with pytest.raises(ValueError):
            m.run([1, 0], params)
    for state in ([1, 1], [1, 0, 0], [np.nan, 0]):
        with pytest.raises(ValueError):
            m.run(state)
    with pytest.raises(ValueError):
        m.run_batch([1, 0])
    with pytest.raises(ValueError):
        result.expectation([[0, 1], [0, 0]])
    with pytest.raises(ValueError):
        m.freeze("missing")
    with pytest.raises(ValueError):
        m.unfreeze("missing")
    with pytest.raises(ValueError):
        m.initialize(scale=-1)
