import numpy as np
import pytest

from photographiqml import MuTA, MuTAClassifier, MuTARegressor, Trainer
from photographiqml.training import finite_difference


def test_nonfinite_training_fails_before_update():
    with pytest.raises(ValueError, match="Objective"):
        Trainer(epochs=1).fit(lambda x: np.nan, [0])
    with pytest.raises(ValueError, match="gradient"):
        Trainer(epochs=1).fit(lambda x: 0, [0], gradient=lambda x: [np.nan])
    with pytest.raises(ValueError, match="compatible shapes"):
        Trainer(epochs=1).fit(lambda x: 0, [0], gradient=lambda x: [1, 2])
    with pytest.raises(ValueError, match="finite vector"):
        Trainer().fit(lambda x: 0, [np.inf])
    with pytest.raises(ValueError):
        finite_difference(lambda x: 0, [0], step=0)


def test_callback_copy_and_validation_history():
    def callback(point, history):
        point[:] = 1000

    values, history = Trainer("sgd", epochs=1, learning_rate=0.1).fit(
        lambda x: x[0] ** 2,
        [1],
        gradient=lambda x: 2 * x,
        validation=lambda x: 2 * x[0] ** 2,
        callback=callback,
    )
    assert values[0] == pytest.approx(0.8)
    assert history.validation_losses == pytest.approx([2, 1.28])


def test_supervised_contract_does_not_broadcast_or_train_gkp():
    with pytest.raises(ValueError, match="logical"):
        MuTAClassifier(MuTA(1, representation="gkp"))
    classifier = MuTAClassifier(MuTA(1))
    with pytest.raises(ValueError, match="labels"):
        classifier.fit([[0], [1]], [0, 2])
    with pytest.raises(ValueError, match="Targets"):
        classifier.fit([[0], [1]], [[0], [1]])
    with pytest.raises(ValueError, match="shape"):
        classifier.fit([[0, 1]], [0])
    with pytest.raises(ValueError, match="fit"):
        MuTARegressor(MuTA(1)).predict([[0]])
