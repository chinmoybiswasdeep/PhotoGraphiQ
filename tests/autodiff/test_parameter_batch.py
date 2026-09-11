import numpy as np
import pytest

import photographiq as pg
from photographiq.autodiff import parameter_batch_expectation

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


def test_batch_and_gradient_against_rotation_formula():
    import jax.numpy as jnp

    pattern = pg.Pattern(inputs=(0,)).append(pg.Rotate(0, pg.Parameter("theta")))
    source = pg.FockInput((1 / np.sqrt(2), 1 / np.sqrt(2)))

    def evaluate(values):
        return parameter_batch_expectation(
            pattern, ("theta",), values, cutoff=4, inputs={0: source}, observable="q"
        )

    values = jnp.array([[0.1], [0.4], [-0.3]])
    np.testing.assert_allclose(evaluate(values), np.cos(np.array(values[:, 0])), atol=1e-12)
    np.testing.assert_allclose(
        jax.grad(lambda x: jnp.sum(evaluate(x)))(values)[:, 0],
        -np.sin(np.array(values[:, 0])),
        atol=1e-12,
    )
    with pytest.raises(ValueError):
        parameter_batch_expectation(pattern, ("wrong",), values, cutoff=4)
    with pytest.raises(ValueError):
        parameter_batch_expectation(pattern, ("theta",), values[:, 0], cutoff=4)
