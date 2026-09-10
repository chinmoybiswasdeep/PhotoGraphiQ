import numpy as np
import pytest

import photographiq as pg
from photographiq import autodiff as ad

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


def test_zero_probability_eager_error_and_traced_nan():
    p = (
        pg.Pattern(inputs=(0,))
        .append(pg.Rotate(0, pg.Parameter("t")))
        .measure(0, pg.PhotonNumber())
    )
    kwargs = dict(cutoff=3, measurement_outcomes={0: 1})
    with pytest.raises(ValueError, match="zero|probability"):
        ad.fock_state(p, {"t": 0.2}, **kwargs)
    result = jax.jit(lambda t: ad.fock_state(p, {"t": t}, **kwargs).density_matrix)(0.2)
    assert np.isnan(np.asarray(result)).all()


def test_conditional_likelihood_and_weighted_derivatives_are_distinct():
    # |00>+|11> passed through loss on measured mode. For count=0,
    # p=1-eta/2, E[n_b]=(1-eta)/(2-eta), p*E=(1-eta)/2.
    source = pg.FockSuperposition.from_mapping({(0, 0): 2**-0.5, (1, 1): 2**-0.5})
    p = (
        pg.Pattern()
        .append(pg.PrepareResource((0, 1), source))
        .append(pg.Loss(0, pg.Parameter("eta")))
        .measure(0, pg.PhotonNumber())
    )

    def values(eta):
        state = ad.fock_state(p, {"eta": eta}, cutoff=4, measurement_outcomes={0: 0})
        mean = jax.numpy.trace(state.density_matrix @ jax.numpy.diag(jax.numpy.arange(4))).real
        return jax.numpy.stack(
            [mean, state.log_likelihood, jax.numpy.exp(state.log_likelihood) * mean]
        )

    for eta in (0.2, 0.6, 0.95):
        derivative = jax.jacrev(values)(eta)
        np.testing.assert_allclose(
            derivative, [-1 / (2 - eta) ** 2, -1 / (2 - eta), -0.5], atol=2e-12
        )
