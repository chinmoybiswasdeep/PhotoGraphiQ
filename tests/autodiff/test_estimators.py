import numpy as np
import pytest

from photographiq import autodiff as ad

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


def test_pathwise_gaussian_bias_and_reproducibility():
    noise = np.random.default_rng(12).normal(size=40000)
    np.testing.assert_array_equal(noise, np.random.default_rng(12).normal(size=40000))

    # x=mu+exp(r)*epsilon; E[x²]=mu²+exp(2r).
    def objective(mu, r):
        samples = jax.vmap(
            lambda e: ad.gaussian_sample(
                jax.numpy.array([mu]),
                jax.numpy.array([[jax.numpy.exp(2 * r)]]),
                jax.numpy.array([e]),
            )[0]
        )(jax.numpy.asarray(noise))
        return jax.numpy.mean(samples**2)

    gradients = np.asarray(jax.grad(objective, argnums=(0, 1))(0.4, 0.2))
    samples = 0.4 + np.exp(0.2) * noise
    contributions = np.stack([2 * samples, 2 * samples * np.exp(0.2) * noise])
    expected = np.array([0.8, 2 * np.exp(0.4)])
    stderr = contributions.std(axis=1, ddof=1) / np.sqrt(len(noise))
    assert np.all(abs(gradients - expected) < 5 * stderr)


def test_bernoulli_score_statistical_bias_variance_and_seed():
    p = 0.3
    samples = np.random.default_rng(37).binomial(1, p, 50000)
    np.testing.assert_array_equal(samples, np.random.default_rng(37).binomial(1, p, 50000))

    def objective(t):
        logs = samples * jax.numpy.log(t) + (1 - samples) * jax.numpy.log1p(-t)
        return ad.score_function_surrogate(
            jax.numpy.asarray(samples, dtype=float), logs, baseline=0.2
        )

    gradient = float(jax.grad(objective)(p))
    contributions = (samples - 0.2) * (samples / p - (1 - samples) / (1 - p))
    assert gradient == pytest.approx(contributions.mean(), abs=2e-12)
    assert abs(gradient - 1) < 5 * contributions.std(ddof=1) / np.sqrt(len(samples))
    assert contributions.var() > 0
