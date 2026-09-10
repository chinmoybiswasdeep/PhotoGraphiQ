import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import photographiq as pg  # noqa: E402
from photographiq import autodiff as ad  # noqa: E402


def test_rotation_gradient_and_finite_difference():
    pattern = pg.Pattern(inputs=(0,)).append(pg.Rotate(0, pg.Parameter("theta")))
    initial = pg.FockInput((2**-0.5, 2**-0.5))

    def objective(theta):
        return ad.expectation(
            pattern, {"theta": theta}, cutoff=5, inputs={0: initial}, observable="q"
        )

    gradient = float(jax.grad(objective)(0.3))
    assert gradient == pytest.approx(-np.sin(0.3), abs=1e-10)
    assert gradient == pytest.approx(
        float((objective(0.30001) - objective(0.29999)) / 0.00002), abs=1e-9
    )


def test_loss_and_branch_likelihood_gradient():
    pattern = pg.Pattern(inputs=(0,)).append(pg.Loss(0, pg.Parameter("eta")))
    initial = {0: pg.FockInput.number(2)}

    def mean(eta):
        return ad.expectation(pattern, {"eta": eta}, cutoff=4, inputs=initial)

    assert float(jax.grad(mean)(0.6)) == pytest.approx(2.0, abs=1e-10)
    pattern.measure(0, pg.PhotonNumber())

    def likelihood(eta):
        return ad.fock_state(
            pattern, {"eta": eta}, cutoff=4, inputs=initial, measurement_outcomes={0: 1}
        ).log_likelihood

    assert float(jax.grad(likelihood)(0.6)) == pytest.approx(1 / 0.6 - 1 / 0.4, abs=1e-10)


def test_homodyne_conditional_and_sampling_primitive():
    resource = pg.FockSuperposition.from_mapping({(0, 0): 2**-0.5, (1, 1): 2**-0.5})
    pattern = (
        pg.Pattern()
        .append(pg.PrepareResource((0, 1), resource))
        .measure(0, pg.Homodyne(pg.Parameter("angle")))
    )

    def objective(angle):
        return ad.expectation(
            pattern, {"angle": angle}, cutoff=4, measurement_outcomes={0: 0.4}, observable="q"
        )

    assert float(jax.grad(objective)(0.2)) == pytest.approx(
        float((objective(0.20001) - objective(0.19999)) / 0.00002), abs=1e-8
    )
    assert (
        float(
            jax.grad(
                lambda mu: ad.gaussian_sample(
                    jax.numpy.array([mu]), jax.numpy.eye(1), jax.numpy.array([0.2])
                )[0]
            )(0.0)
        )
        == 1.0
    )
