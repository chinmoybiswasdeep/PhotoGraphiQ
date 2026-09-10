import numpy as np
import pytest

import photographiq as pg
from photographiq import autodiff as ad

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "command",
    [
        pg.Displace(0, q=0.08, p=0.04),
        pg.Squeeze(0, 0.08),
        pg.CubicPhase(0, 0.01),
        pg.Kerr(0, 0.2),
        pg.QuadraticPhase(0, 0.08),
    ],
)
def test_projected_gates_match_native_at_low_energy(command):
    p = pg.Pattern(inputs=(0,)).append(command)
    inputs = {0: pg.FockInput((2**-0.5, 2**-0.5))}
    native = pg.simulate(p, inputs=inputs, backend="piquasso-fock", cutoff=16)
    traced = ad.fock_state(p, inputs=inputs, cutoff=16)
    np.testing.assert_allclose(traced.density_matrix, native.state.density_matrix, atol=2e-6)


@pytest.mark.parametrize("command", [pg.Entangle("a", "b", 0.08), pg.BeamSplitter("a", "b", 0.2)])
def test_two_mode_ordering_and_output_reduction(command):
    p = pg.Pattern(inputs=("a", "b")).append(command).append(pg.Output(("b",)))
    inputs = {"a": pg.FockInput.number(1)}
    native = pg.simulate(p, inputs=inputs, backend="piquasso-fock", cutoff=6)
    traced = ad.fock_state(p, inputs=inputs, cutoff=6)
    assert traced.nodes == ("b",)
    np.testing.assert_allclose(traced.density_matrix, native.state.density_matrix, atol=1e-5)


def test_parameterized_preparation_signal_and_ladder():
    p = pg.Pattern().append(pg.Prepare(0, pg.Parameter("r")))
    gradient = jax.grad(lambda r: ad.expectation(p, {"r": r}, cutoff=14))(0.1)
    assert float(gradient) == pytest.approx(np.sinh(0.2), abs=1e-8)
    p = (
        pg.Pattern(inputs=(0,))
        .append(pg.Signal("x", 2 * pg.Parameter("t")))
        .append(pg.Displace(0, q=pg.Outcome("x")))
    )
    gradient = jax.grad(lambda t: ad.expectation(p, {"t": t}, observable="q", cutoff=12))(0.1)
    assert float(gradient) == pytest.approx(2, abs=1e-8)
    p = pg.Pattern(inputs=(0,)).append(pg.PhotonSubtract(0))
    assert float(ad.expectation(p, inputs={0: pg.FockInput.number(2)}, cutoff=6)) == pytest.approx(
        1
    )


def test_score_function_expected_gradient():
    # Exact two-point average supplies an independent Bernoulli score identity.
    def surrogate(t):
        probabilities = jax.numpy.array([1 - t, t])
        values = jax.numpy.array([0.0, 1.0])
        return ad.score_function_surrogate(
            2 * jax.lax.stop_gradient(probabilities) * values, jax.numpy.log(probabilities)
        )

    assert float(jax.grad(surrogate)(0.3)) == pytest.approx(1.0)


def test_autodiff_rejects_undefined_execution_contracts():
    p = pg.Pattern(inputs=(0,)).measure(0, pg.Homodyne())
    with pytest.raises(ValueError, match="fixed outcome"):
        ad.fock_state(p, cutoff=6)
    with pytest.raises(TypeError, match="callables"):
        ad.resolve(pg.CallableExpression(lambda records: 1, {}), {})
    with pytest.raises(MemoryError):
        ad.fock_state(pg.Pattern(inputs=(0, 1)), cutoff=12, max_dimension=3)
    with pytest.raises(ValueError, match="cutoff"):
        ad.fock_state(pg.Pattern(), cutoff=True)
