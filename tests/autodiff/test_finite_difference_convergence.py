import numpy as np
import pytest

import photographiq as pg
from photographiq import autodiff as ad

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "name",
    [
        "Rotate",
        "Displace",
        "Squeeze",
        "QuadraticPhase",
        "CubicPhase",
        "Kerr",
        "BeamSplitter",
        "Entangle",
        "Loss",
        "Prepare",
        "Homodyne",
        "PhotonAdd",
        "PhotonSubtract",
    ],
)
def test_gradient_has_finite_difference_window(name):
    t = pg.Parameter("t")
    inputs = {0: pg.FockInput((2**-0.5, 2**-0.5))}
    outcomes = {}
    if name in ("BeamSplitter", "Entangle", "Homodyne"):
        p = pg.Pattern(inputs=(0, 1))
        if name == "Homodyne":
            p.append(pg.Entangle(0, 1, 0.2)).measure(1, pg.Homodyne(t))
            outcomes = {1: 0.4}
        else:
            p.append(getattr(pg, name)(0, 1, t))
    elif name == "Prepare":
        p = pg.Pattern().append(pg.Prepare(0, t))
        inputs = {}
    else:
        p = pg.Pattern(inputs=(0,))
        if name in ("PhotonAdd", "PhotonSubtract"):
            p.append(pg.Displace(0, q=t)).append(getattr(pg, name)(0))
        else:
            p.append(
                pg.Displace(0, q=t, p=t / 2) if name == "Displace" else getattr(pg, name)(0, t)
            )

    # Scalar linear functional includes diagonal and off-diagonal information.
    def objective(t):
        state = ad.fock_state(p, {"t": t}, cutoff=10, inputs=inputs, measurement_outcomes=outcomes)
        size = len(state.basis)
        n = jax.numpy.diag(jax.numpy.asarray([sum(b) for b in state.basis], dtype=float))
        off = jax.numpy.diag(jax.numpy.ones(size - 1), 1)
        observable = n + 0.3 * (off + off.T) + 0.2j * (off - off.T)
        return jax.numpy.trace(state.density_matrix @ observable).real + 0.2 * state.log_likelihood

    point = 0.4 if name == "Loss" else 0.07
    f = jax.jit(objective)
    gradient = float(jax.grad(f)(point))
    steps = np.array([1e-3, 1e-4, 1e-5, 1e-6])
    errors = np.array(
        [abs(float((f(point + h) - f(point - h)) / (2 * h)) - gradient) for h in steps]
    )
    assert np.isfinite(gradient)
    # Require two adjacent accurate steps, not one fortuitous cancellation.
    assert any(np.all(errors[i : i + 2] < 2e-7) for i in range(3)), (name, errors)
    assert errors[2] <= errors[0] + 2e-8, (name, errors)
