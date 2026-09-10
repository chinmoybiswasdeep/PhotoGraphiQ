import numpy as np
import pytest

import photographiq as pg
from photographiq import autodiff as ad

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


def test_kerr_gradient_resolves_nonlinear_level_spacing():
    # Levels 1 and 2 acquire relative phase (2²-1²)*kappa=3*kappa,
    # distinguishing Kerr from a rotation on the 0/1 subspace.
    p = pg.Pattern(inputs=(0,)).append(pg.Kerr(0, pg.Parameter("k")))
    source = pg.FockInput((0, 2**-0.5, 2**-0.5))

    def objective(k):
        return ad.expectation(p, {"k": k}, inputs={0: source}, cutoff=6, observable="q")

    k = 0.23
    assert float(objective(k)) == pytest.approx(np.sqrt(2) * np.cos(3 * k), abs=2e-12)
    assert float(jax.grad(objective)(k)) == pytest.approx(
        -3 * np.sqrt(2) * np.sin(3 * k), abs=2e-12
    )


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("Rotate", 0.3, -np.sin(0.3)),
        ("Displace", 0.2, 1.0),
        ("Squeeze", 0.12, -2 * np.exp(-0.24)),
        ("Kerr", 0.3, -np.sin(0.3)),
        ("CubicPhase", 0.03, 1.0),
    ],
)
def test_analytical_gate_gradients(name, value, expected):
    t = pg.Parameter("t")
    command = pg.Displace(0, q=t) if name == "Displace" else getattr(pg, name)(0, t)
    p = pg.Pattern(inputs=(0,)).append(command)
    inputs = {0: pg.FockInput((2**-0.5, 2**-0.5))} if name in ("Rotate", "Kerr") else {}

    def objective(t):
        if name == "Squeeze":
            state = ad.fock_state(p, {"t": t}, inputs=inputs, cutoff=18)
            # Include external ladder paths in the physical second moment.
            a = np.diag(np.sqrt(np.arange(1, 20)), 1)
            q2 = ((a + a.T) @ (a + a.T))[:18, :18]
            return jax.numpy.trace(state.density_matrix @ jax.numpy.asarray(q2)).real
        return ad.expectation(
            p, {"t": t}, inputs=inputs, cutoff=18, observable="p" if name == "CubicPhase" else "q"
        )

    assert float(jax.grad(objective)(value)) == pytest.approx(expected, abs=2e-8)


def test_cubic_observable_and_gradient_cutoff_convergence():
    p = pg.Pattern(inputs=(0,)).append(pg.CubicPhase(0, pg.Parameter("g")))
    rows = []
    for cutoff in (8, 16, 28):

        def f(g):
            return ad.expectation(p, {"g": g}, cutoff=cutoff, observable="p")

        state = ad.fock_state(p, {"g": 0.15}, cutoff=cutoff)
        rows.append((float(f(0.15)), float(jax.grad(f)(0.15)), float(state.boundary_population)))
        assert min(float(x) for x in state.retained_norms) > 1 - 1e-12
    assert abs(rows[-1][0] - 0.15) < 2e-7
    assert abs(rows[-1][1] - 1) < 2e-6
    assert abs(rows[-1][1] - 1) < abs(rows[0][1] - 1)
    assert rows[-1][2] < rows[0][2]
