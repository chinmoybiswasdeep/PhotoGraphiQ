import numpy as np
import piquasso as pq
import pytest

import photographiq as pg
from tests.v03_reference import basis, gate, physical, vector


@pytest.mark.parametrize(
    "name,value",
    [
        ("Rotate", 0.4),
        ("Displace", (0.15, -0.08)),
        ("Squeeze", 0.08),
        ("QuadraticPhase", 0.1),
        ("CubicPhase", 0.02),
        ("Kerr", 0.3),
        ("BeamSplitter", 0.3),
        ("Entangle", 0.08),
    ],
)
def test_unitary_against_independent_exponential(name, value):
    modes = 2 if name in ("BeamSplitter", "Entangle") else 1
    cutoff = 12
    occupations = basis(modes, cutoff)
    source_basis = ((0,) * modes, (1,) + (0,) * (modes - 1))
    rho = np.array([[0.4, 0.13j], [-0.13j, 0.6]])
    source = pg.FockDensityMatrix(rho, source_basis)
    args = (0, 1, value) if modes == 2 else ((0, *value) if name == "Displace" else (0, value))
    command = getattr(pg, name)(*args)
    out = pg.simulate(
        pg.Pattern(inputs=tuple(range(modes))).append(command),
        initial_state=source,
        backend="piquasso-mixed-fock",
        cutoff=cutoff,
    ).state
    # Larger independent space resolves Gaussian matrix-element tails. Cubic
    # native evolution is itself exp of the projected single-mode q^3.
    ref_basis = basis(modes, cutoff if name == "CubicPhase" else cutoff + 4)
    initial = np.zeros((len(ref_basis), len(ref_basis)), complex)
    ii = [ref_basis.index(b) for b in source_basis]
    initial[np.ix_(ii, ii)] = rho
    u = gate(ref_basis, name, value)
    evolved = u @ initial @ u.conj().T
    ii = [ref_basis.index(b) for b in occupations]
    expected = evolved[np.ix_(ii, ii)]
    assert abs(np.trace(expected) - 1) < 2e-8
    physical(out.density_matrix)
    np.testing.assert_allclose(out.density_matrix, expected, atol=2e-8, rtol=0)


@pytest.mark.parametrize("kind", ["vacuum", "coherent", "cat", "superposition"])
def test_pure_mixed_and_raw_native(kind):
    cutoff = 16
    v = vector(kind, cutoff)
    commands = [pg.Rotate(0, 0.2), pg.CubicPhase(0, 0.02), pg.Kerr(0, 0.3)]
    p = pg.Pattern(inputs=(0,)).extend(commands)
    outputs = [
        pg.simulate(p, inputs={0: pg.FockInput(tuple(v))}, backend=b, cutoff=cutoff).state
        for b in ("piquasso-fock", "piquasso-mixed-fock")
    ]
    assert outputs[0].trace_distance(outputs[1]) < 2e-12
    with pq.Program() as program:
        for i, j in zip(*np.nonzero(np.outer(v, v.conj())), strict=True):
            pq.Q() | pq.DensityMatrix((int(i),), (int(j),), v[i] * v[j].conjugate())
        pq.Q(0) | pq.Phaseshifter(0.2)
        pq.Q(0) | pq.CubicPhase(0.02)
        pq.Q(0) | pq.Kerr(0.3)
    raw = pq.FockSimulator(d=1, config=pq.Config(cutoff=cutoff, hbar=2)).execute(program).state
    np.testing.assert_allclose(outputs[1].density_matrix, raw.density_matrix, atol=2e-12)
