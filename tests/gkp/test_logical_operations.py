import numpy as np
import pytest

import photographiq as pg
from tests.gkp.test_independent_convergence import comb_inner, overlap


def make_state(resource, cutoff):
    return pg.simulate(
        pg.Pattern(inputs=(0,)), inputs={0: resource}, backend="piquasso-fock", cutoff=cutoff
    ).state


@pytest.mark.parametrize("logical", [0, 1, "plus"])
def test_stabilizers_against_closed_integrals(logical):
    resources = [pg.GKPResource(bit, 0.5, 0.45, 6, 4097) for bit in (0, 1)]
    weights = [1.0, 0.0] if logical == 0 else ([0.0, 1.0] if logical == 1 else [1.0, 1.0])
    # superposition() combines individually normalized finite codewords.
    weights = np.array(weights) / np.sqrt([comb_inner(r, r).real for r in resources])

    def inner(dq, dp):
        return sum(
            weights[i] * weights[j] * comb_inner(a, b, dq, dp)
            for i, a in enumerate(resources)
            for j, b in enumerate(resources)
        )

    length = pg.gkp.SPACING
    expected = [inner(2 * length, 0) / inner(0, 0), inner(0, 2 * length) / inner(0, 0)]
    errors = []
    for cutoff in (32, 64):
        source = (
            resources[logical].fock(cutoff)
            if logical != "plus"
            else pg.gkp.superposition(1, 1, cutoff=cutoff, peak_width=0.5, envelope=0.45, peaks=6)
        )
        values = pg.gkp.stabilizers(make_state(source, cutoff))
        errors.append(max(abs(np.array(list(values.values())) - expected)))
    assert errors[-1] < 2e-7
    assert errors[-1] < errors[0]


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_logical_displacement_fidelity_and_repeated_action(pauli):
    cutoff, length = 96, pg.gkp.SPACING
    zero, one = [pg.GKPResource(bit, 0.5, 0.4, 6, 4097) for bit in (0, 1)]
    if pauli == "X":
        source, target = zero.fock(cutoff), one.fock(cutoff)
        expected = abs(overlap(one, zero, dq=length)) ** 2
    else:
        source = pg.gkp.superposition(1, 1, cutoff=cutoff, peak_width=0.5, envelope=0.4, peaks=6)
        target = pg.gkp.superposition(1, -1, cutoff=cutoff, peak_width=0.5, envelope=0.4, peaks=6)
        gram = overlap(zero, one).real
        amplitude = sum(
            si * tj * overlap(a, b, dp=length)
            for si, a in [(1, zero), (-1, one)]
            for tj, b in [(1, zero), (1, one)]
        ) / np.sqrt((2 - 2 * gram) * (2 + 2 * gram))
        expected = abs(amplitude) ** 2
    p = pg.Pattern(inputs=(0,)).append(pg.gkp.logical_displacement(0, pauli))
    output = pg.simulate(p, inputs={0: source}, backend="piquasso-fock", cutoff=cutoff).state
    fidelity = abs(np.vdot(target.amplitudes, output.state_vector)) ** 2
    assert fidelity == pytest.approx(expected, abs=2e-7)
    assert 0.5 < fidelity < 1  # physical envelope deviation, not a loose oracle tolerance
    p.append(pg.gkp.logical_displacement(0, pauli))
    twice = pg.simulate(p, inputs={0: source}, backend="piquasso-fock", cutoff=cutoff).state
    initial = make_state(source, cutoff)
    key = "q_translation" if pauli == "X" else "p_translation"
    assert abs(np.vdot(initial.state_vector, twice.state_vector)) ** 2 == pytest.approx(
        abs(pg.gkp.stabilizers(initial)[key]) ** 2, abs=2e-7
    )
