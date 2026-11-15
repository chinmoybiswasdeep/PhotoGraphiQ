"""Key quantities of the documentation tutorials agree on native and Piquasso backends.

The tutorials run on ``native-fock``/``native-mixed-fock``; this keeps the Piquasso
engines they replaced as a reference. Tolerances are measured differences with margin:
exact-physics quantities agree to roundoff, gates that the engines truncate differently
(squeezing/CZ decompositions in cubic injection and the GKP CZ) to truncation level.
"""

import numpy as np
import pytest

import photographiq as pg

pytest.importorskip("piquasso")  # reference simulator: optional extra
if not {"native-fock", "native-mixed-fock"} <= set(pg.available_backends()):
    pytest.skip("native Fock engine not compiled", allow_module_level=True)

PAIRS = {
    "pure": ("piquasso-fock", "native-fock"),
    "mixed": ("piquasso-mixed-fock", "native-mixed-fock"),
}


def both(kind, run):
    """(Piquasso result, native result) of ``run(backend_name)``."""
    return [run(name) for name in PAIRS[kind]]


def test_photon_number_branch_09():
    resource = pg.FockSuperposition.from_mapping({(0, 0): 2**-0.5, (1, 1): 2**-0.5})
    pattern = pg.Pattern().append(pg.PrepareResource(("a", "b"), resource))
    pattern.measure("a", pg.PhotonNumber())

    def run(backend):
        r = pg.simulate(pattern, backend=backend, cutoff=6, measurement_outcomes={"a": 1})
        return np.exp(r.log_likelihood), r.state.photon_number("b")

    for probability, photons in both("pure", run):
        assert probability == pytest.approx(0.5, abs=1e-12)
        assert photons == pytest.approx(1, abs=1e-12)


def test_photon_subtraction_herald_13():
    theta = 0.2
    pattern = pg.non_gaussian.photon_subtraction(theta=theta)

    def run(backend):
        r = pg.simulate(
            pattern,
            inputs={"in": pg.FockInput.number(2)},
            backend=backend,
            cutoff=6,
            measurement_outcomes={"count": 1},
        )
        return np.exp(r.log_likelihood), r.state.state_vector

    (p_ref, v_ref), (p_nat, v_nat) = both("pure", run)
    assert p_ref == pytest.approx(2 * np.sin(theta) ** 2 * np.cos(theta) ** 2, abs=1e-14)
    assert p_nat == pytest.approx(p_ref, abs=1e-14)
    np.testing.assert_allclose(v_nat, v_ref, atol=1e-12)


def test_cat_wigner_negativity():
    grid = np.linspace(-6, 6, 81)

    def run(backend):
        r = pg.simulate(
            pg.Pattern(inputs=(0,)), inputs={0: pg.CatResource(1.2)}, backend=backend, cutoff=24
        )
        return r.state.wigner(grid, grid)

    reference, native = both("pure", run)
    assert reference.negative_volume > 0.15
    assert native.negative_volume == pytest.approx(reference.negative_volume, abs=1e-12)
    np.testing.assert_allclose(native.values, reference.values, atol=1e-12)


def test_mixed_kerr_loss_21():
    pattern = pg.Pattern(inputs=(0,)).append(pg.Kerr(0, 0.2)).append(pg.Loss(0, 0.6))

    def run(backend):
        r = pg.simulate(pattern, inputs={0: pg.FockInput.number(2)}, backend=backend, cutoff=6)
        return r.state.density_matrix

    reference, native = both("mixed", run)
    np.testing.assert_allclose(np.diag(reference).real[:3], [0.16, 0.48, 0.36], atol=1e-12)
    np.testing.assert_allclose(native, reference, atol=1e-12)


def test_cubic_injection_14_agrees_to_truncation_level():
    pattern = pg.non_gaussian.cubic_injection(gamma=0.02, squeezing=0.2)

    def run(backend):
        r = pg.simulate(pattern, backend=backend, cutoff=24, measurement_outcomes={"m": 0.1})
        return np.exp(r.log_likelihood), r.state.quadrature("in", np.pi / 2)

    (d_ref, m_ref), (d_nat, m_nat) = both("pure", run)
    assert d_nat == pytest.approx(d_ref, abs=1e-8)  # measured 3.3e-9
    np.testing.assert_allclose(m_nat, m_ref, atol=1e-6)  # measured 2.7e-7


def test_gkp_z_readout_38():
    code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)
    pattern = pg.Pattern(inputs=(0,)).measure(0, code.logical_measurement("Z"), key="z")

    def run(backend):
        r = pg.simulate(
            pattern,
            inputs={0: code.zero()},
            backend=backend,
            cutoff=code.cutoff,
            measurement_outcomes={"z": 0.25},
        )
        return r.outcomes["z"].bit, r.measurement_statistics["z"]["value"], r.state.state_vector

    (b_ref, d_ref, v_ref), (b_nat, d_nat, v_nat) = both("pure", run)
    assert b_ref == b_nat == 0
    assert d_nat == pytest.approx(d_ref, abs=1e-12)
    np.testing.assert_allclose(v_nat, v_ref, atol=1e-10)


def test_gkp_cz_readout_44_agrees_to_truncation_level():
    code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)
    v = np.array(code.plus().amplitudes)
    data = {(i, j): v[i] * v[j] for i in range(code.cutoff) for j in range(code.cutoff - i)}
    norm = np.sqrt(sum(abs(a) ** 2 for a in data.values()))
    source = pg.FockSuperposition.from_mapping({k: a / norm for k, a in data.items()})
    pattern = pg.Pattern(inputs=(0, 1)).append(code.logical_cz(0, 1))

    def run(backend):
        r = pg.simulate(pattern, initial_state=source, backend=backend, cutoff=code.cutoff)
        joint = pg.multimode_readout(r.state, {0: "X", 1: "Z"})["joint_probabilities"]
        return np.array([joint[k] for k in sorted(joint)])

    reference, native = both("pure", run)
    for probabilities in (reference, native):
        assert probabilities.sum() == pytest.approx(1, abs=1e-9)
    # CZ: native applies exact P U P matrix elements, Piquasso a truncated decomposition.
    np.testing.assert_allclose(native, reference, atol=1e-7)  # measured 1.9e-8
