"""Scientific plots: data equal PhotoGraphiQ/closed-form values; HTML and static exports."""

import json
import math
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

import photographiq as pg
from photographiq.interactive import (
    covariance_ellipse,
    covariance_heatmap,
    entanglement_diagnostics,
    fidelity_vs_cutoff,
    gaussian_wigner,
    measurement_probabilities,
    optimization_curves,
    photon_number_distribution,
    photon_number_marginal,
    quadrature_density,
    quadrature_distribution,
    runtime_scaling,
    squeezing_ellipses,
    wigner_plot,
)

ALPHA = 0.5 + 0.7j
NODE = shutil.which("node") or "C:/Program Files/nodejs/node.exe"


@pytest.fixture(scope="module")
def coherent():
    """The same coherent state from the Fock (Piquasso) and Gaussian backends."""
    pytest.importorskip("piquasso")
    pattern = pg.Pattern(inputs=(0,))
    inputs = {0: pg.GaussianInput.coherent(ALPHA)}
    fock = pg.simulate(pattern, backend="piquasso-fock", cutoff=14, inputs=inputs, seed=0)
    gauss = pg.simulate(pattern, backend="gaussian", inputs=inputs, seed=0)
    return fock.state, gauss.state


def tmsv(r: float) -> pg.GaussianState:
    c, s = np.cosh(2 * r), np.sinh(2 * r)
    z = np.diag([1.0, -1.0])
    covariance = np.block([[c * np.eye(2), s * z], [s * z, c * np.eye(2)]])
    return pg.GaussianState(np.zeros(4), covariance, ("a", "b"))


def test_photon_numbers_are_poisson(coherent):
    fock, _ = coherent
    marginal = photon_number_marginal(fock, 0)
    mean = abs(ALPHA) ** 2
    poisson = [math.exp(-mean) * mean**n / math.factorial(n) for n in marginal]
    np.testing.assert_allclose(list(marginal.values()), poisson, rtol=0, atol=1e-10)
    plot = photon_number_distribution(fock, 0)
    assert plot.traces[0]["y"] == list(marginal.values())
    with pytest.raises(NotImplementedError):
        photon_number_marginal(coherent[1], 0)


@pytest.mark.parametrize("angle", [0.0, math.pi / 3, -1.1, math.pi / 2])
def test_fock_quadrature_density_is_the_shifted_vacuum(coherent, angle):
    fock, gauss = coherent
    x = np.linspace(-7, 9, 801)
    mean = 2 * (ALPHA * np.exp(-1j * angle)).real  # <q cos + p sin> of a coherent state
    exact = np.exp(-((x - mean) ** 2) / 2) / np.sqrt(2 * np.pi)  # vacuum variance 1
    # Truncated (cutoff 14) coherent state: observed max deviation ~1e-8.
    np.testing.assert_allclose(quadrature_density(fock, 0, angle, x), exact, rtol=0, atol=1e-7)
    np.testing.assert_allclose(quadrature_density(gauss, 0, angle, x), exact, rtol=0, atol=1e-14)


def test_quadrature_histogram_uses_recorded_outcomes():
    pattern = pg.Pattern(inputs=(0,)).measure(0, pg.Homodyne(0.4))
    shots = pg.run_shots(pattern, 400, seed=1, backend="gaussian")
    samples = shots.values(0)
    plot = quadrature_distribution(samples=samples, bins=20)
    counts, edges = np.histogram(samples, bins=20, density=True)
    assert plot.traces[0]["y"] == counts.tolist()
    state = pg.GaussianState(np.zeros(2), np.eye(2), (0,))
    both = quadrature_distribution(state, 0, 0.4, samples=samples, bins=20)
    assert both.traces[0]["name"] == "400 recorded outcomes"


def test_wigner_closed_form_matches_fock(coherent):
    fock, gauss = coherent
    q = np.linspace(-5, 6, 89)
    p = np.linspace(-5, 6, 97)
    gaussian = np.array(wigner_plot(gauss, 0, q, p).traces[0]["z"])
    from_fock = np.array(wigner_plot(fock, 0, q, p).traces[0]["z"])
    # Independent implementations (Laguerre matrix elements vs closed form); observed 4e-8.
    np.testing.assert_allclose(gaussian, from_fock, rtol=0, atol=1e-6)
    squeezed = pg.GaussianState(np.zeros(2), np.diag([np.exp(-1.0), np.exp(1.0)]), (0,))
    grid = np.linspace(-12, 12, 601)
    w = gaussian_wigner(squeezed, grid, grid)
    assert w.captured_mass == pytest.approx(1.0, abs=1e-4)
    vacuum = gaussian_wigner(pg.GaussianState(np.zeros(2), np.eye(2), (0,)), [0, 1], [0, 1])
    assert vacuum.values[0, 0] == pytest.approx(1 / (2 * np.pi), rel=1e-15)


def test_covariance_ellipse_and_heatmap():
    r, theta = 0.6, 0.4
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cov = rot @ np.diag([np.exp(-2 * r), np.exp(2 * r)]) @ rot.T
    state = pg.GaussianState([0.5, -1.0], cov, ("m",))
    points = covariance_ellipse(state, "m") - [0.5, -1.0]
    quadratic = np.einsum("ni,ij,nj->n", points, np.linalg.inv(cov), points)
    np.testing.assert_allclose(quadratic, 1.0, rtol=1e-12)
    radii = np.linalg.norm(points, axis=1)
    assert radii.min() == pytest.approx(np.exp(-r), rel=1e-3)
    assert radii.max() == pytest.approx(np.exp(r), rel=1e-4)
    assert len(squeezing_ellipses(state).traces) == 2
    heat = covariance_heatmap(tmsv(0.3))
    assert heat.traces[0]["z"] == tmsv(0.3).covariance.tolist()
    assert heat.traces[0]["x"] == ["q[a]", "p[a]", "q[b]", "p[b]"]


def test_entanglement_diagnostics_closed_form():
    r = 0.45
    plot = entanglement_diagnostics(tmsv(r))
    negativity, entropy = (t["y"] for t in plot.traces)
    np.testing.assert_allclose(negativity, 2 * r / np.log(2), rtol=1e-12)  # E_N(TMSV)=2r/ln2
    nbar = np.sinh(r) ** 2
    reference = (nbar + 1) * np.log(nbar + 1) - nbar * np.log(nbar)  # thermal entropy
    np.testing.assert_allclose(entropy, reference, rtol=1e-10)


def test_measurement_probabilities_marks_postselection():
    pattern = pg.Pattern(inputs=(0, 1)).measure(0, pg.Homodyne.p()).measure(1)
    result = pg.simulate(pattern, backend="gaussian", seed=2, measurement_outcomes={1: 0.3})
    plot = measurement_probabilities(result)
    (trace,) = plot.traces
    assert trace["name"] == "density" and trace["x"] == ["0", "1 (postselected)"]
    assert trace["y"][1] == pytest.approx(np.exp(-0.045) / np.sqrt(2 * np.pi), rel=1e-12)
    assert measurement_probabilities({0: 0.25, 1: 0.75}).traces[0]["y"] == [0.25, 0.75]


@pytest.mark.filterwarnings("ignore:.*retained Fock norm")
def test_optimization_runtime_and_cutoff_plots():
    from photographiqml.training import History

    history = History(losses=[1.0, 0.5, 0.2], validation_losses=[1.1, 0.6, 0.3])
    plot = optimization_curves(history, log=True)
    assert [t["y"] for t in plot.traces] == [history.losses, history.validation_losses]
    with pytest.raises(ValueError):
        runtime_scaling([1, 2], {"fock": [0.1]})
    pytest.importorskip("piquasso")
    pattern = pg.Pattern(inputs=(0,)).append(pg.Squeeze(0, 0.4))
    inputs = {0: pg.FockInput.number(0)}
    study = pg.cutoff_convergence(pattern, (8, 12, 16, 20), inputs=inputs)
    reference = pg.simulate(pattern, backend="piquasso-fock", cutoff=40, inputs=inputs).state
    infidelity = fidelity_vs_cutoff(study, reference).traces[0]["y"]
    assert all(b < a for a, b in zip(infidelity, infidelity[1:]))  # monotone convergence
    assert infidelity[-1] < 1e-6
    branches = SimpleNamespace(  # different outcomes: not one conditional branch
        rows=[{"cutoff": 4}, {"cutoff": 6}],
        results=[SimpleNamespace(outcomes={1: 0.1}), SimpleNamespace(outcomes={1: 0.2})],
    )
    with pytest.raises(ValueError, match="conditional branches"):
        fidelity_vs_cutoff(branches, reference)


def all_plots():
    state = tmsv(0.3)
    return [
        covariance_heatmap(state),
        squeezing_ellipses(state),
        entanglement_diagnostics(state),
        wigner_plot(state.reduced(("a",)), points=31),
        quadrature_distribution(state, "a", 0.0),
        measurement_probabilities({"0": 0.4, "1": 0.6}),
        runtime_scaling([2, 4, 8], {"gaussian": [1e-3, 2e-3, 5e-3]}),
    ]


def test_plot_html_and_javascript(tmp_path):
    for i, plot in enumerate(all_plots()):
        page = plot.to_html()
        assert "plotly.js-dist-min@2.35.2" in page
        figure = json.loads(re.search(r"const FIGURE = (.*);\n", page).group(1))
        assert figure["data"] == json.loads(json.dumps(plot.traces))
        if os.path.exists(NODE):
            code = re.search(r"<script>(const FIGURE.*?)</script>", page, flags=re.S).group(1)
            path = tmp_path / f"plot{i}.js"
            path.write_text(code, encoding="utf-8")
            check = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True)
            assert check.returncode == 0, check.stderr


@pytest.mark.parametrize("suffix", ["svg", "pdf", "png"])
def test_plot_static_exports(tmp_path, suffix):
    pytest.importorskip("matplotlib")
    for i, plot in enumerate(all_plots()):
        content = plot.save(tmp_path / f"plot{i}.{suffix}").read_bytes()
        if suffix == "svg":
            assert ET.fromstring(content).tag.endswith("svg")
        elif suffix == "pdf":
            assert content.startswith(b"%PDF") and content.rstrip().endswith(b"%%EOF")
        else:
            assert content.startswith(b"\x89PNG")
