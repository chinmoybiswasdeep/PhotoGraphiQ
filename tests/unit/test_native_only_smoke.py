"""Core and native execution with every optional package made unimportable.

A subprocess installs a meta-path blocker for Piquasso, Graphix, JAX, mpmath, CuPy, MentPy
and CUDA-Q (as if they were not installed), then imports ``photographiq`` and
``photographiqml`` and runs the native Gaussian, Fock and mixed-Fock pathways. Missing
optional packages must give errors that name the extra to install.
"""

import subprocess
import sys
import textwrap

import pytest

from photographiq import native

pytestmark = pytest.mark.skipif(
    not {"gaussian", "fock"} <= set(native.components()),
    reason="native gaussian/fock components not built",
)

SCRIPT = textwrap.dedent(
    """
    import importlib.abc, importlib.util, sys
    BLOCK = {"piquasso", "graphix", "jax", "jaxlib", "mpmath", "cupy", "cupyx", "mentpy",
             "cudaq"}

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCK:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    _find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda n, p=None: (
        None if n.split(".")[0] in BLOCK else _find_spec(n, p)
    )
    sys.meta_path.insert(0, Blocker())

    import numpy as np
    import photographiq as pg
    import photographiqml  # noqa: F401
    from photographiq import sampling

    assert set(pg.available_backends()) >= {"gaussian", "native-gaussian", "native-fock",
                                            "native-mixed-fock"}
    assert not any(b.startswith("piquasso") for b in pg.available_backends())

    wire = pg.Pattern(pg.CVGraph.line(2, squeezing=1.0, inputs=(0,))).measure(0, pg.Homodyne.p())
    coherent = {0: pg.GaussianInput.coherent(0.3 + 0.2j)}
    a = pg.simulate(wire, inputs=coherent, backend="gaussian", seed=3)
    b = pg.simulate(wire, inputs=coherent, backend="native-gaussian", seed=3)
    assert np.allclose(a.state.covariance, b.state.covariance)

    herald = pg.non_gaussian.photon_subtraction(theta=0.2)
    fock = pg.simulate(herald, inputs={"in": pg.FockInput.number(2)}, backend="native-fock",
                       cutoff=8, measurement_outcomes={"count": 1})
    assert np.isclose(fock.state.photon_number("in"), 1.0)
    mixed = pg.simulate(herald, inputs={"in": pg.FockInput.number(2)},
                        backend="native-mixed-fock", cutoff=8,
                        measurement_outcomes={"count": 1})
    assert np.isclose(mixed.state.photon_number("in"), 1.0)

    def message(call):
        try:
            call()
        except ImportError as exc:
            return str(exc)
        raise AssertionError("expected ImportError")

    assert "photographiq[piquasso]" in message(
        lambda: pg.simulate(herald, inputs={"in": pg.FockInput.number(2)},
                            backend="piquasso-fock", cutoff=8,
                            measurement_outcomes={"count": 1}))
    state = sampling.squeezed_interferometer_state([0.5, 0.5], np.eye(2))
    assert "photographiq[precision]" in message(
        lambda: sampling.threshold_probability(state, [1, 1], precision=30))
    import photographiq.gpu as gpu
    assert "cupy" in message(gpu.require).lower()
    leaked = sorted(m for m in BLOCK if m in sys.modules)
    assert not leaked, leaked
    print("NATIVE-ONLY OK")
    """
)


def test_core_and_native_paths_run_without_optional_packages(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "NATIVE-ONLY OK" in completed.stdout
