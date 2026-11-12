"""CUDA-Q Dynamics probe. No PhotoGraphiQ execution is implemented on CUDA-Q.

``cudaq`` ships Linux (x86_64/aarch64) wheels; it is not installable on native
Windows (use WSL2). In the development environment ``import cudaq`` fails with
``ModuleNotFoundError: No module named 'cudaq'``.

:func:`cudaq_capabilities` summarises what the documented CUDA-Q Dynamics API
provides (https://nvidia.github.io/cuda-quantum/latest/using/dynamics.html).
These entries are documentation, not verified behaviour: nothing here has been run
against CUDA-Q. Convention warnings for any future translation:

* CUDA-Q truncates each bosonic mode to a local dimension (``dimensions={i: d}``),
  whereas PhotoGraphiQ Fock backends truncate the TOTAL photon number sum(n) < cutoff.
* PhotoGraphiQ uses hbar=2, q = a + a^dag; CUDA-Q documents position = (a^dag + a)/2,
  so translate gates through ladder operators, not through CUDA-Q position/momentum.
"""

from __future__ import annotations

import importlib
import importlib.util

_UNSUPPORTED = (
    "CUDA-Q Dynamics execution of PhotoGraphiQ patterns is not implemented: no "
    "gate-to-Hamiltonian translation or measurement conditioning has been validated."
)


def cudaq_probe() -> dict:
    """Try ``import cudaq`` and report availability, version and the import error."""
    try:
        cudaq = importlib.import_module("cudaq")
    except Exception as exc:  # report any import failure, never raise from a probe
        return {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "version": getattr(cudaq, "__version__", None), "error": None}


def cudaq_capabilities() -> dict:
    """Documented CUDA-Q Dynamics features relevant to bosonic simulation.

    Returns:
        result (dict): ``installed`` (cudaq importable spec found) and ``documented``
            features with the public API that provides them. Unverified here.
    """
    try:
        installed = importlib.util.find_spec("cudaq") is not None
    except (ImportError, ValueError):
        installed = False
    return {
        "installed": installed,
        "verified": False,
        "documented": {
            "bosonic_operators": "cudaq.operators.boson: create, annihilate, number, "
            "position, momentum",
            "truncation": "per-mode local dimension via evolve(..., dimensions={mode: d})",
            "unitary_evolution": "cudaq.evolve with a pure initial cudaq.State",
            "density_matrix_evolution": "cudaq.evolve with a density-matrix initial state",
            "open_system_dynamics": "Lindblad master equation via evolve(collapse_operators=)",
            "time_dependence": "cudaq.Schedule with time-dependent operator callbacks",
            "target": "cudaq.set_target('dynamics') (GPU, cuQuantum/cuDensityMat)",
        },
        "not_provided_by_photographiq": [
            "pattern execution",
            "homodyne/photon-counting conditioning",
            "total-photon-number truncation",
        ],
    }


class CudaqDynamicsBackend:
    """Placeholder registered as 'cudaq-dynamics'; construction always raises.

    Raises:
        ImportError: ``cudaq`` is not importable (with the probe's error).
        NotImplementedError: ``cudaq`` is importable but execution is not implemented.
    """

    capabilities: frozenset[str] = frozenset()

    def __init__(self, cutoff: int | None = None):
        probe = cudaq_probe()
        if not probe["available"]:
            raise ImportError(
                f"The cudaq-dynamics backend requires CUDA-Q ({probe['error']}); CUDA-Q "
                "provides Linux wheels only (pip install cudaq; on Windows use WSL2)."
            )
        raise NotImplementedError(_UNSUPPORTED)
