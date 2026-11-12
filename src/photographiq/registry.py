"""Backend capability registry and allocation-free backend planner.

Each registered backend is described by a :class:`BackendInfo`: an import path
``"module:Class"``, the state representation it evolves, device and precision,
declared numerical approximations and its requirements (optional Python packages
and compiled native components). Capabilities are read from the backend class
attribute ``capabilities``; they are never duplicated here.

Nothing heavy is imported when this module is imported. ``available()`` checks
package specs with :func:`importlib.util.find_spec`, queries
:mod:`photographiq.native` for compiled components and only then imports the
backend module itself (not its optional dependencies, which are lazy).

:func:`select_backend` returns the cheapest available backend whose capabilities
cover :func:`photographiq.capabilities.required_features` of a pattern and that
passes the same allocation-free ``preflight`` used by ``simulate``. It never picks a
backend lacking a required feature; if none qualifies it raises with the reasons.
Cost order: representation (gaussian < pure-fock < mixed-fock), then the ``prefer``
order of implementations, then registration order.

CUDA-Q is not a registered backend: it cannot execute patterns. It is available as the
optional Hamiltonian-level adapter module :mod:`photographiq.cudaq_dynamics`.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any

from .capabilities import CORRELATED_GAUSSIAN_STATE, MIXED_GAUSSIAN_INPUT, required_features

REPRESENTATIONS = ("gaussian", "pure-fock", "mixed-fock")

_TOTAL_TRUNCATION = "Fock truncation by total photon number sum(n) < cutoff"
_RENORMALISATION = (
    "renormalisation after each operation (retained norm recorded; error above norm_tolerance)"
)


@dataclass(frozen=True)
class BackendInfo:
    """Static description of one registered backend.

    Args:
        name: Registry name accepted by ``simulate(backend=...)``.
        import_path: ``"module:Class"`` of the backend class.
        representation: 'gaussian', 'pure-fock' or 'mixed-fock'.
        implementation: Engine family used by the ``prefer`` ordering
            ('numpy', 'piquasso', 'native').
        device: 'cpu' or 'gpu'.
        precision: Declared floating-point type of the state data.
        differentiable: Whether gradients flow through the backend itself. All
            trajectory backends are False; ``photographiq.autodiff`` is a separate
            JAX realization.
        uses_cutoff: Constructor takes ``cutoff=`` (required for Fock backends).
        approximations: Numerical approximations beyond float roundoff.
        python_packages: Importable top-level packages required at run time.
        native_components: Compiled ``photographiq._native`` components required.
    """

    name: str
    import_path: str
    representation: str
    implementation: str
    device: str = "cpu"
    precision: str = "float64"
    differentiable: bool = False
    uses_cutoff: bool = False
    approximations: tuple[str, ...] = ()
    python_packages: tuple[str, ...] = ()
    native_components: tuple[str, ...] = ()

    def __post_init__(self):
        if self.representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation {self.representation!r}")
        if self.device not in ("cpu", "gpu"):
            raise ValueError(f"Unknown device {self.device!r}")

    def unavailable_reason(self) -> str | None:
        """Why the backend cannot be used here, or None when it is available."""
        for package in self.python_packages:
            if not _importable(package):
                return f"Python package {package!r} is not installed"
        if self.native_components:
            try:
                from . import native
            except ImportError as exc:  # e.g. a broken PHOTOGRAPHIQ_NATIVE_PATH build
                return f"photographiq._native failed to load ({exc})"
            if not native.AVAILABLE:
                return f"photographiq._native is not available ({native.load_error})"
            missing = [c for c in self.native_components if c not in native.components()]
            if missing:
                return f"native component(s) {', '.join(missing)} not compiled"
        try:
            self.load()
        except ImportError as exc:
            return f"cannot import {self.import_path}: {exc}"
        return None

    def available(self) -> bool:
        """Whether requirements are met and the backend class imports."""
        return self.unavailable_reason() is None

    def load(self) -> type[Any]:
        """Import and return the backend class (raises ImportError)."""
        module_name, _, class_name = self.import_path.partition(":")
        module = importlib.import_module(module_name)
        try:
            return getattr(module, class_name)
        except AttributeError as exc:
            raise ImportError(f"{module_name} has no attribute {class_name!r}") from exc

    @property
    def capabilities(self) -> frozenset[str]:
        """Capabilities from the backend class attribute (raises ImportError)."""
        return frozenset(self.load().capabilities)


def _info(*args, **kwargs) -> tuple[str, BackendInfo]:
    info = BackendInfo(*args, **kwargs)
    return info.name, info


_REGISTRY: dict[str, BackendInfo] = dict(
    [
        _info(
            "gaussian",
            "photographiq.backends.gaussian:GaussianBackend",
            "gaussian",
            "numpy",
        ),
        _info(
            "piquasso",
            "photographiq.backends.piquasso:PiquassoBackend",
            "gaussian",
            "piquasso",
            python_packages=("piquasso",),
        ),
        _info(
            "piquasso-fock",
            "photographiq.backends.fock:PiquassoFockBackend",
            "pure-fock",
            "piquasso",
            precision="complex128",
            uses_cutoff=True,
            approximations=(_TOTAL_TRUNCATION, _RENORMALISATION),
            python_packages=("piquasso",),
        ),
        _info(
            "piquasso-mixed-fock",
            "photographiq.backends.mixed_fock:MixedFockBackend",
            "mixed-fock",
            "piquasso",
            precision="complex128",
            uses_cutoff=True,
            approximations=(_TOTAL_TRUNCATION, _RENORMALISATION),
            python_packages=("piquasso",),
        ),
        _info(
            "native-gaussian",
            "photographiq.backends.native_gaussian:NativeGaussianBackend",
            "gaussian",
            "native",
            native_components=("gaussian",),
        ),
        _info(
            "native-fock",
            "photographiq.backends.native_fock:NativeFockBackend",
            "pure-fock",
            "native",
            precision="complex128",
            uses_cutoff=True,
            approximations=(_TOTAL_TRUNCATION,),
            native_components=("fock",),
        ),
        _info(
            "native-mixed-fock",
            "photographiq.backends.native_fock:NativeMixedFockBackend",
            "mixed-fock",
            "native",
            precision="complex128",
            uses_cutoff=True,
            approximations=(_TOTAL_TRUNCATION,),
            native_components=("fock",),
        ),
    ]
)

BACKENDS = MappingProxyType(_REGISTRY)
"""Read-only mapping of registry name to :class:`BackendInfo`."""


def _get(name) -> BackendInfo:
    if name == "cudaq-dynamics":
        raise ValueError(
            "Unknown backend: 'cudaq-dynamics'. CUDA-Q cannot execute patterns; it is "
            "available as the Hamiltonian-level adapter module photographiq.cudaq_dynamics"
        )
    if not isinstance(name, str) or name not in _REGISTRY:
        raise ValueError(f"Unknown backend: {name!r}")
    return _REGISTRY[name]


def available_backends() -> tuple[str, ...]:
    """Names of registered backends usable in this environment, in registration order."""
    return tuple(name for name, info in _REGISTRY.items() if info.available())


def describe_backend(name: str) -> dict:
    """Return the registry entry plus availability and class capabilities.

    ``capabilities`` is a sorted list read from the class, or None when the class
    cannot be imported (``unavailable_reason`` then says why).
    """
    info = _get(name)
    reason = info.unavailable_reason()
    try:
        capabilities = sorted(info.capabilities)
    except ImportError:
        capabilities = None
    return {
        **asdict(info),
        "available": reason is None,
        "unavailable_reason": reason,
        "capabilities": capabilities,
    }


def create_backend(name: str, cutoff: int | None = None):
    """Instantiate a registered backend; ``cutoff`` is passed only to Fock-type backends.

    Raises:
        ValueError: Unknown backend name.
        ImportError: Backend module or class cannot be imported.
    """
    info = _get(name)
    missing = [p for p in info.python_packages if not _importable(p)]
    if missing:
        hint = "; ".join(_INSTALL_HINTS.get(p, f"install {p!r}") for p in missing)
        raise ImportError(f"Backend {name!r} is unavailable: {hint}")
    try:
        cls = info.load()
    except ImportError as exc:
        raise ImportError(f"Backend {name!r} is unavailable: {exc}") from exc
    return cls(cutoff=cutoff) if info.uses_cutoff else cls()


_INSTALL_HINTS = {
    "piquasso": "requires the optional dependency 'piquasso': pip install 'photographiq[piquasso]'",
}


def _importable(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def missing_features(name: str, features) -> frozenset[str]:
    """Features in ``features`` that backend ``name`` does not provide.

    Derived requirements (see :mod:`photographiq.capabilities`) are decided from the
    class: mixed Gaussian inputs need a non-Fock or a mixed-Fock backend, correlated
    Gaussian initial states need a ``GaussianBackend`` subclass.
    """
    from .backends.gaussian import GaussianBackend

    cls = _get(name).load()
    capabilities = frozenset(cls.capabilities)
    derived = {
        MIXED_GAUSSIAN_INPUT: "fock_input" not in capabilities or "mixed_fock" in capabilities,
        CORRELATED_GAUSSIAN_STATE: issubclass(cls, GaussianBackend),
    }
    return frozenset(f for f in features if f not in capabilities and not derived.get(f, False))


@dataclass(frozen=True)
class BackendSelection:
    """Outcome of :func:`select_backend`.

    Args:
        name: Selected registry name.
        representation: State representation of the selection.
        features: Required features of the pattern.
        explanation: Human-readable reason for the choice.
        rejected: Cheaper-or-equal candidates that were skipped, with reasons.
    """

    name: str
    representation: str
    features: frozenset[str]
    explanation: str
    rejected: dict[str, str] = field(default_factory=dict)


def _rank(prefer: tuple[str, ...]):
    order = list(_REGISTRY)

    def key(info: BackendInfo):
        implementation = (
            prefer.index(info.implementation) if info.implementation in prefer else len(prefer)
        )
        return REPRESENTATIONS.index(info.representation), implementation, order.index(info.name)

    return key


def select_backend(
    pattern,
    inputs=None,
    *,
    initial_state=None,
    measurement_outcomes=None,
    prefer: tuple[str, ...] = ("native", "numpy"),
    cutoff: int | None = None,
) -> BackendSelection:
    """Pick the cheapest available backend able to execute ``pattern``.

    Candidates are tried in cost order. A candidate is accepted only if it is
    available, its class capabilities cover every required feature, a cutoff is
    supplied when it needs one, and ``preflight`` (resource support versus cutoff,
    measurement/backend compatibility) passes on a freshly constructed instance.

    Args:
        pattern (Pattern): Measurement pattern.
        inputs (dict): Individual input descriptions keyed by input labels.
        initial_state (object): Correlated initial state, as for ``simulate``.
        measurement_outcomes (dict): Postselected outcomes keyed by result key.
        prefer (tuple): Implementation families in preference order within one
            representation; families not listed come afterwards.
        cutoff (int): Exclusive total-photon cutoff; Fock backends are skipped without it.

    Returns:
        result (BackendSelection): Chosen backend and explanation.

    Raises:
        ValueError: Postselection key does not identify a measurement.
        NotImplementedError: No available backend covers the required features;
            the message lists each candidate's missing features or other reason.
    """
    from .capabilities import preflight

    inputs = dict(inputs or {})
    features = required_features(pattern, inputs, initial_state, measurement_outcomes)
    rejected: dict[str, str] = {}
    for info in sorted(_REGISTRY.values(), key=_rank(tuple(prefer))):
        reason = info.unavailable_reason()
        if reason is not None:
            rejected[info.name] = f"unavailable: {reason}"
            continue
        missing = missing_features(info.name, features)
        if missing:
            rejected[info.name] = f"missing {', '.join(sorted(missing))}"
            continue
        if info.uses_cutoff and cutoff is None:
            rejected[info.name] = "needs an explicit cutoff"
            continue
        try:
            engine = create_backend(info.name, cutoff)
            preflight(engine, pattern, inputs, initial_state, measurement_outcomes)
        except (NotImplementedError, ValueError, MemoryError) as exc:
            rejected[info.name] = f"preflight: {exc}"
            continue
        explanation = (
            f"{info.name!r}: cheapest available {info.representation} backend covering "
            f"[{', '.join(sorted(features)) or 'no special features'}]"
        )
        return BackendSelection(info.name, info.representation, features, explanation, rejected)
    details = "; ".join(f"{name}: {why}" for name, why in rejected.items())
    raise NotImplementedError(
        f"No available backend supports [{', '.join(sorted(features))}] ({details})"
    )
