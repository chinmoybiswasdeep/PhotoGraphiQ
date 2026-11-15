"""Optional CUDA-Q Dynamics adapter at the Hamiltonian level (not a pattern backend).

PhotoGraphiQ gates are written as bosonic Hamiltonians H with U = exp(-i H) (t = 1) in
PhotoGraphiQ conventions: hbar = 2, q = a + a^dag, p = -i (a - a^dag). Each generator is
verified against the ``native-fock`` gate in ``tests/cudaq_adapter/test_cudaq_adapter.py`` (up to
the global phase, which native Gaussian gates do not define). A Hamiltonian can be

* turned into a dense matrix on per-mode local dimensions (:func:`to_matrix`),
* evolved by the NumPy/SciPy reference (:func:`reference_evolve`, unitary or Lindblad),
* translated to ``cudaq.operators`` (:func:`to_cudaq`) and evolved by ``cudaq.evolve``
  (:func:`evolve`). CUDA-Q is imported lazily and never replaced by the reference.

The CUDA-Q calls follow the documentation of CUDA-Q 0.16.0 (docs "latest", read
2026-11-11: https://nvidia.github.io/cuda-quantum/latest/using/dynamics.html and the
Python API page): ``cudaq.operators.boson.create/annihilate/identity(target)``,
``cudaq.evolve(hamiltonian, dimensions, schedule, initial_state, collapse_operators=,
observables=, store_intermediate_results=, integrator=)``, ``cudaq.Schedule(steps,
parameters)``, ``cudaq.State.from_data(array)``, ``cudaq.set_target("dynamics")``.
CUDA-Q publishes Linux wheels only; none of this has been executed on CUDA-Q here.

Truncation: CUDA-Q truncates each mode to a local dimension; PhotoGraphiQ Fock backends
keep occupations with TOTAL photon number < cutoff. :func:`per_mode_to_total_photon` and
:func:`total_photon_to_per_mode` map between the two (native basis order).
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import comb, factorial, prod
from typing import Any

import numpy as np

from .autodiff import _basis as _total_photon_basis  # native basis order (tested)

MIN_CUDAQ_VERSION = (0, 16)
"""Version whose documentation was read; older releases are refused (untested)."""

_LINUX_ONLY = (
    "CUDA-Q publishes Linux (x86_64/aarch64) wheels only (pip install cudaq on Linux or "
    "WSL2); it cannot be installed on native Windows"
)

# A normal-ordered monomial: ((mode, creations, annihilations), ...) sorted by mode,
# meaning prod_mode (a_mode^dag)^creations (a_mode)^annihilations. () is the identity.
Monomial = tuple[tuple[int, int, int], ...]


def _mode_product(left: tuple[int, int], right: tuple[int, int]) -> dict[tuple[int, int], int]:
    """a^dag^c1 a^n1 a^dag^c2 a^n2 = sum_k C(n1,k) C(c2,k) k! a^dag^(c1+c2-k) a^(n1+n2-k)."""
    (c1, n1), (c2, n2) = left, right
    return {
        (c1 + c2 - k, n1 + n2 - k): comb(n1, k) * comb(c2, k) * factorial(k)
        for k in range(min(n1, c2) + 1)
    }


def _monomial_product(left: Monomial, right: Monomial) -> dict[Monomial, int]:
    a = {m: (c, n) for m, c, n in left}
    b = {m: (c, n) for m, c, n in right}
    result: dict[Monomial, int] = {(): 1}
    for mode in sorted(a.keys() | b.keys()):
        local = _mode_product(a.get(mode, (0, 0)), b.get(mode, (0, 0)))
        result = {
            key + (((mode, c, n),) if c or n else ()): weight * w
            for key, weight in result.items()
            for (c, n), w in local.items()
        }
    return result


class BosonicHamiltonian:
    """Sum of complex coefficients times normal-ordered products of ladder operators.

    Modes are nonnegative integers (CUDA-Q degrees). ``terms`` maps a normal-ordered
    monomial ``((mode, creations, annihilations), ...)`` to its coefficient; products are
    normal-ordered on construction with [a_i, a_j^dag] = delta_ij.
    """

    def __init__(self, terms: Mapping[Monomial, complex] | None = None):
        self.terms: dict[Monomial, complex] = {}
        for key, value in (terms or {}).items():
            if value != 0:
                self.terms[key] = self.terms.get(key, 0) + complex(value)

    @classmethod
    def ladder(cls, mode: int, dagger: bool) -> BosonicHamiltonian:
        """a_mode^dag if ``dagger`` else a_mode."""
        if not isinstance(mode, int | np.integer) or mode < 0:
            raise ValueError("Mode must be a nonnegative integer")
        return cls({((int(mode), int(dagger), int(not dagger)),): 1.0})

    @classmethod
    def identity(cls, coefficient: complex = 1.0) -> BosonicHamiltonian:
        return cls({(): coefficient})

    @property
    def modes(self) -> tuple[int, ...]:
        return tuple(sorted({m for key in self.terms for m, _, _ in key}))

    def word(self, key: Monomial) -> tuple[tuple[int, bool], ...]:
        """Ladder-operator word (mode, dagger) of a monomial in normal order."""
        creations = [(m, True) for m, c, _ in key for _ in range(c)]
        return tuple(creations + [(m, False) for m, _, n in key for _ in range(n)])

    def dagger(self) -> BosonicHamiltonian:
        return BosonicHamiltonian(
            {tuple((m, n, c) for m, c, n in key): v.conjugate() for key, v in self.terms.items()}
        )

    def is_hermitian(self, atol: float = 1e-12) -> bool:
        return all(abs(v) <= atol for v in (self - self.dagger()).terms.values())

    def __add__(self, other: object) -> BosonicHamiltonian:
        other = _as_hamiltonian(other)
        merged = dict(self.terms)
        for key, value in other.terms.items():
            merged[key] = merged.get(key, 0) + value
        return BosonicHamiltonian(merged)

    __radd__ = __add__

    def __neg__(self) -> BosonicHamiltonian:
        return -1 * self

    def __sub__(self, other: object) -> BosonicHamiltonian:
        return self + (-1 * _as_hamiltonian(other))

    def __rsub__(self, other: object) -> BosonicHamiltonian:
        return _as_hamiltonian(other) - self

    def __mul__(self, other: object) -> BosonicHamiltonian:
        if isinstance(other, int | float | complex | np.number):
            return BosonicHamiltonian({k: v * complex(other) for k, v in self.terms.items()})
        if not isinstance(other, BosonicHamiltonian):
            return NotImplemented
        out: dict[Monomial, complex] = {}
        for k1, v1 in self.terms.items():
            for k2, v2 in other.terms.items():
                for key, weight in _monomial_product(k1, k2).items():
                    out[key] = out.get(key, 0) + v1 * v2 * weight
        return BosonicHamiltonian(out)

    def __rmul__(self, other: object) -> BosonicHamiltonian:
        return self * other  # scalars commute

    def __pow__(self, power: int) -> BosonicHamiltonian:
        result = BosonicHamiltonian.identity()
        for _ in range(power):
            result = result * self
        return result

    def __repr__(self) -> str:
        return f"BosonicHamiltonian({self.terms!r})"


def _as_hamiltonian(value: object) -> BosonicHamiltonian:
    if isinstance(value, BosonicHamiltonian):
        return value
    if isinstance(value, int | float | complex | np.number):
        return BosonicHamiltonian.identity(complex(value))
    raise TypeError(f"Cannot combine BosonicHamiltonian with {type(value).__name__}")


def create(mode: int) -> BosonicHamiltonian:
    return BosonicHamiltonian.ladder(mode, True)


def annihilate(mode: int) -> BosonicHamiltonian:
    return BosonicHamiltonian.ladder(mode, False)


def number(mode: int) -> BosonicHamiltonian:
    return create(mode) * annihilate(mode)


def position(mode: int) -> BosonicHamiltonian:
    """PhotoGraphiQ q = a + a^dag (hbar = 2; CUDA-Q's ``position`` is half of this)."""
    return annihilate(mode) + create(mode)


def momentum(mode: int) -> BosonicHamiltonian:
    """PhotoGraphiQ p = -i (a - a^dag)."""
    return -1j * (annihilate(mode) - create(mode))


# --- gate generators: U = exp(-i H), conventions of photographiq.backends.native_fock ---


def rotation(mode: int, angle: float) -> BosonicHamiltonian:
    """Rotate: exp(i angle n), so H = -angle n."""
    return -angle * number(mode)


def displacement(mode: int, q: float = 0.0, p: float = 0.0) -> BosonicHamiltonian:
    """Displace by (q, p): D(alpha) = exp(alpha a^dag - alpha* a), alpha = (q + i p)/2."""
    alpha = complex(q, p) / 2
    return 1j * (alpha * create(mode) - alpha.conjugate() * annihilate(mode))


def squeezing(mode: int, r: float) -> BosonicHamiltonian:
    """Squeeze (r > 0 squeezes q): exp(r (a^2 - a^dag^2) / 2)."""
    return 0.5j * r * (annihilate(mode) ** 2 - create(mode) ** 2)


def beamsplitter(u: int, v: int, theta: float) -> BosonicHamiltonian:
    """BeamSplitter, q_u' = c q_u - s q_v: exp(theta (a_u a_v^dag - a_u^dag a_v))."""
    return 1j * theta * (annihilate(u) * create(v) - create(u) * annihilate(v))


def controlled_z(u: int, v: int, weight: float = 1.0) -> BosonicHamiltonian:
    """Entangle (weighted CZ): exp(i g q_u q_v / 2), so H = -g q_u q_v / 2."""
    return -0.5 * weight * position(u) * position(v)


def quadratic_phase(mode: int, s: float) -> BosonicHamiltonian:
    """QuadraticPhase: exp(i s q^2 / 4), so H = -s q^2 / 4."""
    return -0.25 * s * position(mode) ** 2


def kerr(mode: int, kappa: float) -> BosonicHamiltonian:
    """Kerr: exp(i kappa n^2), so H = -kappa n^2."""
    return -kappa * number(mode) ** 2


def cross_kerr(u: int, v: int, kappa: float) -> BosonicHamiltonian:
    """Cross-Kerr (backend method only): exp(i kappa n_u n_v)."""
    return -kappa * number(u) * number(v)


def cubic_phase(mode: int, gamma: float) -> BosonicHamiltonian:
    """CubicPhase: exp(i gamma q^3 / 6), so H = -gamma q^3 / 6."""
    return -gamma / 6 * position(mode) ** 3


# --- dense matrices and truncation mapping ---------------------------------------------


def _dims(dimensions: Mapping[int, int], hamiltonians: Iterable[BosonicHamiltonian] = ()):
    modes = tuple(sorted(dimensions))
    if any(not isinstance(dimensions[m], int | np.integer) or dimensions[m] < 1 for m in modes):
        raise ValueError("Local dimensions must be positive integers")
    for h in hamiltonians:
        missing = set(h.modes) - set(modes)
        if missing:
            raise ValueError(f"No local dimension given for mode(s) {sorted(missing)}")
    return modes, tuple(int(dimensions[m]) for m in modes)


def to_matrix(hamiltonian: BosonicHamiltonian, dimensions: Mapping[int, int]) -> np.ndarray:
    """Dense matrix on the tensor product of local dimensions ``{mode: d}``.

    Modes are ordered by increasing label, the lowest label being the most significant
    index (``np.kron`` order, as in the native occupation tuples). Each normal-ordered
    monomial equals its exact projected matrix elements.
    """
    modes, dims = _dims(dimensions, [hamiltonian])
    lowering = {m: np.diag(np.sqrt(np.arange(1, d)), 1) for m, d in zip(modes, dims, strict=True)}
    matrix = np.zeros((prod(dims), prod(dims)), dtype=complex)
    for key, value in hamiltonian.terms.items():
        powers = {m: (c, n) for m, c, n in key}
        factor = np.ones((1, 1), dtype=complex)
        for m in modes:
            c, n = powers.get(m, (0, 0))
            a = lowering[m]
            factor = np.kron(factor, np.linalg.matrix_power(a.T, c) @ np.linalg.matrix_power(a, n))
        matrix += value * factor
    return matrix


def _total_photon_indices(dimensions: Mapping[int, int], cutoff: int) -> np.ndarray:
    _, dims = _dims(dimensions)
    if any(d < cutoff for d in dims):
        raise ValueError(f"Every local dimension must be >= cutoff={cutoff} to hold the basis")
    basis = _total_photon_basis(len(dims), cutoff)
    return np.ravel_multi_index(np.array(basis, dtype=int).T.reshape(len(dims), -1), dims)


def per_mode_to_total_photon(
    state: np.ndarray, dimensions: Mapping[int, int], cutoff: int
) -> np.ndarray:
    """Project a per-mode vector or density matrix onto sum(n) < cutoff (native order).

    The result is not renormalised; its norm (trace) is the retained weight.
    """
    index = _total_photon_indices(dimensions, cutoff)
    state = np.asarray(state)
    return state[index] if state.ndim == 1 else state[np.ix_(index, index)]


def total_photon_to_per_mode(
    state: np.ndarray, dimensions: Mapping[int, int], cutoff: int
) -> np.ndarray:
    """Embed a native total-photon vector or density matrix into per-mode dimensions."""
    index = _total_photon_indices(dimensions, cutoff)
    state = np.asarray(state, dtype=complex)
    if state.shape[0] != index.size:
        raise ValueError(f"State has dimension {state.shape[0]}, basis has {index.size}")
    size = prod(int(d) for d in dimensions.values())
    if state.ndim == 1:
        out = np.zeros(size, dtype=complex)
        out[index] = state
    else:
        out = np.zeros((size, size), dtype=complex)
        out[np.ix_(index, index)] = state
    return out


# --- evolution ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvolutionResult:
    """Final state, optional per-time states and expectation values, and how it ran.

    ``record`` holds ``engine`` ('cudaq' or 'numpy-reference'), the CUDA-Q version and
    the target/simulator/platform reported by CUDA-Q, and ``verified`` (always False:
    the CUDA-Q path has not been validated against the reference on a CUDA-Q install).
    """

    state: np.ndarray
    times: tuple[float, ...]
    intermediate_states: tuple[np.ndarray, ...] | None
    expectation_values: np.ndarray | None
    record: dict = field(default_factory=dict)


def _validate_evolution(hamiltonian, dimensions, times, initial_state, collapse, observables):
    _dims(dimensions, [hamiltonian, *collapse, *observables])
    times = tuple(float(t) for t in times)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("times must contain at least two strictly increasing values")
    size = prod(int(d) for d in dimensions.values())
    state = np.asarray(initial_state, dtype=complex)
    if state.shape not in ((size,), (size, size)):
        raise ValueError(f"initial_state must have shape ({size},) or ({size}, {size})")
    if not hamiltonian.is_hermitian():
        raise ValueError("Hamiltonian is not Hermitian")
    if collapse and state.ndim == 1:
        state = np.outer(state, state.conj())
    return times, state


def _lindbladian(h: np.ndarray, jumps: Sequence[np.ndarray]):
    from scipy import sparse

    eye = sparse.eye_array(h.shape[0], dtype=complex, format="csr")
    h = sparse.csr_array(h)
    # Row-major vec: vec(A rho B) = (A kron B^T) vec(rho).
    out = -1j * (sparse.kron(h, eye) - sparse.kron(eye, h.T))
    for j in map(sparse.csr_array, jumps):
        jj = j.conj().T @ j
        out = out + sparse.kron(j, j.conj()) - 0.5 * (sparse.kron(jj, eye) + sparse.kron(eye, jj.T))
    return sparse.csr_array(out)


def reference_evolve(
    hamiltonian: BosonicHamiltonian,
    dimensions: Mapping[int, int],
    times: Sequence[float],
    initial_state: np.ndarray,
    *,
    collapse_operators: Sequence[BosonicHamiltonian] = (),
    observables: Sequence[BosonicHamiltonian] = (),
    store_intermediate_results: bool = False,
) -> EvolutionResult:
    """NumPy/SciPy reference with the interface of :func:`evolve`.

    Solves d psi/dt = -i H psi, or the Lindblad equation d rho/dt = -i[H, rho] +
    sum_k (L_k rho L_k^dag - {L_k^dag L_k, rho}/2) when ``collapse_operators`` are given
    (loss at rate gamma: ``gamma**0.5 * annihilate(mode)``), by
    ``scipy.sparse.linalg.expm_multiply`` between consecutive ``times`` (H is
    time-independent). States are in :func:`to_matrix` order.
    """
    from scipy.sparse.linalg import expm_multiply

    collapse, observables = tuple(collapse_operators), tuple(observables)
    steps, state = _validate_evolution(
        hamiltonian, dimensions, times, initial_state, collapse, observables
    )
    h = to_matrix(hamiltonian, dimensions)
    mixed = state.ndim == 2
    if mixed:
        generator = _lindbladian(h, [to_matrix(c, dimensions) for c in collapse])
    else:
        generator = -1j * h
    ops = [to_matrix(o, dimensions) for o in observables]

    def expect(s):
        rho = s if s.ndim == 2 else np.outer(s, s.conj())
        return [np.trace(o @ rho) for o in ops]

    states, values = [state], [expect(state)]
    vector = state.reshape(-1)
    for dt in np.diff(steps):
        vector = expm_multiply(generator * dt, vector)
        states.append(vector.reshape(state.shape))
        values.append(expect(states[-1]))
    return EvolutionResult(
        state=states[-1],
        times=steps,
        intermediate_states=tuple(states) if store_intermediate_results else None,
        expectation_values=(
            (np.array(values) if store_intermediate_results else np.array(values[-1]))
            if ops
            else None
        ),
        record={"engine": "numpy-reference", "device": "cpu", "verified": True},
    )


# --- CUDA-Q ------------------------------------------------------------------------------


def cudaq_probe() -> dict:
    """Try ``import cudaq`` and report availability, version and the import error."""
    try:
        cudaq = importlib.import_module("cudaq")
    except Exception as exc:  # report any import failure, never raise from a probe
        return {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "version": getattr(cudaq, "__version__", None), "error": None}


def cudaq_capabilities() -> dict:
    """What this adapter translates, and the documented CUDA-Q features it relies on.

    Returns:
        result (dict): ``installed`` (cudaq spec found), ``verified`` (False: nothing was
            run on CUDA-Q), ``documented`` CUDA-Q features, ``adapter`` features of this
            module and ``not_provided`` (pattern execution, measurement conditioning).
    """
    try:
        installed = importlib.util.find_spec("cudaq") is not None
    except (ImportError, ValueError):
        installed = False
    return {
        "installed": installed,
        "verified": False,
        "documented_version": ".".join(map(str, MIN_CUDAQ_VERSION)),
        "documented": {
            "bosonic_operators": "cudaq.operators.boson: create, annihilate, number, "
            "identity, position, momentum",
            "truncation": "per-mode local dimension via evolve(..., dimensions={mode: d})",
            "unitary_evolution": "cudaq.evolve with a pure initial cudaq.State",
            "density_matrix_evolution": "cudaq.evolve with a density-matrix initial state",
            "open_system_dynamics": "Lindblad master equation via evolve(collapse_operators=)",
            "time_dependence": "cudaq.Schedule with time-dependent operator callbacks",
            "target": "cudaq.set_target('dynamics')",
        },
        "adapter": [
            "gate generators (rotation, displacement, squeezing, beamsplitter, controlled_z, "
            "quadratic_phase, kerr, cross_kerr, cubic_phase)",
            "to_cudaq operator translation",
            "evolve (unitary, Lindblad loss) with reference_evolve as the NumPy reference",
            "per-mode <-> total-photon truncation mapping",
        ],
        "not_provided": ["pattern execution", "homodyne/photon-counting conditioning"],
    }


def _require_cudaq() -> Any:
    try:
        cudaq = importlib.import_module("cudaq")
    except ImportError as exc:
        raise ImportError(f"The CUDA-Q adapter requires cudaq ({exc}); {_LINUX_ONLY}.") from exc
    version = str(getattr(cudaq, "__version__", ""))
    found = re.search(r"(\d+)\.(\d+)", version)
    if found is None or tuple(map(int, found.groups())) < MIN_CUDAQ_VERSION:
        raise ImportError(
            f"cudaq {version or '(unknown version)'} found; the adapter follows CUDA-Q "
            f">= {'.'.join(map(str, MIN_CUDAQ_VERSION))} and refuses older releases"
        )
    return cudaq


def to_cudaq(hamiltonian: BosonicHamiltonian) -> Any:
    """Build the equivalent ``cudaq.operators`` boson expression.

    Each normal-ordered monomial becomes ``coefficient * create(m) * ... * annihilate(m)``
    with ``cudaq.operators.boson``; the constant term uses ``boson.identity()``.

    Raises:
        ImportError: cudaq is missing or older than :data:`MIN_CUDAQ_VERSION`.
    """
    boson = _require_cudaq().operators.boson
    result = None
    for key, value in hamiltonian.terms.items():
        term = boson.identity()
        for mode, dagger in hamiltonian.word(key):
            term = term * (boson.create(mode) if dagger else boson.annihilate(mode))
        term = value * term
        result = term if result is None else result + term
    return boson.empty() if result is None else result


def evolve(
    hamiltonian: BosonicHamiltonian,
    dimensions: Mapping[int, int],
    times: Sequence[float],
    initial_state: np.ndarray,
    *,
    collapse_operators: Sequence[BosonicHamiltonian] = (),
    observables: Sequence[BosonicHamiltonian] = (),
    store_intermediate_results: bool = False,
    integrator: Any = None,
) -> EvolutionResult:
    """Evolve with ``cudaq.evolve`` on the ``dynamics`` target; same interface as
    :func:`reference_evolve`. There is no fallback: without CUDA-Q this raises.

    The initial state is passed through ``cudaq.State.from_data`` in :func:`to_matrix`
    order. Whether CUDA-Q orders degrees the same way is checked at run time by
    comparing ``to_cudaq(probe).to_matrix(dimensions)`` with :func:`to_matrix`; a
    mismatch raises instead of returning a permuted state.

    Raises:
        ImportError: cudaq missing (Linux wheels only) or too old.
        RuntimeError: CUDA-Q's degree ordering differs from this module's.
    """
    cudaq = _require_cudaq()
    collapse, observables = tuple(collapse_operators), tuple(observables)
    steps, state = _validate_evolution(
        hamiltonian, dimensions, times, initial_state, collapse, observables
    )
    probe = BosonicHamiltonian()  # distinct incommensurate weights fix the mode order
    for k, m in enumerate(sorted(dimensions)):
        probe = probe + (np.sqrt(2.0) + k) ** -1 * number(m)
    cudaq_matrix = np.asarray(to_cudaq(probe).to_matrix(dict(dimensions)), dtype=complex)
    if not np.allclose(cudaq_matrix, to_matrix(probe, dimensions), atol=1e-12):
        raise RuntimeError("CUDA-Q degree ordering differs from photographiq.cudaq_dynamics")
    cudaq.set_target("dynamics")
    target = cudaq.get_target()
    result = cudaq.evolve(
        to_cudaq(hamiltonian),
        dict(dimensions),
        cudaq.Schedule(list(steps), ["t"]),
        cudaq.State.from_data(np.ascontiguousarray(state)),
        collapse_operators=[to_cudaq(c) for c in collapse],
        observables=[to_cudaq(o) for o in observables],
        store_intermediate_results=(
            cudaq.IntermediateResultSave.ALL
            if store_intermediate_results
            else cudaq.IntermediateResultSave.NONE
        ),
        integrator=integrator,
    )

    def array(s):
        return np.asarray(s.to_numpy(), dtype=complex).reshape(state.shape)

    values = None
    if observables:
        if store_intermediate_results:
            values = np.array(
                [[v.expectation() for v in step] for step in result.expectation_values()]
            )
        else:
            values = np.array([v.expectation() for v in result.final_expectation_values()])
    return EvolutionResult(
        state=array(result.final_state()),
        times=steps,
        intermediate_states=(
            tuple(array(s) for s in result.intermediate_states())
            if store_intermediate_results
            else None
        ),
        expectation_values=values,
        record={
            "engine": "cudaq",
            "cudaq_version": str(cudaq.__version__),
            "target": getattr(target, "name", None),
            "simulator": getattr(target, "simulator", None),
            "platform": getattr(target, "platform", None),
            "description": getattr(target, "description", None),
            "available_gpus": getattr(cudaq, "num_available_gpus", lambda: None)(),
            "verified": False,
        },
    )
