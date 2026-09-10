"""Optional JAX execution with fixed topology and explicit gradient semantics.

Fock evolution uses projected finite-dimensional generators. It is a separate
numerical realization from native Piquasso; both cutoff and gradients must converge.
No JAX import occurs until a differentiable operation is requested.
"""

from itertools import product
from typing import NamedTuple

import numpy as np

from . import commands as c
from .expressions import CallableExpression, Expr


def _jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "Automatic differentiation requires pip install 'photographiq[autodiff]' (or source extras)"
        ) from exc
    if not jax.config.x64_enabled:
        raise RuntimeError("Enable jax_enable_x64 before execution for scientific precision")
    return jax, jnp


def resolve(value, parameters, records=None):
    """Evaluate PhotoGraphiQ expressions without converting JAX tracers to floats."""
    _, jnp = _jax()
    records = {} if records is None else records
    if isinstance(value, CallableExpression) or callable(value):
        raise TypeError("Differentiable execution requires expression trees, not Python callables")
    if not isinstance(value, Expr):
        return jnp.asarray(value)
    if value.op == "constant":
        return jnp.asarray(value.args[0])
    if value.op == "parameter":
        return parameters[value.args[0]]
    if value.op == "outcome":
        result = records[value.args[0]]
        return result if len(value.args) == 1 else result[value.args[1]]
    ops = {
        "add": jnp.add,
        "sub": jnp.subtract,
        "mul": jnp.multiply,
        "div": jnp.divide,
        "pow": jnp.power,
        "neg": jnp.negative,
        "sin": jnp.sin,
        "cos": jnp.cos,
        "exp": jnp.exp,
        "atan2": jnp.arctan2,
    }
    return ops[value.op](*(resolve(a, parameters, records) for a in value.args))


def _basis(modes, cutoff):
    if modes == 0:
        return ((),)

    def compositions(total, size):
        if size == 1:
            yield (total,)
        else:
            for first in range(total, -1, -1):
                for rest in compositions(total - first, size - 1):
                    yield (first,) + rest

    return tuple(b for n in range(cutoff) for b in compositions(n, modes))


def _operators(basis, mode):
    lookup = {b: i for i, b in enumerate(basis)}
    a = np.zeros((len(basis), len(basis)), complex)
    for j, b in enumerate(basis):
        target_list = list(b)
        target_list[mode] -= 1
        if b[mode]:
            a[lookup[tuple(target_list)], j] = np.sqrt(b[mode])
    return a, a + a.conj().T, -1j * (a - a.conj().T)


class DifferentiableState(NamedTuple):
    """Conditional state, branch likelihood and finite-space diagnostics.

    retained_norms records preparation projection masses before normalization.
    boundary_population is the final weight in the top two total-number shells.
    Unitary projected evolution conserves trace even when cutoff error is large.
    """

    density_matrix: object
    basis: tuple
    nodes: tuple
    log_likelihood: object
    retained_norms: tuple = ()
    boundary_population: object = 0.0


def fock_state(
    pattern, parameters=None, *, cutoff, inputs=None, measurement_outcomes=None, max_dimension=512
):
    """Execute a fixed-outcome Fock pattern using differentiable dense generators.

    Args:
        pattern (Pattern): Validated command sequence with fixed topology.
        parameters (dict): Mapping from names to JAX scalar parameters.
        cutoff (int): Exclusive total-photon cutoff (integer >= 2).
        inputs (dict): Constant node-to-input descriptions; parameterize preparation with gates.
        measurement_outcomes (object): Required fixed results for every measurement.
        max_dimension (object): Dense state-space allocation limit.

    Returns:
        result (DifferentiableState): Normalized conditional state and branch likelihood.

    Raises:
        NotImplementedError: Unsupported measurement or noisy channel.
        ValueError: Invalid cutoff or absent fixed outcomes.

    Example:
        Use ``jax.grad(lambda t: expectation(pattern, {'t': t}, ...))``.
    """
    from math import comb

    from .backends.mixed_fock import MixedFockBackend
    from .measurements import Homodyne, PhotonNumber

    jax, jnp = _jax()
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 2:
        raise ValueError("cutoff must be an integer >= 2")
    pattern.validate()
    parameters = {} if parameters is None else parameters
    inputs, fixed = dict(inputs or {}), dict(measurement_outcomes or {})
    if not set(inputs) <= set(pattern.inputs):
        raise ValueError("Input supplied for a non-input node")
    keys = {x.result_key for x in pattern.commands if isinstance(x, c.Measure)}
    if fixed.keys() != keys:
        raise ValueError("Provide exactly one fixed outcome for every measurement")
    records: dict = {}
    nodes: tuple = ()
    basis: tuple = ((),)
    rho = jnp.ones((1, 1), dtype=jnp.complex128)
    log_likelihood = jnp.asarray(0.0)
    retained_norms = []

    def require(condition, message):
        # Eager calls can raise. During JAX tracing, invalid states are explicitly
        # poisoned with NaNs; no host callback or tracer-to-bool conversion.
        if not isinstance(condition, jax.core.Tracer) and not bool(condition):
            raise ValueError(message)
        return condition

    def normalize(matrix, message, valid=True):
        mass = jnp.trace(matrix).real
        ok = require(jnp.isfinite(mass) & (mass > 0) & valid, message)
        return matrix / jnp.where(ok, mass, jnp.nan)

    def val(x):
        return resolve(x, parameters, records)

    def prepare(new_nodes, state, squeezing=0.0):
        nonlocal rho, basis, nodes
        if (
            comb(len(nodes) + len(new_nodes) + cutoff - 1, len(nodes) + len(new_nodes))
            > max_dimension
        ):
            raise MemoryError("Differentiable Fock dimension exceeds max_dimension")
        # Native preparation is constant. Parameterized squeezing is applied below.
        helper = MixedFockBackend(cutoff, max_matrix_bytes=16 * max_dimension**2)
        helper.prepare_resource(new_nodes, state, squeezing=0.0)
        snap = helper.get_state()
        retained_norms.extend(snap.retained_norms)
        local_basis = snap.basis
        local = jnp.asarray(snap.density_matrix)
        target_basis = _basis(len(nodes) + len(new_nodes), cutoff)
        old_lookup, new_lookup = (
            {b: i for i, b in enumerate(basis)},
            {b: i for i, b in enumerate(local_basis)},
        )
        ii = np.array([old_lookup[b[: len(nodes)]] for b in target_basis])
        jj = np.array([new_lookup[b[len(nodes) :]] for b in target_basis])
        rho = rho[jnp.ix_(ii, ii)] * local[jnp.ix_(jj, jj)]
        mass = jnp.trace(rho).real
        retained_norms.append(mass)
        rho = normalize(
            rho,
            "Fock tensor truncation exceeds tolerance; increase cutoff",
            jnp.abs(mass - 1) <= 1e-3,
        )
        nodes += tuple(new_nodes)
        basis = target_basis
        if state is None and len(new_nodes) == 1:
            a, _, _ = _operators(basis, len(nodes) - 1)
            unitary(
                jax.scipy.linalg.expm(
                    -val(squeezing) * jnp.asarray(a @ a - a.conj().T @ a.conj().T) / 2
                )
            )

    def unitary(u):
        nonlocal rho
        rho = u @ rho @ u.conj().T

    for node in pattern.inputs:
        prepare((node,), inputs.get(node))
    for command in pattern.commands:
        if isinstance(command, c.Prepare):
            prepare((command.node,), command.state, command.squeezing)
            continue
        if isinstance(command, c.PrepareResource):
            prepare(command.nodes, command.state)
            continue
        if isinstance(command, c.Signal):
            records[command.key] = val(command.value)
            continue
        if isinstance(command, c.Output):
            continue
        if isinstance(command, (c.Entangle, c.BeamSplitter)):
            a, q, _ = _operators(basis, nodes.index(command.u))
            b, x, _ = _operators(basis, nodes.index(command.v))
            generator = (
                1j * val(command.weight) * jnp.asarray((q @ x + x @ q) / 4)
                if isinstance(command, c.Entangle)
                # Projected different-mode ladders need not commute at the
                # total-number boundary. Keep the explicitly adjoint ordering.
                else val(command.theta) * jnp.asarray(b.conj().T @ a - a.conj().T @ b)
            )
            unitary(jax.scipy.linalg.expm(generator))
            continue
        mode = nodes.index(command.node)
        a, q, p = _operators(basis, mode)
        number: np.ndarray = np.diag([b[mode] for b in basis])
        generator = None
        if isinstance(command, c.Rotate):
            generator = 1j * val(command.angle) * number
        elif isinstance(command, c.Squeeze):
            generator = val(command.r) * (a @ a - a.conj().T @ a.conj().T) / 2
        elif isinstance(command, c.Displace):
            generator = 0.5j * (val(command.p) * q - val(command.q) * p)
        elif isinstance(command, c.CubicPhase):
            generator = 1j * val(command.gamma) * (q @ q @ q) / 6
        elif isinstance(command, c.QuadraticPhase):
            generator = 1j * val(command.s) * (q @ q) / 4
        elif isinstance(command, c.Kerr):
            generator = 1j * val(command.kappa) * (number @ number)
        elif isinstance(command, (c.PhotonAdd, c.PhotonSubtract)):
            op = jnp.asarray(a.conj().T if isinstance(command, c.PhotonAdd) else a)
            expected = jnp.trace(
                rho @ jnp.asarray(number + np.eye(len(basis)) * isinstance(command, c.PhotonAdd))
            ).real
            rho = op @ rho @ op.conj().T
            rho = normalize(
                rho,
                "Ladder has zero norm or exceeds cutoff",
                expected - jnp.trace(rho).real <= 1e-12,
            )
        elif isinstance(command, c.Loss):
            if command.thermal_photons != 0:
                raise NotImplementedError(
                    "Differentiable loss currently requires a vacuum environment"
                )
            from math import comb as binomial

            eta = val(command.transmissivity)
            result = jnp.zeros_like(rho)
            lookup = {b: i for i, b in enumerate(basis)}
            for lost in range(cutoff):
                op = jnp.zeros_like(rho)
                for j, b in enumerate(basis):
                    if b[mode] >= lost:
                        target_list = list(b)
                        target_list[mode] -= lost
                        value = (
                            np.sqrt(binomial(b[mode], lost))
                            * (1 - eta) ** (lost / 2)
                            * eta ** ((b[mode] - lost) / 2)
                        )
                        op = op.at[lookup[tuple(target_list)], j].set(value)
                result += op @ rho @ op.conj().T
            rho = result
        elif isinstance(command, c.Measure):
            measurement = command.measurement
            outcome = fixed[command.result_key]
            survivor_basis = _basis(len(nodes) - 1, cutoff)
            lookup = {b: i for i, b in enumerate(survivor_basis)}
            op = jnp.zeros((len(survivor_basis), len(basis)), dtype=jnp.complex128)
            if isinstance(measurement, Homodyne):
                if measurement.efficiency != 1 or measurement.noise:
                    raise NotImplementedError(
                        "Differentiable fixed homodyne requires ideal detection"
                    )
                x = jnp.asarray(outcome)
                waves = [jnp.exp(-x * x / 4) / (2 * np.pi) ** 0.25]
                waves.append(x * waves[0])
                for n in range(2, cutoff):
                    waves.append((x * waves[-1] - np.sqrt(n - 1) * waves[-2]) / np.sqrt(n))
                weights = jnp.stack(waves) * jnp.exp(
                    -1j * val(measurement.angle) * jnp.arange(cutoff)
                )
            elif isinstance(measurement, PhotonNumber):
                if (
                    isinstance(outcome, bool)
                    or not isinstance(outcome, int)
                    or not 0 <= outcome < cutoff
                ):
                    raise ValueError("Fixed PNR outcome must be an integer within cutoff")
                weights = jnp.asarray(np.arange(cutoff) == outcome)
            else:
                raise NotImplementedError("Differentiable Fock supports homodyne and PNR")
            for j, b in enumerate(basis):
                op = op.at[lookup[b[:mode] + b[mode + 1 :]], j].set(weights[b[mode]])
            rho = op @ rho @ op.conj().T
            mass = jnp.trace(rho).real
            log_likelihood += jnp.log(mass)
            rho = normalize(rho, "Postselection has zero or invalid probability/density")
            records[command.result_key] = outcome
            basis, nodes = survivor_basis, nodes[:mode] + nodes[mode + 1 :]
        else:
            raise NotImplementedError(
                f"Differentiable execution does not support {type(command).__name__}"
            )
        if generator is not None:
            unitary(jax.scipy.linalg.expm(jnp.asarray(generator)))
    outputs = pattern.outputs
    if nodes != outputs:
        keep = [nodes.index(n) for n in outputs]
        discard = [i for i in range(len(nodes)) if i not in keep]
        target = _basis(len(keep), cutoff)
        lookup = {b: i for i, b in enumerate(target)}
        reduced = jnp.zeros((len(target), len(target)), dtype=jnp.complex128)
        for i, j in product(range(len(basis)), repeat=2):
            if all(basis[i][k] == basis[j][k] for k in discard):
                reduced = reduced.at[
                    lookup[tuple(basis[i][k] for k in keep)],
                    lookup[tuple(basis[j][k] for k in keep)],
                ].add(rho[i, j])
        rho, basis, nodes = reduced, target, outputs
    boundary = jnp.sum(
        jnp.diag(rho).real * jnp.asarray([sum(b) >= max(1, cutoff - 2) for b in basis])
    )
    return DifferentiableState(rho, basis, nodes, log_likelihood, tuple(retained_norms), boundary)


def expectation(pattern, parameters=None, *, observable="photon_number", node=None, **kwargs):
    """Return a differentiable conditional expectation of n, parity, q or p."""
    _, jnp = _jax()
    result = fock_state(pattern, parameters, **kwargs)
    if node is None:
        if len(result.nodes) != 1:
            raise ValueError("Choose an output node for the expectation")
        node = result.nodes[0]
    mode = result.nodes.index(node)
    _, q, p = _operators(result.basis, mode)
    n = np.array([b[mode] for b in result.basis])
    operators = {"photon_number": np.diag(n), "parity": np.diag((-1.0) ** n), "q": q, "p": p}
    if observable not in operators:
        raise ValueError("observable must be photon_number, parity, q, or p")
    return jnp.trace(result.density_matrix @ jnp.asarray(operators[observable])).real


def score_function_surrogate(values, log_probabilities, *, baseline=0.0):
    """Return a loss with likelihood-ratio gradient for sampled discrete branches.

    Samples must come from the parameter-dependent distribution. The baseline
    must be independent of each sampled outcome. Callers supply full joint log
    probabilities, including heralding when estimating unconditional objectives.
    """
    jax, jnp = _jax()
    values, logs = jnp.asarray(values), jnp.asarray(log_probabilities)
    correction = jax.lax.stop_gradient(values - baseline) * (logs - jax.lax.stop_gradient(logs))
    return jnp.mean(values + correction)


def gaussian_sample(mean, covariance, standard_normal):
    """Reparameterize a Gaussian draw with fixed base noise for pathwise gradients.

    Covariance must be positive definite. This primitive does not turn discrete
    photon counts into differentiable samples.
    """
    _, jnp = _jax()
    return jnp.asarray(mean) + jnp.linalg.cholesky(jnp.asarray(covariance)) @ jnp.asarray(
        standard_normal
    )
