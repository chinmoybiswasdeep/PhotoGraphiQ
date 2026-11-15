"""Backend-neutral Gaussian states and input preparation descriptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import xlogy

from .gaussian import omega, rotation


def _symplectic_eigenvalues(covariance):
    # The spectrum of i Omega V is {+-nu_k}; keep one of each pair. With V = L L^T,
    # i L^T Omega L is Hermitian and isospectral, so its eigenvalues have error ~eps*||V||
    # instead of eps*||V||^2 for the non-normal i Omega V (matters for strong squeezing).
    n = len(covariance) // 2
    try:
        lower = np.linalg.cholesky(covariance)
        values = np.linalg.eigvalsh(1j * lower.T @ omega(n) @ lower)
    except np.linalg.LinAlgError:  # not positive definite: unphysical, keep the general path
        values = np.linalg.eigvals(1j * omega(n) @ covariance)
    return np.sort(np.abs(values))[::2]


def _bbp_fidelity(mean1, cov1, mean2, cov2):
    """Squared Uhlmann fidelity of two mixed Gaussian states (Banchi-Braunstein-Pirandola).

    BBP work in units with vacuum covariance I/2, so V_a = V1/2, V_b = V2/2 and
    V_aux = Omega^T (V_a+V_b)^-1 (Omega/4 + V_b Omega V_a). Their determinant
    det[2 (sqrt(I + (V_aux Omega)^-2/4) + I) V_aux] is evaluated spectrally as
    det(2 V_aux) prod_j (1 + sqrt(1 + 1/(4 mu_j^2))), mu_j the eigenvalues of V_aux Omega
    (spectral mapping theorem), so no matrix square root is formed. In these units the
    displacement term becomes exp(-d^T (V1+V2)^-1 d / 2) with d the hbar=2 mean difference.
    The imaginary part of the eigenvalue product cancels analytically and is discarded.
    """
    n = len(mean1) // 2
    om = omega(n)
    va, vb = cov1 / 2, cov2 / 2
    total = va + vb
    aux = om.T @ np.linalg.solve(total, om / 4 + vb @ om @ va)
    mu = np.linalg.eigvals(aux @ om).astype(complex)
    product = np.prod(1 + np.sqrt(1 + 1 / (4 * mu**2))).real
    sign_aux, log_aux = np.linalg.slogdet(2 * aux)
    sign_total, log_total = np.linalg.slogdet(total)
    ratio = product * sign_aux * sign_total * np.exp(log_aux - log_total)
    if not ratio >= 0:
        raise ValueError("Fidelity determinant ratio is not positive")
    delta = np.asarray(mean1) - np.asarray(mean2)
    exponent = delta @ np.linalg.solve(cov1 + cov2, delta)
    return float(np.sqrt(ratio) * np.exp(-0.5 * exponent))


@dataclass(frozen=True)
class FockDensityMatrix:
    """Normalized mixed input on an explicit ordered occupation basis.

    Args:
        matrix (array-like): Hermitian positive semidefinite density matrix with trace one.
        basis (object): Unique nonnegative integer occupation tuples, one per row.

    Raises:
        ValueError: Invalid basis, dimensions, positivity, or normalization.

    Example:
        ``FockDensityMatrix([[0.5, 0], [0, 0.5]], ((0,), (1,)))``.
    """

    matrix: object
    basis: tuple

    def __post_init__(self):
        basis = tuple(tuple(b) for b in self.basis)
        rho = np.asarray(self.matrix, dtype=complex)
        if (
            not basis
            or not basis[0]
            or len(set(basis)) != len(basis)
            or any(len(b) != len(basis[0]) for b in basis)
            or any(
                isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0
                for b in basis
                for n in b
            )
        ):
            raise ValueError("Provide unique, equally sized nonnegative integer occupations")
        if rho.shape != (len(basis), len(basis)) or not np.isfinite(rho).all():
            raise ValueError("Density matrix shape must match the finite occupation basis")
        if not np.allclose(rho, rho.conj().T, atol=1e-12, rtol=0) or not np.isclose(
            np.trace(rho), 1, atol=1e-10, rtol=0
        ):
            raise ValueError("Density matrix must be Hermitian with trace one")
        if np.linalg.eigvalsh(rho).min() < -1e-12:
            raise ValueError("Density matrix must be positive semidefinite")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "matrix", tuple(map(tuple, rho)))

    @property
    def modes(self):
        """Number of modes in each occupation tuple."""
        return len(self.basis[0])

    @classmethod
    def thermal(cls, mean_photons, cutoff):
        """Return a normalized truncated single-mode thermal input.

        The omitted infinite-state weight is ``(nbar/(1+nbar))**cutoff``;
        rebuild this resource when studying cutoff convergence.
        """
        if (
            not np.isfinite(mean_photons)
            or mean_photons < 0
            or isinstance(cutoff, bool)
            or not isinstance(cutoff, int)
            or cutoff < 2
        ):
            raise ValueError("Thermal input requires mean_photons >= 0 and integer cutoff >= 2")
        weights = (mean_photons / (1 + mean_photons)) ** np.arange(cutoff)
        return cls(np.diag(weights / weights.sum()), tuple((n,) for n in range(cutoff)))

    def __repr__(self):
        return f"FockDensityMatrix(modes={self.modes}, basis_size={len(self.basis)})"


@dataclass
class GaussianState:
    """Gaussian statistical moments in interleaved hbar=2 quadratures with ordered labels.

    Args:
        mean (array-like): Interleaved quadrature mean vector.
        covariance (array-like): Statistical covariance, with vacuum covariance I.
        nodes (tuple): Ordered mode labels.

    Raises:
        ValueError: Invalid Gaussian moment dimensions.
        ValueError: Provide one unique label per mode.
        ValueError: Moments must be finite.
        ValueError: Covariance must be symmetric.
        ValueError: Covariance violates the uncertainty relation V+i Omega >= 0.
    """

    mean: np.ndarray
    covariance: np.ndarray
    nodes: tuple = ()

    def __repr__(self):
        preview = repr(self.nodes)
        if len(preview) > 80:
            preview = preview[:77] + "..."
        return f"GaussianState(modes={len(self.nodes)}, nodes={preview}, covariance_shape={self.covariance.shape})"

    def __post_init__(self):
        self.mean = np.array(self.mean, dtype=float, copy=True)
        self.covariance = np.array(self.covariance, dtype=float, copy=True)
        self.nodes = tuple(self.nodes)
        self.validate()

    def validate(self):
        """Validate dimensions, labels, causality or physicality for this object; return self.

        Returns:
            result (object): This validated object.

        Raises:
            ValueError: Invalid Gaussian moment dimensions.
            ValueError: Provide one unique label per mode.
            ValueError: Moments must be finite.
            ValueError: Covariance must be symmetric.
            ValueError: Covariance violates the uncertainty relation V+i Omega >= 0.
        """
        n = self.mean.size
        if self.mean.shape != (n,) or n % 2 or self.covariance.shape != (n, n):
            raise ValueError("Invalid Gaussian moment dimensions")
        if len(self.nodes) != n // 2 or len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Provide one unique label per mode")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.covariance).all():
            raise ValueError("Moments must be finite")
        if not np.allclose(self.covariance, self.covariance.T, atol=1e-10, rtol=0):
            raise ValueError("Covariance must be symmetric")
        if n and np.linalg.eigvalsh(self.covariance + 1j * omega(n // 2)).min() < -1e-7:
            raise ValueError("Covariance violates the uncertainty relation V+i Omega >= 0")
        return self

    def copy(self):
        """Return an independent copy of this object.

        Returns:
            result (object): Independent object copy.
        """
        return GaussianState(self.mean, self.covariance, self.nodes)

    def reduced(self, nodes):
        """Return the partial trace on requested nodes in exactly that label order.

        Args:
            nodes (tuple): Ordered mode labels.

        Returns:
            result (object): Reduced state snapshot.
        """
        nodes = tuple(nodes)
        indices = [
            j
            for node in nodes
            for j in (2 * self.nodes.index(node), 2 * self.nodes.index(node) + 1)
        ]
        return GaussianState(self.mean[indices], self.covariance[np.ix_(indices, indices)], nodes)

    def quadrature(self, node, angle=0.0):
        """Return (mean, variance) of q*cos(angle)+p*sin(angle) on a labelled mode.

        Args:
            node (object): Hashable mode label.
            angle (float): Quadrature or gate angle in radians; expressions allowed where documented.

        Returns:
            result (tuple): Mean and variance of the chosen quadrature.
        """
        i = self.nodes.index(node)
        v = np.array([np.cos(angle), np.sin(angle)])
        return float(v @ self.mean[2 * i : 2 * i + 2]), float(
            v @ self.covariance[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] @ v
        )

    def photon_number(self, node):
        """Return mean photon occupation on a labelled mode.

        Args:
            node (object): Hashable mode label.

        Returns:
            result (float): Mean occupation.
        """
        state = self.reduced((node,))
        return float((np.trace(state.covariance) + state.mean @ state.mean - 2) / 4)

    def parity(self, nodes=None):
        """Return the joint photon-number parity expectation on selected modes.

        Args:
            nodes (tuple): Ordered mode labels.

        Returns:
            result (float): Parity expectation.
        """
        state = self if nodes is None else self.reduced(nodes)
        return float(
            np.exp(-0.5 * state.mean @ np.linalg.solve(state.covariance, state.mean))
            / np.sqrt(np.linalg.det(state.covariance))
        )

    def overlap(self, other):
        """Tr(rho sigma); equals fidelity if at least one state is pure."""
        if self.nodes != other.nodes:
            raise ValueError("State labels/order must agree")
        cov = self.covariance + other.covariance
        delta = self.mean - other.mean
        return float(
            2 ** len(self.nodes)
            * np.exp(-0.5 * delta @ np.linalg.solve(cov, delta))
            / np.sqrt(np.linalg.det(cov))
        )

    def fidelity(self, other, pure_tolerance=1e-9):
        """Squared Uhlmann fidelity (Tr sqrt(sqrt(rho) sigma sqrt(rho)))**2 of two Gaussian states.

        Valid for arbitrary mixed states, unlike ``overlap`` (Tr rho sigma), which agrees
        only when one state is pure. ``other`` must have the same node set; it is reordered
        to this state's label order. If either state is pure (all symplectic eigenvalues
        within ``pure_tolerance`` of 1) the exact identity F = Tr rho sigma is used, because
        the general formula loses accuracy near purity (errors ~1e-6 for squeezing r ~ 3).
        Otherwise it evaluates the Banchi-Braunstein-Pirandola formula (PRL 115, 260501)
        with NumPy, see ``_bbp_fidelity``; for nearly pure mixed states that formula is
        accurate only to roughly sqrt(machine epsilon) * exp(2 r) for squeezing r (observed
        relative errors ~2e-7 at r = 1.5 and ~4e-6 at r = 3 when nu - 1 <= 1e-5). For squeezing
        beyond r ~ 3-4 a rotated float64 covariance cannot resolve purity at all (relative
        error ~eps*exp(4r) in its smallest eigenvalue); treat results there as unreliable.

        Raises:
            ValueError: State node sets differ.
        """
        if set(self.nodes) != set(other.nodes):
            raise ValueError("State node sets must agree")
        other = other.reduced(self.nodes)
        pure = (np.all(abs(s.symplectic_eigenvalues() - 1) < pure_tolerance) for s in (self, other))
        if any(pure):
            return self.overlap(other)
        return _bbp_fidelity(self.mean, self.covariance, other.mean, other.covariance)

    def symplectic_eigenvalues(self):
        """Return the sorted symplectic eigenvalues nu_k of V (vacuum nu = 1, physical nu >= 1)."""
        return _symplectic_eigenvalues(self.covariance)

    def purity(self):
        """Return Tr(rho^2) = det(V)^(-1/2) (hbar=2, vacuum V = I)."""
        return float(np.exp(-0.5 * np.linalg.slogdet(self.covariance)[1]))

    def entropy(self, tolerance=1e-7):
        """Return the von Neumann entropy in nats, sum_k g(nu_k).

        g(nu) = ((nu+1)/2) ln((nu+1)/2) - ((nu-1)/2) ln((nu-1)/2), with g(1) = 0.
        Eigenvalues within ``tolerance`` below 1 are treated as 1 (numerical noise).

        Raises:
            ValueError: A symplectic eigenvalue is below 1 - tolerance (unphysical covariance).
        """
        nu = self.symplectic_eigenvalues()
        if nu.size and nu.min() < 1 - tolerance:
            raise ValueError("Covariance has a symplectic eigenvalue below 1")
        a, b = (np.maximum(nu, 1) + 1) / 2, (np.maximum(nu, 1) - 1) / 2
        return float(np.sum(xlogy(a, a) - xlogy(b, b)))

    def log_negativity(self, nodes):
        """Return the base-2 logarithmic negativity of the bipartition (nodes | rest).

        The partial transpose flips the sign of p on ``nodes``; with nu~_k the symplectic
        eigenvalues of the transposed covariance, E_N = sum_k max(0, -log2 nu~_k).

        Raises:
            ValueError: Unknown or repeated node labels.
        """
        nodes = tuple(nodes)
        if len(set(nodes)) != len(nodes) or not set(nodes) <= set(self.nodes):
            raise ValueError("Provide unique labels of this state")
        flip = np.ones(self.mean.size)
        for node in nodes:
            flip[2 * self.nodes.index(node) + 1] = -1
        nu = _symplectic_eigenvalues(flip[:, None] * self.covariance * flip)
        return float(np.sum(np.maximum(0.0, -np.log2(nu))))


@dataclass(frozen=True)
class GaussianInput:
    """Single-mode Gaussian preparation described by mean and statistical covariance.

    Args:
        mean (array-like): Interleaved quadrature mean vector.
        covariance (array-like): Statistical covariance, with vacuum covariance I.
    """

    mean: tuple = (0.0, 0.0)
    covariance: tuple = ((1.0, 0.0), (0.0, 1.0))

    def state(self, node):
        """Materialize this preparation as a labelled GaussianState.

        Args:
            node (object): Hashable mode label.

        Returns:
            result (GaussianState): Labelled Gaussian moments.
        """
        return GaussianState(np.asarray(self.mean), np.asarray(self.covariance), (node,))

    @classmethod
    def coherent(cls, alpha: complex):
        """Describe a coherent state with mean (2 Re(alpha), 2 Im(alpha)) and vacuum covariance.

        Args:
            alpha (complex): Coherent-state amplitude.

        Returns:
            result (GaussianInput): Coherent preparation description.
        """
        return cls((2 * complex(alpha).real, 2 * complex(alpha).imag))

    @classmethod
    def squeezed(cls, r: float, angle: float = 0.0):
        """Describe a rotated q-squeezed state with principal variances exp(-2r), exp(2r).

        Args:
            r (float): Dimensionless squeezing parameter.
            angle (float): Quadrature or gate angle in radians; expressions allowed where documented.

        Returns:
            result (GaussianInput): Squeezed preparation description.
        """
        rot = rotation(angle)
        cov = rot @ np.diag([np.exp(-2 * r), np.exp(2 * r)]) @ rot.T
        return cls(covariance=tuple(map(tuple, cov)))


@dataclass(frozen=True)
class FockInput:
    """Single-mode normalized amplitudes, from vacuum up to an explicit cutoff."""

    amplitudes: tuple[complex, ...]

    def __repr__(self):
        return f"FockInput(levels={len(self.amplitudes)}, occupied={sum(a != 0 for a in self.amplitudes)})"

    def __post_init__(self):
        a = np.asarray(self.amplitudes, dtype=complex)
        if (
            a.ndim != 1
            or not len(a)
            or not np.isfinite(a).all()
            or not np.isclose(np.vdot(a, a), 1, atol=1e-10, rtol=0)
        ):
            raise ValueError("Fock input amplitudes must be finite and normalized")
        object.__setattr__(self, "amplitudes", tuple(map(complex, a)))

    @classmethod
    def number(cls, n: int):
        """Construct a normalized photon-number basis state.

        Args:
            n (int): Nonnegative photon occupation.

        Raises:
            ValueError: Photon number must be a nonnegative integer.
        """
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError("Photon number must be a nonnegative integer")
        return cls((0j,) * n + (1 + 0j,))

    @classmethod
    def cat(cls, alpha: complex, cutoff: int, parity: int = 1):
        """Construct a normalized finite-Fock projection of |alpha> + parity|-alpha>.

        Args:
            alpha (complex): Coherent-state amplitude.
            cutoff (int): Exclusive total-photon cutoff.
            parity (int): Cat parity +1 or -1.

        Raises:
            ValueError: Invalid cutoff or parity.
            ValueError: Zero cat vector.
            ValueError: Zero cat vector at this cutoff.
        """
        if (
            not isinstance(cutoff, int)
            or isinstance(cutoff, bool)
            or cutoff < 1
            or parity not in (-1, 1)
            or not np.isfinite(alpha)
        ):
            raise ValueError("Invalid cutoff or parity")
        from scipy.special import gammaln

        n = np.arange(cutoff)
        if alpha == 0:
            a = np.zeros(cutoff, dtype=complex)
            a[0] = 1 + parity
        else:
            logabs = n * np.log(abs(alpha)) - gammaln(n + 1) / 2
            allowed = (1 + parity * (-1) ** n) != 0
            if not allowed.any():
                raise ValueError("Zero cat vector at this cutoff")
            logabs -= logabs[allowed].max()
            logabs[~allowed] = -np.inf
            a = np.exp(logabs + 1j * n * np.angle(alpha)) * (1 + parity * (-1) ** n)
        norm = np.linalg.norm(a)
        if norm == 0:
            raise ValueError("Zero cat vector")
        return cls(tuple(a / norm))

    def photon_added(self):
        """Normalized offline a-dagger resource; no heralding probability claim."""
        a = np.r_[0j, np.sqrt(np.arange(1, len(self.amplitudes) + 1)) * self.amplitudes]
        return FockInput(tuple(a / np.linalg.norm(a)))

    def photon_subtracted(self):
        """Normalized offline a resource; subtraction from vacuum is impossible."""
        a = np.sqrt(np.arange(1, len(self.amplitudes))) * self.amplitudes[1:]
        if np.linalg.norm(a) == 0:
            raise ValueError("Photon subtraction has zero norm")
        return FockInput(tuple(a / np.linalg.norm(a)))


@dataclass(frozen=True)
class FockSuperposition:
    """Sparse normalized pure state with explicitly ordered occupation tuples.

    A basis entry (n0, n1, ...) refers to the order of preparation/input labels.
    Zero coefficients are allowed; support must fit the execution cutoff.
    """

    occupations: tuple[tuple[int, ...], ...]
    amplitudes: tuple[complex, ...]

    def __post_init__(self):
        basis = tuple(tuple(b) for b in self.occupations)
        if (
            not basis
            or not basis[0]
            or len(set(basis)) != len(basis)
            or any(len(b) != len(basis[0]) for b in basis)
            or any(
                not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 0
                for b in basis
                for n in b
            )
        ):
            raise ValueError("Invalid or duplicate occupation tuples")
        amplitudes = FockInput(self.amplitudes).amplitudes
        if len(amplitudes) != len(basis):
            raise ValueError("Occupation and amplitude dimensions differ")
        object.__setattr__(self, "occupations", basis)
        object.__setattr__(self, "amplitudes", amplitudes)

    @property
    def modes(self):
        """Number of ordered modes represented by each occupation tuple."""
        return len(self.occupations[0])

    @property
    def amplitude_map(self):
        """Return independent occupation-tuple to complex-amplitude data."""
        return dict(zip(self.occupations, self.amplitudes, strict=True))

    @classmethod
    def from_mapping(cls, amplitudes):
        """Construct a normalized correlated pure resource from an occupation-amplitude mapping.

        Args:
            amplitudes (object): Mapping from occupation tuples to complex amplitudes.
        """
        return cls(tuple(amplitudes), tuple(amplitudes.values()))

    @classmethod
    def number(cls, occupations):
        """Construct a normalized photon-number basis state.

        Args:
            occupations (tuple): Nonnegative integer occupations in mode order.
        """
        return cls((tuple(occupations),), (1 + 0j,))

    @classmethod
    def from_piquasso(cls, state):
        """Copy a normalized public PureFockState; mixed states are rejected."""
        if not hasattr(state, "state_vector"):
            raise TypeError("Only native pure Fock states can be imported")
        return cls.from_mapping(dict(state.fock_amplitudes_map))
