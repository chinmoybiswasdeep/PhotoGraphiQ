"""Backend contract: one trajectory, destructive measurements, labelled modes."""

from abc import ABC, abstractmethod


class BaseBackend(ABC):
    capabilities: frozenset[str] = frozenset()

    def supports(self, feature: str) -> bool:
        """Whether this backend implements a named execution capability."""
        return feature in self.capabilities

    def require(self, *features):
        missing = set(features) - self.capabilities
        if missing:
            raise NotImplementedError(
                f"{type(self).__name__} does not support {', '.join(sorted(missing))}"
            )

    def prepare_resource(self, nodes, state):
        raise NotImplementedError("Correlated Fock resource requires a Fock backend")

    def kerr(self, node, kappa):
        raise NotImplementedError("Kerr requires a Fock backend")

    def quadratic_phase(self, node, s):
        raise NotImplementedError("Quadratic phase is unsupported")

    def ladder(self, node, addition):
        raise NotImplementedError("Ladder operations require a Fock backend")

    @abstractmethod
    def reset(self, seed=None): ...
    @abstractmethod
    def prepare(self, node, squeezing=1.0, state=None): ...
    @abstractmethod
    def entangle(self, u, v, weight=1.0): ...
    @abstractmethod
    def displace(self, node, q=0.0, p=0.0): ...
    @abstractmethod
    def rotate(self, node, angle): ...
    @abstractmethod
    def squeeze(self, node, r): ...
    @abstractmethod
    def beamsplitter(self, u, v, theta): ...
    @abstractmethod
    def measure(self, node, measurement, angle=0.0): ...
    @abstractmethod
    def get_state(self, nodes=None): ...

    def loss(self, node, transmissivity, thermal_photons=0.0):
        raise NotImplementedError("Loss channel is not implemented by this backend")

    def cubic_phase(self, node, gamma):
        raise NotImplementedError("Cubic phase requires a non-Gaussian backend")
