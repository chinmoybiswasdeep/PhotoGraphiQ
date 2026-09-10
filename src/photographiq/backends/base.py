"""Backend contract: one trajectory, destructive measurements, labelled modes."""

from abc import ABC, abstractmethod


class BaseBackend(ABC):
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
