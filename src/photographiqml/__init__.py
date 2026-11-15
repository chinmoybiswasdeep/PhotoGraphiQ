"""PhotoGraphiQML: logical MuTA and explicit photonic research boundaries."""

from photographiq import Parameter

from .ansatz.muta import MuTA
from .ansatz.triangle import TriangleNeuron
from .gkp import GKPBridge
from .kernels import MuTAKernel
from .lowering import GKPPhysicalConfig, lower_muta_to_gkp
from .models import MuTAClassifier, MuTARegressor, QuantumInstrumentModel
from .physical import PhysicalMuTA, PhysicalMuTAResult, compare_logical_physical
from .physical_training import DiscreteSearch
from .training import History, Trainer

__version__ = "0.2.0"
__all__ = [
    "DiscreteSearch",
    "PhysicalMuTA",
    "PhysicalMuTAResult",
    "GKPPhysicalConfig",
    "lower_muta_to_gkp",
    "compare_logical_physical",
    "MuTA",
    "TriangleNeuron",
    "GKPBridge",
    "MuTAKernel",
    "MuTAClassifier",
    "MuTARegressor",
    "QuantumInstrumentModel",
    "Trainer",
    "History",
    "Parameter",
]
