"""PhotoGraphiQ: continuous-variable photonic measurement-based computation."""

from . import gkp, non_gaussian, protocols, synthesis
from .analysis import GaussianChannel, gaussian_channel
from .commands import (
    BeamSplitter,
    CubicPhase,
    Displace,
    Entangle,
    Kerr,
    Loss,
    Measure,
    Output,
    PhotonAdd,
    PhotonSubtract,
    Prepare,
    PrepareResource,
    QuadraticPhase,
    Rotate,
    Signal,
    Squeeze,
)
from .compiler import Circuit, CompilationStep, CompilationTrace, compile_circuit
from .convergence import CutoffStudy, cutoff_convergence
from .encoded import (
    BaseGKPDecoder,
    GKPCode,
    IdealLogicalXYMeasurement,
    LogicalDecodeResult,
    LogicalMeasurementSynthesis,
    LogicalPauliFrame,
    NearestCellDecoder,
    PhysicalGKPReadout,
    modular_effects,
    multimode_readout,
)
from .expressions import CallableExpression, Outcome, Parameter, cos, exp, sin
from .fock_analysis import WignerGrid, fidelity, trace_distance, wigner
from .gkp import GKPResource
from .gkp_studies import GKPMeasurementStudy, SoftDecisionDecoder, measurement_convergence
from .graph import ClusterState, CVGraph
from .instruments import MeasurementInstrument
from .measurements import Generaldyne, Heterodyne, Homodyne, PhotonNumber
from .pattern import Pattern
from .resources import CatResource, CubicPhaseResource
from .simulator import Result, ShotResult, run_shots, sample, simulate
from .states import FockDensityMatrix, FockInput, FockSuperposition, GaussianInput, GaussianState

__version__ = "0.3.1"
from .visualization import visualize_compilation

__all__ = [
    "GKPMeasurementStudy",
    "SoftDecisionDecoder",
    "measurement_convergence",
    "GKPCode",
    "BaseGKPDecoder",
    "NearestCellDecoder",
    "LogicalDecodeResult",
    "LogicalPauliFrame",
    "PhysicalGKPReadout",
    "IdealLogicalXYMeasurement",
    "LogicalMeasurementSynthesis",
    "MeasurementInstrument",
    "modular_effects",
    "multimode_readout",
    "visualize_compilation",
    "gkp",
    "synthesis",
    "GKPResource",
    "CompilationStep",
    "CompilationTrace",
    "FockDensityMatrix",
    "non_gaussian",
    "Kerr",
    "QuadraticPhase",
    "PhotonAdd",
    "PhotonSubtract",
    "PrepareResource",
    "FockSuperposition",
    "CatResource",
    "CubicPhaseResource",
    "CutoffStudy",
    "cutoff_convergence",
    "fidelity",
    "trace_distance",
    "wigner",
    "WignerGrid",
    "GaussianChannel",
    "gaussian_channel",
    "CVGraph",
    "ClusterState",
    "Pattern",
    "Prepare",
    "Entangle",
    "Measure",
    "Displace",
    "Rotate",
    "Squeeze",
    "BeamSplitter",
    "Loss",
    "CubicPhase",
    "Signal",
    "Output",
    "Circuit",
    "compile_circuit",
    "Parameter",
    "Outcome",
    "CallableExpression",
    "sin",
    "cos",
    "exp",
    "Homodyne",
    "Heterodyne",
    "Generaldyne",
    "PhotonNumber",
    "GaussianInput",
    "FockInput",
    "GaussianState",
    "simulate",
    "run_shots",
    "sample",
    "Result",
    "ShotResult",
    "protocols",
]
