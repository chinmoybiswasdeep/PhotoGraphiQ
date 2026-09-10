"""PhotoGraphiQ: continuous-variable photonic measurement-based computation."""

from . import protocols
from .analysis import GaussianChannel, gaussian_channel
from .commands import (
    BeamSplitter,
    CubicPhase,
    Displace,
    Entangle,
    Loss,
    Measure,
    Output,
    Prepare,
    Rotate,
    Signal,
    Squeeze,
)
from .compiler import Circuit, compile_circuit
from .expressions import CallableExpression, Outcome, Parameter, cos, exp, sin
from .graph import ClusterState, CVGraph
from .measurements import Generaldyne, Heterodyne, Homodyne, PhotonNumber
from .pattern import Pattern
from .simulator import Result, ShotResult, run_shots, sample, simulate
from .states import FockInput, GaussianInput, GaussianState

__version__ = "0.1.0"
__all__ = [
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
