"""CV-native measurement-based quantum machine learning (Gaussian and non-Gaussian).

A separate model family from the logical-qubit MuTA and the GKP physical MuTA: classical data
is encoded into CV optical states, processed by trainable Gaussian programs (direct circuits,
their compiled finite-squeezing CV-MBQC channels, or directly trainable weighted clusters with
trainable homodyne angles) or by experimental non-Gaussian Fock programs, and read out by
homodyne statistics followed by an explicit classical head.
"""

from .gaussian import CVFeatureMap, Feature, GaussianProgram, layered_program
from .kernels import CVKernel
from .learning import (
    CVClassifier,
    CVRegressor,
    CVTrainingHistory,
    central_difference,
    complex_step_gradient,
)
from .mbqc import CVCluster
from .nongaussian import NonGaussianProgram
from .resources import pattern_resources, resource_report

__all__ = [
    "CVClassifier",
    "CVFeatureMap",
    "CVKernel",
    "CVCluster",
    "CVRegressor",
    "CVTrainingHistory",
    "Feature",
    "GaussianProgram",
    "NonGaussianProgram",
    "central_difference",
    "complex_step_gradient",
    "layered_program",
    "pattern_resources",
    "resource_report",
]
