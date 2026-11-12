"""Boson sampling and Gaussian boson sampling (hbar = 2, interleaved quadratures).

Matrix functions dispatch to the compiled ``photographiq._native.sampling`` kernels when
available (``backend="auto"``) and to NumPy implementations of the same algorithms
otherwise. See :mod:`.linalg`, :mod:`.unitaries`, :mod:`.boson_sampling` and :mod:`.gbs`
for conventions, formulas and limits.

:func:`photographiq.sampling.sample` draws boson-sampling output patterns from an
interferometer; it is unrelated to :func:`photographiq.sample`, which runs shots of an MBQC
measurement pattern.
"""

from .boson_sampling import (
    collision_probabilities,
    is_collision_free,
    output_distribution,
    output_patterns,
    output_probability,
    sample,
    transition_amplitude,
)
from .gbs import (
    GBSMatrices,
    PNRSamples,
    ThresholdPrecisionWarning,
    TruncatedDistribution,
    apply_loss,
    gbs_matrices,
    pnr_distribution,
    pnr_probability,
    sample_pnr,
    sample_threshold,
    squeezed_interferometer_state,
    threshold_distribution,
    threshold_probability,
)
from .linalg import (
    hafnian,
    hafnian_by_definition,
    loop_hafnian,
    loop_hafnian_by_definition,
    loop_torontonian,
    permanent,
    permanent_by_definition,
    repeat_indices,
    torontonian,
)
from .unitaries import (
    ClementsDecomposition,
    check_unitary,
    clements_decomposition,
    haar_unitary,
    mzi,
    passive_symplectic,
)

__all__ = [
    "ClementsDecomposition",
    "GBSMatrices",
    "PNRSamples",
    "ThresholdPrecisionWarning",
    "TruncatedDistribution",
    "apply_loss",
    "check_unitary",
    "clements_decomposition",
    "collision_probabilities",
    "gbs_matrices",
    "haar_unitary",
    "hafnian",
    "hafnian_by_definition",
    "is_collision_free",
    "loop_hafnian",
    "loop_hafnian_by_definition",
    "loop_torontonian",
    "mzi",
    "output_distribution",
    "output_patterns",
    "output_probability",
    "passive_symplectic",
    "permanent",
    "permanent_by_definition",
    "pnr_distribution",
    "pnr_probability",
    "repeat_indices",
    "sample",
    "sample_pnr",
    "sample_threshold",
    "squeezed_interferometer_state",
    "threshold_distribution",
    "threshold_probability",
    "torontonian",
    "transition_amplitude",
]
