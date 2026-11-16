# PhotoGraphiQML

PhotoGraphiQML provides ideal logical MuTA models and restricted signed-X physical
GKP execution built on PhotoGraphiQ 0.3.1. Begin with the [tutorials](tutorials/index.md)
and [input/output contract](api.md). [Models, backends and validation](guide.md)
summarises the logical and physical layers, backend selection (`piquasso-fock` or the
compiled `native-fock` engine), measurement outcomes, statistical uncertainty, how each
layer is validated, and the limitations.

The [physical architecture](physical/architecture.md), [measurement boundary](physical/supported-measurements.md),
[execution modes](physical/execution-modes.md), [lowering](physical/lowering.md),
[decoding](physical/decoding.md), [Pauli frames](physical/pauli-frames.md),
[convergence](physical/convergence.md) and [discrete training](physical/physical-training.md)
document the v0.2 capability. Earlier research audits record the v0.1 starting point.

The [paper analysis](research/muta-paper-notes.md),
[MentPy audit](research/mentpy-audit.md),
[PhotoGraphiQ audit](research/photographiq-audit.md), and
[mapping](research/muta-mapping.md) explain the scientific boundaries.
The [implementation plan](research/implementation-plan.md) records the
pre-code design. Consult the [release report](release-report.md) for executed
checks and unfinished acceptance criteria.

## Three model families

1. **Ideal logical-qubit MuTA** (`MuTA` and its learners). This is qubit MBQC with
   trainable XY angles, flow, and exact branch contraction. No physical noise is
   modelled. MentPy is the reference.
2. **Finite-energy GKP physical MuTA (restricted)** (`PhysicalMuTA`). Finite GKP modes
   run in a truncated Fock space (`piquasso-fock` or `native-fock`). Measurements are
   signed-X by default, or XY with a supplied magic state, and are decoded with a
   classical Pauli frame. It has no loss model and no physical kernels.
3. **CV-native Gaussian and non-Gaussian learning** (`photographiqml.cv`). CV states are
   encoded from data and processed by trainable Gaussian programs (direct or compiled
   finite-squeezing CV-MBQC), by weighted clusters with trainable homodyne angles
   (`CVCluster`), or experimentally by Kerr/cubic programs on the Fock engine. They are
   read out by homodyne statistics and a classical head. Finite squeezing is the only
   Gaussian imperfection modelled. Gaussian models are efficiently classically simulable,
   and no advantage is claimed. `CVCluster` is not the planned CVMuTA proposal.

The [guide](guide.md#three-model-families) compares the computational model, physical
assumptions, trainable parameters, backends and limitations of each family. The CV
family is documented in [CV-native learning](cv-native/index.md).
