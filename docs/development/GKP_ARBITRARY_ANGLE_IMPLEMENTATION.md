# Arbitrary-angle physical GKP logical measurements: implementation notes

Status: implemented on the local `upgraded` branch (uncommitted). Results and the final
classification are in [GKP_ARBITRARY_ANGLE_FINAL_REPORT.md](GKP_ARBITRARY_ANGLE_FINAL_REPORT.md).

## 1. Audit of the code before this change

Conventions throughout: hbar = 2, interleaved (q, p) quadratures, vacuum covariance I,
total-photon Fock truncation, explicit retained-norm diagnostics.

What existed:

| Component | Location | Behaviour |
|---|---|---|
| Finite-energy codewords | `photographiq/gkp.py` `GKPResource` | Sum of Gaussians, width `peak_width`, envelope `exp(-c^2 envelope^2 / 4)` on centres `c = sqrt(2 pi)(2n + bit)`, `peaks` cells each side; projected to Fock by quadrature (`project`, captured weight returned). |
| Logical readout | `photographiq/encoded.py` `PhysicalGKPReadout` | Z = q homodyne (angle 0), X = p homodyne (angle pi/2), nearest-cell parity decoding (`decode_shift`, upper-cell ties). Y and XY raised `NotImplementedError`. |
| XY factory | `GKPCode.logical_measurement("XY", alpha)` | Only alpha = 0 or pi (mod 2 pi, absolute tolerance 1e-14), returned as X readout with a bit flip. Anything else raised. |
| Logical gates | `GKPCode.logical_gate` | Ideal-lattice Gaussian realizations: X/Z displacements, H = quarter rotation, S = `QuadraticPhase(1)`, CZ = unit `Entangle`. |
| Extension point | `LogicalMeasurementSynthesis.lower` | Raised "No validated magic-state injection or arbitrary logical Rz protocol". |
| Engine dispatch | `backends/fock.py`, `backends/mixed_fock.py` | `measure()` handed a `PhysicalGKPReadout` the live engine (`execute(engine, node, outcome)`), so an instrument can drive the engine itself. |
| PhotoGraphiQML lowering | `photographiqml/lowering.py` | Capability audit per node via `logical_measurement("XY", alpha)`; only 0/pi supported; static `Pattern` with X readouts, `Signal`s for frame-interpreted bits, joint X/Z output POVM. |
| PhysicalMuTA | `photographiqml/physical.py` | `measurement_family="X"` only; categorical {0, pi} initialisation; `DiscreteSearch` coordinate search over {0, pi}. |

So no physical operation for Y or general XY existed, by design.

## 2. Target operation and conventions

The logical observable is `M_L(a) = cos(a) X_L + sin(a) Y_L`, with +1 eigenstate
`|+_a> = (|0> + e^{ia}|1>)/sqrt(2)` (the convention of `IdealLogicalXYMeasurement` and of the
logical MuTA, whose measured node applies `H diag(e^{ia/2}, e^{-ia/2})`, i.e. projection onto
`|+_a>` with outcome 0).

Writing `D(t) = diag(1, e^{it})`, `|+_a> = D(a)|+>`, hence

    P(+) = |<+_a|psi>|^2 = |<+| D(-a) psi>|^2.

**Measuring M_L(a) equals applying D(-a) and then measuring X_L.** (An early draft applied
D(+a); every |+>-input check was blind to it because P(+) is symmetric in a -> -a for |+>. The
six-cardinal-state test of the Y readout caught it: |+i> read as -1. Both the instrument and
the reference were corrected; the generic input used in the experiments has no a -> -a
symmetry.)

Frame rules used by adaptive MBQC: `X M(a) X = M(-a)` and `Z M(a) Z = -M(a)`. In the lowering an
X byproduct flips the sign of the measured angle; a Z byproduct is the classical bit flip
already applied by the frame `Signal`.

Measured physical conventions (native engine, checked numerically):
`QuadraticPhase(+1)` maps the finite |+i~> to |-~> (logical S); `rotate` is
`q -> q cos - p sin`; `entangle(u, v, g)` is `p_u += g q_v, p_v += g q_u`. Hence
`rotate(t, pi/2); entangle(c, t, 1); rotate(t, -pi/2)` is the SUM `q_t -> q_t + q_c,
p_c -> p_c - p_t`, a logical CNOT (control c).

Rotated homodyne is *not* used: a quadrature at angle a is not a measurement of M_L(a) for the
square GKP code.

## 3. Candidate protocols (literature)

| Candidate | Ancilla | Interaction | Measurement | Deterministic? | Feed-forward | Fits the Fock engines? | Complete instrument? |
|---|---|---|---|---|---|---|---|
| Clifford-assisted Y readout (GKP 2001) | none | Gaussian shear S^dagger | p homodyne + cell decoding | yes | none | yes, one mode | yes, a = k pi/2 only |
| Non-Gaussian Z rotation on the data (e.g. cubic-phase based, GKP 2001) | none / resource | cubic or engineered phase | as above | approximate | none | cubic phase exists, but no exact arbitrary-angle construction | no general angle |
| Gate teleportation / magic-state injection (Zhou-Leung-Chuang 2000; Bravyi-Kitaev 2005) | encoded `(|0>+e^{it}|1>)` | logical CNOT (Gaussian SUM) | ancilla q homodyne | heralded per round | conditional correction D(2t) | yes, two modes | yes, with repeat-until-success |
| Programmable ancilla rotations / repeat-until-success (Jones et al. 2012; Paetznick-Svore 2014) | sequence `t, 2t, 4t, ...` | as above | as above | terminates with prob 1 - 2^-K | angle doubling | yes | yes (round budget K reported) |
| All-Gaussian magic for GKP (Baragiola et al. 2019) | GKP Pauli eigenstates + vacuum | Gaussian + GKP measurements, distillation | homodyne | heralded | distillation logic | preparation not simulated here | justifies treating the encoded magic state as an available resource |
| Teleported equatorial measurement in CV-MBQC (Menicucci 2014; Bourassa et al. 2021) | GKP cluster | CZ | homodyne + decoding | yes for Clifford; magic via injection | frame | yes (this lowering) | combines with the above |

Chosen: **Clifford shear for k pi/2**, and **magic-state injection in repeat-until-success
form** for all other angles. Both use only operations the engines implement exactly
(finite-Fock state preparation, Gaussian unitaries, ideal or noisy homodyne).

## 4. Protocol

Let `phi` be the phase of the `D(phi)` still to apply to the data mode (initially `phi = -a`).

1. If `phi = k pi/2` (absolute tolerance 1e-12): `k = 0` nothing; `k = 2` Z, realized as an exact
   outcome relabelling of the final X readout (Pauli frame); `k = 1, 3` S / S^dagger: the
   Gaussian shear `exp(+-i q^2/4)` on a fresh request; after injection rounds a remainder of
   +-pi/2 is injected with the Y-eigenstate ancilla instead (its failure leaves D(pi) = Z,
   again an exact frame flip). Stop.
2. Otherwise (injection round): prepare the finite-energy encoded state
   `(|0~> + e^{i phi}|1~>)/norm` in a fresh mode, apply the logical CNOT (data control), measure
   the ancilla q quadrature, decode its parity m with the nearest-cell rule.
   Ideal-code identity: m = 0 applies `D(phi)`; m = 1 applies `D(-phi)` (no X byproduct because
   the ancilla is measured in Z). Set `phi = 0` (m = 0) or `phi = 2 phi` (m = 1) and return to 1.
3. After `max_rounds` non-Clifford rounds without success the record is a **heralded failure**
   (`heralded_success=False`): the wrong angle was measured. For a generic angle the ideal
   success probability is `1 - 2^-K`. The budget counts non-Clifford rounds only: a Clifford
   remainder reached after earlier rounds is always completed (one Y-state round), so a record
   can use up to K + 1 ancillas. With `protocol="injection"` the first round is always an
   ancilla round, so XY(pi) applies Z physically through an ancilla instead of relabelling.
4. Measure the data p quadrature, decode, XOR the frame flip: logical outcome bit (0 <-> +1).

Feed-forward uses decoded ancilla bits only, never analog residuals. The encoded magic state is
a supplied bosonic resource (a normalized finite-Fock vector built from the same finite
codewords); its preparation is not simulated (see Baragiola et al. 2019 for an all-Gaussian
route from GKP Pauli eigenstates).

Finite-energy consequences measured during development: each Gaussian Clifford (SUM, shear)
distorts the finite envelope; each failed round adds the ancilla's momentum spread to the data
mode (about 1.3-1.4 photons per round at width 0.45), so the Fock cutoff needed grows with the
number of rounds. The default round budget is 4 (generic-angle heralded failure 1/16).

## 5. Implementation

`photographiq.encoded`:

* `EncodedGKPMeasurement` (ABC): instruments the Fock engines execute on the live engine;
  `PhysicalGKPReadout` now derives from it. Both Fock dispatch points (`fock.py`,
  `mixed_fock.py`) dispatch on this base class before angle resolution.
* `GKPLogicalXYMeasurement(angle, code, protocol="auto"|"clifford"|"injection", max_rounds=4,
  decoder, ancilla_transmissivity=1, detector_efficiency=1, homodyne_noise=0)`. `angle` may be a
  classical expression; `simulate()` resolves it (frame-adaptive angles).
* `LogicalXYMeasurementResult` (extends `LogicalDecodeResult`): `bit`, final `raw_outcome`,
  `residual`, `cell`, `rounds` (target phase, raw ancilla outcome, bit, residual, point
  density per round), `heralded_success`, `applied_operations`, `joint_density` (likelihood
  density of the whole analog record), `resource`, `retained_norm_min`, `cutoff`,
  `ancilla_modes`, `readout_probabilities` (exact decoded X probabilities of the pre-readout
  data state: the Rao-Blackwell estimator used by the experiments).
* `GKPCode.logical_measurement("Y" | "XY", protocol=...)` returns the instrument; without
  `protocol` the original restrictions are unchanged. `LogicalMeasurementSynthesis(protocol)`
  lowers to it; without a protocol it still raises.
* Postselection: `measurement_outcomes={key: {"ancilla": [q_1, ...], "data": p}}`; the record
  must contain exactly the rounds the protocol takes.
* Sampling: `sample_quadrature` tabulates the measured mode's exact reduced homodyne density
  (16385 points) and inverts it with the engine's generator; the engine then projects at that
  outcome. Same distribution as the generic adaptive-quadrature sampler, roughly 60x faster
  per GKP readout. Noisy detection keeps the engine's own sampler.
* Capability `encoded_gkp_xy` added to the Fock backends; noise options require `mixed_fock`.

`photographiqml`:

* `GKPPhysicalConfig(xy_protocol="signed-x"|"clifford"|"auto"|"injection",
  max_injection_rounds=4)`. Default `signed-x` reproduces the previous behaviour exactly.
* `physical_capabilities` audits each node for the XY family (Clifford/non-Clifford, ancilla
  rounds) before any Fock allocation; the Hilbert-dimension estimate includes the ancilla mode.
* `lower_muta_to_gkp` emits `GKPLogicalXYMeasurement` per node with the frame-adaptive angle
  `(-1)^x a` (a `CallableExpression` over the interpreted bits of its X/Z sources).
* `PhysicalMuTA(measurement_family="XY")`: continuous initialisation, angle space [0, 2 pi),
  physical-conditional records accept the instrument's analog record dicts.
* `physical_loss`, `physical_gradient` (paired common-random-number central differences) and
  `ContinuousPhysicalSearch` (bounded Brent) in `physical_training.py`. No parameter-shift rule
  is assumed for the physical instrument.

## 6. Validation design

Three levels are always kept separate:

* **Ideal qubit** (`IdealLogicalXYMeasurement`, logical MuTA, `contract_branch`).
* **Untruncated finite-energy reference** — `tests/validation/gkp_xy_reference.py`: position grid,
  codewords from their definition, exact SUM `psi(x_d, x_a - x_d)`, shear `exp(i x^2/4)`, ancilla
  homodyne integrated exactly over decoded cells (second-order cell integration), every
  repeat-until-success branch enumerated. Because the ancilla is measured right after the SUM,
  a decoded round acts on the data density as a Hadamard product with a kernel, so branches are
  exact N x N linear algebra; applied to `|i~><j~|` it gives the instrument on entangled inputs.
* **System under test**: the instrument on `native-fock` / `native-mixed-fock`.

ideal vs reference = physical finite-energy error; reference vs engine = Fock truncation;
engine shot spread = sampling error. Tests: `tests/gkp/test_logical_xy.py`,
`tests/ml/cross_layer/test_physical_xy.py`. Experiments: R60-R67 in
`paper_experiments_ml/14_gkp_xy/`.

## References

* D. Gottesman, A. Kitaev, J. Preskill, Encoding a qubit in an oscillator, Phys. Rev. A 64, 012310 (2001).
* X. Zhou, D. W. Leung, I. L. Chuang, Methodology for quantum logic gate construction, Phys. Rev. A 62, 052316 (2000).
* S. Bravyi, A. Kitaev, Universal quantum computation with ideal Clifford gates and noisy ancillas, Phys. Rev. A 71, 022316 (2005).
* N. C. Jones, J. D. Whitfield, P. L. McMahon, M.-H. Yung, R. Van Meter, A. Aspuru-Guzik, Y. Yamamoto, Faster quantum chemistry simulation on fault-tolerant quantum computers, New J. Phys. 14, 115023 (2012) (programmable ancilla rotations).
* A. Paetznick, K. M. Svore, Repeat-Until-Success: non-deterministic decomposition of single-qubit unitaries, Quantum Inf. Comput. 14, 1277 (2014).
* B. Q. Baragiola, G. Pantaleoni, R. N. Alexander, A. Karanjai, N. C. Menicucci, All-Gaussian universality and fault tolerance with the Gottesman-Kitaev-Preskill code, Phys. Rev. Lett. 123, 200502 (2019).
* N. C. Menicucci, Fault-tolerant measurement-based quantum computing with continuous-variable cluster states, Phys. Rev. Lett. 112, 120504 (2014).
* J. E. Bourassa et al., Blueprint for a scalable photonic fault-tolerant quantum computer, Quantum 5, 392 (2021).
* R. Raussendorf, H. J. Briegel, A one-way quantum computer, Phys. Rev. Lett. 86, 5188 (2001).
* V. Danos, E. Kashefi, P. Panangaden, The measurement calculus, J. ACM 54, 8 (2007).
