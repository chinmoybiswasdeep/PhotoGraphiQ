# Feature matrix

S = supported core path; E = experimental implementation; — = unsupported.
Validation refers to tested regimes, not a proof for every input or parameter.

| Feature | NumPy Gaussian | Piquasso Gaussian | Pure Fock | Mixed Fock | Evidence |
|---|---|---|---|---|---|
| Homodyne | S | S | E | E | analytical conditioning, pure/mixed comparison |
| Noisy homodyne | S | S | — | E | Gaussian moments, convolution density |
| Heterodyne/general-dyne | S | S | — | — | Gaussian analytical references |
| PNR | — | — | E | E | Born probabilities and conditional states |
| CZ, rotation, squeezing | S | S | E | E | matrices, raw native references |
| Cubic/Kerr execution | — | — | E | E | independent Fock references |
| Cat resources | — | — | E | E | parity and Wigner checks |
| Photon subtraction | — | — | E | E | ladder factors; physical tap probabilities |
| Loss/thermal noise | S | S | — | E | analytical moments/binomial loss |
| GKP resources | — | — | E | E | projection/grid checks; no threshold claim |
| Gaussian compilation | S | S | E | E | channel and structural tests |
| Cubic injection/Kerr synthesis | — | — | E | E | finite-resource and synthesis refinement |

JAX is a separate experimental execution API for dense Fock generators, ideal
fixed-outcome homodyne/PNR and vacuum loss. It is not an automatically selected
backend for `simulate`. Arbitrary-order flow search, general POVMs, physical
hardware execution and a fault-tolerant GKP architecture remain unsupported.

## v0.3 scientific acceptance matrix

Exact refers to the stated finite algebra or analytical channel, not a guarantee
that a finite simulation represents every infinite-energy state. "Native" means
an explicit raw Piquasso comparison; shared adapters alone are not independent.

| Feature | Backend | Exact/Approximate | Analytical validation | Independent matrix validation | Raw Piquasso validation | Cutoff convergence | Estimator convergence | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rotation / Kerr | Mixed Fock | Exact within represented support | Phase factors | Density conjugation | Yes | Support invariant | N/A | Experimental |
| Displacement / squeezing / CZ | Mixed Fock | Finite-Fock approximation | Gaussian moments | Padded exponentials | Shared native gates; independent matrices anchor correctness | Resolved low-energy regime | N/A | Experimental |
| Beamsplitter | Mixed Fock / JAX | Exact complete number sector | Binomial weights and coherence signs | Number-conserving generator | Mixed raw gate path; pure/mixed comparisons | Boundary-sector regressions | N/A | Experimental |
| Vacuum loss | Mixed Fock | Exact on finite supported inputs | n=1..4 binomial; coherent attenuation | Kraus/channel references | No separate raw attenuator oracle | Coherent tail resolved | N/A | Experimental |
| Thermal attenuation | Mixed Fock | Truncated amplifier tail | Thermal mean and trace | Thermal-environment dilation | No thermal Fock native support assumed | Explicit retained trace sweep | N/A | Experimental |
| PNR / noisy homodyne | Mixed Fock | Exact finite PNR / numerical homodyne | Born rule / Gaussian Schur complement | Conditional contractions | Pure/mixed consistency | Correlated low-energy regime | Seeded sampling consistency | Experimental |
| Finite GKP comb / logical overlap | Pure / mixed Fock resources | Finite-energy approximation | Closed Gaussian overlap integrals | Independent Hermite projection | Not an independent GKP oracle | Grid, peaks, cutoff varied separately | N/A | Experimental |
| GKP stabilizers / logical shifts | Fock | Finite-energy approximation | Closed displacement integrals / Weyl algebra | Displacement matrices | Inherits native displacement | 32..96 in tested regimes | N/A | Experimental |
| GKP q/p syndrome extraction | Pure Fock | Finite ancilla and Fock approximation | Independent SUM wavefunction kernel | Native evolution compared to kernel | Shares native Gaussian gates | 32,48,72 including failed coarse cases | Fixed branch only | Experimental |
| Deterministic / branch gradients | JAX | Derivative of projected model | Rotation, displacement, squeeze, cubic, Kerr, loss, branch formulas | Boundary and native low-energy comparisons | Through independent/native gate tests | Observable, gradient, boundary recorded | Four finite-difference steps | Experimental |
| Pathwise / score primitives | JAX | Correct estimator under sampling assumptions | Gaussian and Bernoulli identities | N/A | N/A | N/A | Seeded bias, variance, standard errors | Experimental |
| Cubic MBQC injection | Pure Fock compilation | Finite-resource approximation | Filter and outcome density | Direct cubic and Hermite kernel | Existing native injection tests retained | 48,96 plus resource squeezing | Fixed branch only | Approximate |
| Quartic / Kerr synthesis | Ideal emitted primitives | Product-formula approximation | Commutator and Weyl identities | Direct target exponentials | Native primitives tested separately | 32..72; slice refinement | Amplitude order measured near 1/2 | Approximate |
| Arbitrary-unitary certified synthesis | None | N/A | No | No | No | No | No | Unsupported |
| Full stochastic MBQC differentiation | None | N/A | No | No | No | No | No | Unsupported |
| Fault-tolerant GKP thresholds | None | N/A | No | No | No | No | No | Unsupported |

See the [scientific hardening report](../development/v0.3-scientific-hardening.md)
for actual failures, measured regimes, CI evidence and release status. The existing
Gaussian core retains its earlier validation; this pass does not redesign it.
