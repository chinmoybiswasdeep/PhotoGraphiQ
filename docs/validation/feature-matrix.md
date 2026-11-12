# Feature matrix

S = supported core path; E = experimental implementation; — = unsupported.
Validation refers to tested regimes, not a proof for every input or parameter.

"Native Gaussian" is `native-gaussian`. "Pure Fock" covers `piquasso-fock` and
`native-fock`; "Mixed Fock" covers `piquasso-mixed-fock` and `native-mixed-fock`.
The native engines (compiled `photographiq._native`) declare the same
capabilities as their Python/Piquasso counterparts, plus `cross_kerr` on the
native Fock engines. `photographiq.registry.describe_backend(name)` returns the
capabilities a backend class declares; that class attribute, not this page, is
authoritative.

| Feature | NumPy Gaussian | Piquasso Gaussian | Native Gaussian | Pure Fock | Mixed Fock | Evidence |
|---|---|---|---|---|---|---|
| Homodyne | S | S | E | E | E | analytical conditioning, pure/mixed comparison |
| Noisy homodyne | S | S | E | — | E | Gaussian moments, convolution density |
| Heterodyne/general-dyne | S | S | E | — | — | Gaussian analytical references |
| Postselection with likelihoods | S | S | E | E | E | closed-form Gaussian conditioning; converged Fock postselection |
| PNR | — | — | — | E | E | Born probabilities and conditional states |
| Rank-one finite instruments | — | — | — | E | E | complex rank-one conditioning; `native-fock` vs `piquasso-fock` |
| Multi-Kraus instruments / finite-outcome POVMs | — | — | — | — | E | multi-Kraus retained updates (tested on `piquasso-mixed-fock`; on `native-mixed-fock` against an explicit partial trace in `tests/native/test_native_mixed_fock_backend.py`); pure backends reject before allocation |
| Physical GKP X/Z readout | — | — | — | E | E | independent comb integrals; `native-fock` vs `piquasso-fock` |
| CZ, rotation, squeezing | S | S | E | E | E | matrices, raw native references |
| Cubic/Kerr execution | — | — | — | E | E | independent Fock references |
| Cross-Kerr (backend method, no Pattern command) | — | — | — | E (native only) | E (native only) | Piquasso cross-Kerr gate |
| Cat resources | — | — | — | E | E | parity and Wigner checks |
| Photon subtraction | — | — | — | E | E | ladder factors; physical tap probabilities |
| Loss/thermal noise | S | S | E | — | E | analytical moments/binomial loss |
| GKP resources | — | — | — | E | E | projection/grid checks; no threshold claim |
| Gaussian compilation | S | S | E | E | E | channel and structural tests |
| Cubic injection/Kerr synthesis | — | — | — | E | E | finite-resource and synthesis refinement |

**2026-11-12 update.** The native engines now have first-principles references in
`tests/validation/` ([independent validation matrix](independent-validation-matrix.md)):
`native-gaussian` states, gates, channels and homodyne/heterodyne conditioning against
analytic formulas; `native-fock` and `native-mixed-fock` states, gates, photon counting,
homodyne, loss and multi-Kraus instruments against `expm` on enlarged spaces, mpmath and
explicit dilations. One-mode native Gaussian unitaries are checked against 60-digit
references. These native paths are therefore validated (see
[COMPLETE_FUNCTIONALITY.md](../COMPLETE_FUNCTIONALITY.md)). The E entries above remain
accurate for the Piquasso-backed Fock engines and for features without such a reference
(cat resources, photon subtraction, GKP resources, synthesis).

Native Gaussian is checked against the NumPy Gaussian backend trajectory by
trajectory (tests/native/test_native_gaussian_backend.py). The native Fock
engines are checked against Piquasso and exact Gaussian moments
(tests/native). They differ from Piquasso at truncation level for some gates;
see the `photographiq.backends.native_fock` module docstring.

JAX is a separate experimental execution API for dense Fock generators, ideal
fixed-outcome homodyne/PNR and vacuum loss. It is not an automatically selected
backend for `simulate`. Infinite-outcome or continuous POVMs beyond the listed
measurements, physical hardware execution and a fault-tolerant GKP architecture
remain unsupported.

## Modules outside `simulate`

| Module | Status | Scope | Evidence |
|---|---|---|---|
| `photographiq.sampling` (boson sampling, GBS) | E | exact probabilities (permanents, hafnians, loop hafnians, torontonians) and exact sampling (PNR GBS within a photon cutoff, the omitted mass returned as an explicit overflow outcome); CPU only (NumPy or compiled kernels) | independent Fock-space construction, closed forms, Piquasso (tests/sampling) |
| `photographiq.flows` qubit causal flow, gflow, Pauli flow | E | complete layered finders returning maximally delayed layers, plus independent verifiers | statevector determinism of every branch, Graphix cross-check (tests/flow) |
| `photographiq.flows` CV flow (`Model.CV_GAUSSIAN`) | E | real-linear order search (`search_cv_order`) and numerical determinism check by Gaussian postselection | tests/flow/test_flows_cv.py |
| `photographiq.flows` non-Gaussian CV and GKP logical models | — | `analyze` returns `unsupported` | — |
| `photographiq.calculus` | E | qubit measurement calculus, local Cliffords, ZX rewrites, exact on small models; CV view is command-level only (no CV graphical calculus) | exact tensor semantics, Graphix cross-check (tests/calculus) |
| `photographiq.gpu` (CuPy) | E | batched Gaussian kernels only (symplectic update, homodyne conditioning, densities, purity, fidelity); never falls back to CPU silently | GPU and CPU paths against the NumPy Gaussian backend (tests/gpu; GPU tests skip without CuPy or a CUDA device) |
| `photographiq.cudaq_dynamics` adapter | E | Hamiltonian-level adapter, not a backend: gate generators, `to_matrix`, total-photon/per-mode mapping, `reference_evolve` (unitary, Lindblad), `to_cudaq`, `evolve`; the CUDA-Q execution path is unverified | generators and Lindblad loss vs native-fock / native-mixed-fock through the NumPy reference, translation wiring with a fake `cudaq` (tests/cudaq_adapter); real-CUDA-Q tests skip (`cudaq` is not installable on native Windows) |

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
