# Non-Gaussian extension: preimplementation design

This audit precedes implementation. Reference environment: Piquasso 8.0.1,
Graphix 0.4. The existing Gaussian compiler, flow certificates and moment backend
remain the Gaussian layer. The extension is experimental finite-Fock research
software, not a claim of universal or fault-tolerant MBQC.

## A. Current implementation audit

`states.FockInput` validates single-mode normalized vectors and supplies number,
cat and offline photon-added/subtracted resources. `backends.fock` tensors these
into Piquasso's **total photon number < cutoff** space. It executes Gaussian
gates and cubic phase, rejects excessive norm loss, and projects photon counts.
It lacks multimode input, conditional homodyne, capability preflight, Kerr,
online ladder operations, resource injection and convergence/negativity tools.
The pattern IR, causal expressions, safe JSON and mode lifecycle are reusable.
Existing Fock results expose density matrices, probabilities, number and parity.
Gaussian tests and Graphix structural tests must continue passing.

## B. Piquasso capability matrix

| Capability | Piquasso 8.0.1 | Extension decision |
|---|---|---|
| Number/superposition preparation | PureFock NumberState coefficients | Validate and import via public API |
| Gaussian unitaries | PureFock native instructions | Delegate |
| Cubic phase | Native exp(i gamma q^3 / 6), hbar=2 | Preserve convention; independently validate |
| Kerr | Native exp(i xi n^2) | Delegate |
| Photon counting | Native sampling/projection | Label-aware conditional adapter; raw comparison |
| Homodyne | PureFock q sampler returns no conditional state; phi restricted | Wavefunction projection and numerical CDF adapter |
| Create/Annihilate | Occupation shifts with unit coefficients in inspected source | Explicit mathematical ladder adapter with sqrt factors |
| General mixed states/loss | Separate Fock simulator | Reject in this pure-trajectory implementation |
| Cat/cubic resource | Composition of states/instructions | Named finite-resource preparations |
| Wigner/state distance | Native capabilities vary | Explicit normalized output analysis |
| Graphix | Qubit patterns and causal corrections | Structural comparisons only |

Sources: [Piquasso Fock documentation](https://piquasso.readthedocs.io/en/stable/simulators/fock.html),
[native gates](https://piquasso.readthedocs.io/en/stable/instructions/gates.html),
[Graphix](https://graphix.readthedocs.io/en/latest/index.html), inspected pinned
source including `_math/fock.py` and pure simulation steps. No private upstream
API will be imported by production code.

## C. Missing functionality and boundaries

Add sparse correlated pure inputs, capability checks, explicit resource
preparation, conditional ideal homodyne, branch likelihoods, Kerr, normalized
ideal ladder operations, physical heralding patterns, diagnostics and convergence.
No mixed-state injection, lossy Fock trajectories, noisy Fock homodyne,
heterodyne, automatic Gaussian-to-Fock switching, GKP decoding or universal
non-Gaussian compiler is promised. Sequential destructive PNR implements joint
multimode counting with a product of conditional probabilities.

## D. API and mathematical choices

Add `FockSuperposition`, `PrepareResource`, `CubicPhaseResource`, `Kerr`,
`QuadraticPhase`, `PhotonAdd`, `PhotonSubtract`; preserve existing constructors.
Add `supports(feature)` and execution preflight, explicit fixed measurement
outcomes for conditional comparisons, and outcome probability/density records.
Cutoff remains required. Mixed Gaussian/non-Gaussian commands execute wholly in
the explicitly selected Fock backend; no representation switching is inferred.

Use [q,p]=2i, q=a+a†, p=-i(a-a†). `CubicPhase(gamma)` is
exp(i gamma q^3/6), and `QuadraticPhase(s)` is exp(i s q^2/4).
For inverse SUM from input to resource followed by resource q=m, the conditional
wavefunction is psi(q) phi(q+m). A cubic resource has a finite Gaussian envelope
times exp(i gamma q^3/6). Feedforward s=-2 gamma m and Z(-gamma m²) removes
the outcome-dependent polynomial phase, leaving the finite envelope filter.
This is resource-assisted gate injection, not an exact finite-energy unitary.

Homodyne uses normalized Hermite wavefunctions and adaptive quadrature/inverse
CDF in the truncated state. Detection density is not a probability of an exact
real-valued event. Ideal a/a† normalization is not a heralding probability.
Physical subtraction instead couples to vacuum with a beam splitter and counts.

## E. Validation plan

| Feature | Analytical | Independent finite Fock | Raw Piquasso | Graphix | Cutoff |
|---|---|---|---|---|---|
| Number/cat/multimode | number, parity, normalization | coefficient vectors | preparation | lifecycle | cat tails |
| Kerr | phase kappa n² | diagonal exponential | native Kerr | gate ordering | invariant support |
| Cubic | convention, p shift | exp(i gamma q³/6) | native cubic | ordering only | strong gate convergence |
| Ladder | sqrt factors, zero branch | a/a† | shift distinction documented | ordering only | boundary rejection |
| PNR/heralding | binomial probability | projected vectors | PNR | dependencies | branch stability |
| Homodyne/injection | Hermite and psi(q)phi(q+m) | wavefunction integration | native sampler statistics | adaptive causality | fixed outcomes |
| Metrics | vacuum/one-photon Wigner | density matrix distances | moments when available | not applicable | aligned occupations |

Independent dense reference operators live only in tests. Report failure and
unsupported cases, not just happy paths. Strong cubic tests use enlarged cutoffs;
norm conservation alone does not establish convergence of a truncated unitary.

## F. Roadmap

1. Input/command/capability infrastructure and serialization.
2. Fock execution, conditional measurements and diagnostics.
3. Injection/heralding patterns and output analysis.
4. Independent analytical/native/cutoff tests and reproducible experiments.
5. Tutorials, capability audit, revised Quantum-style manuscript and PDF.

## G. Planned files

Modify `src/photographiq/{states,commands,pattern,simulator,serialization,__init__}.py`,
`backends/{base,gaussian,fock}.py`; add `resources.py`, `non_gaussian.py`,
`fock_analysis.py`, `convergence.py`. Add `tests/{fock,non_gaussian,cutoff}/`,
`examples/non_gaussian.py`, `experiments/non_gaussian.py`,
`docs/non_gaussian.md`; revise README, API, validation and specification status.
Update `paper/PhotoGraphiQ.tex`, bibliography, reproducible results and PDF.
If a separate Piquasso-oriented document is requested, label it as a companion
manuscript about this extension, not a revision authorized by Piquasso's authors.
