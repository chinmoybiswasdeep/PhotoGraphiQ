# Foundations of the typed CV diagram calculus (`photographiq.calculus.cv_diagrams`)

This note fixes the semantics of the diagram calculus and gives, for every implemented rewrite,
its statement, assumptions, equivalence class and proof or precise source.

It is **not** a complete CV-ZX calculus. Completeness is neither claimed nor tested, and only
the fragment listed here is implemented.

## 1. Literature and what is (not) adopted

| Work | Content (as read from the primary text) | Use here |
|---|---|---|
| Menicucci, Flammia, van Loock, *Graphical calculus for Gaussian pure states*, PRA 83, 042335 (2011) (MFvL) | Pure Gaussian states as complex-weighted graphs Z = V + iU. Unitary update Z′ = (C + DZ)(A + BZ)⁻¹. q-homodyne deletes a vertex; p-homodyne gives a Schur complement (Sec. III). | Already implemented in `GaussianGraph`. Used as an independent reference. Rule CZ + q-homodyne (§4.9) is the process-level form of vertex deletion. |
| Booth, Markham, *Flow conditions for continuous-variable MBQC*, Quantum 7, 1146 (2023) | Real-linear CV flow: correction-matrix condition, Defs. 5–6, Lemma 7. | Already implemented (`cvflow`, `flows.cv`). Used to certify candidate patterns and for flow-layered schedules. |
| Nagayoshi et al., *ZX graphical calculus for continuous-variable quantum processes*, Phys. Rev. Research 7, 033141 (2025), arXiv:2405.07246 | Generators (Table II): q-spider ∫ds e^{if(s)}\|s…s⟩⟨s…s\|, p-spider, swap, Fourier, inverse and squared Fourier. f is a real polynomial; ħ = 1. Equivalence is up to a nonzero scalar (Def. 13), so scalars are ignored. Rules (Sec. IV.A) include identity (69), fusion (70–71), bialgebra (72–73), Fourier (74–76), copy (77–78), displacement (79–80), antipode (81), squeezing (82–83), quadratic (84–87) and inversion (88–89). Soundness is proved in their App. A.2. Completeness is claimed only for one-mode Gaussian states and operations. | Only the **1 → 1 q-spider** fragment is adopted (`PhasePolynomial` = exp(i f(q̂)), restated with ħ = 2), with fusion (70) and displacement (79) as operator identities (§4.12–4.13). Multi-leg spiders, Fourier boxes, bialgebra, copy, antipode and quadratic rules are **not implemented**. Their rotation convention is q → q cos θ + p sin θ, so R_Nagayoshi(θ) = `gaussian.rotation(−θ)`. |
| Booth, Carette, Comfort, *Complete equational theories for classical and quantum Gaussian relations*, arXiv:2403.10479 | Generators and relations for the hypergraph props of Gaussian relations and positive affine Lagrangian relations (the latter extend Gaussian quantum mechanics with infinitely squeezed states). | Not implemented. Cited as the established complete calculus for the relational (infinite-squeezing) Gaussian fragment. Our calculus works with finite-squeezing channels and instruments instead. |
| Booth, Carette, Comfort, *Graphical symplectic algebra*, arXiv:2401.07914 (FSCD 2026) | Complete presentations of the dagger-compact props of affine Lagrangian and coisotropic relations over any field. Includes an application section "Picturing quantum optics" (6.2.2). | Not implemented. Cited. |
| Shaikh, Yeh, Gogioso, *The Focked-up ZX calculus*, arXiv:2406.02905 | A Fock-basis spider extension. | Not implemented. Cited as a related, different convention. |

Status of the material in this note:

- **Established and cited:** MFvL graph rules, Booth–Markham flow, and the Nagayoshi q-spider fusion and displacement rules (their proofs).
- **Derived here, with proof:** the process-level rules of §4 (elementary operator identities).
- **Numerically investigated conjectures:** none is implemented as a rule.
- **Unsupported:** see §6.

## 2. Conventions

- **Quadratures:** ħ = 2, [q̂, p̂] = 2i, interleaved (q₁, p₁, …), vacuum covariance I.
- **Displacement:** D(a) displaces the quadratures by a = (a_q, a_p).
- **Rotation:** `Rotation(θ)` acts as x → [[cos, −sin], [sin, cos]] x.
- **Squeezing:** `Squeezer(r)` acts as diag(e^{−r}, e^{r}).
- **Shear:** `Shear(s)` = exp(i s q̂²/4), acting as p → p + s q.
- **CZ:** `CZ(g)` = exp(i g q̂_u q̂_v/2), acting as p_u += g q_v and p_v += g q_u.
- **Phase polynomial:** `PhasePolynomial(c)` = exp(i Σ c_k q̂^k), acting as p → p + 2 f′(q). Hence c₁ ↦ p-displacement 2c₁, c₂ ↦ shear 4c₂, and the PhotoGraphiQ cubic phase exp(iγq̂³/6) has c₃ = γ/6.
- **Homodyne:** records q cos θ + p sin θ, plus detector noise (1 − η)/η + σ².
- **Resource preparation:** `Prepare.squeezed(r)` has covariance diag(e^{2r}, e^{−2r}).

## 3. Semantics

**Diagrams.** A diagram is an immutable, causally ordered list of typed boxes on wires of two
types: *bosonic* (linear, never copied) and *real* (classical records, freely copyable, read
only after being written). Typing errors are rejected (`DiagramTypeError`).

**Hybrid Gaussian semantics.** For Gaussian diagrams, every live coordinate (the Wigner
variables of the live modes, and the records) is

  z = A x_in + b + ξ,  with ξ ~ N(0, Y).

Each box updates (A, b, Y) as follows:

- **Preparation:** a new independent block.
- **Unitary:** x → S x + d on its rows.
- **Channel:** rows → X rows + d, Y += Y_box.
- **Homodyne:** replaces the two Wigner coordinates of its mode by one record coordinate h·(q, p), with h = (cos θ, sin θ), and adds variance (1 − η)/η + σ².
- **Feed-forward:** an affine map adding linear functions of records to quadratures.

**Correctness of the homodyne step.** The law of a homodyne outcome on a state with Wigner
function W is the marginal of W along the measured quadrature. The joint distribution of the
outcome and the Wigner variables of the other modes is obtained by integrating W over the
conjugate quadrature of the measured mode only. For Gaussian W this is again Gaussian.
Conditioning on the outcome gives the conditional state (the Schur complement used by
`GaussianBackend.measure`). Detector inefficiency and added noise convolve the outcome with
independent Gaussian noise.

Consequences:

1. The **conditional instrument** (record density and conditional output state) is the
   Gaussian conditioning of the joint law on the records (`HybridGaussian.conditional`).
2. The **averaged channel** is obtained by dropping the record coordinates
   (`HybridGaussian.channel`).
3. Two diagrams have the same instrument for **every** input iff their affine data (A, b, Y)
   restricted to the outputs and records coincide. This is the exact, input-independent test
   used by `compare`. No sampling of inputs is involved.

**Non-Gaussian boxes** have truncated Fock semantics only (`fock.run_fock`). Agreement at
finite cutoff is numerical evidence and is never used as proof of an identity.

## 4. Rewrite rules: statements and proofs

Every rule matches *wire-adjacent* boxes: no box in between touches the matched wires. Moving
a box past such intermediate boxes is justified by rule 4.7.

Equivalence classes:

- **I**: exact instrument. The full conditional instrument is preserved. Global or branch
  phases do not affect density operators.
- **C**: exact channel. The averaged channel is preserved; the rewritten region has no records.
- **L**: logical. The ideal map (X, d) is preserved, but the noise Y and the records change.

**4.1 Rotation fusion (I).** R(α)R(β) = R(α + β). One-parameter subgroup e^{iθn̂}. ∎

**4.2 Squeezing fusion (I).** S(r₁)S(r₂) = S(r₁ + r₂). This holds for same-axis squeezers;
all `Squeezer` boxes squeeze q. One-parameter subgroup generated by (q̂p̂ + p̂q̂). ∎

**4.3 Shear fusion (I).** exp(i s₁q̂²/4) exp(i s₂q̂²/4) = exp(i(s₁ + s₂)q̂²/4), since these
are commuting functions of q̂. ∎

**4.4 Displacement fusion (I, with recorded phase).**

  D(b)D(a) = exp(i(a_q b_p − a_p b_q)/4) D(a + b)  (ħ = 2).

Proof by Baker–Campbell–Hausdorff: [q̂, p̂] = 2i makes the commutator of the generators a
c-number. The phase sign is checked against a dense Fock `expm` (R61, max residual ≈ 1e-16;
`tests/calculus`). ∎

**4.5 Identity removal (I).** Each generator at its identity parameter is the identity:
rotation 0 mod 2π, r = s = g = 0, D(0), BS(0 mod 2π), S = I, d = 0, η = 1, and constant phase
polynomials. A constant phase polynomial contributes the global phase e^{ic₀}, which is
recorded. ∎

**4.6 Symplectic fusion (I).** (S₂, d₂) ∘ (S₁, d₁) = (S₂S₁, S₂d₁ + d₂) for Gaussian
unitaries, embedded on the union of their modes. Up to an unrecorded global phase. ∎

**4.7 Commutation (I).** Boxes on disjoint bosonic wires, where the second does not read a
record written by the first, act on different tensor factors and commute. Classical
dependencies are preserved. ∎

**4.8 Channel fusion (C).** For Gaussian channels: (X₂X₁, X₂Y₁X₂ᵀ + Y₂, X₂d₁ + d₂). Complete
positivity (Y + iΩ − iXΩXᵀ ≥ 0) is preserved under composition and re-checked by
`GaussianChannelBox`. ∎

**4.9 CZ followed by an ideal q-homodyne → feed-forward (I).** Let the homodyne of u be ideal
(η = 1, σ = 0) at θ ∈ {0, π}, with sign s = cos θ. Then

  ⟨m|_u CZ_{uv}(g) = exp(i g s m q̂_v / 2) ⟨m|_u,

which is a p-displacement of v by g s m, conditioned on the record m. The CZ commutes with
q̂_u, so measuring first and displacing afterwards gives the same record law and the same
conditional state for every input. This is the process form of MFvL's q-measurement vertex
deletion. With η < 1 or σ > 0 the record is not a q̂ eigenvalue and the rule fails; the code
rejects it. ∎

**4.10 Preparation followed by homodyne → classical source (I).** A Gaussian preparation with
mean μ and covariance V, measured next by a homodyne h, produces a record with law
N(h·μ, hVhᵀ + (1 − η)/η + σ²) and leaves no quantum wire. Replacing both boxes by a classical
Gaussian source preserves the joint law of all records and outputs.

Combined with 4.9, an ancilla that is prepared, CZ-coupled and q-measured becomes classical
noise driving a feed-forward displacement. This is exact; it is the finite-squeezing noise
channel written classically.

*Recompilation:* `Pattern` has no classical-randomness command, so `to_pattern` re-realises a
source as an ancilla prepared with the same q variance and measured in q. ∎

**4.11 Feed-forward merge (I).** Displacements commute up to a phase. With outcome-dependent
gains the phase depends on the outcomes but is global within each branch, so density
operators are unchanged. Gains and constants add. ∎

**4.12 Phase-polynomial fusion (I; Nagayoshi Eq. 70, 1 → 1 case).**

  exp(i f(q̂)) exp(i g(q̂)) = exp(i(f + g)(q̂)).

For real polynomials f and g these are commuting multiplication operators in the q
representation, hence exact as unitaries on L²(ℝ). ∎

**4.13 Displacement through a phase polynomial (I; Nagayoshi Eq. 79, 1 → 1 case).** With
X(a) the q-displacement, X(a)† q̂ X(a) = q̂ + a, so

  exp(i f(q̂)) X(a) = X(a) exp(i f(q̂ + a)).

The coefficients of f(q + a) are Σ_{n≥k} c_n C(n, k) a^{n−k}, computed exactly. For a
polynomial f, the functional calculus applies on the invariant domain of Schwartz functions,
and both sides are unitary.

The *false* variant commutes exp(i f(q̂)) with X(a) without shifting f. It is not an
identity, because X(a)† q̂ X(a) = q̂ + a. R68 uses it as a negative control: its truncated
infidelity stays ≥ 1e-3 while the true rules converge.

Note that a *p*-displacement is exp(i b q̂/2), itself a function of q̂, so it *does* commute
with exp(i f(q̂)). R68's first run used it as the control by mistake and found exactly that. ∎

**4.14 Teleport-chain resynthesis (L).** A corrected wire step consists of: preparation with
squeezing r; CZ(a, b, 1); ideal homodyne of a at θ; correction q_b −= m / sin θ. Its averaged
channel is x → T(k)x with noise diag(0, e^{−2r}), where k = cot θ (`gaussian.wire_channel`,
exact at every r).

A chain of L steps therefore implements S = T(k_L)⋯T(k₁) exactly at every r. The rule replaces
it with the fewest steps found among:

- 0 steps (S = I);
- 1, 2 or 3 steps, using explicit closed forms checked entrywise;
- the compiler's 4- or 5-step decomposition.

The new product is verified equal to S to 1e-9. Hence X and d are unchanged at every r.

The noise is Σ of the propagated diag(0, e^{−2r}) terms, which depends on the k_i, so it
changes. It decreases in most cases but not all: R62 found an increase in 4 of 60 programs.
The record set changes too. The rule is therefore **logical**, never physical. ∎

**4.15 Flow-layered scheduling (used in R64, not a box rule).** Rebuilding a cluster pattern
from the layered Booth–Markham certificate (`find_cv_flow` → `corrected_pattern`) changes
which corrections are applied, so equivalence is not assumed. R64 measures it: on all 15 grid
clusters, the averaged channel and even the conditional instrument agree with the
sequential-order pattern to ≤ 1e-14, while the dependency depth falls (for example 40 → 20
over three-wire clusters).

## 5. Verification policy

`compare` reports logical, physical and conditional equivalence separately, with the residual
of each matrix and an absolute plus relative tolerance. Experiments and tests check every rule
against references that do not share its code:

- `analysis.gaussian_channel` (source-vector propagation of the executable `Pattern`);
- postselected trajectories on the Gaussian backend (sequential Schur complements);
- `GaussianGraph` (MFvL);
- a dense Fock `expm` reference;
- closed forms: the Poisson herald probability, the coherent-state output, the RBF and Weyl
  phases;
- the untruncated GKP grid model.

Numerical agreement *supports* a proved rule. It is never used *instead of* a proof.

## 6. Not supported (explicitly)

- Nagayoshi multi-leg spiders, Fourier boxes, bialgebra, copy, antipode, quadratic, inversion
  and squeezing rules, and any CV-ZX completeness.
- The relational (infinitely squeezed) semantics of Booth–Carette–Comfort.
- Adaptive (outcome-dependent) homodyne angles, nonlinear feed-forward and vector-record
  feed-forward. Translation rejects them.
- MBQC lowering or optimisation of Kerr gates.
- Rewrites that change heralding probabilities. No such rule exists; heralded diagrams are
  evaluated, not rewritten.
- General non-Gaussian MBQC optimisation.
- GKP logical rewrites: GKP diagrams are executions of existing PhotoGraphiQ primitives only.
- Channel distances: the approximate mode uses an entrywise max|ΔY| budget, which is not a
  diamond-norm bound.
- Fault tolerance and quantum advantage: not claimed.
