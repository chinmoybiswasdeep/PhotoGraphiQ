# Experimental non-Gaussian MBQC

PhotoGraphiQ 0.2 extends the existing pattern, dependency and backend model.
Piquasso 8.0.1 executes physical Fock gates. PhotoGraphiQ supplies labelled
resources, causal measurement updates, nonlinear classical signals and injection
patterns. Graphix remains a **structural** reference, never a numerical CV oracle.
The [preimplementation audit](non_gaussian_design.md) records the design decisions.

## Selecting capabilities and resources

```python
import photographiq as pg
from photographiq.backends.fock import PiquassoFockBackend

backend = PiquassoFockBackend(cutoff=24)
assert backend.supports("cubic_phase")
assert backend.supports("photon_counting")
assert backend.supports("cat_state")
```

The Fock backend requires an explicit integer cutoff `c >= 2`. For `d` modes it
stores occupations with **sum(n) < c**, a space of dimension `comb(d+c-1,d)`.
There is no independent cutoff per mode. `backend.dimension(d)` estimates vector
size; complex128 vectors alone require 16 bytes per amplitude. Native gate
workspaces and density matrices can be much larger. The default allocation guard
rejects dimensions above 1,000,000 and warns at 100,000. This is a vector-space
guard, not a bound on total process memory. Modest one/two-mode studies are the
intended starting point; `experiments/non_gaussian.py` records local wall times.

Non-Gaussian requests require the Fock backend; Gaussian execution rejects them
during capability preflight. Select `backend="piquasso-fock"` explicitly.
Hybrid patterns execute all their Gaussian and non-Gaussian commands in Fock
space. There is no automatic, hidden Gaussian-to-Fock conversion. Pure Gaussian
inputs are explicitly converted when supplied to this backend; mixed ones fail.

```python
resource = pg.FockSuperposition.from_mapping({(0, 0): 2**-0.5, (1, 1): 2**-0.5})
pattern = pg.Pattern().append(pg.PrepareResource(("a", "b"), resource))
result = pg.simulate(pattern, backend="piquasso-fock", cutoff=8)
```

Occupation tuples follow preparation-label order. Correlated inputs can instead
be passed as `initial_state=resource` to a pattern with matching ordered inputs.
`FockSuperposition.number((2,1))` prepares a basis state;
`FockSuperposition.from_piquasso(native_pure_state)` copies normalized public
native amplitudes. Mixed native states are rejected. `FockInput` continues to
represent a single-mode finite vector. Inputs are immutable, finite and normalized;
invalid shapes, occupation values, duplicate bases and excess support fail.

`CatResource(alpha, parity=±1)` regenerates the projected cat at each execution
cutoff and records its retained infinite-state weight. `FockInput.cat` supplies
an explicitly normalized finite vector instead; when comparing cutoffs, rebuild
that vector with a pattern factory. Odd cats at alpha=0 have zero norm and fail.
`CubicPhaseResource(gamma, squeezing=r)` prepares CP(gamma) S(-r)|0> at the
execution cutoff. Finite squeezing and truncation are distinct approximations.

## Conventions, operations and classical control

Use `[q,p]=2i`, vacuum quadrature variance 1, `q=a+a†`, `p=-i(a-a†)`.

| Command | Operation |
|---|---|
| `CubicPhase(node,gamma)` | exp(i gamma q³/6), hence p → p+gamma q² |
| `Kerr(node,kappa)` | exp(i kappa n²) |
| `QuadraticPhase(node,s)` | exp(i s q²/4), hence p → p+s q |
| `Displace(node,q=x,p=z)` | q → q+x, p → p+z |
| `PhotonAdd(node)` | a† psi / norm(a† psi) |
| `PhotonSubtract(node)` | a psi / norm(a psi) |

To represent exp(i lambda q³), pass `gamma=6*lambda`. This preserves the v0.1
and native Piquasso convention. The ladder operations apply the mathematical
sqrt occupation factors. Piquasso 8.0.1 `Create`/`Annihilate` have unit shift
coefficients in the inspected implementation and are not used as these operators.
`ladder_norm_squared` is a normalization diagnostic, **not a success probability**.
Numerically zero ladder norm (≤1e-28) and additions leaving the cutoff fail.

```python
pattern = pg.non_gaussian.photon_subtraction(theta=0.2)
pattern.commands.pop()  # insert a correction before the final Output
pattern.append(pg.Kerr("in", 0.3 * (-1) ** pg.Outcome("count")))
pattern.append(pg.Output(("in",)))
```

Count records are nonnegative integers. The same expression tree handles
polynomials, parity-dependent corrections and adaptive homodyne angles.
Command dependencies, mode lifetimes, scheduling and safe JSON serialization
apply to the extension. A causal DAG is not a non-Gaussian determinism proof.

## Conditional measurements and physical heralding

PNR computes Born probabilities, samples or selects a count, and normalizes the
surviving amplitudes. Sequential PNR on several modes is joint destructive
counting; the product of its conditional probabilities is the joint probability.

Ideal homodyne contracts amplitudes with
`<x_theta|n> = exp(-i n theta) He_n(x) exp(-x²/4) / ((2*pi)**0.25 sqrt(n!))`.
The marginal density is the squared norm of this contraction. Sampling uses
adaptive quadrature and inverse CDF with absolute/relative integration tolerance
1e-10 and root tolerance 1e-10. A finite interval beyond the classical turning
point is accepted only when its integrated mass agrees with one to 1e-7.
This is numerical sampling of an exact finite-expansion conditional state.
Piquasso's native homodyne sampler does not return that conditional state.

```python
result = pg.simulate(
    pg.non_gaussian.photon_subtraction(0.2),
    backend="piquasso-fock",
    cutoff=24,
    inputs={"in": pg.CatResource(1.0)},
    measurement_outcomes={"count": 1},
    seed=42,
)
print(result.measurement_statistics)  # probability and postselected flag
```

The physical tap mixes the input with vacuum then measures the ancilla. For
input |n>, count k has probability C(n,k) sin(theta)^(2k) cos(theta)^(2(n-k)).
Finite-angle single-photon heralding includes attenuation; it is not the ideal
normalized annihilation operation. Postselection is explicit and impossible
outcomes raise errors. It does not simulate the number of trials until success.

`measurement_statistics[key]` contains `kind` (`probability` or `density`),
`value`, and `postselected`. `exp(result.log_likelihood)` is a probability only
when every measurement is discrete. Exact homodyne outcomes have zero event
probability; their reported densities must be integrated over detector bins.
Gaussian backends currently do not supply these likelihood diagnostics.
Seeded runs are reproducible, and `run_shots` spawns independent child seeds.

## Cat and cubic resource injection

`resource_injection` applies inverse SUM from the input to the resource using
Fourier-CZ-Fourier gates, measures the resource q=m, and retains the input.
In the infinite Fock-space description its unnormalized output is
`psi(q) phi(q+m)`. `cat_injection(alpha,parity)` therefore implements a cat
wavefunction filter; no deterministic cat gate is claimed.

For a cubic resource with variance exp(2r), expansion of `(q+m)^3` gives the
conditional phase. `cubic_injection(gamma,r)` corrects it with
`QuadraticPhase(-2 gamma m)` and `Z(-gamma m²)`. Up to global phase, the output is

```
psi(q) exp(i gamma q³/6) exp(-(q+m)²/(4 exp(2r))).
```

The envelope remains: finite-energy injection is a conditional filter followed
by a cubic phase, not an exact unitary cubic gate. The output stays on the input
node; this is gate injection rather than relocation of the input state.
The protocol is related to cubic-resource/adaptive-measurement approaches, but
the particular finite-Fock implementation is validated by its displayed map.

```python
pattern = pg.non_gaussian.cubic_injection(gamma=0.3, squeezing=0.2)
study = pg.cutoff_convergence(pattern, [36, 48, 64], measurement_outcomes={"m": 0.4})
print(study.rows[-1])
```

## Diagnostics and convergence

Outputs expose `state_vector` (pure only), `density_matrix`, ordered `basis`,
`probabilities`, `norm`, `reduced(nodes)`, `photon_number`, `quadrature` and `parity`.
Quadrature second moments include the vacuum boundary term instead of squaring
a prematurely truncated quadrature matrix. Fidelity is **squared Uhlmann
fidelity**; trace distance is half the trace norm. States are aligned by occupation
tuples across cutoffs and require matching ordered output labels.

`state.wigner(q_axis,p_axis,node=...)` computes a single-mode marginal using
analytic displacement matrix elements. Its `values` have shape `(len(p),len(q))`
and normalize over dq dp. `captured_mass` and `negative_volume` use trapezoidal
integration; negative volume is the finite-grid integral of max(-W,0).
It is not automatically corrected for omitted tails. Check both grid refinement
and window expansion before quoting negativity. `grid.plot()` requires Matplotlib.

Every gate/preparation records retained norm and population in the last two
total-photon shells. Norm deviation above 1e-3 (configurable) fails; smaller
deviations above 1e-8 warn before normalization. Boundary population above .02
(configurable) warns independently. These diagnostics are not error bounds.
In particular native cubic phase exponentiates a truncated q³ and remains
unitary even when the cutoff is inadequate.

`cutoff_convergence` reports norms, boundary populations, peak dimensions, photon
and quadrature moments, aligned fidelity, trace distance and probability L1
change. These are adjacent-cutoff comparisons, not proven infinite-space errors.
Different outcomes mark branches incomparable; equal random seeds do not fix a
branch. Use explicit outcomes for conditional comparisons. A factory `c -> pattern`
rebuilds cutoff-dependent resources. Timings are local observations with warmup
effects, not speed claims against other simulators.

## Boundaries and validation

Implemented: pure Fock resources, Gaussian/Fock unitary sequences, ideal PNR and
homodyne, nonlinear signals, finite cat/cubic injection, physical subtraction,
mathematical addition/subtraction, metrics and cutoff studies.
Unsupported: mixed input injection, Fock loss channels, noisy Fock homodyne,
Fock heterodyne/general-dyne, dynamic representation switching, GKP preparation
and decoding, fault tolerance and universal non-Gaussian compilation.
Future resources should be explicit preparation descriptions with capabilities
and independent tests; no apparently functional GKP placeholder is exported.

Tests in `tests/fock`, `tests/non_gaussian`, `tests/cutoff` use analytical
wavefunctions, independent dense operators in tests only, direct public Piquasso
programs and increasing cutoffs. Graphix tests compare only the causal skeleton.
Run `python experiments/non_gaussian.py` to regenerate CSV, figures and provenance,
and `python examples/non_gaussian.py` for a complete tutorial script.


## Hardening: high-order moments and numerical scope

`state.quadrature_moment(node, order, angle=0)` returns raw moments for integer
orders 0 through 4. `state.photon_moment(node, order=2)` supplies photon moments.
`cutoff_convergence(..., high_order_moments=True)` records q?,q?,q?,p?,p?,p?,n?
per output. These observables are optional and currently specific to Fock
snapshots. For a state with stored occupation n<c, ladder paths may temporarily
reach n+order before returning. Those paths are included: powers of PqP would
otherwise give the wrong physical moments at the highest stored level.
These are moments of the normalized finite-support state, not guarantees of
infinite-cutoff convergence; high moments can converge much more slowly than
fidelity. For cubic phase on vacuum the infinite-space targets are
<p>=gamma, <p?>=1+3gamma?, <p?>=gamma+15gamma? and
<p?>=3+10gamma?+105gamma?. Noncommutativity matters for the last two identities.

The fixed homodyne window is justified by finite Fock support: for any normalized
single-mode marginal, p(x) <= sum(n<c) |h_n(x)|? by Cauchy-Schwarz (and convexity
for mixed marginals). It therefore covers all represented states, regardless of
how displacement or squeezing was prepared. Tests include coherent alpha=4,
squeezing r=.7, cats and cubic states at c=96, and an independent Hermite tail
envelope. This does **not** certify an inadequately represented infinite-state
input: alpha=10 at c=24 is rejected by retained norm before sampling. Both total
mass and partial-CDF error estimates are checked; the turning-point bound is not
an unlimited accuracy guarantee at arbitrary enormous cutoff.

The Fock preparation preflight validates mode count, explicit support and
Gaussian physicality before earlier native preparations run. Gaussian purity uses
an explicit 1e-10 absolute determinant tolerance, with zero relative tolerance;
slightly mixed states beyond it are rejected rather than silently purified.
Generic mixed native/reduced states cannot be injected into pure execution.
Pure norms and pure/pure state metrics avoid dense density allocation. Native
partial reduction and mixed/mixed metrics still require dense matrices; the
vector-dimension guard does not make those operations scalable to a million modes
or amplitudes. Plan memory for the specific observable as well as the trajectory.

Squared fidelity is |<psi|phi>|? for pure states and Tr(rho sigma) if either
state is pure. Mixed/mixed uses the squared Uhlmann expression. Comparisons
require identical ordered labels; use `reduced(ordered_labels)` explicitly to
reorder. Empty outputs have unit scalar state and fidelity one. A reduced native
mixed-state representation does not expose a pure vector, even if a special
parameter value happens to give a rank-one matrix.

The cubic correction derivation includes the global phase:

1. The chronological resource rotations and CZ give q_a -> q_a-q_in.
2. Selecting resource q_a=m contracts psi(q) phi(q+m).
3. gamma(q+m)?/6 = gamma q?/6 + gamma m q?/2 + gamma m?q/2 + gamma m?/6.
4. Q(-2gamma m) and Z(-gamma m?) cancel the middle two terms.
5. The global phase exp(i gamma m?/6) is immaterial; the Gaussian envelope remains.

The [hardening audit](hardening-audit.md) and [release report](release-hardening-report.md)
separate observed defects, finite-cutoff sensitivity and unsupported future work.
