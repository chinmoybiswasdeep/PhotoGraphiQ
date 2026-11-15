# Fock truncation and non-Gaussian operations

Non-Gaussian states and gates are simulated in a truncated Fock space by the `native-fock`
(pure), `native-mixed-fock` (density matrix) and optional `piquasso-fock` /
`piquasso-mixed-fock` backends. Truncation is a **numerical approximation** and must be
converged; it is separate from the **physical** noise of finite squeezing.

## Fock truncation

PhotoGraphiQ keeps the occupation tuples with **total** photon number below the cutoff,
$\sum_in_i<c$ (exclusive), not a per-mode cutoff. For $m$ modes the dimension is

$$D=\binom{m+c-1}{m},$$

not $c^{m}$: two modes at $c=20$ give $D=210$ and four modes $D=8855$. Basis order: by
total-photon sector, first mode descending inside each sector. A pure state needs $16D$
bytes, a density matrix $16D^{2}$ bytes before workspaces.

**Retained norm.** After every preparation and gate the backend computes the norm kept inside
the truncated space. If it deviates from 1 by more than `norm_tolerance` (default $10^{-3}$)
the operation raises `ValueError`; otherwise the norm is recorded in
`state.retained_norms`, a `RuntimeWarning` is issued, and the state is renormalized. The
retained norm is a diagnostic, **not** a rigorous error bound on observables: always check
that the observables you report converge with the cutoff (`cutoff_convergence`).

```python
import math
import warnings
import photographiq as pg

print([math.comb(m + 20 - 1, m) for m in (2, 4)], [20**m for m in (2, 4)])
# [210, 8855] [400, 160000]

# A coherent input on the native Fock engine: retained norm vs the exact Poisson tail.
alpha = 1.5
pattern = pg.Pattern(pg.CVGraph.line(1, inputs=(0,), outputs=(0,)))
for cutoff in (9, 12, 16):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # the low-cutoff warning is expected
        state = pg.simulate(pattern, inputs={0: pg.GaussianInput.coherent(alpha)},
                            backend="native-fock", cutoff=cutoff).state
    exact = sum(math.exp(-alpha**2) * alpha ** (2 * n) / math.factorial(n) for n in range(cutoff))
    print(cutoff, round(min(state.retained_norms), 10), round(exact, 10),
          round(state.photon_number(0), 6))
# 9 0.9994499388 0.9994499388 2.246135
# 12 0.9999955337 0.9999955337 2.249956
# 16 0.9999999975 0.9999999975 2.25          (exact mean photon number |alpha|^2 = 2.25)
```

At cutoff 6 the same call raises `ValueError: Fock truncation norm 0.97263465; increase
cutoff=6`.

## Non-Gaussian gates

| Gate | Unitary | Notes |
|---|---|---|
| `CubicPhase(gamma)` | $e^{i\gamma q^{3}/6}$ | default `cubic_phase="truncated_generator"`: $\exp(i\gamma Q^{3}/6)$ with $Q$ the truncated position matrix; `"projected"` uses exact projected elements of $q^{3}$. Both converge to the exact gate |
| `Kerr(kappa)` | $e^{i\kappa n^{2}}$ | diagonal, conserves photon number (exact within the truncation) |
| cross-Kerr (backend method) | $e^{i\kappa n_un_v}$ | diagonal |
| Gaussian gates in Fock space | from the symplectic matrix | native one-mode factorization $R\,S\,R\,D$ with a closed-form squeeze factor; validated to $\le1.9\times10^{-14}$ against 60-digit mpmath ([report](../validation/native-fock-gaussian-unitary.md)) |

## Measurements in Fock space

- **Photon counting** samples the exact finite-space probabilities and projects the
  remaining amplitudes; the outcome is an event with a probability.
- **Homodyne** on the pure-Fock backends projects onto the rotated quadrature eigenstate
  using number-state (Hermite-function) wavefunctions; sampling integrates the outcome
  density numerically. The record has a density. Inefficient or noisy homodyne raises on the
  pure-Fock backends and needs a mixed-Fock backend.
- **Postselection** (`measurement_outcomes=`) fixes an outcome and reports its probability or
  density; it is a heralded or postselected branch, not a deterministic operation.
- **Kraus instruments** (finite-outcome POVMs, custom Kraus maps) run on mixed-Fock backends
  ([mixed-state conditioning](mixed-state-conditioning.md)).

## Non-Gaussian resources and heralding

| Resource | API | Meaning of its normalization |
|---|---|---|
| Fock states, superpositions | `FockInput.number(n)`, `FockSuperposition` | exact within the cutoff |
| Cat states | `CatResource` | normalized truncated superposition of coherent states |
| Cubic-phase resource | `CubicPhaseResource` | finite-squeezing cubic state, injected with nonlinear feed-forward |
| Photon-added / -subtracted inputs | `PhotonAdd`, `PhotonSubtract` | ideal operators, normalized offline; **not** a detector heralding probability |
| Heralded subtraction | `non_gaussian.photon_subtraction(theta)` | a beam splitter tap and photon counting; the herald probability is reported by the measurement |
| Finite-energy GKP | `GKPResource`, `GKPCode` | see [GKP](gkp.md) |

## Mixed states, loss and nonlinear synthesis

Density-matrix evolution, Fock loss and Kraus channels:
[mixed-state conditioning](mixed-state-conditioning.md). Product-formula Kerr and quartic
synthesis and its error budget: [nonlinear synthesis](nonlinear-synthesis.md). Gradients
of fixed-branch Fock execution: [gradient estimators](gradient-estimators.md).

## Historical derivations

The v0.2 derivations of ideal Fock homodyne, ladder factors, finite-envelope cubic injection
and phase-space normalization remain in the
[v0.2 non-Gaussian reference](../non_gaussian.md). That page predates the native C++ Fock
engines and the mixed-Fock backends; where it says a feature is unavailable, this page and
the [feature matrix](../validation/feature-matrix.md) take precedence.
