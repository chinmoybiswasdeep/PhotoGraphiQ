# Issues found while building the experiment suite

Per the task's instructions, nothing here was silently patched. These are
documented observations with minimal reproducers; core source was not
modified as part of building this suite.

## 1. Two validator error messages are unreachable dead code (minor, behavior is still correct)

`Pattern.validate()` (`src/photographiq/pattern.py`) documents two distinct
error messages -- `"Self CZ is invalid"` for `Entangle(u, u, ...)`, and
`"Duplicate outputs"` for an `Output` command that lists the same node
twice -- but neither can actually be raised for the simplest triggering
case. `flow.dependency_graph` is called unconditionally at the very start of
`validate()`, and it builds a "last touch" edge for every quantum node a
command references, iterating `quantum_nodes(command)` without deduplicating
repeats. When a command references the same node twice (`Entangle(0, 0, 1.0)`,
or `Output((0, 0))`), this produces a *self-loop* in the dependency graph,
which `nx.is_directed_acyclic_graph` reports as **not a DAG** -- so
`dependency_graph` raises `"Cyclic command dependencies"` before the
per-command loop ever reaches the specific check further down.

**The pattern is still correctly rejected** -- this is not a silent
correctness bug, only an imprecise/unreachable diagnostic message for this
particular trigger shape. It is possible a *different* code path could still
reach the specific messages (e.g., a duplicate output introduced by
`standardize()` reordering, or a self-CZ constructed some other way that
does not create a literal repeated entry in one command's `quantum_nodes()`
tuple) -- this was not exhaustively searched.

Minimal reproducer (see `03_adaptive_flow/13_invalid_causality.py`,
cases `self_cz` and `duplicate_outputs`):

```python
import photographiq as pg
from photographiq.commands import Entangle, Output

# Raises "Cyclic command dependencies", not "Self CZ is invalid":
pg.Pattern(inputs=(0,)).append(Entangle(0, 0, 1.0)).validate()

# Raises "Cyclic command dependencies", not "Duplicate outputs":
pg.Pattern(inputs=(0,)).append(Output((0, 0))).validate()
```

Recorded (not asserted as a failure) in `results/csv/R13_invalid_causality.csv`.

## 2. Two-mode Fock `GaussianTransform` entangling gate becomes numerically unstable at moderate-to-large cutoff for wide-support inputs

While sizing R28 (GKP syndrome extraction, which entangles two GKP-scale
Fock modes), the native Piquasso `GaussianTransform` instruction used by
`PiquassoFockBackend.entangle` (`src/photographiq/backends/fock.py`) produced
a state with numerically nonsensical norm (as large as ~1.67e6, and
non-monotonically increasing through ~0.97-1.97 immediately before that) for
a two-mode entangling gate at cutoffs from roughly 120 to 140, using a
`peak_width=envelope=0.2` GKP "logical plus" ancilla entangled with a
similarly sharp input. At cutoff=80 with a slightly wider `peak_width=0.35`
resource, the same gate behaves correctly (norm within `1e-3` of one).

**This was not a silent failure**: `PiquassoFockBackend._check` correctly
rejected every one of these cases via its retained-norm tolerance
(`ValueError: Fock truncation norm ...`), so `simulate()` raised cleanly
rather than returning a corrupted state. R28 was designed around a cutoff
(80) and width (0.35) combination confirmed to pass this check. The
takeaway for future experiments: sharper/higher-energy GKP-style resources
sharing one total-photon cutoff across two entangled modes may need
disproportionately larger cutoffs than a single-mode captured-mass estimate
alone would suggest, and the native `GaussianTransform` path should not be
pushed past the point where the norm check starts failing, since its
pre-normalization behavior in that regime is not merely "truncated" but
visibly unstable (super-unity, then divergent, norms).

Minimal reproducer:

```python
import photographiq as pg

for cutoff in (100, 120, 130, 140):
    try:
        plus = pg.gkp.superposition(1, 1, cutoff=cutoff, peak_width=0.2, envelope=0.2)
        pattern = pg.gkp.correction_pattern(quadrature="q", resource=plus)
        code0 = pg.GKPResource(0, peak_width=0.2, envelope=0.2)
        fock0, _ = code0.project(cutoff)
        result = pg.simulate(pattern, backend="piquasso-fock", cutoff=cutoff, inputs={"in": fock0}, seed=0)
        print(cutoff, "norm", result.state.norm)
    except ValueError as exc:
        print(cutoff, "rejected:", exc)
```

Observed on this checkout (piquasso 8.0.1): cutoff=100/110 rejected with
norm just under 1 (~0.98-0.99, expected finite-cutoff truncation); cutoff=120
rejected at norm ~0.97; cutoff=130 rejected at norm ~1.97 (super-unity,
not simple truncation); an earlier ad hoc probe at cutoff=140 raised the
same guard at norm ~1.67e6. Not raised as a suite failure anywhere, since
the safety net worked as designed; recorded here as a numerical-stability
observation for anyone pushing two-mode Fock GKP experiments to larger
cutoffs or sharper widths.

## 3. (Design lesson, not a defect) Single-shot GKP syndrome outcomes are dominated by which ancilla comb peak was sampled

Not a code issue, but worth recording so a future experimenter does not
mistake it for one: a single trajectory's raw ancilla homodyne outcome from
`gkp.correction_pattern` samples across the ancilla's *entire* multi-peak
comb (peaks are spaced by exactly `SPACING`, per `GKPResource.wavefunction`),
so `decode_shift`'s per-cell residual has large shot-to-shot variance at
moderate `peak_width`/`envelope` (~0.35). The *ensemble* (circular) mean
over repeated seeds is centered correctly on the true imposed shift (see
R28's redesigned methodology and its passing result), but any single-shot
comparison to the imposed displacement should not be expected to match
closely without averaging over many trajectories first.

---

# Upgrade migration (2026-11-13): findings while porting R1-R42 to the upgraded software

Context: the R1-R42 scripts were migrated to the native C++ engines (`native-gaussian`,
`native-fock`, `native-mixed-fock`), `photographiq.flows`, the typed IR and the planner. The
historical outputs are frozen in `legacy_reference/441de9a/`. Entries 1-3 above are unchanged.
Migration status per experiment: `UPGRADE_MIGRATION_MATRIX.md`.

## 4. R2 legacy figure compared Tr(N) with the per-quadrature noise (factor-2 mislabel; numbers were correct)

The four-step identity wire adds N = 2 e^(-2r) I, so Tr(N) = 4 e^(-2r). The legacy R2 figure
plotted Tr(N) next to a curve labelled "2e^(-2r) (theory)", which is the per-quadrature value,
so the two curves were a factor 2 apart. The asserted checks were unaffected (they compare
against `gaussian.wire_channel` matrices). Fixed in the script: new columns
`predicted_noise_trace` (4 e^(-2r)) and `noise_trace_error_vs_4exp` (asserted <= 1e-9); the old
column `predicted_added_variance` keeps its historical value and meaning (per quadrature).

## 5. Noisy homodyne SAMPLING on the mixed-Fock engines is still ~8-22 s per shot, also on `native-mixed-fock`

`native-mixed-fock` reuses the Python measurement path of `MixedFockBackend`
(`src/photographiq/backends/mixed_fock.py`): for `Homodyne(noise != 0)` each sampled outcome
integrates the dense density matrix with `scipy.integrate.quad_vec` inside a root find. Measured
on this (shared) machine: 22 s per shot at cutoff 10 on `native-mixed-fock`. The postselected
density of the same measurement (`measurement_outcomes`) costs 0.015 s and agrees with the
analytic Gaussian density N(2 Re alpha, 1 + nu) to 1.1e-8 (R32, 56 points, both mixed engines).
R32 therefore checks the noisy branch deterministically by density and keeps the historical
6-shot noisy sampling only under `--full`. Possible library fix (not applied, outside scope):
sample by inverting a CDF tabulated once per measurement on a grid instead of per-shot
root finding.

## 6. Homodyne sampling on the pure Fock engines is also per-shot expensive (~3-4 s per shot for GKP states)

A probe (100 q-homodyne shots of a finite GKP |0>_L, width 0.35, cutoff 48) took 430 s on
`native-fock` and 330 s on `piquasso-fock` (shared machine); the records were identical between
the engines (max difference 0.0). The cost is the shared Python sampling path, not the C++
kernels. R27 therefore computes GKP readout error probabilities exactly by integrating
|psi(q)|^2 over the wrong decoding cells instead of sampling.

## 7. (Numerical conditioning, not a defect) 100 chained squeezers: engines agree only to ~eps x cond

In R10 the 100-squeezer chain has total logical squeezing r = 29.75 (cond(S) = e^59.5 ~ 7e25,
~258 dB). There `native-gaussian` and `gaussian` give homodyne records that differ by up to
2.4e-4 relative (0.014 absolute), while the compiled map still matches the target to 1.9e-15
relative. This is float64 conditioning of a physically meaningless regime; the trajectory
contract is asserted only on chains with cond(S) <= 1e8 (all others, including 20 squeezers,
agree to <= 1e-14).

## 8. R41's historical cutoff (12) is not converged for every gate; JAX and native-fock differ there by truncation

At cutoff 12 the JAX dense-generator realisation (`photographiq.autodiff`) and `native-fock`
(exact projected matrix elements) give <n> after Squeeze(0.4) that differ by 6.1e-5, and the
cubic-phase expectation changes by 1.3e-2 between cutoff 12 and 40. The historical R41 check is
an internal autodiff-vs-finite-difference consistency check at fixed cutoff and remains valid as
such; the new native cross-check is done at cutoff 40, where forward values agree to 3.8e-15 and
JAX gradients agree with native-fock central finite differences to 6.7e-9 (relative).

## 9. (Expected) causal flows are not unique: Graphix and PhotoGraphiQ pick different valid flows on star graphs

In the new R17 flow comparison, PhotoGraphiQ's causal flow and Graphix's `to_causalflow_or_none`
agree on existence for all five graphs, and on the X-correction support for the line graphs. On
star graphs they choose different leaves as f(centre); Graphix's certificate passes
PhotoGraphiQ's independent `verify_causal_flow`, so both are valid. Support equality is only
asserted where the flow is unique.

## 10. `native-fock` succeeds where `piquasso-fock` stops at the norm guard (two-mode cubic injection)

R34 at cutoff 96 and r = 1.0: `native-fock` keeps the retained norm inside the guard and gives
fidelity 0.9974 to the ideal cubic phase; `piquasso-fock` raises the norm guard. Consistent with
the documented difference (native: exact projected CZ matrix elements; Piquasso: Euler
decomposition truncated after each factor). Wherever both engines succeed, their output states
agree to fidelity >= 0.99999 (R23, R34).

## 11. (Test design) `Pattern.schedule()` returns the listed order for the R12 pattern

The historical R12 check (dependency edges point forward in the schedule) is therefore trivially
satisfied for this pattern. R12 now also executes a genuinely different topological order of the
same DAG (latest-index-first tie breaking) on fixed outcome branches on `gaussian` and
`native-gaussian`; the conditional outputs are identical (difference 0.0).

## 12. R28: native-fock and piquasso-fock syndrome records differ by up to 4.5e-4 (truncation-level SUM-gate difference)

With equal seeds, the GKP syndrome records of the two engines differ by median 3.1e-5 and at most
4.5e-4 over 48 trajectories (delta = -0.6, 0, 0.45; cutoff 80); the difference grows with |record|
(largest for records |m| ~ 4-9, in the low-density tails of the outcome distribution); the
corrected <q> differs by at most 1.7e-3. The tolerance first declared in the migrated script
(1e-6) failed; it had no physical basis. The engines apply the two-mode SUM gate differently at a
finite cutoff (exact projected matrix elements vs Piquasso's Euler decomposition truncated after
each factor), consistent with the 1 - F ~ 8.5e-6 engine difference of the two-mode CZ in R23/R34.
The tolerance is now 1e-3 (< 0.3% of the single-shot syndrome spread ~0.37), justified in the
script; the measured margin is only ~2x, so a larger cutoff or a different seed set could exceed
it -- treat it as a truncation-level agreement statement, not an exact one. The native ensemble
statistics reproduce the historical Piquasso values (circular mean errors 0.0486-0.0534 vs
0.0487-0.0534; mean |<q>| 1.737-1.894 vs 1.737-1.894). A "corrected mean outside the cell" failure
rate tried during migration was removed: |<q>| ~ 1.9 at every delta (also in the historical run) is
not a sound logical-failure indicator; only the syndrome error rate (9/144) is reported.

# Final integration (2026-11-13): findings while running R1-R60 together

## 13. Eigen's own OpenMP threads slowed small native mixed-Fock gates (fixed)

`multiply_fibres` (`cpp/src/fock/kernels.cpp`) parallelises over fibres only above a work
threshold; below it, the region runs serially and Eigen then started its own OpenMP team inside
every fibre product. On an idle 12-thread laptop a cutoff-12 two-mode beam splitter on a density
matrix took 5.5 ms with the default threads vs 0.64 ms single-threaded (cutoff 16: 8.2 vs 3.4 ms).
The native build now defines `EIGEN_DONT_PARALLELIZE` (`cpp/CMakeLists.txt`): 0.72 ms and 3.7 ms.
The native Gaussian benchmark is unchanged within noise. Reported by the R43-R50 work (R47, R49
pinned the runtime to one thread while it was open).

## 14. R59 `--full`: Piquasso's retained-norm guard fires at the moderate GKP resource, cutoff 32

With five seeds, seed 3 at the moderate resource lost 1.05e-3 of norm in `piquasso-fock`'s
truncated CZ at cutoff 32 and PhotoGraphiQ's guard (norm tolerance 1e-3) raised. The default
configuration (two seeds) never reached this case. Outside the validated wide-resource domain the
script now records the trip as data (`engine_cases_guarded`), as its cutoff-convergence section
already did; a trip inside the validated domain still fails the experiment. Consistent with #10 and
R59's finding that Piquasso needs a cutoff of about 48 at the default resource.
