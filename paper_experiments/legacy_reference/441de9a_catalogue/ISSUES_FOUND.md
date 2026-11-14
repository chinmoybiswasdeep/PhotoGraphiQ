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
