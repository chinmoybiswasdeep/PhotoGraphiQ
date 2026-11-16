# Resource costs and verified optimisation

## `resource_cost(diagram)`

`resource_cost` returns a dict with these keys:

| Key | Definition |
|---|---|
| `resource_modes` | modes created by `Prepare` / `PrepareState` |
| `ancilla_modes` | created modes that are not outputs |
| `classical_sources` | number of `ClassicalSource` boxes |
| `entangling_edges` | number of `CZ` and `BeamSplitter` boxes |
| `measurement_count` | number of `Homodyne` and `PhotonCount` boxes |
| `measurement_depth` | longest chain of measurements in the causal DAG (wire order plus record dependencies) |
| `feedforward_commands` | number of `Feedforward` boxes |
| `feedforward_dependencies` | total number of (record, gain) entries over all `Feedforward` boxes |
| `maximum_feedforward_gain` | largest \|gain\| |
| `squeezing_energy` | Σ sinh² r over `Prepare.squeezed` resources (mean photons of a squeezed vacuum) |
| `estimated_optical_energy` | total mean photon number of all prepared Gaussian states, (tr V + \|mean\|²)/4 − 1/2 per mode (ħ = 2) |
| `added_noise` | tr Y / dim of the averaged channel, the mean added noise per output quadrature in vacuum units. NaN for non-Gaussian diagrams or diagrams without outputs. |
| `boxes` | number of boxes |

The units differ: counts, photons and vacuum units. `weighted_cost(cost, weights=None)`
returns Σ_k w_k cost_k. The weights are an explicit user choice, not a universal experimental
cost. The default weights are `resource_modes` 1.0, `entangling_edges` 0.5,
`measurement_depth` 0.5, `feedforward_dependencies` 0.1 and `squeezing_energy` 0.1. To avoid
fixing weights, use `pareto_front(items, keys)`. It returns the indices of the cost dicts that
no other item dominates on `keys` (all keys are minimised).

## `optimize_cv_pattern`

```python
optimize_cv_pattern(source, *, equivalence="physical", objective="resource_modes",
                    strategy="greedy", beam_width=4, max_steps=50, epsilon=0.0,
                    executable=True, weights=None) -> OptimizationResult
```

`source` is a `Pattern` or a `CVDiagram`. The search is bounded and deterministic. Rules and
matches are enumerated in a fixed order, ties are broken by fingerprint, and a fingerprint is
never visited twice. Every candidate is compared with the **original**, not with its parent,
so errors cannot drift.

| `equivalence` | Rules used | A candidate is accepted when |
|---|---|---|
| `"physical"` | the exact Gaussian rules (`commute` excluded) | (X, Y, d) of the averaged channel is unchanged |
| `"logical"` | the exact rules plus `teleport-chain-resynthesis` | (X, d) is unchanged |
| `"approximate"` | as logical | (X, d) is unchanged and max\|Y_original − Y_candidate\| ≤ `epsilon` |

!!! note "epsilon is not a channel distance"
    `epsilon` bounds the entrywise residual of the noise matrix, in vacuum units. It is not a
    diamond-norm bound or any other operational distance.

Other options:

- `objective` is any `resource_cost` key, or `"weighted"` (uses `weights`).
- `strategy="greedy"` keeps one candidate per depth. Any other value runs a beam search of
  width `beam_width`.
- With `executable=True`, a candidate must also recompile with `to_pattern`.

The result has these fields:

- `original`, `best`, `mode`, `objective`;
- `original_cost` and `best_cost`;
- `report`: the `EquivalenceReport` of best against original;
- `history`: per depth, the number of candidates, the best score and the rule applied;
- `explored`, `rejected` and `seconds`;
- `improved`: whether the objective went down.

## What each mode does on compiled patterns

[Tutorial 5](examples.md#5-optimise-a-cv-mbqc-pattern) runs a two-mode circuit
(R, R, S, CZ, R, D) compiled at r = 1:

```text
original: {'resource_modes': 29, 'measurement_depth': 16, 'entangling_edges': 30, 'added_noise': 1.0634}
                physical: improved False explored    0 rejected   0 | resource_modes 29, measurement_depth 16, entangling_edges 30, added_noise 1.063 | logical True physical True Y residual 0
          logical greedy: improved True  explored    3 rejected   0 | resource_modes 22, measurement_depth 11, entangling_edges 23, added_noise 0.9854 | logical True physical False Y residual 0.335
```

- **Physical mode finds nothing to change in a `compile_circuit` pattern.** No exact rule
  even matches there, so this is a reported negative result. Physical mode is useful on
  gate-level diagrams (fusions, identity removal) and on patterns with mergeable feed-forward.
- **Logical mode shortens teleportation chains.** The ideal map is exact. The noise Y and the
  records change, and `report.physical_equivalent` is `False`.
- **Approximate mode** with a tiny `epsilon` rejects every resynthesis whose noise change
  exceeds it.

The noise can go either way. Fewer modes often means less noise, but resynthesised shears can
amplify the noise of the remaining steps. Check `added_noise` or use `pareto_front`.

## Recorded results

Only experiments whose JSON results carry a validation status are quoted here.

| Study | Setting | Result (status `pass`) |
|---|---|---|
| R63 (`63_random_compilation.py`) | 54 random 1–3-mode circuits (depth 2/4/6, 6 seeds), compiled at r = 1, greedy, objective `resource_modes` | Physical mode found nothing to change in any of the 54. Logical mode reduced resource modes in all 54 programs: 3537 → 2819 in total; summed measurement depth 2163 → 1731. Logical residual ≤ 3.4e-14; executed residual of the recompiled patterns ≤ 3.0e-14. Added noise went down in 51 of 54. CV flow certified for all 47 optimised patterns small enough to check (≤ 120 nodes). |
| R64 (`64_scheduling_and_feedforward.py`) | 15 grid clusters (1–3 wires × 3–7 columns) from `cvflow.flow_pattern`, random shears, r = 1 | Feed-forward merge (physical mode, objective `feedforward_commands`): 390 → 120 commands in total. It is exact at the instrument level, and the executed channel residual is ≤ 1.8e-15. Dependencies went 390 → 358. 50 random commutations per cluster never changed the measurement depth. Re-deriving the corrections from the layered `find_cv_flow` certificate reduced the summed depth from 95 to 60 (1 wire: 20 → 20; 2 wires: 35 → 20; 3 wires: 40 → 20). The layered patterns were logically, physically **and** conditionally equivalent to the sequential ones (instrument residual ≤ 9.1e-15). |
| R65 (`65_resource_vs_noise.py`) | 24 random 1–2-mode circuits; every subset of resynthesisable chains; r ∈ {0.5, 1, 1.5, 2} | Every variant was logically equivalent (≤ 8.0e-14). The noise ratio optimised/original was r-independent (spread ≤ 7.6e-13). In 1 of 24 programs the fewest-mode variant was not the quietest. The control, exact ancilla elimination (CZ + q-homodyne → classical source), had an identical channel and instrument. |
| R66 (`66_optimizer_scaling.py`) | 1-mode programs of 2–32 gates and 2-mode programs of 2–12 gates, 3 seeds, greedy vs beam (width 3) | Every result was logically equivalent. Beam search found the same resource count as greedy in all 27 cases. Timings are marked provisional (one machine); the largest greedy search (576 boxes) took 0.15 s. |
| R67 (`67_end_to_end.py`) | circuit R, S, R, CZ, S, R, D, R; r = 1; 2000 trajectories per pattern | Resource modes 38 → 18, measurement depth 26 → 15, entangling edges 39 → 19, added noise 1.487 → 0.824. Each pattern matched its own channel (max \|z\| 1.16 and 2.00). Flow certified (25 → 15 layers). Optimisation time 0.076 s. [Tutorial 6](examples.md#6-recompile-and-execute) is a small version. |

Details: [final implementation report](../../development/CV_ZX_FINAL_IMPLEMENTATION_REPORT.md).

## Flow-layered scheduling and feed-forward merging

There is no rule that reorders measurements to reduce depth. `commute` changes the sequential
order but never the causal DAG, so it never changes `measurement_depth` (R64). The depth
reduction in R64 comes from rebuilding the pattern from the layered CV-flow certificate with
`flows.cv.corrected_pattern(order, correction_map, ...)`. That gives different corrections
with the same ideal map, and on those clusters the same instrument too. The result is then
read back with `from_pattern` and verified with `compare`. `feedforward-merge` is exact and
is available in physical mode.
