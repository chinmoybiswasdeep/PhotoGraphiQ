# Feed-forward

Any gate parameter may be an outcome expression (`pg.Outcome(key)`, `pg.Parameter(name)`,
arithmetic, `sin`, `atan2`, powers, ...). This includes displacements, quadratic phases,
rotations, squeezing, CZ weights and homodyne angles. The executor evaluates the
expression with `photographiq.expressions.resolve` on the **actual** records of the
trajectory, whether those were sampled or postselected. It then applies the resulting
physical Hamiltonian, for example the displacement generator with the evaluated (q, p).

## Measurement records

Records are stored under the command's result key: the node by default, or the `key` of
`pg.Measure(node, measurement, key)`. A homodyne record is the calibrated outcome x. A
photon-count record is the integer n.

## Causality

`Pattern.validate()` rejects a command that depends on a future outcome, and both
`plan()` and `run()` call it:

```python
p = pg.Pattern(inputs=(0,)).extend([pg.Prepare(1, 0.5), pg.Entangle(0, 1), pg.Displace(1, q=pg.Outcome(0))])
p.measure(0, pg.Homodyne(0.0))
run(p.append(pg.Output((1,))), dims=8)   # ValueError: Command 2 depends on a future outcome
```

## Validated cases

| Case | Where | Result |
|---|---|---|
| one-step teleportation, q → −m / sin θ, fixed outcome | HYB-04 | moments vs Gaussian backend at dims 28: mean 2.0e-7, cov 3.4e-6 |
| sampled outcomes, 200 shots, each vs the Gaussian backend postselected on the same outcome | HYB-04 | max per-shot error 1.5e-4 at dims 32 (see the d = 20 caveat in [limitations](limitations.md)) |
| two dependent measurements, quadratic-phase and nonlinear (−γ m²) feed-forward | HYB-05 | three-mode cubic CV-MBQC vs native-fock (4 outcome cases) |
| gains (`Parameter("g") * Outcome(0)`) | `tests/hybrid::test_feedforward_gain_is_applied_to_the_record` | shift = g·m to 1e-3 at dims 16 |
| replay: a sampled record reused as a fixed outcome reproduces the trajectory | `tests/hybrid::test_sampling_is_reproducible_and_matches_density` | identical state |
