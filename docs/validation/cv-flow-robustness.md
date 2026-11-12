# CV-flow robustness

This page covers `photographiq.flows.find_cv_flow` and `flows.analyze` for
`Model.CV_GAUSSIAN`. It explains how the analysis keeps three outcomes apart:

- a flow that is mathematically impossible
- a decision that float64 cannot make
- a certificate, together with the regime that the certificate covers

Tests: `tests/flow/test_flows_cv_robustness.py` (25 tests) and the earlier
`tests/flow/test_flows_cv.py`.

## What can go wrong

The CV-flow search (Booth and Markham, Quantum 7, 1146 (2023)) works backwards
in rounds. In each round it decides, for every unsolved node u, whether
`A[W, C] c = e_u` has a solution. Here A is the weighted adjacency, W is the set
of unsolved nodes and C is the set of solved non-input nodes. Each decision is a
rank decision over the reals. The determinism check then simulates the corrected
pattern at squeezing r, r + `shift` and r + 2 `shift`.

The investigation found the following failure modes.

| Source | Symptom before this change |
|---|---|
| Absolute residual test (`residual <= 1e-10`) in the search | Near-singular weights such as `det = 1e-10` were reported as "no flow" (`not_certified`, `complete=True`), although the float64 data admit a flow with gains of about 1e10. |
| Weights from 1e-4 to 1e4 | A normwise test on unscaled matrices loses about 8 digits. |
| Rank decisions at the rounding level (for example, `0.1 + 0.2` against `0.3`) | Float64 cannot tell a singular value of about 1e-17 from zero. The data are within rounding of a configuration that has no flow. |
| Finite squeezing, r + 2 `shift` > about 9.5 | `GaussianState.validate` uses an absolute bound of -1e-7, and it fails on the raised covariances, so the check cannot run. |
| Small weights (about 0.02) | The outcome-dependent gain is still pre-asymptotic at the levels that are checked: it decays with kappa of about 0.1 instead of 1, which gives a false negative. |
| N_plus with a dynamic range above 1e9 | Anti-squeezed output directions below `tolerance * max` are counted as finite, so the singular-direction count is too low (a false negative). |
| Outcome-dependent displacement | Corrections with gains far above 1/sqrt(eps) make the float64 residual and simulation checks meaningless. The largest gain is now reported in `diagnostics['correction_gain_max']`, with a warning when it exceeds 1/sqrt(eps). |

Measurement ordering was not a failure mode. Solvability is monotone in the
solved set, so the layered search is complete for the given data. The exact
search confirms this: on random graphs it finds the same layers as the float
search (`test_random_graphs_exact_and_float_agree`). The existing brute-force
test over all orders still passes.

## Method

### Rank decisions with margins

Each round works on the correction matrix `M = A[W, C]`:

1. Scale the rows and columns of M by powers of two (exact in float64). This
   does not change whether `e_u` lies in `range(M)`.
2. Normalize M to `||M||_2 = 1` and compute its SVD.
3. Compare two kinds of decision quantity with the rounding threshold
   `tau = max(dims) * eps`:
   - the singular values `sigma_i`
   - for each row u, `s_u = dist(e_u, range_r(M)) * sigma_r`, which is the
     singular value that appending `e_u` creates (up to a factor between
     1/sqrt(2) and 1)

   Both quantities carry a rounding error of about tau.
4. Classify each quantity:
   - `<= tau`: zero
   - `>= sqrt(eps)`: nonzero
   - in between: uncertain

The upper edge sqrt(eps) has two justifications. A singular value below it
gives correction coefficients above 1/sqrt(eps). Their float64 relative error,
cond × eps, is then larger than sqrt(eps), which is about 1.5e-8 and so larger
than the 1e-9 tolerance of the verifier. Such weights are also within a relative
distance of sqrt(eps) of data that give the opposite answer.

A row of M that is exactly zero is decided exactly, because the node has no
edge to any correction column.

`diagnostics['rank_decisions']` reports the following:

| Key | Content |
|---|---|
| `rule` | The decision rule. |
| `uncertain_upper` | The upper edge of the uncertain band, sqrt(eps). |
| `uncertain_float64_decisions` | The number of float64 decisions in the uncertain band. |
| `margin_decades` | `log10(min nonzero quantity / tau)`, the smallest margin of any nonzero decision. |
| `decided_by` | `float64`, `exact` or `exact (automatic)`. |
| `rounds` | For each round: the threshold, the smallest nonzero quantity, the largest zero quantity and the number of uncertain quantities. |

### Exact re-check

`flows/_exact.py` repeats the search in exact rational arithmetic with
`fractions.Fraction`. Every float64 value is a dyadic rational, so the
conversion is exact and the results are proofs for the given weights. The
re-check runs on the same equilibrated systems, so it reproduces the same
minimum-norm vectors exactly. When the matrix is rank-deficient for u, the
re-check returns a rational witness y with `y^T A[W, C] = 0` and `y_u != 0`. The
witnesses are in `diagnostics['witnesses']`, and the tests check them
independently.

Exact rational arithmetic is stronger than any fixed-precision mpmath re-check,
and it needs no optional dependency. Its cost grows with the bit size of the
entries.

The `exact` option controls when the re-check runs:

| `exact` | Behavior |
|---|---|
| `None` (default) | On graphs of at most `EXACT_NODE_LIMIT` (40) nodes, the search is redone exactly when a float64 decision falls in the uncertain band, or when the float64 search finds no flow. This is cheap: about 1 ms for the small cases and 1.5 s for a 31-node wire. |
| `True` | All decisions are exact, at any size. In addition, `exact_ideal_limit` runs (see the next section). |
| `False` | Float64 only. Uncertain decisions give `unknown`. |

### Exact ideal-limit check

`exact_ideal_limit(og, layers, exact_vectors)` runs only with `exact=True`. It
propagates linear Wigner variables exactly:

- the inputs
- for each resource node, an anti-squeezed q (variance e^{2r}) and a squeezed p
  (variance e^{-2r})

It then applies the CZ gates, the records `m_u = k_u q_u + p_u` and the
corrections. Let L be the output rows and R the record rows, and let the
subscript Q denote their anti-squeezed columns. The check then verifies the
following conditions exactly:

1. The flow condition `(A c_u)_v = delta_uv` holds on u's layer and on earlier
   layers.
2. `rank L_Q = |O| - |I|`. This is the rank of the verifier's N_plus.
3. `rank R_Q` equals the number of records. Every record combination then
   carries anti-squeezed noise, so the projected gain and `A - S` are
   O(e^{-2r}), and the decay exponent is exactly 1.
4. `S Omega S^T = Omega` when `|O| = |I|`.

In exact arithmetic, these conditions imply every criterion of
`verify_cv_determinism`. They do not depend on r, so they settle the ideal
limit at any squeezing and are free of the pre-asymptotic decay problem. The
conditions are sufficient but not necessary: a failure means "not shown". The
check rejects every scaled correction vector, including a scale factor of
1 + 2^-60 (see `test_exact_check_matches_closed_form_and_rejects_wrong_corrections`).
It also reproduces the closed-form wire map.

The exact check overrules a float64 check that ran only in two cases:

- The float64 check failed only `decay`.
- The float64 check counted *fewer* singular directions than expected, which is
  the dynamic-range artifact described above.

Missing or wrong corrections give *more* singular directions, and that failure
is never overruled. Any other disagreement gives `unknown`.

### Exposing `shift`

`find_cv_flow(..., shift=3.0)` and `analyze(og, shift=...)` forward `shift` to
`verify_cv_determinism`. The default is unchanged, so existing results are
unchanged.

The trade-off works as follows. The largest level has squeezing r + 2 `shift`,
and above about 9.5 float64 validation fails. A smaller shift therefore reaches
larger r: `shift=1.5` certifies the unit wire from r = 3.5 to 4.5 (tested) and
up to r = 6 (observed). However, the decay exponent is then measured over a
shorter range, so pre-asymptotic cases fail more often. For those cases,
`exact=True` decides the ideal limit independently of r.

## Classification

Every CV result has `diagnostics['decision']`. Every certified result has
`certificate['regimes']`, a mapping from each covered regime to the basis of
the claim. The existing statuses are unchanged.

| Case | `status` | `decision` | Meaning |
|---|---|---|---|
| Ideal deterministic unitary or isometry | `certified` | `certified` | `regimes['ideal_limit']`: "deterministic unitary/isometry: float64 decay check (shift=…)", "exact rational check", both, or "Booth-Markham theorem on the float64 flow certificate (determinism not checked)" when `verify=False`. |
| Finite-squeezing deterministic Gaussian channel | `certified` | `certified` | `regimes['finite_squeezing']`: the outcome-averaged channel (S, N) at the given squeezing, where S is the ideal map. Present only when the float64 check ran at that squeezing. The residual outcome dependence `finite_squeezing_gain` is stated. |
| Postselected only (no flow) | `not_certified` | `impossible` | `diagnostics['proof']` gives the basis: a combinatorial obstruction, an exact rank test with rational witnesses, or a float64 rank test. The float64 test is a backward-error statement: weights perturbed by at most `max(dims) * eps` admit no flow. |
| Unable to certify | `unknown` | `numerically_uncertain` | One of the following: an uncertain rank decision that was not re-checked exactly; a float64 re-certification that failed; or a determinism check that failed or could not run. `reason` names the cause and suggests `exact=True` or `shift`. |
| Unsupported | `unsupported` | (none) | The graph is not `Model.CV_GAUSSIAN`. |

A certificate never rests on numerical luck, for these reasons:

- An uncertain rank decision is never used without the exact re-check.
- The float64 determinism check certifies only when all its criteria pass.
- The exact check is a proof for the stated correction map.
- `finite_squeezing` is claimed only from a float64 check that ran.

## Difficult graphs

The tests assert each classification below.

| Graph | Default | `exact=False` | `exact=True` |
|---|---|---|---|
| Bridge `[[1, 1], [1, 1]]` (singular), `|O| < |I|`, scaled singular bridge | impossible (exact witnesses) | impossible (float64 rank test, margin > 10 decades) | impossible (exact witnesses) |
| Isolated measured node; only inputs solved | impossible (combinatorial) | same | impossible |
| Bridge with `det = 1e-10` or `1e-13` | unknown: the exact re-check finds the flow, but float64 re-certification fails | unknown: an uncertain rank decision | certified, `ideal_limit` only (exact); gain about 1/det and a warning |
| Bridge with `det = eps` (rounding level) | unknown | impossible (float64 backward-error statement) | certified, `ideal_limit` only; gain about 4.5e15 and a warning |
| Wires with w = 1e-4, w = 1e4, alternating 1e-4/1e4; hexagon with mixed weights | unknown: the float64 check cannot run (validation or zero density) | unknown | certified, `ideal_limit` only |
| Wire of 30 nodes; 3×4 and 4×6 grids | certified, both regimes, margin > 10 decades | same | same; the exact and float64 S agree to 1e-9 |
| Unit wire, r = 3.5 to 4.5 | unknown (could not run) | unknown | `shift=1.5`: certified, both regimes, kappa = 1 ± 0.05 |
| Unit wire, r = 8 | unknown | unknown | certified, `ideal_limit` only |
| Slow-decay adversarial graph (weights about 0.02) | unknown (`failed_checks == ('decay',)`) | unknown | certified, both regimes |
| Dynamic-range adversarial graph | unknown (`('singular_directions',)`, count too low) | unknown | certified, both regimes |

### Adversarial set (600 graphs)

The generator draws graphs as follows:

- G(n, p) graphs with n from 3 to 7 and p from 0.3 to 0.8
- signed weights with |w| log-uniform in [0.02, 6.3]
- shears in [-5, 5]
- r in [0.3, 3.4] per node
- random inputs and outputs (seed 2026)

| Default result | Count | With `exact=True` |
|---|---|---|
| certified | 150 | certified |
| impossible, combinatorial obstruction | 178 | impossible |
| impossible, exact rank test (automatic) | 230 | impossible |
| unknown, float64 check could not run (validation at r + 6) | 27 | 27 certified, `ideal_limit` only |
| unknown, `decay` only (pre-asymptotic) | 14 | 14 certified, both regimes |
| unknown, singular-direction count too low (dynamic range) | 1 | certified, both regimes |

With `exact=True`, all 600 graphs are decided: none is left `unknown`. None of
the 408 impossible graphs was uncertain in float64.

## Qubit flows against Graphix

The existing tests already cover this; no new tests were needed.

- `tests/flow/test_flows_qubit.py::test_existence_agrees_with_graphix` runs 150
  random graphs for each notion against Graphix 0.4:

  | Notion | Positive | Negative |
  |---|---|---|
  | Causal flow | 31 | 119 |
  | gflow | 27 | 123 |
  | Pauli flow | 42 | 108 |

  For each positive case, the test also checks Graphix's certificate with the
  independent verifier.
- `tests/graphix_reference/` and `tests/calculus/test_calculus_graphix.py` cover
  correction support, standardization and signal shifting.

## Remaining limits

- The float64 rank test applies normwise thresholds after power-of-two
  equilibration. Badly scaled systems that equilibration does not fix are still
  reported as uncertain on graphs with more than 40 nodes, unless
  `exact=True`.
- A float64 "impossible" on a graph with more than 40 nodes (or with
  `exact=False`) is a backward-error statement, not an exact proof.
- The exact path costs about O(n^4) rational operations on growing integers.
  This takes about 1.5 s for a 31-node wire and is not intended for hundreds of
  nodes.
- The exact ideal-limit check is opt-in (`exact=True`). The default float64
  check still returns `unknown` for r + 2 `shift` above about 9.5, for very
  large or very small weights, and for pre-asymptotic decay. This keeps the
  behavior of existing results unchanged.
- The exact certificate covers the exact rational correction map. The reported
  float `correction_map` is its rounding. When `correction_gain_max` is above
  1/sqrt(eps) (a warning is shown), the physical protocol amplifies records so
  much that only the ideal-limit statement is meaningful.
- `finite_squeezing` is a float64 statement at the given r. No exact
  finite-squeezing channel check exists.
- Unit-efficiency homodyne detection and lossless resources are assumed.
  Losses break the two-term noise model and the exact conditions.
- `GaussianState.validate` was not changed. A relative bound would move the
  threshold only slightly, because the squeezed information itself is lost
  near r + 2 `shift` of about 9.
