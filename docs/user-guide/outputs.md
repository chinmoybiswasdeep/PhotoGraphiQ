# Results and output states

`simulate` returns one `Result`: a conditional quantum state together with the
classical records that produced it. The state is an independent snapshot.

| Attribute | Meaning |
|---|---|
| `outcomes` | Measurement keys mapped to scalar homodyne readings, coordinate pairs, or integer photon counts |
| `records` | Outcomes plus values written by `Signal` commands |
| `state` | Surviving Gaussian or Fock state, with ordered node labels |
| `backend` | Execution backend class name |
| `seed` | Seed supplied to this trajectory |
| `measurement_statistics` | Reported conditional probability/density entries; available on Fock paths |
| `log_likelihood` | Sum of log conditional probabilities/densities; raises if required statistics are absent |
| `physical_displacements` | Count of executed displacement commands, including frame flushes |

Gaussian backends currently do not populate measurement likelihood diagnostics.
An empty dictionary is not evidence of unit probability for a measured pattern.

## Read a deterministic example

```python
import photographiq as pg
pattern = pg.Pattern(inputs=("a",))
result = pg.simulate(pattern, inputs={"a": pg.GaussianInput.coherent(0.3+0.2j)},
                     backend="gaussian", seed=0)
print(result.outcomes)
print(result.state.nodes)
print(result.state.mean)
print(result.state.covariance)
print(result.state.quadrature("a"))
```

```text
{}
('a',)
[0.6 0.4]
[[1. 0.]
 [0. 1.]]
(0.6, 1.0)
```

No measurement occurred, so there are no outcomes. The two means are q and p;
the covariance has vacuum variance in both axes. They are not state amplitudes.

## Gaussian output

`mean` has length 2m, `covariance` shape (2m,2m), and `nodes` fixes mode order.
`quadrature(node, angle=0)` returns (mean, variance), not standard deviation.
`photon_number(node)` returns $\langle n\rangle$; the state above gives 0.13.
`parity(nodes=None)` gives the parity expectation. `reduced(nodes)` traces out
other modes and preserves requested order.

`overlap(other)` returns $\mathrm{Tr}(\rho\sigma)$. It equals fidelity when one
state is pure; it is not general mixed-state Uhlmann fidelity. Labels and order
must match, including when comparing a compiled output to a target state.

## Fock output

```python
pattern = pg.Pattern(inputs=("a",))
result = pg.simulate(pattern, inputs={"a": pg.FockInput.number(1)},
                     backend="piquasso-fock", cutoff=4)
print(result.state.probabilities)
print(result.state.photon_number("a"))
print(result.state.parity())
```

```text
{(0,): 0.0, (1,): 1.0, (2,): 0.0, (3,): 0.0}
1.0
-1.0
```

`basis` specifies occupation ordering. `state_vector` is available for pure
states; a mixed reduced state has no state vector and raises. `density_matrix`
works for pure and mixed outputs but may allocate quadratically. `probabilities`
maps occupation tuples to diagonal probabilities. `quadrature` returns mean and
variance; `quadrature_moment(node, order, angle)` and `photon_moment(node, order)`
support raw moments through order four.

`fidelity(other)` uses squared Uhlmann fidelity; `trace_distance(other)` uses
half the trace norm. Different cutoffs are aligned by occupations, not by array
index. States must have matching node order. `wigner(q,p,node)` returns a
`WignerGrid` with `.plot()`, `.captured_mass` and `.negative_volume`. Window and
grid resolution must both converge.

`retained_norms` and `diagnostics` expose preparation/gate truncation checks and
boundary populations. Norm conservation alone does not certify converged high
moments or gate fidelity. See [cutoff convergence](convergence.md).

## Conditional trajectories, ensembles and postselection

A measurement outcome is a classical reading. A record includes derived signals.
The conditional state is normalized *given* those records. A trajectory is the
complete sequence of readings and state updates. An ensemble averages trajectories.

`run_shots(pattern, shots, seed=...)` uses independent child seeds.
`shots.values(key)` extracts recorded values. `shots.ensemble_state()` includes
both average conditional covariance and covariance of conditional means. For
nonlinear adaptation this is only a Gaussian container of ensemble moments;
average per-trajectory nonlinear observables instead of applying Gaussian formulas
to a generally non-Gaussian mixture.

For photon-count postselection, `exp(result.log_likelihood)` is a branch probability
if every recorded measurement is discrete. An exact real homodyne outcome has
zero point probability: its reported value is a **density**. A density can exceed
one. A detector-bin probability requires integration over that bin. Mixing discrete
and continuous measurements gives a joint probability-density quantity, not a
heralding success rate. Differentiating a normalized selected branch also differs
from differentiating an unconditional expected objective.
