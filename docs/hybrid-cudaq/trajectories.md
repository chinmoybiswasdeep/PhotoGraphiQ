# Trajectories, shots and results

```python
import numpy as np
import photographiq as pg
import photographiq.cudaq_dynamics as cq
from photographiq.hybrid.executor import ensemble_statistics, run, run_shots

gamma, r, theta = 0.15, 0.4, 1.1
p = pg.Pattern(inputs=(0,))
p.extend([pg.Prepare(1, r), pg.Prepare(2, state=pg.CubicPhaseResource(gamma, r)), pg.Entangle(0, 1)])
p.measure(0, pg.Homodyne(theta))
p.displace(1, q=-pg.Outcome(0) / np.sin(theta))
p.extend([pg.Rotate(2, np.pi / 2), pg.Entangle(1, 2, -1.0), pg.Rotate(2, -np.pi / 2)])
p.append(pg.Measure(2, pg.Homodyne(0.0), "m"))
p.extend([pg.QuadraticPhase(1, -2 * gamma * pg.Outcome("m")), pg.Displace(1, p=-gamma * pg.Outcome("m") ** 2)])
p.append(pg.Output((1,))).validate()

inputs = {0: pg.GaussianInput((0.3, 0.1))}
res = run(p, dims=20, inputs=inputs, measurement_outcomes={0: 0.2, "m": -0.3})   # postselected
np.exp(res.log_likelihood)            # joint outcome density 0.05235 (native-fock: 0.05235)
res.expectation(cq.number(0)).real    # <n> of the output mode, 0.1314

shots = run_shots(p, 20, seed=1, dims=16, inputs=inputs)                        # sampled
ensemble_statistics(shots, cq.position(0))   # mean, standard error, "finite-shot estimate"
```

## `run` arguments

| Argument | Meaning |
|---|---|
| `dims` | per-mode local dimension: an int, or `{mode: d, "default": d}` |
| `inputs` | `{input node: GaussianInput | FockInput | FockRep}` (default vacuum) |
| `measurement_outcomes` | postselected records by result key; the others are sampled |
| `seed` | int / Generator; `run_shots` spawns independent child seeds (`SeedSequence`) |
| `engine` | `"numpy-reference"` or `"cudaq"` |
| `mixed` | force density-matrix execution (default: only when the pattern has `Loss`) |
| `resolution` | homodyne bin width (finite resolution) |
| `parameters` | values for `pg.Parameter` symbols |
| `grid` | sampling grid points |

## `HybridResult`

| Field | Content |
|---|---|
| `state` | output `FockRep`, reordered/reduced to `pattern.outputs` (or `None` if none survive) |
| `records` | raw outcomes by key |
| `densities` / `probabilities` | homodyne densities (or bin probabilities) / count probabilities |
| `log_likelihood` | Σ log of the above: the conditional statistics of the branch |
| `retained_weight` | product of the truncation probabilities (bridge, Fock inputs) |
| `tail_mass` | per-homodyne probability outside the sampling window |
| `truncation` | `max_boundary_population`, `max_conditional_boundary` (see [limitations](limitations.md)) |
| `execution_record` | per-operation log: engine, device, modes, seconds, outcome, density, boundary |
| `plan`, `runtime`, `engine` | the dry-run plan, wall time, engine name |
| `expectation(O)` | Tr(ρ O) for a `BosonicHamiltonian` on output positions 0..n−1 |

Ensemble averages over `run_shots` are finite-shot estimates with standard errors. They
are labelled as such and are never presented as exact expectations.
