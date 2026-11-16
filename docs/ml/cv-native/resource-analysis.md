# Resource analysis

Resources are reported separately from the learning objective. The only exception is
the optional penalty term ([training](training.md#resource-penalties)).

## `model.resource_report(X=None)`

Available after `fit` (or `set_params`). Resources are evaluated at the trained
parameters, at the first sample of `X` when data enters gates.

| Field | Meaning | Present for |
| --- | --- | --- |
| `model`, `program`, `optical_modes`, `execution` | Learner and program class, mode count, `ideal` / `physical` | all |
| `trainable_quantum_parameters`, `classical_head_parameters` | Parameter counts | all |
| `cluster` | `wires`, `columns`, `rungs` | `CVCluster` |
| `gate_squeezing_max_r`, `gate_squeezing_max_db` | Largest squeeze-gate r (after bounding) | `GaussianProgram` |
| `resource_modes`, `entangling_edges`, `weighted_edges`, `measurements`, `measurement_depth`, `feedforward_corrections`, `peak_live_modes`, `trajectory_covariance_bytes`, `non_gaussian_operations` | `pattern_resources` of the executable pattern | Gaussian programs and `CVCluster` |
| `resource_squeezing_r`, `resource_squeezing_db` | Resource squeezing of that pattern ($8.686\,r$ dB); `None` for an ideal-mode model | same |
| `structure_compiled_at_r` | Squeezing at which the pattern was built for the counts (counts do not depend on r; an ideal `GaussianProgram` model uses r = 1) | same |
| `added_noise_per_quadrature` | $\operatorname{tr}Y/2n$, vacuum units | `execution == "physical"` |
| `mean_input_photons` | Mean photon number of the encoded inputs over `X` | `X` given |
| `seconds_per_prediction` | Wall time of `moments` on up to 16 samples, per sample | `X` given |
| `moment_memory_bytes` | $3(2n)^2 \cdot 8$ (X, V, Y) | all |
| `fock_cutoff`, `kerr_gates`, `cubic_gates`, `fock_dimension` | Fock execution size | `NonGaussianProgram` |

The executable pattern is `program.pattern(params, squeezing=r)` for a `CVCluster` and
`program.circuit(params, x).compile(squeezing=r)` for a `GaussianProgram`. For an ideal
`GaussianProgram` learner without `squeezing`, the report compiles at r = 1.0 and reports
that value.

## `pattern_resources(pattern)`

Structural counts of any `photographiq.Pattern`:

- Resource modes (`Prepare` commands) and entangling edges.
- Weighted edges (weight different from 1).
- Homodyne measurements.
- Measurement depth: the longest dependency chain counted in measurements.
- Feedforward corrections: commands whose arguments depend on outcomes.
- Peak live modes, and the trajectory covariance memory $(2\cdot\text{peak})^2 \cdot 8$ bytes.
- Kerr and cubic commands.

## What R68 and R74 show

R68 (`R68_cv_architecture.json`): trained two-mode models on the linear-regression split,
both physical at r = 1 (8.686 dB).

| | Compiled layered program | 2 x 4 CVCluster |
| --- | --- | --- |
| Trainable quantum / head parameters | 11 / 3 | 6 / 3 |
| Resource modes = measurements | 130 | 6 |
| Entangling edges (weighted) | 136 (6) | 8 (0) |
| Measurement depth | 73 | 5 |
| Feedforward corrections | 130 | 17 |
| Peak live modes | 3 | 8 |
| Added noise per quadrature | 5.03 | 0.318 |
| Logical-map error of the compiled channel | 4.6e-16 | (n/a) |

R74 (`R74_resource_scaling.json`) structural checks pass. Cluster measurements equal
resource modes, which equal $w(c-1)$, for 1 to 4 wires and 3 to 8 columns. Compiled
resource modes equal measurements, which equal teleportation steps
([compilation](compilation.md)). Measurement depth never exceeds measurements. Timings
are marked PROVISIONAL (single machine, not idle). At the largest sizes the exact channel
took 0.027 s (compiled, 4 modes x 4 layers, 1424 steps) and 0.0049 s (4 x 8 cluster),
against 1.03 s and 0.053 s for one Gaussian-backend trajectory.

Accuracy-resource trade-off (R74; shot-trained 2 x c clusters, S = 100, expected test
MSE from exact moments). Each value comes from one training run, so it is descriptive.

| Columns (measurements) | r = 0.5 | r = 1.0 | r = 1.5 |
| --- | --- | --- | --- |
| 3 (4) | 0.00938 | 0.00878 | 0.00854 |
| 4 (6) | 0.01059 | 0.00890 | 0.00788 |
| 5 (8) | 0.01259 | 0.00979 | 0.00818 |
| 6 (10) | 0.00990 | 0.00885 | 0.00794 |

More squeezing lowered the error for every size. A larger cluster was not better at
r = 0.5.

## Limitations

- Counts describe PhotoGraphiQ's patterns, not an optimised hardware layout. Temporal-mode
  multiplexing and loss are not modelled.
- Timings are single-machine observations.
- The squeezing penalty counts squeeze gates only. Resource-mode squeezing enters through
  the noise penalty and the report.
