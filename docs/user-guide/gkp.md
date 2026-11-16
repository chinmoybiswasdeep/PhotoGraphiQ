# Finite-energy GKP infrastructure

At hbar=2, this square lattice uses logical shift spacing `sqrt(2*pi)` and
stabilizer translations twice that spacing. `GKPResource` builds a normalized
Gaussian comb with separate peak width and envelope controls, then projects it
into Fock space at execution cutoff.

```python
import photographiq as pg
resource = pg.GKPResource(logical=0, peak_width=0.5, envelope=0.5)
input_state, captured_weight = resource.project(48)
print(captured_weight)
pattern = pg.Pattern(inputs=(0,))
result = pg.simulate(pattern, inputs={0: input_state},
                     backend="native-fock", cutoff=48)
print(pg.gkp.stabilizers(result.state))
```

`pg.gkp.superposition(alpha,beta,cutoff=...)` combines the two finite basis states
and normalizes with their actual overlap. They are not assumed exactly orthogonal.
`decode_shift(value)` returns nearest-cell residual and cell parity, with residual
in `[-spacing/2,spacing/2)`. `correction_pattern(resource=logical_plus)` constructs
one finite-ancilla SUM syndrome extraction and modular correction. Its callable
correction is runtime-only and cannot be serialized as JSON or differentiated.

Increase `peaks` to test lattice-sum truncation, `grid_points` for quadrature
resolution, and cutoff for Fock projection. Captured weight measures only the
last of these. Stabilizer expectations need not equal one for finite resources.
This is experimental preparation, diagnostics and elementary correction
infrastructure, not a demonstrated fault-tolerant architecture or threshold study.

Validation includes analytic Gaussian-comb overlap/stabilizer integrals and both
q and p syndrome kernels. For the tested ancilla with peak_width=envelope=0.7,
a shift beyond one cell required cutoff 72 to meet a 2e-6 mean/density accuracy
target. Cutoffs 32 and 48 remain recorded as underresolved examples. Sharper or
higher-energy resources require their own convergence studies.

## Physical logical Y and XY measurements

The logical X and Z readouts are p and q homodyne followed by nearest-cell parity
decoding (`code.logical_measurement("X")`, `code.logical_measurement("Z")`). Without a
`protocol` argument `logical_measurement` keeps its original behaviour exactly: `"XY"` is
accepted only at alpha = 0 or pi (mod 2 pi, absolute tolerance 1e-14) and returns a signed
X readout; `"Y"` and every other angle raise `NotImplementedError`.

Passing an explicit `protocol` returns `pg.GKPLogicalXYMeasurement`, a physical
finite-energy instrument for M_L(a) = cos(a) X_L + sin(a) Y_L. Its +1 eigenstate is
(|0_L> + e^{ia}|1_L>)/sqrt(2), the convention of `IdealLogicalXYMeasurement` and the
logical MuTA. Measuring M_L(a) equals applying D(-a) = diag(1, e^{-ia}) to the data mode
and then reading out X_L, so the instrument applies D(-a) physically and finishes with the
existing p-homodyne X readout.

| `protocol` | Angles | How D(-a) is applied |
| --- | --- | --- |
| `"clifford"` | k pi/2 only (absolute tolerance 1e-12); other angles raise `NotImplementedError` | S or S^dagger as the Gaussian shear exp(+-i q^2/4) (`QuadraticPhase(+-1)`); Z as an exact outcome relabelling (Pauli frame). Deterministic, no ancilla. |
| `"injection"` | any finite angle | Repeat-until-success magic-state injection, also for Clifford angles |
| `"auto"` | any finite angle | Clifford route for k pi/2, injection otherwise |

One injection round prepares the finite-energy encoded state
(|0~> + e^{i phi}|1~>)/norm in a fresh mode, applies the logical CNOT (Gaussian SUM,
data as control), measures the ancilla q quadrature and decodes its parity m. Ideally m = 0
applies D(phi) and m = 1 applies D(-phi); the next round then targets 2 phi. A Clifford
remainder after injection rounds is finished deterministically: pi is a frame flip, and
+-pi/2 is injected with the Y-eigenstate ancilla, whose failure leaves D(pi) = Z, again an
exact frame flip. After `max_rounds` non-Clifford rounds without success the record is a
heralded failure (`heralded_success=False`): the wrong angle was measured. For a generic
angle the ideal-code success probability is 1 - 2^-K for a budget of K rounds; the default
is K = 4. Feed-forward uses decoded ancilla bits only, never analog residuals.

```python
import numpy as np
import photographiq as pg

code = pg.GKPCode(cutoff=56, peak_width=0.6, envelope=0.6, peaks=6, grid_points=2049)
y = code.logical_measurement("Y", protocol="clifford")               # alpha = pi/2
xy = code.logical_measurement("XY", alpha=np.pi / 3, protocol="auto")
same = pg.GKPLogicalXYMeasurement(angle=np.pi / 3, code=code, protocol="auto", max_rounds=4)
also = pg.LogicalMeasurementSynthesis("auto").lower(code, np.pi / 3)

pattern = pg.Pattern(inputs=("d",))
pattern.append(pg.Measure("d", xy, "m"))
pattern.append(pg.Output(()))
record = pg.simulate(pattern, seed=1, backend="native-fock", cutoff=56,
                     inputs={"d": code.encode(np.cos(0.4), np.exp(0.9j) * np.sin(0.4))}
                     ).records["m"]
print(record.bit, record.heralded_success, record.applied_operations)
```

`LogicalMeasurementSynthesis()` without a protocol still raises `NotImplementedError`.
The angle may be a classical expression (for example a `CallableExpression` over earlier
records); `simulate()` resolves it before the instrument runs, which is how adaptive
measurement patterns change the next physical instruction.

### Record

The instrument returns `pg.LogicalXYMeasurementResult`, an extension of
`LogicalDecodeResult`:

| Field | Meaning |
| --- | --- |
| `bit` | Logical M_L(a) outcome (0 for +1) after the instrument's own frame flips |
| `raw_outcome`, `cell`, `residual` | Final data-mode p homodyne and its nearest-cell decoding |
| `frame_correction` | Frame flip applied to the decoded bit (Z remainder or Y-state failure) |
| `angle`, `protocol` | Resolved angle; `"injection"` if any ancilla round ran, else `"clifford"` |
| `rounds` | One dict per injection round: `target_phase`, `raw_outcome` (ancilla q), `bit`, `residual`, `cell`, `density` |
| `heralded_success` | `False` only if the round budget was exhausted |
| `applied_operations` | Readable log, for example `("inject D(-1.0471975512); ancilla bit 0",)` or `("S_dagger",)` |
| `joint_density` | Product of all homodyne point densities: the likelihood density of the full analog record, not a probability |
| `readout_probabilities` | Exact (P(bit 0), P(bit 1)) of the final decoded X readout given the data state just before it; averaging over trajectories is a Rao-Blackwell estimate. `None` with noisy detection |
| `resource`, `cutoff`, `retained_norm_min`, `ancilla_modes` | Magic-ancilla code parameters, engine cutoff, smallest retained Fock norm during the instrument, ancilla mode labels |

### Postselection

A physical XY record is postselected with a dictionary, never with a logical bit:

```python
fixed = pg.simulate(pattern, seed=0, backend="native-fock", cutoff=56,
                    inputs={"d": code.plus()},
                    measurement_outcomes={"m": {"ancilla": [2.4, 0.1], "data": 0.3}})
```

`"ancilla"` lists one q value per injection round and `"data"` is the final p value. The
list must contain exactly the rounds the protocol takes for those values: too few or too
many raise `ValueError`. A Clifford-route measurement takes `{"data": p}` (empty ancilla
list). A scalar is rejected.

### Noise options

`ancilla_transmissivity` (pure loss on each magic ancilla before the CNOT),
`detector_efficiency` and `homodyne_noise` (all homodynes of the instrument) default to the
noiseless values 1, 1 and 0. Any other value adds the `mixed_fock` capability requirement,
so the instrument must run on `native-mixed-fock` or `piquasso-mixed-fock`; a pure engine
rejects it before execution. With noisy detection the engine's own sampler is used and
`readout_probabilities` is `None`.

### Limitations

- The encoded magic state is a supplied finite-Fock vector built from the same finite
  codewords; its preparation is not simulated (Baragiola et al. 2019 describe an
  all-Gaussian route from GKP Pauli eigenstates).
- A rotated homodyne is never used: a quadrature at angle a is not a measurement of
  M_L(a) for the square GKP code.
- Each Gaussian Clifford (SUM, shear) distorts the finite envelope, and each failed round
  adds the ancilla's momentum spread to the data mode, so the cutoff needed grows with the
  number of rounds. The magic-ancilla code cutoff must not exceed the engine cutoff.
- The decoded statistics include the finite-energy error of the resource; they converge to
  the ideal qubit values only as the resource narrows. This is not a fault-tolerant
  protocol and no threshold is claimed.

Design and validation: [implementation notes](../development/GKP_ARBITRARY_ANGLE_IMPLEMENTATION.md),
[final report](../development/GKP_ARBITRARY_ANGLE_FINAL_REPORT.md), experiments R60-R67 in
`paper_experiments_ml/14_gkp_xy/`, tests `tests/gkp/test_logical_xy.py`. For use inside
PhotoGraphiQML see the [XY measurement tutorial](../ml/physical/xy-measurement-tutorial.md).
