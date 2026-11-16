# Logical Pauli-frame propagation

The bridge uses PhotoGraphiQ `LogicalPauliFrame(x_bit,z_bit)` as classical metadata.
For flow successor f(v), interpreted outcome s(v) adds X to f(v) and Z to the
other neighbors of f(v), using MuTA's existing correction map. Contributions compose
by XOR. Frames come from interpreted earlier outcomes, not raw uncorrected bits.

At an X measurement the accumulated Z component flips the decoder bit. The X
component commutes with X readout. The signed-X pi measurement already flips its
raw decoder label through the upstream measurement object; downstream flow does
not apply that sign a second time. Final X or Z output probabilities are permuted
by the appropriate anticommuting frame component.

## Frames at XY measurements

With `measurement_family="XY"` a node measures M(a) = cos(a) X + sin(a) Y. The two frame
rules are

- X M(a) X = M(-a): an X byproduct reflects the measurement angle;
- Z M(a) Z = -M(a): a Z byproduct exchanges the two outcomes.

The lowering therefore makes the measured angle frame-adaptive: a node with frame X^x Z^z
measures the angle (-1)^x a, resolved at run time from earlier interpreted bits (a
`CallableExpression`; see [lowering](lowering.md#xy-family-xy_protocol)). The Z component is
handled exactly as for signed X: it is the classical bit flip applied by the node's
`Signal`, so the physical instruction never contains it. For the signed-X angles 0 and pi
the reflection is trivial (-pi = pi mod 2 pi), so both families agree there. The angle is
recorded in each `LogicalXYMeasurementResult.angle`, so a trajectory can be audited.

The instrument also applies frame flips of its own, before the `Signal` sees the bit, and
reports them in `frame_correction` and `applied_operations`:

- a remaining angle of pi (Z before the final X readout) is an exact outcome relabelling;
- after injection rounds, a remaining +-pi/2 is injected with the Y-eigenstate ancilla; its
  failure leaves exactly D(pi) = Z, which again becomes an outcome flip.

These are classical relabellings of the decoded bit; no displacement is applied to the
oscillator to emulate a logical Pauli. Experiment R64 (`paper_experiments_ml/14_gkp_xy/`)
checks frame-tracked branches against `contract_branch` and the recorded angle (-1)^x a.

## Graph convention

The correction map uses the fully entangled open-graph convention. The just-in-time
physical schedule delays commuting preparations/CZ gates but does not redefine
that convention. Applying an additional CZ propagation to these already graph-based
frame dependencies would double-count corrections. No physical phase-space
displacement is added to emulate a logical Pauli gate.

The independent branch test contracts the two-wire graph in the qubit Hilbert
space for every one of its 256 outcome branches with mixed 0/pi angles. It tracks
the same virtual frames without applying physical corrections, then verifies the
corrected final logical state against the exact MuTA target. This establishes the
discrete flow convention. Finite GKP physical runs separately measure errors of the
actual resource; the logical proof does not erase those errors.

See [lowering](lowering.md) for declared signal dependencies. Raw and interpreted
records both remain available for auditing a trajectory.
