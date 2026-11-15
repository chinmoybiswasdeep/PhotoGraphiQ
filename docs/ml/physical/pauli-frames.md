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
