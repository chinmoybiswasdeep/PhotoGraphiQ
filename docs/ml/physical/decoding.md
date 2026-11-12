# Decoding and physical features

The intermediate physical X instrument measures p and applies PhotoGraphiQ's
nearest-cell decoder. Final Z readout concerns q; X readout concerns p. The lattice
spacing and cell convention are owned by the upstream decoder, never copied into
a separate downstream threshold implementation.

Each `LogicalDecodeResult` retains raw outcome, cell, residual, decoded bit,
decoder identity, frame correction and any supported posterior fields. Nearest
decoding does not estimate confidence, posterior probabilities or leakage, so these
remain `None`. They are not replaced by certainty or zero leakage.

## Calibrated soft features

`GKPBridge.resource_readout("Z", decoder="soft")` uses the public
`SoftDecisionDecoder`: Bayesian discrimination of a declared equal-prior finite
zero/one preparation ensemble. X uses the finite plus/minus ensemble. Its posterior
is a preparation-label probability under that model. It is not a universal posterior
for an arbitrary adaptive graph state. Physical MuTA therefore rejects a soft
feed-forward configuration before allocation. This preserves the meaning of the
ML features instead of attaching unjustified probabilities to arbitrary states.

## Joint outputs

PhotoGraphiQ `multimode_readout` integrates tensor products of physical modular
effects against the full multimode density matrix and then applies logical frame
bit permutations. PhotoGraphiQML retains the resulting joint distribution. Marginals
are derived by summation, never multiplied together to reconstruct a joint state.
Tests compare the two-wire output directly to the public upstream readout and
verify nonzero connected correlations at finite resources.

The scalar supervised wrappers use the first output's decoded Z expectation,
`p(0)-p(1)`, followed by a classical affine/logistic head. They never index Fock
amplitudes as qubit probabilities. Leakage is independently computed through
`GKPCode.leakage` on each output's reduced physical density matrix. Joint
code-subspace leakage remains unknown; marginal leakages are not added or
multiplied to invent it. Gram overlap and preparation projection weights are
reported separately from leakage and decoder confidence.
