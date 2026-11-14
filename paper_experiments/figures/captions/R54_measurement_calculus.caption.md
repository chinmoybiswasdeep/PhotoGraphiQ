**R54 - qubit measurement calculus preserves the full branch semantics.** 120 seeded random
valid non-standard DKP patterns (2 inputs, 3 ancillas, 2 outputs; interleaved N/E/M/X/Z, random
planes/angles, random s/t domains) and 5 flow-derived standard patterns (causal flow,
gflow incl. XZ/YZ planes, Pauli flow). (a) For each of the 1032 outcome branches, Choi
vectors of the original, standardized (NEMC) and signal-shifted patterns from an independent dense
simulator (measuring Z^t X^s O X^s Z^t), compared up to a branch phase: worst 8.3e-17
(standardize), 8.3e-17 (signal shift); branch probabilities differ by at most
1.9e-16. Library `run_branch` vs independent simulator:
2.8e-16; Graphix 0.4 statevector on 1028 branches with a random input:
1 - |overlap| <= 5.6e-16. Dashed: acceptance 1e-10. Negative control: dropping one output
correction is detected (branch deviation > 0.001) in 65/80 patterns. (b) Dependency depth of the
measurement schedule before and after signal shifting. (c) Rewrite-rule applications during
standardization. CV patterns: only command classification and exact affine correction extraction
are validated (31 affine components, worst relative error
2.8e-16; nonlinear feed-forward reported as non-affine);
no CV measurement calculus is claimed. Source: R54.
