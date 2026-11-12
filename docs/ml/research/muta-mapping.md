# Mathematical and implementation mapping

| MuTA concept | MentPy implementation | Meaning | PhotoGraphiQML abstraction | PhotoGraphiQ primitive | Validation |
|---|---|---|---|---|---|
| Triangle | two cross edges on length-5 wires | tip/base four-cycle | TriangleNeuron | GKP resources only after lowering | semantic graph edges |
| Trainable measurement | Ment XY angle=None | cos(a)X+sin(a)Y | logical measurement expression | Parameter; no general XY instrument | qubit contraction and MentPy |
| Fixed measurement | Ment(0) | X basis | zero angle, frozen metadata | different from Homodyne(0) | Table I |
| Wire | many_wires | logical teleportation | (wire,column) nodes and flow | GKP encoded optical mode | input/output order |
| Coupling | tip cross edges | exp(i a XX/2) | connected base-center angle | encoded Clifford + non-Clifford protocol needed | entangling-gate identity |
| Layer | muta + hstack | boundary identification | pivot block schedule | Pattern composition after physical derivation | depth and edge counts |
| Bias | geometry/restricted parameters | inductive bias | connectivity + freeze | parameter bindings | disabled constraint restores full family |
| Dependency | flow correction | X_f Z_neighbors | causal DAG / correction sets | Outcome for physical protocol only | all small adaptive branches |
| Output | qubit vector/density | ordered logical state | logical result | physical Result kept separate | logical fidelity |
| Instrument | controlled Ment + output measurement | CP maps with outcomes | branch probability and conditional state | physical instrument unsupported | completeness and branch states |
| Kernel | feature map overlaps | Eq. 4 and 5 | MuTAKernel | no raw-CV MentPy oracle | PSD and direct Eq. 5 |

## GKP bridge contract

At hbar=2, lattice spacing L=sqrt(2 pi), X_L=exp(-i L p/2),
Z_L=exp(i L q/2); ideal codewords have q=(2s+mu)L. Physical CZ of
unit weight exp(i q1 q2/2) gives (-1)^(mu nu) on these ideal lattice sites.
This identity does not extend exactly to finite peaks and finite envelopes.
Finite codeword Gram matrices need not be identity. Normalizing a
superposition of nonorthogonal codewords is a preparation recipe, not an
isometric encoding channel. Do not implement a decoder by silently taking
overlaps and renormalizing away leakage. Report physical probabilities with
a specified POVM, or use ideal targets explicitly labeled as such.

## CV-native derivation proposal (separate research stage)

A momentum-squeezed ancilla, unit CZ and homodyne q cos(theta)+p sin(theta)
with q feed-forward -m/sin(theta) implement, in the ideal limit,

    (q_out,p_out) = (-cot(theta) q_in - p_in, q_in).
    T(k) = [[-k,-1],[1,0]], k=cot(theta).

At finite squeezing r the unconditional Gaussian channel has added noise
diag(0,exp(-2r)) for one step, propagated under later steps. Four steps at
theta=pi/2 compose to ideal identity. A candidate CV cell may compose such
local steps with weighted physical CZ and displacement bias. Weighted CZ
is continuously removable at g=0 and is symplectic; its physical meaning
differs from MuTA's tunable logical XX. Gaussian-only resources cannot inherit
qubit universality. This is a proposal, not yet CVMuTA: a validated GKP MuTA
bridge is the prerequisite for promoting a separate CVMuTA implementation.
