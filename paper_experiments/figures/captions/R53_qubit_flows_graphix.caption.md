**R53 - qubit causal flow, gflow and Pauli flow against Graphix 0.4.** 9 named open
graphs (textbook successes and failures) and 300 seeded random open graphs per notion
(n = 3..8; planes XY only for causal flow, XY/XZ/YZ for gflow, 40% Pauli labels for Pauli flow).
(a) Existence decisions agree with Graphix's `to_*flow_or_none` on
927/927 (graph, notion) pairs;
negative answers come from complete layered algorithms. Every returned flow (ours and Graphix's)
satisfies the defining conditions (F1-F3, G1-G3, P1-P9) re-checked by an independent set-algebra
checker, and 464 flow-derived patterns were simulated branch by branch by an independent dense
statevector code: all 2^m branches equiprobable with identical output up to phase (worst
deviation 6.7e-16). (b) Jointly certified flows: identical layer partitions in
218 cases, ours never deeper (maximally delayed).
Correction sets need not coincide (flows are not unique): identical sets in
178 of the jointly certified cases. Qubit flow
notions are never applied to CV graphs (`analyze(cv, 'gflow')` returns `unsupported`). Source: R53.
