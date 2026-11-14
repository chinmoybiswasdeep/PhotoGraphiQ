**R55 - local complementation as a local Clifford on graph states (qubit).** (a) Five-cycle graph
state |C5> (|G> = prod CZ |+>^n). (b) Local complementation at vertex 0 (yellow) toggles the edges
inside N(0) (red solid: added, red dotted: removed); U_0 = exp(-i pi/4 X_0) prod_(w in N(0)) exp(+i pi/4 Z_w)
maps |C5> to the new graph state up to the global phase shown. (c) A second complementation at
vertex 2. Over 51 (graph, vertex) pairs on 11 graphs (path, star, cycle, complete,
2x3 cluster, 6 random connected graphs) the independently built U_v gives
1 - |<tau_v G|U_v|G>| <= -2.2e-16; global phases are multiples of pi/4
(observed [0, 1, 2] x pi/4). 44 random LC sequences (length 1-6) were recovered by
`lc_equivalence` with Clifford witnesses verified on statevectors (worst 7.8e-16).
Non-equivalences (path vs star, cycle/cluster vs complete graph) are confirmed by differing
Schmidt-rank vectors, an LU invariant. CV: only the Gaussian graph rules of MFvL 2011 are
validated (covariance error <= 1.3e-15); CV local complementation is not implemented.
Source: R55.
