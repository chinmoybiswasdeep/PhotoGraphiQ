**R51 - CV flow decisions against an independent exact reference.** (a) Library decision for
193 graphs (120 seeded weighted random G(n, p) graphs, n = 4..7, signed
weights |w| log-uniform in [0.2, 3]; plus named linear, 2D cluster, branched and no-certificate
cases): 81 certified, 111 impossible, 1
numerically uncertain. An exact rational forward search over measured prefix sets (independent
of the library's backward layering) agrees on existence for 192/192 decided graphs;
75 impossibility witnesses y^T A[W, C] = 0 were re-checked exactly. The uncertain
case (bridge with det 1e-10) is certified only by `exact=True`, ideal limit only. (b) For each
certified graph, least-squares re-derivation of c_u on the reported order (worst residual
2.3e-15; dashed: acceptance 1e-09) against the largest correction coefficient.
"Impossible" means no admissible order for the real-linear CV-flow condition, not a proof of
nondeterminism by other means. Source: R51.
