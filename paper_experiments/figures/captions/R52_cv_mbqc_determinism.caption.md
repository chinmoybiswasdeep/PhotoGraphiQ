**R52 - CV-MBQC determinism: ideal map, finite-squeezing noise and outcome dependence kept
apart.** Five CV-flow corrected Gaussian patterns (corrections from `find_cv_flow`, executed
as `corrected_pattern`): one-step teleportation (shear 0.35), weighted 4-node wire, 4-mode graph
teleportation, weighted 2x3 cluster and a branched 1 -> 2 isometry, at resource squeezing
r = 0.25..3.5. (a) Added noise N of the outcome-averaged channel
V -> S V S^T + N, projected on F = null(N_+) (the isometry's ideal infinitely squeezed output
quadrature excluded); S equals the closed form / independent linear Wigner model to 3.8e-16 at
every r, N matches the model to 1.9e-16. (b) Residual dependence of the corrected conditional
mean on the records, G = L Sigma R^T (R Sigma R^T)^-1. Fitted slopes d ln/dr (N over r >= 1 /
G over r >= 2): teleport_1step: -2.000 / -1.987, weighted_wire_4: -2.000 / -1.908, graph_teleport_4mode: -2.000 / -1.981, weighted_cluster_2x3: -2.000 / -1.957, branched_isometry: -2.000 / -1.987; dashed: e^{-2r}. (c) Postselected conditional output mean of the
teleported q quadrature versus the record (exact Gaussian postselection; each point is one
branch, not a deterministic run): the slope flattens towards the ideal (S mu)_q as r grows
(teleport N_finite = 0.607 at r = 0.25,
0.000912 at r = 3.5). Library postselected moments agree with
the independent model to 1.4e-15 (relative). At finite r the channel is noisy: never an
ideal unitary. Source: R52 (52_cv_mbqc_determinism.py).
