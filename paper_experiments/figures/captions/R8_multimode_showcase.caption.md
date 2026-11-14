**R8: five-mode circuit compiled to a CV-MBQC pattern.** The circuit R(0.4) on mode 0, S(0.3) on
mode 1, R(-0.6) on mode 2, CZ(0.8) on modes 0-1, D(q=0.2, p=-0.1) on mode 3, BS(0.3) on modes 2-3,
R(0.5) on mode 4 and CZ(-0.5) on modes 3-4 is compiled with resource squeezing r = 1.0 into
138 nodes, 141 weighted CZ edges (grey),
133 homodyne measurements and 134 displacement corrections
(542 commands). Each node is placed at the position of its homodyne in the command
order (x) and on the row of the logical mode it carries (y; nodes of two-mode gates go to the row of the
graph-closest input); colours give the source gate (labels above), red arrows the classical feed-forward
read from `Pattern.dependencies()` (a measurement outcome used by a later correction or measurement angle).
The compiled channel reproduces the independently multiplied 10x10 symplectic map to
3.0e-15 and the displacement to 3.5e-16; the
finite-squeezing noise is reported separately (Tr N = 21, ||N|| =
5.7). Graph metadata (edges and weights, node squeezing, single
measurement of every non-output node, feed-forward order, `OpenGraph` round trip) is consistent. Executed on
`gaussian` and `native-gaussian` (200 trajectories each): ensemble moments within 3.05
shot-noise standard errors of the analytic channel; native and NumPy records agree to
4.4e-15. Source:
`paper_experiments/02_compiler/08_multimode_showcase.py`.
