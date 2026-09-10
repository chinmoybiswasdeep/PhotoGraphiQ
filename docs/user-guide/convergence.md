# Cutoff convergence

`cutoff_convergence(pattern, [c1,c2,...], measurement_outcomes=...)` runs increasing
total-photon cutoffs. The default backend is `piquasso-fock`; choose
`backend="piquasso-mixed-fock"` for density-matrix studies. A callable pattern
factory receives the cutoff so explicitly truncated resources can be rebuilt.

Each row reports norms, maximum boundary population, occupation-aligned fidelity,
trace distance, probability L1 distance, quadrature moments and timing. Optional
`high_order_moments=True` includes raw moments through fourth order and n².

Equal seeds do not guarantee equal sampled branches at different cutoffs. For
state comparisons use fixed outcomes; otherwise inspect the comparability flag.
Converging normalized output fidelity alone can hide an unconverged probability
or density. High moments often need more cutoff than mean photon number.

Run [Tutorial 15](../tutorials/15-cutoff-convergence.md), then vary gate strength,
input energy and resource squeezing. For GKP also refine the wavefunction grid;
for Kerr synthesis refine product-formula steps independently.
