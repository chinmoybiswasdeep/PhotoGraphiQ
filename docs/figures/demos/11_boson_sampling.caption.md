**Boson sampling.** (a) Transition probabilities |U_ij|^2 of a 4-mode Haar-random
interferometer (`sampling.haar_unitary(4, rng=7)`). (b) For the input |1,1,1,0>, exact
output probabilities from permanents (`sampling.output_distribution`, all
20 patterns, collision-free mass 0.3165) against
40000 exact samples from `sampling.sample(..., method="enumerate")` (rng 1; this draws from the same enumerated distribution, so it checks the sampler, not the permanents); error bars are binomial 1-sigma.
Total variation distance over all patterns: 0.0074. Lossless, perfectly indistinguishable
photons and ideal photon-number-resolving detectors.

Source: `scripts/generate_demo_figures.py` (`11_boson_sampling`).
