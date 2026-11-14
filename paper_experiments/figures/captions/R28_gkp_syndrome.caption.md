**R28: finite-ancilla GKP q-syndrome extraction.** A finite-energy |0>_L (peak width = envelope = 0.35) is
displaced by delta in q and passed through the one-ancilla SUM syndrome extraction with modular feed-forward
(`gkp.correction_pattern`, ancilla |+>_L of the same width); `native-fock`, total-photon cutoff 80 shared by
both modes. Left: circular mean (+/- standard error) of the decoded residual over 16 seeded trajectories per
delta versus the ideal delta mod sqrt(2 pi) (max deviation 0.053);
single shots are broad because the finite ancilla's comb peaks are sampled (median circular spread
0.373). Right: mean |<q>| after correction
versus |delta|. Pooled over 144 trajectories: syndrome error rate (|residual error| > sqrt(2 pi)/4)
9/144, Wilson 95% [0.033, 0.115];
0 simulation norm-guard failures. The right panel's |<q>| (~1.9 at every delta, as in the historical
Piquasso run) is not used as a failure measure; its origin (plausibly partial measurement of the data q by the finite
ancilla) is not analysed here. For deltas [-0.6, 0.0, 0.45] the same seeds on `piquasso-fock` give syndrome records within
4.5e-04. Single round,
finite energy, finite ancilla, ideal homodyne: no fault-tolerance or threshold claim. Source:
`paper_experiments/08_gkp/28_gkp_syndrome.py`.
