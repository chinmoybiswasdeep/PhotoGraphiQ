**R48 - Fock-state boson sampling.** (a) Permanent accuracy of the native Gray-code Ryser kernel and
the NumPy multiplicity-aware Ryser (3 seeded submatrices of scipy Haar unitaries per size, scaled to
O(1) entries) against the permutation-sum definition (n <= 8) and Glynn's formula (n = 9..20),
shown as relative error (worst 2.7e-11; the Glynn reference is itself float64);
acceptance is on the error normalised by the Ryser term scale prod_i sum_j |a_ij| (worst 1.5e-16);
bunched permanents (repeated rows/columns) are checked against the definition on the repeated matrix. (b) Single-call runtime
(median of 3 after warm-up; provisional, shared machine): native at n = 20: 7.77e-03 s with
12 OpenMP threads, 4.32e-02 s with 1 thread; the multi-threaded kernel was slower
than 1 thread for n in [1, 2, 3, 4, 5, 6, 7, 8, 9] on this loaded machine
(OpenMP fork/join and spin-wait overhead; see R48 notes). (c) Full output
distributions for 10 (modes, input) cases incl. collision inputs: normalisation (worst
1.2e-15), agreement with an independent Fock-space evolution of the same
interferometer (native sector recurrence, cutoff n + 1; worst 5.0e-16)
and with Piquasso 8.0.1 (worst 5.3e-16). Not shown: the DFT suppression law
holds (suppressed mass <= 6.7e-27 for m = 3..7);
seeded samplers pass Pearson chi-square (p = 0.63, 0.30, 20000 shots).
Seed 20261148. Source: R48 (16_sampling/48_boson_sampling.py).
