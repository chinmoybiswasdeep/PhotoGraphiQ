**R49 - Gaussian boson sampling with PNR detection.** Squeezed vacua (q-squeezed, r per mode) through
scipy Haar interferometers, optional loss and coherent displacement (`squeezed_interferometer_state`).
(a, b) Undisplaced lossless states (m = 2..5 modes, r = 0.3..1.0): hafnian probabilities agree with an
independent Fock-space evolution (product of single-mode squeezed columns, total-photon cutoff, m-mode passive
Fock matrix), which is exact for every pattern below the cutoff: worst absolute error 5.6e-16; the pure
|haf(B)|^2 and mixed haf(A) formulas agree. (b) also: lossy and displaced states (m = 4..12, eta =
0.7..1) against Piquasso 8.0.1 `get_particle_detection_probability`, worst relative difference 6.3e-12 for
P > 1e-12. (c) Displaced states: the loop-hafnian probabilities are approached by the Fock reference as its
cutoff grows (displacement applied after truncation; 2.2e-16 at the largest cutoff, |n| <= 4). (d) Hafnian
and loop hafnian vs the perfect-matching definition. Sum rule: mass difference
1.7e-15 (hafnian mass below the cutoff vs the exact retained Fock mass,
undisplaced cases with cutoff <= 13); patterns with more than 12 photons are not evaluated.
Seed 20261149. Source: R49 (16_sampling/49_gaussian_boson_sampling.py).
