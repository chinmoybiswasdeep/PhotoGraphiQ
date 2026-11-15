**Non-Gaussian (cubic-phase) injection.** (a) `non_gaussian.cubic_injection(0.25,
squeezing=0.5)`: the input `in` is coupled to a finite-energy cubic-phase resource by CZ and
the resource is measured in q (key `m`); the feed-forward is nonlinear (QuadraticPhase(-2
gamma m) and Displace(p = -gamma m^2)), so it is listed as an annotation, not drawn as a
linear correction, and `flows.analyze` returns `unsupported` for this model. (b) Signed
Wigner function of the output of one `native-fock` trajectory (vacuum input, seed 3,
total-photon cutoff 56, smallest retained norm over all operations 0.999804,
3 truncation warnings); wigner function (grid mass 0.9995, negative volume 0.00139). The finite resource implements CP(gamma)
times a Gaussian filter exp(-(q + m)^2 / (4 e^(2r))), a conditional operation, not an exact
unitary cubic gate. Limitations: one trajectory; Fock truncation as stated.

Source: `scripts/generate_demo_figures.py` (`08_non_gaussian_injection`).
