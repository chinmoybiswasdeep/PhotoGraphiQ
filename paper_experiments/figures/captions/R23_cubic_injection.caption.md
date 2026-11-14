**R23: finite-resource cubic-phase injection (m = 0 branch).** Fidelity between the output of
`non_gaussian.cubic_injection` (gamma = 0.15, coherent input alpha = (0.2+0j), ancilla homodyne
postselected at m = 0.0, so the nonlinear feed-forward corrections vanish) and the ideal
CubicPhase(gamma) applied to the same truncated input, built independently with NumPy/SciPy, versus
resource squeezing r. Solid: `native-fock` (main engine); dashed: `piquasso-fock` (reference); colours:
total-photon cutoff [24, 32, 48]. The injection is a conditional filter times a cubic phase, not an exact
unitary; its fidelity rises with r from 0.9469 to
0.9890 on `native-fock`. Cells whose retained Fock
norm left the guard band (norm_tolerance 1e-3) are reported as failures, not plotted: 10
of 18 on `native-fock` and 10 on `piquasso-fock`;
larger r needs a larger cutoff because the squeezed ancilla's photon number grows as sinh^2 r. Where both
engines succeed their outputs agree to fidelity >= 0.999992. Postselected
branch, finite energy: no claim about deterministic or fault-tolerant cubic gates. Source:
`paper_experiments/07_non_gaussian/23_cubic_injection.py`.
