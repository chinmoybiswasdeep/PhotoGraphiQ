**Single-mode CV teleportation (one cluster wire step).** (a) The resource graph of
`protocols.wire((0.0,), squeezing=1.5)`: input 0 is entangled with a p-squeezed mode 1 by
CZ(1) and measured in p (homodyne angle pi/2, shear k = 0); `flows.analyze` certifies the
CV flow (`certified`) and the dashed arrow is the feed-forward the pattern executes,
q += -1 m[0]. (b) One `native-gaussian` trajectory (seed 1) for a coherent
input alpha = (0.8+0.4j); the recorded outcome is m = 2.387. (c) Phase-space
1-sigma ellipses (hbar = 2, vacuum = unit circle): ideal map F (Fourier transform), the
unconditional channel from `gaussian_channel` (S = F to
6.1e-17, added noise Tr N = 0.0498,
i.e. finite squeezing r = 1.5 adds noise; it is not the ideal unitary) and this run's
conditional output state, whose mean is shifted by an outcome-dependent amount at finite
squeezing. Limitations: ideal homodyne detection; no loss.

Source: `scripts/generate_demo_figures.py` (`01_single_mode_teleportation`).
