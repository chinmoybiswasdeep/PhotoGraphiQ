# Physical Fock homodyne instrument

## Quadrature wavefunctions

With ħ = 2 and q = a + a†, the q-eigenfunctions of the number states follow the stable
three-term recurrence:

ψ₀(x) = (2π)^{−1/4} e^{−x²/4},  ψ_{n+1}(x) = (x ψ_n(x) − √n ψ_{n−1}(x)) / √(n+1).

High-order Hermite polynomials are never evaluated. Measuring
x_θ = q cos θ + p sin θ is the same as measuring q after exp(−iθn), so
⟨x_θ = x|n⟩ = e^{−inθ} ψ_n(x).

## Instrument

For a state with coefficients c[…, n, …] on the measured mode:

- conditional (unnormalised) amplitude: ψ̃(x) = Σ_n c[…, n, …] e^{−inθ} ψ_n(x);
- outcome density: p(x) = ‖ψ̃(x)‖². This is a density in the calibrated outcome x, in the same units as the Gaussian backend's records;
- conditional state: ψ̃(x)/√p(x). For a density matrix, ρ̃(x) = ⟨x_θ|ρ|x_θ⟩ on the measured mode, normalised by its trace.

```python
from photographiq.hybrid import homodyne as hd
out = hd.measure(rep, mode, theta, outcome=0.3)            # postselected
out = hd.measure(rep, mode, theta, rng=rng)                 # sampled
out.outcome, out.density, out.state, out.tail_mass, out.conditional_boundary, out.convention
hd.density(rep, mode, theta, xs)                            # p(x) on a grid (uses the reduced state)
```

## Sampling

The outcome is drawn by inverse-CDF sampling on a `grid`-point window (default 4001
points), with linear interpolation inside each cell. The window is the measured
quadrature's mean ± 10σ, widened to cover the truncated mode's support radius 2√d + 4.
The probability outside the window is reported as `tail_mass`. It was below 1e-10 in
HYB-10.

## Finite resolution (bins)

`resolution=w` measures bins [c − w/2, c + w/2) centred on multiples of w. The stated
convention is a fine measurement followed by coarse graining:

- P(bin) = ∫_bin p(x) dx;
- the conditional state is the mixture ∫_bin p(x) ρ(x) dx / P(bin), because the sub-bin record is discarded;
- the integral uses a 64-point midpoint rule.

The result's `convention` string says this explicitly. As w → 0, both P(bin)/w and the
state converge to the ideal instrument as O(w²); HYB-09 measured slopes of 1.99.

## Validation (HYB-03, HYB-09, HYB-10)

| Check | Result |
|---|---|
| vacuum, coherent (θ = 0, 0.7, 2.0), squeezed (r = 0.4, three angles), number states n = 1..4 (scipy Hermite) vs closed forms | max error < 1e-9 |
| two-mode Gaussian conditional state vs the Gaussian backend (pure and lossy/mixed, 7 outcomes, dims 24) | moments ≤ 2.3e-5, densities ≤ 1e-7 |
| non-Gaussian (cubic phase + beam splitter) vs native-fock postselection (dims 30 vs cutoff 40, 5 outcomes) | density ≤ 9e-8, infidelity ≤ 6e-9 |
| binned → ideal | O(w²) (slopes 1.99) |
| 5000 samples of a cubic-phase state | KS 0.010 < 0.023 (1% critical value); reproducible seeds |
