# Noise: loss and thermal channels

`pg.Loss(node, transmissivity=η, thermal_photons=n)` is the Lindblad channel with jump
operators L₁ = √(1 + n) a and L₂ = √n a†, applied for time t = −ln η. On covariances this
channel is V → ηV + (1 − η)(2n + 1)I, the Gaussian backend's loss.

Execution switches to density matrices automatically once a pattern contains `Loss`
(`mixed=True` forces it).

| Engine | Implementation |
|---|---|
| `numpy-reference` | single-mode superoperator Λ = expm(t 𝓛) on the truncated mode, with 𝓛 = Σ_k J_k⊗J_k* − ½J_k†J_k⊗I − ½I⊗(J_k†J_k)ᵀ (row-major vec). It is cached per (η, n, d) and applied by tensor contraction to the lossy mode only. |
| `cudaq` | `cudaq_dynamics.evolve(0, dims, [0, t], ρ, collapse_operators=[...])` on the full state |

For pure loss (n = 0) the truncated Lindbladian is exact: a only lowers the photon number,
so the truncated space is invariant. Λ then equals the Kraus channel
K_k = Σ_n √(C(n,k) η^{n−k}(1−η)^k) |n−k⟩⟨n|, to 1e-10 in `tests/hybrid`. For thermal loss,
the truncated a† is an approximation, and convergence in d has to be checked.

## Validation

| Check | Result |
|---|---|
| Λ vs exact Kraus channel (η = 0.3, 0.8, d = 8) | < 1e-10 |
| thermal Λ vs `reference_evolve` Lindblad (η = 0.7, n = 0.2) | < 1e-8 |
| lossy Kerr teleportation vs native-mixed-fock (cutoff 24) | trace distance < 2e-3 at dims 16 (truncation-limited) |
| HYB-06: lossy teleportation (η = 0.9, 0.7, 0.5 and η = 0.8, n = 0.1) vs Gaussian backend, dims 24 | mean ≤ 7.6e-7, cov ≤ 1.2e-5, density ≤ 7.4e-8 |
| HYB-07: Kerr-enhanced MBQC step with η = 0.85 vs native-mixed-fock (cutoff 26) | trace distance ≤ 2.1e-4, at the native engine's own convergence level |

Inefficient or noisy homodyne (`Homodyne(efficiency<1)` / `noise>0`) is **not**
implemented in the hybrid executor and raises `NotImplementedError`. Model detector loss
with an explicit `Loss` before the homodyne instead.
