# Native Fock: single-mode Gaussian unitary accuracy

Scope: `gaussian_unitary` in `cpp/src/fock/gates.cpp`, which gives the matrix elements
<m|U|n> of the Gaussian unitary with U^dag x U = S x + d (hbar = 2). The native Fock
backends use it for displacement, squeezing, quadratic phase and Gaussian inputs. Pure
`GaussianInput` uses column 0. Mixed `GaussianInput` uses the thermal columns
0..terms-1.

## Problem

Before this change, every Gaussian unitary went through one Bargmann
(generating-function) recurrence, except a pure displacement (S = 1), which used a
Laguerre recurrence. On one mode, that recurrence amplified rounding exponentially in
two cases:

- squeezing and displacement combined in one unitary: errors up to 1.4e2 (table below);
- squeezing alone at large size: rows < 100 of S(r = 1) at size 400 were off by 7.2e-5,
  and at size 150 by 3.3e-11.

Column 0 was always accurate to 1e-15, so pure Gaussian inputs were not affected. The
thermal columns of mixed Gaussian inputs, and the displaced and squeezed gates, were
affected.

## Root cause

The Bargmann recurrence G_{k+e_i} = (b_i G_k + sum_j A_ij sqrt(k_j) G_{k-e_j}) /
sqrt(k_i + 1) is equivalent to building U|n+1> = (U a^dag U^dag) U|n> / sqrt(n+1) column
by column. Rounding error in U|n> that sits at high photon number grows by about
sqrt(N + 1) / sqrt(n + 1) at each step. Errors therefore grow like exp(cutoff / 2),
whatever the pivot. The pivot rule from the earlier fix (most occupied side) only
reduces the growth rate. Where elements are exponentially small relative to their
predecessors (classically forbidden regions, far from the diagonal), the terms cancel
and the amplified rounding is the only thing left. The largest errors lie on the last
row or column (m = 99 or n = 99 at cutoff 100), at n or m between 30 and 72. Operator
ordering and the sqrt(n!) normalisation are not involved: a 60-digit run of the same
recurrence is exact.

## Fix (one mode only)

U = R(phi1) S_r R(phi2) D(beta), where:

1. **Decomposition.** S = rotation(phi1) diag(e^-r, e^r) rotation(phi2) comes from the
   2 x 2 SVD, and S d_beta = d. Rotations are exact phases exp(i phi n). This is exactly
   the operator of the Bargmann form, with the same global phase: D(delta) U_S =
   U_S D(beta), and <0|U_S|0> > 0. In the old path's accurate regime, the old and new
   results agree to 5.4e-14 over 208 random cases with cutoff <= 80 (|alpha| <= 3,
   |r| <= 1, shear, rotation, pure squeeze and pure displacement included). The one
   exception is rotation(2 pi) with |alpha| = 4, where the old Bargmann path was itself
   in error by 1.9e-12.
2. **S_r.** Jacobi form, a new closed form (SU(1,1) analogue of the Laguerre form of D).
   With parity p, b = p - 1/2, m = 2(nu + j) + p, n = 2 nu + p and t = tanh r:
   <m|S_r|n> = sech(r)^(p+1/2) (-t)^j sqrt(nu! G(nu+j+b+1) / ((nu+j)! G(nu+b+1)))
   P_nu^(j, b)(1 - 2 t^2), and <n|S_r|m> = (-1)^j <m|S_r|n>. It is computed by the
   three-term Jacobi recurrence in nu, rescaled to avoid underflow. The argument lies in
   (-1, 1], and the polynomial is never the minimal solution of its recurrence, so
   rounding does not grow.
3. **D(beta).** The existing Laguerre recurrence, now rectangular (L x cutoff).
4. **Product.** The product runs over an internal size L >= cutoff. Because the rows of
   the unitary U_S have norm <= 1, the discarded part satisfies
   |sum_{l >= L} <m|U_S|l><l|D|n>| <= ||P_{>=L} D(beta)|n>|| (Cauchy-Schwarz). L is the
   smallest size for which a **rigorous** bound on that tail is <= 1e-16 for every
   n < cutoff. The bound is the majorant
   M_ln = e^{-x/2} sqrt(l! n!) sum_k |beta|^(l+n-2k) / (k!(l-k)!(n-k)!) >= |<l|D|n>|.
   It satisfies the positive recurrence
   M_{l+1,n} = (|beta| M_ln + sqrt(n) M_{l,n-1}) / sqrt(l+1), with termwise ratio
   M_{l+1,n} / M_ln <= q_l = |beta| sqrt(l+1) / (l+1-n). Since q_l decreases in l, the
   remaining tail after L is <= M_Ln / sqrt(1 - q_L^2).

How tight the bound is, against the exact tail computed from a size-2600 D:

| abs(beta) | cutoff | L (bound) | L needed (exact tail <= 1e-16) | (sqrt(cutoff)+abs(beta))^2 |
|---|---|---|---|---|
| 0.5 | 20 | 47 | 47 | 25 |
| 0.5 | 100 | 137 | 136 | 110 |
| 2 | 20 | 83 | 81 | 42 |
| 2 | 100 | 195 | 187 | 144 |
| 6 | 20 | 194 | 183 | 110 |
| 6 | 100 | 367 | 330 | 256 |
| 10 | 20 | 339 | 314 | 209 |
| 10 | 100 | 576 | 501 | 400 |
| 16 | 20 | 618 | 570 | 419 |
| 16 | 100 | 958 | 815 | 676 |

|beta| <= e^|r| |alpha|. With squeezing only (d = 0) or a pure displacement, no
internal size is needed. Multi-mode unitaries (k >= 2) still use the Bargmann
recurrence (see the limits below).

## Method of validation

- **Reference.** <m|U|n> from the Bargmann recurrence in mpmath at 60 digits, with S
  built exactly symplectic in mpmath (the rounded double S is passed to the engine).
  The reference is checked independently against the Heisenberg relation
  a U = U (T a + R a^dag + delta) on every interior element (residual <= 6e-26).
  60-digit and 90-digit references agree to the last double bit (difference 0.0).
- **Grid.** |alpha| in {0.5, 1, 2, 3, 4, 5, 6}, r in {0.1, 0.3, 0.6, 1.0}, with the
  phases of alpha, phi1 and phi2 uniform at random (seed 2026), and cutoff in
  {20, 40, 60, 80, 100}. The engine is evaluated at each cutoff and compared with the
  top-left block of the reference.

### Max |U - U_ref| (before / after)

| abs(alpha) | r | 20 | 40 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|
| 0.5 | 0.1 | 1.8e-15 / 3.2e-15 | 3.0e-15 / 7.8e-15 | 3.9e-15 / 8.9e-15 | 4.5e-15 / 9.3e-15 | 5.8e-15 / 1.9e-14 |
| 0.5 | 0.3 | 1.2e-15 / 1.5e-15 | 2.2e-15 / 4.8e-15 | 5.6e-15 / 6.7e-15 | 4.7e-14 / 7.0e-15 | 4.5e-13 / 9.4e-15 |
| 0.5 | 0.6 | 3.5e-16 / 1.6e-15 | 1.0e-15 / 6.9e-15 | 9.0e-15 / 1.3e-14 | 1.2e-13 / 1.4e-14 | 1.8e-12 / 1.4e-14 |
| 0.5 | 1.0 | 2.6e-16 / 1.2e-15 | 8.5e-16 / 5.7e-15 | 6.5e-15 / 7.5e-15 | 4.3e-14 / 7.6e-15 | 3.1e-13 / 8.8e-15 |
| 1 | 0.1 | 1.4e-15 / 3.1e-15 | 2.5e-15 / 4.6e-15 | 3.8e-15 / 6.8e-15 | 1.1e-14 / 8.5e-15 | 4.3e-14 / 1.2e-14 |
| 1 | 0.3 | 1.2e-15 / 1.3e-15 | 3.3e-15 / 3.9e-15 | 2.5e-14 / 4.7e-15 | 2.2e-13 / 6.0e-15 | 1.8e-12 / 6.8e-15 |
| 1 | 0.6 | 3.0e-16 / 1.8e-15 | 5.2e-15 / 2.1e-15 | 1.1e-13 / 3.2e-15 | 1.9e-12 / 3.5e-15 | 3.1e-11 / 4.0e-15 |
| 1 | 1.0 | 2.9e-16 / 6.0e-16 | 7.9e-16 / 2.7e-15 | 8.0e-15 / 6.3e-15 | 8.6e-14 / 7.2e-15 | 8.3e-13 / 7.2e-15 |
| 2 | 0.1 | 5.0e-15 / 2.5e-15 | 1.6e-13 / 4.8e-15 | 2.8e-12 / 7.7e-15 | 3.6e-11 / 1.1e-14 | 3.8e-10 / 1.3e-14 |
| 2 | 0.3 | 2.9e-15 / 1.4e-15 | 1.1e-13 / 2.3e-15 | 2.8e-12 / 5.0e-15 | 5.6e-11 / 6.5e-15 | 9.9e-10 / 8.1e-15 |
| 2 | 0.6 | 1.1e-15 / 8.7e-16 | 2.0e-14 / 1.6e-15 | 3.2e-13 / 2.0e-15 | 5.8e-12 / 3.6e-15 | 1.1e-10 / 6.3e-15 |
| 2 | 1.0 | 6.1e-16 / 1.3e-15 | 2.1e-15 / 2.5e-15 | 2.3e-14 / 3.1e-15 | 2.3e-13 / 3.1e-15 | 2.0e-12 / 3.8e-15 |
| 3 | 0.1 | 5.1e-14 / 1.5e-15 | 1.2e-11 / 3.0e-15 | 7.9e-10 / 3.9e-15 | 2.8e-08 / 5.1e-15 | 6.8e-07 / 5.5e-15 |
| 3 | 0.3 | 4.7e-14 / 1.5e-15 | 8.0e-12 / 2.2e-15 | 6.1e-10 / 2.7e-15 | 3.2e-08 / 4.4e-15 | 1.3e-06 / 5.1e-15 |
| 3 | 0.6 | 1.3e-13 / 1.9e-15 | 3.3e-12 / 2.9e-15 | 2.8e-10 / 2.9e-15 | 2.2e-08 / 4.5e-15 | 1.4e-06 / 4.5e-15 |
| 3 | 1.0 | 1.6e-15 / 4.1e-15 | 1.1e-13 / 4.4e-15 | 4.9e-12 / 4.4e-15 | 1.5e-10 / 5.2e-15 | 3.7e-09 / 5.4e-15 |
| 4 | 0.1 | 1.7e-13 / 1.4e-15 | 1.5e-10 / 2.0e-15 | 3.6e-08 / 2.5e-15 | 4.6e-06 / 2.8e-15 | 3.8e-04 / 5.0e-15 |
| 4 | 0.3 | 2.2e-13 / 2.4e-15 | 5.4e-10 / 3.3e-15 | 2.4e-07 / 4.3e-15 | 8.5e-05 / 4.7e-15 | 1.8e-02 / 6.0e-15 |
| 4 | 0.6 | 1.2e-14 / 2.2e-15 | 1.7e-12 / 3.0e-15 | 1.3e-10 / 4.6e-15 | 6.8e-09 / 5.1e-15 | 2.9e-07 / 6.1e-15 |
| 4 | 1.0 | 7.3e-14 / 6.0e-15 | 1.7e-12 / 7.3e-15 | 6.8e-11 / 7.3e-15 | 4.3e-09 / 8.7e-15 | 1.9e-07 / 8.7e-15 |
| 5 | 0.1 | 2.0e-13 / 1.7e-15 | 1.8e-09 / 2.4e-15 | 2.1e-06 / 3.5e-15 | 9.8e-04 / 5.3e-15 | 2.3e-01 / 5.5e-15 |
| 5 | 0.3 | 1.9e-13 / 2.7e-15 | 4.4e-10 / 3.7e-15 | 2.9e-07 / 3.9e-15 | 8.6e-05 / 4.9e-15 | 1.5e-02 / 6.5e-15 |
| 5 | 0.6 | 3.4e-14 / 1.2e-15 | 6.4e-12 / 1.8e-15 | 7.8e-10 / 2.4e-15 | 7.0e-08 / 3.5e-15 | 5.1e-06 / 4.1e-15 |
| 5 | 1.0 | 2.5e-15 / 2.5e-15 | 2.9e-14 / 2.6e-15 | 7.5e-13 / 3.3e-15 | 1.5e-11 / 4.1e-15 | 2.4e-10 / 4.5e-15 |
| 6 | 0.1 | 9.2e-14 / 1.4e-15 | 9.1e-09 / 2.5e-15 | 6.2e-05 / 4.0e-15 | 1.0e-01 / 4.0e-15 | 7.3e+01 / 5.2e-15 |
| 6 | 0.3 | 4.2e-14 / 1.6e-15 | 1.8e-08 / 4.3e-15 | 1.6e-04 / 7.9e-15 | 1.6e-01 / 8.4e-15 | 1.4e+02 / 9.2e-15 |
| 6 | 0.6 | 5.5e-14 / 1.0e-15 | 3.3e-11 / 2.4e-15 | 1.0e-08 / 3.1e-15 | 1.8e-06 / 4.1e-15 | 2.0e-04 / 5.3e-15 |
| 6 | 1.0 | 8.5e-15 / 2.7e-15 | 1.0e-12 / 3.5e-15 | 4.6e-11 / 4.0e-15 | 1.3e-09 / 4.0e-15 | 3.0e-08 / 4.6e-15 |

Location of the maximum before the fix (cutoff 100, |alpha| >= 3): always on the last
row or column, for example (99, 72), (61, 99), (99, 65) and (99, 58). After the fix, the
error has no structure (rounding level).

### Max |U - U_ref| / ||reference column|| (before / after)

| abs(alpha) | r | 20 | 40 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|
| 0.5 | 0.1 | 2.3e-15 / 3.9e-15 | 4.0e-15 / 9.7e-15 | 5.2e-15 / 1.2e-14 | 6.2e-15 / 9.8e-15 | 8.3e-15 / 2.7e-14 |
| 0.5 | 0.3 | 1.6e-15 / 2.0e-15 | 3.0e-15 / 7.4e-15 | 8.9e-15 / 1.1e-14 | 7.4e-14 / 9.7e-15 | 7.1e-13 / 1.5e-14 |
| 0.5 | 0.6 | 5.4e-16 / 2.8e-15 | 1.8e-15 / 1.2e-14 | 1.6e-14 / 2.3e-14 | 1.7e-13 / 2.3e-14 | 2.7e-12 / 2.1e-14 |
| 0.5 | 1.0 | 4.5e-16 / 2.3e-15 | 1.8e-15 / 1.2e-14 | 1.4e-14 / 1.5e-14 | 9.0e-14 / 1.5e-14 | 5.3e-13 / 1.8e-14 |
| 1 | 0.1 | 1.8e-15 / 3.1e-15 | 3.6e-15 / 6.4e-15 | 5.0e-15 / 8.2e-15 | 1.6e-14 / 1.2e-14 | 6.3e-14 / 1.4e-14 |
| 1 | 0.3 | 1.7e-15 / 2.0e-15 | 5.2e-15 / 6.0e-15 | 3.9e-14 / 6.7e-15 | 3.4e-13 / 9.3e-15 | 2.9e-12 / 1.1e-14 |
| 1 | 0.6 | 5.3e-16 / 2.8e-15 | 9.3e-15 / 3.3e-15 | 1.9e-13 / 5.4e-15 | 3.4e-12 / 5.8e-15 | 5.5e-11 / 6.9e-15 |
| 1 | 1.0 | 5.1e-16 / 1.2e-15 | 1.3e-15 / 5.7e-15 | 1.3e-14 / 1.3e-14 | 1.4e-13 / 1.4e-14 | 1.4e-12 / 1.3e-14 |
| 2 | 0.1 | 8.0e-15 / 3.0e-15 | 2.5e-13 / 6.4e-15 | 4.5e-12 / 1.0e-14 | 5.9e-11 / 1.4e-14 | 6.3e-10 / 1.6e-14 |
| 2 | 0.3 | 3.7e-15 / 1.9e-15 | 1.3e-13 / 3.1e-15 | 3.3e-12 / 7.3e-15 | 6.6e-11 / 9.6e-15 | 1.2e-09 / 1.3e-14 |
| 2 | 0.6 | 1.3e-15 / 1.4e-15 | 2.4e-14 / 2.6e-15 | 5.1e-13 / 2.8e-15 | 1.0e-11 / 6.4e-15 | 2.0e-10 / 1.1e-14 |
| 2 | 1.0 | 1.2e-15 / 2.8e-15 | 3.5e-15 / 4.9e-15 | 3.7e-14 / 5.0e-15 | 3.6e-13 / 5.8e-15 | 3.0e-12 / 6.5e-15 |
| 3 | 0.1 | 7.3e-14 / 2.4e-15 | 1.6e-11 / 4.2e-15 | 1.0e-09 / 5.5e-15 | 3.5e-08 / 7.7e-15 | 8.4e-07 / 7.5e-15 |
| 3 | 0.3 | 8.1e-14 / 2.6e-15 | 1.5e-11 / 3.0e-15 | 1.1e-09 / 3.9e-15 | 5.2e-08 / 6.6e-15 | 2.1e-06 / 8.3e-15 |
| 3 | 0.6 | 2.6e-13 / 3.4e-15 | 5.2e-12 / 5.5e-15 | 5.3e-10 / 5.3e-15 | 4.2e-08 / 7.9e-15 | 2.6e-06 / 7.1e-15 |
| 3 | 1.0 | 4.0e-15 / 6.6e-15 | 2.4e-13 / 8.8e-15 | 1.1e-11 / 9.4e-15 | 3.3e-10 / 1.1e-14 | 8.0e-09 / 1.1e-14 |
| 4 | 0.1 | 2.4e-13 / 1.9e-15 | 2.0e-10 / 2.6e-15 | 4.7e-08 / 2.9e-15 | 5.9e-06 / 3.8e-15 | 4.8e-04 / 5.2e-15 |
| 4 | 0.3 | 4.0e-13 / 2.6e-15 | 8.9e-10 / 5.3e-15 | 4.6e-07 / 6.1e-15 | 1.6e-04 / 6.5e-15 | 3.5e-02 / 9.1e-15 |
| 4 | 0.6 | 1.7e-14 / 3.1e-15 | 2.3e-12 / 4.8e-15 | 1.6e-10 / 7.8e-15 | 8.6e-09 / 8.6e-15 | 3.6e-07 / 1.1e-14 |
| 4 | 1.0 | 2.0e-13 / 1.1e-14 | 3.1e-12 / 1.5e-14 | 1.6e-10 / 1.3e-14 | 9.6e-09 / 1.9e-14 | 4.3e-07 / 1.8e-14 |
| 5 | 0.1 | 3.4e-13 / 4.4e-15 | 2.9e-09 / 3.3e-15 | 2.9e-06 / 4.0e-15 | 1.3e-03 / 6.3e-15 | 3.0e-01 / 6.6e-15 |
| 5 | 0.3 | 3.0e-13 / 4.3e-15 | 6.2e-10 / 5.4e-15 | 3.9e-07 / 5.4e-15 | 1.1e-04 / 6.8e-15 | 2.0e-02 / 8.8e-15 |
| 5 | 0.6 | 5.1e-14 / 1.8e-15 | 8.6e-12 / 2.6e-15 | 1.0e-09 / 3.8e-15 | 9.0e-08 / 5.5e-15 | 6.4e-06 / 7.0e-15 |
| 5 | 1.0 | 3.7e-15 / 4.5e-15 | 4.0e-14 / 5.3e-15 | 1.0e-12 / 6.8e-15 | 2.0e-11 / 8.3e-15 | 3.3e-10 / 9.1e-15 |
| 6 | 0.1 | 1.7e-13 / 7.8e-15 | 1.4e-08 / 3.1e-15 | 8.9e-05 / 4.8e-15 | 1.5e-01 / 4.9e-15 | 9.9e+01 / 7.8e-15 |
| 6 | 0.3 | 1.0e-13 / 8.8e-15 | 3.3e-08 / 5.4e-15 | 2.8e-04 / 8.6e-15 | 2.6e-01 / 1.1e-14 | 2.8e+02 / 1.7e-14 |
| 6 | 0.6 | 8.6e-14 / 1.7e-15 | 4.8e-11 / 3.4e-15 | 1.4e-08 / 4.6e-15 | 2.3e-06 / 6.2e-15 | 2.7e-04 / 8.4e-15 |
| 6 | 1.0 | 1.3e-14 / 5.6e-15 | 1.5e-12 / 6.5e-15 | 6.4e-11 / 7.9e-15 | 1.9e-09 / 8.3e-15 | 4.6e-08 / 9.6e-15 |

### Unitarity of the truncated block

Max | ||U e_n||^2 - ||U_ref e_n||^2 | over all columns. The reference norm carries the
exact leakage above the cutoff. In brackets: max | ||U e_n||^2 - 1 | over the columns
whose exact leakage is < 1e-15.

| cutoff | 20 | 40 | 60 | 80 | 100 |
|---|---|---|---|---|---|
| before | 9.1e-14 (1.6e-15) | 3.9e-9 (6.4e-15) | 1.4e-5 (1.1e-14) | 7.2e-2 (1.6e-14) | 2.1e5 (2.2e-14) |
| after | 2.2e-14 (6.7e-16) | 2.9e-14 (4.2e-15) | 3.0e-14 (1.5e-14) | 3.7e-14 (1.8e-14) | 4.7e-14 (3.4e-14) |

### Squeezing alone at large size

r = 1 with phases (0.3, 1.1), rows < 100, columns < L, against an 80-digit reference:

| size L | 150 | 200 | 300 | 400 | 600 |
|---|---|---|---|---|---|
| before | 3.3e-11 | 1.5e-8 | 5.8e-6 | 7.2e-5 | 7.2e-5 |
| after | - | - | - | - | 1.9e-14 |

S(-1) S(1) = 1 on the k x k block at size 800 (S(1)|n> reaches about e^2 n photons, so
this block has no truncation error):

| k | 40 | 50 | 60 | 70 |
|---|---|---|---|---|
| before | 1.5e-12 | 2.2e-11 | 2.8e-10 | 1.1e-8 |
| after | 1.3e-14 | 2.0e-14 | 2.0e-14 | 2.0e-14 |

### Timings

Minimum of 15 calls, one thread, ms. The machine was shared with other jobs, so treat
these figures as orders of magnitude.

| abs(alpha) | r | cutoff | before | after |
|---|---|---|---|---|
| 1 | 0.3 | 20 | 0.02 | 0.13 |
| 4 | 0.3 | 40 | 0.04 | 0.70 |
| 4 | 0.3 | 80 | 0.14 | 2.4 |
| 6 | 0.3 | 100 | 0.20 | 5.5 |
| 6 | 1.0 | 100 | 0.20 | 13 |
| 0 | 0.5 | 100 | 0.21 | 0.17 |
| 2 | 0.3 | 400 | 4.6 | 175 |

The cost with both squeezing and displacement is O(cutoff^2 L). It now pays for
accuracy that the old path did not have: in the slower rows above, the old results were
wrong by up to 1e2. Gates without a displacement (squeezing, quadratic phase) and pure
displacements cost about the same as before. In the backends, the combined case arises
only for `GaussianInput`.

## Tests

- `tests/native/test_native_fock_gaussian_unitary_stability.py` builds 60-digit
  references in the test (about 15 s) and asserts atol 1e-12 for the full matrix and
  3e-14 for column 0. The cases are |alpha| = 4, r = 0.3, cutoff 80 (old: 8.0e-5);
  |alpha| = 6, r = 0.3, cutoff 100 (old: 2.1e2); |alpha| = 3, r = 1, cutoff 100
  (old: 6.7e-9); and |alpha| = 0.5, cutoff 40, where the old path was accurate. The test
  also checks column norms against the exact leakage, squeezing rows at size 400
  (old: 3.3e-10), and the thermal columns of a displaced squeezed input (old: 6.7e-9).
  On the old build, 5 of its 6 tests fail.
- `cpp/tests/test_fock.cpp::test_single_mode_stability` checks a U = U (T a + R a^dag +
  delta) at |alpha| = 6, r = 0.3, cutoff 100 (< 1e-12), the normalisation of leak-free
  columns, and S(-1) S(1) = 1 on 70 x 70 at size 800.

## Remaining limits

- Validated range: |alpha| <= 6, |r| <= 1, cutoff <= 100, and size <= 800 for squeezing
  alone. Outside this range the method and the truncation bound still hold, but the
  rounding of the two recurrences has not been measured. In particular, the Laguerre
  recurrence has been checked only up to |beta| = 9.4 at size 541 (column-norm error
  4e-14).
- **Multi-mode (k >= 2)** unitaries still use the Bargmann recurrence. With a
  displacement it measured 3e-12 at |delta| <= 4, r <= 0.6, cutoff <= 40 (two modes,
  against a product reference). This error is not bounded in general and grows with
  cutoff and squeezing, as on one mode. The backends never request a multi-mode
  displaced unitary (CZ, beam splitter and Gaussian transforms have d = 0), but a direct
  `fock.gaussian_unitary` call can.
- Mixed `GaussianInput` with very large thermal occupation asks for a unitary of size up
  to 4096. With a displacement this now costs O(size^2 L), which takes minutes at the
  largest sizes. The old path was fast there but inaccurate.
