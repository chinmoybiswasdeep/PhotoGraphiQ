# Mathematical model

## Conventions

PhotoGraphiQ conventions throughout: $\hbar = 2$, $[q_j, p_k] = 2i\delta_{jk}$, interleaved
quadratures $(q_1, p_1, \dots, q_n, p_n)$, vacuum covariance $I$.

- Gaussian unitary: $d \to S d + t$, $V \to S V S^T$ with $S$ symplectic.
- Gaussian channel: $d \to X d + t$, $V \to X V X^T + Y$.
- Mean photon number of a Gaussian state: $(\operatorname{tr} V + |d|^2)/4 - n/2$.

## Encoding

`CVFeatureMap` maps one feature to one mode and gives a product state. For feature value
$v = \text{scale}\cdot x_j$ and an optional fixed pre-squeezing $s_0$ (`input_squeezing`):

| Encoding | Mean of mode j | Covariance of mode j |
| --- | --- | --- |
| `displacement` | $q_j = v$ (or $p_j = v$ with `quadrature="p"`) | $S(s_0) S(s_0)^T$ |
| `squeezing` | 0 | $S(v) S(s_0) (S(v) S(s_0))^T$, with $S(r) = \operatorname{diag}(e^{-r}, e^{r})$ and $\lvert v\rvert \le$ `max_squeezing` |
| `rotation` | $R(v)\,(\text{offset}, 0)^T$ | $R(v) S(s_0) (R(v) S(s_0))^T$ |

For coherent displacement encoding the amplitude is $\alpha = v/2$.

## Gate matrices (per mode or mode pair)

$R(\phi) = \begin{pmatrix}\cos\phi & -\sin\phi\\ \sin\phi & \cos\phi\end{pmatrix}$,
$S(r) = \operatorname{diag}(e^{-r}, e^{r})$, shear $P(s) = \begin{pmatrix}1&0\\ s&1\end{pmatrix}$
($p \to p + s q$, the quadratic phase $e^{i s q^2/4}$).
Beam splitter $B(\theta)$ mixes $(q_u, q_v)$ and $(p_u, p_v)$ by the rotation $R(\theta)$.
$CZ(g)$: $p_u \to p_u + g q_v$, $p_v \to p_v + g q_u$. A displacement adds $(q, p)$ to $t$.

## Path A: ideal circuit and compiled channel

The ideal map is the ordered product of gate matrices, with $Y = 0$.

The physical map mirrors `photographiq.compile_circuit`. Each teleportation step $T(k)$
acts as $\begin{pmatrix}-k & -1\\ 1 & 0\end{pmatrix}$ on the mode and adds
$\operatorname{diag}(0, e^{-2r})$ to $Y$ (`photographiq.gaussian.wire_channel`).
Single-mode gates decompose into four shears, or five when $|S_{10}| < 10^{-8}$ (the
compiler's `decompose_symplectic` branch). A CZ is followed by four Fourier steps on each
mode. A displacement uses four Fourier steps (the identity) plus a shift. A beam splitter
is two half rotations, each made of three SUM gates, and each SUM is
$R(\pi/2)\,CZ\,R(-\pi/2)$ on the target. The result is $X = S$ exactly, the same $t$, and
$Y = e^{-2r} Y_1$ with $Y_1$ fixed by the program and its parameters
([compilation](compilation.md)).

## Path B: weighted cluster channel

The CVCluster nodes are grid positions $(w, c)$. Column 0 holds the inputs and the last
column the outputs. Every other node is a p-squeezed resource with variances
$(e^{2r}, e^{-2r})$. Each quadrature is written as a linear form in all initial
quadratures:

1. CZ network: $p_i \mathrel{+}= \sum_j A_{ij} q_j$ (weights 1 along wires, $g$ on rungs).
2. In measurement order (column-major, outputs excluded), node $i$ is measured along
   $q\cos\theta_i + p\sin\theta_i$. The scaled record $m_i/\sin\theta_i$ drives the
   CV-flow correction: each unmeasured node $t$ gets $q_t \mathrel{-}= c_t\, m_i/\sin\theta_i$
   and $p_t \mathrel{-}= (A c)_t\, m_i/\sin\theta_i$, where $c$ comes from the
   Booth–Markham certificate.
3. The output rows restricted to input columns give $X$. The coefficients on the
   resource p quadratures, $L_p$, give $Y = e^{-2r} L_p L_p^T$. If any anti-squeezed
   resource q quadrature reaches the output (above $10^{-9}$), the channel raises
   `RuntimeError`.

Each angle sets the teleportation shear $k_i = \cot\theta_i$. Therefore $\theta$ is
$\pi$-periodic, and $\theta = 0 \bmod \pi$ (a q measurement) is singular. The
displacement $t$ is zero.

## Flow-corrected means

For both paths the unconditional output mean is $X d + t$, which does not depend on $r$.
Finite squeezing changes only the covariance, $X V X^T + Y(r)$. Mean-readout models with
exact moments therefore predict identically at every $r$
([finite squeezing](finite-squeezing.md)).

## Readouts

For readout modes $M$, with output means $\mu$ and covariance $C$:

| `readout` | Features |
| --- | --- |
| `q` | $\mu_{q_m}$, $m \in M$ |
| `p` | $\mu_{p_m}$ |
| `qp` | both; with shots, $S$ is split as $\lfloor S/2\rfloor$ per setting |
| `moments` | $\mu_{q_m}$ and $C_{q_m q_m}$ (exact moments only) |

Shot readout: the sample mean of $S$ homodyne outcomes is drawn exactly as
$\mu + \operatorname{chol}(C_{\text{setting}}/S)\,z$, with fixed standard normals $z$
(common random numbers).

Homodyne head (binary): $P(y=1) = \Phi\big((\mu_{q} - b)/\sqrt{C_{qq}}\big)$ for the first
readout mode. This is the Born probability that one homodyne outcome exceeds the trainable
threshold $b$.

## Kernels

$K(x,x') = \operatorname{Tr}[\rho(x)\rho(x')]$ (overlap), the same value divided by
$\sqrt{\operatorname{Tr}\rho^2\,\operatorname{Tr}\rho'^2}$ (normalized), or the squared
Uhlmann fidelity (fidelity). For coherent displacement encoding,
$K = \exp(-\text{scale}^2 |x - x'|^2/4)$ ([kernels](kernels.md)).

## Non-Gaussian gates

Kerr $e^{i\kappa n^2}$ and cubic phase $e^{i\gamma q^3/6}$ with $q = a + a^\dagger$. These
run on a truncated Fock space with an exclusive total-photon cutoff
([non-Gaussian](non-gaussian.md)).
