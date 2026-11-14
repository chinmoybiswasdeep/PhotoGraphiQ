# CV cluster states, teleportation and feed-forward

This page derives the resource states and the elementary measurement-based step that every
PhotoGraphiQ pattern is built from, and separates the **ideal** (infinite-squeezing)
relations from what a **finite-squeezing** simulation computes. Conventions are those of
[Gaussian states](gaussian.md): $\hbar=2$, $[q,p]=2i$, interleaved ordering, vacuum $V=I$.

## Weighted graphs and canonical CV cluster states

A `CVGraph` is a real symmetric adjacency matrix $A$ with zero diagonal (edge weights
$A_{ij}=g_{ij}$) plus per-node resource squeezing $r_i$, input and output labels. The
**canonical preparation** is:

1. prepare each node in a $p$-squeezed state, $V_i=\mathrm{diag}(e^{2r_i},e^{-2r_i})$;
2. apply $\mathrm{CZ}_{ij}(g_{ij})=\exp(i g_{ij}q_iq_j/2)$ for every edge.

Every CZ leaves $q$ fixed and sends $p\mapsto p+Aq$. Writing $D_q=\mathrm{diag}(e^{2r_i})$,
$D_p=\mathrm{diag}(e^{-2r_i})$ and using grouped order $(q;p)$ for readability, the resource
covariance is

$$V=\begin{pmatrix}D_q & D_qA\\ AD_q & D_p+AD_qA\end{pmatrix}.$$

The **nullifiers** $\delta=p-Aq$ commute with each other and have covariance
$\mathrm{Cov}(\delta)=D_p$. As $r\to\infty$ their variance goes to zero, which is the ideal
CV cluster state. That limit is not a normalizable state: the $q$ variances $e^{2r}$ diverge.
`CVGraph(..., ideal=True)` keeps the nullifier algebra (for flow analysis and symbolic
reasoning) but refuses to build a numerical pattern.

Input nodes replace their resource preparation with the supplied input, so the nullifier
covariance $D_p$ holds only on the resource part; it must not be applied to arbitrary inputs.

```python
import numpy as np
import photographiq as pg

r = 0.7
graph = pg.CVGraph.line(3, squeezing=r)
state = pg.simulate(pg.Pattern(graph), seed=0).state      # prepared resource, nothing measured
N = graph.nullifiers()                                    # rows: delta_i = p_i - sum_j A_ij q_j
print(np.allclose(N @ state.covariance @ N.T, np.exp(-2 * r) * np.eye(3)))   # True
```

Standard topologies (`line`, `ring`, `square`, `rectangular`, `star`, `tree`, `dual_rail`,
`temporal`) are constructors of the same object ([graphs guide](../user-guide/graphs.md)). The
Gaussian-pure-state graph $Z=A+ie^{-2r}I$ of Menicucci, Flammia and van Loock describes the
same resource; see [calculus](calculus.md#gaussian-pure-state-graphical-calculus).

## One teleportation step

Couple an input mode 0 to a $p$-squeezed ancilla 1 with a unit-weight CZ, measure
$m=p_0+k\,q_0$ on mode 0 and apply the correction $X_1(-m)$. Just before the measurement
$p_0'=p_0+q_1$ and $p_1'=p_1+q_0$, so after the correction

$$q_{\rm out}=-k\,q_{\rm in}-p_{\rm in},\qquad p_{\rm out}=q_{\rm in}+p_{\rm anc},\qquad
T(k)=\begin{pmatrix}-k&-1\\ 1&0\end{pmatrix}.$$

**Homodyne angle and feed-forward rescaling.** A homodyne detector measures the normalized
quadrature $q\cos\theta+p\sin\theta$. PhotoGraphiQ therefore measures at
$\theta=\operatorname{atan2}(1,k)$ and feeds forward $m=y/\sin\theta$. Omitting the rescaling
implements the wrong gate whenever $k\ne0$.

**Finite squeezing.** The ancilla contributes $p_{\rm anc}$ with variance $e^{-2r}$. Averaged
over outcomes (the corrected trajectories), one step is the Gaussian channel

$$V_{\rm out}=T(k)\,V_{\rm in}\,T(k)^{T}+N_r,\qquad N_r=\mathrm{diag}(0,e^{-2r}).$$

This is a statement about the unconditional channel. A single conditional trajectory has a
mean that depends on the outcome. An $L$-step wire accumulates noise recursively,
$N_j=T_jN_{j-1}T_j^{T}+N_{r_j}$; four $k=0$ steps give the identity map with isotropic
noise $2e^{-2r}I$:

```python
import numpy as np
import photographiq as pg
from photographiq.gaussian import teleportation_matrix

r, k = 0.8, 0.3
ch = pg.gaussian_channel(pg.protocols.wire([k], squeezing=r))
print(np.allclose(ch.matrix, teleportation_matrix(k)))            # S = T(k)
print(np.allclose(ch.noise, np.diag([0.0, np.exp(-2 * r)])))     # N = diag(0, e^{-2r})

ch4 = pg.gaussian_channel(pg.protocols.wire([0, 0, 0, 0], squeezing=r))
print(np.allclose(ch4.matrix, np.eye(2)), np.allclose(ch4.noise, 2 * np.exp(-2 * r) * np.eye(2)))
```

| Quantity | Ideal limit ($r\to\infty$) | Finite squeezing (what is simulated) |
|---|---|---|
| Resource state | nullifiers exactly zero; not normalizable | Gaussian state with $\mathrm{Cov}(\delta)=e^{-2r}I$ |
| One wire step | unitary $T(k)$ up to a known displacement | channel $(T(k), N_r)$; conditional mean depends on the outcome |
| CV flow certificate | guarantees deterministic, outcome-independent output | does **not** make the step unitary; the certificate states its regime |
| Error budget | none | physical noise $\propto e^{-2r}$; separate from any numerical truncation |

Larger squeezing lowers the noise but raises source energy and the condition number of the
covariance matrix. No numerical ideal limit is hidden behind a finite-$r$ simulation.

## Feed-forward

Feed-forward is classical: commands may depend on earlier outcomes through `Outcome`
expressions (`Displace(q=-Outcome(0))`, adaptive angles, nonlinear functions such as
`pg.sin`). `simulate` evaluates them on the sampled or postselected record; the dependency
graph (`pattern.dependencies()`) combines these classical producer→consumer
edges with the order of operations on each mode. A topological schedule respects the
dependencies but does not prove determinism; that is the job of [flow analysis](flow.md).

## Noise models available

| Source | Model | Where |
|---|---|---|
| Finite resource squeezing | added noise $N_r$ per step | `CVGraph(squeezing=...)`, every compiled pattern |
| Loss / thermal noise | $X=\sqrt\eta I$, $Y=(1-\eta)(2n_{\rm th}+1)I$ | `Loss` command; Fock loss on mixed-Fock backends |
| Detector inefficiency, readout noise | record variance $+(1-\eta)/\eta+\nu_{\rm add}$ | `Homodyne(efficiency=, noise=)` |
| Fock truncation | numerical, not physical | [non-Gaussian theory](non-gaussian.md#fock-truncation) |

## References

N. C. Menicucci et al., PRL 97, 110501 (2006); M. Gu et al., PRA 79, 062318 (2009);
N. C. Menicucci, PRA 83, 062314 (2011); full list in [references](../references.md).
