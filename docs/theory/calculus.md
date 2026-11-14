# Measurement calculus, local Cliffords, ZX and the Gaussian graphical calculus

PhotoGraphiQ implements the standard **qubit** MBQC calculi and one **CV** graphical calculus
(Gaussian pure states). There is no general CV-ZX calculus and no CV analogue of local
complementation; requests for them raise. Code: `src/photographiq/calculus/`.

| Tool | Model | Module | Scope |
|---|---|---|---|
| Measurement calculus (N, E, M, X, Z, S) | qubit | `calculus.qubit` | patterns, standardization, signal shifting, exact branch simulation (≤ 20 live qubits) |
| Local Cliffords, local complementation, LC equivalence | qubit graph states | `calculus.clifford` | exact |
| ZX diagrams | qubit | `calculus.zx` | tensor semantics with exact scalars; a sound rewrite subset |
| Gaussian graphical calculus | CV, pure Gaussian | `calculus.gaussian_graph` | Gaussian unitaries and ideal $q$/$p$ homodyne |
| CV command view, ideal-limit check | CV patterns | `calculus.cv` | affine correction extraction; $S$ vs $N$ separation |

## Qubit measurement calculus

Following Danos, Kashefi and Panangaden (J. ACM 54, 8, 2007), with commands listed in
execution order:

- `N(i)` prepares $|+\rangle$; `E(i, j)` applies $\mathrm{CZ}$;
- `M(i, plane, angle, s_domain, t_domain)` measures the observable $O(\text{plane},\alpha)$
  (XY: $\cos\alpha X+\sin\alpha Y$; YZ: $\cos\alpha Z+\sin\alpha Y$; XZ: $\cos\alpha Z+\sin\alpha X$),
  outcome 0 being the $+1$ eigenvalue. With signals $s,t$ (parities of the domains) the
  effective observable is $Z^{t}X^{s}OX^{s}Z^{t}$;
- `X(i, domain)`, `Z(i, domain)` are conditional Pauli corrections; `S(i, domain)` is the
  classical signal command.

Pauli conjugation keeps the plane and changes the angle (XY: $X:\alpha\to-\alpha$,
$Z:\alpha\to\alpha+\pi$), so the absorption rules $M_i^{s,t}X_i^{d}=M_i^{s\oplus d,t}$ and
$M_i^{s,t}Z_i^{d}=M_i^{s,t\oplus d}$ are exact. `standardize` brings a pattern to
$N\,E\,M\,C$ order, `signal_shift` moves the outcome-flipping part of each measurement domain
into signals (XY: the $t$-domain; YZ and XZ: the $s$-domain), `run_branch` / `enumerate_branches`
compute each branch's Kraus operator, and `is_deterministic` compares them.
`pattern_from_flow` builds the pattern of a certified causal flow, gflow or Pauli flow
([flow theory](flow.md)).

## Local Cliffords and local complementation

Local complementation at $v$ complements the edges among the neighbours of $v$. The graph
states satisfy $|G\star v\rangle=U_v|G\rangle$ with the local Clifford
$U_v=\sqrt{-iX_v}\prod_{w\sim v}\sqrt{iZ_w}$ (Van den Nest, Dehaene, De Moor 2004).
`lc_equivalence(G, H)` searches for a sequence of local complementations and returns it with a
numerical residual, or `None`.

## ZX calculus (qubit subset)

A Z spider with phase $\alpha$ and $n$ legs is $|0\cdots0\rangle\langle0\cdots0|+e^{i\alpha}|1\cdots1\rangle\langle1\cdots1|$;
an X spider is the same in the $|\pm\rangle$ basis. `Diagram.matrix()` evaluates the diagram
densely (exponential cost) with exact scalars. Implemented rewrites: spider fusion, self-loop
removal, identity removal, colour change, Pauli state copy, $2\times2$ bialgebra, Hopf and
parallel-Hadamard cancellation. Each rewrite checks its precondition, multiplies the scalar
by the exact factor and is recorded in `Diagram.trace`. Not implemented: $\pi$-copy, local
complementation/pivoting as ZX rules, and any CV-ZX identities.

```python
import networkx as nx
import numpy as np
from photographiq.calculus import lc_equivalence, zx

print(lc_equivalence(nx.star_graph(3), nx.complete_graph(4)).sequence)   # (0,)

d = zx.Diagram()
a, b = d.add_vertex("Z", 0.4), d.add_vertex("Z", 0.7)
d.add_edge(d.add_input(), a); d.add_edge(a, b); d.add_edge(b, d.add_output())
fused = zx.fuse(d, a, b)
print(np.allclose(fused.matrix(), d.matrix()), round(fused.phase(a), 6))   # True 1.1
```

## Gaussian-pure-state graphical calculus

Menicucci, Flammia and van Loock (PRA 83, 042335, 2011) represent a zero-mean $N$-mode
Gaussian pure state by a complex symmetric matrix $Z=V+iU$ with $U>0$. In PhotoGraphiQ
conventions the position wavefunction is $\psi(q)\propto\exp(\tfrac{i}{4}q^{T}Zq)$ and

$$\mathrm{cov}=\begin{pmatrix}U^{-1}&U^{-1}V\\ VU^{-1}&U+VU^{-1}V\end{pmatrix}\quad(qq/pp\text{ blocks}).$$

Vacuum is $Z=iI$ and the $p$-squeezed CV cluster state is $Z=A+ie^{-2r}I$, where $r$ is the
resource squeezing. A Gaussian unitary with symplectic blocks $\begin{pmatrix}A&B\\C&D\end{pmatrix}$
acts as $Z\mapsto(C+DZ)(A+BZ)^{-1}$; ideal $q$/$p$ homodyne is a Schur complement on $Z$.
Displacements are tracked exactly in a separate mean vector. Mixed states, loss and
non-Gaussian elements are outside its scope.

The finite-squeezing wire-shortening rule shows the ideal/finite distinction directly:
measuring two interior nodes of a four-node wire in $p$ leaves a single edge of weight
$\to-1$ only as $r\to\infty$.

```python
import numpy as np
from photographiq.calculus import GaussianGraph

wire = np.diag([1.0, 1.0, 1.0], 1)
wire = wire + wire.T
for r in (1.0, 4.0):
    Z = GaussianGraph.cluster(wire, r).measure_p(1, 0.0).measure_p(2, 0.0).Z
    print(r, np.round(Z, 6).tolist())
# 1.0 [[0.268236j, (-0.982014+0j)], [(-0.982014+0j), 0.268236j]]
# 4.0 [[0.000671j, (-1+0j)], [(-1+0j), 0.000671j]]
```

## CV command view and ideal-limit check

`calculus.cv.command_view(pattern)` classifies the commands of a CV pattern and extracts
displacement corrections as exact affine forms of outcomes (non-affine expressions are
reported, never approximated). `ideal_limit_check` evaluates the pattern's channel
$V\mapsto SV S^{T}+N$ at increasing squeezing and reports the deviation of $S$ from the ideal
map separately from $N$.

## References

V. Danos, E. Kashefi, P. Panangaden, J. ACM 54, 8 (2007); M. Van den Nest, J. Dehaene,
B. De Moor, PRA 69, 022316 (2004); B. Coecke and A. Kissinger, *Picturing Quantum Processes*
(2017); N. C. Menicucci, S. T. Flammia, P. van Loock, PRA 83, 042335 (2011). User guides:
[MBQC calculus](../user-guide/mbqc-calculus.md),
[Gaussian graphical calculus](../user-guide/gaussian-graphical-calculus.md).
