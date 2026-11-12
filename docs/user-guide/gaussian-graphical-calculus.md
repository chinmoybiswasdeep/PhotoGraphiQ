# Gaussian graphical calculus

`photographiq.calculus.gaussian_graph` implements the graphical calculus for
Gaussian pure states of N. C. Menicucci, S. T. Flammia and P. van Loock,
"Graphical calculus for Gaussian pure states", Phys. Rev. A 83, 042335 (2011),
arXiv:1007.0725. This page calls the paper MFvL. Equation numbers refer to
arXiv v3.

A pure Gaussian state with zero mean is a complex symmetric matrix
`Z = V + iU` with `U > 0`. Read `Z` as the adjacency matrix of an undirected
graph with complex edge weights and self-loops. Gaussian unitaries and ideal
quadrature homodyne measurements act on this matrix as graph rewrite rules. The
module tracks the displacement exactly in a separate mean vector. Every rule is
checked against the covariance algebra of `photographiq.gaussian` and against
the conditioning in `GaussianBackend`.

This is not a CV-ZX calculus and has no link to the qubit ZX subset in
`photographiq.calculus.zx`. Every example on this page was run in the project
environment. The printed output is shown below each example.

## Conventions

The repository conventions are:

| Item | Value |
|---|---|
| Commutator | `hbar = 2`, `[q, p] = 2i`, `p = -2i d/dq` |
| Ordering | interleaved `(q0, p0, q1, p1, ...)` |
| Vacuum covariance | `I` |
| Coherent mean | `(2 Re a, 2 Im a)` |
| CZ(g) | `p_u += g q_v`, `p_v += g q_u` (`gaussian.cz`) |
| p-squeezed resource | covariance `diag(e^{2r}, e^{-2r})` (`GaussianBackend.prepare(node, r)`) |
| Gate squeezing | `gaussian.squeezing(r) = diag(e^{-r}, e^{r})` |
| Rotation | `gaussian.rotation(t) = [[c, -s], [s, c]]` |

MFvL uses `hbar = 1` with vacuum covariance `I/2` (Eq. 2.15). It writes
`x = (q; p)` in block order.

## The Z–V correspondence (derived)

**Wave function.** MFvL Eqs. 2.21 and 2.22 give the state as
`psi(x) ∝ exp((i/2) x^T Z x)` in `hbar = 1` units. Our quadratures are
`q = sqrt(2) x`, so in our units:

    psi(q) ∝ exp((i/4) q^T Z q),        Z = V + iU,  U = Im Z > 0.

`Z` is the same matrix in both conventions. Only the prefactor in the exponent
changes. The vacuum is `Z = iI`: `psi ∝ exp(-q^2/4)` and
`|psi|^2 ∝ exp(-q^2/2)`, so `Var q = 1`.

**Nullifiers.** With `p = -2i d/dq`, `p psi = (Z q) psi`. So `(p - Z q) psi = 0`
(MFvL Eq. 2.35). The `hbar` factors cancel.

**Covariance.** `|psi|^2 ∝ exp(-q^T U q / 2)`, so `cov_qq = U^{-1}`. From
`p psi = Z q psi`, the symmetrized moments are `<{q, p^T}>/2 = U^{-1} V` and
`cov_pp = U + V U^{-1} V`. In qq/pp blocks:

    cov = [[U^{-1},     U^{-1} V        ],
           [V U^{-1},   U + V U^{-1} V ]]          (hbar = 2, vacuum I)

This is exactly `2 x` MFvL Eq. 2.19. Equivalently, `cov = S_Z S_Z^T` with
`S_Z = [[U^{-1/2}, 0], [V U^{-1/2}, U^{1/2}]]` (MFvL Eq. 2.18). The state is
pure: `(Omega cov)^2 = -I` holds, and the test suite checks it.

**Inverse.** MFvL Eqs. 2.25 and 2.26, times 2, give `U = (cov_qq)^{-1}` and
`V = cov_pq U`. A covariance matrix gives a graph only when rebuilding `cov`
from this `Z` reproduces it. That happens exactly when the state is pure.
`from_covariance` checks this to a relative max-norm tolerance (default
`rtol = 1e-9`). It raises an error for mixed and unphysical inputs.

**Resource and cluster states.** `prepare(node, r)` has `cov_qq = e^{2r}`, so
`U = e^{-2r}`. `Z = i e^{-2r}`, which is the same as MFvL. `CZ(g)` adds `g` to
`Z_uv`. The canonical approximate CV cluster state is therefore
`Z = A + i e^{-2r} I` (MFvL Eq. 2.29), unchanged by the change of units.
`Z -> A` as `r -> inf` (Eq. 2.30).

**Unitaries.** A Gaussian unitary with Heisenberg action `x -> S x` has a
symplectic matrix that does not depend on `hbar`, because `q` and `p` scale the
same way. In qq/pp blocks `S = [[A, B], [C, D]]`, it maps the nullifier row
`(-Z, I)` to `(-Z, I) S^{-1}`. The kernel of that row is the Lagrangian
subspace `col[I; Z]`, and it maps to `S col[I; Z] = col[I; Z'] (A + B Z)`.
So:

    Z' = (C + D Z)(A + B Z)^{-1}                       (MFvL Eq. 2.24)

The mean transforms as `mean -> S mean`.

## API

```python
from photographiq.calculus import GaussianGraph
```

| Member | Meaning |
|---|---|
| `GaussianGraph(Z, nodes=None, mean=None)` | Validated state `D(mean) |Z>`. `Z` must be complex symmetric with `Im Z` positive definite. Instances are immutable. |
| `.Z`, `.nodes`, `.mean` | Graph matrix (read-only), labels (row order), interleaved `hbar = 2` mean |
| `vacuum(nodes)` | `Z = iI` |
| `squeezed(nodes, r)` | `Z = i e^{-2r} I`, the backend's p-squeezed resource |
| `cluster(adjacency, r, nodes=None)` | `Z = A + i e^{-2r} I` (Eq. 2.29). `A` must be real symmetric with a zero diagonal. |
| `from_covariance(cov, nodes=None, mean=None, rtol=1e-9)`, `from_state(GaussianState)` | Graph of a pure state. Raises `ValueError` if the state is mixed. |
| `to_covariance()`, `to_state()` | Covariance (interleaved) and `GaussianState` |
| `edges(atol=0)` | `{(u, v): Z_uv}` for `u <= v`. Diagonal entries are self-loops. |
| `apply(S, nodes)` | Any interleaved symplectic on `nodes`, by Eq. 2.24. Raises if `S` is not symplectic. |
| `cz(u, v, g)` | `Z_uv += g` (Eq. 3.23) |
| `quadratic_phase(node, s)` | Shear `p += s q`: `Z_kk += s` (Eqs. 3.13–3.14) |
| `rotate(node, t)` | `gaussian.rotation(t)`, which is MFvL `S_phase(-t)` (Eqs. 3.17–3.18) |
| `squeeze(node, r)` | `gaussian.squeezing(r)`, which is MFvL `S_squeeze(-r)`: `Z_kk e^{2r}`, edges at `k` times `e^r` (Eqs. 3.15–3.16) |
| `beamsplitter(u, v, t)` | `gaussian.beamsplitter(t)`, which is MFvL `S_BS(t)` (Eqs. 3.10–3.12) |
| `measure_q(node, m)` | Ideal q homodyne: delete the vertex (Sec. III.D–E) |
| `measure_p(node, m)` | Ideal p homodyne: Schur complement (Sec. III.D–E, Eq. 3.20 followed by deletion) |
| `tensor(other)` | Direct sum `Z ⊕ Z'` (disjoint union). Labels must be disjoint. |
| `relabel(mapping)`, `reorder(nodes)` | Rename nodes, or permute them to `P Z P^T` (same state) |

The convenience gates use the same matrices as `GaussianBackend`, so a backend
gate sequence and the matching graph sequence describe the same state.

## Measurement rules (derived)

Split the modes into the measured node `k` and the rest `r`. Write
`z = Z_rk`. Start with the zero-mean graph state.

**q homodyne, outcome m** (MFvL Sec. III.D: "deleting its jth row and column").
Projecting onto `<q_k = m|` evaluates `psi` at `q_k = m`:

    psi'(q_r) ∝ exp((i/4) q_r^T Z_rr q_r + (i/2) m z^T q_r)

So `Z' = Z_rr`, which is vertex deletion and does not depend on the outcome.
The outcome only appears in the linear term `b = m z`.

**p homodyne, outcome m** (MFvL Sec. III.D–E: inverse Fourier transform, then
q measurement). With `<p = m|q> = e^{-i m q / 2}` in `hbar = 2` units, the
Gaussian integral over `q_k` converges because `Im Z_kk > 0`:

    integral dq_k e^{-i m q_k/2} psi = exp((i/4) q_r^T (Z_rr - z z^T / Z_kk) q_r + (i/2) (m z / Z_kk)^T q_r)

So `Z' = Z_rr - z z^T / Z_kk` is the Schur complement. `Z_kk` is never zero.
This matches the MFvL figure for measuring `p` (`W - t^{-1} r r^T`) and the
Fourier rule, Eq. 3.20.

**Displacement (tracked, not out of scope).** A conditional wave function
`exp((i/4) q^T Z' q + (i/2) b^T q)` has mean `q = -U'^{-1} Im b`. From
`p psi = (Z' q + b) psi` the momentum mean is `p = Re Z' q + Re b`. For a state
with prior mean `mu`, write `D(mu) = D(mu_r) ⊗ D(mu_k)`. `D(mu_k)` only shifts
the outcome to `m - mu_{q_k}` (or `m - mu_{p_k}`) and adds a phase. `D(mu_r)`
adds `mu_r`. So the module returns the exact conditional mean
`mu_r + (q, p)(m - mu_k)`, which matches the backend's Kalman update.

**Other quadratures.** A homodyne that records `q cos t + p sin t` is
`g.rotate(node, -t).measure_q(node, m)`, because `rotation(-t)` maps
`q -> q cos t + p sin t`. The tests compare it with
`GaussianBackend.measure(node, Homodyne(), t)`.

## Examples

### Cluster state, covariance and round trip

```python
import numpy as np
from photographiq.calculus import GaussianGraph

g = GaussianGraph.cluster([[0, 1, 0], [1, 0, 1], [0, 1, 0]], r=1.0, nodes="abc")
print(np.round(g.Z, 4))
print(np.round(g.to_covariance()[:2, :2], 4))
print(GaussianGraph.from_covariance(g.to_covariance(), g.nodes).edges(atol=1e-12))
```

```text
[[0.+0.1353j 1.+0.j     0.+0.j    ]
 [1.+0.j     0.+0.1353j 1.+0.j    ]
 [0.+0.j     1.+0.j     0.+0.1353j]]
[[7.3891 0.    ]
 [0.     7.5244]]
{('a', 'a'): 0.1353352832366127j, ('a', 'b'): (1+0j), ('b', 'b'): 0.1353352832366127j, ('b', 'c'): (1+0j), ('c', 'c'): 0.1353352832366127j}
```

Node `a` has `Var q = e^{2}` and `Var p = e^{-2} + e^{2}` (`U + V U^{-1} V`).
The `e^{2}` term is the antisqueezed `q` noise of its single neighbour, which CZ
copies into `p_a`.

### Unitaries against the Gaussian backend

```python
import numpy as np
from photographiq.backends.gaussian import GaussianBackend
from photographiq.calculus import GaussianGraph

g = GaussianGraph.cluster([[0, 1], [1, 0]], r=0.5, nodes="ab")
h = g.squeeze("a", 0.3).rotate("b", 0.4).beamsplitter("a", "b", 0.2).cz("a", "b", -0.7)
backend = GaussianBackend()
for n in "ab":
    backend.prepare(n, 0.5)
backend.entangle("a", "b", 1.0)
backend.squeeze("a", 0.3)
backend.rotate("b", 0.4)
backend.beamsplitter("a", "b", 0.2)
backend.entangle("a", "b", -0.7)
print(np.round(h.Z, 4))
print(np.abs(h.to_covariance() - backend.get_state().covariance).max())
```

```text
[[0.1794+0.6863j 0.6949+0.2759j]
 [0.6949+0.2759j 0.9298+0.5244j]]
7.771561172376096e-16
```

### Homodyne rules with outcome-dependent means

```python
import numpy as np
from photographiq.backends.gaussian import GaussianBackend
from photographiq.calculus import GaussianGraph
from photographiq.measurements import Homodyne

g = GaussianGraph.cluster([[0, 1, 0], [1, 0, 1], [0, 1, 0]], r=1.0, nodes="abc")
q = g.measure_q("b", 0.5)  # vertex deletion
p = g.measure_p("b", 0.5)  # Schur complement
print(q.edges(atol=1e-12))
print({k: complex(np.round(v, 4)) for k, v in p.edges(atol=1e-12).items()})
print(np.round(p.mean, 4))
backend = GaussianBackend()
backend.set_state(g.to_state())
backend.measure("b", Homodyne(), np.pi / 2, outcome=0.5)
s = backend.get_state()
print(np.abs(p.to_covariance() - s.covariance).max(), np.abs(p.mean - s.mean).max())
```

```text
{('a', 'a'): 0.1353352832366127j, ('c', 'c'): 0.1353352832366127j}
{('a', 'a'): 7.5244j, ('a', 'c'): 7.3891j, ('c', 'c'): 7.5244j}
[0.2477 0.     0.2477 0.    ]
6.217248937900877e-15 2.498001805406602e-16
```

Measuring `p` on the middle of a 3-node wire creates the imaginary edge
`i e^{2r}`. This edge grows without bound as `r -> inf`. In the ideal limit the
neighbours are left in a `q_a + q_c = m` eigenstate, which is not an ideal
cluster graph. This is the MFvL remark that phase shifts can make the result
non-representable in the ideal formalism.

### Ideal limit: wire shortening

Measuring `p` on both inner nodes of a 4-node unit-weight wire gives exactly
`Z' = -X/(1+e^2) + i e (2+e^2)/(1+e^2) I` with `e = e^{-2r}`. As `r -> inf`,
this tends to a single edge of weight -1 between the end nodes:

```python
import numpy as np
from photographiq.calculus import GaussianGraph

wire = np.diag([1.0, 1.0, 1.0], 1)
wire = wire + wire.T
for r in (1.0, 2.0, 4.0):
    z = GaussianGraph.cluster(wire, r).measure_p(1, 0.0).measure_p(2, 0.0).Z
    print(r, np.round(z, 6).tolist())
```

```text
1.0 [[0.268236j, (-0.982014+0j)], [(-0.982014+0j), 0.268236j]]
2.0 [[0.036625j, (-0.999665+0j)], [(-0.999665+0j), 0.036625j]]
4.0 [[0.000671j, (-1+0j)], [(-1+0j), 0.000671j]]
```

The finite-squeezing noise `Im Z'` is about `2 e^{-2r}`, twice that of one
resource mode.

### Composition and rejected inputs

```python
import numpy as np
from photographiq.calculus import GaussianGraph

a = GaussianGraph.squeezed(["x"], 0.5)
b = GaussianGraph.vacuum(["y"]).relabel({"y": "z"})
print(a.tensor(b).nodes, np.round(a.tensor(b).Z, 4).tolist())
for bad in (lambda: GaussianGraph.from_covariance(2 * np.eye(2)),
            lambda: GaussianGraph([[1.0]]),
            lambda: GaussianGraph.vacuum("a").apply(np.diag([2.0, 2.0]), ("a",))):
    try:
        bad()
    except ValueError as e:
        print(e)
```

```text
('x', 'z') [[0.3679j, 0j], [0j, 1j]]
covariance is not a pure Gaussian state (mixed states have no Gaussian graph): relative residual 7.50e-01 > 1e-09, det(V)^(-1/2) = 0.5
Im Z must be positive definite (normalizable pure state)
Transformation must be symplectic (S Omega S^T = Omega)
```

## Validation

The tests are in `tests/calculus/test_gaussian_graph.py`. Errors are max-norm
errors relative to `max(1, |reference|)`. The maxima below come from 200 random
seeds. The states are random products of squeezers (`|r| <= 0.8`), rotations,
beam splitters and CZ gates acting on the vacuum, with 1 to 5 modes and random
means.

| Check | Reference | Max error |
|---|---|---|
| `cov -> Z -> cov` | input covariance | 1.9e-15 |
| `Z -> cov -> Z` | input `Z` | 1.2e-15 |
| `cz`, `rotate`, `squeeze`, `beamsplitter`, `quadratic_phase` | `GaussianBackend` gates (covariance and mean) | 1.0e-15 to 3.2e-15 |
| `apply(S, nodes)` with random 2-mode `S` | `GaussianBackend.transform` | 1.8e-14 |
| `measure_q` | `GaussianBackend.measure(Homodyne(), 0)`, fixed outcome | 1.8e-15 (covariance and mean) |
| `measure_p` | `GaussianBackend.measure(Homodyne(), pi/2)` | 2.0e-15 covariance, 2.6e-15 mean |
| `rotate(-t)` then `measure_q` | `GaussianBackend.measure(Homodyne(), t)` | 2.6e-15 covariance, 4.4e-15 mean |

The test tolerance is `1e-10`. Each check is a few dense `O(N^3)` steps on
matrices with condition number below about `1e4`, so the expected error is
about `eps * cond`, roughly `1e-12`.

The suite also checks:

- the MFvL closed-form local rules (Eqs. 3.14, 3.16, 3.18, 3.20, 3.23, 3.12) against the generic Eq. 2.24;
- the cluster state against backend `prepare` plus `entangle`;
- `(Omega cov)^2 = -I`;
- `Z_r -> A`;
- the wire-shortening limit.

The wire-shortening tolerance is `64 eps e^{2r}`, not `eps`. The intermediate
graph has `Im Z ~ e^{2r}`, and the second Schur step cancels it down to about
`2 e^{-2r}`. At `r = 8` the absolute error is `1.6e-9`, so the tiny imaginary
(noise) part is only accurate to about 1%. Sequential `p` measurements through
very strongly squeezed intermediate states lose relative accuracy in `Im Z`.

The negative tests reject:

- a thermal state, a lossy state and a reduced (mixed) state;
- `Im Z = 0`, `Im Z` indefinite, and a Hermitian `Z` that is not symmetric;
- non-symplectic and complex gate matrices, and repeated gate modes;
- tensor products with overlapping labels;
- non-finite outcomes.

## Not implemented

- **Mixed states.** Lossy, thermal or reduced states have no `Z` and are
  rejected. Use `GaussianBackend` for them. MFvL Appendix D only sketches a
  generalization.
- **Non-ideal measurements.** Finite-efficiency or noisy homodyne, heterodyne
  and general-dyne measurements are not implemented, because they produce mixed
  states.
- **Non-Gaussian elements.** Photon counting, cubic phase, GKP and cat states
  are not implemented.
- **General CV-ZX or graphical symplectic algebra.** For example, the
  Booth–Carette–Comfort calculus is not implemented, and there is no
  translation to or from the qubit ZX diagrams in `photographiq.calculus.zx`.
- **Other MFvL constructions.** The MFvL H-graph constructions (Eqs.
  2.32–2.34), the closest-cluster-state optimization (Sec. IV.A, Eqs. 4.1–4.3)
  and the entanglement measure are not implemented.
- **Ideal (`r = inf`) graphs.** These are rejected because `Im Z` must be
  positive definite. Ideal limits are only studied as `r -> inf` sequences.
