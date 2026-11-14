# Compilation derivations

How `Circuit.compile` / `compile_circuit` turns gates into teleportation steps, and how the
Gaussian channel of a pattern is extracted. Step-level derivations are in
[CV cluster states](cluster-states.md); nonlinear (cubic, Kerr, quartic) synthesis is in
[nonlinear synthesis](nonlinear-synthesis.md).

## Single-mode Gaussian gates as shear sequences

Every one-mode symplectic $S=\begin{pmatrix}A&B\\C&D\end{pmatrix}$, $\det S=1$, with $C\ne0$
factorizes into four teleportation steps with shears $k$ listed chronologically
$[0,\ (1-D)/C,\ C,\ (1-A)/C]$:

$$S=T(k_4)\,T(k_3)\,T(k_2)\,T(k_1),\qquad T(k)=\begin{pmatrix}-k&-1\\1&0\end{pmatrix}.$$

When $C$ is near zero the compiler prepends $T(0)$ and decomposes $S\,T(0)^{-1}$ instead, giving
five steps. It recomputes the product and checks it against $S$, and it raises for
near-singular requests rather than returning an unstable decomposition. Each step consumes
one resource node and adds noise $N_r$, so a compiled gate is the channel
$(S, N_{\rm total})$, never the ideal unitary.

Circuit rotation angles and squeezing values must be numeric at compile time. Symbolic
angles, squeezing and couplings remain available in hand-built patterns and wires.

## Two-mode gates

- **Logical CZ**: a resource edge between the two current wire nodes, followed by four-step
  identity transport on each wire.
- **SUM**: a CZ conjugated by Fourier transforms on the target.
- **Beam splitter**: with $U$ and $L$ the two SUM directions, the $q$-coordinate rotation by
  $\phi$ is $U(-\tan\tfrac\phi2)\,L(\sin\phi)\,U(-\tan\tfrac\phi2)$. Two half-angle
  decompositions avoid the singularity at $\phi=\pi$.

The compiler emits only preparation, entanglement, measurement and displacement-correction
commands; no direct rotation or squeezing gate is inserted as a shortcut. This is
deliberately resource-expensive and faithful to the MBQC model. The `CompilationTrace`
returned with `return_trace=True` maps every source gate to its resource nodes, measurement
commands and correction commands.

## Extracting the Gaussian channel of a pattern

`gaussian_channel(pattern)` propagates affine Wigner variables through a fixed-angle homodyne
pattern, independently of the conditional simulator, and returns $(S,N,d)$ with

$$\mu_{\rm out}=S\mu_{\rm in}+d,\qquad V_{\rm out}=SV_{\rm in}S^{T}+N .$$

It rejects general-dyne measurements, nonlinear feed-forward, outcome-dependent gates and
Fock resources; trajectories (`simulate`) still support all of those.

```python
import numpy as np
import photographiq as pg

circuit = pg.Circuit(1).rotate(0, 0.3).squeeze(0, 0.15)
for r in (0.8, 2.0):
    ch = pg.gaussian_channel(circuit.compile(squeezing=r))
    print(r, np.round(ch.matrix, 6).tolist(), round(float(np.trace(ch.noise)), 6))
# 0.8 [[0.822266, -0.254357], [0.343345, 1.109943]] 2.035714
# 2.0 [[0.822266, -0.254357], [0.343345, 1.109943]] 0.184676
# The map S does not depend on r; Tr N falls by e^{-2(2.0-0.8)} = 1/11.02.
```

## Pattern rewriting

`Pattern.standardize` only moves preparations and entanglers past disjoint commands whose
classical inputs are already available, merges adjacent displacements and removes numeric
zero shifts. It makes no full normal-form claim; the qubit standardization of the
measurement calculus is a separate tool ([calculus](calculus.md)).

## Beyond Gaussian gates

The universal target is Gaussian plus cubic phase. Cubic gates are applied by injecting a
cubic resource with nonlinear feed-forward; Kerr and quartic gates are approximated by
product formulas with explicit error budgets ([nonlinear synthesis](nonlinear-synthesis.md),
[universal compilation guide](../user-guide/universal-compilation.md)). A universal gate set
does not imply exact finite-resource gates or arbitrary-unitary synthesis.
