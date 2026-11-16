# Typed CV diagrams: introduction

`photographiq.calculus.cv_diagrams` is a typed **Gaussian process calculus** with hybrid
classical-quantum semantics. A diagram is a causally ordered list of generator boxes on
labelled wires. Wires carry quantum modes or classical real-valued records. Every Gaussian
diagram has an exact semantics as a unitary, as an outcome-averaged channel, and as a
conditional instrument. Rewrite rules change a diagram, and each rule states which of these
semantics it preserves.

The package has three layers:

| Layer | What it gives you | Pages |
|---|---|---|
| Gaussian | Typed generators, exact hybrid Gaussian semantics, rewrite rules with declared equivalence classes and proofs | [Semantics](mathematical-semantics.md), [Generators](gaussian-generators.md), [Rewrite rules](rewrite-rules.md), [Verification](rewrite-verification.md) |
| MBQC | `Pattern` and `Circuit` translation, resource costs, CV-flow checks, verified rewrite search in `physical`, `logical` and `approximate` modes | [MBQC translation](mbqc-translation.md), [Resource optimisation](resource-optimization.md), [Finite squeezing](finite-squeezing.md) |
| Non-Gaussian (restricted) | 1 → 1 phase-polynomial q-spiders, Kerr, Fock preparation, photon counting, post-selection, finite-energy GKP preparation and parity readout, evaluated on the truncated Fock engines | [Non-Gaussian](non-gaussian.md), [GKP](gkp.md) |

The proofs of every rule are in the theory note
[Foundations of the typed CV diagram calculus](../../theory/CV_ZX_FOUNDATIONS.md). This
guide describes how to use the code and repeats only the statements.

## What this is not

This is **not** a complete CV-ZX calculus. It makes no claim of completeness, universality,
fault tolerance or quantum advantage. It is separate from four related things:

| Related object | Relation |
|---|---|
| Qubit ZX subset, `photographiq.calculus.zx` ([MBQC calculus](../mbqc-calculus.md)) | Unrelated and unchanged. Qubit spiders are not CV generators. |
| `GaussianGraph` ([Gaussian graphical calculus](../gaussian-graphical-calculus.md)), after Menicucci, Flammia and van Loock, PRA 83, 042335 (2011) | Pure Gaussian *states* as complex-weighted graphs. Diagrams describe *processes* (channels and instruments). The two connect through `from_gaussian_graph` and `to_gaussian_graph`; see [GaussianGraph integration](gaussian-graph-integration.md). |
| Nagayoshi et al., "ZX graphical calculus for continuous-variable quantum processes", Phys. Rev. Research 7, 033141 (2025) | Their calculus has q and p spiders with polynomial phase functions, uses ħ = 1, and treats diagrams as equal up to nonzero scalars. They claim completeness only for one-mode Gaussian states and operations. Only their **1 → 1 q-spider** is adopted here, as `PhasePolynomial` = exp(i f(q)) with ħ = 2. Their fusion rule (Eq. 70) and displacement rule (Eq. 79) are used as operator identities. Their rotation convention is q → q cos θ + p sin θ, which has the opposite sign to PhotoGraphiQ's `gaussian.rotation`. |
| Booth, Carette and Comfort: complete equational theories for Gaussian relations (arXiv:2403.10479) and Graphical Symplectic Algebra (arXiv:2401.07914) | Complete calculi for the relational (infinite-squeezing) Gaussian fragment. Not implemented. This package works with finite-squeezing channels and instruments instead. |

## Conventions

The conventions are PhotoGraphiQ's (see [Conventions](../../getting-started/conventions.md)):

| Item | Value |
|---|---|
| Commutator | ħ = 2, [q, p] = 2i |
| Ordering | interleaved (q₀, p₀, q₁, p₁, …) |
| Vacuum covariance | I |
| `Rotation(θ)` | `gaussian.rotation`: q → q cos θ − p sin θ |
| `Squeezer(r)` | diag(e^{−r}, e^{r}) |
| `Shear(s)` | exp(i s q²/4): p → p + s q |
| `CZ(u, v, g)` | exp(i g q_u q_v / 2): p_u += g q_v, p_v += g q_u |
| `Prepare.squeezed(mode, r)` | p-squeezed resource, covariance diag(e^{2r}, e^{−2r}) |

## A first diagram

One corrected teleportation step. A resource mode `"a"` is prepared, entangled with the input
by CZ, the input is measured in p, and the record corrects q of the resource:

```python
import numpy as np
from photographiq.calculus.cv_diagrams import (
    CZ, CVDiagram, Feedforward, Homodyne, Prepare, evaluate,
)

step = CVDiagram(
    inputs=(0,),
    boxes=(
        Prepare.squeezed("a", 0.5),
        CZ(0, "a", 1.0),
        Homodyne(0, np.pi / 2, "m"),
        Feedforward("a", (("m", (-1.0, 0.0)),)),
    ),
    outputs=("a",),
)
X, Y, d = evaluate(step).channel()
print(np.round(X, 12).tolist(), np.round(Y, 6).tolist())
# [[-0.0, -1.0], [1.0, 0.0]] [[0.0, 0.0], [0.0, 0.367879]]
```

The ideal map is a quarter rotation. The finite squeezing adds noise
diag(0, e^{−2r}) = diag(0, 0.367879). [Tutorial 1](examples.md#1-build-a-gaussian-diagram)
continues this example.

## Where to go next

- [Mathematical semantics](mathematical-semantics.md): what `evaluate` computes.
- [Gaussian generators](gaussian-generators.md): every box and its parameters.
- [Rewrite rules](rewrite-rules.md): the rule table.
- [Examples](examples.md): nine runnable tutorials with their output.
- [Limitations](limitations.md): what is not supported.
