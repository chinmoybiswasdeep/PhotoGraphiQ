# Universal gate-set compilation

The target combines Gaussian operations with cubic-phase resource injection.
`Circuit.cubic_phase(mode, gamma)` lowers to a finite-energy cubic ancilla, inverse
SUM, homodyne measurement and outcome-dependent quadratic/displacement corrections.
`Circuit.kerr(mode, kappa)` first synthesizes quartics using cubic commutators.

```python
import photographiq as pg
circuit = pg.Circuit(1).cubic_phase(0, 0.01)
pattern, trace = circuit.compile(squeezing=0.2, return_trace=True)
result = pg.simulate(pattern, backend="piquasso-fock", cutoff=24,
                     measurement_outcomes={("cubic", 1): 0.0})
```

`pg.synthesis.quadrature_polynomial(terms, steps=...)` accepts terms
`(coefficient, quadrature_angle, degree)` with degrees 1–4. It targets
`exp(i sum(coefficient*x_angle**degree))`. `synthesize_kerr` uses a Weyl-ordered
quartic identity for n². Product-formula steps control synthesis refinement.

Universality describes the Gaussian-plus-cubic computational gate set. The v0.3
front end does not accept arbitrary black-box unitaries or arbitrary-degree
polynomial expressions. Quartic/Kerr synthesis is experimental and can generate
very large patterns. There is no automatic certified tolerance solver.

`SynthesisReport` exposes `approximate`, `order`, `steps`, `primitive_count`,
`target_gate` and `target_parameter`. `order` means the leading amplitude-error
power in inverse slice count: 0.5 for the complete quartic commutator and Kerr
construction, 1 for a general Lie split of noncommuting lower-degree terms, and
`None` for an exact single ideal quadrature power. This is an asymptotic model on
resolved, energy-controlled states, not a certified bound. The symmetric splitting
inside one auxiliary pulse does not make the entire construction second order.

`trace.steps[i].synthesis_report` retains the report for a compiled source gate.
Kerr compilation emits an approximation warning. Cubic trace metadata separately
identifies the finite-resource injection approximation; an exact ideal cubic
primitive does not mean its finite-ancilla realization is an exact unitary.

Validate three different limits: synthesis step refinement, Fock cutoff refinement,
and finite-resource squeezing. Increasing squeezing alone can worsen truncation.
An injection output retains a physical finite-energy envelope; it is not an exact
deterministic unitary. Compare fixed measurement branches when assessing state
convergence, and compare probabilities/densities as well as normalized states.
