# PhotoGraphiQ

Continuous-variable photonic measurement-based quantum computing in Python.
PhotoGraphiQ represents weighted cluster resources, adaptive measurement patterns,
classical feed-forward and Gaussian gate compilation above a Piquasso backend.
An independent NumPy backend and analytical channel engine check the physics.
GraphiX is used for **structural** validation, never as a CV numerical oracle.

```sh
python -m pip install -e '.[dev]'
python -m pytest
python examples/gaussian_compilation.py
```

Python 3.11–3.14 is supported by the package metadata; the local reference
environment is Python 3.12, Piquasso 8.0.1 and GraphiX 0.4. GraphiX and matplotlib
are optional validation/visualization dependencies. Piquasso is a runtime dependency.

```python
import photographiq as pg

pattern = pg.Circuit(1).rotate(0, 0.4).squeeze(0, 0.2).compile(squeezing=1.2)
result = pg.simulate(
    pattern,
    inputs={0: pg.GaussianInput.coherent(0.3 + 0.1j)},
    seed=42,
    frame=True,
)
print(result.outcomes)
print(result.state.mean, result.state.covariance)

# Exact unconditional channel for fixed-angle, affine Gaussian patterns:
channel = pg.gaussian_channel(pattern)
print(channel.matrix, channel.noise)
```

Implemented features include weighted graph families and arbitrary labels;
typed commands; safe symbolic parameter/outcome expressions and declared callables;
exact homodyne, heterodyne and general Gaussian conditioning; displacement frames;
causal scheduling; total-order CV-flow certificates; Gaussian circuit compilation;
finite-squeezing channel analysis; photon loss and detector inefficiency;
versioned JSON; graph/dependency diagrams; and independent repeated trajectories.
Experimental Fock execution supports number, cat and arbitrary pure Fock inputs,
photon counting, Gaussian gates and cubic-phase resources with explicit cutoff checks.

**Conventions:** `[q,p]=2i`, interleaved `(q0,p0,q1,p1,...)`, statistical vacuum
covariance `V=I`, radians, `CZ(g): p_i -> p_i + g*q_j`, and positive resource
squeezing means momentum squeezed. Piquasso's covariance is `2*V`.

Finite squeezing is physical by default. A conditional output is not an ideal
unitary output, and a mixture of adaptive Gaussian trajectories need not be
Gaussian. `ensemble_state()` returns its first two moments in a Gaussian container.
The Piquasso backend uses native physical gates and PhotoGraphiQ's exact measurement
adapter; see the documented upstream detector-convention discrepancies.

The release does not implement adaptive Fock homodyne, GKP error correction,
universal non-Gaussian compilation, arbitrary-order CV-flow search, hardware
temporal scheduling or automatic differentiation. Unsupported operations raise
clear exceptions. The beam-splitter compiler prioritizes a transparent correct
decomposition over resource efficiency.

* [User guide and tutorials](docs/guide.md)
* [Mathematics and derivations](docs/theory.md)
* [API and extension guide](docs/api.md)
* [Design matrix and inspected source](docs/design.md)
* [Validation and limitations](docs/validation.md)
* [Coverage of the supplied specification](docs/specification-status.md)
* [References](docs/references.md)
* [Quantum-style paper source](paper/PhotoGraphiQ.tex) and [PDF](paper/PhotoGraphiQ.pdf)
* [Reproducible paper data](paper/results/finite_squeezing.csv)

Reproduce the paper figures with `python experiments/reproduce.py`. Build the
manuscript by running `pdflatex PhotoGraphiQ.tex` twice from `paper/`; the official
Quantum class is included under its original LaTeX Project Public License.
The manuscript is a software paper draft, not a claim of journal acceptance.

MIT license for original software. Upstream source references and bundled research
papers retain their own licenses. See [CONTRIBUTING.md](CONTRIBUTING.md).
