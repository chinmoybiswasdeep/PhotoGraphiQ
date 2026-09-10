# PhotoGraphiQ

**Continuous-variable photonic measurement-based quantum computing in Python.**

[![Tests](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/chinmoybiswasdeep/PhotoGraphiQ/branch/main/graph/badge.svg)](https://codecov.io/gh/chinmoybiswasdeep/PhotoGraphiQ)
[![Python](https://img.shields.io/badge/Python-3.11%E2%80%933.14-blue)](pyproject.toml)
[![License](https://img.shields.io/github/license/chinmoybiswasdeep/PhotoGraphiQ)](LICENSE)
[![Docs build](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/actions/workflows/docs.yml)
[![Ruff](https://img.shields.io/badge/code%20style-Ruff-261230)](https://docs.astral.sh/ruff/)
[![Stars](https://img.shields.io/github/stars/chinmoybiswasdeep/PhotoGraphiQ)](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/stargazers)
[![Issues](https://img.shields.io/github/issues/chinmoybiswasdeep/PhotoGraphiQ)](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/issues)
[![Pull requests](https://img.shields.io/github/issues-pr/chinmoybiswasdeep/PhotoGraphiQ)](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/pulls)

![Circuit to MBQC compilation with gate-to-resource correspondence](docs/assets/circuit-to-mbqc.svg)

PhotoGraphiQ connects optical circuits to labelled cluster resources, adaptive
measurements and classical corrections. Build a reusable experiment, simulate its
conditional outputs, and inspect finite-squeezing noise or Fock-cutoff convergence.

## Why PhotoGraphiQ?

Piquasso supplies photonic state evolution. PhotoGraphiQ adds the CV-MBQC layer:
weighted graphs, causal patterns, compilation, feed-forward, resource injection,
diagnostics and independent validation. PhotoGraphiQML is a separate downstream
direction; no ML framework is required by the core package.

## Install from source

Python 3.11–3.14. Until a package release is verified, use the source checkout:

```sh
git clone https://github.com/chinmoybiswasdeep/PhotoGraphiQ.git
cd PhotoGraphiQ
python -m pip install '.[visualization]'
```

The v0.3 development branch is `feature/v0.3`; check it out before installation
while these changes are awaiting release. Piquasso 8.0.1 is a core dependency.
`[validation]` adds Graphix, `[autodiff]` adds JAX, `[docs]` builds the website,
and `pip install -e '.[dev]'` installs an editable development environment.

## Five-minute start

```python
import photographiq as pg

graph = pg.CVGraph.line(2, squeezing=1.0, inputs=(0,))
pattern = pg.Pattern(graph).measure(0, pg.Homodyne.p())
pattern.displace(1, q=-pg.Outcome(0))
result = pg.simulate(pattern, inputs={0: pg.GaussianInput.coherent(0.3+0.2j)}, seed=7)
print(result.outcomes)              # sampled readings, not amplitudes
print(result.state.quadrature(1))   # q mean and variance
pattern.draw(output="cluster.svg")
```

This is one Fourier wire step with finite resource noise. For identity transport,
use `pg.protocols.identity()`. [Read the quickstart](docs/getting-started/quickstart.md).

## Circuit → MBQC

```python
circuit = pg.Circuit(1).rotate(0, 0.3).squeeze(0, 0.15)
compiled, trace = circuit.compile(squeezing=0.8, return_trace=True)
pg.visualize_compilation(circuit, compiled, trace=trace, output="compilation.svg")
channel = pg.gaussian_channel(compiled)
print(channel.matrix, channel.noise)  # ideal linear map and physical added noise
```

Matching colors link source gates to generated resources. The trace identifies
measurements and corrections. PNG, SVG and PDF export use optional Matplotlib.

## Non-Gaussian experiments

```python
pattern = pg.non_gaussian.photon_subtraction(theta=0.2)
result = pg.simulate(pattern, inputs={"in": pg.FockInput.number(2)},
                     backend="piquasso-fock", cutoff=8,
                     measurement_outcomes={"count": 1})
print(result.state.photon_number("in"))  # one photon, conditioned on the herald
```

v0.3 also introduces experimental density-matrix evolution, Gaussian-plus-cubic
compilation with quartic/Kerr synthesis, finite-energy GKP resources and optional
JAX differentiation. These paths have explicit numerical and feature limits.

## Documentation and examples

- [Documentation home](docs/index.md) and [installation](docs/getting-started/installation.md)
- [24 executable tutorials](docs/tutorials/index.md), [10 demo projects](examples/projects/README.md), and [notebooks](notebooks)
- [Inputs](docs/user-guide/inputs.md), [outputs](docs/user-guide/outputs.md), and [API reference](docs/api/index.md)
- [Feature matrix](docs/validation/feature-matrix.md), [performance](docs/performance.md), and [roadmap](ROADMAP.md)

The Pages workflow targets `https://chinmoybiswasdeep.github.io/PhotoGraphiQ/`.
Until deployment is verified, use the repository documentation or run
`python -m mkdocs serve` with the docs extra. Badge status comes from the named
services; coverage may remain unavailable until Codecov is activated.

## Validation and research status

Conventions: `[q,p]=2i`, interleaved quadratures, vacuum statistical covariance I,
radians, and positive resource squeezing means momentum squeezed. A coherent
amplitude alpha has means `(2 Re(alpha), 2 Im(alpha))`.

Analytical CV theory, independent NumPy/SciPy references and raw Piquasso programs
check numerical physics. Graphix checks **MBQC structure**, never CV amplitudes or
quadrature distributions. See the [validation policy](docs/validation/index.md)
and preserved [v0.2 hardening evidence](docs/release-hardening-report.md).

Finite squeezing is physical. Fock truncation, synthesis steps, GKP grids and
gradient estimators each require their own convergence checks. A selected
homodyne outcome reports a density, not an event probability. A universal target
gate set does not imply exact finite-resource gates or arbitrary-unitary synthesis.

## Citation and contribution

Use [CITATION.cff](CITATION.cff) and include the version/commit used. Cite relevant
upstream software and protocols listed in [references](docs/references.md).
No DOI or journal acceptance is claimed. The [paper source](paper/PhotoGraphiQ.tex)
and historical validation artifacts remain available for reproducibility.

Contributions are welcome: read [CONTRIBUTING.md](CONTRIBUTING.md), the
[code of conduct](CODE_OF_CONDUCT.md), and [security policy](SECURITY.md).
Original software is [MIT licensed](LICENSE); bundled third-party research
material retains its original licensing.
