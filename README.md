# PhotoGraphiQ

**A software platform for continuous-variable (CV) photonic measurement-based quantum
computing (MBQC): build weighted cluster resources, compile circuits into adaptive
measurement patterns, simulate them with validated Gaussian and Fock engines, and analyse
flow, corrections and finite-squeezing noise.**

[![Tests](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/actions/workflows/ci.yml)
[![Docs build](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/Python-3.11%E2%80%933.14-blue)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-0.3.1-informational)](CHANGELOG.md)
[![License](https://img.shields.io/github/license/chinmoybiswasdeep/PhotoGraphiQ)](LICENSE)
[![Ruff](https://img.shields.io/badge/code%20style-Ruff-261230)](https://docs.astral.sh/ruff/)

> **Status, stated honestly.** The badges above report the remote GitHub Actions runs on
> `main` (Linux, Python 3.11–3.14). The current upgrade work on the `upgraded` branch was
> validated **locally on Windows 11** (2026-11-12: 1721 tests passed, 13 skipped, 0 failed;
> `ruff`, `mypy`, `mkdocs --strict` clean) and has **not yet been re-run on remote CI**.
> Per-feature evidence is in the
> [master validation status](docs/validation/MASTER_VALIDATION_STATUS.md).

![PhotoGraphiQ architecture: user API, IR, compiler, backend planner, numerical engines, results](docs/figures/architecture/figure_A_complete_architecture.svg)

*Figure A: every box cites the code that implements it; the figure is generated and checked
against the source by `scripts/generate_architecture_figures.py`
([all architecture figures](docs/architecture/README.md)).*

## Contents

[What it does](#what-it-does) ·
[Feature status](#feature-status) ·
[Installation](#installation) ·
[Quickstart](#quickstart) ·
[Capabilities by example](#capabilities-by-example) ·
[Scientific validation](#scientific-validation) ·
[Benchmarks](#benchmarks) ·
[Manuscript experiments](#manuscript-experiments) ·
[Documentation](#documentation) ·
[Citation and licence](#citation-and-licence)

## What it does

In CV-MBQC the information lives in the quadratures of optical modes. A squeezed,
entangled **resource graph** supplies the connectivity; **homodyne measurements** and
**outcome-dependent displacement corrections** (feed-forward) perform the computation.
PhotoGraphiQ makes each of those objects explicit and inspectable:

```text
Circuit ──compile──▶ Pattern (prepare · entangle · measure · correct) ──simulate──▶ Result
   ▲                     ▲                                                  │
CVGraph / OpenGraph ── flows.analyze (certify an order, derive corrections)   ├─ outcomes, densities
                                                                            ├─ conditional state
backend registry: gaussian · native-gaussian · native-fock · native-mixed-fock └─ diagnostics
```

| Layer | What you get | Where |
|---|---|---|
| Resource graphs | `CVGraph` (weighted, finite squeezing, nullifiers), standard topologies, `OpenGraph` IR for CV and qubit models | [graphs](docs/user-guide/graphs.md) |
| Patterns and compilation | command IR with symbolic `Outcome` feed-forward; Gaussian circuit → teleportation-step compiler with a provenance trace; Gaussian-plus-cubic target with experimental Kerr/quartic synthesis | [patterns](docs/user-guide/patterns.md), [compilation](docs/user-guide/compilation.md) |
| Simulation | NumPy and native C++20 Gaussian, pure-Fock and mixed-Fock engines behind one capability-checked backend contract | [backends](docs/user-guide/backends.md) |
| Flow and corrections | causal flow, gflow, Pauli flow (qubit); CV flow with certified / impossible / numerically-uncertain classification | [flow](docs/user-guide/flow.md) |
| Calculus | qubit measurement calculus, local Cliffords, a qubit ZX subset, and the Gaussian-pure-state graphical calculus | [MBQC calculus](docs/user-guide/mbqc-calculus.md), [Gaussian graphical calculus](docs/user-guide/gaussian-graphical-calculus.md) |
| Boson sampling and GBS | permanents, hafnians, loop hafnians, torontonians (native C++), exact distributions and samplers | [sampling](docs/user-guide/sampling.md) |
| Non-Gaussian and GKP | Fock inputs, cat and cubic resources, heralded photon subtraction, finite-energy GKP codewords, readout and decoding | [non-Gaussian](docs/user-guide/non-gaussian.md), [GKP](docs/user-guide/gkp.md) |
| Visualization | static SVG/PNG/PDF drawings and interactive, draggable graph views with flow, correction and execution-trace overlays | [visualization](docs/user-guide/visualization.md), [interactive](docs/user-guide/interactive.md) |
| Acceleration | optional CuPy batched Gaussian and Fock kernels; a CUDA-Q Hamiltonian adapter (execution unverified) | [GPU](docs/gpu.md), [CUDA-Q](docs/cudaq.md) |
| PhotoGraphiQML | measurement-based learning (MuTA) with logical-to-physical GKP lowering | [PhotoGraphiQML](docs/ml/index.md) |

One distribution ships two import packages. `photographiq` (the core) needs only NumPy,
SciPy and NetworkX plus its compiled extension; `photographiqml` builds on it, and the core
never imports it. Piquasso is an **optional** reference simulator, not a runtime requirement.

## Feature status

Status labels follow the [master validation status](docs/validation/MASTER_VALIDATION_STATUS.md)
and the [complete functionality reference](docs/COMPLETE_FUNCTIONALITY.md).

| Status | Features |
|---|---|
| **Validated** (independent reference, passing) | NumPy and native Gaussian engines (max error 6.5e-13 vs analytic/`expm`); native pure and mixed Fock gates (5e-16 vs mpmath / enlarged-space `expm`); boson sampling and GBS kernels (incl. a 382-case threshold-probability matrix); causal/g/Pauli flow (360/360 agreement with Graphix); CV flow with exact rational witnesses; measurement calculus, local Cliffords, ZX subset; Gaussian graphical calculus; CuPy GPU kernels (correctness, max deviation 7.2e-16); PhotoGraphiQML logical MuTA vs MentPy |
| **Experimental** | density-matrix (mixed-Fock) evolution with Kraus instruments; Gaussian-plus-cubic compilation with Kerr/quartic synthesis; finite-energy GKP; JAX differentiation of fixed-outcome Fock execution; CUDA-Q adapter (gate translation validated, **CUDA-Q execution unverified**) |
| **Restricted** | physical PhotoGraphiQML lowering (signed-X angle family only); multi-mode displaced native Fock Gaussian unitaries (no derived error bound); threshold GBS in double precision with many clicks at low squeezing (use `precision="auto"`); interactive views verified in Microsoft Edge only |
| **Unsupported** (rejected by the code) | general CV-ZX calculus; CV local complementation; Fock heterodyne/general-dyne; GPU pattern simulation and GPU outcome sampling; GPU permanents/hafnians (measured, not shipped); arbitrary-angle physical MuTA; physical GKP Y/magic-angle measurements |

Physics caveats that hold everywhere: finite squeezing is **physical noise**, not a numerical
error; Fock truncation (total photon number, `sum(n) < cutoff`) is a **numerical
approximation** whose convergence you must check; a homodyne outcome has a probability
*density*, not an event probability; postselecting an outcome is not a deterministic
operation.

## Installation

Python 3.11–3.14, from source (no PyPI release is claimed):

```sh
git clone https://github.com/chinmoybiswasdeep/PhotoGraphiQ.git
cd PhotoGraphiQ
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e .                            # core + native C++ engines
python -c "import photographiq as pg; print(pg.available_backends())"
```

Installing from source compiles `photographiq._native` (C++20 compiler, CMake ≥ 3.24; Eigen is
fetched at a pinned hash). If the extension cannot be loaded, the pure-Python parts and the
NumPy `gaussian` backend still work and the native backends raise an `ImportError` that
explains how to build them. Optional extras: `piquasso`, `precision`, `gpu-cuda12` / `gpu-cuda13`,
`visualization`, `validation`, `autodiff`, `ml-validation`, `ml-experiments`, `docs`, `dev`.

| Platform | Support level |
|---|---|
| Linux (x86_64) | remote CI (GitHub Actions, Python 3.11–3.14) |
| Windows 11 | verified locally (MSVC, Python 3.11–3.14, RTX 4050 GPU) |
| macOS | untested |

Full instructions, including GPU extras, offline native builds and a check that runs real
backend operations: [installation guide](docs/getting-started/installation.md).

## Quickstart

One Fourier step along a two-node CV wire, with a sampled homodyne outcome and its
feed-forward correction:

```python
import photographiq as pg

graph = pg.CVGraph.line(2, squeezing=1.0, inputs=(0,))          # finite squeezing r = 1
pattern = pg.Pattern(graph).measure(0, pg.Homodyne.p())          # measure p on node 0
pattern.displace(1, q=-pg.Outcome(0))                            # feed-forward correction
result = pg.simulate(pattern, inputs={0: pg.GaussianInput.coherent(0.3 + 0.2j)}, seed=7)
print(result.outcomes)              # {0: 0.4035...}: a sample from a density
print(result.state.quadrature(1))   # (mean, variance) of q on the output mode
```

`simulate` uses the NumPy `gaussian` backend by default; pass `backend="native-gaussian"`
for the C++ engine (identical seeded trajectories). Continue with the
[five-minute quickstart](docs/getting-started/quickstart.md).

## Capabilities by example

Every snippet below was executed against version 0.3.1.

### Circuit → CV-MBQC compilation, with finite-squeezing noise kept separate

![A circuit compiled into its MBQC resource, colour-matched gate by gate](docs/assets/circuit-to-mbqc.svg)

```python
import photographiq as pg

circuit = pg.Circuit(1).rotate(0, 0.3).squeeze(0, 0.15)
compiled = circuit.compile(squeezing=0.8)          # teleportation steps T(k) on a resource wire
channel = pg.gaussian_channel(compiled)
print(channel.matrix)   # intended symplectic map S of the circuit
print(channel.noise)    # added noise N from finite squeezing; N -> 0 only as r -> infinity
```

The compiler emits only preparation, entanglement, measurement and correction commands; a
`CompilationTrace` links each source gate to the resources it generated
([compilation guide](docs/user-guide/compilation.md), [theory](docs/theory.md)).

### Native C++ simulation and the backend registry

```python
import photographiq as pg

pattern = pg.non_gaussian.photon_subtraction(theta=0.2)
result = pg.simulate(pattern, inputs={"in": pg.FockInput.number(2)},
                     backend="native-fock", cutoff=8, measurement_outcomes={"count": 1})
print(result.state.photon_number("in"))               # 1.0 on the heralded branch
print(pg.describe_backend("native-fock")["approximations"])
```

| Backend | State | Engine | Requires |
|---|---|---|---|
| `gaussian` (default) | means + covariance | NumPy | core |
| `native-gaussian` | means + covariance | C++20, Eigen, OpenMP | compiled `gaussian` component |
| `native-fock` | pure, total-photon cutoff | C++20 | compiled `fock` component |
| `native-mixed-fock` | density matrix, total-photon cutoff | C++20 | compiled `fock` component |
| `piquasso`, `piquasso-fock`, `piquasso-mixed-fock` | as above | Piquasso 8.0.1 | `[piquasso]` extra |

`pg.select_backend(pattern)` picks a backend that supports every operation the pattern
needs, and unsupported requests are rejected before any state is allocated
([backend comparison](docs/user-guide/backends.md)).

### Flow, determinism and corrections

```python
import networkx as nx
from photographiq import flows
from photographiq.ir import HomodyneSpec, Model, OpenGraph

g = nx.Graph()
g.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
og = OpenGraph(Model.CV_GAUSSIAN, g, (0,), (3,),
               {0: HomodyneSpec(0.2), 1: HomodyneSpec(-0.4), 2: HomodyneSpec(0.7)},
               {n: 0.6 for n in g})
res = flows.analyze(og)
print(res.status, res.layers)      # certified ((0,), (1,), (2,))
print(res.correction_map[0])       # {1: {'q': -1.0, 'p': -0.0}, 2: {'q': -0.0, 'p': -1.0}}
```

CV flow results are **certified** (for the ideal infinite-squeezing model, regime stated),
**impossible** (exact rational witness) or **numerically uncertain**. A qubit flow
certificate is not a CV certificate, and failure to find a flow is not a proof of
nondeterminism ([flow guide](docs/user-guide/flow.md)).

### Interactive visualization

```python
import photographiq as pg
from photographiq.interactive import GraphView
from photographiq.ir import OpenGraph

graph = pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,))
view = GraphView(OpenGraph.from_cvgraph(graph, {0: 0.3, 1: -0.5, 2: 0.0}), title="CV wire")
html = view.construct().to_html()          # self-contained page: drag nodes, step through stages
```

Stages *construct → flow → corrections → execution* are driven by real flow results and
simulation traces ([interactive guide](docs/user-guide/interactive.md)).

### Boson sampling and Gaussian boson sampling

```python
from photographiq import sampling

U = sampling.haar_unitary(4, rng=1)
print(sampling.output_probability(U, [1, 1, 0, 0], [0, 1, 1, 0]))     # |Perm(U_ST)|^2
state = sampling.squeezed_interferometer_state([0.5, 0.5, 0, 0], U, transmissivity=0.9)
print(sampling.threshold_probability(state, [1, 0, 1, 0]))             # torontonian
```

Threshold probabilities carry an error estimate and accept `precision=` (mpmath) for the
regimes where double precision fails ([sampling guide](docs/user-guide/sampling.md),
[accuracy study](docs/validation/gbs-threshold-accuracy.md)).

### Measurement calculus, local Cliffords and ZX (qubit subset)

```python
import networkx as nx
import numpy as np
from photographiq.calculus import lc_equivalence, zx

print(lc_equivalence(nx.star_graph(3), nx.complete_graph(4)).sequence)   # (0,)
d = zx.Diagram()
a, b = d.add_vertex("Z", 0.4), d.add_vertex("Z", 0.7)
d.add_edge(d.add_input(), a); d.add_edge(a, b); d.add_edge(b, d.add_output())
print(np.allclose(zx.fuse(d, a, b).matrix(), d.matrix()))               # spider fusion is exact
```

These act on **qubit** patterns and graph states. For CV, PhotoGraphiQ provides the
Gaussian-pure-state graphical calculus (`calculus.GaussianGraph`, Menicucci–Flammia–van
Loock); there is no general CV-ZX calculus.

### GPU acceleration and its limits

```python
import numpy as np
from photographiq import gpu

device = "gpu" if gpu.available() else "cpu"      # explicit: never a silent fallback
B, n = 64, 4
means = np.zeros((B, 2 * n))
covs = np.broadcast_to(np.eye(2 * n), (B, 2 * n, 2 * n)).copy()
res = gpu.condition_homodyne(means, covs, mode=0, angles=np.zeros(B),
                             outcomes=np.linspace(-1, 1, B), efficiency=0.9, device=device)
print(res.execution.device)                        # e.g. 'cuda:0', read from the output buffer
```

- **What exists:** optional CuPy kernels for *batched* Gaussian updates (symplectic maps,
  homodyne conditioning, purity, fidelity) and batched Fock local gates. They compose
  cuBLAS/cuSOLVER/cuSPARSE calls; there are **no custom CUDA kernels**.
- **When it helps:** many states at once (B ≥ 32) or very large single Fock states
  (D ≳ 4×10⁴). Single small states are slower on the GPU than the native C++ engine.
- **What does not run on the GPU:** whole-pattern simulation, outcome sampling, and the
  BS/GBS matrix functions.
- **CUDA-Q:** `photographiq.cudaq_dynamics` translates gate Hamiltonians to CUDA-Q
  Dynamics. The translation is validated against `native-fock`; execution on a real CUDA-Q
  installation (Linux only) is **unverified**. It is not a pattern backend
  ([CUDA-Q adapter](docs/cudaq.md)).

### PhotoGraphiQML: measurement-based learning

```python
import numpy as np
import photographiqml as pqml

model = pqml.MuTA(2, 1, one_column=True)
print(model.run([1, 0, 0, 0], {"alpha.w1.c1": np.pi / 2}).probabilities)   # [0.5, 0, 0, 0.5]
```

Logical MuTA is an ideal qubit-level model, checked against MentPy. Physical execution
lowers a restricted angle family onto finite-energy GKP resources and reports raw records,
decoded probabilities and the logical target separately. See the
[PhotoGraphiQML documentation](docs/ml/index.md) and the
[core/ML contract](docs/development/photographiqml-contract.md).

## Scientific validation

Numerical physics is checked against **independent references**: closed-form CV results,
dense NumPy/SciPy and `expm` computations on enlarged spaces, mpmath at 40–60 digits,
cross-backend agreement (NumPy ↔ C++ ↔ Piquasso ↔ CuPy) and exact rational arithmetic. Graphix
checks MBQC **structure** (flows, orders, dependencies), never CV amplitudes. MentPy checks
logical MuTA.

- [Master validation status](docs/validation/MASTER_VALIDATION_STATUS.md): requirement-by-requirement status and evidence
- [Independent validation matrix](docs/validation/independent-validation-matrix.md) and [numerical policy](docs/validation/numerical-policy.md)
- [Feature matrix](docs/validation/feature-matrix.md) and [complete functionality reference](docs/COMPLETE_FUNCTIONALITY.md)

Conventions used everywhere: ħ = 2, `[q, p] = 2i`, interleaved quadratures
`(q0, p0, q1, p1, …)`, vacuum covariance `I`, homodyne at angle θ measures
`q cos θ + p sin θ`, Fock truncation by total photon number
([conventions](docs/getting-started/conventions.md)).

## Benchmarks

Measured on one Windows laptop (i5-12450HX, RTX 4050 Laptop GPU) on 2026-11-12; outputs are
checked for agreement before anything is timed. Raw data:
[`benchmarks/results_20261112/`](benchmarks/results_20261112/README.md); tables and
methodology: [Phase 4 report](docs/validation/PHASE4_MANUSCRIPT_REPRODUCIBILITY.md).

| Comparison | Result |
|---|---|
| Native Gaussian vs NumPy, batched (like-for-like) | 1.2–10× |
| Native Fock vs Piquasso, per gate (73 cases) | median 44× |
| Native vs Piquasso, 8-mode wire at cutoff 16 | 4.8 ms vs 90 ms (pure); 58 ms vs 2.31 s (mixed) |
| BS/GBS kernels vs Piquasso | comparable or slower (e.g. hafnian n = 24: 23 ms vs 15 ms) |
| Flow search vs Graphix | correctness-checked reimplementation, not a speed claim |

See [performance](docs/performance.md) for the cost model and [benchmarks](benchmarks/README.md)
to re-run them.

## Manuscript experiments

| Suite | Directory | IDs | Scope |
|---|---|---|---|
| PhotoGraphiQ core | [`paper_experiments/`](paper_experiments/README.md) | PGQ-R… ([index](paper_experiments/EXPERIMENT_INDEX.md)) | CV-MBQC transport, compilation, feed-forward, non-Gaussian, GKP, cross-validation, convergence, native engines, sampling, flow and calculus, performance |
| PhotoGraphiQML | [`paper_experiments_ml/`](paper_experiments_ml/README.md) | PGML-R1–R59 | logical MuTA, MentPy validation, training, kernels, GKP lowering, physical statistics |

Both suites start at R1, so always cite the suite prefix. Each result file carries a
`.metadata.json` provenance record (commit, dirty flag, source hash, packages, hardware).

```sh
python scripts/reproduce_manuscript.py --list        # what would run
python scripts/reproduce_manuscript.py               # fast validation (minutes)
python scripts/reproduce_manuscript.py --mode full   # benchmarks + safe experiment subsets
python paper_experiments/run_all_safe.py             # core suite, fast subset
```

`reproduce_manuscript.py` runs the suites from copies, so recorded results are never
overwritten.

## Documentation

| Start | Learn | Reference | Research |
|---|---|---|---|
| [Installation](docs/getting-started/installation.md) | [Tutorials](docs/tutorials/index.md) | [API reference](docs/api/index.md) | [Theory](docs/theory/index.md) |
| [Quickstart](docs/getting-started/quickstart.md) | [Demo projects](docs/examples/index.md) | [Backends](docs/user-guide/backends.md) | [Validation](docs/validation/index.md) |
| [Concepts](docs/getting-started/concepts.md) | [Notebooks](notebooks) | [Complete functionality](docs/COMPLETE_FUNCTIONALITY.md) | [Performance](docs/performance.md) |
| [Conventions](docs/getting-started/conventions.md) | [BS/GBS notebook](examples/notebooks/boson_sampling_and_gbs_benchmark.ipynb) | [Native build](docs/development/native-build.md) | [References](docs/references.md) |

Build the website locally with `python -m pip install -e ".[docs]"` and
`python -m mkdocs serve`. The Pages workflow targets
`https://chinmoybiswasdeep.github.io/PhotoGraphiQ/`; until a deployment is verified, use the
repository documentation.

## Citation and licence

Cite the software with [CITATION.cff](CITATION.cff), giving the version **and commit** you
used, and cite the upstream methods and software listed in
[references](docs/references.md) (see [citing](docs/citing.md)). No DOI or journal
acceptance is claimed.

Original software is [MIT licensed](LICENSE). Eigen is used under MPL-2.0 only; bundled
third-party research material keeps its original licence. Contributions are welcome: read
[CONTRIBUTING.md](CONTRIBUTING.md), the [code of conduct](CODE_OF_CONDUCT.md) and the
[security policy](SECURITY.md).
