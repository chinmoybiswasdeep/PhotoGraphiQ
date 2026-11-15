# Complete functionality reference

Every feature of PhotoGraphiQ and PhotoGraphiQML on the `upgraded` branch, with its Python API,
implementation, supported backends, CPU/GPU support, mathematical assumptions, an executed
example, the tests that cover it and an honest status. Examples were run in a Windows
development environment with the native extension built and Piquasso, Graphix, JAX, mpmath
and CuPy installed (CUDA-Q, MentPy and Playwright were not available). The document was
brought up to date with the audit of 2026-11-12 (release engineering, independent
validation and feature completion; see [Phase 1](validation/PHASE1_RELEASE_AUDIT.md),
[Phase 2](validation/PHASE2_SCIENTIFIC_AUDIT.md) and
[Phase 3](validation/PHASE3_FEATURE_COMPLETION.md)); evidence produced outside this
environment (MentPy, the real-browser check) is labelled as such where it is cited.

Conventions throughout: hbar = 2, `[q, p] = 2i`, interleaved quadratures `(q0, p0, q1, p1, ...)`,
vacuum covariance `I`, Fock truncation by total photon number `sum n < cutoff`.

## Status labels

| Status | Meaning | Features |
|---|---|---|
| **Implemented and validated** | implemented, and tests compare it with an independent reference (closed form, brute-force definition, `expm` on an enlarged space, Piquasso, Graphix, mpmath, MentPy, another backend) | 168 |
| **Implemented but experimental** | works and is tested, but validation is limited or the code labels it experimental | 91 |
| **Partially supported** | some cases work, others raise (stated per feature) | 8 |
| **Unsupported** | explicitly rejected by the code (the error is stated) | 13 |
| **Planned** | not implemented; never counted as complete | 2 |

The counts are the first status label of each per-feature `Status` row (282 rows), recounted
by script on 2026-11-12. The **Planned** rows of the Section 36 table are roadmap items, not
feature entries, and are not counted. Until 2026-11-11 every Fock path was listed as
experimental because the code and `docs/validation/feature-matrix.md` call Fock execution
experimental; native Fock entries with a first-principles reference in `tests/validation/`
are now validated, and each such Tests row says why (Section 13).

Postselected results are never presented as deterministic, and no GPU execution is claimed
where the code runs on the CPU. Benchmark timings are not quoted here; see
`benchmarks/README.md`, the 2026-11-12 reproduction bundle `benchmarks/results_20261112/`
and the result files under `benchmarks/`.

## Contents

1. [Software overview](#1-software-overview)
2. [Architecture](#2-architecture)
3. [Package/module hierarchy](#3-packagemodule-hierarchy)
4. [Public APIs](#4-public-apis)
5. [Graph construction](#5-graph-construction)
6. [Gaussian state preparation](#6-gaussian-state-preparation)
7. [Non-Gaussian state preparation](#7-non-gaussian-state-preparation)
8. [Gaussian gates](#8-gaussian-gates)
9. [Non-Gaussian gates](#9-non-gaussian-gates)
10. [Measurements](#10-measurements)
11. [Adaptive feedforward](#11-adaptive-feedforward)
12. [Native C++ Gaussian backend](#12-native-c-gaussian-backend)
13. [Native C++ Fock backend](#13-native-c-fock-backend)
14. [Mixed-Fock simulation](#14-mixed-fock-simulation)
15. [GPU execution](#15-gpu-execution)
16. [CUDA-Q support](#16-cuda-q-support)
17. [Sampling algorithms](#17-sampling-algorithms)
18. [Boson Sampling](#18-boson-sampling)
19. [Gaussian Boson Sampling](#19-gaussian-boson-sampling)
20. [Hafnian and Torontonian routines](#20-hafnian-and-torontonian-routines)
21. [Measurement calculus](#21-measurement-calculus)
22. [Local Clifford transformations](#22-local-clifford-transformations)
23. [ZX calculus](#23-zx-calculus)
24. [Flow and gflow](#24-flow-and-gflow)
25. [Pauli flow](#25-pauli-flow)
26. [CV determinism](#26-cv-determinism)
27. [GKP operations](#27-gkp-operations)
28. [PhotoGraphiQML](#28-photographiqml)
29. [Optimization algorithms](#29-optimization-algorithms)
30. [Interactive visualization](#30-interactive-visualization)
31. [Manuscript figures](#31-manuscript-figures)
32. [Benchmarks](#32-benchmarks)
33. [Examples](#33-examples)
34. [Scientific validation](#34-scientific-validation)
35. [Known limitations](#35-known-limitations)
36. [Future development](#36-future-development)

## 1. Software overview

PhotoGraphiQ simulates continuous-variable (CV) photonic measurement-based quantum computation: weighted CV cluster resources, adaptive homodyne/heterodyne/photon-counting patterns, Gaussian and non-Gaussian gates, finite-energy GKP encodings, and analysis tools. One distribution (`photographiq` 0.3.1, Python >= 3.11, < 3.15) ships two import packages: `photographiq` (core) and `photographiqml` (0.2.0, the PhotoGraphiQML learning layer). Global conventions used everywhere below: hbar = 2, q = a + a^dag, p = -i(a - a^dag), [q, p] = 2i, interleaved quadrature order (q0, p0, q1, p1, ...), statistical covariance with vacuum covariance V = I, and Fock truncation by total photon number sum(n) < cutoff.

### Core package `photographiq`

| Field | Value |
|---|---|
| Description | Pattern construction, compilation, backend selection, trajectory simulation and analysis for CV measurement-based computation. Required dependencies are only NumPy, SciPy and NetworkX. |
| Python API | `import photographiq as pg`; `pg.__version__` (`"0.3.1"`) |
| Implementation | `src/photographiq/` |
| Backends | `gaussian` (default, NumPy) is always available; others listed in Section 2 |
| CPU / GPU | CPU (NumPy) by default; CPU (C++/OpenMP) through the native backends; GPU only through the separate `photographiq.gpu` module (CuPy) |
| Assumptions | hbar = 2, [q,p] = 2i, xpxp ordering, vacuum covariance I, total-photon cutoff |
| Tests | `tests/` (unit, analytical, cross-backend, Piquasso and Graphix reference suites) |
| Status | **Implemented and validated** (Gaussian core; the Fock, non-Gaussian and GKP paths carry their own, mostly experimental, statuses in Sections 6 to 11) |

Example:

```python
import photographiq as pg, photographiqml as pml
print(pg.__version__, pml.__version__)
print(pg.available_backends())
# 0.3.1 0.2.0
# ('gaussian', 'piquasso', 'piquasso-fock', 'piquasso-mixed-fock', 'native-gaussian', 'native-fock', 'native-mixed-fock')
```

### Extension package `photographiqml`

| Field | Value |
|---|---|
| Description | Logical MuTA (measurement-based) learning models and a restricted signed-X physical GKP execution path built on public PhotoGraphiQ commands. It depends on `photographiq` only. |
| Python API | `import photographiqml as pml`; `pml.MuTA`, `pml.PhysicalMuTA`, `pml.lower_muta_to_gkp`, `pml.Trainer` (full list in Section 4) |
| Implementation | `src/photographiqml/` |
| Backends | Logical models are ideal qubit calculations (no CV backend); `PhysicalMuTA` runs on Fock backends (`GKPPhysicalConfig(backend="piquasso-fock")` by default) |
| CPU / GPU | CPU only |
| Assumptions | Logical layer is an ideal qubit model; the physical layer uses finite-energy GKP resources and nearest-cell decoding; no fault-tolerance claim |
| Tests | `tests/ml/` (lowering, muta, gkp, training, cross_layer, mentpy_reference, paper_experiments; the MentPy reference tests skip here because MentPy is not installed); `tests/native/test_native_fock_cross_layer.py` (physical MuTA on native Fock vs Piquasso) |
| Status | **Implemented but experimental** (package-level summary; individual features in Sections 28 and 29 carry their own statuses, from validated logical components to unsupported physical cases) |

Example: see `docs/ml/index.md` and `examples/ml/`.

### Optional compiled extension `photographiq._native`

| Field | Value |
|---|---|
| Description | C++20 engines (Eigen, OpenMP, pybind11) built by scikit-build-core from `cpp/` with components `gaussian`, `fock` and `sampling`. Pure-Python functionality never requires it. |
| Python API | `photographiq.native.AVAILABLE`, `photographiq.native.components()`, `photographiq.native.build_info()`, `photographiq.native.require(component)` |
| Implementation | `src/photographiq/native.py`; `cpp/src/core/`, `cpp/src/gaussian/`, `cpp/src/fock/`, `cpp/src/sampling/`, `cpp/src/bindings/` |
| Backends | `native-gaussian`, `native-fock`, `native-mixed-fock`; native sampling kernels in `photographiq.sampling` |
| CPU / GPU | CPU (C++/OpenMP) |
| Assumptions | Same conventions as the Python backends (`build_info()["hbar"] == 2.0`) |
| Tests | `tests/native/` (native Gaussian vs NumPy backend and closed forms; native Fock vs Piquasso, `scipy.linalg.expm` in enlarged spaces and closed forms); `cpp/tests/test_gaussian.cpp`, `cpp/tests/test_fock.cpp`, `cpp/tests/test_sampling.cpp` |
| Status | **Implemented and validated** (the Fock engines share the experimental status of all Fock paths, Section 2) |

Example:

```python
from photographiq import native
print(native.AVAILABLE, native.components())
print(native.build_info()["openmp"], native.build_info()["cxx_standard"])
# True ('gaussian', 'fock', 'sampling')
# True 202002
```

### Optional third-party integrations

| Field | Value |
|---|---|
| Description | Optional extras: `piquasso` (Piquasso 8.0.1 backends and reference checks), `validation` (Graphix 0.4, structural reference), `autodiff` (JAX, plus Piquasso 8.0.1, which prepares every input and `Prepare` state before the JAX evolution), `precision` (mpmath for threshold probabilities), `gpu-cuda12` / `gpu-cuda13` (CuPy), `visualization` (matplotlib), `ml-validation` (MentPy 0.1.0a15), `ml-experiments` (matplotlib, scikit-learn, Piquasso). CUDA-Q is not an extra; `photographiq.cudaq_dynamics` is an optional Hamiltonian-level adapter that imports `cudaq` lazily. |
| Python API | `photographiq._optional.LazyModule` (lazy import with install hints); `pip install 'photographiq[piquasso]'` etc. |
| Implementation | `src/photographiq/_optional.py`, `pyproject.toml` (`[project.optional-dependencies]`), `src/photographiq/cudaq_dynamics.py` |
| Backends | `piquasso`, `piquasso-fock`, `piquasso-mixed-fock` need Piquasso; CUDA-Q is not a backend (adapter only) |
| CPU / GPU | CPU for all integrations except CuPy (GPU, `photographiq.gpu`) |
| Assumptions | Requesting a backend whose package is missing raises `ImportError` with the install command; importing `photographiq` imports no optional package |
| Tests | `tests/cross_backend/test_registry_planner.py::test_registry_import_is_lazy`, `tests/cudaq_adapter/test_cudaq_adapter.py::test_cudaq_missing_raises_import_error` |
| Status | **Partially supported** (Piquasso, Graphix, JAX, mpmath, CuPy integrations work; the CUDA-Q path of the CUDA-Q adapter is unverified) |

Example:

```python
from photographiq._optional import piquasso
print(piquasso.available)
# True
```

## 2. Architecture

The data flow is: user API (graphs, patterns, circuits, protocol builders) -> intermediate representations (typed graph IR `OpenGraph` in `src/photographiq/ir.py`; command-level `Pattern`) -> compiler (`Circuit.compile`, `compile_circuit`) -> capability analysis and backend registry/planner (`src/photographiq/capabilities.py`, `src/photographiq/registry.py`) -> trajectory engines (backends) -> results (`Result`, `ShotResult`) -> analysis and visualization. PhotoGraphiQML sits on top as an extension that lowers its models to public PhotoGraphiQ commands. The registered engines are:

| Registry name | Representation | Implementation | Device | Needs |
|---|---|---|---|---|
| `gaussian` (default) | means/covariances | NumPy | CPU | core |
| `piquasso` | means/covariances | Piquasso gates + exact NumPy conditioning | CPU | Piquasso |
| `native-gaussian` | means/covariances | C++20 | CPU | native `gaussian` |
| `piquasso-fock` | pure Fock vector | Piquasso | CPU | Piquasso |
| `native-fock` | pure Fock vector | C++20 | CPU | native `fock` |
| `piquasso-mixed-fock` | density matrix | Piquasso | CPU | Piquasso |
| `native-mixed-fock` | density matrix | C++20 | CPU | native `fock` |

### User API layer

| Field | Value |
|---|---|
| Description | Users build computations from `CVGraph` resources, explicit command lists in `Pattern`, gate-level `Circuit` objects, or ready-made builders in `photographiq.protocols` and `photographiq.non_gaussian`. |
| Python API | `pg.CVGraph(...)`, `pg.Pattern(graph=None, *, inputs=())`, `pg.Circuit(modes)`, `pg.protocols.*`, `pg.non_gaussian.*` |
| Implementation | `src/photographiq/graph.py`, `src/photographiq/pattern.py`, `src/photographiq/compiler.py`, `src/photographiq/protocols.py`, `src/photographiq/non_gaussian.py` |
| Backends | Backend-neutral |
| CPU / GPU | CPU only (construction is pure Python) |
| Assumptions | Labels are arbitrary hashable objects; JSON serialization supports primitive labels and tuples of them |
| Tests | `tests/graphix_reference/test_structure.py::test_wire_connectivity_io_measurement_order_and_domains` (structure checked against Graphix) |
| Status | **Implemented and validated** |

Example: see Sections 5 and 8.

### Typed graph IR `OpenGraph`

| Field | Value |
|---|---|
| Description | Backend-independent typed open graph shared by flow analysis, measurement calculus and interactive visualization. It fixes the computational model (`Model.CV_GAUSSIAN`, `CV_NONGAUSSIAN`, `QUBIT`, `GKP_LOGICAL`) because measurement semantics and corrections differ between models. |
| Python API | `pg.OpenGraph(model, graph, inputs=(), outputs=(), measurements={}, squeezing={})`, `OpenGraph.from_cvgraph(graph, shears=None, *, model=Model.CV_GAUSSIAN)`, `.to_cvgraph()`, `.to_dict()`, `OpenGraph.from_dict(data)`, `.adjacency(order=None)`; `photographiq.ir.HomodyneSpec(shear)`, `photographiq.ir.QubitMeasurement(plane, angle, pauli=None)` |
| Implementation | `src/photographiq/ir.py` |
| Backends | Not executed directly; `to_cvgraph()` (CV models only) produces an executable resource |
| CPU / GPU | CPU only |
| Assumptions | CV homodyne angle is `atan2(1, shear)`; qubit graphs must be unweighted and carry no squeezing; outputs are never measured; layout coordinates are not stored |
| Tests | `tests/unit/test_ir_open_graph.py` (conversion, JSON round trip with tuple labels, model-semantics rejection) |
| Status | **Implemented but experimental** (tests are round-trip and rejection checks; no independent reference applies to the container itself) |

Example:

```python
import photographiq as pg
graph = pg.CVGraph.line(3, squeezing=0.8, inputs=(0,), outputs=(2,))
og = pg.OpenGraph.from_cvgraph(graph, shears={0: 0.0, 1: 0.5})
print(og.model.value, og.measured)
print(og.to_dict()["measurements"])
# cv-gaussian (0, 1)
# {'0': {'shear': 0.0}, '1': {'shear': 0.5}}
```

### Command-level IR `Pattern`

| Field | Value |
|---|---|
| Description | Ordered, backend-neutral command list (`Prepare`, `Entangle`, gates, `Measure`, `Signal`, `Output`, ...) with ordered input labels. `validate()` checks labels, mode lifetimes and classical causality; `dependencies()` returns the command DAG. |
| Python API | `pg.Pattern(graph=None, *, inputs=())`, `.append(command)`, `.extend(commands)`, `.measure(node, measurement=None, *, key=None)`, `.displace(node, *, q=0.0, p=0.0)`, `.validate()`, `.dependencies()`, `.schedule()`, `.standardize()`, `.inspect()`, `.outputs`, `.parameters`, `.graph`, `.to_json(path=None)`, `Pattern.from_json(text)`, `.draw(**kwargs)` |
| Implementation | `src/photographiq/pattern.py`, `src/photographiq/commands.py`, `src/photographiq/flow.py`, `src/photographiq/serialization.py` |
| Backends | All executing backends |
| CPU / GPU | CPU only |
| Assumptions | `Output` must be the final command; measurement keys default to the node label and must be unique; `measure()` defaults to `Homodyne.q()` |
| Tests | `tests/graphix_reference/test_structure.py`, `tests/graphix_reference/test_labelled_structures.py` (graph, I/O, measurement order and domains vs Graphix); `tests/unit/test_core.py::test_future_and_cycle_and_missing_modes`, `::test_json_roundtrip_and_security` |
| Status | **Implemented and validated** |

Example:

```python
import photographiq as pg
p = pg.Pattern(pg.CVGraph.line(2, squeezing=1.0))
p.measure(0, pg.Homodyne.p()).displace(1, p=pg.Outcome(0))
print(p.validate())
print(p.inspect())
# Pattern(inputs=0, outputs=1, commands=5)
# 0: Prepare(node=0, squeezing=1.0, state=None)
# 1: Prepare(node=1, squeezing=1.0, state=None)
# 2: Entangle(u=0, v=1, weight=1.0)
# 3: Measure(node=0, measurement=Homodyne(angle=1.5707963267948966, efficiency=1.0, noise=0.0), key=None)
# 4: Displace(node=1, q=0.0, p=Expr(op='outcome', args=(0,)))
```

### Compiler

| Field | Value |
|---|---|
| Description | Lowers a gate-level `Circuit` to a finite-squeezing teleportation `Pattern` (single-mode symplectics via four or five shear steps, CZ and beamsplitter via transport, cubic/Kerr via resource injection or synthesis). This document has no separate compilation section; see [compilation](user-guide/compilation.md) and [universal compilation](user-guide/universal-compilation.md). |
| Python API | `pg.Circuit(modes).displace/rotate/squeeze/gaussian/cz/beamsplitter/identity/cubic_phase/kerr(...)`, `Circuit.compile(squeezing=1.0, *, return_trace=False, synthesis_steps=8)`, `pg.compile_circuit(circuit, *, squeezing=1.0, return_trace=False, synthesis_steps=8)`, `photographiq.compiler.decompose_symplectic(matrix)` |
| Implementation | `src/photographiq/compiler.py`, `src/photographiq/synthesis.py` |
| Backends | Output patterns of Gaussian circuits run on all backends; cubic/Kerr lowering needs a Fock backend |
| CPU / GPU | CPU only |
| Assumptions | Finite resource squeezing r leaves physical noise; the ideal map is reached only as r -> infinity |
| Tests | `tests/protocols/test_gaussian_protocols.py::test_exact_compiler_matrix`, `tests/analytical/test_channels.py::test_compiled_gaussian_gate_channel` (analytical channel), `tests/piquasso_reference/test_raw_piquasso.py::test_ideal_compiler_target_against_raw_gaussian_circuit` (Piquasso) |
| Status | **Implemented and validated** (Gaussian lowering; cubic/Kerr synthesis is experimental, `src/photographiq/synthesis.py`) |

Example:

```python
import photographiq as pg
pattern = pg.Circuit(1).rotate(0, 0.3).compile(squeezing=1.0)
print(pattern)
print(pg.gaussian_channel(pattern).matrix.round(6))
# Pattern(inputs=1, outputs=1, commands=17)
# [[ 0.955336 -0.29552 ]
#  [ 0.29552   0.955336]]
```

### Capability analysis and backend planner

| Field | Value |
|---|---|
| Description | `required_features` lists every capability a pattern needs without allocating a state; `preflight` checks a backend instance before any preparation; `select_backend` returns the cheapest available backend whose class capabilities cover the features and whose preflight passes, and raises `NotImplementedError` with per-candidate reasons otherwise. |
| Python API | `pg.select_backend(pattern, inputs=None, *, initial_state=None, measurement_outcomes=None, prefer=("native", "numpy"), cutoff=None)` -> `BackendSelection(name, representation, features, explanation, rejected)`; `photographiq.capabilities.required_features(pattern, inputs=None, initial_state=None, measurement_outcomes=None)`; `photographiq.capabilities.preflight(engine, pattern, inputs, initial_state=None, measurement_outcomes=None)`; `photographiq.registry.missing_features(name, features)` |
| Implementation | `src/photographiq/capabilities.py`, `src/photographiq/registry.py` |
| Backends | Considers every registered backend; cost order gaussian < pure-fock < mixed-fock, then `prefer`, then registration order |
| CPU / GPU | CPU only |
| Assumptions | Fock candidates are skipped when no `cutoff` is given; derived requirements `mixed_gaussian_input` and `correlated_gaussian_state` are decided from the backend class |
| Tests | `tests/cross_backend/test_registry_planner.py` (`test_required_features_basic`, `test_planner_feature_check_agrees_with_preflight`, `test_select_gaussian_pattern`, `test_select_non_gaussian_pattern`, `test_select_mixed_pattern`, `test_select_raises_for_unsupported_combination`, `test_select_rejects_cutoff_too_small_via_preflight`) |
| Status | **Implemented but experimental** (behavioural tests only; no independent reference applies) |

Example:

```python
import photographiq as pg
from photographiq.capabilities import required_features
p = pg.Pattern(pg.CVGraph.line(2)).measure(0, pg.PhotonNumber())
print(sorted(required_features(p)))
sel = pg.select_backend(p, cutoff=8)
print(sel.name)
print(sel.rejected["gaussian"])
# ['photon_counting']
# native-fock
# missing photon_counting
```

### Backend registry

| Field | Value |
|---|---|
| Description | Static `BackendInfo` entries (import path, representation, implementation family, device, precision, approximations, required packages and native components). Capabilities are read from each backend class, never duplicated. Nothing heavy is imported until a backend is requested. |
| Python API | `pg.available_backends()`, `pg.describe_backend(name)`, `photographiq.registry.create_backend(name, cutoff=None)`, `photographiq.registry.BACKENDS` |
| Implementation | `src/photographiq/registry.py` |
| Backends | The seven entries in the table above |
| CPU / GPU | CPU only (registry) |
| Assumptions | `differentiable` is False for every trajectory backend; Fock entries declare the approximation "Fock truncation by total photon number sum(n) < cutoff" |
| Tests | `tests/cross_backend/test_registry_planner.py::test_entry_resolves_or_reports_reason`, `::test_capabilities_are_read_from_class`, `::test_fock_entries_declare_total_photon_truncation`, `::test_registry_import_is_lazy` |
| Status | **Implemented but experimental** (behavioural tests only) |

Example:

```python
import photographiq as pg
info = pg.describe_backend("native-fock")
print(info["representation"], info["implementation"], info["device"], info["uses_cutoff"])
print(info["approximations"])
# pure-fock native cpu True
# ('Fock truncation by total photon number sum(n) < cutoff',)
```

### Gaussian trajectory engines

| Field | Value |
|---|---|
| Description | Exact Gaussian evolution of means and covariances with destructive conditioning by the Schur complement; supports homodyne (ideal and noisy), heterodyne, generaldyne, loss, quadratic phase and postselection. |
| Python API | `pg.simulate(..., backend="gaussian" | "piquasso" | "native-gaussian")`; classes `photographiq.backends.GaussianBackend`, `photographiq.backends.PiquassoBackend`, `photographiq.backends.native_gaussian.NativeGaussianBackend` |
| Implementation | `src/photographiq/backends/gaussian.py`, `src/photographiq/backends/piquasso.py`, `src/photographiq/backends/native_gaussian.py`; `cpp/src/gaussian/operations.cpp`, `cpp/src/gaussian/conditioning.cpp` |
| Backends | `gaussian`, `piquasso`, `native-gaussian` |
| CPU / GPU | CPU (NumPy) for `gaussian`/`piquasso`; CPU (C++/OpenMP) for `native-gaussian` |
| Assumptions | hbar = 2, xpxp, V_vacuum = I; the three engines consume random numbers identically, so equal seeds give equal trajectories up to float rounding |
| Tests | `tests/analytical/test_measurements.py::test_exact_homodyne_schur_complement` (closed form), `tests/analytical/test_primitives.py` (vs Piquasso), `tests/native/test_native_gaussian_backend.py::test_random_patterns_match_numpy_backend`, `::test_postselection_matches_numpy_backend` |
| Status | **Implemented and validated** |

Example: see Section 10 (homodyne example on all backends).

### Fock trajectory engines

| Field | Value |
|---|---|
| Description | Pure-state (`piquasso-fock`, `native-fock`) and density-matrix (`piquasso-mixed-fock`, `native-mixed-fock`) trajectories with explicit total-photon truncation, retained-norm diagnostics and boundary-population warnings. They add Fock inputs, cubic phase, Kerr, ladder operations, photon counting and finite instruments. |
| Python API | `pg.simulate(..., backend="native-fock", cutoff=N)` etc.; classes `photographiq.backends.fock.PiquassoFockBackend(cutoff=None, *, norm_tolerance=1e-3, max_dimension=1_000_000, boundary_warning=0.02)`, `photographiq.backends.mixed_fock.MixedFockBackend(cutoff=None, *, max_matrix_bytes=256_000_000, **kwargs)`, `photographiq.backends.native_fock.NativeFockBackend(cutoff=None, *, cubic_phase="truncated_generator", cubic_tolerance=1e-12, **kwargs)`, `photographiq.backends.native_fock.NativeMixedFockBackend` (same signature). `cutoff` defaults to `None` only syntactically: an integer cutoff >= 2 is required, and otherwise each backend raises `ValueError("<registry name> requires an explicit total-photon cutoff >= 2")` naming itself (`piquasso-fock`, `piquasso-mixed-fock`, `native-fock` or `native-mixed-fock`, from the private class attribute `_registry_name`). Before the fix, all four said `piquasso-fock`; `tests/regression/test_validation_messages.py::test_missing_cutoff_names_the_backend` checks each. These classes are not re-exported by `photographiq.backends`, which exports only `BaseBackend`, `GaussianBackend` and `PiquassoBackend`. |
| Implementation | `src/photographiq/backends/fock.py`, `src/photographiq/backends/mixed_fock.py`, `src/photographiq/backends/native_fock.py`; `cpp/src/fock/basis.cpp`, `cpp/src/fock/gates.cpp`, `cpp/src/fock/kernels.cpp` |
| Backends | `piquasso-fock`, `native-fock`, `piquasso-mixed-fock`, `native-mixed-fock` |
| CPU / GPU | CPU (Piquasso/NumPy) or CPU (C++/OpenMP) |
| Assumptions | Total-photon truncation sum(n) < cutoff; renormalisation after each operation with the retained norm recorded (error below `norm_tolerance`); native engines apply exact Gaussian matrix elements as P U P, Piquasso decomposes some Gaussian gates, so the two differ at truncation level and agree as the cutoff grows |
| Tests | `tests/native/test_native_fock_backend.py` (vs Piquasso and exact Gaussian moments, `test_piquasso_difference_vanishes_with_cutoff`), `tests/native/test_native_fock_kernels.py` (closed forms, `expm`), `tests/mixed_fock/`, `tests/fock/` |
| Status | **Implemented but experimental** (`src/photographiq/backends/fock.py` and `docs/validation/feature-matrix.md` label the Fock paths experimental; this architecture entry covers the Piquasso engines too, while the native engines are validated feature by feature in Sections 13 and 14 since 2026-11-12) |

Example: see Sections 7 and 9.

### CUDA-Q Dynamics adapter

| Field | Value |
|---|---|
| Description | Optional Hamiltonian-level adapter, not a registered backend (CUDA-Q cannot execute patterns). Gate generators in PhotoGraphiQ conventions, dense matrices, the per-mode/total-photon truncation mapping, a NumPy/SciPy `reference_evolve`, `to_cudaq` and `evolve` (calls `cudaq.evolve`). See [CUDA-Q adapter](cudaq.md) and Section 16. |
| Python API | `photographiq.cudaq_dynamics` (`BosonicHamiltonian`, generators, `to_matrix`, `per_mode_to_total_photon`, `total_photon_to_per_mode`, `reference_evolve`, `to_cudaq`, `evolve`, `cudaq_probe`, `cudaq_capabilities`); `photographiq.backends.cudaq_dynamics` re-exports the probe functions |
| Implementation | `src/photographiq/cudaq_dynamics.py` |
| Backends | None (`simulate(backend="cudaq-dynamics")` raises `ValueError` naming the adapter module) |
| CPU / GPU | Reference on CPU; the CUDA-Q path has not been run |
| Assumptions | CUDA-Q ships Linux wheels only. CUDA-Q truncates per mode and documents position = (a^dag + a)/2; the adapter goes through ladder operators with q = a + a^dag |
| Tests | `tests/cudaq_adapter/test_cudaq_adapter.py`, `tests/cross_backend/test_registry_planner.py::test_cudaq_is_an_adapter_not_a_backend` |
| Status | **Implemented but experimental** (conventions validated against native-fock; CUDA-Q execution unverified) |

### Results layer

| Field | Value |
|---|---|
| Description | `simulate` returns one conditional trajectory as `Result(outcomes, records, state, backend, seed, physical_displacements, measurement_statistics)` with `log_likelihood`; `run_shots` returns `ShotResult` with `values(key)` and `ensemble_state()`. |
| Python API | `pg.simulate(pattern, *, backend="gaussian", parameters=None, inputs=None, seed=None, frame=False, cutoff=None, initial_state=None, measurement_outcomes=None)`, `pg.run_shots(pattern, shots, *, seed=None, **kwargs)` (alias `pg.sample`), `Result.log_likelihood`, `ShotResult.values(key)`, `ShotResult.ensemble_state()` |
| Implementation | `src/photographiq/simulator.py` |
| Backends | All executing backends; `ensemble_state()` requires Gaussian trajectory states |
| CPU / GPU | CPU only |
| Assumptions | `Result.backend` is the backend class name (for example `"GaussianBackend"`), not the registry name, as the `Result` docstring now states; `log_likelihood` is a joint density in calibrated outcome units when continuous measurements are present, not an event probability; `ensemble_state()` is a moment match and the mixture need not be Gaussian |
| Tests | `tests/protocols/test_gaussian_postselection.py::test_sampled_density_is_analytic_density`, `tests/cross_backend/test_cross_backend_agreement.py::test_homodyne_postselection_density_closed_form`, `tests/protocols/test_gaussian_protocols.py::test_braunstein_kimble_ensemble` (closed form), `tests/analytical/test_measurements.py::test_seed_zero_and_independent_shots` |
| Status | **Implemented and validated** |

Example:

```python
import photographiq as pg
p = pg.Pattern(pg.CVGraph.line(2, squeezing=0.5)).measure(0, pg.Homodyne.p())
r = pg.simulate(p, seed=3)
print(r)
print(r.outcomes, r.measurement_statistics[0]["kind"], round(r.log_likelihood, 6))
# Result(backend='GaussianBackend', outcomes=1, records=1, output_modes=1, seed=3)
# {0: 3.5853792684546093} density -3.565078
```

### Visualization layer

| Field | Value |
|---|---|
| Description | Static matplotlib diagrams of graphs, patterns, circuits, dependencies and compilation provenance, and the staged interactive `GraphView` (Cytoscape.js/Plotly.js HTML, or matplotlib export). Numbers shown are computed in Python. Interactive visualization is covered in Section 30; static plots in [visualization](user-guide/visualization.md). |
| Python API | `Pattern.draw(**kwargs)`, `Circuit.draw(**kwargs)`, `pg.visualize_compilation(circuit, compiled, *, trace=None, output=None, layout="spring", style="default")`, `photographiq.visualization.draw_graph/draw_pattern/draw_circuit/draw_commands/draw_dependencies`, `photographiq.interactive.GraphView` |
| Implementation | `src/photographiq/visualization.py`, `src/photographiq/interactive/` |
| Backends | Backend-neutral (reads patterns and `Result` objects) |
| CPU / GPU | CPU only |
| Assumptions | Layout coordinates are visualization metadata and never change semantics |
| Tests | `tests/integration/test_visualization.py`, `tests/visualization/` |
| Status | **Implemented but experimental** (rendering and data-plumbing tests; no independent reference applies) |

Example: see `tests/integration/test_visualization.py` and `docs/user-guide/visualization.md`.

### PhotoGraphiQML as an extension

| Field | Value |
|---|---|
| Description | PhotoGraphiQML reuses `photographiq.Parameter` for trainable symbols and lowers signed-X MuTA layers to public PhotoGraphiQ commands (`lower_muta_to_gkp`) after a capability audit, then executes them on Fock backends with GKP resources and decoding. It does not modify core semantics. |
| Python API | `pml.lower_muta_to_gkp(model, parameters=None, *, config=None)`, `pml.GKPPhysicalConfig(cutoff=48, peak_width=0.6, envelope=0.6, peaks=6, grid_points=2049, decoder="nearest", backend="piquasso-fock", max_dimension=100000, max_matrix_bytes=256000000)`, `pml.PhysicalMuTA(n_wires=1, n_layers=1, *, physical_config=None, measurement_family="X", **kwargs)` |
| Implementation | `src/photographiqml/lowering.py`, `src/photographiqml/physical.py`; contract in `docs/development/photographiqml-contract.md` |
| Backends | Fock backends for the physical path |
| CPU / GPU | CPU only |
| Assumptions | Signed-X family only on the physical path; finite-energy GKP, nearest-cell decoding |
| Tests | `tests/ml/lowering/`, `tests/ml/cross_layer/`, `tests/native/test_native_fock_cross_layer.py::test_physical_muta_conditional_run_matches_piquasso` |
| Status | **Implemented but experimental** (summary; per-feature statuses of the lowering, audit and physical execution are in Section 28) |

Example:

```python
import photographiq as pg, photographiqml as pml
print(len(pg.__all__), len(pml.__all__))
print(pml.Parameter is pg.Parameter)
# 85 16
# True
```

## 3. Package/module hierarchy

This section is a reference listing of every module, one line each; feature statuses are given in the feature sections. Two infrastructure helpers that are not covered elsewhere are documented as features at the end.

### Module table: `photographiq`

| Module | Purpose |
|---|---|
| `photographiq` (`__init__.py`) | Public namespace (85 names in `__all__`), `__version__ = "0.3.1"` |
| `photographiq._optional` | `LazyModule`: lazy optional imports with installation hints |
| `photographiq._native` (compiled) | C++ extension with components `gaussian`, `fock`, `sampling` (built from `cpp/`) |
| `photographiq.analysis` | `GaussianChannel`, `gaussian_channel`: unconditional (S, N) channel of fixed-angle affine Gaussian patterns |
| `photographiq.autodiff` | Optional JAX execution with fixed topology and explicit gradient semantics (not imported by `import photographiq`) |
| `photographiq.backends` | Backend package; exports `BaseBackend`, `GaussianBackend`, `PiquassoBackend`, ... |
| `photographiq.backends.base` | `BaseBackend` contract: labelled modes, destructive measurement, capability checks |
| `photographiq.backends.gaussian` | `GaussianBackend`: NumPy reference Gaussian engine with exact conditioning |
| `photographiq.backends.piquasso` | `PiquassoBackend`: Piquasso Gaussian gates with the exact NumPy measurement adapter; `import_state` |
| `photographiq.backends.fock` | `PiquassoFockBackend`, `FockResultState`: pure-Fock trajectories with total-photon truncation |
| `photographiq.backends.mixed_fock` | `MixedFockBackend`: density-matrix trajectories, thermal loss, noisy homodyne, instruments |
| `photographiq.backends.native_gaussian` | `NativeGaussianBackend` plus native diagnostics and batched kernels (`batch_apply_symplectic`, `batch_condition_homodyne`, ...) |
| `photographiq.backends.native_fock` | `NativeFockBackend`, `NativeMixedFockBackend` (C++ engines; adds `cross_kerr`) |
| `photographiq.cudaq_dynamics` | Optional CUDA-Q Dynamics adapter at the Hamiltonian level (not a backend); `photographiq.backends.cudaq_dynamics` is a compatibility shim |
| `photographiq.calculus` | Qubit measurement calculus, Clifford group, ZX diagrams, CV command view, Gaussian graphical calculus |
| `photographiq.calculus.qubit` | DKP qubit calculus: `QubitPattern`, standardization, signal shifting, flow-to-pattern, exact branch simulation |
| `photographiq.calculus.clifford` | 24-element single-qubit Clifford group, local complementation, LC equivalence |
| `photographiq.calculus.zx` | Scalar-exact qubit ZX diagrams and a sound rewrite subset |
| `photographiq.calculus.cv` | Command-level view of CV patterns and ideal-limit Gaussian identity checks |
| `photographiq.calculus.gaussian_graph` | `GaussianGraph`: restricted Gaussian graphical calculus for pure Gaussian states (Menicucci, Flammia and van Loock 2011); not CV-ZX |
| `photographiq.capabilities` | `required_features`, `preflight`: allocation-free pattern analysis |
| `photographiq.commands` | Command dataclasses (`Prepare`, `Entangle`, `Measure`, gates, `Signal`, `Output`, ...) |
| `photographiq.compiler` | `Circuit`, `compile_circuit`, `decompose_symplectic`, `CompilationStep`, `CompilationTrace` |
| `photographiq.convergence` | `cutoff_convergence`, `CutoffStudy`: reproducible cutoff comparisons on any Fock backend (`backend=`, default `piquasso-fock`) |
| `photographiq.cvflow` | `certify_cv_flow`, `flow_pattern`: supplied-order real-linear CV-flow certificate and corrections |
| `photographiq.encoded` | GKP code (`GKPCode`), decoders, `LogicalPauliFrame`, `PhysicalGKPReadout`, `modular_effects`, `multimode_readout` |
| `photographiq.expressions` | `Expr`, `Parameter`, `Outcome`, `CallableExpression`, `sin`, `cos`, `exp`, `atan2`, `resolve` |
| `photographiq.flow` | Command DAG scheduling (`dependency_graph`, `topological_schedule`); not a determinism certificate |
| `photographiq.flows` | `analyze` plus causal flow, gflow, Pauli flow (qubit) and CV flow search/verification |
| `photographiq.flows.causal` / `.gflow` / `.pauli` / `.cv` / `.result` / `._common` / `._exact` | One flow notion per module; `FlowAnalysisResult`; shared GF(2) helpers; exact rational CV-flow re-check |
| `photographiq.fock_analysis` | `fidelity`, `trace_distance`, `wigner`, `WignerGrid`, `quadrature_moment` |
| `photographiq.fock_measurements` | Hermite wavefunctions and ideal homodyne projection of Fock amplitudes |
| `photographiq.gaussian` | Phase-space matrices: `omega`, `rotation`, `squeezing`, `cz`, `beamsplitter`, `teleportation_matrix`, `wire_channel` |
| `photographiq.gkp` | `GKPResource`, `superposition`, `logical_displacement`, `decode_shift`, `stabilizers`, `correction_pattern` |
| `photographiq.gkp_studies` | `SoftDecisionDecoder`, `GKPMeasurementStudy`, `measurement_convergence` |
| `photographiq.gpu` | Optional CuPy batched Gaussian kernels (`device.py`, `gaussian.py`) and batched Fock local gates (`fock.py`); not imported by `import photographiq` |
| `photographiq.graph` | `CVGraph` (alias `ClusterState`) |
| `photographiq.instruments` | `MeasurementInstrument`, `destructive_instrument`, `density_matrix` |
| `photographiq.interactive` | Staged interactive views: `graph_view`, `layouts`, `overlays`, `plots`, `static`, `trace`, `editor`, `_html`, `_template` |
| `photographiq.ir` | `Model`, `Plane`, `QubitMeasurement`, `HomodyneSpec`, `OpenGraph` |
| `photographiq.measurements` | `Homodyne`, `Heterodyne`, `Generaldyne`, `PhotonNumber`, `MeasurementProtocol` |
| `photographiq.native` | Loader for `photographiq._native`: `AVAILABLE`, `components()`, `build_info()`, `require()` |
| `photographiq.non_gaussian` | `photon_subtraction`, `resource_injection`, `cat_injection`, `cubic_injection` |
| `photographiq.pattern` | `Pattern` |
| `photographiq.protocols` | `wire`, `identity`, `displacement`, `rotation`, `squeeze`, `gaussian`, `entangling`, `teleportation`, `adaptive` |
| `photographiq.registry` | `BackendInfo`, `BACKENDS`, `available_backends`, `describe_backend`, `create_backend`, `select_backend`, `missing_features` |
| `photographiq.resources` | `CubicPhaseResource`, `CatResource` (built at the execution cutoff) |
| `photographiq.sampling` | Boson sampling and GBS: `linalg`, `unitaries`, `boson_sampling`, `gbs` |
| `photographiq.serialization` | Versioned allowlisted JSON (`dumps`, `loads`); never eval or pickle |
| `photographiq.simulator` | `simulate`, `run_shots` (`sample`), `Result`, `ShotResult` |
| `photographiq.states` | `GaussianState`, `GaussianInput`, `FockInput`, `FockSuperposition`, `FockDensityMatrix` |
| `photographiq.synthesis` | Experimental product-formula synthesis: `quadrature_polynomial`, `synthesize_kerr`, `SynthesisReport` |
| `photographiq.visualization` | Matplotlib diagrams: `draw_graph`, `draw_pattern`, `draw_circuit`, `draw_commands`, `draw_dependencies`, `visualize_compilation` |

### Module table: `photographiqml`

| Module | Purpose |
|---|---|
| `photographiqml` (`__init__.py`) | Public namespace (16 names), `__version__ = "0.2.0"` |
| `photographiqml.ansatz.muta` | `MuTA`: logical MuTA geometry and flow with explicit representation boundaries |
| `photographiqml.ansatz.triangle` | `TriangleNeuron`: paper-layer geometry in semantic coordinates |
| `photographiqml.diagnostics` | Representation-specific diagnostics |
| `photographiqml.expressivity` | Exact small logical Pauli Lie closure and local pure-state Fisher diagnostics |
| `photographiqml.gkp` | `GKPBridge`: controlled GKP resource bridge |
| `photographiqml.kernels` | `MuTAKernel`: feature map realized through a logical MuTA layer |
| `photographiqml.logical` | Ideal qubit target operations (not a CV simulator) |
| `photographiqml.lowering` | `GKPPhysicalConfig`, `lower_muta_to_gkp`: capability-audited lowering to PhotoGraphiQ commands |
| `photographiqml.models` | `MuTAClassifier`, `MuTARegressor`, `QuantumInstrumentModel` |
| `photographiqml.parameters` | Trainability metadata around PhotoGraphiQ expressions |
| `photographiqml.physical` | `PhysicalMuTA`, `PhysicalMuTAResult`, `compare_logical_physical` |
| `photographiqml.physical_training` | `DiscreteSearch`: bounded categorical search |
| `photographiqml.training` | `Trainer`, `History`: deterministic objective training |
| `photographiqml.validation` | Small independent logical validation and optional MentPy comparison |

### Module table: compiled sources `cpp/`

| Path | Purpose |
|---|---|
| `cpp/src/core/symplectic.cpp` | Shared symplectic helpers |
| `cpp/src/gaussian/operations.cpp`, `conditioning.cpp`, `diagnostics.cpp`, `batch.cpp` | In-place Gaussian gates, Schur-complement conditioning, spectral diagnostics, OpenMP batched kernels |
| `cpp/src/fock/basis.cpp`, `gates.cpp`, `kernels.cpp` | Total-photon basis and ranking, Gaussian/cubic/phase gate matrices, pure and density kernels |
| `cpp/src/sampling/permanent.cpp`, `hafnian.cpp`, `torontonian.cpp` | Matrix functions for boson sampling and GBS |
| `cpp/src/bindings/module.cpp`, `gaussian_bindings.cpp`, `fock_bindings.cpp`, `sampling_bindings.cpp` | pybind11 bindings |
| `cpp/tests/test_gaussian.cpp`, `test_fock.cpp`, `test_sampling.cpp` | C++ unit tests |

### Lazy optional imports

| Field | Value |
|---|---|
| Description | Optional packages are wrapped in `LazyModule` objects that import on first attribute access and raise `ImportError` naming the extra to install. |
| Python API | `photographiq._optional.LazyModule(name, extra)`, `.available` (property), `.load()` |
| Implementation | `src/photographiq/_optional.py` |
| Backends | Used by the Piquasso-backed backends |
| CPU / GPU | CPU only |
| Assumptions | Private module (leading underscore); not part of the public API |
| Tests | `tests/cross_backend/test_registry_planner.py::test_registry_import_is_lazy` |
| Status | **Implemented but experimental** (behavioural test only) |

Example:

```python
from photographiq._optional import piquasso
print(piquasso.available)
# True
```

### Native extension loader

| Field | Value |
|---|---|
| Description | Loads `photographiq._native` (or a build at `PHOTOGRAPHIQ_NATIVE_PATH`) and reports which components were compiled. |
| Python API | `photographiq.native.AVAILABLE`, `photographiq.native.load_error`, `photographiq.native.components()`, `photographiq.native.build_info()`, `photographiq.native.require(component)` |
| Implementation | `src/photographiq/native.py` |
| Backends | `native-gaussian`, `native-fock`, `native-mixed-fock`, native sampling |
| CPU / GPU | CPU (C++/OpenMP) |
| Assumptions | `require` raises `ImportError` when a component is missing; the registry then reports the backend as unavailable |
| Tests | `tests/cross_backend/test_registry_planner.py::test_native_names_resolve_or_raise_import_error` |
| Status | **Implemented but experimental** (behavioural test only) |

Example:

```python
from photographiq import native
print(native.AVAILABLE, native.components())
# True ('gaussian', 'fock', 'sampling')
```

## 4. Public APIs

The top-level `photographiq` namespace exports exactly the 85 names below (`photographiq.__all__`), grouped by purpose. `atan2`, `HomodyneSpec`, `QubitMeasurement` and `Plane` are public but live only in their modules (`photographiq.expressions`, `photographiq.ir`). `photographiq.gpu` and `photographiq.autodiff` are not imported by `import photographiq` and must be imported explicitly. This section is a reference listing; statuses are in the feature sections.

### Name table: top-level `photographiq`

| Purpose | Names |
|---|---|
| Graphs and IR | `CVGraph`, `ClusterState` (alias of `CVGraph`), `OpenGraph`, `Model`, `Pattern` |
| Commands | `Prepare`, `PrepareResource`, `Entangle`, `Displace`, `Rotate`, `Squeeze`, `BeamSplitter`, `QuadraticPhase`, `Loss`, `CubicPhase`, `Kerr`, `PhotonAdd`, `PhotonSubtract`, `Measure`, `Signal`, `Output` |
| Measurements | `Homodyne`, `Heterodyne`, `Generaldyne`, `PhotonNumber`, `MeasurementInstrument` |
| States and resources | `GaussianInput`, `GaussianState`, `FockInput`, `FockSuperposition`, `FockDensityMatrix`, `CatResource`, `CubicPhaseResource`, `GKPResource` |
| Classical expressions | `Parameter`, `Outcome`, `CallableExpression`, `sin`, `cos`, `exp` |
| Execution and results | `simulate`, `run_shots`, `sample` (alias of `run_shots`), `Result`, `ShotResult` |
| Backend registry | `available_backends`, `describe_backend`, `select_backend` |
| Compilation | `Circuit`, `compile_circuit`, `CompilationStep`, `CompilationTrace`, `visualize_compilation` |
| Analysis | `GaussianChannel`, `gaussian_channel`, `CutoffStudy`, `cutoff_convergence`, `fidelity`, `trace_distance`, `wigner`, `WignerGrid` |
| GKP encoding | `GKPCode`, `BaseGKPDecoder`, `NearestCellDecoder`, `SoftDecisionDecoder`, `LogicalDecodeResult`, `LogicalPauliFrame`, `PhysicalGKPReadout`, `IdealLogicalXYMeasurement`, `LogicalMeasurementSynthesis`, `GKPMeasurementStudy`, `measurement_convergence`, `modular_effects`, `multimode_readout` |
| Subpackages and modules | `protocols`, `cvflow`, `non_gaussian`, `gkp`, `synthesis`, `calculus`, `flows`, `interactive`, `ir`, `registry`, `sampling` |

### Name table: top-level `photographiqml`

| Purpose | Names |
|---|---|
| Logical models | `MuTA`, `TriangleNeuron`, `MuTAKernel`, `MuTAClassifier`, `MuTARegressor`, `QuantumInstrumentModel` |
| Training | `Trainer`, `History`, `DiscreteSearch` |
| Physical GKP path | `GKPBridge`, `GKPPhysicalConfig`, `lower_muta_to_gkp`, `PhysicalMuTA`, `PhysicalMuTAResult`, `compare_logical_physical` |
| Re-export | `Parameter` (the same object as `photographiq.Parameter`) |

Example (verifies the namespace sizes and the shared `Parameter`):

```python
import photographiq as pg, photographiqml as pml
print(len(pg.__all__), len(pml.__all__))
print(pml.Parameter is pg.Parameter)
# 85 16
# True
```

## 5. Graph construction

CV resources are weighted, labelled undirected graphs whose nodes carry momentum resource squeezing and whose edges are weighted CZ interactions. `Pattern(graph)` turns a graph into `Prepare` and `Entangle` commands; `OpenGraph` gives a typed view for flow analysis and visualization.

### `CVGraph` from a NetworkX graph, with incremental editing

| Field | Value |
|---|---|
| Description | Wraps an undirected simple NetworkX graph (edge attribute `weight`, node attribute `squeezing`) with ordered input and output labels; `add_node` and `add_edge` validate a trial copy before committing. |
| Python API | `pg.CVGraph(graph=None, *, squeezing=1.0, inputs=(), outputs=(), ideal=False)` (`squeezing` scalar or dict per node), `.add_node(node, squeezing=1.0)`, `.add_edge(u, v, weight=1.0)`, `.nodes`, `.ancillas`, `.copy()`, `.validate()`; alias `pg.ClusterState` |
| Implementation | `src/photographiq/graph.py` |
| Backends | Backend-neutral |
| CPU / GPU | CPU only |
| Assumptions | Directed graphs, multigraphs, self-loops (local quadratic phases, not resource edges), nonfinite or negative squeezing and zero/nonfinite weights raise `ValueError`; edges default to weight 1 |
| Tests | `tests/unit/test_core.py::test_graph_labels_and_families`, `tests/graphix_reference/test_labelled_structures.py::test_arbitrary_cv_labels_with_explicit_graphix_bijection` (Graphix structure) |
| Status | **Implemented and validated** |

Example:

```python
import networkx as nx, photographiq as pg
g = nx.Graph(); g.add_edge("left", "right", weight=0.7)
graph = pg.CVGraph(g, squeezing={"left": 0.7, "right": 1.2}, inputs=("left",))
print(graph, graph.ancillas)
print(graph.adjacency())
h = pg.CVGraph().add_node("a", squeezing=0.5).add_node("b").add_edge("a", "b", weight=-0.3)
try:
    h.add_edge("a", "a")
except ValueError as exc:
    print("ValueError:", exc)
# CVGraph(nodes=2, edges=1, inputs=1, outputs=0) ('right',)
# [[0.  0.7]
#  [0.7 0. ]]
# ValueError: Self edges are local quadratic phases, not resource edges
```

### `CVGraph.from_adjacency`

| Field | Value |
|---|---|
| Description | Builds a labelled resource from a real symmetric weighted adjacency matrix with zero diagonal. |
| Python API | `pg.CVGraph.from_adjacency(adjacency, labels=None, **kwargs)` |
| Implementation | `src/photographiq/graph.py` |
| Backends | Backend-neutral |
| CPU / GPU | CPU only |
| Assumptions | Symmetry tolerance 1e-12 absolute; labels default to `range(n)`; non-square, nonfinite, asymmetric or nonzero-diagonal input raises `ValueError` |
| Tests | `tests/unit/test_core.py::test_invalid_adjacency`, `tests/analytical/test_primitives.py::test_weighted_graph_nullifiers` (closed-form covariance of a weighted 3-node graph on `gaussian` and `piquasso`) |
| Status | **Implemented and validated** |

Example:

```python
import photographiq as pg
graph = pg.CVGraph.from_adjacency([[0, -0.4], [-0.4, 0]], labels=["in", (1, 2)])
print(graph.nodes)
try:
    pg.CVGraph.from_adjacency([[0, 1], [0, 0]])
except ValueError as exc:
    print("ValueError:", exc)
# ('in', (1, 2))
# ValueError: Adjacency must be symmetric with zero diagonal
```

### Standard topologies

| Field | Value |
|---|---|
| Description | Constructors for line, ring, star, rectangular grid, square grid and balanced tree resources, plus `temporal` (a time-labelled line) and `dual_rail` (a 2 x bins ladder). |
| Python API | `pg.CVGraph.line(modes, **kw)`, `.ring(modes, **kw)` (modes >= 3), `.star(leaves, **kw)`, `.rectangular(rows, columns, **kw)`, `.square(size, **kw)`, `.tree(branching, height, **kw)`, `.temporal(bins, **kw)`, `.dual_rail(bins, **kw)` |
| Implementation | `src/photographiq/graph.py` |
| Backends | Backend-neutral |
| CPU / GPU | CPU only |
| Assumptions | Grid nodes are `(row, column)` tuples; `temporal` and `dual_rail` are topologies only, not optical hardware schedules or calibrated circuits |
| Tests | `tests/unit/test_core.py::test_graph_labels_and_families` (calls `validate()` on line, ring, star, tree, rectangular and dual_rail; no edge-set check). `tests/graphix_reference/test_labelled_structures.py::test_label_aware_graph_families_and_measurement_order` checks path, cycle, star and grid graphs against Graphix, but it builds them with NetworkX generators passed to `CVGraph(graph)`, not with these constructors. `square` and `temporal` have no dedicated test. |
| Status | **Implemented but experimental** (no test compares the constructors' edge sets with an independent reference) |

Example (node and edge counts):

```python
import photographiq as pg
graphs = [pg.CVGraph.line(4), pg.CVGraph.ring(4), pg.CVGraph.star(3), pg.CVGraph.rectangular(2, 3),
          pg.CVGraph.square(2), pg.CVGraph.tree(2, 2), pg.CVGraph.temporal(3), pg.CVGraph.dual_rail(3)]
print("; ".join(f"{len(g.nodes)} {g.network.number_of_edges()}" for g in graphs))
# 4 3; 4 4; 4 3; 6 7; 4 4; 7 6; 3 2; 6 7
```

### Symbolic resources and ideal graphs

| Field | Value |
|---|---|
| Description | Node squeezing and edge weights may be `Parameter` expressions bound at execution time; `ideal=True` marks a symbolic infinite-squeezing specification that cannot be turned into a physical pattern. |
| Python API | `pg.CVGraph(..., squeezing=pg.Parameter("r"))`, `graph.network[u][v]["weight"] = pg.Parameter("g")`, `.adjacency(parameters=None)`, `pg.CVGraph(..., ideal=True)` |
| Implementation | `src/photographiq/graph.py`, `src/photographiq/expressions.py` |
| Backends | All (parameters are resolved before the backend sees them) |
| CPU / GPU | CPU only |
| Assumptions | `Pattern(graph)` with `ideal=True` raises `ValueError`; unbound parameters raise `ValueError` in `simulate` |
| Tests | `tests/regression/test_inputs_and_frames.py::test_symbolic_flow_coefficients_are_frozen` (analytic channel with a bound symbolic weight), `tests/unit/test_core.py::test_parameters_and_callable_dependency`, `::test_invalid_state_and_requests` |
| Status | **Implemented and validated** |

Example:

```python
import photographiq as pg
g = pg.CVGraph.line(2, squeezing=pg.Parameter("r"))
g.network[0][1]["weight"] = pg.Parameter("g")
print(g.adjacency({"g": 0.6}))
print(sorted(pg.Pattern(g).parameters))
try:
    pg.Pattern(pg.CVGraph.line(2, ideal=True))
except ValueError as exc:
    print("ValueError:", exc)
# [[0.  0.6]
#  [0.6 0. ]]
# ['g', 'r']
# ValueError: Ideal graph is symbolic; choose finite squeezing to build a physical pattern
```

### Adjacency and nullifiers

| Field | Value |
|---|---|
| Description | `adjacency` returns the weighted adjacency A in node order; `nullifiers` returns rows of delta_i = p_i - sum_j A_ij q_j in interleaved quadrature coordinates, for physical and ideal graphs. |
| Python API | `CVGraph.adjacency(parameters=None)`, `CVGraph.nullifiers(parameters=None)` |
| Implementation | `src/photographiq/graph.py` |
| Backends | Backend-neutral (check against any Gaussian backend state) |
| CPU / GPU | CPU only |
| Assumptions | For resource squeezing r per node, the nullifier covariance is diag(exp(-2 r_i)) |
| Tests | `tests/analytical/test_primitives.py::test_weighted_graph_nullifiers` (closed form) |
| Status | **Implemented and validated** |

Example:

```python
import numpy as np, photographiq as pg
graph = pg.CVGraph.line(3, squeezing=1.0)
V = pg.simulate(pg.Pattern(graph)).state.covariance
print(np.round(graph.nullifiers() @ V @ graph.nullifiers().T, 6))
print(round(np.exp(-2), 6))
# [[0.135335 0.       0.      ]
#  [0.       0.135335 0.      ]
#  [0.       0.       0.135335]]
# 0.135335
```

### Graph-to-pattern conversion and the pattern connectivity view

| Field | Value |
|---|---|
| Description | `Pattern(graph)` emits one `Prepare(node, squeezing)` per non-input node and one `Entangle(u, v, weight)` per edge; `Pattern.graph` reconstructs the entangling connectivity of any pattern, adding the weights of repeated CZ commands. |
| Python API | `pg.Pattern(graph)`, `Pattern.graph`, `Pattern.outputs` |
| Implementation | `src/photographiq/pattern.py` |
| Backends | Backend-neutral |
| CPU / GPU | CPU only |
| Assumptions | Input nodes are supplied externally (not prepared); `Pattern.graph` aggregates repeated CZ edges and is not a temporal simulation; if the graph declares outputs, the surviving modes must match them (else `ValueError` asking for an `Output` command) |
| Tests | `tests/graphix_reference/test_labelled_structures.py::test_label_aware_graph_families_and_measurement_order`, `tests/graphix_reference/test_structure.py::test_wire_connectivity_io_measurement_order_and_domains` (connectivity vs Graphix) |
| Status | **Implemented and validated** |

Example:

```python
import photographiq as pg
p = pg.Pattern(pg.CVGraph.line(2)).append(pg.Entangle(0, 1, 0.5))
print(p.graph.adjacency())
# [[0.  1.5]
#  [1.5 0. ]]
```

### Typed open graphs for CV and qubit models

| Field | Value |
|---|---|
| Description | `OpenGraph` builds typed graphs directly: CV models carry `HomodyneSpec(shear)` measurements and squeezing, qubit models carry `QubitMeasurement(plane, angle, pauli)` with unweighted edges. It is the input of `photographiq.flows.analyze` and `photographiq.interactive.GraphView`. |
| Python API | `pg.OpenGraph(model, graph, inputs, outputs, measurements, squeezing)`, `pg.Model.CV_GAUSSIAN` / `CV_NONGAUSSIAN` / `QUBIT` / `GKP_LOGICAL`, `photographiq.ir.QubitMeasurement(plane="XY", angle=0.0, pauli=None)`, `photographiq.ir.Plane` |
| Implementation | `src/photographiq/ir.py` |
| Backends | Not executable except through `to_cvgraph()` for CV models |
| CPU / GPU | CPU only |
| Assumptions | Qubit observables: XY cos(a)X + sin(a)Y, YZ cos(a)Z + sin(a)Y, XZ cos(a)Z + sin(a)X (outcome 0 is the +1 eigenstate, as in Graphix 0.4); weighted qubit graphs raise `ValueError` |
| Tests | `tests/unit/test_ir_open_graph.py::test_qubit_model_semantics_are_enforced`, `::test_cvgraph_conversion_preserves_weights_squeezing_and_roles` |
| Status | **Implemented but experimental** |

Example:

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, QubitMeasurement
og = OpenGraph(Model.QUBIT, nx.path_graph(3), (0,), (2,),
               {0: QubitMeasurement("XY", 0.2), 1: QubitMeasurement("XY", 0.0, pauli="X")})
print(og.model.value, og.measured, og.to_dict()["measurements"]["1"])
try:
    OpenGraph(Model.QUBIT, nx.Graph([(0, 1, {"weight": 0.5})]), (0,), (1,))
except ValueError as exc:
    print("ValueError:", exc)
# qubit (0, 1) {'plane': 'XY', 'angle': 0.0, 'pauli': 'X'}
# ValueError: qubit graphs are unweighted; edge (0, 1)
```

## 6. Gaussian state preparation

Gaussian inputs are described per mode by `GaussianInput(mean, covariance)` or jointly by a correlated `GaussianState`. Pattern inputs that are not supplied default to vacuum; resource modes created by `Prepare(node, squeezing=r)` are momentum-squeezed vacua.

### Vacuum default inputs

| Field | Value |
|---|---|
| Description | Every pattern input without an entry in `inputs=` is prepared as vacuum (mean 0, V = I). |
| Python API | `pg.simulate(pattern, inputs=None, ...)`; `pg.GaussianInput()` |
| Implementation | `src/photographiq/simulator.py`, `src/photographiq/states.py` |
| Backends | All executing backends |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) for native backends |
| Assumptions | Vacuum covariance I in hbar = 2 units |
| Tests | `tests/analytical/test_primitives.py::test_vacuum_squeeze_displacement_rotation` (closed form, NumPy and Piquasso), `tests/analytical/test_measurements.py::test_vacuum_measurement_statistics` |
| Status | **Implemented and validated** |

Example:

```python
import photographiq as pg
r = pg.simulate(pg.Pattern(inputs=(0,)), seed=0)
print(r.state.quadrature(0))
# (0.0, 1.0)
```

### Resource-squeezed preparation

| Field | Value |
|---|---|
| Description | `Prepare(node, squeezing=r)` creates a fresh mode in the momentum-squeezed vacuum with V = diag(exp(2r), exp(-2r)), the canonical CV cluster resource. An explicit `state=` overrides the squeezing. |
| Python API | `pg.Prepare(node, squeezing=1.0, state=None)` |
| Implementation | `src/photographiq/commands.py`, `src/photographiq/backends/gaussian.py` (`GaussianInput.squeezed(-r)`), `src/photographiq/backends/fock.py` |
| Backends | All executing backends (Fock backends represent it within the cutoff) |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | Finite squeezing r >= 0; the `Squeeze` gate with positive r squeezes q, whereas resource preparation uses -r |
| Tests | `tests/analytical/test_primitives.py::test_weighted_graph_nullifiers` (closed form), `tests/fock/test_homodyne_validation.py` (Fock representation) |
| Status | **Implemented and validated** (Gaussian backends; Fock representation is experimental) |

Example:

```python
import numpy as np, photographiq as pg
s = pg.simulate(pg.Pattern().append(pg.Prepare("a", squeezing=0.5))).state
print(np.round(s.covariance, 6), np.round([np.exp(1.0), np.exp(-1.0)], 6))
# [[2.718282 0.      ]
#  [0.       0.367879]] [2.718282 0.367879]
```

### Coherent states

| Field | Value |
|---|---|
| Description | `GaussianInput.coherent(alpha)` describes D(alpha)|0> with mean (2 Re alpha, 2 Im alpha) and V = I. |
| Python API | `pg.GaussianInput.coherent(alpha)` |
| Implementation | `src/photographiq/states.py` |
| Backends | All executing backends |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | hbar = 2: <q> = 2 Re alpha, <p> = 2 Im alpha, <n> = |alpha|^2 |
| Tests | `tests/analytical/test_primitives.py::test_beamsplitter_and_loss_agree` (NumPy vs Piquasso), `tests/cross_backend/test_cross_backend_agreement.py::test_gaussian_to_fock_reference_matches_coherent_closed_form` (closed form) |
| Status | **Implemented and validated** |

Example:

```python
import photographiq as pg
c = pg.GaussianInput.coherent(0.3 + 0.2j)
r = pg.simulate(pg.Pattern(inputs=(0,)), inputs={0: c})
print(c.mean, r.state.photon_number(0))
# (0.6, 0.4) 0.13
```

### Squeezed states

| Field | Value |
|---|---|
| Description | `GaussianInput.squeezed(r, angle)` describes a rotated q-squeezed vacuum with principal variances exp(-2r), exp(2r). |
| Python API | `pg.GaussianInput.squeezed(r, angle=0.0)` |
| Implementation | `src/photographiq/states.py` |
| Backends | All executing backends |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | Covariance R(angle) diag(exp(-2r), exp(2r)) R(angle)^T, det V = 1 |
| Tests | `tests/native/test_native_gaussian_diagnostics.py::test_fidelity_closed_forms` (fidelity 1/cosh r), `tests/cross_backend/test_cross_backend_agreement.py::test_homodyne_postselection_density_closed_form` (closed form across backends) |
| Status | **Implemented and validated** |

Example:

```python
import numpy as np, photographiq as pg
s = pg.GaussianInput.squeezed(r=0.4, angle=0.2)
print(np.round(s.covariance, 6))
print(round(np.linalg.det(s.covariance), 12))
# [[ 0.519435 -0.345845]
#  [-0.345845  2.155435]]
# 1.0
```

### Thermal and other mixed Gaussian inputs

| Field | Value |
|---|---|
| Description | Any physical single-mode covariance can be supplied, e.g. thermal V = (2 nbar + 1) I. Mixed inputs (det V != 1) are accepted by Gaussian and mixed-Fock backends and rejected by pure-Fock backends. |
| Python API | `pg.GaussianInput(mean=(0.0, 0.0), covariance=((1.0, 0.0), (0.0, 1.0)))`, `GaussianInput.state(node)` |
| Implementation | `src/photographiq/states.py`; `src/photographiq/backends/mixed_fock.py` (Piquasso Gaussian density matrix) and `src/photographiq/backends/native_fock.py` (Williamson form with summed thermal weights) |
| Backends | `gaussian`, `piquasso`, `native-gaussian`, `piquasso-mixed-fock`, `native-mixed-fock`; `piquasso-fock` and `native-fock` raise `NotImplementedError` "Mixed Gaussian inputs require a mixed Fock backend" |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | V + i Omega >= 0 is enforced when the input becomes a state (`GaussianInput.state()`, `simulate`, `Prepare(state=...)`), raising `ValueError` "Covariance violates the uncertainty relation V+i Omega >= 0"; constructing `GaussianInput(...)` itself does not check it. Native mixed-Fock thermal weights are summed until the omitted tail is below 1e-15 |
| Tests | `tests/analytical/test_gaussian_diagnostics.py::test_thermal_entropy_closed_form_and_symplectic_invariance` (closed form), `tests/native/test_native_mixed_fock_backend.py::test_thermal_gaussian_input_is_geometric_and_physical`, `tests/mixed_fock/test_evolution.py::test_gaussian_thermal_and_noisy_vacuum`, `tests/fock/test_preflight_hardening.py` (pure-Fock rejection) |
| Status | **Implemented and validated** (Gaussian backends; mixed-Fock execution is experimental) |

Example:

```python
import photographiq as pg
th = pg.GaussianInput(covariance=((1.2, 0), (0, 1.2)))
st = th.state(0)
print(st.photon_number(0), round(st.purity(), 6))
try:
    pg.simulate(pg.Pattern(inputs=(0,)), inputs={0: th}, backend="piquasso-fock", cutoff=6)
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
# 0.09999999999999998 0.833333
# NotImplementedError: Mixed Gaussian inputs require a mixed Fock backend
```

### Correlated Gaussian initial states

| Field | Value |
|---|---|
| Description | A multimode `GaussianState` ordered by `pattern.inputs` is injected as `initial_state=`; off-diagonal covariance blocks carry correlations. |
| Python API | `pg.GaussianState(mean, covariance, nodes=())`, `pg.simulate(pattern, initial_state=state)`, `GaussianState.reduced(nodes)` |
| Implementation | `src/photographiq/simulator.py`, `src/photographiq/backends/gaussian.py` (`set_state`) |
| Backends | `gaussian`, `piquasso`, `native-gaussian` only; Fock backends raise `NotImplementedError` "Correlated Gaussian injection requires a Gaussian backend"; combining with `inputs=` raises `ValueError` |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | `nodes` must equal `pattern.inputs` exactly; mean shape (2n,), covariance (2n, 2n) |
| Tests | `tests/regression/test_inputs_and_frames.py::test_correlated_input_and_output_reordering`, `tests/native/test_native_gaussian_backend.py::test_correlated_initial_state_and_run_shots` (native vs NumPy), `tests/cross_backend/test_registry_planner.py::test_select_correlated_gaussian_state_needs_gaussian_backend_class` |
| Status | **Implemented and validated** |

Example:

```python
import numpy as np, photographiq as pg
from photographiq.backends import GaussianBackend
b = GaussianBackend(); b.prepare("a", 0.3); b.prepare("b", 0.4); b.entangle("a", "b", 0.6)
initial = b.get_state()
p = pg.Pattern(inputs=("a", "b")).append(pg.Output(("b", "a")))
r = pg.simulate(p, initial_state=initial)
print(r.state.nodes, np.allclose(r.state.covariance, initial.reduced(("b", "a")).covariance))
try:
    pg.simulate(p, initial_state=initial, backend="native-fock", cutoff=6)
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
# ('b', 'a') True
# NotImplementedError: Correlated Gaussian injection requires a Gaussian backend
```

### Import of external Piquasso Gaussian states

| Field | Value |
|---|---|
| Description | `PiquassoBackend.import_state` converts a Piquasso `GaussianState` with a declared source hbar into PhotoGraphiQ units; the resulting `.get_state()` snapshot can be used as `initial_state`. |
| Python API | `photographiq.backends.PiquassoBackend().import_state(state, nodes, *, source_hbar)` |
| Implementation | `src/photographiq/backends/piquasso.py` |
| Backends | `piquasso` (needs Piquasso) |
| CPU / GPU | CPU (NumPy/Piquasso) |
| Assumptions | mean scaled by sqrt(2/source_hbar), covariance = sigma / source_hbar; the source hbar must be declared (no default) |
| Tests | `tests/piquasso_reference/test_raw_piquasso.py::test_external_state_hbar_conversion` (raw Piquasso) |
| Status | **Implemented and validated** |

Example:

```python
import piquasso as pq, numpy as np
from photographiq.backends import PiquassoBackend
with pq.Program() as prog:
    pq.Q(0) | pq.Squeezing(r=0.3)
native = pq.GaussianSimulator(d=1, config=pq.Config(hbar=1.0)).execute(prog).state
b = PiquassoBackend(); b.import_state(native, ("x",), source_hbar=1.0)
print(np.round(b.get_state().covariance, 6), np.round([np.exp(-0.6), np.exp(0.6)], 6))
# [[0.548812 0.      ]
#  [0.       1.822119]] [0.548812 1.822119]
```

### Gaussian inputs on Fock backends

| Field | Value |
|---|---|
| Description | Pure `GaussianInput`s and resource squeezing are expanded into the truncated Fock basis, so Gaussian preparations can be combined with non-Gaussian operations. |
| Python API | `pg.simulate(pattern, inputs={...: pg.GaussianInput...}, backend="native-fock", cutoff=N)` |
| Implementation | `src/photographiq/backends/fock.py` (`_gaussian_vector`, Piquasso squeezing/phase/displacement), `src/photographiq/backends/native_fock.py` with `cpp/src/fock/gates.cpp` (one-mode Gaussian unitary as R S R D, Section 13) |
| Backends | `piquasso-fock`, `native-fock` (pure inputs only); `piquasso-mixed-fock`, `native-mixed-fock` (pure and mixed) |
| CPU / GPU | CPU (NumPy/Piquasso) or CPU (C++/OpenMP) |
| Assumptions | Total-photon truncation; retained norm below `1 - norm_tolerance` raises `ValueError`; a boundary-population warning is emitted when the top two shells hold more than `boundary_warning`. Mixed inputs use the thermal columns of the unitary, which the 2026-11-12 stability fix made accurate (Section 13). |
| Tests | `tests/native/test_native_fock_backend.py::test_gaussian_pattern_matches_exact_gaussian_moments`, `tests/native/test_native_fock_kernels.py::test_coherent_and_squeezed_vacuum_closed_forms` (closed forms), `tests/non_gaussian/test_execution.py::test_gaussian_fock_gates_against_independent_operators`; `tests/validation/test_val_fock.py::test_number_coherent_squeezed_states_analytic` (mpmath amplitudes and the regularised incomplete gamma function for the retained norm) and `::test_displaced_squeezed_and_mixed_gaussian_inputs` (`expm` on a 90-level space, displaced squeezed pure input on `native-fock` and thermal displaced squeezed input on `native-mixed-fock`, 1e-12); `tests/native/test_native_fock_gaussian_unitary_stability.py::test_thermal_columns_of_displaced_squeezed_input` (60-digit mpmath). Upgraded to validated on 2026-11-12 because of these independent references on the native engines; the Piquasso engines are checked only as a secondary comparison there. |
| Status | **Implemented and validated** (native engines; Piquasso-backed Fock inputs remain a secondary check, and `docs/validation/feature-matrix.md` still marks Fock paths E) |

Example: see Section 8 (`Displace` on `native-fock`).

## 7. Non-Gaussian state preparation

Non-Gaussian inputs and resources are finite Fock descriptions; all of them need a Fock backend (`piquasso-fock`, `native-fock`, `piquasso-mixed-fock`, `native-mixed-fock`) and an explicit `cutoff`. On Gaussian backends they are rejected by preflight with `NotImplementedError` naming the missing capability (`fock_input`, `cat_state`, `cubic_phase`, `multimode_fock`, `mixed_fock`). All Fock paths are labelled experimental in `docs/validation/feature-matrix.md`.

### Photon-number states

| Field | Value |
|---|---|
| Description | `FockInput.number(n)` describes |n>. |
| Python API | `pg.FockInput.number(n)` |
| Implementation | `src/photographiq/states.py`, `src/photographiq/backends/fock.py` |
| Backends | Fock backends; Gaussian backends raise `NotImplementedError` "GaussianBackend does not support fock_input" |
| CPU / GPU | CPU (NumPy/Piquasso) or CPU (C++/OpenMP) |
| Assumptions | n < cutoff; nonnegative integer (bool rejected) |
| Tests | `tests/integration/test_fock.py::test_fock_number_and_counting`, `tests/non_gaussian/test_execution.py::test_raw_piquasso_count_statistics` (raw Piquasso) |
| Status | **Implemented but experimental** |

Example:

```python
import photographiq as pg
r = pg.simulate(pg.Pattern(inputs=(0,)), inputs={0: pg.FockInput.number(2)}, backend="native-fock", cutoff=6)
print(r.state.photon_number(0), r.state.parity())
try:
    pg.simulate(pg.Pattern(inputs=(0,)), inputs={0: pg.FockInput.number(2)})
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
# 2.0 1.0
# NotImplementedError: GaussianBackend does not support fock_input
```

### Arbitrary single-mode Fock amplitudes

| Field | Value |
|---|---|
| Description | `FockInput(amplitudes)` stores sum_n c_n |n> in ascending occupation order. |
| Python API | `pg.FockInput(amplitudes)` |
| Implementation | `src/photographiq/states.py` |
| Backends | Fock backends |
| CPU / GPU | CPU (NumPy/Piquasso) or CPU (C++/OpenMP) |
| Assumptions | Coefficients must be finite and normalized to 1e-10 (no silent normalization); the vector keeps its own truncation when the execution cutoff grows |
| Tests | `tests/non_gaussian/test_execution.py::test_cubic_independent_and_raw_native` (independent matrix and raw Piquasso), `tests/fock/test_states_and_metrics.py` |
| Status | **Implemented but experimental** |

Example:

```python
import photographiq as pg
print(pg.FockInput((2**-0.5, 1j * 2**-0.5)))
try:
    pg.FockInput((1.0, 1.0))
except ValueError as exc:
    print("ValueError:", exc)
# FockInput(levels=2, occupied=2)
# ValueError: Fock input amplitudes must be finite and normalized
```

### Cat states

| Field | Value |
|---|---|
| Description | Normalized finite-Fock projection of |alpha> + parity |-alpha>. `FockInput.cat` fixes the projection cutoff; `CatResource` is rebuilt at the execution cutoff and records the captured weight, which is the form to use in cutoff-convergence studies. |
| Python API | `pg.FockInput.cat(alpha, cutoff, parity=1)`, `pg.CatResource(alpha, parity=1)` (used as `Prepare(node, state=...)` or an input) |
| Implementation | `src/photographiq/states.py`, `src/photographiq/resources.py`, `src/photographiq/backends/fock.py` |
| Backends | Fock backends |
| CPU / GPU | CPU (NumPy/Piquasso) or CPU (C++/OpenMP) |
| Assumptions | Log-space amplitudes for stability; odd cat at alpha = 0 raises `ValueError` |
| Tests | `tests/fock/test_cat_wigner_hardening.py::test_cat_phase_normalization_and_cutoff_convergence`, `::test_wigner_analytic_cats_parity_and_grid_convergence` (analytic Wigner), `tests/cross_backend/test_cross_backend_agreement.py::test_cat_photon_counting_closed_form` (closed form), `tests/fock/test_states_and_metrics.py::test_cat_projection_weight_and_tiny_odd_cat` |
| Status | **Implemented but experimental** |

Example (even cat: parity +1 and <n> = |alpha|^2 tanh |alpha|^2):

```python
import numpy as np, photographiq as pg
p = pg.Pattern().append(pg.Prepare(0, state=pg.CatResource(1.0, parity=1)))
r = pg.simulate(p, backend="native-fock", cutoff=20)
print(round(r.state.parity(), 12), round(r.state.photon_number(0), 9), round(np.tanh(1.0), 9))
# 1.0 0.761594156 0.761594156
```

### Offline photon-added and photon-subtracted inputs

| Field | Value |
|---|---|
| Description | Normalized a^dag|psi> or a|psi> computed on the input description before execution. No heralding probability is implied. |
| Python API | `FockInput.photon_added()`, `FockInput.photon_subtracted()` |
| Implementation | `src/photographiq/states.py` |
| Backends | Fock backends (as inputs) |
| CPU / GPU | CPU only |
| Assumptions | Subtraction from vacuum raises `ValueError` "Photon subtraction has zero norm" |
| Tests | `tests/non_gaussian/test_execution.py::test_ideal_ladders_sqrt_factors_and_failures`, `tests/piquasso_reference/test_fock_convergence.py::test_offline_photon_resources` |
| Status | **Implemented but experimental** |

Example (probabilities of a|psi> and a^dag|psi> for |psi> = sqrt(0.4)|1> + sqrt(0.6)|3>):

```python
import numpy as np, photographiq as pg
src = pg.FockInput((0, np.sqrt(0.4), 0, np.sqrt(0.6)))
print(np.round(np.abs(src.photon_subtracted().amplitudes) ** 2, 6))
print(np.round(np.abs(src.photon_added().amplitudes) ** 2, 6))
# [0.181818 0.       0.818182]
# [0.   0.   0.25 0.   0.75]
```

### Correlated multimode pure Fock resources

| Field | Value |
|---|---|
| Description | `FockSuperposition` is a sparse normalized pure state on ordered occupation tuples, used as `initial_state` or prepared mid-pattern on new nodes by `PrepareResource`. `from_piquasso` imports a public Piquasso `PureFockState`. |
| Python API | `pg.FockSuperposition(occupations, amplitudes)`, `.from_mapping(amplitudes)`, `.number(occupations)`, `.from_piquasso(state)`, `.amplitude_map`; `pg.PrepareResource(nodes, state)` |
| Implementation | `src/photographiq/states.py`, `src/photographiq/backends/fock.py` (`prepare_resource`) |
| Backends | Fock backends (capability `multimode_fock`) |
| CPU / GPU | CPU (NumPy/Piquasso) or CPU (C++/OpenMP) |
| Assumptions | Occupation tuples follow preparation-label order; support with sum(n) >= cutoff raises `ValueError` before allocation; mixed Piquasso states are rejected by `from_piquasso` (`TypeError`) |
| Tests | `tests/fock/test_states_and_metrics.py::test_sparse_multimode_import_reduction_and_counting`, `tests/non_gaussian/test_execution.py::test_homodyne_entangled_conditional_state_and_angle` (closed-form conditional state) |
| Status | **Implemented but experimental** |

Example:

```python
import photographiq as pg
bell = pg.FockSuperposition.from_mapping({(0, 0): 2**-0.5, (1, 1): 2**-0.5})
r = pg.simulate(pg.Pattern(inputs=("a", "b")), initial_state=bell, backend="native-fock", cutoff=8)
print({k: round(v, 6) for k, v in r.state.probabilities.items() if v > 1e-12})
q = pg.Pattern().append(pg.PrepareResource(("x", "y"), pg.FockSuperposition.number((2, 1))))
print(pg.simulate(q, backend="native-fock", cutoff=8).state.photon_number("x"))
# {(0, 0): 0.5, (1, 1): 0.5}
# 2.0
```

### Mixed Fock density matrices

| Field | Value |
|---|---|
| Description | `FockDensityMatrix(matrix, basis)` is a normalized mixed input on an explicit ordered occupation basis; `thermal(mean_photons, cutoff)` builds a truncated, renormalized thermal state. |
| Python API | `pg.FockDensityMatrix(matrix, basis)`, `pg.FockDensityMatrix.thermal(mean_photons, cutoff)`, `.modes` |
| Implementation | `src/photographiq/states.py`, `src/photographiq/backends/mixed_fock.py` |
| Backends | `piquasso-mixed-fock`, `native-mixed-fock`; pure-Fock backends raise `NotImplementedError` "... does not support mixed_fock" |
| CPU / GPU | CPU (NumPy/Piquasso) or CPU (C++/OpenMP) |
| Assumptions | Hermitian, PSD (eigenvalues >= -1e-12), trace one; the thermal construction omits weight (nbar/(1+nbar))**cutoff and must be rebuilt for each cutoff |
| Tests | `tests/mixed_fock/test_state_physicality.py::test_density_preparations_and_permuted_basis`, `tests/mixed_fock/test_evolution.py::test_guards_and_serialization`, `tests/mixed_fock/test_validation.py::test_mixed_resource_preflight_and_impossible_branches` |
| Status | **Implemented but experimental** |

Example:

```python
import photographiq as pg
mixed = pg.FockDensityMatrix([[0.3, 0], [0, 0.7]], ((0,), (1,)))
r = pg.simulate(pg.Pattern(inputs=(0,)), initial_state=mixed, backend="native-mixed-fock", cutoff=6)
print(round(r.state.photon_number(0), 12))
th = pg.FockDensityMatrix.thermal(mean_photons=0.1, cutoff=12)
print(th, round(th.matrix[0][0].real, 6))
try:
    pg.simulate(pg.Pattern(inputs=(0,)), initial_state=mixed, backend="native-fock", cutoff=6)
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
# 0.7
# FockDensityMatrix(modes=1, basis_size=12) 0.909091
# NotImplementedError: NativeFockBackend does not support mixed_fock
```

### Cubic-phase resource states

| Field | Value |
|---|---|
| Description | `CubicPhaseResource(gamma, squeezing=r)` prepares CP(gamma) S(-r)|0> with CP = exp(i gamma q^3/6), the resource consumed by cubic injection. |
| Python API | `pg.CubicPhaseResource(gamma, squeezing=0.0)` |
| Implementation | `src/photographiq/resources.py`, `src/photographiq/backends/fock.py` (prepares squeezed vacuum, then applies `cubic_phase`) |
| Backends | Fock backends (capability `cubic_phase`) |
| CPU / GPU | CPU (NumPy/Piquasso) or CPU (C++/OpenMP) |
| Assumptions | [q, p] = 2i; finite gamma, squeezing >= 0; larger r raises the cutoff cost |
| Tests | `tests/integration/test_fock.py::test_cat_and_cubic_resource`, `tests/cutoff/test_convergence.py::test_strong_cubic_convergence_not_just_unit_norm` |
| Status | **Implemented but experimental** |

Example:

```python
import photographiq as pg
p = pg.Pattern().append(pg.Prepare(0, state=pg.CubicPhaseResource(gamma=0.2, squeezing=0.3)))
r = pg.simulate(p, backend="native-fock", cutoff=30)
print(round(r.state.quadrature(0)[0], 6), r.state.retained_norms[-1] > 0.999)
# 0.0 True
```

### Finite-energy GKP resource states

| Field | Value |
|---|---|
| Description | `GKPResource` is a finite Gaussian comb (peak width, envelope, number of peaks, quadrature grid) projected into the Fock basis at the execution cutoff; `project(cutoff)` also returns the captured weight. Full GKP coverage is in Section 27. |
| Python API | `pg.GKPResource(logical=0, peak_width=0.4, envelope=0.4, peaks=8, grid_points=4097)`, `.wavefunction()`, `.project(cutoff)`, `.fock(cutoff)`; `photographiq.gkp.superposition(alpha, beta, *, cutoff, **resource_options)`; `pg.GKPCode(...).zero()/one()/plus()/minus()/encode(alpha, beta)` |
| Implementation | `src/photographiq/gkp.py`, `src/photographiq/encoded.py` |
| Backends | Fock backends (capability `fock_input`) |
| CPU / GPU | CPU only |
| Assumptions | Logical spacing sqrt(2 pi) at [q, p] = 2i; peaks, grid and cutoff are independent numerical controls; captured weight near one diagnoses the Fock projection only; no fault-tolerance threshold is established |
| Tests | `tests/gkp/test_resources.py`, `tests/gkp/test_independent_convergence.py` |
| Status | **Implemented but experimental** (`docs/user-guide/gkp.md` calls GKP preparation experimental) |

Example:

```python
import photographiq as pg
state, mass = pg.GKPResource(0, peak_width=0.4, envelope=0.4).project(48)
print(len(state.amplitudes), round(mass, 6))
# 48 1.0
```

### Heralded photon subtraction

| Field | Value |
|---|---|
| Description | Physical vacuum-tap beamsplitter plus photon counting on the tap. No postselection is implicit: the count is sampled unless `measurement_outcomes={"count": 1}` selects the heralded branch. |
| Python API | `pg.non_gaussian.photon_subtraction(theta=0.1, *, input_node="in", ancilla="tap", key="count")` |
| Implementation | `src/photographiq/non_gaussian.py` |
| Backends | Fock backends (capability `photon_counting`) |
| CPU / GPU | CPU (NumPy/Piquasso) or CPU (C++/OpenMP) |
| Assumptions | For input |n>, count k has probability binomial(n, k) sin(theta)^(2k) cos(theta)^(2(n-k)); the heralded map includes attenuation and differs from the ideal normalized `a` at finite theta; a postselected result is a conditional branch with probability `exp(log_likelihood)`, not a deterministic operation |
| Tests | `tests/non_gaussian/test_execution.py::test_physical_heralding_probability_and_nonlinear_signal` (closed-form probability), `tests/non_gaussian/test_heralding_limits.py::test_all_count_kraus_amplitudes_and_probabilities`, `::test_weak_tap_approaches_ideal_subtraction_quantitatively` |
| Status | **Implemented but experimental** |

Example:

```python
import numpy as np, photographiq as pg
theta = 0.31
p = pg.non_gaussian.photon_subtraction(theta)
r = pg.simulate(p, backend="native-fock", cutoff=8, inputs={"in": pg.FockInput.number(3)},
                measurement_outcomes={"count": 1})
print(round(np.exp(r.log_likelihood), 9), round(3 * np.sin(theta)**2 * np.cos(theta)**4, 9))
print(round(r.state.photon_number("in"), 9), r.measurement_statistics["count"])
# 0.229638276 0.229638276
# 2.0 {'kind': 'probability', 'value': 0.2296382762840627, 'postselected': True}
```

## 8. Gaussian gates

Gaussian gates are pattern commands applied directly to modes (and, in compiled MBQC patterns, realized by teleportation through the cluster). On Gaussian backends they are exact symplectic updates (`GaussianBackend.transform` checks symplecticity); on Fock backends they are truncated Fock operators, which are experimental. Gate parameters may be `Parameter` or `Outcome` expressions (Section 11).

### Displacement `Displace`

| Field | Value |
|---|---|
| Description | Translates (q, p) by real amounts in hbar = 2 units. |
| Python API | `pg.Displace(node, q=0.0, p=0.0)`, `Pattern.displace(node, *, q=0.0, p=0.0)` |
| Implementation | `src/photographiq/commands.py`, `src/photographiq/backends/gaussian.py`, `src/photographiq/backends/fock.py` (`pq.Displacement` with alpha = (q + i p)/2), `src/photographiq/backends/native_fock.py`, `cpp/src/fock/gates.cpp` |
| Backends | All executing backends |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | alpha = (q + i p)/2; nonfinite values raise `ValueError`; with `frame=True` displacements are deferred (Section 11) |
| Tests | `tests/analytical/test_primitives.py::test_vacuum_squeeze_displacement_rotation` (closed form, NumPy and Piquasso), `tests/non_gaussian/test_execution.py::test_gaussian_fock_gates_against_independent_operators`, `tests/native/test_native_fock_kernels.py::test_coherent_and_squeezed_vacuum_closed_forms` |
| Status | **Implemented and validated** (Gaussian backends; native Fock engines against `expm` in `tests/validation/test_val_fock.py`, Section 13; Piquasso Fock execution experimental) |

Example:

```python
import photographiq as pg
p = pg.Pattern(inputs=(0,)).append(pg.Displace(0, q=0.7, p=-0.2))
for b in ("gaussian", "native-fock"):
    r = pg.simulate(p, backend=b, cutoff=20)
    print(b, [round(x, 8) for x in (r.state.quadrature(0)[0], r.state.quadrature(0, angle=1.5707963267948966)[0])])
# gaussian [0.7, -0.2]
# native-fock [0.7, -0.2]
```

### Phase rotation `Rotate`

| Field | Value |
|---|---|
| Description | Phase-space rotation R(theta) = [[cos, -sin], [sin, cos]] acting on (q, p). |
| Python API | `pg.Rotate(node, angle)` |
| Implementation | `src/photographiq/gaussian.py` (`rotation`), `src/photographiq/backends/gaussian.py`, `src/photographiq/backends/fock.py` (`pq.Phaseshifter`), `src/photographiq/backends/native_fock.py` (exact phase) |
| Backends | All executing backends |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | Angles in radians; exact on Fock backends (number-diagonal phase) |
| Tests | `tests/analytical/test_primitives.py::test_vacuum_squeeze_displacement_rotation` (closed form, NumPy and Piquasso), `tests/non_gaussian/test_sensitivity.py::test_rotation_quadratic_conventions_against_independent_exponential` (Fock matrix exponential, `piquasso-fock`) |
| Status | **Implemented and validated** (Gaussian backends; native Fock engines against `expm` in `tests/validation/test_val_fock.py`, Section 13; Piquasso Fock execution experimental) |

Example:

```python
import numpy as np, photographiq as pg
p = pg.Pattern(inputs=(0,)).extend([pg.Displace(0, q=1.0), pg.Rotate(0, np.pi / 2)])
print(np.round(pg.simulate(p).state.mean, 12))
# [0. 1.]
```

### Single-mode squeezing `Squeeze`

| Field | Value |
|---|---|
| Description | Squeezing gate S(r) = diag(exp(-r), exp(r)); positive r squeezes q. |
| Python API | `pg.Squeeze(node, r)` |
| Implementation | `src/photographiq/gaussian.py` (`squeezing`), `src/photographiq/backends/gaussian.py`, `src/photographiq/backends/fock.py` (`pq.Squeezing`), `src/photographiq/backends/native_fock.py` |
| Backends | All executing backends |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | Resource preparation uses -r (momentum squeezing), the gate uses +r; Fock execution truncates |
| Tests | `tests/analytical/test_primitives.py::test_vacuum_squeeze_displacement_rotation`, `tests/native/test_native_fock_kernels.py::test_gaussian_matrix_elements_are_numerically_stable_at_large_cutoff` (native squeezing matrix vs `expm` of the exact generator in a 700-dimensional space, atol 1e-12 at cutoff 80) |
| Status | **Implemented and validated** (Gaussian backends; native Fock engines against `expm` in `tests/validation/test_val_fock.py`, Section 13; Piquasso Fock execution experimental) |

Example:

```python
import numpy as np, photographiq as pg
p = pg.Pattern(inputs=(0,)).append(pg.Squeeze(0, 0.4))
print(np.round(np.diag(pg.simulate(p).state.covariance), 6), np.round(np.exp([-0.8, 0.8]), 6))
# [0.449329 2.225541] [0.449329 2.225541]
```

### Weighted controlled-Z `Entangle`

| Field | Value |
|---|---|
| Description | Weighted CZ(g) = exp(i g q_u q_v / 2): p_u -> p_u + g q_v and p_v -> p_v + g q_u. It is the resource-edge interaction of `CVGraph`. |
| Python API | `pg.Entangle(u, v, weight=1.0)` |
| Implementation | `src/photographiq/gaussian.py` (`cz`), `src/photographiq/backends/gaussian.py`, `src/photographiq/backends/fock.py` (`pq.GaussianTransform`), `src/photographiq/backends/native_fock.py` |
| Backends | All executing backends |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | u != v required: a backend called directly raises `ValueError` "Repeated gate modes", and inside a pattern `Pattern.validate()` raises `ValueError("Self CZ is invalid: command {i} entangles {u!r} with itself")` (before the fix it reported "Cyclic command dependencies"); Piquasso-Fock applies CZ through a decomposition with truncation after each factor, native Fock applies exact projected matrix elements |
| Tests | `tests/analytical/test_primitives.py::test_weighted_graph_nullifiers` (closed form), `tests/piquasso_reference/test_fock_convergence.py::test_fock_cz_raw_reference_and_cutoff_convergence` (raw Piquasso), `tests/protocols/test_gaussian_postselection.py::test_cz_pair_closed_form_postselection`, `tests/regression/test_validation_messages.py::test_self_cz_is_not_reported_as_a_cycle` |
| Status | **Implemented and validated** (Gaussian backends; native Fock engines against `expm` in `tests/validation/test_val_fock.py`, Section 13; Piquasso Fock execution experimental) |

Example:

```python
import numpy as np, photographiq as pg
p = pg.Pattern(inputs=(0, 1)).extend([pg.Displace(0, q=1.0), pg.Entangle(0, 1, 0.5)])
print(np.round(pg.simulate(p).state.mean, 12))
# [1.  0.  0.  0.5]
```

### Beamsplitter `BeamSplitter`

| Field | Value |
|---|---|
| Description | Real beamsplitter: q_u' = cos(theta) q_u - sin(theta) q_v, q_v' = sin(theta) q_u + cos(theta) q_v (same for p). |
| Python API | `pg.BeamSplitter(u, v, theta)` |
| Implementation | `src/photographiq/gaussian.py` (`beamsplitter`), `src/photographiq/backends/gaussian.py`, `src/photographiq/backends/fock.py` (`pq.Beamsplitter(theta, phi=0)`), `src/photographiq/backends/native_fock.py` (exact per photon-number sector) |
| Backends | All executing backends |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | Passive, number-conserving, so exact within a total-photon cutoff |
| Tests | `tests/analytical/test_primitives.py::test_beamsplitter_and_loss_agree` (NumPy vs Piquasso), `tests/native/test_native_fock_kernels.py::test_hong_ou_mandel`, `::test_passive_gates_are_exact_and_unitary_per_sector` |
| Status | **Implemented and validated** (Gaussian backends; native Fock engines against `expm` in `tests/validation/test_val_fock.py`, Section 13; Piquasso Fock execution experimental) |

Example:

```python
import numpy as np, photographiq as pg
p = pg.Pattern(inputs=(0, 1)).extend([pg.Displace(0, q=1.0), pg.BeamSplitter(0, 1, np.pi / 4)])
print(np.round(pg.simulate(p).state.mean, 6))
# [0.707107 0.       0.707107 0.      ]
```

### Quadratic phase `QuadraticPhase`

| Field | Value |
|---|---|
| Description | exp(i s q^2 / 4): p -> p + s q (a shear). |
| Python API | `pg.QuadraticPhase(node, s)` |
| Implementation | `src/photographiq/backends/gaussian.py` (`quadratic_phase`), `src/photographiq/backends/fock.py` (`pq.QuadraticPhase`), `src/photographiq/backends/native_fock.py` |
| Backends | All executing backends (capability `quadratic_phase`) |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | [q, p] = 2i; Fock execution truncates |
| Tests | `tests/analytical/test_cvflow_multimode.py::test_quadratic_phase_channel_matches_analytic_map_and_simulate` (analytic map), `tests/non_gaussian/test_sensitivity.py::test_rotation_quadratic_conventions_against_independent_exponential` (matrix exponential) |
| Status | **Implemented and validated** (Gaussian backends; native Fock engines against `expm` in `tests/validation/test_val_fock.py`, Section 13; Piquasso Fock execution experimental) |

Example:

```python
import numpy as np, photographiq as pg
p = pg.Pattern(inputs=(0,)).extend([pg.Displace(0, q=1.0), pg.QuadraticPhase(0, 0.3)])
print(np.round(pg.simulate(p).state.mean, 12))
# [1.  0.3]
```

### Loss and thermal noise `Loss`

| Field | Value |
|---|---|
| Description | Attenuation with intensity transmissivity eta and thermal environment occupation n_th: mean -> sqrt(eta) mean, V -> eta V + (1 - eta)(2 n_th + 1) I. A Gaussian channel, not a unitary gate. |
| Python API | `pg.Loss(node, transmissivity, thermal_photons=0.0)` |
| Implementation | `src/photographiq/backends/gaussian.py` (`loss`), `src/photographiq/backends/mixed_fock.py` (explicit Kraus: pure attenuation followed by a quantum-limited amplifier for thermal noise), `src/photographiq/backends/native_fock.py` |
| Backends | `gaussian`, `piquasso`, `native-gaussian`, `piquasso-mixed-fock`, `native-mixed-fock`; pure-Fock backends raise `NotImplementedError` "... does not support loss" |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | 0 <= eta <= 1, n_th >= 0; the mixed-Fock thermal amplifier tail is truncated at the cutoff |
| Tests | `tests/analytical/test_primitives.py::test_beamsplitter_and_loss_agree` (vs Piquasso), `tests/mixed_fock/test_loss_channels.py::test_binomial_channel_all_limits`, `::test_thermal_loss_against_environment_dilation` (environment-dilation reference), `tests/native/test_native_mixed_fock_backend.py::test_loss_matches_exact_gaussian_moments` |
| Status | **Implemented and validated** (Gaussian backends; `native-mixed-fock` loss against a beam-splitter dilation, Section 14; `piquasso-mixed-fock` experimental) |

Example:

```python
import numpy as np, photographiq as pg
p = pg.Pattern(inputs=(0,)).append(pg.Loss(0, transmissivity=0.6, thermal_photons=0.2))
print(np.round(pg.simulate(p).state.covariance, 6))
print(round(pg.simulate(p, backend="native-mixed-fock", cutoff=16).state.photon_number(0), 9), 0.4 * 0.2)
try:
    pg.simulate(p, backend="native-fock", cutoff=8)
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
# [[1.16 0.  ]
#  [0.   1.16]]
# 0.08 0.08000000000000002
# NotImplementedError: NativeFockBackend does not support loss
```

### Arbitrary symplectic transform (backend method)

| Field | Value |
|---|---|
| Description | `GaussianBackend.transform(nodes, matrix)` applies any real symplectic matrix to the listed modes. There is no Pattern command for it; patterns use the gate commands above or `Circuit.gaussian` (compiled). |
| Python API | `photographiq.backends.GaussianBackend().transform(nodes, matrix)` |
| Implementation | `src/photographiq/backends/gaussian.py`, `src/photographiq/gaussian.py` (`embed`, `omega`) |
| Backends | `GaussianBackend` and `NativeGaussianBackend` objects (C++ override); `PiquassoBackend` inherits the NumPy `transform` unchanged (Piquasso is not used for it) |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | Non-symplectic matrices or repeated modes raise `ValueError` |
| Tests | `tests/protocols/test_covariance_symmetry.py::test_long_random_gate_sequence` (on `GaussianBackend` every CZ, beamsplitter, squeeze and rotation goes through `transform`; 600 gates are compared with plain symplectic matrix products), `tests/native/test_native_gaussian_backend.py::test_errors_match_numpy_backend` (both `transform` implementations reject `2 * I`); `tests/native/test_native_gaussian_diagnostics.py::test_batch_symplectic_matches_dense_reference` uses `GaussianBackend.transform` as the reference for the batched kernel |
| Status | **Implemented and validated** |

Example:

```python
import numpy as np
from photographiq.backends import GaussianBackend
b = GaussianBackend(); b.prepare(0, 0)
try:
    b.transform((0,), np.array([[2.0, 0.0], [0.0, 1.0]]))
except ValueError as exc:
    print("ValueError:", exc)
# ValueError: Transformation must be symplectic
```

### Teleportation-based (measurement-based) Gaussian gates

| Field | Value |
|---|---|
| Description | Ready-made finite-squeezing cluster patterns that realize Gaussian gates by homodyne teleportation with feed-forward: wire steps, identity, displacement, rotation, squeezing, general single-mode symplectic and two-mode CZ. The unconditional channel (S, N) exposes the finite-squeezing noise. This document has no separate compilation section; see [compilation](user-guide/compilation.md) and [universal compilation](user-guide/universal-compilation.md). |
| Python API | `pg.protocols.wire(shears=(0.0,), *, squeezing=1.0)`, `.identity(*, squeezing=1.0)`, `.displacement(q=0.0, p=0.0, *, squeezing=1.0)`, `.rotation(angle, *, squeezing=1.0)`, `.squeeze(r, *, squeezing=1.0)`, `.gaussian(matrix, *, squeezing=1.0)`, `.entangling(weight=1.0, *, squeezing=1.0)`; `pg.gaussian_channel(pattern, *, parameters=None)` |
| Implementation | `src/photographiq/protocols.py`, `src/photographiq/compiler.py`, `src/photographiq/analysis.py`, `src/photographiq/gaussian.py` (`teleportation_matrix`, `wire_channel`) |
| Backends | All executing backends |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | Each wire step adds momentum noise exp(-2r); the ideal gate is the r -> infinity limit; `gaussian_channel` requires fixed (outcome-independent) measurement angles |
| Tests | `tests/protocols/test_gaussian_protocols.py::test_exact_compiler_matrix`, `::test_cluster_ensemble_channel`, `tests/analytical/test_channels.py::test_wire_noise_and_bk_noise_exact` (closed form) |
| Status | **Implemented and validated** |

Example (compiled squeezing gate: ideal matrix plus finite-squeezing noise):

```python
import numpy as np, photographiq as pg
ch = pg.gaussian_channel(pg.protocols.squeeze(0.3, squeezing=1.0))
print(np.round(ch.matrix, 6))
print(np.round(ch.noise, 4))
# [[ 0.740818 -0.      ]
#  [ 0.        1.349859]]
# [[0.2839 0.1003]
#  [0.1003 0.5173]]
```

## 9. Non-Gaussian gates

Non-Gaussian gates require a Fock backend and an explicit cutoff; Gaussian backends reject them at preflight with `NotImplementedError` naming the capability (for example "GaussianBackend does not support cubic_phase"). All are labelled experimental in `docs/validation/feature-matrix.md`. The simulator contains code that would materialize a deferred displacement frame before these gates, but `frame=True` is only accepted on Gaussian backends, which reject these gates at preflight, so in practice frames and non-Gaussian gates never meet (Section 11).

### Cubic phase `CubicPhase`

| Field | Value |
|---|---|
| Description | exp(i gamma q^3 / 6). `piquasso-fock` uses Piquasso's expm(i gamma Q^3/6) with the truncated position matrix; the native backends offer `cubic_phase="truncated_generator"` (default, identical to Piquasso to 1e-13) or `cubic_phase="projected"` (exact projected matrix elements by a converged quadrature, so leakage appears in the retained-norm diagnostics). |
| Python API | `pg.CubicPhase(node, gamma)`; `NativeFockBackend(cutoff, *, cubic_phase="truncated_generator" | "projected", cubic_tolerance=1e-12)` |
| Implementation | `src/photographiq/backends/fock.py`, `src/photographiq/backends/native_fock.py`; `cpp/src/fock/gates.cpp` |
| Backends | Fock backends (capability `cubic_phase`) |
| CPU / GPU | CPU (NumPy/Piquasso) or CPU (C++/OpenMP) |
| Assumptions | [q, p] = 2i; the truncated generator is unitary on the truncated space and never reports leakage; both definitions converge to the exact gate with the cutoff |
| Tests | `tests/non_gaussian/test_execution.py::test_cubic_independent_and_raw_native` (independent matrix, raw Piquasso, atol 2e-13), `tests/native/test_native_fock_kernels.py::test_cubic_phase_projected_against_direct_integration`, `tests/native/test_native_fock_backend.py::test_cubic_phase_converges_to_exact_state_and_matches_piquasso_definition`, `tests/fock/test_high_moments.py::test_cubic_high_moments_converge_to_noncommuting_analytic_identity` |
| Status | **Implemented but experimental** |

Example:

```python
import numpy as np, photographiq as pg
src = pg.FockInput((np.sqrt(0.6), 1j * np.sqrt(0.4)))
p = pg.Pattern().extend([pg.Prepare(0, state=src), pg.CubicPhase(0, 0.2)])
a = pg.simulate(p, backend="piquasso-fock", cutoff=36).state.state_vector
b = pg.simulate(p, backend="native-fock", cutoff=36).state.state_vector
print(round(abs(np.vdot(a, b)) ** 2, 12))
try:
    pg.simulate(pg.Pattern(inputs=(0,)).append(pg.CubicPhase(0, 0.2)))
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
# 1.0
# NotImplementedError: GaussianBackend does not support cubic_phase
```

### Kerr `Kerr`

| Field | Value |
|---|---|
| Description | Self-Kerr exp(i kappa n^2), an exact number-diagonal phase. |
| Python API | `pg.Kerr(node, kappa)` |
| Implementation | `src/photographiq/backends/fock.py` (`pq.Kerr`), `src/photographiq/backends/native_fock.py` |
| Backends | Fock backends (capability `kerr`) |
| CPU / GPU | CPU (NumPy/Piquasso) or CPU (C++/OpenMP) |
| Assumptions | Exact within the represented support (support is cutoff-invariant) |
| Tests | `tests/non_gaussian/test_execution.py::test_kerr_exact_nontrivial_and_native` (closed form, raw Piquasso), `tests/non_gaussian/test_sensitivity.py::test_kerr_sensitivity_and_cutoff_invariant_support` |
| Status | **Implemented but experimental** |

Example:

```python
import numpy as np, photographiq as pg
src = pg.FockInput(tuple(np.ones(3) / np.sqrt(3)))
p = pg.Pattern().extend([pg.Prepare(0, state=src), pg.Kerr(0, 0.71)])
v = pg.simulate(p, backend="native-fock", cutoff=6).state.state_vector[:3]
print(np.allclose(v, np.array(src.amplitudes) * np.exp(1j * 0.71 * np.arange(3) ** 2)))
# True
```

### Cross-Kerr (backend method only)

| Field | Value |
|---|---|
| Description | exp(i kappa n_u n_v) on two modes. There is no Pattern command; it is available only as a method on native Fock backend objects. |
| Python API | `NativeFockBackend(cutoff).cross_kerr(u, v, kappa)`, `NativeMixedFockBackend(cutoff).cross_kerr(u, v, kappa)` |
| Implementation | `src/photographiq/backends/native_fock.py`; `cpp/src/fock/kernels.cpp` |
| Backends | `native-fock`, `native-mixed-fock` (capability `cross_kerr`); not reachable through `simulate` |
| CPU / GPU | CPU (C++/OpenMP) |
| Assumptions | Exact phase on the truncated basis |
| Tests | `tests/native/test_native_fock_backend.py::test_cross_kerr_matches_piquasso_cross_kerr_gate` (Piquasso), `tests/native/test_native_fock_kernels.py::test_phase_kernels_kerr_and_cross_kerr` |
| Status | **Partially supported** (backend method works; no Pattern command, so it cannot be used in patterns, serialized or planned) |

Example (relative phase kappa * 1 * 1 on |11>):

```python
import numpy as np, photographiq as pg
from photographiq.backends.native_fock import NativeFockBackend
b = NativeFockBackend(cutoff=6)
b.prepare_resource(("u", "v"), pg.FockSuperposition.from_mapping({(0, 0): 2**-0.5, (1, 1): 2**-0.5}))
b.cross_kerr("u", "v", 0.5)
st = b.get_state()
s, basis = st.state_vector, st.basis
print(round(float(np.angle(s[basis.index((1, 1))] / s[basis.index((0, 0))])), 12))
# 0.5
```

### Ideal photon addition and subtraction `PhotonAdd`, `PhotonSubtract`

| Field | Value |
|---|---|
| Description | Normalized ideal a^dag or a applied during the pattern. Diagnostics record the pre-normalization norm <psi|a a^dag|psi> or <psi|a^dag a|psi>. These are not physical heralding channels (use `non_gaussian.photon_subtraction` for that). |
| Python API | `pg.PhotonAdd(node)`, `pg.PhotonSubtract(node)` |
| Implementation | `src/photographiq/backends/fock.py` (`ladder`), `src/photographiq/backends/mixed_fock.py` (`ladder`) |
| Backends | Fock backends (capabilities `photon_addition`, `photon_subtraction`) |
| CPU / GPU | CPU (NumPy) |
| Assumptions | Subtraction from vacuum raises `ValueError` "Ideal ladder operation has zero norm"; addition that leaves the cutoff raises `ValueError` "Photon addition exceeds cutoff; increase cutoff" |
| Tests | `tests/non_gaussian/test_execution.py::test_ideal_ladders_sqrt_factors_and_failures` (independent ladder matrices), `tests/native/test_native_mixed_fock_backend.py::test_ladder_and_counting_match_piquasso_mixed`, `tests/mixed_fock/test_validation.py::test_mixed_ladder_weighting` |
| Status | **Implemented but experimental** |

Example:

```python
import numpy as np, photographiq as pg
src = pg.FockInput((0, np.sqrt(0.4), 0, np.sqrt(0.6)))
for cmd in (pg.PhotonAdd(0), pg.PhotonSubtract(0)):
    st = pg.simulate(pg.Pattern().extend([pg.Prepare(0, state=src), cmd]), backend="native-fock", cutoff=9).state
    print(type(cmd).__name__, round(st.photon_number(0), 6), round(st.diagnostics[-1]["ladder_norm_squared"], 6))
try:
    pg.simulate(pg.Pattern().extend([pg.Prepare(0, 0), pg.PhotonSubtract(0)]), backend="native-fock", cutoff=8)
except ValueError as exc:
    print("ValueError:", exc)
# PhotonAdd 3.5 3.2
# PhotonSubtract 1.636364 2.2
# ValueError: Ideal ladder operation has zero norm
```

### Resource injection (conditional filter)

| Field | Value |
|---|---|
| Description | Inverse SUM between the input and a resource mode, then a q homodyne on the resource: psi(q) -> psi(q) phi(q + m). For a cat resource this is `cat_injection`. The result is a measurement-conditioned filter, not a deterministic gate. |
| Python API | `pg.non_gaussian.resource_injection(resource, *, input_node="in", ancilla="resource", key="m")`, `pg.non_gaussian.cat_injection(alpha=1.0, parity=1, **kwargs)` |
| Implementation | `src/photographiq/non_gaussian.py` |
| Backends | Fock backends |
| CPU / GPU | CPU (NumPy/Piquasso) or CPU (C++/OpenMP) |
| Assumptions | The inverse SUM is realized by Rotate / Entangle(-1) / Rotate and truncates on Fock backends; outcome m is sampled unless postselected |
| Tests | `tests/non_gaussian/test_injection.py::test_resource_injection_independent_wavefunction_and_fixed_branch` (independent wavefunction), `::test_raw_piquasso_resource_entanglement_before_measurement` (raw Piquasso), `::test_regression_ci_inverse_sum_decomposition_converges` |
| Status | **Implemented but experimental** |

Example:

```python
import photographiq as pg
p = pg.non_gaussian.cat_injection(alpha=1.0, parity=1)
r = pg.simulate(p, backend="native-fock", cutoff=30, seed=4)
print(sorted(r.outcomes), r.measurement_statistics["m"]["kind"], r.state.nodes)
# ['m'] density ('in',)
```

### Cubic-phase injection with nonlinear feed-forward

| Field | Value |
|---|---|
| Description | Injects a `CubicPhaseResource`; given resource outcome m, applies the outcome-dependent corrections QuadraticPhase(-2 gamma m) and Displace(p = -gamma m^2), leaving CP(gamma) times the finite-squeezing envelope exp(-(q + m)^2 / (4 exp(2r))). |
| Python API | `pg.non_gaussian.cubic_injection(gamma, squeezing=0.0, *, input_node="in", ancilla="resource", key="m")` |
| Implementation | `src/photographiq/non_gaussian.py` |
| Backends | Fock backends |
| CPU / GPU | CPU (NumPy/Piquasso) or CPU (C++/OpenMP) |
| Assumptions | The envelope is physical filtering; it disappears only as r -> infinity, which increases the cutoff cost; finite-resource, finite-cutoff and quadrature errors are tracked separately in tests |
| Tests | `tests/non_gaussian/test_injection.py::test_regression_ci_cubic_injection_error_budget`, `::test_independent_injection_quadrature_error_separate_from_cutoff`, `tests/cutoff/test_convergence.py::test_finite_cubic_injection_cutoff_convergence`, `tests/graphix_reference/test_non_gaussian_structure.py::test_cubic_injection_has_corresponding_causal_skeleton` |
| Status | **Implemented but experimental** |

Example:

```python
import photographiq as pg
p = pg.non_gaussian.cubic_injection(0.1, squeezing=0.2)
print([type(c).__name__ for c in p.commands])
r = pg.simulate(p, backend="native-fock", cutoff=40, measurement_outcomes={"m": 0.0})
print(r.state.nodes, r.measurement_statistics["m"]["postselected"])
# ['Prepare', 'Rotate', 'Entangle', 'Rotate', 'Measure', 'QuadraticPhase', 'Displace', 'Output']
# ('in',) True
```

### Synthesized nonlinear gates (pointer)

| Field | Value |
|---|---|
| Description | Product-formula synthesis of quadrature polynomials and Kerr from Gaussian and cubic primitives, used by `Circuit.kerr` and `Circuit.cubic_phase` lowering. This document has no separate compilation section; see [compilation](user-guide/compilation.md) and [universal compilation](user-guide/universal-compilation.md). |
| Python API | `photographiq.synthesis.quadrature_polynomial(terms, *, steps=8, modes=1, mode=0)`, `photographiq.synthesis.synthesize_kerr(kappa, *, steps=8, modes=1, mode=0)`, `Circuit.compile(..., synthesis_steps=8)` |
| Implementation | `src/photographiq/synthesis.py`, `src/photographiq/compiler.py` |
| Backends | Fock backends |
| CPU / GPU | CPU only |
| Assumptions | Finite-energy injection, finite-Fock projection and product-formula error are separate limits; no returned step count is an a-priori operator-norm guarantee |
| Tests | `tests/compilation/test_synthesis.py`, `tests/compilation/test_kerr_validation.py`, `tests/compilation/test_quartic_validation.py` |
| Status | **Implemented but experimental** (the module docstring says "Experimental") |

Example: see `docs/tutorials/22-universal-compilation.md`.

## 10. Measurements

All pattern measurements are destructive: `Measure(node, measurement, key)` removes the mode and records one classical value under `key` (default: the node label). Outcomes are sampled from the backend's random generator unless fixed by `measurement_outcomes` (postselection, Section 11). Each recorded outcome stores its conditional probability (discrete) or density (continuous) in `Result.measurement_statistics`. There is no dedicated threshold-detector measurement class; see the threshold entry below.

### Ideal homodyne on Gaussian backends

| Field | Value |
|---|---|
| Description | Destructive measurement of q cos(theta) + p sin(theta); the remaining modes are conditioned exactly by the Schur complement. |
| Python API | `pg.Homodyne(angle=0.0, efficiency=1.0, noise=0.0)`, `pg.Homodyne.q()`, `pg.Homodyne.p()`, `Pattern.measure(node, measurement=None, *, key=None)`; direct backend call `backend.measure(node, pg.Homodyne(theta), angle=None, *, outcome=None)` |
| Implementation | `src/photographiq/measurements.py`, `src/photographiq/backends/gaussian.py` (`measure`), `src/photographiq/backends/native_gaussian.py`, `cpp/src/gaussian/conditioning.cpp`; `src/photographiq/backends/base.py` (`resolve_angle`, shared by every backend's `measure`) |
| Backends | `gaussian`, `piquasso`, `native-gaussian` |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | Angle may be an expression of earlier outcomes; vacuum outcome variance 1; zero or nonfinite variance raises `ValueError`. `simulate` resolves the angle and passes it explicitly. A direct `backend.measure(node, Homodyne(theta))` without `angle=` now measures at theta (before the 2026-11-12 fix it silently measured q). An explicit `angle=` that contradicts a nonzero numeric `Homodyne.angle` raises `ValueError("angle=... contradicts Homodyne.angle=...")`; an explicit angle with the default `Homodyne()` (angle 0) still wins; an expression angle without `angle=` raises `ValueError` (run the pattern with `simulate`). The same rule applies on the Fock and mixed-Fock backends. |
| Tests | `tests/analytical/test_measurements.py::test_exact_homodyne_schur_complement` (closed form), `::test_vacuum_measurement_statistics`, `tests/protocols/test_gaussian_postselection.py::test_sampled_density_is_analytic_density`, `tests/native/test_native_gaussian_backend.py::test_tiny_variance_homodyne_closed_form`; `tests/validation/test_val_gaussian.py::test_homodyne_conditioning_and_density` (conditional state and density against a physical dilation conditioned through the precision matrix, all three Gaussian backends, 6.5e-13); `tests/regression/test_measure_angle.py` (`test_homodyne_angle_used_without_explicit_angle` on every backend class, `test_explicit_angle_contradicting_homodyne_raises`, `test_explicit_angle_with_default_homodyne_still_wins`) |
| Status | **Implemented and validated** |

Example (postselected density of a vacuum homodyne at x = 0.5, on four backends):

```python
import numpy as np, photographiq as pg
p = pg.Pattern().append(pg.Prepare(0, 0)).measure(0, pg.Homodyne(0.3))
for b in ("gaussian", "native-gaussian", "native-fock", "native-mixed-fock"):
    r = pg.simulate(p, backend=b, cutoff=8, measurement_outcomes={0: 0.5})
    print(b, round(np.exp(r.log_likelihood), 10))
print(round(np.exp(-0.125) / np.sqrt(2 * np.pi), 10))
# gaussian 0.3520653268
# native-gaussian 0.3520653268
# native-fock 0.3520653268
# native-mixed-fock 0.3520653268
# 0.3520653268
```

Example (direct backend call; the `Homodyne` angle is used, a contradicting `angle=` raises):

```python
import numpy as np, photographiq as pg
from photographiq.backends import GaussianBackend
e = GaussianBackend(); e.prepare(0, state=pg.GaussianInput.coherent(1j))   # <q> = 0, <p> = 2
e.measure(0, pg.Homodyne(np.pi / 2), outcome=2.0)                           # no angle= given
print(round(e.last_measurement["value"], 4), round(1 / np.sqrt(2 * np.pi), 4))  # N(2; 2, 1): p was measured
e.prepare(0, state=pg.GaussianInput.coherent(1j))
try:
    e.measure(0, pg.Homodyne(np.pi / 2), 0.0, outcome=2.0)
except ValueError as exc:
    print("ValueError:", exc)
# 0.3989 0.3989
# ValueError: angle=0.0 contradicts Homodyne.angle=1.5707963267948966
```

### Ideal homodyne on Fock backends

| Field | Value |
|---|---|
| Description | Contracts Fock amplitudes (or density matrices) with Hermite wavefunctions; sampling inverts the numerically integrated CDF by Brent root finding, and the conditional state of the remaining modes is returned. |
| Python API | `pg.Homodyne(angle)` with `backend="piquasso-fock" | "native-fock" | "piquasso-mixed-fock" | "native-mixed-fock"` |
| Implementation | `src/photographiq/fock_measurements.py` (`wavefunctions`, `homodyne_projection`), `src/photographiq/backends/fock.py`, `src/photographiq/backends/mixed_fock.py` |
| Backends | Fock backends |
| CPU / GPU | CPU (NumPy/SciPy) |
| Assumptions | Integration window 2 sqrt(cutoff) + 10; normalization and partial-CDF error budgets of 1e-7 raise `ArithmeticError` instead of sampling a wrong tail; tolerances control numerical inversion, not the Hilbert-space cutoff. A direct `measure` call without `angle=` uses the `Homodyne` angle (`resolve_angle`, see the previous entry). |
| Tests | `tests/fock/test_homodyne_validation.py::test_number_homodyne_density_cdf_and_normalized_condition` (closed form), `::test_displaced_rotated_homodyne_distribution`, `::test_unrepresented_displacement_fails_instead_of_sampling_wrong_tail`, `tests/non_gaussian/test_execution.py::test_homodyne_entangled_conditional_state_and_angle`, `tests/protocols/test_gaussian_postselection.py::test_adaptive_wire_matches_fock_postselection` (Gaussian vs Fock); `tests/validation/test_val_fock.py::test_fock_homodyne_density_convention` (closed-form coherent-state density at three angles on all four Fock backends, 6.7e-16; Piquasso rows are a secondary check); `tests/regression/test_measure_angle.py`. Upgraded to validated on 2026-11-12 because of the independent closed-form references on both native engines. |
| Status | **Implemented and validated** (`docs/validation/feature-matrix.md` still marks Fock homodyne E) |

Example: see the four-backend example above.

### Inefficient and noisy homodyne

| Field | Value |
|---|---|
| Description | Detector efficiency eta in (0, 1] and added classical noise variance; outcomes are calibrated to the incident quadrature, so the outcome variance is Var + (1 - eta)/eta + noise. |
| Python API | `pg.Homodyne(angle, efficiency=eta, noise=sigma2)` |
| Implementation | `src/photographiq/backends/gaussian.py`; `src/photographiq/backends/mixed_fock.py` (loss of eta before the projector, rescaled wavefunctions, Gaussian convolution for `noise` by `quad_vec`) |
| Backends | `gaussian`, `piquasso`, `native-gaussian`, `piquasso-mixed-fock`, `native-mixed-fock`; pure-Fock backends raise `NotImplementedError` "... does not support noisy_homodyne" |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) for native Gaussian |
| Assumptions | 0 < eta <= 1 and finite noise >= 0 (`ValueError` otherwise); the mixed-Fock noise convolution raises `ArithmeticError` if it does not converge |
| Tests | `tests/analytical/test_measurements.py::test_detector_efficiency_and_generaldyne` (variance 2 at eta = 0.5), `tests/protocols/test_gaussian_postselection.py::test_lossy_inefficient_noisy_homodyne_matches_mixed_fock` (Gaussian vs mixed Fock), `tests/regression/test_inputs_and_frames.py::test_virtual_measurement_and_loss`; `tests/validation/test_val_gaussian.py::test_homodyne_conditioning_and_density` (loss-beam-splitter dilation with a vacuum ancilla, 3.6e-14); `tests/validation/test_val_fock.py::test_fock_homodyne_density_convention` (mixed-Fock density with eta = 0.6 and noise 0.2 against the closed form N(mu, 1 + (1 - eta)/eta + noise), 5.6e-16). The mixed-Fock qualifier was dropped on 2026-11-12 because of this closed-form test. |
| Status | **Implemented and validated** (Gaussian and mixed-Fock backends) |

Example (2000 sampled shots, so the variance is a statistical estimate):

```python
import photographiq as pg
p = pg.Pattern().append(pg.Prepare(0, 0)).measure(0, pg.Homodyne(efficiency=0.5, noise=0.1))
v = pg.run_shots(p, 2000, seed=23).values(0)
print(round(v.var(), 3), "expected", 1 + (1 - 0.5) / 0.5 + 0.1)
try:
    pg.simulate(p, backend="native-fock", cutoff=8)
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
# 2.129 expected 2.1
# NotImplementedError: NativeFockBackend does not support noisy_homodyne
```

### Heterodyne

| Field | Value |
|---|---|
| Description | Destructive joint (q, p) measurement with outcome covariance V + I (vacuum: 2I); the outcome is a `(q, p)` tuple, so feed-forward must select a component with `Outcome(key, component=...)` (`Pattern.validate()` rejects a read without it, see "Outcome references"). |
| Python API | `pg.Heterodyne()`; `pg.Outcome(key, component=0 | 1)` |
| Implementation | `src/photographiq/measurements.py`, `src/photographiq/backends/gaussian.py` |
| Backends | `gaussian`, `piquasso`, `native-gaussian` only; Fock backends raise `NotImplementedError` "... does not support heterodyne" |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | Postselected outcomes must be finite real pairs |
| Tests | `tests/analytical/test_measurements.py::test_vacuum_measurement_statistics` (variance 2), `tests/protocols/test_gaussian_postselection.py::test_cz_pair_closed_form_phase_space_postselection` (closed form) |
| Status | **Implemented and validated** |

Example:

```python
import photographiq as pg
p = pg.Pattern(pg.CVGraph.line(2, squeezing=0.5)).measure(0, pg.Heterodyne())
r = pg.simulate(p, seed=1)
print(type(r.outcomes[0]).__name__, len(r.outcomes[0]), r.measurement_statistics[0]["kind"])
try:
    pg.simulate(p, backend="native-mixed-fock", cutoff=6)
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
# tuple 2 density
# NotImplementedError: NativeMixedFockBackend does not support heterodyne
```

### Generaldyne

| Field | Value |
|---|---|
| Description | Gaussian phase-space measurement with a physical seed covariance sigma; outcomes have covariance V + sigma. Heterodyne is the case sigma = I. |
| Python API | `pg.Generaldyne(covariance)` |
| Implementation | `src/photographiq/measurements.py`, `src/photographiq/backends/gaussian.py` |
| Backends | `gaussian`, `piquasso`, `native-gaussian` only |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | The seed covariance must satisfy V + i Omega >= 0 (`ValueError` otherwise). The outcome is a `(q, p)` tuple; as for heterodyne, `Pattern.validate()` rejects feed-forward that reads it without `component=`. |
| Tests | `tests/protocols/test_gaussian_postselection.py::test_cz_pair_closed_form_phase_space_postselection` (closed-form conditional mean and outcome density for squeezed generaldyne seeds), `tests/analytical/test_measurements.py::test_detector_efficiency_and_generaldyne` (runs and rejects an unphysical seed). `tests/piquasso_reference/test_raw_piquasso.py::test_raw_generaldyne_conditional_state` checks only Piquasso's own generaldyne convention, not PhotoGraphiQ's. |
| Status | **Implemented and validated** |

Example (sampled variances, 2000 shots):

```python
import numpy as np, photographiq as pg
p = pg.Pattern().append(pg.Prepare(0, 0)).measure(0, pg.Generaldyne(np.diag([0.5, 2.0])))
v = pg.run_shots(p, 2000, seed=5).values(0)
print(np.round(v.var(axis=0), 2), "expected", [1.5, 3.0])
try:
    pg.Generaldyne(np.eye(2) * 0.2)
except ValueError as exc:
    print("ValueError:", exc)
# [1.54 2.82] expected [1.5, 3.0]
# ValueError: Covariance violates the uncertainty relation V+i Omega >= 0
```

### Photon-number-resolving counting

| Field | Value |
|---|---|
| Description | Destructive photon counting; returns an integer, records the Born probability and conditions the remaining modes. |
| Python API | `pg.PhotonNumber()` |
| Implementation | `src/photographiq/measurements.py`, `src/photographiq/backends/fock.py`, `src/photographiq/backends/mixed_fock.py` |
| Backends | Fock backends (capability `photon_counting`); Gaussian backends raise `NotImplementedError` "GaussianBackend does not support photon_counting" (GBS photon statistics of Gaussian states are in `photographiq.sampling`) |
| CPU / GPU | CPU (NumPy) |
| Assumptions | Outcomes are 0 .. cutoff-1; postselecting a zero-probability count raises `ValueError` |
| Tests | `tests/integration/test_fock.py::test_fock_number_and_counting`, `tests/non_gaussian/test_execution.py::test_raw_piquasso_count_statistics` (raw Piquasso), `tests/mixed_fock/test_conditioning.py::test_diagonal_count_conditioning`, `tests/cross_backend/test_cross_backend_agreement.py::test_cat_photon_counting_closed_form`; `tests/validation/test_val_fock.py::test_photon_counting_probability_and_conditional_state` (Born probability and conditional state computed independently in NumPy, `native-fock` and `native-mixed-fock`, 2.2e-16). Upgraded to validated on 2026-11-12 because of this independent reference. |
| Status | **Implemented and validated** (`docs/validation/feature-matrix.md` still marks PNR E) |

Example (single photon on a balanced beamsplitter, vacuum heralded on mode 1):

```python
import numpy as np, photographiq as pg
p = pg.Pattern(inputs=(0, 1)).append(pg.BeamSplitter(0, 1, np.pi / 4)).measure(1, pg.PhotonNumber())
r = pg.simulate(p, backend="native-fock", cutoff=4, inputs={0: pg.FockInput.number(1)},
                measurement_outcomes={1: 0})
print(r.measurement_statistics[1], round(r.state.photon_number(0), 12))
try:
    pg.simulate(p)
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
# {'kind': 'probability', 'value': 0.5000000000000001, 'postselected': True} 1.0
# NotImplementedError: GaussianBackend does not support photon_counting
```

### Threshold (click) detection

| Field | Value |
|---|---|
| Description | No dedicated pattern measurement exists. A click/no-click detector can be written as a `MeasurementInstrument` (effects |0><0| and I - |0><0|), which needs mixed-Fock conditioning because the click effect is not rank one. Separately, `photographiq.sampling` computes threshold (torontonian) probabilities and samples for Gaussian states outside the pattern model. |
| Python API | `pg.MeasurementInstrument({"no-click": (P0,), "click": (P1, P2, ...)})`; `photographiq.sampling.threshold_probability(state, clicks, *, backend="auto", precision=None, return_error=False)`, `sample_threshold(state, shots, *, rng=None, backend="auto")`, `threshold_distribution(state, *, backend="auto")` |
| Implementation | `src/photographiq/instruments.py`; `src/photographiq/sampling/gbs.py`, `cpp/src/sampling/torontonian.cpp` |
| Backends | Pattern instrument: `piquasso-mixed-fock`, `native-mixed-fock` (pure-Fock backends raise `NotImplementedError` "... does not support custom_kraus_instrument"; Gaussian backends cannot run it); GBS functions: any `GaussianState` |
| CPU / GPU | CPU (NumPy) for the instrument; CPU (NumPy or C++) for GBS threshold functions |
| Assumptions | Instrument input dimension must equal the backend cutoff (single mode); threshold probabilities carry an absolute first-order rounding bound (Section 20), and `ThresholdPrecisionWarning` is raised for negative results, results consistent with zero within that bound, and estimated relative error above 1e-6 |
| Tests | `tests/gkp/instruments/test_instruments.py::test_multikraus_partial_measurement_preserves_mixedness_and_rejects_pure_preflight` (`piquasso-mixed-fock`); `tests/native/test_native_mixed_fock_backend.py::test_multi_kraus_instrument_matches_explicit_partial_trace` (`native-mixed-fock`, against an explicit partial trace); `tests/sampling/test_sampling_gbs.py::test_threshold_normalization_and_vacuum_inclusion_exclusion`, `::test_threshold_multiprecision_matches_closed_forms` (mpmath `precision=30` path vs closed forms, rel 1e-15), `::test_threshold_rounding_bound_holds_against_multiprecision`, `::test_threshold_flags_negative_and_zero_results` |
| Status | **Partially supported** (in patterns only via a user-built instrument on mixed-Fock backends; no `Threshold` measurement class) |

Example:

```python
import numpy as np, photographiq as pg
cutoff = 6
vac = np.zeros((cutoff, cutoff)); vac[0, 0] = 1
instr = pg.MeasurementInstrument({"no-click": (vac,), "click": tuple(np.outer(e, e) for e in np.eye(cutoff)[1:])})
print(sorted(instr.required_capabilities))
p = pg.Pattern(inputs=(0,)).measure(0, instr, key="det")
for n in (0, 2):
    r = pg.simulate(p, backend="native-mixed-fock", cutoff=cutoff, inputs={0: pg.FockInput.number(n)}, seed=0)
    print(n, r.outcomes["det"])
try:
    pg.simulate(p, backend="native-fock", cutoff=cutoff)
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
# ['custom_kraus_instrument']
# 0 no-click
# 2 click
# NotImplementedError: NativeFockBackend does not support custom_kraus_instrument
```

### Finite-outcome instruments `MeasurementInstrument`

| Field | Value |
|---|---|
| Description | A complete single-mode instrument given as outcome label -> Kraus matrices. In a pattern it is destructive (the measured mode is traced out, other modes keep their conditional state). Instruments with rank-one effects run on pure-Fock backends; general ones need mixed Fock. `probabilities` and `conditional` also work standalone on vectors or density matrices. |
| Python API | `pg.MeasurementInstrument(kraus)`, `.effects`, `.labels`, `.required_capabilities`, `.probabilities(state)`, `.conditional(state, outcome)` |
| Implementation | `src/photographiq/instruments.py` (`MeasurementInstrument`, `destructive_instrument`) |
| Backends | Rank-one: all Fock backends (`rank_one_instrument`); general Kraus: `piquasso-mixed-fock`, `native-mixed-fock` (`custom_kraus_instrument`) |
| CPU / GPU | CPU (NumPy) |
| Assumptions | Effects must sum to the identity to 1e-10 (include failure outcomes); input dimension must equal the backend cutoff (`ValueError` "Instrument input dimension must equal backend cutoff"); effects alone do not define the update |
| Tests | `tests/gkp/instruments/test_instruments.py::test_kraus_is_not_effect_update`, `::test_complex_rank_one_entangled_conditioning`, `::test_validation_probabilities_and_impossible_branch`, `::test_multikraus_partial_measurement_preserves_mixedness_and_rejects_pure_preflight` (multi-Kraus on `piquasso-mixed-fock`), `tests/native/test_native_fock_backend.py::test_instrument_and_gkp_readout_match_piquasso` (rank-one, `native-fock` vs `piquasso-fock`), `tests/native/test_native_mixed_fock_backend.py::test_multi_kraus_instrument_matches_explicit_partial_trace` (random 2-outcome, 4-Kraus instrument on `native-mixed-fock`: probabilities and conditional state against an explicit partial trace, `1e-12`). Upgraded to validated on 2026-11-12: general Kraus instruments have an independent reference (explicit Kraus algebra or an explicit partial trace) on both mixed backends, and rank-one instruments a closed-form conditional state (`test_complex_rank_one_entangled_conditioning`). |
| Status | **Implemented and validated** (`docs/validation/feature-matrix.md` still marks instruments E; rank-one instruments on `native-fock` are compared with Piquasso only) |

Example:

```python
import numpy as np, photographiq as pg
cutoff = 4
proj = [np.outer(e, e) for e in np.eye(cutoff)]
instr = pg.MeasurementInstrument({k: (proj[k],) for k in range(cutoff)})
print(sorted(instr.required_capabilities))
p = pg.Pattern(inputs=(0,)).measure(0, instr)
r = pg.simulate(p, backend="native-fock", cutoff=cutoff, inputs={0: pg.FockInput.number(1)})
print(r.outcomes, r.measurement_statistics[0]["kind"])
try:
    pg.simulate(p, backend="native-fock", cutoff=5)
except ValueError as exc:
    print("ValueError:", exc)
# ['rank_one_instrument']
# {0: 1} probability
# ValueError: Instrument input dimension must equal backend cutoff
```

### Physical GKP X/Z readout

| Field | Value |
|---|---|
| Description | Ideal homodyne of q (Z basis) or p (X basis) followed by modular decoding and an optional Pauli-frame correction; the record is a `LogicalDecodeResult` (bit, raw outcome, cell, residual, ...). Full GKP coverage is in Section 27. |
| Python API | `pg.PhysicalGKPReadout(basis="Z", decoder=NearestCellDecoder(), frame=LogicalPauliFrame(), flip=0)`, `pg.GKPCode(...).logical_measurement(basis="Z", *, alpha=None, **kwargs)`, `pg.SoftDecisionDecoder(code, basis="Z", prior=(0.5, 0.5))` |
| Implementation | `src/photographiq/encoded.py`, `src/photographiq/gkp_studies.py` |
| Backends | Fock backends (capabilities `encoded_gkp_xz`, `homodyne`) |
| CPU / GPU | CPU (NumPy) |
| Assumptions | Postselection fixes the raw analog value, never the decoded bit; leakage cannot be inferred from one outcome and is `None`; Y and general XY readout raise `NotImplementedError` (XY only for alpha = 0 mod pi, mapped to X) |
| Tests | `tests/gkp/measurements/test_encoded.py::test_physical_analog_result_and_feedforward`, `::test_unsupported_measurements_never_lower_to_rotated_homodyne`, `::test_frame_against_qubit_conjugation`, `tests/gkp/measurements/test_encoded_boundaries.py` |
| Status | **Partially supported** (X and Z readout implemented and experimental; Y and arbitrary XY raise `NotImplementedError`) |

Example:

```python
import photographiq as pg
code = pg.GKPCode(peak_width=0.4, envelope=0.4, cutoff=48)
p = pg.Pattern(inputs=(0,)).measure(0, code.logical_measurement("Z"), key="z")
r = pg.simulate(p, backend="native-fock", cutoff=48, inputs={0: code.one()}, seed=2)
print(r.outcomes["z"].bit, r.outcomes["z"].decoder)
try:
    code.logical_measurement("Y")
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
# 1 nearest-cell
# NotImplementedError: Only physical GKP X/Z readout is implemented; Y/XY need validation
```

### Logical Y and arbitrary XY measurement of GKP qubits

| Field | Value |
|---|---|
| Description | No physical measurement of the GKP logical Y or a general XY observable exists. `IdealLogicalXYMeasurement` is a two-dimensional ideal-qubit reference (outcome probabilities of cos(alpha) X + sin(alpha) Y on a qubit state), not an oscillator POVM and not usable in `Measure`; `LogicalMeasurementSynthesis` is an extension point whose `lower` always raises. |
| Python API | `pg.IdealLogicalXYMeasurement(alpha).probabilities(state)`, `pg.LogicalMeasurementSynthesis().lower(code, alpha, **kwargs)`, `pg.GKPCode(...).logical_measurement("XY", alpha=...)` |
| Implementation | `src/photographiq/encoded.py` |
| Backends | None for physical execution; the reference is a NumPy calculation on 2 x 2 states |
| CPU / GPU | CPU only |
| Assumptions | `logical_measurement("XY", alpha)` maps alpha = 0 mod pi to X readout (with `flip=1` at pi) and raises `NotImplementedError` "Physical logical XY supports only alpha=0 mod pi; no Y or injection" otherwise; `lower` raises `NotImplementedError` "No validated magic-state injection or arbitrary logical Rz protocol"; putting the reference in a pattern fails `validate()` with "Unsupported measurement description" |
| Tests | `tests/gkp/measurements/test_encoded.py::test_ideal_reference` (qubit observable closed form), `::test_unsupported_measurements_never_lower_to_rotated_homodyne`, `tests/gkp/measurements/test_encoded_boundaries.py` |
| Status | **Unsupported** (physical Y/XY readout raises `NotImplementedError`; only the ideal qubit reference is implemented) |

Example:

```python
import numpy as np, photographiq as pg
print(pg.IdealLogicalXYMeasurement(np.pi / 4).probabilities([1, 0]))
try:
    pg.LogicalMeasurementSynthesis().lower(pg.GKPCode(), np.pi / 4)
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
try:
    pg.GKPCode().logical_measurement("XY", alpha=np.pi / 4)
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
# (0.4999999999999999, 0.5000000000000001)
# NotImplementedError: No validated magic-state injection or arbitrary logical Rz protocol
# NotImplementedError: Physical logical XY supports only alpha=0 mod pi; no Y or injection
```

### Measurement statistics and likelihoods

| Field | Value |
|---|---|
| Description | Every backend records, per measurement key, `{"kind": "probability" | "density", "value": ..., "postselected": bool}`; `Result.log_likelihood` sums the logarithms. |
| Python API | `Result.measurement_statistics`, `Result.log_likelihood` |
| Implementation | `src/photographiq/simulator.py`, `last_measurement` in each backend |
| Backends | All executing backends |
| CPU / GPU | CPU only |
| Assumptions | With any continuous measurement the value is a joint density in calibrated outcome units, not an event probability; `log_likelihood` raises `NotImplementedError` if a backend did not report every measurement |
| Tests | `tests/protocols/test_gaussian_postselection.py::test_sampled_density_is_analytic_density`, `::test_far_tail_postselection_raises_instead_of_minus_infinite_likelihood`, `tests/non_gaussian/test_execution.py::test_homodyne_entangled_conditional_state_and_angle` (closed-form density) |
| Status | **Implemented and validated** |

Example: see the four-backend homodyne example and the photon-counting example above.

### Extensible measurement protocol

| Field | Value |
|---|---|
| Description | `Measure` accepts any object satisfying `MeasurementProtocol` (`required_capabilities`, `validate()`, `validate_backend(engine)`); backends dispatch on the concrete type and raise `NotImplementedError` for types they do not know. |
| Python API | `photographiq.measurements.MeasurementProtocol` |
| Implementation | `src/photographiq/measurements.py`, `src/photographiq/pattern.py` (`validate` rejects non-protocol objects with `NotImplementedError` "Unsupported measurement description") |
| Backends | Backend-specific |
| CPU / GPU | CPU only |
| Assumptions | Capability checks run in preflight before any state allocation |
| Tests | `tests/fock/test_preflight_hardening.py::test_mixed_channels_rejected_before_preparation`, `tests/cross_backend/test_registry_planner.py::test_required_features_basic` |
| Status | **Implemented but experimental** (behavioural tests only) |

Example: see `src/photographiq/encoded.py` (`PhysicalGKPReadout` implements the protocol).

## 11. Adaptive feedforward

Measurement outcomes become classical records that later commands may use through expression trees. The simulator evaluates each expression when its command executes, after its producers. Two execution modes must be distinguished: sampling (`seed`, `run_shots`) draws outcomes from the Born rule, while postselection (`measurement_outcomes`) fixes outcomes and returns the corresponding conditional branch with its probability or density; a postselected result is never a deterministic operation.

### Outcome references `Outcome`

| Field | Value |
|---|---|
| Description | `Outcome(key)` refers to a prior measurement or `Signal`; vector outcomes (heterodyne, generaldyne) require `component=0` (q) or `1` (p). Usable in any numeric command field, including measurement angles. |
| Python API | `pg.Outcome(key, component=None)` |
| Implementation | `src/photographiq/expressions.py`, `src/photographiq/simulator.py` (`resolve`) |
| Backends | All executing backends |
| CPU / GPU | CPU only |
| Assumptions | A reference to a future or missing key fails `Pattern.validate()`. Reading a `Heterodyne` or `Generaldyne` record (always a `(q, p)` tuple) without `component=` also fails `validate()`, and therefore `simulate`. The check walks every expression tree of later commands, including nested expressions, measurement angles and `Signal` values, and raises `ValueError("Command {i} reads the Heterodyne outcome 'm', a (q, p) pair, without component=; use Outcome('m', component=0) for q or 1 for p")`. Before the fix this passed `validate()` and failed at execution with a `TypeError` from `float()`. Still caught only at execution: a component other than 0 or 1 on a pair (`IndexError`), `component=` on a scalar Homodyne or Signal record (`TypeError`), and reads inside a `CallableExpression`. |
| Tests | `tests/graphix_reference/test_labelled_structures.py::test_multiple_adaptive_domains_corrections_and_causal_order` (domains vs Graphix), `tests/regression/test_inputs_and_frames.py::test_virtual_measurement_and_loss`, `tests/regression/test_validation_messages.py::test_two_component_outcome_without_component_rejected` (Heterodyne and Generaldyne, nested expression; components 0 and 1 still simulate), `::test_scalar_outcome_without_component_still_valid` |
| Status | **Implemented and validated** |

Example:

```python
import photographiq as pg
p = pg.Pattern(pg.CVGraph.line(2, squeezing=0.8)).measure(0, pg.Homodyne.p())
p.displace(1, q=-pg.Outcome(0))
r = pg.simulate(p, seed=7)
print(round(r.records[0], 6), p.dependencies().has_edge(3, 4))
h = pg.Pattern(pg.CVGraph.line(2)).measure(0, pg.Heterodyne()).displace(1, q=pg.Outcome(0, component=1))
print(round(pg.simulate(h, seed=7).records[0][1], 6))
bad = pg.Pattern(pg.CVGraph.line(2)).measure(0, pg.Heterodyne()).displace(1, q=pg.Outcome(0))
try:
    bad.validate()
except ValueError as exc:
    print("ValueError:", exc)
# 0.002793 True
# 0.003592
# ValueError: Command 4 reads the Heterodyne outcome 0, a (q, p) pair, without component=; use Outcome(0, component=0) for q or 1 for p
```

### External parameters `Parameter`

| Field | Value |
|---|---|
| Description | Named real scalars bound at execution through `parameters={...}`; `Pattern.parameters` lists the required names. |
| Python API | `pg.Parameter(name)`, `pg.simulate(pattern, parameters={...})`, `Pattern.parameters` |
| Implementation | `src/photographiq/expressions.py`, `src/photographiq/simulator.py` |
| Backends | All executing backends |
| CPU / GPU | CPU only |
| Assumptions | Unbound names raise `ValueError` "Unbound parameters: [...]"; trajectory backends do not differentiate parameters (`photographiq.autodiff` is the separate JAX path) |
| Tests | `tests/unit/test_core.py::test_parameters_and_callable_dependency`, `tests/protocols/test_gaussian_protocols.py::test_adaptive_backend_agreement` (NumPy vs Piquasso) |
| Status | **Implemented and validated** |

Example:

```python
import photographiq as pg
p = pg.protocols.adaptive(squeezing=0.8)
print(sorted(p.parameters))
try:
    pg.simulate(p)
except ValueError as exc:
    print("ValueError:", exc)
print(len(pg.simulate(p, parameters={"k": 0.3}, seed=10).outcomes))
# ['k']
# ValueError: Unbound parameters: ['k']
# 2
```

### Expression arithmetic and functions

| Field | Value |
|---|---|
| Description | Immutable expression trees with +, -, *, /, **, unary minus and `sin`, `cos`, `exp`, `atan2`; dependencies and parameters are tracked explicitly. Expressions are never evaluated from strings. |
| Python API | `pg.sin(x)`, `pg.cos(x)`, `pg.exp(x)`, `photographiq.expressions.atan2(y, x)`, `Expr.dependencies`, `Expr.parameters`, `Expr.evaluate(records, parameters)` |
| Implementation | `src/photographiq/expressions.py` |
| Backends | All executing backends |
| CPU / GPU | CPU only |
| Assumptions | A NaN, infinite, overflowing or complex result raises `ValueError("Expression must evaluate to a finite real scalar")` (module constant `_NOT_FINITE_REAL`). An `OverflowError` (for example `10**1000`, `exp(1000)`, or a value such as `10**400` bound to a `Parameter` or recorded as an int outcome) or a `ZeroDivisionError` is chained as `__cause__`, and a complex result such as `(-8.0) ** (1/3)` is also converted; before the fix these surfaced as the raw Python errors or a `TypeError`. Other `TypeError`s are not converted. Callers: `select_backend` records such a pattern as a preflight rejection for each backend, and the CV determinism check reports it as undecided. The interactive overlay now raises this `ValueError` for a non-affine `Outcome ** 0.5`, where it used to raise a misleading `NotImplementedError`. The JAX table in `autodiff.resolve` is separate and still returns inf or NaN silently. A string constant is not parsed (it fails at execution with `ValueError` from `float()`), but `float()` accepts a string parameter value such as `"1.5"`. Angles are in radians. |
| Tests | `tests/protocols/test_gaussian_postselection.py::test_adaptive_wire_matches_fock_postselection` (measurement angle `atan2(1, 0.2 + 0.5 * Outcome)` and correction `-Outcome / sin(angle)`, Gaussian backend vs `piquasso-fock`), `tests/unit/test_core.py::test_json_roundtrip_and_security` (parameter expressions survive serialization with identical trajectories), `tests/regression/test_validation_messages.py::test_arithmetic_errors_become_documented_value_error` (four cases, checks `__cause__`), `::test_out_of_range_or_complex_values_become_documented_value_error`, `::test_finite_expression_unchanged`. `cos` has no dedicated test; `exp` is exercised only by the overflow case. |
| Status | **Implemented and validated** (arithmetic, `sin` and `atan2` through the cross-backend test; `cos` and `exp` untested) |

Example:

```python
from photographiq.expressions import Outcome, Parameter, atan2, cos, exp, sin
e = 2 * sin(Outcome("m")) + cos(Parameter("t")) ** 2 - exp(0) / atan2(1, 1)
print(sorted(e.dependencies), sorted(e.parameters))
print(round(e.evaluate({"m": 0.5}, {"t": 0.0}), 12))
for expr, rec in ((exp(Outcome("m")), {"m": 1000.0}), (1 / Outcome("m"), {"m": 0.0})):
    try:
        expr.evaluate(rec, {})
    except ValueError as exc:
        print("ValueError:", exc, "from", type(exc.__cause__).__name__)
# ['m'] ['t']
# 0.685611532473
# ValueError: Expression must evaluate to a finite real scalar from OverflowError
# ValueError: Expression must evaluate to a finite real scalar from ZeroDivisionError
```

### Runtime callables `CallableExpression`

| Field | Value |
|---|---|
| Description | Wraps an arbitrary Python function of the records with a mandatory declared dependency set; the function only receives the declared records. A plain Python callable placed in a command passes `Pattern.validate()` but is rejected at execution with `TypeError` "Wrap callables in CallableExpression with declared dependencies". |
| Python API | `pg.CallableExpression(function, dependencies)` |
| Implementation | `src/photographiq/expressions.py` |
| Backends | All executing backends |
| CPU / GPU | CPU only |
| Assumptions | Access to an undeclared record raises `KeyError`; patterns containing callables cannot be serialized (`TypeError`); `standardize()` does not merge displacements that contain callables |
| Tests | `tests/unit/test_core.py::test_parameters_and_callable_dependency`, `::test_json_roundtrip_and_security`, `tests/regression/test_rewrites.py::test_callable_corrections_remain_valid_under_standardization` (standardized vs original pattern) |
| Status | **Implemented but experimental** (behavioural tests only; no test compares a callable-driven trajectory with an independent reference) |

Example:

```python
import photographiq as pg
f = pg.CallableExpression(lambda rec: 0.2 + rec[0] ** 2, dependencies={0})
p = pg.Pattern(pg.CVGraph.line(2, squeezing=0.4)).measure(0).measure(1, pg.Homodyne(f))
print(len(pg.simulate(p, seed=8).outcomes))
try:
    p.to_json()
except TypeError as exc:
    print("TypeError:", exc)
try:
    pg.CallableExpression(lambda rec: rec[2], {1}).evaluate({1: 1, 2: 2}, {})
except KeyError as exc:
    print("KeyError:", exc)
# 2
# TypeError: Callables are runtime-only and cannot be serialized
# KeyError: 2
```

### Classical registers `Signal`

| Field | Value |
|---|---|
| Description | `Signal(key, value)` stores an evaluated expression as a new classical record usable by later commands; `Result.records` holds signals and outcomes, `Result.outcomes` only measurements. |
| Python API | `pg.Signal(key, value)` |
| Implementation | `src/photographiq/commands.py`, `src/photographiq/simulator.py` |
| Backends | All executing backends |
| CPU / GPU | CPU only |
| Assumptions | Keys are unique across measurements and signals (`ValueError` "Duplicate classical key") |
| Tests | `tests/graphix_reference/test_labelled_structures.py::test_s_t_domains_and_transitive_classical_signal_support` (Graphix domains), `tests/unit/test_core.py::test_future_and_cycle_and_missing_modes` |
| Status | **Implemented and validated** |

Example:

```python
import photographiq as pg
p = pg.Pattern(pg.CVGraph.line(2, squeezing=0.5)).measure(0, pg.Homodyne.p())
p.append(pg.Signal("half", 0.5 * pg.Outcome(0))).displace(1, p=pg.Outcome("half"))
r = pg.simulate(p, seed=1)
print(sorted(r.records, key=str), round(r.records["half"] / r.records[0], 12), list(r.outcomes))
# [0, 'half'] 0.5 [0]
```

### Causality validation and scheduling

| Field | Value |
|---|---|
| Description | `Pattern.validate()` rejects future dependencies, cycles, missing producers, duplicate keys, use of measured or unprepared modes, and reads of a `(q, p)` pair outcome without `component=`. A pass that runs before the dependency graph is built rejects commands that name one mode twice: `Entangle(u, u)` gives "Self CZ is invalid: command {i} entangles {u!r} with itself", and `BeamSplitter(u, u)`, `Output((0, 0))` or duplicate nodes in `PrepareResource` give "Command {i} ({Type}) repeats a mode: ...". Before the fix all of these were misreported as "Cyclic command dependencies" (`photographiq.flow.dependency_graph` still adds a self-loop for a repeated mode, so `dependencies()` and `schedule()` called without `validate()` still reject such a pattern with "Cyclic command dependencies"). `dependencies()` returns the command DAG (mode lifetimes plus classical edges) and `schedule()` a lexicographic topological order. This is scheduling only, not a determinism certificate. |
| Python API | `Pattern.validate()`, `Pattern.dependencies()`, `Pattern.schedule()`; `photographiq.flow.dependency_graph(commands)`, `photographiq.flow.topological_schedule(commands)` |
| Implementation | `src/photographiq/pattern.py`, `src/photographiq/flow.py` |
| Backends | Backend-neutral (runs inside `simulate`) |
| CPU / GPU | CPU only |
| Assumptions | Determinism of CV patterns is analysed separately by `photographiq.cvflow` / `photographiq.flows` |
| Tests | `tests/unit/test_core.py::test_future_and_cycle_and_missing_modes`, `tests/graphix_reference/test_structure.py::test_wire_connectivity_io_measurement_order_and_domains` (Graphix domains), `tests/regression/test_validation_messages.py::test_self_cz_is_not_reported_as_a_cycle`, `::test_repeated_mode_is_not_reported_as_a_cycle` (BeamSplitter, Output), `::test_genuine_cycle_still_reported` |
| Status | **Implemented and validated** |

Example:

```python
import photographiq as pg
p = pg.Pattern(pg.CVGraph.line(2))
p.measure(0, pg.Homodyne(pg.Outcome(1))).measure(1)
try:
    p.validate()
except ValueError as exc:
    print("ValueError:", exc)
q = pg.Pattern().extend([pg.Signal("a", pg.Outcome("b")), pg.Signal("b", pg.Outcome("a"))])
try:
    q.validate()
except ValueError as exc:
    print("ValueError:", exc)
try:
    pg.Pattern(inputs=(0,)).extend([pg.Prepare(1), pg.Entangle(1, 1)]).validate()
except ValueError as exc:
    print("ValueError:", exc)
print(pg.protocols.wire((0.0, 0.0)).schedule())
# ValueError: Command 3 depends on a future outcome
# ValueError: Cyclic command dependencies
# ValueError: Self CZ is invalid: command 1 entangles 1 with itself
# (0, 1, 2, 3, 4, 5, 6, 7, 8)
```

### Displacement frames (`frame=True`)

| Field | Value |
|---|---|
| Description | Defers `Displace` commands, propagates the pending shift through subsequent Gaussian gates and loss, and adds it to homodyne/heterodyne/generaldyne records virtually instead of physically displacing; remaining shifts are materialized at the end of the pattern. (The code also flushes before cubic, Kerr and ladder commands, but that path is unreachable because `frame=True` requires a Gaussian backend, which rejects those commands.) `Result.physical_displacements` counts the displacements actually applied. |
| Python API | `pg.simulate(pattern, frame=True)` |
| Implementation | `src/photographiq/simulator.py` (`pending`, `propagate`, `flush`) |
| Backends | `gaussian`, `piquasso`, `native-gaussian` only; other backends raise `NotImplementedError` "Exact displacement-frame tracking requires a Gaussian backend" |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | With `frame=True`, postselected outcomes are given in the same calibrated units as the returned records (including the deferred shift) |
| Tests | `tests/regression/test_inputs_and_frames.py::test_virtual_measurement_and_loss` (frame vs physical, atol 1e-12), `tests/regression/test_rewrites.py::test_standardization_and_frame_equivalence`, `tests/protocols/test_gaussian_postselection.py::test_frame_and_physical_displacements_postselect_identically` |
| Status | **Implemented and validated** |

Example:

```python
import numpy as np, photographiq as pg
p = pg.Pattern(pg.CVGraph.line(2, squeezing=0.5))
p.displace(0, q=0.4, p=0.3).displace(1, q=0.1, p=-0.2)
p.append(pg.Loss(0, 0.8, 0.1)).measure(0, pg.Homodyne(0.3, efficiency=0.8)).displace(1, q=pg.Outcome(0))
a, b = (pg.simulate(p, seed=37, frame=f) for f in (False, True))
print(np.isclose(a.outcomes[0], b.outcomes[0]), np.allclose(a.state.mean, b.state.mean))
print(a.physical_displacements, b.physical_displacements)
try:
    pg.simulate(p, backend="native-mixed-fock", cutoff=6, frame=True)
except NotImplementedError as exc:
    print("NotImplementedError:", exc)
# True True
# 3 1
# NotImplementedError: Exact displacement-frame tracking requires a Gaussian backend
```

### Postselection versus sampling

| Field | Value |
|---|---|
| Description | Without `measurement_outcomes`, outcomes are sampled with the seeded generator (`seed=0` is valid; `run_shots` spawns independent child seeds). With `measurement_outcomes={key: value}`, those outcomes are fixed and the result is the conditional branch; its probability or density is in `measurement_statistics` / `log_likelihood`. Replaying a sampled outcome reproduces the sampled trajectory. |
| Python API | `pg.simulate(pattern, seed=..., measurement_outcomes={...})`, `pg.run_shots(pattern, shots, *, seed=None, **kwargs)` |
| Implementation | `src/photographiq/simulator.py`, `measure(..., outcome=...)` in each backend, `src/photographiq/capabilities.py` (`postselection` feature, key check) |
| Backends | All executing backends (capability `postselection`) |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | Postselection keys must identify measurements (`ValueError`); zero-probability or zero-density branches raise `ValueError` rather than returning -inf likelihood; a postselected result describes one conditional branch, not a deterministic operation, and its success probability is not 1 |
| Tests | `tests/protocols/test_gaussian_postselection.py::test_postselection_replays_sampled_trajectory`, `::test_cz_pair_closed_form_postselection` (closed form), `::test_adaptive_wire_matches_fock_postselection` (Gaussian vs Fock), `::test_gaussian_postselection_guards`, `tests/cross_backend/test_registry_planner.py::test_select_bad_postselection_key` |
| Status | **Implemented and validated** (Gaussian backends; Fock postselection experimental) |

Example:

```python
import numpy as np, photographiq as pg
p = pg.Pattern(pg.CVGraph.line(2, squeezing=0.6)).measure(0, pg.Homodyne.p()).displace(1, q=pg.Outcome(0))
sampled = pg.simulate(p, seed=11)
replay = pg.simulate(p, measurement_outcomes={0: sampled.outcomes[0]})
print(np.allclose(sampled.state.mean, replay.state.mean), replay.measurement_statistics[0]["postselected"])
print(np.isclose(sampled.log_likelihood, replay.log_likelihood))
# True True
# True
```

### Ensembles over sampled trajectories

| Field | Value |
|---|---|
| Description | `run_shots` returns independent seeded trajectories; `values(key)` collects one record across shots and `ensemble_state()` moment-matches the unconditional ensemble (corrections included). |
| Python API | `pg.run_shots(pattern, shots, *, seed=None, **kwargs)`, `ShotResult.values(key)`, `ShotResult.ensemble_state()`, `ShotResult.trajectories` |
| Implementation | `src/photographiq/simulator.py` |
| Backends | All executing backends; `ensemble_state()` needs Gaussian trajectory states (`NotImplementedError` otherwise) |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | The moment-matched state need not describe a Gaussian mixture exactly |
| Tests | `tests/protocols/test_gaussian_protocols.py::test_braunstein_kimble_ensemble`, `::test_cluster_ensemble_channel` (closed-form channel), `tests/analytical/test_measurements.py::test_seed_zero_and_independent_shots` |
| Status | **Implemented and validated** |

Example (Braunstein-Kimble teleportation of a coherent state, 400 shots; the closed form is V_out = V_in + 2 exp(-2r) I):

```python
import numpy as np, photographiq as pg
shots = pg.run_shots(pg.protocols.teleportation(squeezing=1.0), 400, seed=3,
                     inputs={0: pg.GaussianInput.coherent(0.5)})
ens = shots.ensemble_state()
print(np.round(ens.mean, 2), np.round(np.diag(ens.covariance), 2), round(1 + 2 * np.exp(-2), 3))
# [ 1.01 -0.03] [1.26 1.24] 1.271
```

### Pattern standardization

| Field | Value |
|---|---|
| Description | Moves preparations and CZ commands earlier only across disjoint commands and merges adjacent displacements on the same mode; the result is re-validated. |
| Python API | `Pattern.standardize()` |
| Implementation | `src/photographiq/pattern.py` |
| Backends | Backend-neutral |
| CPU / GPU | CPU only |
| Assumptions | Does not promise a full N-E-M-X-Z normal form for arbitrary adaptive patterns; displacements with callables are not merged |
| Tests | `tests/regression/test_rewrites.py::test_standardization_and_frame_equivalence`, `tests/graphix_reference/test_structure.py::test_safe_preparation_standardization_matches_graphix` (Graphix) |
| Status | **Implemented and validated** |

Example:

```python
import numpy as np, photographiq as pg
p = pg.Pattern().extend([pg.Prepare(0, 0.4), pg.Displace(0, 0.2, 0.3), pg.Prepare(1, 0.5), pg.Entangle(0, 1, 0.7)])
p.measure(0, pg.Homodyne.p()).extend([pg.Displace(1, pg.Outcome(0), 0.1), pg.Displace(1, 0.3, -0.1)])
s = p.standardize()
print(len(p.commands), len(s.commands))
print(np.allclose(pg.simulate(p, seed=15).state.mean, pg.simulate(s, seed=15).state.mean))
# 7 6
# True
```

### Flow-derived corrections (CV flow)

| Field | Value |
|---|---|
| Description | `certify_cv_flow` solves the real-linear correction conditions (Booth and Markham 2023) for a supplied total measurement order; `flow_pattern` builds the pattern with homodyne angles atan2(1, shear) and outcome-proportional displacement corrections. Flow search and verification over open graphs (`photographiq.flows`) belong to the flow section. |
| Python API | `photographiq.cvflow.certify_cv_flow(graph, order, *, parameters=None, tolerance=1e-10)` -> `CVFlow(order, corrections, residuals)` or `None`; `photographiq.cvflow.flow_pattern(graph, order, *, shears=None, parameters=None)` |
| Implementation | `src/photographiq/cvflow.py` |
| Backends | Resulting patterns run on all backends |
| CPU / GPU | CPU only |
| Assumptions | The certificate concerns the ideal infinite-squeezing model, not unitarity at finite r; no order search in this module; symbolic weights are frozen at certification |
| Tests | `tests/analytical/test_cvflow.py::test_weighted_real_flow_certificate`, `::test_flow_pattern_and_streaming_wire_have_same_channel`, `tests/analytical/test_cvflow_multimode.py::test_multimode_flow_pattern_matches_graph_teleportation_oracle` (independent oracle), `tests/graphix_reference/test_flow_structure.py::test_line_correction_support_agrees` (Graphix support) |
| Status | **Implemented and validated** |

Example:

```python
import numpy as np, photographiq as pg
from photographiq.cvflow import certify_cv_flow, flow_pattern
g = pg.CVGraph.line(2, inputs=(0,), outputs=(1,))
print(certify_cv_flow(g, [0]).corrections)
print(np.round(pg.gaussian_channel(flow_pattern(g, [0])).matrix, 12))
# {0: {1: 1.0}}
# [[-0. -1.]
#  [ 1.  0.]]
```

### Teleportation corrections (Braunstein-Kimble protocol)

| Field | Value |
|---|---|
| Description | Optical teleportation with two squeezed modes, two beamsplitters, q and p homodyne, and correction Displace(2, sqrt(2) m_q, sqrt(2) m_p). Distinct from the CZ cluster wire. |
| Python API | `pg.protocols.teleportation(*, squeezing=1.0)` |
| Implementation | `src/photographiq/protocols.py` |
| Backends | All executing backends |
| CPU / GPU | CPU (NumPy) / CPU (C++/OpenMP) |
| Assumptions | Ensemble output V_out = V_in + 2 exp(-2r) I; a single trajectory is a conditional state |
| Tests | `tests/protocols/test_gaussian_protocols.py::test_braunstein_kimble_ensemble`, `tests/analytical/test_channels.py::test_wire_noise_and_bk_noise_exact` (closed form) |
| Status | **Implemented and validated** |

Example: see "Ensembles over sampled trajectories" above.

### GKP syndrome extraction with modular feed-forward (pointer)

| Field | Value |
|---|---|
| Description | One finite-ancilla SUM syndrome extraction: the ancilla (a user-supplied finite logical-plus GKP `FockInput`) is coupled to the input by a SUM gate (unit-weight CZ conjugated by ancilla rotations), measured in q, and a `CallableExpression` displaces the input by minus the nearest-lattice residual `decode_shift(m)[0]`. The p variant conjugates the whole protocol by rotations. The output is a measurement-conditioned state, not a deterministic error correction. Full coverage is in Section 27. |
| Python API | `photographiq.gkp.correction_pattern(*, quadrature="q", resource=None, input_node="in", ancilla="gkp", key="syndrome")`, `photographiq.gkp.decode_shift(outcome)` |
| Implementation | `src/photographiq/gkp.py` |
| Backends | Fock backends (the ancilla is a `FockInput`) |
| CPU / GPU | CPU (NumPy/Piquasso) or CPU (C++/OpenMP) |
| Assumptions | `resource=None` raises `ValueError` (the ancilla and its cutoff must be supplied and tested); nearest-cell decoding with half-cell ties mapped to the upper cell, not an optimal finite-energy decoder; finite ancilla and Fock truncation errors remain |
| Tests | `tests/gkp/test_correction.py::test_finite_syndrome_correction_matches_wavefunction_filter`, `tests/gkp/test_syndrome_validation.py::test_syndrome_filter_probability_and_corrected_mean` (independent wavefunction-kernel reference) |
| Status | **Implemented but experimental** (`docs/validation/feature-matrix.md`: "GKP q/p syndrome extraction ... Experimental") |

Example:

```python
import photographiq as pg
from photographiq.gkp import correction_pattern
p = correction_pattern(quadrature="q", resource=pg.GKPCode(cutoff=40).plus())
print([type(c).__name__ for c in p.commands])
print(pg.gkp.decode_shift(0.3), pg.gkp.decode_shift(2.8))
# ['Prepare', 'Rotate', 'Entangle', 'Rotate', 'Measure', 'Displace', 'Output']
# (0.3, 0) (0.2933717253689996, 1)
```

### Logical Pauli frames for GKP feed-forward (pointer)

| Field | Value |
|---|---|
| Description | `LogicalPauliFrame(x_bit, z_bit)` tracks an ideal logical X^x Z^z frame through H, S and CZ and corrects decoded GKP bits; `PhysicalGKPReadout(frame=...)` applies it. Full coverage is in Section 27. |
| Python API | `pg.LogicalPauliFrame(x_bit=0, z_bit=0)`, `.compose(other)`, `.hadamard()`, `.phase()`, `.cz(other)`, `.correction(basis)`, `.xy_angle(alpha)` |
| Implementation | `src/photographiq/encoded.py` |
| Backends | Classical bookkeeping (used with Fock backends) |
| CPU / GPU | CPU only |
| Assumptions | Ideal frame modulo global phase; no envelope correction |
| Tests | `tests/gkp/measurements/test_encoded.py::test_frame_against_qubit_conjugation` (qubit conjugation reference) |
| Status | **Implemented and validated** |

Example: see `docs/tutorials/45-gkp-frames.md`.

## 12. Native C++ Gaussian backend

The compiled extension `photographiq._native` is built from `cpp/` (C++20, Eigen 3.4.0, pybind11, optional OpenMP) by scikit-build-core during installation; see [native build](development/native-build.md). It has three components, `gaussian`, `fock` and `sampling`. The Python package works without it: the NumPy `gaussian` backend is the default, and native backends raise `ImportError` with build instructions when the extension or a component is missing. All backends, native or not, are found through one registry.

### Backend registry and planner

| Field | Value |
|---|---|
| Description | Static description of every registered backend (representation, implementation family, device, precision, declared approximations, requirements). It also includes an allocation-free planner that picks the cheapest available backend whose class capabilities cover a pattern's required features and whose `preflight` passes. |
| Python API | `photographiq.available_backends()`, `photographiq.describe_backend(name)`, `photographiq.select_backend(pattern, inputs=None, *, initial_state=None, measurement_outcomes=None, prefer=("native", "numpy"), cutoff=None)` returning `photographiq.registry.BackendSelection(name, representation, features, explanation, rejected)`; also `photographiq.registry.create_backend(name, cutoff=None)`, `photographiq.capabilities.required_features(pattern, inputs=None, initial_state=None, measurement_outcomes=None)`, `photographiq.registry.missing_features(name, features)`, `photographiq.registry.BACKENDS` (read-only mapping of `BackendInfo`) |
| Implementation | `src/photographiq/registry.py`, `src/photographiq/capabilities.py` |
| Backends | Registry entries: `gaussian`, `piquasso`, `piquasso-fock`, `piquasso-mixed-fock`, `native-gaussian`, `native-fock`, `native-mixed-fock` |
| CPU / GPU | CPU only (metadata) |
| Assumptions | Capabilities are read from the backend class attribute `capabilities`, never duplicated. Cost order: representation (gaussian < pure-fock < mixed-fock), then the `prefer` order of implementation families, then registration order. Fock backends are skipped unless `cutoff` is given. Availability checks use `importlib.util.find_spec` and `photographiq.native.components()`; importing the registry imports neither Piquasso, CUDA-Q nor JAX. If no backend qualifies, `select_backend` raises `NotImplementedError` and lists each candidate's reason. `simulate(backend="<name>")` resolves names through the same registry (`create_backend`); an unknown name raises `ValueError("Unknown backend: ...")`. |
| Tests | `tests/cross_backend/test_registry_planner.py` (`test_entry_resolves_or_reports_reason`, `test_capabilities_are_read_from_class`, `test_registry_import_is_lazy`, `test_planner_feature_check_agrees_with_preflight` (the capability check is compared against each backend's real `preflight`), `test_select_gaussian_pattern`, `test_select_non_gaussian_pattern`, `test_select_mixed_pattern`, `test_select_raises_for_unsupported_combination`, `test_select_rejects_cutoff_too_small_via_preflight`) |
| Status | **Implemented but experimental** (behavioural tests only: the capability check is compared with each backend's own `preflight`, which is internal consistency, not an independent reference; matches Section 2) |

```python
import photographiq as pg
print(pg.available_backends())
# ('gaussian', 'piquasso', 'piquasso-fock', 'piquasso-mixed-fock', 'native-gaussian', 'native-fock', 'native-mixed-fock')
d = pg.describe_backend("native-fock")
print(d["representation"], d["implementation"], d["device"], d["available"], d["approximations"])
# pure-fock native cpu True ('Fock truncation by total photon number sum(n) < cutoff',)
p = pg.Pattern(inputs=(0,)).append(pg.Squeeze(0, 0.4)).append(pg.Measure(0, pg.Homodyne(0.0)))
print(pg.select_backend(p).explanation)
# 'native-gaussian': cheapest available gaussian backend covering [homodyne]
```

The printed `available_backends()` tuple depends on the environment. This output is from the development machine: native extension built, Piquasso installed, CUDA-Q absent.

### Native extension loader and build information

| Field | Value |
|---|---|
| Description | Loads `photographiq._native`, or a development build named by the environment variable `PHOTOGRAPHIQ_NATIVE_PATH`, and reports what was compiled. |
| Python API | `photographiq.native.AVAILABLE`, `photographiq.native.components()`, `photographiq.native.build_info()`, `photographiq.native.require(component)` |
| Implementation | `src/photographiq/native.py`, `cpp/src/bindings/module.cpp`, `cpp/CMakeLists.txt` |
| Backends | `native-gaussian`, `native-fock`, `native-mixed-fock`, and the native sampling kernels |
| CPU / GPU | CPU (C++); no CUDA toolkit is used |
| Assumptions | Components are chosen at build time with `PGQ_COMPONENTS` (default `gaussian;fock;sampling`). OpenMP is used when the compiler provides it (`PGQ_USE_OPENMP=ON`, the default). `photographiq.native.load_error` holds the import error when the extension is missing; `components()` then returns `()` and `build_info()` returns `{}`. `require` raises `ImportError` naming the build command when the extension or a component is missing. `build_info()` reports compiler, build type, C++ standard, OpenMP, Eigen version and `hbar` (2.0). |
| Tests | `native.require` is called by `tests/native/test_native_fock_backend.py`, `test_native_fock_cross_layer.py`, `test_native_fock_kernels.py`, `test_native_gaussian_diagnostics.py` and `test_native_mixed_fock_backend.py`; `tests/native/test_native_gaussian_benchmark_smoke.py` reads `native.components()`. `tests/cross_backend/test_registry_planner.py::test_native_names_resolve_or_raise_import_error` checks the `ImportError` branch only when the extension is absent; with the extension built (as here) it takes the success branch. No test reads `build_info()` (the benchmark scripts record it). C++ unit tests are in `cpp/tests/` (built with `-DPGQ_BUILD_CPP_TESTS=ON`, run with `ctest`, see [native build](development/native-build.md)); they were not run for this document. |
| Status | **Implemented but experimental** (no test exercises the missing-extension path or `build_info()` in a built environment) |

```python
import photographiq.native as native
print(native.components())
# ('gaussian', 'fock', 'sampling')
info = native.build_info()
print(info["openmp"], info["cxx_standard"], info["eigen"], info["hbar"])
# True 202002 3.4.0 2.0
```

### NativeGaussianBackend (`native-gaussian`)

| Field | Value |
|---|---|
| Description | Drop-in subclass of the NumPy `GaussianBackend` whose state updates run in compiled kernels that update a contiguous mean vector and a Fortran-ordered covariance in place. A gate on k of n modes costs O(n k) instead of the dense O(n^3) `S V S^T`. |
| Python API | `photographiq.simulate(pattern, backend="native-gaussian", ...)`, `photographiq.run_shots(...)`, `photographiq.backends.native_gaussian.NativeGaussianBackend()` |
| Implementation | `src/photographiq/backends/native_gaussian.py`; `cpp/src/gaussian/operations.cpp`, `cpp/src/gaussian/conditioning.cpp`, `cpp/src/bindings/gaussian_bindings.cpp` |
| Backends | `native-gaussian` |
| CPU / GPU | CPU (C++, Eigen). Single-state kernels are serial; OpenMP is used only by the batched kernels (next features). |
| Assumptions | Same semantics as `GaussianBackend`: hbar=2, [q,p]=2i, interleaved xpxp, statistical covariance with vacuum covariance I, exact Schur-complement homodyne/heterodyne/generaldyne conditioning, homodyne records calibrated to the incident quadrature, loss with thermal photons, quadratic phase, postselection with densities, displacement frames (`frame=True`). Outcomes are sampled in Python from natively computed moments with the same `numpy.random.Generator` calls as `GaussianBackend`, so equal seeds give equal trajectories up to summation order. External covariances are symmetrised on load. Postselection on an outcome with zero numerical density raises `ValueError`. The compiled module also has a general Gaussian-channel kernel `photographiq._native.gaussian.apply_channel(mean, cov, modes, X, Y)` that rejects non-completely-positive (X, Y); no Python API or backend method wraps it (tested by `tests/native/test_native_gaussian_diagnostics.py::test_channel_kernel_matches_dense_algebra_and_rejects_non_cp`). |
| Tests | `tests/native/test_native_gaussian_backend.py` (`test_random_patterns_match_numpy_backend` and `test_postselection_matches_numpy_backend`, against the NumPy reference backend at rtol 1e-12; `test_near_singular_conditioning_large_squeezing`, r = 8, rtol 1e-8; `test_tiny_variance_homodyne_closed_form`, closed form; `test_errors_match_numpy_backend`; `test_snapshots_are_independent_of_backend_storage`); `tests/cross_backend/test_cross_backend_agreement.py` (`test_gaussian_pattern_agreement`, against Piquasso and the Fock backends); `tests/validation/test_val_gaussian.py` (added 2026-11-12, references computed from first principles inside the test: states and purities, gate symplectics built as `expm(2 Omega M)` from each generator, loss and thermal loss against analytic X, Y and a beam-splitter dilation, homodyne and heterodyne conditioning against a physical dilation, the heterodyne density against the Husimi function, the adaptive wire against the analytic map T(k), physicality of random 5-mode circuits; largest error 6.5e-13 against a tolerance of 1e-11, see [independent validation matrix](validation/independent-validation-matrix.md)) |
| Status | **Implemented and validated** |

```python
import numpy as np
import photographiq as pg
from photographiq.cvflow import flow_pattern
graph = pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,))
pattern = flow_pattern(graph, (0, 1, 2), shears={0: 0.3, 1: -0.5, 2: 0.0})
a = pg.simulate(pattern, backend="gaussian", seed=7)
b = pg.simulate(pattern, backend="native-gaussian", seed=7)
print(max(abs(a.outcomes[k] - b.outcomes[k]) for k in a.outcomes) < 1e-10)
# True
print(np.max(np.abs(a.state.covariance - b.state.covariance)) < 1e-10)
# True
```

### Native Gaussian diagnostics

| Field | Value |
|---|---|
| Description | Compiled versions of the `GaussianState` diagnostics: sorted symplectic eigenvalues, purity, von Neumann entropy, base-2 logarithmic negativity of a bipartition, and squared Uhlmann fidelity. |
| Python API | `photographiq.backends.native_gaussian.symplectic_eigenvalues(state)`, `purity(state)`, `entropy(state, tolerance=1e-7)`, `log_negativity(state, nodes)`, `fidelity(state, other, pure_tolerance=1e-9)` |
| Implementation | `src/photographiq/backends/native_gaussian.py`; `cpp/src/gaussian/diagnostics.cpp` |
| Backends | Any `GaussianState` (from `gaussian`, `native-gaussian` or `piquasso`) |
| CPU / GPU | CPU (C++) |
| Assumptions | hbar=2, vacuum covariance I, so the vacuum symplectic eigenvalue is 1. Symplectic eigenvalues use a Cholesky factorisation followed by a Hermitian eigensolver. Purity is det(V)^(-1/2). Entropy is in nats and raises `ValueError` for an eigenvalue below 1 - tolerance. Fidelity reorders `other` to `state`'s labels; it uses the overlap formula when one state is pure and the general mixed-state formula otherwise. |
| Tests | `tests/native/test_native_gaussian_diagnostics.py` (`test_spectral_diagnostics_match_williamson_and_numpy` (Williamson decomposition and NumPy), `test_tmsv_entropy_and_negativity_closed_forms` (two-mode squeezed vacuum closed forms), `test_thermal_entropy_and_unphysical_rejection`, `test_fidelity_closed_forms`, `test_fidelity_matches_numpy_states_and_symmetry`, `test_fidelity_matches_piquasso` (Piquasso)); `tests/validation/test_val_gaussian.py::test_thermal_entropy_and_native_diagnostics`, `::test_mixed_state_spectral_diagnostics`, `::test_tmsv_log_negativity` (independent closed forms on the native functions) |
| Status | **Implemented and validated** |

```python
import numpy as np
import photographiq as pg
from photographiq.backends import native_gaussian as ng
r = 0.5
pattern = (pg.Pattern(inputs=())
           .append(pg.Prepare(0, squeezing=r))                         # p-squeezed
           .append(pg.Prepare(1, state=pg.GaussianInput.squeezed(r)))  # q-squeezed
           .append(pg.BeamSplitter(0, 1, np.pi / 4)))
state = pg.simulate(pattern, backend="native-gaussian").state
print(round(ng.log_negativity(state, (0,)), 6), round(2 * r / np.log(2), 6))  # TMSV: E_N = 2r/ln 2
# 1.442695 1.442695
print(round(ng.purity(state), 12), round(state.log_negativity((0,)), 6))
# 1.0 1.442695
```

### Native batched Gaussian kernels

| Field | Value |
|---|---|
| Description | Functional API over a batch of B independent states. It applies one symplectic matrix, or one per state, to listed mode indices, computes calibrated homodyne moments, and conditions each state on its given homodyne outcome. |
| Python API | `photographiq.backends.native_gaussian.batch_apply_symplectic(means, covariances, modes, matrices)`, `batch_homodyne_moments(means, covariances, mode, angles, efficiency=1.0, noise=0.0)`, `batch_condition_homodyne(means, covariances, mode, angles, outcomes, efficiency=1.0, noise=0.0)` |
| Implementation | `src/photographiq/backends/native_gaussian.py`; `cpp/src/gaussian/batch.cpp` (`#pragma omp parallel for` over the batch) |
| Backends | Standalone functions on arrays (means `(B, 2n)`, covariances `(B, 2n, 2n)`); not a backend |
| CPU / GPU | CPU (C++/OpenMP) |
| Assumptions | Modes are integer indices, not labels. The kernels use the symmetric part (V + V^T)/2 of each covariance. A non-symplectic matrix or bad shapes raise `ValueError`; a non-integer mode index raises `TypeError` from the binding. An outcome whose density underflows returns density 0 without raising (unlike `GaussianBackend.measure`). `batch_condition_homodyne` conditions on **given** outcomes, so a batch is a set of postselected branches with their outcome densities, not a deterministic operation. There is no batched trajectory runner; use `simulate`/`run_shots`. |
| Tests | `tests/native/test_native_gaussian_diagnostics.py` (`test_batch_symplectic_matches_dense_reference` (dense S V S^T), `test_batch_homodyne_matches_numpy_backend` (NumPy `GaussianBackend.measure`), `test_batch_kernels_use_the_symmetric_part`); `tests/native/test_native_gaussian_benchmark_smoke.py`; `tests/gpu/test_gpu_gaussian.py::test_matches_native_batch_kernels` |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq.backends import native_gaussian as ng
B = 5
means = np.zeros((B, 4)); covs = np.broadcast_to(np.eye(4), (B, 4, 4)).copy()
S = np.diag([np.exp(-0.3), np.exp(0.3)])                 # squeeze mode index 1
m2, c2 = ng.batch_apply_symplectic(means, covs, [1], S)
print(c2.shape, np.allclose(c2[:, 2, 2], np.exp(-0.6)))
# (5, 4, 4) True
mo, co, dens = ng.batch_condition_homodyne(m2, c2, 1, angles=0.0, outcomes=np.linspace(-1, 1, B))
print(co.shape, np.round(dens, 4))                       # five postselected branches
# (5, 2, 2) [0.2165 0.4288 0.5385 0.4288 0.2165]
```

Runtime comparisons are made by `benchmarks/gaussian/bench_native_gaussian.py`; methodology is in `benchmarks/README.md`. No timing numbers are quoted here.

### Photon counting on Gaussian backends

| Field | Value |
|---|---|
| Description | Photon-number measurement on a Gaussian (covariance-matrix) backend. |
| Python API | `pg.Measure(node, pg.PhotonNumber())` with `backend="gaussian"` or `"native-gaussian"` |
| Implementation | `src/photographiq/backends/native_gaussian.py` (`NativeGaussianBackend.measure`), `src/photographiq/backends/gaussian.py` (`GaussianBackend.measure`), `src/photographiq/capabilities.py` (`preflight`) |
| Backends | None of the Gaussian backends; use `native-fock`, `piquasso-fock` or a mixed-Fock backend |
| CPU / GPU | Not applicable |
| Assumptions | The conditional state after photon counting is non-Gaussian. The capability check in `simulate`/`select_backend` rejects the pattern before execution. A direct `NativeGaussianBackend.measure` call raises `NotImplementedError("Photon counting has non-Gaussian conditional states; use native-fock, native-mixed-fock, piquasso-fock or piquasso-mixed-fock")`. A direct NumPy `GaussianBackend.measure` call raises `NotImplementedError("Photon counting has non-Gaussian conditional states; use a Fock backend (native-fock, native-mixed-fock, piquasso-fock or piquasso-mixed-fock)")`; before the fix it named only `piquasso-fock`. |
| Tests | `tests/native/test_native_gaussian_backend.py::test_errors_match_numpy_backend`; `tests/cross_backend/test_registry_planner.py::test_planner_feature_check_agrees_with_preflight`; `tests/regression/test_validation_messages.py::test_photon_counting_messages_mention_native_fock_backends` (the `NativeGaussianBackend` message and that of `photographiq.interactive.photon_number_marginal`; the `GaussianBackend` wording was checked by hand for this section) |
| Status | **Unsupported** |

Example: see `tests/native/test_native_gaussian_backend.py::test_errors_match_numpy_backend`.

## 13. Native C++ Fock backend

`native-fock` (pure states) and `native-mixed-fock` (density matrices, section 14) keep every Python code path of the Piquasso-backed Fock backends: preparation, renormalisation with retained-norm diagnostics, photon counting, homodyne, instruments, GKP readout and ladder operations. Only the state object and the gate kernels (and, for `native-mixed-fock`, the loss kernel and density-matrix construction) are replaced, by `photographiq._native.fock`, and Piquasso is never imported. Truncation is by total photon number (sum n < cutoff), with states stored in Piquasso's basis order.

Status policy in this section. Until 2026-11-11 every Fock path was listed as experimental, because `src/photographiq/backends/fock.py` and `docs/validation/feature-matrix.md` call Fock execution experimental. On 2026-11-12 the independent validation suite `tests/validation/test_val_fock.py` (see [independent validation matrix](validation/independent-validation-matrix.md)) added references computed from first principles for both native engines: number, coherent and squeezed states against mpmath amplitudes; every gate (displacement, rotation, squeezing, beam splitter, CZ, quadratic phase, Kerr, cross-Kerr, cubic phase) against `scipy.linalg.expm` on an enlarged per-mode space, with the truncation handled exactly as P U psi / norm(P U psi); photon counting, partial traces, homodyne densities and loss as a dilation. The largest gate error is 5e-16 against a tolerance of 1e-12. Entries covered by these references are now **Implemented and validated**, and each Tests row states the reason. The feature matrix still marks the Fock paths E. Piquasso Fock engines appear in that suite only as a secondary check.

### NativeFockBackend (`native-fock`)

| Field | Value |
|---|---|
| Description | Pure-state truncated Fock trajectory backend on the C++ engine, with the semantics and diagnostics of `piquasso-fock` plus a `cross_kerr` capability. |
| Python API | `photographiq.simulate(pattern, backend="native-fock", cutoff=c, ...)`; `photographiq.backends.native_fock.NativeFockBackend(cutoff=None, *, cubic_phase="truncated_generator", cubic_tolerance=1e-12, **kwargs)` with `kwargs` `norm_tolerance=1e-3`, `max_dimension=1_000_000`, `boundary_warning=0.02` (passed to `PiquassoFockBackend`) |
| Implementation | `src/photographiq/backends/native_fock.py`, `src/photographiq/backends/fock.py` (inherited Python paths); `cpp/src/fock/basis.cpp`, `cpp/src/fock/gates.cpp`, `cpp/src/fock/kernels.cpp`, `cpp/src/bindings/fock_bindings.cpp` |
| Backends | `native-fock` |
| CPU / GPU | CPU (C++; gate application uses OpenMP only above a fixed work threshold, 2^22, in `cpp/src/fock/kernels.cpp`); measurement conditioning and sampling run in Python/NumPy |
| Assumptions | hbar=2, q = a + a^dag, p = -i(a - a^dag), total-photon cutoff sum n < cutoff (cutoff >= 2, required). After each operation the state is renormalised, and the retained norm and the boundary population (weight at total photon number >= cutoff - 2) are recorded in `diagnostics`. A norm loss above `norm_tolerance` raises `ValueError`, and a boundary population above `boundary_warning` warns. Gate matrices are defined up to a global phase. `max_dimension` raises `MemoryError` before allocation. |
| Tests | `tests/native/test_native_fock_backend.py` (`test_gaussian_pattern_matches_exact_gaussian_moments` (exact Gaussian moments, atol 2e-7), `test_non_gaussian_patterns_match_piquasso`, `test_piquasso_difference_vanishes_with_cutoff`, `test_sampled_trajectories_match_piquasso`, `test_truncation_errors_warnings_and_snapshots`, `test_native_backends_never_import_piquasso`); `tests/native/test_native_fock_cross_layer.py` (Piquasso); `tests/cross_backend/test_cross_backend_agreement.py` (`test_cat_photon_counting_closed_form`, `test_homodyne_postselection_density_closed_form`); `tests/validation/test_val_fock.py` (states against mpmath, all eight gates and the cubic phase against enlarged-space `expm`, counting, homodyne density, partial trace, inner-product preservation of number-conserving gates, norm and positivity after a 21-gate sequence). Upgraded to validated on 2026-11-12 because of this independent suite. |
| Status | **Implemented and validated** (`src/photographiq/backends/fock.py` and the feature matrix still call Fock execution experimental) |

```python
import numpy as np
import photographiq as pg
hom = pg.Pattern(inputs=(0, 1)).append(pg.BeamSplitter(0, 1, np.pi / 4))   # Hong-Ou-Mandel
inputs = {0: pg.FockInput.number(1), 1: pg.FockInput.number(1)}
state = pg.simulate(hom, inputs=inputs, backend="native-fock", cutoff=5).state
print({k: round(v, 12) for k, v in state.probabilities.items() if v > 1e-15})
# {(2, 0): 0.5, (0, 2): 0.5}
pattern = (pg.Pattern(inputs=(0,)).append(pg.Squeeze(0, 0.2))
           .append(pg.CubicPhase(0, 0.1)).append(pg.Kerr(0, 0.05)))
one = {0: pg.FockInput.number(1)}
a = pg.simulate(pattern, inputs=one, backend="native-fock", cutoff=20).state
b = pg.simulate(pattern, inputs=one, backend="piquasso-fock", cutoff=20).state
print(f"{1 - pg.fidelity(a, b):.1e}")
# 0.0e+00
```

### Exact Gaussian gate matrix elements

| Field | Value |
|---|---|
| Description | Displacement, squeezing, weighted CZ (`Entangle`), quadratic phase, beam splitter and Gaussian inputs use exact Fock matrix elements of the full Gaussian unitary, applied as P U P on the truncated space. One-mode unitaries (U^dag x U = S x + d) are factorised as U = R(phi1) S_r R(phi2) D(beta): the rotations are exact phases, S = rotation(phi1) diag(e^-r, e^r) rotation(phi2) comes from the 2 x 2 SVD, the squeeze factor uses a closed form in Jacobi polynomials evaluated by a stable three-term recurrence, D(beta) uses a rectangular Laguerre recurrence, and the product runs over an internal size L >= cutoff chosen by a rigorous bound on the displacement tail (discarded part <= 1e-16 for every column). Multi-mode unitaries (CZ, beam splitter, general transforms) use the Bargmann recurrence. |
| Python API | Pattern commands `pg.Displace`, `pg.Squeeze`, `pg.Entangle`, `pg.QuadraticPhase`, `pg.BeamSplitter`, `pg.Rotate`; Gaussian inputs `pg.GaussianInput`; kernel `photographiq.native.require("fock").gaussian_unitary(symplectic, displacement, cutoff)` |
| Implementation | `cpp/src/fock/gates.cpp` (`gaussian_unitary`, `single_mode_unitary`, `displacement_tail_cutoff`, `displacement_matrix`), `src/photographiq/backends/native_fock.py` (`_NativeGates`) |
| Backends | `native-fock`, `native-mixed-fock` |
| CPU / GPU | CPU (C++) |
| Assumptions | Same operator and global phase as the Bargmann form (`<0\|U_S\|0>` real positive); in the old path's accurate regime the old and new results agree to 5.4e-14. One-mode accuracy: at most 1.9e-14 against 60-digit mpmath for abs(alpha) <= 6, abs(r) <= 1, cutoff <= 100 (squeezing alone up to size 800); outside that range the method and the tail bound still hold but rounding has not been measured. Before the fix the same grid reached 8.5e-5 at abs(alpha) = 4, r = 0.3, cutoff 80 and 1.4e2 at abs(alpha) = 6, cutoff 100 (the Bargmann column recurrence amplifies rounding by about sqrt(N + 1)/sqrt(n + 1) per step). The combined squeeze-and-displace cost is O(cutoff^2 L); in the backends it arises only for `GaussianInput`, and very hot mixed Gaussian inputs (size up to 4096) can take minutes. **Multi-mode displaced unitaries** still use the Bargmann recurrence with no general error bound (3e-12 measured at abs(delta) <= 4, r <= 0.6, cutoff <= 40 on two modes). The backends never request one (CZ, beam splitter and Gaussian transforms have d = 0), but a direct `gaussian_unitary` call can. Piquasso applies CZ, quadratic phase and general Gaussian transforms through an Euler (Bloch-Messiah) decomposition, truncating after every factor, so at a fixed cutoff the two engines differ at truncation level and both converge as the cutoff grows. Rotation is an exact phase. Details: [native Fock Gaussian unitary](validation/native-fock-gaussian-unitary.md). |
| Tests | `tests/native/test_native_fock_gaussian_unitary_stability.py` (60-digit mpmath references built in the test, about 15 s: `test_combined_squeeze_displacement_matches_high_precision` at abs(alpha) = 4, r = 0.3, cutoff 80; abs(alpha) = 6, r = 0.3, cutoff 100; abs(alpha) = 3, r = 1, cutoff 100; abs(alpha) = 0.5, cutoff 40; atol 1e-12 for the full matrix and 3e-14 for column 0, plus column norms against the exact leakage; `test_squeezing_rows_stay_accurate_at_large_size`; `test_thermal_columns_of_displaced_squeezed_input`; 5 of its 6 tests fail on the old build); `cpp/tests/test_fock.cpp::test_single_mode_stability` (Heisenberg relation at abs(alpha) = 6, r = 0.3, cutoff 100; S(-1) S(1) = 1 on 70 x 70 at size 800); `tests/validation/test_val_fock.py::test_gate_against_enlarged_expm` (every gate against `expm` on an enlarged space); `tests/native/test_native_fock_kernels.py` (`test_coherent_and_squeezed_vacuum_closed_forms`, `test_passive_gates_are_exact_and_unitary_per_sector`, `test_hong_ou_mandel`, `test_projected_gaussian_gates_are_contractions`, `test_gaussian_matrix_elements_are_numerically_stable_at_large_cutoff`, `test_heisenberg_action_matches_symplectic` (symplectic action), `test_local_gates_on_multimode_states_match_piquasso_truncation_semantics`). Upgraded to validated on 2026-11-12 because of the mpmath and `expm` references. |
| Status | **Implemented and validated** (one-mode unitaries and every gate the backends request; direct multi-mode displaced `gaussian_unitary` calls have no error bound) |

Example (the formerly inaccurate case abs(alpha) = 4, r = 0.3, cutoff 80, against `expm` on a 400-level space):

```python
import numpy as np
from scipy.linalg import expm
from photographiq import native
F = native.require("fock")
alpha, r, cutoff, big = 4.0 * np.exp(0.7j), 0.3, 80, 400          # the formerly inaccurate case
a = np.diag(np.sqrt(np.arange(1, big)), 1)
ref = expm(alpha * a.T - np.conj(alpha) * a) @ expm(r * (a @ a - a.T @ a.T) / 2)
U = F.gaussian_unitary(np.diag([np.exp(-r), np.exp(r)]), [2 * alpha.real, 2 * alpha.imag], cutoff)
print(U.shape, f"{np.abs(U - ref[:cutoff, :cutoff]).max():.1e}")
# (80, 80) 3.9e-15
```

### Cubic phase: two matrix definitions

| Field | Value |
|---|---|
| Description | The cubic phase exp(i gamma q^3/6) can be built in two ways. `"truncated_generator"` (default) is Piquasso's expm(i gamma Q^3/6) of the truncated position matrix. It is unitary on the truncated space and never reports leakage. `"projected"` builds the exact projected elements of the exact gate by a converged nested trapezoidal rule, so leakage above the cutoff appears in the retained-norm diagnostics. |
| Python API | `pg.CubicPhase(node, gamma)`; selection via `NativeFockBackend(cutoff, cubic_phase="projected", cubic_tolerance=1e-12)` or `NativeMixedFockBackend(...)` passed as `backend=` |
| Implementation | `cpp/src/fock/gates.cpp`, `src/photographiq/backends/native_fock.py` (`_cubic_phase_matrix`, LRU-cached) |
| Backends | `native-fock`, `native-mixed-fock` (`piquasso-fock` provides only the truncated-generator form) |
| CPU / GPU | CPU (C++) |
| Assumptions | Both forms converge to the exact gate as the cutoff grows. The truncated-generator form equals the Gauss-Hermite (DVR) quadrature at the cutoff nodes and is identical to Piquasso to about 1e-13. `cubic_tolerance` must lie in (0, 1e-6). With the projected form, leakage can trigger the `norm_tolerance` error where Piquasso continues silently. Non-finite gamma raises `ValueError`. |
| Tests | `tests/native/test_native_fock_kernels.py` (`test_cubic_phase_projected_against_direct_integration` (direct numerical integration), `test_cubic_phase_heisenberg_action`, `test_cubic_truncated_generator_matches_piquasso_and_converges_to_projected` (Piquasso), `test_projected_cubic_phase_reports_its_truncation_error_exactly`); `tests/native/test_native_fock_backend.py` (`test_cubic_phase_converges_to_exact_state_and_matches_piquasso_definition`, `test_cubic_phase_moments_converge_for_both_definitions`); `tests/validation/test_val_fock.py::test_cubic_phase_both_definitions` (truncated generator against `expm` of the truncated q^3, 4.4e-16; projected form against P V P from an independent quadrature, 2.9e-16; 1 - F below 1e-6 at cutoff 28 for both) and `::test_cubic_reference_quadrature_converges`. Upgraded to validated on 2026-11-12 because of these independent references. |
| Status | **Implemented and validated** (feature matrix: E) |

```python
import photographiq as pg
from photographiq.backends.native_fock import NativeFockBackend
from photographiq.backends.fock import PiquassoFockBackend
be = NativeFockBackend(cutoff=4, cubic_phase="projected")
print(be.cubic_phase_method, sorted(be.capabilities - PiquassoFockBackend.capabilities))
# projected ['cross_kerr']
```

### Kerr and cross-Kerr phases

| Field | Value |
|---|---|
| Description | Kerr exp(i kappa n^2) and cross-Kerr exp(i kappa n_u n_v) as exact diagonal phases. |
| Python API | `pg.Kerr(node, kappa)` (Pattern command); cross-Kerr only as the backend method `NativeFockBackend.cross_kerr(u, v, kappa)` / `NativeMixedFockBackend.cross_kerr(u, v, kappa)` |
| Implementation | `src/photographiq/backends/native_fock.py`; `cpp/src/fock/kernels.cpp` (`phase_pure`, `phase_mixed`) |
| Backends | Kerr: `native-fock`, `native-mixed-fock`, `piquasso-fock`, `piquasso-mixed-fock`. Cross-Kerr: native Fock backends only |
| CPU / GPU | CPU (C++; `phase_mixed` uses OpenMP above the work threshold, `phase_pure` is serial) |
| Assumptions | There is no cross-Kerr Pattern command, so cross-Kerr cannot be used from `simulate`; call the method on a backend instance whose state you prepared directly. |
| Tests | `tests/native/test_native_fock_backend.py` (`test_kerr_cross_kerr_and_hong_ou_mandel` (closed-form phases), `test_cross_kerr_matches_piquasso_cross_kerr_gate` (Piquasso `CrossKerr`)); `tests/native/test_native_fock_kernels.py::test_phase_kernels_kerr_and_cross_kerr`; `tests/validation/test_val_fock.py::test_gate_against_enlarged_expm` (Kerr 1.8e-16 and cross-Kerr 9.0e-17 against `expm`, both native engines) and `::test_number_conserving_gates_preserve_inner_products`. Upgraded to validated on 2026-11-12 because of these independent references. |
| Status | **Implemented and validated** (Kerr on the native engines). Cross-Kerr is **Partially supported**: backend method only, no Pattern command (as in Section 9) |

Example: see `tests/native/test_native_fock_backend.py::test_kerr_cross_kerr_and_hong_ou_mandel`.

### Measurements on the native pure Fock backend

| Field | Value |
|---|---|
| Description | Destructive ideal homodyne (sampled or postselected), photon counting, measurement instruments and physical GKP readout on pure states, inherited from `PiquassoFockBackend.measure`. |
| Python API | `pg.Measure(node, pg.Homodyne(angle))`, `pg.Measure(node, pg.PhotonNumber())`, `Pattern.measure(node, pg.MeasurementInstrument(...))`, `pg.PhysicalGKPReadout`; `measurement_outcomes=` for postselection |
| Implementation | `src/photographiq/backends/fock.py` (`measure`), `src/photographiq/fock_measurements.py` (`homodyne_projection`), `src/photographiq/instruments.py`, `src/photographiq/encoded.py` (`PhysicalGKPReadout.execute`). Conditioning reads the amplitude dictionary `fock_amplitudes_map` in Python; no native measurement kernel is used. |
| Backends | `native-fock`, `piquasso-fock` |
| CPU / GPU | CPU (Python/NumPy conditioning; for `native-fock` the state vector is stored by the C++ engine) |
| Assumptions | Homodyne with `efficiency != 1` or `noise != 0` raises `NotImplementedError("Noisy Fock homodyne requires mixed-state conditioning")`; use a mixed-Fock backend. Heterodyne and generaldyne raise `NotImplementedError("Fock backend supports ideal Homodyne and PhotonNumber measurements only")`. Postselected outcomes give one conditional branch; its probability or density is in `Result.measurement_statistics`. |
| Tests | `tests/native/test_native_fock_backend.py` (`test_sampled_trajectories_match_piquasso`, `test_instrument_and_gkp_readout_match_piquasso`); `tests/cross_backend/test_cross_backend_agreement.py::test_homodyne_postselection_density_closed_form` (closed form); `tests/validation/test_val_fock.py` (`test_photon_counting_probability_and_conditional_state`, `test_fock_homodyne_density_convention`, `test_partial_trace_by_explicit_contraction`); `tests/regression/test_measure_angle.py` (a direct `measure` uses the `Homodyne` angle) |
| Status | **Partially supported** (ideal homodyne, photon counting, instruments and GKP readout work; noisy/inefficient homodyne, heterodyne and generaldyne raise `NotImplementedError` on pure Fock backends) |

Example: see `tests/native/test_native_fock_backend.py::test_instrument_and_gkp_readout_match_piquasso`.

Runtime comparisons of `native-fock`/`native-mixed-fock` against the Piquasso engines are made by `benchmarks/fock/bench_fock.py`, which checks the agreement of the two engines' output states before timing (Section 32); see `benchmarks/README.md`.

## 14. Mixed-Fock simulation

Covers the Piquasso-based `piquasso-mixed-fock` and the native `native-mixed-fock` density-matrix backends.

Two backends evolve truncated density matrices. `piquasso-mixed-fock` applies gates with Piquasso's `FockSimulator` and does loss, conditioning and ladder operations locally in NumPy/SciPy. `native-mixed-fock` reuses all of that Python code and replaces the state storage and the gate and loss kernels with C++. Both truncate by total photon number. A density matrix has D^2 complex entries, with D = comb(modes + cutoff - 1, modes). See also [mixed Fock](user-guide/mixed-fock.md).

### Density-matrix evolution with Piquasso (`piquasso-mixed-fock`)

| Field | Value |
|---|---|
| Description | Mixed-state trajectory backend. It supports pure and mixed inputs, Gaussian and non-Gaussian gates (cubic phase, Kerr), loss, noisy homodyne, photon counting, instruments, ladder operations and postselection. |
| Python API | `photographiq.simulate(pattern, backend="piquasso-mixed-fock", cutoff=c, ...)`; `photographiq.backends.mixed_fock.MixedFockBackend(cutoff=None, *, max_matrix_bytes=256_000_000, **kwargs)` with `kwargs` `norm_tolerance`, `max_dimension`, `boundary_warning` (as for `piquasso-fock`) |
| Implementation | `src/photographiq/backends/mixed_fock.py` |
| Backends | `piquasso-mixed-fock` (requires the optional `piquasso` package) |
| CPU / GPU | CPU (Piquasso for gates; NumPy/SciPy for loss, conditioning and sampling) |
| Assumptions | hbar=2, total-photon cutoff sum n < cutoff, renormalisation after each operation with the retained trace recorded. Normalising a truncated tail does not turn it into physical loss, so inspect `diagnostics`. Capabilities add `mixed_fock`, `loss`, `noisy_homodyne`, `custom_kraus_instrument`, `finite_outcome_povm` and `mixed_conditional_update` to those of `piquasso-fock`. `state_vector` is unavailable; use `density_matrix`, moments, Wigner functions or reductions. |
| Tests | `tests/mixed_fock/test_evolution.py`, `tests/mixed_fock/test_independent_unitaries.py::test_unitary_against_independent_exponential` (independent matrix exponential), `tests/mixed_fock/test_cross_backend.py::test_overlapping_gaussian_regime_across_backends` (Gaussian backend), `tests/mixed_fock/test_state_physicality.py`, `tests/mixed_fock/test_validation.py` (`test_mixed_convergence_and_nonlinear_pure_limit`, against the pure-state limit); `tests/validation/test_val_fock.py` runs this backend as a secondary check only (tolerance 1e-10; CZ deviates by 1.5e-5, its documented Euler-factor truncation) |
| Status | **Implemented but experimental** (kept: the Fock paths are labelled experimental in `src/photographiq/backends/fock.py` and `docs/validation/feature-matrix.md`, and the 2026-11-12 independent suite treats the Piquasso engines as a secondary check; tests use an independent matrix exponential, the Gaussian backend and the pure-state limit) |

```python
import photographiq as pg
pattern = pg.Pattern(inputs=(0,)).append(pg.Loss(0, transmissivity=0.6))
two = {0: pg.FockInput.number(2)}
for name in ("piquasso-mixed-fock", "native-mixed-fock"):
    state = pg.simulate(pattern, inputs=two, backend=name, cutoff=5).state
    print(name, [round(float(p), 12) for k, p in sorted(state.probabilities.items()) if p > 1e-15])
# piquasso-mixed-fock [0.16, 0.48, 0.36]
# native-mixed-fock [0.16, 0.48, 0.36]
```

The binomial reference is P(k) = C(2,k) 0.6^k 0.4^(2-k) = 0.16, 0.48, 0.36.

### Native density-matrix evolution (`native-mixed-fock`)

| Field | Value |
|---|---|
| Description | `MixedFockBackend` semantics with a C++ density-matrix engine: gates as rho -> U rho U^dag on the touched modes, partial traces, photon-number marginals, thermal loss and physicality reports (`result.state.native.physicality()`: Hermiticity residual, trace, extreme eigenvalues; an internal attribute, not a documented API). Adds `cross_kerr`. |
| Python API | `photographiq.simulate(pattern, backend="native-mixed-fock", cutoff=c, ...)`; `photographiq.backends.native_fock.NativeMixedFockBackend(cutoff=None, *, cubic_phase="truncated_generator", cubic_tolerance=1e-12, **kwargs)` with `kwargs` `max_matrix_bytes` and the pure-backend settings |
| Implementation | `src/photographiq/backends/native_fock.py` (`NativeMixedFockBackend`, `NativeMixedFockState`); `cpp/src/fock/kernels.cpp` (`apply_mixed`, `phase_mixed`, `reduced_mixed`, `marginal_mixed`, `thermal_loss`, `physicality`) |
| Backends | `native-mixed-fock` (no Piquasso needed) |
| CPU / GPU | CPU (C++/OpenMP for gates and loss; Python/NumPy/SciPy for conditioning and sampling) |
| Assumptions | As for `piquasso-mixed-fock`. Gaussian gates use exact matrix elements (section 13), so at a fixed cutoff results differ from Piquasso's Euler decomposition at truncation level. A mixed Gaussian input is built as rho = U (sum_k p_k P_k) U^dag (P_k the Fock projector on k) from the Williamson form, with thermal weights summed until the omitted trace-norm tail is below 1e-15. If the number of thermal terms or the cutoff exceeds 4096, `NotImplementedError` is raised. |
| Tests | `tests/native/test_native_mixed_fock_backend.py` (`test_pure_inputs_give_the_pure_backend_state`, `test_noisy_channels_and_mixed_inputs_match_piquasso_mixed` (Piquasso mixed), `test_ladder_and_counting_match_piquasso_mixed`, `test_loss_matches_exact_gaussian_moments` (exact Gaussian moments), `test_thermal_gaussian_input_is_geometric_and_physical` (geometric distribution), `test_states_stay_physical`, `test_multi_kraus_instrument_matches_explicit_partial_trace` (explicit partial trace)); `tests/native/test_native_fock_kernels.py` (`test_mixed_kernel_equals_conjugation_and_reduction_is_partial_trace`, `test_parallel_density_kernel_matches_serial_pure_kernel`); `tests/validation/test_val_fock.py` on `native-mixed-fock` (states against mpmath, every gate against enlarged-space `expm`, loss and thermal loss against a beam-splitter dilation with a thermal environment (7.5e-16), counting and conditional states, partial trace with reordering, inefficient and noisy homodyne densities, Hermiticity, trace and positivity after a 21-gate sequence). Upgraded to validated on 2026-11-12 because of this independent suite. |
| Status | **Implemented and validated** (feature matrix: E) |

Example: see the binomial-loss example above (both mixed backends).

### Mixed and thermal inputs

| Field | Value |
|---|---|
| Description | Density-matrix inputs in a sparse local basis and single-mode mixed Gaussian inputs (thermal or squeezed thermal), tensored into the global truncated basis. |
| Python API | `pg.FockDensityMatrix(matrix, basis)`, `pg.GaussianInput(mean=(0.0, 0.0), covariance=...)` with a covariance above the vacuum (thermal or squeezed thermal), passed through `inputs=` or `pg.Prepare(node, state=...)` |
| Implementation | `src/photographiq/backends/mixed_fock.py` (`validate_preparation`, `prepare_resource`, `_gaussian_density`), `src/photographiq/backends/native_fock.py` (`_gaussian_density`) |
| Backends | `piquasso-mixed-fock`, `native-mixed-fock`; Gaussian backends for mixed Gaussian inputs. Pure Fock backends reject mixed Gaussian inputs (`NotImplementedError: Mixed Gaussian inputs require a mixed Fock backend`). |
| CPU / GPU | CPU |
| Assumptions | A `FockDensityMatrix` basis must match the resource's mode count and lie below the total cutoff, otherwise `ValueError`. A multi-mode `GaussianInput` is rejected (`ValueError: GaussianInput requires one mode`). The thermal tail beyond the cutoff is truncated and renormalised. |
| Tests | `tests/mixed_fock/test_evolution.py` (`test_binomial_loss_and_mixed_input`, `test_gaussian_thermal_and_noisy_vacuum`), `tests/mixed_fock/test_state_physicality.py::test_density_preparations_and_permuted_basis`, `tests/mixed_fock/test_loss_channels.py::test_thermal_tail_diagnostics_and_cutoff_convergence`, `tests/native/test_native_mixed_fock_backend.py::test_thermal_gaussian_input_is_geometric_and_physical`; `tests/validation/test_val_fock.py::test_displaced_squeezed_and_mixed_gaussian_inputs` (thermal displaced squeezed `GaussianInput` on `native-mixed-fock` against `expm`, 2.2e-16) |
| Status | **Implemented but experimental** (kept: the independent reference covers mixed Gaussian inputs on `native-mixed-fock` only, not `FockDensityMatrix` inputs or `piquasso-mixed-fock`; the feature matrix labels Fock paths experimental) |

Example: see `tests/mixed_fock/test_evolution.py::test_binomial_loss_and_mixed_input`.

### Loss and thermal loss channel

| Field | Value |
|---|---|
| Description | Attenuator with intensity transmissivity eta and environment occupation n_env. It is built from explicit Kraus operators as a vacuum attenuator eta/G followed by a quantum-limited amplifier with G = 1 + (1 - eta) n_env, which keeps inter-mode correlations. |
| Python API | `pg.Loss(node, transmissivity, thermal_photons=0.0)` |
| Implementation | `src/photographiq/backends/mixed_fock.py` (`loss`), `cpp/src/fock/kernels.cpp` (`thermal_loss`, used by `native-mixed-fock`) |
| Backends | `piquasso-mixed-fock`, `native-mixed-fock`; Gaussian backends (covariance form) |
| CPU / GPU | CPU (NumPy for `piquasso-mixed-fock`, C++ for `native-mixed-fock`) |
| Assumptions | Requires 0 <= transmissivity <= 1 and thermal_photons >= 0, else `ValueError`. The amplifier step can push weight above the cutoff. It is truncated, renormalised and reported as a retained-norm warning. |
| Tests | `tests/mixed_fock/test_loss_channels.py` (`test_binomial_channel_all_limits` (binomial closed form, n = 1..4), `test_coherent_attenuation` (coherent closed form), `test_thermal_loss_against_environment_dilation` (beam-splitter dilation with a thermal environment), `test_thermal_tail_diagnostics_and_cutoff_convergence`); `tests/native/test_native_fock_kernels.py::test_thermal_loss_closed_forms`; `tests/native/test_native_mixed_fock_backend.py::test_loss_matches_exact_gaussian_moments`; `tests/validation/test_val_fock.py::test_loss_matches_beam_splitter_dilation` (`native-mixed-fock`, n_env = 0 and 0.15, against a beam-splitter dilation with a thermal environment, 7.5e-16; retained trace 2.9e-15). Upgraded to validated on 2026-11-12: both mixed backends now have an independent dilation reference. |
| Status | **Implemented and validated** (feature matrix: E; the Gaussian covariance form is validated, Section 8) |

```python
import photographiq as pg
two = {0: pg.FockInput.number(2)}
p2 = (pg.Pattern(inputs=(0,)).append(pg.Loss(0, transmissivity=0.8, thermal_photons=0.1))
      .append(pg.Measure(0, pg.PhotonNumber(), key="n")))
r = pg.simulate(p2, inputs=two, backend="native-mixed-fock", cutoff=10, seed=3)
print(r.outcomes, r.measurement_statistics["n"])     # one sampled outcome and its probability
# {'n': 1} {'kind': 'probability', 'value': 0.3260878723339176, 'postselected': False}
```

### Conditional photon counting and homodyne on density matrices

| Field | Value |
|---|---|
| Description | Destructive photon counting and homodyne on one mode of a density matrix, sampled or postselected. Inefficient homodyne is applied as loss on the detected mode. Electronic noise is applied as a classical Gaussian convolution of the conditional operator, not as a random displacement of the surviving state. |
| Python API | `pg.Measure(node, pg.PhotonNumber())`, `pg.Measure(node, pg.Homodyne(angle, efficiency=eta, noise=s))`, `simulate(..., measurement_outcomes={key: value})`; `pg.PhysicalGKPReadout` and `MeasurementInstrument` are dispatched by the same `measure` |
| Implementation | `src/photographiq/backends/mixed_fock.py` (`measure`), `src/photographiq/fock_measurements.py` (`wavefunctions`) |
| Backends | `piquasso-mixed-fock`, `native-mixed-fock` |
| CPU / GPU | CPU (NumPy/SciPy) |
| Assumptions | Homodyne outcomes are calibrated to the incident quadrature, as in `GaussianBackend`. Homodyne sampling inverts the numerically integrated CDF (`scipy.integrate.quad` plus `brentq`). It raises `ArithmeticError` if the density is not normalised to 1e-7 or the noisy convolution does not converge. Photon counts must be integers below the cutoff. A postselected outcome with zero numerical probability raises `ValueError`. Postselected results are one conditional branch, with their probability or density in `measurement_statistics`. |
| Tests | `tests/mixed_fock/test_conditioning.py` (`test_diagonal_count_conditioning`, `test_correlated_gaussian_homodyne_schur_complement` (Gaussian Schur complement, with efficiency and noise)), `tests/mixed_fock/test_validation.py::test_mixed_homodyne_sampling_is_reproducible_and_gaussian`, `tests/mixed_fock/test_evolution.py::test_entangled_pnr_and_homodyne`; `tests/validation/test_val_fock.py::test_photon_counting_probability_and_conditional_state` and `::test_fock_homodyne_density_convention` (eta = 0.6, noise 0.2: closed-form density on both mixed backends). Upgraded to validated on 2026-11-12 because of these independent references. |
| Status | **Implemented and validated** (feature matrix: E) |

Example: see `tests/mixed_fock/test_conditioning.py::test_correlated_gaussian_homodyne_schur_complement`.

### Photon addition and subtraction on density matrices

| Field | Value |
|---|---|
| Description | Ideal heralded a^dag rho a or a rho a^dag, renormalised. The squared ladder norm is recorded in `diagnostics`. |
| Python API | `pg.PhotonAdd(node)`, `pg.PhotonSubtract(node)` |
| Implementation | `src/photographiq/backends/mixed_fock.py` (`ladder`) |
| Backends | `piquasso-mixed-fock`, `native-mixed-fock` (pure Fock backends have their own `ladder`) |
| CPU / GPU | CPU (NumPy) |
| Assumptions | A zero-norm ladder operation raises `ValueError`. Photon addition that would exceed the cutoff raises `ValueError("Photon addition exceeds cutoff; increase cutoff")`. These are heralded, non-deterministic operations whose success weight is reported. |
| Tests | `tests/mixed_fock/test_validation.py::test_mixed_ladder_weighting`, `tests/native/test_native_mixed_fock_backend.py::test_ladder_and_counting_match_piquasso_mixed` (Piquasso mixed) |
| Status | **Implemented but experimental** (the Fock paths are labelled experimental in `src/photographiq/backends/fock.py` and `docs/validation/feature-matrix.md`; matches Section 9) |

Example: see `tests/mixed_fock/test_validation.py::test_mixed_ladder_weighting`.

### Measurement instruments and Kraus maps on density matrices

| Field | Value |
|---|---|
| Description | General destructive instruments with several Kraus operators per outcome (`custom_kraus_instrument`, `finite_outcome_povm`). Only mixed backends can represent their non-pure conditional states. |
| Python API | `pg.MeasurementInstrument(kraus)` with `kraus = {outcome: (K1, K2, ...)}`, used in `Pattern.measure(node, instrument, key=None)` |
| Implementation | `src/photographiq/instruments.py` (`destructive_instrument(..., mixed=True)`), `src/photographiq/backends/mixed_fock.py` |
| Backends | `piquasso-mixed-fock`, `native-mixed-fock`; pure Fock backends accept rank-one instruments only (preflight rejects multi-Kraus outcomes) |
| CPU / GPU | CPU (NumPy) |
| Assumptions | The Kraus update sum_k K rho K^dag is used, not the effect update. Outcomes are single-mode and destructive. |
| Tests | `tests/gkp/instruments/test_instruments.py` (`test_kraus_is_not_effect_update` (explicit Kraus algebra), `test_complex_rank_one_entangled_conditioning`, `test_multikraus_partial_measurement_preserves_mixedness_and_rejects_pure_preflight` (on `piquasso-mixed-fock`), `test_invalid_instruments`); `tests/native/test_native_mixed_fock_backend.py::test_multi_kraus_instrument_matches_explicit_partial_trace` (on `native-mixed-fock`: a random 2-outcome, 4-Kraus instrument on mode A of a lossy, displaced, squeezed two-mode state; outcome probabilities and the conditional density matrix of mode B against an explicit index-sum partial trace of the unmeasured state, both to `1e-12`). Both mixed backends now have an independent multi-Kraus test; before this test, `native-mixed-fock` multi-Kraus instruments had been checked only by hand. Upgraded to validated on 2026-11-12 because explicit Kraus sums and an explicit partial trace are independent references (matches Section 10). |
| Status | **Implemented and validated** (feature matrix: E) |

Example: see `tests/gkp/instruments/test_instruments.py::test_complex_rank_one_entangled_conditioning`.

### Heterodyne and generaldyne on mixed-Fock backends

| Field | Value |
|---|---|
| Description | Phase-space (heterodyne/generaldyne) measurements on a truncated density matrix. |
| Python API | `pg.Measure(node, pg.Heterodyne())` / `pg.Generaldyne(...)` with a mixed-Fock backend |
| Implementation | `src/photographiq/backends/mixed_fock.py` (`measure`) |
| Backends | None of the Fock backends; available on `gaussian` and `native-gaussian` |
| CPU / GPU | Not applicable |
| Assumptions | Raises `NotImplementedError("Mixed Fock supports PhotonNumber and Homodyne")`, and the capability check rejects such patterns before execution. |
| Tests | `tests/cross_backend/test_registry_planner.py::test_planner_feature_check_agrees_with_preflight` |
| Status | **Unsupported** |

Example: not applicable.

### Density-matrix memory guard

| Field | Value |
|---|---|
| Description | Refuses to allocate a density matrix larger than a configured byte budget. |
| Python API | `MixedFockBackend(cutoff, max_matrix_bytes=256_000_000)`, `NativeMixedFockBackend(cutoff, max_matrix_bytes=...)`; pass the instance as `backend=` |
| Implementation | `src/photographiq/backends/mixed_fock.py` (`_guard`) |
| Backends | `piquasso-mixed-fock`, `native-mixed-fock` |
| CPU / GPU | CPU |
| Assumptions | The bound is 16 D^2 bytes for one complex128 matrix. Gate workspaces need additional memory that the guard does not count. Exceeding it raises `MemoryError` before allocation. `max_matrix_bytes` must be an integer >= 16. |
| Tests | `tests/mixed_fock/test_evolution.py::test_guards_and_serialization`, `tests/mixed_fock/test_validation.py::test_mixed_resource_preflight_and_impossible_branches` |
| Status | **Implemented but experimental** (behavioural guard tests only; no independent reference applies) |

Example: see `tests/mixed_fock/test_evolution.py::test_guards_and_serialization`.

## 15. GPU execution

Covers `photographiq.gpu` (CuPy): what runs on the GPU and what does not.

`photographiq.gpu` provides **batched Gaussian kernels** (many independent states, given as means `(B, 2n)` and covariances `(B, 2n, 2n)`) and, since 2026-11-12, **batched Fock local-gate kernels** (`photographiq.gpu.fock`: pure states `(B, D)` or density matrices `(B, D, D)` in the native total-photon basis), processed at once on an NVIDIA GPU through CuPy. The kernels are compositions of cuBLAS batched GEMM, cuSOLVER batched factorisations, cuSPARSE sparse-dense products and CuPy element-wise operations; there are no custom compiled CUDA kernels. `simulate`, `run_shots`, every registered backend, all measurement conditioning and all sampling run on the CPU. Importing `photographiq.gpu` does not import CuPy. Every kernel takes `device="gpu"` (default) or `device="cpu"`. With `device="gpu"` it raises if CuPy or a CUDA device is missing and never falls back to the CPU silently. Full details are in [GPU acceleration](gpu.md).

### Device discovery and execution record

| Field | Value |
|---|---|
| Description | Reports whether CuPy and a CUDA device are usable, describes the device, and attaches an `Execution` record to every kernel result naming the device that produced it. |
| Python API | `photographiq.gpu.available()`, `photographiq.gpu.device_info()`, `photographiq.gpu.require()`, `photographiq.gpu.Execution(kernel, device, backend, device_name, dtype, returned_on_device)`, `photographiq.gpu.INSTALL_HINT` |
| Implementation | `src/photographiq/gpu/device.py`, `src/photographiq/_optional.py` |
| Backends | Not a backend; applies to all `photographiq.gpu` kernels |
| CPU / GPU | GPU via photographiq.gpu (CuPy); the CPU path reports `device="cpu"` |
| Assumptions | `Execution.device` is `"cuda:<id>"`, read from the output buffer's own device (`array.device.id`), not assumed. `require()` raises `ImportError` (CuPy missing) or `RuntimeError` (no usable device). The `ImportError` reads "photographiq.gpu requires CuPy; " followed by `INSTALL_HINT`: "install the CuPy wheel matching your NVIDIA driver: pip install cupy-cuda12x (CUDA 12) or pip install 'cupy-cuda13x[ctk]' (CUDA 13); the extras gpu-cuda12 / gpu-cuda13 install the same wheels". Before the fix the CUDA 13 hint named plain `cupy-cuda13x` and did not mention the extras. There is no `"auto"` device, and other values raise `ValueError("device must be 'gpu' or 'cpu'")`. `on_device=True` returns CuPy arrays (work may still be queued on the stream); with `device="cpu"` it raises `ValueError`. |
| Tests | `tests/gpu/test_gpu_device.py` (`test_import_does_not_import_cupy`, `test_missing_cupy_raises_with_install_hint`, `test_no_device_raises_instead_of_falling_back`, `test_device_argument_is_explicit`, `test_device_info_describes_the_current_device`, `test_gpu_results_report_and_live_on_the_gpu` (buffer memory type checked with `cupy.cuda.runtime.pointerGetAttributes`), `test_device_resident_inputs_stay_on_device`) |
| Status | **Implemented and validated** |

```python
from photographiq import gpu
print(gpu.available())
# True
info = gpu.device_info()
print(info["device_name"], info["compute_capability"], info["cupy_version"])
# NVIDIA GeForce RTX 4050 Laptop GPU 8.9 14.2.0
```

### Batched symplectic update

| Field | Value |
|---|---|
| Description | d -> S d and V -> S V S^T on listed modes for every state, with one S for all states or one per state. This is a local O(B n k) update with an exactly symmetric output. |
| Python API | `photographiq.gpu.apply_symplectic(means, covariances, modes, matrices, *, device="gpu", dtype="float64", on_device=False)` returning `GaussianBatch(means, covariances, execution)` |
| Implementation | `src/photographiq/gpu/gaussian.py` |
| Backends | Standalone arrays (reference: `GaussianBackend.transform`) |
| CPU / GPU | GPU via photographiq.gpu (CuPy); `device="cpu"` runs the same array code in NumPy |
| Assumptions | hbar=2, interleaved xpxp, vacuum covariance I, integer mode indices (`operator.index`, so a float index raises `TypeError`). Symplecticity is checked in float64 before any cast (`ValueError`). Input covariances are assumed symmetric and physical; there is no O(n^3) physicality check. Inputs may be NumPy or CuPy arrays. |
| Tests | `tests/gpu/test_gpu_gaussian.py` (`test_apply_symplectic_matches_reference_backend` (NumPy `GaussianBackend`, batch sizes 1/7/33, single-mode, multimode, permuted and all-mode gates), `test_gpu_and_cpu_paths_agree` (rtol 1e-12), `test_matches_native_batch_kernels` (native C++ kernels), `test_invalid_input_raises`) |
| Status | **Implemented and validated** |

Example: see `tests/gpu/test_gpu_gaussian.py::test_apply_symplectic_matches_reference_backend`.

### Batched homodyne moments, log-density and conditioning

| Field | Value |
|---|---|
| Description | Mean and variance of the calibrated homodyne record, outcome log-likelihoods, and Schur-complement conditioning of every state on its **given** outcome (mode removed), with the outcome densities. |
| Python API | `photographiq.gpu.homodyne_moments(means, covariances, mode, angles, efficiency=1.0, noise=0.0, *, device, dtype, on_device)` returning `HomodyneMoments`; `photographiq.gpu.homodyne_log_density(means, covariances, mode, angles, outcomes, efficiency=1.0, noise=0.0, ...)` returning `BatchValues`; `photographiq.gpu.condition_homodyne(means, covariances, mode, angles, outcomes, efficiency=1.0, noise=0.0, ...)` returning `HomodyneConditioning(means, covariances, log_densities, densities, execution)` |
| Implementation | `src/photographiq/gpu/gaussian.py` |
| Backends | Standalone arrays (reference: `GaussianBackend.measure(..., outcome=x)`, `scipy.stats.norm.logpdf`) |
| CPU / GPU | GPU via photographiq.gpu (CuPy); `device="cpu"` uses NumPy |
| Assumptions | Record q cos t + p sin t with variance Var + (1 - eta)/eta + noise; 0 < efficiency <= 1. Outcomes must be finite. A batch conditioned on chosen outcomes is a set of **postselected branches**, not sampled trajectories, and there is no GPU sampling. Unlike `GaussianBackend.measure`, an outcome whose density underflows to 0 does not raise: `densities` is 0 and `log_densities` stays finite. |
| Tests | `tests/gpu/test_gpu_gaussian.py` (`test_condition_homodyne_matches_reference_backend` (NumPy reference, efficiencies 1/0.9/0.5, noise 0/0.2), `test_conditioned_states_are_physical` (V + i Omega >= 0), `test_gpu_and_cpu_paths_agree`, `test_matches_native_batch_kernels`) |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import gpu
B, n = 4, 2
means = np.zeros((B, 2 * n))
covs = np.broadcast_to(np.diag([np.exp(-1.0), np.exp(1.0), 1.0, 1.0]), (B, 2 * n, 2 * n)).copy()
x = np.linspace(-1, 1, B)
res = gpu.condition_homodyne(means, covs, mode=0, angles=0.0, outcomes=x, efficiency=0.9)
print(res.covariances.shape, type(res.means).__name__, np.round(res.densities, 6))
# (4, 2, 2) ndarray [0.202957 0.513305 0.513305 0.202957]
print(res.execution)
# Execution(kernel='condition_homodyne', device='cuda:0', backend='cupy', device_name='NVIDIA GeForce RTX 4050 Laptop GPU', dtype='float64', returned_on_device=False)
cpu = gpu.condition_homodyne(means, covs, mode=0, angles=0.0, outcomes=x, efficiency=0.9, device="cpu")
print(cpu.execution.device, np.allclose(cpu.densities, res.densities, rtol=1e-12))
# cpu True
```

### Batched purity and symplectic eigenvalues

| Field | Value |
|---|---|
| Description | Purity det(V)^(-1/2) by batched `slogdet`, and symplectic eigenvalues by Cholesky followed by a Hermitian eigensolver, for every covariance in the batch. |
| Python API | `photographiq.gpu.purity(covariances, *, device="gpu", dtype="float64", on_device=False)`, `photographiq.gpu.symplectic_eigenvalues(covariances, ...)`, both returning `BatchValues(values, execution)` |
| Implementation | `src/photographiq/gpu/gaussian.py` |
| Backends | Standalone arrays (reference: `GaussianState.purity`, `GaussianState.symplectic_eigenvalues`) |
| CPU / GPU | GPU via photographiq.gpu (CuPy); `device="cpu"` uses NumPy |
| Assumptions | Vacuum covariance I. `symplectic_eigenvalues` raises `ValueError` for non-positive-definite input. Non-finite diagnostics raise `ValueError`. |
| Tests | `tests/gpu/test_gpu_gaussian.py::test_diagnostics_match_gaussian_state` (NumPy `GaussianState`) |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import gpu
covs = np.broadcast_to(np.diag([np.exp(-1.0), np.exp(1.0), 1.0, 1.0]), (4, 4, 4)).copy()
p = gpu.purity(covs)
print(p.values, p.execution.device)
# [1. 1. 1. 1.] cuda:0
```

### Batched fidelity

| Field | Value |
|---|---|
| Description | Squared Uhlmann fidelity between paired states: the overlap formula when one state of the pair is pure, otherwise the general mixed-state formula (BBP, Banchi-Braunstein-Pirandola). |
| Python API | `photographiq.gpu.fidelity(means1, covariances1, means2, covariances2, pure_tolerance=1e-9, *, device="gpu", dtype="float64", on_device=False)` returning `BatchValues` |
| Implementation | `src/photographiq/gpu/gaussian.py` |
| Backends | Standalone arrays (reference: `GaussianState.fidelity`) |
| CPU / GPU | GPU via photographiq.gpu (CuPy), but **not fully batched on the GPU**: CuPy's `linalg.eigvals` loops in Python over the B matrices, one cuSOLVER call each |
| Assumptions | The general eigensolver in the mixed-state formula is ill-conditioned for nearly pure states. The test tolerance is rtol 1e-8 for fidelity, against rtol 1e-11 for the other kernels. |
| Tests | `tests/gpu/test_gpu_gaussian.py::test_diagnostics_match_gaussian_state` (NumPy `GaussianState.fidelity`), `test_gpu_and_cpu_paths_agree` |
| Status | **Implemented and validated** |

Example: see `tests/gpu/test_gpu_gaussian.py::test_diagnostics_match_gaussian_state`.

### float32 precision option

| Field | Value |
|---|---|
| Description | Optional single-precision execution of every GPU-module kernel, recorded in `execution.dtype`. |
| Python API | `dtype="float32"` on any `photographiq.gpu` kernel (default `"float64"`) |
| Implementation | `src/photographiq/gpu/gaussian.py` |
| Backends | Standalone arrays |
| CPU / GPU | GPU via photographiq.gpu (CuPy) or CPU (NumPy) |
| Assumptions | Used only when requested; other dtypes raise `ValueError("dtype must be float64 or float32")`. Symplecticity checks still run in float64 before the cast (atol 1e-5 when the matrices are given as float32, else 1e-8). The test bounds the float32 deviation from float64 at rtol 1e-4 (conditioning) and 1e-5 (symplectic update); [GPU acceleration](gpu.md) reports a typical relative error of about 5e-8 at moderate squeezing, growing with squeezing (not asserted by a test). GeForce-class GPUs run FP64 at a small fraction of their FP32 rate, which limits float64 speedups. |
| Tests | `tests/gpu/test_gpu_gaussian.py::test_float32_is_explicit_and_close` (within 1e-4 / 1e-5 of float64) |
| Status | **Implemented but experimental** (validated only against float64 on moderate test states) |

Example: see `tests/gpu/test_gpu_gaussian.py::test_float32_is_explicit_and_close`.

### Batched Fock local gates (`photographiq.gpu.fock`)

| Field | Value |
|---|---|
| Description | Applies one local gate to a batch of Fock states in the native engine's total-photon basis (sum n < cutoff, D = C(modes + cutoff - 1, modes)). `apply_operator` maps psi -> P U P psi for pure batches `(B, D)` and rho -> (P U P) rho (P U P)^dag for density-matrix batches `(B, D, D)`. The k-mode operator acts on every fibre of fixed occupations of the other modes as its top-left block, which is one sparse D x D matrix G applied by one cuSPARSE sparse-dense product (two for density matrices). `apply_phase` applies diagonal gates (Kerr, rotation, cross-Kerr) element-wise. |
| Python API | `photographiq.gpu.apply_operator(states, modes, cutoff, local, op, *, device="gpu", dtype="complex128", on_device=False)`, `photographiq.gpu.apply_phase(states, modes, cutoff, local, phases, *, device="gpu", dtype="complex128", on_device=False)`, both returning `photographiq.gpu.FockBatch(states, execution)` |
| Implementation | `src/photographiq/gpu/fock.py` (basis and ranks from the native engine's `basis` and `rank`, cached CSR structure per `(modes, cutoff, local)`; only the values are gathered from `op` per call) |
| Backends | Standalone arrays, not a `simulate` backend (reference: the native kernels `apply_pure`, `apply_mixed`, `phase_pure`, `phase_mixed`). Needs the native `fock` component for the basis. |
| CPU / GPU | GPU via `photographiq.gpu` (CuPy/cuSPARSE); `device="cpu"` runs the same algorithm with `scipy.sparse` |
| Assumptions | Same contract as the Gaussian kernels: `device="gpu"` raises without CuPy or a device and never falls back to the CPU; `execution` names the device that produced the result; complex64 only when requested. No renormalisation and no diagnostics; the input is not modified. Gate matrices come from the native engine (`gaussian_unitary`, `cubic_phase_*`). A batch whose shape does not match `(modes, cutoff)`, an operator that is not `(S, S)` with S = C(k + cutoff - 1, k), or repeated, out-of-range or non-integer local modes raise `ValueError`. Measured on an RTX 4050 Laptop GPU (`benchmarks/gpu/bench_gpu_fock.py`, shared machine): batched pure states (B >= 32) and one large pure state (D = 42 504) gain over the native loop, and density-matrix batches gain at kernel level; **single small states are slower on the GPU**, phase gates gain nothing end to end over NumPy, and host-device transfers dominate end-to-end time for large batches. These kernels do not accelerate `simulate`: one trajectory at a time is the wrong shape for a GPU. See [GPU acceleration](gpu.md) for the figures. |
| Tests | `tests/gpu/test_gpu_fock.py` (29 tests: `test_gates_match_native_engine` (displacement, squeezing, beam splitter, CZ, Kerr, cross-Kerr, both cubic-phase definitions, against the native kernels), `test_arbitrary_operators_and_non_hermitian_matrices_match_native` (index maps), `test_gpu_matches_cpu_path`, `test_norm_and_trace`, `test_gate_sequence_matches_native_backend` (whole `NativeFockBackend` / `NativeMixedFockBackend` gate sequences), `test_complex64_is_explicit_and_bounded`, `test_basis_and_argument_errors`, `test_missing_cupy_raises_instead_of_falling_back`, `test_gpu_results_report_and_live_on_the_gpu`). Largest deviation from the native kernels 7.2e-16 (GPU, complex128) and 2.4e-7 (complex64); norms and traces agree to 3.3e-16. The reference is the native C++ engine, itself validated against `expm` (Section 13). |
| Status | **Implemented and validated** (complex128; complex64 is checked only against complex128 within 1e-5) |

```python
import numpy as np
from photographiq import gpu, native
F = native.require("fock")
modes, cutoff, B = 3, 8, 64
D = F.dimension(modes, cutoff)
rng = np.random.default_rng(0)
psi = rng.normal(size=(B, D)) + 1j * rng.normal(size=(B, D))
psi /= np.linalg.norm(psi, axis=1, keepdims=True)
U = np.asfortranarray(F.gaussian_unitary(np.diag([np.exp(-0.2), np.exp(0.2)]), [0.3, 0.1], cutoff))
res = gpu.apply_operator(psi, modes, cutoff, [1], U)              # device="gpu"
ref = psi.copy()
for row in ref:                                                    # native kernel, state by state
    col = row.reshape(-1, 1).copy(); F.apply_pure(col, modes, cutoff, [1], U); row[:] = col[:, 0]
print(D, res.states.shape, f"{np.abs(res.states - ref).max():.1e}")
print(res.execution.device, res.execution.dtype)
# 120 (64, 120) 6.2e-17
# cuda:0 complex128
```

### Pattern simulation and sampling on the GPU

| Field | Value |
|---|---|
| Description | GPU execution of single trajectories (`simulate`, `run_shots`), sampling of measurement outcomes, and GPU permanents, hafnians and torontonians for boson sampling and GBS. |
| Python API | None. `simulate(..., backend="gpu")` raises `ValueError: Unknown backend: 'gpu'`. No GPU backend is registered; CUDA-Q is an adapter module, not a backend (section 16). |
| Implementation | Not implemented |
| Backends | None |
| CPU / GPU | CPU only (all registered backends and `photographiq.sampling`) |
| Assumptions | Reasons from [GPU acceleration](gpu.md). Single small trajectories are dominated by launch and synchronisation latency; use `native-gaussian`. CuPy random streams differ from NumPy's, so sampled trajectories would not be reproducible across devices. GPU sampling kernels were measured on 2026-11-12 and judged not justified, so nothing ships: batched CuPy permanents gain only for large batches of small matrices (a gain that mostly reflects native per-call overhead), single permanents at n = 20 and 24 and a single torontonian at n = 16 are slower than the native kernels, and no current workload evaluates large batches of same-size matrices. A native batched kernel should come first. The JAX realization `photographiq.autodiff` has no device option and runs wherever JAX places arrays; in the documented environment JAX reports only `CpuDevice`, and no GPU run of it has been tested. |
| Tests | Not applicable |
| Status | **Unsupported** |

Example: not applicable. Benchmarks of the batched kernels (NumPy vs native C++/OpenMP vs GPU, transfers timed separately) are `benchmarks/gpu/bench_gpu_gaussian.py` and `benchmarks/gpu/bench_gpu_fock.py`, with methodology in `benchmarks/gpu/README.md` and `benchmarks/README.md` (Section 32). No timing numbers are quoted.

## 16. CUDA-Q support

Implemented in `src/photographiq/cudaq_dynamics.py` as an optional Hamiltonian-level adapter. Full description: [CUDA-Q adapter](cudaq.md).

CUDA-Q publishes Linux (x86_64/aarch64) wheels only, so it cannot be installed on native Windows (use WSL2). It is not installed in the environment used for this document, so the CUDA-Q execution path is **unverified**. CUDA-Q is not a registered backend: it cannot execute patterns (no measurement conditioning), so the former `cudaq-dynamics` registry entry was removed.

### CUDA-Q Dynamics adapter

| Field | Value |
|---|---|
| Description | `BosonicHamiltonian` (normal-ordered sums of ladder-operator products, Hermiticity check); generators H with U = exp(-i H) for rotation, displacement, squeezing, beam splitter, CZ, quadratic phase, Kerr, cross-Kerr and cubic phase; `to_matrix`; per-mode/total-photon mapping; `reference_evolve` (unitary, or Lindblad with collapse operators); `to_cudaq` (builds `cudaq.operators.boson` expressions); `evolve` (calls `cudaq.evolve` on the `dynamics` target, no fallback). |
| Python API | `photographiq.cudaq_dynamics.evolve(hamiltonian, dimensions, times, initial_state, *, collapse_operators=(), observables=(), store_intermediate_results=False, integrator=None)` and `reference_evolve` with the same interface (no `integrator`), returning `EvolutionResult(state, times, intermediate_states, expectation_values, record)` |
| Implementation | `src/photographiq/cudaq_dynamics.py`; `src/photographiq/backends/cudaq_dynamics.py` re-exports `cudaq_probe`, `cudaq_capabilities` |
| Backends | Not a backend |
| CPU / GPU | `reference_evolve`: CPU (SciPy `expm_multiply`). `evolve`: whatever CUDA-Q reports in `record` (never run here) |
| Assumptions | hbar = 2, q = a + a^dag, p = -i (a - a^dag). Each generator matches the `native-fock` gate up to a global phase (the native Gaussian gates fix the phase by a real positive vacuum amplitude). Lindblad loss with collapse operator sqrt(gamma) a matches the native loss channel with T = exp(-gamma t). Without `cudaq`, `to_cudaq` and `evolve` raise `ImportError` naming CUDA-Q's Linux-only wheels; cudaq older than 0.16 is refused. |
| Tests | `tests/cudaq_adapter/test_cudaq_adapter.py` (the directory is not called `tests/cudaq`, which would shadow the real package; 29 collected, 28 pass here: every generator against native-fock on enlarged per-mode dimensions to 1.4e-13 with the global phase aligned and sign-flipped generators rejected, Lindblad loss against native-mixed-fock to 6e-17, basis mapping, Hermiticity, missing-cudaq and old-version errors, translation wiring with a fake `cudaq`; `test_real_cudaq_evolve_matches_reference` skips without `cudaq`); `tests/cross_backend/test_registry_planner.py::test_cudaq_is_an_adapter_not_a_backend` |
| Status | **Implemented but experimental** (conventions validated through the NumPy/SciPy reference path; no CUDA-Q call has run, so CUDA-Q execution is unverified) |

```python
import numpy as np
import photographiq as pg
import photographiq.cudaq_dynamics as cq
dims = {0: 8}
psi = np.zeros(8, complex)
psi[1] = 1
out = cq.reference_evolve(cq.rotation(0, 0.3), dims, [0.0, 1.0], psi)
print(np.round(out.state[1], 6), out.record["engine"])
# (0.955336+0.29552j) numpy-reference
try:
    cq.evolve(cq.rotation(0, 0.3), dims, [0.0, 1.0], psi)
except ImportError:
    print("ImportError")
# ImportError
try:
    pg.simulate(pg.Pattern(inputs=(0,)), backend="cudaq-dynamics")
except ValueError as exc:
    print("ValueError:", exc)
# ValueError: Unknown backend: 'cudaq-dynamics'. CUDA-Q cannot execute patterns; it is available as the Hamiltonian-level adapter module photographiq.cudaq_dynamics
```

### CUDA-Q availability probe

| Field | Value |
|---|---|
| Description | `cudaq_probe()` tries `import cudaq` and reports availability, version and the import error (never raises). `cudaq_capabilities()` summarises the documented CUDA-Q Dynamics features the adapter relies on, with `verified: False`. |
| Python API | `photographiq.cudaq_dynamics.cudaq_probe()`, `cudaq_capabilities()` (also importable from `photographiq.backends.cudaq_dynamics`) |
| Implementation | `src/photographiq/cudaq_dynamics.py` |
| Backends | Not a backend |
| CPU / GPU | CPU only |
| Assumptions | Any import exception is reported as `"<Type>: <message>"`. |
| Tests | `tests/cross_backend/test_registry_planner.py::test_cudaq_reported_without_claiming_execution`, `::test_cudaq_is_an_adapter_not_a_backend` |
| Status | **Implemented but experimental** |

```python
from photographiq.cudaq_dynamics import cudaq_probe
print(cudaq_probe())
# {'available': False, 'version': None, 'error': "ModuleNotFoundError: No module named 'cudaq'"}
```

## 17. Sampling algorithms

`photographiq.sampling` draws exact samples from boson-sampling and Gaussian-boson-sampling output distributions. It is a standalone module: it does not go through `photographiq.simulate` and is unrelated to `photographiq.sample(pattern, shots)`, which runs shots of an MBQC pattern. Every sampler here is exact for the stated model; the GBS samplers enumerate the full (or truncated) distribution first, so their cost grows with the number of patterns. No chain-rule or approximate large-scale GBS sampler is implemented. User guide: [boson sampling and GBS](user-guide/sampling.md).

### Clifford & Clifford boson sampler

| Field | Value |
|---|---|
| Description | Exact Fock-state boson sampling by Clifford & Clifford (2018), algorithm B, for inputs with at most one photon per mode. Each step draws the next output mode from weights built from `(k-1)`-minor permanents. |
| Python API | `photographiq.sampling.sample(unitary, inputs, shots, *, rng=None, method="auto", backend="auto")` with `method="clifford"`; returns an `int64` array of shape `(shots, modes)`. `method="auto"` (the default) picks this sampler for collision-free inputs. |
| Implementation | `src/photographiq/sampling/boson_sampling.py` (`sample`, `_clifford_shot`); minor permanents via `src/photographiq/sampling/linalg.py` and `cpp/src/sampling/permanent.cpp`. |
| Backends | Standalone module (not a `simulate` backend). `backend=` selects the permanent kernel: `"native"` (C++), `"python"` (NumPy) or `"auto"`. |
| CPU / GPU | CPU only. Per-shot loop in Python; permanents in C++/OpenMP (native) or NumPy. |
| Assumptions | Interferometer convention `a_j^dag -> sum_k U_kj a_k^dag`; `U` must be unitary to `atol=1e-10` (`check_unitary`). Each `(k-1)`-minor permanent is evaluated separately rather than by the paper's single Laplace pass, so the permanent work is `O(n sum_k k 2^k)` for `n` photons. Raises `ValueError("Clifford & Clifford sampling here requires at most one photon per mode")` for a bunched input with `method="clifford"`. |
| Tests | `tests/sampling/test_sampling_boson.py::test_exact_samplers_reproduce_distribution` (cases `(1, 1, 1, 0, 0)` with `"clifford"` and `(1, 0, 1, 0, 1)` with `"auto"`, which also selects this sampler; chi-square test, 8000 shots, p > 1e-3, against the exact enumerated distribution `output_distribution`, which is itself checked against a creation-operator expansion and Piquasso); `::test_chi_square_check_rejects_distinguishable_particles` (negative control: distinguishable-particle samples are rejected with p < 1e-6); `::test_clifford_rejects_collision_inputs`. |
| Status | **Implemented and validated** |

```python
from photographiq import sampling as S
U = S.haar_unitary(4, rng=1)
print(S.sample(U, [1, 1, 1, 0], 3, rng=7).tolist())
# [[1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 1, 0]]
```

### Enumeration boson sampler

| Field | Value |
|---|---|
| Description | Exact sampling from the fully enumerated output distribution over all `C(n+m-1, n)` patterns. Works for any input occupation, including inputs with several photons in one mode. |
| Python API | `photographiq.sampling.sample(unitary, inputs, shots, *, rng=None, method="auto", backend="auto")` with `method="enumerate"`; `method="auto"` picks it for inputs with collisions. |
| Implementation | `src/photographiq/sampling/boson_sampling.py` (`sample`, `output_distribution`). |
| Backends | Standalone module; permanent kernel selected by `backend=` (`"auto"`, `"native"`, `"python"`). |
| CPU / GPU | CPU only (NumPy sampling over C++/OpenMP or NumPy permanents). |
| Assumptions | Intended for small systems: one permanent per pattern. `output_distribution` raises `ValueError` if the probabilities do not sum to 1 within `1e-9`. Unknown `method` raises `ValueError`. |
| Tests | `tests/sampling/test_sampling_boson.py::test_exact_samplers_reproduce_distribution`, case `(2, 1, 0, 0)` with `"enumerate"` (chi-square, 8000 shots, against `output_distribution`, which is checked against a creation-operator expansion and Piquasso). This is the only sampler test of this method; the `"auto"` case `(1, 0, 1, 0, 1)` is collision-free and runs Clifford & Clifford. |
| Status | **Implemented and validated** |

```python
from photographiq import sampling as S
U = S.haar_unitary(4, rng=1)
print(S.sample(U, [2, 1, 0, 0], 3, rng=7, method="enumerate").tolist())
# [[1, 0, 2, 0], [1, 1, 1, 0], [1, 0, 2, 0]]
```

### PNR GBS sampler

| Field | Value |
|---|---|
| Description | Exact photon-number-resolving samples of a Gaussian state, drawn from the enumerated PNR distribution of all patterns with total photon number below `cutoff`. The omitted mass is an explicit outcome, returned as a row of `-1`; it is never resampled or redistributed. |
| Python API | `photographiq.sampling.sample_pnr(state, shots, cutoff, *, rng=None, backend="auto")` returns `PNRSamples(samples, cutoff, truncated_mass)` with property `overflow` (number of `-1` rows). |
| Implementation | `src/photographiq/sampling/gbs.py` (`sample_pnr`, `PNRSamples`, `pnr_distribution`). |
| Backends | Standalone module; hafnian kernels selected by `backend=`. Input is a `photographiq.states.GaussianState`. |
| CPU / GPU | CPU only (NumPy or C++/OpenMP hafnians). |
| Assumptions | hbar = 2, interleaved `(q0, p0, q1, p1, ...)`, vacuum covariance `I`. Total-photon cutoff: patterns with `sum n < cutoff` are enumerated (`C(cutoff-1+m, m)` patterns). Before sampling, the in-cutoff probabilities and the truncated mass go through `_sampling_weights` (shared with `sample_threshold`): a negative entry no larger in size than `_CLIP_RTOL = 1e-12` times the total absolute mass is treated as rounding and clipped to 0, and the vector is renormalised; a larger negative raises `ValueError("Probability ... is negative beyond rounding (1e-12 x total mass ...): the state is not physical or the evaluation lost accuracy")`. The docstring states this. The 1e-12 threshold is justified in the code comment: the largest rounding error of a single probability seen in the tests is 2.9e-15 (16-photon PNR). Clipping changes the sampled distribution by at most 1e-12 per clipped outcome in total variation. Example: for the two-mode squeezed vacuum with `r = 0.8`, `transmissivity=1 - 1e-13` and `cutoff=8`, the mixed formula gives `P(2, 5) = -2.27e-18` where the exact value is 0. That entry is clipped, never drawn, and sampling succeeds (before the fix this raised NumPy's `ValueError("Probabilities are not non-negative")`). |
| Tests | `tests/sampling/test_sampling_gbs.py::test_pnr_sampler_reports_truncated_mass` (binomial test of the overflow count against `truncated_mass`, chi-square of the in-cutoff counts against `pnr_distribution`; the PNR probabilities are validated against closed forms, an independent Fock-space construction and Piquasso, see section 19); `::test_pnr_sampler_clips_rounding_level_negatives` (the nearly pure TMSV above: 2000 shots, no negative pattern drawn); `::test_samplers_raise_for_negatives_beyond_rounding` (`-1e-17` clipped, `-1e-9` against a total mass of 1 raises). |
| Status | **Implemented and validated** |

```python
from photographiq import sampling as S
state = S.squeezed_interferometer_state([0.9, 0.7], S.haar_unitary(2, rng=2), [0.9, 0.9])
res = S.sample_pnr(state, 1000, cutoff=6, rng=4)
print(res.samples.shape, res.overflow, round(res.truncated_mass, 6))
# (1000, 2) 75 0.067295
```

### Threshold GBS sampler

| Field | Value |
|---|---|
| Description | Exact threshold-detector (click / no-click) samples from the probabilities of all `2^m` click patterns. |
| Python API | `photographiq.sampling.sample_threshold(state, shots, *, rng=None, backend="auto")` returns an `int64` array of shape `(shots, modes)` with entries 0/1. |
| Implementation | `src/photographiq/sampling/gbs.py` (`sample_threshold`, `threshold_distribution`). |
| Backends | Standalone module; threshold sum by native or NumPy kernel (`backend=`). |
| CPU / GPU | CPU only. |
| Assumptions | Enumerates all `2^m` patterns, so only small `m` is practical. Negative probabilities no larger in size than `1e-12` times the total absolute mass are clipped to 0 and the vector is renormalised before sampling. A larger negative raises `ValueError("Probability ... is negative beyond rounding ...")`; the old code clipped every negative without a check. The docstring states this, and the same helper `_sampling_weights` is used by `sample_pnr`. The worst negatives observed by the sampling fix's reviewer grow with the mode count (-5.9e-15 at 12 modes, -2.4e-14 at 14 modes, squeezing 0.1), so the check should be reachable only at about 18-20 modes, where full enumeration is already too slow. Patterns whose value is negative or consistent with zero emit a `ThresholdPrecisionWarning` (section 20) while the distribution is built. |
| Tests | `tests/sampling/test_sampling_gbs.py::test_threshold_sampler_matches_distribution` (chi-square, 20000 shots, against `threshold_distribution`, whose entries are validated against Piquasso and a Fock-space reference); `::test_samplers_raise_for_negatives_beyond_rounding` (a monkeypatched distribution with a `-1e-6` entry raises). |
| Status | **Implemented and validated** |

```python
from photographiq import sampling as S
state = S.squeezed_interferometer_state([0.9, 0.7], S.haar_unitary(2, rng=2), [0.9, 0.9])
print(S.sample_threshold(state, 4, rng=0).tolist())
# [[0, 1], [0, 0], [0, 0], [0, 0]]
```

### Truncation reporting

| Field | Value |
|---|---|
| Description | Truncated PNR distributions report the captured probability and the omitted mass (`1 - captured_mass`, the probability of all patterns with total photon number `>= cutoff`) instead of renormalising. |
| Python API | `photographiq.sampling.TruncatedDistribution(probabilities, cutoff, captured_mass)` with derived field `truncated_mass`; returned by `pnr_distribution(state, cutoff, *, backend="auto")`. `PNRSamples.truncated_mass` and `PNRSamples.overflow` carry it into samples. |
| Implementation | `src/photographiq/sampling/gbs.py` (`TruncatedDistribution`, `pnr_distribution`, `PNRSamples`). |
| Backends | Standalone module. |
| CPU / GPU | CPU only. |
| Assumptions | Total-photon cutoff `sum n < cutoff`. `truncated_mass` can be negative at the `1e-15` level from rounding (`sample_pnr` clips it to 0 under the same `1e-12` rule as the probabilities). `cutoff` must be a positive integer (`ValueError`). |
| Tests | `tests/sampling/test_sampling_gbs.py::test_pnr_normalization_and_truncated_mass` (pattern count `C(cutoff-1+3, 3)`; `0 <= truncated_mass <= <n>/cutoff`, the Markov bound from the covariance-derived mean photon number; strictly decreasing in cutoff; truncated mass `< 1e-5` at cutoff 10; partial mean photon number matches `GaussianState.photon_number` to `1e-4`). |
| Status | **Implemented and validated** |

```python
from photographiq import sampling as S
state = S.squeezed_interferometer_state([0.4, 0.3, 0.2], S.haar_unitary(3, rng=1))
d = S.pnr_distribution(state, cutoff=4)
print(len(d.probabilities), d.cutoff, round(d.captured_mass, 12), round(d.truncated_mass, 12))
# 20 4 0.98380143068 0.01619856932
```

## 18. Boson Sampling

Fock-state boson sampling through a passive linear interferometer: transition amplitudes are permanents of submatrices of `U` with repeated rows and columns, evaluated directly from the multiplicities. The module also provides the interferometer tools used to set up such experiments (Haar-random unitaries, Clements decomposition, passive symplectics).

### Haar-random unitaries

| Field | Value |
|---|---|
| Description | Haar-distributed `m x m` unitary from the QR decomposition of a complex Ginibre matrix, with the phases of `diag(R)` moved into `Q` (Mezzadri 2007). |
| Python API | `photographiq.sampling.haar_unitary(modes, rng=None)` |
| Implementation | `src/photographiq/sampling/unitaries.py` |
| Backends | Standalone module. |
| CPU / GPU | CPU (NumPy). |
| Assumptions | `rng` is a seed, a `numpy.random.Generator` or `None`. `modes` must be a positive integer (`ValueError`). |
| Tests | `tests/sampling/test_sampling_unitaries.py::test_haar_unitary_is_unitary_and_reproducible`, `::test_haar_statistics` (Kolmogorov-Smirnov test of `abs(U_00)^2` against the analytical Beta(1, m-1) law, zero mean and the analytical fourth moment `2/(m(m+1))`). |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
U = S.haar_unitary(3, rng=5)
print(np.allclose(U.conj().T @ U, np.eye(3)))
# True
```

### Interferometer convention and passive symplectics

| Field | Value |
|---|---|
| Description | Unitary check, Mach-Zehnder element and conversion of an interferometer unitary to its interleaved-quadrature symplectic matrix, in the convention shared with every PhotoGraphiQ backend. |
| Python API | `photographiq.sampling.check_unitary(unitary, atol=1e-10)`, `photographiq.sampling.mzi(modes, u, v, theta, phi)`, `photographiq.sampling.passive_symplectic(unitary)` |
| Implementation | `src/photographiq/sampling/unitaries.py` |
| Backends | Standalone module. |
| CPU / GPU | CPU (NumPy). |
| Assumptions | `a_k -> sum_j U_kj a_j` (Heisenberg picture); `Rotate(phi)` is `U = exp(i phi)`; `BeamSplitter(theta)` is `[[cos, -sin], [sin, cos]]`; `a = (q + i p)/2` (hbar = 2); 2x2 symplectic blocks `[[Re U_jk, -Im U_jk], [Im U_jk, Re U_jk]]`. `check_unitary` raises `ValueError("Matrix is not unitary")` or `ValueError("Expected a square matrix")`. |
| Tests | `tests/sampling/test_sampling_unitaries.py::test_passive_symplectic_matches_gate_conventions` (against the gate symplectics `photographiq.gaussian.rotation` / `beamsplitter`; symplectic and orthogonal), `::test_mzi_is_beamsplitter_after_rotation`, `::test_clements_rejects_non_unitary`. |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
s = S.passive_symplectic(S.haar_unitary(3, rng=5))
print(s.shape, np.allclose(s @ s.T, np.eye(6)))
# (6, 6) True
```

### Clements decomposition

| Field | Value |
|---|---|
| Description | Rectangular-mesh decomposition of a unitary into `m(m-1)/2` Mach-Zehnder elements and output phases (Clements et al., Optica 3, 1460 (2016)), exportable as a PhotoGraphiQ gate list. |
| Python API | `photographiq.sampling.clements_decomposition(unitary, atol=1e-10)` returns `ClementsDecomposition(modes, elements, output_phases)` with methods `unitary()` and `operations()` (`("rotate", mode, angle)` / `("beamsplitter", u, v, theta)` tuples). |
| Implementation | `src/photographiq/sampling/unitaries.py` |
| Backends | Standalone module. The gate list uses the `Rotate` / `BeamSplitter` conventions of the Gaussian backend. |
| CPU / GPU | CPU (NumPy). |
| Assumptions | Elements are in application (time) order. Reconstruction error is checked; `RuntimeError` if it exceeds `max(atol, 1e3 m eps)`. |
| Tests | `tests/sampling/test_sampling_unitaries.py::test_clements_reconstructs_unitary_and_photographiq_gates` (m = 1, 2, 3, 4, 7, 10: reconstruction to `1e-13`, and the product of `photographiq.gaussian` gate symplectics equals `passive_symplectic(U)` to `1e-13`). |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
U = S.haar_unitary(3, rng=5)
d = S.clements_decomposition(U)
print(len(d.elements), np.abs(d.unitary() - U).max() < 1e-13)
# 3 True
```

### Transition amplitudes and output probabilities

| Field | Value |
|---|---|
| Description | Single amplitude `<t| W(U) |s> = Perm(U_{t,s}) / sqrt(prod s_i! prod t_j!)` and probability `P(t|s) = abs(Perm(U_{t,s}))^2 / (prod s_i! prod t_j!)`, where `U_{t,s}` repeats row `k` `t_k` times and column `j` `s_j` times. |
| Python API | `photographiq.sampling.transition_amplitude(unitary, inputs, outputs, *, backend="auto")`, `photographiq.sampling.output_probability(unitary, inputs, outputs, *, backend="auto")` |
| Implementation | `src/photographiq/sampling/boson_sampling.py`; permanent from `src/photographiq/sampling/linalg.py` / `cpp/src/sampling/permanent.cpp`. |
| Backends | Standalone module; `backend=` selects native or NumPy permanents. |
| CPU / GPU | CPU (C++/OpenMP or NumPy). |
| Assumptions | Photon number is conserved: the amplitude is `0` if `sum t != sum s`. The repeated matrix is never built (multiplicity-aware Ryser formula). Occupations must be `m` nonnegative integers (`ValueError`). |
| Tests | `tests/sampling/test_sampling_boson.py::test_amplitudes_match_creation_operator_expansion` (independent monomial expansion of `prod_j (sum_k U_kj a_k^dag)^{s_j}`, tolerance `1e-12`), `::test_matches_piquasso_fock_simulation` (Piquasso `PureFockSimulator`, `1e-12`), `::test_distribution_normalization_and_hong_ou_mandel` (HOM: `P(1,1) < 1e-30`, `P(2,0) = 0.5`). |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
bs = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
print(S.transition_amplitude(bs, (1, 1), (2, 0)))
print(S.output_probability(S.haar_unitary(4, rng=1), [1, 1, 0, 0], [0, 1, 1, 0]))
# (-0.7071067811865474+0j)
# 0.06908404014531996
```

### Output distributions and collision statistics

| Field | Value |
|---|---|
| Description | Exact distribution over all `C(n+m-1, n)` output patterns, pattern enumeration, and the split of probability mass into collision-free and bunched outcomes. |
| Python API | `photographiq.sampling.output_distribution(unitary, inputs, *, backend="auto")` (dict pattern -> probability), `photographiq.sampling.output_patterns(modes, photons)`, `photographiq.sampling.is_collision_free(pattern)`, `photographiq.sampling.collision_probabilities(distribution)` |
| Implementation | `src/photographiq/sampling/boson_sampling.py` |
| Backends | Standalone module. |
| CPU / GPU | CPU only. |
| Assumptions | Patterns are returned in lexicographic order. `output_distribution` raises `ValueError` if the total differs from 1 by more than `1e-9`. |
| Tests | `tests/sampling/test_sampling_boson.py::test_distribution_normalization_and_hong_ou_mandel` (pattern count, normalisation to `1e-13`, collision split sums to 1), `::test_output_patterns_enumeration`, `::test_matches_piquasso_fock_simulation` (Piquasso). |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
bs = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
print(S.output_distribution(bs, (1, 1)))
d = S.output_distribution(S.haar_unitary(4, rng=1), [1, 1, 1, 0])
print(len(d), S.collision_probabilities(d))
# {(0, 2): 0.4999999999999998, (1, 1): 0.0, (2, 0): 0.4999999999999998}
# 20 {'collision_free': 0.47610419972468104, 'collision': 0.5238958002753178}
```

## 19. Gaussian Boson Sampling

Photon-number-resolving (PNR) and threshold-detector probabilities of PhotoGraphiQ Gaussian states, including mixed (lossy) and displaced states. PNR probabilities are (loop) hafnians; threshold probabilities are evaluated by an inclusion-exclusion sum over vacuum projections, conditioned on the dark modes (a real form of the loop torontonian). Conventions: hbar = 2, interleaved quadratures, vacuum covariance `I`, `a_j = (q_j + i p_j)/2`.

### GBS input states

| Field | Value |
|---|---|
| Description | Builds the standard GBS input: single-mode squeezers on vacuum, an interferometer, per-mode pure loss, then an optional coherent displacement. |
| Python API | `photographiq.sampling.squeezed_interferometer_state(squeezing, unitary, transmissivity=1.0, displacement=None)` returns a `photographiq.states.GaussianState` with nodes `0..m-1`. |
| Implementation | `src/photographiq/sampling/gbs.py`; squeezing symplectic from `src/photographiq/gaussian.py`. |
| Backends | Standalone module; the resulting `GaussianState` is the same class the Gaussian backend produces. |
| CPU / GPU | CPU (NumPy). |
| Assumptions | `Squeeze(r) = diag(e^{-r}, e^{r})` (`r > 0` squeezes q); `a -> U a`; loss applied after the interferometer and before the displacement; displacement `alpha_i` adds `(2 Re alpha_i, 2 Im alpha_i)` to the mean. One squeezing and one displacement per mode (`ValueError` otherwise). |
| Tests | `tests/sampling/test_sampling_gbs.py::test_matches_piquasso_gaussian_simulation` (covariance and mean equal Piquasso `GaussianSimulator` with `hbar=2` to `1e-14`), `::test_two_mode_squeezed_vacuum_closed_form`. |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
st = S.squeezed_interferometer_state([1.0], np.eye(1))
print(S.pnr_probability(st, [2]))
# 0.18794405337586959
```

### GBS matrices

| Field | Value |
|---|---|
| Description | Complex-amplitude representation of a Gaussian state: `sigma_Q`, the GBS matrix `A = X (I - sigma_Q^{-1})^*`, `beta`, loop vector `gamma = sigma_Q^{-1} beta` and vacuum probability `p_0`, in block `(a, a^dag)` order. Precomputing it avoids rebuilding it for every pattern. |
| Python API | `photographiq.sampling.gbs_matrices(state)` returns `GBSMatrices(sigma_q, a, beta, gamma, vacuum_probability, mean, covariance)` with properties `modes`, `displaced`, `pure`. |
| Implementation | `src/photographiq/sampling/gbs.py` |
| Backends | Standalone module. `pnr_probability` and `threshold_probability` accept a `GBSMatrices` in place of the state. |
| CPU / GPU | CPU (NumPy). |
| Assumptions | `A` is the complex conjugate of thewalrus' `A` (same probabilities; do not mix `gamma` from one convention with `A` from the other). `pure` is true only if the off-diagonal block of `A` is below `100 eps max(1, max abs(A))`, so nearly pure states (for example loss `1 - 1e-11`) take the exact mixed formula. |
| Tests | `tests/sampling/test_sampling_gbs.py::test_gbs_matrices_structure` (`p_0` against `exp(-beta^dag sigma_Q^{-1} beta / 2) / sqrt(det sigma_Q)` to `1e-13`, symmetry of `A`, Hermiticity of `sigma_Q`, purity flags), `::test_auto_method_uses_the_exact_mixed_formula_for_nearly_pure_states`. |
| Status | **Implemented and validated** |

```python
from photographiq import sampling as S
st = S.squeezed_interferometer_state([0.4, 0.3, 0.2], S.haar_unitary(3, rng=1),
                                     transmissivity=0.9, displacement=[0.1, 0, 0])
g = S.gbs_matrices(st)
print(g.modes, g.pure, g.displaced, round(g.vacuum_probability, 12))
# 3 False True 0.85905730507
```

### PNR probabilities

| Field | Value |
|---|---|
| Description | Probability of a photon-number pattern `n`: `P(n) = p_0 lhaf(A_{n+n}; gamma_{n+n}) / prod n_i!` for mixed states (hafnian when the mean is zero); for pure states `P(n) = p_0 abs(lhaf(B_n; gamma_n))^2 / prod n_i!`. |
| Python API | `photographiq.sampling.pnr_probability(state, pattern, *, method="auto", backend="auto")` with `method` in `"auto"`, `"pure"`, `"mixed"`; `state` is a `GaussianState` or `GBSMatrices`. |
| Implementation | `src/photographiq/sampling/gbs.py`; hafnians in `src/photographiq/sampling/linalg.py` and `cpp/src/sampling/hafnian.cpp`. |
| Backends | Standalone module; `backend=` selects native or NumPy hafnians. |
| CPU / GPU | CPU (C++/OpenMP or NumPy). |
| Assumptions | Pattern ordered as `state.nodes`. Repeated photons are handled by duplicating indices: cost `2^{N/2}` (pure) or `2^N` (mixed) power-trace terms for `N` photons. `method="pure"` on a mixed state raises `ValueError("State is not pure; use method='mixed'")`. |
| Tests | `tests/sampling/test_sampling_gbs.py::test_single_mode_squeezed_vacuum_closed_form` (closed form, `abs=1e-14`, `rel=1e-11`, n < 12, both methods), `::test_two_mode_squeezed_vacuum_closed_form`, `::test_coherent_state_is_poissonian` (Poisson, `rel=1e-12`), `::test_pnr_matches_independent_fock_construction` (dense Fock-space evolution at per-mode cutoff 36, displaced and lossy, `abs=1e-13`), `::test_matches_piquasso_gaussian_simulation` (Piquasso `get_particle_detection_probability`, `abs=1e-13`), `::test_matches_piquasso_pure_fock_simulation` (Piquasso `PureFockSimulator`). All run for `backend="python"` and `"native"`. |
| Status | **Implemented and validated** |

```python
from photographiq import sampling as S
pure = S.squeezed_interferometer_state([0.4, 0.3, 0.2], S.haar_unitary(3, rng=1))
print(S.pnr_probability(pure, [1, 1, 0], method="pure"),
      S.pnr_probability(pure, [1, 1, 0], method="mixed"))
# 0.02604195986063082 0.026041959860630817
```

### Threshold probabilities

| Field | Value |
|---|---|
| Description | Probability of a click pattern (0 = no click, 1 = at least one photon). With clicked set `S` and dark set `Z`: `P(S) = P_Z ([S empty] + sum_{T subset S} (-1)^{abs(T)} w_T)`, where each vacuum-projection term enters through its deviation `w_T = expm1(...)` from 1. This is equal to `p_0 ltor(O_S, gamma'_S)`. |
| Python API | `photographiq.sampling.threshold_probability(state, clicks, *, backend="auto", precision=None, rtol=1e-6, return_error=False, return_info=False)`; returns a float, `(value, relative_error)` with `return_error=True`, or `(value, ThresholdInfo)` with `return_info=True` (section 20). `precision` is `None` (double), an `int` number of digits (mpmath) or `"auto"` (double, recomputed in mpmath when needed; section 20). |
| Implementation | `src/photographiq/sampling/gbs.py` (`_dark_conditioned`, `_vacuum_sum`, `_sum_bound`, `_threshold_double`, `_vacuum_sum_mp`); native sum `threshold_vacuum_sum` in `cpp/src/sampling/torontonian.cpp`. |
| Backends | Standalone module; `backend=` selects native or NumPy summation and is validated on every path, including `precision=` (`ValueError("Unknown backend ...")`; `backend="native"` raises `ImportError` without the native extension even with `precision=`); `precision=` uses mpmath (section 20). |
| CPU / GPU | CPU (C++/OpenMP or NumPy; mpmath is pure Python). |
| Assumptions | Cost `2^{clicks}` small determinants or eigenvalue problems. The dark-mode conditioning reuses one Cholesky factor of `M_ZZ` (`scipy.linalg.cho_solve`). The alternating sum loses digits as the number of clicks grows and `P` shrinks; see the rounding bound in section 20. Clicks must be 0/1 (`ValueError("Threshold patterns contain only 0 and 1")`). Both summation paths raise `ValueError("Conditioned covariance block is not positive definite")` when a block `I + E_T` is not positive definite (native: C++ `PhysicsError`, a `std::invalid_argument`; NumPy: an eigenvalue `<= -1`; before the fix the NumPy path returned `nan`). The native kernel also raises `ValueError("Too many clicked modes")` above 40 clicked modes; the NumPy path has no mode limit. A non-positive-definite block cannot arise from a valid `GaussianState`, whose constructor rejects covariances that violate `V + i Omega >= 0`. |
| Tests | `tests/sampling/test_sampling_gbs.py::test_two_mode_squeezed_vacuum_closed_form` (`P(0,0) = 1/cosh^2 r` to `rel=1e-12`), `::test_coherent_state_is_poissonian`, `::test_pnr_matches_independent_fock_construction` (sums of Fock-space PNR probabilities over the click support, `abs=1e-12`), `::test_matches_piquasso_gaussian_simulation` (Piquasso `get_threshold_detection_probability`, `abs=1e-13`), `::test_threshold_normalization_and_vacuum_inclusion_exclusion` (inclusion-exclusion over `GaussianState.overlap` with vacuum, `abs=1e-13`), `::test_python_vacuum_sum_raises_like_native_for_non_positive_blocks` (same message on both backends), `::test_threshold_precision_validates_backend`. Accuracy matrix: `benchmarks/gaussian_boson_sampling/threshold_reference_matrix.py` against an independent 40-digit mpmath implementation, 382 cases (section 20 and [GBS threshold accuracy](validation/gbs-threshold-accuracy.md)). |
| Status | **Implemented and validated** |

```python
import math, numpy as np
from photographiq import sampling as S
bs = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
tmsv = S.squeezed_interferometer_state([0.8, -0.8], bs)
print(S.threshold_probability(tmsv, [0, 0]), 1 / math.cosh(0.8) ** 2)
# 0.5590551677322436 0.5590551677322438
```

### Threshold distribution

| Field | Value |
|---|---|
| Description | Probabilities of all `2^m` click patterns. |
| Python API | `photographiq.sampling.threshold_distribution(state, *, backend="auto")` (dict pattern -> probability) |
| Implementation | `src/photographiq/sampling/gbs.py` |
| Backends | Standalone module. |
| CPU / GPU | CPU only. |
| Assumptions | Small `m` only. Rounding can produce entries of order `1e-16` where the exact value is 0, and slightly negative values; each such entry emits a `ThresholdPrecisionWarning` ("consistent with zero" or "negative beyond", section 20). |
| Tests | `tests/sampling/test_sampling_gbs.py::test_threshold_normalization_and_vacuum_inclusion_exclusion` (sum to 1 within `1e-13`; each entry against vacuum-overlap inclusion-exclusion). |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
bs = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
print(S.threshold_distribution(S.squeezed_interferometer_state([0.8, -0.8], bs)))
# {(0, 0): 0.5590551677322436, (0, 1): 1.8620277577560473e-16, (1, 0): 1.8620277577560473e-16, (1, 1): 0.4409448322677561}
# (stderr, twice: ThresholdPrecisionWarning: Threshold probability 1.862e-16 is consistent with zero
#  within its absolute rounding bound 6.7e-16 (alternating inclusion-exclusion sum over 1 clicked modes); ...)
```

### Photon loss

| Field | Value |
|---|---|
| Description | Per-mode pure-loss channel on a Gaussian state: `V -> eta V + (1 - eta) I` on the 2x2 blocks and `d -> sqrt(eta) d`. |
| Python API | `photographiq.sampling.apply_loss(state, transmissivity)` (scalar or one value per mode); also the `transmissivity=` argument of `squeezed_interferometer_state`. |
| Implementation | `src/photographiq/sampling/gbs.py` |
| Backends | Standalone module (operates on `GaussianState`). |
| CPU / GPU | CPU (NumPy). |
| Assumptions | `eta` is intensity transmission in `[0, 1]`; otherwise `ValueError("Transmissivities must lie in [0, 1]")`. Zero-temperature environment only (no thermal photons in this function). |
| Tests | `tests/sampling/test_sampling_gbs.py::test_loss_is_binomial_thinning` (against binomial thinning of lossless PNR distributions), `::test_pnr_matches_independent_fock_construction` (Fock reference with thinning), `::test_matches_piquasso_gaussian_simulation` (Piquasso `Attenuator`). |
| Status | **Implemented and validated** |

```python
from photographiq import sampling as S
tmsv = S.squeezed_interferometer_state([0.8, -0.8], [[2**-0.5, -2**-0.5], [2**-0.5, 2**-0.5]])
print(S.threshold_probability(S.apply_loss(tmsv, [0.9, 0.5]), [1, 0]))
# 0.14551309357464634
```

(The `precision=30` value is `0.1455130935746464`. The last digits changed when the dark-mode conditioning moved to `scipy.linalg.cho_solve`; the reported relative bound is 1.0e-14.)

## 20. Hafnian and Torontonian routines

Matrix functions behind boson sampling and GBS. Each public function takes `backend="auto"` (compiled `photographiq._native.sampling` kernels when built, NumPy otherwise), `"native"` (raises `ImportError` if the component is not compiled) or `"python"`; any other value raises `ValueError("Unknown backend ...")`. Both implementations use the same algorithms and compensated (Neumaier / `math.fsum`) summation. All routines run on the CPU; there is no GPU path.

### Permanent

| Field | Value |
|---|---|
| Description | Permanent of a square matrix, or of the matrix with row `i` repeated `rows[i]` times and column `j` repeated `columns[j]` times, by Ryser's inclusion-exclusion formula with multiplicities (`prod_j (c_j + 1)` terms instead of `2^N`; the cheaper side is enumerated). |
| Python API | `photographiq.sampling.permanent(matrix, rows=None, columns=None, *, backend="auto")` |
| Implementation | `src/photographiq/sampling/linalg.py` (`permanent`, `_ryser_repeated`); `cpp/src/sampling/permanent.cpp` (reflected mixed-radix Gray order, OpenMP over a fixed number of chunks). |
| Backends | `backend="native"`, `"python"`, `"auto"`. |
| CPU / GPU | CPU (C++/OpenMP) or CPU (NumPy). |
| Assumptions | With multiplicities the matrix may be rectangular. `ValueError` for a non-square matrix without multiplicities, unequal multiplicity sums, negative or non-integer multiplicities, or non-finite entries. The native kernel raises `ValueError("Permanent enumeration too large")` above `9e18` terms. |
| Tests | `tests/sampling/test_sampling_linalg.py::test_permanent_matches_definition` (n = 0..7, against the permutation sum, `rtol=1e-12`), `::test_permanent_with_multiplicities_matches_repeated_matrix`, `::test_permanent_identities` (`Perm(J_n) = n!`, diagonal), `::test_permanent_matches_piquasso` (Piquasso `piquasso._math.permanent`, n up to 12 and with multiplicities, `1e-12`), `::test_permanent_rejects_inconsistent_multiplicities`. |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
m = np.array([[1, 2], [3, 4]])
print(S.permanent(m), S.permanent(m, backend="python"), S.permanent_by_definition(m))
# (10+0j) (10+0j) (10+0j)
```

### Hafnian

| Field | Value |
|---|---|
| Description | Hafnian of a symmetric matrix (sum over perfect matchings; diagonal ignored) by the Glynn-type power-trace formula (Bjorklund, Gupt, Quesada 2019), with power traces from explicit matrix powers. Cost `O(n^4 2^n)` for a `2n x 2n` matrix. |
| Python API | `photographiq.sampling.hafnian(matrix, *, backend="auto")` |
| Implementation | `src/photographiq/sampling/linalg.py` (`hafnian`, `_power_trace`); `cpp/src/sampling/hafnian.cpp` (OpenMP over sign vectors). |
| Backends | `backend="native"`, `"python"`, `"auto"`. |
| CPU / GPU | CPU (C++/OpenMP) or CPU (NumPy). |
| Assumptions | Symmetry is checked to `1e-12 max(1, max abs)`; otherwise `ValueError("Hafnians are defined for symmetric matrices")`. Odd dimension returns `0`. Native limit: 40 pairs (`ValueError("Hafnian dimension too large")`). |
| Tests | `tests/sampling/test_sampling_linalg.py::test_hafnians_match_definition` (d = 0..10, against the perfect-matching sum, `rtol=1e-12`), `::test_hafnian_identities` (`haf([[0, B], [B^T, 0]]) = Perm(B)`, `haf(J_2n) = (2n-1)!!`), `::test_hafnians_match_piquasso` (Piquasso `hafnian_with_reduction`, d = 4, 8, 12). |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
J = np.ones((6, 6))
print(S.hafnian(J), S.hafnian_by_definition(J))
# (15+0j) (15+0j)
```

### Loop hafnian

| Field | Value |
|---|---|
| Description | Loop hafnian (sum over matchings with loops; diagonal entries are loop weights), by the power-trace formula with the loop correction of Bjorklund, Cygan, Koivisto, Kaski. Used for displaced GBS states. |
| Python API | `photographiq.sampling.loop_hafnian(matrix, diagonal=None, *, backend="auto")`; `diagonal` replaces the matrix diagonal as loop weights. |
| Implementation | `src/photographiq/sampling/linalg.py`; `cpp/src/sampling/hafnian.cpp`. |
| Backends | `backend="native"`, `"python"`, `"auto"`. |
| CPU / GPU | CPU (C++/OpenMP) or CPU (NumPy). |
| Assumptions | Odd dimension is padded with an isolated vertex of loop weight 1. Same symmetry check as `hafnian`; `diagonal` must be a finite vector of matching size (`ValueError`). |
| Tests | `tests/sampling/test_sampling_linalg.py::test_hafnians_match_definition` (matchings-with-loops sum), `::test_hafnian_identities` (telephone numbers for the all-ones matrix up to n = 10; zero loops give the hafnian), `::test_hafnians_match_piquasso` (Piquasso `loop_hafnian_with_reduction`). |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
J = np.ones((6, 6))
print(S.loop_hafnian(J), S.loop_hafnian_by_definition(J))
# (76+0j) (76+0j)
```

### Torontonian

| Field | Value |
|---|---|
| Description | `Tor(O) = sum_{Z subset [n]} (-1)^{n-abs(Z)} / sqrt(det(I - O_Z))` for a `2n x 2n` matrix. |
| Python API | `photographiq.sampling.torontonian(matrix, *, backend="auto")` |
| Implementation | `src/photographiq/sampling/linalg.py` (`_torontonian`); `cpp/src/sampling/torontonian.cpp` (LU determinants, OpenMP over subset chunks). |
| Backends | `backend="native"`, `"python"`, `"auto"`. |
| CPU / GPU | CPU (C++/OpenMP) or CPU (NumPy). |
| Assumptions | Block ordering `(a_1..a_n, a_1^dag..a_n^dag)`: `O_Z` keeps rows/columns `{i, i+n : i in Z}` (Piquasso pairs `(2i, 2i+1)` instead). Principal square root. `ValueError("Torontonians take a 2n x 2n matrix")` for odd dimension; native limit 40 modes. `threshold_probability` does not call this function; it uses the dark-mode-conditioned real form (section 19). |
| Tests | `tests/sampling/test_sampling_linalg.py::test_torontonian_product_identities` (direct-sum matrices with a closed-form product, n = 0..5, `1e-12`), `::test_torontonians_match_piquasso` (Piquasso `torontonian` after reordering, `1e-11`). |
| Status | **Implemented and validated** |

```python
import math, numpy as np
from photographiq import sampling as S
print(S.torontonian(np.diag([0.3, 0.3])), 1 / math.sqrt(0.49) - 1)
# (0.4285714285714286+0j) 0.4285714285714286
```

### Loop torontonian

| Field | Value |
|---|---|
| Description | `ltor(O, g) = sum_Z (-1)^{n-abs(Z)} exp(g_Z^T (I - O_Z)^{-1} g_Z^* / 2) / sqrt(det(I - O_Z))` (displaced threshold detection). |
| Python API | `photographiq.sampling.loop_torontonian(matrix, gamma, *, backend="auto")` |
| Implementation | `src/photographiq/sampling/linalg.py`; `cpp/src/sampling/torontonian.cpp`. |
| Backends | `backend="native"`, `"python"`, `"auto"`. |
| CPU / GPU | CPU (C++/OpenMP) or CPU (NumPy). |
| Assumptions | Block ordering as for `torontonian`; `gamma` must be a finite vector of length `2n` (`ValueError`). |
| Tests | `tests/sampling/test_sampling_linalg.py::test_torontonian_product_identities` (closed-form product; zero `gamma` reduces to `Tor`), `::test_torontonians_match_piquasso` (Piquasso `loop_torontonian`, `1e-11`). |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
g = np.array([0.2 + 0.1j, 0.2 - 0.1j])
print(S.loop_torontonian(np.diag([0.3, 0.3]), g, backend="python"),
      S.loop_torontonian(np.diag([0.3, 0.3]), g, backend="native"))
# (0.5343449010232797+0j) (0.5343449010232797+0j)
```

### By-definition reference implementations

| Field | Value |
|---|---|
| Description | Direct evaluation of the defining combinatorial sums, used as independent oracles for small sizes: permutations (`O(n! n)`, n <= 8), perfect matchings (`(2n-1)!!` terms, 2n <= 12) and matchings with loops. |
| Python API | `photographiq.sampling.permanent_by_definition(matrix)`, `photographiq.sampling.hafnian_by_definition(matrix)`, `photographiq.sampling.loop_hafnian_by_definition(matrix)` |
| Implementation | `src/photographiq/sampling/linalg.py` |
| Backends | Pure Python / NumPy only (no native version). |
| CPU / GPU | CPU only. |
| Assumptions | No symmetry check in the by-definition hafnians. Factorial cost; small matrices only. There is no by-definition torontonian; its references are closed-form products and Piquasso. |
| Tests | Used as the oracle in `tests/sampling/test_sampling_linalg.py::test_permanent_matches_definition` and `::test_hafnians_match_definition`; their agreement with the Ryser and power-trace kernels, which in turn match Piquasso (`::test_permanent_matches_piquasso`, `::test_hafnians_match_piquasso`), cross-checks them. |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
print(S.permanent_by_definition(np.ones((4, 4))), S.loop_hafnian_by_definition(np.ones((3, 3))))
# (24+0j) (4+0j)
```

### Repeated-index matrices

| Field | Value |
|---|---|
| Description | Returns the matrix with row and column `i` repeated `repetitions[i]` times (the explicit `A_n` of a photon pattern). The probability APIs do not need it: `permanent` takes multiplicities directly and `pnr_probability` indexes with `np.repeat`. |
| Python API | `photographiq.sampling.repeat_indices(matrix, repetitions)` |
| Implementation | `src/photographiq/sampling/linalg.py` |
| Backends | Pure NumPy. |
| CPU / GPU | CPU (NumPy). |
| Assumptions | `repetitions` must be nonnegative integers, one per row (`ValueError`). |
| Tests | None: no test in the suite calls `repeat_indices`, and no other library code uses it. Only the example below has been run. |
| Status | **Implemented but experimental** |

```python
import numpy as np
from photographiq import sampling as S
print(S.repeat_indices(np.arange(4).reshape(2, 2), [2, 1]).tolist())
# [[0, 0, 1], [0, 0, 1], [2, 2, 3]]
```

### Native kernels vs Python implementations

| Field | Value |
|---|---|
| Description | The compiled `photographiq._native.sampling` submodule exposes `permanent`, `permanent_repeated`, `hafnian`, `loop_hafnian`, `torontonian`, `loop_torontonian`, `threshold_vacuum_sum`, `max_threads` and `set_num_threads`. The GIL is released during evaluation; `std::invalid_argument` maps to `ValueError`. Work is split into a fixed number of chunks summed in fixed order, so results do not depend on the thread count. |
| Python API | `backend=` argument of every matrix function and probability API; `photographiq.native.require("sampling")` returns the kernel module; `photographiq.native.components()` lists compiled components. |
| Implementation | `src/photographiq/sampling/linalg.py` (`_kernels` dispatch), `src/photographiq/native.py`, `cpp/src/sampling/*.cpp`, `cpp/src/bindings/sampling_bindings.cpp`. C++ unit test `cpp/tests/test_sampling.cpp` (built only with `PGQ_BUILD_CPP_TESTS=ON`). |
| Backends | `"native"` (C++20, Eigen, OpenMP if available), `"python"` (NumPy), `"auto"`. |
| CPU / GPU | CPU (C++/OpenMP) vs CPU (NumPy). No GPU code; `photographiq.gpu` does not cover these routines. |
| Assumptions | `"auto"` falls back to NumPy silently when the `sampling` component is not compiled; `"native"` then raises `ImportError`. `set_num_threads` is process wide. Timing comparisons: run `benchmarks/boson_sampling/bench_boson_sampling.py` and `benchmarks/gaussian_boson_sampling/bench_gbs.py`, see `benchmarks/README.md`. |
| Tests | `tests/sampling/test_sampling_native.py::test_native_agrees_with_python_at_larger_sizes` (permanent n = 14, hafnian / loop hafnian / torontonian 16 x 16, `1e-10` relative), `::test_results_do_not_depend_on_thread_count` (bitwise equality for 1, 3 and max threads; covers `permanent`, `hafnian` and `torontonian` only), `::test_native_errors_map_to_value_error`; every test in `tests/sampling/test_sampling_linalg.py` and `tests/sampling/test_sampling_gbs.py` that takes the `backend` fixture runs for both; `tests/sampling/test_sampling_benchmark_smoke.py` (benchmark rows: relative errors `< 1e-8` in the smoke configuration). |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import native, sampling as S
print(native.components())
a = np.random.default_rng(0).normal(size=(8, 8)); a = a + a.T
print(abs(S.hafnian(a, backend="native") - S.hafnian(a, backend="python")) < 1e-10)
# ('gaussian', 'fock', 'sampling')
# True
```

### Threshold rounding-error bound

| Field | Value |
|---|---|
| Description | `threshold_probability(..., return_error=True)` returns `B / abs(P)`, where `B` is an absolute first-order rounding bound covering both the alternating inclusion-exclusion sum and the dark-mode conditioning: `B = P_Z (eps sum_T abs(w_T) + S) + P_Z C + abs(P) (eps_Z + u)` with unit roundoff `u = eps/2`. `C = (1 + sqrt3)/8 norm(V_c) Tr abs(dV_c) + norm(V_c)^{1/2} norm(de) / 2` propagates the Cholesky backward error of `M_ZZ` (Higham, Theorems 10.3-10.4) through a sensitivity bound on the conditional click probability. `eps_Z` bounds the relative error of `P_Z`. `S = u sum_T (1 + w_T) (sum_i abs(log1p mu_i(E_T)) / 2 + q_T)`, times `max(1, 16 / sqrt(2^k - 1))` for `k` clicks, models rounding inside each exponent. Without `S` the old estimate missed that error: at squeezing 0.05 with 6 clicks the error was 4.2e-16 against an estimate of 5.2e-17. The module docstring documents the formula. |
| Python API | `photographiq.sampling.threshold_probability(state, clicks, *, return_error=True)` returns `(value, error)` with `error = B / abs(value)`, or `inf` if `value == 0` |
| Implementation | `src/photographiq/sampling/gbs.py` (`_dark_conditioned`, `_sum_bound`, `_threshold_double`, constant `_SUM_SAFETY = 16.0`); `cpp/src/sampling/torontonian.cpp` (`threshold_vacuum_sum` returns the sum and `sum_T abs(w_T)`; the kernel is unchanged and the bound is computed in Python around it). |
| Backends | `backend="native"`, `"python"`; with `precision=`, the estimate is `P_Z mp.eps sum_T abs(1 + w_T)`. It does not strictly cover rounding in the conditioning step, which is negligible at 30 or more digits. |
| CPU / GPU | CPU only. |
| Assumptions | The conditioning part (`C`, `eps_Z`) is a derived first-order bound. Finite differences check the click-sensitivity inequality, which at the vacuum is tight to within 1.37x. The alternating-sum part `S` is first-order and empirical: it is calibrated against mpmath, not proven. The factor `max(1, 16 / sqrt(2^k - 1))` is a calibration for 1 to 8 clicks, where the per-term error of the eigensolver otherwise exceeded the model by up to 3.9x. It is 1.002 at k = 8 and 1 from k = 9. With this factor, the sum error stayed below 0.52 of its share in 4140 random states on both backends (k = 1-9, squeezing 5e-5 to 2, transmissivity 0.5-1, with and without displacement). A fresh check found the total error below 0.43 of `B` in 4000 further evaluations. The bound is relative to the covariance and mean as stored: rounding of the stored covariance is not included, the same contract as `precision=`. It is loose for strong squeezing, up to about 6e4 times the old estimate (`norm(V_c) Tr abs(dV_c)` term), so `B / abs(P) > 1e-6` can warn for small but accurate probabilities of strongly squeezed states. The bound is conservative. The orchestrator's measurements put it at 180x to 6e4x the actual error on realistic states. A check made for this section on 37 random states (4-10 modes, squeezing 0.2-1.5, transmissivity 0.5-1, 1-8 clicks, native backend, against `precision=30`) gave 20x to 1.1e4x, with a median of 400x. On the three 24/28/32-mode `threshold_mp_reference.py` cases (r = 0.5, eta = 0.9, 12/14/16 clicks) it is 480x to 1260x the actual error against the 50-digit references; there the relative bound is 1.4e-4, 2.0e-2 and 3.6. On the 16-mode / 8-click test case it is 7.7x the old estimate (1.98e-5 instead of 2.58e-6), against an actual relative error of 2.0e-6. |
| Tests | `tests/sampling/test_sampling_gbs.py::test_threshold_rounding_bound_holds_against_multiprecision` (12 cases x 2 backends against `precision=30`: small squeezing, displaced, large squeezing (1.2), lossless displaced, no dark modes, three zero-probability patterns, and few clicks (k = 1, 2, 3) at squeezing below 0.01; asserts `abs(value - ref) <= B` and `error == B / abs(value)`); `::test_threshold_flags_negative_and_zero_results`; `::test_conditional_click_sensitivity_bounds` (central differences on 12 random states, and the vacuum value `dF/dV_qq = 1/4` against the bound 0.34). `::test_threshold_error_estimate_bounds_multiprecision_reference` (16 modes, 8 clicks, r = 0.2, eta = 0.9, reference `P = 1.1329737713467e-09` from `precision=30`; asserts `1e-6 < error < 3e-5`, that the warning is issued, and `abs(value - ref) <= error * ref`). Its upper limit was raised from `1e-5` to `3e-5`, and the docstring gives the reason: the limit only caps how pessimistic the bound may be, and the bound now also covers dark-mode conditioning and eigensolver rounding, which the earlier, tighter estimate (2.6e-6) missed. No honest bound fits under 1e-5 on that case, because the old estimate plus the exact per-term rounding already fills that budget. All 63 tests in `tests/sampling/test_sampling_gbs.py` pass on both backends. **Reference matrix (2026-11-12):** `benchmarks/gaussian_boson_sampling/threshold_reference_matrix.py` compared both backends with an independent 40-digit mpmath implementation (separate code from `precision=`) on 382 cases: 6 to 24 modes, 1 to 16 clicks, r from 0.05 to 1.5, transmissivity 0.5 to 1, with and without displacement, plus 22 zero-probability patterns. The bound held in every case on both backends (`abs(double - ref) <= B + reference uncertainty`). Its looseness `B / actual` has median 620 (native) and 1100 (Python), ranging from 5.2 to 8.7e6 (native) and 12 to 1.3e7 (Python); it is tightest at small squeezing and few clicks and loosest at large squeezing. No warning was missed: every case with relative error above 1e-6 warned (80/80 native, 72/72 Python), with 37 native and 45 Python false alarms. Double precision failed (relative error above 1e-6) in 80 of 360 random cases (native) and 72 (Python), all at r <= 0.5 with many clicks; Piquasso 8.0.1 fails in the same regime (82 of 360) and returned 0.0 for 14 nonzero probabilities without a warning. Data: `benchmarks/gaussian_boson_sampling/results/threshold_reference_matrix_20261111-233720.json`; see [GBS threshold accuracy](validation/gbs-threshold-accuracy.md). |
| Status | **Implemented but experimental** (the alternating-sum part of the bound, including the factor `max(1, 16 / sqrt(2^k - 1))`, is calibrated against mpmath, not proven; it held on all 382 reference-matrix cases) |

Example: see the `ThresholdPrecisionWarning` example below.

### ThresholdPrecisionWarning

| Field | Value |
|---|---|
| Description | `RuntimeWarning` subclass issued by `threshold_probability` in three cases, checked in order. (1) The value is negative by more than `B`: "Threshold probability ... is negative beyond its absolute rounding bound ...: the stored covariance and mean are not a physical state to this accuracy (e.g. a covariance below the vacuum by rounding of its entries)". (2) `abs(value) <= B`: "... is consistent with zero within its absolute rounding bound ...; pass precision=<digits> for a multiprecision evaluation". (3) Otherwise `B / abs(value) > 1e-6`: "... has estimated relative rounding error ...; pass precision=<digits> ...". So every negative result is flagged. |
| Python API | `photographiq.sampling.ThresholdPrecisionWarning` |
| Implementation | `src/photographiq/sampling/gbs.py` (`threshold_probability`) |
| Backends | All threshold paths (native, Python, mpmath). |
| CPU / GPU | CPU only. |
| Assumptions | Relative threshold `1e-6`, or `rtol` with `precision="auto"`; depends on the bound of the previous entry. On the 382-case reference matrix no warning was missed (every case above 1e-6 relative error warned), and 37 native / 45 Python warnings were false alarms with actual errors between 3.5e-11 and 8.2e-7, the cost of the bound's margin. Examples on `product = apply_loss(squeezed_interferometer_state([0.5, 0.0], np.eye(2)), 0.9)`, a lossy squeezed mode next to a vacuum mode. Pattern `(0, 1)` gives `-4.93e-17` with `B = 1.9e-31` and warns "negative beyond". That value is exact for the stored covariance, because the rounded variance `0.9 + 0.1` of mode 1 is below the vacuum. Before the fix this pattern gave no warning. Pattern `(1, 1)` gives exactly 0 (`error = inf`) and warns "consistent with zero", where the old warning was the generic one. `threshold_distribution` and `sample_threshold` can emit the warning once per pattern. Because the bound is loose for strongly squeezed states, case (3) can fire for small but accurate probabilities there. |
| Tests | `tests/sampling/test_sampling_gbs.py::test_threshold_flags_negative_and_zero_results` (both backends: `(0, 1)` warns "negative beyond"; `(1, 1)` and a TMSV value of `-2.5e-17` inside its bound warn "consistent with zero within"; an ordinary TMSV `(1, 1)` stays silent); `::test_threshold_error_estimate_bounds_multiprecision_reference` (`pytest.warns(ThresholdPrecisionWarning, match="precision=")`; the test passes, see the previous entry). |
| Status | **Implemented but experimental** (cases (1) to (3) are triggered by the calibrated bound of the previous entry) |

```python
import warnings, numpy as np
from photographiq import sampling as S
state = S.squeezed_interferometer_state(np.full(16, 0.2), S.haar_unitary(16, rng=3), 0.9)
clicks = [1] * 8 + [0] * 8
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    p, err = S.threshold_probability(state, clicks, return_error=True)
print(p, err, caught[0].category.__name__)
# 1.132975991264507e-09 1.9834915903645424e-05 ThresholdPrecisionWarning
```

(Default backend, native here. The `precision=30` reference is `1.1329737713467033e-09`, so the actual relative error is 2.0e-6, within the reported 2.0e-5 and about 10x below it. Before the fix this example printed `1.13297519146997e-09 2.580299906299249e-06`.)

### Multiprecision threshold probabilities (`precision=`)

| Field | Value |
|---|---|
| Description | Evaluates the dark-mode-conditioned inclusion-exclusion sum in mpmath with the requested number of decimal digits instead of double precision. |
| Python API | `photographiq.sampling.threshold_probability(state, clicks, *, precision=<digits>)`; optional extra `photographiq[precision]` (`mpmath>=1.3`). |
| Implementation | `src/photographiq/sampling/gbs.py` (`_vacuum_sum_mp`; since 2026-11-12 it walks the 2^k subsets depth first and extends one Cholesky factor per added mode, O(k^2) multiprecision operations per subset instead of a fresh determinant) |
| Backends | mpmath only for the evaluation. `backend=` is still validated: an unknown name raises `ValueError("Unknown backend ...")`, and `backend="native"` raises `ImportError` if the native extension is missing. Before the fix, `precision=20, backend="bogus"` returned a value. |
| CPU / GPU | CPU only (pure Python and single threaded; the incremental Cholesky path is much faster than the former determinant-per-subset path, but 16 clicks still take minutes per probability; timings are in [GBS threshold accuracy](validation/gbs-threshold-accuracy.md), not here). |
| Assumptions | `precision` must be an `int >= 16` (`ValueError("precision is a number of decimal digits >= 16")`); `ImportError` with an install hint if mpmath is missing. The covariance and mean are used exactly as stored in double precision, so the result is exact for the stored state, not for the ideal one. The returned error estimate `P_Z mp.eps sum_T abs(1 + w_T)` does not strictly include rounding in the conditioning step, which is negligible at 30 or more digits. |
| Tests | `tests/sampling/test_sampling_gbs.py::test_threshold_multiprecision_matches_closed_forms` (TMSV `P(0,0) = 1/cosh^2 r`, `P(1,1)`, coherent-state click product, all to `rel=1e-15`; rejects `precision=8`), `::test_threshold_precision_validates_backend`; used as the reference in `::test_threshold_rounding_bound_holds_against_multiprecision` and `::test_threshold_error_estimate_bounds_multiprecision_reference` (both pass). `benchmarks/gaussian_boson_sampling/threshold_mp_reference.py` is a separate 50-digit reference for benchmark cases. |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
state = S.squeezed_interferometer_state(np.full(16, 0.2), S.haar_unitary(16, rng=3), 0.9)
print(S.threshold_probability(state, [1] * 8 + [0] * 8, precision=30))
# 1.1329737713467033e-09
```

### Adaptive precision (`precision="auto"`, `ThresholdInfo`)

| Field | Value |
|---|---|
| Description | Evaluates in double precision first and recomputes in mpmath only when the double bound exceeds the tolerance (`B > rtol abs(P)`). The number of digits is chosen from the measured cancellation, `ceil(log10(X / max(abs(P), B)) + log10(1/rtol)) + 10` (20 to 400), where `X = P_Z (2^k + sum_T abs(w_T))` is the term scale of the mpmath sum; if mpmath's own estimate still exceeds `rtol`, it runs once more with more digits. `return_info=True` reports which path ran. |
| Python API | `photographiq.sampling.threshold_probability(state, clicks, *, precision="auto", rtol=1e-6, return_info=False)`; `photographiq.sampling.ThresholdInfo(path, digits, error, double_error)` (`path` is `"double"` or `"mpmath"`; `digits` is `None` on the double path; `error` is what `return_error=True` returns; `double_error` is the double bound `B / abs(P)`, `None` for `precision=<digits>`) |
| Implementation | `src/photographiq/sampling/gbs.py` (`threshold_probability`, `ThresholdInfo`, `_threshold_double`, `_vacuum_sum_mp`) |
| Backends | Standalone sampling module; the double pass uses `backend=` (native or NumPy), the mpmath pass needs the `precision` extra |
| CPU / GPU | CPU only |
| Assumptions | Opt-in: existing calls and return types are unchanged. In `auto` mode `rtol` (in (0, 1), else `ValueError`) is also the warning threshold. `return_info` cannot be combined with `return_error` (`ValueError`). A missing mpmath raises `ImportError` naming `photographiq[precision]` on every `auto` call, not only on calls that need it, so behaviour does not depend on the data. Zero patterns with `B = 0` stay in double and return the exact 0; other zero patterns can only be shown consistent with zero. The mpmath estimate `P_Z mp.eps sum abs(terms)` ignores the error growth of the Cholesky factorisation; the `+10`-digit margin covered it on the reference matrix. `threshold_distribution` and `sample_threshold` do not take `precision`. On the 382-case reference matrix, 133 cases went to mpmath (25 to 51 digits) and 249 stayed in double; the largest relative error against the 40-digit reference was 3.1e-8 on the double path and 7.6e-10 on the mpmath path (109 cases with reference uncertainty below 1e-10), below `rtol = 1e-6` everywhere, no random case warned, and the actual error never exceeded the mpmath estimate. |
| Tests | `tests/sampling/test_sampling_gbs.py::test_threshold_auto_precision_stays_in_double_when_accurate` (both backends, including a zero pattern), `::test_threshold_auto_precision_switches_to_mpmath` (16 modes, 8 clicks, r = 0.2: double bound 2e-5 above `rtol`, mpmath path; `rtol=1e-4` stays in double; `rtol=0` raises), `::test_threshold_precision_without_mpmath_names_the_extra`; reference matrix `benchmarks/gaussian_boson_sampling/threshold_reference_matrix.py` (Section 32) |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
state = S.squeezed_interferometer_state(np.full(16, 0.2), S.haar_unitary(16, rng=3), 0.9)
clicks = [1] * 8 + [0] * 8
p, info = S.threshold_probability(state, clicks, precision="auto", return_info=True)
print(p, info.path, info.digits, f"{info.double_error:.1e}", f"{info.error:.1e}")
tmsv = S.squeezed_interferometer_state([0.8, -0.8], np.array([[1, -1], [1, 1]]) / np.sqrt(2))
print(S.threshold_probability(tmsv, [0, 0], precision="auto", return_info=True)[1])
# 1.1329737713467033e-09 mpmath 28 2.0e-05 4.3e-18
# ThresholdInfo(path='double', digits=None, error=5.912305713032234e-15, double_error=5.912305713032234e-15)
```

The first value equals the `precision=30` reference of the previous entry; no `ThresholdPrecisionWarning` is issued, because the returned value meets `rtol`.

## 21. Measurement calculus

`photographiq.calculus` provides a qubit measurement calculus after Danos, Kashefi and Panangaden (DKP, J. ACM 54, 8 (2007)) and a command-level view of CV patterns. All of it is pure Python/NumPy, runs on the CPU, and does not go through a simulation backend. On the CV side there are `command_view` and `ideal_limit_check` (this section) and, since 2026-11-12, the restricted Gaussian graphical calculus of Menicucci, Flammia and van Loock for pure Gaussian states (`GaussianGraph`, Section 22). No general CV-ZX calculus or graphical symplectic algebra is implemented.

### Qubit command patterns (`QubitPattern`)

| Field | Value |
|---|---|
| Description | Ordered qubit patterns built from the commands `N` (prepare \|+>), `E` (CZ), `M` (plane/angle measurement with s- and t-domains), `X`, `Z` (domain-conditioned Paulis) and `S` (classical signal shift). The constructor checks that the pattern is well defined and causal. |
| Python API | `photographiq.calculus.qubit.N(node)`, `E(u, v)`, `M(node, plane=Plane.XY, angle=0.0, s_domain=frozenset(), t_domain=frozenset())`, `X(node, domain=frozenset())`, `Z(node, domain=frozenset())`, `S(node, domain=frozenset())`; `QubitPattern(inputs=(), commands=(), outputs=None)` with `.validate()`, `.measured`, `.dependencies()`, `.dependency_graph()`, `.schedule()`, `.is_standard()`, `.reorder_measurements(order)`, `.copy(commands=None)` |
| Implementation | `src/photographiq/calculus/qubit.py` |
| Backends | Backend-independent (no simulation backend is used) |
| CPU / GPU | CPU only (pure Python) |
| Assumptions | Angles in radians. Measured observables: XY `cos(a) X + sin(a) Y`, YZ `cos(a) Z + sin(a) Y`, XZ `cos(a) Z + sin(a) X`; outcome 0 is the +1 eigenvalue (same planes as Graphix 0.4, which uses units of pi). Commands are listed in execution order, the reverse of DKP operator notation. `outputs=None` infers the unmeasured qubits. `validate()` raises `ValueError` for duplicate inputs or outputs, re-preparation, a CZ on one qubit, use of an unprepared or measured qubit, domains that read unmeasured qubits, an `S` on an unmeasured qubit or with its own signal in the domain, and outputs that differ from the unmeasured qubits; it raises `TypeError` for an object that is not a calculus command. `reorder_measurements` requires a standard pattern. |
| Tests | `tests/calculus/test_calculus_qubit.py::test_basis_is_eigenbasis_of_documented_observable`, `::test_domain_table_equals_pauli_conjugation` (domain table against explicit Pauli conjugation), `::test_validation_rejects_indefinite_patterns`, `::test_dependencies_and_schedule`; `tests/calculus/test_calculus_graphix.py::test_graphix_plane_convention_is_ours` (Graphix 0.4) |
| Status | **Implemented and validated** |

```python
from photographiq.calculus.qubit import N, E, M, X, QubitPattern
p = QubitPattern((1,), [N(2), E(1, 2), M(1, "XY", -0.6), X(2, {1})])
print(p, p.is_standard(), p.schedule())
# QubitPattern(inputs=(1,), outputs=(2,), commands=4) True (frozenset({1}),)
try:
    QubitPattern((0,), [M(0), X(0, {0})])
except ValueError as e:
    print(e)
# Command 1 acts on an unprepared or measured qubit
```

### Exact branch simulation

| Field | Value |
|---|---|
| Description | Runs a qubit pattern for fixed binary outcomes and returns the branch Kraus operator, which maps all input states at once. `enumerate_branches` returns all `2**m` branches. |
| Python API | `photographiq.calculus.qubit.run_branch(pattern, outcomes) -> Branch`; `enumerate_branches(pattern) -> list[Branch]`; `branch_operators(pattern, nodes=None) -> dict`; `Branch(outcomes, signals, operator)` with `Branch.probability(state=None)` (default: maximally mixed input); `MAX_QUBITS = 20`; helpers `observable(plane, angle)`, `measurement_basis(plane, angle, outcome)`, `effective_angle(plane, angle, s, t)` |
| Implementation | `src/photographiq/calculus/qubit.py` |
| Backends | Backend-independent (its own dense NumPy statevector) |
| CPU / GPU | CPU (NumPy) |
| Assumptions | Exact, no sampling. Big-endian, first listed qubit most significant. The cost is exponential in the number of qubits that are live at the same time plus the inputs; more than `MAX_QUBITS = 20` raises `ValueError`. Branches are keyed by the final signals; each branch is defined only up to a phase, which cannot be observed. |
| Tests | `tests/calculus/test_calculus_graphix.py::test_branch_states_match_graphix_statevector` (Graphix 0.4 statevector simulator, at least 50 branches, overlap error < 1e-10); `tests/calculus/test_calculus_qubit.py::test_run_branch_probabilities_sum_to_one`, `::test_long_line_runs_with_two_live_qubits` (31-qubit line against the closed-form product of J gates) |
| Status | **Implemented and validated** |

```python
from photographiq.calculus.qubit import N, E, M, X, QubitPattern, enumerate_branches, run_branch
p = QubitPattern((1,), [N(2), E(1, 2), M(1, "XY", -0.6), X(2, {1})])
print([round(b.probability(), 6) for b in enumerate_branches(p)], run_branch(p, {1: 0}).operator.shape)
# [0.5, 0.5] (2, 2)
```

### Pattern equivalence

| Field | Value |
|---|---|
| Description | Compares two patterns as linear maps, branch by branch: the Kraus operators must agree up to a phase per branch, and the branch probabilities (`K^dagger K`) must agree with no phase freedom. |
| Python API | `photographiq.calculus.equivalent(a, b, *, atol=1e-10) -> EquivalenceReport` (fields `equivalent`, `max_deviation`, `max_probability_deviation`, `phases`; truthy when equivalent) |
| Implementation | `src/photographiq/calculus/qubit.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU (NumPy) |
| Assumptions | Both patterns must have the same ordered inputs and outputs and the same measured set; otherwise `ValueError`. Uses the dense branch simulator, so the same size limit applies. |
| Tests | Positive cases only: `tests/calculus/test_calculus_qubit.py::test_each_rewrite_rule_preserves_the_map`, `::test_standardization_preserves_map_and_reaches_nemc`, `::test_signal_shifting_preserves_map_and_empties_flipping_domains`, `::test_reordering_measurements_along_dependencies`, `::test_causal_flow_line_pattern_is_composition_of_j_gates`. The branch operators it compares are checked against the Graphix 0.4 statevector in `tests/calculus/test_calculus_graphix.py`, but no test asserts that two different maps are reported as not equivalent. |
| Status | **Implemented but experimental** |

```python
from photographiq.calculus.qubit import N, E, M, X, QubitPattern
from photographiq.calculus import equivalent
p = QubitPattern((1,), [N(2), E(1, 2), M(1, "XY", -0.6), X(2, {1})])
q = QubitPattern((1,), [N(2), E(1, 2), M(1, "XY", -0.5), X(2, {1})])
print(equivalent(p, p).equivalent, equivalent(p, q).equivalent)
# True False
```

### Strong determinism check (`is_deterministic`)

| Field | Value |
|---|---|
| Description | Checks strong determinism: every branch equals `2**(-m/2)` times one common isometry, up to a phase per branch. Returns that isometry. |
| Python API | `photographiq.calculus.is_deterministic(pattern, *, atol=1e-10) -> DeterminismReport` (fields `deterministic`, `isometry`, `max_deviation`) |
| Implementation | `src/photographiq/calculus/qubit.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU (NumPy) |
| Assumptions | Ideal qubits; enumerates all `2**m` branches (exponential in the number of measurements). |
| Tests | `tests/calculus/test_calculus_qubit.py::test_j_pattern_implements_h_rz` (closed form `J(a) = H Rz(a)`), `::test_dkp_cnot_pattern` (closed-form CNOT), `::test_gflow_patterns_are_deterministic_for_all_planes` (brute-force gflow, with a negative control that drops corrections) |
| Status | **Implemented and validated** |

```python
from photographiq.calculus.qubit import N, E, M, X, QubitPattern, is_deterministic
p = QubitPattern((1,), [N(2), E(1, 2), M(1, "XY", -0.6), X(2, {1})])
r = is_deterministic(p)
print(r.deterministic, r.max_deviation < 1e-12)
# True True
print(is_deterministic(QubitPattern((1,), [N(2), E(1, 2), M(1, "XY", -0.6)])).deterministic)
# False
```

### Standardization (`standardize`)

| Field | Value |
|---|---|
| Description | Rewrites a pattern to NEMC standard form (`N* E* M* (X\|Z)* S*`) with the DKP rules EX, EZ, MX, MZ, signal commutation and commutation of commands on disjoint qubits. `rewrite_at` applies a single rule. |
| Python API | `photographiq.calculus.standardize(pattern, *, trace=None) -> QubitPattern` (`trace` receives `(rule, index)` pairs); `photographiq.calculus.qubit.rewrite_at(pattern, index) -> (rule, QubitPattern)` |
| Implementation | `src/photographiq/calculus/qubit.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU only (pure Python) |
| Assumptions | Rule names recorded in `trace`: `EX`, `EZ`, `MX`, `MZ`, `commute`, `S-commute`. EX, EZ, signal and commutation steps are exact operator identities; MX and MZ hold up to a phase per branch. X and Z corrections keep their relative order; adjacent corrections of one kind on one qubit are merged and empty ones dropped. `rewrite_at` raises `ValueError` when no rule applies at the index. Graphix puts Z before X in standard form, so the comparison with Graphix uses XOR-ed domains per output. |
| Tests | `tests/calculus/test_calculus_graphix.py::test_standardization_matches_graphix` (Graphix 0.4 `standardize`, 40 random patterns); `tests/calculus/test_calculus_qubit.py::test_standardization_preserves_map_and_reaches_nemc` (map preserved, idempotent), `::test_each_rewrite_rule_preserves_the_map`, `::test_reordering_measurements_along_dependencies` |
| Status | **Implemented and validated** |

```python
from photographiq.calculus.qubit import N, E, M, X, QubitPattern
from photographiq.calculus import standardize, equivalent
p = QubitPattern((0,), [N(1), E(0, 1), M(0, "XY", 0.3), X(1, {0}),
                        N(2), E(1, 2), M(1, "XY", 0.5), X(2, {1})])
trace = []
s = standardize(p, trace=trace)
print(p.is_standard(), s.is_standard(), sorted({rule for rule, _ in trace}))
# False True ['EX', 'MX', 'commute']
print(equivalent(p, s).equivalent, [type(c).__name__ for c in s.commands])
# True ['N', 'N', 'E', 'E', 'M', 'M', 'Z', 'X']
```

### Signal shifting (`signal_shift`)

| Field | Value |
|---|---|
| Description | DKP signal shifting: removes from each measurement domain the part that flips the outcome (XY: t-domain; YZ and XZ: s-domain), rewrites later domains in raw outcomes, and appends `S` commands that restore the original signals, so branch keys are unchanged. |
| Python API | `photographiq.calculus.signal_shift(pattern) -> QubitPattern` |
| Implementation | `src/photographiq/calculus/qubit.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU only (pure Python) |
| Assumptions | The input pattern must not contain `S` commands; otherwise `NotImplementedError`. Makes the same per-plane choice as Graphix 0.4 (`shift_signals(method="direct")`). |
| Tests | `tests/calculus/test_calculus_graphix.py::test_signal_shifting_matches_graphix` (Graphix 0.4, 40 random patterns); `tests/calculus/test_calculus_qubit.py::test_signal_shifting_preserves_map_and_empties_flipping_domains` |
| Status | **Implemented and validated** |

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, QubitMeasurement
from photographiq.flows import find_causal_flow
from photographiq.calculus import pattern_from_flow, signal_shift, equivalent
from photographiq.calculus.qubit import M, S
og = OpenGraph(Model.QUBIT, nx.path_graph(4), (0,), (3,),
               {n: QubitMeasurement("XY", 0.2 * (n + 1)) for n in range(3)})
pat = pattern_from_flow(og, find_causal_flow(og))
sh = signal_shift(pat)
print([(c.node, set(c.s_domain), set(c.t_domain)) for c in sh.commands if isinstance(c, M)])
# [(0, set(), set()), (1, {0}, set()), (2, {1}, set())]
print([(c.node, set(c.domain)) for c in sh.commands if isinstance(c, S)], equivalent(pat, sh).equivalent)
# [(2, {0})] True
```

### Flow to pattern (`pattern_from_flow`)

| Field | Value |
|---|---|
| Description | Builds the standard pattern of a qubit open graph from a correction table or a certified `FlowAnalysisResult`: `N` for non-inputs, `E` for every edge, `M` in the given order with s/t domains from the X/Z corrections, then the output corrections. |
| Python API | `photographiq.calculus.pattern_from_flow(graph, corrections, order=None) -> QubitPattern`; `corrections` is `{node: {'x': set, 'z': set}}`, the transposed `{'x': {...}, 'z': {...}}` table, or an object with `correction_map` and `measurement_order` |
| Implementation | `src/photographiq/calculus/qubit.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU only (pure Python) |
| Assumptions | Covers causal-flow and gflow results; Pauli-flow results are in section 25. Needs `Model.QUBIT` (`ValueError` otherwise). A result whose status is not `certified` is rejected (`ValueError`). An order that does not list every measured qubit exactly once, a correction on an unknown qubit and a correction on an earlier-measured qubit in a supplied table raise `ValueError` (this includes raw Pauli-flow tables; results from Pauli flow arrive without those components, section 25, as the docstring now says). The `pauli` tag of a measurement is ignored; plane and angle define it. Determinism is not assumed: check it with `is_deterministic`. A transposed table is rejected as ambiguous when qubits are named `'x'` and `'z'`. |
| Tests | `tests/calculus/test_calculus_qubit.py::test_causal_flow_line_pattern_is_composition_of_j_gates` (closed-form product of J gates), `::test_gflow_patterns_are_deterministic_for_all_planes` (brute-force gflow, `calculus_random.brute_force_gflow`), `::test_pattern_from_flow_accepts_flow_analysis_results`, `::test_pattern_from_flow_rejects_cv_graphs`, `::test_correction_table_named_x_z_is_rejected_as_ambiguous`, `::test_pattern_from_pauli_flow_results_is_deterministic` (section 25) |
| Status | **Implemented and validated** |

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, QubitMeasurement
from photographiq.flows import find_gflow
from photographiq.calculus import pattern_from_flow, is_deterministic
og = OpenGraph(Model.QUBIT, nx.path_graph(3), (0,), (2,),
               {0: QubitMeasurement("XY", 0.4), 1: QubitMeasurement("XY", 1.1)})
pat = pattern_from_flow(og, find_gflow(og))
print(pat.is_standard(), is_deterministic(pat).deterministic)
# True True
```

### CV command-level view (`command_view`, `affine_form`)

| Field | Value |
|---|---|
| Description | Classifies each command of a CV `Pattern` as quantum, feed-forward, measurement or classical; lists the outcome variables (continuous or discrete, number of components, homodyne angle when fixed); extracts displacement corrections as exact affine forms of outcomes `c0 + sum_k c_k m_k`. Expressions that are not affine are kept verbatim. |
| Python API | `photographiq.calculus.cv.command_view(pattern, parameters=None) -> CVPatternView`; `affine_form(value, parameters=None) -> AffineForm \| None`; `AffineForm.evaluate(records)`; `DisplacementCorrection.affine` |
| Implementation | `src/photographiq/calculus/cv.py` |
| Backends | Backend-independent (read-only view of a `Pattern`) |
| CPU / GPU | CPU only (pure Python) |
| Assumptions | Nothing is approximated: nonlinear dependence on outcomes, unbound parameters and `CallableExpression` give `None` (the raw expression is kept). `PhotonNumber` outcomes are discrete; heterodyne and generaldyne outcomes have two components. |
| Tests | `tests/calculus/test_calculus_cv.py::test_command_view_roles_outcomes_and_corrections`, `::test_cv_flow_corrections_are_exact_affine_forms` (reference: direct evaluation of the original `Displace` expressions with `resolve`), `::test_nonaffine_and_unbound_expressions_are_kept_verbatim`, `::test_photon_counting_outcomes_are_discrete` |
| Status | **Implemented and validated** |

```python
from photographiq.calculus.cv import command_view, wire_step_pattern
view = command_view(wire_step_pattern(0.3, 1.0))
print([c.role.value for c in view.commands])
# ['quantum', 'quantum', 'measurement', 'feedforward']
print(view.corrections[0].affine, view.corrections[0].q.coefficients)
# True {(0, None): -1.044030650891055}
```

### CV ideal-limit identity check (`ideal_limit_check`)

| Field | Value |
|---|---|
| Description | For each resource squeezing r, evaluates the unconditional Gaussian channel `V -> S V S^T + N` of `build(r)` and reports the deviation of `S` from an ideal symplectic target separately from the added noise `N` (optionally against an analytic `expected_noise(r)`). Nothing is extrapolated. `wire_step_pattern(k, r)` builds one CV wire step. |
| Python API | `photographiq.calculus.cv.ideal_limit_check(build, target, squeezings, *, parameters=None, expected_noise=None) -> IdealLimitReport` (`points`, `max_matrix_error`, `noise_decreasing`); `wire_step_pattern(k, squeezing) -> Pattern` |
| Implementation | `src/photographiq/calculus/cv.py` (uses `src/photographiq/analysis.py::gaussian_channel`) |
| Backends | Gaussian channel analysis (`photographiq.analysis.gaussian_channel`), not a registered backend |
| CPU / GPU | CPU (NumPy) |
| Assumptions | hbar = 2, vacuum covariance I. For one wire step `S = T(k) = [[-k, -1], [1, 0]]` exactly at every finite r and `N = diag(0, e^{-2r})`. The channel is outcome-averaged: `matrix_error == 0` does not mean that each corrected trajectory implements the target at finite r. Adaptive or non-Gaussian patterns raise `NotImplementedError` (as in `gaussian_channel`); a target of the wrong shape raises `ValueError`. |
| Tests | `tests/calculus/test_calculus_cv.py::test_wire_step_ideal_map_is_exact_and_noise_is_analytic` (closed form `T(k)` and `diag(0, e^{-2r})`; matrix error < 1e-14, noise error < 1e-15), `::test_compiled_identity_matches_closed_form_wire_channel` (`gaussian.wire_channel`), `::test_ideal_limit_check_rejects_shape_mismatch_and_adaptive_patterns` |
| Status | **Implemented and validated** |

```python
from photographiq.calculus.cv import ideal_limit_check, wire_step_pattern
from photographiq.gaussian import teleportation_matrix
rep = ideal_limit_check(lambda r: wire_step_pattern(0.3, r), teleportation_matrix(0.3), [0.5, 1.0, 2.0])
print(rep.max_matrix_error < 1e-14, rep.noise_decreasing, [round(p.noise_norm, 4) for p in rep.points])
# True True [0.3679, 0.1353, 0.0183]
```

## 22. Local Clifford transformations

`photographiq.calculus.clifford` implements the single-qubit Clifford group, qubit graph states, local complementation with the local Clifford that implements it, and an exhaustive LC-equivalence search for small labelled graphs. Local complementation of CV (weighted) graph states is explicitly not implemented. For CV graph states, `photographiq.calculus.gaussian_graph` (last entry of this section) implements how Gaussian unitaries and ideal homodyne measurements rewrite the complex adjacency matrix of a pure Gaussian state. It is placed here because it is the CV counterpart of local graph-state transformations; it is not a ZX calculus (Section 23).

### Single-qubit Clifford group (`Clifford`)

| Field | Value |
|---|---|
| Description | The 24 single-qubit Cliffords modulo global phase, generated breadth-first from H and S, with a multiplication table, inverses and Pauli conjugation. |
| Python API | `photographiq.calculus.Clifford(index)`; `Clifford.from_matrix(matrix)`, `Clifford.named(word)` (word over `I, X, Y, Z, H, S`, matrix product), `Clifford.all()`, `.matrix`, `.name`, `a @ b`, `.inverse()`, `.conjugate(pauli) -> (sign, P')` |
| Implementation | `src/photographiq/calculus/clifford.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU (NumPy) |
| Assumptions | `Clifford.name` is a shortest H/S word read as a matrix product (`"HS"` means `H @ S`, S acts first). `from_matrix` raises `ValueError` for non-unitary or non-Clifford matrices. |
| Tests | `tests/calculus/test_calculus_clifford.py::test_group_has_24_distinct_unitaries_and_closed_table`, `::test_pauli_conjugation_is_a_signed_permutation`, `::test_non_clifford_is_rejected` (reference: explicit 2x2 matrix products) |
| Status | **Implemented and validated** |

```python
from photographiq.calculus import Clifford
print(len(Clifford.all()), Clifford.named("HS"), Clifford.named("H").conjugate("X"))
# 24 Clifford(4, 'HS') (1, 'Z')
```

### Local complementation (`local_complement`, `lc_cliffords`)

| Field | Value |
|---|---|
| Description | `local_complement(G, v)` returns `tau_v(G)`, which toggles every edge inside the neighbourhood of v. `lc_cliffords(G, v)` returns the local Cliffords of `U_v = exp(-i pi/4 X_v) prod_{w in N(v)} exp(+i pi/4 Z_w)`, with `U_v \|G> = \|tau_v(G)>` up to a global phase. |
| Python API | `photographiq.calculus.local_complement(graph, v) -> nx.Graph`; `photographiq.calculus.clifford.lc_cliffords(graph, v) -> dict`; `graph_state(graph, order=None)`; `apply_local_cliffords(state, cliffords, order)` |
| Implementation | `src/photographiq/calculus/clifford.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU (NumPy) |
| Assumptions | Unweighted qubit graph states `\|G> = prod CZ \|+>^n`, big-endian in the given node order (Van den Nest, Dehaene and De Moor, PRA 69, 022316 (2004)). Dense statevectors, so small graphs only. |
| Tests | `tests/calculus/test_calculus_clifford.py::test_graph_state_is_stabilized_by_generators` (stabilizer generators), `::test_local_complementation_unitary_maps_graph_states` (statevector proportionality and stabilizers of `tau_v(G)`, random graphs up to 6 nodes) |
| Status | **Implemented and validated** |

```python
import networkx as nx, numpy as np
from photographiq.calculus import local_complement
from photographiq.calculus.clifford import lc_cliffords, graph_state, apply_local_cliffords
star = nx.star_graph(3)
tau = local_complement(star, 0)
order = tuple(star)
out = apply_local_cliffords(graph_state(star, order), lc_cliffords(star, 0), order)
print(sorted(tau.edges), round(abs(np.vdot(graph_state(tau, order), out)), 12))
# [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] 1.0
```

### LC equivalence of labelled graph states (`lc_equivalence`)

| Field | Value |
|---|---|
| Description | Breadth-first search of the LC orbit of `g1` for `g2` on the same labelled vertices. Returns a witness (sequence of local complementations and the total local Clifford per vertex), checked on statevectors, or `None` after the orbit is exhausted. |
| Python API | `photographiq.calculus.lc_equivalence(g1, g2, *, max_nodes=10, max_orbit=1_000_000, verify_up_to=12) -> LCEquivalence \| None` (fields `sequence`, `cliffords`, `residual`) |
| Implementation | `src/photographiq/calculus/clifford.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU (NumPy) |
| Assumptions | Exponential search; small graphs only. Vertex relabelling (graph isomorphism) is not considered. Different vertex sets or more than `max_nodes` nodes raise `ValueError`; an orbit larger than `max_orbit` raises `RuntimeError` (never a silent negative answer). `residual` is `None` above `verify_up_to` nodes. |
| Tests | `tests/calculus/test_calculus_clifford.py::test_lc_equivalence_recovers_random_sequences` (random LC sequences; witness checked by stabilizers), `::test_lc_inequivalence_agrees_with_cut_rank_invariant` (independent GF(2) cut-rank invariant; one negative pair, star vs path on 4 nodes) |
| Status | **Implemented and validated** |

```python
import networkx as nx
from photographiq.calculus import lc_equivalence
w = lc_equivalence(nx.star_graph(3), nx.complete_graph(4))
print(w.sequence, w.residual < 1e-12, lc_equivalence(nx.star_graph(3), nx.path_graph(4)))
# (0,) True None
```

### CV local complementation

| Field | Value |
|---|---|
| Description | A placeholder that raises: qubit local complementation does not transfer to weighted CV graph states, and no sound CV analogue has been derived or verified in this package. |
| Python API | `photographiq.calculus.clifford.cv_local_complementation(*args, **kwargs)` |
| Implementation | `src/photographiq/calculus/clifford.py` |
| Backends | None |
| CPU / GPU | Not applicable |
| Assumptions | Always raises `NotImplementedError` (the docstring explains why: CV local Gaussian unitaries act on weighted adjacency matrices by symplectic Moebius-type maps, Menicucci, Flammia and van Loock, PRA 83, 042335 (2011)). |
| Tests | `tests/calculus/test_calculus_clifford.py::test_cv_local_complementation_is_explicitly_unsupported` |
| Status | **Unsupported** |

```python
from photographiq.calculus.clifford import cv_local_complementation
try:
    cv_local_complementation()
except NotImplementedError as e:
    print(str(e).splitlines()[0])
# Not implemented: qubit LC does not transfer to CV graph states.
```

### Gaussian graphical calculus (`GaussianGraph`)

| Field | Value |
|---|---|
| Description | The graphical calculus for Gaussian pure states of Menicucci, Flammia and van Loock (PRA 83, 042335 (2011), arXiv:1007.0725; "MFvL"). A zero-mean pure Gaussian state is a complex symmetric matrix Z = V + iU with U > 0, read as the adjacency matrix of a graph with complex edge weights and self-loops; the displacement is tracked exactly in a separate mean vector. Gaussian unitaries act by the linear-fractional update Z' = (C + D Z)(A + B Z)^-1 (MFvL Eq. 2.24), an ideal q homodyne deletes the vertex, and an ideal p homodyne is a Schur-complement update with the conditional mean tracked. In hbar = 2 conventions the covariance is [[U^-1, U^-1 V], [V U^-1, U + V U^-1 V]] (qq/pp blocks), twice MFvL Eq. 2.19. **This is not a CV-ZX calculus**: it has no spiders, no rewrite system beyond these rules, and no link to the qubit ZX diagrams of Section 23. |
| Python API | `photographiq.calculus.GaussianGraph(Z, nodes=None, mean=None)` (immutable; `.Z`, `.nodes`, `.mean`); constructors `vacuum(nodes)`, `squeezed(nodes, r)`, `cluster(adjacency, r, nodes=None)` (Z = A + i e^{-2r} I, MFvL Eq. 2.29), `from_covariance(cov, nodes=None, mean=None, rtol=1e-9)`, `from_state(GaussianState)`; `to_covariance()`, `to_state()`, `edges(atol=0)`; gates `apply(S, nodes)`, `cz(u, v, g)`, `quadratic_phase(node, s)`, `rotate(node, t)`, `squeeze(node, r)`, `beamsplitter(u, v, t)`; measurements `measure_q(node, m)`, `measure_p(node, m)`; `tensor(other)`, `relabel(mapping)`, `reorder(nodes)` |
| Implementation | `src/photographiq/calculus/gaussian_graph.py` |
| Backends | Backend-independent; the convenience gates use the same matrices as `GaussianBackend`, and `to_state()` returns a `GaussianState` for any Gaussian backend |
| CPU / GPU | CPU (NumPy, dense O(N^3)) |
| Assumptions | hbar = 2, interleaved (q0, p0, ...), vacuum covariance I. Pure states only: `from_covariance` raises `ValueError("covariance is not a pure Gaussian state (mixed states have no Gaussian graph) ...")` when the rebuilt covariance differs by more than `rtol`; Z must be complex symmetric with Im Z positive definite, gates must be real symplectic, gate modes distinct, `tensor` labels disjoint and outcomes finite (each `ValueError`). Ideal (r = inf) graphs are rejected; ideal limits are studied as r -> inf sequences. Sequential p measurements through very strongly squeezed intermediate states lose relative accuracy in Im Z (at r = 8 the wire-shortening noise term is accurate to about 1%). Not implemented: mixed states, finite-efficiency or noisy homodyne, heterodyne and general-dyne, non-Gaussian elements (photon counting, cubic phase, GKP, cats), general CV-ZX or graphical symplectic algebra (for example Booth-Carette-Comfort), translation to qubit ZX, and the MFvL H-graph constructions, closest-cluster optimisation and entanglement measure. The determinism question for CV graphs is answered by CV flow (Section 26), not by this calculus. Guide: [Gaussian graphical calculus](user-guide/gaussian-graphical-calculus.md). |
| Tests | `tests/calculus/test_gaussian_graph.py` (213 tests, references computed by direct Gaussian numerics in `photographiq.gaussian` and `GaussianBackend`, not by the module): `cov -> Z -> cov` and `Z -> cov -> Z` round trips, every gate and `apply` with random symplectics against `GaussianBackend` covariance and mean, `measure_q` / `measure_p` / rotated homodyne against `GaussianBackend.measure` conditioning, the MFvL closed-form local rules (Eqs. 3.12, 3.14, 3.16, 3.18, 3.20, 3.23) against Eq. 2.24, purity `(Omega cov)^2 = -I`, the cluster state against backend `prepare` plus `entangle`, the wire-shortening limit, and rejection of mixed, unphysical and non-symplectic inputs. Largest error 2e-14 over 200 random seeds (tolerance 1e-10). |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq.backends.gaussian import GaussianBackend
from photographiq.calculus import GaussianGraph
from photographiq.measurements import Homodyne
wire = np.diag([1.0, 1.0, 1.0], 1); wire = wire + wire.T
g = GaussianGraph.cluster(wire, r=1.0)                       # Z = A + i e^{-2r} I
h = g.squeeze(0, 0.3).cz(0, 3, -0.7).measure_p(1, 0.5)      # Eq. 2.24 updates, then Schur complement
backend = GaussianBackend(); backend.set_state(g.to_state())
backend.squeeze(0, 0.3); backend.entangle(0, 3, -0.7)
backend.measure(1, Homodyne(), np.pi / 2, outcome=0.5)
s = backend.get_state()
print(h.nodes, np.abs(h.to_covariance() - s.covariance).max() < 1e-12, np.abs(h.mean - s.mean).max() < 1e-12)
for r in (1.0, 4.0):                                        # wire shortening: Z' -> single edge -1
    print(r, np.round(GaussianGraph.cluster(wire, r).measure_p(1, 0.0).measure_p(2, 0.0).Z, 6).tolist())
try:
    GaussianGraph.from_covariance(2 * np.eye(2))
except ValueError as e:
    print(str(e).split(":")[0])
# (0, 2, 3) True True
# 1.0 [[0.268236j, (-0.982014+0j)], [(-0.982014+0j), 0.268236j]]
# 4.0 [[0.000671j, (-1+0j)], [(-1+0j), 0.000671j]]
# covariance is not a pure Gaussian state (mixed states have no Gaussian graph)
```

## 23. ZX calculus

`photographiq.calculus.zx` implements qubit ZX diagrams with exact tensor semantics, including scalars, and a small set of sound rewrites. Each rewrite checks its precondition, multiplies `Diagram.scalar` by the exact factor and appends a `RewriteStep` to `Diagram.trace`. The implemented rules (with their `trace` names) are: spider fusion (`fuse`), self-loop removal (`self-loop`), identity removal including Hadamard-Hadamard cancellation through edge composition (`identity`), colour change (`color-change`), Pauli state copy (`copy`), the 2x2 bialgebra (`bialgebra`), Hopf (`hopf`), and cancellation of parallel Hadamard edges (`hadamard-pair`). Not implemented (no API exists): pi-copy, ZX local complementation and pivoting, circuit extraction, full simplification strategies, and any CV-ZX identities. The bialgebra of other arities is rejected with `RewriteError`. Tensor evaluation is dense and exponential. The only CV graphical calculus in the package is the restricted Gaussian graphical calculus `GaussianGraph` (Section 22); it is not a CV-ZX calculus and has no translation to or from these qubit diagrams.

### ZX diagrams and tensor semantics (`Diagram`)

| Field | Value |
|---|---|
| Description | Open ZX multigraph of Z/X spiders with phases, boundary vertices, and simple or Hadamard edges (self-loops allowed). `matrix()` returns the exact `2**n_out x 2**n_in` linear map times `scalar`. Sequential and parallel composition. |
| Python API | `photographiq.calculus.Diagram()`; `.add_vertex(kind, phase=0.0)`, `.add_edge(u, v, kind=EdgeType.SIMPLE)`, `.add_input()`, `.add_output()`, `.matrix()`, `.then(other)`, `.tensor(other)`, `.copy()`, `.validate()`, `.spiders()`, `.kind(v)`, `.phase(v)`, `.incident(v)`, `.degree(v)`, `.between(u, v, kind=None)`, `.loops(v)`, `.to_networkx()`; attributes `scalar`, `trace` (list of `zx.RewriteStep(rule, vertices, factor)`); `zx.VertexType` (`BOUNDARY`, `Z`, `X`), `zx.EdgeType` (`SIMPLE`, `HADAMARD`) |
| Implementation | `src/photographiq/calculus/zx.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU (NumPy `einsum`, greedy pairwise contraction) |
| Assumptions | Coecke-Kissinger conventions with exact scalars: Z spider `\|0..0><0..0\| + e^{ia} \|1..1><1..1\|`; X spider is H on every leg of a Z spider; Hadamard edge `[[1,1],[1,-1]]/sqrt(2)`. Each boundary has exactly one edge. First boundary most significant. |
| Tests | `tests/calculus/test_calculus_zx.py::test_spider_semantics` (closed-form spider matrices), `::test_composition_and_tensor_product` (matrix product and Kronecker product) |
| Status | **Implemented and validated** |

Example: see the rewrite example below.

### Sound rewrites (fusion, self-loops, identity, colour change, copy, Hopf, Hadamard pair)

| Field | Value |
|---|---|
| Description | `fuse` (same-colour spiders on a simple edge; phases add; factor 1), `remove_self_loops` (simple loop factor 1; each Hadamard loop adds pi and a factor 1/sqrt 2), `remove_identity` (phase-0 spider with two plain edges; edges compose, so H-H becomes simple), `color_change` (toggles incident edges; factor 1), `copy_state` (one-leg Pauli state through an opposite-colour spider; factor `e^{i a alpha} 2^{(1-n)/2}`), `hopf` (two parallel simple Z-X edges removed; factor 1/2), `cancel_hadamard_pair` (two parallel Hadamard edges between same-colour spiders; factor 1/2). |
| Python API | `photographiq.calculus.zx.fuse(d, u, v)`, `remove_self_loops(d, v)`, `remove_identity(d, v)`, `color_change(d, v)`, `copy_state(d, state, v)`, `hopf(d, u, v)`, `cancel_hadamard_pair(d, u, v)`; each returns a new `Diagram`; `RewriteError` (a `ValueError`) on a failed precondition |
| Implementation | `src/photographiq/calculus/zx.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU only (pure Python) |
| Assumptions | Rewrites never mutate their input. `copy_state` copies only Pauli states (phase 0 or pi). |
| Tests | `tests/calculus/test_calculus_zx.py::test_every_rewrite_is_scalar_exact_on_random_diagrams` (dense tensor of the diagram before and after, including the scalar; all 8 rules, 25 random diagrams each), `::test_rewrite_preconditions_are_enforced` |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq.calculus import zx
from photographiq.calculus.zx import Diagram, RewriteError
d = Diagram()
a, b = d.add_vertex("Z", 0.4), d.add_vertex("Z", 0.7)
d.add_edge(d.add_input(), a); d.add_edge(a, b); d.add_edge(b, d.add_output())
f = zx.fuse(d, a, b)
print(np.allclose(f.matrix(), d.matrix()), round(f.phase(a), 6), f.trace)
# True 1.1 [RewriteStep(rule='fuse', vertices=(0, 1), factor=(1+0j))]
h = Diagram()
z, x = h.add_vertex("Z"), h.add_vertex("X")
h.add_edge(h.add_input(), z); h.add_edge(z, x); h.add_edge(z, x); h.add_edge(x, h.add_output())
h2 = zx.hopf(h, z, x)
print(np.allclose(h2.matrix(), h.matrix()), h2.scalar)
# True (0.5+0j)
try:
    zx.fuse(h, z, x)
except RewriteError as e:
    print(e)
# Fusion needs spiders of the same colour
```

### Bialgebra rule

| Field | Value |
|---|---|
| Description | Bialgebra rewrite for a phase-0 Z spider and a phase-0 X spider joined by exactly one simple edge (factor sqrt 2). |
| Python API | `photographiq.calculus.zx.bialgebra(d, u, v)` |
| Implementation | `src/photographiq/calculus/zx.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU only (pure Python) |
| Assumptions | Only the 2x2 case is implemented: each spider must have exactly two other legs; other arities raise `RewriteError("Only the 2x2 bialgebra is implemented")`. Nonzero phases, self-loops or a non-simple link also raise `RewriteError`. |
| Tests | `tests/calculus/test_calculus_zx.py::test_every_rewrite_is_scalar_exact_on_random_diagrams` (tensor equality), `::test_rewrite_preconditions_are_enforced` |
| Status | **Partially supported** |

Example: see `tests/calculus/test_calculus_zx.py`.

### Spider simplification (`simplify_spiders`)

| Field | Value |
|---|---|
| Description | Applies self-loop removal, fusion and identity removal until a fixed point is reached; the trace records every step. |
| Python API | `photographiq.calculus.zx.simplify_spiders(d, *, max_steps=10_000) -> Diagram` |
| Implementation | `src/photographiq/calculus/zx.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU only (pure Python) |
| Assumptions | Not a complete simplifier: it does not apply colour change, copy, bialgebra or Hopf. Raises `RuntimeError` if `max_steps` is exceeded. |
| Tests | `tests/calculus/test_calculus_zx.py::test_simplify_spiders_preserves_tensor_and_records_trace` (tensor equality), `::test_j_pattern_simplifies_to_two_spiders` (closed form `H Rz(a)/sqrt 2` up to phase; the test asserts at most 3 spiders remain) |
| Status | **Implemented and validated** |

Example: see the `pattern_branch_diagram` example below.

### Graph-state and pattern-branch diagrams

| Field | Value |
|---|---|
| Description | `graph_state_diagram(G)` builds a diagram equal to `\|G>` (Z spiders, Hadamard edges, exact scalar `2^{(\|E\|-n)/2}`). `pattern_branch_diagram(pattern, outcomes)` builds a diagram whose matrix equals the Kraus operator of one branch of a `QubitPattern`. |
| Python API | `photographiq.calculus.zx.graph_state_diagram(graph, order=None) -> Diagram`; `pattern_branch_diagram(pattern, outcomes) -> Diagram` |
| Implementation | `src/photographiq/calculus/zx.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU (NumPy) |
| Assumptions | One diagram per outcome branch: no symbolic outcome variables. Measurement effects use the domain-adjusted angle of `effective_angle`. |
| Tests | `tests/calculus/test_calculus_zx.py::test_graph_state_diagram_equals_statevector` (reference `clifford.graph_state`), `::test_pattern_branch_diagrams_equal_kraus_operators` (reference `qubit.run_branch`, before and after `simplify_spiders`) |
| Status | **Implemented and validated** |

```python
import numpy as np, networkx as nx
from photographiq.calculus import zx
from photographiq.calculus.clifford import graph_state
from photographiq.calculus.qubit import N, E, M, X, QubitPattern, run_branch
print(np.allclose(zx.graph_state_diagram(nx.path_graph(3)).matrix()[:, 0], graph_state(nx.path_graph(3))))
# True
p = QubitPattern((1,), [N(2), E(1, 2), M(1, "XY", -0.6), X(2, {1})])
bd = zx.pattern_branch_diagram(p, {1: 1})
s = zx.simplify_spiders(bd)
print(np.allclose(s.matrix(), run_branch(p, {1: 1}).operator), len(bd.spiders()), len(s.spiders()))
# True 5 2
```

## 24. Flow and gflow

Flow analysis works on the typed graph IR `photographiq.ir.OpenGraph`. `photographiq.flows.analyze` applies only the flow notion that belongs to the graph's `Model`: causal flow, gflow and Pauli flow (GF(2), unweighted) for `Model.QUBIT`, and real-linear CV flow for `Model.CV_GAUSSIAN` (section 26). Every finder returns a `FlowAnalysisResult` and checks its certificate with an independent verifier before returning it. A result of `not_certified` is never a proof of nondeterminism; for CV graphs `diagnostics['decision'] == "impossible"` proves only that no CV flow exists for the given weights (Section 26). The Graphix comparison benchmark is `benchmarks/graphix/bench_graphix_mbqc.py` (see `benchmarks/README.md`); no timing numbers are quoted here. The guide page is [MBQC calculus and flows](user-guide/mbqc-calculus.md).

### Typed open graph IR (`OpenGraph`)

| Field | Value |
|---|---|
| Description | Backend-independent graph-level IR: graph with weighted edges, ordered inputs and outputs, one measurement spec per measured node, CV resource squeezing, and a `Model` that is part of the data. |
| Python API | `photographiq.ir.OpenGraph(model, graph, inputs=(), outputs=(), measurements={}, squeezing={})` (also `photographiq.OpenGraph`); `.nodes`, `.measured`, `.weight(u, v)`, `.adjacency(order=None)`, `.validate()`, `.copy()`, `.to_dict()`, `OpenGraph.from_dict(data)`, `OpenGraph.from_cvgraph(graph, shears=None, *, model=Model.CV_GAUSSIAN)`, `.to_cvgraph()`; `ir.Model` (`CV_GAUSSIAN`, `CV_NONGAUSSIAN`, `QUBIT`, `GKP_LOGICAL`), `ir.Plane` (`XY`, `YZ`, `XZ`), `ir.QubitMeasurement(plane=XY, angle=0.0, pauli=None)`, `ir.HomodyneSpec(shear=0.0)` (`.angle = atan2(1, shear)`) |
| Implementation | `src/photographiq/ir.py` |
| Backends | Backend-independent; `to_cvgraph()` gives an executable `CVGraph` (CV models only) |
| CPU / GPU | CPU only (pure Python, NetworkX) |
| Assumptions | Models are never interchanged: qubit graphs must be unweighted (`ValueError`), measurement specs must match the model (`TypeError`), squeezing is CV-only, outputs are not measured, no self-loops, edge weights finite and nonzero. Layout coordinates are not stored. `from_dict` turns JSON lists back into tuple labels. |
| Tests | `tests/unit/test_ir_open_graph.py::test_cvgraph_conversion_preserves_weights_squeezing_and_roles` (round trip compared with the original `CVGraph` network and an explicit adjacency matrix), `::test_json_round_trip_restores_tuple_labels`, `::test_qubit_model_semantics_are_enforced` |
| Status | **Implemented but experimental** (tests are round-trip and rejection checks; no independent reference applies to the container itself; matches Sections 2 and 5) |

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, QubitMeasurement
og = OpenGraph(Model.QUBIT, nx.path_graph(4), (0,), (3,), {n: QubitMeasurement("XY", 0.3) for n in range(3)})
print(og.measured, OpenGraph.from_dict(og.to_dict()).to_dict() == og.to_dict())
# (0, 1, 2) True
try:
    OpenGraph(Model.QUBIT, nx.Graph([(0, 1, {"weight": 2.0})]), (0,), (1,), {0: QubitMeasurement()})
except ValueError as e:
    print(e)
# qubit graphs are unweighted; edge (0, 1)
```

### Flow dispatcher (`analyze`)

| Field | Value |
|---|---|
| Description | Runs the flow analysis that matches the model. In `auto` mode on qubit graphs it tries causal flow, then gflow, then Pauli flow (only if some measurement has a `pauli` tag) and returns the first certified result; otherwise a `not_certified` result with the earlier attempts in `diagnostics['attempts']`. |
| Python API | `photographiq.flows.analyze(open_graph, algorithm="auto", **options) -> FlowAnalysisResult`; `flows.ALGORITHMS = ("auto", "causal", "gflow", "pauli", "cv")` |
| Implementation | `src/photographiq/flows/__init__.py` |
| Backends | Backend-independent (CV verification uses the `gaussian` backend, section 26) |
| CPU / GPU | CPU (NumPy) |
| Assumptions | Qubit algorithms on CV graphs, `cv` on qubit graphs, and any algorithm on `CV_NONGAUSSIAN`/`GKP_LOGICAL` return status `unsupported` (no exception). An unknown algorithm name raises `ValueError`; keyword options passed for a qubit algorithm (including `auto` on a qubit graph) raise `TypeError`. The `cv` check runs before the options check, so `analyze(qubit_graph, "cv", verify=False, tolerance=1e-8)` returns `unsupported`; before the fix it raised `TypeError: Unexpected options for qubit flow analysis`. Unknown options passed with `cv` on a qubit graph are therefore ignored, as for the non-Gaussian and GKP models. |
| Tests | `tests/flow/test_flows_qubit.py::test_model_and_input_validation` (also asserts the `cv`-with-options case and that `auto` with options still raises `TypeError`), `::test_no_flow_is_not_reported_as_nondeterminism`, `::test_gflow_without_causal_flow_xy_only` |
| Status | **Implemented and validated** |

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, QubitMeasurement
from photographiq import flows
tri = OpenGraph(Model.QUBIT, nx.complete_graph(3), (0, 1), (2,), {0: QubitMeasurement(), 1: QubitMeasurement()})
res = flows.analyze(tri)
print(res.status, res.diagnostics["reason"])
# not_certified no certificate from causal, gflow; this does not prove nondeterminism
```

### Causal flow

| Field | Value |
|---|---|
| Description | Finds the maximally delayed causal flow (Danos and Kashefi, PRA 74, 052310 (2006)) with the Mhalla-Perdrix algorithm (ICALP 2008), which is complete; `verify_causal_flow` checks (F1)-(F3), domain, codomain and injectivity directly. |
| Python API | `photographiq.flows.find_causal_flow(og) -> FlowAnalysisResult` (`certificate['flow']`: node -> f(node)); `photographiq.flows.verify_causal_flow(og, flow, layers) -> list[str]` (empty list means valid) |
| Implementation | `src/photographiq/flows/causal.py`, `src/photographiq/flows/_common.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU only (pure Python) |
| Assumptions | `Model.QUBIT` only; XY-plane measurements only (non-XY nodes give `not_certified` with `complete=True`). Ideal qubits and a noiseless CZ graph state. Missing measurement specs raise `ValueError`. O(n m). |
| Tests | `tests/flow/test_flows_qubit.py::test_existence_agrees_with_graphix` (Graphix `to_causalflow_or_none`, 150 random graphs), `::test_random_flows_verify_and_are_deterministic_by_statevector` (independent NumPy statevector simulation of every branch with reference qubits; overlap and probability errors < 1e-12), `::test_line_has_causal_flow_with_textbook_flow_function_and_is_deterministic`, `::test_rectangular_cluster_has_causal_flow`, `::test_dropping_corrections_breaks_determinism`, `::test_verifiers_reject_corrupted_certificates` |
| Status | **Implemented and validated** |

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, QubitMeasurement
from photographiq import flows
og = OpenGraph(Model.QUBIT, nx.path_graph(4), (0,), (3,), {n: QubitMeasurement("XY", 0.3) for n in range(3)})
r = flows.find_causal_flow(og)
print(r.status, r.certificate["flow"], r.layers, r.correction_map[1])
# certified {2: 3, 1: 2, 0: 1} ((0,), (1,), (2,)) {'x': frozenset({2}), 'z': frozenset({3})}
print(flows.verify_causal_flow(og, r.certificate["flow"], r.layers))
# []
```

### Generalised flow (gflow)

| Field | Value |
|---|---|
| Description | Finds the maximally delayed plane-aware gflow (Browne, Kashefi, Mhalla, Perdrix, NJP 9, 250 (2007); XY/XZ/YZ form of Backens et al., Quantum 5, 421 (2021)) by a backward layered GF(2) search, which is complete. `verify_gflow` checks (G1)-(G3) with plain set operations. |
| Python API | `photographiq.flows.find_gflow(og) -> FlowAnalysisResult` (`certificate['correction_sets']`: node -> g(node)); `photographiq.flows.verify_gflow(og, sets, layers) -> list[str]`; internals `flows.gflow.solve_gf2(matrix, rhs)`, `flows.gflow.layered_search(og, labels)` |
| Implementation | `src/photographiq/flows/gflow.py`, `src/photographiq/flows/_common.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU (NumPy GF(2) elimination) |
| Assumptions | `Model.QUBIT` only; planar measurements; ideal qubits. `correction_map` lists X corrections on `g(u) \ {u}` and Z corrections on `Odd(g(u)) \ {u}`, restricted to later nodes and outputs. Worst case O(n^5). |
| Tests | `tests/flow/test_flows_qubit.py::test_existence_agrees_with_graphix` (Graphix `to_gflow_or_none`; our verifier also accepts Graphix's certificates, and our layering is never deeper), `::test_random_flows_verify_and_are_deterministic_by_statevector` (independent statevector), `::test_gflow_without_causal_flow_xy_only`, `::test_yz_and_xz_planes_need_gflow`; `tests/calculus/test_calculus_qubit.py::test_pattern_from_flow_accepts_flow_analysis_results` (brute-force gflow) |
| Status | **Implemented and validated** |

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, QubitMeasurement
from photographiq import flows
g = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 3), (2, 4), (3, 4)])
og = OpenGraph(Model.QUBIT, g, (3, 2), (0, 4), {n: QubitMeasurement("XY", 0.5) for n in (1, 2, 3)})
print(flows.find_causal_flow(og).status, flows.find_gflow(og).status)
# not_certified certified
r = flows.analyze(og)
print(r.algorithm.split()[0], r.layers, flows.verify_gflow(og, r.certificate["correction_sets"], r.layers))
# gflow ((2, 3), (1,)) []
```

### Command DAG scheduling (`photographiq.flow`)

| Field | Value |
|---|---|
| Description | Builds the dependency DAG of a CV command list (mode lifetimes and classical record dependencies) and returns a topological order of command indices. This is causal scheduling only: a DAG is not a flow or determinism certificate. |
| Python API | `photographiq.flow.dependency_graph(commands) -> nx.DiGraph`; `photographiq.flow.topological_schedule(commands) -> tuple`; used by `Pattern.dependencies()` and `Pattern.schedule()` |
| Implementation | `src/photographiq/flow.py`, `src/photographiq/pattern.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU only (NetworkX) |
| Assumptions | Duplicate classical keys, keys with no producer and cyclic dependencies raise `ValueError`. Lexicographic topological sort. |
| Tests | `tests/graphix_reference/test_labelled_structures.py::test_multiple_adaptive_domains_corrections_and_causal_order` (classical dependencies equal the s/t domains of the equivalent Graphix pattern; hand-derived DAG edges; the schedule respects every edge), `tests/graphix_reference/test_structure.py::test_wire_connectivity_io_measurement_order_and_domains`, `tests/unit/test_core.py::test_future_and_cycle_and_missing_modes` (cyclic dependencies rejected) |
| Status | **Implemented and validated** |

```python
import photographiq as pg
from photographiq.flow import topological_schedule
pattern = pg.Pattern(pg.CVGraph.line(3, squeezing=0.8, inputs=(0,), outputs=(2,)))
pattern.measure(0); pattern.measure(1)
print(topological_schedule(pattern.commands))
# (0, 1, 2, 3, 4, 5)
```

## 25. Pauli flow

Pauli flow extends gflow so that Pauli-measured nodes may be corrected out of order. A node counts as a Pauli measurement only when `QubitMeasurement.pauli` is set; the plane and angle must then describe that Pauli observable.

### Pauli flow finder and verifier

| Field | Value |
|---|---|
| Description | Finds the maximally delayed Pauli flow (Simmons, EPTCS 343 (2021) 50, Def. 2.4, equivalent to Mitosek-Backens 2024) with the layered GF(2) search plus the P1-P3 relaxations; complete for Pauli-flow existence. `verify_pauli_flow` checks (P1)-(P9) directly. `pauli_label` returns the X/Y/Z label after checking it against plane and angle, or the plane name when `pauli` is `None`. |
| Python API | `photographiq.flows.find_pauli_flow(og) -> FlowAnalysisResult` (`certificate['correction_sets']`, `certificate['labels']`); `photographiq.flows.verify_pauli_flow(og, sets, layers) -> list[str]`; `photographiq.flows.pauli_label(m, *, atol=1e-9) -> str` |
| Implementation | `src/photographiq/flows/pauli.py`, `src/photographiq/flows/gflow.py` (`layered_search`) |
| Backends | Backend-independent |
| CPU / GPU | CPU (NumPy GF(2) elimination) |
| Assumptions | `Model.QUBIT` only; ideal qubits. A `pauli` tag that disagrees with plane and angle raises `ValueError` ("disagrees"). Corrections on earlier-measured Pauli nodes act as phases and are omitted from `correction_map`. Nodes in one layer may correct each other (X/Y nodes). |
| Tests | `tests/flow/test_flows_qubit.py::test_existence_agrees_with_graphix` (Graphix `to_pauliflow_or_none`, 150 random graphs; our verifier accepts Graphix's certificates), `::test_random_flows_verify_and_are_deterministic_by_statevector` (independent statevector), `::test_pauli_flow_needed_for_x_measurements_in_xz_plane` (Graphix documentation example), `::test_pauli_flow_with_y_and_z_measurements_is_deterministic`, `::test_model_and_input_validation` |
| Status | **Implemented and validated** |

```python
import math, networkx as nx
from photographiq.ir import Model, OpenGraph, QubitMeasurement
from photographiq import flows
xm = QubitMeasurement("XZ", math.pi / 2, "X")  # X Pauli written in the XZ plane
og = OpenGraph(Model.QUBIT, nx.path_graph(3), (0,), (2,), {0: xm, 1: xm})
print(flows.pauli_label(xm), flows.find_gflow(og).status)
# X not_certified
r = flows.analyze(og)
print(r.algorithm.split()[0], r.certificate["correction_sets"], r.layers)
# pauli-flow {0: frozenset({1}), 1: frozenset({2})} ((0, 1),)
```

### Pauli flow to `QubitPattern`

| Field | Value |
|---|---|
| Description | `pattern_from_flow` accepts a certified Pauli-flow result and builds a standard pattern from its `correction_map`. |
| Python API | `photographiq.calculus.pattern_from_flow(og, find_pauli_flow(og))` |
| Implementation | `src/photographiq/calculus/qubit.py`, `src/photographiq/flows/_common.py` (`correction_map`) |
| Backends | Backend-independent |
| CPU / GPU | CPU (NumPy) |
| Assumptions | The `pauli` tag is ignored by `pattern_from_flow`; plane and angle define the measurement. Pauli flow may place `X_{p(u)}` / `Z_{Odd(p(u))}` components on earlier-measured qubits. Conditions P1-P3 allow this only where the component commutes with that qubit's Pauli measurement (X on X, Z on Z, XZ ~ Y on Y), so the component only changes a branch's global phase. `_common.correction_map` drops these components, so a `FlowAnalysisResult` from Pauli flow is accepted. A raw Pauli-flow table that keeps them raises `ValueError` ("... earlier ..."), as for any supplied table. The `pattern_from_flow` docstring now states both cases; the code was already correct and is unchanged. Components on same-layer Pauli qubits that come later in the flattened order are kept as domains and are harmless. The result is not assumed deterministic; check it with `is_deterministic`. Evidence from the fixer and an independent reviewer: on 200 random graphs (seed 7) Pauli flow was certified on 63, and on 400 more (seed 123) on 36. Every resulting pattern is strongly deterministic, and every one that also has gflow is `equivalent` to the gflow pattern. |
| Tests | `tests/calculus/test_calculus_qubit.py::test_pattern_from_pauli_flow_results_is_deterministic` (200 random graphs, about half of the non-input nodes Pauli-measured: raw tables with earlier components raise `ValueError` and those targets are Pauli-measured; every pattern is deterministic; equivalence with the gflow pattern where gflow exists; negative control: dropping one outcome's output corrections breaks determinism; requires at least 40 certified, 10 with earlier components and 10 with Pauli flow only). The correction maps are also checked by an independent statevector simulation in `tests/flow/test_flows_qubit.py::test_random_flows_verify_and_are_deterministic_by_statevector`. |
| Status | **Implemented and validated** |

```python
import math, networkx as nx
from photographiq.ir import Model, OpenGraph, QubitMeasurement
from photographiq import flows
from photographiq.calculus import pattern_from_flow, is_deterministic
xm = QubitMeasurement("XZ", math.pi / 2, "X")
og = OpenGraph(Model.QUBIT, nx.path_graph(3), (0,), (2,), {0: xm, 1: xm})
r = flows.find_pauli_flow(og)
print(r.correction_map[0], is_deterministic(pattern_from_flow(og, r)).deterministic)
# {'x': frozenset({1}), 'z': frozenset({2})} True
```

## 26. CV determinism

For `Model.CV_GAUSSIAN` graphs, determinism means the real-linear CV flow of Booth and Markham (Quantum 7, 1146 (2023), Defs. 5-6 and Lemma 7) on the weighted adjacency, with homodyne measurements and displacement feed-forward. The qubit GF(2) rules are never used as a CV certificate. CV determinism holds only in the infinite-squeezing limit; at finite squeezing the package reports the added noise and the remaining outcome dependence. Non-Gaussian and GKP-logical graphs are not analysed. The restricted Gaussian graphical calculus (`GaussianGraph`, Section 22) describes how measurements rewrite a pure Gaussian graph state, but it does not decide determinism; that is the job of CV flow.

Since 2026-11-12 every CV result is classified in `diagnostics['decision']` as **certified** (with the regimes it covers in `certificate['regimes']`: `ideal_limit`, `finite_squeezing` or both), **impossible** (proof in `diagnostics['proof']`: a combinatorial obstruction, an exact rank test with rational witnesses in `diagnostics['witnesses']`, or a float64 backward-error statement) or **numerically_uncertain**. The statuses are unchanged except one behaviour change: a float64 rank decision that falls in the uncertain band now gives `unknown`, where it previously gave `not_certified`. On 600 adversarial graphs the defaults give 150 certified, 408 impossible and 42 unknown; in that run `exact=True` certified 41 of the 42 (27 for the ideal limit only, 14 in both regimes). The remaining dynamic-range case is resolved by a rule added afterwards and covered by a regression test; the 600-graph run was not repeated after that change. Evidence: [CV-flow robustness](validation/cv-flow-robustness.md).

### `FlowAnalysisResult` (fields and statuses)

| Field | Value |
|---|---|
| Description | Common frozen result of every flow and determinism analysis. Fields: `status`, `model`, `algorithm`, `measurement_order` (total order, earliest first), `layers` (partial order as layers of measured nodes; outputs are not listed), `correction_map`, `certificate`, `assumptions`, `diagnostics`; property `certified`. Statuses: `certified` (certificate found and independently verified), `not_certified` (no certificate for the named flow notion; `diagnostics['complete']` says whether the finder is complete (all finders here report `True`; the CV search is complete in exact arithmetic), and this is never a proof of nondeterminism), `unknown` (undecided, e.g. an uncertain float64 rank decision, or the numerical check failed or could not run), `unsupported` (model/algorithm outside scope). CV results also carry `diagnostics['decision']` (`certified`, `impossible`, `numerically_uncertain`), `diagnostics['proof']` for impossible results, `diagnostics['rank_decisions']` (rule, uncertain band, margin in decades, `decided_by`) and `certificate['regimes']` for certified ones. |
| Python API | `photographiq.flows.FlowAnalysisResult(status, model, algorithm, measurement_order=(), layers=(), correction_map={}, certificate={}, assumptions=(), diagnostics={})`; `.certified`; `photographiq.flows.result.STATUSES`; `photographiq.flows.result.unsupported(model, algorithm, reason)` |
| Implementation | `src/photographiq/flows/result.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU only (pure Python) |
| Assumptions | Any other status raises `ValueError`. Qubit `correction_map`: `{node: {'x': frozenset, 'z': frozenset}}`. CV `correction_map`: `{node: {target: {'q': cq, 'p': cp}}}`, meaning `q += cq*m`, `p += cp*m` with `m = record / sin(theta)`. |
| Tests | `tests/flow/test_flows_cv.py::test_random_weighted_graphs_existence_matches_brute_force_over_orders` (`certified` iff a brute-force search over all orders finds a certificate), `::test_weights_decide_cv_flow_on_the_same_skeleton`, `::test_corrections_respect_layers_and_large_squeezing_is_undecided_not_an_error` (`unknown`), `::test_large_squeezing_is_certified_or_undecided_never_failed` (`certified` or `unknown`, never a failed check); `tests/flow/test_flows_cv_robustness.py` (25 tests: decisions, proofs, witnesses and regimes on the graph set of [CV-flow robustness](validation/cv-flow-robustness.md)); `tests/flow/test_flows_qubit.py::test_existence_agrees_with_graphix`, `::test_no_flow_is_not_reported_as_nondeterminism`, `::test_model_and_input_validation` (`unsupported`) |
| Status | **Implemented and validated** |

Example: see the `find_cv_flow` example below.

### CV flow search and certification (`find_cv_flow`)

| Field | Value |
|---|---|
| Description | Searches an admissible measurement order by backward real-linear layering, re-certifies it with the independent supplied-order checker `certify_cv_flow`, builds the corrected finite-squeezing pattern from exactly the reported correction map, and (by default) verifies determinism numerically. Also reached through `analyze(og)` or `analyze(og, "cv", ...)`. |
| Python API | `photographiq.flows.find_cv_flow(og, *, tolerance=1e-10, verify=True, seed=0, verify_tolerance=1e-9, shift=3.0, exact=None) -> FlowAnalysisResult`; `analyze(og, shift=..., exact=...)` forwards both; `certificate` keys `total_order` (a `cvflow.CVFlow`), `layer_corrections`, `shears`, `checks`, `determinism` (the `verify_cv_determinism` report), `regimes`; `diagnostics` keys `decision`, `proof`, `witnesses`, `rank_decisions`, `correction_gain_max` |
| Implementation | `src/photographiq/flows/cv.py`, `src/photographiq/flows/_exact.py` (exact rational re-check), `src/photographiq/cvflow.py` |
| Backends | Order search and certificate: backend-independent (NumPy SVD, `fractions.Fraction` for the exact path). Verification: `gaussian_channel` and `simulate(backend="gaussian")` |
| CPU / GPU | CPU (NumPy; pure Python for the exact path) |
| Assumptions | Homodyne measurements at angle `atan2(1, shear)` on a weighted graph; unit-efficiency detection; missing resource squeezing defaults to r = 1.0. Rank decisions are made on the power-of-two-equilibrated, normalised correction matrix: a singular value (or appended-row quantity) at most `max(dims) * eps` is zero, at least `sqrt(eps)` is nonzero, and anything in between is uncertain; `tolerance` is kept only for backward compatibility. `exact=None` (default) redoes the search in exact rational arithmetic on graphs of at most 40 nodes (`EXACT_NODE_LIMIT`) when a decision is uncertain or no flow is found; `exact=True` makes every decision exact at any size and also runs `exact_ideal_limit` (next entry); `exact=False` is float64 only. `shift` is the squeezing increment of the float64 determinism check (levels r, r + shift, r + 2 shift); keep r + 2 shift below about 9. Nodes in one layer do not correct each other. `certified` requires the algebraic certificate and, if `verify=True`, the float64 or exact ideal-limit check; `regimes['finite_squeezing']` is claimed only from a float64 check that ran at that squeezing. A failed or impossible numerical check gives `unknown`, never `certified`, and so does an uncertain rank decision that was not re-checked exactly (before 2026-11-12 it gave `not_certified`). Corrections with gains above 1/sqrt(eps) are reported in `diagnostics['correction_gain_max']` with a `diagnostics['warning']` entry; only the ideal-limit statement is then meaningful. The following measurements at the default `shift=3.0` predate the new options. The numerical check runs only at moderate resource squeezing. On the four-node unit-weight wire of the example (shift 3), re-run after the fix, every r from 2.5 to 3.4 in steps of 0.1 is certified, and so is r = 3.3 + 1e-12. Every r from 3.5 to 4.5, and r = 3.999999999999, 4.000000000000002, 5 and 6, gives `unknown` (check could not run: the raised-squeezing covariances fail the absolute -1e-7 tolerance of `GaussianState.validate` in float64). Before the fix r = 3.3 gave `unknown` because the numerical check failed. The fixer's sweep on the wire without shears had the same false negative at r = 3.1. The cause was a fixed absolute bound on the law-of-total-covariance check, now replaced (see `verify_cv_determinism`). Near r = 3.5 to 4.5 the outcome can flip between certified and could-not-run for tiny changes of r (for example, without shears, 4.0 against 4.000000000000002); either result is safe. On random weighted graphs the could-not-run regime was seen from about r = 2.9. Treat `unknown` at high squeezing as undecided. `verify=False` adds the assumption "numerical determinism verification skipped". Missing `HomodyneSpec` or a non-positive `tolerance` raises `ValueError`. O(n^4). |
| Tests | `tests/flow/test_flows_cv.py::test_wire_order_search_matches_teleportation_closed_form` (closed form `gaussian.wire_channel`, atol 1e-12), `::test_weighted_two_node_correction_closed_form` (closed form `[[0, -1/w], [w, 0]]`), `::test_multimode_graph_teleportation_matches_closed_form` (Booth-Markham closed-form `C_Z(A) (+) F D(s)`), `::test_random_weighted_graphs_existence_matches_brute_force_over_orders` (brute force over all orders), `::test_cv_flow_and_qubit_gflow_are_different_notions`, `::test_found_order_is_admissible_where_supplied_reverse_order_is_not`; `tests/flow/test_flows_cv_robustness.py` (`test_impossible_graphs_carry_exact_witnesses` (witnesses re-checked independently with `Fraction`), `test_combinatorial_obstruction`, `test_near_singular_weights_are_uncertain_or_resolved_exactly`, `test_rounding_level_determinant_float_and_exact_answers_differ_and_are_labelled`, `test_extreme_weights_float_check_cannot_run_exact_check_certifies`, `test_long_wires_and_grids_certified_in_both_regimes`, `test_large_squeezing_certified_with_smaller_shift` (unit wire r = 3.5 to 4.5 with `shift=1.5`, kappa = 1 +- 0.05), `test_very_large_squeezing_ideal_limit_only_by_exact_check`, `test_random_graphs_exact_and_float_agree`, `test_float64_false_negatives_resolved_by_exact_check`, `test_failed_float_checks_are_named_and_never_overruled_for_extra_directions`) |
| Status | **Implemented and validated** |

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, HomodyneSpec
from photographiq import flows
g = nx.Graph(); g.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
og = OpenGraph(Model.CV_GAUSSIAN, g, (0,), (3,),
               {0: HomodyneSpec(0.2), 1: HomodyneSpec(-0.4), 2: HomodyneSpec(0.7)}, {n: 0.6 for n in g})
r = flows.analyze(og)
print(r.status, r.layers, r.correction_map[0])
# certified ((0,), (1,), (2,)) {1: {'q': -1.0, 'p': -0.0}, 2: {'q': -0.0, 'p': -1.0}}
print(flows.analyze(og, "gflow").status)
# unsupported
```

### Exact CV-flow re-check and exact ideal-limit check (`exact=`, `exact_ideal_limit`)

| Field | Value |
|---|---|
| Description | Separates three outcomes that float64 alone cannot keep apart: a flow that is impossible, a decision float64 cannot make, and a certificate with the regime it covers. The exact path repeats the layered search in rational arithmetic (`fractions.Fraction`; every float64 weight is a dyadic rational, so the conversion is exact and the result is a proof for the given weights) and returns rational witnesses y with y^T A[W, C] = 0 and y_u != 0 when a node cannot be corrected. `exact_ideal_limit` propagates linear Wigner variables exactly (inputs; anti-squeezed q and squeezed p per resource node), applies the CZ gates, records and corrections, and checks the flow condition, rank L_Q = abs(O) - abs(I), full rank of the record rows R_Q, and S Omega S^T = Omega when abs(O) = abs(I). These conditions imply every criterion of `verify_cv_determinism` and do not depend on r, so they settle the ideal limit at any squeezing. |
| Python API | `photographiq.flows.find_cv_flow(og, exact=None \| True \| False, shift=3.0)`, `photographiq.flows.analyze(og, exact=..., shift=...)`; `photographiq.flows.exact_ideal_limit(og, layers, exact_vectors) -> dict`; `photographiq.flows.cv.EXACT_NODE_LIMIT` (40) |
| Implementation | `src/photographiq/flows/_exact.py`, `src/photographiq/flows/cv.py` (`_search`, `exact_ideal_limit`) |
| Backends | Backend-independent (no simulation) |
| CPU / GPU | CPU only (pure Python rational arithmetic) |
| Assumptions | The exact ideal-limit conditions are sufficient, not necessary: a failure means "not shown". The exact check overrules a float64 check only when the float64 check failed `decay` alone or counted fewer singular directions than expected (the dynamic-range artifact); more singular directions (missing or wrong corrections) are never overruled, and any other disagreement gives `unknown`. The exact certificate covers the exact rational correction map; the reported float `correction_map` is its rounding. `finite_squeezing` is never certified exactly; it remains a float64 statement at the given r. A float64 "impossible" on a graph above 40 nodes, or with `exact=False`, is a backward-error statement (weights perturbed by at most `max(dims) * eps` admit no flow), not an exact proof. The exact path costs about O(n^4) rational operations on growing integers (seconds for a 31-node wire) and is not meant for hundreds of nodes. Unit-efficiency detection and lossless resources are assumed. A smaller `shift` reaches larger r in the float64 check (`shift=1.5` certifies the unit wire up to r = 4.5 in the tests) but measures the decay exponent over a shorter range. |
| Tests | `tests/flow/test_flows_cv_robustness.py::test_exact_check_matches_closed_form_and_rejects_wrong_corrections` (reproduces the closed-form wire map; rejects every scaled correction vector, including a factor 1 + 2^-60), `::test_impossible_graphs_carry_exact_witnesses`, `::test_very_large_squeezing_ideal_limit_only_by_exact_check` (unit wire at r = 8), `::test_random_graphs_exact_and_float_agree`, `::test_float64_false_negatives_resolved_by_exact_check`, `::test_failed_float_checks_are_named_and_never_overruled_for_extra_directions` |
| Status | **Implemented and validated** |

```python
import networkx as nx
from photographiq import flows
from photographiq.ir import HomodyneSpec, Model, OpenGraph
def cv_graph(edges, inputs, outputs, squeezing=0.8):
    g = nx.Graph(); g.add_weighted_edges_from(edges)
    return OpenGraph(Model.CV_GAUSSIAN, g, inputs, outputs,
                     {n: HomodyneSpec() for n in g if n not in outputs}, {n: squeezing for n in g})
def bridge(d):   # inputs u, v; outputs a, b; correction matrix [[1, 1], [1, 1 + d]], det = d
    return cv_graph([("u", "a", 1.0), ("u", "b", 1.0), ("v", "a", 1.0), ("v", "b", 1.0 + d)], ("u", "v"), ("a", "b"))
r = flows.analyze(bridge(0.0))
print(r.status, r.diagnostics["decision"], r.diagnostics["witnesses"])
for exact in (None, True):
    r = flows.analyze(bridge(1e-10), exact=exact)
    print(exact, r.status, r.diagnostics["decision"], sorted(r.certificate.get("regimes", {})))
print(f"{r.diagnostics['correction_gain_max']:.1e}", "warning" in r.diagnostics)
w = cv_graph([(i, i + 1, 1.0) for i in range(4)], (0,), (4,), squeezing=4.0)
for shift in (3.0, 1.5):
    r = flows.find_cv_flow(w, shift=shift)
    print(shift, r.status, sorted(r.certificate.get("regimes", {})))
# not_certified impossible {'u': {'u': Fraction(-1, 2), 'v': Fraction(1, 2)}, 'v': {'u': Fraction(-1, 2), 'v': Fraction(1, 2)}}
# None unknown numerically_uncertain []
# True certified certified ['ideal_limit']
# 1.0e+10 True
# 3.0 unknown []
# 1.5 certified ['finite_squeezing', 'ideal_limit']
```

The singular bridge is provably impossible (rational witnesses). The bridge with determinant 1e-10 is `unknown` by default (the exact re-check finds the flow, but the float64 re-certification fails) and certified for the ideal limit only with `exact=True`; its correction gain of about 1/det is reported with a `diagnostics['warning']` entry (not a Python warning). The unit wire at r = 4 cannot be checked in float64 with the default `shift=3.0` and is certified in both regimes with `shift=1.5`.

### CV order search (`search_cv_order`)

| Field | Value |
|---|---|
| Description | The backward layered search alone: a node joins the next-earlier layer when `A[W, C] c = e_u` is solvable (A the weighted adjacency, W the unsolved nodes, C the solved non-input nodes). Since 2026-11-12 solvability is decided by the float64 rank test with margins of `find_cv_flow` (relative thresholds on the equilibrated matrix), not by `tolerance`, which is kept for backward compatibility; this function does not re-check uncertain decisions (use `find_cv_flow`). |
| Python API | `photographiq.flows.search_cv_order(og, *, tolerance=1e-10) -> (layers, vectors, unsolved, max_residual)` |
| Implementation | `src/photographiq/flows/cv.py` |
| Backends | Backend-independent |
| CPU / GPU | CPU (NumPy) |
| Assumptions | Does not verify anything; an empty `unsolved` means an admissible order was found. Graph inputs are never correction columns. |
| Tests | `tests/flow/test_flows_cv.py::test_random_weighted_graphs_existence_matches_brute_force_over_orders` (through `analyze(verify=False)`; brute force over all orders with `certify_cv_flow`) |
| Status | **Implemented and validated** |

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, HomodyneSpec
from photographiq import flows
g = nx.Graph(); g.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
og = OpenGraph(Model.CV_GAUSSIAN, g, (0,), (3,), {n: HomodyneSpec() for n in range(3)}, {n: 0.6 for n in g})
layers, vectors, unsolved, worst = flows.search_cv_order(og)
print(layers, unsolved)
# ((0,), (1,), (2,)) ()
```

### Numerical determinism verification (`verify_cv_determinism`)

| Field | Value |
|---|---|
| Description | Checks numerically that a corrected Gaussian pattern is deterministic in the infinite-squeezing limit. It raises every resource squeezing by x in (0, `shift`, 2 `shift`), computes the outcome-averaged channel (S, N), and obtains conditional output moments by exact Gaussian postselection on fixed records. The checks are: S independent of r; N fits `e^{2x} N_plus + e^{-2x} N_minus`; conditional covariance independent of the records; law of total covariance; outcome gain and conditional map error decaying like `exp(-2 kappa x)` with kappa >= `min_exponent`; S symplectic when square; `rank(N_plus) = \|outputs\| - \|inputs\|`. |
| Python API | `photographiq.flows.verify_cv_determinism(pattern, *, shift=3.0, seed=0, tolerance=1e-9, min_exponent=0.9, probes=2) -> dict` (keys include `ideal_limit_deterministic`, `matrix`, `noise`, `symplectic_residual`, `finite_squeezing_noise`, `ideal_singular_noise`, `ideal_singular_directions`, `expected_singular_directions`, `finite_squeezing_gain`, `finite_squeezing_conditional_map_error`, `decay_exponents`, `levels`, `tolerance`); helpers `flows.resource_graph(og)`, `flows.cv.corrected_pattern(graph, order, corrections, shears)` |
| Implementation | `src/photographiq/flows/cv.py` |
| Backends | `gaussian` (NumPy) through `simulate(..., backend="gaussian", measurement_outcomes=...)` and `analysis.gaussian_channel` |
| CPU / GPU | CPU (NumPy) |
| Assumptions | hbar = 2, vacuum covariance I. The postselected runs are a numerical certificate only; they are not deterministic execution. At finite r the conditional output still depends on the records (reported as `finite_squeezing_gain`), and the added noise is reported separately from the ideal map. Directions whose variance diverges as r -> infinity are projected out before the outcome dependence is assessed. Loss or detector inefficiency would break the two-term noise model and fail the check. Probes are random directions from `seed`. The law-of-total-covariance check is `spread_min >= -(scale + spread_rounding)` with `scale = tolerance * max(magnitude, abs(N))`. The term `spread_rounding = len(commands) * dim(total) * eps * peak`, reported per level as `levels[k]['spread_rounding']`, uses `peak`, the largest absolute covariance entry of any intermediate state in the conditional runs. A `GaussianBackend` subclass records `peak` without changing the arithmetic. The old fixed absolute bound (-1e-9) caused false negatives at r of about 3.1 to 3.3, where the quantity's float64 rounding is about eps x e^{2(r + 2 shift)}. `spread_rounding` is a heuristic first-order estimate, not a rigorous bound. On the wire the true error is 0.004 to 0.03 of it (against a 90-digit mpmath model). On adversarial graphs the error reached about 1670 times it, through anti-squeezed output directions and ill-conditioned Schur complements, but the `scale` term absorbed it in every case: `spread_min` never fell below -0.023 (scale + allowance) on 300 graphs, and there was no false negative on 600 more. Loosening this check cannot certify a wrong pattern, because the identity holds for every Gaussian pattern. Nondeterminism is caught by the unchanged decay-exponent and singular-direction checks: wrong corrections (p or q scaled by 0, 0.5, 1 + 1e-3 or 1 + 1e-6, q and p swapped) are still rejected at r = 3.1, 3.3 and 3.45. Example allowances on the wire with shears 0.2, -0.4, 0.7 and r = 0.6 are 4.3e-14, 1.7e-11 and 6.7e-9 at the three levels; at r + 6 of about 9.3 the allowance is about 1.5e-6, so the check is nearly blind there. The other checks still use absolute float64 tolerances, and `GaussianState.validate` (absolute -1e-7) makes the raised levels fail above about r = 3.5. There the result is `unknown` ("could not run"), and near the threshold it can flip with tiny changes of r (see `find_cv_flow`). Since 2026-11-12, `find_cv_flow` and `analyze` forward `shift` to this check, and `exact=True` replaces it by the exact ideal-limit check, which does not depend on r (previous entries). A few adversarial graphs (7 of 600, for example r = 2.49 with abs(S) = 283) are certified algebraically but give a decay exponent of about 0.65, so the result is `unknown`. These are slow-decay false negatives of the decay check. The `covariance_outcome_dependence` check measures exactly 0 and needs no allowance. |
| Tests | `tests/flow/test_flows_cv.py::test_wire_order_search_matches_teleportation_closed_form` (matrix and noise against the closed form `gaussian.wire_channel`, atol 1e-12; decay exponent within 0.05 of kappa = 1; finite-squeezing gain > 0.1 reported), `::test_broken_feed_forward_fails_ideal_limit_check` (four broken feed-forward variants rejected), `::test_random_certified_graphs_pass_ideal_limit_verification`, `::test_corrections_respect_layers_and_large_squeezing_is_undecided_not_an_error` (r in {6, 6 + 1e-12, 7.3, 8, 10}, where e^{-2(r+6)} is below eps x e^{2(r+6)}: `unknown`, "could not run", a recorded `ValueError`), `::test_large_squeezing_is_certified_or_undecided_never_failed` (r = 3.0 to 4.5 plus 3.3 + 1e-12, 4.0 and 4.000000000000002: certified with exponent about 1 or "could not run", never a failed check; r < 3.45 must be certified; fails on the old code), `::test_wrong_correction_is_still_rejected_at_large_squeezing` (r = 3.1 and 3.3, q corrections scaled by 0, 0.5 and 1 + 1e-3: not deterministic, extra singular directions). `::test_conditional_simulation_agrees_with_unconditional_channel_on_average` (4000 sampled-outcome trajectories against `gaussian_channel`) checks the conditional Gaussian simulation that the verifier relies on, not the verifier itself. |
| Status | **Implemented and validated** |

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, HomodyneSpec
from photographiq import flows
from photographiq.cvflow import flow_pattern
g = nx.Graph(); g.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
og = OpenGraph(Model.CV_GAUSSIAN, g, (0,), (3,), {n: HomodyneSpec() for n in range(3)}, {n: 0.6 for n in g})
rep = flows.verify_cv_determinism(flow_pattern(flows.resource_graph(og), (0, 1, 2), shears={0: 0.2, 1: -0.4, 2: 0.7}))
print(rep["ideal_limit_deterministic"], {k: round(v, 3) for k, v in rep["decay_exponents"].items()}, rep["finite_squeezing_gain"] > 0.1)
# True {'gain': 0.999, 'conditional_map_error': 0.999} True
```

### Supplied-order CV flow certificate (`certify_cv_flow`, `flow_pattern`)

| Field | Value |
|---|---|
| Description | Checks the Booth-Markham correction-matrix condition for one given total order and returns the correction vectors with residuals (`CVFlow`), or `None`. `flow_pattern` builds the finite-squeezing protocol for a certified order. It does not search orders. |
| Python API | `photographiq.cvflow.certify_cv_flow(graph, order, *, parameters=None, tolerance=1e-10) -> CVFlow \| None`; `photographiq.cvflow.flow_pattern(graph, order, *, shears=None, parameters=None) -> Pattern`; `cvflow.CVFlow(order, corrections, residuals)` |
| Implementation | `src/photographiq/cvflow.py` |
| Backends | Certificate: backend-independent. The returned `Pattern` runs on any backend that supports its commands. |
| CPU / GPU | CPU (NumPy) |
| Assumptions | Works on a `CVGraph`, not an `OpenGraph` (use `flows.resource_graph(og)`). The order must list each non-output node once (`ValueError`). `flow_pattern` raises `ValueError` if the order has no certificate and freezes the numerical edge weights. The certificate concerns the ideal limit, not finite-r unitarity. |
| Tests | `tests/analytical/test_cvflow.py::test_weighted_real_flow_certificate` (closed form), `::test_flow_pattern_and_streaming_wire_have_same_channel` (`protocols.wire`), `::test_no_flow_for_isolated_measured_node`; `tests/graphix_reference/test_flow_structure.py::test_line_correction_support_agrees` (correction support against Graphix causal flow) |
| Status | **Implemented and validated** |

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, HomodyneSpec
from photographiq import flows
from photographiq.cvflow import certify_cv_flow
g = nx.Graph(); g.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
og = OpenGraph(Model.CV_GAUSSIAN, g, (0,), (3,), {n: HomodyneSpec() for n in range(3)}, {n: 0.6 for n in g})
graph = flows.resource_graph(og)
print(certify_cv_flow(graph, (2, 1, 0)), certify_cv_flow(graph, (0, 1, 2)).residuals)
# None (0.0, 0.0, 0.0)
```

### Determinism of non-Gaussian and GKP-logical graphs

| Field | Value |
|---|---|
| Description | No determinism analysis exists for `Model.CV_NONGAUSSIAN` or `Model.GKP_LOGICAL`: displacement feed-forward does not commute with non-Gaussian elements, and GKP logical flows would need a finite-GKP error model. |
| Python API | `photographiq.flows.analyze(og)` and `find_cv_flow(og)` return status `unsupported` |
| Implementation | `src/photographiq/flows/__init__.py`, `src/photographiq/flows/cv.py` |
| Backends | None |
| CPU / GPU | Not applicable |
| Assumptions | No exception is raised; the result has `status == "unsupported"` and a reason in `diagnostics['reason']`. |
| Tests | `tests/flow/test_flows_qubit.py::test_model_and_input_validation` |
| Status | **Unsupported** |

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, HomodyneSpec
from photographiq import flows
print(flows.analyze(OpenGraph(Model.CV_NONGAUSSIAN, nx.path_graph(2), (0,), (1,), {0: HomodyneSpec()})).status)
# unsupported
```

## 27. GKP operations

PhotoGraphiQ implements finite-energy square-lattice GKP resources in a truncated Fock basis, physical X/Z readout by homodyne detection plus classical decoding, Gaussian realizations of ideal-lattice logical gates, and finite-dimensional measurement instruments. The project classifies this layer as experimental ([feature matrix](validation/feature-matrix.md)). It does not implement a fault-tolerant architecture, a threshold study, physical Y or arbitrary XY measurement, or magic-state injection. All conventions use hbar=2 and [q,p]=2i, so the logical shift spacing is L = sqrt(2*pi) (`photographiq.gkp.SPACING`) and the stabilizer translations are 2L.

### Finite-energy GKP resource (`GKPResource`)

| Field | Value |
|---|---|
| Description | A Gaussian comb of peaks at q = L(2k + bit), each with standard deviation `peak_width`, weighted by a Gaussian envelope `exp(-q_k^2 envelope^2 / 4)`. It is evaluated on an explicit q grid and projected onto the Fock basis by Simpson quadrature against Hermite functions. |
| Python API | `photographiq.GKPResource(logical=0, peak_width=0.4, envelope=0.4, peaks=8, grid_points=4097)`; `.wavefunction()`; `.project(cutoff) -> (FockInput, captured_weight)`; `.fock(cutoff)`; `photographiq.gkp.superposition(alpha, beta, *, cutoff, **resource_options)` |
| Implementation | `src/photographiq/gkp.py` |
| Backends | Backend-independent NumPy/SciPy construction. The resulting `FockInput` runs on `piquasso-fock`, `piquasso-mixed-fock`, `native-fock` and `native-mixed-fock`. A `GKPResource` can also be passed directly as a `Prepare`/`inputs` state (capability `fock_input`; `gaussian` raises `NotImplementedError: GaussianBackend does not support fock_input`) and is serialized by `photographiq.serialization.dumps`/`loads`. |
| CPU / GPU | CPU (NumPy/SciPy) |
| Assumptions | hbar=2, L = sqrt(2*pi). Finite peaks and envelope. `project` raises `ValueError` if the captured weight is non-finite, below 1e-14 or above 1 + 1e-6. `grid_points` must be odd and >= 129, and the grid spacing must be at most `peak_width/4` (otherwise `ValueError: GKP peaks are unresolved`). `captured_weight` only diagnoses the Fock projection; `peaks` and `grid_points` need separate refinement. `superposition` normalizes using the actual nonzero overlap of the two finite codewords. |
| Tests | `tests/gkp/test_resources.py::test_comb_projection_and_decoder`; `tests/gkp/test_independent_convergence.py::test_finite_logical_overlap_matches_closed_gaussian_integral` and `::test_grid_peak_and_cutoff_are_independent` (reference: closed-form Gaussian overlap integrals and an independent Hermite projection) |
| Status | **Implemented but experimental** |

```python
import photographiq as pg
r = pg.GKPResource(0, peak_width=0.5, envelope=0.5)
state, weight = r.project(48)
print(len(state.amplitudes), round(weight, 6))
# 48 1.0
```

### Finite GKP code (`GKPCode`)

| Field | Value |
|---|---|
| Description | Holds the two finite codewords for one resource setting. Provides logical states, the nonidentity Gram matrix, the Gram-aware code-space projector, code-subspace leakage and one-mode diagnostics. |
| Python API | `photographiq.GKPCode(peak_width=0.4, envelope=0.4, cutoff=48, peaks=8, grid_points=4097)`; `.zero()`, `.one()`, `.plus()`, `.minus()`, `.encode(alpha, beta)`, `.gram`, `.code_projector`, `.leakage(state)`, `.diagnostics(state)`, `.resource(bit)` |
| Implementation | `src/photographiq/encoded.py` (resources from `src/photographiq/gkp.py`) |
| Backends | Backend-independent. The states it produces run on the four Fock backends. |
| CPU / GPU | CPU (NumPy) |
| Assumptions | The codewords overlap (`gram[0,1]` is nonzero), so `encode` is not an isometric encoder. `code_projector` raises `ValueError` if the Gram matrix is numerically singular (minimum eigenvalue < 1e-10). `cutoff` is the exclusive Fock cutoff. |
| Tests | `tests/gkp/measurements/test_encoded.py::test_resources_gram_projector_and_gate_interface`; `tests/gkp/measurements/test_encoded_boundaries.py::test_invalid_encoded_parameters_fail_explicitly` (reference for the overlap: closed Gaussian integrals in `tests/gkp/test_independent_convergence.py`) |
| Status | **Implemented but experimental** |

```python
import numpy as np, photographiq as pg
code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=48, peaks=6, grid_points=2049)
print(np.round(code.gram, 6))
print(code.leakage(code.encode(1, 1j).amplitudes) < 1e-12)
# [[1.      +0.j 0.122965+0.j]
#  [0.122965+0.j 1.      +0.j]]
# True
```

### Stabilizer expectations

| Field | Value |
|---|---|
| Description | Estimates the two stabilizer expectations on one reduced mode of a Fock state: `q_translation` = <exp(-i L p)> (a shift of q by 2L) and `p_translation` = <exp(i L q)> (a shift of p by 2L). |
| Python API | `photographiq.gkp.stabilizers(state, node=None) -> {"q_translation": complex, "p_translation": complex}` |
| Implementation | `src/photographiq/gkp.py` |
| Backends | Any Fock result state (`piquasso-fock`, `piquasso-mixed-fock`, `native-fock`, `native-mixed-fock`) |
| CPU / GPU | CPU (NumPy/SciPy `expm`) |
| Assumptions | The displacement matrices use ladder operators padded to 2*cutoff+32 and are then truncated. Finite resources give expectations below one. Check the Fock cutoff independently, especially for sharp grids. |
| Tests | `tests/gkp/test_logical_operations.py::test_stabilizers_against_closed_integrals` (reference: closed displacement integrals); `tests/gkp/test_conventions.py::test_weyl_logical_anticommutation_and_stabilizer_commutation` (Weyl algebra) |
| Status | **Implemented but experimental** |

```python
import photographiq as pg
code = pg.GKPCode(peak_width=0.5, envelope=0.5, cutoff=48)
res = pg.simulate(pg.Pattern(inputs=(0,)), inputs={0: code.zero()},
                  backend="piquasso-fock", cutoff=48)
print({k: round(v.real, 4) for k, v in pg.gkp.stabilizers(res.state).items()})
# {'q_translation': 0.3834, 'p_translation': 0.4559}
```

### Logical X and Z displacements

| Field | Value |
|---|---|
| Description | Returns a `Displace` command that translates q (X) or p (Z) by L. It acts as the logical Pauli on the ideal lattice, up to a global phase. |
| Python API | `photographiq.gkp.logical_displacement(node, pauli="X")`; `GKPCode.logical_gate(node, "X" or "Z")` |
| Implementation | `src/photographiq/gkp.py`, `src/photographiq/encoded.py` |
| Backends | All backends that support `Displace`. On GKP states it is meaningful only on the Fock backends. |
| CPU / GPU | CPU |
| Assumptions | The finite-energy envelope is displaced as well, so the command does not exactly permute the finite codewords. The test asserts that the fidelity to the target codeword lies strictly between 0.5 and 1 and matches the closed-form overlap. |
| Tests | `tests/gkp/test_logical_operations.py::test_logical_displacement_fidelity_and_repeated_action` (reference: closed-form displaced-comb overlaps, agreement to 2e-7 at cutoff 96) |
| Status | **Implemented but experimental** |

```python
import photographiq as pg
print(pg.GKPCode().logical_gate(0, "X"))
# Displace(node=0, q=2.5066282746310002, p=0.0)
```

### Logical H, S and CZ (ideal-lattice Gaussian realizations)

| Field | Value |
|---|---|
| Description | H is `Rotate(node, pi/2)`, S is `QuadraticPhase(node, 1.0)` and CZ is `Entangle(u, v, 1.0)`. These are the Gaussian unitaries that act as the logical Clifford gates on ideal lattice states. |
| Python API | `GKPCode.logical_gate(node, gate)` with `gate` in `"X"`, `"Z"`, `"H"`, `"S"`; `GKPCode.logical_cz(u, v)` |
| Implementation | `src/photographiq/encoded.py` |
| Backends | Fock backends for GKP inputs: `piquasso-fock`, `piquasso-mixed-fock`, `native-fock`, `native-mixed-fock` |
| CPU / GPU | CPU (Piquasso, or C++/OpenMP for the native engines) |
| Assumptions | On finite-energy codewords these gates are approximate. The H/S test asserts only a logical fidelity above 0.4 for broad resources (0.55, 0.5). It requires the Gaussian action itself to match an independent reference to 2e-8 at cutoff 80 (H: the Fock-diagonal phase exp(i pi n/2); S: multiplication by exp(i q^2/4) evaluated by quadrature with independent Hermite functions). These gate tests run `PiquassoFockBackend` with a relaxed `norm_tolerance` (0.05 for H/S, 0.1 for CZ). The two-mode CZ must reach an infidelity below 1e-7 against direct multiplication by exp(i q1 q2 / 2) at total cutoff 64. `logical_cz(u, u)` raises `ValueError`. |
| Tests | `tests/gkp/logical_gates/test_finite_reference.py::test_finite_gate_against_quadrature_reference` and `::test_ideal_lattice_phase_and_cz_identity` (ideal-lattice phase identities); `tests/gkp/multimode/test_cz_reference.py::test_two_mode_cz_independent_reference` (reference: Fock-diagonal phase and independent Hermite/quadrature integrals; `piquasso-fock` only) |
| Status | **Implemented but experimental** |

```python
import photographiq as pg
code = pg.GKPCode()
print(code.logical_gate(0, "H"), code.logical_gate(0, "S"))
print(code.logical_cz(0, 1))
# Rotate(node=0, angle=1.5707963267948966) QuadraticPhase(node=0, s=1.0)
# Entangle(u=0, v=1, weight=1.0)
```

### Non-Clifford logical gates (T, arbitrary rotations)

| Field | Value |
|---|---|
| Description | No logical gate other than X, Z, H, S and CZ is provided. |
| Python API | `GKPCode.logical_gate(node, "T")` raises `NotImplementedError("Only ideal-lattice X, Z, H and S realizations exist")` |
| Implementation | `src/photographiq/encoded.py` |
| Backends | None |
| CPU / GPU | N/A |
| Assumptions | A cubic-phase or magic-state route would need its own validated protocol. None exists. |
| Tests | `tests/gkp/measurements/test_encoded.py::test_resources_gram_projector_and_gate_interface` (asserts the `NotImplementedError`) |
| Status | **Unsupported** |

```python
import photographiq as pg
try:
    pg.GKPCode().logical_gate(0, "T")
except NotImplementedError as e:
    print("NotImplementedError:", e)
# NotImplementedError: Only ideal-lattice X, Z, H and S realizations exist
```

### Physical logical X/Z readout (`PhysicalGKPReadout`)

| Field | Value |
|---|---|
| Description | A destructive ideal homodyne detection (Z: q quadrature, angle 0; X: p quadrature, angle pi/2) followed by a decoder and an optional Pauli-frame flip. The result is a `LogicalDecodeResult` with `bit`, `raw_outcome`, `cell`, `residual`, `decoder`, `confidence`, `probabilities`, `code_subspace_leakage` and `frame_correction`. |
| Python API | `photographiq.PhysicalGKPReadout(basis="Z", decoder=NearestCellDecoder(), frame=LogicalPauliFrame(), flip=0)`; `GKPCode.logical_measurement(basis="Z", *, alpha=None, **kwargs)`; used through `Pattern.measure(node, readout, key=...)` |
| Implementation | `src/photographiq/encoded.py`; the measurement is executed by `src/photographiq/backends/fock.py`, `mixed_fock.py` and `native_fock.py` |
| Backends | Requires the capabilities `encoded_gkp_xz` and `homodyne`: `piquasso-fock`, `piquasso-mixed-fock`, `native-fock`, `native-mixed-fock`. On `gaussian` the call raises `NotImplementedError: GaussianBackend does not support encoded_gkp_xz`. |
| CPU / GPU | CPU (Piquasso, or C++/OpenMP for the native engines) |
| Assumptions | Ideal detector. Passing `measurement_outcomes={key: value}` postselects the raw analog value. The result is a conditional branch with a probability density (`measurement_statistics[key]["kind"] == "density"`), not a deterministic logical bit. Leakage cannot be inferred from one analog outcome and is returned as `None`. |
| Tests | `tests/gkp/measurements/test_encoded.py::test_physical_analog_result_and_feedforward` (piquasso-fock and piquasso-mixed-fock); `tests/gkp/convergence/test_modular_reference.py::test_physical_readout_converges_to_independent_continuous_reference` (reference: an independent continuous comb density, agreement below 3e-7); `tests/gkp/multimode/test_cz_reference.py::test_cz_measurement_step_has_correct_conditional_state_and_density` (independent quadrature bra contraction); `tests/native/test_native_fock_backend.py::test_instrument_and_gkp_readout_match_piquasso` (native-fock against Piquasso) |
| Status | **Implemented but experimental** |

```python
import photographiq as pg
code = pg.GKPCode(cutoff=48)
p = pg.Pattern(inputs=(0,)).measure(0, code.logical_measurement("X"), key="x")
r = pg.simulate(p, inputs={0: code.plus()}, backend="piquasso-fock", cutoff=48,
                measurement_outcomes={"x": 0.2})       # postselected analog branch
o = r.outcomes["x"]
print(o.bit, o.raw_outcome, o.cell, round(o.residual, 6), o.decoder)
print(r.measurement_statistics["x"]["kind"])
# 0 0.2 0 0.2 nearest-cell
# density
```

### Signed-X measurement through `logical_measurement("XY", alpha)`

| Field | Value |
|---|---|
| Description | An XY-plane measurement request is accepted only at the angles where it equals physical X readout: alpha = 0 mod 2*pi (X) and alpha = pi mod 2*pi (X with `flip=1`). |
| Python API | `GKPCode.logical_measurement("XY", alpha=...)` |
| Implementation | `src/photographiq/encoded.py` |
| Backends | Same as physical readout (four Fock backends) |
| CPU / GPU | CPU |
| Assumptions | The reduced angle is compared with an absolute tolerance of 1e-14. A nearby trainable angle is never rounded to X. Every other finite angle raises `NotImplementedError("Physical logical XY supports only alpha=0 mod pi; no Y or injection")`. A non-finite or non-scalar alpha raises `ValueError`. |
| Tests | `tests/gkp/measurements/test_encoded.py::test_unsupported_measurements_never_lower_to_rotated_homodyne`; `tests/ml/lowering/test_preflight.py::test_authoritative_upstream_tolerance` |
| Status | **Partially supported** |

```python
import photographiq as pg
code = pg.GKPCode()
for a in (0.0, 3.141592653589793):
    m = code.logical_measurement("XY", alpha=a); print(m.basis, m.flip)
try:
    code.logical_measurement("XY", alpha=0.3)
except NotImplementedError as e:
    print("NotImplementedError:", e)
# X 0
# X 1
# NotImplementedError: Physical logical XY supports only alpha=0 mod pi; no Y or injection
```

### Physical Y, arbitrary XY and logical measurement synthesis

| Field | Value |
|---|---|
| Description | Physical Y readout, arbitrary-angle XY readout and magic-state or logical-Rz synthesis are not implemented. `LogicalMeasurementSynthesis` exists only as an extension point. |
| Python API | `GKPCode.logical_measurement("Y")` and `PhysicalGKPReadout("Y")` raise `NotImplementedError("Only physical GKP X/Z readout is implemented; Y/XY need validation")`; `photographiq.LogicalMeasurementSynthesis().lower(code, alpha)` raises `NotImplementedError("No validated magic-state injection or arbitrary logical Rz protocol")` |
| Implementation | `src/photographiq/encoded.py` |
| Backends | None |
| CPU / GPU | N/A |
| Assumptions | The project's roadmap ([PhotoGraphiQML roadmap](ml/ROADMAP.md)) lists validated arbitrary XY measurement and injection instruments as future work. |
| Tests | `tests/gkp/measurements/test_encoded.py::test_unsupported_measurements_never_lower_to_rotated_homodyne` |
| Status | **Unsupported** |

```python
import photographiq as pg
try:
    pg.GKPCode().logical_measurement("Y")
except NotImplementedError as e:
    print("NotImplementedError:", e)
# NotImplementedError: Only physical GKP X/Z readout is implemented; Y/XY need validation
```

### Ideal logical XY reference measurement

| Field | Value |
|---|---|
| Description | Outcome probabilities of an ideal two-dimensional qubit measurement of cos(alpha) X + sin(alpha) Y. It is a reference for comparison, not a finite oscillator POVM. |
| Python API | `photographiq.IdealLogicalXYMeasurement(alpha).probabilities(state)` (state: 2-vector or 2x2 density matrix) |
| Implementation | `src/photographiq/encoded.py`, `src/photographiq/instruments.py` (`density_matrix`) |
| Backends | None (pure NumPy qubit arithmetic) |
| CPU / GPU | CPU (NumPy) |
| Assumptions | The input must be normalized, Hermitian and positive (`ValueError` otherwise). It is not available as a pattern measurement. |
| Tests | `tests/gkp/measurements/test_encoded.py::test_ideal_reference` (reference: closed-form (1 + <O>)/2 for six qubit states and five angles) |
| Status | **Implemented and validated** |

```python
import numpy as np, photographiq as pg
m = pg.IdealLogicalXYMeasurement(np.pi / 4)
print(np.round(m.probabilities(np.array([1, 1j]) / np.sqrt(2)), 6))
# [0.853553 0.146447]
```

### Nearest-cell decoding

| Field | Value |
|---|---|
| Description | Maps a real quadrature outcome to the nearest lattice cell. It returns the residual in [-L/2, L/2) and the cell parity as the logical bit. |
| Python API | `photographiq.gkp.decode_shift(outcome) -> (residual, parity)`; `photographiq.NearestCellDecoder().decode(outcome) -> LogicalDecodeResult`; `GKPCode.decode(outcome, decoder=None)` |
| Implementation | `src/photographiq/gkp.py`, `src/photographiq/encoded.py` |
| Backends | Classical post-processing; used by `PhysicalGKPReadout` on all four Fock backends |
| CPU / GPU | CPU |
| Assumptions | An exact half-cell tie maps to the upper cell. The comparison is made in the original coordinate so that adjacent floating-point values at a boundary keep the correct parity. A non-finite outcome raises `ValueError`. This is a decision rule, not an optimal finite-energy decoder, and it returns `confidence=None`. |
| Tests | `tests/gkp/test_conventions.py::test_decoder_cells_and_ties` and `::test_decoder_adjacent_floats_at_boundary`; `tests/gkp/measurements/test_encoded.py::test_decoder_cells_boundaries` (reference: the definition of the cell index, for cells up to +/-10000) |
| Status | **Implemented and validated** |

```python
import photographiq as pg
r = pg.NearestCellDecoder().decode(2.5 * pg.gkp.SPACING)   # tie -> upper cell
print(r.cell, r.bit, round(r.residual, 12))
print(pg.gkp.decode_shift(-2.6))
# 3 1 -1.253314137316
# (-0.09337172536899985, 1)
```

### Soft-decision (Bayesian ensemble) decoding

| Field | Value |
|---|---|
| Description | The posterior over the labels of an equal-prior (or user-prior) ensemble of zero/one (Z) or plus/minus (X) preparations. It uses the actual finite-Fock homodyne densities of those codewords. |
| Python API | `photographiq.SoftDecisionDecoder(code, basis="Z", prior=(0.5, 0.5)).decode(outcome)` |
| Implementation | `src/photographiq/gkp_studies.py` |
| Backends | Classical post-processing; usable as the `decoder=` of `PhysicalGKPReadout` |
| CPU / GPU | CPU (NumPy) |
| Assumptions | The posterior concerns the preparation label of the stated ensemble, not arbitrary or entangled logical states. Only X/Z bases (`NotImplementedError` otherwise). The prior must be two positive values summing to one. An outcome outside the numerical likelihood support raises `ValueError`. |
| Tests | `tests/gkp/decoders/test_soft_and_studies.py::test_soft_decoder_matches_continuous_likelihoods_and_calibrated_draws` (reference: independent continuous likelihoods, posterior agreement to 3e-5 on 1500 synthetic draws); `tests/gkp/measurements/test_encoded_boundaries.py::test_soft_frame_probabilities_reverse_with_labels` |
| Status | **Implemented but experimental** |

```python
import photographiq as pg
code = pg.GKPCode(0.55, 0.5, 48, 6, 2049)
r = pg.SoftDecisionDecoder(code, "Z", prior=(0.5, 0.5)).decode(0.3)
print(r.bit, r.decoder, [round(float(x), 6) for x in r.probabilities])
# 0 finite-fock-ensemble-bayes [0.999764, 0.000236]
```

### Logical Pauli frames

| Field | Value |
|---|---|
| Description | A classical X^x Z^z frame (modulo global phase). It supports composition and propagation through H, S and CZ, the measurement-angle update for XY, and the bit correction for X/Y/Z readout. |
| Python API | `photographiq.LogicalPauliFrame(x_bit=0, z_bit=0)`; `.compose(other)`, `.hadamard()`, `.phase()`, `.cz(other)`, `.xy_angle(alpha)`, `.correction(basis)`; `PhysicalGKPReadout(frame=...)` |
| Implementation | `src/photographiq/encoded.py` |
| Backends | Classical; applied by `PhysicalGKPReadout` and `multimode_readout(frames=...)` |
| CPU / GPU | CPU |
| Assumptions | The frame has its ideal logical meaning. It does not correct physical envelope distortion. Entries must be bits (`ValueError` otherwise). |
| Tests | `tests/gkp/measurements/test_encoded.py::test_frame_against_qubit_conjugation` (reference: explicit 2x2 Pauli conjugation) |
| Status | **Implemented and validated** |

```python
import photographiq as pg
f = pg.LogicalPauliFrame(1, 0)
print(f.hadamard(), f.phase())
print(f.correction("X"), f.correction("Z"), f.correction("Y"))
# LogicalPauliFrame(x_bit=0, z_bit=1) LogicalPauliFrame(x_bit=1, z_bit=1)
# 0 1 1
```

### Modular homodyne POVM (`modular_effects`)

| Field | Value |
|---|---|
| Description | The two Fock-compressed effects obtained by integrating the homodyne POVM over all even and all odd lattice cells. Contracting them with a density matrix gives the probability of each decoded bit without sampling. |
| Python API | `photographiq.modular_effects(cutoff, basis="Z") -> ndarray (2, cutoff, cutoff)` |
| Implementation | `src/photographiq/encoded.py` |
| Backends | Backend-independent (NumPy/SciPy `quad_vec`) |
| CPU / GPU | CPU |
| Assumptions | The cell window extends beyond the oscillator turning point and is checked against the identity (`ArithmeticError` if incomplete or if integration fails). It discards the analog record, so it differs from a conditional trajectory with a known raw outcome. Only X/Z (`NotImplementedError`). |
| Tests | `tests/gkp/convergence/test_modular_reference.py::test_physical_readout_converges_to_independent_continuous_reference` and `::test_grid_and_peak_refinement_on_decoded_probabilities` (reference: independent continuous comb densities) |
| Status | **Implemented but experimental** |

```python
import numpy as np, photographiq as pg
code = pg.GKPCode(0.55, 0.5, 48, 6, 2049)
E = pg.modular_effects(48, "Z")
v = np.array(code.encode(1, 0).amplitudes)
print(np.round(np.einsum("i,kij,j->k", v.conj(), E, v).real, 6))
# [0.977296 0.022704]
```

### Joint multimode readout (`multimode_readout`)

| Field | Value |
|---|---|
| Description | Exact joint modular X/Z bit probabilities of all modes of a finite Fock state, with optional per-mode Pauli frames. It also returns the marginals derived from the joint distribution. |
| Python API | `photographiq.multimode_readout(state, bases, *, frames=None)` |
| Implementation | `src/photographiq/encoded.py` |
| Backends | Any Fock result state |
| CPU / GPU | CPU (NumPy) |
| Assumptions | It never multiplies marginals, so correlations are kept. The cost grows quadratically with the Fock dimension. `code_subspace_leakage` and `residuals` are returned as `None`. A basis must be given for every mode. |
| Tests | `tests/gkp/multimode/test_cz_reference.py::test_two_mode_cz_independent_reference` (normalization of the joint and marginal distributions); `tests/gkp/multimode/test_convergence_metrics.py::test_joint_fidelity_and_leakage_convergence_are_distinct`; `tests/gkp/measurements/test_encoded_boundaries.py::test_mixed_input_instrument_and_multimode_contract` |
| Status | **Implemented but experimental** |

```python
import photographiq as pg
code = pg.GKPCode(0.9, 0.9, 24, 4, 1025)
p = pg.Pattern(inputs=(0, 1)).append(code.logical_cz(0, 1))
r = pg.simulate(p, inputs={0: code.plus(), 1: code.zero()}, backend="piquasso-fock", cutoff=24)
out = pg.multimode_readout(r.state, {0: "X", 1: "Z"})
print({k: round(v, 4) for k, v in out["joint_probabilities"].items()})
# RuntimeWarning: GaussianTransform: retained Fock norm 0.99998177; test a larger cutoff
# {(0, 0): 0.7149, (0, 1): 0.046, (1, 0): 0.121, (1, 1): 0.1181}
# (Ideal lattice: (0, 0) with probability 1. Broad 0.9-width peaks misclassify.)
```

### Measurement convergence studies (`measurement_convergence`)

| Field | Value |
|---|---|
| Description | Varies one parameter (`cutoff`, `grid_points`, `peaks`, `peak_width` or `envelope`) and reports, for each value: decoded probabilities, per-cell residual moments, code-subspace leakage, fidelity to the previous value and stabilizers. |
| Python API | `photographiq.measurement_convergence(code, values, *, axis="cutoff", basis="Z", coefficients=(1, 0), backend="piquasso-fock") -> GKPMeasurementStudy(axis, rows)` |
| Implementation | `src/photographiq/gkp_studies.py` |
| Backends | `backend=` prepares the codeword state: `piquasso-fock` (default, kept for reproducibility of recorded studies) or `native-fock` (since 2026-11-12, no Piquasso needed); any other name raises `ValueError("backend must be piquasso-fock or native-fock")`. The two give identical study rows in the example below. |
| CPU / GPU | CPU |
| Assumptions | The study never certifies convergence. Width and envelope sweeps change the physical resource; they are not numerical refinements. At least two strictly increasing values are required. |
| Tests | `tests/gkp/decoders/test_soft_and_studies.py::test_measurement_study_tracks_more_than_projection_mass` and `::test_invalid_decoder_and_study_requests` |
| Status | **Implemented but experimental** |

```python
import photographiq as pg
code = pg.GKPCode(0.55, 0.5, 24, 6, 2049)
for backend in ("piquasso-fock", "native-fock"):
    study = pg.measurement_convergence(code, (24, 32), basis="Z", backend=backend)
    for row in study.rows:
        print(backend, row["cutoff"], [round(x, 5) for x in row["probabilities"]], row["fidelity_to_previous"] is None)
# piquasso-fock 24 [0.97733, 0.02267] True
# piquasso-fock 32 [0.97729, 0.02271] False
# native-fock 24 [0.97733, 0.02267] True
# native-fock 32 [0.97729, 0.02271] False
```

### Finite-ancilla syndrome extraction and correction (`correction_pattern`)

| Field | Value |
|---|---|
| Description | Builds one SUM-gate syndrome extraction with a finite logical-plus GKP ancilla, q homodyne of the ancilla, and a modular feed-forward displacement `-decode_shift(outcome)[0]` on the data mode. The p version Fourier-conjugates the whole protocol. |
| Python API | `photographiq.gkp.correction_pattern(*, quadrature="q", resource=None, input_node="in", ancilla="gkp", key="syndrome") -> Pattern` |
| Implementation | `src/photographiq/gkp.py` |
| Backends | Any Fock backend: the pattern ran here on `piquasso-fock`, `native-fock`, `piquasso-mixed-fock` and `native-mixed-fock`. The tests validate it on `piquasso-fock` only. |
| CPU / GPU | CPU |
| Assumptions | `resource` is required (`ValueError` otherwise). The correction is a Python `CallableExpression`: JSON serialization raises `TypeError: Callables are runtime-only and cannot be serialized`, and `autodiff` rejects it with `TypeError`. The validated regime is narrow: for the tested ancilla (peak_width = envelope = 0.7), a shift beyond one cell needed cutoff 72 to meet the 2e-6 target ([GKP guide](user-guide/gkp.md)). There is no fault-tolerance or threshold claim. |
| Tests | `tests/gkp/test_resources.py::test_gkp_syndrome_pattern`; `tests/gkp/test_correction.py::test_finite_syndrome_correction_matches_wavefunction_filter`; `tests/gkp/test_syndrome_validation.py::test_syndrome_filter_probability_and_corrected_mean` (reference: an independent SUM wavefunction kernel) |
| Status | **Implemented but experimental** |

```python
import photographiq as pg
anc = pg.gkp.superposition(1, 1, cutoff=32, peak_width=0.7, envelope=0.7)
p = pg.gkp.correction_pattern(quadrature="q", resource=anc)
print([type(c).__name__ for c in p.commands])
# ['Prepare', 'Rotate', 'Entangle', 'Rotate', 'Measure', 'Displace', 'Output']
```

### Finite Kraus measurement instruments (`MeasurementInstrument`)

| Field | Value |
|---|---|
| Description | A complete finite-outcome instrument given as `{label: (K1, K2, ...)}`. It checks completeness (effects sum to the identity), computes outcome probabilities and the retained conditional update. When used in a pattern it destructively measures one mode and keeps the conditional state of the others. |
| Python API | `photographiq.MeasurementInstrument(kraus)`; `.effects`, `.probabilities(state)`, `.conditional(state, outcome)`, `.required_capabilities`; `Pattern.measure(node, instrument, key=...)` |
| Implementation | `src/photographiq/instruments.py`; execution in `src/photographiq/backends/fock.py`, `mixed_fock.py` |
| Backends | Rank-one effects (`rank_one_instrument`): `piquasso-fock`, `native-fock`, `piquasso-mixed-fock`, `native-mixed-fock`. General multi-Kraus instruments (`custom_kraus_instrument`) are accepted only by `piquasso-mixed-fock` and `native-mixed-fock`; pure backends reject them before execution. The instrument dimension must equal the backend cutoff (`ValueError`). |
| CPU / GPU | CPU |
| Assumptions | Effects that do not sum to the identity raise `ValueError` (include failure outcomes explicitly). An impossible outcome raises `ValueError`. `conditional` keeps the measured system, whereas pattern `Measure` traces it out. |
| Tests | `tests/gkp/instruments/test_instruments.py` (`test_kraus_is_not_effect_update`, `test_complex_rank_one_entangled_conditioning`, `test_multikraus_partial_measurement_preserves_mixedness_and_rejects_pure_preflight`, `test_invalid_instruments`, `test_validation_probabilities_and_impossible_branch`; reference: explicit Kraus sums and hand-computed conditional states; the multi-Kraus case runs on `piquasso-mixed-fock`); `tests/native/test_native_fock_backend.py::test_instrument_and_gkp_readout_match_piquasso` (rank-one, `native-fock` vs `piquasso-fock`); `tests/native/test_native_mixed_fock_backend.py::test_multi_kraus_instrument_matches_explicit_partial_trace` (random 2-outcome, 4-Kraus instrument on `native-mixed-fock`, against an explicit partial trace, `1e-12`) |
| Status | **Implemented but experimental** (multi-Kraus instruments are now independently tested on both mixed backends; the Fock execution paths keep the project's experimental label) |

```python
import numpy as np, photographiq as pg
inst = pg.MeasurementInstrument({0: (np.diag([1, 0]),), 1: (np.diag([0, 1]),)})
print(sorted(inst.required_capabilities))
print(inst.probabilities(np.array([0.6, 0.8])))
# ['rank_one_instrument']
# {0: 0.36, 1: 0.6400000000000001}
```

### Unambiguous code discrimination instrument

| Field | Value |
|---|---|
| Description | A finite POVM with outcomes 0, 1, `"inconclusive"` and `"outside-code"`, built from the dual basis of the nonorthogonal finite codewords with equal reciprocal-vector weights. |
| Python API | `GKPCode.discrimination_instrument() -> MeasurementInstrument` |
| Implementation | `src/photographiq/encoded.py` |
| Backends | As `MeasurementInstrument` (rank-one check applies) |
| CPU / GPU | CPU |
| Assumptions | It is a mathematical instrument, not an optical synthesis, and it is not modular homodyne detection. |
| Tests | `tests/gkp/instruments/test_instruments.py::test_finite_code_dual_povm_distinguishes_ambiguity_from_leakage` |
| Status | **Implemented but experimental** |

```python
import numpy as np, photographiq as pg
code = pg.GKPCode(0.55, 0.5, 24, 6, 2049)
p = code.discrimination_instrument().probabilities(np.array(code.zero().amplitudes))
print({k: round(v, 4) for k, v in p.items()})
# {0: 0.877, 1: 0.0, 'inconclusive': 0.123, 'outside-code': -0.0}
```

### Fault-tolerant GKP architecture and thresholds

| Field | Value |
|---|---|
| Description | No fault-tolerant GKP architecture, concatenated code, threshold estimate or optimal finite-energy decoder exists. |
| Python API | None |
| Implementation | None (the `src/photographiq/gkp.py` docstring states that the module does not establish a fault-tolerance threshold) |
| Backends | None |
| CPU / GPU | N/A |
| Assumptions | The root `ROADMAP.md` requires broader syndrome/decoder studies and independent finite-energy correction evidence before any fault-tolerance claim. The [feature matrix](validation/feature-matrix.md) marks "Fault-tolerant GKP thresholds" as unsupported. |
| Tests | None |
| Status | **Planned** |

Example: none (not implemented).

## 28. PhotoGraphiQML

`photographiqml` (version 0.2.0, shipped in the same distribution as `photographiq`) implements the measurement-based quantum machine learning ansatz MuTA. There are two execution paths. The ideal logical path runs on qubit statevectors in NumPy. The restricted physical path lowers the model to a PhotoGraphiQ GKP `Pattern` and executes it on a Fock backend, either `piquasso-fock` (the default) or `native-fock`. The physical path supports only signed-X measurements (angles 0 and pi). Arbitrary-angle physical MuTA, entangled physical inputs and mixed-Fock lowering are rejected before any Fock allocation. See [PhotoGraphiQML](ml/index.md) and the [physical architecture](ml/physical/architecture.md).

### Triangle neuron geometry (`TriangleNeuron`)

| Field | Value |
|---|---|
| Description | One paper layer (n wires, one pivot): a 5-column wire graph per wire, plus edges from the pivot's column-1 site to columns 0 and 2 of each connected wire. Nodes are `(wire, column)`. |
| Python API | `photographiqml.TriangleNeuron(n_wires=2, pivot=0, connections=None)`; `.graph`, `.connections`, `.input_nodes`, `.output_nodes`, `.trainable_nodes`, `.draw(ax=None)` |
| Implementation | `src/photographiqml/ansatz/triangle.py` |
| Backends | None (NetworkX graph) |
| CPU / GPU | CPU |
| Assumptions | `connections=None` connects every non-pivot wire. An empty tuple gives disconnected wires. `draw` shows the logical graph, not an optical layout. |
| Tests | `tests/ml/muta/test_logical.py::test_triangle_and_table_one` (reference: paper Table 1 node, edge and parameter counts, and closed-form exp(i phi XX/2)); `tests/ml/muta/test_logical.py::test_bad_triangle` |
| Status | **Implemented and validated** |

```python
from photographiqml import TriangleNeuron
t = TriangleNeuron(3, pivot=0)
print(t.graph.number_of_nodes(), t.graph.number_of_edges(), t.connections)
# 15 16 (1, 2)
```

### MuTA model construction

| Field | Value |
|---|---|
| Description | Composes triangle layers into a MuTA open graph with a causal flow, X/Z correction sets, a dependency DAG, a measurement order and named measurement-angle parameters `alpha.w<wire>.c<column>`. It follows MentPy's block-count convention. |
| Python API | `photographiqml.MuTA(n_wires=2, n_layers=1, *, one_column=False, restrict_trainable=False, representation="logical", connections=None)`; `.graph`, `.flow`, `.corrections`, `.measurement_order`, `.n_parameters`, `.expected_parameter_count`, `.summary()`, `.draw(ax=None)` |
| Implementation | `src/photographiqml/ansatz/muta.py` |
| Backends | None (graph construction) |
| CPU / GPU | CPU |
| Assumptions | `one_column=False` gives `n_wires` paper layers per requested layer. `restrict_trainable=True` freezes the column-3 measurements. Custom `connections` require `one_column=True`. |
| Tests | `tests/ml/muta/test_logical.py::test_topology` (reference: closed-form node, edge and parameter counts, bipartiteness and DAG order); `tests/ml/mentpy_reference/test_reference.py::test_exact_semantic_reference` (reference: the MentPy `templates.muta` graph at the pinned commit; skipped when MentPy is absent, as here, and run by `.github/workflows/ml-reference.yml`) |
| Status | **Implemented and validated** |

```python
from photographiqml import MuTA
m = MuTA(2, 1)
print(m.paper_depth, m.n_parameters, m.expected_parameter_count, len(m.graph), m.graph.number_of_edges())
print(MuTA(2, 1, restrict_trainable=True).n_parameters)
# 2 16 16 18 20
# 12
```

### Ideal logical execution

| Field | Value |
|---|---|
| Description | Executes the MuTA flow on an n-qubit statevector, without measurement randomness (Appendix B frontier translation): for each measured node it applies CZ to every remaining junction neighbour on the frontier, then H diag(exp(i alpha/2), exp(-i alpha/2)) to its wire. Returns a `LogicalResult` with `.state`, `.probabilities`, `.density_matrix` and `.expectation(observable)`. |
| Python API | `MuTA.run(input_state, parameters=None)`; `MuTA.unitary(parameters=None)`; `MuTA.run_batch(states, parameters=None)` |
| Implementation | `src/photographiqml/logical.py` |
| Backends | None (NumPy qubit simulator; no PhotoGraphiQ backend is used) |
| CPU / GPU | CPU (NumPy) |
| Assumptions | The input must be normalized (no implicit normalization). `unitary` is limited to 10 wires. Parameters are given as a name mapping or as a vector over the trainable (non-frozen) names. |
| Tests | `tests/ml/muta/test_logical.py::test_all_adaptive_branches` (reference: an independent full graph-state branch contraction, `photographiqml.validation.contract_branch`, over all 256 outcome branches); `::test_depth_inclusion_and_composition`; `tests/ml/mentpy_reference/test_reference.py::test_numerical_reference` (MentPy `PatternSimulator`; skipped here) |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiqml import MuTA
from photographiqml.validation import contract_branch
m = MuTA(1, 1); theta = m.initialize(seed=3)
out = m.run([1, 0], theta).state
ref, prob = contract_branch(m, [1, 0], theta)
print(np.round(out, 6), round(abs(np.vdot(ref, out)), 12), round(prob, 6))
# [0.980357+0.121299j 0.013778-0.154906j] 1.0 0.0625
```

### Parameters, freezing and serialization

| Field | Value |
|---|---|
| Description | Named parameters backed by `photographiq.Parameter` symbols, with freeze/unfreeze, seeded Gaussian initialization, JSON save/load (schema 1 logical, schema 2 physical) and topology checks on load. |
| Python API | `MuTA.parameters()`, `.trainable_parameters()`, `.freeze(names, value=None)`, `.unfreeze(names)`, `.initialize(seed=None, scale=0.1)`, `.state_dict()`, `.save(path)`, `MuTA.load(path)` |
| Implementation | `src/photographiqml/parameters.py`, `src/photographiqml/ansatz/muta.py` |
| Backends | None |
| CPU / GPU | CPU |
| Assumptions | Rebinding a frozen parameter to a new value raises `ValueError`. Non-finite values are rejected. Loading a schema-1 `gkp` resource model never converts it to physical execution. |
| Tests | `tests/ml/muta/test_parameters.py::test_binding_freezing_serialization` and `::test_seed_and_shapes` (reference: round-trip equality with the saved model state); `tests/ml/cross_layer/test_contract.py::test_physical_schema_and_legacy_migration` |
| Status | **Implemented and validated** |

Example: see `tests/ml/muta/test_parameters.py` and `examples/ml/tutorials/25.py`.

### Resource-only GKP representation (`representation="gkp"` / `"gkp-resource"`)

| Field | Value |
|---|---|
| Description | A MuTA labelled as a GKP resource model. It holds the topology and parameters but cannot be executed. `"gkp"` is the deprecated spelling. |
| Python API | `MuTA(..., representation="gkp-resource")`; `MuTA.run` raises `NotImplementedError` ("Physical GKP MuTA requires a validated logical XY measurement/injection instrument and decoder ..."). `representation="gkp"` also emits `DeprecationWarning`. Any other representation raises `ValueError` ("Choose logical or gkp-resource; use PhysicalMuTA for gkp-physical. CVMuTA is a separate proposal"). |
| Implementation | `src/photographiqml/ansatz/muta.py` |
| Backends | None |
| CPU / GPU | N/A |
| Assumptions | Physical execution uses the separate `PhysicalMuTA` class. |
| Tests | `tests/ml/gkp/test_bridge.py::test_representation_boundary`; `tests/ml/cross_layer/test_contract.py::test_physical_schema_and_legacy_migration` |
| Status | **Unsupported** |

```python
import warnings
from photographiqml import MuTA
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always"); m = MuTA(1, representation="gkp")
try:
    m.run([1, 0])
except NotImplementedError:
    print([x.category.__name__ for x in w], "NotImplementedError")
# ['DeprecationWarning'] NotImplementedError
```

### GKP resource bridge (`GKPBridge`)

| Field | Value |
|---|---|
| Description | A one-mode helper with four uses: it encodes a logical qubit state into finite GKP codewords, reports codeword diagnostics, returns the ideal logical target of a model, and returns a one-mode X/Z resource readout with a nearest or soft decoder. `GKPBridge.run` delegates to the physical runner. |
| Python API | `photographiqml.GKPBridge(cutoff=32, peak_width=0.4, envelope=0.4, peaks=8, grid_points=4097)`; `.encode(logical_state)`, `.code`, `.diagnostics()`, `.logical_target(model, input_state, parameters=None)`, `.run(model, input_state=None, parameters=None, *, config=None, **options)`, `.resource_readout(basis="Z", *, decoder="nearest")`, `.measurement_convergence(values, **options)` |
| Implementation | `src/photographiqml/gkp.py` |
| Backends | Encoding is NumPy. `run` uses the backend of `GKPPhysicalConfig`; `measurement_convergence` uses `piquasso-fock`. |
| CPU / GPU | CPU |
| Assumptions | `encode` is not an isometric channel (the codewords overlap). `logical_target` is labelled `representation="logical"`, never finite GKP. `diagnostics()["physical_muta_validated"]` is always `False`. `run` refuses resource-only models. |
| Tests | `tests/ml/gkp/test_bridge.py::test_resource_projection_and_execution`, `::test_projection_cutoff_and_grid_refinement`, `::test_representation_boundary`; `tests/ml/cross_layer/test_physical_execution.py::test_hard_soft_resource_decoder_and_resource_study` |
| Status | **Implemented but experimental** |

```python
from photographiqml import GKPBridge, MuTA
b = GKPBridge(cutoff=32, peak_width=0.6, envelope=0.6, peaks=6, grid_points=2049)
d = b.diagnostics()
print(round(d["codeword_overlap"], 6), d["physical_muta_validated"])
print(b.resource_readout("X", decoder="soft").decoder.__class__.__name__)
print(b.logical_target(MuTA(1), [1, 0]).representation)
# 0.174489 False
# SoftDecisionDecoder
# logical
```

### Physical configuration and PhotoGraphiQ contract check

| Field | Value |
|---|---|
| Description | `GKPPhysicalConfig` fixes the resource (cutoff, peak width, envelope, peaks, grid), the decoder, the backend and the allocation budgets. Constructing it first runs `check_photographiq_contract()`, which checks that every required PhotoGraphiQ API name exists and that the version satisfies 0.3.1 <= v < 0.4. |
| Python API | `photographiqml.GKPPhysicalConfig(cutoff=48, peak_width=0.6, envelope=0.6, peaks=6, grid_points=2049, decoder="nearest", backend="piquasso-fock", max_dimension=100000, max_matrix_bytes=256000000)`; `photographiqml.lowering.check_photographiq_contract()`; `photographiqml.lowering.REQUIRED_API`; `photographiqml.lowering.VALIDATED_BACKENDS == ("piquasso-fock", "native-fock")` |
| Implementation | `src/photographiqml/lowering.py` |
| Backends | `backend` must be `"piquasso-fock"` (default) or `"native-fock"`. Any other name, including `native-mixed-fock` and `gaussian`, raises `ValueError` ("... mixed Fock lowering remains unsupported"). |
| CPU / GPU | CPU |
| Assumptions | `cutoff` is the exclusive total-photon cutoff. The defaults are research starting points, not validated precision settings. A missing API name or a wrong version raises `ImportError` that names the audited commit. |
| Tests | `tests/ml/cross_layer/test_contract.py::test_public_contract_and_dependency`; `tests/ml/lowering/test_preflight.py::test_config_validation_and_roundtrip`; `tests/ml/cross_layer/test_native_backend.py::test_config_accepts_native_fock_and_keeps_piquasso_default` |
| Status | **Implemented and validated** |

```python
from photographiqml import GKPPhysicalConfig
from photographiqml.lowering import check_photographiq_contract
print(check_photographiq_contract())
print(GKPPhysicalConfig().backend)
try:
    GKPPhysicalConfig(backend="native-mixed-fock")
except ValueError as e:
    print("ValueError:", e)
# {'version': '0.3.1', 'audited_commit': 'db07f9f9bf47da841bfa6b206562c5a3ffb121d3'}
# piquasso-fock
# ValueError: Physical MuTA validates backend in ('piquasso-fock', 'native-fock') only; mixed Fock lowering remains unsupported
```

### Capability audit and preflight

| Field | Value |
|---|---|
| Description | Checks every bound angle with the upstream `GKPCode.logical_measurement("XY", alpha)` rule and estimates the peak number of live modes, the Hilbert dimension and the workspace bytes, all without any Fock projection. `require_supported` raises on the first unsupported node or on a budget or decoder violation. |
| Python API | `MuTA.physical_capabilities(parameters=None, *, config=None)`; `photographiqml.lowering.physical_capabilities(model, parameters=None, *, config=None)`; `photographiqml.lowering.require_supported(report)`; capability-only drawing `MuTA.draw_physical(parameters=None, *, config=None, ax=None)` / `photographiqml.lowering.visualize_lowering(model, parameters=None, *, config=None, ax=None)` (unsupported nodes drawn red, needs `matplotlib`) |
| Implementation | `src/photographiqml/lowering.py` |
| Backends | None (classical audit) |
| CPU / GPU | CPU |
| Assumptions | Angles are accepted only at 0 or pi mod 2*pi with absolute tolerance 1e-14. `decoder="soft"` is rejected for flow execution (its posteriors are not calibrated for adaptive node states). The memory figures are estimates; a request above `max_dimension` or `max_matrix_bytes` is rejected (`ValueError` from `require_supported`). |
| Tests | `tests/ml/lowering/test_preflight.py::test_reject_every_angle_before_any_fock_allocation`, `::test_authoritative_upstream_tolerance` (reference: the upstream `GKPCode.logical_measurement` rule), `::test_resource_and_decoder_guards_preallocation`, `::test_execution_request_guards_preallocation`; `tests/ml/cross_layer/test_physical_execution.py::test_draw_capability_failure_without_projection` |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiqml import MuTA, GKPPhysicalConfig
cfg = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
rep = MuTA(1, 1).physical_capabilities([0, np.pi, 0, 0.3], config=cfg)
print(rep["supported"], [r["parameter"] for r in rep["unsupported_nodes"]], rep["peak_live_modes"], rep["hilbert_dimension"])
# False ['alpha.w0.c3'] 2 300
```

### Physical lowering to a GKP pattern (`lower_muta_to_gkp`)

| Field | Value |
|---|---|
| Description | Lowers a signed-X MuTA to a public PhotoGraphiQ `Pattern`: `Prepare(node, GKPCode.plus())` just before a node is needed, unit-weight `Entangle` (logical CZ) for every graph edge, `Measure(node, PhysicalGKPReadout("X", ...))` with a nearest-cell decoder, and a `Signal(("bit", node), CallableExpression(...))` that records the interpreted bit: the decoded bit XOR the X-basis correction of the classical Pauli frame built from earlier interpreted bits. The frame is virtual; no physical correction displacement is applied. Returns a `PhysicalLoweringResult` with the pattern, code, key maps, frame dependencies and audit. |
| Python API | `photographiqml.lower_muta_to_gkp(model, parameters=None, *, config=None)`; `photographiqml.lowering.node_frame(model, node, records)` |
| Implementation | `src/photographiqml/lowering.py` |
| Backends | The pattern uses only public commands and runs through `photographiq.simulate` / `run_shots` on the configured backend. |
| CPU / GPU | CPU |
| Assumptions | The audit is completed before the first codeword allocation. The frame signals are Python callables, so the pattern is not JSON-serializable. `convergence` is always `{"certified": False}`. |
| Tests | `tests/ml/lowering/test_preflight.py::test_lowering_semantic_maps_and_commands`; `tests/ml/cross_layer/test_frames.py::test_all_signed_x_graph_branches_with_virtual_frames`; `tests/ml/cross_layer/test_physical_execution.py::test_conditional_against_independent_public_pattern` (reference: an independently assembled linear-chain pattern) |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiqml import MuTA, GKPPhysicalConfig, lower_muta_to_gkp
cfg = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
low = lower_muta_to_gkp(MuTA(1, 1), [0, np.pi, 0, np.pi], config=cfg)
names = [type(c).__name__ for c in low.pattern.commands]
print(len(names), names[:4], names[-2:])
# 17 ['Prepare', 'Entangle', 'Measure', 'Signal'] ['Signal', 'Output']
```

### Physical MuTA execution (`PhysicalMuTA`)

| Field | Value |
|---|---|
| Description | A finite-GKP MuTA restricted to the categorical angles {0, pi}. `mode="physical-conditional"` runs one trajectory with an explicit analog outcome for every measured node. This is a postselected branch, not a deterministic result. `mode="physical-shots"` samples `shots` trajectories with `photographiq.run_shots`. Its `decoded_joint_probabilities` are the mean over trajectories of the conditional joint output POVM (a Rao-Blackwell estimator); `sampled_output_bits` are drawn from each trajectory's joint POVM. The result also carries the logical target, frames, measurement records, standard errors and diagnostics (Gram, leakage, backend retained norms). |
| Python API | `photographiqml.PhysicalMuTA(n_wires=1, n_layers=1, *, physical_config=None, measurement_family="X", **kwargs)`; `.run(input_state, parameters=None, **options)`, which forwards to `photographiqml.physical.run_physical(model, input_state, parameters=None, *, config=None, mode="physical-shots", shots=1, seed=0, analog_outcomes=None, output_basis="Z")`; `.initialize(seed=None, scale=None)` (random 0/pi choices; any `scale` raises `ValueError`); `.angle_space()`; `.state_dict()` (schema 2); `photographiqml.compare_logical_physical(result)` |
| Implementation | `src/photographiqml/physical.py` |
| Backends | `piquasso-fock` (default) and `native-fock`, selected by `GKPPhysicalConfig.backend`. Mixed Fock is rejected. |
| CPU / GPU | CPU (Piquasso, or the C++/OpenMP native Fock engine) |
| Assumptions | Supported subset: signed-X angles, product inputs and X/Z output. The input must be a normalized product of single-qubit states (an entangled input raises `NotImplementedError`). The final readout is X or Z only. `unitary()` and `run_batch()` raise `NotImplementedError` because a finite-GKP instrument is not a logical unitary. The backend's norm guard can reject a supported pattern at low cutoff (`ValueError: Fock truncation norm ...; increase cutoff`). `compare_logical_physical` compares decoded statistics only (`finite_state_fidelity` is `None`). |
| Tests | `tests/ml/cross_layer/test_physical_execution.py::test_conditional_against_independent_public_pattern` (reference: an independently built pattern), `::test_physical_shots_repeatability_and_error_estimates`, `::test_two_wire_joint_readout_retains_correlations`, `::test_physical_interface_boundaries` |
| Status | **Partially supported** |

```python
import numpy as np
from photographiqml import PhysicalMuTA, GKPPhysicalConfig, compare_logical_physical
cfg = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
m = PhysicalMuTA(1, physical_config=cfg)
res = m.run([1, 0], [0, np.pi, 0, np.pi], mode="physical-conditional",
            analog_outcomes=dict.fromkeys(m.measurement_order, 0.0))   # postselected branch
print(res.mode, res.physical_result.backend)
print({k: round(v, 4) for k, v in res.decoded_joint_probabilities.items()})
print(round(compare_logical_physical(res)["total_variation_distance"], 4))
# (stderr also shows RuntimeWarning lines "GaussianTransform: retained Fock norm 0.99998...; test a larger cutoff")
# physical-conditional PiquassoFockBackend
# {(0,): 0.8272, (1,): 0.1728}
# 0.1728
```

### Native-backend physical execution and its verification

| Field | Value |
|---|---|
| Description | With `GKPPhysicalConfig(backend="native-fock")` the lowered pattern runs on `NativeFockBackend`. That class reuses all the Python preparation, homodyne, renormalization and GKP readout code of `PiquassoFockBackend` and replaces only the state object and the gate kernels. The CZ is applied as the exact projected operator P U P instead of Piquasso's per-Euler-factor truncation. |
| Python API | `GKPPhysicalConfig(..., backend="native-fock")`; `PhysicalMuTA(...).run(...)`; the executed engine is visible as `result.physical_result.backend` (conditional) or `result.physical_result.trajectories[i].backend` (shots) |
| Implementation | `src/photographiqml/lowering.py`, `src/photographiqml/physical.py`, `src/photographiq/backends/native_fock.py`, `cpp/src/fock/` |
| Backends | `native-fock` (requires the compiled `photographiq._native` fock component; Piquasso is not imported by this engine) |
| CPU / GPU | CPU (C++/OpenMP) |
| Assumptions | The two engines agree only up to their truncation errors. Per-gate retained norms differ, so near the strict norm guard one engine can raise "increase cutoff" while the other continues ([physical architecture](ml/physical/architecture.md)). Asserted in tests at cutoff 32 with peak width and envelope 0.9 and four peaks: decoded-probability difference < 1e-4 (1 wire) and < 2e-4 (2 wires); final-state infidelity < 2e-6; decoded bits, sampled output bits and empirical probabilities identical. |
| Tests | `tests/ml/cross_layer/test_native_backend.py` (`test_one_wire_conditional_matches_piquasso`, `test_one_wire_shots_sample_identically`, `test_two_wire_conditional_matches_piquasso`; reference: `piquasso-fock`). Each comparison now asserts which engine actually ran: the helper `_assert_engine` checks, for every trajectory of both runs, that `Result.backend` is `"PiquassoFockBackend"` or `"NativeFockBackend"` and that the state object comes from `piquasso` or `photographiq.backends.native_fock`. The old check read only `resource_config.backend`, which echoes the request and passed even when `NativeFockBackend` was swapped for `PiquassoFockBackend`. `tests/native/test_native_fock_cross_layer.py::test_physical_muta_conditional_run_matches_piquasso` (substitutes `NativeFockBackend` for `piquasso-fock` by monkeypatching, then asserts `physical_result.backend == "NativeFockBackend"` against `"PiquassoFockBackend"`) and `::test_physical_muta_difference_vanishes_with_cutoff` (the engine difference shrinks between cutoffs 24 and 32) |
| Status | **Implemented but experimental** (agreement with `piquasso-fock` is asserted, but the native Fock engine it runs on is labelled experimental, Section 13) |

```python
import numpy as np
from photographiqml import PhysicalMuTA, GKPPhysicalConfig
cfg = GKPPhysicalConfig(cutoff=32, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025,
                        backend="native-fock")
m = PhysicalMuTA(1, physical_config=cfg)
res = m.run([1, 0], [0, np.pi, 0, np.pi], mode="physical-shots", shots=2, seed=1)
print(res.physical_result.trajectories[0].backend, res.sampled_output_bits)
print(res.diagnostics["estimator"])
# (stderr also shows RuntimeWarning lines "GaussianTransform: retained Fock norm 0.99999...; test a larger cutoff")
# NativeFockBackend [(0,), (0,)]
# mean conditional joint POVM (Rao-Blackwell estimator)
```

### Arbitrary-angle physical MuTA

| Field | Value |
|---|---|
| Description | Physical execution or lowering with any measurement angle other than 0 or pi mod 2*pi. |
| Python API | `lower_muta_to_gkp`, `PhysicalMuTA.run` and `GKPBridge.run` raise `photographiqml.lowering.UnsupportedPhysicalMeasurement` (a `NotImplementedError` subclass) naming the first offending parameter. `PhysicalMuTA(measurement_family=...)` accepts only `"X"` (`ValueError`). `PhysicalMuTA.initialize(scale=...)` raises `ValueError`. |
| Implementation | `src/photographiqml/lowering.py`, `src/photographiqml/physical.py` |
| Backends | None |
| CPU / GPU | N/A |
| Assumptions | The rejection happens in the capability audit, before any Fock allocation. Validated arbitrary XY instruments are on the [roadmap](ml/ROADMAP.md). |
| Tests | `tests/ml/lowering/test_preflight.py::test_reject_every_angle_before_any_fock_allocation`; `tests/ml/cross_layer/test_physical_execution.py::test_physical_interface_boundaries` |
| Status | **Unsupported** |

```python
import numpy as np
from photographiqml import MuTA, GKPPhysicalConfig, lower_muta_to_gkp
cfg = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
try:
    lower_muta_to_gkp(MuTA(1, 1), [0, 0, 0, 0.3], config=cfg)
except NotImplementedError as e:
    print(type(e).__name__, str(e)[:80])
# UnsupportedPhysicalMeasurement Physical GKP MuTA cannot lower alpha.w0.c3=0.29999999999999999 rad: PhotoGraphiQ
```

### Other physical-path boundaries (entangled inputs, mixed Fock, soft decoder)

| Field | Value |
|---|---|
| Description | Three requests are rejected before any Fock allocation: entangled logical inputs (`NotImplementedError`), mixed-Fock backends (`ValueError` at config construction) and `decoder="soft"` for flow execution (`ValueError` from `require_supported`). |
| Python API | `PhysicalMuTA.run(...)`, `GKPPhysicalConfig(...)` |
| Implementation | `src/photographiqml/lowering.py` (`product_input`, `physical_capabilities`), `src/photographiqml/physical.py` |
| Backends | None |
| CPU / GPU | N/A |
| Assumptions | Correlated input preparation and joint code-subspace leakage are roadmap items. |
| Tests | `tests/ml/lowering/test_preflight.py::test_entangled_input_is_not_product_encoded` and `::test_resource_and_decoder_guards_preallocation`; `tests/ml/cross_layer/test_physical_execution.py::test_mixed_backend_fails_before_projection` |
| Status | **Unsupported** |

```python
import numpy as np
from photographiqml import PhysicalMuTA, GKPPhysicalConfig
cfg = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
m = PhysicalMuTA(2, physical_config=cfg)
try:
    m.run(np.array([1, 0, 0, 1]) / np.sqrt(2), np.zeros(m.n_parameters), shots=1)
except NotImplementedError as e:
    print("NotImplementedError:", e)
# NotImplementedError: Entangled physical input encoding is unsupported in v0.2; supply normalized product logical states
```

### Whole-model physical convergence study

| Field | Value |
|---|---|
| Description | Audits a complete sweep of one axis (`cutoff`, `grid_points`, `peaks`, `peak_width` or `envelope`) and then runs the physical model at each value. For each value it reports decoded probabilities, the change from the previous value, the logical comparison and the backend diagnostics. |
| Python API | `PhysicalMuTA.physical_convergence(input_state, values, **options)`; `photographiqml.physical.validate_physical_model(model, input_state, values, *, axis="cutoff", parameters=None, config=None, **run_options)` |
| Implementation | `src/photographiqml/physical.py` |
| Backends | The configured backend (`piquasso-fock` or `native-fock`) |
| CPU / GPU | CPU |
| Assumptions | `certified` is always `False`. `peak_width` and `envelope` are labelled "physical resource change", the other axes "numerical refinement". In shot mode the deltas include sampling error. `validate_physical_model` uses `config`, else `model.physical_config`, and builds the default `GKPPhysicalConfig()` only when the model has no `physical_config` (before the fix the default was always built). An unknown axis raises `ValueError("Unknown convergence axis")`; fewer than two or non-increasing values raise `ValueError`. |
| Tests | `tests/ml/cross_layer/test_physical_execution.py::test_independent_whole_model_study`; `tests/ml/cross_layer/test_native_backend.py::test_validate_physical_model_builds_default_config_only_when_needed` (the default is never built when the model or the call supplies a config; skips without the native extension or Piquasso) |
| Status | **Implemented but experimental** |

Example: see `examples/ml/tutorials/41_physical_convergence.py`.

### MuTA quantum kernel (`MuTAKernel`)

| Field | Value |
|---|---|
| Description | The paper's Eq. 5 feature map, realized through a two-wire one-column logical MuTA. The kernel is the squared overlap of the feature states, k(x, x') = abs(<phi(x), phi(x')>)^2, with Gram-matrix and PSD diagnostics. |
| Python API | `photographiqml.MuTAKernel()`; `.features(X)`, `kernel(X1, X2)` (the instance is callable, `MuTAKernel.__call__`), `.gram_matrix(X)`, `.diagnostics(X)` |
| Implementation | `src/photographiqml/kernels.py` |
| Backends | None (logical NumPy) |
| CPU / GPU | CPU |
| Assumptions | Inputs must have shape (N, 2). No implicit feature scaling. |
| Tests | `tests/ml/training/test_learning.py::test_kernel_equation_and_psd` (reference: the closed-form Rz/XX feature state); `tests/ml/mentpy_reference/test_derived_outputs.py::test_kernel_against_reference` (MentPy; skipped here) |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiqml import MuTAKernel
k = MuTAKernel(); X = np.array([[0.1, 0.2], [0.5, -0.3], [1.0, 0.7]])
print(np.round(k.gram_matrix(X), 6))
# [[1.       0.993792 0.785152]
#  [0.993792 1.       0.81675 ]
#  [0.785152 0.81675  1.      ]]
```

### Supervised wrappers (`MuTAClassifier`, `MuTARegressor`)

| Field | Value |
|---|---|
| Description | A product Ry angle encoding of classical features, a MuTA, the first-wire Z expectation and a trainable affine head (regressor: mean squared error; classifier: logistic loss). For a `PhysicalMuTA` the measurement angles are searched with `DiscreteSearch`, and only the classical head is fitted continuously (BFGS) on shot-estimated features. |
| Python API | `photographiqml.MuTARegressor(model, *, trainer=None, seed=0, physical_options=None)`; `photographiqml.MuTAClassifier(...)`; `.fit(X, y)`, `.predict(X)`, `.predict_proba(X)` (classifier), `.score(X, y)` (classifier); helpers `photographiqml.models.angle_encode(features)`, `photographiqml.models.haar_states(n_wires, n_samples, seed=0)`, `photographiqml.models.infidelity(predictions, targets)` |
| Implementation | `src/photographiqml/models.py` |
| Backends | Logical: NumPy. Physical: the `GKPPhysicalConfig` backend. |
| CPU / GPU | CPU |
| Assumptions | Physical models require `physical_options` with `mode="physical-shots"`, `shots` and `seed`, and require `DiscreteSearch` (passing `Trainer` raises `ValueError`). The physical features are stochastic estimates. Classifier labels must be 0 or 1. |
| Tests | `tests/ml/training/test_learning.py::test_classifier_and_regressor`; `tests/ml/cross_layer/test_discrete.py::test_supervised_physical_guards` and `::test_physical_supervised_head_real_shots`; `tests/ml/training/test_contracts.py::test_supervised_contract_does_not_broadcast_or_train_gkp` |
| Status | **Implemented but experimental** |

```python
import numpy as np
from photographiqml import MuTA, MuTAClassifier, Trainer
X = np.array([[0.1], [0.2], [2.9], [3.0]]); y = np.array([0, 0, 1, 1])
clf = MuTAClassifier(MuTA(1, 1), trainer=Trainer("adam", epochs=60, learning_rate=0.1)).fit(X, y)
print(clf.predict(X), clf.score(X, y))
# [0 0 1 1] 1.0
```

### Quantum instrument model

| Field | Value |
|---|---|
| Description | A logical MuTA followed by a destructive computational-basis measurement of one wire. It returns, for each outcome, the probability and the normalized post-measurement state of the remaining wires. |
| Python API | `photographiqml.QuantumInstrumentModel(model, measured_wire=0).run(input_state, parameters=None)` |
| Implementation | `src/photographiqml/models.py` |
| Backends | Logical NumPy only. A non-logical model raises `NotImplementedError`. |
| CPU / GPU | CPU |
| Assumptions | It does not reproduce the paper's learned teleportation instrument with variational intermediate controls (docstring). |
| Tests | `tests/ml/training/test_learning.py::test_instrument_and_qfi` (reference: closed-form branches of a Bell input, probabilities 1/2 and computational-basis post-measurement states); `tests/ml/mentpy_reference/test_derived_outputs.py::test_instrument_marginals_against_reference` (MentPy; skipped here) |
| Status | **Implemented and validated** |

```python
from photographiqml import MuTA, QuantumInstrumentModel
m = MuTA(2, 1, one_column=True); theta = m.initialize(seed=1, scale=0.5)
print([round(b.probability, 6) for b in QuantumInstrumentModel(m, 0).run([1, 0, 0, 0], theta)])
# [0.971925, 0.028075]
```

### Logical diagnostics (concurrence, QFI, Lie closure)

| Field | Value |
|---|---|
| Description | Pure two-qubit concurrence; pure-state QFI 4 Var(H); and the exact dimension of the real Lie algebra generated by i times the given Pauli strings (binary symplectic closure, 1 to 6 qubits). |
| Python API | `photographiqml.diagnostics.concurrence(state)`, `photographiqml.diagnostics.pure_qfi(state, generator)`, `photographiqml.expressivity.pauli_lie_dimension(generators)` |
| Implementation | `src/photographiqml/diagnostics.py`, `src/photographiqml/expressivity.py` |
| Backends | Logical NumPy |
| CPU / GPU | CPU |
| Assumptions | Concurrence is a qubit measure, not a CV entanglement measure. Pauli strings must have equal length, 1 to 6 qubits (`ValueError` otherwise). |
| Tests | `tests/ml/training/test_diagnostics.py::test_lie_algebra_regimes` (reference: known dimensions 3 = su(2), 6 = su(2)+su(2), 15 = su(4)); `tests/ml/muta/test_logical.py::test_triangle_and_table_one` (concurrence against closed-form abs(sin phi)); `tests/ml/training/test_learning.py::test_instrument_and_qfi` (QFI of a closed-form state = 1) |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiqml.diagnostics import pure_qfi
from photographiqml.expressivity import pauli_lie_dimension
print(round(pure_qfi(np.array([1, 1]) / np.sqrt(2), np.diag([1, -1]) / 2), 6))
print(pauli_lie_dimension(["XI", "ZZ"]), pauli_lie_dimension(["XX", "ZZ"]))
# 1.0
# 3 2
```

### Local state Fisher matrix (`state_fisher`)

| Field | Value |
|---|---|
| Description | The local pure-state QFI matrix 4 Re(<d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi>) of a logical MuTA output, from central finite-difference state derivatives. |
| Python API | `photographiqml.expressivity.state_fisher(model, input_state, parameters, step=1e-05)` |
| Implementation | `src/photographiqml/expressivity.py` |
| Backends | Logical NumPy |
| CPU / GPU | CPU |
| Assumptions | Finite differences (truncation error O(step^2)); `parameters` must be the full active vector. Its rank is a local numerical diagnostic, not an effective dimension or a layer-inclusion proof (docstring). |
| Tests | `tests/ml/training/test_diagnostics.py::test_fisher_psd_and_refinement` (symmetry, positive semidefiniteness, rank bound and agreement between steps 1e-5 and 1e-6; no independent reference) |
| Status | **Implemented but experimental** |

Example: see `tests/ml/training/test_diagnostics.py::test_fisher_psd_and_refinement` and `examples/ml/tutorials/24.py`.

### MentPy reference helpers

| Field | Value |
|---|---|
| Description | Builds the pinned MentPy MuTA template with an explicit semantic label map and compares density matrices from MentPy's `PatternSimulator` with `MuTA.run`. |
| Python API | `photographiqml.validation.mentpy_reference(model, *, fix_measurements=False)`, `photographiqml.validation.compare_mentpy(model, input_state, parameters=None)`, `photographiqml.validation.contract_branch(model, input_state, parameters=None, outcomes=None)` |
| Implementation | `src/photographiqml/validation.py` |
| Backends | MentPy `numpy-sv` (optional extra `ml-validation`, `mentpy==0.1.0a15`); `contract_branch` is pure NumPy and limited to 16 graph nodes |
| CPU / GPU | CPU |
| Assumptions | MentPy is not installed in the project environment used for this document: there the helpers raise `ImportError("MentPy validation requires photographiq[ml-validation]")`. On 2026-11-12 the comparison was run in an isolated Windows venv with MentPy 0.1.0a15 from the CI's exact commit `63c3d83e` (non-editable install, `numba` 0.67.0): see [PhotoGraphiQML MentPy validation](validation/photographiqml-mentpy-validation.md). Only full connectivity is supported (`ValueError`). `contract_branch` is itself a reference implementation used to check `MuTA.run`. |
| Tests | `tests/ml/muta/test_core_dependency.py::test_core_without_mentpy_import` and `::test_mentpy_helpers_name_the_install_extra` (missing-dependency path, run here); `tests/ml/mentpy_reference/` (34/34 passed in the isolated MentPy venv, together with 154/154 of `tests/ml`; skipped in the project venv; also run in `.github/workflows/ml-reference.yml`); `contract_branch` is exercised by `tests/ml/muta/test_logical.py::test_all_adaptive_branches`. Upgraded to validated on 2026-11-12 because the comparison against MentPy, an independent implementation, was executed and passed. |
| Status | **Implemented and validated** (against MentPy in an isolated environment; not re-run in the project venv) |

Example: see `examples/ml/tutorials/20.py` (requires MentPy).

### CV-native MuTA (CVMuTA)

| Field | Value |
|---|---|
| Description | A MuTA whose cells are continuous-variable gates rather than GKP-encoded qubits. |
| Python API | None. `MuTA(representation=...)` accepts no CV option (`ValueError` "... CVMuTA is a separate proposal"). |
| Implementation | None (`examples/ml/tutorials/29.py` only checks one Gaussian teleportation ingredient and prints "CVMuTA is a proposal") |
| Backends | None |
| CPU / GPU | N/A |
| Assumptions | Listed as item 3 of the [PhotoGraphiQML roadmap](ml/ROADMAP.md). |
| Tests | None |
| Status | **Planned** |

Example: none (not implemented).

## 29. Optimization algorithms

PhotoGraphiQML provides deterministic first-order and quasi-Newton optimizers for scalar objectives of a parameter vector, a central finite-difference gradient, and a categorical coordinate search for physical signed-X models. `photographiq.autodiff` provides optional JAX derivatives of a fixed-topology, fixed-outcome, finite-cutoff Fock model, together with two stochastic gradient-estimator primitives. Neither layer differentiates through sampled photonic trajectories.

### Adam

| Field | Value |
|---|---|
| Description | Bias-corrected Adam (beta1 = 0.9, beta2 = 0.999, eps = 1e-8) with a fixed learning rate. |
| Python API | `photographiqml.Trainer(optimizer="adam", epochs=100, learning_rate=0.05).fit(objective, initial, *, gradient=None, validation=None, callback=None) -> (x, History)` |
| Implementation | `src/photographiqml/training.py` |
| Backends | Any objective (logical MuTA objectives run in NumPy) |
| CPU / GPU | CPU (NumPy) |
| Assumptions | The objective is deterministic and scalar. The default gradient is a central finite difference. A non-finite objective or gradient raises `ValueError` before the update. `History` records losses, validation losses, gradient norms, parameter norms and elapsed seconds, starting at the initial point. |
| Tests | `tests/ml/training/test_learning.py::test_optimizer[adam]` and `::test_gradient_and_gate_learning` (reaches infidelity < 1e-4 on a gate-learning task); `tests/ml/training/test_contracts.py::test_nonfinite_training_fails_before_update` and `::test_callback_copy_and_validation_history` |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiqml import Trainer
x, h = Trainer("adam", epochs=200, learning_rate=0.1).fit(lambda x: float(np.sum((x - 1.0) ** 2)), [0.0, 3.0])
print(np.round(x, 4), len(h.losses))
# [1.     1.0001] 201
```

### Plain gradient descent (SGD)

| Field | Value |
|---|---|
| Description | x <- x - learning_rate * gradient, applied once per epoch on the full deterministic objective. |
| Python API | `photographiqml.Trainer(optimizer="sgd", epochs=..., learning_rate=...)` |
| Implementation | `src/photographiqml/training.py` |
| Backends | Any objective |
| CPU / GPU | CPU |
| Assumptions | Despite the name there is no mini-batching or sampling: it is deterministic full-batch gradient descent. Mini-batches are a [roadmap](ml/ROADMAP.md) item. |
| Tests | `tests/ml/training/test_learning.py::test_optimizer[sgd]` |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiqml import Trainer
x, h = Trainer("sgd", epochs=200, learning_rate=0.1).fit(lambda x: float(np.sum((x - 1.0) ** 2)), [0.0, 3.0])
print(np.round(x, 4), len(h.losses))
# [1. 1.] 201
```

### L-BFGS-B

| Field | Value |
|---|---|
| Description | SciPy `minimize(method="L-BFGS-B")` with `maxiter=epochs` and the trainer's gradient as the Jacobian. |
| Python API | `photographiqml.Trainer(optimizer="lbfgs", epochs=...)` |
| Implementation | `src/photographiqml/training.py` |
| Backends | Any objective |
| CPU / GPU | CPU (SciPy) |
| Assumptions | `learning_rate` is ignored. No bounds are passed. The history callback evaluates the objective and gradient again at each iterate, so there are extra evaluations beyond SciPy's own. |
| Tests | `tests/ml/training/test_learning.py::test_optimizer[lbfgs]` |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiqml import Trainer
x, h = Trainer("lbfgs", epochs=200).fit(lambda x: float(np.sum((x - 1.0) ** 2)), [0.0, 3.0])
print(np.round(x, 4), len(h.losses))
# [1. 1.] 3
```

### Central finite-difference gradient

| Field | Value |
|---|---|
| Description | (f(x + h e_i) - f(x - h e_i)) / (2h) for each coordinate. This is the default gradient of `Trainer`. |
| Python API | `photographiqml.training.finite_difference(objective, x, step=1e-06)` |
| Implementation | `src/photographiqml/training.py` |
| Backends | Any deterministic objective |
| CPU / GPU | CPU |
| Assumptions | Not exact: truncation error O(h^2) plus round-off of order eps/h. It needs 2n objective evaluations. It is meaningless on a shot-noise objective. |
| Tests | `tests/ml/training/test_learning.py::test_gradient_and_gate_learning` (reference: the two-term parameter-shift rule, computed in the test, agreement to 1e-8) |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiqml.training import finite_difference
print(finite_difference(lambda x: float(np.sin(x[0]) * x[1]), [0.3, 2.0]))
print(np.array([np.cos(0.3) * 2.0, np.sin(0.3)]))
# [1.91067298 0.29552021]
# [1.91067298 0.29552021]
```

### User-supplied gradient callable

| Field | Value |
|---|---|
| Description | `Trainer.fit(..., gradient=callable)` replaces the finite difference with any derivative the caller supplies, for example an analytic gradient or a JAX gradient of an `autodiff` objective. |
| Python API | `Trainer.fit(objective, initial, *, gradient=...)` |
| Implementation | `src/photographiqml/training.py` |
| Backends | Any |
| CPU / GPU | CPU (unless the supplied callable itself uses another device) |
| Assumptions | The trainer checks only shape and finiteness. Correctness of the derivative is the caller's responsibility. |
| Tests | `tests/ml/training/test_contracts.py::test_callback_copy_and_validation_history` (one SGD step with gradient 2x from x = 1 gives exactly 0.8; validation history [2, 1.28]); `::test_nonfinite_training_fails_before_update` |
| Status | **Implemented and validated** |

```python
import numpy as np, photographiqml as pqml
p, h = pqml.Trainer("lbfgs", epochs=20).fit(lambda x: np.sum(x * x), [1, 2], gradient=lambda x: 2 * x)
print(p.round(6))
# [0. 0.]
```

### Discrete coordinate search over {0, pi}

| Field | Value |
|---|---|
| Description | Coordinate search for `PhysicalMuTA`: starting from a categorical point, each sweep flips one angle at a time between 0 and pi and keeps the flip if the loss decreases. Every evaluation is recorded. |
| Python API | `photographiqml.DiscreteSearch(sweeps=1, seed=0).fit(model, objective, initial=None) -> photographiqml.physical_training.DiscreteSearchResult(parameters, loss, evaluations, seed)`; with `initial=None` the start point is `model.initialize(seed)` |
| Implementation | `src/photographiqml/physical_training.py` |
| Backends | The objective's own; physical objectives run on `piquasso-fock` or `native-fock` |
| CPU / GPU | CPU |
| Assumptions | A local search, not a global optimizer. With a shot-based objective the loss is a stochastic estimate: reusing seeds helps comparisons but does not remove estimator noise, so the selected candidate should be re-checked at independent seeds (docstring). It never rounds a continuous solution. The initial vector must contain exact 0/pi values (`ValueError`), and the model must be a `PhysicalMuTA`. |
| Tests | `tests/ml/cross_layer/test_discrete.py::test_discrete_coordinate_search` (finds the known minimum of a closed-form categorical objective; 9 evaluations for 2 sweeps over 4 angles) |
| Status | **Implemented but experimental** |

```python
import numpy as np
from photographiqml import DiscreteSearch, PhysicalMuTA
target = np.array([np.pi, 0, 0, np.pi])
res = DiscreteSearch(sweeps=1, seed=0).fit(PhysicalMuTA(1), lambda x: float(np.sum(x != target)),
                                           initial=np.zeros(4))
print(res.parameters.tolist(), res.loss, len(res.evaluations))
# [3.141592653589793, 0.0, 0.0, 3.141592653589793] 0.0 5
```

### Continuous optimization of physical measurement angles

| Field | Value |
|---|---|
| Description | Gradient-based training of physical signed-X angles is not allowed, because the angles are categorical. |
| Python API | `MuTARegressor(PhysicalMuTA(...), trainer=Trainer())` raises `ValueError("Physical angles require DiscreteSearch; Adam cannot optimize categorical measurements")` |
| Implementation | `src/photographiqml/models.py` |
| Backends | None |
| CPU / GPU | N/A |
| Assumptions | The supported workflow trains a logical model, then validates or searches the physical model ([physical training](ml/physical/physical-training.md)). |
| Tests | `tests/ml/cross_layer/test_discrete.py::test_supervised_physical_guards` |
| Status | **Unsupported** |

```python
from photographiqml import PhysicalMuTA, MuTARegressor, Trainer
try:
    MuTARegressor(PhysicalMuTA(1), trainer=Trainer())
except ValueError as e:
    print("ValueError:", e)
# ValueError: Physical angles require DiscreteSearch; Adam cannot optimize categorical measurements
```

### JAX automatic differentiation of fixed-branch Fock patterns

| Field | Value |
|---|---|
| Description | A separate differentiable executor that rebuilds a validated `Pattern` with dense projected generators in JAX. It computes the conditional state of a fixed measurement branch, the branch log-likelihood and the expectations of n, parity, q or p. `jax.grad` gives exact derivatives of this truncated finite-cutoff model, which are not exact derivatives of the infinite-dimensional physics. |
| Python API | `photographiq.autodiff.fock_state(pattern, parameters=None, *, cutoff, inputs=None, measurement_outcomes=None, max_dimension=512) -> DifferentiableState`; `photographiq.autodiff.expectation(pattern, parameters=None, *, observable="photon_number", node=None, **kwargs)` |
| Implementation | `src/photographiq/autodiff.py` |
| Backends | Its own JAX dense-Fock path. It is not a `simulate()` backend, and NumPy `simulate()` is not differentiable. Every preparation (inputs and `Prepare`) goes through the Piquasso-backed `MixedFockBackend` to build the constant initial density matrix, so Piquasso must be installed as well. |
| CPU / GPU | CPU (JAX). JAX on GPU is not configured or tested by the project. |
| Assumptions | Optional extra `autodiff` (`jax>=0.4.35` and `piquasso==8.0.1`). Piquasso is a real dependency: every input and `Prepare`/`PrepareResource` state is built through `MixedFockBackend`, and without Piquasso `fock_state`/`expectation` raise its `ImportError`. The extra has included Piquasso since 2026-11-11, and the module docstring says so. JAX is imported only when a differentiable operation is called (`ImportError` naming the extra otherwise). `jax_enable_x64` must be enabled (`RuntimeError` otherwise). Total-photon cutoff, fixed topology, and an explicit outcome for every measurement. Supported operations: Gaussian gates, cubic phase, Kerr, photon addition and subtraction, vacuum loss, ideal homodyne and PNR. Thermal loss, noisy homodyne, other measurements and Python-callable expressions raise. Zero-probability branches raise eagerly and give NaN under tracing. Known limitation: `autodiff.resolve` evaluates expressions with its own JAX operator table, so it stays traceable but does not apply `Expr.evaluate`'s finiteness check. An overflowing, infinite, NaN or zero-division expression gives `inf` or `nan` instead of `ValueError("Expression must evaluate to a finite real scalar")`. |
| Tests | `tests/autodiff/test_analytical_gradients.py` (reference: closed-form gate gradients for rotation, displacement, squeezing, Kerr and cubic phase, to 2e-8 or 2e-12); `tests/autodiff/test_autodiff_execution.py::test_projected_gates_match_native_at_low_energy` (Piquasso backends); `tests/autodiff/test_finite_difference_convergence.py`; `tests/autodiff/test_postselection_gradients.py`; `tests/autodiff/test_boundary_regressions.py`; `tests/autodiff/test_gradients.py` |
| Status | **Implemented but experimental** |

```python
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np, photographiq as pg
from photographiq import autodiff as ad
p = pg.Pattern(inputs=(0,)).append(pg.Rotate(0, pg.Parameter("t")))
src = pg.FockInput((2**-0.5, 2**-0.5))
f = lambda t: ad.expectation(p, {"t": t}, inputs={0: src}, cutoff=8, observable="q")
print(float(f(0.3)), float(jax.grad(f)(0.3)), -np.sin(0.3))
# 0.9553364891256059 -0.2955202066613395 -0.29552020666133955
```

### Batched parameter expectations (`vmap`)

| Field | Value |
|---|---|
| Description | `jax.vmap` of `expectation` over rows of parameter values for a fixed pattern. |
| Python API | `photographiq.autodiff.parameter_batch_expectation(pattern, parameter_names, parameter_values, **kwargs)` |
| Implementation | `src/photographiq/autodiff.py` |
| Backends | JAX dense-Fock path |
| CPU / GPU | CPU (JAX) |
| Assumptions | The names must bind every pattern parameter exactly once. It covers `expectation` only: no sampling, decoders or custom instruments. |
| Tests | `tests/autodiff/test_parameter_batch.py::test_batch_and_gradient_against_rotation_formula` (reference: closed-form rotation formula) |
| Status | **Implemented but experimental** |

```python
import jax
jax.config.update("jax_enable_x64", True)
import photographiq as pg
from photographiq import autodiff as ad
p = pg.Pattern(inputs=(0,)).append(pg.Rotate(0, pg.Parameter("t")))
print(ad.parameter_batch_expectation(p, ["t"], [[0.0], [0.3]],
      inputs={0: pg.FockInput((2**-0.5, 2**-0.5))}, cutoff=8, observable="q"))
# [1.         0.95533649]
```

### Score-function (likelihood-ratio) gradient estimator

| Field | Value |
|---|---|
| Description | Returns a surrogate loss whose forward value is mean(f) and whose JAX gradient is the REINFORCE estimator mean((f - b) * grad log p) + mean(grad f), for samples drawn from the parameter-dependent distribution. |
| Python API | `photographiq.autodiff.score_function_surrogate(values, log_probabilities, *, baseline=0.0)` |
| Implementation | `src/photographiq/autodiff.py` |
| Backends | JAX (backend-independent primitive) |
| CPU / GPU | CPU (JAX) |
| Assumptions | Stochastic. It is unbiased only if the samples come from the parameter-dependent distribution and the baseline does not depend on each sampled outcome. The caller supplies full joint log-probabilities, including heralding. It is not wired into `simulate` or `run_shots`. |
| Tests | `tests/autodiff/test_estimators.py::test_bernoulli_score_statistical_bias_variance_and_seed` (reference: the exact Bernoulli derivative, within 5 standard errors); `tests/autodiff/test_autodiff_execution.py::test_score_function_expected_gradient` |
| Status | **Implemented but experimental** |

```python
import jax, numpy as np
jax.config.update("jax_enable_x64", True)
from photographiq import autodiff as ad
s = np.random.default_rng(0).binomial(1, 0.3, 20000).astype(float)
loss = lambda t: ad.score_function_surrogate(s, s * jax.numpy.log(t) + (1 - s) * jax.numpy.log1p(-t), baseline=0.2)
print(round(float(jax.grad(loss)(0.3)), 4), "exact d E[x]/dt = 1")
# 1.0049 exact d E[x]/dt = 1
```

### Pathwise (reparameterized Gaussian) estimator

| Field | Value |
|---|---|
| Description | `mean + cholesky(covariance) @ standard_normal`, differentiable in the mean and covariance for fixed base noise. |
| Python API | `photographiq.autodiff.gaussian_sample(mean, covariance, standard_normal)` |
| Implementation | `src/photographiq/autodiff.py` |
| Backends | JAX |
| CPU / GPU | CPU (JAX) |
| Assumptions | Stochastic Monte Carlo estimator. The covariance must be positive definite. It does not make discrete photon counts differentiable. |
| Tests | `tests/autodiff/test_estimators.py::test_pathwise_gaussian_bias_and_reproducibility` (reference: the exact derivatives of E[x^2], within 5 standard errors) |
| Status | **Implemented but experimental** |

```python
import jax, numpy as np
jax.config.update("jax_enable_x64", True)
from photographiq import autodiff as ad
eps = np.random.default_rng(1).normal(size=20000)
g = jax.grad(lambda mu: jax.numpy.mean(jax.vmap(lambda e: ad.gaussian_sample(
    jax.numpy.array([mu]), jax.numpy.eye(1), jax.numpy.array([e]))[0] ** 2)(eps)))(0.4)
print(round(float(g), 4), "exact = 0.8")
# 0.7774 exact = 0.8   (Monte Carlo estimate; standard error about 0.014)
```

### Full stochastic MBQC differentiation

| Field | Value |
|---|---|
| Description | Differentiating through sampled measurement trajectories of `simulate` / `run_shots`, through decoders or through physical MuTA. |
| Python API | None. `autodiff.fock_state` requires fixed outcomes for every measurement (`ValueError` otherwise). `autodiff.resolve` raises `TypeError` for Python callables, such as the frame signals of lowered physical MuTA. |
| Implementation | None |
| Backends | None |
| CPU / GPU | N/A |
| Assumptions | The [feature matrix](validation/feature-matrix.md) lists "Full stochastic MBQC differentiation" as unsupported. Valid autodiff and stochastic estimators for training are item 4 of the [PhotoGraphiQML roadmap](ml/ROADMAP.md). |
| Tests | `tests/autodiff/test_autodiff_execution.py::test_autodiff_rejects_undefined_execution_contracts` |
| Status | **Unsupported** |

Example: see `tests/autodiff/test_autodiff_execution.py::test_autodiff_rejects_undefined_execution_contracts`.

## 30. Interactive visualization

Covers `photographiq.interactive`.

`photographiq.interactive` shows measurement-based computations as staged, immutable views of an `OpenGraph`: construct, flow, corrections, execution. It also provides scientific plots. Python is the authoritative model: every number is computed by PhotoGraphiQ, and the browser only displays and edits. HTML pages load Cytoscape.js 3.30.2 or Plotly.js 2.35.2 from pinned jsDelivr CDN URLs, or inline a local copy for offline use. The Python `plotly` package is not needed. Static figures use Matplotlib and need no browser. In this section, "validated" means the tests compare what is displayed with an independent PhotoGraphiQ computation (certified flow, simulator records, closed-form physics). See the [interactive visualization guide](user-guide/interactive.md).

**Real-browser verification (2026-11-12).** `scripts/check_interactive_browser.py OUT_DIR [--channel msedge|chrome]` drives the pages with real mouse, wheel and keyboard input through Playwright (1.65.1, using the installed Edge 157 or Chrome; no browser download), takes screenshots and compares what is drawn with the Python model, the flow result, the `Pattern` commands and a real `simulate` trace. It writes `*.html`, `*.png` and `report.json` and exits 1 on any failure; `tests/visualization/test_interactive_browser_playwright.py` runs the same checks and skips without Playwright, a browser or Cytoscape.js. All 9 checks pass (node dragging, panning and zooming, edit round trip through `apply_edits`, construction/flow/corrections stages, execution animation, layout scale on small graphs and a 6 x 6 cluster, static SVG/PDF/PNG exports) after four fixes in `src/photographiq/interactive/`: correction-arc labels are no longer struck through by longer arcs (shorter arcs drawn on top, opaque label backgrounds); recorded outcomes of measured nodes are no longer drawn at 50 % opacity; the Selection panel refreshes after an edit; static SVGs declare a sans-serif fallback after DejaVu Sans, so labels no longer fall back to Times. Remaining limits: straight edges and fixed-bend arcs can pass under other nodes (a dependency 0 -> 2 under node 1 reads like 0 -> 1 -> 2; an edge between collinear nodes is hidden), three-line outcome labels overflow the 42 px node, the flow panel can push the execution controls below the fold at 1280 x 800, labels on the 6 x 6 cluster are 8.8 px at default zoom, static `#k` order labels touch correction arcs on the CV wire, overlays are not recomputed after an edit, and touch input, Firefox/WebKit, high-DPI rendering and the Plotly plot pages were not covered by the real-browser check. Evidence and screenshots: [interactive visualization browser check](validation/interactive-visualization-browser.md). The script needs Playwright, which is not a declared extra and is not installed in the project venv, so it was not re-run for this document.

### Staged graph view (`GraphView`)

| Field | Value |
|---|---|
| Description | Immutable view of an `OpenGraph` with four stages: `construct` (vertices, weighted CZ edges, input/output/measured/ancilla roles, squeezing and measurement metadata), `flow`, `corrections` and `execution`. Each `with_*` call returns a new view. It renders to a self-contained HTML page (Cytoscape.js) or a notebook iframe. |
| Python API | `photographiq.interactive.GraphView(graph, layout=None, stage="construct", ..., title="PhotoGraphiQ graph")`; `.construct()`, `.with_layout(layout="publication", **options)`, `.elements(stage=None)`, `.payload(stage=None, *, editable=True)`, `.to_html(stage=None, *, editable=True, cytoscape_js=None)`, `.save_html(path, stage=None, **options)`, `.show(stage=None, **options)` (options are passed to `to_html`), `._repr_html_()`; `photographiq.interactive.STAGES` |
| Implementation | `src/photographiq/interactive/graph_view.py`, `src/photographiq/interactive/_template.py`, `src/photographiq/interactive/_html.py` |
| Backends | Independent of backends (display of `OpenGraph`) |
| CPU / GPU | CPU only (Python); rendering in the viewer's browser |
| Assumptions | Requesting a stage whose overlay is missing raises `ValueError`. Node labels whose `str()` collide raise `ValueError`. JSON embedded in `<script>` is escaped against `</script>` injection. `show()` writes a temporary `.html` file and opens the default browser. In notebooks the page is shown as a `srcdoc` iframe. |
| Tests | `tests/visualization/test_interactive_graph.py` (`test_construct_stage_shows_graph_only`, `test_html_is_injection_safe_and_offline_capable`, `test_generated_javascript_parses` (Node.js `--check`)); `tests/visualization/test_interactive_browser.py::test_editor_and_animation_in_browser` (headless Edge/Chrome); `tests/visualization/test_interactive_notebook.py::test_notebook_executes` (`examples/notebooks/interactive_graph_visualization.ipynb`); `tests/visualization/test_interactive_browser_playwright.py` and `scripts/check_interactive_browser.py` (real mouse and keyboard input in Edge, 2026-11-12; skip without Playwright) |
| Status | **Implemented and validated** |

```python
import photographiq as pg
from photographiq.flows import analyze
from photographiq.interactive import GraphView
from photographiq.ir import OpenGraph
graph = pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,))
open_graph = OpenGraph.from_cvgraph(graph, {0: 0.3, 1: -0.5, 2: 0.0})
view = GraphView(open_graph, title="Four-node CV wire").construct()
html = view.to_html()
print("cdn.jsdelivr.net/npm/cytoscape@3.30.2" in html)
# True
```

### Pattern and CVGraph conversion

| Field | Value |
|---|---|
| Description | Builds the drawn resource graph from a `CVGraph` (with wire-step homodyne shears) or from a `Pattern`. Anything that cannot be drawn faithfully is reported as a note or a node annotation instead of being approximated. |
| Python API | `GraphView.from_cvgraph(graph, shears=None, **options)`, `GraphView.from_pattern(pattern, *, parameters=None, **options)`; `photographiq.interactive.measurement_text(spec)` |
| Implementation | `src/photographiq/interactive/graph_view.py` (`_pattern_graph`, `_wire_step`) |
| Backends | Independent of backends |
| CPU / GPU | CPU only |
| Assumptions | Repeated CZs on one edge are aggregated into one weight, with a note. If other operations act on the endpoints in between, the note says the aggregate need not equal the executed operation. Cancelling CZs draw no edge. Beam splitters are listed, not drawn. Homodyne angles that are not wire-step shears (atan2(1, k) to 1e-12 rad), adaptive angles and non-homodyne measurements are annotated. The model is `CV_GAUSSIAN` only if every command is Gaussian. |
| Tests | `tests/visualization/test_interactive_graph.py` (`test_pattern_conversion_reports_what_it_cannot_draw`, `test_repeated_cz_conversion_is_never_silent`, `test_nonlinear_feed_forward_is_rejected`) |
| Status | **Implemented and validated** |

Example: see `tests/visualization/test_interactive_graph.py::test_pattern_conversion_reports_what_it_cannot_draw`.

### Layouts

| Field | Value |
|---|---|
| Description | Deterministic 2-D node positions: grid, circular, hierarchical (breadth-first depth from the inputs), layered (by flow layers or measurement order), force (method name `"force"`, function `force_directed`; NetworkX spring layout, seeded), publication (aligned integer grid with barycentre ordering), manual. Positions can be saved and loaded as JSON. |
| Python API | `photographiq.interactive.compute_layout(graph, method="publication", **options)`, `grid(graph, columns=None)`, `circular(graph)`, `hierarchical(graph)`, `layered(graph, *, order=None, layers=None)`, `force_directed(graph, *, seed=0, iterations=200)`, `publication(graph, *, order=None, layers=None, sweeps=4)`, `manual(graph, positions)`, `Layout(positions)` (`to_dict`, `to_json`, `from_dict`, `from_json`, `require`), `LAYOUTS`; `export_layout(layout, path=None)`, `import_layout(source, graph)` |
| Implementation | `src/photographiq/interactive/layouts.py`, `src/photographiq/interactive/editor.py` |
| Backends | Independent of backends |
| CPU / GPU | CPU only |
| Assumptions | Coordinates are display metadata and never change graph semantics. The y axis grows upward. The publication layout reduces edge crossings but does not minimise them. Non-finite positions, missing nodes and an unknown method name (for example `"force_directed"`) raise `ValueError`. Positions for nodes absent from the graph raise `ValueError` in `import_layout`/`apply_edits`, but `manual` and `GraphView(layout=...)` accept them silently. Node labels whose `str()` collide raise `ValueError`, because JSON keys are `str(node)`. The hierarchical layout accepts graphs whose disconnected components have labels that cannot be compared with each other (for example `"a"` next to `0` and `1`). |
| Tests | `tests/visualization/test_interactive_graph.py` (`test_layouts_are_deterministic_and_semantics_free`, `test_force_layout_seed_and_layered_columns`, `test_tuple_labels_and_layout_files`, `test_hierarchical_layout_accepts_mixed_label_types`); `tests/visualization/test_interactive_browser_playwright.py` and `scripts/check_interactive_browser.py` (real mouse and keyboard input in Edge, 2026-11-12; skip without Playwright): no node or label boxes overlap on the 4-node wire, the 6-node qubit graph and the 6 x 6 cluster (8.8 px labels at default zoom) |
| Status | **Implemented but experimental** (tests check determinism, coverage and that semantics are unchanged; layout quality has no reference, and edges can pass under nodes) |

```python
import photographiq as pg
from photographiq.interactive import compute_layout
from photographiq.ir import OpenGraph
open_graph = OpenGraph.from_cvgraph(pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,)), {})
print(compute_layout(open_graph, "grid").positions[1])
# (1.0, 0.0)
```

### Flow overlay

| Field | Value |
|---|---|
| Description | Stage B: displays a flow analysis exactly as reported. It shows classical dependency arrows from the `correction_map`, the measurement order and layers, the certification status, the algorithm and the assumptions. |
| Python API | `GraphView.with_flow(result)`, `GraphView.flow_arrows()`; `photographiq.interactive.FlowOverlay.from_result(result, graph)` (accepts a `photographiq.flows.FlowAnalysisResult`, a dict or any object with the same attributes) |
| Implementation | `src/photographiq/interactive/overlays.py` (`FlowOverlay`), `src/photographiq/interactive/graph_view.py` |
| Backends | Independent of backends |
| CPU / GPU | CPU only |
| Assumptions | No flow is derived here. A model mismatch between the result and the graph, or unknown nodes in the order, layers or corrections, raises `ValueError`. Outcome keys that are not graph nodes are listed in the notes without an arrow. |
| Tests | `tests/visualization/test_interactive_graph.py` (`test_flow_arrows_equal_reported_dependencies`, `test_flow_result_is_checked_against_graph`, `test_real_flow_analysis_overlays` (output of `photographiq.flows.analyze`)); `tests/visualization/test_interactive_browser_playwright.py` and `scripts/check_interactive_browser.py` (real mouse and keyboard input in Edge, 2026-11-12; skip without Playwright): drawn dependency arrows equal `flow_arrows()` and the node layers equal `flow.layers` |
| Status | **Implemented and validated** |

```python
import photographiq as pg
from photographiq.flows import analyze
from photographiq.interactive import GraphView
from photographiq.ir import OpenGraph
open_graph = OpenGraph.from_cvgraph(pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,)),
                                    {0: 0.3, 1: -0.5, 2: 0.0})
flow = analyze(open_graph)
print(flow.status, flow.measurement_order)
# certified (0, 1, 2)
print(sorted(GraphView(open_graph).with_flow(flow).flow_arrows()))
# [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
```

### Correction overlay

| Field | Value |
|---|---|
| Description | Stage C: per-node correction formulas and labelled arrows. Qubit graphs show X/Z domains of earlier outcomes. CV graphs show displacements as linear combinations of raw records m[k]. Sources are a flow result, a correction map, or a `Pattern`, whose outcome-dependent `Displace` commands are what it actually executes. |
| Python API | `GraphView.with_corrections(source=None, *, parameters=None)`, `GraphView.correction_arrows()`; `photographiq.interactive.pattern_corrections(pattern, parameters=None)`, `normalize_corrections(correction_map, graph)`, `Correction(target, kind, x, z, q, p, q0, p0)` with `.formula()`, `.terms()`, `.evaluate(records)` |
| Implementation | `src/photographiq/interactive/overlays.py` |
| Backends | Independent of backends |
| CPU / GPU | CPU only |
| Assumptions | CV flow coefficients refer to m / sin(theta), with theta = atan2(1, k). The overlay multiplies them by sqrt(1 + k^2), so they match raw records in patterns and traces. Pattern feed-forward must be affine in scalar outcomes, checked at random points: non-affine or opaque `CallableExpression` feed-forward raises `ValueError`, and non-scalar outcomes raise `NotImplementedError`. A bare number as a correction raises `TypeError`. |
| Tests | `tests/visualization/test_interactive_graph.py` (`test_correction_map_normalization`, `test_cv_corrections_equal_certified_flow` (against `photographiq.cvflow.certify_cv_flow`), `test_nonlinear_feed_forward_is_rejected`); `tests/visualization/test_interactive_browser_playwright.py` and `scripts/check_interactive_browser.py` (real mouse and keyboard input in Edge, 2026-11-12; skip without Playwright): drawn arrows and labels equal `pattern_corrections(pattern)` and the outcome-dependent `Displace` commands of the `Pattern`, and shorter arcs are drawn above longer ones (fix F1) |
| Status | **Implemented and validated** |

```python
import photographiq as pg
from photographiq.flows import analyze
from photographiq.interactive import GraphView
from photographiq.ir import OpenGraph
open_graph = OpenGraph.from_cvgraph(pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,)),
                                    {0: 0.3, 1: -0.5, 2: 0.0})
view = GraphView(open_graph).with_flow(analyze(open_graph)).with_corrections()
for target, c in view.corrections.items():
    print(target, c.formula())
# 1 q += -1.044 m[0]
# 2 q += -1.118 m[1]; p += -1.044 m[0]
# 3 q += -1 m[2]; p += -1.118 m[1]
```

### Execution trace and animation

| Field | Value |
|---|---|
| Description | Stage D: the ordered steps of **one recorded trajectory**, built from the real pattern commands and a simulator `Result`. Measurement steps carry the recorded outcomes, and feed-forward arguments are evaluated from `Result.records` as `simulate` does. The HTML page animates the steps with Play, Pause, Step and Reset. |
| Python API | `photographiq.interactive.execution_trace(pattern, result, *, parameters=None)` returning `ExecutionTrace(steps, backend, seed, outcomes, physical_displacements)` (`.postselected`, `.to_dict()`), `TraceStep(index, kind, nodes, label, values, depends_on, outcome_key, outcome, statistic)`; `GraphView.with_execution(trace)` |
| Implementation | `src/photographiq/interactive/trace.py`, `src/photographiq/interactive/_template.py` |
| Backends | Any backend whose `Result` records the pattern's measurements |
| CPU / GPU | CPU only |
| Assumptions | Postselected measurements are labelled `[postselected: conditional branch]`, and the page notes "this is one conditional branch, not a deterministic run". With `frame=True` the trace shows the command arguments, not the moment the deferred frame was physically applied. A `Result` that does not record exactly the pattern's measurement keys raises `ValueError`. A trace with nodes absent from the graph is rejected by `with_execution`. |
| Tests | `tests/visualization/test_interactive_graph.py` (`test_execution_trace_follows_simulator` (simulator records), `test_postselected_branch_is_labelled`); `tests/visualization/test_interactive_browser.py::test_editor_and_animation_in_browser`; `tests/visualization/test_interactive_browser_playwright.py` and `scripts/check_interactive_browser.py` (real mouse and keyboard input in Edge, 2026-11-12; skip without Playwright): each step label equals the trace step and each displayed outcome equals `Result.outcomes` to the displayed digits; outcome text stays opaque (fix F2) |
| Status | **Implemented and validated** |

```python
import photographiq as pg
from photographiq.cvflow import flow_pattern
from photographiq.interactive import execution_trace
graph = pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,))
pattern = flow_pattern(graph, (0, 1, 2), shears={0: 0.3, 1: -0.5, 2: 0.0})
result = pg.simulate(pattern, backend="gaussian", seed=7)
trace = execution_trace(pattern, result)
print(len(trace.steps), [s.label for s in trace.steps if s.kind == "measure"][0])
# 15 Measure[Homodyne] 0 (angle=1.279) -> 0 = 0.005422
post = pg.simulate(pattern, backend="gaussian", seed=7, measurement_outcomes={0: 0.0})
print([s.label for s in execution_trace(pattern, post).steps if s.postselected][0])
# Measure[Homodyne] 0 (angle=1.279) -> 0 = 0 [postselected: conditional branch]
```

### Browser editor and edit round trip

| Field | Value |
|---|---|
| Description | The HTML page (`editable=True`) can add and remove nodes, add edges and set weights, toggle inputs and outputs, and set squeezing, shear or qubit measurements. **Export JSON** downloads `{"graph": OpenGraph.to_dict(), "layout": {...}}`, which Python re-validates. |
| Python API | `GraphView.to_html(editable=True)`; `photographiq.interactive.apply_edits(data)` returning `(OpenGraph, Layout)` (JSON text, file path or dict) |
| Implementation | `src/photographiq/interactive/_template.py` (JavaScript editor), `src/photographiq/interactive/editor.py` |
| Backends | Independent of backends |
| CPU / GPU | CPU only |
| Assumptions | Edits in the browser are not trusted. `apply_edits` rebuilds the graph through `OpenGraph.from_dict`, so an invalid edit (zero weight, weighted qubit edge, measured output, unknown node) raises as it would in Python. JSON lists used as labels are turned back into tuples. |
| Tests | `tests/visualization/test_interactive_graph.py` (`test_apply_edits_round_trip_and_revalidation`, `test_tuple_labels_and_layout_files`); `tests/visualization/test_interactive_browser.py::test_editor_and_animation_in_browser` (drives the editor in headless Edge/Chrome; skips without a browser or offline); `tests/visualization/test_interactive_browser_playwright.py` and `scripts/check_interactive_browser.py` (real mouse and keyboard input in Edge, 2026-11-12; skip without Playwright): edits made with real clicks and typing round-trip through `apply_edits`, an invalid edit is rejected, and the Selection panel refreshes after an edit (fix F3) |
| Status | **Implemented and validated** |

```python
import json
import photographiq as pg
from photographiq.interactive import GraphView, apply_edits
from photographiq.ir import OpenGraph
view = GraphView(OpenGraph.from_cvgraph(pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,)), {}))
exported = {"graph": view.graph.to_dict(), "layout": view.layout.to_dict()}
edited, layout = apply_edits(json.dumps(exported))
print(edited.to_dict() == view.graph.to_dict())
# True
exported["graph"]["edges"][0]["weight"] = 0
try:
    apply_edits(exported)
except ValueError as exc:
    print(exc)
# Edge (0, 1) needs a finite nonzero weight
```

### Static export (SVG, PDF, PNG)

| Field | Value |
|---|---|
| Description | Matplotlib rendering of any stage, with the same styling as the HTML view: solid grey CZ edges with weight labels, dashed blue dependency arrows, dotted orange correction arrows with coefficients, square inputs, diamond outputs, circular measured nodes. The format is chosen from the file suffix. |
| Python API | `GraphView.save(path, stage=None, **options)`; `photographiq.interactive.draw_static(view, path=None, stage=None, *, ax=None, figsize=None, dpi=300, formulas=True, title=None)` |
| Implementation | `src/photographiq/interactive/static.py` |
| Backends | Independent of backends |
| CPU / GPU | CPU only (Matplotlib; no browser, no pyplot state when saving) |
| Assumptions | Requires Matplotlib. Text stays text in SVG (`svg.fonttype: none`), and PDF uses Type 42 fonts. Without `path`, the Axes is returned. |
| Tests | `tests/visualization/test_interactive_graph.py` (`test_static_exports_are_valid_files` (SVG parsed as XML, PDF/PNG signatures, presence of edge and correction elements), `test_static_flow_export`); `tests/visualization/test_interactive_browser_playwright.py` and `scripts/check_interactive_browser.py` (real mouse and keyboard input in Edge, 2026-11-12; skip without Playwright): the SVG rendered in Edge, the PDF rendered with `pdftoppm` and the PNG were inspected, and the SVG font declaration ends with a generic `sans-serif` fallback (fix F4) |
| Status | **Implemented but experimental** (automated tests check file validity, element presence and the font fallback; readability was checked by inspection, which is not an independent reference) |

```python
import tempfile, pathlib
import photographiq as pg
from photographiq.cvflow import flow_pattern
from photographiq.interactive import GraphView
graph = pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,))
pattern = flow_pattern(graph, (0, 1, 2), shears={0: 0.3, 1: -0.5, 2: 0.0})
view = GraphView.from_pattern(pattern).with_corrections(pattern)
out = pathlib.Path(tempfile.mkdtemp())
print(view.save(out / "wire.svg", stage="corrections").name)
# wire.svg
```

### State plots: photon number, quadrature, Wigner

| Field | Value |
|---|---|
| Description | Bar chart of a mode's photon-number marginal, with the retained truncated mass in the title. Ideal homodyne density of q cos t + p sin t, optionally overlaid on a normalised histogram of recorded outcomes. Signed Wigner heatmap with the grid mass and negative volume in the title. |
| Python API | `photographiq.interactive.photon_number_marginal(state, node)`, `photon_number_distribution(state, node)`, `quadrature_density(state, node, angle, x)`, `quadrature_distribution(state=None, node=None, angle=0.0, *, samples=None, x=None, bins=40)`, `gaussian_wigner(state, q, p, node=None)`, `wigner_plot(state, node=None, q=None, p=None, *, points=121, extent=5.0)` |
| Implementation | `src/photographiq/interactive/plots.py` (uses `src/photographiq/fock_analysis.py`, `src/photographiq/fock_measurements.py`) |
| Backends | Photon numbers: Fock-backend states (`native-fock`, `piquasso-fock`, mixed-Fock), otherwise `NotImplementedError("Photon-number distributions need a Fock state (native-fock, native-mixed-fock, piquasso-fock or piquasso-mixed-fock backend)")`. Quadrature and Wigner plots: Gaussian and Fock states |
| CPU / GPU | CPU only (NumPy; rendering by Plotly.js or Matplotlib) |
| Assumptions | hbar=2, q = a + a^dag, vacuum covariance I. The Gaussian Wigner function is the closed form normalised to 1 over dq dp. The Fock quadrature density is exact for the truncated state. Detector inefficiency is not included in the ideal density, but a histogram of recorded `samples` includes any detector noise. `photon_number_marginal` on a state without Fock probabilities (for example a `GaussianState`) raises `NotImplementedError("Photon-number distributions need a Fock state (native-fock, native-mixed-fock, piquasso-fock or piquasso-mixed-fock backend)")`; before the fix it named only the Piquasso backends. |
| Tests | `tests/visualization/test_interactive_plots.py` (`test_photon_numbers_are_poisson` (Poisson closed form), `test_fock_quadrature_density_is_the_shifted_vacuum` (closed form), `test_quadrature_histogram_uses_recorded_outcomes`, `test_wigner_closed_form_matches_fock` (Gaussian closed form vs Fock Wigner)); `tests/regression/test_validation_messages.py::test_photon_counting_messages_mention_native_fock_backends` (the error message) |
| Status | **Implemented and validated** |

```python
import math
import photographiq as pg
from photographiq.interactive import photon_number_marginal, photon_number_distribution
state = pg.simulate(pg.Pattern(inputs=(0,)).append(pg.Displace(0, q=2.0, p=0.0)),
                    backend="native-fock", cutoff=15).state          # coherent, alpha = 1
pn = photon_number_marginal(state, 0)
print([round(pn[n], 6) for n in range(4)], [round(math.exp(-1) / math.factorial(n), 6) for n in range(4)])
# [0.367879, 0.367879, 0.18394, 0.061313] [0.367879, 0.367879, 0.18394, 0.061313]
print(photon_number_distribution(state, 0).layout["title"]["text"])
# Photon-number distribution, mode 0 (retained mass 1.000000)
```

### Gaussian-state plots: covariance, ellipses, entanglement

| Field | Value |
|---|---|
| Description | Heatmap of the statistical covariance matrix (labels q[n], p[n] in interleaved order). One-sigma covariance ellipses per mode, drawn with dashed vacuum circles. Per-mode bars of the base-2 log-negativity of (mode, rest) and the reduced entropy in nats, with the global purity in the title. |
| Python API | `photographiq.interactive.covariance_heatmap(state)`, `covariance_ellipse(state, node, points=181)`, `squeezing_ellipses(state, nodes=None)`, `entanglement_diagnostics(state)` |
| Implementation | `src/photographiq/interactive/plots.py` |
| Backends | `GaussianState` from `gaussian`, `native-gaussian` or `piquasso` |
| CPU / GPU | CPU only |
| Assumptions | Vacuum covariance I, interleaved xpxp. The ellipse is the set (x - mu)^T V^-1 (x - mu) = 1 of the reduced 2x2 covariance. |
| Tests | `tests/visualization/test_interactive_plots.py` (`test_covariance_ellipse_and_heatmap`, `test_entanglement_diagnostics_closed_form` (TMSV closed form)); `tests/visualization/test_interactive_browser.py::test_plots_render_in_browser` |
| Status | **Implemented and validated** |

```python
import photographiq as pg
from photographiq.cvflow import flow_pattern
from photographiq.interactive import covariance_heatmap, wigner_plot
graph = pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,))
pattern = flow_pattern(graph, (0, 1, 2), shears={0: 0.3, 1: -0.5, 2: 0.0})
state = pg.simulate(pattern, backend="gaussian", seed=7).state
print(covariance_heatmap(state).layout["title"]["text"])
# Covariance matrix (vacuum V = I)
print(wigner_plot(state, 3, points=81, extent=8).layout["title"]["text"])
# Wigner function, mode 3 (grid mass 1.0000, negative volume 0)
```

### Run and numerics plots

| Field | Value |
|---|---|
| Description | Recorded measurement likelihoods of a `Result` (probabilities for discrete outcomes, densities for continuous ones, postselected keys labelled), or bars of a plain mapping outcome -> probability. Optimisation curves (losses, validation losses, gradient norms) of a `photographiqml` `History`. Infidelity versus Fock cutoff for a `CutoffStudy`. Log-log runtime-versus-size curves for user-supplied timings. |
| Python API | `photographiq.interactive.measurement_probabilities(source)`, `optimization_curves(history, *, log=False)`, `fidelity_vs_cutoff(study, reference=None)`, `runtime_scaling(sizes, timings, *, xlabel="modes")` |
| Implementation | `src/photographiq/interactive/plots.py` |
| Backends | Independent of backends (plots results) |
| CPU / GPU | CPU only |
| Assumptions | Continuous-outcome densities are in calibrated units, not probabilities. `fidelity_vs_cutoff` never compares rows from different conditional branches: with a `reference`, differing outcomes across rows raise `ValueError`, and without one non-comparable rows are skipped. Infidelities are floored at 1e-17 for the log axis. `runtime_scaling` plots timings it is given and does not measure them. |
| Tests | `tests/visualization/test_interactive_plots.py` (`test_measurement_probabilities_marks_postselection`, `test_optimization_runtime_and_cutoff_plots`) |
| Status | **Implemented and validated** |

```python
import photographiq as pg
from photographiq.cvflow import flow_pattern
from photographiq.interactive import measurement_probabilities
graph = pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,))
pattern = flow_pattern(graph, (0, 1, 2), shears={0: 0.3, 1: -0.5, 2: 0.0})
post = pg.simulate(pattern, backend="gaussian", seed=7, measurement_outcomes={0: 0.0})
print(measurement_probabilities(post).traces[0]["x"])
# ['0 (postselected)', '1', '2']
```

### Plot rendering and export

| Field | Value |
|---|---|
| Description | Every plot function returns a `Plot`, a restricted Plotly figure description (trace types `scatter`, `bar`, `heatmap`). It renders as self-contained HTML with Plotly.js, as a notebook iframe, or as a static Matplotlib figure (SVG/PDF/PNG). |
| Python API | `photographiq.interactive.Plot(traces, layout)`; `.to_html(*, plotly_js=None, height=460)`, `.save_html(path, **options)`, `.show(**options)`, `.draw(ax=None)`, `.save(path, *, dpi=300)` |
| Implementation | `src/photographiq/interactive/plots.py`, `src/photographiq/interactive/_html.py` |
| Backends | Independent of backends |
| CPU / GPU | CPU only |
| Assumptions | No physics is evaluated in JavaScript. `plotly_js=path` inlines a local Plotly.js copy for offline pages. Other trace types raise `NotImplementedError` in the Matplotlib renderer. |
| Tests | `tests/visualization/test_interactive_plots.py` (`test_plot_html_and_javascript`, `test_plot_static_exports`); `tests/visualization/test_interactive_browser.py::test_plots_render_in_browser` (Plotly.js in headless Edge/Chrome) |
| Status | **Implemented but experimental** (tests check that pages render and files are valid; no rendered-image reference) |

```python
import tempfile, pathlib
import photographiq as pg
from photographiq.interactive import squeezing_ellipses
state = pg.GaussianState([0.4, -0.2], [[0.5, 0.1], [0.1, 2.1]], ("m",))
out = pathlib.Path(tempfile.mkdtemp())
plot = squeezing_ellipses(state)
print(plot.save(out / "ellipses.pdf").suffix, plot.save_html(out / "ellipses.html").suffix)
# .pdf .html
```

## 31. Manuscript figures

Eight architecture figures (A–H) are generated from a structured description of the
implemented code. Every box cites a module, class, function, file or C++ symbol, and the
generator fails when a cited symbol no longer exists. The figures are drawn as diagrams,
not screenshots. Captions are generated together with the figures. See
[architecture figures](architecture/README.md) for the format and conventions.

### Architecture figure generator

| Field | Value |
|---|---|
| Description | Renders figures A–H from `docs/architecture/figures.json` to SVG and PDF, writes `captions.md` and the figure index, and validates code references, layout and print readability. |
| Python API | Command line: `python scripts/generate_architecture_figures.py [--check] [--report-sizes] [--preview DIR]`. Module functions `validate(spec)`, `check_ref(ref)`, `render(figure)` and `print_sizes(figure_id, width_in, texts)`. |
| Implementation | `scripts/generate_architecture_figures.py` (source `docs/architecture/figures.json`) |
| Backends | Not applicable |
| CPU / GPU | CPU (Matplotlib) |
| Assumptions | Figures are placed at 7.0 in text width. Text is at least 7 pt for titles and at least 6 pt for details, edge labels, legend and lane labels after fit and print scaling. Output is deterministic: text is converted to paths, SVG ids are fixed and no dates are written. |
| Tests | `tests/unit/test_architecture_figures.py`: `test_every_reference_exists`, `test_validation_rejects_missing_code`, `test_edge_crossing_detection`, `test_print_readability_check`, `test_wide_figure_with_tiny_text_fails_and_lanes_render` |
| Status | **Implemented and validated** |

Example:

```text
$ python scripts/generate_architecture_figures.py --check
8 figures, 484 references to existing code
```

The generator refuses to write into `docs/` while any check fails. The checks are:
- a label overflows its box;
- an edge crosses a box other than its endpoints;
- an edge label overlaps a box or another edge label;
- the legend overlaps the diagram;
- any text falls below the print minimum.

`PGQ_FIGURES_SKIP_PRINT_CHECK=1` turns only the print check into a warning, for use
while a figure is being redesigned.

### Figures A–H

| Field | Value |
|---|---|
| Description | Publication figures of the software architecture, each with a manuscript caption that explains the interfaces between components. |
| Python API | Generated files under `docs/figures/architecture/`: `figure_<id>_<slug>.svg` and `.pdf`, plus `captions.md` |
| Implementation | `docs/architecture/figures.json` (editable source), rendered by `scripts/generate_architecture_figures.py` |
| Backends | Not applicable |
| CPU / GPU | Not applicable |
| Assumptions | Line styles: dark-blue solid for quantum information, ochre dashed for classical outcomes and feed-forward, grey solid for data or control flow, grey dotted for optional dependencies. Optional components (Piquasso, Graphix, mpmath, MentPy, CuPy, CUDA-Q, JAX) have dashed borders. |
| Tests | `tests/unit/test_architecture_figures.py` (references, layout, print size). The factual content was checked against the code by independent reviewers. |
| Status | **Implemented and validated** |

| Figure | File stem | Content |
|---|---|---|
| A | `figure_A_complete_architecture` | User API → quantum IR → compiler → backend planner → numerical engines → results → visualization, as labelled bands. PhotoGraphiQML is an extension band on top. Boson sampling and `photographiq.gpu` appear as direct entry points that bypass the planner. |
| B | `figure_B_cv_mbqc_pipeline` | Input states → graph construction → entanglement → flow/dependency analysis → measurement calculus → execution schedule → measurements → feed-forward → outputs. Quantum and classical paths use different line styles. |
| C | `figure_C_simulation_architecture` | Python public API, native loader and C++ bindings; native Gaussian, Fock and mixed-Fock simulators; CPU optimization layers (Eigen, OpenMP, compensated summation); GPU module; CUDA-Q adapter; sampling kernels. |
| D | `figure_D_photonic_qml` | A PhotoGraphiQML training iteration: parameterized MuTA pattern → logical or (experimental) physical GKP representation → simulation → observable → loss → finite-difference gradient → optimizer (Adam, SGD, L-BFGS-B; `DiscreteSearch` for physical models) → updated parameters. |
| E | `figure_E_graph_compilation` | Open graph → flow analysis (causal flow, gflow, Pauli flow, CV flow) → measurement order → corrections → standardization → signal shifting → executable pattern, with qubit and CV routes side by side. |
| F | `figure_F_boson_sampling` | Fock input → interferometer → output pattern → permanent. Separately, Gaussian state → interferometer → PNR or threshold detection → (loop) hafnian or dark-mode-conditioned threshold probability, with Python and native kernel dispatch. |
| G | `figure_G_backend_selection` | `required_features`, `BackendInfo` registry, `select_backend` cost order and rejection reasons; `simulate(backend=name)` never auto-selects; explicit GPU path with no CPU fallback. |
| H | `figure_H_validation_architecture` | Analytical references, independent implementations, Piquasso, Graphix, mpmath and MentPy (optional) feeding the test suites, benchmarks and release gates. |

All eight figures pass the print check at 7.0 in width. Figure A is 9.62 in tall; placed
at 6.8 in width it fits a standard text page with all text at 6.0 pt or larger.

Limitations recorded in the captions:
- **Figure D:** PhotoGraphiQML has no autodiff or parameter-shift gradient, and the physical path is experimental.
- **Figure E:** no cost-driven pattern optimizer exists, and LC and ZX rewrites are analysis-only.
- **Figure C:** CUDA-Q is an adapter module, not a backend; its CUDA-Q path is unverified.

## 32. Benchmarks

Every benchmark script lives under `benchmarks/`. The methodology, caveats and output layout are documented in `benchmarks/README.md`, `benchmarks/gpu/README.md` and `benchmarks/graphix/README.md`. This section quotes no timing numbers. The reference run is the 2026-11-12 reproduction bundle `benchmarks/results_20261112/` (last entry of this section), made with `scripts/reproduce_manuscript.py --mode full` on a nearly idle machine; cite timings from there, not from this document, and never from `results/superseded/`. Each script compares only implementations that compute the same task on identical inputs and records agreement before timing (`bench_fock.py` has done so since the 2026-11-11 fix). Results are JSON (plus SVG/PDF figures for some scripts) under the script's `results/` directory unless `--output`/`--out` is given.

### Shared benchmark helpers (`bench_common.py`)

| Field | Value |
|---|---|
| Description | Environment metadata, the timing protocol, the tracemalloc memory peak, JSON writing and figure helpers, shared by the boson-sampling, GBS, threshold mpmath reference, native Gaussian, GPU and Graphix scripts; `fock/bench_fock.py` uses `environment`, `rel_err`, `stamp` and `write_json` but not `time_call`. |
| Python API | `bench_common.environment(seed=None, **extra)`, `bench_common.time_call(fn, *, repeats=7, warmup=2, min_sample_s=2e-3, budget_s=5.0)`, `bench_common.peak_python_bytes(fn)`, `bench_common.write_json(payload, path)` (script-local module, not part of the package) |
| Implementation | `benchmarks/bench_common.py` |
| Backends | N/A |
| CPU / GPU | CPU (records native and OpenMP/numba thread counts) |
| Assumptions | `time_call` reports the first call separately as `first_call_s` (never a sample), uses one call to estimate cost, then `warmup` untimed calls and `repeats` samples. Each sample is the mean of `inner` calls so that it lasts at least 2 ms. It reports median, IQR, min and mean. `repeats` shrinks (to at least 3) to stay within `budget_s`; if one call times (warmup + 4) exceeds the budget, there is no warm-up and the samples are single calls. Metadata: UTC timestamp, CPU model, logical CPUs, platform, Python, versions of photographiq, piquasso, graphix, numpy, scipy and numba, native components and build info, thread environment variables, precision, seed, and the git revision with a dirty flag (read-only git calls). tracemalloc does not see C/C++ or numba allocations, so the memory figure is a lower bound for those implementations. |
| Tests | `tests/sampling/test_sampling_benchmark_smoke.py::test_time_call_excludes_cold_first_call` (a synthetic function with a known 0.3 s cold first call); also exercised by the smoke tests of the benchmark scripts below |
| Status | **Implemented and validated** |

Example: see the smoke tests above.

### Boson sampling (`boson_sampling/bench_boson_sampling.py`)

| Field | Value |
|---|---|
| Description | Four task families: permanents of n x n complex matrices; single output probabilities P(t given s), at API level and at kernel level (rectangular permanent with multiplicities); full output distributions; and exact sampling throughput. Compared implementations: PhotoGraphiQ native (C++/OpenMP Ryser), PhotoGraphiQ Python (NumPy Ryser) and Piquasso 8. |
| Python API | `python benchmarks/boson_sampling/bench_boson_sampling.py [--smoke] [--full] [--seed N] [--output path.json] [--no-figures]` |
| Implementation | `benchmarks/boson_sampling/bench_boson_sampling.py`; it measures `src/photographiq/sampling/` and `cpp/src/` sampling kernels |
| Backends | PhotoGraphiQ sampling module (native and Python) against Piquasso `PassiveSimulator` / `ParticleNumberMeasurement` |
| CPU / GPU | CPU only |
| Assumptions | Equivalent tasks only. Piquasso program execution is timed separately as setup. Correctness reference: the permutation sum for n <= 8, otherwise PhotoGraphiQ Python where run, otherwise Piquasso. Samplers report photon-number conservation and the total-variation distance to the exact distribution. Piquasso's distribution basis also holds patterns with fewer photons (exactly zero here). |
| Tests | `tests/sampling/test_sampling_benchmark_smoke.py::test_benchmark_smoke` (smoke configuration: valid JSON, environment keys, all four task families present, positive timings, and every recorded `rel_error`, `rel_error_vs_piquasso` and `rel_error_vs_python` below 1e-8) |
| Status | **Implemented and validated** |

```bash
python benchmarks/boson_sampling/bench_boson_sampling.py --smoke --output <scratch>/bs.json --no-figures
# Ran here through the smoke test (passed). Writes results/boson_sampling_<config>_<stamp>.json by default.
```

### Gaussian boson sampling (`gaussian_boson_sampling/bench_gbs.py`)

| Field | Value |
|---|---|
| Description | Matrix functions (hafnian, loop hafnian, torontonian, loop torontonian) and single-pattern GBS probabilities: PNR (`pnr_probability` with `method="auto"` and `"mixed"`, and native with precomputed `GBSMatrices`) and threshold (`threshold_probability`). Sweeps cover modes, photon number, squeezing, loss and displacement. Compared: PhotoGraphiQ native, PhotoGraphiQ Python and Piquasso 8. |
| Python API | `python benchmarks/gaussian_boson_sampling/bench_gbs.py [--smoke] [--full] [--seed N] [--output path.json] [--no-figures]` |
| Implementation | `benchmarks/gaussian_boson_sampling/bench_gbs.py` |
| Backends | PhotoGraphiQ `sampling` (native and Python) against Piquasso `GaussianSimulator` |
| CPU / GPU | CPU only |
| Assumptions | The same matrix is passed to every implementation. For the torontonian the index permutation (0,2,..,1,3,..) is applied because Piquasso pairs (2i, 2i+1) and PhotoGraphiQ pairs (i, i+n). GBS states are built by both libraries and the covariances are checked equal. Rows with precomputed `GBSMatrices` have no Piquasso counterpart and are labelled as such. Each threshold row records PhotoGraphiQ's own `estimated_rel_error`. Since the threshold fix this is `B / abs(P)` from the absolute rounding bound (Section 20), which is larger than the old estimate, so it is not comparable with result files written before 2026-11-11. The benchmark's `rel_error_vs_piquasso` for threshold rows mixes the errors of both libraries; use `threshold_mp_reference.py` for an absolute reference. |
| Tests | `tests/sampling/test_sampling_benchmark_smoke.py::test_benchmark_smoke` (smoke configuration: hafnian, torontonian, PNR and threshold rows present; recorded relative errors against Piquasso and the Python implementation below 1e-8) |
| Status | **Implemented and validated** |

```bash
python benchmarks/gaussian_boson_sampling/bench_gbs.py --smoke --output <scratch>/gbs.json --no-figures
# Ran here through the smoke test (passed). Default output: results/gbs_<config>_<stamp>.json and figures_<config>/.
```

### Threshold-probability mpmath reference (`gaussian_boson_sampling/threshold_mp_reference.py`)

| Field | Value |
|---|---|
| Description | A 50-digit mpmath inclusion-exclusion reference, P(clicks on S) = sum over T subset of S of (-1)^#T P(vacuum on Z u T), with Z the modes outside S, for chosen `bench_gbs` cases. It reports each implementation's actual relative error and PhotoGraphiQ's estimated error (`estimated_relerr_photographiq_*`, from `threshold_probability(..., return_error=True)`). Since the threshold fix this estimate is the relative form `B / abs(P)` of the absolute rounding bound (Section 20), so it is larger than in result files written before 2026-11-11. |
| Python API | `python benchmarks/gaussian_boson_sampling/threshold_mp_reference.py 24,12,0.9 28,14,0.9 ... [--output PATH]` (modes,clicks,eta; r = 0.5 and seed 11 fixed); `parse_args(argv)` (argparse, `parse_intermixed_args`, so cases may follow the option) |
| Implementation | `benchmarks/gaussian_boson_sampling/threshold_mp_reference.py` |
| Backends | mpmath (extra `precision`), Python multiprocessing |
| CPU / GPU | CPU only |
| Assumptions | hbar=2, zero mean, P(vacuum on R) = 2^#R / sqrt(det(V_R + I)), where #R is the number of modes in R. The reference is computed from each library's own double-precision covariance (`ref_cov_sensitivity` shows the effect). The cost is 2^clicks mp determinants. Output goes to `--output`, by default `results/threshold_mp_reference_<stamp>.json` next to the script. At least one case is required. Each case is parsed by an argparse type, so a wrong field count or a non-numeric field (`24,12`, `24,12,0.9,1`, `a,b,c`) gives a usage error. Before the fix `--output x.json` crashed with `ValueError: could not convert string to float: '--output'`. |
| Tests | `tests/native/test_fock_benchmark_smoke.py::test_threshold_mp_reference_output_option` (option parsing, default `None`, usage errors; it does not run the reference, and it sits behind that file's native-fock module-level skip). The numbers are not tested: the script is itself a reference computation. |
| Status | **Implemented but experimental** |

```bash
python benchmarks/gaussian_boson_sampling/threshold_mp_reference.py 8,4,0.9 --output <scratch>/mp.json
# Ran here (2 s; nothing written under benchmarks/): relerr native 1.8e-13, python 2.1e-13, piquasso 1.1e-13;
# estimated_relerr_photographiq_native 2.05e-11.
```

### Threshold-probability accuracy matrix (`gaussian_boson_sampling/threshold_reference_matrix.py`)

| Field | Value |
|---|---|
| Description | Accuracy matrix of GBS threshold probabilities against an independent 40-digit mpmath implementation. Grid: modes {4, 8, 12, 16, 20, 24}, clicks 1 to min(modes, 16), all five squeezings {0.05, 0.2, 0.5, 1.0, 1.5} per (modes, clicks), each with a seeded random transmissivity in {1.0, 0.9, 0.5}, displacement on or off, a seeded Haar interferometer and a random click pattern (360 cases), plus 22 zero-probability cases (a vacuum mode untouched by the interferometer clicks). Per case it records `threshold_probability` on the native and Python backends (value, relative bound, absolute bound B, warning), `precision="auto"` (path, digits, value) and Piquasso `get_threshold_detection_probability` on the identical program. |
| Python API | `python benchmarks/gaussian_boson_sampling/threshold_reference_matrix.py [--smoke] [--procs N] [--output PATH]` |
| Implementation | `benchmarks/gaussian_boson_sampling/threshold_reference_matrix.py` (reference: dark-mode Schur complement in mpmath, then a depth-first walk over the 2^k subsets that extends one Cholesky factor per added mode; shares no code with PhotoGraphiQ's `precision=` path) |
| Backends | PhotoGraphiQ sampling module (native, Python, `auto`), Piquasso 8.0.1, mpmath (extra `precision`), Python multiprocessing |
| CPU / GPU | CPU only |
| Assumptions | The reference is evaluated from each library's own stored double-precision covariance and mean, taken exactly; its own uncertainty `10^(3 - 40) sum abs(terms) / abs(P)` is recorded per case (below 1e-10 in 356 of 360 random cases). The bound "holds" when `abs(double - ref) <= B + reference uncertainty`. The full run used 12 processes; the run time is recorded in the result file, not here. Results of the 2026-11-11 run (`results/threshold_reference_matrix_20261111-233720.json`) are summarised in Section 20 and [GBS threshold accuracy](validation/gbs-threshold-accuracy.md). Output defaults to `results/threshold_reference_matrix_<stamp>.json` next to the script. |
| Tests | None for the script; it is itself a reference computation. The smoke run below checks that it executes. |
| Status | **Implemented but experimental** (no automated test; the numbers it produces are the evidence for Section 20) |

```bash
python benchmarks/gaussian_boson_sampling/threshold_reference_matrix.py --smoke --output <scratch>/trm.json
# [1/42 <elapsed>] m=4 k=1 r=0.05 eta=0.5 disp=0 random: P=6.366e-04 native=1.0218594154096595e-15 held=True/True
# ... (42 smoke cases; nothing written under benchmarks/)
```

```python
import json, sys
R = json.load(open(sys.argv[1]))["results"]          # the smoke JSON written above
print(len(R), all(c[b]["bound_held"] for c in R for b in ("native", "python")))
print(sorted({c["auto"]["path"] for c in R}))
# 42 True
# ['double', 'mpmath']
```

### Native Gaussian engine (`gaussian/bench_native_gaussian.py`)

| Field | Value |
|---|---|
| Description | Single-state beam-splitter gates and destructive homodyne, and batched gates and homodyne over B states. Compared: `GaussianBackend` (dense NumPy, O(n^3)), a NumPy O(nk) local update, `NativeGaussianBackend` (C++), the native OpenMP batch kernels, a batched NumPy implementation of the same algorithm, and a Python loop. |
| Python API | `python benchmarks/gaussian/bench_native_gaussian.py [--smoke] [--full] [--seed N] [--output path.json]` |
| Implementation | `benchmarks/gaussian/bench_native_gaussian.py`; measures `src/photographiq/backends/gaussian.py`, `src/photographiq/backends/native_gaussian.py`, `cpp/src/` Gaussian kernels |
| Backends | `gaussian`, `native-gaussian` and the native batch functions |
| CPU / GPU | CPU (NumPy, and C++/OpenMP) |
| Assumptions | Identical inputs. Before timing, every output array is compared with native and the run aborts if the scaled difference exceeds 1e-12 (`max_scaled_diff_vs_native` per row). `speedups` name both sides. `python-loop / native-batch` also includes O(n^3) re-validation and is labelled an upper bound. Native batches use all OpenMP threads, so the ratios are not single-core comparisons. Results in `gaussian/results/superseded/` must not be cited. |
| Tests | `tests/native/test_native_gaussian_benchmark_smoke.py::test_native_gaussian_benchmark_smoke`, `::test_agreement_check_rejects_wrong_output`, `::test_chain_state_matches_dense_reference` |
| Status | **Implemented and validated** |

```bash
python benchmarks/gaussian/bench_native_gaussian.py --smoke --output <scratch>/ng.json
# Ran here through the smoke test (passed). Default output: gaussian/results/native_gaussian_<config>_<stamp>.json.
```

### Native Fock engines (`fock/bench_fock.py`)

| Field | Value |
|---|---|
| Description | Per-gate runtime (displace, squeeze, rotate, Kerr, cubic phase, beam splitter, CZ, quadratic phase, plus thermal loss on the mixed engines) and per-pattern runtime (`wire`, `network`) over modes and cutoff for `native-fock` / `native-mixed-fock` against `piquasso-fock` / `piquasso-mixed-fock`. It also measures per-pattern memory in fresh subprocesses and OpenMP thread scaling of a CZ kernel (5-mode pure state, 3-mode density matrix), and accuracy vs cutoff (`convergence` rows, untimed): for a two-mode squeezed, displaced beam-splitter circuit, max over modes of |<n>_Fock - <n>_exact| against the untruncated `GaussianBackend`, plus the product of retained norms and the largest boundary population, for `native` and `piquasso`. Piquasso-backed loss is not timed above dimension 105 (the row is marked `skipped`). |
| Python API | `python benchmarks/fock/bench_fock.py [--smoke \| --quick] [--out PATH]`; `run(config="full", output=None)` |
| Implementation | `benchmarks/fock/bench_fock.py`; measures `src/photographiq/backends/native_fock.py`, `cpp/src/fock/` |
| Backends | `native-fock`, `native-mixed-fock`, `piquasso-fock`, `piquasso-mixed-fock` |
| CPU / GPU | CPU (C++/OpenMP against Piquasso/NumPy) |
| Assumptions | Correctness gate. Before anything is timed, every gate and pattern case runs once, untimed, on the native engine and its Piquasso counterpart. Each row stores `native_vs_piquasso_trace_distance` (patterns also store `native_vs_piquasso_statistics_rel_diff`, via `bench_common.rel_err`) next to `agreement_tol`, and a mismatch raises `RuntimeError`, as does a zero reference statistic. Tolerances: `ROUNDING_TOL = 1e-12` where both engines apply the same truncated operator (single-mode gates, beam splitter, thermal loss). Observed at most 1.3e-15 at benchmark sizes, and at most 2.2e-15 for single-mode gates and the beam splitter on number states up to cutoff 30. For the CZ and whole patterns, where the truncation schemes differ, the tolerance is `max(1e-12, sqrt(p_boundary))`, with `p_boundary` the largest population at total photon number >= cutoff - 2 recorded by either engine. This is an empirical scale, not a proven bound. Deviation / sqrt(p_boundary) for a single CZ: at most 0.047 over 358 number-state inputs at the default `norm_tolerance` 1e-3, which the benchmark uses; at most 0.098 over 240 squeezed-coherent inputs; but 1.39 at `norm_tolerance` 0.9 (cutoff 16, input (0.3\|0> + \|13>)\|1>, g = 1). So the tolerance holds only for states that the default norm guard admits. Observed deviation / tolerance at every benchmark size: at most 0.0098 (CZ), 0.015 (pattern states) and 0.024 (pattern statistics). Piquasso-backed loss above dimension 105 is neither timed nor checked (`agreement_skipped`). Thread rows have no Piquasso counterpart: each child returns a checksum, and `checksum_rel_diff_vs_first` must be at most 1e-12. Timing uses its own timer, not `bench_common.time_call`: the prepared state is restored outside the timed region before each repetition and again after each gate. Before the fix each gate was timed on the previous gate's output. There are warm-up calls, and the script reports median and IQR. `psutil` is optional and no extra declares it: without it, the memory fields, `physical_cpus` and the CPU-load field are `None`. Memory is the psutil `peak_wset` (Windows only, otherwise `None`) and the baseline RSS. Thread scaling sets `OMP_NUM_THREADS` in subprocesses. Metadata comes from `bench_common.environment` (keys include `timestamp_utc`, `packages`, `native_fock_max_threads`, `physical_cpus`, `memory_bytes`, `system_cpu_percent_at_start`; load is recorded, not controlled). Output defaults to `fock/results/fock-<host>-<stamp>.json`. The file in `fock/results/superseded/` must not be cited. |
| Tests | `tests/native/test_fock_benchmark_smoke.py` (skips without the native fock component or Piquasso): `::test_fock_benchmark_smoke` (smoke run, all agreement fields within tolerance), `::test_agreement_check_rejects_wrong_output` (a Kerr kappa error of 1e-6 fails the rounding tolerance, a kappa of 0.3 instead of 0.2 fails the truncation tolerance, and a wrong or zero reference statistic raises `RuntimeError`), `::test_memory_fields_are_none_without_psutil`. The smoke test also asserts that the convergence error decreases strictly with the cutoff for both engines and that the two engines agree to 1e-12 at each cutoff. Engine agreement is also tested in `tests/native/test_native_fock_backend.py` and `tests/native/test_native_mixed_fock_backend.py`. |
| Status | **Implemented but experimental** (the CZ/pattern agreement tolerance is empirical) |

```bash
python benchmarks/fock/bench_fock.py --smoke --out <scratch>/fock_smoke.json
# Executed here (exit 0, about 30 s): JSON with keys config, environment, gates, patterns, threads, convergence; every
# row carries agreement_tol (largest gate deviation / tolerance 0.0021). Timings not quoted.
```

### GPU batched Gaussian kernels (`gpu/bench_gpu_gaussian.py`)

| Field | Value |
|---|---|
| Description | Three batched tasks on B copies of an n-mode squeezed lossy CZ chain: `local_symplectic`, `dense_symplectic` (a different random symplectic matrix per state) and destructive `homodyne`. Compared: `photographiq.gpu` with `device="cpu"` (NumPy, same algorithm), the native C++ batch kernels (float64 only) and CuPy (`gpu_end_to_end`, `gpu_kernel`, `gpu_h2d`, `gpu_d2h`). |
| Python API | `python benchmarks/gpu/bench_gpu_gaussian.py [--modes ...] [--batches ...] [--dtypes float64 float32] [--budget S] [--max-bytes B] [--smoke] [--output path.json]` |
| Implementation | `benchmarks/gpu/bench_gpu_gaussian.py`; measures `src/photographiq/gpu/` and `src/photographiq/backends/native_gaussian.py` |
| Backends | `photographiq.gpu` (CuPy), the native Gaussian batch kernels, NumPy |
| CPU / GPU | GPU via `photographiq.gpu` (CuPy) and CPU (NumPy, C++/OpenMP). Without CuPy or a device the GPU rows are not run, and the script prints that there is no CPU fallback. |
| Assumptions | Before timing, each output is compared with the float64 NumPy result (`max_rel_deviation_vs_numpy_float64`; for float32 this measures precision loss). The first call is excluded from the samples (CuPy compilation). `gpu_kernel` includes the device synchronizations of the validity checks. Inputs above `--max-bytes` are skipped and listed. The JSON adds `gpu` (`photographiq.gpu.device_info()`) and `native_gaussian_max_threads`. GeForce FP64 throughput is a small fraction of FP32; compare dtypes before drawing conclusions. `--smoke` timings are meaningless. |
| Tests | None for the script. GPU kernels: `tests/gpu/` |
| Status | **Implemented but experimental** |

```bash
python benchmarks/gpu/bench_gpu_gaussian.py --smoke --output <scratch>/gpu_smoke.json
# Executed here (RTX 4050, CuPy installed): rows numpy, native, gpu_end_to_end, gpu_kernel,
# gpu_h2d, gpu_d2h; environment includes gpu.device_name and CUDA versions (timings not quoted).
```

### GPU batched Fock gates (`gpu/bench_gpu_fock.py`)

| Field | Value |
|---|---|
| Description | One local gate on B Fock states in the total-photon basis: `displace` (1-mode matrix from the native `gaussian_unitary`), `cz` (2-mode matrix) and `kerr` (diagonal phase; pure states only), on pure batches `(B, D)` and density-matrix batches `(B, D, D)`. Compared: `native_loop` (the native `apply_pure` / `apply_mixed` / `phase_*` state by state on a fresh copy), `numpy` (`photographiq.gpu.fock` with `device="cpu"`, scipy CSR), `gpu_end_to_end` (NumPy in and out), `gpu_kernel` (device-resident, `on_device=True`, synchronised), `gpu_h2d`, `gpu_d2h`. |
| Python API | `python benchmarks/gpu/bench_gpu_fock.py [--batches ...] [--mixed-batches ...] [--dtypes complex128 complex64] [--budget S] [--max-bytes B] [--smoke] [--output PATH]` |
| Implementation | `benchmarks/gpu/bench_gpu_fock.py`; measures `src/photographiq/gpu/fock.py` against `photographiq._native.fock` |
| Backends | `photographiq.gpu.fock` (CuPy/cuSPARSE and scipy), native Fock kernels |
| CPU / GPU | GPU via CuPy and CPU (scipy, C++) |
| Assumptions | Before timing, every implementation's output is compared with the complex128 `native_loop` result (relative to the largest entry) and the run stops above 1e-12 (complex128) or 1e-5 (complex64). Timing uses `bench_common.time_call`; the first call, which builds the cached index maps and the cuSPARSE handle, is reported separately and is never a sample. `peak_device_pool_bytes` is the CuPy memory-pool footprint of one `gpu_kernel` call. The 2026-11-11 run (`benchmarks/gpu/results/gpu_fock_20261111-232613.json`) was on a shared machine, so factors below about 2 are noise; the 2026-11-12 run is in the bundle (`bench-gpu-fock.json`). `--smoke` timings are meaningless. |
| Tests | None for the script; the kernels are tested in `tests/gpu/test_gpu_fock.py` (Section 15) |
| Status | **Implemented but experimental** |

```bash
python benchmarks/gpu/bench_gpu_fock.py --smoke --output <scratch>/gpufock.json
# Executed here (RTX 4050, CuPy installed, exit 0): one line per row, e.g.
# mixed m=2 c= 5 D=    15 B=   2 complex64  cz       gpu_kernel      median ... ms   (timings not quoted)
```

```python
import json, sys
d = json.load(open(sys.argv[1]))                       # the smoke JSON written above
rows = d["results"]
print(d["smoke"], len(rows), sorted({r["implementation"] for r in rows}))
dev = [r.get("max_rel_deviation_vs_native_complex128") for r in rows if r["dtype"] == "complex128"]
print(max(x for x in dev if x is not None) < 1e-12)
# True 88 ['gpu_d2h', 'gpu_end_to_end', 'gpu_h2d', 'gpu_kernel', 'native_loop', 'numpy']
# True
```

### Qubit MBQC against Graphix (`graphix/bench_graphix_mbqc.py`)

| Field | Value |
|---|---|
| Description | Qubit MBQC only: causal-flow, gflow and Pauli-flow finding, standardization, signal shifting, and fixed-branch statevector simulation. Compared: PhotoGraphiQ `photographiq.flows` and `photographiq.calculus.qubit` against Graphix 0.4. |
| Python API | `python benchmarks/graphix/bench_graphix_mbqc.py [--full] [--seed N] [--repeats N] [--warmup N] [--cap SECONDS] [--no-plots] [--out-dir DIR]` (`--out-dir`, added 2026-11-12, keeps a reproduction run from writing into the repository) |
| Implementation | `benchmarks/graphix/bench_graphix_mbqc.py` |
| Backends | No CV backend. Qubit tooling against Graphix (extra `validation`). |
| CPU / GPU | CPU only |
| Assumptions | PhotoGraphiQ finders always verify their certificate, so they are compared with Graphix search plus `check_well_formed`; Graphix search-only time is a separate column. Each certificate is checked by both verifiers. Map equivalence of standardization and shifting is checked by exact branch simulation on small sizes. Not applicable, and never compared: any CV/Gaussian/GBS task, PhotoGraphiQ's full Kraus map, Graphix tensor-network or density-matrix backends. Output: `graphix/results/graphix_mbqc_fast_<stamp>.json` (or `graphix_mbqc_full_<stamp>.json`), with `.svg` and `.pdf` figures. |
| Tests | `tests/flow/test_flows_graphix_benchmark_smoke.py::test_graphix_benchmark_tiny_run` |
| Status | **Implemented and validated** |

```bash
python benchmarks/graphix/bench_graphix_mbqc.py --no-plots
# A tiny configuration ran here through the smoke test (passed); the fast run takes minutes.
```

### Benchmark figures (`plot_results.py`)

| Field | Value |
|---|---|
| Description | Draws the figures of the native Gaussian, Fock and GPU benchmarks from their JSON without recomputing anything. It produces runtime vs modes (Gaussian, per task and implementation, with an IQR band), per-gate runtime vs Fock-space dimension (pure and mixed, native vs Piquasso), whole-pattern runtime and process peak memory, OpenMP thread scaling, accuracy vs cutoff (Fock vs exact Gaussian), and CPU (NumPy, native) vs GPU (CuPy kernel, end to end) runtime vs batch size in float64 and float32. The sampling and Graphix benchmarks draw their own figures. |
| Python API | `python benchmarks/plot_results.py [--gaussian PATH] [--fock PATH] [--gpu PATH]` (default: the newest non-smoke JSON of each); `plot_gaussian(data, out)`, `plot_fock(data, out)`, `plot_gpu(data, out)` |
| Implementation | `benchmarks/plot_results.py` (uses `bench_common.new_axes`, `series_style`, `save_figure`) |
| Backends | Not applicable |
| CPU / GPU | CPU (Matplotlib) |
| Assumptions | Plots the median, with the IQR as a band. SVG and PDF go to `<benchmark>/results/figures_<json stem>/`. |
| Tests | `tests/unit/test_benchmark_plots.py::test_every_plot_renders` (synthetic JSON, all 10 files written) |
| Status | **Implemented but experimental** |

```bash
python benchmarks/plot_results.py
# wrote benchmarks/gaussian/results/figures_native_gaussian_default_<stamp>/gaussian_runtime_vs_modes.svg (and .pdf), ...
```

### Manuscript reproduction driver (`scripts/reproduce_manuscript.py`)

| Field | Value |
|---|---|
| Description | Reproduces the evidence behind the manuscript as a list of named steps (test suites, figure check, benchmark smoke and full runs, the threshold mpmath reference, the manuscript experiment suites and their comparison with the recorded results). Each step runs in a subprocess with a fixed seed where it takes one; its log goes to `<output>/logs/`, its artifacts to `<output>/artifacts/`, and `<output>/summary.json` records the environment, the git revision and dirty flag, each command, its exit code, run time and status. |
| Python API | `python scripts/reproduce_manuscript.py [--mode fast\|full] [--output DIR] [--only STEP ...] [--list]` |
| Implementation | `scripts/reproduce_manuscript.py` |
| Backends | Whatever each step uses; optional dependencies are checked per step |
| CPU / GPU | CPU; the GPU benchmark steps need CuPy and a CUDA device |
| Assumptions | `fast` (default, minutes) runs the validation test suites and benchmark smoke runs; `full` (hours) adds the full benchmarks and the experiment suites. A step whose optional dependency is missing is recorded as `skipped` with the reason, never as passed; the exit code is non-zero if any executed step failed. Nothing is written into the repository: outputs go to `--output` (default `build/reproduction/<timestamp>`, git-ignored), and the experiment suites run unchanged from copies. |
| Tests | None (the driver's own runs are its check; the 2026-11-12 full run is the bundle below) |
| Status | **Implemented but experimental** |

```bash
python scripts/reproduce_manuscript.py --list
# tests-validation                 needs=pytest                 independent Gaussian/Fock validation matrix
# tests-native                     needs=pytest                 native engines agree with references
# ... (14 steps in fast mode, ending with)
# bench-gpu-fock-smoke             needs=cupy                   benchmark runs
```

### Recorded-result comparison (`scripts/compare_recorded_results.py`)

| Field | Value |
|---|---|
| Description | Compares re-run experiment JSON with the recorded results value by value. Timing, timestamp, environment and provenance fields are excluded; every other number, string and boolean is compared (numbers with `rtol`/`atol`, NaN equals NaN). Keys or list lengths present in only one file are counted per side, so schema drift between script versions is reported, not hidden. |
| Python API | `python scripts/compare_recorded_results.py RERUN_DIR RECORDED_DIR [--since EPOCH_SECONDS] [--rtol 1e-9] [--atol A] [--output report.json]` |
| Implementation | `scripts/compare_recorded_results.py` |
| Backends | Not applicable (JSON comparison) |
| CPU / GPU | CPU |
| Assumptions | Only files modified after `--since` are compared (default: the `.copied_at` marker written by `reproduce_manuscript.py`), so a file that was copied but never regenerated is not counted as reproduced. Exit code 0 only if every compared value agrees. |
| Tests | `tests/unit/test_compare_recorded_results.py::test_compare_counts_only_regenerated_files_and_detects_differences` |
| Status | **Implemented but experimental** (behavioural test only) |

```bash
python scripts/compare_recorded_results.py paper_experiments_ml/results/json paper_experiments_ml/results/json --since 0 --output <scratch>/cmp.json
# OK   R10_trainability_audit.json: {'compared': 93, 'identical': 93, 'close': 0, 'different': 0, 'only_in_rerun': 0, 'only_in_recorded': 0, 'length_mismatch': 0}
# ... (one OK line per file; a self-comparison, so every file agrees)
# 60/60 regenerated result files agree with the recorded values
```

### Reproduction bundle of 2026-11-12 (`benchmarks/results_20261112/`)

| Field | Value |
|---|---|
| Description | The reference benchmark and reproduction run, made with `scripts/reproduce_manuscript.py --mode full` on 2026-11-12 (run `full-20261112-041139`). It holds `summary.json`, the JSON (and figures) of every benchmark (`bench-boson-sampling`, `bench-gbs`, `bench-native-gaussian`, `bench-fock`, `bench-gpu-gaussian`, `bench-gpu-fock`, `graphix/`, `threshold_mp_reference.json`) and the experiment comparisons `paper_experiments*-comparison*.json`. |
| Python API | Files only; see `benchmarks/results_20261112/README.md` |
| Implementation | `benchmarks/results_20261112/` |
| Backends | All benchmarked implementations (native, NumPy, Piquasso 8.0.1, Graphix 0.4, CuPy 14.2.0) |
| CPU / GPU | Intel Core i5-12450HX (12 logical CPUs) and NVIDIA RTX 4050 Laptop GPU, Windows 11, Python 3.12.14, MSVC 19.50 with OpenMP |
| Assumptions | Working tree on branch `upgraded` (HEAD `f30321f` plus the uncommitted 2026-11-12 changes, so `git_dirty: true`). Timings vary by roughly 30 % between runs for small cases on this laptop; accuracy fields are deterministic. The earlier result files under each `benchmarks/*/results/` (2026-11-10/11) are kept as previous runs. Of 27 steps, 25 passed and 2 failed: `paper_experiments_ml-safe` failed because 9 MentPy-dependent experiments cannot import MentPy in the project venv, and `paper_experiments-compare` reported differences. In the core comparison 14 of 25 files agree at rtol 1e-9 (15 with atol 1e-12); the others differ mainly by schema drift (keys or rows present in only one file, because the experiment scripts changed after the recorded run), plus changed error messages in R13 (validation messages fixed on 2026-11-11), a different shot count in R1 and 1e-16-level covariance errors in R15. All 28 compared PhotoGraphiQML files agree. This document quotes no timing numbers; cite them from the bundle. |
| Tests | The bundle is evidence, not code; its `tests-*` steps ran the test suites (all passed) |
| Status | **Implemented but experimental** (a single run on one machine; two steps failed as stated) |

```python
import collections, json, pathlib
root = pathlib.Path("benchmarks/results_20261112")
summary = json.loads((root / "summary.json").read_text())
print(summary["mode"], collections.Counter(s["status"] for s in summary["steps"]))
print([s["name"] for s in summary["steps"] if s["status"] != "passed"])
for name in ("paper_experiments-comparison", "paper_experiments-comparison-atol", "paper_experiments_ml-comparison"):
    files = json.loads((root / f"{name}.json").read_text())["files"]
    print(name, dict(collections.Counter(f["status"] for f in files.values())))
# full Counter({'passed': 25, 'failed': 2})
# ['paper_experiments-compare', 'paper_experiments_ml-safe']
# paper_experiments-comparison {'agrees': 14, 'differs': 11}
# paper_experiments-comparison-atol {'agrees': 15, 'differs': 10}
# paper_experiments_ml-comparison {'agrees': 28}
```

## 33. Examples

Runnable examples are spread over several directories. Unless noted, they need only the core dependencies (`numpy`, `scipy`, `networkx`) plus `matplotlib` (extra `visualization`) for figures. Since 2026-11-12 the Fock tutorials, projects and notebooks run on the native engines (`native-fock` / `native-mixed-fock`), which need only the compiled extension; only tutorial 24 (through `photographiq.autodiff`) and the two top-level Fock scripts still need the `piquasso` extra. The migration was made because Numba 0.68 crashed when reloading Piquasso's cached kernels in fresh processes (Section 35); every printed or asserted quantity was compared, and the results are equal to roundoff except three examples whose squeezing/CZ truncation differs by at most 3.3e-9. Graphix examples need `validation`, JAX examples need `autodiff`, MentPy cross-checks need `ml-validation`, and the ML experiment suites use `ml-experiments`. Examples are documentation, not validation evidence. Where a test or checker executes them, it checks only that they run and that their own assertions hold.

### Top-level example scripts (`examples/*.py`)

| Field | Value |
|---|---|
| Description | Four standalone scripts. |
| Python API | `python examples/<name>.py` |
| Implementation | `examples/` |
| Backends | `gaussian`; `piquasso-fock` for two scripts |
| CPU / GPU | CPU |
| Assumptions | Run from the repository root with the package installed. |
| Tests | None (executed manually here: `examples/first_cluster.py` ran and printed mean and covariance) |
| Status | **Implemented but experimental** |

- `examples/first_cluster.py`: measures a three-node CV line graph with feed-forward and prints the conditional output mean and covariance (sampled outcomes).
- `examples/gaussian_compilation.py`: compiles a Gaussian circuit and separates its ideal map from finite-resource noise.
- `examples/adaptive_and_fock.py`: compares adaptive homodyne readings with integer photon counts (needs `piquasso`).
- `examples/non_gaussian.py`: non-Gaussian resources on `piquasso-fock` (needs `piquasso`).

Example: see `examples/first_cluster.py`.

### Core tutorials (`docs/tutorials/*.md` with `examples/tutorials/*.py`)

| Field | Value |
|---|---|
| Description | 37 tutorials. Each documentation page in `docs/tutorials/` has a matching script in `examples/tutorials/`, and `examples/tutorials/tutorials.ipynb` combines them in one notebook. Index: [tutorials](tutorials/index.md); GKP index: [GKP logical tutorials](tutorials/gkp-logical-index.md). |
| Python API | `python examples/tutorials/<NN>-<name>.py` |
| Implementation | `examples/tutorials/`, `docs/tutorials/` |
| Backends | `gaussian`; `native-fock` / `native-mixed-fock` where marked; tutorial 24 needs `autodiff` (JAX plus Piquasso) |
| CPU / GPU | CPU |
| Assumptions | `scripts/check_docs_examples.py` executes every script in a temporary directory with a 180 s timeout (`--skip-autodiff` skips 24). The CI docs workflow runs it with `--projects --notebooks`. With Piquasso made unimportable, every tutorial except 24 passes (Phase 1 audit). |
| Tests | `scripts/check_docs_examples.py` (not pytest; all tutorials, 10 projects and 5 notebooks passed in the 2026-11-12 local gate run); `tests/piquasso_reference/test_tutorial_backends.py` (optional, needs Piquasso: tutorials 09, 13, 14, 21, 38, 44 and the cat-Wigner example on the native and Piquasso backends, equal to roundoff or to truncation level); `tests/unit/test_native_only_smoke.py::test_core_and_native_paths_run_without_optional_packages` (native Gaussian, Fock and mixed-Fock paths in a subprocess with Piquasso, Graphix, JAX, mpmath, CuPy, MentPy and CUDA-Q blocked). Executed here: `examples/tutorials/23-gkp.py` (ran; printed captured weight, stabilizers and a decoded cell). |
| Status | **Implemented but experimental** |

- 01 first-cluster: first CV cluster state.
- 02 teleportation: CV quantum teleportation.
- 03 quantum-wire: one-dimensional quantum wire.
- 04 adaptive-homodyne: adaptive homodyne measurement.
- 05 feedforward: classical feed-forward.
- 06 finite-squeezing: finite squeezing and noise.
- 07 gaussian-compilation: Gaussian gate compilation.
- 08 two-mode-entangling: two-mode entangling computation.
- 09 photon-number: photon-number measurement (`native-fock`).
- 10 kerr: Kerr nonlinearity (`native-fock`).
- 11 cubic-phase: cubic-phase gate (`native-fock`).
- 12 cat-resource: cat-state resource (`native-fock`).
- 13 photon-subtraction: heralded photon subtraction (`native-fock`).
- 14 cubic-injection: cubic resource injection (`native-fock`).
- 15 cutoff-convergence: cutoff convergence (`native-fock`, through `cutoff_convergence(..., backend="native-fock")`).
- 16 wigner-analysis: Wigner-function analysis (`native-fock`).
- 17 graphix-validation: Graphix structural validation (`validation` extra, `graphix==0.4`).
- 18 custom-topology: custom CV graph topology.
- 19 reusable-pattern: building a reusable Pattern.
- 20 compilation-visualization: circuit-to-MBQC compilation and visualization.
- 21 mixed-fock: mixed Fock evolution (`native-mixed-fock`).
- 22 universal-compilation: compiling a nonlinear resource injection (`native-fock`).
- 23 gkp: finite-energy GKP resources (`native-fock`).
- 24 automatic-differentiation: JAX gradients (`autodiff` extra; `photographiq.autodiff` also needs `piquasso` to prepare inputs).
- 37 gkp-finite-codewords: finite-energy logical zero and one.
- 38 gkp-z-readout: logical Z from q homodyne (`native-fock`).
- 39 gkp-x-readout: logical X from p homodyne (`native-fock`).
- 40 gkp-soft-decoding: soft GKP decoding.
- 41 gkp-displacements: logical X and Z displacements (`native-fock`).
- 42 gkp-hadamard: GKP Hadamard (`native-fock`).
- 43 gkp-phase: GKP phase gate (`native-fock`).
- 44 gkp-cz: two-mode logical CZ (`native-fock`).
- 45 gkp-frames: logical Pauli frames (`native-fock`).
- 46 gkp-instrument: encoded finite measurement instrument.
- 47 gkp-magic-boundary: the pi/4 measurement boundary (shows the `NotImplementedError`).
- 48 gkp-convergence: independent GKP convergence axes (`measurement_convergence(..., backend="native-fock")`).
- 49 gkp-downstream-step: preparing a downstream physical measurement step (`native-fock`).

Example: see `examples/tutorials/23-gkp.py`.

### Demo projects (`examples/projects/*/main.py`)

| Field | Value |
|---|---|
| Description | Ten self-contained projects, each with a README and a `main.py` that accepts `--output` to keep results outside the source tree. |
| Python API | `python examples/projects/<name>/main.py [--output DIR]` |
| Implementation | `examples/projects/` |
| Backends | `gaussian`; `native-fock` for three projects |
| CPU / GPU | CPU |
| Assumptions | All use `matplotlib`. `cubic-convergence` calls `cutoff_convergence(..., backend="native-fock")`. No project needs Piquasso since 2026-11-12. |
| Tests | `scripts/check_docs_examples.py --projects` (not pytest; runs each `main.py --output <tmp>` with a 180 s timeout and checks that a two-column `data.csv` and `figure.svg` were written) |
| Status | **Implemented but experimental** |

- `adaptive-calculator`: binds an adaptive parameter and extracts repeated-trajectory means.
- `cat-wigner`: compares finite-grid Wigner negativity over cat amplitudes (`native-fock`).
- `cubic-convergence`: tracks fourth momentum moments over increasing Fock cutoffs (`native-fock`).
- `finite-squeezing-wire`: measures accumulated channel noise at fixed resource squeezing.
- `gaussian-compiler`: inspects finite-resource noise for a family of compiled rotations.
- `graph-experiment`: varies a CZ edge weight and inspects resource correlations.
- `heralded-subtraction`: sweeps a vacuum tap and keeps its one-photon herald probability (`native-fock`).
- `reservoir-features`: maps coherent input amplitudes to a small observable feature table, without ML dependencies.
- `teleportation-fidelity`: sweeps the exact unconditional teleportation channel against a coherent target.
- `topology-comparison`: compares the total photon occupation of equal-squeezing resource families.

Example: see `examples/projects/README.md`.

### Example notebooks (`examples/notebooks/`)

| Field | Value |
|---|---|
| Description | Two notebooks. |
| Python API | Open in Jupyter |
| Implementation | `examples/notebooks/` |
| Backends | Sampling module; Piquasso for comparisons |
| CPU / GPU | CPU |
| Assumptions | The benchmark notebook runs the benchmark code in a smaller `notebook` configuration. Its timings are machine-dependent and are not quoted here. |
| Tests | `tests/visualization/test_interactive_notebook.py::test_notebook_executes` (interactive notebook only) |
| Status | **Implemented but experimental** |

- `boson_sampling_and_gbs_benchmark.ipynb`: API tour, validation and benchmarks for boson sampling and GBS (needs `piquasso` and `matplotlib`; displays Graphix results if present).
- `interactive_graph_visualization.ipynb`: interactive graph, overlay and plot views from `photographiq.interactive` (needs `nbclient`/`ipykernel` to execute).

Example: see `tests/visualization/test_interactive_notebook.py`.

### Quick-start notebooks (`notebooks/`, `notebooks/ml/`)

| Field | Value |
|---|---|
| Description | Short three-cell notebooks mirroring selected tutorials. |
| Python API | Open in Jupyter |
| Implementation | `notebooks/`, `notebooks/ml/` |
| Backends | `gaussian`; `native-fock` for 04 and 05 (since 2026-11-12) |
| CPU / GPU | CPU |
| Assumptions | Requires `matplotlib`. |
| Tests | `scripts/check_docs_examples.py --notebooks` (executes `notebooks/*.ipynb`); `scripts/ml/execute_tutorials.py` (executes `notebooks/ml/*.ipynb`). Neither is pytest. |
| Status | **Implemented but experimental** |

- `notebooks/01_quickstart.ipynb`: your first CV cluster state.
- `notebooks/02_teleportation.ipynb`: CV quantum teleportation.
- `notebooks/03_gaussian_compilation.ipynb`: Gaussian gate compilation.
- `notebooks/04_non_gaussian.ipynb`: heralded photon subtraction (`native-fock`).
- `notebooks/05_cutoff_convergence.ipynb`: cutoff convergence (`native-fock`, through `cutoff_convergence(..., backend="native-fock")`).
- `notebooks/ml/01_quickstart.ipynb`: first MuTA model.
- `notebooks/ml/02_muta.ipynb`: triangle neuron anatomy.
- `notebooks/ml/03_gate_learning.ipynb`: gate learning.
- `notebooks/ml/04_quantum_kernel.ipynb`: MuTA quantum kernel.
- `notebooks/ml/05_gkp_resources.ipynb`: comparing qubit, GKP and CV MuTA.

Example: see `notebooks/01_quickstart.ipynb`.

### PhotoGraphiQML tutorials (`docs/ml/tutorials/*.md` with `examples/ml/tutorials/*.py`)

| Field | Value |
|---|---|
| Description | 44 tutorials (01-30 logical MuTA and training; 31-44 the physical signed-X GKP bridge), with an all-in-one Colab notebook `examples/ml/PhotoGraphiQML_tutorials.ipynb`. Index: [ML tutorials](ml/tutorials/index.md). |
| Python API | `python examples/ml/tutorials/<NN>.py` |
| Implementation | `examples/ml/tutorials/`, `docs/ml/tutorials/`, executed by `scripts/ml/execute_tutorials.py` |
| Backends | Logical NumPy; `piquasso-fock` for the physical tutorials (32-44, through `PhysicalMuTA`/`GKPBridge` defaults or explicit `backend="piquasso-fock"`) |
| CPU / GPU | CPU |
| Assumptions | The tutorials' assertions are run by `scripts/ml/execute_tutorials.py`. Three tutorials were retitled so that the titles match what they do: 17 "Training with a supplied gradient" (an L-BFGS gradient callable; no automatic differentiation), 18 "Finite-squeezing noise of a Gaussian wire" (one channel diagnostic, not a noisy MuTA model) and 29 "CV teleportation cell (CVMuTA proposal)" (one ingredient of the proposed cell; CVMuTA is not implemented). The new titles are in `docs/ml/tutorials/17.md`, `18.md`, `29.md` and `index.md`, in the generator `scripts/ml/create_learning_materials.py`, and in the headings of `examples/ml/PhotoGraphiQML_tutorials.ipynb` (17 there reads "Supplying an analytic gradient (L-BFGS)"). |
| Tests | `scripts/ml/execute_tutorials.py` (not pytest). Executed here: `34_restricted_physical_muta.py` and `42_unsupported_arbitrary_xy_why_it_fails.py` (both ran; 42 prints the expected `UnsupportedPhysicalMeasurement` message). |
| Status | **Implemented but experimental** |

- 01 What is MB-QML?; 02 MuTA from the paper; 03 First MuTA model; 04 Triangle neuron anatomy; 05 Multi-wire MuTA; 06 Multi-layer MuTA; 07 Trainable measurement angles; 08 Classical feature encoding; 09 Quantum state inputs; 10 Gate learning.
- 11 State classification; 12 Classical classification; 13 Quantum instrument learning; 14 MuTA quantum kernel; 15 Training with Adam; 16 Finite-difference validation; 17 Training with a supplied gradient; 18 Finite-squeezing noise of a Gaussian wire; 19 GKP logical MuTA; 20 MentPy cross-validation (needs `ml-validation`).
- 21 Expressivity vs depth; 22 Tunable entanglement; 23 Bias engineering; 24 Fisher information; 25 Model serialization; 26 Custom observable readout; 27 Custom loss; 28 Training callbacks; 29 CV teleportation cell (CVMuTA proposal); 30 Comparing qubit, GKP and CV MuTA.
- 31 PhotoGraphiQ 0.3.1 encoded interface; 32 Physical GKP X measurement (`piquasso`); 33 Physical GKP Z measurement (`piquasso`); 34 Restricted physical MuTA; 35 Zero-angle physical MuTA; 36 Pauli-frame propagation; 37 Hard vs soft decoding; 38 Multi-mode joint readout; 39 Physical shots and statistical error; 40 Logical vs physical comparison; 41 Physical convergence; 42 Unsupported arbitrary XY: why it fails; 43 Logical training to physical validation; 44 Discrete physical-angle search. (34-44 use `PhysicalMuTA` on `piquasso-fock`.)

Example: see `examples/ml/tutorials/34_restricted_physical_muta.py`.

### PhotoGraphiQML experiments (`experiments/ml/`)

| Field | Value |
|---|---|
| Description | Reproducible finite-resource evidence scripts for the ML layer. |
| Python API | `python experiments/ml/<name>/main.py`, `python experiments/ml/restricted_physical.py` |
| Implementation | `experiments/ml/` |
| Backends | Logical NumPy; `piquasso-fock` for `restricted_physical.py` |
| CPU / GPU | CPU |
| Assumptions | The scripts state that they do not certify convergence. |
| Tests | None |
| Status | **Implemented but experimental** |

- `experiments/ml/paper_reproduction/`: logical gate-learning reproduction with `config.json` and saved `results/` (needs `matplotlib` and MentPy: it calls `compare_mentpy` at checkpoints). Its README now gives the run path `python experiments/ml/paper_reproduction/main.py` and the install `.[ml-validation,ml-experiments]`.
- `experiments/ml/kernel_classification/`: MuTA-kernel classification of synthetic classical datasets against classical baselines (needs `scikit-learn` and `matplotlib`). Its README now gives `python experiments/ml/kernel_classification/main.py` and `.[ml-experiments]`. `docs/ml/release-report-v0.1.md` still cites the old `experiments/paper_reproduction/...` and `experiments/kernel_classification/...` paths.
- `experiments/ml/restricted_physical.py`: restricted physical MuTA evidence (bridge, classifier, discrete search) without a convergence certificate (`piquasso`).

Example: see `experiments/ml/paper_reproduction/README.md`.

### Core manuscript experiment suite (`paper_experiments/`)

| Field | Value |
|---|---|
| Description | Experiments R1-R42 in 13 topic folders (core MBQC, compiler, adaptive flow, Piquasso validation, Graphix validation, backend cross-checks, non-Gaussian, GKP, noise, convergence, performance, serialization, autodiff). They write CSV, JSON and metadata under `results/` and figures under `figures/`. `PhotoGraphiQ_Manuscript_Experiments.ipynb` combines all scripts (generated by `build_notebook.py`). |
| Python API | `python paper_experiments/run_all_safe.py` (lightweight subset); `python paper_experiments/<folder>/<script>.py` |
| Implementation | `paper_experiments/` (see `EXPERIMENT_INDEX.md`, `STATUS.md`, `ISSUES_FOUND.md`, `MANUSCRIPT_MAP.md`) |
| Backends | `gaussian`, Piquasso-backed Fock backends, Graphix (R17), JAX (R41, R42) |
| CPU / GPU | CPU |
| Assumptions | Needs the `dev` extra. Without Graphix or JAX, R17 and R41-R42 print `SKIPPED` and exit 0. R18, R20-R23, R26-R28 and R33-R34 use Fock cutoffs up to 96 and take minutes. `run_all_safe.py` skips the experiments tagged optional dependency, large Fock or long benchmark. The performance experiments' timings are machine-dependent and are not quoted here. |
| Tests | None in pytest; `STATUS.md` records the latest run |
| Status | **Implemented but experimental** |

Example: see `paper_experiments/README.md`.

### PhotoGraphiQML manuscript experiment suite (`paper_experiments_ml/`)

| Field | Value |
|---|---|
| Description | 59 experiments (R1-R48, R_PERF, R50-R59) in 13 folders: logical foundations, MentPy validation, parameter contracts, gradients and training, kernels, expressivity, GKP resources, physical lowering, Piquasso validation, physical statistics, convergence, performance and extended studies. Each experiment has a declared oracle class (A-E, N/A) and tolerances. `PhotoGraphiQML_Manuscript_Experiments.ipynb` is a Colab notebook. |
| Python API | `python paper_experiments_ml/run_all_safe.py`, `run_all_full.py`, `generate_tables.py`, `verify_evidence.py`, `build_notebook.py` |
| Implementation | `paper_experiments_ml/` (see `EXPERIMENT_INDEX.md`, `EXPERIMENT_CONTRACTS.json`, `EXPERIMENT_MANIFEST.json`, `STATUS.md`, `ISSUES_FOUND.md`) |
| Backends | Logical NumPy, `piquasso-fock`, MentPy (02_mentpy_validation), scikit-learn, Plotly (notebook atlas) |
| CPU / GPU | CPU |
| Assumptions | Needs `ml-experiments`, plus `ml-validation` for the MentPy experiments (MentPy is not installed in this environment). The physical-shot experiments take minutes. |
| Tests | `tests/ml/paper_experiments/test_experiment_consistency.py` (structure, contracts, evidence schema and status validation; no re-execution of the physics) |
| Status | **Implemented but experimental** |

Example: see `paper_experiments_ml/README.md`.

## 34. Scientific validation

PhotoGraphiQ is validated in layers. Each layer compares the code with a reference that does not share the code path under test: closed forms, brute-force definitions, external libraries (Piquasso, Graphix, mpmath, MentPy) or a second implementation. The project's own statement of this policy is in [validation philosophy](validation/index.md), [numerical policy](validation/numerical-policy.md) and the [feature matrix](validation/feature-matrix.md). The policy says that a same-backend round trip checks API integrity but is not evidence of physical correctness, that Graphix is never a continuous-variable numerical oracle, and that "a configured CI workflow is not a passing run". The test runs quoted below were made on the development machine (Windows 11, Python venv, native extension built, Piquasso, Graphix, JAX, mpmath and CuPy installed, CUDA-Q and MentPy absent; the MentPy comparison was run in a separate isolated venv). The GitHub workflow `.github/workflows/ci.yml` runs on `ubuntu-latest` only, and this document does not claim a passing remote run: the 2026-11-12 CI fixes (notebook lint, the `numba<0.68` constraint, installed-package test imports) were reproduced locally on Windows but not pushed. The 2026-11-12 audit added the layers listed after the existing ones in this section: a first-principles validation matrix for the Gaussian and Fock engines, a 60-digit reference for the native one-mode Gaussian unitary, a 382-case reference matrix for threshold probabilities (in the mpmath entry), exact rational CV-flow checks, a real-browser check of the interactive pages, the MentPy comparison, and the release gates. Audit reports: [Phase 1](validation/PHASE1_RELEASE_AUDIT.md), [Phase 2](validation/PHASE2_SCIENTIFIC_AUDIT.md), [Phase 3](validation/PHASE3_FEATURE_COMPLETION.md).

### Tolerance policy

`docs/validation/numerical-policy.md` sets one acceptance strategy per kind of test. It also says: "Never relax a tolerance merely to make CI pass", keep failing regimes and document their error mechanism, and treat coverage as "a software execution metric, not a numerical-accuracy certificate".

| Test class | Error budget and acceptance strategy (from `docs/validation/numerical-policy.md`) | Example in the suite |
|---|---|---|
| Exact finite algebra | Typically 1e-13 to 2e-12 absolute; complete number sectors and phases preserved | `tests/cross_backend/test_cross_backend_agreement.py` uses `ROUNDOFF = 1e-10` for Gaussian moments across Gaussian backends; `tests/sampling/test_sampling_gbs.py::test_two_mode_squeezed_vacuum_closed_form` uses `rel=1e-12` |
| Finite-Fock approximation | Reference and production cutoffs resolved separately; failed coarse rows retained | The `tests/cross_backend/test_cross_backend_agreement.py` docstring tabulates the observed error at cutoff c and c-4 and sets each tolerance from the observed convergence (for example `wire` at cutoff 16: rho error 1.5e-4, tolerance 5e-4) |
| Integration | Grid/window refined independently; analytic Gaussian integrals as an extra oracle | `tests/gkp/test_independent_convergence.py::test_grid_peak_and_cutoff_are_independent`; `tests/non_gaussian/test_finite_squeezing_kraus_reference.py::test_oracle_grid_is_converged` |
| Stochastic estimator | Fixed seeds, measured standard errors, a five-standard-error check, variance reported | `tests/autodiff/test_estimators.py` (`abs(gradient - expected) < 5 * stderr`); chi-square p > 1e-3 in `tests/sampling/` |
| Product formula | Phase-aligned amplitude error fitted over several slice counts above the numerical floor | `tests/compilation/test_quartic_validation.py::test_quartic_commutator_amplitude_order` |

### Map of test directories to references

| `tests/` subdirectory | Main independent references used |
|---|---|
| `analytical/` | Closed forms: Schur-complement homodyne conditioning, detector efficiency and generaldyne moments, wire and Braunstein-Kimble noise channels, nullifiers, CV-flow channels; the Banchi-Braunstein-Pirandola fidelity formula (`test_gaussian_diagnostics.py`, module skips without Piquasso) |
| `autodiff/` | Closed-form gate gradients, refined finite differences, Piquasso backends at low energy, five-standard-error estimator checks |
| `calculus/` | Exact statevectors and Kraus maps, ZX tensor semantics, Graphix 0.4 statevector (`test_calculus_graphix.py`); direct Gaussian covariance algebra and `GaussianBackend` conditioning for `GaussianGraph` (`test_gaussian_graph.py`) |
| `compilation/` | Direct target exponentials, commutator and Weyl identities, refinement of slice counts and cutoffs |
| `cudaq_adapter/` | `native-fock` / `native-mixed-fock` gates and loss through the NumPy/SciPy reference path (no CUDA-Q call) |
| `cross_backend/` | NumPy `gaussian` backend converted to a Fock density matrix through Piquasso's Gaussian-state `density_matrix`; cat-counting and homodyne-density closed forms; registry behaviour |
| `cutoff/` | Cutoff refinement (fidelity to the previous cutoff, not only unit norm) |
| `flow/` | Graphix 0.4 flow finders (`to_causalflow_or_none`, `to_gflow_or_none`, `to_pauliflow_or_none`), independent statevector determinism, CV teleportation channels, exact rational witnesses and the exact ideal-limit check (`test_flows_cv_robustness.py`) |
| `fock/` | `tests/fock/reference.py`, a dense test-only oracle built with `scipy.linalg.expm` ("never imported by the production package"); closed forms |
| `gkp/` | Closed Gaussian overlap integrals, independent Hermite projection, independent continuous comb density, independent SUM wavefunction kernel, explicit Kraus sums |
| `gpu/` | NumPy `GaussianBackend` / `GaussianState`, native C++ batch kernels, native C++ Fock kernels and backends (`test_gpu_fock.py`) |
| `graphix_reference/` | Graphix 0.4 structure only: labels, edges, measurement order, correction domains |
| `integration/` | End-to-end behaviour (no independent numerical reference) |
| `ml/` | Closed-form logical unitaries (`muta/test_logical.py::test_triangle_and_table_one`), `piquasso-fock` against `native-fock`, contract checks; MentPy (`mentpy_reference/`, skipped here) |
| `mixed_fock/` | Independent matrix exponential, Gaussian backend, binomial and coherent closed forms, thermal-environment dilation, pure-state limit, physicality (trace, Hermiticity, positivity) |
| `native/` | NumPy backends, Piquasso, exact Gaussian moments, `expm` of exact generators, direct numerical integration, 60-digit mpmath matrix elements (`test_native_fock_gaussian_unitary_stability.py`) |
| `non_gaussian/` | Finite-squeezing Kraus oracle, closed forms, heralding limits |
| `piquasso_reference/` | Raw Piquasso 8.0.1 programs |
| `protocols/` | Closed-form channels and postselection densities, mixed-Fock and Fock postselection |
| `regression/` | Frozen behaviour and error messages (`test_validation_messages.py`: self-CZ, repeated modes, pair outcomes without `component=`, backend names in the missing-cutoff error, non-finite expressions; `test_measure_angle.py`: a direct `measure` uses the `Homodyne` angle); regression only, no independent reference |
| `sampling/` | By-definition permutation and matching sums, closed forms, Piquasso, chi-square tests, mpmath |
| `unit/` | Round trips and rejection checks (no independent reference); `test_native_only_smoke.py` runs the native paths with every optional package blocked |
| `validation/` | References computed from first principles inside the test (closed forms, `scipy.linalg.expm` on enlarged spaces, mpmath amplitudes, physical dilations, precision-matrix conditioning); no PhotoGraphiQ function builds a reference |
| `visualization/` | Rendering and data plumbing (no independent reference); `test_interactive_browser_playwright.py` compares drawn elements with flows, patterns and simulator traces in a real browser (skips without Playwright) |

### Analytical and closed-form references

| Field | Value |
|---|---|
| Description | Tests compare outputs with formulas that are derived independently of the implementation: Gaussian moments, Schur-complement conditioning, noise channels, Born probabilities, closed-form GBS and boson-sampling probabilities, gate gradients and GKP overlap integrals. |
| Python API | Not an API; the references are written inside the tests (for example `exp(-2r)`, `1/cosh^2 r`, `tanh(r)^(2n)/cosh^2 r`) |
| Implementation | `tests/analytical/`, `tests/protocols/`, closed-form cases in `tests/sampling/`, `tests/mixed_fock/`, `tests/gkp/`, `tests/autodiff/test_analytical_gradients.py` |
| Backends | All backends that the tested feature supports |
| CPU / GPU | CPU (NumPy); the GPU tests reuse the NumPy references |
| Assumptions | hbar=2, [q,p]=2i, interleaved xpxp, vacuum covariance I. A closed form validates only the regime it covers; finite-cutoff results are compared with resolved references, and the gap between the two is identified explicitly (numerical policy). |
| Tests | `tests/analytical/test_measurements.py::test_exact_homodyne_schur_complement`, `::test_detector_efficiency_and_generaldyne`; `tests/analytical/test_channels.py::test_wire_noise_and_bk_noise_exact`; `tests/analytical/test_primitives.py::test_weighted_graph_nullifiers`; `tests/sampling/test_sampling_gbs.py::test_two_mode_squeezed_vacuum_closed_form` (`rel=1e-12` for P(no click) = 1/cosh^2 r); `tests/mixed_fock/test_loss_channels.py::test_binomial_channel_all_limits`; `tests/gkp/test_independent_convergence.py::test_finite_logical_overlap_matches_closed_gaussian_integral`; `tests/autodiff/test_analytical_gradients.py` (2e-8 or 2e-12) |
| Status | **Implemented and validated** |

```python
import numpy as np
import photographiq as pg
r = 0.4
s = pg.simulate(pg.Pattern(inputs=(0,)).append(pg.Squeeze(0, r)), backend="gaussian").state
print(np.round(s.covariance, 12))
print(np.allclose(np.diag(s.covariance), [np.exp(-2 * r), np.exp(2 * r)], atol=1e-14))
# [[0.44932896 0.        ]
#  [0.         2.22554093]]
# True
```

### Brute-force and by-definition references

| Field | Value |
|---|---|
| Description | Small cases are checked against the defining sums or against dense operators built independently of the production code: permutation and perfect-matching sums for permanents and hafnians, dense `expm` Fock operators, independent matrix exponentials for density-matrix gates, and exact statevector/Kraus evaluation of qubit patterns and ZX diagrams. |
| Python API | `photographiq.sampling.permanent_by_definition(matrix)`, `hafnian_by_definition(matrix)`, `loop_hafnian_by_definition(matrix)`; test-only oracle `tests/fock/reference.py` |
| Implementation | `src/photographiq/sampling/linalg.py` (by-definition sums); `tests/fock/reference.py`; `tests/mixed_fock/test_independent_unitaries.py`; exact branch simulation in `src/photographiq/calculus/qubit.py` used as the map oracle |
| Backends | Standalone sampling module; Fock and mixed-Fock backends; qubit calculus |
| CPU / GPU | CPU only |
| Assumptions | Factorial cost: permanents n <= 8, hafnians 2n <= 12. There is no by-definition torontonian; its references are closed-form products and Piquasso (Section 20). `tests/fock/reference.py` is never imported by the package. |
| Tests | `tests/sampling/test_sampling_linalg.py::test_permanent_matches_definition`, `::test_hafnians_match_definition`; `tests/sampling/test_sampling_boson.py::test_amplitudes_match_creation_operator_expansion`; `tests/mixed_fock/test_independent_unitaries.py::test_unitary_against_independent_exponential`; `tests/calculus/test_calculus_qubit.py::test_each_rewrite_rule_preserves_the_map`, `::test_standardization_preserves_map_and_reaches_nemc`; `tests/calculus/test_calculus_zx.py::test_every_rewrite_is_scalar_exact_on_random_diagrams`, `::test_pattern_branch_diagrams_equal_kraus_operators` |
| Status | **Implemented and validated** |

```python
import numpy as np
from photographiq import sampling as S
a = np.random.default_rng(0).normal(size=(6, 6)) + 1j * np.random.default_rng(1).normal(size=(6, 6))
print(abs(S.permanent(a) - S.permanent_by_definition(a)) / abs(S.permanent_by_definition(a)) < 1e-12)
# True
```

### Raw Piquasso references

| Field | Value |
|---|---|
| Description | Programs written directly in Piquasso 8.0.1, outside PhotoGraphiQ's adapters, serve as references for Gaussian states and conditioning, Fock evolution, boson-sampling and GBS probabilities, and the native C++ Fock engines. |
| Python API | Backends `piquasso`, `piquasso-fock`, `piquasso-mixed-fock` (these use Piquasso internally); `photographiq.backends.PiquassoBackend().import_state(state, nodes, *, source_hbar)` for hbar conversion |
| Implementation | `tests/piquasso_reference/`; Piquasso comparisons in `tests/sampling/`, `tests/native/`, `tests/mixed_fock/`, `tests/autodiff/` |
| Backends | Reference for `gaussian`, `native-gaussian`, `native-fock`, `native-mixed-fock` and the sampling module |
| CPU / GPU | CPU only |
| Assumptions | Requires the `piquasso` extra (pinned `piquasso==8.0.1`); the tests skip without it. The feature matrix states that "shared adapters alone are not independent": only an explicit raw Piquasso program counts as a Piquasso reference. Piquasso's Euler (Bloch-Messiah) truncation differs from the native exact projected Gaussian gates at fixed cutoff, so Fock comparisons use cutoff convergence rather than equality (Section 13). Piquasso pairs torontonian indices (2i, 2i+1), PhotoGraphiQ (i, i+n). |
| Tests | `tests/piquasso_reference/test_raw_piquasso.py` (`test_raw_graph_state_and_displacement`, `test_raw_generaldyne_conditional_state`, `test_ideal_compiler_target_against_raw_gaussian_circuit`, `test_external_state_hbar_conversion`); `tests/piquasso_reference/test_fock_convergence.py::test_fock_cz_raw_reference_and_cutoff_convergence`; `tests/sampling/test_sampling_gbs.py::test_matches_piquasso_gaussian_simulation` (`abs=1e-13`); `tests/sampling/test_sampling_boson.py::test_matches_piquasso_fock_simulation`; `tests/native/test_native_fock_backend.py::test_non_gaussian_patterns_match_piquasso` |
| Status | **Implemented and validated** |

```python
import numpy as np
import piquasso as pq
import photographiq as pg
r = 0.3
with pq.Program() as prog:
    pq.Q(0) | pq.Squeezing(r)
raw = pq.GaussianSimulator(d=1, config=pq.Config(hbar=2.0)).execute(prog).state
mine = pg.simulate(pg.Pattern(inputs=(0,)).append(pg.Squeeze(0, r)), backend="gaussian").state
print(np.abs(raw.xpxp_covariance_matrix / 2 - mine.covariance).max() < 1e-14)
# True
```

### Graphix references

| Field | Value |
|---|---|
| Description | Graphix 0.4 is used in two different roles. For qubit MBQC it is a numerical and structural reference: flow existence, certificates, standardization, signal shifting and branch statevectors. For CV patterns it checks only MBQC structure (labels, edges, input/output roles, measurement order, s/t correction domains); it never checks a continuous-variable number. |
| Python API | `photographiq.flows.find_causal_flow`, `find_gflow`, `find_pauli_flow`, `photographiq.calculus.qubit` (compared objects); Graphix `OpenGraph.to_gflow_or_none()` etc. (reference) |
| Implementation | `tests/graphix_reference/`, `tests/flow/test_flows_qubit.py`, `tests/calculus/test_calculus_graphix.py`; benchmark `benchmarks/graphix/bench_graphix_mbqc.py` |
| Backends | Qubit calculus and flows; structure of CV `Pattern` objects (no CV backend is compared) |
| CPU / GPU | CPU only |
| Assumptions | Requires the `validation` extra (`graphix==0.4`); the tests skip without it. Plane conventions match Graphix 0.4 (`test_graphix_plane_convention_is_ours`), which uses angles in units of pi. Agreement in CV structure says nothing about Gaussian or Fock numerics. |
| Tests | `tests/flow/test_flows_qubit.py::test_existence_agrees_with_graphix` (150 random open graphs per finder; PhotoGraphiQ's verifier accepts Graphix's certificates); `tests/calculus/test_calculus_graphix.py` (`test_branch_states_match_graphix_statevector`, `test_standardization_matches_graphix`, `test_signal_shifting_matches_graphix`, `test_graphix_plane_convention_is_ours`); `tests/graphix_reference/test_structure.py`, `test_labelled_structures.py`, `test_flow_structure.py`, `test_non_gaussian_structure.py` (CV structure only) |
| Status | **Implemented and validated** (qubit MBQC numerics and structure; CV patterns: structure only) |

```python
import networkx as nx
from graphix.fundamentals import Plane as GPlane
from graphix.opengraph import OpenGraph as GOpenGraph
import photographiq as pg
from photographiq.ir import Model, QubitMeasurement
from photographiq.flows import find_gflow
g = nx.path_graph(4)
m = {v: QubitMeasurement("XY", 0.0) for v in (0, 1, 2)}
og = pg.OpenGraph(Model.QUBIT, g, inputs=(0,), outputs=(3,), measurements=m)
theirs = GOpenGraph(g, [0], [3], {v: GPlane.XY for v in (0, 1, 2)}).to_gflow_or_none()
print(find_gflow(og).certified, theirs is not None)
# True True
```

### mpmath multiprecision reference

| Field | Value |
|---|---|
| Description | The threshold (torontonian-type) inclusion-exclusion sum loses digits as clicks increase and probabilities shrink. mpmath evaluates the same sum with a chosen number of decimal digits, both as a user option and as a test and benchmark reference. |
| Python API | `photographiq.sampling.threshold_probability(state, clicks, *, precision=<digits> \| "auto")`; `benchmarks/gaussian_boson_sampling/threshold_mp_reference.py` (50 digits); `benchmarks/gaussian_boson_sampling/threshold_reference_matrix.py` (independent 40-digit implementation, 382 cases) |
| Implementation | `src/photographiq/sampling/gbs.py` (`_vacuum_sum_mp`) |
| Backends | Standalone sampling module. The mpmath evaluation does not use the native or NumPy kernels, but `backend=` is still validated (`ValueError("Unknown backend ...")`; `backend="native"` raises `ImportError` without the native extension). Before the fix `precision=20, backend="bogus"` returned a value. |
| CPU / GPU | CPU only (pure Python) |
| Assumptions | Requires the `precision` extra (`mpmath>=1.3`). The covariance is used exactly as stored in double precision, so the reference is exact for the stored state, not for the ideal one. The double-precision rounding bound is checked against `precision=30` on 12 cases (small and large squeezing, displaced, lossless, no dark modes, three zero-probability patterns, 1 to 3 clicks at tiny squeezing) on both backends, plus the 16-mode / 8-click case. The alternating-sum part of that bound was calibrated against mpmath (4140 random states), so these references are also its calibration data. Its own error estimate `P_Z mp.eps sum_T abs(1 + w_T)` does not strictly include rounding in the conditioning step, which is negligible at 30 or more digits. |
| Tests | `tests/sampling/test_sampling_gbs.py::test_threshold_multiprecision_matches_closed_forms` (`rel=1e-15`), `::test_threshold_rounding_bound_holds_against_multiprecision` (12 cases x 2 backends), `::test_threshold_error_estimate_bounds_multiprecision_reference` (16 modes, 8 clicks; `1e-6 < error < 3e-5`), `::test_threshold_precision_validates_backend`, `::test_threshold_auto_precision_switches_to_mpmath`. Reference matrix (2026-11-12, separate code from `precision=`): over 382 cases the double-precision bound was never violated on either backend, no warning was missed, double precision failed only at r <= 0.5 with many clicks (as did Piquasso), and `precision="auto"` stayed below `rtol = 1e-6` everywhere (Section 20). |
| Status | **Implemented and validated** |

```python
import math
import numpy as np
from photographiq import sampling as S
r = 0.8
tmsv = S.squeezed_interferometer_state([r, -r], np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2))
exact = 1 - 1 / math.cosh(r) ** 2
print(abs(S.threshold_probability(tmsv, [1, 1], precision=30) / exact - 1) < 1e-15)
# True
```

### MentPy reference (PhotoGraphiQML)

| Field | Value |
|---|---|
| Description | Logical MuTA models are compared with the MentPy MuTA template: graph, input/output nodes, trainable nodes and measurement order exactly, and density matrices from MentPy's `PatternSimulator` numerically. |
| Python API | `photographiqml.validation.mentpy_reference(model, *, fix_measurements=False)`, `photographiqml.validation.compare_mentpy(model, input_state, parameters=None)` |
| Implementation | `src/photographiqml/validation.py`; `tests/ml/mentpy_reference/`; workflow `.github/workflows/ml-reference.yml` |
| Backends | MentPy `numpy-sv` against `MuTA.run` (logical, ideal qubit) |
| CPU / GPU | CPU only |
| Assumptions | MentPy is not installed in the project venv, so `tests/ml/mentpy_reference/` skips there. On 2026-11-11/12 it was run in an isolated Windows venv (Python 3.12.14, non-editable install, MentPy 0.1.0a15 from the workflow's git commit `63c3d83e495696b4491c9d376dab7e3e6cf6c863`, `numba` 0.67.0): 34/34 reference tests and 154/154 of `tests/ml` passed, and all 24 scratch-copy experiments, including the MentPy-dependent ones, passed with every non-timing value bit-identical to the recorded JSON. In physical MuTA, `native-fock` and `piquasso-fock` probabilities differ by at most 1.5e-4 at cutoff 24 and 9e-5 at cutoff 32 (truncation scale) with identical decoded bits. The numerical test asserts a discrepancy below 1e-10. Evidence: [PhotoGraphiQML MentPy validation](validation/photographiqml-mentpy-validation.md). |
| Tests | `tests/ml/mentpy_reference/test_reference.py` (`test_exact_semantic_reference`, `test_numerical_reference`), `tests/ml/mentpy_reference/test_derived_outputs.py` (`test_kernel_against_reference`, `test_instrument_marginals_against_reference`); in this environment only `tests/ml/muta/test_core_dependency.py` (missing-dependency path) runs |
| Status | **Implemented and validated** (run in an isolated venv against MentPy; not run in the project venv, and no passing CI run is claimed) |

Example: see `tests/ml/mentpy_reference/test_reference.py` (requires MentPy).

### Cross-backend agreement

| Field | Value |
|---|---|
| Description | The same pattern is run on every available registered backend whose capabilities cover it. Gaussian-representation backends must agree with the NumPy `gaussian` backend to float roundoff. Fock and mixed-Fock backends are compared with the Gaussian result converted to a Fock density matrix by an independent code path, with tolerances derived from observed cutoff convergence. |
| Python API | `photographiq.simulate(pattern, backend=<name>, ...)` for every name in `photographiq.available_backends()` |
| Implementation | `tests/cross_backend/test_cross_backend_agreement.py`, `tests/mixed_fock/test_cross_backend.py`, `tests/protocols/test_covariance_symmetry.py`, `tests/protocols/test_gaussian_postselection.py` |
| Backends | `gaussian`, `piquasso`, `native-gaussian`, `piquasso-fock`, `native-fock`, `piquasso-mixed-fock`, `native-mixed-fock` |
| CPU / GPU | CPU only |
| Assumptions | `docs/validation/index.md`: Gaussian backends share a measurement adapter, so agreement among them alone does not validate detector statistics; analytical moments and densities supply that evidence. Backends lacking a feature are skipped with the missing feature named; `test_expected_coverage` pins which backends must run each case. |
| Tests | `tests/cross_backend/test_cross_backend_agreement.py` (`test_gaussian_pattern_agreement` (Gaussian backends `atol=1e-10`; Fock tolerances from the docstring table), `test_wire_cutoff_convergence_justifies_tolerance`, `test_cat_photon_counting_closed_form`, `test_homodyne_postselection_density_closed_form`, `test_expected_coverage`); `tests/protocols/test_gaussian_postselection.py::test_lossy_inefficient_noisy_homodyne_matches_mixed_fock`; `tests/mixed_fock/test_cross_backend.py::test_overlapping_gaussian_regime_across_backends` |
| Status | **Implemented and validated** |

```python
import numpy as np
import photographiq as pg
p = (pg.Pattern(inputs=(0, 1)).append(pg.Squeeze(0, 0.3)).append(pg.Entangle(0, 1, 0.7))
     .append(pg.Measure(0, pg.Homodyne(0.4), key="m")))
ref = pg.simulate(p, backend="gaussian", seed=5)
for name in ("native-gaussian", "piquasso"):
    r = pg.simulate(p, backend=name, seed=5)
    print(name, abs(r.outcomes["m"] - ref.outcomes["m"]) < 1e-10,
          np.abs(r.state.covariance - ref.state.covariance).max() < 1e-10)
# native-gaussian True True
# piquasso True True
```

### Native C++ against NumPy

| Field | Value |
|---|---|
| Description | Each compiled component is checked against a Python implementation of the same quantity. `native-gaussian` is checked against `gaussian` with equal seeds. The `sampling` kernels are checked against the NumPy permanent/hafnian/torontonian code, with results independent of thread count. The native Fock engines are checked against Piquasso and exact references (Section 13). |
| Python API | `backend="native-gaussian"` vs `"gaussian"`; `backend="native"` vs `"python"` on every `photographiq.sampling` matrix function; `photographiq.native.components()`, `photographiq.native.build_info()` |
| Implementation | `cpp/src/gaussian/`, `cpp/src/fock/`, `cpp/src/sampling/`; Python counterparts in `src/photographiq/backends/` and `src/photographiq/sampling/linalg.py` |
| Backends | `native-gaussian`, `native-fock`, `native-mixed-fock`, native sampling kernels |
| CPU / GPU | CPU (C++/OpenMP) against CPU (NumPy) |
| Assumptions | Equal seeds give equal trajectories up to floating-point summation order (`src/photographiq/backends/native_gaussian.py` docstring reports about 1e-13). Thread-count independence is asserted bitwise for `permanent`, `hafnian` and `torontonian` only. `cpp/tests/test_sampling.cpp` is built only with `PGQ_BUILD_CPP_TESTS=ON`. |
| Tests | `tests/native/test_native_gaussian_backend.py` (`test_random_patterns_match_numpy_backend`, `test_postselection_matches_numpy_backend`, `test_errors_match_numpy_backend`, `test_near_singular_conditioning_large_squeezing`, `test_tiny_variance_homodyne_closed_form`); `tests/sampling/test_sampling_native.py` (`test_native_agrees_with_python_at_larger_sizes` (1e-10 relative), `test_results_do_not_depend_on_thread_count`); `tests/sampling/test_sampling_gbs.py::test_python_vacuum_sum_raises_like_native_for_non_positive_blocks` (both threshold kernels raise the same `ValueError` for a non-positive-definite block; the NumPy path used to return NaN); `tests/native/test_native_fock_kernels.py`, `tests/native/test_native_mixed_fock_backend.py` (including `test_multi_kraus_instrument_matches_explicit_partial_trace`, against an explicit partial trace) |
| Status | **Implemented and validated** (the native Fock engines are now also validated against first-principles references, Section 13) |

```python
import numpy as np
from photographiq import sampling as S
a = np.random.default_rng(0).normal(size=(10, 10)); a = a + a.T
h_native, h_python = S.hafnian(a, backend="native"), S.hafnian(a, backend="python")
print(abs(h_native - h_python) / abs(h_python) < 1e-12)
# True
```

### GPU kernels against NumPy

| Field | Value |
|---|---|
| Description | Every numerical GPU test runs for `device="cpu"` and `device="gpu"`. Gaussian results are compared with the NumPy `GaussianBackend` / `GaussianState` state by state and with the native C++ batch kernels; Fock results (`photographiq.gpu.fock`) with the native C++ Fock kernels and whole native backend gate sequences. |
| Python API | `photographiq.gpu.apply_symplectic(...)`, `condition_homodyne(...)`, `purity(...)`, `symplectic_eigenvalues(...)`, `fidelity(...)` with `device="gpu"` or `"cpu"` |
| Implementation | `src/photographiq/gpu/`; `tests/gpu/` |
| Backends | Not a `simulate` backend; batched Gaussian arrays only |
| CPU / GPU | GPU via photographiq.gpu (CuPy) compared with CPU (NumPy) |
| Assumptions | float64 by default; float32 is checked only against float64 within rtol 1e-4 (conditioning) and 1e-5 (symplectic update), and is experimental (Section 15). GPU cases skip when CuPy or a device is missing. The result's `execution.device` is read from the output buffer. |
| Tests | `tests/gpu/test_gpu_gaussian.py` (`test_apply_symplectic_matches_reference_backend`, `test_condition_homodyne_matches_reference_backend`, `test_diagnostics_match_gaussian_state`, `test_gpu_and_cpu_paths_agree` (rtol 1e-12), `test_matches_native_batch_kernels`, `test_float32_is_explicit_and_close`); `tests/gpu/test_gpu_device.py::test_gpu_results_report_and_live_on_the_gpu`; `tests/gpu/test_gpu_fock.py` (29 tests; 7.2e-16 complex128, 2.4e-7 complex64) |
| Status | **Implemented and validated** (float64 / complex128; float32 and complex64 experimental) |

```python
import numpy as np
from photographiq import gpu
B, n = 4, 2
means = np.zeros((B, 2 * n)); covs = np.repeat(np.eye(2 * n)[None], B, axis=0)
r = 0.3; S = np.diag([np.exp(-r), np.exp(r)])
mats = np.repeat(S[None], B, axis=0)
g = gpu.apply_symplectic(means, covs, [0], mats, device="gpu")
c = gpu.apply_symplectic(means, covs, [0], mats, device="cpu")
print(g.execution.device, c.execution.device, np.abs(g.covariances - c.covariances).max() < 1e-12)
# cuda:0 cpu True
```

### Truncation and resource-refinement checks

| Field | Value |
|---|---|
| Description | Finite representations are refined one control at a time: the total-photon Fock cutoff, quadrature grids, GKP peak counts, product-formula slice counts and resource squeezing. Tests require the error against an independent reference to decrease, not only the norm to be close to one. |
| Python API | `photographiq.cutoff_convergence(pattern, cutoffs, *, seed=0, high_order_moments=False, backend="piquasso-fock", **kwargs)` (any of `piquasso-fock`, `piquasso-mixed-fock`, `native-fock`, `native-mixed-fock` since 2026-11-12; the default is kept for reproducibility); `photographiq.measurement_convergence(..., backend="piquasso-fock" \| "native-fock")`; backend `diagnostics` (retained norm, boundary population); `norm_tolerance`, `boundary_warning`, `max_dimension` backend options |
| Implementation | `src/photographiq/backends/fock.py` (`_check`), `src/photographiq/backends/mixed_fock.py`, `src/photographiq/convergence.py`; tests in `tests/cutoff/`, `tests/gkp/`, `tests/compilation/` |
| Backends | `piquasso-fock`, `native-fock`, `piquasso-mixed-fock`, `native-mixed-fock`; GKP resources; synthesis |
| CPU / GPU | CPU only |
| Assumptions | Total-photon cutoff sum n < cutoff. Norm monitoring "detects many cutoff problems but is not a rigorous Fock error certificate" (`docs/validation.md`). Convergence is shown for the tested regimes only. A resolved infinite-space reference and the exponential of a projected generator can differ at finite cutoff, and the tests keep the two apart. |
| Tests | `tests/cutoff/test_convergence.py::test_strong_cubic_convergence_not_just_unit_norm`, `::test_finite_cubic_injection_cutoff_convergence`, `::test_truncation_warnings_and_rejection`; `tests/cross_backend/test_cross_backend_agreement.py::test_wire_cutoff_convergence_justifies_tolerance`; `tests/piquasso_reference/test_fock_convergence.py::test_fock_cz_raw_reference_and_cutoff_convergence`; `tests/gkp/test_independent_convergence.py::test_grid_peak_and_cutoff_are_independent`; `tests/gkp/convergence/test_modular_reference.py::test_grid_and_peak_refinement_on_decoded_probabilities`; `tests/compilation/test_quartic_validation.py::test_quartic_refinement_and_independent_target` |
| Status | **Implemented but experimental** (refinement evidence holds for tested regimes only and is not a certificate) |

```python
import warnings
import photographiq as pg
alpha = 1.5
for cutoff in (6, 10, 14, 20):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # retained-norm diagnostics
            st = pg.simulate(pg.Pattern(inputs=(0,)), backend="native-fock", cutoff=cutoff,
                             inputs={0: pg.GaussianInput.coherent(alpha)}).state
    except ValueError as exc:
        print(cutoff, "ValueError:", exc)
        continue
    mean_n = sum(k[0] * p for k, p in st.probabilities.items())
    print(cutoff, f"{abs(mean_n - alpha**2):.1e}")
# 6 ValueError: Fock truncation norm 0.97263465; increase cutoff=6
# 10 9.7e-04
# 14 1.4e-06
# 20 9.6e-12
```

### Seeded statistical checks

| Field | Value |
|---|---|
| Description | Samplers and stochastic estimators are checked against exact distributions or expected values with fixed seeds: chi-square goodness of fit for boson-sampling and GBS samplers, and a five-standard-error bound for gradient estimators. A negative control checks that the chi-square test rejects distinguishable particles. |
| Python API | `photographiq.sampling.sample(...)`, `sample_pnr(...)`, `sample_threshold(...)`; `photographiq.autodiff` estimators |
| Implementation | `tests/sampling/test_sampling_boson.py`, `tests/sampling/test_sampling_gbs.py`, `tests/autodiff/test_estimators.py`, `tests/analytical/test_measurements.py::test_seed_zero_and_independent_shots` |
| Backends | Sampling module; JAX autodiff executor; Gaussian trajectory backends |
| CPU / GPU | CPU only |
| Assumptions | Thresholds such as p > 1e-3 and 5 standard errors bound the false-failure rate for the fixed seed; they are not proofs. Shots must greatly exceed the number of outcome patterns for the total-variation figures in benchmarks to be meaningful. |
| Tests | `tests/sampling/test_sampling_boson.py::test_exact_samplers_reproduce_distribution` (8000 shots, chi-square), `::test_chi_square_check_rejects_distinguishable_particles`; `tests/autodiff/test_estimators.py::test_pathwise_gaussian_bias_and_reproducibility`, `::test_bernoulli_score_statistical_bias_variance_and_seed` |
| Status | **Implemented and validated** |

```python
import numpy as np
from scipy import stats
from photographiq import sampling as S
U = S.haar_unitary(3, rng=7)
exact = S.output_distribution(U, [1, 1, 0])
samples = S.sample(U, [1, 1, 0], 8000, rng=np.random.default_rng(1))
keys = sorted(exact)
counts = {k: 0 for k in keys}
for row in map(tuple, samples):
    counts[row] += 1
obs = np.array([counts[k] for k in keys]); exp = 8000 * np.array([exact[k] for k in keys])
print(stats.chisquare(obs, exp * obs.sum() / exp.sum()).pvalue > 1e-3)
# True
```

### Benchmark correctness checks

| Field | Value |
|---|---|
| Description | Each benchmark computes a correctness value in an untimed call before timing, so that a fast but wrong implementation cannot appear in the results. Boson-sampling and GBS rows carry `rel_error` against the defining sum (small sizes), the PhotoGraphiQ Python implementation or Piquasso. `bench_native_gaussian.py` aborts if any output differs from native by more than 1e-12 (scaled). `bench_fock.py` runs every gate and pattern case once on the native engine and on its Piquasso counterpart and raises `RuntimeError` if the trace distance (and, for patterns, the relative difference of each outcome statistic) exceeds the row's `agreement_tol`; thread rows compare checksums across thread counts. `bench_gpu_gaussian.py` records `max_rel_deviation_vs_numpy_float64`; `bench_gpu_fock.py` stops if any output deviates from the complex128 native loop by more than 1e-12 (complex64: 1e-5). The Graphix benchmark checks every certificate with both verifiers and checks map equivalence by exact branch simulation. `threshold_mp_reference.py` gives each implementation's actual threshold error against 50-digit mpmath, and `threshold_reference_matrix.py` the actual error, the bound and the warning of every case against an independent 40-digit implementation. `scripts/compare_recorded_results.py` compares regenerated experiment values with the recorded ones. |
| Python API | `benchmarks/boson_sampling/bench_boson_sampling.py`, `benchmarks/gaussian_boson_sampling/bench_gbs.py`, `benchmarks/gaussian_boson_sampling/threshold_mp_reference.py`, `benchmarks/gaussian/bench_native_gaussian.py`, `benchmarks/fock/bench_fock.py`, `benchmarks/gpu/bench_gpu_gaussian.py`, `benchmarks/graphix/bench_graphix_mbqc.py` (scripts, not package API) |
| Implementation | `benchmarks/` (methodology in `benchmarks/README.md`) |
| Backends | Native and NumPy implementations, Piquasso 8, Graphix 0.4, CuPy |
| CPU / GPU | CPU; `bench_gpu_gaussian.py` also GPU via CuPy |
| Assumptions | `bench_fock.py` uses `ROUNDING_TOL = 1e-12` where both engines apply the same truncated operator, and `max(1e-12, sqrt(p_boundary))` for the CZ and whole patterns, where the truncation schemes differ. The second tolerance is an empirical scale, not a proven bound: it held with a ratio of at most 0.098 on 598 test inputs at the default norm guard, but reached 1.39 at `norm_tolerance=0.9` (Section 32). The GBS `rel_error_vs_piquasso` for threshold rows mixes the errors of both libraries. The threshold rows' `estimated_rel_error` is the new, larger rounding bound, so it is not comparable with result files written before 2026-11-11; the figures quoted in `benchmarks/README.md` (2.1e-5, 2.9e-3, 0.52) are the old estimate from such a file. No timing numbers are quoted in this document; the reference run is the 2026-11-12 bundle `benchmarks/results_20261112/` (Section 32). |
| Tests | `tests/sampling/test_sampling_benchmark_smoke.py::test_benchmark_smoke` (every recorded relative error below 1e-8 in the smoke configuration), `::test_time_call_excludes_cold_first_call`; `tests/native/test_native_gaussian_benchmark_smoke.py` (`test_native_gaussian_benchmark_smoke`, `test_agreement_check_rejects_wrong_output`, `test_chain_state_matches_dense_reference`); `tests/native/test_fock_benchmark_smoke.py` (`test_fock_benchmark_smoke`, `test_agreement_check_rejects_wrong_output`, `test_memory_fields_are_none_without_psutil`, and `test_threshold_mp_reference_output_option` for the argument parsing of `threshold_mp_reference.py`); `tests/flow/test_flows_graphix_benchmark_smoke.py::test_graphix_benchmark_tiny_run`. No test covers `bench_gpu_gaussian.py`, `bench_gpu_fock.py`, `threshold_reference_matrix.py` or the numbers of `threshold_mp_reference.py`; `tests/unit/test_compare_recorded_results.py` covers the comparison script. |
| Status | **Implemented and validated** (boson-sampling, GBS, native Gaussian and Graphix scripts; the Fock, GPU and mpmath-reference scripts are experimental, Section 32) |

Example: see `tests/sampling/test_sampling_benchmark_smoke.py` and `tests/native/test_native_gaussian_benchmark_smoke.py`.

### Independent validation matrix for the Gaussian and Fock engines (`tests/validation/`)

| Field | Value |
|---|---|
| Description | Peer-review audit (2026-11-12) of the NumPy `gaussian`, C++ `native-gaussian`, `native-fock` and `native-mixed-fock` engines, with the Piquasso backends as a secondary check. Every reference value is computed inside the test from first principles with NumPy, SciPy or mpmath; no PhotoGraphiQ function builds a reference, and the engines' own outputs and diagnostics are only read back. It also checks every convention against the source (hbar = 2, interleaved ordering, vacuum I, coherent mean, squeezing, CZ, beam splitter, rotation, homodyne angle, detector inefficiency, heterodyne, loss, total-photon truncation) and found them consistent on all seven backends. |
| Python API | Not an API: `pytest -q -p no:cacheprovider tests/validation` |
| Implementation | `tests/validation/test_val_gaussian.py`, `tests/validation/test_val_fock.py`; table in [independent validation matrix](validation/independent-validation-matrix.md) |
| Backends | `gaussian`, `native-gaussian`, `piquasso` (Gaussian); `native-fock`, `native-mixed-fock`, with `piquasso-fock` and `piquasso-mixed-fock` as a secondary check (looser tolerances: 1e-10, and 1e-4 for their truncated CZ) |
| CPU / GPU | CPU |
| Assumptions | Gaussian: states, gate symplectics built as `expm(2 Omega M)` from each generator, loss against analytic X, Y and a beam-splitter dilation, homodyne (with efficiency and noise) and heterodyne conditioning against a physical dilation conditioned through the precision matrix, heterodyne density against the Husimi function, feed-forward against T(k), physicality of random 5-mode circuits, TMSV log-negativity; largest error 6.5e-13 (tolerance 1e-11). Fock: states against mpmath and the regularised incomplete gamma function, all gates against `expm` on 40/56 (one mode) and 26/32 (two modes) levels per mode with the reference checked between sizes (below 4e-16), truncation handled exactly as P U psi / norm(P U psi), counting, partial traces, homodyne densities, loss as a dilation, unitarity of number-conserving gates and physicality after 21 gates; largest gate error 5e-16 (tolerance 1e-12). Caveat: the `piquasso` Gaussian backend reuses NumPy code for input preparation, measurements and diagnostics, so those rows re-test NumPy. The audit found one API defect (a direct `measure` ignored `Homodyne.angle`, fixed, Section 10) and one misleading comment (mixed-Fock noise is convolved in calibrated units). |
| Tests | The suite itself: 130 collected, 128 pass and 2 skip (Piquasso Fock has no cross-Kerr) |
| Status | **Implemented and validated** |

```bash
python -m pytest -q -p no:cacheprovider -rs tests/validation
# SKIPPED [2] tests\validation\test_val_fock.py:308: Piquasso Fock backends have no cross-Kerr method
# 128 passed, 2 skipped in <elapsed>
```

### High-precision reference for the native one-mode Gaussian unitary

| Field | Value |
|---|---|
| Description | Matrix elements <m\|U\|n> of the native one-mode Gaussian unitary are compared with a Bargmann recurrence evaluated in mpmath at 60 digits, with S built exactly symplectic in mpmath. The reference is itself checked against the Heisenberg relation a U = U (T a + R a^dag + delta) on every interior element (residual at most 6e-26), and 60- and 90-digit references agree to the last double bit. |
| Python API | Not an API: `pytest tests/native/test_native_fock_gaussian_unitary_stability.py`; C++ check `cpp/tests/test_fock.cpp::test_single_mode_stability` (built with `-DPGQ_BUILD_CPP_TESTS=ON`) |
| Implementation | `tests/native/test_native_fock_gaussian_unitary_stability.py`; study in [native Fock Gaussian unitary](validation/native-fock-gaussian-unitary.md) |
| Backends | `photographiq._native.fock.gaussian_unitary`, used by `native-fock` and `native-mixed-fock` |
| CPU / GPU | CPU (mpmath) |
| Assumptions | Grid of the study: abs(alpha) in {0.5, ..., 6}, r in {0.1, 0.3, 0.6, 1.0}, random phases (seed 2026), cutoff 20 to 100: at most 1.9e-14 after the fix, against up to 1.4e2 before. Squeezing alone was checked up to size 800. Multi-mode displaced unitaries are not covered (Section 13). |
| Tests | The file itself (6 tests; 5 fail on the build before the fix) |
| Status | **Implemented and validated** |

```bash
python -m pytest -q -p no:cacheprovider tests/native/test_native_fock_gaussian_unitary_stability.py
# 6 passed in <elapsed>
```

### Exact rational CV-flow checks

| Field | Value |
|---|---|
| Description | CV-flow decisions are checked in exact rational arithmetic: impossibility through rational witnesses that the tests verify independently with `fractions.Fraction`, and the ideal limit through `exact_ideal_limit`, which is compared with the closed-form wire map and rejects every scaled correction vector. Qubit flows remain checked against Graphix 0.4 on 150 random graphs per notion (causal 31 positive / 119 negative, gflow 27 / 123, Pauli 42 / 108). |
| Python API | `photographiq.flows.find_cv_flow(og, exact=True)`, `photographiq.flows.exact_ideal_limit` (Section 26) |
| Implementation | `tests/flow/test_flows_cv_robustness.py`; study in [CV-flow robustness](validation/cv-flow-robustness.md) |
| Backends | Backend-independent |
| CPU / GPU | CPU |
| Assumptions | Covers the adversarial graph set of the study (near-singular and rounding-level determinants, weights from 1e-4 to 1e4, long wires and grids, large squeezing, slow decay, dynamic range) and 600 random adversarial graphs. The exact check covers the ideal limit only; `finite_squeezing` stays a float64 statement. |
| Tests | The file itself (25 tests) plus `tests/flow/test_flows_cv.py` and `tests/flow/test_flows_qubit.py::test_existence_agrees_with_graphix` |
| Status | **Implemented and validated** |

```bash
python -m pytest -q -p no:cacheprovider tests/flow/test_flows_cv_robustness.py
# 25 passed in <elapsed>
```

### Real-browser check of the interactive pages

| Field | Value |
|---|---|
| Description | Drives the interactive HTML pages with real mouse, wheel and keyboard input in Edge or Chrome through Playwright and compares what is drawn with the Python model, the flow result, the `Pattern` commands and a real `simulate` trace (Section 30). |
| Python API | `python scripts/check_interactive_browser.py OUT_DIR [--channel msedge\|chrome]` |
| Implementation | `scripts/check_interactive_browser.py`, `tests/visualization/test_interactive_browser_playwright.py`; report in [interactive visualization browser check](validation/interactive-visualization-browser.md) with screenshots in `docs/validation/assets/interactive/` |
| Backends | Independent of backends (the CV wire uses `simulate(..., backend="gaussian", seed=7)`) |
| CPU / GPU | CPU; the viewer's browser |
| Assumptions | Needs Playwright (not a declared extra) and network access to the pinned Cytoscape.js CDN or `PHOTOGRAPHIQ_CYTOSCAPE_JS`. Run once on 2026-11-12 with Playwright 1.65.1 and Edge 157 in an isolated venv: all 9 checks passed after four fixes. Layout quality and readability are judged by inspection, not by an independent reference. |
| Tests | `tests/visualization/test_interactive_browser_playwright.py` (skips here: Playwright is not installed in the project venv) |
| Status | **Implemented but experimental** (a single recorded run outside the project venv; visual checks are by inspection) |

```bash
python -m pytest -q -p no:cacheprovider -rs tests/visualization/test_interactive_browser_playwright.py
# SKIPPED [1] tests\visualization\test_interactive_browser_playwright.py:12: could not import 'playwright.sync_api': No module named 'playwright'
# 1 skipped in <elapsed>
```

### Release gates, Python-version matrix and native-only smoke test

| Field | Value |
|---|---|
| Description | Release engineering checks of 2026-11-12. `experiments/validate_release.py` runs pytest with coverage, ruff check, ruff format, mypy, the package build and the physics-evidence checks; `scripts/check_public_api.py` checks the public exports; `scripts/check_docs_examples.py --projects --notebooks` runs every tutorial, project and notebook. A native-only smoke test runs the native Gaussian, Fock and mixed-Fock paths in a subprocess with Piquasso, Graphix, JAX, mpmath, CuPy, MentPy and CUDA-Q blocked, and checks the "install the extra" errors. Tests now import the installed package (`pythonpath = ["src"]` was removed), so native tests also run against a non-editable wheel. |
| Python API | `python experiments/validate_release.py`; `python scripts/check_public_api.py`; `python scripts/check_docs_examples.py --projects --notebooks`; `pytest tests/unit/test_native_only_smoke.py` |
| Implementation | `experiments/validate_release.py`, `scripts/`, `tests/unit/test_native_only_smoke.py`, `tests/regression/test_measure_angle.py`, `pyproject.toml`; report in [Phase 1 release audit](validation/PHASE1_RELEASE_AUDIT.md) |
| Backends | All registered backends |
| CPU / GPU | CPU |
| Assumptions | Local Windows runs only. Gate run (Python 3.12): 1273 passed, 14 skipped, coverage 93.90 % (floor 90 %); 85 public exports checked; all tutorials, 10 projects and 5 notebooks passed. Fresh non-editable installs on Python 3.11, 3.13 and 3.14: 1645 passed and 61 skipped each after the `numba<0.68` constraint (one Numba 0.68 crash before it). The remote CI was not re-run, because the fixes are not pushed, and no Linux environment was available. |
| Tests | `tests/unit/test_native_only_smoke.py::test_core_and_native_paths_run_without_optional_packages`; `tests/regression/test_measure_angle.py` (6 tests) |
| Status | **Implemented but experimental** (software gates, not an independent physics reference; remote CI not re-run) |

```bash
python -m pytest -q -p no:cacheprovider tests/unit/test_native_only_smoke.py tests/regression/test_measure_angle.py
# 7 passed in <elapsed>
```

## 35. Known limitations

This section collects limitations stated in Sections 1 to 33, in the code and in the project documentation. Each row names the affected feature, what goes wrong or is refused, and what to do instead. "Raises" means that the code rejects the request explicitly. None of these cases produces a silently wrong answer unless the row says so.

### Unsupported and partially supported features

| Limitation | Affected feature | Consequence | Workaround |
|---|---|---|---|
| No CUDA-Q pattern execution | `photographiq.cudaq_dynamics` (Section 16) | CUDA-Q is not a backend; `simulate(backend="cudaq-dynamics")` raises `ValueError` naming the adapter. The adapter evolves Hamiltonians only; without `cudaq`, `to_cudaq`/`evolve` raise `ImportError` naming the Linux-only wheels. Its CUDA-Q path is unverified. | Use the CPU Fock backends for patterns; `reference_evolve` for Hamiltonian dynamics |
| No GPU trajectory or sampling execution | `simulate`, `run_shots`, all registered backends, `photographiq.sampling` (Section 15) | `simulate(..., backend="gpu")` raises `ValueError: Unknown backend: 'gpu'`; all pattern execution, conditioning and sampling run on the CPU. GPU permanents, hafnians and torontonians were measured and not shipped (single matrices slower than native; batched gains only for many small same-size matrices, which no workload uses) | `native-gaussian` (C++/OpenMP) for single trajectories; `photographiq.gpu` batched Gaussian kernels for many states with given outcomes and `photographiq.gpu.fock` for batched Fock gates |
| GPU Fock kernels help only batched workloads | `photographiq.gpu.apply_operator`, `apply_phase` (Section 15) | A single small state is slower on the GPU than the native kernel; phase gates gain nothing end to end over NumPy; host-device transfers dominate end-to-end time for large batches; no renormalisation or diagnostics; the kernels do not accelerate `simulate` | Keep batches on the device (`on_device=True`); use the native engine for single states |
| Photon counting on Gaussian backends | `PhotonNumber` with `gaussian`, `piquasso`, `native-gaussian` | Rejected by the capability check; a direct `measure` call raises `NotImplementedError("Photon counting has non-Gaussian conditional states; use ...")`, naming `native-fock`, `native-mixed-fock`, `piquasso-fock` and `piquasso-mixed-fock` | A Fock or mixed-Fock backend; GBS probabilities via `photographiq.sampling` |
| Noisy or inefficient homodyne on pure Fock backends | `piquasso-fock`, `native-fock` | `NotImplementedError("Noisy Fock homodyne requires mixed-state conditioning")` | `piquasso-mixed-fock` or `native-mixed-fock` |
| Heterodyne and generaldyne on Fock backends | All four Fock backends | `NotImplementedError` ("Fock backend supports ideal Homodyne and PhotonNumber measurements only" / "Mixed Fock supports PhotonNumber and Homodyne") | Gaussian backends |
| No threshold detector measurement class | Click detection inside patterns (Section 10) | **Partially supported**: only through a user-built `MeasurementInstrument` on mixed-Fock backends | Build the instrument {\|0><0\|, I - \|0><0\|}; use `photographiq.sampling.threshold_probability` for Gaussian states |
| Cross-Kerr has no Pattern command | Cross-Kerr (Sections 9 and 13) | **Partially supported**: method `cross_kerr(u, v, kappa)` on native Fock backend instances only; cannot be used in patterns, serialized or planned | Call the method on a backend instance |
| Multi-Kraus instruments need mixed states | `MeasurementInstrument` on pure Fock backends | Preflight rejects outcomes with more than one Kraus operator; instrument dimension must equal the cutoff (`ValueError`) | `piquasso-mixed-fock` or `native-mixed-fock` |
| Mixed Gaussian inputs on pure Fock backends | `GaussianInput` with covariance above vacuum | `NotImplementedError("Mixed Gaussian inputs require a mixed Fock backend")` | A mixed-Fock backend |
| Physical GKP Y and arbitrary XY readout | `GKPCode.logical_measurement`, `PhysicalGKPReadout` (Sections 10 and 27) | `"Y"` raises `NotImplementedError("Only physical GKP X/Z readout is implemented; Y/XY need validation")`; `"XY"` works only at alpha = 0 or pi (mod 2 pi, tolerance 1e-14), otherwise `NotImplementedError` | `IdealLogicalXYMeasurement` as an ideal qubit reference (not an oscillator POVM) |
| No magic-state injection or logical Rz | `LogicalMeasurementSynthesis().lower(...)`; non-Clifford logical gates | Always raises `NotImplementedError("No validated magic-state injection or arbitrary logical Rz protocol")`; only X, Z, H, S and CZ exist | None in the package |
| No fault-tolerant GKP architecture or threshold | GKP layer (Section 27) | No concatenated code, threshold estimate or optimal finite-energy decoder; the feature matrix lists "Fault-tolerant GKP thresholds" as unsupported | None; see Section 36 |
| CV local complementation | `photographiq.calculus.clifford.cv_local_complementation` | Always raises `NotImplementedError` (no sound CV analogue derived or verified) | Qubit `local_complement` / `lc_equivalence` for qubit graphs only |
| Bialgebra rule limited to 2x2 | `photographiq.calculus.zx` | **Partially supported**: other arities, nonzero phases, self-loops raise `RewriteError` | Other sound rewrites (Section 23) |
| No determinism analysis for non-Gaussian or GKP-logical graphs | `photographiq.flows.analyze` with `Model.CV_NONGAUSSIAN` / `Model.GKP_LOGICAL` | No exception; result `status == "unsupported"` with a reason | None; `not_certified` is never a proof of nondeterminism either |
| Unconditional channel analyzer needs affine feed-forward | `photographiq.analysis` | `NotImplementedError` for outcome-dependent Gaussian gates, callable channels, vector outcomes, non-Gaussian preparation and non-homodyne measurements | Sample trajectories with `run_shots` |
| Physical MuTA restricted to signed-X | `PhysicalMuTA`, `lower_muta_to_gkp`, `GKPBridge.run` (Section 28) | **Partially supported**: angles other than 0 or pi raise `UnsupportedPhysicalMeasurement`; entangled inputs raise `NotImplementedError`; mixed-Fock backends and `decoder="soft"` raise `ValueError`; `unitary()` and `run_batch()` raise `NotImplementedError`; final readout X or Z only | Train the logical `MuTA`, then validate the physical model at {0, pi} |
| Resource-only GKP MuTA cannot execute | `MuTA(representation="gkp-resource")` (`"gkp"` is deprecated) | `MuTA.run` raises `NotImplementedError` ("Physical GKP MuTA requires a validated logical XY measurement/injection instrument and decoder ...") | `PhysicalMuTA` |
| Physical quantum-output instrument model | `QuantumInstrumentModel` | Requires logical MuTA; physical version raises `NotImplementedError` | Logical model only |
| No continuous optimization of physical angles | PhotoGraphiQML training (Section 29) | Angles are categorical; gradient training of physical angles is refused | Discrete coordinate search over {0, pi} |
| No differentiation through sampled trajectories | Full stochastic MBQC differentiation (Section 29) | Unsupported; `photographiq.autodiff` differentiates fixed branches only | Fixed-branch JAX gradients, score-function and pathwise estimators (experimental) |
| JAX executor covers a subset | `photographiq.autodiff` | Thermal loss (`"Differentiable loss currently requires a vacuum environment"`), noisy homodyne, other measurements and Python callables raise; `jax_enable_x64` required (`RuntimeError`) | Restrict to vacuum loss, ideal homodyne and PNR |
| No noise models beyond loss and detector noise | All backends | Phase diffusion, mode mismatch and complete hardware noise models are not implemented (`docs/specification-status.md`) | Model them externally through `Loss`, detector efficiency/noise or custom instruments |
| Runtime callables are not portable | `CallableExpression`, PhotoGraphiQML frame signals | Patterns using them cannot be serialized to JSON (rejected, no pickle or eval) | Use `Parameter` and expression arithmetic |
| Some outcome reads are checked only at execution | `Pattern.validate()` and `Outcome(key, component=...)` | `validate()` rejects a Heterodyne/Generaldyne `(q, p)` record read without `component=`. It does not reject a component other than 0 or 1 on such a key (`IndexError` at execution), `component=` on a scalar Homodyne or `Signal` key (`TypeError` at execution), or reads inside a `CallableExpression`, which cannot be checked without running it. `float()` also accepts a string parameter value such as `"1.5"`. | Use `component=0` (q) or `1` (p) on pair outcomes only; run the pattern once on a cheap backend |
| JAX expression evaluation does not check finiteness | `photographiq.autodiff.resolve` (used by `fock_state`, `expectation`) | **Silently wrong value possible**: it uses its own traceable JAX operator table, so an overflowing, infinite, NaN or zero-division expression returns `inf` or `nan` instead of the `ValueError("Expression must evaluate to a finite real scalar")` raised by `Expr.evaluate` | Check `Expr.evaluate` on the bound values first, or check the JAX result with `jnp.isfinite` |

### Truncation, finite resources and precision

| Limitation | Affected feature | Consequence | Workaround |
|---|---|---|---|
| Total-photon truncation sum n < cutoff | All Fock and mixed-Fock backends, `pnr_distribution`, GKP resources | States are renormalised after each operation; the retained norm and the boundary population (weight at total photon number >= cutoff - 2) are recorded in `diagnostics`. Norm loss above `norm_tolerance` (default 1e-3) raises `ValueError("Fock truncation norm ...; increase cutoff")`; boundary population above `boundary_warning` (0.02) warns. Renormalising a truncated tail does not turn it into physical loss. | Repeat at increasing cutoffs (`cutoff_convergence`) and inspect `diagnostics`; norm monitoring is not a rigorous error certificate |
| Engines truncate differently | `native-fock` / `native-mixed-fock` vs `piquasso-fock` / `piquasso-mixed-fock` | Native uses exact projected Gaussian matrix elements; Piquasso truncates after each Euler factor. Results differ at truncation level, and near the norm guard one engine can raise while the other continues. | Compare engines only at converged cutoffs |
| Two cubic-phase definitions | `CubicPhase` on native Fock backends | `"truncated_generator"` (default, as Piquasso) never reports leakage; `"projected"` reports leakage and may trip the norm guard. Both converge with the cutoff. | Choose the definition explicitly and check convergence |
| Multi-mode displaced Gaussian unitaries have no error bound | `photographiq._native.fock.gaussian_unitary` with k >= 2 modes and d != 0 (Section 13) | The one-mode case was fixed on 2026-11-12 (at most 1.9e-14 against 60-digit mpmath for abs(alpha) <= 6, abs(r) <= 1, cutoff <= 100; rounding outside that range not measured). Multi-mode unitaries still use the Bargmann recurrence, whose rounding error is not bounded in general and grows with cutoff and squeezing (3e-12 measured at abs(delta) <= 4, r <= 0.6, cutoff <= 40 on two modes). The backends never request such a unitary (CZ, beam splitter and Gaussian transforms have d = 0); only a direct kernel call can. Hot mixed `GaussianInput`s (size up to 4096) now cost O(size^2 L) and can take minutes | Apply the displacement as a separate one-mode gate; compare with a larger cutoff |
| Memory guards | Fock and mixed-Fock backends | `max_dimension` (default 1,000,000) raises `MemoryError` before allocation; mixed-Fock `max_matrix_bytes` (default 256,000,000, counts 16 D^2 bytes, not gate workspaces) raises `MemoryError`; native mixed thermal inputs needing more than 4096 Fock terms raise `NotImplementedError` | Fewer live modes (lazy `Prepare`), lower cutoff after a convergence check, or a Gaussian backend |
| Finite squeezing | Cluster resources, teleportation gadgets, CV flow and determinism (Sections 8, 11, 26) | Finite-squeezing gates add Gaussian noise (for one wire step N = diag(0, e^{-2r})). CV determinism verification uses postselected runs as a numerical certificate only; they are not deterministic execution, and the conditional output remains noisy at finite r | Report the channel (S, N) from `ideal_limit_check` or the compiled pattern; increase r to approach the ideal limit |
| CV determinism undecided at large squeezing | `verify_cv_determinism`, `find_cv_flow`, `analyze(..., "cv")` (Section 26) | The false "unknown" results at r of about 3.1 to 3.5 are fixed: the law-of-total-covariance check now has a rounding allowance that scales with the largest covariance entry. For r above about 3.5 (on the wire; from about 2.9 on some random graphs) the raised levels r + 3 and r + 6 still fail `GaussianState.validate`, whose absolute -1e-7 eigenvalue bound does not scale with the covariance. The result is then `unknown` ("could not run"), never a false certification, and near the threshold it can flip between "could not run" and certified with tiny changes of r (4.0 against 4.000000000000002). A few adversarial graphs (7 of 600) give `unknown` from a slow decay exponent (about 0.65) although the flow is certified. The allowance itself is a first-order estimate, not a rigorous bound. | Pass `shift=1.5` to `find_cv_flow` or `analyze` (exposed since 2026-11-12; certifies the unit wire up to r = 4.5 in the tests), or `exact=True` for the exact ideal-limit check, which does not depend on r; otherwise treat `unknown` as undecided; `unknown` is never a proof of nondeterminism |
| CV-flow certification limits | `find_cv_flow`, `analyze(..., "cv")`, `exact_ideal_limit` (Section 26) | An uncertain float64 rank decision gives `unknown` (it gave `not_certified` before 2026-11-12). The automatic exact re-check runs only on graphs of at most 40 nodes; above that, or with `exact=False`, an "impossible" result is a float64 backward-error statement, not a proof, and badly scaled systems stay uncertain. The exact path costs about O(n^4) rational operations and is not meant for hundreds of nodes. The exact ideal-limit conditions are sufficient, not necessary. `finite_squeezing` certificates are float64 statements only (no exact finite-squeezing check), and with correction gains above 1/sqrt(eps) only the ideal-limit statement is meaningful. Unit-efficiency detection and lossless resources are assumed. The 600-graph adversarial run was not repeated after the last rule change | Use `exact=True` on small graphs; read `diagnostics['decision']`, `diagnostics['proof']` and `certificate['regimes']` |
| Postselected results are single branches | `measurement_outcomes=`, `mode="physical-conditional"` | The result is one conditional branch, reported with its probability or density in `measurement_statistics`; it is not a deterministic outcome | Sample with `run_shots` or physical shots |
| Finite-energy GKP | `GKPResource`, `GKPCode`, decoders | Codewords overlap (`encode` is not isometric); leakage cannot be inferred from one analog outcome (`None`); `captured_weight` diagnoses only the Fock projection, and peaks and grid need separate refinement | Refine cutoff, `grid_points` and `peaks` separately (`measurement_convergence`, `validate_physical_model`) |
| Product-formula synthesis is approximate | Cubic/Kerr/quartic synthesis (`src/photographiq/synthesis.py`) | Finite-energy injection, Fock projection and product-formula error are separate limits; no returned step count is an a-priori operator-norm guarantee | Refine slice counts and cutoffs; compare with direct targets |
| Double-precision threshold probabilities | `threshold_probability`, `threshold_distribution`, `sample_threshold` | The alternating inclusion-exclusion sum loses digits as clicks grow and P shrinks. On the 382-case reference matrix double precision failed (relative error above 1e-6) in 80 of 360 random cases (native) and 72 (Python), all at r <= 0.5 with many clicks; Piquasso fails in the same regime and can return 0.0 for nonzero probabilities without a warning. The `return_error=True` value is `B / abs(P)` for an absolute bound `B` whose conditioning part is derived and whose alternating-sum part is first order and calibrated against mpmath (factor `max(1, 16 / sqrt(2^k - 1))`), not proven; it held in every comparison so far. Negative results and results within `B` of zero always warn; `precision=` cannot resolve a value that is exactly zero, and a value negative beyond `B` means the stored covariance is unphysical at this accuracy. The bound does not include rounding of the stored covariance itself. The native kernel refuses more than 40 clicked modes. `threshold_distribution` and `sample_threshold` do not take `precision`, and mpmath is pure Python (16 clicks take minutes per probability) | `precision="auto"` (recomputes in mpmath only when needed; below `rtol` on every reference-matrix case) or `precision=<digits>`, both exact for the stored double-precision covariance (extra `precision`) |
| Threshold rounding bound is loose | `threshold_probability(..., return_error=True)`, `ThresholdPrecisionWarning` | On the reference matrix `B / actual` has median 620 (native) and 1100 (Python) and reaches 8.7e6 / 1.3e7, loosest at large squeezing. The warning (relative bound above 1e-6) therefore raised 37 native and 45 Python false alarms whose actual errors were 3.5e-11 to 8.2e-7; no real failure went unwarned | Treat a warning as "not certified", not as "wrong"; use `precision="auto"` to settle it |
| Exact GBS samplers enumerate | `sample_pnr`, `sample_threshold` | Cost grows with the number of output patterns; no chain-rule or approximate large-scale sampler exists; PNR samples beyond the cutoff appear as `-1` rows (`overflow`). Both samplers clip negative probabilities no larger than 1e-12 of the total mass (rounding) and raise `ValueError("... negative beyond rounding ...")` for larger ones; `sample_threshold` used to clip every negative silently. Observed threshold negatives grow with the mode count (-2.4e-14 at 14 modes), so the raise is expected only near 18 to 20 modes, where enumeration is already too slow. | Keep modes and cutoff small; read `truncated_mass` |
| Exponential matrix functions | Permanent, hafnian, torontonian | Exponential cost; by-definition oracles limited to n <= 8 (permanent) and 2n <= 12 (hafnian) | Native kernels (`backend="native"`) for larger sizes |
| float32 GPU option | `photographiq.gpu` with `dtype="float32"` | Validated only against float64 on moderate states (rtol 1e-4 / 1e-5); error grows with squeezing | Use float64 |
| Truncated JAX derivatives | `photographiq.autodiff` | Gradients are exact for the truncated finite-cutoff model, not for the infinite-dimensional physics | Check both the observable and its gradient at increasing cutoff |

### Validation scope

| Limitation | Affected feature | Consequence | Workaround |
|---|---|---|---|
| Graphix checks only MBQC structure for CV | `tests/graphix_reference/` and the `validation` extra | Agreement covers labels, edges, measurement order and correction domains; Graphix is never a CV numerical oracle (for qubit MBQC it also checks flows and branch statevectors) | Use the analytical, Piquasso and cross-backend layers for CV numerics |
| Shared measurement adapter | Gaussian backends | Agreement among `gaussian`, `piquasso` and `native-gaussian` alone does not validate detector statistics | Analytical moments and densities (`tests/analytical/`) |
| Experimental labels | Piquasso Fock execution, non-Gaussian gates in patterns, synthesis, GKP, JAX differentiation, PhotoGraphiQML physical path | The code and `docs/validation/feature-matrix.md` call all Fock execution experimental. Since 2026-11-12 this document lists the native Fock engines and their gates, states, counting, homodyne, loss and instruments as validated because `tests/validation/` gives them first-principles references; the feature matrix has not been changed to match | Treat other Fock results as research-grade; repeat convergence studies for new regimes |
| MentPy comparison only in an isolated venv | PhotoGraphiQML logical models | MentPy is absent from the project venv, so the reference tests skip there and 9 MentPy-dependent manuscript experiments fail in the reproduction bundle. The comparison passed (34/34) in an isolated venv on 2026-11-11/12; the workflow installs a git commit while the extra pins `mentpy==0.1.0a15` | Install MentPy (`ml-validation` extra or the commit in `tests/ml/mentpy_reference/requirements.txt`) and run `tests/ml/mentpy_reference` |
| Behavioural tests only | `OpenGraph`, registry and planner, layouts, static exports, plot rendering, lazy imports, memory guard | Checked for round trips, rejections and file validity, with no independent reference | None needed for correctness of physics; read statuses in Sections 2, 5, 14 and 30 |
| Fock benchmark agreement tolerance is empirical | `benchmarks/fock/bench_fock.py` | The script now checks native against Piquasso before timing. For the CZ and whole patterns the tolerance `max(1e-12, sqrt(p_boundary))` is an empirical scale, not a bound; it failed (ratio 1.39) on a heavily truncated state at `norm_tolerance=0.9`, so it holds only for states the default norm guard (1e-3) admits | Keep the default `norm_tolerance`; rely also on `tests/native/` for engine agreement |
| Some native mixed-Fock paths lack a direct comparison | `native-mixed-fock` | Multi-Kraus instruments are now tested against an explicit partial trace, but rank-one instruments and physical GKP readout are compared with Piquasso only on the pure engines (`native-fock` vs `piquasso-fock`); `native-mixed-fock` shares the Python measurement code but has no dedicated test for them | Cross-check against `piquasso-mixed-fock` for new cases |
| No passing remote CI claimed | `.github/workflows/ci.yml`, `ml-reference.yml` | Workflows are configured for `ubuntu-latest` only; this document reports local runs on Windows. The 2026-11-12 fixes for the failing `tests` and `documentation` workflows (notebook lint and format, the Numba 0.68 crash, installed-package imports) were reproduced locally but not pushed, so the remote CI has not been re-run, and no Linux environment was available | Run `experiments/validate_release.py` locally; a push by the maintainer is needed to confirm CI |
| CUDA-Q execution unverified | `photographiq.cudaq_dynamics.to_cudaq`, `evolve` (Section 16) | No CUDA-Q call has ever run (Linux-only wheels; no WSL or Docker here). The documented signatures, degree ordering, NumPy input on the `dynamics` target, integrator accuracy and reported device are unchecked; `test_real_cudaq_evolve_matches_reference` skips. Only the conventions are validated, through the NumPy/SciPy reference path | Run `tests/cudaq_adapter` on Linux with CUDA-Q >= 0.16 before relying on `evolve` |
| Interactive layout limits | `photographiq.interactive` pages and static exports (Section 30) | Straight edges and fixed-bend arcs can pass under other nodes, so a dependency can read like a chain through an intermediate node, and an edge between collinear nodes is hidden; three-line outcome labels overflow the node; the flow panel can push the execution controls below the fold; labels on a 6 x 6 cluster are 8.8 px at default zoom; static `#k` labels touch correction arcs; overlays are not recomputed after an edit; touch, Firefox/WebKit, high-DPI and the Plotly pages were not checked in a real browser | Hover arrows to read the true dependency, move nodes, zoom in; re-run the analysis in Python after edits |

### Optional dependencies

| Limitation | Affected feature | Consequence | Workaround |
|---|---|---|---|
| Piquasso is optional and pinned | `piquasso`, `piquasso-fock`, `piquasso-mixed-fock`; raw-Piquasso tests; `photographiq.autodiff` input preparation; `PhysicalMuTA` default backend | Requesting these without Piquasso raises `ImportError` with the install command; reference tests skip | `pip install 'photographiq[piquasso]'` (`piquasso==8.0.1`); native Fock backends do not import Piquasso |
| Native extension may be absent | `native-gaussian`, `native-fock`, `native-mixed-fock`, `backend="native"` in `photographiq.sampling` | Native backends raise `ImportError` with build instructions; `backend="auto"` silently falls back to NumPy | Build from source with a C++20 compiler and CMake >= 3.24 ([native build](development/native-build.md)) |
| Other optional extras | Graphix (`validation`), JAX plus Piquasso (`autodiff`; Piquasso was added on 2026-11-11 because it prepares every input), mpmath (`precision`), CuPy (`gpu-cuda12`: `cupy-cuda12x`; `gpu-cuda13`: `cupy-cuda13x[ctk]`, which also pulls the CUDA 13 toolkit), matplotlib (`visualization`), MentPy (`ml-validation`), scikit-learn (`ml-experiments`) | Lazy imports raise `ImportError` naming the extra; the dependent tests skip | Install the named extra; `dev` installs Graphix, Piquasso and mpmath but not JAX, CuPy or MentPy |
| CUDA-Q is not an extra | `photographiq.cudaq_dynamics` | No install path is declared; CUDA-Q publishes Linux wheels only | WSL2 or Linux (Hamiltonian evolution only, unverified) |
| Numba < 0.68 constraint | Extras that install Piquasso (`piquasso`, `ml-experiments`, `autodiff`, `dev`) | With `numba` 0.68.1 / `llvmlite` 0.50.0, Piquasso's cached hafnian kernels compile in the first process and every later process aborts while loading the cache (`LLVM ERROR: Symbol not found: .numba.unresolved...`). The extras therefore pin `numba<0.68` (0.67.0 / 0.49.0 pass); no upstream issue was found | Keep the constraint until a fixed Numba release is verified (reproduction in the `pyproject.toml` comment); the native backends do not use Numba |

### Platform notes

| Limitation | Affected feature | Consequence | Workaround |
|---|---|---|---|
| CUDA-Q is Linux-only | `photographiq.cudaq_dynamics` | Not installable on native Windows; the probe reports `ModuleNotFoundError` here | WSL2 or Linux |
| GPU module needs an NVIDIA GPU and a matching CuPy wheel | `photographiq.gpu` | `device="gpu"` raises `ImportError` (CuPy missing; the message names `cupy-cuda12x`, `'cupy-cuda13x[ctk]'` and the matching extras) or `RuntimeError` (no device) and never falls back to the CPU; GeForce-class GPUs run FP64 at a small fraction of their FP32 rate | `device="cpu"`; pick `gpu-cuda12` or `gpu-cuda13` to match the driver |
| No custom CUDA kernels | `photographiq.gpu` | Kernels are compositions of cuBLAS/cuSOLVER batched calls and CuPy element-wise operations | None needed for correctness |
| OpenMP is optional | Native kernels | Without compiler OpenMP support the kernels run serially; native Fock gates use OpenMP only above a 2^22 work threshold | Check `photographiq.native.build_info()["openmp"]` |
| Build-time downloads | Native extension | CMake fetches Eigen 3.4.0 unless an offline source is supplied | `-Ccmake.define.FETCHCONTENT_SOURCE_DIR_EIGEN=/path/to/eigen-3.4.0` |
| Python version range | Package | `requires-python = ">=3.11,<3.15"` | Use Python 3.11 to 3.14 |
| Windows-only memory metric | `benchmarks/fock/bench_fock.py` | psutil `peak_wset` is reported on Windows only (`None` elsewhere). psutil is optional and no extra declares it; without it all memory fields, `physical_cpus` and the CPU-load field are `None` | Install psutil; use the baseline RSS on other platforms |

## 36. Future development

The items below come only from the project's roadmaps (`ROADMAP.md`, identical to `docs/roadmap.md`; [PhotoGraphiQML roadmap](ml/ROADMAP.md)), from code that raises `NotImplementedError` with a message or docstring that states the missing work, from the **Planned** entries in Sections 27 and 28, and from the 2026-11-12 audit reports and comments that name the remaining work ([Phase 1](validation/PHASE1_RELEASE_AUDIT.md), [Phase 3](validation/PHASE3_FEATURE_COMPLETION.md), the validation pages they link, and `pyproject.toml`). No `TODO` or `FIXME` comment exists in `src/` or `cpp/`. The roadmaps state: "These are research directions without promised dates" and "None of these items imply quantum advantage or an error-correction threshold." No item below has a date or a commitment. Defects fixed on 2026-11-11 (sampler clipping, the threshold rounding bound and its warnings, pattern validation messages, CV determinism false negatives at r of about 3.1 to 3.5, `analyze(qubit_graph, "cv", ...)`, the Fock benchmark correctness gate, the `autodiff` extra and the GPU install hint) and on 2026-11-12 (the ignored `Homodyne` angle in direct `measure` calls, the one-mode native Gaussian-unitary instability, the CI lint and Numba failures, the four interactive-page defects, uncertain CV rank decisions reported as `not_certified`) are not listed, and neither are the 2026-11-12 completions (`precision="auto"`, `photographiq.gpu.fock`, `GaussianGraph`, the CV-flow `shift=`/`exact=` options, native backends for the convergence helpers). The open problems those fixes left are in Section 35, because no roadmap item covers them.

| Item | Source | Current behaviour | Status |
|---|---|---|---|
| Compilation with better resource efficiency, approximation control and temporal representations | `ROADMAP.md`, `docs/roadmap.md` ("Later compilation work can improve ...") | Gaussian compilation favours transparency over noise/resource efficiency; synthesis is approximate (Section 35) | **Planned** |
| Broader GKP syndrome/decoder studies and independent finite-energy correction evidence, before any fault-tolerance claim | `ROADMAP.md`; Section 27 "Fault-tolerant GKP architecture and thresholds" | No fault-tolerant architecture, threshold or optimal decoder | **Planned** |
| Faithful logical MuTA validation: tied measurement groups, bounds, mixed quantum inputs, noisier resource branches, the learned teleportation instrument, and independent training of the pinned MentPy reference | PhotoGraphiQML roadmap item 1 | Logical MuTA exists; MentPy comparison covers checkpoints only | **Planned** |
| Validated arbitrary logical XY measurement and injection instruments for the physical bridge | PhotoGraphiQML roadmap item 2; `src/photographiq/encoded.py` ("Y/XY need validation"); `src/photographiqml/ansatz/muta.py` ("requires a validated logical XY measurement/injection instrument and decoder") | Signed-X only; other angles raise `UnsupportedPhysicalMeasurement` | **Planned** |
| Correlated input preparation and joint code-subspace leakage; expanded resource and whole-pattern convergence studies | PhotoGraphiQML roadmap item 2; Section 28 "Other physical-path boundaries" | Entangled inputs raise `NotImplementedError` | **Planned** |
| CV-native MuTA (CVMuTA): ideal cell map, finite-resource channel, identity embeddings, validation against PhotoGraphiQ physics | PhotoGraphiQML roadmap item 3; Section 28 "CV-native MuTA" | Not implemented | **Planned** |
| Training with valid autodiff, stochastic estimators, mini-batches, noise-aware readouts, callbacks and optimizer/model checkpointing | PhotoGraphiQML roadmap item 4 | Fixed-branch JAX gradients and experimental estimators; full stochastic MBQC differentiation unsupported | **Planned** |
| Full paper reproduction, ten larger demo projects, the full release acceptance checklist, and the Python CI matrix on hosted runners | PhotoGraphiQML roadmap item 5 | Partial reproductions in `experiments/ml/` and `paper_experiments_ml/`; CI configured, no passing run claimed | **Planned** |
| Reservoir computing and recurrent models | PhotoGraphiQML roadmap ("future directions") | Not implemented | **Planned** |
| Magic-state injection or arbitrary logical Rz protocol behind the `LogicalMeasurementSynthesis` extension point | `src/photographiq/encoded.py` (`lower` raises "No validated magic-state injection or arbitrary logical Rz protocol") | Always raises | **Planned** |
| Physical quantum-output instrument model with a separately validated post-measurement map | `src/photographiqml/models.py` (`QuantumInstrumentModel` raises "a physical quantum-output instrument needs a separately validated post-measurement map") | Logical only | **Planned** |
| A sound CV analogue of local complementation, verified by covariance/nullifier comparison | `src/photographiq/calculus/clifford.py` (`cv_local_complementation` docstring: "a sound CV analogue must be derived and verified ..., which this package has not done") | Always raises `NotImplementedError` | **Planned** |
| CUDA-Q execution of patterns (measurement conditioning on CUDA-Q) and a verified CUDA-Q run of the adapter | `src/photographiq/cudaq_dynamics.py` (the gate-to-Hamiltonian translation exists and is validated against native-fock through the NumPy reference) | Adapter only; CUDA-Q path not run | **Planned** |
| Lift the `numba<0.68` constraint once a Numba release that reloads cached Piquasso kernels is verified | `pyproject.toml` comment ("Remove once a fixed numba is verified"); [Phase 1 audit](validation/PHASE1_RELEASE_AUDIT.md), unresolved blockers | The extras that install Piquasso pin `numba<0.68` | **Planned** |
| Confirm the 2026-11-12 CI fixes on the hosted Linux runners | [Phase 1 audit](validation/PHASE1_RELEASE_AUDIT.md) ("A push, made by the maintainer, is required to confirm green CI on ubuntu") | Fixes reproduced on Windows only; remote CI not re-run | **Planned** |
| A general error bound (or a stable algorithm) for displaced multi-mode Gaussian unitaries | [native Fock Gaussian unitary](validation/native-fock-gaussian-unitary.md), remaining limits; [Phase 3](validation/PHASE3_FEATURE_COMPLETION.md), "Still unsupported" | Bargmann recurrence without a general bound; never requested by the backends | **Planned** |
| Exact certification of the finite-squeezing CV channel | [CV-flow robustness](validation/cv-flow-robustness.md), remaining limits ("No exact finite-squeezing channel check exists"); [Phase 3](validation/PHASE3_FEATURE_COMPLETION.md), "Still unsupported" | `finite_squeezing` certificates are float64 statements; the exact check covers the ideal limit only | **Planned** |
| A native batched kernel for permanents, hafnians and torontonians, before any GPU sampling kernel is reconsidered | [Phase 3](validation/PHASE3_FEATURE_COMPLETION.md), section 3.3 ("a native batched kernel should come first"); [GPU acceleration](gpu.md) | GPU sampling measured and not shipped; native kernels evaluate one matrix per call | **Planned** |
| Layout-aware edge routing for the interactive pages | [interactive visualization browser check](validation/interactive-visualization-browser.md), remaining limits ("Fixing this needs layout-aware edge routing, which was not attempted") | Straight edges and fixed-bend arcs can pass under other nodes | **Planned** |
