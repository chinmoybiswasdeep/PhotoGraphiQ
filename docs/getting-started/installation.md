# Installation

PhotoGraphiQ 0.3.1 supports Python 3.11–3.14 and is installed from a source checkout. No
PyPI release is claimed.

## Platform support

| Platform | Support level | Evidence |
|---|---|---|
| Linux x86_64 | Supported: remote CI | GitHub Actions `ubuntu-latest`, Python 3.11–3.14 (`.github/workflows/ci.yml`) |
| Windows 11 x86_64 | Supported: verified locally | MSVC 19.50 native build; full suite on Python 3.11–3.14 (2026-11-12, [validation status](../validation/MASTER_VALIDATION_STATUS.md)) |
| macOS | **Untested** | The code is portable C++20 and Python, but no build or test has been run |

The upgrade work on the `upgraded` branch was validated locally on Windows and has not been
re-run on remote CI yet.

## What gets installed

| Layer | Contents | Needed for |
|---|---|---|
| Runtime dependencies | NumPy, SciPy, NetworkX | everything |
| `photographiq._native` (compiled during install) | C++20 engines: `gaussian`, `fock`, `sampling` (Eigen, optional OpenMP) | `native-gaussian`, `native-fock`, `native-mixed-fock` backends and the native BS/GBS kernels |
| Optional extras | see the [table below](#optional-extras) | reference backends, GPU, plots, validation, autodiff, docs |

The **native C++ backend** is part of the normal source install and runs on the CPU. It is
different from the **optional GPU kernels** (`photographiq.gpu`), which use CuPy on an NVIDIA
GPU, ship no compiled CUDA code and do not need `nvcc`. If the native build fails or is
skipped, the NumPy `gaussian` backend, graphs, compilation, flows, calculus and visualization
still work; the native backends then raise an `ImportError` that explains how to build them.

## 1. Create an environment

```sh
git clone https://github.com/chinmoybiswasdeep/PhotoGraphiQ.git
cd PhotoGraphiQ
python -m venv .venv
```

Activate it with `source .venv/bin/activate` (Linux, macOS) or
`.venv\Scripts\Activate.ps1` (Windows PowerShell).

## 2. Install a C++20 toolchain (for the native engines)

| Tool | Version | How to get it |
|---|---|---|
| C++ compiler | C++20 (GCC 11+, Clang 14+, MSVC 19.3x+) | Linux: `sudo apt install build-essential` (or your distribution's GCC); Windows: Visual Studio Build Tools with "Desktop development with C++"; macOS: Xcode command-line tools (untested) |
| CMake | ≥ 3.24 | `python -m pip install cmake` works on every platform |
| scikit-build-core, pybind11 | ≥ 0.10, ≥ 2.13 | installed automatically by pip (build requirements) |
| Eigen | 3.4.0 | used from the system if found, otherwise downloaded at a pinned SHA-256 |
| OpenMP | optional | used when the compiler provides it |

No CUDA toolkit is needed. On Windows, CMake finds Visual Studio Build Tools automatically.
**Offline builds:** download `eigen-3.4.0.tar.gz`, extract it, and pass
`-Ccmake.define.FETCHCONTENT_SOURCE_DIR_EIGEN=/path/to/eigen-3.4.0` to `pip install`. Component
selection, C++ unit tests and troubleshooting are in
[building the native engines](../development/native-build.md).

## 3. Install

| Goal | Command (from the repository root) |
|---|---|
| CPU core with native engines | `python -m pip install .` |
| Development (editable, tests, lint, Piquasso, Graphix, mpmath) | `python -m pip install -e ".[dev]"` |
| Documentation and notebooks | `python -m pip install -e ".[docs]"` |
| Everything used by CI | `python -m pip install -e ".[dev,autodiff]"` |

An editable install (`-e`) follows source changes in `src/`; C++ changes in `cpp/` need a
reinstall to rebuild the extension. Quote the extras in every shell.

### Optional extras

| Extra | Adds | Used by |
|---|---|---|
| `piquasso` | Piquasso 8.0.1, numba < 0.68 | `piquasso`, `piquasso-fock`, `piquasso-mixed-fock` reference backends; the default backend of `cutoff_convergence` and of physical PhotoGraphiQML |
| `precision` | mpmath | multiprecision threshold (torontonian) probabilities |
| `gpu-cuda12` / `gpu-cuda13` | `cupy-cuda12x` / `cupy-cuda13x[ctk]` | batched Gaussian and Fock kernels in `photographiq.gpu` |
| `visualization` | Matplotlib | static drawings and plots |
| `validation` | Graphix 0.4 | MBQC structure cross-checks |
| `autodiff` | JAX, Piquasso | differentiable fixed-outcome Fock execution (Piquasso prepares the inputs) |
| `ml-validation` | MentPy 0.1.0a15 | independent logical MuTA reference tests |
| `ml-experiments` | Matplotlib, scikit-learn, Piquasso | the PhotoGraphiQML manuscript suite |
| `docs` | MkDocs Material, mkdocstrings, nbclient | building this website |
| `dev` | pytest, ruff, mypy, build, Matplotlib, Graphix, nbformat, Piquasso, mpmath | the full test suite |

`numba < 0.68` is pinned because numba 0.68 crashes when reloading Piquasso's cached kernels
(see `pyproject.toml`).

### CPU-only installs

Every install is CPU-only unless you add a GPU extra: the native engines run on the CPU and
use OpenMP threads when the compiler provides OpenMP (the standard `OMP_NUM_THREADS`
variable limits them). A source install needs the C++ toolchain above; the pure-Python
fallback described in [What gets installed](#what-gets-installed) applies when the extension
cannot be loaded, not as a way to skip the build.

### GPU extras (NVIDIA, optional)

```sh
python -m pip install -e ".[gpu-cuda12]"   # driver with CUDA 12
python -m pip install -e ".[gpu-cuda13]"   # driver with CUDA 13 (CUDA runtime from PyPI)
```

GPU kernels never fall back to the CPU silently: `device="gpu"` raises if CuPy or a device is
missing. They accelerate batched workloads only; see [GPU acceleration](../gpu.md).

### CUDA-Q

`photographiq.cudaq_dynamics` is a Hamiltonian-level adapter, not a backend. CUDA-Q publishes
Linux wheels only, so it cannot be used on native Windows or macOS; its execution path has
not been verified on any platform. Its NumPy/SciPy reference path needs no CUDA-Q
([CUDA-Q adapter](../cudaq.md)).

## 4. Check the installation with real backend operations

Importing is not enough: this check runs one finite-squeezing teleportation step on the
NumPy, native Gaussian and native Fock engines and compares them.

```python
import numpy as np
import photographiq as pg

print("photographiq", pg.__version__, "| native:", pg.native.components())
print("backends:", pg.available_backends())

# One finite-squeezing wire step (r = 0.3) with a coherent input and feed-forward.
graph = pg.CVGraph.line(2, squeezing=0.3, inputs=(0,))
pattern = pg.Pattern(graph).measure(0, pg.Homodyne.p())
pattern.displace(1, q=-pg.Outcome(0))
coh = {0: pg.GaussianInput.coherent(0.3 + 0.2j)}

# 1. NumPy reference vs native C++ Gaussian engine: same seed, same trajectory.
ref = pg.simulate(pattern, inputs=coh, seed=7)                    # backend="gaussian"
nat = pg.simulate(pattern, inputs=coh, seed=7, backend="native-gaussian")
assert ref.outcomes == nat.outcomes
assert np.allclose(nat.state.covariance, ref.state.covariance, atol=1e-12)
print("native-gaussian == gaussian")

# 2. The same conditional branch on the native pure-Fock engine, converging with cutoff.
mean_ref, var_ref = ref.state.quadrature(1)
for cutoff in (20, 30, 40):
    fock = pg.simulate(pattern, inputs=coh, backend="native-fock", cutoff=cutoff,
                       measurement_outcomes=ref.outcomes)
    mean, var = fock.state.quadrature(1)
    print(f"cutoff {cutoff}: |d<q>| = {abs(mean - mean_ref):.1e}, "
          f"|dVar| = {abs(var - var_ref):.1e}, min retained norm = {min(fock.state.retained_norms):.10f}")
assert abs(mean - mean_ref) < 1e-6 and abs(var - var_ref) < 1e-6

# 3. Heralded photon subtraction (non-Gaussian) on the native pure-Fock engine.
sub = pg.simulate(pg.non_gaussian.photon_subtraction(theta=0.2),
                  inputs={"in": pg.FockInput.number(2)}, backend="native-fock", cutoff=8,
                  measurement_outcomes={"count": 1})
assert abs(sub.state.photon_number("in") - 1.0) < 1e-12
print("installation OK")
```

Output on the Windows reference machine (the Fock rows also emit `RuntimeWarning`s about the
retained norm at cutoffs 20 and 30; that is the truncation diagnostic working as intended):

```text
photographiq 0.3.1 | native: ('gaussian', 'fock', 'sampling')
backends: ('gaussian', 'piquasso', 'piquasso-fock', 'piquasso-mixed-fock', 'native-gaussian', 'native-fock', 'native-mixed-fock')
native-gaussian == gaussian
cutoff 20: |d<q>| = 1.2e-05, |dVar| = 1.6e-04, min retained norm = 0.9999782317
cutoff 30: |d<q>| = 2.4e-08, |dVar| = 2.7e-06, min retained norm = 0.9999998489
cutoff 40: |d<q>| = 4.5e-10, |dVar| = 4.4e-08, min retained norm = 0.9999999989
installation OK
```

The Fock result converges to the exact Gaussian one as the total-photon cutoff grows. The
Gaussian answer has no truncation; the Fock answer does, and the retained norm reports it.
The `piquasso*` entries appear only with the `piquasso` extra.

**With the `piquasso` extra**, the reference backend consumes random numbers identically:

```python
import numpy as np
import photographiq as pg

graph = pg.CVGraph.line(2, squeezing=0.3, inputs=(0,))
pattern = pg.Pattern(graph).measure(0, pg.Homodyne.p())
pattern.displace(1, q=-pg.Outcome(0))
coh = {0: pg.GaussianInput.coherent(0.3 + 0.2j)}
a = pg.simulate(pattern, inputs=coh, seed=7, backend="piquasso")
b = pg.simulate(pattern, inputs=coh, seed=7)
assert np.isclose(a.outcomes[0], b.outcomes[0], rtol=1e-12)          # equal to rounding
assert np.allclose(a.state.covariance, b.state.covariance, atol=1e-12)
print("piquasso backend agrees with gaussian")
```

**With a GPU extra**, run a kernel on the device and compare it with the CPU path:

```python
import numpy as np
from photographiq import gpu

print(gpu.device_info())
rng = np.random.default_rng(0)
B, n = 32, 3
means = rng.normal(size=(B, 2 * n))
covs = np.broadcast_to(np.eye(2 * n), (B, 2 * n, 2 * n)).copy()
x = rng.normal(size=B)
on_gpu = gpu.condition_homodyne(means, covs, mode=1, angles=0.4, outcomes=x, efficiency=0.9)
on_cpu = gpu.condition_homodyne(means, covs, mode=1, angles=0.4, outcomes=x, efficiency=0.9,
                                device="cpu")
assert on_gpu.execution.device.startswith("cuda")             # raises earlier if no GPU
assert np.allclose(on_gpu.covariances, on_cpu.covariances, rtol=1e-12, atol=1e-14)
print("GPU OK:", on_gpu.execution)
```

`photographiq.native.build_info()` reports the compiler, C++ standard, OpenMP and Eigen
versions of the compiled extension.

## 5. Development checks

```sh
python -m pytest                                          # full test suite
python -m ruff check . && python -m ruff format --check .
python -m mypy
python scripts/check_public_api.py                        # public exports import
python scripts/check_docs_examples.py --projects          # tutorials and demo projects run
python -m mkdocs build --strict                           # docs (needs the docs extra)
python experiments/validate_release.py --output .validation   # all CI gates in one run
```

GPU tests skip with a reason when CuPy or a device is missing; Piquasso-reference tests
need the `piquasso` extra (included in `dev`).

## 6. Reproduce the manuscript evidence

```sh
python scripts/reproduce_manuscript.py --list          # show the steps, run nothing
python scripts/reproduce_manuscript.py                 # fast validation (minutes)
python scripts/reproduce_manuscript.py --mode full     # benchmarks + safe experiment subsets
python paper_experiments/run_all_safe.py               # PhotoGraphiQ core suite, fast subset
python paper_experiments_ml/run_all_safe.py            # PhotoGraphiQML suite, fast subset
```

`reproduce_manuscript.py` writes to `build/reproduction/<timestamp>/` and runs the experiment
suites from copies, so recorded results stay unchanged; a step whose optional dependency is
missing is recorded as skipped, never as passed. See
[`paper_experiments/README.md`](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/blob/main/paper_experiments/README.md)
and the
[experiment index](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/blob/main/paper_experiments/EXPERIMENT_INDEX.md).

## Troubleshooting

- **`photographiq._native is not available`**: no working C++20 compiler or CMake during
  install. Install them and reinstall.
- **`ImportError` naming an extra** (for example `photographiq[piquasso]`): the requested
  backend or feature needs that optional dependency.
- **Wrong interpreter**: check `python -m pip show photographiq` with the same interpreter that
  runs your script.
- **Stale native build**: delete `build/` and reinstall.
