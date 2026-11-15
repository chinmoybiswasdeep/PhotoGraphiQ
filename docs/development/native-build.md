# Building the native engines

PhotoGraphiQ's compiled engines live in `cpp/` and are built into the extension
module `photographiq._native` by [scikit-build-core](https://scikit-build-core.readthedocs.io)
and CMake whenever the package is installed from source. Everything else in the
package is pure Python: if the extension is missing, the NumPy `gaussian` backend,
graph tools, flows, calculus and visualization still work, and native backends raise
an `ImportError` explaining how to build them.

## Requirements

| Tool | Version | Notes |
| --- | --- | --- |
| C++ compiler | C++20 | MSVC 19.3x or later, GCC 11+, Clang 14+ |
| CMake | 3.24+ | `pip install cmake` works |
| pybind11 | 2.13+ | build requirement, installed automatically by pip |
| scikit-build-core | 0.10+ | build backend, installed automatically by pip |
| Eigen | 3.4.0 | found on the system, otherwise downloaded at a pinned SHA-256 |
| OpenMP | optional | used when the compiler provides it (`PGQ_USE_OPENMP=ON`) |

No CUDA toolkit is needed. GPU acceleration is optional and provided separately
(see [GPU acceleration](../gpu.md)).

## Install

```sh
python -m pip install -e ".[dev]"
python -c "import photographiq.native as n; print(n.components(), n.build_info())"
```

`components()` lists the compiled components (`gaussian`, `fock`, `sampling`) and
`build_info()` reports the compiler, build type, C++ standard, OpenMP and Eigen versions.

On Windows, CMake locates Visual Studio Build Tools automatically; a "Developer
Command Prompt" is not required.

### Offline builds

Download `eigen-3.4.0.tar.gz` (SHA-256
`8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72`), extract it, and
point CMake at it:

```sh
python -m pip install -e . -Ccmake.define.FETCHCONTENT_SOURCE_DIR_EIGEN=/path/to/eigen-3.4.0
```

### Choosing components

```sh
python -m pip install -e . "-Ccmake.define.PGQ_COMPONENTS=gaussian;sampling"
```

## Development builds and C++ tests

Configure an out-of-tree build with C++ unit tests:

```sh
cmake -S cpp -B build/dev -DPGQ_BUILD_CPP_TESTS=ON \
      -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/dev --config Release
ctest -C Release --test-dir build/dev
```

To use that build without reinstalling, point the loader at the compiled file:

```sh
PHOTOGRAPHIQ_NATIVE_PATH=build/dev/Release/_native.cp312-win_amd64.pyd python -m pytest tests/native
```

## Layout

| Path | Contents |
| --- | --- |
| `cpp/include/photographiq/core` | shared types and conventions (hbar = 2, interleaved quadratures, vacuum V = I) |
| `cpp/include/photographiq/gaussian`, `cpp/src/gaussian` | covariance-matrix engine |
| `cpp/include/photographiq/fock`, `cpp/src/fock` | truncated Fock-space pure and mixed kernels |
| `cpp/include/photographiq/sampling`, `cpp/src/sampling` | permanent, hafnian, loop hafnian, torontonian |
| `cpp/src/bindings` | pybind11 bindings, one file per component |
| `cpp/tests` | C++ unit tests (`ctest`) |

Eigen is used only through its MPL-2.0 licensed code (`EIGEN_MPL2_ONLY`).

## Troubleshooting

- *"photographiq._native is not available"*: the package was installed without a
  working compiler. Install a C++20 compiler and reinstall.
- *"Native component 'fock' was not compiled"*: rebuild with that component in
  `PGQ_COMPONENTS`.
- Builds are cached in `build/{wheel_tag}`; delete that directory to force a clean
  rebuild.
