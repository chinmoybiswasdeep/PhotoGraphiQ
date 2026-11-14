# Roadmap

- v0.2 established the validated Gaussian and experimental pure-Fock core.
- v0.3 adds documentation, visualization, provenance and experimental mixed-state,
  nonlinear-synthesis, GKP and differentiable execution infrastructure.
- Later compilation work can improve resource efficiency, approximation control
  and temporal representations.
- GKP development needs broader syndrome/decoder studies and independent
  finite-energy correction evidence before fault-tolerance claims.
- 0.3.1 adds native C++20 Gaussian, Fock and sampling engines, the backend registry,
  qubit flows and calculus with CV flow, interactive visualization, optional CuPy GPU
  kernels and a CUDA-Q Hamiltonian adapter (execution unverified).
- Open items: remote CI re-run of the upgrade, Linux/macOS native-build evidence, CUDA-Q
  execution on Linux, an error bound for multi-mode displaced native Fock unitaries.
- PhotoGraphiQML now lives in this repository as the `photographiqml` package; the
  core still introduces no ML framework dependency.

These are research directions without promised dates. Feature status is in the
[matrix](validation/feature-matrix.md), not inferred from the roadmap.
