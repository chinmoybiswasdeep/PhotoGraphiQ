# Installation

The NumPy reference engine needs only the base package (NumPy and SciPy):

```bash
pip install -e .
python -c "import photographiq.hybrid.executor as h; print(h.ENGINES)"
# ('numpy-reference', 'cudaq')
```

## CUDA-Q engine (Linux only)

`engine="cudaq"` needs NVIDIA CUDA-Q ≥ 0.16 (`cudaq_dynamics.MIN_CUDAQ_VERSION`). CUDA-Q
publishes wheels only for Linux x86_64 and aarch64, so on Windows it needs WSL2:

```bash
pip install "cudaq>=0.16"
python -c "import photographiq.cudaq_dynamics as cq; print(cq.cudaq_probe())"
python -m pytest tests/cudaq_adapter tests/hybrid -v      # the real-CUDA-Q tests must run, not skip
```

Without CUDA-Q, `run(..., engine="cudaq")` raises `ImportError`. The executor never falls
back to NumPy when CUDA-Q was requested.

Status on the development machine: **blocked**. It runs Windows 11 with an RTX 4050
Laptop GPU and has no WSL. See
[CUDA-Q real runtime validation](../validation/CUDAQ_REAL_RUNTIME_VALIDATION.md).

The GPU target is whatever `cudaq.get_target()` reports, and every CUDA-Q evolution
stores it in its execution record. A GPU result should be described as such only after
that record confirms a GPU simulator.
