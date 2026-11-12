"""CuPy device discovery and the execution record attached to every GPU-module result.

CuPy is imported lazily: importing :mod:`photographiq.gpu` never imports it; the first call
that needs the device (``available``, ``device_info`` or a kernel with ``device="gpu"``)
does. Nothing in this module falls back to the CPU: ``require`` raises when CuPy or a CUDA
device is missing, and only an explicit ``device="cpu"`` runs the NumPy implementation.
"""

from __future__ import annotations

import functools
import platform
from dataclasses import dataclass
from types import ModuleType

from .._optional import LazyModule

INSTALL_HINT = (
    "install the CuPy wheel matching your NVIDIA driver: pip install cupy-cuda12x (CUDA 12) "
    "or pip install 'cupy-cuda13x[ctk]' (CUDA 13); the extras gpu-cuda12 / gpu-cuda13 "
    "install the same wheels"
)


class _CuPy(LazyModule):
    def load(self) -> ModuleType:
        try:
            return super().load()
        except ImportError as exc:
            raise ImportError(f"photographiq.gpu requires CuPy; {INSTALL_HINT}") from exc


cupy = _CuPy("cupy", "gpu")


def available() -> bool:
    """True if CuPy imports and at least one CUDA device is visible (imports CuPy)."""
    if not cupy.available:
        return False
    try:
        return cupy.load().cuda.runtime.getDeviceCount() > 0
    except (ImportError, RuntimeError):  # missing CUDA DLLs, cudaErrorNoDevice, driver too old
        return False


def require() -> ModuleType:
    """Return the CuPy module, or raise ImportError / RuntimeError (never falls back)."""
    cp = cupy.load()
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except RuntimeError as exc:
        raise RuntimeError(f"CuPy is installed but no usable CUDA device was found: {exc}") from exc
    if count < 1:
        raise RuntimeError("CuPy is installed but no CUDA device is visible")
    return cp


def _version(code: int) -> str:
    return f"{code // 1000}.{(code % 1000) // 10}"


@functools.lru_cache(maxsize=None)
def _device_name(device_id: int) -> str:
    props = cupy.load().cuda.runtime.getDeviceProperties(device_id)
    return props["name"].decode()


def device_info() -> dict:
    """Backend, CuPy/CUDA versions, name, compute capability and memory of the current device."""
    cp = require()
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    free, total = device.mem_info
    return {
        "backend": "cupy",
        "cupy_version": cp.__version__,
        "cuda_runtime_version": _version(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver_version": _version(cp.cuda.runtime.driverGetVersion()),
        "device_count": cp.cuda.runtime.getDeviceCount(),
        "device_id": device.id,
        "device_name": props["name"].decode(),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "multiprocessors": props["multiProcessorCount"],
        "total_memory_bytes": int(total),
        "free_memory_bytes": int(free),
    }


@dataclass(frozen=True)
class Execution:
    """Where a kernel actually ran.

    ``device`` is ``"cuda:<id>"`` read from the output buffer's own device, or ``"cpu"`` for
    the NumPy path. ``returned_on_device`` says whether the result arrays are CuPy arrays.
    """

    kernel: str
    device: str
    backend: str
    device_name: str
    dtype: str
    returned_on_device: bool


def gpu_execution(kernel: str, array, dtype: str, on_device: bool) -> Execution:
    device_id = array.device.id
    return Execution(kernel, f"cuda:{device_id}", "cupy", _device_name(device_id), dtype, on_device)


def cpu_execution(kernel: str, dtype: str) -> Execution:
    return Execution(kernel, "cpu", "numpy", platform.processor() or "cpu", dtype, False)
