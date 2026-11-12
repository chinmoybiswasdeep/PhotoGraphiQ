"""Device discovery and the no-silent-fallback guarantee of photographiq.gpu."""

import subprocess
import sys
import types

import numpy as np
import pytest

from photographiq import gpu
from photographiq.gpu import device as gpu_device

needs_gpu = pytest.mark.skipif(not gpu.available(), reason="CuPy/CUDA device missing")


def small_batch():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4, 4))
    return rng.normal(size=(4, 4)), a @ np.swapaxes(a, 1, 2) + np.eye(4)


def kernels(means, covs, **kw):
    s = np.array([[1.0, 0.0], [0.4, 1.0]])
    return {
        "apply_symplectic": lambda: gpu.apply_symplectic(means, covs, [1], s, **kw),
        "condition_homodyne": lambda: gpu.condition_homodyne(means, covs, 0, 0.2, 0.1, **kw),
        "homodyne_moments": lambda: gpu.homodyne_moments(means, covs, 0, 0.2, **kw),
        "homodyne_log_density": lambda: gpu.homodyne_log_density(means, covs, 1, 0.2, 0.1, **kw),
        "purity": lambda: gpu.purity(covs, **kw),
        "symplectic_eigenvalues": lambda: gpu.symplectic_eigenvalues(covs, **kw),
        "fidelity": lambda: gpu.fidelity(means, covs, means, covs, **kw),
    }


def test_import_does_not_import_cupy():
    code = "import sys, photographiq, photographiq.gpu; print('cupy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_missing_cupy_raises_with_install_hint(monkeypatch):
    monkeypatch.setattr(gpu_device, "cupy", gpu_device._CuPy("photographiq_no_such_cupy", "gpu"))
    assert not gpu.available()
    with pytest.raises(ImportError, match="cupy-cuda12x.*cupy-cuda13x"):
        gpu.device_info()
    means, covs = small_batch()
    for name, call in kernels(means, covs, device="gpu").items():
        with pytest.raises(ImportError, match="requires CuPy"):
            call()
    # only an explicit device="cpu" runs, and it says so
    for name, call in kernels(means, covs, device="cpu").items():
        execution = call().execution
        assert (execution.kernel, execution.device, execution.backend) == (name, "cpu", "numpy")


def test_no_device_raises_instead_of_falling_back(monkeypatch):
    def no_device():
        raise RuntimeError("cudaErrorNoDevice: no CUDA-capable device is detected")

    fake = types.SimpleNamespace(
        available=True,
        load=lambda: types.SimpleNamespace(
            cuda=types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=no_device))
        ),
    )
    monkeypatch.setattr(gpu_device, "cupy", fake)
    assert not gpu.available()
    means, covs = small_batch()
    with pytest.raises(RuntimeError, match="no usable CUDA device"):
        gpu.condition_homodyne(means, covs, 0, 0.0, 0.0)


def test_device_argument_is_explicit():
    means, covs = small_batch()
    with pytest.raises(ValueError, match="'gpu' or 'cpu'"):
        gpu.purity(covs, device="auto")
    with pytest.raises(ValueError, match="requires device='gpu'"):
        gpu.purity(covs, device="cpu", on_device=True)


@needs_gpu
def test_device_info_describes_the_current_device():
    import cupy as cp

    info = gpu.device_info()
    assert info["backend"] == "cupy" and info["cupy_version"] == cp.__version__
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    assert info["device_name"] == props["name"].decode()
    assert info["compute_capability"] == f"{props['major']}.{props['minor']}"
    assert 0 < info["free_memory_bytes"] <= info["total_memory_bytes"]
    code = cp.cuda.runtime.runtimeGetVersion()
    assert info["cuda_runtime_version"] == f"{code // 1000}.{(code % 1000) // 10}"


@needs_gpu
def test_gpu_results_report_and_live_on_the_gpu():
    import cupy as cp

    current = cp.cuda.Device().id
    name = gpu.device_info()["device_name"]
    means, covs = small_batch()
    for kw in ({}, {"on_device": True}):
        for kernel, call in kernels(means, covs, **kw).items():
            result = call()
            execution = result.execution
            assert execution.kernel == kernel and execution.backend == "cupy"
            assert execution.device == f"cuda:{current}" and execution.device_name == name
            assert execution.returned_on_device == bool(kw)
            arrays = [v for k, v in vars(result).items() if k != "execution"]
            for array in arrays:
                if not kw:
                    assert isinstance(array, np.ndarray)
                    continue
                assert isinstance(array, cp.ndarray)
                attributes = cp.cuda.runtime.pointerGetAttributes(array.data.ptr)
                assert attributes.type == cp.cuda.runtime.memoryTypeDevice
                assert attributes.device == current


@needs_gpu
def test_device_resident_inputs_stay_on_device():
    import cupy as cp

    means, covs = small_batch()
    d_means, d_covs = cp.asarray(means), cp.asarray(covs)
    on = gpu.condition_homodyne(d_means, d_covs, 1, 0.3, 0.2, 0.9, on_device=True)
    off = gpu.condition_homodyne(means, covs, 1, 0.3, 0.2, 0.9)
    np.testing.assert_array_equal(on.covariances.get(), off.covariances)
    # the CPU path refuses device arrays instead of copying them silently
    with pytest.raises(TypeError):
        gpu.purity(d_covs, device="cpu")
