"""HYB-01 (real CUDA-Q runtime) and HYB-11 (GPU performance): environment evidence.

HYB-01 passes only if ``cudaq`` imports, ``cudaq_dynamics.evolve`` runs and agrees with the
NumPy reference (the executable checks live in tests/cudaq_adapter and tests/hybrid and are
skip-guarded). HYB-11 requires HYB-01 plus a confirmed GPU target. Otherwise both are recorded
as BLOCKED with the probe output; no runtime or speed-up number is produced.
"""

import importlib.util
import platform
import shutil
import subprocess

from hyb_support import save

import photographiq.cudaq_dynamics as cq


def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).replace("\x00", "").strip()  # wsl.exe writes UTF-16
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"


def main():
    probe = cq.cudaq_probe()
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cudaq_importable": importlib.util.find_spec("cudaq") is not None,
        "probe": probe,
        "capabilities": cq.cudaq_capabilities(),
        "min_cudaq_version": ".".join(map(str, cq.MIN_CUDAQ_VERSION)),
        "nvidia_smi": _run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        )
        if shutil.which("nvidia-smi")
        else "nvidia-smi not found",
        "wsl": _run(["wsl", "--status"]) if shutil.which("wsl") else "wsl.exe not found",
        "pip_index": _run(["python", "-m", "pip", "index", "versions", "cudaq"]),
    }
    real = env["cudaq_importable"] and probe.get("available", False)
    status = "pending-real-run" if real else "blocked"
    reason = (
        "cudaq importable: run pytest tests/cudaq_adapter tests/hybrid -k real_cudaq to execute the checks"
        if real
        else "CUDA-Q is not importable here (Linux-only wheels; no WSL/Linux runtime installed)"
    )
    for name, extra in (
        ("HYB01_cudaq_real_runtime", {}),
        ("HYB11_gpu_performance", {"note": "GPU performance is measured only after HYB-01 passes"}),
    ):
        save(
            [{"experiment": name, "status": status, "reason": reason}],
            name,
            configuration={"environment": env, **extra},
            validation={"status": status, "reason": reason},
        )
        print(f"{name}: {status} ({reason})")


if __name__ == "__main__":
    main()
