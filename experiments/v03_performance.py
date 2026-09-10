"""Supplement v0.3 evidence with dense preparation and compiler allocation costs."""

import argparse
import json
import platform
import time
import tracemalloc
import warnings
from pathlib import Path

import photographiq as pg


def measure(function):
    tracemalloc.start()
    start = time.perf_counter()
    result = function()
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, seconds, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for modes, cutoff in ((1, 16), (1, 32), (2, 8), (2, 12), (3, 5), (3, 7)):
        pattern = pg.Pattern(inputs=tuple(range(modes))).append(pg.Rotate(0, 0.2))
        inputs = {n: pg.GaussianInput.coherent(0.2 + 0.1j) for n in range(modes)}

        def run():
            return pg.simulate(pattern, inputs=inputs, backend="piquasso-mixed-fock", cutoff=cutoff)

        run()
        result, seconds, peak = measure(run)
        dimension = len(result.state.basis)
        rows.append(
            {
                "kind": "dense coherent density preparation and rotation",
                "modes": modes,
                "cutoff": cutoff,
                "dimension": dimension,
                "matrix_bytes": 16 * dimension**2,
                "runtime_seconds": seconds,
                "tracemalloc_peak_bytes": peak,
            }
        )
        print(f"Dense mixed modes={modes}, cutoff={cutoff}: {seconds:.3f}s", flush=True)
    for steps in (1, 2, 4):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pattern, seconds, peak = measure(
                lambda: pg.Circuit(1).kerr(0, 0.002).compile(synthesis_steps=steps)
            )
        rows.append(
            {
                "kind": "Kerr MBQC compilation",
                "modes": 1,
                "steps": steps,
                "commands": len(pattern.commands),
                "runtime_seconds": seconds,
                "tracemalloc_peak_bytes": peak,
            }
        )
        print(f"Kerr steps={steps}: {len(pattern.commands)} commands, {seconds:.3f}s", flush=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "allocation_scope": "Python and registered NumPy allocations, not process RSS or all native workspaces; warm mixed backend; concurrent validation can affect wall times",
                "measurements": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
