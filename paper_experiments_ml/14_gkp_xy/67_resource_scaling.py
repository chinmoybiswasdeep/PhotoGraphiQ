"""R67: Resource and accuracy scaling of the physical logical-XY instrument (Experiment H).

Scientific question: what does the instrument cost, and what limits it numerically?

(a) Runtime and memory vs Fock cutoff (native-fock, medium resource) for a deterministic
    Clifford Y readout and a postselected one-round injection record (median of 3 repeats;
    tracemalloc peak of Python-side allocations plus the two-mode Fock vector size; native
    workspaces are not captured by tracemalloc).
(b) Energy growth: each failed injection round adds the ancilla's momentum spread to the data
    mode through the CNOT. A postselected chain of failures (ancilla q at the odd cell centre)
    is followed for up to 6 rounds at cutoff 112; the data-mode mean photon number and the
    minimum retained Fock norm are recorded after every round.
(c) Heralded success vs round budget K at alpha = pi/3: exact branch sum on the untruncated
    grid for K = 0..6 against 1 - 2^-K, and the engine's sampled heralded fraction for
    K = 1..4 (32 records each, broad resource, cutoff 64).

Acceptance: grid heralded success equals 1 - 2^-K within 1e-3; the engine fraction lies within
4 binomial SE (floored at 1/shots) of 1 - 2^-K; timings and photon growth are descriptive
(timing is machine dependent and provisional unless PGQ_IDLE_MACHINE=1).
"""

import time
import tracemalloc
import warnings
from math import comb

import numpy as np
from xy_support import GENERIC, code, grid, plt, run_single, save

import photographiq as pg
from photographiq.encoded import apply_logical_cnot
from photographiq.registry import create_backend

NAME = "R67_resource_scaling"
CUTOFFS = (56, 64, 80, 96, 112)
SPACING = float(np.sqrt(2 * np.pi))


def timed(function, repeats=3):
    seconds, peaks = [], []
    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        function()
        seconds.append(time.perf_counter() - start)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    return float(np.median(seconds)), int(max(peaks))


def energy_growth(cutoff=112, rounds=6, phase=np.pi / 3):
    gkp = code("medium", cutoff)
    engine = create_backend("native-fock", cutoff)
    engine.reset(0)
    engine.prepare("d", state=gkp.plus())
    rows = []
    for k in range(rounds):
        ancilla = ("a", k)
        try:
            engine.prepare(ancilla, state=gkp.encode(1, np.exp(1j * phase)))
            apply_logical_cnot(engine, "d", ancilla)
        except ValueError as exc:  # retained-norm guard: this cutoff cannot hold another round
            rows.append({"failed_rounds": k + 1, "refused": str(exc), "cutoff": cutoff})
            break
        engine.measure(ancilla, pg.Homodyne(0.0), 0.0, outcome=SPACING)  # odd cell: heralded failure
        reduced = engine.get_state(("d",))
        occupations = np.asarray(reduced.basis)[:, 0]
        photons = float(np.real(np.trace(np.asarray(reduced.density_matrix) @ np.diag(occupations))))
        rows.append({"failed_rounds": k + 1, "data_mean_photons": photons,
                     "min_retained_norm": float(min(engine.retained_norms)), "cutoff": cutoff})
        phase *= 2
    return rows


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    rows = []
    for cutoff in CUTOFFS:
        y_seconds, y_peak = timed(lambda: run_single("medium", np.pi / 2, (1, 1j), protocol="clifford", cutoff=cutoff))
        inj_seconds, inj_peak = timed(
            lambda: run_single("medium", np.pi / 3, GENERIC, cutoff=cutoff, outcomes={"ancilla": [0.0], "data": 0.3})
        )
        rows.append({"study": "cost", "cutoff": cutoff, "clifford_y_seconds": y_seconds,
                     "clifford_y_python_peak_bytes": y_peak, "injection_round_seconds": inj_seconds,
                     "injection_python_peak_bytes": inj_peak, "two_mode_dimension": comb(cutoff + 1, 2),
                     "two_mode_vector_bytes": 16 * comb(cutoff + 1, 2)})
        print(f"cutoff {cutoff}: Y {y_seconds:.2f}s, injection {inj_seconds:.2f}s", flush=True)

    growth = energy_growth()
    rows += [{"study": "energy_growth", **r} for r in growth]
    print("growth", [round(r.get("data_mean_photons", float("nan")), 2) for r in growth], flush=True)

    model = grid("broad")
    state = model.encode(*GENERIC)
    herald_ok = True
    for budget in range(7):
        protocol_rows = model.protocol_map(model.operator(state), np.pi / 3, max_rounds=budget)
        success = sum(r["value"].real for r in protocol_rows if r["success"])
        herald_ok &= abs(success - (1 - 2.0**-budget)) < 1e-3
        rows.append({"study": "heralding", "source": "grid", "max_rounds": budget, "heralded_success": success,
                     "expected": 1 - 2.0**-budget})
    for budget in (1, 2, 3, 4):
        records, _ = run_single("broad", np.pi / 3, GENERIC, shots=32, seed=670 + budget, max_rounds=budget)
        fraction = float(np.mean([r.heralded_success for r in records]))
        expected = 1 - 2.0**-budget
        se = max(np.sqrt(expected * (1 - expected) / 32), 1 / 32)
        herald_ok &= abs(fraction - expected) <= 4 * se
        rows.append({"study": "heralding", "source": "engine", "max_rounds": budget, "heralded_success": fraction,
                     "expected": expected, "se": se, "mean_rounds": float(np.mean([len(r.rounds) for r in records]))})
        print(f"budget {budget}: engine heralded {fraction:.3f} expected {expected:.3f}", flush=True)
    status = "pass" if herald_ok else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 3, figsize=(13, 3.8))
    cost = [r for r in rows if r["study"] == "cost"]
    axes[0].semilogy([r["cutoff"] for r in cost], [r["clifford_y_seconds"] for r in cost], "o-", label="Clifford Y (1 mode)")
    axes[0].semilogy([r["cutoff"] for r in cost], [r["injection_round_seconds"] for r in cost], "s-", label="one injection round (2 modes)")
    axes[0].set(xlabel="Fock cutoff", ylabel="Median wall time (s)", title="Cost vs cutoff (provisional timing)")
    axes[0].legend(fontsize=7)
    growth_rows = [r for r in rows if r["study"] == "energy_growth" and "refused" not in r]
    axes[1].plot([r["failed_rounds"] for r in growth_rows], [r["data_mean_photons"] for r in growth_rows], "o-", label="data-mode <n>")
    twin = axes[1].twinx()
    twin.semilogy([r["failed_rounds"] for r in growth_rows], [1 - r["min_retained_norm"] + 1e-16 for r in growth_rows], "s:", color="C3", label="1 - retained norm")
    axes[1].set(xlabel="Consecutive failed rounds", ylabel="Data-mode mean photon number", title="Energy growth (medium, cutoff 112)")
    twin.set_ylabel("1 - min retained norm")
    herald = [r for r in rows if r["study"] == "heralding"]
    grid_rows = [r for r in herald if r["source"] == "grid"]
    engine_rows = [r for r in herald if r["source"] == "engine"]
    axes[2].plot([r["max_rounds"] for r in grid_rows], [r["heralded_success"] for r in grid_rows], "o-", label="grid (exact)")
    axes[2].errorbar([r["max_rounds"] for r in engine_rows], [r["heralded_success"] for r in engine_rows],
                     yerr=[r["se"] for r in engine_rows], fmt="s", label="engine (32 records)")
    axes[2].plot(range(7), [1 - 2.0**-k for k in range(7)], "k:", lw=0.8, label="$1-2^{-K}$")
    axes[2].set(xlabel="Round budget K", ylabel="P(heralded success)", title="Repeat-until-success, $\\alpha=\\pi/3$")
    axes[2].legend(fontsize=7)
    save(rows, NAME, {"oracle_class": "A", "status": status, "heralding_ok": herald_ok,
                      "timing_status": common_timing()}, fig)
    print(f"R67 status={status}")
    if status != "pass":
        raise AssertionError("R67 acceptance failed")


def common_timing():
    import os

    return "final (idle machine declared)" if os.environ.get("PGQ_IDLE_MACHINE") == "1" else "PROVISIONAL"


if __name__ == "__main__":
    main()
