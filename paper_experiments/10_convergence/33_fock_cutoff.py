"""R33: Fock cutoff convergence.

Runs representative non-Gaussian workloads (cat state, photon subtraction,
cubic phase, GKP resource) at strictly increasing Fock cutoffs using
``photographiq.convergence.cutoff_convergence``, and reports retained norm,
boundary population, photon number, adjacent-cutoff fidelity/trace distance
(from the study itself) plus fidelity to the HIGHEST tested cutoff (computed
separately here). The highest tested cutoff is a reference point, not an
exact/infinite-cutoff ground truth -- convergence must still be judged from
the trend, not assumed from the last value's smallness alone.

Upgrade migration (2026-11):

* every original workload runs on ``native-fock`` (main) and ``piquasso-fock``
  (historical engine); the fidelity between the two engines' states at each
  cutoff is reported and must be >= 1 - ENGINE_INFIDELITY (declared below);
* two DEMANDING Gaussian workloads with an exact reference were added, because
  the original low-energy workloads converge already at the lowest cutoff:
  (a) ``displaced_squeezed_gates``: vacuum -> Squeeze(0.5) -> Displace(q = 8,
  i.e. alpha = 4), mean photon number ~16, cutoffs 40..120, on ``native-fock``
  and ``piquasso-fock``; (b) ``thermal_displaced_squeezed``: a mixed Gaussian
  input (thermal 0.2, r = 0.3, alpha = 3) on ``native-mixed-fock`` and
  ``piquasso-mixed-fock`` -- this is the combined squeeze+displace Gaussian
  unitary whose native implementation was re-derived in this tree
  (docs/validation/native-fock-gaussian-unitary.md). Their photon number and
  <q> are compared with the exact Gaussian moments at every cutoff. Asserted:
  the native engines reach relative photon-number error <= NATIVE_EXACT_TOLERANCE
  at the highest cutoff. Piquasso's values are REPORTED (Piquasso 8.0.1's
  displacement/squeezing matrices are documented to lose accuracy at large
  cutoffs; see backends/native_fock.py), not asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

ENGINES = ("native-fock", "piquasso-fock")
ENGINE_INFIDELITY = 1e-6
NATIVE_EXACT_TOLERANCE = 1e-8

WORKLOADS = {
    "cat_state": dict(
        pattern=pg.Pattern(inputs=()).append(pg.Prepare(0, state=pg.CatResource(1.5, parity=1))).append(pg.Output((0,))),
        cutoffs=[16, 24, 32, 48, 64],
        node=0,
        kwargs={},
    ),
    "photon_subtraction": dict(
        pattern=pg.non_gaussian.photon_subtraction(0.3),
        cutoffs=[8, 12, 16, 24, 32],
        node="in",
        kwargs={"inputs": {"in": pg.FockInput.number(2)}, "measurement_outcomes": {"count": 1}},
    ),
    "cubic_phase": dict(
        pattern=pg.Pattern(inputs=(0,)).append(pg.CubicPhase(0, 0.15)).append(pg.Output((0,))),
        cutoffs=[16, 24, 32, 48, 64],
        node=0,
        kwargs={"inputs": {0: pg.GaussianInput.coherent(0.3 + 0.0j)}},
    ),
    "gkp_resource": dict(
        pattern=pg.Pattern(inputs=()).append(pg.Prepare(0, state=pg.GKPResource(0, peak_width=0.4, envelope=0.4))).append(pg.Output((0,))),
        cutoffs=[24, 32, 48, 64, 96],
        node=0,
        kwargs={},
    ),
}

_THERMAL, _R_MIXED, _Q_MIXED = 0.2, 0.3, 6.0
DEMANDING = {
    "displaced_squeezed_gates": dict(
        pattern=pg.Pattern(inputs=(0,)).append(pg.Squeeze(0, 0.5)).append(pg.Displace(0, q=8.0)).append(pg.Output((0,))),
        cutoffs=[40, 60, 80, 100, 120],
        engines=("native-fock", "piquasso-fock"),
        kwargs={},
        exact_mean=np.array([8.0, 0.0]),
        exact_cov=np.diag([np.exp(-1.0), np.exp(1.0)]),
    ),
    "thermal_displaced_squeezed": dict(
        pattern=pg.Pattern(inputs=(0,)).append(pg.Output((0,))),
        cutoffs=[40, 60, 80],
        engines=("native-mixed-fock", "piquasso-mixed-fock"),
        kwargs={"inputs": {0: pg.GaussianInput(
            (_Q_MIXED, 0.0),
            tuple(map(tuple, (2 * _THERMAL + 1) * np.diag([np.exp(-2 * _R_MIXED), np.exp(2 * _R_MIXED)]))),
        )}},
        exact_mean=np.array([_Q_MIXED, 0.0]),
        exact_cov=(2 * _THERMAL + 1) * np.diag([np.exp(-2 * _R_MIXED), np.exp(2 * _R_MIXED)]),
    ),
}


def main():
    plt = common.setup_style()
    all_rows = {}
    finals = {}
    for name, spec in WORKLOADS.items():
        for engine in ENGINES:
            study = pg.cutoff_convergence(spec["pattern"], spec["cutoffs"], backend=engine, **spec["kwargs"])
            final_state = study.results[-1].state
            rows = []
            for row, result in zip(study.rows, study.results, strict=True):
                last = row["cutoff"] == spec["cutoffs"][-1]
                rows.append(
                    {
                        "workload": name,
                        "backend": engine,
                        "cutoff": row["cutoff"],
                        "dimension": common.fock_dimension(len(result.state.nodes), row["cutoff"]),
                        "seconds": row["seconds"],
                        "norm": row["norm"],
                        "minimum_retained_norm": row["minimum_retained_norm"],
                        "maximum_boundary_population": row["maximum_boundary_population"],
                        "photon_number": row["photon_numbers"].get(spec["node"]),
                        "fidelity_to_previous": row["fidelity_to_previous"],
                        "trace_distance_to_previous": row["trace_distance_to_previous"],
                        "fidelity_to_highest_cutoff": 1.0 if last else pg.fidelity(result.state, final_state),
                        "trace_distance_to_highest_cutoff": 0.0 if last else pg.trace_distance(result.state, final_state),
                        "state": result.state,
                    }
                )
            all_rows[(name, engine)] = rows
            print(f"  {name} [{engine}]: " + ", ".join(f"c={r['cutoff']}:F={r['fidelity_to_highest_cutoff']:.6f}" for r in rows), flush=True)
        for mine, theirs in zip(all_rows[(name, ENGINES[0])], all_rows[(name, ENGINES[1])], strict=True):
            agreement = pg.fidelity(mine["state"], theirs["state"])
            mine["native_vs_piquasso_fidelity"] = theirs["native_vs_piquasso_fidelity"] = agreement

    demanding_rows = []
    for name, spec in DEMANDING.items():
        exact_n = (np.trace(spec["exact_cov"]) + spec["exact_mean"] @ spec["exact_mean"] - 2) / 4
        for engine in spec["engines"]:
            run = common.safe_cutoff_run(
                lambda spec=spec, engine=engine: pg.cutoff_convergence(spec["pattern"], spec["cutoffs"], backend=engine, **spec["kwargs"]),
                label=f"{name} {engine}",
            )
            if not run["ok"]:
                demanding_rows.append({"workload": name, "backend": engine, "cutoff": None, "ok": False, "error": run["error"]})
                continue
            for row, result in zip(run["value"].rows, run["value"].results, strict=True):
                state = result.state
                q_mean, q_var = state.quadrature(0)
                demanding_rows.append({
                    "workload": name, "backend": engine, "cutoff": row["cutoff"], "ok": True,
                    "exact_photon_number": float(exact_n),
                    "photon_number": state.photon_number(0),
                    "relative_photon_number_error": abs(state.photon_number(0) - exact_n) / exact_n,
                    "q_mean_error": abs(q_mean - spec["exact_mean"][0]),
                    "q_variance_error": abs(q_var - spec["exact_cov"][0, 0]),
                    "minimum_retained_norm": row["minimum_retained_norm"],
                    "maximum_boundary_population": row["maximum_boundary_population"],
                    "seconds": row["seconds"],
                })
        finals[name] = spec["cutoffs"][-1]
    common.save_csv(demanding_rows, "R33_fock_cutoff_demanding")

    flat_rows = [{k: v for k, v in row.items() if k != "state"} for rows in all_rows.values() for row in rows]
    failures = [r for r in flat_rows if r["native_vs_piquasso_fidelity"] < 1 - ENGINE_INFIDELITY]
    failures += [
        r for r in demanding_rows
        if r["backend"].startswith("native") and (not r["ok"] or (r["cutoff"] == finals[r["workload"]]
                                                                and r["relative_photon_number_error"] > NATIVE_EXACT_TOLERANCE))
    ]
    common.save_result(flat_rows, "R33_fock_cutoff", extra={
        "workloads": list(WORKLOADS), "engines": list(ENGINES), "engine_infidelity_tolerance": ENGINE_INFIDELITY,
        "demanding_workloads": {k: {"cutoffs": v["cutoffs"], "engines": list(v["engines"])} for k, v in DEMANDING.items()},
        "native_exact_tolerance": NATIVE_EXACT_TOLERANCE, "demanding_rows": demanding_rows})
    if failures:
        common.save_csv(failures, "R33_fock_cutoff_FAILURES")
        raise AssertionError(f"{len(failures)} R33 checks failed; see FAILURES csv")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, name in zip(axes.flat[:4], WORKLOADS, strict=True):
        for engine, style in zip(ENGINES, ("o-", "x--"), strict=True):
            rows = all_rows[(name, engine)][:-1]
            ax.semilogy([r["cutoff"] for r in rows], [max(1e-16, 1 - r["fidelity_to_highest_cutoff"]) for r in rows],
                        style, label=engine)
        ax.set_xlabel("Fock cutoff")
        ax.set_ylabel("1 - F(state, highest cutoff)")
        ax.set_title(f"{name} (ref. c={WORKLOADS[name]['cutoffs'][-1]})", fontsize=9.5)
        ax.legend(fontsize=7)
    for ax, name in zip(axes.flat[4:], DEMANDING, strict=True):
        for engine, style in zip(DEMANDING[name]["engines"], ("o-", "x--"), strict=True):
            rows = [r for r in demanding_rows if r["workload"] == name and r["backend"] == engine and r["ok"]]
            ax.semilogy([r["cutoff"] for r in rows], [max(1e-17, r["relative_photon_number_error"]) for r in rows],
                        style, label=engine)
        ax.set_xlabel("Fock cutoff")
        ax.set_ylabel("|<n> - exact| / exact")
        ax.set_title(f"{name} (exact Gaussian reference)", fontsize=9.5)
        ax.legend(fontsize=7)
    fig.suptitle("Fock cutoff convergence: first four panels use the highest cutoff as reference, last two the exact Gaussian moments"
                 " (values below 1e-16 are exact zeros drawn at the floor)", fontsize=10)
    common.save_figure(fig, "R33_fock_cutoff")
    plt.close(fig)

    def at(name, engine, cutoff):
        return [r for r in demanding_rows if r["workload"] == name and r["backend"] == engine and r.get("cutoff") == cutoff]

    ds = {e: at("displaced_squeezed_gates", e, 120) for e in DEMANDING["displaced_squeezed_gates"]["engines"]}
    th = {e: at("thermal_displaced_squeezed", e, 80) for e in DEMANDING["thermal_displaced_squeezed"]["engines"]}
    fmt = lambda rows: f"{rows[0]['relative_photon_number_error']:.1e}" if rows else "n/a"  # noqa: E731
    common.save_caption(
        "R33_fock_cutoff",
        f"""
**R33: Fock cutoff convergence.** Top row and bottom left: infidelity of each state to the state at the highest
tested cutoff for a cat state (alpha = 1.5), heralded photon subtraction from |2>, the cubic phase (gamma = 0.15) on a
coherent state and a finite-energy GKP resource, on `native-fock` (circles) and `piquasso-fock` (crosses); the
highest cutoff is a reference, not the exact state, so convergence is judged from the trend. The two engines agree at
every cutoff to fidelity >= {min(r['native_vs_piquasso_fidelity'] for r in flat_rows):.12f}. Bottom middle/right:
demanding Gaussian workloads with an exact reference -- vacuum squeezed (r = 0.5) then displaced to alpha = 4
(<n> = {at('displaced_squeezed_gates', 'native-fock', 120)[0]['exact_photon_number']:.3f}, cutoffs 40-120) and a thermal
(0.2) displaced (alpha = 3) squeezed (r = 0.3) mixed input on the density-matrix engines (cutoffs 40-80). Relative error of
<n> at the highest cutoff: {fmt(ds['native-fock'])} (native-fock) vs {fmt(ds['piquasso-fock'])} (piquasso-fock) and
{fmt(th['native-mixed-fock'])} (native-mixed-fock) vs {fmt(th['piquasso-mixed-fock'])} (piquasso-mixed-fock). Truncation
is never hidden: retained norms and boundary populations are in the result files. Model: hbar = 2, total-photon cutoff.
Source: `paper_experiments/10_convergence/33_fock_cutoff.py`.
""",
    )

    common.print_summary(
        "R33 Fock cutoff convergence",
        workloads=list(WORKLOADS) + list(DEMANDING),
        min_final_fidelity=min(rows[-2]["fidelity_to_highest_cutoff"] for rows in all_rows.values() if len(rows) > 1),
        min_native_vs_piquasso_fidelity=min(r["native_vs_piquasso_fidelity"] for r in flat_rows),
        demanding_at_highest_cutoff={f"{r['workload']}[{r['backend']}]": r["relative_photon_number_error"]
                                     for r in demanding_rows if r.get("cutoff") == finals[r["workload"]]},
    )


if __name__ == "__main__":
    main()
