"""HYB-05 (primary acceptance experiment): three-mode cubic-phase CV-MBQC vs native-fock.

Pattern (hyb_support.three_mode_cubic): coherent input (0.3, 0.1) on mode 0; mode 1 prepared
p-squeezed (r = 0.4); mode 2 prepared as CubicPhaseResource(gamma, r = 0.4) (bridge + cubic
Hamiltonian); CZ(0, 1); homodyne mode 0 at theta = 1.1 with correction q -> -m0 / sin(theta);
inverse-SUM coupling of the resource; q homodyne of mode 2 -> m; quadratic-phase and nonlinear
feed-forward QP(-2 gamma m), Z(-gamma m^2). Three modes are live together; two measurements;
three outcome-dependent corrections.

The same Pattern runs on the hybrid executor (per-mode dims 10..28, truncated-generator gates)
and on native-fock (total-photon cutoffs 20..48, the reference) with identical postselected
outcomes; piquasso-fock (cutoffs 20, 26) is recorded as an independent engine where it passes its
norm guard. The two truncations differ, so agreement is a statement about the converged limit.

Acceptance (declared before running): for every (gamma, outcomes) case
  (1) infidelity(hybrid largest dims, native ref) <= max(1e-5, 10 * infidelity(native floor, native ref));
  (2) |joint density(hybrid largest dims) / joint density(native ref) - 1| <= 1e-4;
  (3) the hybrid infidelity at the largest dims is below the one at dims 10 (convergence).
ref / floor = largest / second-largest native cutoff. Recorded deviations: (a) the first run used
cutoffs 16..40; native-fock's norm guard rejected cutoffs 16 and (gamma = 0.3) 26 and 32, so the
run aborted before any comparison; the sweep was extended to 48. (b) With cutoffs <= 48 and dims
<= 28, gamma = 0.3 FAILED criterion (2) (2.8e-4) while native itself moved by 3.6e-4 between
cutoffs 40 and 48 (unconverged reference); the gamma = 0.3 sweep was extended to native 72 and
dims 40. Both engines converge slowly and non-monotonically at gamma = 0.3 (strong q^3 phase).
"""

import numpy as np
from hyb_support import COLORS, Timer, infidelity, pg, plt, save, simulate, three_mode_cubic

from photographiq.hybrid.executor import run
from photographiq.hybrid.states import FockRep

INPUT = {0: pg.GaussianInput((0.3, 0.1))}
DIMS = (10, 12, 16, 20, 24, 28)
CUTOFFS = (16, 20, 26, 32, 40, 48)
CASES = [
    (0.15, {0: 0.2, "m": -0.3}, DIMS, CUTOFFS),
    (0.15, {0: -0.6, "m": 0.5}, DIMS, CUTOFFS),
    (0.15, {0: 1.0, "m": 0.0}, DIMS, CUTOFFS),
    (0.3, {0: 0.2, "m": -0.3}, DIMS + (32, 36, 40), (40, 48, 56, 64, 72)),
]


def observables(vec):
    rep = FockRep(vec / np.linalg.norm(vec), (0,), (len(vec),))
    (q, p), _ = rep.moments()
    return {
        "q": float(q),
        "p": float(p),
        "n": float(np.sum(np.arange(len(vec)) * np.abs(rep.data) ** 2)),
    }


def main():
    rows, checks = [], {}
    for gamma, outcomes, dims, cutoffs in CASES:
        p = three_mode_cubic(gamma)
        label = f"gamma={gamma}, m0={outcomes[0]}, m={outcomes['m']}"
        native = {}
        for c in cutoffs:
            try:
                with Timer() as t:
                    native[c] = simulate(
                        p,
                        backend="native-fock",
                        cutoff=c,
                        inputs=INPUT,
                        measurement_outcomes=outcomes,
                    )
                native[c].seconds = t.seconds
            except ValueError as exc:  # native norm guard: recorded, never dropped
                rows.append({"case": label, "engine": "native-fock", "size": c, "error": str(exc)})
        ref = native[cutoffs[-1]].state.state_vector
        ref_density = float(np.exp(native[cutoffs[-1]].log_likelihood))
        for c, r in native.items():
            rows.append(
                {
                    "case": label,
                    "engine": "native-fock",
                    "size": c,
                    "hilbert_dimension": len(r.state.native.state_vector)
                    if hasattr(r.state.native, "state_vector")
                    else None,
                    "joint_density": float(np.exp(r.log_likelihood)),
                    "infidelity_to_reference": infidelity(r.state.state_vector, ref),
                    "seconds": r.seconds,
                    **observables(r.state.state_vector),
                }
            )
        for c in (20, 26):
            try:
                with Timer() as t:
                    r = simulate(
                        p,
                        backend="piquasso-fock",
                        cutoff=c,
                        inputs=INPUT,
                        measurement_outcomes=outcomes,
                    )
                rows.append(
                    {
                        "case": label,
                        "engine": "piquasso-fock",
                        "size": c,
                        "joint_density": float(np.exp(r.log_likelihood)),
                        "infidelity_to_reference": infidelity(r.state.state_vector, ref),
                        "seconds": t.seconds,
                        **observables(r.state.state_vector),
                    }
                )
            except (ValueError, ImportError, NotImplementedError) as exc:  # recorded, never dropped
                rows.append(
                    {
                        "case": label,
                        "engine": "piquasso-fock",
                        "size": c,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        for d in dims:
            with Timer() as t:
                h = run(p, dims=d, inputs=INPUT, measurement_outcomes=outcomes)
            rows.append(
                {
                    "case": label,
                    "engine": "hybrid (numpy-reference)",
                    "size": d,
                    "hilbert_dimension": d**3,
                    "peak_bytes": h.plan.peak_bytes,
                    "joint_density": float(np.exp(h.log_likelihood)),
                    "infidelity_to_reference": infidelity(h.state.data, ref),
                    "seconds": t.seconds,
                    "lost_weight": 1 - h.retained_weight,
                    **h.truncation,
                    **observables(h.state.data),
                }
            )
            rows[-1].pop("note", None)
        hyb = {
            r["size"]: r for r in rows if r["case"] == label and r["engine"].startswith("hybrid")
        }
        floor = infidelity(native[cutoffs[-2]].state.state_vector, ref)
        top = hyb[dims[-1]]
        checks[label] = {
            "infidelity": top["infidelity_to_reference"] <= max(1e-5, 10 * floor),
            "density": abs(top["joint_density"] / ref_density - 1) <= 1e-4,
            "converges": top["infidelity_to_reference"] < hyb[10]["infidelity_to_reference"],
            "reference_cutoff": cutoffs[-1],
            "hybrid_dims": dims[-1],
            "native_floor": floor,
            "hybrid_infidelity": top["infidelity_to_reference"],
            "density_rel_error": abs(top["joint_density"] / ref_density - 1),
        }
        print(label, checks[label], flush=True)
    status = (
        "pass"
        if all(v["infidelity"] and v["density"] and v["converges"] for v in checks.values())
        else "fail"
    )

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 3, figsize=(7.0, 2.5))
    colours = [COLORS["hybrid"], COLORS["native"], COLORS["gaussian"], COLORS["accent"]]
    for (gamma, outcomes, _, cutoffs), colour in zip(CASES, colours, strict=True):
        label = f"gamma={gamma}, m0={outcomes[0]}, m={outcomes['m']}"
        h = [r for r in rows if r["case"] == label and r["engine"].startswith("hybrid")]
        n = [
            r
            for r in rows
            if r["case"] == label
            and r["engine"] == "native-fock"
            and r["size"] != cutoffs[-1]
            and "error" not in r
        ]
        axes[0].semilogy(
            [r["size"] for r in h],
            [max(r["infidelity_to_reference"], 1e-16) for r in h],
            "o-",
            color=colour,
            label=f"$\\gamma$ {gamma}, m=({outcomes[0]}, {outcomes['m']})",
        )
        axes[0].semilogy(
            [r["size"] for r in n],
            [max(r["infidelity_to_reference"], 1e-16) for r in n],
            "s:",
            color=colour,
            ms=2.5,
        )
        ref = [
            r
            for r in rows
            if r["case"] == label and r["engine"] == "native-fock" and r["size"] == cutoffs[-1]
        ][0]["joint_density"]
        axes[1].semilogy(
            [r["size"] for r in h],
            [abs(r["joint_density"] / ref - 1) + 1e-16 for r in h],
            "o-",
            color=colour,
        )
        axes[2].loglog(
            [r["hilbert_dimension"] for r in h], [r["seconds"] for r in h], "o-", color=colour
        )
    axes[0].set(
        xlabel="Per-mode dim (o) / total cutoff (s)",
        ylabel="Infidelity to native reference",
        title="Output state",
    )
    axes[0].legend(fontsize=5)
    axes[1].set(
        xlabel="Per-mode dimension",
        ylabel=r"$|p_\mathrm{hyb}/p_\mathrm{native} - 1|$",
        title="Joint outcome density",
    )
    axes[2].set(
        xlabel="Hybrid Hilbert dimension ($d^3$)",
        ylabel="Seconds per trajectory (CPU)",
        title="Hybrid runtime (NumPy)",
    )
    save(
        rows,
        "HYB05_three_mode_cubic_mbqc",
        configuration={
            "cases": [[g, {str(k): v for k, v in o.items()}, d, c] for g, o, d, c in CASES],
            "input": [0.3, 0.1],
            "graph_weights": {"CZ(0,1)": 1.0, "CZ(1,2) between R(pi/2) frames": -1.0},
            "measurement_angles": {"0": 1.1, "m (mode 2)": 0.0},
            "correction_gains": {
                "q on 1 per m0": "-1/sin(1.1)",
                "QP on 1 per m": "-2 gamma",
                "p on 1 per m^2": "-gamma",
            },
            "resource_squeezing": {"mode 1": 0.4, "cubic resource": 0.4},
            "r": 0.4,
            "theta": 1.1,
            "engine": "numpy-reference (CPU)",
        },
        validation={"status": status, "checks": checks},
        figure=fig,
        caption="Three-mode cubic-phase CV-MBQC: hybrid executor vs native-fock (cutoff sweeps), joint densities, CPU runtime.",
    )
    print(f"HYB05 status={status}")
    if status != "pass":
        raise AssertionError("HYB-05 acceptance failed")


if __name__ == "__main__":
    main()
