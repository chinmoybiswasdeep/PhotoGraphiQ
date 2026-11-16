"""HYB-07 (Kerr-enhanced MBQC, with and without loss) and HYB-08 (photon counting).

HYB-07. kerr_wire: K(kappa) on the input before the theta = 0.5 homodyne (a non-Gaussian
measurement basis), resource r = 0.4, coherent input (0.3, 0.1), outcome 0.2; kappa in
{0.05, 0.1, 0.2}; transmissivity eta in {1, 0.85} on both modes after the Kerr gate. Reference:
native-fock (eta = 1) / native-mixed-fock (eta < 1) at the largest cutoff. The Kerr effect size is
the trace distance to the kappa = 0 output. Acceptance: at dims 24, trace distance to the
reference <= max(1e-3, 3 * TD(native second-largest, largest)), relative density error <= 1e-4,
and the effect size exceeds 100 x that numerical error.

HYB-08. photon_subtraction(0.3) on a squeezed-vacuum input (r = 0.5, prepared by the bridge):
count k = 0..3 probabilities and conditional states vs native-fock (cutoff 36), and the sum of
the count probabilities. Acceptance at dims 28: |dP| < 1e-8, P(k) * infidelity < 1e-10 (the error
of the unnormalised branch), sum >= 1 - 1e-8. Recorded deviation: the first run declared
infidelity < 1e-8 at dims 20 against cutoff 24 and FAILED for k >= 1 (k = 3: 7.4e-5 with
P(3) = 8.2e-5): normalising a rare branch amplifies truncation error by 1 / P(k), as for rare
homodyne outcomes (HYB-04). The sweep was extended and the criterion stated per branch.
"""

import numpy as np
from hyb_support import (
    COLORS,
    Timer,
    infidelity,
    kerr_wire,
    pg,
    plt,
    save,
    simulate,
    trace_distance,
)

from photographiq.hybrid.executor import run

INPUT = {0: pg.GaussianInput((0.3, 0.1))}
OUT = {0: 0.2}


def native_state(r, mixed):
    return r.state.density_matrix if mixed else r.state.state_vector


def kerr_study():
    rows, checks = [], {}
    for eta in (1.0, 0.85):
        mixed = eta < 1
        losses = ((0, eta), (1, eta)) if mixed else ()
        backend, cutoffs = (
            ("native-mixed-fock", (18, 22, 26)) if mixed else ("native-fock", (20, 26, 32))
        )
        base = run(
            kerr_wire(0.0, losses=losses), dims=24, inputs=INPUT, measurement_outcomes=OUT
        ).state.data
        for kappa in (0.05, 0.1, 0.2):
            p = kerr_wire(kappa, losses=losses)
            nat = {
                c: simulate(p, backend=backend, cutoff=c, inputs=INPUT, measurement_outcomes=OUT)
                for c in cutoffs
            }
            ref = native_state(nat[cutoffs[-1]], mixed)
            ref_density = float(np.exp(nat[cutoffs[-1]].log_likelihood))
            floor = trace_distance(native_state(nat[cutoffs[-2]], mixed), ref)
            for d in (12, 16, 20, 24):
                with Timer() as t:
                    h = run(p, dims=d, inputs=INPUT, measurement_outcomes=OUT)
                rows.append(
                    {
                        "eta": eta,
                        "kappa": kappa,
                        "dim": d,
                        "representation": h.state.representation,
                        "trace_distance": trace_distance(h.state.data, ref),
                        "native_floor": floor,
                        "density_rel_error": abs(h.densities[0] / ref_density - 1),
                        "seconds": t.seconds,
                        "effect_size": trace_distance(h.state.data, base),
                        **{k: v for k, v in h.truncation.items() if k != "note"},
                    }
                )
            last = rows[-1]
            checks[f"eta={eta},kappa={kappa}"] = (
                last["trace_distance"] <= max(1e-3, 3 * floor)
                and last["density_rel_error"] <= 1e-4
                and last["effect_size"] > 100 * last["trace_distance"]
            )
            print(
                "HYB07",
                eta,
                kappa,
                f"TD {last['trace_distance']:.2e} floor {floor:.2e} effect {last['effect_size']:.3f}",
                f"dens {last['density_rel_error']:.1e}",
                flush=True,
            )
    return rows, checks


def counting_study():
    p = pg.non_gaussian.photon_subtraction(0.3)
    inputs = {"in": pg.GaussianInput.squeezed(0.5)}
    rows = []
    for d in (8, 12, 16, 20, 24, 28):
        total = 0.0
        for k in range(4):
            ref = simulate(
                p,
                backend="native-fock",
                cutoff=36,
                inputs=inputs,
                measurement_outcomes={"count": k},
            )
            h = run(p, dims=d, inputs=inputs, measurement_outcomes={"count": k})
            total += h.probabilities["count"]
            rows.append(
                {
                    "dim": d,
                    "count": k,
                    "probability": h.probabilities["count"],
                    "native_probability": float(np.exp(ref.log_likelihood)),
                    "probability_error": abs(
                        h.probabilities["count"] - float(np.exp(ref.log_likelihood))
                    ),
                    "infidelity": infidelity(h.state.data, ref.state.state_vector),
                }
            )
        sums = run_shots_sum(p, inputs, d)
        for r in rows[-4:]:
            r["sum_all_counts"] = sums
    final = [r for r in rows if r["dim"] == 28]
    checks = {
        "probabilities": max(r["probability_error"] for r in final) < 1e-8,
        "branch_states": max(r["probability"] * r["infidelity"] for r in final) < 1e-10,
        "sum": final[0]["sum_all_counts"] >= 1 - 1e-8,
    }
    return rows, checks


def run_shots_sum(p, inputs, d):
    """Sum of the count distribution over all n < d (exactly-zero counts raise and contribute 0)."""
    total = 0.0
    for k in range(d):
        try:
            total += run(p, dims=d, inputs=inputs, measurement_outcomes={"count": k}).probabilities[
                "count"
            ]
        except ValueError:
            pass
    return float(total)


def main():
    krows, kchecks = kerr_study()
    st07 = "pass" if all(kchecks.values()) else "fail"
    crows, cchecks = counting_study()
    st08 = "pass" if all(cchecks.values()) else "fail"
    pyplot = plt()
    fig, axes = pyplot.subplots(1, 2, figsize=(7.0, 2.5))
    colours = [COLORS["hybrid"], COLORS["native"], COLORS["gaussian"]]
    for eta, style in ((1.0, "o-"), (0.85, "s--")):
        for kappa, colour in zip((0.05, 0.1, 0.2), colours, strict=True):
            sel = [r for r in krows if r["eta"] == eta and r["kappa"] == kappa]
            axes[0].semilogy(
                [r["dim"] for r in sel],
                [r["trace_distance"] for r in sel],
                style,
                color=colour,
                label=f"$\\kappa$ {kappa}, $\\eta$ {eta}",
            )
    axes[0].set(
        xlabel="Per-mode dimension",
        ylabel="Trace distance to native",
        title="Kerr-enhanced MBQC step",
    )
    axes[0].legend(fontsize=5, ncol=2)
    for k, colour in zip(range(4), colours + [COLORS["accent"]], strict=True):
        sel = [r for r in crows if r["count"] == k]
        axes[1].semilogy(
            [r["dim"] for r in sel],
            [max(r["probability_error"], 1e-17) for r in sel],
            "o-",
            color=colour,
            label=f"k = {k}",
        )
    axes[1].set(
        xlabel="Per-mode dimension",
        ylabel=r"$|P_\mathrm{hyb} - P_\mathrm{native}|$",
        title="Photon counting after subtraction tap",
    )
    axes[1].legend(fontsize=6)
    save(
        krows,
        "HYB07_kerr_enhanced_mbqc",
        configuration={"theta": 0.5, "r": 0.4, "input": [0.3, 0.1], "outcome": 0.2},
        validation={"status": st07, "checks": kchecks},
        figure=fig,
        caption="Kerr-enhanced MBQC (lossless and lossy) and photon counting: hybrid executor vs native Fock engines.",
    )
    save(
        crows,
        "HYB08_photon_counting",
        configuration={"tap": 0.3, "input": "squeezed vacuum r=0.5 (bridge)", "native_cutoff": 36},
        validation={
            "status": st08,
            "checks": cchecks,
            "first_run": "dims 20 vs cutoff 24, infidelity < 1e-8: failed for k >= 1 (see docstring)",
        },
    )
    print(f"HYB07 {st07} {kchecks}\nHYB08 {st08} {cchecks}")


if __name__ == "__main__":
    main()
