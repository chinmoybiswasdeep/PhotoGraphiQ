"""R63: Encoded measurement back-action on an entangled GKP pair (Experiment D).

Scientific question: is the physical XY instrument a correct measurement *instrument* and not
only a correct outcome distribution? One half (data) of the finite-energy Bell state
(|0~0~> + |1~1~>)/norm is measured at angle alpha; the remaining partner must collapse to the
logical state +-(cos alpha, -sin alpha, 0) on the Bloch sphere (ideal qubit), with the sign
set by the logical outcome.

Partner tomography uses the same physical readouts for every level: X (p cells), Y (Gaussian
S^dagger shear then p cells), Z (q cells), giving decoded Bloch components.

* Reference (grid, untruncated): the instrument's map is linear, so applying it to the
  codeword operators |i~><j~| of the data mode gives exact coefficients c_b(i, j) and the
  exact conditional partner operator sum_ij c_b(i, j) |i~><j~| / norm^2 (no sampling).
* Engine (native-fock, broad resource, cutoff 56, three live modes during injection): each
  sampled trajectory leaves the partner's conditional state; its decoded Bloch vector is
  computed exactly and averaged within each logical outcome b (sampling SE reported).

Ideal limit: the reference (heralded records) is evaluated at the broad, medium and sharp
resources.

Acceptance: engine vs reference (broad, all records) within 4 SE + 0.01 for each Bloch
component and outcome with at least 4 records; for every (alpha, b) the heralded reference's
direction error and 1 - |Bloch| decrease from broad to medium to sharp, and the sharp direction
error is <= 0.1 rad.

Revision note: the first run asserted a 0.2 rad direction tolerance at the broad resource only.
Engine and reference agreed everywhere, but the broad reference is far from ideal (strong
contraction and a +X bias of the decoded partner from the finite envelope: a q-comb's central
momentum peak is over-weighted in X decoding). That criterion was replaced by the ideal-limit
trend above, as in R61.
"""

import numpy as np
from xy_support import bell_resource, code, grid, ideal_partner_bloch, partner_bloch, pg, plt, save

NAME = "R63_measurement_back_action"
ANGLES = (np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3)
RESOURCE, CUTOFF, SHOTS = "broad", 56, 24


def reference_branches(alpha, resource=RESOURCE, heralded_only=False):
    model = grid(resource)
    words = [model.codeword(b) for b in (0, 1)]
    overlap = float(np.trapezoid(words[0] * words[1], model.x))
    norm2 = 2 + 2 * overlap**2
    out = {}
    for b in (0, 1):
        bloch = np.zeros(3, complex)
        weight = 0.0
        for i in (0, 1):
            for j in (0, 1):
                rows = model.protocol_map(np.outer(words[i], words[j]), alpha, max_rounds=4)
                coefficient = sum(r["value"] for r in rows if r["bit"] == b and (r["success"] or not heralded_only))
                partner = np.outer(words[i], words[j])
                bloch += coefficient * np.array(model.bloch(partner))
                weight += (coefficient * np.trapezoid(words[i] * words[j], model.x)).real
        out[b] = {"probability": weight / norm2, "bloch": (bloch / weight).real}
    return out


def engine_records(alpha, seed):
    gkp = code(RESOURCE, CUTOFF)
    pattern = pg.Pattern(inputs=("p", "d"))
    instrument = gkp.logical_measurement("XY", alpha=alpha, protocol="auto", max_rounds=4)
    pattern.append(pg.Measure("d", instrument, "m"))
    pattern.append(pg.Output(("p",)))
    shots = pg.run_shots(pattern, SHOTS, seed=seed, backend="native-fock", cutoff=CUTOFF,
                         initial_state=bell_resource(gkp))
    return [(s.records["m"], partner_bloch(s.state, "p", CUTOFF)) for s in shots.trajectories]


def main():
    import warnings

    warnings.simplefilter("ignore", RuntimeWarning)
    rows, failures = [], []
    for alpha in ANGLES:
        reference = reference_branches(alpha)
        records = engine_records(alpha, seed=63)
        for b in (0, 1):
            ideal = np.array(ideal_partner_bloch(alpha, b))
            ref = reference[b]["bloch"]
            values = np.array([bloch for record, bloch in records if record.bit == b])
            row = {"alpha": float(alpha), "bit": b, "reference_probability": reference[b]["probability"],
                   "engine_count": len(values), "ideal_bloch": ideal.tolist(), "reference_bloch": ref.tolist(),
                   "reference_length": float(np.linalg.norm(ref)),
                   "reference_direction_error": float(abs(np.angle(np.exp(1j * (np.arctan2(ref[1], ref[0])
                                                                              - np.arctan2(ideal[1], ideal[0])))))),
                   "heralded_fraction": float(np.mean([r.heralded_success for r, _ in records]))}
            if len(values) >= 4:
                mean = values.mean(axis=0)
                se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
                ok = bool(np.all(np.abs(mean - ref) <= 4 * se + 0.01))
                row |= {"engine_bloch": mean.tolist(), "engine_se": se.tolist(), "pass": ok}
                if not ok:
                    failures.append((float(alpha), b))
            rows.append(row)
            print(f"a={alpha:.3f} b={b} ideal={np.round(ideal, 3)} ref={np.round(ref, 3)} "
                  f"eng={np.round(row.get('engine_bloch', [np.nan] * 3), 3)} n={len(values)}", flush=True)
    limit_rows = []
    for alpha in ANGLES:
        per_resource = {name: reference_branches(alpha, name, heralded_only=True) for name in ("broad", "medium", "sharp")}
        for b in (0, 1):
            ideal = np.array(ideal_partner_bloch(alpha, b))
            errors, deficits = [], []
            for name in ("broad", "medium", "sharp"):
                vector = per_resource[name][b]["bloch"]
                direction = float(abs(np.angle(np.exp(1j * (np.arctan2(vector[1], vector[0])
                                                            - np.arctan2(ideal[1], ideal[0]))))))
                errors.append(direction)
                deficits.append(float(1 - np.linalg.norm(vector)))
                limit_rows.append({"alpha": float(alpha), "bit": b, "resource": name, "heralded_bloch": vector.tolist(),
                                   "direction_error": direction, "length_deficit": deficits[-1]})
            trend = all(y < x for x, y in zip(errors, errors[1:])) and all(y < x for x, y in zip(deficits, deficits[1:]))
            if not trend or errors[-1] > 0.1:
                failures.append((float(alpha), b, "ideal-limit"))
            print(f"a={alpha:.3f} b={b} heralded direction errors {np.round(errors, 3)} length deficits {np.round(deficits, 3)}", flush=True)
    rows += [{"kind": "ideal_limit", **r} for r in limit_rows]
    status = "pass" if not failures else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 2, figsize=(9, 4.2))
    circle = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(np.cos(circle), np.sin(circle), color="0.8", lw=0.8)
    engine_rows = [r for r in rows if "kind" not in r]
    for k, row in enumerate(engine_rows):
        color = f"C{ANGLES.index(row['alpha'])}"
        ideal, ref = row["ideal_bloch"], row["reference_bloch"]
        axes[0].plot(*ideal[:2], "o", mfc="none", color=color, ms=9)
        axes[0].plot(*ref[:2], "x", color=color, ms=7)
        if "engine_bloch" in row:
            axes[0].errorbar(*row["engine_bloch"][:2], xerr=row["engine_se"][0], yerr=row["engine_se"][1],
                             fmt="s", color=color, ms=4)
    axes[0].set(xlabel="Partner <X>", ylabel="Partner <Y>", aspect="equal",
                title="Partner Bloch vector (o ideal, x grid, ■ engine)")
    for name, marker in (("broad", "o"), ("medium", "s"), ("sharp", "^")):
        sub = [r for r in rows if r.get("kind") == "ideal_limit" and r["resource"] == name]
        axes[1].semilogy(range(len(sub)), [max(r["direction_error"], 1e-4) for r in sub], marker + "-", label=name)
    axes[1].set(xticks=range(2 * len(ANGLES)),
                xticklabels=[f"{a:.2f}/b{b}" for a in ANGLES for b in (0, 1)],
                ylabel="Heralded partner direction error (rad)", title="Ideal limit (grid reference)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(fontsize=7)
    save(rows, NAME, {"oracle_class": "B", "status": status, "failures": failures, "resource": RESOURCE,
                      "cutoff": CUTOFF, "shots": SHOTS}, fig)
    print(f"R63 status={status} failures={failures}")
    if status != "pass":
        raise AssertionError("R63 acceptance failed")


if __name__ == "__main__":
    main()
