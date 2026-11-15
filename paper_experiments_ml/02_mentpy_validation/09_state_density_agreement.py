"""R9: Random numerical state and density-matrix agreement against MentPy
across multiple seeds and model sizes.

Scientific question: For random logical input states and random trainable
angles, does MuTA.run's output density matrix agree numerically with an
independently executed MentPy PatternSimulator run on the semantically
mapped circuit, across a range of model sizes and random seeds?

Theory/equations: photographiqml.validation.compare_mentpy constructs the
MentPy reference (mentpy_reference, fixing frozen columns explicitly),
simulates it with mp.PatternSimulator(backend="numpy-sv", output_form="dm")
over a flow-respecting schedule window, and returns the max absolute
density-matrix entrywise difference against MuTA.run's own density_matrix.

Functionality tested: MuTA.run (photographiqml.logical.execute) vs.
mp.PatternSimulator numpy statevector backend (mentpy), via
validation.compare_mentpy.

Oracle and independence class: B (independent external implementation).

Exact/approximate/statistical status: exact, up to floating-point roundoff
(both are deterministic numpy statevector simulations of the same finite
circuit).

Primary metric: max density-matrix entrywise absolute error, aggregated over
n_seeds independent random (state, parameter) draws per model configuration.

Declared acceptance condition: max error across the full seed x configuration
grid is below tol = declare_tolerance(scale=1, safety_factor=1000) (a larger
safety factor accounts for MentPy's own independent linear-algebra pipeline,
distinct floating-point operation ordering, and its window-limited pattern
simulator).

Expected cost: light-to-moderate (up to 3-wire, 2-layer models; 8 seeds each).

Manuscript destination: Main text (Fig. 3, headline MentPy agreement plot).

Scientific limitations: MentPy validates the ideal logical qubit layer only.
The numpy statevector backend defaults to the zero-outcome branch and
disallows random outcomes (docs/research/mentpy-audit.md); this experiment
therefore validates deterministic execution, not adaptive branch sampling
(see R4/R5 for exhaustive/independent branch checks within PhotoGraphiQML
itself, and R12 for instrument branch agreement).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common

from photographiqml import MuTA
from photographiqml.models import haar_states
from photographiqml.validation import MENTPY_COMMIT, compare_mentpy

EXPERIMENT_ID = "R9"

N_SEEDS = 8


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=1000.0)
    rows = []
    configs = [
        {"n_wires": 1, "n_layers": 1, "one_column": True},
        {"n_wires": 1, "n_layers": 3, "one_column": True},
        {"n_wires": 2, "n_layers": 1, "one_column": True},
        {"n_wires": 2, "n_layers": 1, "one_column": False},
        {"n_wires": 2, "n_layers": 2, "one_column": True},
        {"n_wires": 3, "n_layers": 1, "one_column": True},
    ]
    for cfg in configs:
        model = MuTA(cfg["n_wires"], cfg["n_layers"], one_column=cfg["one_column"])
        for seed in range(N_SEEDS):
            state = haar_states(cfg["n_wires"], 1, seed=1000 * seed + cfg["n_wires"])[0]
            parameters = model.initialize(seed=2000 * seed + cfg["n_layers"], scale=2.0)
            error = compare_mentpy(model, state, parameters)
            rows.append({**cfg, "seed": seed, "density_matrix_error": error})

    max_error = max(r["density_matrix_error"] for r in rows)
    status = "pass" if max_error < tol else "fail"

    common.save_result(
        rows,
        "R9_state_density_agreement",
        extra={
            "protocol": "compare_mentpy: MuTA.run vs. MentPy PatternSimulator (numpy-sv, output_form=dm)",
            "oracle_class": "B",
            "status_category": "exact",
            "mentpy_commit": MENTPY_COMMIT,
            "tolerance": tol,
            "acceptance_condition": f"max density-matrix error < {tol:.3e}",
            "max_error": max_error,
            "n_seeds_per_config": N_SEEDS,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "B", "status": status},
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    configs_labels = [
        f"n{c['n_wires']}L{c['n_layers']}{'1c' if c['one_column'] else 'nc'}" for c in configs
    ]
    for i, cfg in enumerate(configs):
        errors = [
            r["density_matrix_error"]
            for r in rows
            if r["n_wires"] == cfg["n_wires"]
            and r["n_layers"] == cfg["n_layers"]
            and r["one_column"] == cfg["one_column"]
        ]
        ax.scatter([i] * len(errors), errors, color=common.COLORS["mentpy"], alpha=0.7, s=18)
    ax.axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    ax.set_yscale("log")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs_labels, rotation=45, ha="right", fontsize=7)
    ax.set(
        title=f"R9: MuTA vs. MentPy density-matrix agreement (status={status})",
        ylabel="max |Delta rho|",
    )
    ax.legend()
    common.save_figure(fig, "R9_state_density_agreement")
    plt.close(fig)

    common.print_summary(
        "R9 state/density agreement",
        n_points=len(rows),
        tol=tol,
        max_error=max_error,
        status=status,
    )
    if status != "pass":
        raise AssertionError(f"R9 failed: max_error={max_error} tol={tol}")


if __name__ == "__main__":
    main()
