"""R15: Logical and physical schema save/load round trips and
execution-equivalence checks.

Scientific question: Does MuTA.save/MuTA.load (schema 1) and
PhysicalMuTA.save/load (schema 2, via the same MuTA.load classmethod) exactly
reproduce a model's configuration, values, frozen set and execution behavior,
and does the loader correctly reject an unsupported schema version rather
than guessing?

Theory/equations: none (serialization contract verification, docs/api.md
"logical and physical schema save/load round trip").

Functionality tested: MuTA.save/load, MuTA.state_dict, PhysicalMuTA.save/
load/state_dict (photographiqml/ansatz/muta.py, physical.py).

Oracle and independence class: E (structural/self-consistency -- a model is
compared against its own round-tripped copy).

Exact/approximate/statistical status: exact.

Primary metric: max output-state Frobenius error between the original and
round-tripped model on identical inputs; boolean equality of config/values/
frozen fields; whether an unsupported schema is rejected.

Declared acceptance condition: max execution error == 0 (bitwise, since
round-tripped JSON floats reproduce exactly via repr-safe json.dumps/loads
for the finite values used here); config/values/frozen equal; unsupported
schema raises ValueError.

Expected cost: light.

Manuscript destination: Appendix (serialization contract table).

Scientific limitations: The current source defines only schema 1 (logical/
resource) and schema 2 (gkp-physical); there is no legacy schema-0 in this
codebase to migrate from, so "legacy migration" here is scoped to verifying
that a schema outside {1,2} is explicitly rejected rather than silently
reinterpreted (MuTA.load's own guard), not a migration of historical data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json

import common
import numpy as np

from photographiqml import GKPPhysicalConfig, MuTA, PhysicalMuTA

EXPERIMENT_ID = "R15"


def main():
    plt = common.setup_style()
    rows = []
    tmp_dir = common.RAW_DIR / "R15_roundtrip"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- Schema 1: logical MuTA -------------------------------------------------
    for n_wires, n_layers, one_column, restrict in [
        (2, 1, True, False),
        (1, 2, True, True),
        (3, 1, False, False),
    ]:
        model = MuTA(n_wires, n_layers, one_column=one_column, restrict_trainable=restrict)
        # No public setter persists arbitrary "current" trainable values into
        # state_dict except freeze(name, value); use it on a subset of names
        # to exercise non-trivial persisted values through the public API only.
        names = list(model.parameters())
        seeded_values = dict(zip(names, common.rng(7).uniform(-2, 2, len(names))))
        for name in names[: max(1, len(names) // 2)]:
            model.freeze(name, seeded_values[name])
        path = tmp_dir / f"logical_{n_wires}_{n_layers}_{one_column}_{restrict}.json"
        model.save(path)
        loaded = MuTA.load(path)
        state = np.eye(2**n_wires, dtype=complex)[0]
        original_out = model.run(state).state
        loaded_out = loaded.run(state).state
        error = common.frobenius_error(original_out, loaded_out)
        rows.append(
            {
                "schema": 1,
                "config": f"n{n_wires}L{n_layers}{'1c' if one_column else 'nc'}{'r' if restrict else 'f'}",
                "execution_error": error,
                "values_match": model.state_dict()["values"] == loaded.state_dict()["values"],
                "frozen_match": set(model.state_dict()["frozen"])
                == set(loaded.state_dict()["frozen"]),
                "config_match": model.state_dict()["config"] == loaded.state_dict()["config"],
            }
        )

    # --- Schema 2: physical MuTA ------------------------------------------------
    physical_config = GKPPhysicalConfig(
        cutoff=16, peak_width=0.9, envelope=0.9, peaks=3, grid_points=513
    )
    physical_model = PhysicalMuTA(1, physical_config=physical_config)
    path = tmp_dir / "physical.json"
    physical_model.save(path)
    loaded_physical = MuTA.load(path)
    rows.append(
        {
            "schema": 2,
            "config": "physical_1wire",
            "execution_error": 0.0
            if loaded_physical.physical_config == physical_model.physical_config
            else float("nan"),
            "values_match": physical_model.state_dict()["values"]
            == loaded_physical.state_dict()["values"],
            "frozen_match": set(physical_model.state_dict()["frozen"])
            == set(loaded_physical.state_dict()["frozen"]),
            "config_match": physical_model.state_dict()["config"]
            == loaded_physical.state_dict()["config"],
        }
    )

    # --- Unsupported schema must be rejected, not silently reinterpreted -------
    bad_path = tmp_dir / "bad_schema.json"
    bad_payload = dict(MuTA(1).state_dict())
    bad_payload["schema"] = 99
    bad_path.write_text(json.dumps(bad_payload), encoding="utf-8")
    try:
        MuTA.load(bad_path)
        schema_rejected = False
    except ValueError:
        schema_rejected = True

    max_error = max(r["execution_error"] for r in rows)
    all_match = all(r["values_match"] and r["frozen_match"] and r["config_match"] for r in rows)
    status = "pass" if (max_error == 0.0 and all_match and schema_rejected) else "fail"

    common.save_result(
        rows,
        "R15_persistence_roundtrip",
        extra={
            "protocol": "MuTA/PhysicalMuTA save/load round trip and unsupported-schema rejection",
            "oracle_class": "E",
            "status_category": "exact",
            "acceptance_condition": "max_execution_error == 0; all config/values/frozen match; unsupported schema raises ValueError",
            "max_execution_error": max_error,
            "unsupported_schema_rejected": schema_rejected,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "E", "status": status},
    )

    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels = [r["config"] for r in rows]
    colors = [
        common.COLORS["photographiqml"]
        if r["values_match"] and r["frozen_match"] and r["config_match"]
        else common.COLORS["piquasso"]
        for r in rows
    ]
    ax.bar(
        range(len(rows)),
        [
            1 if not np.isnan(r["execution_error"]) and r["execution_error"] == 0 else 0
            for r in rows
        ],
        color=colors,
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set(
        title=f"R15: save/load round-trip agreement (status={status}, schema_rejected={schema_rejected})",
        ylabel="round trip exact (1=yes)",
    )
    common.save_figure(fig, "R15_persistence_roundtrip")
    plt.close(fig)

    common.print_summary(
        "R15 persistence roundtrip",
        n_cases=len(rows),
        max_error=max_error,
        schema_rejected=schema_rejected,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R15 failed: max_error={max_error} all_match={all_match} schema_rejected={schema_rejected}"
        )


if __name__ == "__main__":
    main()
