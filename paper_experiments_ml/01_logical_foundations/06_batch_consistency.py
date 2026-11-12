"""R6: run_batch versus repeated scalar run, including empty batches and
complex normalized inputs.

Scientific question: Does MuTA.run_batch produce exactly the same output as
calling MuTA.run once per row (same shared parameters), for arbitrary complex
normalized inputs, and does it correctly handle a zero-row batch without
error?

Theory/equations: run_batch is documented (docs/api.md) as
"normalized complex rows (N,2**n), shared parameters -> complex (N,2**n);
empty batch supported". This is a self-consistency/structural property, not
a new physics claim.

Functionality tested: MuTA.run_batch vs. MuTA.run (photographiqml/ansatz/muta.py).

Oracle and independence class: E (structural/self-consistency test) --
run_batch is checked against the same model's own scalar run, not an
external or independently derived reference.

Exact/approximate/statistical status: exact (bitwise-identical arithmetic
path is expected, since run_batch's implementation literally calls run per
row; this experiment verifies that documented contract holds, including at
its two declared edge cases).

Primary metric: max per-row Frobenius error between run_batch and repeated
run; behavior on an empty batch (shape and absence of exception).

Declared acceptance condition: max per-row error == 0 (bitwise, since both
paths execute identical code); run_batch([]) returns shape (0, 2**n) without
raising.

Expected cost: light.

Manuscript destination: Appendix (API contract verification table).

Scientific limitations: This is a self-consistency check of the public
interface, not independent physics validation (see R1-R5, R7-R9 for that).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import MuTA
from photographiqml.models import haar_states

EXPERIMENT_ID = "R6"


def main():
    plt = common.setup_style()
    rows = []
    for n_wires in (1, 2, 3):
        for n_batch in (0, 1, 5, 17):
            model = MuTA(n_wires, 2)
            parameters = model.initialize(seed=n_wires * 100 + n_batch, scale=1.0)
            states = (
                haar_states(n_wires, n_batch, seed=n_wires * 100 + n_batch + 7)
                if n_batch
                else np.empty((0, 2**n_wires), dtype=complex)
            )
            batch_output = model.run_batch(states, parameters)
            shape_ok = batch_output.shape == (n_batch, 2**n_wires)
            if n_batch:
                scalar_outputs = np.array([model.run(s, parameters).state for s in states])
                error = common.frobenius_error(batch_output, scalar_outputs)
            else:
                error = 0.0
            rows.append(
                {
                    "n_wires": n_wires,
                    "n_batch": n_batch,
                    "shape_ok": shape_ok,
                    "error": error,
                }
            )

    max_error = max(r["error"] for r in rows)
    all_shapes_ok = all(r["shape_ok"] for r in rows)
    status = "pass" if (max_error == 0.0 and all_shapes_ok) else "fail"

    common.save_result(
        rows,
        "R6_batch_consistency",
        extra={
            "protocol": "run_batch vs. repeated scalar run, including n_batch=0 edge case",
            "oracle_class": "E",
            "status_category": "exact",
            "acceptance_condition": "max_error == 0 (bitwise) and all output shapes correct",
            "max_error": max_error,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "E", "status": status},
    )

    fig, ax = plt.subplots(figsize=(6, 3.6))
    labels = [f"n{r['n_wires']}b{r['n_batch']}" for r in rows]
    colors = [
        common.COLORS["photographiqml"]
        if r["shape_ok"] and r["error"] == 0
        else common.COLORS["piquasso"]
        for r in rows
    ]
    # All-zero values render invisibly as bars; use markers so every tested
    # case is visibly plotted, not just an empty axes.
    ax.scatter(range(len(rows)), [r["error"] for r in rows], color=colors, zorder=3)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set(
        title=f"R6: run_batch vs. repeated run, {len(rows)} cases (status={status})",
        ylabel="Frobenius error",
    )
    common.save_figure(fig, "R6_batch_consistency")
    plt.close(fig)

    common.print_summary(
        "R6 batch consistency", n_cases=len(rows), max_error=max_error, status=status
    )
    if status != "pass":
        raise AssertionError(f"R6 failed: max_error={max_error} all_shapes_ok={all_shapes_ok}")


if __name__ == "__main__":
    main()
