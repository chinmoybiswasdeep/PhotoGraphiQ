"""R2: Identity wire versus finite squeezing.

Sweeps resource squeezing r for the package's actual four-step identity wire
(``photographiq.protocols.identity``, i.e. four Fourier teleportation steps
whose ideal composed map is identity) and measures how the output mean error
and added covariance noise shrink as r grows.

Exact functionality: ``gaussian_channel`` computes the unconditional affine
map (S, N, d) analytically for the fixed-angle pattern; per docs/theory.md an
L-step wire accumulates noise recursively and four k=0 steps implement
identity with noise 2*exp(-2r)*I. That closed form is reproduced here
independently (via ``photographiq.gaussian.wire_channel``, a separate
function from the pattern-level analyzer) as a second oracle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.gaussian import wire_channel  # noqa: E402

SQUEEZING_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
INPUT_ALPHA = 0.4 + 0.3j


def main():
    plt = common.setup_style()
    rows = []
    for r in SQUEEZING_VALUES:
        pattern = pg.protocols.identity(squeezing=r)
        output_node = pattern.outputs[0]
        channel = pg.gaussian_channel(pattern)

        # Independent closed-form oracle: four k=0 wire steps.
        closed_form_matrix, closed_form_noise = wire_channel([0.0] * 4, r)

        input_state = pg.GaussianInput.coherent(INPUT_ALPHA)
        applied = channel.apply(input_state.state(0))

        matrix_error = common.frobenius_error(channel.matrix, np.eye(2))
        oracle_matrix_error = common.frobenius_error(channel.matrix, closed_form_matrix)
        oracle_noise_error = common.frobenius_error(channel.noise, closed_form_noise)

        mean_error = common.frobenius_error(applied.mean, np.array([2 * INPUT_ALPHA.real, 2 * INPUT_ALPHA.imag]))
        added_noise_trace = float(np.trace(channel.noise))

        single = pg.simulate(pattern, backend="gaussian", inputs={0: input_state}, seed=7)
        _, q_var = single.state.quadrature(output_node)
        _, p_var = single.state.quadrature(output_node, np.pi / 2)

        rows.append(
            {
                "squeezing_r": r,
                "identity_matrix_error": matrix_error,
                "channel_vs_closed_form_matrix_error": oracle_matrix_error,
                "channel_vs_closed_form_noise_error": oracle_noise_error,
                "output_mean_error": mean_error,
                "added_variance_trace": added_noise_trace,
                "predicted_added_variance": 2 * np.exp(-2 * r),
                "output_q_variance": q_var,
                "output_p_variance": p_var,
                "log10_added_variance": common.safe_log10(added_noise_trace),
            }
        )

    tolerance = 1e-9
    failures = [r for r in rows if r["channel_vs_closed_form_matrix_error"] > tolerance
                or r["channel_vs_closed_form_noise_error"] > tolerance or r["identity_matrix_error"] > tolerance]
    if failures:
        common.save_csv(failures, "R2_identity_finite_squeezing_FAILURES")
        raise AssertionError("Identity wire channel disagreed with the closed-form oracle; see FAILURES csv")

    common.save_result(
        rows,
        "R2_identity_finite_squeezing",
        extra={"input_alpha": [INPUT_ALPHA.real, INPUT_ALPHA.imag], "protocol": "four-step identity wire"},
    )

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    rs = np.asarray(SQUEEZING_VALUES)
    ax.semilogy(rs, [r["added_variance_trace"] for r in rows], "o-", label="Measured Tr(N)", color="#24677b")
    ax.semilogy(rs, [r["predicted_added_variance"] for r in rows], "--", label=r"$2e^{-2r}$ (theory)", color="#c66d27")
    ax.set_xlabel("Resource squeezing r")
    ax.set_ylabel("Added output variance (vacuum units)")
    ax.legend()
    ax.set_title("Four-step identity wire: finite-squeezing noise")
    common.save_figure(fig, "R2_identity_finite_squeezing")
    plt.close(fig)

    common.print_summary(
        "R2 identity vs finite squeezing",
        squeezing_values=SQUEEZING_VALUES,
        max_channel_vs_closed_form_error=max(r["channel_vs_closed_form_matrix_error"] for r in rows),
        max_added_variance=max(r["added_variance_trace"] for r in rows),
        min_added_variance=min(r["added_variance_trace"] for r in rows),
    )


if __name__ == "__main__":
    main()
