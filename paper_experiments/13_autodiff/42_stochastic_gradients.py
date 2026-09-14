"""R42: Conditional / estimator gradients (experimental, statistical estimators).

Tests the two generic Monte Carlo gradient-estimator primitives in
``photographiq.autodiff`` against known analytical identities:

1. ``score_function_surrogate`` (REINFORCE/likelihood-ratio estimator) on a
   parametrized Bernoulli branch, compared to the exact analytic gradient of
   E[f(X)] for X ~ Bernoulli(sigmoid(theta)).
2. ``gaussian_sample`` (pathwise/reparameterization primitive) on a simple
   quadratic objective, compared to the exact analytic pathwise gradient.

Both are Monte Carlo estimators: agreement is statistical (bias and standard
error over repeated seeds), not exact-precision agreement. This is
explicitly experimental functionality (docs/validation/feature-matrix.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

if not common.HAS_JAX:
    print("SKIPPED: jax is not installed; see requirements_notes.md")
    sys.exit(0)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from photographiq.autodiff import gaussian_sample, score_function_surrogate  # noqa: E402

N_SAMPLES = 20000
N_REPEATS = 10
THETA0 = 0.3
F_ONE, F_ZERO = 2.0, -1.0  # values attached to the two Bernoulli branches


def score_function_case():
    def analytic_gradient(theta):
        p = jax.nn.sigmoid(theta)
        return (F_ONE - F_ZERO) * p * (1 - p)

    exact_gradient = float(analytic_gradient(THETA0))

    rows = []
    for repeat in range(N_REPEATS):
        rng = np.random.default_rng(1000 + repeat)
        p0 = float(jax.nn.sigmoid(THETA0))
        outcomes = rng.random(N_SAMPLES) < p0  # samples from Bernoulli(p(theta0))
        values = jnp.where(jnp.asarray(outcomes), F_ONE, F_ZERO)

        def loss(theta, outcomes=outcomes, values=values):
            p = jax.nn.sigmoid(theta)
            log_probs = jnp.where(jnp.asarray(outcomes), jnp.log(p), jnp.log1p(-p))
            return score_function_surrogate(values, log_probs)

        estimate = float(jax.grad(loss)(jnp.asarray(THETA0)))
        rows.append(estimate)

    rows = np.array(rows)
    return {
        "estimator": "score_function_surrogate",
        "exact_gradient": exact_gradient,
        "mean_estimate": float(rows.mean()),
        "bias": float(rows.mean() - exact_gradient),
        "standard_error": float(rows.std(ddof=1) / np.sqrt(N_REPEATS)),
        "n_samples_per_repeat": N_SAMPLES,
        "n_repeats": N_REPEATS,
    }


def pathwise_case():
    sigma = 0.7

    def analytic_gradient(theta):
        return 2.0 * theta  # d/dtheta E_z[(theta+sigma*z)^2] = 2*theta since E[z]=0

    exact_gradient = float(analytic_gradient(THETA0))

    rows = []
    for repeat in range(N_REPEATS):
        rng = np.random.default_rng(2000 + repeat)
        z_batch = rng.standard_normal((N_SAMPLES, 1))

        def loss(theta, z_batch=z_batch):
            mean = jnp.array([theta])
            covariance = jnp.array([[sigma**2]])
            samples = jax.vmap(lambda z: gaussian_sample(mean, covariance, z))(jnp.asarray(z_batch))
            return jnp.mean(samples[:, 0] ** 2)

        estimate = float(jax.grad(loss)(THETA0))
        rows.append(estimate)

    rows = np.array(rows)
    return {
        "estimator": "gaussian_sample (pathwise)",
        "exact_gradient": exact_gradient,
        "mean_estimate": float(rows.mean()),
        "bias": float(rows.mean() - exact_gradient),
        "standard_error": float(rows.std(ddof=1) / np.sqrt(N_REPEATS)),
        "n_samples_per_repeat": N_SAMPLES,
        "n_repeats": N_REPEATS,
    }


def main():
    plt = common.setup_style()
    results = [score_function_case(), pathwise_case()]

    # A statistically consistent unbiased estimator should have |bias| within
    # a few standard errors of zero; this is not an exact-precision check.
    failures = [r for r in results if abs(r["bias"]) > 5 * r["standard_error"] + 1e-6]
    if failures:
        common.save_json(failures, "R42_stochastic_gradients_FAILURES")
        raise AssertionError("An estimator's bias exceeded its statistical tolerance; see FAILURES json")

    common.save_json({"results": results}, "R42_stochastic_gradients")
    common.write_metadata("R42_stochastic_gradients")

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    names = [r["estimator"] for r in results]
    ax.errorbar(
        range(len(results)), [r["mean_estimate"] for r in results],
        yerr=[5 * r["standard_error"] for r in results], fmt="o", capsize=4, color="#24677b", label="Estimate +/- 5 SE",
    )
    ax.plot(range(len(results)), [r["exact_gradient"] for r in results], "x", markersize=10, color="#c66d27", label="Exact gradient")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Gradient estimate")
    ax.legend(fontsize=8)
    ax.set_title("Stochastic gradient estimators (experimental)")
    common.save_figure(fig, "R42_stochastic_gradients")
    plt.close(fig)

    common.print_summary("R42 conditional / estimator gradients", results=results)


if __name__ == "__main__":
    main()
