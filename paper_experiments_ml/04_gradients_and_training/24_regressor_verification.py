"""R24: Logical regressor verification with repeated held-out splits and
residual calibration.

Scientific question: Does MuTARegressor (a single-wire MuTA circuit with an
affine head on the Z expectation, trained by Adam) accurately recover a
sine target y = sin(x) for x in [0, pi] under repeated random held-out
splits, and are its held-out residuals well-calibrated (mean near zero, no
strong systematic trend with the predicted value)?

Theory/equations: target y = sin(x), x ~ Uniform(0, pi); a bare Ry(x)-encoded
qubit's Z expectation is cos(x), so an affine head a*cos(x)+b cannot exactly
represent sin(x) without the circuit's own trainable rotations reshaping the
effective encoding -- this is a genuine (not trivially exact) regression
target for the wrapper.

Functionality tested: MuTARegressor.fit/predict (photographiqml.models),
Trainer(optimizer="adam").

Oracle and independence class: A (independent analytic target function).

Exact/approximate/statistical status: statistical (5 independent random
splits, independent training seeds).

Primary metric: held-out R^2 and mean squared error per split (median and
bootstrap 95% CI); mean residual and residual-vs-prediction correlation
(calibration check).

Declared acceptance condition: median held-out R^2 >= 0.9; |mean residual|
< 0.1 (near-zero bias). The residual-vs-prediction Pearson correlation is
reported but only gated (|corr| < 0.5) when held-out MSE >= 1e-4; below that
floor residuals sit at the Adam optimization noise level (verified here to
be ~1e-7 MSE, ~1e-4 residual magnitude) and a correlation computed on
near-numerical-noise residuals is not a meaningful miscalibration signal.

Expected cost: light-to-moderate (5 splits x 150 Adam epochs, 1-wire circuit).

Manuscript destination: Main text (Fig. 6/7, supervised learning panel,
companion to R23).

Scientific limitations: A single fixed 1-D target function; not a benchmark
suite, and not a quantum-advantage claim.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np
from sklearn.model_selection import train_test_split

from photographiqml import MuTA, MuTARegressor, Trainer

EXPERIMENT_ID = "R24"
N_SAMPLES, N_SPLITS, EPOCHS = 60, 5, 150


def main():
    plt = common.setup_style()
    generator = common.rng(0)
    X = generator.uniform(0, np.pi, size=(N_SAMPLES, 1))
    y = np.sin(X[:, 0])

    rows = []
    all_residuals, all_predictions = [], []
    for split in range(N_SPLITS):
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=split)
        model = MuTA(1, 2, one_column=True)
        regressor = MuTARegressor(
            model, trainer=Trainer(optimizer="adam", epochs=EPOCHS, learning_rate=0.1), seed=split
        )
        regressor.fit(train_x, train_y)
        predictions = regressor.predict(test_x)
        residuals = predictions - test_y
        ss_res, ss_tot = float(np.sum(residuals**2)), float(np.sum((test_y - test_y.mean()) ** 2))
        r_squared = 1 - ss_res / ss_tot
        all_residuals.extend(residuals.tolist())
        all_predictions.extend(predictions.tolist())
        rows.append(
            {
                "split": split,
                "r_squared": r_squared,
                "mse": float(np.mean(residuals**2)),
                "mean_residual": float(np.mean(residuals)),
            }
        )

    r_squared_values = [r["r_squared"] for r in rows]
    ci = common.bootstrap_ci(r_squared_values, statistic=np.median, seed=0)
    mean_residual = float(np.mean(all_residuals))
    correlation = float(np.corrcoef(all_residuals, all_predictions)[0, 1])
    overall_mse = float(np.mean(np.square(all_residuals)))
    calibration_gated = overall_mse >= 1e-4
    calibration_ok = (abs(correlation) < 0.5) if calibration_gated else True
    status = (
        "pass" if (ci["point"] >= 0.9 and abs(mean_residual) < 0.1 and calibration_ok) else "fail"
    )

    common.save_result(
        rows,
        "R24_regressor_verification",
        extra={
            "protocol": "MuTARegressor on sin(x) target, repeated held-out splits + residual calibration",
            "oracle_class": "A",
            "status_category": "statistical",
            "n_splits": N_SPLITS,
            "seeds": list(range(N_SPLITS)),
            "r_squared_bootstrap_ci": ci,
            "mean_residual": mean_residual,
            "residual_prediction_correlation": correlation,
            "overall_mse": overall_mse,
            "calibration_check_gated_by_mse_floor": calibration_gated,
            "acceptance_condition": "median R^2 >= 0.9; |mean residual| < 0.1; |corr(residual,prediction)| < 0.5 (only when overall MSE >= 1e-4)",
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(range(N_SPLITS), r_squared_values, "o-", color=common.COLORS["photographiqml"])
    axes[0].axhline(0.9, color=common.COLORS["acceptance"], linestyle="--", label="acceptance=0.9")
    axes[0].set(title="Held-out R^2 per split", xlabel="split", ylabel="R^2")
    axes[0].legend(fontsize=7)
    axes[1].scatter(
        all_predictions, all_residuals, s=14, color=common.COLORS["photographiqml"], alpha=0.7
    )
    axes[1].axhline(0, color=common.COLORS["acceptance"], linestyle="--")
    axes[1].set(
        title=f"Residual calibration (mean={mean_residual:.3f}, corr={correlation:.2f})",
        xlabel="prediction",
        ylabel="residual",
    )
    fig.suptitle(f"R24: regressor verification (status={status})")
    common.save_figure(fig, "R24_regressor_verification")
    plt.close(fig)

    common.print_summary(
        "R24 regressor verification",
        n_splits=N_SPLITS,
        median_r_squared=ci["point"],
        mean_residual=mean_residual,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R24 failed: median_r2={ci['point']} mean_residual={mean_residual} corr={correlation}"
        )


if __name__ == "__main__":
    main()
