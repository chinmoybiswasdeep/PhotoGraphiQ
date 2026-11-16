"""Physical MuTA training: categorical {0, pi} search and continuous XY-angle tools.

DiscreteSearch serves the signed-X family. physical_loss, physical_gradient and
ContinuousPhysicalSearch serve measurement_family="XY" (finite-energy sampled objectives with
common random numbers); none of them is a logical optimizer or assumes a parameter-shift rule.
"""

from dataclasses import dataclass

import numpy as np

from .lowering import physical_capabilities, require_supported


@dataclass
class DiscreteSearchResult:
    parameters: np.ndarray
    loss: float
    evaluations: list
    seed: int | None


class DiscreteSearch:
    """Coordinate search over {0, pi}; never round an arbitrary trained solution.

    The caller supplies a physical objective with explicit shots/seeds and
    resource settings. Reusing seeds helps comparisons but does not remove
    estimator noise. Revalidate the selected candidate at independent seeds.
    """

    def __init__(self, sweeps=1, seed=0):
        if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 0:
            raise ValueError("sweeps must be a nonnegative integer")
        self.sweeps, self.seed = sweeps, seed

    def fit(self, model, objective, initial=None):
        from .physical import PhysicalMuTA

        if not isinstance(model, PhysicalMuTA):
            raise ValueError("Discrete physical search requires PhysicalMuTA")
        point = np.array(
            list(model.initialize(self.seed).values()) if initial is None else initial, dtype=float
        )
        if point.shape != (model.n_parameters,) or not np.isin(point, [0, np.pi]).all():
            raise ValueError("Initial search vector must contain exact categorical 0/pi choices")
        require_supported(physical_capabilities(model, point, config=model.physical_config))
        evaluations = []

        def evaluate(values):
            loss = float(objective(values.copy()))
            if not np.isfinite(loss):
                raise ValueError("Discrete objective must be finite")
            evaluations.append({"parameters": values.tolist(), "loss": loss})
            return loss

        best = evaluate(point)
        for _ in range(self.sweeps):
            for index in range(len(point)):
                candidate = point.copy()
                candidate[index] = np.pi if point[index] == 0 else 0
                loss = evaluate(candidate)
                if loss < best:
                    point, best = candidate, loss
        return DiscreteSearchResult(point, best, evaluations, self.seed)


@dataclass
class PhysicalLossEstimate:
    """Finite-energy physical loss sum_b w_b P(b) and its shot standard error."""

    value: float
    standard_error: float
    per_shot: np.ndarray
    probabilities: dict
    shots: int
    seed: int


def _require_continuous(model):
    from .physical import PhysicalMuTA

    if not isinstance(model, PhysicalMuTA) or model.measurement_family != "XY":
        raise ValueError(
            "Continuous physical training requires PhysicalMuTA(measurement_family='XY')"
        )


def physical_loss(model, input_state, parameters, weights, *, shots=16, seed=0, output_basis="Z"):
    """Physical-shots loss, linear in the decoded output distribution.

    ``weights`` maps output bit tuples to real weights (missing labels weigh zero), so the loss
    is the expectation of a classical function of the decoded output. Each trajectory's
    conditional joint POVM row gives one unbiased per-shot value (Rao-Blackwell estimator);
    the standard error is their spread. Reusing ``seed`` across parameter points gives common
    random numbers. This is a sampled finite-energy loss, never the ideal logical loss.
    """
    _require_continuous(model)
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 2:
        raise ValueError("physical_loss needs at least two shots for a standard error")
    result = model.run(
        input_state,
        parameters,
        mode="physical-shots",
        shots=shots,
        seed=seed,
        output_basis=output_basis,
    )
    labels = list(result.decoded_joint_probabilities)
    vector = np.array([float(weights.get(label, 0.0)) for label in labels])
    per_shot = np.asarray(result.diagnostics["per_shot_joint_probabilities"]) @ vector
    return PhysicalLossEstimate(
        float(per_shot.mean()),
        float(per_shot.std(ddof=1) / np.sqrt(shots)),
        per_shot,
        dict(result.decoded_joint_probabilities),
        shots,
        seed,
    )


def physical_gradient(
    model, input_state, parameters, name, weights, *, step=0.1, shots=16, seed=0, output_basis="Z"
):
    """Central finite difference of the physical loss in one angle, with paired shot SE.

    Both points use the same seed (common random numbers), so the per-shot differences are
    paired and their spread is the gradient's standard error. The physical instrument's
    angle dependence is not the ideal qubit's, so no parameter-shift rule is assumed.
    Returns (gradient, standard_error).
    """
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    base = dict(model._parameters.bind(parameters))
    options = {"shots": shots, "seed": seed, "output_basis": output_basis}
    plus = physical_loss(model, input_state, base | {name: base[name] + step}, weights, **options)
    minus = physical_loss(model, input_state, base | {name: base[name] - step}, weights, **options)
    differences = (plus.per_shot - minus.per_shot) / (2 * step)
    return float(differences.mean()), float(differences.std(ddof=1) / np.sqrt(shots))


@dataclass
class ContinuousSearchResult:
    angle: float
    loss: PhysicalLossEstimate
    evaluations: list


class ContinuousPhysicalSearch:
    """Bounded one-angle minimisation of the physical loss (other angles fixed).

    Brent's bounded method on a sampled objective with common random numbers. The returned
    angle is a finite-energy optimum for this seed and resource; revalidate it at independent
    seeds. Optimisation success is not evidence of any quantum learning advantage.
    """

    def __init__(self, shots=16, seed=0, maxiter=12, xatol=0.05):
        if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        self.shots, self.seed, self.maxiter, self.xatol = shots, seed, maxiter, xatol

    def fit(self, model, input_state, parameters, name, weights, bounds=(0.0, 2 * np.pi)):
        from scipy.optimize import minimize_scalar

        _require_continuous(model)
        base = dict(model._parameters.bind(parameters))
        evaluations = []

        def objective(angle):
            estimate = physical_loss(
                model,
                input_state,
                base | {name: float(angle)},
                weights,
                shots=self.shots,
                seed=self.seed,
            )
            evaluations.append(
                {"angle": float(angle), "loss": estimate.value, "se": estimate.standard_error}
            )
            return estimate.value

        found = minimize_scalar(
            objective,
            bounds=bounds,
            method="bounded",
            options={"maxiter": self.maxiter, "xatol": self.xatol},
        )
        final = physical_loss(
            model,
            input_state,
            base | {name: float(found.x)},
            weights,
            shots=self.shots,
            seed=self.seed,
        )
        return ContinuousSearchResult(float(found.x), final, evaluations)
