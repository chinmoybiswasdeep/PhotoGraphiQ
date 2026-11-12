"""Bounded categorical search, separate from continuous logical optimizers."""

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
