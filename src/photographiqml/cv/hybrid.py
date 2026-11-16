"""Optional hybrid Gaussian-Fock forward inference for non-Gaussian CV programs (experimental).

Executes ``NonGaussianProgram.pattern(...)`` on ``photographiq.hybrid.executor.run`` (per-mode
truncation, the CUDA-Q Dynamics representation) with the same Gaussian feature-map inputs as
``NonGaussianProgram.run``. Forward inference only: no loss, gradient or training path, and
no MuTA / GKP model is affected. ``engine="cudaq"`` needs CUDA-Q (Linux) and never falls back.
"""

from __future__ import annotations

import numpy as np

import photographiq as pg
import photographiq.cudaq_dynamics as cq
from photographiq.hybrid.executor import run


def hybrid_moments(program, params, x, feature_map, dims, modes, *, engine="numpy-reference", **kw):
    """(means, variances, retained weight, truncation diagnostics) of q, p on ``modes``.

    ``kw`` passes ``cubic`` / ``injection_squeezing`` to ``program.pattern`` and anything else
    (e.g. ``measurement_outcomes`` for injected cubic gates, ``seed``) to the executor.
    """
    mean, cov = feature_map.moments(x)
    inputs = {
        j: pg.GaussianInput(
            tuple(mean[2 * j : 2 * j + 2]),
            tuple(map(tuple, cov[2 * j : 2 * j + 2, 2 * j : 2 * j + 2])),
        )
        for j in range(program.modes)
    }
    pattern_kw = {k: kw.pop(k) for k in ("cubic", "injection_squeezing") if k in kw}
    res = run(
        program.pattern(params, x, **pattern_kw), dims=dims, inputs=inputs, engine=engine, **kw
    )
    if res.state is None:
        raise ValueError("The program has no surviving output modes")
    out = list(res.state.modes)
    means, variances = [], []
    for m in modes:
        k = out.index(m)
        for op in (cq.position(k), cq.momentum(k)):
            first = res.expectation(op).real
            means.append(first)
            variances.append(res.expectation(op * op).real - first**2)
    return np.array(means), np.array(variances), res.retained_weight, res.truncation
