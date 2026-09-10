import numpy as np

import photographiq as pg


def test_standardization_and_frame_equivalence():
    p = pg.Pattern().extend(
        [pg.Prepare(0, 0.4), pg.Displace(0, 0.2, 0.3), pg.Prepare(1, 0.5), pg.Entangle(0, 1, 0.7)]
    )
    p.measure(0, pg.Homodyne.p())
    p.extend(
        [
            pg.Displace(1, pg.Outcome(0), 0.1),
            pg.Displace(1, 0.3, -0.1),
            pg.Rotate(1, 0.4),
            pg.Squeeze(1, 0.2),
        ]
    )
    results = [
        pg.simulate(q, seed=15, frame=frame)
        for q, frame in [(p, False), (p.standardize(), False), (p, True)]
    ]
    for result in results[1:]:
        np.testing.assert_allclose(result.state.mean, results[0].state.mean, atol=1e-11)
        np.testing.assert_allclose(result.state.covariance, results[0].state.covariance, atol=1e-11)
    assert results[-1].physical_displacements < results[0].physical_displacements


def test_callable_corrections_remain_valid_under_standardization():
    p = pg.Pattern(pg.CVGraph.line(2, squeezing=0.4)).measure(0)
    p.displace(1, q=pg.CallableExpression(lambda records: records[0] ** 2, {0}))
    p.displace(1, q=0.2)
    a = pg.simulate(p, seed=8)
    b = pg.simulate(p.standardize(), seed=8)
    np.testing.assert_allclose(a.state.mean, b.state.mean, atol=1e-12)
