import numpy as np

import photographiq as pg


def test_large_states_have_compact_representations():
    state = pg.GaussianState(np.zeros(100), np.eye(100), tuple(range(50)))
    assert len(repr(state)) < 180
    assert len(repr(pg.FockInput.number(100))) < 80
    pattern = pg.Pattern(pg.CVGraph.line(50))
    assert len(repr(pattern)) < 80
    assert len(repr(pattern.graph)) < 80
