"""Soft GKP decoding. See the matching documentation for physical limitations."""

import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

decoder = pg.SoftDecisionDecoder(code, "Z")
for raw in (-2.6, 0.1, 1.2, 2.6):
    result = decoder.decode(raw)
    print(result)
    assert abs(sum(result.probabilities) - 1) < 1e-12
    assert result.confidence == max(result.probabilities)
print("nearest-cell:", code.decode(1.2))
