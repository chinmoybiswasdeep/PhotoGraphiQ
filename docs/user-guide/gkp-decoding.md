# Decoders and classical frames

```python
import photographiq as pg
code = pg.GKPCode()
print(code.decode(-2.6))
soft = pg.SoftDecisionDecoder(code, "Z", prior=(0.5, 0.5))
print(soft.decode(-2.6))
frame = pg.LogicalPauliFrame(1, 0).hadamard()
measurement = pg.PhysicalGKPReadout("X", frame=frame)
```

Nearest-cell returns no confidence; soft decoding supplies a posterior under its
stated zero/one or plus/minus ensemble. It is not a universal decoder for arbitrary
logical states. Residual always describes the raw nearest cell, even when a frame
or posterior changes the output bit. `frame_correction` records a label flip.

Use a `CallableExpression` to read `.bit` into classical `Signal` feed-forward.
`Outcome("key")` arithmetic expects a scalar and should not be applied directly
to a structured decode result. Frames compose modulo global phase and propagate
through H, S and CZ; their ideal logical meaning does not remove physical envelope
distortion. See [decoder theory](../theory/gkp-decoding.md).
