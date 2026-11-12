# Classical feed-forward


```python
pattern = pg.protocols.adaptive(squeezing=0.8)
result = pg.simulate(pattern, parameters={"k": 0.3}, seed=10, frame=True)
print(result.outcomes)

from photographiq.expressions import CallableExpression

angle = CallableExpression(lambda records: 0.2 + records[0] ** 2, dependencies={0})
```

`Outcome(key)` refers to an earlier measurement or `Signal`. Heterodyne outcomes
need `Outcome(key, component=0)` or `component=1`. Expressions support arithmetic,
sin, cos, exp and atan2. A callable only receives its declared record subset and
cannot be serialized. Parameters are externally bound real scalars; there is no
string evaluation, and `simulate()` itself is not differentiable. Gradients with
respect to parameters come from the separate JAX path `photographiq.autodiff`,
which evaluates expression trees (not Python callables) on fixed, postselected
measurement branches; see [automatic differentiation](automatic-differentiation.md).
`Signal('name', expression)` stores
a new classical register, and `Result.records` contains both signals and outcomes.
Duplicate keys, missing producers, cycles and future dependencies are rejected.


See the [API reference](../api/index.md) and [tutorials](../tutorials/index.md) for executable examples.
