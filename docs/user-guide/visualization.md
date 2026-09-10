# Visualization and export

Install the visualization extra. Drawing imports matplotlib only when requested.
`graph.draw` is not required: use `draw_graph(graph)` from the visualization module.
`pattern.draw()` and `circuit.draw()` return Axes; `visualize_compilation` returns
a Figure. Export through `output="name.svg"` or `ax.figure.savefig(...)`.

Graph/pattern layouts support spring, circular, grid and a complete manual
`positions` mapping. `labels=False` reduces clutter. Styles are `default` and
`monochrome`. Input nodes are square, surviving outputs diamond-shaped and PNR
nodes triangular; Gaussian measurements show angles and keys. Dashed arrows show
classical outcome/signal dependencies. Order labels refer to command indices.

`draw_commands(pattern)` displays the physical optical instruction sequence,
including preparations, CP, Kerr and measurement glyphs. Label lanes continue
across the drawing for readability; a line after destructive measurement does
not imply that the quantum mode survives.

Compilation figures color gates and generated resource nodes consistently and
list measurement/correction indices. Retain the trace when serializing a pattern;
modified circuits or patterns are rejected rather than given a false mapping.
Overview figures are limited to 24 gates and 150 nodes; inspect synthesis blocks
separately beyond that size.

Connectivity aggregates repeated CZ edges and cannot describe the complete
temporal semantics of a pattern containing interleaved gates and injections.
Use the command sequence and `draw_dependencies(pattern)` alongside it. SVG and
PDF preserve vector geometry; PNG is useful for slide previews.
