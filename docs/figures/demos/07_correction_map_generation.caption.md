**Correction-map generation.** (a) `flows.analyze` on a four-node CV wire
(`CVGraph.line(4, squeezing=1.5)`, homodyne shears k = 0.3, -0.5, 0): certified CV flow, one
blue arrow per (outcome, corrected node) pair of its correction map. (b) The same map as
displacement corrections in terms of the raw homodyne records m[k] (flow coefficients are
per rescaled record m / sin(theta); the overlay multiplies by sqrt(1 + k^2)): node 1: q += -1.044 m[0]; node 2: q += -1.118 m[1]; p += -1.044 m[0]; node 3: q += -1 m[2]; p += -1.118 m[1].
These equal the `Displace` commands of `cvflow.flow_pattern` for this order: True.
(c) Binary corrections of the qubit graph of demo 06 from its causal flow (X on the flow
successor, Z on its other neighbours; exponents are XORs of outcome bits): node 1: X^(m[0]); node 2: Z^(m[0]); node 5: X^(m[1]) Z^(m[0]); node 3: X^(m[2]); node 4: X^(m[3]) Z^(m[2]).
Formulas are shown only where they are read from a correction map or from executed commands.

Source: `scripts/generate_demo_figures.py` (`07_correction_map_generation`).
