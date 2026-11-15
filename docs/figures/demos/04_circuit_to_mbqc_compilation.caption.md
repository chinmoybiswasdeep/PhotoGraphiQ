**Circuit-to-MBQC compilation.** (a) `Circuit(2).rotate(0, 0.4).squeeze(1, 0.3).cz(0, 1,
0.8)` drawn by `draw_circuit`. (b) The weighted graph of `compile(squeezing=1.5,
return_trace=True)` (`GraphView.from_pattern`); each shaded box encloses the resource nodes
that the `CompilationTrace` attributes to one source gate
(gate 0 R: [2, 3, 4, 5]; gate 1 S: [6, 7, 8, 9, 10]; gate 2 cz: [11, 12, 13, 14, 15, 16, 17, 18]).
`pg.visualize_compilation` draws the same provenance with a spring layout. The compiled pattern has
19 modes, 18 CZ edges, 17 homodyne
measurements and 17 outcome-dependent corrections. Single-mode gates are
decomposed into four or five shear (wire) steps by `decompose_symplectic`; CZ adds four
identity steps per mode. Finite squeezing makes every step a noisy channel. The interactive
page shows the same graph with stage selection.

Source: `scripts/generate_demo_figures.py` (`04_circuit_to_mbqc_compilation`).
