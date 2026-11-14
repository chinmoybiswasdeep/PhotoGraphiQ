**PhotoGraphiQML training.** A one-wire logical MuTA model (`photographiqml.MuTA(1)`,
4 parameters) trained with `Trainer(epochs=100)` to reproduce the
rotation exp(0.4 i X) on 7 Haar-random input states (`haar_states(1, 7, 42)`); the loss is
the mean infidelity, final value 1.94e-06. The curve is drawn by
`interactive.optimization_curves`. This is a logical (qubit-level) model; lowering continuous
angles to a physical GKP execution is a separate, explicitly bounded step
(`MuTA.physical_capabilities`) and is not shown here.

Source: `scripts/generate_demo_figures.py` (`13_photographiqml`).
