# Logical gate-learning reproduction

Run `python experiments/ml/paper_reproduction/main.py` after installing the pinned
MentPy reference and the plotting dependencies (`.[ml-validation,ml-experiments]`).
`config.json` specifies seeds, split, optimizer and resources. Results retain every
training and test curve, parameters, reference errors and runtimes; rerunning overwrites
these outputs.

Targets follow Fig. 3: a Haar unitary on the first wire and IsingXX(pi/2).
The configured 20-seed run follows the paper's run count, but is not an exact
reproduction of its statistics or optimizer settings. MentPy verifies checkpoint
output density matrices but is not independently trained here. Finite-energy
GKP training is unsupported. No claim of quantum advantage is made.
