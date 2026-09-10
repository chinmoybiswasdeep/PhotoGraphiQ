# Concepts without the algebra

A **mode** is an oscillator, labelled by an integer, string or another hashable
Python object. Its **quadratures** q and p play the roles of position and momentum.
An **input** carries your data; an **ancilla** supplies a resource for computation.

A **CVGraph** describes resource connectivity and squeezing. An edge of weight g
represents a controlled-Z interaction. A **Pattern** adds preparation, gates,
measurements and corrections in causal order. Measuring a mode destroys it in
these execution backends; the result becomes a classical record.

**Feed-forward** uses an earlier outcome to choose a correction or later angle.
Changing an angle using an outcome makes the pattern adaptive. A **Circuit** is a
gate-oriented description; compilation replaces its operations with MBQC gadgets.

A simulation returns one **conditional trajectory**. Repeating it gives an
ensemble. Those objects answer different questions: a single output is conditioned
on its records, whereas an unconditional channel averages over all records.

Begin with the Gaussian backend for large, low-cost covariance calculations.
Use a Fock backend for photon counting, cats, cubic gates and Kerr. Mixed-state
Fock execution is needed for lossy non-Gaussian trajectories. The total-photon
cutoff controls represented occupation space, not the physical resource squeezing.
