# References and provenance

1. Z. Kolarovszki et al., **Piquasso: A Photonic Quantum Computer Simulation
   Software Platform**, Quantum **9**, 1708 (2025).
   https://doi.org/10.22331/q-2025-04-15-1708.
   Supplied PDF: `papers/Kolarovszki et al. - 2025 - ...pdf`.
2. S. Sunami and M. Fukushima, **Graphix: optimizing and simulating
   measurement-based quantum computation on local-Clifford decorated graph**,
   arXiv:2212.11975. https://arxiv.org/abs/2212.11975.
   The supplied PDF is the 2022 v1 preprint, used for architectural context.
3. M. Gu, C. Weedbrook, N. C. Menicucci, T. C. Ralph and P. van Loock,
   **Quantum computing with continuous-variable clusters**, Phys. Rev. A **79**,
   062318 (2009). https://doi.org/10.1103/PhysRevA.79.062318.
4. N. C. Menicucci et al., **Universal Quantum Computation with
   Continuous-Variable Cluster States**, Phys. Rev. Lett. **97**, 110501 (2006).
   https://doi.org/10.1103/PhysRevLett.97.110501.
5. R. I. Booth and D. Markham, **Flow conditions for continuous variable
   measurement-based quantum computing**, Quantum **7**, 1146 (2023).
   https://doi.org/10.22331/q-2023-10-19-1146.
   Correction-matrix definitions 5–6 and linearity lemma 7 motivate `cvflow.py`.
6. S. L. Braunstein and H. J. Kimble, **Teleportation of Continuous Quantum
   Variables**, Phys. Rev. Lett. **80**, 869 (1998).
   https://doi.org/10.1103/PhysRevLett.80.869.

Documentation inspected on 2026-09-10:
https://piquasso.readthedocs.io/en/stable/index.html and
https://graphix.readthedocs.io/en/latest/index.html.
Exact upstream commits and inspected modules are listed in [design.md](design.md).
Source checkouts are local research aids under ignored `.research/`, not copied
into the distributed package. Original code is MIT; the supplied PDFs and
`paper/quantumarticle.cls` retain their upstream licensing.

The official class was downloaded from
https://github.com/quantum-journal/quantum-journal/blob/master/quantumarticle.cls.
Quantum author instructions: https://quantum-journal.org/instructions/authors/.


Non-Gaussian extension references:

* P. Marek, R. Filip and A. Furusawa, Deterministic implementation of weak quantum
  cubic nonlinearity, Physical Review A 84, 053802 (2011).
  https://doi.org/10.1103/PhysRevA.84.053802
* K. Miyata et al., Implementation of a quantum cubic gate by adaptive
  non-Gaussian measurement, Physical Review A 93, 022301 (2016).
  https://doi.org/10.1103/PhysRevA.93.022301

These motivate resource/adaptive approaches. The PhotoGraphiQ finite-envelope
injection map is derived explicitly and is not claimed to reproduce all details
of either experimental proposal.
