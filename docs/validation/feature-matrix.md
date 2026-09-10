# Feature matrix

S = supported core path; E = experimental implementation; — = unsupported.
Validation refers to tested regimes, not a proof for every input or parameter.

| Feature | NumPy Gaussian | Piquasso Gaussian | Pure Fock | Mixed Fock | Evidence |
|---|---|---|---|---|---|
| Homodyne | S | S | E | E | analytical conditioning, pure/mixed comparison |
| Noisy homodyne | S | S | — | E | Gaussian moments, convolution density |
| Heterodyne/general-dyne | S | S | — | — | Gaussian analytical references |
| PNR | — | — | E | E | Born probabilities and conditional states |
| CZ, rotation, squeezing | S | S | E | E | matrices, raw native references |
| Cubic/Kerr execution | — | — | E | E | independent Fock references |
| Cat resources | — | — | E | E | parity and Wigner checks |
| Photon subtraction | — | — | E | E | ladder factors; physical tap probabilities |
| Loss/thermal noise | S | S | — | E | analytical moments/binomial loss |
| GKP resources | — | — | E | E | projection/grid checks; no threshold claim |
| Gaussian compilation | S | S | E | E | channel and structural tests |
| Cubic injection/Kerr synthesis | — | — | E | E | finite-resource and synthesis refinement |

JAX is a separate experimental execution API for dense Fock generators, ideal
fixed-outcome homodyne/PNR and vacuum loss. It is not an automatically selected
backend for `simulate`. Arbitrary-order flow search, general POVMs, physical
hardware execution and a fault-tolerant GKP architecture remain unsupported.
