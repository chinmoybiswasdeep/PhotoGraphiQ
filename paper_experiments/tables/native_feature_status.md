| feature | native_gaussian | native_fock | native_mixed_fock | gpu_cupy | suite_evidence |
|---|---|---|---|---|---|
| Gaussian states, gates, loss | V | V | V | V (batched) | R43,R45,R47,R58 |
| Homodyne / heterodyne conditioning | V | V (homodyne) | V (homodyne) | V (homodyne, batched) | R43,R45,R47,R58 |
| Adaptive feed-forward patterns | V | V | V | - | R1-R14,R43,R52,R60 |
| Non-Gaussian gates (Kerr, cubic) | - | V | V | E (Fock P U P) | R22-R25,R45,R46 |
| Photon counting | - | V | V | - | R19,R20,R45,R47 |
| GKP resources | - | E | E | - | R26-R28,R59 |
| Boson sampling / GBS kernels | n/a | n/a | n/a | - (measured, not justified) | R48-R50 |
| CUDA-Q execution | - | - | - | - | none (adapter translation only; execution unverified) |
