# R60-R67: physical arbitrary-angle GKP logical measurements (extension catalogue)

These studies validate `pg.GKPLogicalXYMeasurement` and its use in `PhotoGraphiQML`
(`PhysicalMuTA(measurement_family="XY")`). They are an **extension catalogue**: the canonical
evidence contract of this suite (`publication.EXPECTED_IDS`, `EXPERIMENT_INDEX.md`,
`EXPERIMENT_CONTRACTS.json`, `EXPERIMENT_MANIFEST.json`, the generated notebook and `STATUS.md`)
still covers R1-R59 and R_PERF only. Adding R60-R67 there requires a publication run that
regenerates contracts, manifest, notebook and status together; until then these scripts are run
individually and write the usual CSV/JSON/metadata and figures under `../results` and
`../figures` with an `R6x_` prefix.

Design and protocol: [docs/development/GKP_ARBITRARY_ANGLE_IMPLEMENTATION.md](../../docs/development/GKP_ARBITRARY_ANGLE_IMPLEMENTATION.md).
Results and classification: [docs/development/GKP_ARBITRARY_ANGLE_FINAL_REPORT.md](../../docs/development/GKP_ARBITRARY_ANGLE_FINAL_REPORT.md).

Every study separates three levels: the ideal qubit, the untruncated finite-energy grid
reference (`tests/validation/gkp_xy_reference.py`), and the Fock engine (system under test).
Ideal-vs-reference is physical finite-energy error, reference-vs-engine is Fock truncation, and
shot spread is sampling error.

| ID | Script | Question | Reference / oracle | Acceptance (declared in the script) | Cost |
|---|---|---|---|---|---|
| R60 | `60_physical_y_measurement.py` | Physical Y on the six cardinal states (Clifford shear and Y-state injection), three resources | grid (B), ideal (A) | engine-grid within 2e-3 (+4 SE); Y-eigenstate finite-energy error decreases with quality | ~10 min |
| R61 | `61_arbitrary_angle_probabilities.py` | P(+) over 0 <= alpha < 2 pi; heralded vs unconditional | grid (B), ideal (A) | engine-grid within 4 SE + 2e-3 at 8 angles; heralded error decreases broad -> sharp, <= 0.03 at sharp; heralding 1 - 2^-4 | ~20 min |
| R62 | `62_finite_energy_convergence.py` | Width, envelope, peaks (physical); Fock cutoff (numerical); ancilla loss (physical noise, mixed Fock) | grid (B) | largest cutoff matches grid; eta = 1 matches noiseless; loss degrades P(+) | ~15 min |
| R63 | `63_measurement_back_action.py` | Partner state after measuring half of a GKP Bell pair | grid operator map (B), ideal (A) | engine-grid within 4 SE + 0.01 per Bloch component; grid direction within 0.2 rad of ideal | ~20 min |
| R64 | `64_adaptive_mbqc.py` | Previous outcome changes the next physical instruction | `contract_branch` (A) | ideal frame-tracked branches exact; recorded angle (-1)^x a; adaptive beats non-adaptive in the x = 1 branch | ~5 min |
| R65 | `65_physical_muta_execution.py` | Two-layer PhysicalMuTA with non-Pauli angles vs logical target | logical MuTA (N/A, descriptive) | valid output, heralded >= 0.8, medium not worse than broad (2 SE) | ~20 min |
| R66 | `66_physical_muta_learning.py` | Optimise one continuous angle on the physical simulator | exact logical loss (A) | found angle within 0.05 of the logical optimum loss; FD gradient signs; improvement at fresh seed | ~35 min |
| R67 | `67_resource_scaling.py` | Runtime, memory, energy growth per failed round, heralding vs round budget | grid (A for heralding) | grid heralding = 1 - 2^-K; engine within 4 SE | ~15 min |

Run one study: `python 60_physical_y_measurement.py` from this folder (native Fock engines
required; MentPy is not needed). Timing results are provisional unless `PGQ_IDLE_MACHINE=1`.
