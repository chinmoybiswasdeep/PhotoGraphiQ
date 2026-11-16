"""Physical arbitrary-angle GKP logical XY measurement, alone and inside a PhysicalMuTA.

Measures M_L(a) = cos(a) X_L + sin(a) Y_L at the non-Clifford angle a = pi/3 on one
finite-energy GKP mode with pg.GKPLogicalXYMeasurement (repeat-until-success magic-state
injection), then runs a one-wire PhysicalMuTA with measurement_family="XY".

The resource is deliberately wide (peak_width = envelope = 0.6) so the example runs in
seconds; its decoded statistics carry the finite-energy error of that resource and are
not converged. Conventions: hbar = 2, total-photon Fock cutoff.
"""

import warnings

import numpy as np

import photographiq as pg
from photographiqml import GKPPhysicalConfig, PhysicalMuTA, compare_logical_physical

warnings.simplefilter("ignore", RuntimeWarning)

CUTOFF = 56
ANGLE = np.pi / 3
LOGICAL = (np.cos(0.4), np.exp(0.9j) * np.sin(0.4))  # a generic logical input state

# --- 1. One GKP mode measured at a non-Clifford angle -------------------------------------
code = pg.GKPCode(cutoff=CUTOFF, peak_width=0.6, envelope=0.6, peaks=6, grid_points=2049)
measurement = code.logical_measurement("XY", alpha=ANGLE, protocol="auto")  # max_rounds=4
pattern = pg.Pattern(inputs=("d",))
pattern.append(pg.Measure("d", measurement, "m"))
pattern.append(pg.Output(()))
options = {"backend": "native-fock", "cutoff": CUTOFF, "inputs": {"d": code.encode(*LOGICAL)}}

ideal = pg.IdealLogicalXYMeasurement(ANGLE).probabilities(np.array(LOGICAL))
print(f"ideal qubit P(+1), P(-1) at a = pi/3: {ideal[0]:.4f}, {ideal[1]:.4f}")

record = pg.simulate(pattern, seed=1, **options).records["m"]
print("\none sampled trajectory")
print("  bit (0 <-> +1):        ", record.bit)
print(
    "  rounds:                ",
    [(round(r["target_phase"], 4), round(r["raw_outcome"], 4), r["bit"]) for r in record.rounds],
)
print("  heralded_success:      ", record.heralded_success)
print("  applied_operations:    ", record.applied_operations)
print("  readout_probabilities: ", tuple(round(p, 4) for p in record.readout_probabilities))
print("  final p outcome, cell: ", round(record.raw_outcome, 4), record.cell)

shots = pg.run_shots(pattern, 8, seed=7, **options)
records = [t.records["m"] for t in shots.trajectories]
rao_blackwell = np.mean([r.readout_probabilities for r in records], axis=0)
print("\n8 shots")
print("  bits:                  ", [r.bit for r in records])
print("  injection rounds used: ", [len(r.rounds) for r in records])
print("  all heralded success:  ", all(r.heralded_success for r in records))
print(f"  mean readout P(+1):     {rao_blackwell[0]:.4f}  (finite-energy, unconverged)")

# Postselection fixes the analog record: one q value per injection round, then the data p.
fixed = pg.simulate(
    pattern, seed=0, measurement_outcomes={"m": {"ancilla": [2.4, 0.1], "data": 0.3}}, **options
).records["m"]
print("\npostselected record {'ancilla': [2.4, 0.1], 'data': 0.3}")
print("  ancilla bits:          ", [r["bit"] for r in fixed.rounds])
print("  bit, joint_density:    ", fixed.bit, f"{fixed.joint_density:.4g}")

# --- 2. A one-wire PhysicalMuTA with the XY measurement family ----------------------------
config = GKPPhysicalConfig(
    cutoff=CUTOFF,
    peak_width=0.6,
    envelope=0.6,
    peaks=6,
    grid_points=2049,
    backend="native-fock",
    xy_protocol="auto",
    max_dimension=200_000,
)
model = PhysicalMuTA(1, 1, one_column=True, measurement_family="XY", physical_config=config)
angles = dict(zip(model.parameters(), (0.0, 0.0, np.pi / 4, 0.0)))
audit = model.physical_capabilities(angles, config=config)
print("\nPhysicalMuTA audit")
print(
    "  supported, peak modes, dimension:",
    audit["supported"],
    audit["peak_live_modes"],
    audit["hilbert_dimension"],
)
print(
    "  ancilla rounds per node:        ",
    [row["ancilla_rounds_max"] for row in audit["angle_audit"]],
)
result = model.run([1, 0], angles, mode="physical-shots", shots=3, seed=2)
print(
    "  decoded P(z):                   ",
    {k: round(v, 4) for k, v in result.decoded_joint_probabilities.items()},
)
print(
    "  rounds per node and shot:       ",
    [[len(r.rounds) for r in shot.values()] for shot in result.measurement_records],
)
# Three shots of a wide resource: the distance is dominated by trajectory spread and
# finite-energy error, so it is an execution check, not a performance number.
print(
    "  TV distance to logical target:  ",
    round(compare_logical_physical(result)["total_variation_distance"], 4),
)
