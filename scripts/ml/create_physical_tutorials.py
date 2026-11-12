"""Generate executable physical tutorials and capture their actual stdout."""

import contextlib
import io
import subprocess
import sys
from pathlib import Path

# Repository-root anchored, so running from any cwd only touches the PhotoGraphiQML trees.
ROOT = Path(__file__).resolve().parents[2]

CONFIG = """from photographiqml import GKPPhysicalConfig, PhysicalMuTA
config = GKPPhysicalConfig(cutoff=24, peak_width=.9, envelope=.9,
                           peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
"""
CONDITIONAL = """result = model.run([1, 0], mode="physical-conditional",
                   analog_outcomes=dict.fromkeys(model.measurement_order, 0.0))
"""
READOUT = """import photographiq as pg
code = pg.GKPCode(cutoff=24, peak_width=.9, envelope=.9, peaks=4, grid_points=1025)
pattern = pg.Pattern(inputs=(0,))
pattern.append(pg.Measure(0, code.logical_measurement(BASIS), "readout"))
result = pg.simulate(pattern, inputs={0: PREPARATION}, cutoff=24,
                     backend="piquasso-fock", measurement_outcomes={"readout": .2})
record = result.records["readout"]
assert record.raw_outcome == .2 and record.bit in (0, 1)
assert record.confidence is None
print(record)
"""

TUTORIALS = [
    (
        31,
        "PhotoGraphiQ 0.3.1 encoded interface",
        "Check public encoded primitives before building a physical model.",
        "Finite codewords overlap; normalized encoding is not an exact isometry.",
        """import numpy as np
from photographiqml import GKPBridge
from photographiqml.lowering import check_photographiq_contract
print(check_photographiq_contract())
bridge = GKPBridge(cutoff=24, peak_width=.9, envelope=.9, peaks=4, grid_points=1025)
assert np.isclose(np.linalg.norm(bridge.encode([1, 0]).amplitudes), 1)
print(bridge.diagnostics())
""",
        "The normalized vector is a one-mode finite Fock resource.",
        "Preparation alone does not validate an MBQC computation.",
    ),
    (
        32,
        "Physical GKP X measurement",
        "Execute p readout on a finite GKP plus state.",
        "Logical X is a physical p-homodyne instrument with modular decoding.",
        READOUT.replace("BASIS", '"X"').replace("PREPARATION", "code.plus()"),
        "The .2 value is a conditional analog outcome, not a logical bit.",
        "One analog branch has no shot standard error or convergence certificate.",
    ),
    (
        33,
        "Physical GKP Z measurement",
        "Execute q readout on a finite GKP zero state.",
        "Logical Z is a physical q-homodyne instrument with modular decoding.",
        READOUT.replace("BASIS", '"Z"').replace("PREPARATION", "code.zero()"),
        "Z is an output/resource measurement; it is not an intermediate MuTA XY angle.",
        "Hard decoding leaves posterior confidence unknown.",
    ),
    (
        34,
        "Restricted physical MuTA",
        "Inspect signed-X lowering and categorical parameters.",
        "Every bound angle is audited before finite-resource construction.",
        CONFIG
        + """from photographiqml import lower_muta_to_gkp
angles = model.initialize(seed=4)
lowering = lower_muta_to_gkp(model, angles, config=config)
lowering.pattern.validate()
assert lowering.audit["supported"]
print(angles)
print({k: lowering.audit[k] for k in ("peak_live_modes", "hilbert_dimension", "cz_operations")})
""",
        "The lowering is an executable Pattern with explicit mode and signal mappings.",
        "This categorical family does not implement arbitrary-angle MuTA.",
    ),
    (
        35,
        "Zero-angle physical MuTA",
        "Run a complete finite-energy conditional chain.",
        "A zero-angle logical identity can still have finite-resource decoded errors.",
        CONFIG
        + CONDITIONAL
        + """assert result.representation == "gkp-physical"
assert not result.convergence["certified"]
print(result.decoded_joint_probabilities)
print(result.diagnostics["marginal_code_subspace_leakage"])
""",
        "All zero analog outcomes specify one physical conditional trajectory.",
        "These statistics are not the unconditional identity-channel error.",
    ),
    (
        36,
        "Pauli-frame propagation",
        "Inspect virtual corrections from flow records.",
        "An X readout anticommutes with the Z component of its logical frame.",
        CONFIG
        + """from photographiqml.lowering import node_frame
records = {( "bit", v): int(i % 2) for i, v in enumerate(model.measurement_order)}
frame = node_frame(model, model.output_nodes[0], records)
assert frame.correction("X") == frame.z_bit
assert frame.correction("Z") == frame.x_bit
print(frame)
""",
        "Frames alter interpretation without a physical analog displacement.",
        "This algebra example supplies interpreted bits; real runs obtain them from the decoder.",
    ),
    (
        37,
        "Hard vs soft decoding",
        "Compare unknown hard confidence to a calibrated posterior.",
        "The soft model discriminates a specified preparation ensemble.",
        """from photographiqml import GKPBridge
bridge = GKPBridge(cutoff=24, peak_width=.9, envelope=.9, peaks=4, grid_points=1025)
hard = bridge.resource_readout().decoder.decode(.2)
soft = bridge.resource_readout(decoder="soft").decoder.decode(.2)
assert hard.confidence is None and soft.confidence is not None
assert abs(sum(soft.probabilities) - 1) < 1e-12
print("hard confidence:", hard.confidence)
print("preparation posterior:", soft.probabilities)
""",
        "Soft values concern equal-prior zero/one preparation labels.",
        "They are not calibrated for arbitrary adaptive MuTA node states.",
    ),
    (
        38,
        "Multi-mode joint readout",
        "Retain correlations of two physical outputs.",
        "A tensor POVM acts on the full density matrix before marginalization.",
        CONFIG
        + "model = PhysicalMuTA(2, physical_config=config)\n"
        + CONDITIONAL.replace("[1, 0]", "[1, 0, 0, 0]")
        + """p = result.decoded_joint_probabilities
assert len(p) == 4 and abs(sum(p.values()) - 1) < 1e-9
marginals = list(result.decoded_marginals.values())
connected = p[(0, 0)] - marginals[0][0] * marginals[1][0]
assert abs(connected) > 1e-5
print(p)
print("connected probability:", connected)
""",
        "The joint distribution is not reconstructed from independent marginals.",
        "Broad finite codewords and a fixed analog branch limit the physical interpretation.",
    ),
    (
        39,
        "Physical shots and statistical error",
        "Sample reproducible physical trajectories.",
        "Averaged conditional probabilities and sampled bit frequencies are different estimators.",
        CONFIG
        + """result = model.run([1, 0], shots=3, seed=14)
assert len(result.sampled_output_bits) == 3
assert result.standard_errors is not None
print("mean conditional POVM:", result.decoded_joint_probabilities)
print("standard errors:", result.standard_errors)
print("sampled bit frequencies:", result.empirical_probabilities)
""",
        "Each trajectory samples homodyne outcomes; final bits use the joint output POVM.",
        "Three shots illustrate the API, not a statistically precise estimate.",
    ),
    (
        40,
        "Logical vs physical comparison",
        "Compare decoded statistics without a false state overlap.",
        "Logical qubit and physical Fock vectors occupy different Hilbert spaces.",
        CONFIG
        + CONDITIONAL
        + """from photographiqml import compare_logical_physical
comparison = compare_logical_physical(result)
assert comparison["finite_state_fidelity"] is None
assert comparison["total_variation_distance"] >= 0
print(comparison)
""",
        "Total variation compares decoded probabilities to an ideal logical target.",
        "The comparison remains conditional and does not certify a quantum channel.",
    ),
    (
        41,
        "Physical convergence",
        "Vary numerical grid resolution at fixed resources.",
        "Numerical convergence and changes in physical squeezing are distinct studies.",
        CONFIG
        + """study = model.physical_convergence([1, 0], [513, 1025], axis="grid_points",
    mode="physical-conditional", analog_outcomes=dict.fromkeys(model.measurement_order, 0.0))
assert not study["certified"]
print(study["axis_type"])
print([row["max_probability_delta"] for row in study["rows"]])
""",
        "Only the integration grid changes in this whole-pattern calculation.",
        "A small grid delta does not establish cutoff, peak-sum or finite-squeezing accuracy.",
    ),
    (
        42,
        "Unsupported arbitrary XY why it fails",
        "Exercise the preserved safety boundary.",
        "Rotating a homodyne angle does not synthesize arbitrary logical XY measurement.",
        CONFIG
        + """try:
    model.run([1, 0], {"alpha.w0.c1": .37})
except NotImplementedError as error:
    print(error)
else:
    raise AssertionError("Unsupported physical measurement executed")
""",
        "The exception is raised before finite GKP projection and Fock simulation.",
        "Arbitrary angles require a separately validated measurement-synthesis instrument.",
    ),
    (
        43,
        "Logical training to physical validation",
        "Audit a continuously trained logical candidate.",
        "Logical optimizer success says nothing about physical measurement support.",
        """import numpy as np
from photographiqml import MuTA, Trainer, GKPBridge
model = MuTA(1)
parameters, history = Trainer(epochs=2).fit(
    lambda p: float((model.run([1, 0], p).probabilities[1] - .3)**2), np.full(4, .5))
audit = model.physical_capabilities(parameters)
assert not audit["supported"]
try:
    GKPBridge().run(model, [1, 0], parameters)
except NotImplementedError as error:
    print(error)
else:
    raise AssertionError("Continuous candidate was silently altered")
""",
        "The candidate is rejected unchanged; it is not rounded into the discrete family.",
        "Two logical optimizer steps toward a target output probability demonstrate the workflow, not converged task-learning performance.",
    ),
    (
        44,
        "Discrete physical-angle search",
        "Evaluate categorical candidates with physical shots.",
        "Coordinate flips preserve exact 0/pi measurements; the physical loss has sampling noise.",
        CONFIG
        + """import numpy as np
from photographiqml.physical_training import DiscreteSearch
def objective(angles):
    result = model.run([1, 0], angles, shots=1, seed=14)
    return result.decoded_joint_probabilities[(1,)]
search = DiscreteSearch(sweeps=1, seed=14).fit(model, objective, initial=np.zeros(4))
assert set(search.parameters) <= {0, np.pi}
assert len(search.evaluations) == 5
print("selected:", search.parameters)
print("estimated error:", search.loss)
""",
        "Every objective evaluation executes a supported finite GKP pattern.",
        "One shot and a relabeling-only family cannot establish expressive trainability.",
    ),
]


def main():
    links = []
    for number, title, goal, theory, code, interpretation, limitations in TUTORIALS:
        lines = code.splitlines()
        imports = [line for line in lines if line.startswith(("import ", "from "))]
        body = [line for line in lines if not line.startswith(("import ", "from "))]
        code = "\n".join(imports + [""] + body) + "\n"
        slug = f"{number}_" + title.lower().replace(".", "").replace(" ", "_")
        path = ROOT / "examples" / "ml" / "tutorials" / f"{slug}.py"
        path.write_text(code, encoding="utf-8")
        subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--fix", str(path)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [sys.executable, "-m", "ruff", "format", str(path)], check=True, capture_output=True
        )
        code = path.read_text(encoding="utf-8")
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(code, str(path), "exec"), {"__name__": "__main__"})
        document = f"""# {number}. {title}

## Goal

{goal}

## Theory

{theory}

## Code

Run this standalone example after installing the pinned dependency and package.

```python
{code.rstrip()}
```

## Output

Captured from execution; upstream truncation warnings go to stderr and remain active.

```text
{capture.getvalue().rstrip()}
```

## Validation

The assertions above execute during generation and in `scripts/ml/execute_tutorials.py`.
See [physical architecture](../physical/architecture.md) and the
[release report](../release-report.md) for independent logical and physical checks.

## Physical interpretation

{interpretation}

## Limitations

{limitations}
"""
        (ROOT / "docs" / "ml" / "tutorials" / f"{slug}.md").write_text(document, encoding="utf-8")
        links.append(f"- [{number}. {title}]({slug}.md)")
        print(path, flush=True)
    index = ROOT / "docs" / "ml" / "tutorials" / "index.md"
    old = index.read_text(encoding="utf-8").split("## Physical GKP tutorials")[0]
    index.write_text(
        old.rstrip() + "\n\n## Physical GKP tutorials\n\n" + "\n".join(links) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
