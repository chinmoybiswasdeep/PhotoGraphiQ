"""Capability-audited lowering of signed-X MuTA to public PhotoGraphiQ commands."""

from dataclasses import asdict, dataclass, field
from math import comb

import numpy as np

import photographiq as pg

from .logical import statevector

PHOTOGRAPHIQ_COMMIT = "db07f9f9bf47da841bfa6b206562c5a3ffb121d3"
REQUIRED_API = (
    "GKPCode",
    "PhysicalGKPReadout",
    "LogicalPauliFrame",
    "LogicalDecodeResult",
    "BaseGKPDecoder",
    "NearestCellDecoder",
    "SoftDecisionDecoder",
    "multimode_readout",
    "measurement_convergence",
    "Parameter",
    "Pattern",
    "Prepare",
    "Measure",
    "Signal",
    "CallableExpression",
    "Output",
    "simulate",
    "run_shots",
)


VALIDATED_BACKENDS = ("piquasso-fock", "native-fock")


def check_photographiq_contract():
    missing = [name for name in REQUIRED_API if not hasattr(pg, name)]
    version = tuple(int(x) for x in pg.__version__.split(".")[:3])
    if missing or not (0, 3, 1) <= version < (0, 4, 0):
        raise ImportError(
            f"Physical MuTA requires PhotoGraphiQ >=0.3.1,<0.4; found {pg.__version__}; missing {missing}. Install audited commit {PHOTOGRAPHIQ_COMMIT}"
        )
    return {"version": pg.__version__, "audited_commit": PHOTOGRAPHIQ_COMMIT}


@dataclass(frozen=True)
class GKPPhysicalConfig:
    """Resource and allocation policy. Cutoff is exclusive total photon number."""

    cutoff: int = 48
    peak_width: float = 0.6
    envelope: float = 0.6
    peaks: int = 6
    grid_points: int = 2049
    decoder: str = "nearest"
    backend: str = "piquasso-fock"
    max_dimension: int = 100_000
    max_matrix_bytes: int = 256_000_000

    def __post_init__(self):
        check_photographiq_contract()
        self.code()  # Validation only: GKPCode does not project until requested.
        if self.decoder not in ("nearest", "soft"):
            raise ValueError("decoder must be nearest or soft")
        # native-fock is validated against piquasso-fock in tests/ml/cross_layer
        # (test_native_backend.py); piquasso-fock stays the default for reproducibility.
        if self.backend not in VALIDATED_BACKENDS:
            raise ValueError(
                f"Physical MuTA validates backend in {VALIDATED_BACKENDS} only; mixed Fock lowering remains unsupported"
            )
        for name in ("max_dimension", "max_matrix_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def code(self):
        return pg.GKPCode(
            cutoff=self.cutoff,
            peak_width=self.peak_width,
            envelope=self.envelope,
            peaks=self.peaks,
            grid_points=self.grid_points,
        )

    def to_dict(self):
        return asdict(self)


class UnsupportedPhysicalMeasurement(NotImplementedError):
    pass


def _schedule(model):
    """Prepare only future neighbors needed by the next measurement."""
    prepared = set(model.input_nodes)
    alive = set(prepared)
    edges = set()
    operations = []
    peak = len(alive)
    for node in model.measurement_order:
        for other in sorted(model.graph.neighbors(node)):
            if other not in prepared:
                operations.append(("prepare", other))
                prepared.add(other)
                alive.add(other)
                peak = max(peak, len(alive))
            edge = frozenset((node, other))
            if edge not in edges:
                operations.append(("cz", (node, other)))
                edges.add(edge)
        operations.append(("measure", node))
        alive.remove(node)
    if alive != set(model.output_nodes) or len(edges) != model.graph.number_of_edges():
        raise ValueError("Unsupported open-graph schedule: output or edge mismatch")
    return operations, peak


def physical_capabilities(model, parameters=None, *, config=None):
    """Audit every bound angle using upstream tolerance, without Fock projection."""
    config = config or getattr(model, "physical_config", None) or GKPPhysicalConfig()
    code = config.code()
    angles = model._parameters.bind(parameters)
    audit = []
    for node in model.measurement_order:
        name = model.parameter_name(node)
        angle = angles[name]
        try:
            readout = code.logical_measurement("XY", alpha=angle)
            audit.append(
                {
                    "node": node,
                    "parameter": name,
                    "angle": angle,
                    "supported": True,
                    "basis": "X",
                    "flip": readout.flip,
                }
            )
        except NotImplementedError:
            audit.append(
                {
                    "node": node,
                    "parameter": name,
                    "angle": angle,
                    "supported": False,
                    "basis": None,
                    "flip": None,
                }
            )
    _, peak = _schedule(model)
    dimension = comb(config.cutoff + peak - 1, peak)
    output_dimension = comb(config.cutoff + model.n_wires - 1, model.n_wires)
    # Joint POVM contraction allocates several output-size complex matrices.
    readout_bytes = 64 * output_dimension**2
    projection_bytes = 64 * config.grid_points * (config.cutoff + 2 * config.peaks + 1)
    unsupported = [row for row in audit if not row["supported"]]
    reasons = []
    if config.decoder != "nearest":
        reasons.append(
            "Soft ensemble posteriors are not calibrated for adaptive MuTA node states; use nearest for flow and resource_readout for calibrated soft features"
        )
    if (
        dimension > config.max_dimension
        or max(readout_bytes, projection_bytes) > config.max_matrix_bytes
    ):
        reasons.append(
            "Fock allocation exceeds configured dimension/matrix budget; lower cutoff or width, or explicitly raise the budget"
        )
    return {
        "supported": not unsupported and not reasons,
        "angle_audit": audit,
        "unsupported_nodes": unsupported,
        "reasons": reasons,
        "decoder": config.decoder,
        "backend": config.backend,
        "required_operations": [
            "finite GKP plus",
            "unit CZ",
            "physical X",
            "logical frame",
            "joint X/Z output POVM",
        ],
        "resource_nodes": len(model.graph),
        "measurements": len(model.flow),
        "cz_operations": model.graph.number_of_edges(),
        "peak_live_modes": peak,
        "hilbert_dimension": dimension,
        "output_dimension": output_dimension,
        "vector_bytes": 16 * dimension,
        "joint_readout_workspace_bytes_estimate": readout_bytes,
        "projection_workspace_bytes_estimate": projection_bytes,
        "frame_updates": sum(1 + len(z) for _, z in model.corrections.values()),
    }


def require_supported(report):
    if report["unsupported_nodes"]:
        row = report["unsupported_nodes"][0]
        raise UnsupportedPhysicalMeasurement(
            f"Physical GKP MuTA cannot lower {row['parameter']}={row['angle']:.17g} rad: "
            "PhotoGraphiQ 0.3.1 supports physical XY only at 0 and pi modulo 2pi "
            "(absolute tolerance 1e-14). Use representation='logical' or an explicit "
            "discrete X family; other angles require a validated LogicalMeasurementSynthesis protocol."
        )
    if report["reasons"]:
        raise ValueError("; ".join(report["reasons"]))


def product_input(input_state, n_wires):
    """Factor a normalized product input; reject entanglement, never discard it."""
    array = np.asarray(input_state, dtype=complex)
    if array.shape == (n_wires, 2):
        product_factors = tuple(statevector(v, 1) for v in array)
        joint = np.array([1], dtype=complex)
        for v in product_factors:
            joint = np.kron(joint, v)
        return joint, product_factors
    joint = statevector(array, n_wires)
    remainder = joint
    factors = []
    for _ in range(n_wires - 1):
        u, singular, vh = np.linalg.svd(remainder.reshape(2, -1), full_matrices=False)
        if singular[1] > 1e-10:
            raise NotImplementedError(
                "Entangled physical input encoding is unsupported in v0.2; supply normalized product logical states"
            )
        factors.append(u[:, 0])
        remainder = singular[0] * vh[0]
    factors.append(remainder)
    return joint, tuple(factors)


def node_frame(model, node, records):
    """Frame in the fully entangled open-graph convention, from interpreted bits."""
    frame = pg.LogicalPauliFrame()
    for source, (successor, z_targets) in model.corrections.items():
        if node == successor or node in z_targets:
            bit = int(records[("bit", source)])
            frame = frame.compose(
                pg.LogicalPauliFrame(
                    bit if node == successor else 0, bit if node in z_targets else 0
                )
            )
    return frame


@dataclass
class PhysicalLoweringResult:
    pattern: pg.Pattern
    code: pg.GKPCode
    input_modes: tuple
    output_modes: tuple
    measurement_keys: dict
    node_map: dict
    frame_dependencies: dict
    audit: dict
    resource_config: GKPPhysicalConfig
    decoder: pg.BaseGKPDecoder
    convergence: dict = field(default_factory=lambda: {"certified": False})


def lower_muta_to_gkp(model, parameters=None, *, config=None):
    """Return an inspectable physical Pattern after complete capability audit."""
    config = config or getattr(model, "physical_config", None) or GKPPhysicalConfig()
    report = physical_capabilities(model, parameters, config=config)
    require_supported(report)
    code = config.code()
    decoder = pg.NearestCellDecoder()
    readouts = {
        row["node"]: code.logical_measurement("XY", alpha=row["angle"], decoder=decoder)
        for row in report["angle_audit"]
    }
    pattern = pg.Pattern(inputs=model.input_nodes)
    keys = {node: ("raw", node) for node in model.measurement_order}
    dependencies = {
        node: tuple(
            source
            for source, (successor, z) in model.corrections.items()
            if node == successor or node in z
        )
        for node in model.graph
    }
    plus = code.plus()  # First Fock allocation: every angle/resource guard already passed.
    for operation, value in _schedule(model)[0]:
        if operation == "prepare":
            pattern.append(pg.Prepare(value, state=plus))
        elif operation == "cz":
            pattern.append(code.logical_cz(*value))
        else:
            node = value
            key = keys[node]
            pattern.append(pg.Measure(node, readouts[node], key))
            declared = frozenset({key} | {("bit", v) for v in dependencies[node]})

            def interpreted(records, node=node, key=key):
                return records[key].bit ^ node_frame(model, node, records).correction("X")

            pattern.append(pg.Signal(("bit", node), pg.CallableExpression(interpreted, declared)))
    pattern.append(pg.Output(model.output_nodes)).validate()
    return PhysicalLoweringResult(
        pattern,
        code,
        model.input_nodes,
        model.output_nodes,
        keys,
        {v: v for v in model.graph},
        dependencies,
        report,
        config,
        decoder,
    )


def visualize_lowering(model, parameters=None, *, config=None, ax=None):
    """Capability-only visualization; unsupported nodes are red before allocation."""
    import matplotlib.pyplot as plt
    import networkx as nx

    report = physical_capabilities(model, parameters, config=config)
    bad = {row["node"] for row in report["unsupported_nodes"]}
    if ax is None:
        _, ax = plt.subplots(figsize=(8, model.n_wires + 2))
    labels = {
        v: f"{v}\n{'unsupported XY' if v in bad else 'X / decoder' if v in model.flow else 'X/Z output'}"
        for v in model.graph
    }
    nx.draw(
        model.graph,
        {v: (v[1], -v[0]) for v in model.graph},
        labels=labels,
        node_color=["tomato" if v in bad else "skyblue" for v in model.graph],
        ax=ax,
    )
    ax.set_title("Logical node = GKP mode; flow signals -> classical Pauli frame")
    return ax
