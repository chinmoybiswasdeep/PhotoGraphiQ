"""Finite-energy signed-X MuTA execution with separate quantum and decoded results."""

from dataclasses import dataclass, field, replace
from itertools import product
from time import perf_counter

import numpy as np

import photographiq as pg

from .ansatz.muta import MuTA
from .logical import H, execute, local_gate
from .lowering import (
    GKPPhysicalConfig,
    lower_muta_to_gkp,
    node_frame,
    physical_capabilities,
    product_input,
    require_supported,
)


@dataclass
class PhysicalMuTAResult:
    physical_result: object
    decoded_joint_probabilities: dict
    decoded_marginals: dict
    logical_target: object
    frames: list
    measurement_records: list
    resource_config: GKPPhysicalConfig
    lowering: object
    mode: str
    shots: int
    standard_errors: dict | None
    sampled_output_bits: list
    empirical_probabilities: dict | None
    empirical_standard_errors: dict | None
    diagnostics: dict
    convergence: dict = field(
        default_factory=lambda: {
            "certified": False,
            "reason": "A single resource setting is not a convergence study",
        }
    )
    representation: str = "gkp-physical"


def _check_execution(model, input_state, mode, shots, analog_outcomes, output_basis):
    if mode not in ("physical-conditional", "physical-shots"):
        raise ValueError(
            "Choose physical-conditional or physical-shots; logical-exact uses MuTA.run"
        )
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer")
    if output_basis not in ("X", "Z"):
        raise NotImplementedError("Final physical readout supports X/Z only")
    if mode == "physical-conditional":
        if (
            shots != 1
            or analog_outcomes is None
            or set(analog_outcomes) != set(model.measurement_order)
        ):
            raise ValueError(
                "physical-conditional requires shots=1 and one explicit analog outcome per measured logical node"
            )
        if not all(
            np.isscalar(v) and np.isreal(v) and np.isfinite(v) for v in analog_outcomes.values()
        ):
            raise ValueError(
                "Analog outcomes must be finite real scalars, not requested logical bits"
            )
    elif analog_outcomes is not None:
        raise ValueError(
            "physical-shots samples outcomes; analog postselection belongs to physical-conditional"
        )
    return product_input(input_state, model.n_wires)


def run_physical(
    model,
    input_state,
    parameters=None,
    *,
    config=None,
    mode="physical-shots",
    shots=1,
    seed=0,
    analog_outcomes=None,
    output_basis="Z",
):
    config = config or getattr(model, "physical_config", None) or GKPPhysicalConfig()
    # Complete audit and request checks precede lowering's first codeword allocation.
    report = physical_capabilities(model, parameters, config=config)
    require_supported(report)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    joint_input, factors = _check_execution(
        model, input_state, mode, shots, analog_outcomes, output_basis
    )
    start = perf_counter()
    lowering = lower_muta_to_gkp(model, parameters, config=config)
    code = lowering.code
    encoded = {
        node: code.encode(*factor) for node, factor in zip(model.input_nodes, factors, strict=True)
    }
    kwargs = {"inputs": encoded, "backend": config.backend, "cutoff": config.cutoff}
    if mode == "physical-conditional":
        kwargs["measurement_outcomes"] = {
            lowering.measurement_keys[node]: float(v) for node, v in analog_outcomes.items()
        }
        raw_result = pg.simulate(lowering.pattern, seed=seed, **kwargs)
        trajectories = [raw_result]
    else:
        raw_result = pg.run_shots(lowering.pattern, shots, seed=seed, **kwargs)
        trajectories = raw_result.trajectories
    simulation_seconds = perf_counter() - start
    decode_start = perf_counter()
    frames, records, rows, leakage, boundaries = [], [], [], [], []
    labels = tuple(product((0, 1), repeat=model.n_wires))
    for trajectory in trajectories:
        frame = {v: node_frame(model, v, trajectory.records) for v in model.output_nodes}
        frames.append(frame)
        decoded = {}
        for node, key in lowering.measurement_keys.items():
            raw = trajectory.records[key]
            correction = node_frame(model, node, trajectory.records).correction("X")
            decoded[node] = replace(
                raw,
                bit=int(trajectory.records[("bit", node)]),
                frame_correction=raw.frame_correction ^ correction,
            )
        records.append(decoded)
        readout = pg.multimode_readout(
            trajectory.state, dict.fromkeys(model.output_nodes, output_basis), frames=frame
        )
        rows.append([readout["joint_probabilities"][bits] for bits in labels])
        leakage.append(
            {
                node: code.leakage(trajectory.state.reduced((node,)).density_matrix)
                for node in model.output_nodes
            }
        )
        boundaries.append(list(trajectory.state.diagnostics))
    values = np.asarray(rows)
    mean = values.mean(axis=0)
    probabilities = dict(zip(labels, mean.tolist()))
    marginals = {
        node: tuple(
            float(sum(p for bits, p in probabilities.items() if bits[w] == bit)) for bit in (0, 1)
        )
        for w, node in enumerate(model.output_nodes)
    }
    standard_errors = (
        dict(zip(labels, (values.std(axis=0, ddof=1) / np.sqrt(shots)).tolist()))
        if mode == "physical-shots" and shots > 1
        else None
    )
    # Each final sample is drawn from the joint conditional POVM, never marginals.
    samples, empirical, empirical_se = [], None, None
    if mode == "physical-shots":
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(shots + 1)[-1])
        for row in values:
            if np.min(row) < -1e-9 or not np.isclose(row.sum(), 1, atol=1e-9, rtol=0):
                raise ArithmeticError(
                    "Joint output POVM probabilities are unresolved; refine cutoff"
                )
            # Remove machine-roundoff negativity only for categorical sampling.
            p = np.maximum(row, 0)
            samples.append(labels[rng.choice(len(labels), p=p / p.sum())])
        empirical = {bits: samples.count(bits) / shots for bits in labels}
        empirical_se = (
            {bits: float(np.sqrt(p * (1 - p) / shots)) for bits, p in empirical.items()}
            if shots > 1
            else None
        )
    target = execute(model, joint_input, model._parameters.bind(parameters))
    input_weights = [code.resource(bit).project(code.cutoff)[1] for bit in (0, 1)]
    diagnostics = {
        "codeword_gram": code.gram,
        "basis_projection_weights": input_weights,
        "input_code_subspace_leakage": [
            code.leakage(np.asarray(v.amplitudes)) for v in encoded.values()
        ],
        "marginal_code_subspace_leakage": leakage,
        "joint_code_subspace_leakage": None,
        "backend_diagnostics": boundaries,
        "output_basis": output_basis,
        "decoder": "nearest-cell",
        "decoder_confidence": None,
        "simulation_seconds": simulation_seconds,
        "readout_seconds": perf_counter() - decode_start,
        "resource_accounting": report,
        "estimator": "conditional joint POVM"
        if mode == "physical-conditional"
        else "mean conditional joint POVM (Rao-Blackwell estimator)",
        "pair_correlations": {
            (i, j): float(
                sum(((-1) ** (bits[i] + bits[j])) * p for bits, p in probabilities.items())
            )
            for i in range(model.n_wires)
            for j in range(i + 1, model.n_wires)
        },
    }
    return PhysicalMuTAResult(
        raw_result,
        probabilities,
        marginals,
        target,
        frames,
        records,
        config,
        lowering,
        mode,
        shots,
        standard_errors,
        samples,
        empirical,
        empirical_se,
        diagnostics,
    )


def compare_logical_physical(result):
    """Compare decoded output statistics only; no Fock/qubit overlap."""
    state = result.logical_target.state
    n = len(result.decoded_marginals)
    if result.diagnostics["output_basis"] == "X":
        for wire in range(n):
            state = local_gate(state, H, wire, n)
    ideal = abs(state) ** 2
    actual = np.array(list(result.decoded_joint_probabilities.values()))
    differences = []
    for w, marginal in enumerate(result.decoded_marginals.values()):
        weights = np.array([(-1) ** ((index >> (n - 1 - w)) & 1) for index in range(2**n)])
        differences.append(float(marginal[0] - marginal[1] - ideal @ weights))
    return {
        "total_variation_distance": float(np.sum(abs(actual - ideal)) / 2),
        "observable_differences": differences,
        "comparison_mode": result.mode,
        "finite_state_fidelity": None,
        "marginal_code_subspace_leakage": result.diagnostics["marginal_code_subspace_leakage"],
        "convergence": result.convergence,
    }


def validate_physical_model(
    model, input_state, values, *, axis="cutoff", parameters=None, config=None, **run_options
):
    """Whole-pattern study; numerical axes and physical resource axes stay distinct."""
    if not config:  # the default is built only when the model has no physical_config
        config = model.physical_config if hasattr(model, "physical_config") else GKPPhysicalConfig()
    if axis not in ("cutoff", "grid_points", "peaks", "peak_width", "envelope"):
        raise ValueError("Unknown convergence axis")
    values = tuple(values)
    if len(values) < 2 or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("Supply at least two strictly increasing sweep values")
    # Audit the entire sweep before executing its first point.
    configs = [replace(config, **{axis: value}) for value in values]
    for current in configs:
        require_supported(physical_capabilities(model, parameters, config=current))
    rows, previous = [], None
    for value, current in zip(values, configs, strict=True):
        result = run_physical(model, input_state, parameters, config=current, **run_options)
        probability = np.array(list(result.decoded_joint_probabilities.values()))
        rows.append(
            {
                "value": value,
                "probabilities": probability.tolist(),
                "max_probability_delta": None
                if previous is None
                else float(np.max(abs(previous - probability))),
                "comparison": compare_logical_physical(result),
                "standard_errors": result.standard_errors,
                "prediction": int(np.argmax(probability)),
                "backend_diagnostics": result.diagnostics["backend_diagnostics"],
            }
        )
        previous = probability
    return {
        "axis": axis,
        "axis_type": "physical resource change"
        if axis in ("peak_width", "envelope")
        else "numerical refinement",
        "rows": rows,
        "certified": False,
        "interpretation": "Fixed analog branch deltas are conditional; shot deltas also include sampling error. No automatic convergence certification.",
    }


class PhysicalMuTA(MuTA):
    """Restricted finite-GKP MuTA with a categorical signed-X measurement family."""

    def __init__(
        self, n_wires=1, n_layers=1, *, physical_config=None, measurement_family="X", **kwargs
    ):
        if measurement_family != "X":
            raise ValueError("PhysicalMuTA currently supports measurement_family='X' (0/pi) only")
        if kwargs.pop("representation", "gkp-physical") != "gkp-physical":
            raise ValueError("PhysicalMuTA requires representation='gkp-physical'")
        super().__init__(n_wires, n_layers, **kwargs)
        self.representation = "gkp-physical"
        self.physical_config = physical_config or GKPPhysicalConfig()
        if not isinstance(self.physical_config, GKPPhysicalConfig):
            raise TypeError("physical_config must be GKPPhysicalConfig")
        self.measurement_family = measurement_family

    def initialize(self, seed=None, scale=None):
        if scale is not None:
            raise ValueError(
                "Physical angles are categorical; continuous initialization is unsupported"
            )
        return dict(
            zip(
                self.trainable_parameters(),
                np.random.default_rng(seed).choice([0.0, np.pi], self.n_parameters),
            )
        )

    def angle_space(self):
        return {name: (0.0, float(np.pi)) for name in self.trainable_parameters()}

    def run(self, input_state, parameters=None, **options):
        return run_physical(self, input_state, parameters, config=self.physical_config, **options)

    def unitary(self, parameters=None):
        raise NotImplementedError(
            "A finite-GKP instrument is not a logical unitary; use GKPBridge.logical_target"
        )

    def run_batch(self, states, parameters=None):
        raise NotImplementedError(
            "Use explicit per-input physical shots; physical results are not logical statevectors"
        )

    def state_dict(self):
        data = super().state_dict()
        data["schema"] = 2
        data["physical_config"] = self.physical_config.to_dict()
        data["measurement_family"] = self.measurement_family
        return data

    def physical_convergence(self, input_state, values, **options):
        return validate_physical_model(
            self, input_state, values, config=self.physical_config, **options
        )
