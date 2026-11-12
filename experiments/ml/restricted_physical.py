"""Reproducible finite-resource evidence, without a convergence certification."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from photographiqml import GKPBridge, GKPPhysicalConfig, PhysicalMuTA
from photographiqml.models import MuTAClassifier
from photographiqml.physical import compare_logical_physical
from photographiqml.physical_training import DiscreteSearch


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def main():
    config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
    evidence = {
        "resource_config": asdict(config),
        "benchmarks": [],
        "convergence": {},
        "classification": [],
        "claims": "Restricted signed-X execution only; broad, overlapping finite codewords. No universality, convergence certification or advantage claim.",
    }
    for wires in (1, 2):
        model = PhysicalMuTA(wires, physical_config=config)
        start = perf_counter()
        result = model.run(
            [1] + [0] * (2**wires - 1),
            mode="physical-conditional",
            analog_outcomes=dict.fromkeys(model.measurement_order, 0.0),
        )
        evidence["benchmarks"].append(
            {
                "wires": wires,
                "paper_layers": model.paper_depth,
                "mode": result.mode,
                "shots": 1,
                "wall_seconds": perf_counter() - start,
                "probabilities": result.decoded_joint_probabilities,
                "comparison": compare_logical_physical(result),
                "diagnostics": result.diagnostics,
                "memory_measurement": "Analytic allocation estimates only; peak process memory not measured",
            }
        )
    model = PhysicalMuTA(physical_config=config)
    try:
        model.run([1, 0], shots=16, seed=2026)
    except ValueError as error:
        if "Fock truncation norm" not in str(error):
            raise
        evidence["rejected_resolution"] = {
            "cutoff": 24,
            "shots": 16,
            "seed": 2026,
            "error": str(error),
        }
    shot_config = replace(config, cutoff=40)
    model = PhysicalMuTA(physical_config=shot_config)
    result = model.run([1, 0], shots=16, seed=2026)
    evidence["shot_study"] = {
        "shots": result.shots,
        "seed": 2026,
        "resource_config": asdict(shot_config),
        "probabilities": result.decoded_joint_probabilities,
        "standard_errors": result.standard_errors,
        "empirical_probabilities": result.empirical_probabilities,
        "empirical_standard_errors": result.empirical_standard_errors,
        "samples": result.sampled_output_bits,
        "comparison": compare_logical_physical(result),
        "diagnostics": result.diagnostics,
    }
    model = PhysicalMuTA(physical_config=config)
    for axis, values in {
        "cutoff": [20, 24, 28],
        "grid_points": [513, 1025],
        "peaks": [3, 4],
        "peak_width": [0.85, 0.9],
        "envelope": [0.85, 0.9],
    }.items():
        evidence["convergence"][axis] = model.physical_convergence(
            [1, 0],
            values,
            axis=axis,
            mode="physical-conditional",
            analog_outcomes=dict.fromkeys(model.measurement_order, 0.0),
        )
    bridge = GKPBridge(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
    evidence["resource_convergence"] = asdict(bridge.measurement_convergence([20, 24, 28]))
    model = PhysicalMuTA(physical_config=shot_config)
    for seed in (7, 19):
        # A small execution demo, not an out-of-sample performance estimate.
        X, y = [[0.0], [0.2], [2.9], [np.pi]], [0, 0, 1, 1]
        classifier = MuTAClassifier(
            model,
            trainer=DiscreteSearch(sweeps=1, seed=seed),
            physical_options={"mode": "physical-shots", "shots": 2, "seed": seed},
        )
        classifier.fit(X, y)
        classifier.physical_options["seed"] = seed + 1000  # Independent validation trajectories.
        prediction = classifier.predict(X)
        evidence["classification"].append(
            {
                "training_seed": seed,
                "validation_seed": seed + 1000,
                "shots_per_input": 2,
                "resource_config": asdict(shot_config),
                "inputs": X,
                "labels": y,
                "predictions": prediction,
                "training_input_accuracy_new_trajectories": float(np.mean(prediction == y)),
                "search": asdict(classifier.history),
                "convergence": classifier.convergence,
                "physical_diagnostics": classifier.physical_diagnostics,
            }
        )
    path = Path(__file__).resolve().parents[2] / "docs" / "ml" / "physical" / "evidence.json"
    path.write_text(json.dumps(serializable(evidence), indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
