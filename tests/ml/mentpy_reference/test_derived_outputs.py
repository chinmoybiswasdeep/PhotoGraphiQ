import numpy as np
import pytest

pytest.importorskip("mentpy")

import mentpy as mp

from photographiqml import MuTA, MuTAKernel, QuantumInstrumentModel
from photographiqml.validation import mentpy_reference


def reference_state(model, parameters):
    ref, mapping = mentpy_reference(model, fix_measurements=True)
    values = model._parameters.bind(parameters)
    simulator = mp.PatternSimulator(
        ref, input_state=np.array([1, 0, 0, 0]), backend="numpy-sv", window_size=5
    )
    return simulator.run(
        [values[model.parameter_name(mapping[v])] for v in ref.trainable_nodes], output_form="sv"
    )


def test_kernel_against_reference():
    kernel = MuTAKernel()
    data = np.array([[0.2, -0.5], [1.1, 0.4]])
    states = []
    for x0, x1 in data:
        states.append(
            reference_state(
                kernel.model,
                {
                    "alpha.w0.c0": x0,
                    "alpha.w1.c0": x1,
                    "alpha.w1.c1": np.cos(x0) * np.cos(x1),
                    "alpha.w0.c2": x0,
                    "alpha.w1.c2": x1,
                },
            )
        )
    expected = abs(np.vdot(*states)) ** 2
    assert kernel.gram_matrix(data)[0, 1] == pytest.approx(expected)


def test_instrument_marginals_against_reference():
    model = MuTA(2, one_column=True)
    params = model.initialize(7, 1)
    state = reference_state(model, params).reshape(2, 2)
    branches = QuantumInstrumentModel(model).run([1, 0, 0, 0], params)
    for bit, branch in enumerate(branches):
        probability = np.vdot(state[bit], state[bit]).real
        assert branch.probability == pytest.approx(probability)
        assert abs(np.vdot(branch.state, state[bit] / np.sqrt(probability))) ** 2 == pytest.approx(
            1
        )
