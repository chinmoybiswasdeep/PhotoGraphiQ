import numpy as np
from scipy.linalg import expm

import photographiq as pg


def direct(circuit, cutoff=32):
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    q, n = a + a.T, np.diag(np.arange(cutoff))
    state = np.zeros(cutoff, complex)
    state[0], state[1] = 2**-0.5, 2**-0.5
    for name, _, params in circuit.gates:
        if name == "symplectic":
            s = params[0]
            if np.allclose(s[0], [1, 0]):
                u = expm(1j * s[1, 0] * (q @ q) / 4)
            else:
                u = expm(1j * np.arctan2(s[1, 0], s[0, 0]) * n)
        elif name == "cubic_phase":
            u = expm(1j * params[0] * (q @ q @ q) / 6)
        else:
            raise AssertionError(name)
        state = u @ state
    return state


def test_kerr_synthesis_refines():
    errors = []
    for steps in (1, 4, 16):
        circuit, report = pg.synthesis.synthesize_kerr(0.002, steps=steps)
        actual = direct(circuit)
        expected = np.zeros(32, complex)
        expected[0], expected[1] = 2**-0.5, np.exp(0.002j) * 2**-0.5
        errors.append(1 - abs(np.vdot(expected, actual)) ** 2)
        assert report.primitive_count == len(circuit.gates)
    assert errors[-1] < errors[0] / 3
    assert errors[-1] < 1e-7


def test_trace_and_cubic_injection():
    circuit = pg.Circuit(1).rotate(0, 0.2).cubic_phase(0, 0.01)
    pattern, trace = circuit.compile(return_trace=True, squeezing=0.2)
    assert pattern.to_json() == circuit.compile(squeezing=0.2).to_json()
    assert trace.steps[0].source_gate == "R"
    assert trace.steps[0].source_parameters == (0.2,)
    assert trace.steps[1].source_gate == "cubic_phase"
    assert len(trace.steps[1].resource_nodes) == 1
    assert trace.steps[1].corrections
    assert not any(isinstance(c, pg.CubicPhase) for c in pattern.commands)
    assert any(
        isinstance(c, pg.Prepare) and isinstance(c.state, pg.CubicPhaseResource)
        for c in pattern.commands
    )


def test_trace_preserves_runtime_callable_compilation():
    correction = pg.CallableExpression(lambda records: records[0], {0})
    circuit = pg.Circuit(1).displace(0, q=correction)
    pattern, trace = circuit.compile(return_trace=True)
    result = pg.simulate(pattern, backend="gaussian", seed=7)
    assert len(trace.steps) == 1
    assert np.isfinite(result.state.mean).all()
