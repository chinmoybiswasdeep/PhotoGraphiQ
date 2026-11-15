"""Author and execute concise tutorials; outputs are captured, never invented."""

import contextlib
import io
import json
from pathlib import Path

import nbformat

# Repository-root anchored, so running from any cwd only touches the PhotoGraphiQML trees.
ROOT = Path(__file__).resolve().parents[2]

PRELUDE = "import numpy as np\nimport photographiqml as pqml\n"

# Title, theory, executable example, limitations. Examples deliberately use
# only implemented APIs; research gates demonstrate an explicit refusal.
LESSONS = [
    (
        "What is MB-QML?",
        "Measurement angles parameterize a logical map. An all-X four-measurement wire implements identity after Pauli correction.",
        "m = pqml.MuTA(1)\nr = m.run([1, 0])\nassert np.allclose(r.probabilities, [1, 0])\nprint(r.probabilities.round(6))",
        "This is ideal qubit computation; it has no finite-squeezing parameter.",
    ),
    (
        "MuTA from the paper",
        "Table I obtains IsingXX(-phi) from the base-center measurement. On |00>, phi=pi/2 creates a maximally entangled logical state.",
        "m = pqml.MuTA(2, one_column=True)\nr = m.run([1, 0, 0, 0], {'alpha.w1.c1': np.pi/2})\nassert np.allclose(r.probabilities, [0.5, 0, 0, 0.5])\nprint(r.probabilities.round(6))",
        "The sign convention is exp(+i phi XX/2); angles are not homodyne settings.",
    ),
    (
        "First MuTA model",
        "The public model exposes geometry, parameters and ideal output states separately. Wire zero is the most significant tensor factor.",
        "m = pqml.MuTA(2, one_column=True)\nassert m.n_parameters == 8\nprint(m.summary())",
        "The full paper model trains four angles per wire. MentPy's default restriction is not adopted silently.",
    ),
    (
        "Triangle neuron anatomy",
        "The MuTA triangle is a tip joined to both ends of a three-site base, forming a four-cycle in a bipartite graph.",
        "cell = pqml.TriangleNeuron(2)\nassert cell.graph.has_edge((0,1), (1,0))\nassert cell.graph.has_edge((0,1), (1,2))\nprint(sorted(cell.graph.edges))",
        "Five sites describe each wire segment, not the cardinality of the named triangle.",
    ),
    (
        "Multi-wire MuTA",
        "A full layer rotates the choice of tip through the wires under MentPy's API convention. Connections within each paper layer share one tip.",
        "m = pqml.MuTA(3, 1)\nassert m.paper_depth == 3\nassert len(m.graph) == 39\nprint(m.summary())",
        "Arbitrarily adding tips to one block can destroy flow. The constructor enforces admissible blocks.",
    ),
    (
        "Multi-layer MuTA",
        "Composition identifies output sites with subsequent inputs, so each paper layer adds four sites per wire.",
        "m = pqml.MuTA(2, 2, one_column=True)\nassert len(m.graph) == 18\nassert m.output_nodes == ((0,8), (1,8))\nprint(m.output_nodes)",
        "Integer MentPy labels are not stable under stacking; semantic coordinates are.",
    ),
    (
        "Trainable measurement angles",
        "PhotoGraphiQ Parameter expressions identify angles while separate metadata records trainability and frozen values.",
        "m = pqml.MuTA(1)\nm.freeze('alpha.w0.c3', 0)\nassert m.n_parameters == 3\nprint(list(m.trainable_parameters()))\nm.unfreeze('alpha.w0.c3')\nassert m.n_parameters == 4",
        "Tied parameter groups and bounds are pending. Frozen values cannot be overridden by run.",
    ),
    (
        "Classical feature encoding",
        "Product Ry encoding maps a real feature vector to one qubit per feature. Each supplied value is interpreted as radians.",
        "from photographiqml.models import angle_encode\ns = angle_encode([0, np.pi])\nassert np.allclose(abs(s)**2, [0,1,0,0])\nprint(abs(s).round(6))",
        "This product encoder differs from the paper's measurement-based kernel feature map. No standardization is implicit.",
    ),
    (
        "Quantum state inputs",
        "A normalized entangled input bypasses classical feature encoding and is transformed coherently in wire order.",
        "s = np.array([1,0,0,1])/np.sqrt(2)\nr = pqml.MuTA(2).run(s)\nassert abs(np.vdot(s,r.state))**2 > 1-1e-12\nprint(r.probabilities.round(6))",
        "Mixed logical inputs are not implemented. Gaussian/Fock inputs belong to PhotoGraphiQ physical execution.",
    ),
    (
        "Gate learning",
        "Average pure-state infidelity is the paper's supervised objective. The example fits a one-qubit X rotation on independently sampled states.",
        "from photographiqml.models import haar_states, infidelity\nfrom scipy.linalg import expm\nfrom photographiqml.logical import X\nm = pqml.MuTA(1)\ns = haar_states(1,7,42)\nt = s @ expm(0.4j*X).T\ndef loss(p):\n    return infidelity(m.run_batch(s,p),t)\np,h = pqml.Trainer(epochs=100).fit(loss, np.zeros(4))\nassert h.losses[-1] < 1e-4\nprint(round(h.losses[-1],6))",
        "Use the larger reproduction script for 20-seed two-qubit targets and reference checkpoints.",
    ),
    (
        "State classification",
        "QFI for a pure probe and known Hermitian generator is four times its variance. This supplies labels, not a proof of learned generalization.",
        "from photographiqml.diagnostics import pure_qfi\nfrom photographiqml.logical import Z\nH = (np.kron(Z,np.eye(2))+np.kron(np.eye(2),Z))/2\ns = np.array([1,0,0,1])/np.sqrt(2)\nqfi = pure_qfi(s,H)\nassert np.isclose(qfi,4)\nprint(qfi)",
        "The paper's tied-angle polynomial-head classifier remains pending; h=Z/2 keeps SQL=2.",
    ),
    (
        "Classical classification",
        "A trainable MuTA followed by first-wire Z expectation and a logistic head yields probabilities for labels zero and one.",
        "X = [[0],[0.1],[3.0],[3.14]]\ny = [1,1,0,0]\nc = pqml.MuTAClassifier(pqml.MuTA(1),trainer=pqml.Trainer(epochs=60))\nc.fit(X,y)\nassert c.score(X,y) == 1\nprint(c.predict([[0.05],[3.1]]))",
        "Training-set accuracy is only an API smoke test. Research classification requires held-out sets and baselines.",
    ),
    (
        "Quantum instrument learning",
        "An instrument must return outcome probabilities together with conditional quantum states. A final destructive Z measurement is a valid simple instrument.",
        "m = pqml.QuantumInstrumentModel(pqml.MuTA(2,one_column=True))\nb = m.run(np.array([1,0,0,1])/np.sqrt(2))\nassert np.isclose(sum(x.probability for x in b),1)\nprint([(x.outcome,round(x.probability,6)) for x in b])",
        "This demonstrates instrument semantics, not the paper's trained teleportation protocol or controlled intermediate measurements.",
    ),
    (
        "MuTA quantum kernel",
        "The kernel is squared feature-state overlap. Its Gram matrix is PSD because it is an inner product between pure-state density operators.",
        "k = pqml.MuTAKernel()\nX = [[0,0],[1,0],[0,1]]\nd = k.diagnostics(X)\nassert d['minimum_eigenvalue'] > -1e-10\nprint(k.gram_matrix(X).round(4))",
        "The feature map is specifically Eq. 5 for two features. PSD alone does not establish useful classification.",
    ),
    (
        "Training with Adam",
        "Adam uses bias-corrected first and second gradient moments. Each fit resets optimizer state; initial parameters are explicit.",
        "p,h = pqml.Trainer(epochs=180).fit(lambda p: np.sum((p-0.3)**2),[1,-1])\nassert np.linalg.norm(p-0.3) < 0.001\nprint(p.round(4))",
        "Current training is full-batch on deterministic objectives. Physical measurement randomness needs a separate estimator.",
    ),
    (
        "Finite-difference validation",
        "Central differences have second-order truncation error for smooth objectives until floating-point cancellation dominates.",
        "from photographiqml.training import finite_difference\np = np.array([0.3,-0.2])\ng = finite_difference(lambda x: np.sum(x**2),p)\nassert np.allclose(g,2*p,atol=1e-8)\nprint(g.round(6))",
        "Tests additionally check a logical infidelity gradient against its valid Pauli parameter-shift rule.",
    ),
    (
        "Training with a supplied gradient",
        "The trainer accepts a supplied derivative callable. This example uses an analytical derivative to illustrate that interface; no autodiff backend is claimed.",
        "p,h = pqml.Trainer('lbfgs',epochs=20).fit(lambda x: np.sum(x*x), [1,2],gradient=lambda x: 2*x)\nassert np.linalg.norm(p)<1e-8\nprint(p.round(6))",
        "Automatic differentiation through logical or stochastic physical execution is pending. A callable is not automatically an autodiff implementation.",
    ),
    (
        "Finite-squeezing noise of a Gaussian wire",
        "PhotoGraphiQ's unconditional Gaussian wire channel propagates independent finite-resource noise. Four zero-shear steps have ideal identity but nonzero noise.",
        "from photographiq.gaussian import wire_channel\nS,N = wire_channel([0]*4,1.0)\nassert np.allclose(S,np.eye(2))\nassert np.trace(N)>0\nprint(N.round(6))",
        "This direct PhotoGraphiQ calculation is a physical channel diagnostic, not a CVMuTA or GKP gate implementation.",
    ),
    (
        "GKP logical MuTA",
        "The bridge distinguishes an ideal logical target from a finite-energy resource. Full GKP MuTA must fail explicitly until a measurement/injection instrument and decoder are available.",
        "m = pqml.MuTA(1,representation='gkp')\nr = pqml.GKPBridge().logical_target(m,[1,0])\nassert r.representation == 'logical'\ntry:\n    m.run([1,0])\nexcept NotImplementedError:\n    print('Legacy resource-only execution is unsupported; ideal target remains logical')\nelse:\n    raise AssertionError('Unexpected physical execution')",
        "A finite resource and a logical target do not constitute validated finite-energy execution.",
    ),
    (
        "MentPy cross-validation",
        "The optional pinned reference compares semantic topology and logical density matrices. It is not a reference for arbitrary CV amplitudes.",
        "import importlib.util\nif importlib.util.find_spec('mentpy'):\n    from photographiqml.validation import compare_mentpy\n    m = pqml.MuTA(2,one_column=True)\n    error = compare_mentpy(m,[1,0,0,0],m.initialize(2))\n    assert error<1e-10\n    print('Logical density agreement:',error<1e-10)\nelse:\n    print('Optional MentPy reference not installed')",
        "Fixed measurements are repaired explicitly before numerical comparison, and unmodified upstream trainability is audited separately.",
    ),
    (
        "Expressivity vs depth",
        "All-X identity insertion proves weak inclusion when another logical layer is appended. It does not promise strictly larger families at every depth.",
        "a,b = pqml.MuTA(2,1),pqml.MuTA(2,2)\np = a.initialize(3)\nassert np.allclose(a.unitary(p),b.unitary(p))\nprint('Identity extension verified')",
        "This identity fails as a statement about finite-resource CV channels because the appended layer adds noise.",
    ),
    (
        "Tunable entanglement",
        "For exp(i phi XX/2)|00>, concurrence is |sin(phi)|, vanishing at zero and pi. This statement is specific to pure two-qubit outputs.",
        "from photographiqml.diagnostics import concurrence\nm = pqml.MuTA(2,one_column=True)\nphi = 0.7\nc = concurrence(m.run([1,0,0,0],{'alpha.w1.c1':phi}).state)\nassert np.isclose(c,abs(np.sin(phi)))\nprint(round(c,6))",
        "Do not use logical concurrence on Fock amplitudes or infer that every input is entangled by an entangling gate.",
    ),
    (
        "Bias engineering",
        "The paper's inductive bias is imposed through graph connectivity and angle restrictions. Explicitly freezing an angle changes the variational family.",
        "m = pqml.MuTA(2,one_column=True,connections=())\nassert m.graph.number_of_edges()==8\nm.freeze('alpha.w0.c1',0)\nassert m.n_parameters==7\nm.unfreeze('alpha.w0.c1')\nassert m.n_parameters==8\nprint('Topology and angle constraints are explicit')",
        "A CV displacement offset is a different physical notion of bias. Tied parameters are still pending.",
    ),
    (
        "Fisher information",
        "The local pure-state QFI matrix removes the unobservable global-phase direction. Its rank is a local sensitivity diagnostic.",
        "from photographiqml.expressivity import state_fisher\nm = pqml.MuTA(1)\nF = state_fisher(m,[1,0],list(m.initialize(1).values()))\nassert np.linalg.eigvalsh(F).min()>-1e-9\nprint('Local rank:',np.linalg.matrix_rank(F,tol=1e-8))",
        "This numerical matrix is not a proof of statistical effective dimension or global reachability.",
    ),
    (
        "Model serialization",
        "JSON retains topology, stored values and frozen metadata without executing expressions or loading pickle objects.",
        "import tempfile\nfrom pathlib import Path\nm = pqml.MuTA(1)\nm.freeze('alpha.w0.c1',0.3)\nwith tempfile.TemporaryDirectory() as d:\n    p = Path(d)/'model.json'\n    m.save(p)\n    restored = pqml.MuTA.load(p)\n    assert np.allclose(m.unitary(),restored.unitary())\nprint('Logical model round trip verified')",
        "Fitted classifier heads and optimizer state are not included in MuTA.save.",
    ),
    (
        "Custom observable readout",
        "A logical output expectation is evaluated against a finite Hermitian matrix in wire order. Hermiticity and dimensions are checked.",
        "m = pqml.MuTA(1)\nr = m.run([1,0])\nvalue = r.expectation(np.diag([1,-1]))\nassert np.isclose(value,1)\nprint(value)",
        "Physical quadratures, photon number and parity use PhotoGraphiQ state readouts and different units.",
    ),
    (
        "Custom loss",
        "A deterministic scalar callable can combine model predictions with task-specific targets. Here a quadratic objective verifies the optimizer interface.",
        "target = np.array([0.5,-0.4])\ndef loss(p):\n    return float(np.mean((p-target)**2))\np,h = pqml.Trainer('lbfgs',epochs=20).fit(loss,[0,0])\nassert loss(p)<1e-10\nprint(round(loss(p),8))",
        "The trainer cannot infer whether a custom loss is scientifically meaningful or whether a supplied gradient is valid.",
    ),
    (
        "Training callbacks",
        "Callbacks receive a copy of current parameters and cumulative history, including the initial point. They can record experiment diagnostics.",
        "seen = []\ndef callback(p,h):\n    seen.append((len(h.losses)-1,float(np.linalg.norm(p))))\np,h = pqml.Trainer(epochs=3).fit(lambda p: np.sum(p*p),[1],callback=callback)\nassert len(seen)==4\nprint([step for step,norm in seen])",
        "Early stopping through callback return values and optimizer checkpoint restoration are not implemented.",
    ),
    (
        "CV teleportation cell (CVMuTA proposal)",
        "A derived CV teleportation step has T(k)=[[-k,-1],[1,0]] with k=cot(theta). Four zero-shear steps are ideal identity, a potential composable cell ingredient.",
        "from photographiq.gaussian import teleportation_matrix\nT = teleportation_matrix(0)\nassert np.allclose(np.linalg.matrix_power(T,4),np.eye(2))\nprint('CV cell ingredient verified; CVMuTA is a proposal')",
        "This is not an implemented CVMuTA. Its physical channel, expressivity and non-Gaussian extension require independent derivation after GKP validation.",
    ),
    (
        "Comparing qubit, GKP and CV MuTA",
        "Logical fidelity, finite-codeword projection and Gaussian channel noise answer different questions. Keep each metric attached to its representation.",
        "r = pqml.GKPBridge(cutoff=24,grid_points=2049).diagnostics()\nassert not r['physical_muta_validated']\nassert all(0<w<=1.000001 for w in r['captured_weights'])\nprint('GKP captured weights:',np.round(r['captured_weights'],5))",
        "Projection convergence does not certify GKP-MuTA gate convergence; MentPy does not validate raw CV physics.",
    ),
]


def main():
    tutorials = ROOT / "docs" / "ml" / "tutorials"
    examples = ROOT / "examples" / "ml" / "tutorials"
    notebooks = ROOT / "notebooks" / "ml"
    for directory in (tutorials, examples, notebooks):
        directory.mkdir(parents=True, exist_ok=True)
    index = [
        "# Tutorials",
        "",
        "These concise tutorials execute supported paths and clearly identify research gates.",
        "The tutorials on autodiff, learned instruments, GKP and CVMuTA describe current",
        "limitations; their prerequisite examples do not complete those research stages.",
        "",
    ]
    for number, (title, theory, example, limitation) in enumerate(LESSONS, 1):
        stem = f"{number:02d}"
        code = PRELUDE + example + "\n"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(code, f"tutorial-{stem}", "exec"), {})
        (examples / f"{stem}.py").write_text(code, encoding="utf-8")
        figure = (
            "gkp_resource"
            if number in (19, 30)
            else "entanglement"
            if number == 22
            else "triangle"
            if number == 4
            else "muta"
        )
        content = f"# {number:02d} {title}\n\n## Goal\n\nExplore {title.lower()} with an explicit representation contract.\n\n## Theory\n\n{theory}\n\n## Code\n\nRun from the installed repository environment.\n\n```python\n{code}```\n\n## Output\n\nCaptured from this example during documentation generation:\n\n```text\n{output.getvalue()}```\n\n## Visualization\n\n![Related resource or logical graph](../assets/{figure}.svg)\n\nThe figure shows {'finite GKP resource projection' if figure == 'gkp_resource' else 'logical resource structure or its entanglement diagnostic'}; it is not a hardware layout.\n\n## Validation\n\nThe assertions above are executed by `scripts/ml/execute_tutorials.py`. The scientific\nreference hierarchy is documented in [the mapping](../research/muta-mapping.md).\n\n## Limitations\n\n{limitation}\n\n## Things to try\n\nChange the explicit numerical values or supported model size, rerun the assertions,\nand explain whether the expected identity or diagnostic should still hold. Record\nthe seed and representation whenever comparing results.\n"
        (tutorials / f"{stem}.md").write_text(content, encoding="utf-8")
        index.append(f"- [{title}]({stem}.md)")
    (tutorials / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    for lesson, name in [
        (3, "01_quickstart"),
        (4, "02_muta"),
        (10, "03_gate_learning"),
        (14, "04_quantum_kernel"),
        (30, "05_gkp_resources"),
    ]:
        title, theory, code, limitation = LESSONS[lesson - 1]
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(f"# {title}\n\n{theory}"),
                nbformat.v4.new_code_cell(PRELUDE + code),
                nbformat.v4.new_markdown_cell(
                    f"## Validation and limitations\n\nAssertions test the documented behavior.\n\n{limitation}\n\nSee the corresponding tutorial for the graph and validation hierarchy."
                ),
            ]
        )
        notebook.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nbformat.write(notebook, notebooks / f"{name}.ipynb")
    (ROOT / "docs" / "ml" / "tutorial-manifest.json").write_text(
        json.dumps(
            {
                "tutorials": len(LESSONS),
                "notebooks": 5,
                "scope": "Supported examples and explicit research boundaries",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
