**R56 - qubit ZX-calculus subset: rewrites are exact including scalars.** Top: textbook diagrams
(green Z, red X spiders, phases in units of pi; dashed blue edges with yellow boxes are Hadamard
edges; black dots are boundaries, inputs left, outputs right). Bottom: after the library rewrite;
the exact scalar factor recorded by the rewrite and the deviation of the after-matrix from the
before-matrix are shown. Matrices come from an independent einsum evaluator that builds X spiders
directly from |+>/|-> projectors, compared with no global-phase or normalisation freedom. Over all
621 checks (textbook rules, 600 random rewrites on random diagrams with rule counts
{'fuse': 140, 'identity': 55, 'color-change': 288, 'self-loop': 25, 'hopf': 63, 'hadamard-pair': 17, 'copy': 9, 'bialgebra': 3}, `simplify_spiders` on graph-state diagrams vs CZ statevectors, and MBQC branch diagrams
vs `run_branch` Kraus operators) the worst relative deviation is 1.1e-15 (rewrites),
3.2e-16 (library vs independent
evaluator), 2.3e-16 (graph state / MBQC semantics). Qubit ZX subset only:
no CV-ZX, pi-copy or ZX local complementation/pivoting. Source: R56.
