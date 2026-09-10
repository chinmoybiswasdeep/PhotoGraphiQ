# PhotoGraphiQ manuscript

`PhotoGraphiQ-v0.2.pdf` is the revised manuscript including the non-Gaussian extension.
`PhotoGraphiQ.tex` (including `non_gaussian.tex`) is the complete software paper draft.
`PhotoGraphiQ.pdf` is the previous Gaussian-focused manuscript retained because
it was locked by an open PDF viewer during this update. `quantumarticle.cls` is
the unmodified official Quantum document class (LPPL 1.3c or later, see its header).
The draft has collective contributor metadata; actual authors and affiliations
must be finalized by the authors before submission. AI assistance is disclosed.

From the repository root:

```sh
python experiments/reproduce.py
python experiments/non_gaussian.py
python -m pytest --junitxml=paper/results/tests.xml
cd paper
pdflatex -interaction=nonstopmode -halt-on-error PhotoGraphiQ.tex
pdflatex -interaction=nonstopmode -halt-on-error PhotoGraphiQ.tex
```

No BibTeX step is needed: DOI-linked references are embedded. The generated
CSV and JSON contain actual simulation data and exact dependency versions.
The Gaussian finite-squeezing fidelity is analytical, while covariance markers
are Monte Carlo estimates. Non-Gaussian fidelity is computed from finite-Fock
state outputs at adjacent cutoffs; Wigner figures are finite-grid diagnostics. PDF production is typesetting, not journal acceptance or peer review.

Alternatively, `tectonic --keep-logs paper/PhotoGraphiQ.tex` from the repository
root builds the PDF with a portable TeX distribution. The source supports both
pdfLaTeX and the XeTeX engine used by Tectonic.
