# PhotoGraphiQ manuscript

`PhotoGraphiQ.tex` is the complete software paper draft. `quantumarticle.cls` is
the unmodified official Quantum document class (LPPL 1.3c or later, see its header).
The draft has collective contributor metadata; actual authors and affiliations
must be finalized by the authors before submission. AI assistance is disclosed.

From the repository root:

```sh
python experiments/reproduce.py
python -m pytest --junitxml=paper/results/tests.xml
cd paper
pdflatex -interaction=nonstopmode -halt-on-error PhotoGraphiQ.tex
pdflatex -interaction=nonstopmode -halt-on-error PhotoGraphiQ.tex
```

No BibTeX step is needed: DOI-linked references are embedded. The generated
CSV and JSON contain actual simulation data and exact dependency versions.
The plotted fidelity is analytical, while covariance markers are Monte Carlo
estimates. PDF production is typesetting, not journal acceptance or peer review.

Alternatively, `tectonic --keep-logs paper/PhotoGraphiQ.tex` from the repository
root builds the PDF with a portable TeX distribution. The source supports both
pdfLaTeX and the XeTeX engine used by Tectonic.
