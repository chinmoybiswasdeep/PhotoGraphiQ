# Documentation development

Install the docs extra, then run `python -m mkdocs serve`. Build with `python -m mkdocs build --strict`. Tutorial pages include their executable source with snippets; update source and captured output together using `python scripts/check_docs_examples.py --write-outputs`. Add new tutorial metadata, explanation, interpretation and exercises. Generate figures with `python scripts/generate_docs_figures.py`. Notebook execution is checked with nbclient.

See [numerical policy](../validation/numerical-policy.md), [API](../api/index.md) and the [feature matrix](../validation/feature-matrix.md).
