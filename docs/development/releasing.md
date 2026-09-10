# Release procedure

Run all validation gates, executable docs and notebook checks. Review actual measured coverage and remote matrix results. Build wheel and sdist, install the wheel into an isolated environment and run a smoke example. Synchronize pyproject, package version, citation and changelog. GitHub Pages and Codecov require repository-side setup. Release tags and package publication should follow successful review; a local version bump is not publication.

See [numerical policy](../validation/numerical-policy.md), [API](../api/index.md) and the [feature matrix](../validation/feature-matrix.md).
