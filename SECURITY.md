# Security policy

Report suspected vulnerabilities privately through GitHub's **Security → Report
a vulnerability** feature when enabled. If unavailable, use GitHub Support to
request a private reporting route; do not publish exploit details in an issue.

Include the affected commit/version, Python and dependency versions, a minimal
reproduction, impact and any proposed fix. Dependency vulnerabilities and unsafe
serialization behavior are in scope. Patterns use allowlisted JSON, not pickle
or arbitrary expression evaluation; report any way to bypass those restrictions.

General numerical errors belong in the physics/numerical issue template. Security
fixes target the current development branch; older research snapshots may require
upgrading. No response-time guarantee is implied for this volunteer project.
