# Repository conventions

This checkout contains methodology, aggregate QPU results and local simulator
evidence. Do not add third-party QPU observations or their per-row derivatives.
Keep source QASM and serialized source circuit graphs external.

Preserve recorded labels, predictions, splits and execution contracts. Compare
methods on the same successful held-out rows within an engine, clock and quality
context. Keep adaptations distinct from paper-exact reproduction.

Use semantic filenames. Historical IDs belong in provenance metadata, not new
dated report folders. Keep logs, caches, temporary outputs and environments out
of Git. Update `provenance/files.json` when published files change.

`scripts/verify.py` checks the public inventory and documentation. It does not
authorize training, simulator timing or QPU execution.
