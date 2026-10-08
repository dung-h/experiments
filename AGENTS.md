# Repository conventions

The only current real-QPU evaluation dataset is 4,515 observations:
Ma–Li 340, Qonductor 230 and QPack 3,945, in 165 conservative groups.
Use docs/data_preprocessing.md and protocol/real_qpu_dataset.json for selection
and input qualification. Do not restore other cohort counts or scores as current
results. Fresh-fit QPU execution and saved-evidence QA are complete. Publish
only the validated aggregate tables; do not launch another experiment while
editing reports or checking the publication.

This checkout publishes methodology, aggregate dataset information and local
simulator evidence. Do not add third-party QPU observations, row-level derivatives,
source QASM or serialized source circuit graphs. Preserve local source evidence
and frozen running contracts; publication changes must not alter an active run.

Compare methods on identical successful held-out rows, keep assigned denominators
and input/clock qualifications visible, and distinguish adaptations from original
methods. Never describe reconstructed recipes as exact historical logical QASM.

Use semantic filenames without task/date/version suffixes for maintained reports.
Keep logs, caches, temporary outputs and environments out of Git. Refresh
provenance/files.json after changes. scripts/verify.py checks public files and
documentation; it does not authorize fitting, timing or QPU execution.
