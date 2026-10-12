# Provenance and validation index

| File | Role |
| --- | --- |
| [files.json](files.json) | Live public byte-hash inventory, excluding itself |
| [upstream.json](upstream.json) | Upstream identities and external terms |
| [real_qpu.json](real_qpu.json) | Parent aggregate lineage; private inputs not bundled |
| [publication.json](publication.json) | Original reduced-publication receipt, not the whole new snapshot |
| [completed_science.json](completed_science.json) | Source/published hashes, semantic mapping and declared transformations |
| [complete_grid_publication.json](complete_grid_publication.json) | Byte-exact aggregate supplement and saved-row selection |
| [completed_science_validation.json](completed_science_validation.json) | Historical prepublication check counts/scope |

Source hashes identify immutable producing evidence/code in the development
workspace. Published hashes identify public copies/views. Markdown link
adaptation or aggregate-key selection may make them differ; retained scientific
values are unchanged and the transformation is declared.

Do not repair historical hashes to fit newer documentation.
[Closure erratum](../results/research_questions/closure/provenance_resolution.md)
resolves earlier namespace false positives and pathless declarations.
[Public checks](../tests/README.md) establish saved-evidence consistency, not
independent QPU retraining. Private source paths are provenance, not downloads.

Navigation may refresh files.json. Historical export receipts, frozen
protocols, metric tables and source mappings remain unchanged.
