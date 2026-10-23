# Post-freeze Track-A reports and code: distribution scope

This package publishes the completed sample-efficiency study's aggregate results, related raw-R²/target-boundary audits, tabular source/compiled comparison, and current planning documents. Exact producing Python source is retained. New method plans are explicitly drafts, not experimental results.

## Included

- Frozen aggregate effect/performance tables, fold/source/seed summaries and figures.
- Historical saved-run completion receipts and aggregate validation metadata.
- Source code for extraction, training, aggregation and independent tests.
- Audit prose, feature/target/split descriptions and explicit methodological limitations.
- An explicit file allowlist and byte-hash index, verifiable with standard-library Python.

## Local prerequisites deliberately excluded

- Ma–Li/Qonductor/QPack observation rows and row-level targets/predictions.
- QASM archives, serialized graphs, feature tables, split/subset membership and per-group error tables.
- Checkpoints and raw training/inference logs.
- Bootstrap archives containing group identities and per-group derivatives.

File references and hashes of omitted artifacts identify the local scientific evidence. They do not mean those files are distributed. The historical completion receipt is retained with its original hashes; it is not relabeled as a full public-reproduction receipt. In particular, the private saved-output tests require those omitted artifacts and will not pass in a public-only checkout. The public verifier checks byte integrity and consistency of the supplied aggregates only.

The sample-efficiency runner imports the frozen A1 runner and requires the pinned upstream Ma–Li checkout, qualified local panel, pretrained checkpoints and recorded environments. Minimal producing A1 source is supplied as a dependency; restricted data and model states are not. A new clone is not a self-contained training package.

## Report-package verification

From a checkout containing this package:

```bash
python3 scripts/verify_postfreeze_track_a_completion.py
git diff --check
```

These checks execute no model, simulator or QPU job. The verifier checks the selected file hashes, aggregate treatment deltas, original raw-R² candidates and presence/coverage metadata. It does not independently replay the private predictions.
