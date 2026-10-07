# Reproduction scope

This checkout publishes methodology, aggregate QPU results and simulator
evidence measured by this project. It is not a self-contained real-QPU
training dataset or a one-command end-to-end replication package.

## What can be checked now

```bash
python3 scripts/verify.py
python3 -B -m unittest discover -s tests
```

The verifier checks file hashes, inventory completeness, local documentation
links and the data exclusions. It requires only Python's standard library.
The tests check these invariants on small fixtures. Neither command fits a
model, executes a simulator, downloads data or contacts a QPU.

The simulator tree includes local attempts, target reductions, quality records,
fold assignments, predictions and metrics. The two reader CSVs contain the
reported aggregate scores. The figures and slide deck are retained outputs;
the verifier checks their bytes, not their full regeneration.

## External inputs

The original Ma–Li, Qonductor and QPack real-QPU observations are not
redistributed. Their per-observation QPU features, graphs, partition ledgers
and predictions are also excluded. Original QASM archives, serialized source
DAG payloads and native SDK binaries are not bundled. A circuit filename,
source path or digest in a measurement record identifies an external input;
it is not the circuit itself.

Sources and pinned revisions are listed in
[upstream metadata](../provenance/upstream.json). The
[preprocessing guide](data_preprocessing.md) describes the source joins,
reconstruction tiers, unit conversions and missingness. The
[feature dictionary](data_and_feature_dictionary.md) specifies feature order
and transformations. Obtain the corresponding inputs from their sources before
attempting fresh preprocessing or QPU fitting.

## Retained implementations

`methods/` contains project implementations and adapters used in the study.
Original runner paths are retained in code and recorded commands. These files
explain the experiment; they are not all portable entry points for this reduced
checkout. External input trees, upstream helper modules and engine environments
must be resolved before execution. Missing data is never replaced with invented
runtime labels or silently imputed features.

Fresh neural fits require CUDA and the recorded partitions, seeds and
fit-only transforms. Simulator measurement requires the selected engine,
precision, shots, timeout, workspace, warmup and quality policy. Installing
measurement dependencies is not necessary to read or verify these results.
See [training and measurement](training_and_measurement.md).

The new Pasqal runner remains gated. Its reference threshold and resume paths
passed synthetic tests in the local evidence package, but no new measurement
was performed. The archived dependency-lock digest was a failed `pip freeze`
diagnostic, not a complete lock. A clean prospective CUDA environment still
needs to be resolved and verified before any new run. The saved 18-cell analog
pilot remains a companion study, not a digital-panel predictor.

## Evidence selection

[Publication metadata](../provenance/publication.json) records the source
snapshot and omitted files. The inventory (not bundled) hashes
this checkout. Earlier private-package replay receipts and independent
scientific reviews describe those earlier evidence snapshots, not a fresh
Git-clone training or simulator reproduction of this public subset.
