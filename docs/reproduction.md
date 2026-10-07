# Reproduction scope

This checkout publishes methodology, aggregate QPU results and simulator
evidence measured by this project. It is not a self-contained real-QPU
training dataset or a one-command end-to-end replication package.

| Operation | Current public-checkout support |
| --- | --- |
| File integrity and documentation checks | Included, standard-library commands below |
| Saved simulator score checks | Included for all 49 predictor rows; checks published fields against retained targets/predictions |
| QPU score recomputation | Not supported here: row-level targets, predictions and split membership are excluded |
| Bootstrap or checkpoint replay | Not performed by the public checks; retained intervals and checkpoint metadata are historical evidence |
| Fresh extraction, neural fitting or simulator measurement | Not turnkey: external circuits/helpers, frozen inputs and engine environments must first be resolved |

## What can be checked now

```bash
python3 scripts/verify.py
python3 -B scripts/check_simulator_results.py
python3 -B -m unittest discover -s tests
```

The verifier checks file hashes, inventory completeness, local documentation
links and the data exclusions. It requires only Python's standard library.
The tests check these invariants on small fixtures. None of these commands fits a
model, executes a simulator, downloads data or contacts a QPU.

The numerical checker independently recomputes MAE, R², scored counts and
coverage for all 49 simulator predictor rows from saved OOF predictions and
target ledgers. It also checks other published error metrics where present,
exact-hash alias consistency, recorded folds and available three-seed reductions.
Dense neural records retain seed bounds rather than individual predictions, so
their bounds are checked without claiming to reconstruct those seed medians.
Fixed and joint MPS quality counts, the 18-cell analog quality record and 603
saved TN estimate/warm ratios are checked separately. This is a saved-evidence
check, not a new timing run, checkpoint replay or bootstrap reproduction.

The [counting rules](methodology.md#counting-evaluation-outcomes) distinguish
emitted predictions from scored targets. Predictor coverage is scored/assigned,
not necessarily predicted/assigned.

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

`protocol/reproduction.json` covers only the optional Pasqal source-construction
and measurement adapters. It is not a readiness registry for every method.
Source construction has a pinned Python function but no bundled command-line
dispatcher. It does not establish a working EMU-MPS timing environment.

The new Pasqal runner remains gated. Its reference threshold and resume paths
passed synthetic tests in the local evidence package, but no new measurement
was performed. The archived dependency-lock digest was a failed `pip freeze`
diagnostic, not a complete lock. A clean prospective CUDA environment still
needs to be resolved and verified before any new run. The saved 18-cell analog
pilot remains a companion study, not a digital-panel predictor.

## Evidence selection

[Publication metadata](../provenance/publication.json) records the source
snapshot and omitted files. [The inventory](../provenance/files.json) hashes
this checkout. Earlier private-package replay receipts and independent
scientific reviews describe those earlier evidence snapshots, not a fresh
Git-clone training or simulator reproduction of this public subset.

`source_sha256` in publication metadata identifies the imported source snapshot.
It is not a claim that maintained documentation and corrected reader metadata
still have those original bytes. The current inventory records their present
contents. The QCSim reader correction changes attempted/predicted counts for
three controls from 143 to 150 and unattempted counts from seven to zero.
Scored counts stay 143, coverage stays 143/150, and MAE/R², saved predictions,
labels, folds and historical manifests are unchanged.

## Release checks

Keep this selection separate from the development worktree. Do not add QPU
row-level data, source circuit archives or serialized source graphs while fixing
documentation. Recompute the current file inventory after authorized changes,
then run all three verification commands above. Test the exact pending release
commit in a fresh checkout before pushing, without installing simulator engines
or executing experiment runners. A historical clone receipt is not that check.
