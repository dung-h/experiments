# Ma--Li DAG runtime prediction: local reproduction

Run date: 2026-09-11
Reference: Ma and Li, *Understanding and Estimating the Execution Time of
Quantum Circuits*, arXiv:2411.15631.

**Document status:** verified local reproduction checkpoint (2026-09-12). This
file is the canonical source for the metrics and scope of the experiments
listed below. It must not be read as claiming an exact or universal estimator.

**Data provenance:** upstream Ma--Li repository commit
`32c392a6ece276f1ff046d4e30052d0571ff6dc6`; local code and generated artifacts
are separate working-tree additions. The upstream `data/` tree contains the
public QASM files and runtime CSVs; generated `.npy` files are local artifacts
and are not silently treated as upstream data.

## What was reproduced

The inputs are the public Ma--Li replication-package data: the 1,510 QASM
circuits and the four runtime CSVs (1,510 rows for each simulator, plus 192
Osaka and 148 Kyoto real-device rows). The CSV labels and circuit files are
used directly; no new QPU jobs were submitted.

The public repository does not contain the processed graph tensors,
`total_ibm_standardization.npy`, or historical Osaka/Kyoto calibration JSON.
Consequently, simulator features are rebuilt from the public QASM and fake
backend snapshots, while the 340 real-device labels are adapted using the
offline fake-snapshot path described below. This is a reproducible public-data
rebuild plus an offline proxy for historical backend metadata, not a
bit-for-bit reconstruction of the authors' missing processed artifact.

The local runner uses the model structure described by Ma--Li:

- 3 Graph Transformer (`TransformerConv`) layers over 178-dimensional node
  features;
- global average pooling followed by the 41-dimensional global feature MLP;
- four fully connected layers producing one runtime prediction;
- fixed seeded 8:1:1 split, Adam (`5e-4`, `1e-4` weight decay), 500 epochs;
- raw-seconds MSE, NMSE and R-squared, plus MAE/RMSE and log-runtime metrics.

The paper reports, for the combined FakeWashington/FakeSherbrooke simulator
experiment (3,020 points), MSE `0.081`, R-squared `0.943`, and NMSE `0.057`.
It describes 178 graph features, 41 global features, three Graph Transformer
layers, and a fixed 8:1:1 split. See the paper's model and evaluation
sections: <https://arxiv.org/html/2411.15631#S5.SS2>.

This is only the **pretraining** stage of the paper's transfer-learning
experiment. Ma--Li first save this simulator checkpoint, then initialise a
new model from it and fine-tune on 340 real-QPU observations (192 Osaka and
148 Kyoto). Their real-device evaluation uses ten folds; in each fold the
held-out tenth is the test set and the other nine tenths are split 8:1 into
fine-tuning train/validation data. The protocol is described in the paper's
evaluation section: <https://arxiv.org/html/2411.15631#S5.SS4>.

## Local protocol

| Item | Local value |
|---|---|
| Samples | 3,020 (1,510 Washington + 1,510 Sherbrooke) |
| Target | Ma--Li `time_taken`, backend-labelled simulator runtime |
| Seed | 1234 |
| Split | 2,416 / 302 / 302 |
| Layers | 3 |
| Batch | 32 (128 OOM on the 16 GB GPU) |
| Epochs | 500 |
| Runtime device | RTX 5070 Ti, CUDA 12.8 |
| PyTorch/PyG | 2.7.1 / 2.6.1 |

The generated graph artifact uses the memory-safe in-place standardization
pipeline. Global features are standardized per feature; this avoids the
upstream helper's flattening of all global vectors into one scalar stream, so
the run is architecturally equivalent but not byte-for-byte identical to the
original helper implementation.

## Canonical artifacts and integrity

Use the following files when producing follow-up reports. Do not substitute a
smoke-run directory for the full-run directories.

| Purpose | Canonical artifact | SHA-256 |
|---|---|---|
| Simulator metrics | `experiments/results/full_result.json` | `43b6871491627f9b6a0980940c57d8117f1a793c865c041ea61b6d45bf5e26eb` |
| Real-label proxy metrics | `experiments/results/adaptation_fake_full/fake_pretrained_result.json` | `4c9ff24d1ebfbad8194991ebabe852472ed4d5c16f39a6cb17b8a1be725b0c6d` |
| Simulator manifest | `data/training_data_manifest.csv` | `58ab3bb9003361875349082f221689c04f7cc8f070eecfcc00af0cb6e5822500` |
| Osaka/Kyoto proxy manifest | `data/osaka_kyoto_fake_manifest.csv` | `9363951b35bd840b9a5e2a3ba31bd34346ca2da9e9e826da8aec159e3a89a223` |

The large `.npy` tensors are generated inputs for these runs, not files from
the upstream public repository. Their manifests, model initialization mode,
backend-property kind, seed, split and device must be reported together with
any metric.

## Metrics

The best validation checkpoint was epoch 67:

| Split/checkpoint | MSE (s²) | RMSE (s) | MAE (s) | R² | NMSE | log RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Best validation | 0.1067 | 0.3266 | 0.1358 | 0.9882 | 0.0118 | 0.0985 |
| Independent test | 1.4653 | 1.2105 | 0.2267 | 0.4262 | 0.5738 | 0.1410 |

The test prediction was evaluated with the best validation checkpoint, not the
last epoch. The full training trajectory and predictions are in
`experiments/results/`.

## Why the test R-squared is low

The 302-point test fold contains unusually large Sherbrooke runtimes that the
model underpredicts:

| Circuit | Actual (s) | Predicted (s) | Absolute error (s) |
|---|---:|---:|---:|
| `qpeinexact_indep_qiskit_30` | 22.623 | 3.589 | 19.034 |
| `portfolioqaoa_indep_qiskit_10` | 12.968 | 5.634 | 7.334 |

Removing only the largest one of these errors raises test R² from `0.426` to
`0.740`; removing both raises it to `0.841`. This is an audit diagnostic, not
an acceptable way to report the final metric.

The same trained model evaluated on alternative 10%-fold indexings gives R²
from about `0.944` to `0.986`, while the seed-1234 fold gives `0.426`. This
shows that a row-level random split is highly sensitive to rare runtime
outliers and backend tails. It also explains how a result near the paper's
`0.943` can coexist with a poor test fold without changing the model.

## Conclusion

The DAG Graph Transformer contribution is working: it reaches validation
R² `0.988` and low log-runtime error on the current extract. However, the
paper's simulator headline is **not yet reproduced as a defensible independent
test result** under this exact local split: test R² is `0.426`.

The paper's reported real-QPU result (R² about `0.905`) must not be compared
with the local simulator-only `0.426`: they are different stages. The offline
real-label adaptation has now been run using the public 340 labels and a
versioned fake Osaka/Kyoto snapshot. Its aggregate held-out R² is `0.5922`
with a large fold standard deviation (`0.8317`), so it is not evidence of a
universal estimator and is not claimed as an exact reproduction of the paper.
The exact historical real-QPU feature tensor remains unavailable because the
public package does not ship the calibration snapshot used when those jobs
were collected.

Before claiming replication, the next checks are:

1. reproduce the paper's original circuit-row ordering and preprocessing
   byte-for-byte, including its global-feature standardization;
2. run the same architecture across multiple fixed seeds and report mean/std;
3. report source/device-held-out and circuit-family/workflow-group splits;
4. if the authors' historical Osaka/Kyoto properties or processed tensor are
   obtained, rerun the same 340-point fine-tuning with that artifact; otherwise
   retain the fake-snapshot result as an explicitly labelled offline proxy and
   compare it against a from-scratch baseline.

The executable is `experiments/replicate_mali_dag.py`; resume checkpoints are
written under `experiments/results/`.

## Offline real-label adaptation path

The 340 real labels do not require another QPU submission. The repository now
also contains versioned `FakeOsaka` and `FakeKyoto` property snapshots under
`data/fake_backend_properties/`. These snapshots provide the 127-qubit
topology and T1/T2 metadata needed by the DAG encoder; the target remains the
already-recorded Osaka/Kyoto `time_taken` value. The resulting run is named
`fake_snapshot_adaptation`, because the snapshot date is not guaranteed to be
the same calibration state used when the historical measurements were made.

The graph builder accepts this cache through
`--backend-properties-dir`, and `experiments/adapt_mali_real_qpu.py` loads the
simulator checkpoint and performs the paper's 10-fold 8:1:1 fine-tuning. A
five-epoch one-fold smoke run completed successfully (test R² `0.953`); this
was only an implementation check.

The full fake-snapshot adaptation (10 folds, 500 epochs, batch 32, pretrained
initialisation) completed in 2.18 hours on the RTX 5070 Ti. Aggregate held-out
metrics are:

| Metric | Mean | Std |
|---|---:|---:|
| MSE (s²) | 1.8698 | 4.3644 |
| RMSE (s) | 0.9609 | 0.9729 |
| MAE (s) | 0.5483 | 0.1894 |
| R² | 0.5922 | 0.8317 |
| NMSE | 0.4078 | 0.8317 |
| log MAE | 0.0554 | 0.0098 |
| log RMSE | 0.0803 | 0.0328 |

Nine folds have R² between `0.762` and `0.934`; fold 10 is `-1.899` because
`qwalk-noancilla_indep_qiskit_9` is overpredicted (`13.70s` actual vs `36.03s`
predicted). Thus the mean R² is not a stable universal-estimator claim. The
raw fold checkpoints and metrics are under
`experiments/results/adaptation_fake_full/`. A matched from-scratch control
has now been run on the same fake-snapshot artifact and protocol: its aggregate
held-out R² is `0.8855 ± 0.0461`, versus `0.5922 ± 0.8317` for the pretrained
run. This is an offline control result, not evidence that transfer learning is
generally harmful; it shows that the current simulator checkpoint can impose a
harmful prior for an extreme real-device circuit. See
`experiments/MA_LI_TRANSFER_BASELINE_COMPARISON.md` for the full comparison.

The fold-10 failure is not caused by a different test split between the two
initializations. Both models use the same indices: Kyoto's copy of
`qwalk-noancilla_indep_qiskit_9` is in training and Osaka's copy is in test.
For this circuit, the local global features and all non-calibration graph
columns are identical across the two devices; only the four T1/T2 columns
differ. The pretrained model predicts 13.355 s for Kyoto but 36.031 s for
Osaka, whereas the from-scratch model predicts 13.461 s and 12.872 s. Setting
T1/T2 to their standardized mean makes the pretrained predictions equal at
19.402 s, but changes the from-scratch predictions only to 13.213 s. This is
strong diagnostic evidence that the pretrained model is using backend-dependent
T1/T2 values to extrapolate a simulator-derived runtime prior; it is not proof
that T1/T2 are intrinsically predictive of real-QPU runtime.

## Reporting guardrails

- Quote `0.426` only as the seed-1234 simulator independent-test R² for this
  local preprocessing, not as the paper's simulator headline.
- Quote `0.5922 ± 0.8317` only as the fake-snapshot, pretrained, 10-fold
  offline adaptation result; it is not the paper's `0.905` real-QPU result.
- Keep simulator, real-QPU, and fake-snapshot proxy targets distinct. Never
  collapse them into a universal `runtime_seconds` target.
- Preserve the run metadata (dataset commit, artifact path, backend-property
  source, seed, split, epochs, batch size and software versions) when adding
  new experiments.

## Changelog

- **2026-09-12:** audited the upstream repository tree and git history; added
  the public-data/proxy distinction, canonical artifact hashes and reporting
  guardrails. Confirmed that the full 10-fold fake-snapshot adaptation had
  completed and that no additional QPU jobs were submitted.
- **2026-09-11:** completed the 500-epoch simulator pretraining and the full
  500-epoch, 10-fold offline adaptation run.
- **2026-09-16:** completed the matched random-initialization control. The
  from-scratch run achieved R² `0.8855 ± 0.0461`, while the pretrained
  fake-snapshot run achieved `0.5922 ± 0.8317`; the difference is dominated by
  the held-out qwalk outlier in pretrained fold 10.
