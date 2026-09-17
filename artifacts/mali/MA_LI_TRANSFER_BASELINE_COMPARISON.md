# Ma--Li transfer-learning control: pretrained versus from-scratch

Run date: 2026-09-16
Reference: Ma and Li, *Understanding and Estimating the Execution Time of
Quantum Circuits* (arXiv:2411.15631)

## Scope

This is an additional control experiment for the public-data reproduction. It
does **not** replace the paper's two-stage protocol. Both runs use the same
340 recorded Osaka/Kyoto labels, the same locally generated FakeOsaka/FakeKyoto
feature artifact, model, optimizer, seed, folds, and stopping rule. The only
change is model initialization:

| Run | Initialization |
|---|---|
| Pretrained | simulator checkpoint (`full_best.pt`) |
| From scratch | random initialization |

The backend properties are a versioned fake snapshot, not the historical IBM
calibration state. Therefore these numbers are an offline adaptation proxy,
not the paper's exact real-QPU result.

## Protocol

```text
data:        data/osaka_kyoto_fake_standardization.npy
labels:      192 Osaka + 148 Kyoto (340 total)
split:       10 held-out folds; 272 train / 34 validation / 34 test per fold
seed:        1234
model:       3 Graph Transformer layers, graph + global features
optimizer:   Adam, lr=5e-4, weight_decay=1e-4
epochs:      500
batch:       32
device:      RTX 5070 Ti
```

The from-scratch run was executed with:

```bash
.venv-mali-gpu/bin/python experiments/adapt_mali_real_qpu.py \
  --data data/osaka_kyoto_fake_standardization.npy \
  --manifest data/osaka_kyoto_fake_manifest.csv \
  --init random --properties-kind fake --folds 10 --epochs 500 \
  --batch-size 32 --seed 1234 --device cuda --cpu-threads 8 \
  --output-dir experiments/results/adaptation_fake_full_from_scratch
```

## Aggregate held-out results

Values are mean +/- standard deviation across the 10 test folds.

| Metric | Pretrained | From scratch | From scratch - pretrained |
|---|---:|---:|---:|
| MSE (s^2) | 1.8698 +/- 4.3644 | 0.3739 +/- 0.1092 | -1.4958 |
| RMSE (s) | 0.9609 +/- 0.9729 | 0.6057 +/- 0.0842 | -0.3552 |
| MAE (s) | 0.5483 +/- 0.1894 | 0.4783 +/- 0.0564 | -0.0700 |
| R^2 | 0.5922 +/- 0.8317 | **0.8855 +/- 0.0461** | **+0.2933** |
| NMSE | 0.4078 +/- 0.8317 | 0.1145 +/- 0.0461 | -0.2933 |
| log MAE | 0.0554 +/- 0.0098 | 0.0522 +/- 0.0070 | -0.0032 |
| log RMSE | 0.0803 +/- 0.0328 | 0.0671 +/- 0.0136 | -0.0132 |

## Fold-level R^2

| Fold | Pretrained | From scratch |
|---:|---:|---:|
| 1 | 0.9002 | 0.9252 |
| 2 | 0.8497 | 0.8265 |
| 3 | 0.8905 | 0.9134 |
| 4 | 0.8566 | 0.8895 |
| 5 | 0.9343 | 0.9363 |
| 6 | 0.8409 | 0.8474 |
| 7 | 0.7616 | 0.8035 |
| 8 | 0.9228 | 0.9209 |
| 9 | 0.8644 | 0.8568 |
| 10 | **-1.8991** | **0.9358** |

The pretrained mean is dominated by fold 10. The from-scratch run has no
negative fold and has substantially lower fold-to-fold variance.

The aggregate gap should not be interpreted as a uniform improvement on every
fold. Excluding fold 10, the mean R² is `0.8690` for pretrained and `0.8799`
for from-scratch (difference `+0.0109`), while the full-run difference is
`+0.2933`. Thus most of the apparent transfer penalty is a single catastrophic
outlier; the remaining folds are broadly comparable.

## Fold-10 qwalk diagnostic

The held-out fold contains `qwalk-noancilla_indep_qiskit_9` on Osaka:

```text
actual:       13.6962 s
pretrained:   36.0312 s
from scratch: 12.8724 s
```

For the pretrained model, this one error contributes 498.85 of the fold's
508.59 squared-error total (about 98.1%), producing R^2 = -1.8991. The
from-scratch model predicts the same sample within 0.824 s, giving fold-10
R^2 = 0.9358. The simulator labels for this circuit are much larger
(approximately 42.43 s on FakeWashington and 73.24 s on FakeSherbrooke) than
the real-QPU labels (approximately 13.70 s), so simulator pretraining can
transfer the wrong runtime scale for this circuit.

The split itself is identical for both runs, and it contains the Kyoto copy of
the same qwalk circuit in training while the Osaka copy is in test. For this
circuit, the local feature artifact has identical global features and
identical non-calibration graph columns across the two backends; only the four
T1/T2 columns differ. The saved fold-10 models make the following predictions:

| Model/input | Kyoto train | Osaka test |
|---|---:|---:|
| Pretrained, full features | 13.355 s | 36.031 s |
| From scratch, full features | 13.461 s | 12.872 s |
| Pretrained, T1/T2 set to standardized mean | 19.402 s | 19.402 s |
| From scratch, T1/T2 set to standardized mean | 13.213 s | 13.213 s |

This intervention is not a retraining experiment, but it is strong evidence
that the pretrained model is using the backend-dependent T1/T2 values to create
the Osaka/Kyoto divergence, whereas the from-scratch model is comparatively
insensitive to them. It explains why seeing the Kyoto qwalk row in training
does not protect the pretrained model: the model fits the Kyoto point but
extrapolates incorrectly when the Osaka calibration features change.

## Interpretation and guardrails

1. On this exact fake-snapshot/prose-8:1:1 protocol, random initialization
   outperformed the available simulator-pretrained checkpoint: R^2 0.8855
   versus 0.5922, with much lower variance.
2. This does **not** prove that pretraining is harmful in general. It is one
   seed, one feature reconstruction, one fake calibration snapshot, and 340
   small real-label observations.
3. The result shows that the current simulator-to-real transfer can impose a
   harmful prior for extreme circuits. A from-scratch control was necessary to
   reveal that the earlier low mean was not simply a limit of the DAG model.
4. Neither result should be compared directly with the paper's reported
   real-QPU R^2 ~0.905 as an exact replication: the historical properties,
   original processed tensor/order, and the paper/code split discrepancy are
   unresolved.

## Assessment of the missing paper baseline

The absence of a real-QPU from-scratch control is a methodological omission if
the goal is to attribute accuracy or sample efficiency to transfer learning.
The paper's reported R² is still a valid result for its proposed pipeline, but
without a matched random-initialization run it cannot establish that
pretraining caused the improvement, or even that it was better than direct
training on the 340 labels.

The local evidence points to three interacting explanations for the observed
difference:

1. **Simulator-to-QPU scale mismatch.** The qwalk circuit is approximately
   42.43/73.24 s on FakeWashington/FakeSherbrooke but 13.70 s on Osaka. The
   pretrained output head therefore starts with a strong but misleading
   high-runtime prior; the from-scratch model learns directly from the real
   target scale.
2. **Feature normalization/domain mismatch.** The simulator checkpoint and
   Osaka/Kyoto artifact are standardized from different populations. This is
   required by the available public data pipeline, but it is not guaranteed to
   preserve the input scale expected by the pretrained weights.
3. **Small, strict held-out folds.** Each test fold has only 34 rows and the
   local manuscript-style protocol trains on 272 real samples. A rare extreme
   circuit can therefore dominate one fold, while a larger 306-row
   code-style training fold might behave differently.

These are hypotheses supported by the diagnostics, not a proof of the paper
authors' internal cause. A definitive attribution would require the historical
processed tensor and a matched multi-seed comparison under both the paper prose
protocol and the public repository's CV implementation.

## Artifacts

- Pretrained aggregate: `experiments/results/adaptation_fake_full/fake_pretrained_result.json`
- From-scratch aggregate: `experiments/results/adaptation_fake_full_from_scratch/fake_random_result.json`
- From-scratch fold checkpoints/predictions: `experiments/results/adaptation_fake_full_from_scratch/`
- Runner: `experiments/adapt_mali_real_qpu.py`
