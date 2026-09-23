# Quantum Rings runtime-prediction replication: reporting summary

Report date: 2026-09-18  
Repository: `iQuHACK/2026-Quantum-Rings`  
Paper under replication: [arXiv:2606.11620](https://arxiv.org/abs/2606.11620)

This document is the short reporting path for the replication work. It separates
the paper's reported numbers from experiments rerun on the public repository,
and links every result to its raw data, script, and machine-readable output.

## 1. What was replicated

The paper describes a predictor that selects an approximation threshold and
estimates simulator runtime from an OpenQASM circuit and execution context.
The public artifact supplies the circuits, labels, and runner outputs, but not
the complete paper implementation or private holdout. The replication therefore
has three levels:

1. **Public-artifact replication:** reproduce the feature extraction, target
   construction, circuit-level split, baselines, and family-aware MLP.
2. **Protocol audit:** test whether the reported neural architecture is needed
   by adding Random Forest, ExtraTrees, Gradient Boosting, context-wise scores,
   and leave-one-family-out diagnostics.
3. **Public MQT pretraining:** replace the earlier fold-local family classifier
   with a frozen classifier trained on an independent 200-circuit MQT Bench
   set. This is a public approximation, not the authors' exact private set.

The Quantum Rings simulator itself was not rerun in this replication because
the public repository does not include the credentialed runner, the private
truth file, or the exact paper-protocol forward labels.

## 2. Data and target semantics

### Evaluation data

`data/hackathon_public.json` contains 36 OpenQASM circuits and four execution
contexts per circuit:

| Dimension | Values | Rows |
|---|---|---:|
| Circuit | 36 QASM files | 36 |
| Backend | CPU, GPU | 2 |
| Precision | single, double | 2 |
| Total | circuit × backend × precision | **144** |

The same QASM file appears in all four contexts. All four rows stay in the
same fold; this prevents a circuit from leaking through another backend or
precision row.

### Two runtime protocols

The paper and the public challenge do not use the same target:

| Protocol | Threshold target | Runtime label | Status |
|---|---:|---|---|
| Paper-style | mirror fidelity ≥ 0.75 | mirror-sweep wall time at the first 0.75 crossing | proxy; 140 rows have this source, 4 use documented fallback |
| Challenge/public | mirror fidelity ≥ 0.99 | 10,000-shot forward runtime at public selected threshold | available public target |

The 0.75 runtime is not a measured 10,000-shot forward runtime at 0.75. It
must be called a mirror-sweep proxy in any presentation. The 0.99 target is a
different simulator-runtime task and should not be pooled with the paper-style
result.

### Independent MQT pretraining data

The paper describes approximately 200 separate MQT Bench circuits for the
family classifier. The exact private files and weights are not published in
the Quantum Rings repository. We generated a transparent public approximation:

- 200 QASM circuits, 20 per family;
- DJ, GHZ, W-State, Graph State, Grover, QFT, QFT Entangled, QPE, QNN, VQE;
- 5–24 qubits, MQT Bench `alg` level;
- MQT Bench 2.2.3, Qiskit 2.5.2;
- deterministic per-circuit seeds and SHA-256 hashes in the manifest;
- zero duplicate QASM hashes against the 36 evaluation circuits.

Raw QASM and regeneration instructions are in
[`data/mqt_family_pretraining/`](data/mqt_family_pretraining/README.md).

## 3. Replication pipeline

### Feature extraction

Each QASM circuit is converted to the 32-feature representation implemented in
[`experiments/replicate_paper.py`](experiments/replicate_paper.py):

- circuit size and depth statistics;
- total, one-qubit, and two-qubit gate counts;
- gate composition and complexity ratios;
- execution context (CPU/GPU and single/double precision);
- interaction-graph and local-structure features;
- algorithm fingerprints;
- RCM/cut-pressure features.

Count-like features are transformed with logarithmic scaling before model
training. Runtime is modelled in log seconds; original-seconds MAE, R² and
median relative error are also reported.

### Splits and models

- circuit-level 5-fold cross-validation, seed 0;
- all four rows for one QASM remain in one fold;
- threshold classifier and runtime regressor are evaluated separately;
- baseline models: Random Forest, ExtraTrees, Gradient Boosting;
- neural ablations: no-family MLP, graph-feature MLP, family-conditioned MLP,
  and decayed cross-entropy;
- family-aware model: embedding + FiLM modulation + residual heads + direct
  feature shortcuts;
- MQT integration: MLP classifier trained once on the independent MQT set,
  frozen before evaluation, then compared with mapped oracle family labels.

## 4. Results rerun on the public artifact

### Paper-style target 0.75

The first table is the paper-aligned public-artifact result. Runtime is the
mirror-sweep proxy described above.

| Model | Threshold exact | Within ±1 rung | R²(log runtime) | Median relative error |
|---|---:|---:|---:|---:|
| Random Forest (`replicate_paper`) | 68.06% | 86.81% | 0.748 | 34.8% |
| MLP, no family | 47.92% | 75.69% | −0.593 | 78.5% |
| Family MLP, predicted family | 59.03% | 83.33% | −0.253 | 69.0% |
| Family MLP, oracle family | 61.11% | 80.56% | −0.512 | 77.3% |

The follow-up tree audit, using the same folds, found Gradient Boosting to be
the strongest runtime baseline (`R²(log) = 0.757`, median relative error
36.7%). Its threshold Random Forest reached 70.83% exact and 86.81% within one
rung.

### Challenge/public target 0.99

This is the public 10,000-shot forward-runtime task.

| Model | Threshold exact | Within ±1 rung | R²(log runtime) | Median relative error |
|---|---:|---:|---:|---:|
| Random Forest | 61.11% | 84.03% | 0.417 | 50.2% |
| ExtraTrees | not the threshold headline | not the threshold headline | **0.712** | **38.8%** |
| Gradient Boosting | not the threshold headline | not the threshold headline | 0.578 | 37.9% |
| MLP, no family | 40.97% | 67.36% | −0.786 | 67.7% |
| Family MLP, predicted family | 63.19% | 81.94% | −0.856 | 67.1% |
| Family MLP, oracle family | 61.11% | 77.08% | −1.095 | 68.9% |

The threshold and runtime columns come from the same public rows, but they are
different objectives. ExtraTrees is listed for runtime because it is the best
runtime regressor in the audit; it was not the threshold classifier headline.
The audit's incremental MLP variants produced runtime `R²(log)` values from
0.102 down to −0.891 depending on graph/family/loss configuration; none
exceeded the tree baselines.

### Context effect

The aggregate score hides strong backend/precision effects. For the paper-style
0.75 proxy, Random Forest runtime results were:

| Context | R²(log runtime) | Median relative error |
|---|---:|---:|
| CPU + double | 0.405 | 53.2% |
| CPU + single | 0.453 | 42.4% |
| GPU + double | 0.841 | 24.3% |
| GPU + single | 0.888 | 22.8% |

For the public 0.99 forward target, the corresponding R²(log) values were
0.329, 0.478, 0.284 and 0.389. This context reversal is evidence that the two
runtime targets should not be treated as one universal simulator-runtime label.

### Public MQT classifier and frozen integration

The independent MQT classifier used 160 training and 40 held-out circuits.
The MLP reached **97.5% (39/40)**; a Random Forest reference reached 90.0%.
The exact held-out predictions are in
[`results/mqt_family_classifier/heldout_predictions.csv`](results/mqt_family_classifier/heldout_predictions.csv).

When the frozen classifier was applied to the 36 Quantum Rings evaluation
circuits, only 26/36 had labels that mapped to the ten-family vocabulary:

| Target | Family input | Threshold exact | Within ±1 | R²(log runtime) | Median relative error |
|---|---|---:|---:|---:|---:|
| 0.75 | MQT-predicted | 56.94% | 81.94% | −0.198 | 63.8% |
| 0.75 | mapped oracle | 58.33% | 74.31% | −0.317 | 79.3% |
| 0.99 | MQT-predicted | 52.78% | 78.47% | −0.010 | 64.0% |
| 0.99 | mapped oracle | 57.64% | 69.44% | −0.842 | 65.8% |

The MQT classifier itself scored 33.33% over all evaluation circuits and
46.15% among mapped known families. Thus the independent pretraining removes
evaluation-set leakage, but it does not establish cross-distribution family
transfer.

## 5. Main findings for the presentation

1. The public artifact can be reproduced at the data, feature, split, and
   baseline level. The paper's headline numbers cannot be claimed as exactly
   reproduced because the private runner, weights, 0.75 forward labels, and
   holdout truth are absent.
2. Tree models are the strongest runtime baselines on the public data. The
   reconstructed family-aware neural model is not shown to be necessary for
   runtime prediction.
3. Family conditioning helps threshold classification more consistently than
   runtime regression. The improvement is not robust under the independent MQT
   family classifier.
4. CPU/GPU and precision are not cosmetic metadata. Their per-context scores
   differ substantially, and the ordering changes between mirror proxy and
   forward runtime.
5. The public MQT classifier can reproduce the reported **97.5% scale** on a
   transparent held-out set, but this numerical match is not evidence that the
   authors' private set or weights were recovered.
6. The defensible estimator claim is domain-specific: a simulator-runtime
   predictor for this public Quantum Rings artifact. It is not a universal
   runtime estimator and must not be pooled with QPU execution or cloud
   turnaround targets.

## 6. Raw data, reports, and result files

| Purpose | Raw/source artifact | Reproduction script | Result/report |
|---|---|---|---|
| Public challenge labels | [`data/hackathon_public.json`](data/hackathon_public.json) | [`replicate_paper.py`](experiments/replicate_paper.py) | [`results/paper_replication.json`](results/paper_replication.json) |
| Public holdout schema | [`data/holdout_public.json`](data/holdout_public.json) | validation scripts only; truth/QASM are organizer-private | no local leaderboard score |
| 32-feature / paper-style replication | `circuits/*.qasm` | [`experiments/replicate_paper.py`](experiments/replicate_paper.py) | [`REPLICATION_REPORT.md`](REPLICATION_REPORT.md) |
| Protocol audit | same public JSON/QASM | [`experiments/protocol_audit.py`](experiments/protocol_audit.py) | [`results/PROTOCOL_AUDIT_REPORT.md`](results/PROTOCOL_AUDIT_REPORT.md), [`results/protocol_audit.json`](results/protocol_audit.json) |
| Independent family pretraining | [`data/mqt_family_pretraining/`](data/mqt_family_pretraining/) | [`generate_mqt_family_pretraining.py`](experiments/generate_mqt_family_pretraining.py) | [`results/MQT_FAMILY_PRETRAINING_REPORT.md`](results/MQT_FAMILY_PRETRAINING_REPORT.md) |
| Frozen MQT family classifier | generated QASM + manifest | [`train_mqt_family_classifier.py`](experiments/train_mqt_family_classifier.py) | [`results/mqt_family_classifier/`](results/mqt_family_classifier/) |
| MQT-conditioned evaluation | public challenge JSON/QASM + frozen joblib | [`replicate_with_mqt_family.py`](experiments/replicate_with_mqt_family.py) | [`results/mqt_pretrained_family_replication.json`](results/mqt_pretrained_family_replication.json) |

The raw and derived artifacts are committed in the repository. The binary
`mqt_family_classifier.joblib` is the frozen model used for the integration
result; it is not the paper authors' weight file.

## 7. Reproduction commands

Use the pinned environment in
[`experiments/requirements-reproduction.txt`](experiments/requirements-reproduction.txt):

```bash
cd <repo-root>

python \
  experiments/replicate_paper.py --target both --seed 0 \
  --output results/paper_replication.json

python \
  experiments/protocol_audit.py --target both --seed 0 \
  --output results/protocol_audit.json

python \
  experiments/train_mqt_family_classifier.py

python \
  experiments/replicate_with_mqt_family.py --target both --seed 0 \
  --output results/mqt_pretrained_family_replication.json
```

The MQT generation step requires the separate environment listed in
[`experiments/requirements-mqt-pretraining.txt`](experiments/requirements-mqt-pretraining.txt);
once the committed QASM and manifest are present, it is not needed to rerun
the classifier.

The two main pipelines were rerun after the report was assembled. Their
metrics are identical to the committed results after ignoring the expected
`repo_commit` provenance field; the refreshed JSON files now point to commit
`e0f4591`.

## 8. Integrity hashes

These hashes identify the inputs and main result files used for this report:

```text
data/hackathon_public.json
65101f39da3a64f024c9637fdcc5d713f4bd4a748f9cb319306199c66137c899

data/holdout_public.json
6f73189575b6a1a9e92e9462727e711c8956494a58b61a82f8fba7728167df61

data/mqt_family_pretraining/manifest.json
d0e28759257efd616ba1bac0ee6a7fe70283a578323824e324ae01d18cf1f3cc

results/protocol_audit.json
c8b81dc0fa6c1386704113de5ef9912bdc5f9b4a7f031817069b5a706592585a

results/paper_replication.json
b9bd992b734158aae22cc732e4eccb081348303df0ca35cb3d6b9d6f36c72525

results/mqt_pretrained_family_replication.json
87cce89ea912b5bf80f283a2a17fb473be0c9767e82ab72e5c75be3e3550e244
```
