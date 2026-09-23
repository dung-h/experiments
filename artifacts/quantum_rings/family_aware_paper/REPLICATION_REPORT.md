# Replication report: Family-Aware Residual Architecture

Replication recorded: 2026-09-18
Repository: [`iQuHACK/2026-Quantum-Rings`](https://github.com/iQuHACK/2026-Quantum-Rings)  
Baseline audit commit: `e4fc97b` (`Report Quantum Rings protocol audit findings`);
the public-MQT follow-up is recorded in the files and commits after that point.
Paper: [arXiv:2606.11620](https://arxiv.org/abs/2606.11620)

## Executive result

The repository is a data/challenge artifact, not the complete implementation
of the paper. I reconstructed the feature pipeline and a paper-aligned
family-conditioned residual/FiLM MLP, and evaluated it with circuit-level
5-fold cross-validation. The strongest independent baseline on the public
artifact is Random Forest; the reconstructed neural model does not reproduce
the paper's headline score. This is an honest artifact replication, not a
claim that the paper's private training setup was recovered.

## Important protocol mismatch

The paper formulates threshold selection at mirror fidelity **>= 0.75** and
reports a 10-class ladder (`star`, 1, 2, ..., 256). The challenge repository's
public data uses `selection.target = 0.99` for all 144 rows and its README
scores the first rung reaching **>= 0.99**. The two protocols produce
different threshold labels:

| Protocol | Rows | `star`/no rung | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper-style target 0.75 | 144 | 4 | 54 | 46 | 6 | 17 | 13 | 0 | 4 | 0 |
| Challenge target 0.99 | 144 | 7 | 52 | 44 | 10 | 8 | 17 | 4 | 0 | 2 |

All public 10,000-shot `forward` timings were selected using the 0.99
protocol. Therefore a true paper-protocol forward-runtime label at the 0.75
crossing is not present. For the paper-style run below, I use the mirror sweep
wall time at the first 0.75 crossing and mark it as a proxy; four no-crossing
rows use the forward timing as a documented fallback.

## Public data audit

- 36 QASM circuits, 144 circuit × processor × precision rows;
- 20 fine-grained family labels in the repository (the paper describes 10
  primary families and a 21-class embedding with an unknown class);
- contexts: `CPU/GPU × single/double`;
- 137 rows have `status = ok`, 7 are `no_threshold_met` under the public 0.99
  selection;
- QASM is OpenQASM 2.0 and parses successfully with Qiskit 2.5.2;
- no official model source, trained weights, or the paper's separate ~200
  circuit family-classifier pretraining set is included.

Every circuit has all four execution contexts in the public data:

| Context | Rows |
|---|---:|
| CPU + single | 36 |
| CPU + double | 36 |
| GPU + single | 36 |
| GPU + double | 36 |

The headline metrics below aggregate the 144 rows. The circuit-level split
keeps all four rows for a QASM file in the same fold; it is not a context-held-
out experiment.

The paper's arXiv source also states that code and trained models are
available **upon request**. Thus the correct status is “not publicly bundled
or linked,” rather than “confirmed unavailable.”

## Reconstructed method

[`experiments/replicate_paper.py`](experiments/replicate_paper.py) implements:

1. 32 features matching the paper's categories: basic statistics, 12 gate
   counts, execution context, complexity ratios, interaction-graph/local
   features, three algorithm fingerprints, and five RCM/cut-pressure features;
2. logarithmic scaling of count-like features;
3. circuit-level 5-fold CV (all four contexts of each QASM file remain in one
   fold);
4. Random Forest classifier/regressor baselines;
5. a no-family MLP and a family-aware MLP with family embedding, FiLM
   modulation, additive residual heads, direct shortcuts, decayed threshold
   loss, and log-runtime MSE;
6. a fold-local Random Forest family classifier as the original transparent
   replacement for the unavailable private family pretraining;
7. a separate public MQT Bench generator, frozen MLP classifier, and integration
   runner in `experiments/generate_mqt_family_pretraining.py`,
   `experiments/train_mqt_family_classifier.py`, and
   `experiments/replicate_with_mqt_family.py`.

Run both protocols with:

```bash
cd <repo-root>
PYTHONPATH=. python \
  experiments/replicate_paper.py --target both \
  --output results/paper_replication.json
```

The model/evaluation package subset is recorded in
[`experiments/requirements-reproduction.txt`](experiments/requirements-reproduction.txt).

## Results

Metrics are aggregated over all held-out rows from the five circuit folds.
`±1` means one adjacent rung in the ordered threshold ladder; `R²(log)` is
the runtime score in the log domain used for training.

### Paper-style threshold target 0.75

| Model | Exact threshold | Within ±1 rung | Mean rung distance | R²(log runtime) | Median relative runtime error |
|---|---:|---:|---:|---:|---:|
| Random Forest | **68.06%** | **86.81%** | **0.50** | **0.748** | **34.8%** |
| MLP, no family | 47.92% | 75.69% | 1.21 | −0.593 | 78.5% |
| Family MLP, predicted family | 59.03% | 83.33% | 0.79 | −0.253 | 69.0% |
| Family MLP, oracle family | 61.11% | 80.56% | 0.87 | −0.512 | 77.3% |

The fold-local family classifier reaches **38.89%** circuit accuracy, versus
the paper's reported 97.5% pretrained classifier. This gap is expected because
the paper's pretraining circuits are not supplied and many repository families
have only one circuit.

The follow-up public-MQT experiment removes the data-leakage concern in that
replacement: its MLP is trained on 200 independently generated MQT circuits,
then frozen before evaluation. It reaches 97.5% on its own held-out MQT split;
the transfer result and its limitations are reported separately in
[`results/MQT_FAMILY_PRETRAINING_REPORT.md`](results/MQT_FAMILY_PRETRAINING_REPORT.md).

### Challenge target 0.99

| Model | Exact threshold | Within ±1 rung | Mean rung distance | R²(log runtime) | Median relative runtime error |
|---|---:|---:|---:|---:|---:|
| Random Forest | **61.11%** | **84.03%** | **0.69** | **0.417** | **50.2%** |
| MLP, no family | 40.97% | 67.36% | 1.51 | −0.786 | 67.7% |
| Family MLP, predicted family | 63.19% | 81.94% | 0.80 | −0.856 | 67.1% |
| Family MLP, oracle family | 61.11% | 77.08% | 0.91 | −1.095 | 68.9% |

The raw JSON outputs are [`results/paper_replication_paper.json`](results/paper_replication_paper.json) and [`results/paper_replication_challenge.json`](results/paper_replication_challenge.json).

## Interpretation

The paper reports 79.5% exact threshold accuracy, 91.2% within one rung, and
R² = 0.82 runtime correlation. Those values are not independently reproduced
from this public repository. The main blockers are concrete:

1. target semantics differ (0.75 in the paper vs 0.99 in the challenge data);
2. paper-protocol forward timings are absent from the public JSON;
3. the official model implementation/weights are not in the repo (the paper
   says they can be requested from the authors); and
4. the paper's exact separate family-classifier pretraining data and weights
   are not included in the public artifact. A documented public MQT Bench
   approximation is now provided separately; it must not be described as the
   authors' original set.

The public artifact is still useful for our Quantum Runtime Estimator: it is a
simulator-only, approximate tensor-network domain with explicit threshold,
fidelity, CPU/GPU, precision, mirror-runtime, and forward-runtime fields. It
should be evaluated as its own simulator-runtime track, not pooled with QPU
execution, cloud turnaround, or workflow/JCT targets.

## Protocol gap audit: paper versus this replication

The following differences prevent the current result from being called an
exact reproduction of the paper's headline numbers:

| Component | Paper protocol | Current setup | Status |
|---|---|---|---|
| Evaluation circuits | 36 circuits, described as 10 primary families | Same 36 QASM files, but 20 fine-grained repository labels and one custom CutBell circuit | Partial |
| Execution contexts | CPU/GPU × single/double | All four contexts, 36 rows each | Matched |
| Threshold target | Minimum rung at mirror fidelity ≥ 0.75 | Derived at 0.75, but public selection was generated at 0.99 | Proxy labels |
| Runtime target | Runtime for the circuit–threshold pair | Mirror-sweep wall time at the 0.75 crossing; no matching 10,000-shot forward label | Proxy target |
| Quantum Rings execution | Simulator runs generate the sweep and forward truth | Public SDK exists, but is not bundled in this repo and has not been rerun in the current credentialed environment | Not reproduced |
| Family pretraining | Separate approximately 200 MQT Bench circuits; frozen MLP classifier | Publicly generated 200-circuit MQT approximation; exact author set unavailable | Public approximation |
| Feature heuristics | Exact author implementation is not released | 32 paper-shaped features; fingerprint penalties and RCM/cut definitions are transparent approximations | Partial |
| Optimization | AdamW, 5e-3 learning rate, warm-up, batch size 32, early stopping | Full-batch 160-epoch stable training at 3e-4; no warm-up or early stopping | Not matched |
| Final inference | Five-fold ensemble averaging | Out-of-fold predictions, one model per held-out fold; no ensemble averaging | Not matched |
| Ablations | Baseline, graph, family, decayed-CE and tree-baseline ladder | RF, no-family MLP, predicted-family MLP and oracle-family MLP | Incomplete |
| Leaderboard holdout | Hidden QASM and private truth | `data/holdout_public.json` has task metadata only | Unavailable |

The most consequential gap is runtime semantics: a mirror sweep runtime is not
the same target as a 10,000-shot forward runtime at the paper's 0.75-selected
threshold. The second is family pretraining: the paper's exact 97.5%
classifier result cannot be tested against the authors' private circuit set.
Our generated public MQT set reaches 97.5% on its own held-out split, but its
transfer accuracy on the evaluation circuits is only 46.15% among mapped known
families. These gaps make a direct comparison between our `R²` and the paper's
`R² = 0.82` misleading even when the circuit split is correct. See
[`results/MQT_FAMILY_PRETRAINING_REPORT.md`](results/MQT_FAMILY_PRETRAINING_REPORT.md)
for the public pretraining protocol and integration result.

The implementation also does not yet reproduce the paper's incremental ablation
table or report metrics separately for CPU-single, CPU-double, GPU-single and
GPU-double. The aggregate 144-row score is valid for the public artifact, but it
does not show whether the model transfers between execution contexts.

## Quantum Rings SDK availability audit

The simulator is not absent from the public ecosystem. Quantum Rings currently
distributes [`QuantumRingsLib`/`quantumrings`](https://pypi.org/project/quantumrings/)
through PyPI and documents [CPU, GPU and hybrid backends](https://www.quantumrings.com/doc/usage/backends.html).
Running it requires a [Quantum Rings token and account name](https://www.quantumrings.com/doc/start/credentials.html);
the iQuHACK repository intentionally describes an offline challenge and does
not vendor the SDK, runner or credentials. The current reproduction
environment uses Python 3.10, while the current [SDK documentation](https://www.quantumrings.com/doc/install/install_cpu.html)
requires Python 3.11–3.14, so it must be installed in a separate environment.

The SDK can therefore support a future execution rerun, but installing the
current package would not by itself make the paper exact: the hackathon SDK
version, runner settings, host, timeout policy and private pretraining data are
not recorded in the public artifact. Any new SDK run must be reported as a
fresh Quantum Rings validation, with its version and credentialed backend
explicitly recorded.

## Additional methodological feedback audit

The external feedback identifies weaknesses in the paper's experiment design,
not only missing implementation artifacts:

1. **Runtime evaluation is under-baselined.** The paper reports tree models for
   threshold classification but does not report matching Random Forest,
   gradient-boosting or ExtraTrees runtime-regression results on the same
   target and folds. Consequently, `R² = 0.82` does not establish that the
   neural family-aware architecture is necessary for runtime prediction.
2. **The runtime row definition is underspecified.** The paper says runtime is
   predicted for a circuit–threshold pair, but the 32-feature list does not
   include threshold, while a threshold-removal ablation is mentioned later.
   The exact runtime input and number of circuit–threshold training rows cannot
   be reconstructed from the six-page description.
3. **Fold ensembling is ambiguous.** The paper says final predictions are
   averaged across five fold models but does not state whether reported
   validation values are strict out-of-fold predictions. Averaging models that
   were trained on a validation circuit would make the metric optimistic.
4. **Family generalization is not tested.** Circuit-level CV holds out circuit
   instances, not entire algorithm families. It therefore does not show
   transfer to an unseen family. Several per-family results have `N = 1` and
   measure context variation of one circuit rather than family generalization.
5. **Family taxonomy is not fully specified.** The classifier description,
   primary-family table and 21-class embedding use different granularities;
   the mapping is not included in the public artifact.
6. **Related-work coverage needs updating.** The paper does not discuss
   Maestro (arXiv:2512.04216), a prior predictive simulator-runtime system.
   The defensible novelty claim is therefore family-conditioned multi-task
   parameter selection in a small-data Quantum Rings setting, not the first
   quantum simulator runtime predictor.

These points are now treated as limitations of the paper's evidence, not as
failures of our code. Our replication keeps strict circuit-level out-of-fold
metrics and labels the 0.75 runtime as a proxy; a follow-up validation should
add runtime tree baselines, explicit threshold conditioning, per-context
metrics and leave-one-family-out evaluation.

## Independent rerun record

The protocol was rerun on 2026-09-18 from the committed public JSON and QASM
files using the pinned reproduction environment in
`experiments/requirements-reproduction.txt` (`Qiskit 2.5.2`,
`scikit-learn 1.7.2`, `PyTorch 2.14.0+cpu`, `networkx 3.4.2`). The reported
metrics are byte-for-byte identical to the previous local result for seed 0;
the JSON artifacts now record the source revision and run date rather than the
initial repository commit.

```bash
cd <repo-root>
PYTHONPATH=. python \
  experiments/replicate_paper.py --target both --seed 0 \
  --output results/paper_replication.json
```

The split is circuit-level: all four backend/precision rows for one QASM file
remain in the same fold. This is the key leakage-control condition for this
small dataset and must not be replaced by a row-wise random split.

## Reproducibility boundary

This work reproduces the data audit, feature extraction, split policy, baseline
comparisons, and a paper-aligned architecture. It does not rerun the Quantum
Rings simulator itself or regenerate the private truth/holdout labels. The
challenge README explicitly says holdout QASM and truth are organizer-only;
`data/holdout_public.json` contains IDs plus processor/precision but no circuit
identity or labels. Consequently, no valid leaderboard score can be computed
locally from the supplied files.
