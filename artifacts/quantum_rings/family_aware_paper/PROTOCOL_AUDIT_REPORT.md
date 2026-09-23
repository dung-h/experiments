# Follow-up protocol audit

**Audit recorded:** 2026-09-18  
**Source revision:** `f6d9fe3` (`experiments/protocol_audit.py`)  
**Dataset:** `data/hackathon_public.json` (36 circuits × 4 contexts = 144 rows)

This audit follows the methodological feedback in `feedback.log`. It keeps the
same circuit-level five-fold split as the replication, then adds runtime tree
baselines, an incremental MLP ablation, per-context metrics, leave-one-family-
out stress testing and a threshold-conditioned runtime diagnostic.

## Reproduce

From the repository root:

```bash
python experiments/protocol_audit.py \
  --target both --seed 0 \
  --output results/protocol_audit.json
```

The audit uses the pinned environment in
`experiments/requirements-reproduction.txt`. The paper-target 0.75 runtime is
the documented mirror-sweep proxy; the challenge-target 0.99 runtime is the
public 10,000-shot forward label.

## Results: paper-style target 0.75

### Runtime tree baselines

All regressors use the same 32 features and the same circuit-level folds. The
regression target is log runtime.

| Model | R²(log runtime) | Median relative error |
|---|---:|---:|
| Random Forest | 0.744 | 36.5% |
| ExtraTrees | 0.722 | 48.1% |
| Gradient Boosting | **0.757** | **36.7%** |

The threshold Random Forest baseline reaches 70.83% exact accuracy and 86.81%
within one rung.

### Incremental MLP audit

| Configuration | Exact threshold | R²(log runtime) |
|---|---:|---:|
| Baseline MLP, global features | 47.92% | −0.228 |
| + graph features | 48.61% | −0.592 |
| + family classifier | 59.03% | −0.249 |
| + decayed cross-entropy | **60.42%** | **−0.209** |

Family conditioning improves threshold selection, but none of the reconstructed
MLP variants is a competitive runtime regressor on this public proxy target.
The tree baselines are substantially stronger for runtime.

### Runtime by context: Random Forest

| Context | R²(log runtime) | Median relative error |
|---|---:|---:|
| CPU + double | 0.405 | 53.2% |
| CPU + single | 0.453 | 42.4% |
| GPU + double | **0.841** | **24.3%** |
| GPU + single | **0.888** | **22.8%** |

These are mirror-runtime proxy metrics, not measured forward runtimes at the
paper's 0.75 threshold. The aggregate score hides a clear CPU/GPU difference.

### Leave-one-family-out stress test

Using the repository's 20 fine-grained family labels, the held-out-family RF
achieves threshold exact accuracy 63.19% and runtime R²(log) 0.758. This is not
evidence of reliable family transfer: many families contain one circuit, and
the taxonomy is not the paper's fully specified primary-family mapping. It is a
stress test rather than a paper-comparable metric.

## Results: challenge target 0.99

### Runtime tree baselines

| Model | R²(log runtime) | Median relative error |
|---|---:|---:|
| Random Forest | 0.410 | 49.2% |
| ExtraTrees | **0.712** | **38.8%** |
| Gradient Boosting | 0.578 | 37.9% |

For comparison, the MLP ablation runtime scores are 0.102 (baseline), −0.731
(+graph), −0.850 (+family) and −0.891 (+decayed CE). The tree baselines again
provide the stronger public-data runtime result.

### Threshold-conditioned runtime diagnostic

This uses the actual public forward runtime at the 0.99-selected threshold.

| Runtime features | R²(log runtime) | Median relative error |
|---|---:|---:|
| No threshold input | 0.410 | 49.2% |
| True threshold input (oracle) | 0.407 | 46.3% |
| Predicted threshold input | 0.408 | 46.3% |

Adding threshold did not improve the model on these public labels. This is
consistent with the paper's qualitative threshold-removal observation, but it
does not resolve the paper's missing 0.75 forward-runtime target.

### Runtime by context: Random Forest

| Context | R²(log runtime) | Median relative error |
|---|---:|---:|
| CPU + double | 0.329 | 50.3% |
| CPU + single | 0.478 | 42.9% |
| GPU + double | 0.284 | 50.2% |
| GPU + single | 0.389 | 42.5% |

The context ordering changes between the mirror proxy and the public forward
target. This is another reason not to pool the two runtime semantics.

## Finding

The feedback's strongest criticism is supported: the paper's neural family-aware
architecture is not established as the necessary runtime estimator by its
reported baseline table. On the available public artifact, tree regressors are
the most reliable runtime baselines, while family conditioning primarily helps
threshold classification. The paper's claim remains a method result for its
private Quantum Rings protocol, not a universally demonstrated runtime-model
advantage.

The audit does not change the replication boundary. It cannot regenerate the
Quantum Rings simulator labels, the private family-classifier pretraining set,
the paper-protocol forward timings or the hidden holdout truth.
