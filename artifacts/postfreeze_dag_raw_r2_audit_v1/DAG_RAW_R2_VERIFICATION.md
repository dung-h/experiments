# Study R coarsened-DAG raw-scale R² verification

## Scope and provenance

This is a read-only post-freeze audit of the historical Study R coarsened-DAG topology screen. It uses only the saved three-seed mean OOF table at `artifacts/validation/mali_physical_dag_v1/topology_controls_validation/seed_mean_oof_predictions.csv`. It does not alter Study R, Scientific Freeze v2, model weights, QASM, splits, or predictions.

The saved screen is explicitly a `CURRENT_FAKE_SNAPSHOT_PROXY`, not a reconstruction of the historical physical circuit or calibration state ([corpus report](../validation/mali_physical_dag_v1/REPORT.md), lines 1–18; [builder](../../experiments/build_mali_physical_dag.py), lines 1–12). A newer, separate 4,515-observation / 165-group Track-A topology ablation exists at `artifacts/postclosure_topology_ablation/v1/`; it is authoritative for its own frozen Track-A protocol but does **not** supersede these Study R numerical results because its cohort, representation, and evaluation protocol differ. No numbers from the two studies are combined here.

### Observation and evaluation semantics

* **Target:** Ma–Li archived `time_taken` / observed `result.time_taken`, stored as `target_seconds` in seconds ([builder](../../experiments/build_mali_physical_dag.py), lines 54–74). The build receipt records 1024 shots and queue exclusion; any more detailed timing-boundary decomposition is not established by this topology artifact ([build summary](../validation/mali_physical_dag_v1/build_summary.json)).
* **Population:** 340 stored observations: 192 Osaka and 148 Kyoto rows, associated with 170 raw-QASM SHA-256 groups. The reconstructed proxy has 300 backend-specific graphs; it is not “all thesis QPU observations.”
* **Split:** five grouped-QASM folds. A raw-QASM hash is held out wholesale in its fold; Osaka/Kyoto rows sharing that hash remain together ([split code](../../experiments/train_mali_layer_dag_rnn.py), lines 132–137; [aggregation code](../../experiments/evaluate_mali_topology_controls.py), lines 80–84).
* **OOF and seeds:** all four variants supply exactly one held-out prediction for every row and each seed. The reported analysis prediction is the **arithmetic mean of three independently trained seed predictions per row** (1234, 2025, 31415), then scored; it is not a selected deployable model and not an average of three metric values ([aggregation code](../../experiments/evaluate_mali_topology_controls.py), lines 57–75; [canonical report](../validation/mali_physical_dag_v1/topology_controls_validation/REPORT.md), lines 1–3).
* **Weights:** point metrics are pooled row means (`np.mean`); no sample weights are applied. Uncertainty resamples raw-QASM groups, retaining every row for a sampled group.

Alignment checks passed: each variant has the identical 340 `row_id`s, targets agree exactly across variants, all predictions and targets are finite/non-negative, and every raw-QASM group is present in the saved pairing. See `verification_receipt.json`.

## Recomputed metrics

The candidate raw-scale R² values are verified exactly (within floating-point representation) against the saved canonical `summary.json`. Raw R² is calculated in seconds, not derived from log-R². Log-R² is independently recalculated as a consistency check after applying `log1p` separately to saved seconds-scale target and prediction values. Definitions are implemented in `research/postfreeze_dag_raw_r2_audit_v1/verify_dag_raw_r2.py` and match the historical metric implementation ([metrics](../../experiments/train_mali_layer_dag_rnn.py), lines 117–129).

| Variant | MAE (s) | RMSE (s) | Raw R² | log1p-runtime R² | Mean signed error (s) |
|---|---:|---:|---:|---:|---:|
| Node-only | 0.814544 | 1.057305 | 0.693008 | 0.766643 | -0.095938 |
| True DAG | 0.844832 | 1.120833 | 0.655009 | 0.738250 | -0.155041 |
| Reversed DAG | 0.896700 | 1.182247 | 0.616167 | 0.699120 | -0.106102 |
| Shuffled DAG | 0.852959 | 1.122301 | 0.654104 | 0.728509 | -0.082543 |

The original public table reported MAE and log-R², not raw R² ([canonical report](../validation/mali_physical_dag_v1/topology_controls_validation/REPORT.md), lines 5–16). That is a reporting difference, not a numerical discrepancy: raw R² was already serialized in its `summary.json`.

## Paired raw-scale bootstrap

This audit ran 5,000 deterministic paired bootstrap replicates (seed `20261021`) over the 170 raw-QASM hashes. A replicate concatenates all row(s) from each selected hash, recalculates pooled MAE and pooled R² on that full resample, and never averages per-group R². All 5,000 replicates were valid; none had zero target variance. “Improvement fraction” below is an empirical bootstrap fraction, not a p-value or posterior probability.

| Contrast (True − control) | ΔMAE (s) [95% CI] | Δ raw R² [95% CI] | Empirical MAE-improvement fraction | Empirical raw-R²-improvement fraction |
|---|---:|---:|---:|---:|
| Node-only | +0.030288 [-0.014092, +0.078419] | -0.038000 [-0.078653, -0.002723] | 0.0944 | 0.0168 |
| Shuffled DAG | -0.008127 [-0.060468, +0.043630] | +0.000905 [-0.032328, +0.033281] | 0.6270 | 0.5308 |
| Reversed DAG | -0.051867 [-0.098220, -0.007796] | +0.038842 [+0.008500, +0.071401] | 0.9880 | 0.9952 |

The full machine-readable outputs are `dag_raw_metrics.csv`, `dag_raw_paired_comparisons.csv`, and `bootstrap_draws_and_deltas.npz`.

### Backend-composition sensitivity (descriptive only)

The direction against a true-DAG advantage is not confined to one archived backend: relative to node-only, true-DAG raw R² is lower by `0.050487` on Kyoto (130 hash groups represented) and `0.029157` on Osaka (170 hash groups represented); the corresponding MAE changes are `+0.023787 s` and `+0.035299 s`. These are pooled-within-backend descriptions, not separately powered inferential tests, because hashes can be represented on both backends and the original inference unit remains raw QASM.

## Interpretation of the topology statements

| Statement | Verdict | Evidence and limitation |
|---|---|---|
| S1. True DAG outperforms reversed adjacency. | **SUPPORTED, bounded** | Both raw-R² and MAE paired CIs favor true over reversed. This only establishes directional information for this coarsened architecture/proxy cohort. |
| S2. True DAG meaningfully outperforms shuffled adjacency. | **NOT ESTABLISHED** | Point differences are essentially zero and both paired CIs cross zero. This is not evidence of equivalence. |
| S3. True DAG meaningfully outperforms node-only. | **NOT ESTABLISHED; raw-R² direction favors node-only** | The true-minus-node raw-R² CI is entirely below zero; the MAE CI includes zero. The result does not support true-DAG superiority. |
| S4. Preserving the true DAG is necessary for accurate QPU runtime estimation. | **NOT ESTABLISHED** | One coarsened proxy architecture cannot establish necessity across representations, targets, or execution contexts. |
| S5. The tested coarsened representation fails to demonstrate incremental predictive value from true dependency edges. | **SUPPORTED, bounded** | True DAG does not robustly beat either required node-only or shuffled control. This is the appropriate negative conclusion for this experiment. |

## Thesis-ready wording

> In the historical 340-observation Ma–Li coarsened-DAG screen, true dependency orientation outperformed reversed adjacency, but true-DAG message passing did not establish an incremental advantage over node-only or shuffled-adjacency controls. Raw-scale R² reproduced the same bounded conclusion: 0.655 for true DAG versus 0.693 for node-only. This does not show that circuit topology is universally irrelevant; it shows that the tested 256-bin representation and two-layer message-passing operator did not demonstrate a robust topology-specific predictive gain for this proxy cohort.

