# IBM usage formulas evaluated against frozen Track-A labels

Date: 2026-10-27. Scope: arithmetic evaluation of fixed formulas on saved data. No training, fitted correction, QPU/simulator execution, or frozen-artifact modification.

## Formulas and target

IBM's [workload-usage guide](https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time) provides both a quick approximation and a duration-based baseline. We applied these formulas directly, without calibration:

- **IBM quick formula:** `2 + 0.00035 × shots` seconds.
- **IBM duration + default delay:** `2 + shots × (saved circuit duration + 0.00025)` seconds.

Both calculations assume one circuit and one sub-job per observation. The second uses the saved Qiskit nominal single-shot duration and a fixed 250 µs repetition delay. Historical repetition delays, initialization options/durations, mitigation expansion and internal sub-job partitioning are not established; no extra initialization term is invented. These configuration assumptions are declared in [PROTOCOL.md](PROTOCOL.md), which was written before calculating scores. We do not round predictions to whole seconds.

The formula itself is IBM's published formula. The duration input in the second calculation comes from a saved backend snapshot rather than certified execution-day calibration. Neither calculation is a value retrieved from IBM's remote `job.usage_estimation` property. That property returns a server estimate for a submitted job ([API documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/runtime-job-v2)).

The comparison uses all **4,515 observations / 165 structural groups** in the authoritative qualified Track-A panel. The targets are the saved provider `Result.time_taken` labels in seconds: Ma–Li mean of three runs, Qonductor one-circuit jobs, QPack circuit execution records converted from milliseconds. All labels and source aggregation remain unchanged. This comparison is against that observed field; it does not prove that historical `Result.time_taken` and modern IBM QPU-lock usage have identical internal accounting. See [target lineage](../RESULT_TIME_TAKEN_PROVENANCE_NOTE.md).

## Actual computed results

MAE means average absolute error in seconds. Pooled MAE gives every observation equal weight; group-balanced MAE first averages errors inside each structural group and then gives the 165 groups equal weight.

| Formula | Pooled MAE (s) | Group-balanced MAE (s) | Source-macro MAE (s) | RMSE (s) | Raw R² | Mean signed error (s) |
|---|---:|---:|---:|---:|---:|---:|
| IBM quick | 2.809364 | 5.488259 | 3.235294 | 3.198050 | −3.684787 | −2.774125 |
| IBM duration + default delay | 2.727686 | 1.529070 | 1.819117 | 2.966255 | −3.030289 | −2.695485 |

Negative signed error indicates underprediction. The duration-based calculation lowers pooled MAE by **0.081678 s**, with source behavior shown below. This is a descriptive difference; no bootstrap significance claim is made.

| Source | Rows | Groups | Quick MAE (s) | Duration + delay MAE (s) |
|---|---:|---:|---:|---:|
| Ma–Li | 340 | 148 | 6.096523 | 1.467514 |
| Qonductor | 230 | 11 | 0.976437 | 1.056087 |
| QPack MCP | 3,945 | 6 | 2.632923 | 2.933751 |

For Ma–Li, circuit-specific duration substantially improves the fixed quick approximation. For Qonductor and QPack, this configured duration-based estimate has larger MAE. The large change in group-balanced MAE reflects the 148 Ma–Li groups; pooled MAE is dominated by QPack's 3,945 rows. Neither weighting should be substituted silently for the other.

## Interpretation

IBM's published formulas produce fully numerical, reproducible predictions on the saved inputs. Under the declared configuration they still leave substantial errors against the frozen observed labels. The calculation does not identify whether the remaining error comes from approximate circuit duration, historical delay/options, sub-job count, the typical circuit length in the quick formula, or differences between usage accounting and the observed target. Those causes require evidence beyond the residual itself.

The formulas were not fitted on labels, so there are no trained fold-specific models or hyperparameter-selection steps. Original group IDs are retained for balanced scoring. This is an additional post-freeze diagnostic, not a replacement of any frozen analytical benchmark result.

## Reproduction and validation

From the repository root:

```bash
python3 docs/audits/qpu_timing_boundary/ibm_usage_formula_20261027/evaluate.py
```

Saved outputs:

- [predictions.csv](predictions.csv): exact IDs, targets, source/group, shots, durations, provenance and both predictions.
- [metrics.csv](metrics.csv): overall and per-source metrics, including bias and error tails.
- [validation_receipt.json](validation_receipt.json): input/code/output hashes, HEAD, environment and coverage checks.
- [independent_verification.json](independent_verification.json): **13,569 checks PASS**, using independent arithmetic expressions and `math.fsum` reductions; validates target/group alignment, formulas, MAE, group-balanced MAE and raw R².

No commit or push is part of this calculation.
