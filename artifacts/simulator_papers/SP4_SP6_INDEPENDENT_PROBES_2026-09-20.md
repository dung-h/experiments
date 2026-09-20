# SP-4 to SP-6 independent probes

**Local date:** 20 September 2026.

> **Historical initial probe.** This 45/12/20-row snapshot is retained for
> provenance. The expanded, current result set is
> [`SIMULATOR_ESTIMATOR_VALIDATION_REPORT_2026-09-20.md`](SIMULATOR_ESTIMATOR_VALIDATION_REPORT_2026-09-20.md):
> it contains 98 feasible plan candidates, 24 EMU-MPS threshold labels and 52
> precision pairs, with explicit failure artifacts.

| ID | Local target | Completed result | Boundary |
| --- | --- | --- | --- |
| SP-4 plan ranking | Warm GPU scalar TN contraction | 45 candidate rows; minimum FLOP selected the fastest candidate in 14/15 groups; largest regret 1.048×. | Three explicit plans, not the author candidate generator or learned ranker. |
| SP-5 family-aware | Required EMU-MPS bond-dimension rung at fidelity 0.99 | 12 labels; shared and inferred-family residual both achieve 41.7% exact and 100.0% within one rung. | Threshold target, not QASM runtime; known family in each training fold only. |
| SP-6 precision | Warm GPU scalar TN contraction | 20 complex64/complex128 pairs; median complex128/complex64 time is 3.369×; max relative scalar difference is `1.936e-06`. | No SGEMM emulation, automatic selector, RCS/Sycamore corpus or fidelity study. |

All warm-contraction targets exclude construction, path search and first call.
These rows are not pooled with VQCSim, Aer, Pauli propagation, Ma-Li,
Qonductor or QPU service-runtime targets.

## Reproduction commands

```bash
ROOT=/path/to/quantum-runtime-estimator-replications
export CUTENSORNET_PYTHON="$ROOT/.venv-cutensornet/bin/python"
bash "$ROOT/tracks/cutensornet/code/with_cutensornet_env.sh" "$ROOT/experiments/simulator_papers/run_explicit_plan_ranking_proxy.py" --output "$ROOT/artifacts/simulator_papers/explicit_plan_ranking_proxy_20260920/candidates.csv"
python3 "$ROOT/experiments/simulator_papers/evaluate_explicit_plan_ranking_proxy.py" "$ROOT/artifacts/simulator_papers/explicit_plan_ranking_proxy_20260920/candidates.csv" --output-prefix "$ROOT/artifacts/simulator_papers/explicit_plan_ranking_proxy_20260920/evaluation"

python /path/to/python-with-pandas-and-scikit-learn "$ROOT/experiments/simulator_papers/evaluate_family_aware_threshold_proxy.py" "$ROOT/artifacts/pasqal_emu_mps/threshold_sweep_labels.csv" --output-prefix "$ROOT/artifacts/simulator_papers/family_aware_emu_mps_proxy_20260920/evaluation"

bash "$ROOT/tracks/cutensornet/code/with_cutensornet_env.sh" "$ROOT/experiments/simulator_papers/run_precision_selection_proxy.py" --output "$ROOT/artifacts/simulator_papers/precision_selection_proxy_20260920/pairs.csv"
python3 "$ROOT/experiments/simulator_papers/evaluate_precision_selection_proxy.py" "$ROOT/artifacts/simulator_papers/precision_selection_proxy_20260920/pairs.csv" --output-prefix "$ROOT/artifacts/simulator_papers/precision_selection_proxy_20260920/evaluation"
```

The detailed row-level summaries are in the three `evaluation.md` files beside
the raw outputs. SP-5 used Python 3.10.21, pandas 2.3.3 and scikit-learn 1.7.2.
