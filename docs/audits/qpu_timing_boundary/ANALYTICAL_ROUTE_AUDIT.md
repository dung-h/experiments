# Analytical-route and clock audit

## Routes inspected

The frozen contract separates the evaluation clock (`archived_observed_service_execution_time`) from every analytical output clock ([`qualified_logical_benchmark.json`](../../../benchmark_v1/protocol/qualified_logical_benchmark.json#L40-L45)).  The current artifacts therefore do **not** support treating a good calibrated score as validation of the raw physical/scheduled quantity.

| Route | Circuit / duration inputs | Calculation and exclusions | Output clock | Snapshot / status |
|---|---|---|---|---|
| QCRE snapshot critical path | Frozen compiled/submitted unitary view; backend Target duration table | DAG topological longest-finish recurrence. `barrier` and `snapshot` are zero-duration; explicit `delay` handling reads delay duration; unsupported operations fail rather than being imputed. See [`run_archived_analytical_wave_v3.py`](../../../benchmark_v1/scripts/run_archived_analytical_wave_v3.py#L166-L195). The qualified contract explicitly omits terminal measurement; barriers omitted and nonunitary/classical operations unsupported ([`qualified_logical_benchmark.json`](../../../benchmark_v1/protocol/qualified_logical_benchmark.json#L40-L45)). | `scheduled_critical_path_seconds` | Analytical proxy. Gate-duration method card says it is not provider service time ([`qcre_gate_duration.json`](../../../benchmark_v1/protocol/methods/qcre_gate_duration.json#L2-L13)). |
| Qiskit nominal duration | Same compiled circuit and frozen Target | `compiled.estimate_duration(target=backend.target, unit="s")` ([`run_archived_analytical_wave_v3.py`](../../../benchmark_v1/scripts/run_archived_analytical_wave_v3.py#L399-L441)). | `scheduled_single_shot_seconds` | The Qiskit card explicitly forbids equivalence to `Result.time_taken` ([`qiskit_estimate_duration.json`](../../../benchmark_v1/protocol/methods/qiskit_estimate_duration.json#L2-L13)). |
| Shot-scaled QCRE/Qiskit | Above raw output plus ledger shots | Exactly `shots * raw_duration` ([`run_archived_analytical_wave_v3.py`](../../../benchmark_v1/scripts/run_archived_analytical_wave_v3.py#L429-L441)). No measured repetition/inter-shot service model is added. | `shot_scaled_scheduled_seconds` | Non-native adaptation; not evidence of linear physical/service scaling. |
| Affine/log-affine calibrated variants | Raw or shot-scaled output plus outer-train target labels | Outer-train-only fit; outputs projected as declared calibration variants ([`run_archived_analytical_wave_v3.py`](../../../benchmark_v1/scripts/run_archived_analytical_wave_v3.py#L850-L863)). | `calibrated_observed_service_seconds` | Learned adaptation, never native API/paper output ([`method_fidelity_registry_v2.json`](../../../benchmark_v1/registry/method_fidelity_registry_v2.json#L31-L36), #L164-L191). |
| Scholten nominal proxy | Generic compiled wire-path depth, shots, reported nominal throughput | `shots × generic depth / reported CLOPS`; measurement/reset/delay included; barrier/snapshot excluded. It is not original QV-normalized effective depth ([`docs/methodology.md`](../../methodology.md#L253-L265)). | `nominal_clops_v_like_seconds` | Only 3,399/4,515 rows covered; nominal, not job-day CLOPS. |

## QCRE versus Qiskit

Under this frozen Target, QCRE and Qiskit single-shot values are intentionally numerically equivalent to floating-point tolerance, and their scaled/calibrated descendants are likewise equivalent when fit on identical training rows ([`method_fidelity_registry_v2.json`](../../../benchmark_v1/registry/method_fidelity_registry_v2.json#L31-L36)).  Numerical agreement therefore validates an implementation cross-check, **not** two independent physical estimators and not target-boundary correctness.

## Inputs and unmodeled components

The raw routes use nominal Target instruction durations and a compiled scheduling view.  They do not provide a measured row-level accounting of provider initialization, job/service handling, repeated-shot control, result assembly, queueing, compiler/transpiler work, or teardown.  The protocol says snapshot assets are nominal tier-3 contexts, not job-day calibration ([`docs/methodology.md`](../../methodology.md#L219-L232)).  Therefore none of those components may be estimated by subtraction and relabeled “overhead.”

Reset and measurement deserve special care: the QCRE qualified route explicitly removes terminal measurement from its unitary view; the Scholten proxy instead includes measurement/reset/delay in generic wire depth.  These are distinct representations and clocks, so differences cannot identify the isolated effect of measurement or reset.

## Calibration boundary

Calibration materially changes the object being evaluated.  It is leakage-safe because fits are outer-train-only, but it is still a label-trained bridge from a nominal schedule to observed service seconds.  A calibrated result can demonstrate predictive usefulness under the declared panel/splits; it cannot establish that the raw analytical formula already predicted observed runtime or that calibration recovered physical overhead.
