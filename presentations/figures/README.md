# Biểu đồ cho báo cáo benchmark

These four figures use the `two_domain_scorecard_v3` tables and the locked
`docs/presentation.md`. They compare compatible row sets and clocks, not
methods across the QPU and simulator domains. PNG and SVG exports retain
their original bytes. Source table paths below are relative to
`artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/`.

The original plot source has SHA-256
`06a379812d9fac2bbad7f14f72606f62af966f0a8761ec5bc0b915835d6a1b6f`.
The packaged [source](../source/plot_report_figures.py) changes only input and
output paths, output-directory safeguards, and accompanying documentation.
Its hash is recorded in `../manifest.json`.
Locked-claims source: `docs/presentation.md` SHA-256 `814fbadacd2db4eaf73d8a316ddc557ca0794e54218b6f37d73401c45931acbf`.

## Figure 1: Real-QPU source-local paired error

Files: `fig1_qpu_paired_mae_by_source.png` (SHA-256 `a99c90cf8de5a0daa36b16014cdaeaf7248b675993463168898f4b131edb511d`), `fig1_qpu_paired_mae_by_source.svg` (SHA-256 `1173484dc6c4ea6e305b1e964e13e94c87ffe43a3b3d19674d411aa9196ea47c`).
Units: seconds
Clock: archived_observed_service_execution_time
Exact row set/denominator: `[{"n": 340, "shared_row_set_hash": "3ad175f3b2cf88f52f9f0dedd404684fbff696b2d776f8d55538083dcdeac8e7", "source_id": "mali_real_qpu"}, {"n": 4481, "shared_row_set_hash": "a785df4abc98e96ad899bdf4f53837806715232d440bc620de74b88639c74655", "source_id": "qonductor_single_circuit_ibm"}, {"n": 3945, "shared_row_set_hash": "d8f5438afbc5179bc99a96672022627411c6d6ead8597fcee6b9abbe383a36d7", "source_id": "qpack_mcp"}]`
Caption and limitations: Units: seconds. Row set: the three source rows for candidate=mali_graph_architecture_v3_large and reference=unified_polynomial_v3; shared n=340 Ma–Li, 4,481 Qonductor, 3,945 QPack. Each row's shared_row_set_hash is in the source CSV. Evaluation/output clock: archived_observed_service_execution_time. Limits: source slices are not source-specific training; Ma–Li interval crosses zero, Qonductor coverage is asymmetric, and this does not isolate graph structure.
Sources:
- `qpu/archived_method_pairwise_comparisons.csv` — SHA-256 `46feff9632a48b7c852a938c0e5a153209d700e0574a5d8cd685c609ab5b1313`

## Figure 2: Real-QPU prediction coverage

Files: `fig2_qpu_prediction_coverage.png` (SHA-256 `d8dc1661d0bd415eda6c33f8bcad935eb4fed424b832504a138b89225c20f365`), `fig2_qpu_prediction_coverage.svg` (SHA-256 `d7131064da9e90793806dfe8e2ffac7389b90a3a1566a0decf81c5fec001cc0b`).
Units: successful predictions / assigned observations; percent
Clock: method-specific; see method_output_clock in the source table
Exact row set/denominator: `{"assigned_per_method": 8767, "method_ids": ["scholten_nominal_single_circuit_adaptation", "scholten_nominal_single_circuit_adaptation_outer_train_affine_seconds", "scholten_nominal_single_circuit_adaptation_outer_train_log_affine_seconds", "hyb_hanas_one_circuit_effective_cost_adaptation", "hyb_hanas_one_circuit_effective_cost_adaptation_outer_train_affine_seconds", "hyb_hanas_one_circuit_effective_cost_adaptation_outer_train_log_affine_seconds", "hyb_hanas_shot_linear_effective_seconds", "hyb_hanas_shot_linear_effective_seconds_outer_train_affine_seconds", "hyb_hanas_shot_linear_effective_seconds_outer_train_log_affine_seconds", "mali_graph_architecture_v3", "mali_graph_architecture_v3_large", "qcre_shot_scaled_schedule_seconds", "qcre_shot_scaled_schedule_seconds_outer_train_affine_seconds", "qcre_shot_scaled_schedule_seconds_outer_train_log_affine_seconds", "qcre_snapshot_critical_path", "qcre_snapshot_critical_path_outer_train_affine_seconds", "qcre_snapshot_critical_path_outer_train_log_affine_seconds", "qiskit_estimate_duration_snapshot", "qiskit_estimate_duration_snapshot_outer_train_affine_seconds", "qiskit_estimate_duration_snapshot_outer_train_log_affine_seconds", "qiskit_shot_scaled_schedule_seconds", "qiskit_shot_scaled_schedule_seconds_outer_train_affine_seconds", "qiskit_shot_scaled_schedule_seconds_outer_train_log_affine_seconds", "unified_polynomial_v3"], "method_rows": 24}`
Caption and limitations: Units: fraction of assigned observations, displayed as percent and counts. Exact row set: all 24 rows in qpu/archived_method_coverage.csv; assigned denominator is 8,767 per method. Clock differs by method_output_clock as recorded in the table. Limit: coverage only, not accuracy; methods have different fidelity classes and output clocks, so no ranking is implied.
Sources:
- `qpu/archived_method_coverage.csv` — SHA-256 `ee408a85b5fef378103a2ada6709c791f9e66a8db192e15445bdd6e9803aa915`

## Figure 3: CUDA-Q MPS under a fixed quality gate

Files: `fig3_cudaq_mps_quality_gated_clocks.png` (SHA-256 `30a34a68ee74b8a2f010459d78da86ae7585a97cae40aff1d6e114551f633b85`), `fig3_cudaq_mps_quality_gated_clocks.svg` (SHA-256 `d6b5f69803fabb98490a52298d8954f7463f9ca12f06ca993dc716c0c77bbfaf`).
Units: milliseconds per successful session-cell median
Clock: first_execution and warm_execution shown separately
Exact row set/denominator: `[{"attempted_cells": 486, "clock": "first_execution", "configuration_id": "cudaq_nvidia_mps_fp64", "failure_cells": 24, "ok_cells": 462, "stratum": "core_q2_q9"}, {"attempted_cells": 486, "clock": "warm_execution", "configuration_id": "cudaq_nvidia_mps_fp64", "failure_cells": 24, "ok_cells": 462, "stratum": "core_q2_q9"}, {"attempted_cells": 126, "clock": "first_execution", "configuration_id": "cudaq_nvidia_mps_fp64", "failure_cells": 15, "ok_cells": 111, "stratum": "frontier_q10_q16"}, {"attempted_cells": 126, "clock": "warm_execution", "configuration_id": "cudaq_nvidia_mps_fp64", "failure_cells": 15, "ok_cells": 111, "stratum": "frontier_q10_q16"}]`
Caption and limitations: Units: milliseconds per successful session-cell median. Exact row set: CUDA-Q MPS fp64, quality policy fp64_bond16_cutoff1e-10_gesvdj_fidelity_ge_0.99; four per_configuration_metrics rows where method_output_clock equals evaluation_target_clock, crossing core/frontier with first/warm clock. Denominators are 486 core and 126 frontier session-cells per clock; availability table reports 462/486 and 111/126 successful cells per clock. Limit: medians are among successful rows; failure counts include all terminal failure classes and no engine ranking is made.
Sources:
- `simulator/per_configuration_metrics.csv` — SHA-256 `f8628e32fb0a0cecc8f06eb76a77b9e14e25b4ba8c288b2e72b8227d117d757a`
- `simulator/availability_matrix.csv` — SHA-256 `f3b48ecc2967d5bf8d2ce3b0368dc5e1c8523ceda5626070cde3979df1880458`
Quality policy: `fp64_bond16_cutoff1e-10_gesvdj_fidelity_ge_0.99`

## Figure 4: Maestro pilot timing spread

Files: `fig4_maestro_pilot_timing_spread.png` (SHA-256 `3bba23dd81ae9bb202c432a1baa3f93f3d059b3761cf54cec8959e1765e93987`), `fig4_maestro_pilot_timing_spread.svg` (SHA-256 `b93794d61c76e1bdcdc07cc4edd52899ab6f38c349ae92a4e06916f33dc4ef6a`).
Units: seconds; logarithmic x-axis
Clock: maestro_qcsim_process_isolated_reported_execution
Exact row set/denominator: `[{"candidate": "statevector", "cell_id": "35d04914103e48a1e741ec2a608c7dd3c4e1fa811dc3f3fa41bf07fa86b31bc6", "operation_class": "one_qubit_noncommuting", "operation_repeats": 256, "stability_status": "unavailable"}, {"candidate": "statevector", "cell_id": "d69b0d8ec6c08da40f8aabbd898064001b538c9572998a3c39a0893e299124a9", "operation_class": "one_qubit_noncommuting", "operation_repeats": 512, "stability_status": "unavailable"}, {"candidate": "statevector", "cell_id": "ea5c38a39f65bca076245b349fe4a06a3ae264f6a6734fcf9041e1236ff9072f", "operation_class": "one_qubit_noncommuting", "operation_repeats": 1024, "stability_status": "unavailable"}, {"candidate": "statevector", "cell_id": "cf265906872b55f394d45c6cb56c013bd3f3e4dd775e9d6e23e496b75c32e06a", "operation_class": "two_qubit_wrapper_control", "operation_repeats": 256, "stability_status": "unavailable"}, {"candidate": "statevector", "cell_id": "6f72abf103d1a994f7ca0a0a4ff63139e19e8679cafbcae30fa14382e66115a0", "operation_class": "two_qubit_wrapper_control", "operation_repeats": 512, "stability_status": "unavailable"}, {"candidate": "statevector", "cell_id": "ae05f4262cc29de97a8bac964bc5e9eb7f03ac11cd96bf1503cd168b254cc24a", "operation_class": "two_qubit_wrapper_control", "operation_repeats": 1024, "stability_status": "unavailable"}, {"candidate": "statevector", "cell_id": "f8d2f3cb7e573cdc305fd343f1513dd3f5ef0bfa6d69e73b25ab988061430526", "operation_class": "two_qubit_cx_interleaved", "operation_repeats": 256, "stability_status": "unavailable"}, {"candidate": "statevector", "cell_id": "16c0a6ad5539f3051d685aa89945dab16ee116dcd4a4f7f7feb128c894534dc0", "operation_class": "two_qubit_cx_interleaved", "operation_repeats": 512, "stability_status": "unavailable"}, {"candidate": "statevector", "cell_id": "7286cf5792fb186c880edf1a928c25a299cdf75a3446fbb2081082d9c2459a77", "operation_class": "two_qubit_cx_interleaved", "operation_repeats": 1024, "stability_status": "unavailable"}, {"candidate": "mps_fixed_chi", "cell_id": "0fe4a26d1979391e2ab2e63a80c549850f735faab7e3e6df249a9459fd45c723", "operation_class": "one_qubit_noncommuting", "operation_repeats": 512, "stability_status": "ok"}]`
Caption and limitations: Units: seconds, logarithmic x-axis. Exact row set: all 10 rows in simulator/maestro_pilot_summary.csv, each with timed_n=15; row identity is cell_id. Clock: maestro_qcsim_process_isolated_reported_execution. Lines show min_reported_seconds to max_reported_seconds; symbols show the three session_medians_seconds. Limits: this was a stability pilot (9 statevector failures, 1 MPS pass), not predictor accuracy or common-panel execution; cause of timing spread was not established.
Sources:
- `simulator/maestro_pilot_summary.csv` — SHA-256 `d6107c87721695537cee9c67f3e7c51ab916f620a665b532487d1374e7d02ad5`

## Reproduction

From repository root, after installing `presentations/requirements-figures.txt`:

```bash
.venv-presentation/bin/python presentations/source/plot_report_figures.py
```

The script reads committed scorecard tables and the presentation brief and
writes under the new `work/presentation-figures/` directory. It does not
alter these exported figures or benchmark evidence. See [setup](../README.md).
