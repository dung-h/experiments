#!/usr/bin/env python3
"""Validate the versioned replication capsule without external dependencies."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "README.md",
    "RESULTS.md",
    "REPRODUCTION.md",
    "upstream/upstream.lock.json",
    "artifacts/manifest.json",
    "tracks/qonductor/qonductor_local_replication.patch",
    "tracks/mali/README.md",
    "tracks/cdaa_qcre/README.md",
    "tracks/cdaa_qcre/data/instruction_durations/SNAPSHOT_MANIFEST.csv",
    "tracks/cdaa_qcre/independent/sqgm/exp-in/exp_eagle.json",
    "tracks/cdaa_qcre/independent/sqgm/exp-in/exp_heron.json",
    "tracks/cutensornet/code/run_cutensornet_runtime_benchmark.py",
    "tracks/cutensornet/code/with_cutensornet_env.sh",
    "experiments/qonductor_mapping_audit.py",
    "experiments/mali_qcre_proxy.py",
    "experiments/mali_qcre_validation.py",
    "experiments/mali_qcre_seed_sensitivity.py",
    "experiments/run_mali_direct_estimator.py",
    "experiments/requirements-mali-direct-estimator.txt",
    "experiments/README.md",
    "experiments/simulator_runtime_v1/README.md",
    "experiments/simulator_runtime_v1/requirements-evaluation.txt",
    "experiments/simulator_runtime_v1/run_torch_statevector_matrix.py",
    "experiments/simulator_runtime_v1/evaluate_torch_statevector_matrix.py",
    "experiments/simulator_runtime_v1/summarize_torch_statevector_matrix.py",
    "experiments/simulator_papers/README.md",
    "experiments/simulator_papers/run_zero_setup_pps_canary.py",
    "experiments/simulator_papers/run_zero_setup_pps_grid.py",
    "experiments/simulator_papers/evaluate_zero_setup_pps_estimator.py",
    "experiments/simulator_papers/evaluate_vqcsim_runtime_estimator.py",
    "experiments/simulator_papers/evaluate_aer_runtime_intervals.py",
    "experiments/simulator_papers/relativize_vqcsim_records.py",
    "experiments/simulator_papers/run_explicit_plan_ranking_proxy.py",
    "experiments/simulator_papers/evaluate_explicit_plan_ranking_proxy.py",
    "experiments/simulator_papers/evaluate_family_aware_threshold_proxy.py",
    "experiments/simulator_papers/run_precision_selection_proxy.py",
    "experiments/simulator_papers/evaluate_precision_selection_proxy.py",
    "experiments/simulator_papers/evaluate_precision_policy.py",
    "experiments/simulator_papers/analyze_validation_pass.py",
    "tracks/pasqal_emu_mps/scripts/run_threshold_extension.py",
    "artifacts/mali/fold10_qwalk_outlier.json",
    "artifacts/qonductor/reproduction_metrics.json",
    "artifacts/validation/qonductor_mapping/QONDUCTOR_MAPPING_AUDIT.md",
    "artifacts/validation/qonductor_mapping/qonductor_mapping_audit.json",
    "artifacts/validation/mali_qcre_proxy/MALI_QCRE_PROXY_REPORT.md",
    "artifacts/validation/mali_qcre_proxy/mali_qcre_proxy_features.csv",
    "artifacts/validation/mali_qcre_proxy/mali_qcre_proxy_summary.json",
    "artifacts/validation/mali_qcre_final/MALI_QCRE_FINAL_VALIDATION_REPORT.md",
    "artifacts/validation/mali_qcre_final/mali_qcre_grouped_validation_rows.csv",
    "artifacts/validation/mali_qcre_final/mali_qcre_validation_summary.json",
    "artifacts/validation/mali_qcre_seed_sensitivity/MALI_QCRE_SEED_SENSITIVITY_REPORT.md",
    "artifacts/validation/mali_qcre_seed_sensitivity/mali_qcre_seed_sensitivity_rows.csv",
    "artifacts/validation/mali_qcre_seed_sensitivity/mali_qcre_seed_sensitivity_summary.json",
    "artifacts/validation/mali_direct_estimator/MALI_DIRECT_COMPILED_ESTIMATOR_REPORT.md",
    "artifacts/validation/mali_direct_estimator/mali_direct_estimator_per_fold.csv",
    "artifacts/validation/mali_direct_estimator/mali_direct_estimator_oof_predictions.csv",
    "artifacts/validation/mali_direct_estimator/mali_direct_estimator_summary.csv",
    "artifacts/validation/mali_direct_estimator/mali_direct_estimator_strict_backend_metrics.csv",
    "artifacts/validation/mali_direct_estimator/mali_direct_estimator_provenance.json",
    "artifacts/cdaa_qcre/independent_all_metrics.csv",
    "artifacts/cutensornet/cutensornet_runtime_benchmark.csv",
    "artifacts/azizov_independent/p0_smoke.csv",
    "artifacts/azizov_independent/p0_smoke.environment.json",
    "artifacts/azizov_independent/P0_REPORT.md",
    "artifacts/azizov_independent/p0_baselines.json",
    "tracks/azizov_independent/README.md",
    "tracks/azizov_independent/scripts/build_manifest.py",
    "tracks/azizov_independent/scripts/run_p0_smoke.py",
    "tracks/azizov_independent/scripts/summarize_p0.py",
    "tracks/azizov_independent/scripts/train_p0_baselines.py",
    "tracks/azizov_independent/scripts/compare_feature_blocks.py",
    "tracks/azizov_independent/scripts/run_matrix.py",
    "tracks/azizov_independent/scripts/summarize_matrix.py",
    "tracks/azizov_independent/scripts/build_feasible_subset.py",
    "tracks/azizov_independent/scripts/evaluate_subset_splits.py",
    "tracks/azizov_independent/scripts/run_resource_canary.py",
    "artifacts/azizov_independent/p1_q16_stratified3.csv",
    "artifacts/azizov_independent/p1_q16_stratified3.environment.json",
    "artifacts/azizov_independent/P1_Q16_STRATIFIED3_REPORT.md",
    "artifacts/azizov_independent/p1_q16_stratified3_baselines.json",
    "artifacts/azizov_independent/p1_q16_stratified3_feature_ablation.json",
    "artifacts/azizov_independent/FEASIBLE_Q9_REPORT.md",
    "artifacts/azizov_independent/feasible_q9.csv",
    "artifacts/azizov_independent/rejected_local_screen.csv",
    "artifacts/azizov_independent/feasible_q9_runtime.csv",
    "artifacts/azizov_independent/feasible_q9_runtime.environment.json",
    "artifacts/azizov_independent/feasible_q9_baselines.json",
    "artifacts/azizov_independent/feasible_q9_feature_ablation.json",
    "artifacts/azizov_independent/FEASIBLE_Q9_SPLIT_EVALUATION.md",
    "artifacts/azizov_independent/FEASIBLE_Q9_SPLIT_EVALUATION.json",
    "artifacts/azizov_independent/Q10_Q16_RESOURCE_CANARY_PARTIAL_REPORT.md",
    "artifacts/azizov_independent/q10_q16_resource_canary_partial.csv",
    "artifacts/azizov_independent/q10_q16_resource_canary_partial.environment.json",
    "artifacts/azizov_independent/Q10_Q16_HIGHMEM_CHECK_REPORT.md",
    "artifacts/azizov_independent/q10_q16_highmem_canary.csv",
    "artifacts/azizov_independent/q10_q16_highmem_canary.environment.json",
    "artifacts/azizov_independent/Q10_Q16_30MIN_INSIGHT_REPORT.md",
    "artifacts/azizov_independent/q10_q16_30min_canary.csv",
    "artifacts/simulator_papers/INITIAL_LOCAL_REPRODUCTION_REPORT_2026-09-20.md",
    "artifacts/simulator_papers/SIMULATOR_ESTIMATOR_VALIDATION_REPORT_2026-09-20.md",
    "artifacts/simulator_papers/DEEP_ANALYSIS_2026-09-20.md",
    "artifacts/simulator_papers/DEEP_ANALYSIS_2026-09-20.json",
    "artifacts/simulator_papers/vqcsim_rq2_gpu_fp32_16_24_20260920/records.csv",
    "artifacts/simulator_papers/vqcsim_rq2_gpu_fp32_16_24_20260920/records.json",
    "artifacts/simulator_papers/vqcsim_rq2_gpu_fp32_16_24_20260920/manifest.json",
    "artifacts/simulator_papers/vqcsim_rq2_gpu_fp32_16_24_20260920/run_summary.json",
    "artifacts/simulator_papers/zero_setup_pps_local_canary_20260920.jsonl",
    "artifacts/simulator_papers/zero_setup_pps_local_canary_20260920.environment.json",
    "artifacts/simulator_papers/SP4_SP6_INDEPENDENT_PROBES_2026-09-20.md",
    "artifacts/simulator_papers/explicit_plan_ranking_proxy_20260920/candidates.csv",
    "artifacts/simulator_papers/explicit_plan_ranking_proxy_20260920/evaluation.json",
    "artifacts/simulator_papers/family_aware_emu_mps_proxy_20260920/evaluation.csv",
    "artifacts/simulator_papers/family_aware_emu_mps_proxy_20260920/evaluation.json",
    "artifacts/simulator_papers/precision_selection_proxy_20260920/pairs.csv",
    "artifacts/simulator_papers/precision_selection_proxy_20260920/evaluation.json",
    "artifacts/simulator_papers/vqcsim_rq2_gpu_fp32_mirror_16_24_20260920/records.csv",
    "artifacts/simulator_papers/vqcsim_rq2_gpu_fp32_mirror_16_24_20260920/run_summary.json",
    "artifacts/simulator_papers/vqcsim_estimator_base_mirror_20260920/evaluation.json",
    "artifacts/simulator_papers/aer_interval_validation_20260920/evaluation.json",
    "artifacts/simulator_papers/zero_setup_pps_grid_20260920.jsonl",
    "artifacts/simulator_papers/zero_setup_pps_estimator_20260920/evaluation.json",
    "artifacts/simulator_papers/explicit_plan_ranking_expanded_20260920/candidates_connected.csv",
    "artifacts/simulator_papers/explicit_plan_ranking_expanded_20260920/candidates_connected.failures.json",
    "artifacts/simulator_papers/explicit_plan_ranking_expanded_20260920/evaluation_connected.json",
    "artifacts/simulator_papers/emu_mps_threshold_extension_20260920/threshold_labels.csv",
    "artifacts/simulator_papers/family_aware_emu_mps_expanded_20260920/evaluation.json",
    "artifacts/simulator_papers/precision_selection_expanded_20260920/records.csv",
    "artifacts/simulator_papers/precision_selection_expanded_20260920/policy.json",
    "artifacts/simulator_runtime_v1/torch_dense_statevector_v1/manifest.json",
    "artifacts/simulator_runtime_v1/torch_dense_statevector_v1/records.jsonl",
    "artifacts/simulator_runtime_v1/torch_dense_statevector_v1/records.csv",
    "artifacts/simulator_runtime_v1/torch_dense_statevector_v1/results_summary.json",
    "artifacts/simulator_runtime_v1/torch_dense_statevector_v1/REPORT.md",
    "artifacts/simulator_runtime_v1/torch_dense_statevector_v1/evaluation/aggregate_metrics.csv",
    "artifacts/simulator_runtime_v1/torch_dense_statevector_v1/evaluation/coverage.csv",
    "artifacts/simulator_runtime_v1/torch_dense_statevector_v1/evaluation/oof_predictions.csv",
    "artifacts/simulator_runtime_v1/torch_dense_statevector_v1/evaluation/per_fold_metrics.csv",
    "artifacts/simulator_runtime_v1/torch_dense_statevector_v1/evaluation/summary.json",
)

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def main() -> int:
    for relative in REQUIRED:
        require((ROOT / relative).is_file(), f"missing required artifact: {relative}")

    lock = json.loads((ROOT / "upstream/upstream.lock.json").read_text())
    require(set(lock["tracks"]) == {
        "mali", "qonductor", "cdaa", "qcre", "cutensornet", "vqcsim", "zero_setup"
    },
            "unexpected upstream lock tracks")
    artifact_manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
    require(len(artifact_manifest["artifacts"]) == 31,
            "unexpected replication artifact manifest size")

    qonductor = json.loads((ROOT / "artifacts/qonductor/reproduction_metrics.json").read_text())
    execution = qonductor["execution_time"]
    require(abs(execution["qonductor_vs_real"]["r2"] - 0.9386347722408748) < 1e-12,
            "Qonductor regression R2 fixture changed")
    require(abs(execution["dag_vs_real"]["r2"] - 0.8848516371292685) < 1e-12,
            "Qonductor DAG R2 fixture changed")

    mapping = json.loads(
        (ROOT / "artifacts/validation/qonductor_mapping/qonductor_mapping_audit.json").read_text()
    )
    require(mapping["csv"]["rows"] == 100, "Qonductor mapping CSV row count changed")
    require(mapping["csv"]["has_stable_join_key"] is False,
            "Qonductor mapping unexpectedly gained a stable join key")

    with (ROOT / "artifacts/validation/mali_qcre_proxy/mali_qcre_proxy_features.csv").open(
        newline=""
    ) as handle:
        proxy_rows = list(csv.DictReader(handle))
    require(len(proxy_rows) == 340, "Ma-Li/QCRE proxy row count changed")
    require({row["backend_label"] for row in proxy_rows} == {"osaka", "kyoto"},
            "Ma-Li/QCRE proxy backend labels changed")
    require(all(not row["qasm_path"].startswith("/") for row in proxy_rows),
            "Ma-Li/QCRE proxy contains non-portable absolute QASM paths")
    proxy_summary = json.loads(
        (ROOT / "artifacts/validation/mali_qcre_proxy/mali_qcre_proxy_summary.json").read_text()
    )
    require(proxy_summary["n"] == 340, "Ma-Li/QCRE proxy summary row count changed")
    software = proxy_summary["software_and_backend_metadata"]
    require(software["qiskit_version"] == "1.4.1", "Qiskit proxy version changed")
    require(software["qiskit_ibm_runtime_version"] == "0.36.1",
            "Qiskit IBM Runtime proxy version changed")
    for backend_name, backend in software["backends"].items():
        require(backend["num_qubits"] == 127,
                f"{backend_name} FakeBackend qubit count changed")
        require(len(backend["coupling_map_edges"]) == 144,
                f"{backend_name} FakeBackend coupling map changed")
        require(backend["basis_gates"] == ["ecr", "id", "rz", "sx", "x"],
                f"{backend_name} FakeBackend basis gates changed")
    require(software["backends"]["osaka"]["backend_version"] == "1.0.16",
            "FakeOsaka backend snapshot version changed")
    require(software["backends"]["kyoto"]["backend_version"] == "1.2.29",
            "FakeKyoto backend snapshot version changed")
    require(
        abs(
            proxy_summary["features"]["physical_depth"]["five_fold_log_calibration"]["r2_seconds"]
            - 0.7847371915381873
        )
        < 1e-12,
        "Ma-Li/QCRE proxy metric fixture changed",
    )

    final_summary = json.loads(
        (ROOT / "artifacts/validation/mali_qcre_final/mali_qcre_validation_summary.json").read_text()
    )
    require(final_summary["split"]["n_unique_logical_circuits"] == 170,
            "Ma-Li grouped logical-circuit count changed")
    require(final_summary["duration_audit"]["coverage_status"] == "complete",
            "Ma-Li duration coverage is no longer complete")
    require(
        abs(final_summary["single_feature_grouped_cv"]["physical_depth"]["aggregate"]["r2_seconds"]
            - 0.7879012317585139) < 1e-12,
        "Ma-Li grouped physical-depth fixture changed",
    )

    seed_summary = json.loads(
        (ROOT / "artifacts/validation/mali_qcre_seed_sensitivity/mali_qcre_seed_sensitivity_summary.json").read_text()
    )
    require(seed_summary["seeds"] == [1234, 2025, 31415],
            "Ma-Li sensitivity seed list changed")
    require(
        seed_summary["stability"]["physical_depth"]["std_r2"] < 0.01,
        "Ma-Li physical-depth seed sensitivity became unstable",
    )

    with (ROOT / "artifacts/validation/mali_direct_estimator/mali_direct_estimator_summary.csv").open(
        newline=""
    ) as handle:
        direct_summary = list(csv.DictReader(handle))
    require(len(direct_summary) == 28, "unexpected Ma-Li direct-estimator summary size")
    require(
        {row["split"] for row in direct_summary}
        == {
            "grouped_logical_circuit_5fold",
            "paired_backend_transfer_diagnostic",
            "backend_and_logical_circuit_held_out",
            "family_held_out",
        },
        "Ma-Li direct-estimator split set changed",
    )
    direct_index = {(row["split"], row["model"]): row for row in direct_summary}
    direct_grouped = direct_index[("grouped_logical_circuit_5fold", "compiled_ridge")]
    direct_strict = direct_index[("backend_and_logical_circuit_held_out", "compiled_ridge")]
    require(int(direct_grouped["n_oof_rows"]) == 340,
            "Ma-Li direct-estimator grouped OOF coverage changed")
    require(abs(float(direct_grouped["r2_log"]) - 0.8777095522833426) < 1e-12,
            "Ma-Li direct-estimator grouped compiled Ridge fixture changed")
    require(abs(float(direct_strict["r2_log"]) - 0.8741660138813845) < 1e-12,
            "Ma-Li direct-estimator strict compiled Ridge fixture changed")
    with (ROOT / "artifacts/validation/mali_direct_estimator/mali_direct_estimator_oof_predictions.csv").open(
        newline=""
    ) as handle:
        direct_predictions = list(csv.DictReader(handle))
    require(len(direct_predictions) == 9_520,
            "Ma-Li direct-estimator OOF prediction coverage changed")
    direct_provenance = json.loads(
        (ROOT / "artifacts/validation/mali_direct_estimator/mali_direct_estimator_provenance.json").read_text()
    )
    require(direct_provenance["input"]
            == "artifacts/validation/mali_qcre_proxy/mali_qcre_proxy_features.csv",
            "Ma-Li direct-estimator provenance is no longer portable")
    require(direct_provenance["n_logical_circuit_groups"] == 170,
            "Ma-Li direct-estimator logical-group count changed")
    require(direct_provenance["software"] == {
        "python": "3.14.4",
        "numpy": "2.5.3",
        "pandas": "3.0.5",
        "scikit_learn": "1.9.0",
    }, "Ma-Li direct-estimator software fixture changed")

    with (ROOT / "artifacts/cutensornet/cutensornet_runtime_benchmark.csv").open(newline="") as handle:
        cutn_rows = list(csv.DictReader(handle))
    require(len(cutn_rows) == 25, "unexpected cuTensorNet baseline row count")
    require({"cutensornet_runtime_est_s", "contract_gpu_median_s", "end_to_end_first_s"}
            <= set(cutn_rows[0]), "cuTensorNet target columns changed")

    with (ROOT / "artifacts/cdaa_qcre/independent_all_metrics.csv").open(newline="") as handle:
        cdaa_rows = list(csv.DictReader(handle))
    require(len(cdaa_rows) == 315, "unexpected CDAA/QCRE independent metric row count")

    with (ROOT / "artifacts/azizov_independent/p0_smoke.csv").open(newline="") as handle:
        azizov_rows = list(csv.DictReader(handle))
    require(len(azizov_rows) == 168, "unexpected Azizov P0 row count")
    require({row["status"] for row in azizov_rows} == {"ok"},
            "Azizov P0 contains an unexpected failure status")
    require({row["backend_class"] for row in azizov_rows} == {"FakeWashingtonV2", "FakeSherbrooke"},
            "Azizov P0 backend coverage changed")
    azizov_baselines = json.loads(
        (ROOT / "artifacts/azizov_independent/p0_baselines.json").read_text()
    )
    require(azizov_baselines["input_rows"] == 168,
            "Azizov P0 baseline input row count changed")
    azizov_ablation = json.loads(
        (ROOT / "artifacts/azizov_independent/p0_feature_ablation.json").read_text()
    )
    require(set(azizov_ablation["blocks"]) == {"source", "compiled", "hybrid"},
            "Azizov P0 feature blocks changed")
    with (ROOT / "artifacts/azizov_independent/p1_q16_stratified3.csv").open(newline="") as handle:
        azizov_p1_rows = list(csv.DictReader(handle))
    require(len(azizov_p1_rows) == 504, "unexpected Azizov P1 subset row count")
    require({row["status"] for row in azizov_p1_rows} == {"ok"},
            "Azizov P1 subset contains an unexpected failure status")
    p1_baselines = json.loads(
        (ROOT / "artifacts/azizov_independent/p1_q16_stratified3_baselines.json").read_text()
    )
    require(p1_baselines["input_rows"] == 504,
            "Azizov P1 baseline input row count changed")
    with (ROOT / "artifacts/azizov_independent/feasible_q9.csv").open(newline="") as handle:
        eligible_rows = list(csv.DictReader(handle))
    with (ROOT / "artifacts/azizov_independent/rejected_local_screen.csv").open(newline="") as handle:
        rejected_rows = list(csv.DictReader(handle))
    require(len(eligible_rows) == 149 and len(rejected_rows) == 1253,
            "Azizov local-screen manifest counts changed")
    with (ROOT / "artifacts/azizov_independent/feasible_q9_runtime.csv").open(newline="") as handle:
        feasible_runtime_rows = list(csv.DictReader(handle))
    require(len(feasible_runtime_rows) == 1192,
            "Azizov feasible runtime row count changed")
    require({row["status"] for row in feasible_runtime_rows} == {"ok"},
            "Azizov feasible subset contains a timeout or error")
    feasible_baselines = json.loads(
        (ROOT / "artifacts/azizov_independent/feasible_q9_baselines.json").read_text()
    )
    require(feasible_baselines["input_rows"] == 1192,
            "Azizov feasible baseline input count changed")
    feasible_split_eval = json.loads(
        (ROOT / "artifacts/azizov_independent/FEASIBLE_Q9_SPLIT_EVALUATION.json").read_text()
    )
    require(feasible_split_eval["input_rows"] == 1192,
            "Azizov feasible split-evaluation input count changed")
    require(set(feasible_split_eval["splits"]) == {
        "random_row",
        "grouped_circuit",
        "family_held_out",
        "backend_held_out_FakeSherbrooke",
        "backend_held_out_FakeWashingtonV2",
    }, "Azizov feasible split-evaluation split set changed")
    with (ROOT / "artifacts/azizov_independent/q10_q16_resource_canary_partial.csv").open(
        newline=""
    ) as handle:
        canary_rows = list(csv.DictReader(handle))
    require(len(canary_rows) == 7, "unexpected partial q10-q16 canary row count")
    require(sum(row["status"] == "ok" for row in canary_rows) == 3,
            "q10-q16 canary success count changed")
    require(sum(row["status"] == "resource_limit" for row in canary_rows) == 4,
            "q10-q16 canary resource-limit count changed")
    require(max(float(row["peak_rss_mb"]) for row in canary_rows) > 16_000,
            "q10-q16 canary RSS evidence changed")
    canary_env = json.loads(
        (ROOT / "artifacts/azizov_independent/q10_q16_resource_canary_partial.environment.json").read_text()
    )
    require(canary_env["processed_rows"] == 7 and canary_env["completion_status"].startswith("aborted"),
            "q10-q16 canary provenance changed")
    with (ROOT / "artifacts/azizov_independent/q10_q16_highmem_canary.csv").open(
        newline=""
    ) as handle:
        highmem_rows = list(csv.DictReader(handle))
    require(len(highmem_rows) == 2, "unexpected high-memory canary row count")
    require({row["status"] for row in highmem_rows} == {"resource_limit"},
            "high-memory canary status changed")
    require(all(row["stop_reason"].startswith("wall_timeout") for row in highmem_rows),
            "high-memory canary stop reason changed")
    highmem_env = json.loads(
        (ROOT / "artifacts/azizov_independent/q10_q16_highmem_canary.environment.json").read_text()
    )
    require(highmem_env["max_rss_gib"] == 36.0 and highmem_env["processed_rows"] == 2,
            "high-memory canary provenance changed")
    with (ROOT / "artifacts/azizov_independent/q10_q16_30min_canary.csv").open(
        newline=""
    ) as handle:
        long_timeout_rows = list(csv.DictReader(handle))
    require(len(long_timeout_rows) == 2, "unexpected 30-minute q10-q16 canary row count")
    require({row["status"] for row in long_timeout_rows} == {"timeout"},
            "30-minute q10-q16 canary must remain right-censored timeouts")
    require(all(row["stop_reason"].startswith("wall_timeout") for row in long_timeout_rows),
            "30-minute q10-q16 canary stop reason changed")
    require(min(float(row["monitor_wall_s"]) for row in long_timeout_rows) >= 1_800,
            "30-minute q10-q16 canary duration evidence changed")
    require(max(float(row["peak_rss_mb"]) for row in long_timeout_rows) > 16_000,
            "30-minute q10-q16 canary RSS evidence changed")

    vqcsim_root = ROOT / "artifacts/simulator_papers/vqcsim_rq2_gpu_fp32_16_24_20260920"
    with (vqcsim_root / "records.csv").open(newline="") as handle:
        vqcsim_rows = list(csv.DictReader(handle))
    require(len(vqcsim_rows) == 45, "unexpected VQCSim local-canary row count")
    require({row["terminal_status"] for row in vqcsim_rows} == {"ok"},
            "VQCSim local canary contains a non-ok row")
    require(all(row["parity_passed"] == "True" for row in vqcsim_rows),
            "VQCSim local canary lost a parity pass")
    require({row["family"] for row in vqcsim_rows}
            == {"qaoa", "qnn", "vqe_real_amp", "vqe_su2", "vqe_two_local"},
            "VQCSim local-canary family set changed")
    require({int(row["size"]) for row in vqcsim_rows} == set(range(16, 25)),
            "VQCSim local-canary width coverage changed")
    vqcsim_path_fields = (
        "qasm3_path", "qasm3_bound_path", "qasm2_parameterized_path", "qasm2_bound_path"
    )
    require(all(not row[field].startswith("/") for row in vqcsim_rows for field in vqcsim_path_fields),
            "VQCSim records contain non-portable absolute QASM paths")
    vqcsim_summary = json.loads((vqcsim_root / "run_summary.json").read_text())
    require(vqcsim_summary["stats"]["ok_records"] == 45,
            "VQCSim local-canary summary changed")

    pps_path = ROOT / "artifacts/simulator_papers/zero_setup_pps_local_canary_20260920.jsonl"
    pps_rows = [json.loads(line) for line in pps_path.read_text().splitlines() if line.strip()]
    require(len(pps_rows) == 25 and all("error" not in row for row in pps_rows),
            "Zero-Setup PPS local-canary success coverage changed")
    expected_deltas = {1e-2, 5e-3, 1e-3, 5e-4, 1e-4}
    require({float(row["delta"]) for row in pps_rows} == expected_deltas,
            "Zero-Setup PPS delta grid changed")
    require(all(sum(float(row["delta"]) == delta for row in pps_rows) == 5
                for delta in expected_deltas),
            "Zero-Setup PPS repeat count changed")
    pps_environment = json.loads(
        (ROOT / "artifacts/simulator_papers/zero_setup_pps_local_canary_20260920.environment.json").read_text()
    )
    require(pps_environment["pauli_prop"] == "0.2.1" and pps_environment["qiskit"] == "2.5.2",
            "Zero-Setup PPS environment changed")

    plan_root = ROOT / "artifacts/simulator_papers/explicit_plan_ranking_proxy_20260920"
    with (plan_root / "candidates.csv").open(newline="") as handle:
        plan_rows = list(csv.DictReader(handle))
    require(len(plan_rows) == 45 and {row["status"] for row in plan_rows} == {"ok"},
            "explicit plan-ranking proxy coverage changed")
    require({row["candidate_plan"] for row in plan_rows}
            == {"native_time_tuned", "left_fold", "right_fold"},
            "explicit plan-ranking candidate set changed")
    plan_summary = json.loads((plan_root / "evaluation.json").read_text())
    require(plan_summary["n_complete_circuit_groups"] == 15,
            "explicit plan-ranking group count changed")
    require(0.9 <= plan_summary["flop_exact_fastest_rate"] <= 1.0,
            "explicit plan-ranking selection rate is outside checked range")

    family_root = ROOT / "artifacts/simulator_papers/family_aware_emu_mps_proxy_20260920"
    with (family_root / "evaluation.csv").open(newline="") as handle:
        family_rows = list(csv.DictReader(handle))
    require(len(family_rows) == 24,
            "family-aware EMU-MPS proxy row count changed")
    require({row["method"] for row in family_rows}
            == {"shared_numeric", "inferred_family_residual"},
            "family-aware EMU-MPS proxy methods changed")
    family_summary = json.loads((family_root / "evaluation.json").read_text())
    require(family_summary["n_labels"] == 12,
            "family-aware EMU-MPS label count changed")
    require(family_summary["split"].startswith("leave-one-size-out"),
            "family-aware EMU-MPS split changed")

    precision_root = ROOT / "artifacts/simulator_papers/precision_selection_proxy_20260920"
    with (precision_root / "pairs.csv").open(newline="") as handle:
        precision_rows = list(csv.DictReader(handle))
    require(len(precision_rows) == 40 and {row["precision"] for row in precision_rows}
            == {"complex64", "complex128"},
            "precision proxy pair coverage changed")
    precision_summary = json.loads((precision_root / "evaluation.json").read_text())
    require(precision_summary["n_pairs"] == 20,
            "precision proxy matched-pair count changed")
    require(2.0 < precision_summary["complex128_over_complex64_time_ratio_median"] < 5.0,
            "precision proxy median time ratio outside checked range")

    vqcsim_mirror_root = ROOT / "artifacts/simulator_papers/vqcsim_rq2_gpu_fp32_mirror_16_24_20260920"
    with (vqcsim_mirror_root / "records.csv").open(newline="") as handle:
        vqcsim_mirror_rows = list(csv.DictReader(handle))
    require(len(vqcsim_mirror_rows) == 45 and {row["terminal_status"] for row in vqcsim_mirror_rows} == {"ok"},
            "VQCSim mirror extension coverage changed")
    require(all(row["parity_passed"] == "True" for row in vqcsim_mirror_rows),
            "VQCSim mirror extension lost a parity pass")
    vqcsim_expanded = json.loads(
        (ROOT / "artifacts/simulator_papers/vqcsim_estimator_base_mirror_20260920/evaluation.json").read_text()
    )
    require(vqcsim_expanded["n_rows"] == 90 and vqcsim_expanded["dataset_variants"] == ["base", "mirror"],
            "VQCSim base/mirror estimator coverage changed")

    aer_interval = json.loads(
        (ROOT / "artifacts/simulator_papers/aer_interval_validation_20260920/evaluation.json").read_text()
    )
    require(aer_interval["n_successful_rows"] == 1192 and len(aer_interval["splits"]) == 3,
            "Aer interval validation coverage changed")
    aer_by_split = {row["split"]: row for row in aer_interval["splits"]}
    require(aer_by_split["backend_held_out"]["interval_coverage"] < 0.5,
            "Aer backend-held-out interval unexpectedly no longer exposes domain shift")

    pps_grid = [
        json.loads(line)
        for line in (ROOT / "artifacts/simulator_papers/zero_setup_pps_grid_20260920.jsonl").read_text().splitlines()
        if line.strip()
    ]
    require(len(pps_grid) == 36 and all("error" not in row for row in pps_grid),
            "Zero-Setup PPS controlled grid coverage changed")
    require({int(row["num_trotter_steps"]) for row in pps_grid} == {4, 8, 12, 20},
            "Zero-Setup PPS Trotter grid changed")
    pps_estimator = json.loads(
        (ROOT / "artifacts/simulator_papers/zero_setup_pps_estimator_20260920/evaluation.json").read_text()
    )
    require(pps_estimator["n_successful_rows"] == 36 and len(pps_estimator["splits"]) == 9,
            "Zero-Setup PPS estimator evaluation changed")

    plan_expanded_root = ROOT / "artifacts/simulator_papers/explicit_plan_ranking_expanded_20260920"
    with (plan_expanded_root / "candidates_connected.csv").open(newline="") as handle:
        plan_expanded_rows = list(csv.DictReader(handle))
    plan_failures = json.loads((plan_expanded_root / "candidates_connected.failures.json").read_text())
    require(len(plan_expanded_rows) == 98 and len(plan_failures) == 7,
            "expanded explicit-plan feasibility coverage changed")
    plan_expanded = json.loads((plan_expanded_root / "evaluation_connected.json").read_text())
    require(plan_expanded["n_complete_circuit_groups"] == 15,
            "expanded explicit-plan group coverage changed")
    require(abs(plan_expanded["flop_exact_fastest_rate"] - 0.8) < 1e-12,
            "expanded explicit-plan FLOP baseline fixture changed")

    emu_extension_root = ROOT / "artifacts/simulator_papers/emu_mps_threshold_extension_20260920"
    with (emu_extension_root / "threshold_labels.csv").open(newline="") as handle:
        emu_extension_rows = list(csv.DictReader(handle))
    require(len(emu_extension_rows) == 12,
            "EMU-MPS threshold extension label count changed")
    require(all(row["reference_censored"] == "False" for row in emu_extension_rows),
            "EMU-MPS threshold extension gained a censored reference")
    emu_expanded = json.loads(
        (ROOT / "artifacts/simulator_papers/family_aware_emu_mps_expanded_20260920/evaluation.json").read_text()
    )
    require(emu_expanded["n_labels"] == 24 and emu_expanded["results"]["shared_numeric"]["n"] == 24,
            "expanded EMU-MPS threshold evaluation changed")

    precision_expanded_root = ROOT / "artifacts/simulator_papers/precision_selection_expanded_20260920"
    with (precision_expanded_root / "records.csv").open(newline="") as handle:
        precision_expanded_rows = list(csv.DictReader(handle))
    require(len(precision_expanded_rows) == 64 and {row["precision"] for row in precision_expanded_rows} == {"complex64", "complex128"},
            "precision extension raw coverage changed")
    precision_policy = json.loads((precision_expanded_root / "policy.json").read_text())
    require(precision_policy["n_pairs"] == 52 and precision_policy["unsafe_complex64_pairs"] == 28,
            "precision-policy matched-pair fixture changed")

    deep_analysis = json.loads(
        (ROOT / "artifacts/simulator_papers/DEEP_ANALYSIS_2026-09-20.json").read_text()
    )
    require(deep_analysis["vqcsim"]["n_matched_base_mirror_pairs"] == 45,
            "deep VQCSim paired diagnostic coverage changed")
    require(deep_analysis["aer"]["backend_held_out_by_backend"]
            and len(deep_analysis["aer"]["backend_held_out_by_backend"]) == 2,
            "deep Aer backend diagnostic coverage changed")
    require(deep_analysis["plan_ranking"]["failure_kinds"] == {"memory_limit": 6, "unsupported": 1},
            "deep plan-feasibility diagnostic changed")
    require(deep_analysis["precision"]["family_held_out_unsafe_complex64_selections"] == 2,
            "deep precision holdout diagnostic changed")

    dense_root = ROOT / "artifacts/simulator_runtime_v1/torch_dense_statevector_v1"
    with (dense_root / "records.csv").open(newline="") as handle:
        dense_rows = list(csv.DictReader(handle))
    require(len(dense_rows) == 96, "unexpected dense-statevector matrix row count")
    require({row["status"] for row in dense_rows} == {"ok"},
            "dense-statevector matrix contains a non-ok row")
    require(len({row["circuit_id"] for row in dense_rows}) == 24,
            "dense-statevector logical-circuit coverage changed")
    require({row["context_id"] for row in dense_rows} == {
        "cpu:complex64", "cpu:complex128", "cuda:complex64", "cuda:complex128"
    }, "dense-statevector execution-context coverage changed")
    dense_summary = json.loads((dense_root / "results_summary.json").read_text())
    require(dense_summary["successful_rows"] == 96,
            "dense-statevector summary success count changed")
    require(dense_summary["logical_circuits"] == 24,
            "dense-statevector summary circuit count changed")
    require(dense_summary["maximum_output_difference"] < 1e-5,
            "dense-statevector scalar sanity difference became unexpectedly large")
    with (dense_root / "evaluation/aggregate_metrics.csv").open(newline="") as handle:
        dense_metrics = list(csv.DictReader(handle))
    dense_metric_index = {(row["split"], row["model"]): row for row in dense_metrics}
    require(len(dense_metrics) == 15,
            "dense-statevector aggregate metric coverage changed")
    grouped_hgb = dense_metric_index[("circuit_group", "hist_gradient_log")]
    width_hgb = dense_metric_index[("width_held_out", "hist_gradient_log")]
    require(int(grouped_hgb["n_test"]) == 96 and float(grouped_hgb["r2_log"]) > 0.9,
            "dense-statevector grouped HGB fixture changed")
    require(float(width_hgb["r2_log"]) < 0.1,
            "dense-statevector width holdout no longer records the tree extrapolation failure")

    with (ROOT / "tracks/cdaa_qcre/data/instruction_durations/SNAPSHOT_MANIFEST.csv").open(
        newline=""
    ) as handle:
        snapshots = list(csv.DictReader(handle))
    require(len(snapshots) == 6, "unexpected instruction-duration snapshot count")
    for row in snapshots:
        path = ROOT / "tracks/cdaa_qcre/data/instruction_durations" / row["packaged_file"]
        require(path.is_file(), f"missing duration snapshot: {path.name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == row["sha256"], f"duration snapshot hash changed: {path.name}")

    for path in (ROOT / "tracks/cdaa_qcre/independent").rglob("*.py"):
        require(
            "/home/server/Documents/cdaa" not in path.read_text(encoding="utf-8"),
            f"non-portable absolute CDAA path remains: {path.relative_to(ROOT)}",
        )

    outlier = json.loads((ROOT / "artifacts/mali/fold10_qwalk_outlier.json").read_text())
    require(outlier["split"]["seed"] == 1234, "Ma-Li outlier seed changed")
    require(outlier["split"]["fold"] == 10, "Ma-Li outlier fold changed")
    require(outlier["circuit"] == "qwalk-noancilla_indep_qiskit_9",
            "Ma-Li outlier circuit changed")
    require(abs(outlier["predictions_seconds"]["pretrained_osaka_test"] - 36.0312) < 1e-4,
            "Ma-Li pretrained fixture changed")
    require(abs(outlier["predictions_seconds"]["from_scratch_osaka_test"] - 12.8724) < 1e-4,
            "Ma-Li from-scratch fixture changed")

    print("OK: replication reports, fixtures, provenance lock, Ma-Li seed-1234 diagnostic and dense-statevector matrix verified")
    print("This check does not rerun GPU timing, retrain models or contact external services.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
