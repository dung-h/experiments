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
    "experiments/mali_split_audit.py",
    "experiments/build_mali_physical_dag.py",
    "experiments/evaluate_mali_physical_baselines.py",
    "experiments/train_mali_layer_dag_rnn.py",
    "experiments/build_mali_graph_summary_features.py",
    "experiments/evaluate_mali_feature_ablation.py",
    "experiments/mali_qwalk_forensics.py",
    "experiments/build_mali_critical_subgraph_features.py",
    "experiments/evaluate_mali_critical_subgraph.py",
    "experiments/mali_uncertainty_calibration.py",
    "experiments/train_mali_topology_controls.py",
    "experiments/evaluate_mali_topology_controls.py",
    "experiments/mali_strict_transfer_logical_screen.py",
    "experiments/evaluate_mali_strict_transfer_screen.py",
    "experiments/mali_ood_and_duplicate_validation.py",
    "experiments/requirements-mali-direct-estimator.txt",
    "experiments/README.md",
    "docs/MA_LI_DEEP_RUNTIME_ESTIMATOR_PROTOCOL_2026-09-21.md",
    "experiments/simulator_runtime_v1/README.md",
    "experiments/simulator_runtime_v1/requirements-evaluation.txt",
    "experiments/simulator_runtime_v1/run_torch_statevector_matrix.py",
    "experiments/simulator_runtime_v1/evaluate_torch_statevector_matrix.py",
    "experiments/simulator_runtime_v1/summarize_torch_statevector_matrix.py",
    "experiments/simulator_runtime_v2/README.md",
    "experiments/simulator_runtime_v2/EXPERIMENT_PROTOCOL.md",
    "experiments/simulator_runtime_v2/requirements-evaluation.txt",
    "experiments/simulator_runtime_v2/features.py",
    "experiments/simulator_runtime_v2/run_dense_statevector_v2.py",
    "experiments/simulator_runtime_v2/build_corpus.py",
    "experiments/simulator_runtime_v2/evaluate_runtime_v2.py",
    "experiments/simulator_runtime_v2/summarize_runtime_v2.py",
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
    "artifacts/validation/mali_deep_protocol_v1/split_audit/REPORT.md",
    "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv",
    "artifacts/validation/mali_deep_protocol_v1/split_audit/split_summary.json",
    "artifacts/validation/mali_physical_dag_v1/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/manifest.csv",
    "artifacts/validation/mali_physical_dag_v1/graph_records.csv",
    "artifacts/validation/mali_physical_dag_v1/build_summary.json",
    "artifacts/validation/mali_physical_dag_v1/graph_summary_features.csv",
    "artifacts/validation/mali_physical_dag_v1/graph_summary_features.json",
    "artifacts/validation/mali_physical_dag_v1/baseline_evaluation/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/baseline_evaluation/metrics.csv",
    "artifacts/validation/mali_physical_dag_v1/baseline_evaluation/oof_predictions.csv",
    "artifacts/validation/mali_physical_dag_v1/baseline_evaluation/summary.json",
    "artifacts/validation/mali_physical_dag_v1/layer_dag_rnn/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/layer_dag_rnn/metrics.csv",
    "artifacts/validation/mali_physical_dag_v1/layer_dag_rnn/oof_predictions.csv",
    "artifacts/validation/mali_physical_dag_v1/layer_dag_rnn/summary.json",
    "artifacts/validation/mali_physical_dag_v1/feature_ablation/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/feature_ablation/metrics.csv",
    "artifacts/validation/mali_physical_dag_v1/feature_ablation/oof_predictions.csv",
    "artifacts/validation/mali_physical_dag_v1/feature_ablation/summary.json",
    "artifacts/validation/mali_physical_dag_v1/qwalk_forensics/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/qwalk_forensics/qwalk_rows.csv",
    "artifacts/validation/mali_physical_dag_v1/qwalk_forensics/qwalk_feature_comparison.csv",
    "artifacts/validation/mali_physical_dag_v1/qwalk_forensics/summary.json",
    "artifacts/validation/mali_physical_dag_v1/critical_subgraph_features.csv",
    "artifacts/validation/mali_physical_dag_v1/critical_subgraph_features.json",
    "artifacts/validation/mali_physical_dag_v1/critical_subgraph_evaluation/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/critical_subgraph_evaluation/metrics.csv",
    "artifacts/validation/mali_physical_dag_v1/critical_subgraph_evaluation/oof_predictions.csv",
    "artifacts/validation/mali_physical_dag_v1/critical_subgraph_evaluation/summary.json",
    "artifacts/validation/mali_physical_dag_v1/uncertainty_calibration/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/uncertainty_calibration/duplicate_label_pairs.csv",
    "artifacts/validation/mali_physical_dag_v1/uncertainty_calibration/conformal_intervals.csv",
    "artifacts/validation/mali_physical_dag_v1/uncertainty_calibration/summary.json",
    "artifacts/validation/mali_physical_dag_v1/topology_controls_grouped_cpu_screen/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/topology_controls_grouped_cpu_screen/oof_predictions.csv",
    "artifacts/validation/mali_physical_dag_v1/topology_controls_grouped_cpu_seeds_2025_31415/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/topology_controls_grouped_cpu_seeds_2025_31415/oof_predictions.csv",
    "artifacts/validation/mali_physical_dag_v1/topology_controls_validation/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/topology_controls_validation/seed_mean_oof_predictions.csv",
    "artifacts/validation/mali_physical_dag_v1/topology_controls_validation/summary.json",
    "artifacts/validation/mali_physical_dag_v1/duplicate_aware_validation/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/duplicate_aware_validation/oof_predictions.csv",
    "artifacts/validation/mali_physical_dag_v1/duplicate_aware_validation/summary.json",
    "artifacts/validation/mali_physical_dag_v1/ood_uncertainty_validation/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/ood_uncertainty_validation/intervals.csv",
    "artifacts/validation/mali_physical_dag_v1/ood_uncertainty_validation/summary.json",
    "artifacts/validation/mali_strict_transfer_logical_v1/REPORT.md",
    "artifacts/validation/mali_strict_transfer_logical_v1/summary.json",
    "artifacts/validation/mali_strict_transfer_logical_v1/real_transfer_metrics.csv",
    "artifacts/validation/mali_strict_transfer_logical_v1/real_transfer_oof_predictions.csv",
    "artifacts/validation/mali_strict_transfer_logical_v1/evaluation/REPORT.md",
    "artifacts/validation/mali_strict_transfer_logical_v1/evaluation/seed_mean_oof_predictions.csv",
    "artifacts/validation/mali_strict_transfer_logical_v1/evaluation/summary.json",
    "experiments/mali_family_feature_space_audit.py",
    "experiments/mali_family_ood_common.py",
    "experiments/evaluate_mali_unseen_family_transfer.py",
    "experiments/evaluate_mali_simulator_assisted_family_transfer.py",
    "experiments/evaluate_mali_azizov_compiled_transfer.py",
    "experiments/evaluate_mali_azizov_transpiled_dag_transfer.py",
    "experiments/mali_transpiled_dag_common.py",
    "experiments/build_mali_ws_compiled_features.py",
    "experiments/build_mali_ws_physical_dag.py",
    "experiments/mali_graph_intervention_audit.py",
    "docs/FINDINGS_CLUSTERS.md",
    "docs/DAILY_WORK_LOG_2026-09-17_TO_2026-09-22.md",
    "docs/experiment_tracker.csv",
    "docs/REMOVED_AND_OUT_OF_SCOPE.md",
    "artifacts/validation/mali_family_ood_v1/REPORT.md",
    "artifacts/validation/mali_family_ood_v1/unseen_family_transfer/summary.json",
    "artifacts/validation/mali_family_ood_v1/simulator_assisted_transfer/summary.json",
    "artifacts/validation/mali_azizov_compiled_transfer_v1/REPORT.md",
    "artifacts/validation/mali_azizov_compiled_transfer_v1/summary.json",
    "artifacts/validation/mali_azizov_transpiled_dag_v1/REPORT.md",
    "artifacts/validation/mali_azizov_transpiled_dag_v1/summary.json",
    "artifacts/mali/graph_intervention_audit_20260921/REPORT.md",
    "artifacts/mali/graph_intervention_audit_20260921/metadata_and_metrics.json",
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
    "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/manifest.json",
    "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/records.jsonl",
    "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/records.csv",
    "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/canonical_corpus.csv",
    "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/canonical_corpus.summary.json",
    "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/results_summary.json",
    "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/REPORT.md",
    "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/evaluation_final/REPORT.md",
    "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/evaluation_final/aggregate_metrics.csv",
    "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/evaluation_final/oof_predictions.csv",
    "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/evaluation_final/per_fold_metrics.csv",
    "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/evaluation_final/resource_envelope.csv",
    "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/evaluation_final/resource_frontier_prediction.csv",
    "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/evaluation_final/summary.json",
    "docs/PAPER_REPO_EXPERIMENT_MAP_2026-09-21.md",
    "experiments/cudaq_runtime/run_cudaq_matrix.py",
    "experiments/cudaq_runtime/merge_cudaq_matrix.py",
    "experiments/cudaq_runtime/evaluate_cudaq_matrix.py",
    "artifacts/cudaq_runtime/cudaq_matrix_20260921/REPORT.md",
    "artifacts/cudaq_runtime/cudaq_matrix_20260921/cudaq_matrix.csv",
    "artifacts/cudaq_runtime/cudaq_matrix_20260921/cudaq_matrix.environment.json",
    "artifacts/cudaq_runtime/cudaq_matrix_20260921/evaluation/summary_warm_sample_median_s.csv",
    "artifacts/cudaq_runtime/cudaq_matrix_20260921/evaluation/summary_first_sample_s.csv",
    "experiments/cudaq_runtime/run_cudaq_mps_matrix.py",
    "experiments/cudaq_runtime/evaluate_cudaq_mps_matrix.py",
    "artifacts/cudaq_runtime/cudaq_mps_20260921/REPORT.md",
    "artifacts/cudaq_runtime/cudaq_mps_20260921/mps_matrix.csv",
    "artifacts/cudaq_runtime/cudaq_mps_20260921/mps_matrix.environment.json",
    "artifacts/cudaq_runtime/cudaq_mps_20260921/evaluation/mps_runtime_summary.csv",
    "artifacts/cudaq_runtime/cudaq_mps_20260921/evaluation/mps_runtime_metrics.csv",
    "artifacts/cudaq_runtime/cudaq_mps_20260921/evaluation/mps_runtime_evaluation.json",
    "tracks/quantum_rings/README.md",
    "tracks/quantum_rings/clone_upstreams.sh",
    "artifacts/quantum_rings/REPORT.md",
    "artifacts/quantum_rings/summary.json",
    "artifacts/quantum_rings/community_spirit_sprinters_evaluation.json",
    "artifacts/quantum_rings/community_softlocked_evaluation.json",
    "artifacts/quantum_rings/community_softlocked_training_table_cv.json",
    "tracks/quantum_rings/family_aware_paper/README.md",
    "tracks/quantum_rings/family_aware_paper/experiments/replicate_paper.py",
    "tracks/quantum_rings/family_aware_paper/experiments/protocol_audit.py",
    "tracks/quantum_rings/family_aware_paper/experiments/requirements-reproduction.txt",
    "artifacts/quantum_rings/family_aware_paper/README.md",
    "artifacts/quantum_rings/family_aware_paper/REPLICATION_SUMMARY_REPORT.md",
    "artifacts/quantum_rings/family_aware_paper/paper_replication.json",
    "artifacts/quantum_rings/family_aware_paper/protocol_audit.json",
    "artifacts/quantum_rings/family_aware_paper/mqt_pretrained_family_replication.json",
    "artifacts/quantum_rings/family_aware_paper/mqt_family_classifier/mqt_family_classifier.joblib",
    "artifacts/quantum_rings/family_aware_paper/mqt_family_classifier/mqt_family_classifier.json",
)

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def main() -> int:
    for relative in REQUIRED:
        require((ROOT / relative).is_file(), f"missing required artifact: {relative}")

    lock = json.loads((ROOT / "upstream/upstream.lock.json").read_text())
    require(set(lock["tracks"]) == {
        "mali", "qonductor", "cdaa", "qcre", "cutensornet", "vqcsim", "zero_setup",
        "quantum_rings"
    },
            "unexpected upstream lock tracks")
    artifact_manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
    require(len(artifact_manifest["artifacts"]) == 49,
            "unexpected replication artifact manifest size")
    require({c["id"] for c in artifact_manifest["clusters"]} == {
        "mali_azizov", "quantum_rings_solutions", "family_aware_paper"
    }, "finding-cluster index changed")

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

    quantum_rings = json.loads((ROOT / "artifacts/quantum_rings/summary.json").read_text())
    require(quantum_rings["contest"]["public_rows"] == 144,
            "Quantum Rings public row count changed")
    require(quantum_rings["contest"]["machine_profile_present"] is False,
            "Quantum Rings contest unexpectedly gained a machine profile")
    spirit_r2 = quantum_rings["spirit_sprinters"]["runtime_all"]["r2_log_runtime"]
    require(abs(spirit_r2 - 0.742841382483534) < 1e-12,
            "Spirit Sprinters public log-R2 fixture changed")
    soft_r2 = quantum_rings["softlocked"]["public_144"]["all_contexts"]["runtime"]["r2_log_runtime"]
    require(abs(soft_r2 + 25.460883474577965) < 1e-12,
            "SoftLocked public log-R2 fixture changed")
    in_domain = quantum_rings["softlocked"]["in_domain_training_table"]["runtime_r2_seconds"]
    require(abs(in_domain - 0.9673809598542934) < 1e-12,
            "SoftLocked in-domain R2 fixture changed")

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

    split_audit = json.loads(
        (ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/split_summary.json").read_text()
    )
    require(split_audit["n_rows"] == 340 and split_audit["n_qasm_hashes"] == 170,
            "Ma-Li deep split inventory changed")
    require(split_audit["paired_hashes"] == 130 and split_audit["n_family_components"] == 9,
            "Ma-Li deep paired/family inventory changed")
    require(split_audit["strict_hash_overlap_max"] == 0,
            "Ma-Li strict split has circuit-hash overlap")

    physical_summary = json.loads(
        (ROOT / "artifacts/validation/mali_physical_dag_v1/build_summary.json").read_text()
    )
    require(physical_summary["n_label_rows"] == 340 and physical_summary["n_unique_graphs"] == 300,
            "Ma-Li physical-DAG corpus inventory changed")
    require(physical_summary["totals"] == {
        "nodes": 67404145,
        "edges": 79673830,
        "active_qubits_max": 127,
        "duration_missing_nodes": 0,
    }, "Ma-Li physical-DAG totals changed")

    physical_eval = json.loads(
        (ROOT / "artifacts/validation/mali_physical_dag_v1/baseline_evaluation/summary.json").read_text()
    )
    require(physical_eval["models"] == ["qcre", "compiled", "physical_graph", "rich_summary"],
            "Ma-Li physical baseline model set changed")
    require(abs(physical_eval["pooled_metrics"]["grouped_qasm"]["rich_summary"]["r2_log1p_seconds"] - 0.8896) < 1e-4,
            "Ma-Li rich static baseline fixture changed")
    layer_summary = json.loads(
        (ROOT / "artifacts/validation/mali_physical_dag_v1/layer_dag_rnn/summary.json").read_text()
    )
    require(layer_summary["model"] == "physics_guided_coarsened_layer_DAG_GRU_residual"
            and layer_summary["max_bins"] == 256,
            "Ma-Li layer-DAG pilot provenance changed")
    ablation = json.loads(
        (ROOT / "artifacts/validation/mali_physical_dag_v1/feature_ablation/summary.json").read_text()
    )
    require(set(ablation["feature_blocks"]) == {
        "qcre", "compiled", "graph_size", "layer", "timing", "edge", "opcode", "rich_summary"
    }, "Ma-Li feature-ablation block set changed")
    require(abs(ablation["pooled_metrics"]["grouped_qasm"]["rich_summary"]["r2_log1p_seconds"] - 0.8895291914860881) < 1e-12,
            "Ma-Li feature-ablation rich-summary fixture changed")
    qwalk = json.loads(
        (ROOT / "artifacts/validation/mali_physical_dag_v1/qwalk_forensics/summary.json").read_text()
    )
    require(qwalk["n_qwalk_rows"] == 2 and qwalk["qwalk_rows"] == ["osaka:33", "kyoto:215"],
            "Ma-Li QWalk forensics inventory changed")
    critical_eval = json.loads(
        (ROOT / "artifacts/validation/mali_physical_dag_v1/critical_subgraph_evaluation/summary.json").read_text()
    )
    require(abs(critical_eval["pooled_metrics"]["grouped_qasm"]["rich_plus_critical"]["r2_log1p_seconds"] - 0.9040886649438724) < 1e-12,
            "Ma-Li critical-subgraph grouped fixture changed")
    uncertainty = json.loads(
        (ROOT / "artifacts/validation/mali_physical_dag_v1/uncertainty_calibration/summary.json").read_text()
    )
    require(uncertainty["duplicate_cells"] == 40 and uncertainty["duplicate_rows"] == 80,
            "Ma-Li duplicate-label inventory changed")
    require(abs(uncertainty["duplicate_label_noise"]["absolute_diff_median_seconds"] - 0.4171114753333338) < 1e-12,
            "Ma-Li duplicate-label noise fixture changed")
    topology = json.loads(
        (ROOT / "artifacts/validation/mali_physical_dag_v1/topology_controls_validation/summary.json").read_text()
    )
    require(topology["seeds"] == [1234, 2025, 31415] and topology["n_qasm_hashes"] == 170,
            "Ma-Li topology-control seed/group inventory changed")
    topology_delta = topology["paired_group_bootstrap"]["true_dag_minus_node_only"]
    require(abs(topology_delta["ensemble_delta_log_r2"] + 0.0283929182403706) < 1e-12,
            "Ma-Li topology-control log-R2 fixture changed")
    require(topology_delta["probability_log_r2_better_than_node_only"] < 0.05,
            "Ma-Li topology control unexpectedly supports true DAG")
    strict_transfer = json.loads(
        (ROOT / "artifacts/validation/mali_strict_transfer_logical_v1/summary.json").read_text()
    )
    require(strict_transfer["source_rows_after_filter"] == 2636
            and strict_transfer["removed_real_qasm_hashes"] == 170
            and strict_transfer["removed_source_rows_matching_real_qasm"] == 384,
            "Ma-Li strict transfer leakage-filter inventory changed")
    strict_eval = json.loads(
        (ROOT / "artifacts/validation/mali_strict_transfer_logical_v1/evaluation/summary.json").read_text()
    )
    require(strict_eval["seeds"] == [1234, 2025, 31415]
            and strict_eval["n_rows"] == 340 and strict_eval["n_qasm_hashes"] == 170,
            "Ma-Li strict transfer seed/group inventory changed")
    require(strict_eval["aggregate_seed_mean_predictions"]["frozen_transfer"]["mae_seconds"]
            > strict_eval["aggregate_seed_mean_predictions"]["scratch"]["mae_seconds"],
            "Ma-Li frozen transfer unexpectedly beats scratch")
    duplicate_aware = json.loads(
        (ROOT / "artifacts/validation/mali_physical_dag_v1/duplicate_aware_validation/summary.json").read_text()
    )
    require(duplicate_aware["n_rows"] == 340 and duplicate_aware["n_exact_cells"] == 300,
            "Ma-Li duplicate-aware inventory changed")
    require(duplicate_aware["pooled"]["strict_all"]["cell_mean_noise_weighted"]["mae_seconds"]
            < duplicate_aware["pooled"]["strict_all"]["row_weighted"]["mae_seconds"],
            "Ma-Li duplicate-aware strict fixture changed")
    ood = json.loads(
        (ROOT / "artifacts/validation/mali_physical_dag_v1/ood_uncertainty_validation/summary.json").read_text()
    )
    grouped_ood = ood["intervals"]["grouped_qasm_support_stratified_alpha_0.1"]
    require(ood["n_rows"] == 340 and grouped_ood["qwalk_coverage"] == 0.0,
            "Ma-Li OOD interval fixture changed")
    require(grouped_ood["qwalk_median_width_seconds"]
            > ood["intervals"]["grouped_qasm_global_alpha_0.1"]["qwalk_median_width_seconds"],
            "Ma-Li OOD stratification no longer widens QWalk interval")

    compiled_transfer = json.loads(
        (ROOT / "artifacts/validation/mali_azizov_compiled_transfer_v1/summary.json").read_text()
    )
    require(compiled_transfer["coverage"]["n_sim_ws"] == 3020
            and compiled_transfer["coverage"]["n_qpu"] == 340,
            "Ma-Li compiled-transfer coverage changed")
    require(abs(compiled_transfer["sim_grouped"]["sim_ws_T2"]["r2_log1p_seconds"]
                - 0.4348365036165025) < 1e-12,
            "Ma-Li compiled-transfer simulator T2 fixture changed")
    require(abs(compiled_transfer["transfer_incl"]["qpu_scratch_T2"]["r2_log1p_seconds"]
                - 0.8744650476488727) < 1e-12,
            "Ma-Li compiled-transfer QPU-scratch T2 fixture changed")
    require(abs(compiled_transfer["transfer_incl"]["matched_incl_T2"]["r2_log1p_seconds"]
                - 0.35325364810606574) < 1e-12,
            "Ma-Li compiled-transfer affine T2 fixture changed")

    dag_transfer = json.loads(
        (ROOT / "artifacts/validation/mali_azizov_transpiled_dag_v1/summary.json").read_text()
    )
    require(dag_transfer["n_qpu"] == 340 and dag_transfer["n_sim"] == 3020,
            "Ma-Li transpiled-DAG coverage changed")
    require(abs(dag_transfer["metrics"]["qpu_scratch_transpiled_dag"]["r2_log1p_seconds"]
                - 0.9046986222821181) < 1e-12,
            "Ma-Li transpiled-DAG QPU-scratch fixture changed")
    require(abs(dag_transfer["metrics"]["sim_transpiled_dag"]["r2_log1p_seconds"]
                - 0.7787466977930845) < 1e-12,
            "Ma-Li transpiled-DAG simulator fixture changed")
    require(abs(dag_transfer["metrics"]["sim_transpiled_dag_affine_qpu"]["r2_log1p_seconds"]
                + 2.6288523063619085) < 1e-12,
            "Ma-Li transpiled-DAG affine fixture changed")
    require(abs(dag_transfer["metrics"]["sim_transpiled_dag_finetune_qpu"]["r2_log1p_seconds"]
                - 0.8750076986105026) < 1e-12,
            "Ma-Li transpiled-DAG fine-tune fixture changed")

    family_c = json.loads(
        (ROOT / "artifacts/validation/mali_family_ood_v1/unseen_family_transfer/summary.json").read_text()
    )
    require(family_c["n_qpu_rows"] == 340 and family_c["n_components"] == 9,
            "Ma-Li family-OOD C inventory changed")
    require(abs(family_c["pooled_all_components"]["scratch_T3"]["r2_log1p_seconds"]
                - 0.8700313238166076) < 1e-12,
            "Ma-Li family-OOD C T3 fixture changed")
    require(abs(family_c["pooled_excluding_qwalk"]["scratch_T1"]["r2_log1p_seconds"]
                - 0.8588729692273314) < 1e-12,
            "Ma-Li family-OOD C T1 fixture changed")
    require(family_c["experiment_b_status"] == "blocked_no_qpu_labels_for_12_missing_families",
            "Ma-Li historical Experiment B status changed")

    family_b2 = json.loads(
        (ROOT / "artifacts/validation/mali_family_ood_v1/simulator_assisted_transfer/summary.json").read_text()
    )
    require(family_b2["experiment"] == "B2_simulator_assisted_hardware_unseen_family",
            "Ma-Li family-OOD B2 experiment id changed")
    require(abs(family_b2["pooled_excluding_qwalk"]["sim_T2proxy_incl_affine_qpu"]["r2_log1p_seconds"]
                - 0.24825559477681192) < 1e-12,
            "Ma-Li family-OOD B2 T2-incl fixture changed")
    require(family_b2["max_abs_scratch_log_r2_delta_vs_experiment_c"] == 0.0,
            "Ma-Li family-OOD B2 scratch no longer matches C")

    paper_rep = json.loads(
        (ROOT / "artifacts/quantum_rings/family_aware_paper/paper_replication.json").read_text()
    )
    require(paper_rep["paper"] == "arXiv:2606.11620"
            and paper_rep["replication_recorded"] == "2026-09-18",
            "family-aware paper provenance changed")
    require(abs(paper_rep["protocols"]["0.75"]["models"]["rf"]["runtime"]["r2_log_runtime"]
                - 0.7482836995550155) < 1e-12,
            "family-aware 0.75 RF fixture changed")
    require(abs(paper_rep["protocols"]["0.75"]["models"]["family_predicted"]["runtime"]["r2_log_runtime"]
                + 0.25276169665440307) < 1e-12,
            "family-aware 0.75 family-MLP fixture changed")
    require(abs(paper_rep["protocols"]["0.99"]["models"]["family_predicted"]["runtime"]["r2_log_runtime"]
                + 0.8561158663118285) < 1e-12,
            "family-aware 0.99 family-MLP fixture changed")

    protocol_audit = json.loads(
        (ROOT / "artifacts/quantum_rings/family_aware_paper/protocol_audit.json").read_text()
    )
    require(abs(protocol_audit["protocols"]["0.75"]["tree_baselines"]["runtime"]["gradient_boosting"]["r2_log_runtime"]
                - 0.7573463411144011) < 1e-12,
            "family-aware 0.75 Gradient Boosting fixture changed")
    require(abs(protocol_audit["protocols"]["0.99"]["tree_baselines"]["runtime"]["extra_trees"]["r2_log_runtime"]
                - 0.7115234168134515) < 1e-12,
            "family-aware 0.99 ExtraTrees fixture changed")

    mqt_family = json.loads(
        (ROOT / "artifacts/quantum_rings/family_aware_paper/mqt_pretrained_family_replication.json").read_text()
    )
    require(abs(mqt_family["protocols"]["0.75"]["evaluation_family_accuracy_all"]
                - 1 / 3) < 1e-12,
            "family-aware frozen MQT transfer accuracy changed")

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

    dense_v2_root = ROOT / "artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920"
    with (dense_v2_root / "records.csv").open(newline="") as handle:
        dense_v2_raw = list(csv.DictReader(handle))
    require(len(dense_v2_raw) == 112, "unexpected dense-statevector v2 raw row count")
    require(sum(row["status"] == "ok" for row in dense_v2_raw) == 108,
            "dense-statevector v2 successful raw coverage changed")
    require(sum(row["status"] == "resource_limit" for row in dense_v2_raw) == 4,
            "dense-statevector v2 resource-limit coverage changed")
    require({row["family"] for row in dense_v2_raw} == {"random_matching", "random_star"},
            "dense-statevector v2 family coverage changed")
    with (dense_v2_root / "canonical_corpus.csv").open(newline="") as handle:
        dense_v2_corpus = list(csv.DictReader(handle))
    require(len(dense_v2_corpus) == 208, "unexpected dense-statevector v2 canonical row count")
    require(sum(row["status"] == "ok" for row in dense_v2_corpus) == 204,
            "dense-statevector v2 canonical duration coverage changed")
    require({row["source_run"] for row in dense_v2_corpus} == {"v1_reference", "v2_structural"},
            "dense-statevector v2 source provenance changed")
    require(len({row["circuit_id"] for row in dense_v2_corpus}) == 68,
            "dense-statevector v2 logical-circuit coverage changed")
    require({int(row["num_qubits"]) for row in dense_v2_corpus} == {16, 20, 24, 26, 28},
            "dense-statevector v2 width coverage changed")

    with (dense_v2_root / "evaluation_final/aggregate_metrics.csv").open(newline="") as handle:
        dense_v2_metrics = list(csv.DictReader(handle))
    dense_v2_index = {(row["split"], row["model"]): row for row in dense_v2_metrics}
    logical_grouped = dense_v2_index[("circuit_group", "logical_hgb")]
    analytical_width = dense_v2_index[("width_held_out", "analytical_calibrated_ridge")]
    logical_width = dense_v2_index[("width_held_out", "logical_hgb")]
    require(int(logical_grouped["n_test"]) == 204 and float(logical_grouped["r2_log"]) > 0.97,
            "dense-statevector v2 circuit-group logical estimator changed")
    require(float(analytical_width["r2_log"]) > float(logical_width["r2_log"]),
            "dense-statevector v2 width-extrapolation finding changed")

    with (dense_v2_root / "evaluation_final/resource_frontier_prediction.csv").open(
        newline=""
    ) as handle:
        frontier_rows = list(csv.DictReader(handle))
    frontier = {(row["context_id"], int(row["num_qubits"])): row for row in frontier_rows}
    c128_q28 = frontier[("cuda:complex128", 28)]
    c64_q28 = frontier[("cuda:complex64", 28)]
    require(c128_q28["raw_statevector_fits"] == "True" and c128_q28["envelope_fits"] == "False",
            "dense-statevector v2 q28 complex128 envelope boundary changed")
    require(int(c128_q28["observed_resource_limit"]) == 4,
            "dense-statevector v2 q28 complex128 resource evidence changed")
    require(c64_q28["envelope_fits"] == "True" and int(c64_q28["observed_ok"]) == 4,
            "dense-statevector v2 q28 complex64 feasibility evidence changed")

    cudaq_root = ROOT / "artifacts/cudaq_runtime/cudaq_matrix_20260921"
    with (cudaq_root / "cudaq_matrix.csv").open(newline="") as handle:
        cudaq_rows = list(csv.DictReader(handle))
    require(len(cudaq_rows) == 44, "CUDA-Q pilot row count changed")
    require({row["status"] for row in cudaq_rows} == {"ok"},
            "CUDA-Q pilot contains a non-ok row")
    require({row["target_name"] for row in cudaq_rows} == {"cpu_fp64", "gpu_fp32", "gpu_fp64"},
            "CUDA-Q target coverage changed")
    require({row["family"] for row in cudaq_rows} == {
        "ghz", "hea", "qaoa_cycle", "random_brickwork"
    }, "CUDA-Q family coverage changed")
    require(all(float(row["first_sample_s"]) > 0 for row in cudaq_rows),
            "CUDA-Q first-call timings are not positive")
    require(all(float(row["warm_sample_median_s"]) > 0 for row in cudaq_rows),
            "CUDA-Q warm-call timings are not positive")
    cudaq_manifest = json.loads((cudaq_root / "cudaq_matrix.environment.json").read_text())
    require(cudaq_manifest["cudaq_version"].startswith("CUDA-Q Version 0.15.1"),
            "CUDA-Q environment version changed")

    mps_root = ROOT / "artifacts/cudaq_runtime/cudaq_mps_20260921"
    with (mps_root / "mps_matrix.csv").open(newline="") as handle:
        mps_rows = list(csv.DictReader(handle))
    require(len(mps_rows) == 48, "CUDA-Q MPS pilot row count changed")
    require({row["status"] for row in mps_rows} == {"ok"},
            "CUDA-Q MPS pilot contains a non-ok row")
    require({row["mps_max_bond_config"] for row in mps_rows} == {"2", "4", "8", "16"},
            "CUDA-Q MPS bond-cap coverage changed")
    require({row["family"] for row in mps_rows} == {
        "ghz", "hea", "qaoa_cycle", "random_brickwork"
    }, "CUDA-Q MPS family coverage changed")
    require(all(float(row["state_warm_median_s"]) > 0 for row in mps_rows),
            "CUDA-Q MPS state timings are not positive")
    require(all(float(row["sample_warm_median_s"]) > 0 for row in mps_rows),
            "CUDA-Q MPS sample timings are not positive")
    mps_evaluation = json.loads(
        (mps_root / "evaluation/mps_runtime_evaluation.json").read_text()
    )
    require(mps_evaluation["rows_used"] == 48,
            "CUDA-Q MPS evaluator row count changed")
    require("observed_max_bond" in mps_evaluation["excluded_post_run_fields"],
            "CUDA-Q MPS leakage guard changed")

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

    print("OK: replication reports, fixtures, provenance lock, Ma-Li seed-1234 diagnostic, dense-statevector matrix, CUDA-Q dense pilot and CUDA-Q MPS pilot verified")
    print("This check does not rerun GPU timing, retrain models or contact external services.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
