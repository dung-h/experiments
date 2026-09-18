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
    "experiments/qonductor_mapping_audit.py",
    "experiments/mali_qcre_proxy.py",
    "experiments/mali_qcre_validation.py",
    "experiments/mali_qcre_seed_sensitivity.py",
    "experiments/README.md",
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
)

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def main() -> int:
    for relative in REQUIRED:
        require((ROOT / relative).is_file(), f"missing required artifact: {relative}")

    lock = json.loads((ROOT / "upstream/upstream.lock.json").read_text())
    require(set(lock["tracks"]) == {"mali", "qonductor", "cdaa", "qcre", "cutensornet"},
            "unexpected upstream lock tracks")
    artifact_manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
    require(len(artifact_manifest["artifacts"]) == 15,
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

    print("OK: replication reports, fixtures, provenance lock and Ma-Li seed-1234 diagnostic verified")
    print("This check does not rerun GPU timing, retrain models or contact external services.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
