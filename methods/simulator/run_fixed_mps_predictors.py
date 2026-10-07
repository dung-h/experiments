#!/usr/bin/env python3
"""Run the frozen fixed-chi MPS predictors on the R9 recovered target.

The original S85 runner remains byte-pinned and supplies its existing graph,
Ridge, and median fit implementation.  This adapter replaces only the target
loader, records the R9 manifest SHA in every adapter receipt, and aggregates a
new target revision in a new output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
S85_PATH = ROOT / "benchmark_v1/scripts/run_mps_fixed_chi16_runtime_adaptation_v1.py"
S85_SCRIPT_DIR = S85_PATH.parent
DEFAULT_TARGET_MANIFEST = ROOT / "work/benchmark_recovery/r9_mps/targets_recovered_v1/manifest.json"
DEFAULT_OUTPUT = ROOT / "work/benchmark_recovery/r9_mps/fixed_chi16_recovered_v1"
FAMILY_PLAN = ROOT / "work/family_aware_joint_mps_ladder_v1/preflight"

if str(S85_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(S85_SCRIPT_DIR))
import run_mps_fixed_chi16_runtime_adaptation_v1 as s85  # noqa: E402

from recovered_target_inputs import (  # noqa: E402
    EXPECTED_HASHES,
    EXPECTED_C44_FOLD_COUNTS,
    EXPECTED_FAMILY_FOLD_COUNTS,
    RecoveredTargets,
    canonical_assignment_sha256,
    load_recovered_targets,
    read_csv,
    read_json,
    sha256_file,
)


def family_assignment_from_frozen_plan(plan_dir: Path | None = None) -> dict[str, int]:
    plan_dir = plan_dir or FAMILY_PLAN
    plan_dir = plan_dir.resolve()
    manifest_path = plan_dir / "preflight_manifest.json"
    table_path = plan_dir / "hash_feature_manifest.csv"
    manifest = read_json(manifest_path)
    expected = manifest.get("files", {}).get(table_path.name)
    if not expected or expected != sha256_file(table_path):
        raise ValueError("frozen family-holdout feature manifest hash mismatch")
    source_hash = manifest.get("input_hashes", {}).get("s85_static_feature_loader_sha256")
    if source_hash != sha256_file(S85_PATH):
        raise ValueError("family-holdout assignment was generated from another S85 loader")
    rows = read_csv(table_path)
    result: dict[str, int] = {}
    for row in rows:
        digest = row["source_qasm_sha256"]
        fold = int(row["family_holdout_fold_diagnostic_only"])
        if digest in result:
            raise ValueError(f"duplicate frozen family assignment for {digest}")
        result[digest] = fold
    if len(result) != EXPECTED_HASHES:
        raise ValueError("frozen family-holdout assignment must cover 150 unique hashes")
    return result


def check_split_authority(bundle: RecoveredTargets, fold_by_hash: dict[str, int],
                          plan_dir: Path | None = None) -> None:
    if fold_by_hash != bundle.fixed_fold_by_hash:
        raise ValueError("R9 fixed targets changed the frozen C44 hash/fold assignment")
    family = family_assignment_from_frozen_plan(plan_dir)
    if family != bundle.family_fold_by_hash:
        raise ValueError("R9 targets changed the frozen family-component holdout assignment")
    c44_pin = bundle.manifest["split_assignments"]["c44_split_assignment_sha256"]
    family_pin = bundle.manifest["split_assignments"]["family_holdout_split_assignment_sha256"]
    if (canonical_assignment_sha256(fold_by_hash, expected_counts=EXPECTED_C44_FOLD_COUNTS) != c44_pin
            or canonical_assignment_sha256(family, expected_counts=EXPECTED_FAMILY_FOLD_COUNTS) != family_pin):
        raise ValueError("R9 manifest split pins differ from the independently loaded frozen splits")


def fixed_target_map(bundle: RecoveredTargets) -> dict[str, dict[str, Any]]:
    return {row["source_qasm_sha256"]: dict(row) for row in bundle.fixed_rows}


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def load_fixed_inputs(bundle: RecoveredTargets, plan_dir: Path | None = None):
    """Join recovered labels to the already pinned feature/split inputs."""
    metadata, old_targets, graph_records, feature_by_hash = s85.load_inputs()
    fold_by_hash = metadata["fold_by_hash"]
    check_split_authority(bundle, fold_by_hash, plan_dir)
    old_by_hash = {row["source_qasm_sha256"]: row for row in old_targets}
    current_by_hash = fixed_target_map(bundle)
    if set(current_by_hash) != set(old_by_hash) or set(feature_by_hash) != set(old_by_hash):
        raise ValueError("recovered target hashes differ from the frozen S85 panel/features")

    feature_columns = [*s85.GLOBAL_NAMES, "node_count_after_strip", "edge_count_after_strip"]
    for digest in sorted(current_by_hash):
        target = current_by_hash[digest]
        old = old_by_hash[digest]
        if int(target["fold"]) != int(fold_by_hash[digest]):
            raise ValueError(f"recovered target moved C44 fold for {digest}")
        if target.get("prior_target_status") != old.get("target_status"):
            raise ValueError(f"prior target status differs from immutable old target for {digest}")
        previous = str(target.get("prior_target_seconds", ""))
        if old["target_status"] == "runtime_observed":
            if not _same_number(previous, old["target_seconds"]):
                raise ValueError(f"prior finite runtime differs from the old target for {digest}")
            if not _same_number(target["target_seconds"], old["target_seconds"]):
                raise ValueError(f"R9 changed a previously finite fixed-chi target for {digest}")
            if target["quality_status_separate"] != old["quality_status_separate"]:
                raise ValueError(f"R9 changed old quality annotation for {digest}")
        elif previous:
            raise ValueError(f"prior unavailable fixed target has a fabricated prior runtime for {digest}")
        globals_ = feature_by_hash[digest]["globals"]
        graph = feature_by_hash[digest]["graph"]
        for name in s85.GLOBAL_NAMES:
            if name not in target or not _same_number(target[name], globals_[name]):
                raise ValueError(f"R9 changed frozen structural feature {name} for {digest}")
        for name, actual in (("node_count_after_strip", len(graph["nodes"])),
                             ("edge_count_after_strip", len(graph["edge_src"]))):
            if name not in target or int(target[name]) != actual:
                raise ValueError(f"R9 changed frozen graph feature {name} for {digest}")

    # R2's recovered fixed target is required to fill exactly the six old
    # unavailable rows, while retaining the old 144 finite labels unchanged.
    recovered = [digest for digest, row in old_by_hash.items()
                 if row["target_status"] != "runtime_observed"]
    if len(recovered) != 6 or any(current_by_hash[digest]["target_status"] != "runtime_observed" for digest in recovered):
        raise ValueError("R9 fixed target must recover precisely the six old unavailable rows")
    return metadata, list(current_by_hash.values()), graph_records, feature_by_hash


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _path_label(path: Path) -> str:
    """Avoid machine-local absolute paths in receipts for external restored inputs."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return f"external-input/{resolved.name}"


def prepare(output_dir: Path, manifest_path: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite R9 fixed-MPS output: {output_dir}")
    bundle = load_recovered_targets(manifest_path)
    metadata, targets, _graphs, features = load_fixed_inputs(bundle)
    fold_by_hash = metadata["fold_by_hash"]
    rows = []
    for target in sorted(targets, key=lambda row: row["source_qasm_sha256"]):
        digest = target["source_qasm_sha256"]
        graph = features[digest]["graph"]
        row = {**target, **features[digest]["globals"],
               "node_count_after_strip": len(graph["nodes"]),
               "edge_count_after_strip": len(graph["edge_src"])}
        rows.append(row)
    output_dir.mkdir(parents=True)
    target_table = output_dir / "hash_targets_and_static_features.csv"
    _write_csv(target_table, rows)
    base_hashes = s85.input_hashes()
    input_hashes = {**base_hashes, "r9_target_manifest_sha256": bundle.manifest_sha256,
                    "r9_fixed_target_table_sha256": bundle.fixed_table_sha256}
    manifest = {
        "artifact_id": "r9-fixed-mps-recovered-target-preflight-v1",
        "status": "CPU_preflight_PASS_CUDA_not_started",
        "target_revision_id": bundle.target_revision_id,
        "r9_target_manifest_path": _path_label(bundle.manifest_path),
        "r9_target_manifest_sha256": bundle.manifest_sha256,
        "source_runner_lineage": bundle.runner_lineage,
        "target_table_sha256": sha256_file(target_table),
        "split_assignments": bundle.manifest["split_assignments"],
        "assigned_hashes": len(rows),
        "finite_runtime_labels": sum(row["target_status"] == "runtime_observed" for row in rows),
        "quality_status_counts": dict(Counter(row["quality_status_separate"] for row in rows)),
        "target_reduction": "R9 manifest fixed_chi16_targets.csv; labels joined by exact source QASM hash",
        "s85_input_sha256": base_hashes,
        "input_sha256": input_hashes,
        "runner_sha256": sha256_file(S85_PATH),
        "cuda_training_environment_preflight": None,
        "training_authorized": False,
        "model_configuration": {"folds": list(s85.FROZEN_FOLDS), "seeds": list(s85.SEEDS),
                                 "graph_epochs": 500, "graph_batch_size": 32,
                                 "ridge_alpha": 1.0, "cpu_baselines": ["train_fold_median", "ridge_alpha_1"]},
        "created_by_adapter_sha256": sha256_file(Path(__file__).resolve()),
    }
    qa = {
        "status": "PASS", "cuda_started": False, "training_performed": False,
        "target_manifest_sha256": bundle.manifest_sha256,
        "target_table_sha256": manifest["target_table_sha256"],
        "split_assignment_sha256": bundle.manifest["split_assignments"]["c44_split_assignment_sha256"],
        "assigned_hashes": len(rows), "finite_runtime_labels": len(rows),
        "feature_fold_hash_join_exact": True, "old_finite_labels_preserved": 144,
        "formerly_unavailable_labels_recovered": 6,
    }
    _write_json(output_dir / "preflight_manifest.json", manifest)
    _write_json(output_dir / "preflight_qa.json", qa)
    _write_json(output_dir / "r9_target_input_receipt.json", {
        "target_manifest_sha256": bundle.manifest_sha256,
        "source_runner_lineage": bundle.runner_lineage,
        "fixed_target_table_sha256": bundle.fixed_table_sha256,
        "materialized_feature_target_table_sha256": manifest["target_table_sha256"],
        "status": "CPU_preflight_PASS",
    })
    return manifest


def _refresh_bundle_and_inputs(manifest_path: Path, output_dir: Path):
    bundle = load_recovered_targets(manifest_path)
    metadata, targets, graphs, features = load_fixed_inputs(bundle)
    r9_receipt = read_json(output_dir / "r9_target_input_receipt.json")
    if r9_receipt.get("target_manifest_sha256") != bundle.manifest_sha256:
        raise ValueError("R9 target manifest differs from the prepared preflight")
    preflight = read_json(output_dir / "preflight_manifest.json")
    if preflight.get("r9_target_manifest_sha256") != bundle.manifest_sha256:
        raise ValueError("prepared fixed-MPS preflight pins another target manifest")
    if preflight.get("input_sha256") != {
        **s85.input_hashes(), "r9_target_manifest_sha256": bundle.manifest_sha256,
        "r9_fixed_target_table_sha256": bundle.fixed_table_sha256,
    }:
        raise ValueError("one or more fixed-MPS inputs changed after preflight")
    target_path = output_dir / "hash_targets_and_static_features.csv"
    if sha256_file(target_path) != preflight.get("target_table_sha256"):
        raise ValueError("materialized fixed target/features table changed after preflight")

    original_hashes = s85.input_hashes
    original_load = s85.load_inputs

    def pinned_hashes():
        now = load_recovered_targets(manifest_path)
        if now.manifest_sha256 != bundle.manifest_sha256:
            raise ValueError("R9 target manifest changed during a fit/aggregate stage")
        return {**original_hashes(), "r9_target_manifest_sha256": now.manifest_sha256,
                "r9_fixed_target_table_sha256": now.fixed_table_sha256}

    def recovered_load():
        now = load_recovered_targets(manifest_path)
        if now.manifest_sha256 != bundle.manifest_sha256:
            raise ValueError("R9 target manifest changed during target loading")
        base_metadata, _old, base_graphs, base_features = original_load()
        if base_metadata["fold_by_hash"] != metadata["fold_by_hash"]:
            raise ValueError("frozen C44 assignments changed after R9 preflight")
        return base_metadata, targets, base_graphs, base_features

    s85.input_hashes = pinned_hashes
    s85.load_inputs = recovered_load
    return bundle, metadata, targets, features, original_hashes, original_load


def cuda_preflight(output_dir: Path, manifest_path: Path) -> None:
    bundle, _metadata, _targets, _features, old_hashes, old_load = _refresh_bundle_and_inputs(manifest_path, output_dir)
    try:
        # This step only fingerprints the already installed CUDA environment;
        # it does not launch a model fit or simulator timing workload.
        with s85.exclusive_compute_lock():
            environment = s85.cuda_training_environment()
    finally:
        s85.input_hashes = old_hashes
        s85.load_inputs = old_load
    path = output_dir / "preflight_manifest.json"
    preflight = read_json(path)
    if preflight["r9_target_manifest_sha256"] != bundle.manifest_sha256:
        raise ValueError("R9 manifest pin changed during CUDA preflight")
    preflight["cuda_training_environment_preflight"] = environment
    preflight["status"] = "PASS_CUDA_environment_fingerprint_recorded"
    _write_json(path, preflight)
    qa_path = output_dir / "preflight_qa.json"
    qa = read_json(qa_path)
    qa["cuda_environment_fingerprint_recorded"] = True
    _write_json(qa_path, qa)


def fit(output_dir: Path, manifest_path: Path) -> dict[str, Any]:
    bundle, metadata, targets, _features, old_hashes, old_load = _refresh_bundle_and_inputs(manifest_path, output_dir)
    target_by_hash = {row["source_qasm_sha256"]: row for row in targets}
    preflight = read_json(output_dir / "preflight_manifest.json")
    expected_cuda = preflight.get("cuda_training_environment_preflight")
    if not isinstance(expected_cuda, dict):
        raise ValueError("run cuda-preflight before R9 fixed-MPS fitting")
    fold_reports = []
    try:
        for fold in s85.FROZEN_FOLDS:
            fold_dir = output_dir / f"fold_{fold}"
            if fold_dir.exists():
                report = s85.validate_fold_integrity(
                    output_dir, fold, metadata["fold_by_hash"], target_by_hash, expected_cuda)
                prior = read_json(fold_dir / "r9_target_input_receipt.json")
                if prior.get("target_manifest_sha256") != bundle.manifest_sha256:
                    raise ValueError(f"existing fold {fold} is pinned to another R9 target revision")
                fold_reports.append(report)
                continue
            # The base function uses the same source DAG/model/training code and
            # receives recovered targets only through the pinned in-memory loader.
            s85.fit_fold(output_dir, fold)
            qa_path = fold_dir / f"fold{fold}_qa.json"
            qa = read_json(qa_path)
            qa["r9_target_manifest_sha256"] = bundle.manifest_sha256
            _write_json(qa_path, qa)
            environment_path = fold_dir / "environment.json"
            env = read_json(environment_path)
            env["r9_target_manifest_sha256"] = bundle.manifest_sha256
            _write_json(environment_path, env)
            receipt = {"status": "PASS", "fold": fold,
                       "target_manifest_sha256": bundle.manifest_sha256,
                       "target_table_sha256": bundle.fixed_table_sha256,
                       "prediction_sha256": sha256_file(fold_dir / f"fold{fold}_predictions.csv")}
            _write_json(fold_dir / "r9_target_input_receipt.json", receipt)
            report = s85.validate_fold_integrity(
                output_dir, fold, metadata["fold_by_hash"], target_by_hash, expected_cuda)
            fold_reports.append(report)
            if fold == 0:
                print(json.dumps({"fold0_checkpoint": "TECHNICAL_QA_PASS",
                                  "continuation": "automatically_continue_folds_1_through_4",
                                  "metrics_consulted": False,
                                  "target_manifest_sha256": bundle.manifest_sha256}, sort_keys=True))
        receipt = {"artifact_id": "r9-fixed-mps-recovered-fit-v1",
                   "status": "five_fold_fit_technical_qa_pass_aggregation_pending",
                   "target_manifest_sha256": bundle.manifest_sha256,
                   "source_runner_lineage": bundle.runner_lineage,
                   "target_revision_id": bundle.target_revision_id,
                   "folds": fold_reports,
                   "training_methods": ["train_fold_median", "ridge_alpha_1",
                                        *[f"graph_seed_{seed}" for seed in s85.SEEDS],
                                        "graph_median_three_seeds"],
                   "metrics_quality_used_for_continuation": False}
        _write_json(output_dir / "r9_fit_receipt.json", receipt)
        return receipt
    finally:
        s85.input_hashes = old_hashes
        s85.load_inputs = old_load


def _metric(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    if not actual:
        return {"n": 0, "mae_seconds": None, "median_absolute_error_seconds": None,
                "log1p_mae": None, "r2": None, "p90_absolute_error_seconds": None,
                "p99_absolute_error_seconds": None, "max_absolute_error_seconds": None}
    import numpy as np
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    errors = np.abs(y - p)
    denominator = float(np.square(y - y.mean()).sum())
    return {"n": int(len(y)), "mae_seconds": float(errors.mean()),
            "median_absolute_error_seconds": float(np.median(errors)),
            "log1p_mae": float(np.abs(np.log1p(y) - np.log1p(p)).mean()),
            "r2": float(1.0 - np.square(y - p).sum() / denominator) if denominator else None,
            "p90_absolute_error_seconds": float(np.quantile(errors, .90)),
            "p99_absolute_error_seconds": float(np.quantile(errors, .99)),
            "max_absolute_error_seconds": float(errors.max())}


def aggregate(output_dir: Path, manifest_path: Path) -> dict[str, Any]:
    bundle, metadata, targets, _features, old_hashes, old_load = _refresh_bundle_and_inputs(manifest_path, output_dir)
    try:
        fit_receipt = read_json(output_dir / "r9_fit_receipt.json")
        if fit_receipt.get("status") != "five_fold_fit_technical_qa_pass_aggregation_pending" or fit_receipt.get("target_manifest_sha256") != bundle.manifest_sha256:
            raise ValueError("all five R9 fixed-MPS folds must pass on this exact target manifest")
        fold_by_hash = metadata["fold_by_hash"]
        targets_by_hash = {row["source_qasm_sha256"]: row for row in targets}
        combined: list[dict[str, str]] = []
        fold_hashes: dict[str, str] = {}
        for fold in s85.FROZEN_FOLDS:
            fold_dir = output_dir / f"fold_{fold}"
            receipt = read_json(fold_dir / "r9_target_input_receipt.json")
            if receipt.get("target_manifest_sha256") != bundle.manifest_sha256:
                raise ValueError(f"fixed-MPS fold {fold} target receipt mismatch")
            path = fold_dir / f"fold{fold}_predictions.csv"
            if sha256_file(path) != receipt.get("prediction_sha256"):
                raise ValueError(f"fixed-MPS fold {fold} prediction hash mismatch")
            rows = read_csv(path)
            expected = {digest for digest, assigned in fold_by_hash.items() if assigned == fold}
            hashes = [row.get("source_qasm_sha256", "") for row in rows]
            if len(hashes) != len(set(hashes)) or set(hashes) != expected:
                raise ValueError(f"fixed-MPS fold {fold} predictions do not match frozen test hashes")
            for row in rows:
                digest = row["source_qasm_sha256"]
                target = targets_by_hash[digest]
                if int(row["fold"]) != fold or row["target_status"] != target["target_status"] or row["quality_status_separate_audit_only"] != target["quality_status_separate"]:
                    raise ValueError(f"fixed-MPS fold {fold} target/split mismatch for {digest}")
                if not _same_number(row["target_seconds"], target["target_seconds"]):
                    raise ValueError(f"fixed-MPS fold {fold} runtime target mismatch for {digest}")
            fold_hashes[str(fold)] = sha256_file(path)
            combined.extend(rows)
        combined.sort(key=lambda row: row["source_qasm_sha256"])
        if len(combined) != EXPECTED_HASHES or {row["source_qasm_sha256"] for row in combined} != set(fold_by_hash):
            raise ValueError("fixed-MPS OOF must contain each of the 150 exact hashes once")
        methods = ["train_fold_median", "ridge_alpha_1", *[f"graph_seed_{s}" for s in s85.SEEDS], "graph_median_three_seeds"]
        finite_rows = [row for row in combined if targets_by_hash[row["source_qasm_sha256"]]["target_status"] == "runtime_observed"]
        quality_rows = [row for row in finite_rows if targets_by_hash[row["source_qasm_sha256"]]["quality_status_separate"] == "quality_pass"]
        metrics = []
        for population, selected in (("all_finite_runtime_targets", finite_rows), ("quality_pass_runtime_targets", quality_rows)):
            for method in methods:
                metric = _metric([float(row["target_seconds"]) for row in selected],
                                 [float(row[f"pred_{method}_seconds"]) for row in selected])
                metrics.append({"population": population, "method_id": method, **metric})
        output = output_dir / "r9_aggregate"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite fixed-MPS aggregate: {output}")
        output.mkdir()
        prediction_path = output / "five_fold_oof_predictions.csv"
        _write_csv(prediction_path, combined)
        _write_csv(output / "method_metrics.csv", metrics)
        summary = {
            "artifact_id": "r9-fixed-mps-recovered-oof-v1",
            "status": "five_fold_oof_complete_not_promoted",
            "target_revision_id": bundle.target_revision_id,
            "target_manifest_sha256": bundle.manifest_sha256,
            "source_runner_lineage": bundle.runner_lineage,
            "target_clock_id": s85.TARGET_ID,
            "split_assignment_sha256": bundle.manifest["split_assignments"]["c44_split_assignment_sha256"],
            "assigned_hashes": len(combined), "finite_runtime_targets": len(finite_rows),
            "quality_pass_runtime_targets": len(quality_rows),
            "quality_failed_or_other_finite_targets": len(finite_rows) - len(quality_rows),
            "fold_prediction_sha256": fold_hashes,
            "oof_predictions_sha256": sha256_file(prediction_path),
            "metrics": metrics,
            "quality_status_used_as_predictor": False,
            "unavailable_target_imputation": False,
            "training_metrics_used_for_fold_continuation": False,
        }
        _write_json(output / "manifest.json", summary)
        return summary
    finally:
        s85.input_hashes = old_hashes
        s85.load_inputs = old_load


def main(argv: list[str] | None = None) -> int:
    global FAMILY_PLAN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "cuda-preflight", "fit-five-fold-oof", "aggregate"), required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, default=FAMILY_PLAN,
                        help="Frozen family-component split plan inside this checkout")
    args = parser.parse_args(argv)
    FAMILY_PLAN = args.plan_dir.resolve()
    if args.action == "preflight":
        result = prepare(args.output_dir, args.target_manifest)
    elif args.action == "cuda-preflight":
        cuda_preflight(args.output_dir.resolve(), args.target_manifest.resolve())
        result = {"status": "CUDA_environment_fingerprint_recorded", "training_started": False}
    elif args.action == "fit-five-fold-oof":
        result = fit(args.output_dir.resolve(), args.target_manifest.resolve())
    else:
        result = aggregate(args.output_dir.resolve(), args.target_manifest.resolve())
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
