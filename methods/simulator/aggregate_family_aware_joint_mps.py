#!/usr/bin/env python3
"""Validate five frozen folds and aggregate joint MPS runtime/quality OOF results.

The C44 evaluation and family-component holdout diagnostic are written to
separate tables.  This script does not fit models or rewrite fold artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "benchmark_v1/protocol/family_aware_joint_runtime_quality_mps_v1.json"
PLAN = ROOT / "work/family_aware_joint_mps_ladder_v1/preflight"
MEASUREMENTS = ROOT / "artifacts/benchmark_v3/simulator/family_aware_joint_mps_ladder_v1/full"
TRAINING = ROOT / "artifacts/benchmark_v3/simulator/family_aware_joint_mps_ladder_v1/training"
METHODS = ("family_conditioned", "family_agnostic_ablation")
CHI = (2, 4, 8, 16, 32, 64)
SEEDS = (42, 1234, 31415)
ATTEMPT_STATUSES = {"ok", "timeout", "resource_limit", "worker_error",
                    "unsupported", "adapter_error", "technical_error"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_measurement_ledger(summary: dict[str, Any], attempts: list[dict[str, Any]],
                                expected_hashes: set[str]) -> dict[str, Any]:
    """Validate a complete terminal ledger while preserving partial outcomes.

    A full campaign can have a PARTIAL scientific result even when every
    assigned hash/rung/session identity reached a terminal state. Such rows
    remain unavailable targets; they are never dropped or imputed.
    """
    if summary.get("scope") != "full" or summary.get("status") not in {"PASS", "PARTIAL"}:
        raise ValueError("MPS measurement campaign must be a full PASS or PARTIAL run")
    expected = {(digest, chi, f"session-{session}")
                for digest in expected_hashes for chi in CHI for session in (1, 2, 3)}
    identities = []
    statuses = []
    for row in attempts:
        try:
            identities.append((str(row["source_qasm_sha256"]), int(row["max_bond"]),
                               str(row["session_id"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("MPS attempt ledger contains a malformed identity") from exc
        status = row.get("status")
        if status not in ATTEMPT_STATUSES:
            raise ValueError(f"MPS attempt ledger contains an unknown status: {status}")
        if row.get("attempt_terminal") is not True:
            raise ValueError("MPS attempt ledger contains a nonterminal row")
        statuses.append(status)
    if len(attempts) != len(expected) or len(set(identities)) != len(identities) or set(identities) != expected:
        raise ValueError("MPS full attempt ledger has duplicate, missing, or extra identities")
    status_counts = {status: statuses.count(status) for status in sorted(set(statuses))}
    technical_errors = len(statuses) - status_counts.get("ok", 0)
    if summary.get("attempt_count") != len(attempts):
        raise ValueError("MPS run summary attempt count differs from its ledger")
    if summary.get("technical_error_count") != technical_errors:
        raise ValueError("MPS run summary technical-error count differs from its ledger")
    if summary["status"] == "PASS" and technical_errors != 0:
        raise ValueError("MPS run is labeled PASS despite terminal technical failures")
    if summary["status"] == "PARTIAL" and technical_errors == 0:
        raise ValueError("MPS run is labeled PARTIAL without a terminal technical failure")
    normalized_sha = hashlib.sha256(
        (json.dumps(attempts, sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()
    if summary.get("attempt_ledger_sha256") != normalized_sha:
        raise ValueError("MPS full attempt ledger digest differs from run summary")
    return {"measurement_campaign_status": summary["status"],
            "measurement_attempt_count": len(attempts),
            "measurement_target_hashes": len(expected_hashes),
            "measurement_technical_error_count": technical_errors,
            "measurement_attempt_status_counts": status_counts}


def metric_row(split_id: str, method_id: str, scope: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    runtime = []
    for row in rows:
        if row.get("target_status") != "runtime_observed":
            continue
        actual = finite_float(row.get("target_runtime_seconds"))
        predicted = finite_float(row.get("runtime_median_seed_seconds"))
        if actual is not None and predicted is not None and actual >= 0 and predicted >= 0:
            runtime.append((actual, predicted))
    errors = [abs(actual - predicted) for actual, predicted in runtime]
    log_errors = [abs(math.log1p(actual) - math.log1p(predicted)) for actual, predicted in runtime]
    actuals = [actual for actual, _ in runtime]
    predictions = [predicted for _, predicted in runtime]
    mean_actual = statistics.mean(actuals) if actuals else None
    ss_total = sum((value - mean_actual) ** 2 for value in actuals) if actuals else 0.0
    ss_error = sum((actual - predicted) ** 2 for actual, predicted in runtime)
    observed_quality = [row for row in rows if str(row.get("target_quality_pass", "")).strip().lower() in {"true", "false"}]
    quality_pairs = [(str(row["target_quality_pass"]).strip().lower() == "true",
                      finite_float(row.get("quality_mean_seed_probability"))) for row in observed_quality]
    quality_pairs = [(actual, predicted) for actual, predicted in quality_pairs
                     if predicted is not None and 0.0 <= predicted <= 1.0]
    correct = [actual == (predicted >= 0.5) for actual, predicted in quality_pairs]
    positives = [(actual, predicted) for actual, predicted in quality_pairs if actual]
    negatives = [(actual, predicted) for actual, predicted in quality_pairs if not actual]
    recalls = ([sum(predicted >= 0.5 for _actual, predicted in positives) / len(positives)] if positives else [])
    recalls += ([sum(predicted < 0.5 for _actual, predicted in negatives) / len(negatives)] if negatives else [])
    tp = sum(actual and predicted >= 0.5 for actual, predicted in quality_pairs)
    fp = sum((not actual) and predicted >= 0.5 for actual, predicted in quality_pairs)
    fn = sum(actual and predicted < 0.5 for actual, predicted in quality_pairs)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0 if precision is not None and recall is not None else None
    return {
        "split_id": split_id, "method_id": method_id, "scope": scope,
        "assigned_rows": len(rows), "runtime_evaluated_rows": len(runtime),
        "runtime_coverage_of_assigned": len(runtime) / len(rows) if rows else 0.0,
        "runtime_mae_seconds": statistics.mean(errors) if errors else None,
        "runtime_median_absolute_error_seconds": statistics.median(errors) if errors else None,
        "runtime_log1p_mae": statistics.mean(log_errors) if log_errors else None,
        "runtime_r2": 1.0 - ss_error / ss_total if runtime and ss_total > 0 else None,
        "runtime_p90_absolute_error_seconds": percentile(errors, 0.90),
        "runtime_p99_absolute_error_seconds": percentile(errors, 0.99),
        "runtime_max_absolute_error_seconds": max(errors) if errors else None,
        "quality_evaluated_rows": len(quality_pairs),
        "quality_coverage_of_assigned": len(quality_pairs) / len(rows) if rows else 0.0,
        "quality_brier_score": statistics.mean((predicted - float(actual)) ** 2 for actual, predicted in quality_pairs) if quality_pairs else None,
        "quality_accuracy_at_0_5": statistics.mean(correct) if correct else None,
        "quality_balanced_accuracy_at_0_5": statistics.mean(recalls) if recalls else None,
        "quality_precision_at_0_5": precision, "quality_recall_at_0_5": recall,
        "quality_f1_at_0_5": f1,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _numeric_minimum_chi(value: Any) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed in CHI else None


def select_hash_configurations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method_hash: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method_hash[(row["method_id"], row["source_qasm_sha256"])].append(row)
    decisions = []
    for (method, digest), group in sorted(by_method_hash.items()):
        by_chi = {int(row["max_bond"]): row for row in group}
        if set(by_chi) != set(CHI):
            raise ValueError(f"selection requires all six rung predictions for {method}/{digest}")
        minimum_values = {str(row.get("minimum_passing_chi", "")).strip() for row in group}
        if len(minimum_values) != 1:
            raise ValueError(f"inconsistent measured minimum-passing-chi label for {digest}")
        actual_minimum = _numeric_minimum_chi(next(iter(minimum_values)))
        passing = [chi for chi in CHI
                   if finite_float(by_chi[chi].get("quality_mean_seed_probability")) is not None
                   and float(by_chi[chi]["quality_mean_seed_probability"]) >= 0.5]
        selected = min(passing) if passing else None
        selected_row = by_chi[selected] if selected is not None else None
        oracle_row = by_chi[actual_minimum] if actual_minimum is not None else None
        decisions.append({
            "method_id": method, "source_qasm_sha256": digest,
            "selected_chi": selected if selected is not None else "",
            "selection_status": "selected" if selected is not None else "abstain_no_rung_probability_at_0_5",
            "selected_quality_probability": (selected_row["quality_mean_seed_probability"] if selected_row else ""),
            "selected_target_status": (selected_row["target_status"] if selected_row else ""),
            "selected_target_quality_pass": (selected_row["target_quality_pass"] if selected_row else ""),
            "selected_actual_runtime_seconds": (selected_row["target_runtime_seconds"] if selected_row else ""),
            "selected_predicted_runtime_seconds": (selected_row["runtime_median_seed_seconds"] if selected_row else ""),
            "actual_minimum_passing_chi": actual_minimum if actual_minimum is not None else "",
            "minimum_passing_chi_label": next(iter(minimum_values)),
            "oracle_target_status": (oracle_row["target_status"] if oracle_row else ""),
            "oracle_actual_runtime_seconds": (oracle_row["target_runtime_seconds"] if oracle_row else ""),
            "oracle_predicted_runtime_seconds": (oracle_row["runtime_median_seed_seconds"] if oracle_row else ""),
            "selection_is_exact_minimum": selected == actual_minimum if selected is not None and actual_minimum is not None else "",
            "selection_abs_log2_chi_error": (abs(math.log2(selected) - math.log2(actual_minimum))
                if selected is not None and actual_minimum is not None else ""),
        })
    return decisions


def selection_metric_rows(split_id: str, decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for method in METHODS:
        rows = [row for row in decisions if row["method_id"] == method]
        selected = [row for row in rows if row["selection_status"] == "selected"]
        selected_quality = [row for row in selected
                            if str(row["selected_target_quality_pass"]).strip().lower() in {"true", "false"}]
        selected_runtime = [row for row in selected if row["selected_target_status"] == "runtime_observed"
                            and finite_float(row["selected_actual_runtime_seconds"]) is not None]
        achievable = [row for row in rows if row["actual_minimum_passing_chi"] != ""]
        exact = [row for row in achievable if row["selection_is_exact_minimum"] is True]
        log_errors = [float(row["selection_abs_log2_chi_error"]) for row in achievable
                      if row["selection_abs_log2_chi_error"] not in ("", None)]
        oracle = [row for row in rows if row["actual_minimum_passing_chi"] != ""
                  and row["oracle_target_status"] == "runtime_observed"
                  and finite_float(row["oracle_actual_runtime_seconds"]) is not None
                  and finite_float(row["oracle_predicted_runtime_seconds"]) is not None]
        selected_errors = [abs(float(row["selected_predicted_runtime_seconds"])
                               - float(row["selected_actual_runtime_seconds"])) for row in selected_runtime]
        oracle_errors = [abs(float(row["oracle_predicted_runtime_seconds"])
                             - float(row["oracle_actual_runtime_seconds"])) for row in oracle]
        quality_passes = [str(row["selected_target_quality_pass"]).strip().lower() == "true"
                          for row in selected_quality]
        result.append({
            "split_id": split_id, "method_id": method, "scope": "predicted_selection_same_rung",
            "assigned_hashes": len(rows), "selected_hashes": len(selected),
            "selection_coverage": len(selected) / len(rows) if rows else 0.0,
            "selection_abstentions": len(rows) - len(selected),
            "selected_target_quality_observed_hashes": len(selected_quality),
            "selected_quality_observation_coverage": len(selected_quality) / len(selected) if selected else 0.0,
            "selected_quality_pass_rate": statistics.mean(quality_passes) if quality_passes else None,
            "selected_quality_violation_rate": 1.0 - statistics.mean(quality_passes) if quality_passes else None,
            "same_selected_rung_runtime_hashes": len(selected_runtime),
            "same_selected_rung_runtime_mae_seconds": statistics.mean(selected_errors) if selected_errors else None,
            "oracle_minimum_passing_hashes": len(oracle),
            "oracle_minimum_passing_runtime_mae_seconds": statistics.mean(oracle_errors) if oracle_errors else None,
            "quality_achievable_hashes": len(achievable),
            "minimum_passing_chi_exact_accuracy": len(exact) / len(achievable) if achievable else None,
            "minimum_passing_chi_mean_abs_log2_error": statistics.mean(log_errors) if log_errors else None,
        })
    return result


def paired_group_bootstrap(
    split_id: str, target: str, metric: str,
    grouped_losses: dict[str, list[tuple[float, float]]],
    replicates: int = 10000,
) -> dict[str, Any]:
    import numpy as np
    hashes = sorted(digest for digest, values in grouped_losses.items() if values)
    base = {"split_id": split_id, "target": target, "metric": metric,
            "reference_method_id": METHODS[1], "candidate_method_id": METHODS[0],
            "cluster_column": "source_qasm_sha256", "shared_hashes": len(hashes),
            "shared_rows": sum(len(grouped_losses[digest]) for digest in hashes),
            "observed_delta_candidate_minus_reference": None,
            "bootstrap_mean_delta": None, "ci_95_low": None, "ci_95_high": None,
            "replicates_requested": replicates, "replicates_completed": 0,
            "status": "unavailable_no_shared_hashes"}
    if not hashes:
        return base
    left = [loss for digest in hashes for loss, _ in grouped_losses[digest]]
    right = [loss for digest in hashes for _, loss in grouped_losses[digest]]
    observed = statistics.mean(left) - statistics.mean(right)
    registry = read_json(ROOT / "benchmark_v1/registry/seed_registry.json")
    stream = str(registry["streams"]["bootstrap"])
    label = f"family-joint-mps|{split_id}|{target}|{metric}|{METHODS[0]}_vs_{METHODS[1]}"
    seed = int.from_bytes(hashlib.sha256(
        f"{registry['root_seed']}|{stream}|{label}|0".encode()).digest()[:4], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(hashes), size=len(hashes))
        a_sum = b_sum = total = 0
        for group_index in sampled:
            for loss_a, loss_b in grouped_losses[hashes[int(group_index)]]:
                a_sum += loss_a
                b_sum += loss_b
                total += 1
        deltas[index] = a_sum / total - b_sum / total
    low, high = np.quantile(deltas, [0.025, 0.975], method="linear")
    base.update({"observed_delta_candidate_minus_reference": observed,
                 "bootstrap_mean_delta": float(deltas.mean()), "ci_95_low": float(low),
                 "ci_95_high": float(high), "replicates_completed": replicates,
                 "bootstrap_seed": seed, "bootstrap_stream": stream,
                 "bootstrap_rng": "numpy.PCG64", "quantile_method": "linear", "status": "evaluated"})
    return base


def paired_comparisons(split_id: str, rows: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell = {(row["method_id"], row["source_qasm_sha256"], int(row["max_bond"])): row for row in rows}
    by_decision = {(row["method_id"], row["source_qasm_sha256"]): row for row in decisions}
    comparisons = []
    for target, collect in (
        ("same_actual_rung", lambda digest: [
            (abs(float(by_cell[(METHODS[0], digest, chi)]["runtime_median_seed_seconds"])
                 - float(by_cell[(METHODS[0], digest, chi)]["target_runtime_seconds"])),
             abs(float(by_cell[(METHODS[1], digest, chi)]["runtime_median_seed_seconds"])
                 - float(by_cell[(METHODS[1], digest, chi)]["target_runtime_seconds"])))
            for chi in CHI if by_cell[(METHODS[0], digest, chi)]["target_status"] == "runtime_observed"
            and by_cell[(METHODS[1], digest, chi)]["target_status"] == "runtime_observed"]),
        ("predicted_selection_same_hash", lambda digest: [
            (abs(float(a["selected_predicted_runtime_seconds"]) - float(a["selected_actual_runtime_seconds"])),
             abs(float(b["selected_predicted_runtime_seconds"]) - float(b["selected_actual_runtime_seconds"])))
            for a, b in [(by_decision[(METHODS[0], digest)], by_decision[(METHODS[1], digest)])]
            if a["selection_status"] == b["selection_status"] == "selected"
            and a["selected_target_status"] == b["selected_target_status"] == "runtime_observed"]),
        ("oracle_minimum_passing_same_rung", lambda digest: [
            (abs(float(a["oracle_predicted_runtime_seconds"]) - float(a["oracle_actual_runtime_seconds"])),
             abs(float(b["oracle_predicted_runtime_seconds"]) - float(b["oracle_actual_runtime_seconds"])))
            for a, b in [(by_decision[(METHODS[0], digest)], by_decision[(METHODS[1], digest)])]
            if a["actual_minimum_passing_chi"] == b["actual_minimum_passing_chi"] != ""
            and a["oracle_target_status"] == b["oracle_target_status"] == "runtime_observed"]),
        ("rung_quality_brier", lambda digest: [
            ((float(by_cell[(METHODS[0], digest, chi)]["quality_mean_seed_probability"])
              - (str(by_cell[(METHODS[0], digest, chi)]["target_quality_pass"]).lower() == "true")) ** 2,
             (float(by_cell[(METHODS[1], digest, chi)]["quality_mean_seed_probability"])
              - (str(by_cell[(METHODS[1], digest, chi)]["target_quality_pass"]).lower() == "true")) ** 2)
            for chi in CHI if str(by_cell[(METHODS[0], digest, chi)]["target_quality_pass"]).lower() in {"true", "false"}
            and str(by_cell[(METHODS[1], digest, chi)]["target_quality_pass"]).lower() in {"true", "false"}]),
    ):
        groups = {digest: collect(digest) for digest in sorted({row["source_qasm_sha256"] for row in rows})}
        comparisons.append(paired_group_bootstrap(split_id, target, "mean_absolute_error_seconds" if "runtime" in target or "same_" in target else "brier_score", groups))
    return comparisons


def _read_fold(fold_dir: Path, split_id: str, fold: int, expected_all_hashes: set[str],
               expected_test_hashes: set[str],
               protocol_sha: str, plan_sha: str, measurement_sha: str,
               trainer_sha: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = fold_dir / "run_manifest.json"
    prediction_path = fold_dir / "oof_predictions.csv"
    transform_path = fold_dir / "fold_transform_and_support.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed_fold_technical_qa_pass":
        raise ValueError(f"fold {split_id}/{fold} is not a completed technical-QA pass")
    expected = {"split_id": split_id, "fold": fold, "protocol_sha256": protocol_sha,
                "plan_sha256": plan_sha, "measurement_attempts_sha256": measurement_sha,
                "runner_sha256": trainer_sha}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"fold {split_id}/{fold} {key} differs from frozen input")
    if manifest.get("quality_labels_used_as_inputs") is not False or manifest.get("test_runtime_or_fidelity_used_for_fit") is not False:
        raise ValueError(f"fold {split_id}/{fold} does not attest train/test target separation")
    if not transform_path.is_file() or sha(transform_path) != manifest.get("transform_sha256"):
        raise ValueError(f"fold {split_id}/{fold} transform/support artifact hash mismatch")
    transform = read_json(transform_path)
    train_hashes, test_hashes = set(transform.get("train_hashes", [])), set(transform.get("test_hashes", []))
    if train_hashes & test_hashes or train_hashes | test_hashes != expected_all_hashes:
        raise ValueError(f"fold {split_id}/{fold} train/test hash partition is invalid")
    if test_hashes != expected_test_hashes:
        raise ValueError(f"fold {split_id}/{fold} test hashes differ from its frozen fold assignment")
    if not prediction_path.is_file() or sha(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError(f"fold {split_id}/{fold} OOF prediction hash mismatch")
    rows = read_csv(prediction_path)
    expected_pairs = {(digest, chi, method) for digest in expected_test_hashes
                      for chi in CHI for method in METHODS}
    actual_pairs = {(row["source_qasm_sha256"], int(row["max_bond"]), row["method_id"]) for row in rows}
    if len(rows) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise ValueError(f"fold {split_id}/{fold} prediction identity coverage mismatch")
    if any(row.get("split_id") != split_id or int(row.get("fold", -1)) != fold for row in rows):
        raise ValueError(f"fold {split_id}/{fold} prediction split/fold labels mismatch")
    for row in rows:
        target_status = str(row.get("target_status", ""))
        if target_status != "runtime_observed" and target_status != "unavailable" and not target_status.startswith("unavailable_"):
            raise ValueError(f"unknown target status in {split_id}/{fold}: {row.get('target_status')}")
        if row.get("target_status") == "runtime_observed":
            if finite_float(row.get("target_runtime_seconds")) is None:
                raise ValueError(f"observed runtime missing/nonfinite in {split_id}/{fold}")
            if str(row.get("target_quality_pass", "")).strip().lower() not in {"true", "false"}:
                raise ValueError(f"observed quality label missing in {split_id}/{fold}")
        elif str(row.get("target_runtime_seconds", "")).strip() or str(row.get("target_quality_pass", "")).strip():
            raise ValueError(f"unavailable target must not carry runtime/quality labels in {split_id}/{fold}")
        for seed in SEEDS:
            if finite_float(row.get(f"runtime_seed_{seed}_seconds")) is None or finite_float(row.get(f"quality_seed_{seed}_probability")) is None:
                raise ValueError(f"nonfinite/missing seed prediction in {split_id}/{fold}")
    return rows, manifest


def aggregate(plan_dir: Path, measurement_dir: Path, training_root: Path) -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    protocol_sha = sha(PROTOCOL)
    plan_manifest = read_json(plan_dir / "preflight_manifest.json")
    if plan_manifest.get("protocol_sha256") != protocol_sha:
        raise ValueError("family plan protocol hash differs from current frozen protocol")
    for name, expected in plan_manifest["files"].items():
        if sha(plan_dir / name) != expected:
            raise ValueError(f"family plan materialization changed: {name}")
    plan_sha = sha(plan_dir / "preflight_manifest.json")
    measurement_summary_path = measurement_dir / "run_summary.json"
    attempts_path = measurement_dir / "attempt_records.jsonl"
    measurement_summary = read_json(measurement_summary_path)
    measurement_sha = sha(attempts_path)
    attempts = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    feature_rows = read_csv(plan_dir / "hash_feature_manifest.csv")
    expected_all = {row["source_qasm_sha256"] for row in feature_rows}
    if len(expected_all) != 150:
        raise ValueError("expected exactly 150 unique exact-QASM hashes")
    measurement_validation = validate_measurement_ledger(measurement_summary, attempts, expected_all)
    method_metrics, selection_metrics, family_metrics = [], [], []
    paired_metrics, output_rows, selection_outputs = [], {}, {}
    fold_manifests = {}
    trainer_sha = sha(ROOT / "benchmark_v1/scripts/train_family_aware_joint_mps_v1.py")
    for split_id, fold_column, output_name in (
        ("c44", "c44_fold", "c44_oof_predictions.csv"),
        ("family_holdout", "family_holdout_fold_diagnostic_only", "family_holdout_diagnostic_oof_predictions.csv"),
    ):
        collected: list[dict[str, Any]] = []
        for fold in range(5):
            expected_fold_hashes = {row["source_qasm_sha256"] for row in feature_rows if int(row[fold_column]) == fold}
            fold_dir = training_root / split_id / f"fold_{fold}"
            rows, manifest = _read_fold(fold_dir, split_id, fold, expected_all, expected_fold_hashes,
                                        protocol_sha, plan_sha, measurement_sha, trainer_sha)
            collected.extend(rows)
            fold_manifests[f"{split_id}_fold_{fold}"] = {
                "run_manifest_sha256": sha(fold_dir / "run_manifest.json"),
                "prediction_sha256": sha(fold_dir / "oof_predictions.csv"),
                "transform_sha256": sha(fold_dir / "fold_transform_and_support.json"),
                "test_hashes": manifest["test_hashes"], "training_hashes": manifest["training_hashes"],
            }
        expected_rows = 150 * len(CHI) * len(METHODS)
        if len(collected) != expected_rows:
            raise ValueError(f"{split_id} OOF assigned denominator is {len(collected)}, expected {expected_rows}")
        collected.sort(key=lambda row: (row["source_qasm_sha256"], int(row["max_bond"]), row["method_id"]))
        output_rows[output_name] = collected
        decisions = select_hash_configurations(collected)
        if len(decisions) != 150 * len(METHODS):
            raise ValueError(f"{split_id} selection decision denominator changed")
        selection_name = ("c44_selection_decisions.csv" if split_id == "c44"
                          else "family_holdout_selection_diagnostic.csv")
        selection_outputs[selection_name] = decisions
        selection_metrics.extend(selection_metric_rows(split_id, decisions))
        paired_metrics.extend(paired_comparisons(split_id, collected, decisions))
        for method in METHODS:
            method_rows = [row for row in collected if row["method_id"] == method]
            method_metrics.append(metric_row(split_id, method, "all_rungs", method_rows))
            for chi in CHI:
                method_metrics.append(metric_row(split_id, method, f"chi_{chi}",
                    [row for row in method_rows if int(row["max_bond"]) == chi]))
        family_method_rows = [row for row in collected if row["method_id"] == METHODS[0]]
        by_hash: dict[str, dict[str, str]] = {}
        for row in family_method_rows:
            by_hash.setdefault(row["source_qasm_sha256"], row)
        labeled = [row for row in by_hash.values() if row.get("family_component_diagnostic_only")]
        for seed in SEEDS:
            correct = [row[f"predicted_family_seed_{seed}"] == row["family_component_diagnostic_only"] for row in labeled]
            family_metrics.append({"split_id": split_id, "diagnostic": "family_component_classifier_accuracy",
                                   "seed": seed, "hashes": len(labeled),
                                   "accuracy": statistics.mean(correct) if correct else None,
                                   "family_component_is_prediction_input": False})

    return {"protocol_sha256": protocol_sha, "protocol_id": protocol["protocol_id"],
            "plan_sha256": plan_sha, "measurement_attempts_sha256": measurement_sha,
            "measurement_summary_sha256": sha(measurement_summary_path),
            **measurement_validation,
            "trainer_sha256": trainer_sha, "fold_manifests": fold_manifests,
            "outputs": output_rows, "selection_outputs": selection_outputs,
            "method_metrics": method_metrics, "selection_metrics": selection_metrics,
            "paired_comparisons": paired_metrics,
            "family_prediction_diagnostic": family_metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=PLAN)
    parser.add_argument("--measure-dir", type=Path, default=MEASUREMENTS)
    parser.add_argument("--training-root", type=Path, default=TRAINING)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite aggregate output: {output}")
    result = aggregate(args.plan_dir.resolve(), args.measure_dir.resolve(), args.training_root.resolve())
    output.mkdir(parents=True)
    for name, rows in result.pop("outputs").items():
        write_csv(output / name, rows)
    for name, rows in result.pop("selection_outputs").items():
        write_csv(output / name, rows)
    write_csv(output / "method_metrics.csv", result.pop("method_metrics"))
    write_csv(output / "selection_metrics.csv", result.pop("selection_metrics"))
    write_csv(output / "paired_comparisons.csv", result.pop("paired_comparisons"))
    write_csv(output / "family_prediction_diagnostic.csv", result.pop("family_prediction_diagnostic"))
    manifest = {"artifact_id": "family-aware-joint-mps-oof-aggregate-v1", "status": "PASS",
                **result, "files": {path.name: sha(path) for path in sorted(output.glob("*.csv"))}}
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "measurement_campaign_status": manifest["measurement_campaign_status"],
                      "measurement_technical_error_count": manifest["measurement_technical_error_count"],
                      "output_dir": str(output),
                      "c44_rows": 150 * len(CHI) * len(METHODS),
                      "family_holdout_diagnostic_rows": 150 * len(CHI) * len(METHODS)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
