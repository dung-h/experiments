#!/usr/bin/env python3
"""Aggregate frozen exact-hash OOF outputs without crossing simulator clocks.

This creates an additive E6 artifact only. Aer/Azizov and fixed-MPS/chi16
results are summarized in separate tables and paired comparisons. It never
re-fits models, imputes labels/predictions, or changes source artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - deployment diagnostic
    raise RuntimeError("E6 paired bootstrap requires NumPy") from exc


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
ARTIFACT_ID = "predictive-runtime-aggregate-v1"
EXPECTED_AER_METHOD_OUTPUT_CLOCK = "predicted_noisy_aer_warm_execution_seconds"
EXPECTED_MPS_METHOD_OUTPUT_CLOCK = "predicted_cudaq_mps_fp64_bond16_warm_state_execution_seconds"
DEFAULT_AER_DIR = ROOT / "artifacts/benchmark_v3/simulator/azizov_common_core_adaptation"
DEFAULT_MPS_E3_DIR = ROOT / "artifacts/benchmark_v3/simulator/mps_fixed_chi16_runtime_oof"
DEFAULT_MPS_E5_DIR = ROOT / "artifacts/benchmark_v3/simulator/family_residual_fixed_mps_runtime"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1"
PANEL_ID = "sim_common_q16_v1"
EXPECTED_FOLDS = {0: 37, 1: 28, 2: 26, 3: 25, 4: 34}
E4_NEURAL_METHOD_IDS = {
    f"azizov_gnn_{view}_seed_{seed}"
    for view in ("source", "hybrid", "transpiled")
    for seed in (42, 1234, 31415)
}
E4_CLASSICAL_METHOD_IDS = {
    f"classical_{model}_{view}"
    for model in ("linear_regression", "ridge", "svr_rbf", "random_forest", "xgboost")
    for view in ("source", "hybrid", "transpiled")
}
E4_BASE_METHOD_IDS = E4_NEURAL_METHOD_IDS | E4_CLASSICAL_METHOD_IDS
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_QUANTILE_RULE = "numpy.quantile(method='linear', q=[0.025,0.975])"
EXPECTED_E5_SHA256 = {
    "five_fold_oof_predictions.csv": "06effa52a1b5737f45ee4dbbe0d17112b201d753670b960a9815bdaee74998d8",
    "method_metrics.json": "96ee542f3222df336641af4db2ad3b7a75ecfb8da7956dca96a7ac5191350038",
    "paired_bootstrap.csv": "e344874849376f2ad947922584cf192524ca0cce1ddeb5efd2e76fd7d0646204",
    "paired_hash_comparisons.csv": "9a4f93d1ce1b98adebf13d5201f93413c6cfb5174887dd5c2179d55ccc9d2972",
}

E4_REQUIRED = (
    "method_metrics.csv",
    "coverage_and_terminal_statuses.csv",
    "same_hash_shared_row_pairs.csv",
    "labels_core.csv",
    "fold_assignments_and_hash_audit.csv",
    "representation_manifest.json",
    "source_hashes.json",
    "run_manifest.json",
    "environment_and_cuda_lock.json",
    "fold_0_qa.json",
)
E3_REQUIRED = (
    "five_fold_oof_predictions.csv",
    "five_fold_oof_summary.json",
    "preflight_manifest.json",
    "preflight_qa.json",
    "five_fold_execution_receipt.json",
    "hash_targets_and_static_features.csv",
)
E5_REQUIRED = (
    "five_fold_oof_predictions.csv",
    "method_metrics.json",
    "paired_bootstrap.csv",
    "paired_hash_comparisons.csv",
    "five_fold_oof_summary.json",
    "materialization_manifest.json",
    "execution_environment.json",
    "five_fold_execution_receipt.json",
    "hash_targets_and_static_features.csv",
)

METRIC_NAMES = (
    "mae_seconds",
    "medae_seconds",
    "rmse_seconds",
    "log1p_mae_seconds",
    "r2_seconds",
    "p90_absolute_error_seconds",
    "p99_absolute_error_seconds",
    "max_absolute_error_seconds",
)

COVERAGE_FIELDS = [
    "domain_id", "artifact_id", "evaluation_target_clock", "method_output_clock", "source_artifact", "method_id",
    "source_method_column", "source_sha256", "fold", "panel_member_count", "member_ids_json",
    "target_status", "quality_status", "target_seconds", "prediction_seconds",
    "method_terminal_status", "score_status", "terminal_reason",
]
METRICS_FIELDS = [
    "domain_id", "artifact_id", "evaluation_target_clock", "method_output_clock", "source_artifact", "source_method_column",
    "method_id", "population", "assigned_hashes", "finite_target_hashes", "quality_pass_hashes",
    "quality_failed_finite_hashes", "target_unavailable_hashes", "prediction_hashes_assigned",
    "scored_hashes", "coverage_of_assigned", "coverage_of_finite_targets",
    *METRIC_NAMES, "status",
]
BOOTSTRAP_FIELDS = [
    "domain_id", "artifact_id", "evaluation_target_clock", "method_output_clock", "population", "comparison_label",
    "candidate_method_id", "reference_method_id", "assigned_hashes", "candidate_predicted_hashes",
    "reference_predicted_hashes", "shared_hashes", "shared_hash_set_sha256",
    "observed_mae_difference_seconds", "bootstrap_mean_mae_difference_seconds",
    "bootstrap_ci_low_seconds", "bootstrap_ci_high_seconds", "bootstrap_replicates_requested",
    "bootstrap_replicates_completed", "bootstrap_seed", "bootstrap_stream", "bootstrap_group_column",
    "percentile_rule", "coverage_asymmetric", "status",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _number(value: Any, *, field: str, allow_blank: bool = False) -> float | None:
    if value is None or value == "":
        if allow_blank:
            return None
        raise ValueError(f"missing numeric value for {field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value for {field}: {value!r}")
    return number


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _digest_sorted_hashes(hashes: Iterable[str]) -> str:
    payload = "\n".join(sorted(hashes)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_hash(value: str, *, source: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"invalid exact-QASM SHA-256 in {source}: {value!r}")


def _require_files(directory: Path, names: Sequence[str], *, role: str) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{role} artifact is incomplete at {directory}: missing {', '.join(missing)}")


def _float_equal(left: Any, right: Any, *, field: str) -> bool:
    left_number = _number(left, field=field, allow_blank=True)
    right_number = _number(right, field=field, allow_blank=True)
    if left_number is None or right_number is None:
        return left_number is right_number
    return math.isclose(left_number, right_number, rel_tol=1e-10, abs_tol=1e-12)


def _target_summary(labels: Sequence[Mapping[str, str]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    by_hash: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    by_member: dict[str, dict[str, str]] = {}
    for row in labels:
        digest = str(row.get("source_sha256", ""))
        member_id = str(row.get("panel_member_id", ""))
        _assert_hash(digest, source="Aer labels_core.csv")
        if not member_id or member_id in by_member:
            raise ValueError(f"empty or duplicate Aer panel member id: {member_id!r}")
        fold = int(row["fold"])
        observed = _number(row.get("observed_seconds"), field="observed_seconds")
        if observed is None or observed < 0:
            raise ValueError(f"invalid Aer target for {member_id}")
        by_hash[digest].append(row)
        by_member[member_id] = dict(row)

    summary: dict[str, dict[str, Any]] = {}
    for digest, members in by_hash.items():
        folds = {int(row["fold"]) for row in members}
        clocks = {str(row.get("target_clock", "")) for row in members}
        if len(folds) != 1 or len(clocks) != 1 or "" in clocks:
            raise ValueError(f"Aer aliases disagree on fold/clock for {digest}")
        target = _median([float(row["observed_seconds"]) for row in members])
        member_ids = sorted(str(row["panel_member_id"]) for row in members)
        summary[digest] = {
            "fold": folds.pop(),
            "target_seconds": target,
            "target_status": "runtime_observed",
            "quality_status": "not_applicable",
            "target_clock": clocks.pop(),
            "member_ids": member_ids,
        }
    return summary, by_member


def collapse_aer_member_ledger(
    labels: Sequence[Mapping[str, str]],
    ledger: Sequence[Mapping[str, str]],
    *,
    validate_frozen_panel: bool = True,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]], str]:
    """Join E4's member ledger to hash labels and verify aliases before dedup."""
    hash_info, member_info = _target_summary(labels)
    methods = sorted({str(row.get("method_id", "")) for row in ledger})
    if not methods or "" in methods:
        raise ValueError("E4 terminal ledger contains no method IDs")
    expected_keys = {(method, member) for method in methods for member in member_info}
    actual_keys: set[tuple[str, str]] = set()
    by_method_hash: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in ledger:
        method_id = str(row.get("method_id", ""))
        member_id = str(row.get("panel_member_id", ""))
        key = (method_id, member_id)
        if key in actual_keys:
            raise ValueError(f"duplicate E4 method/member terminal record: {key}")
        actual_keys.add(key)
        if member_id not in member_info:
            raise ValueError(f"E4 terminal ledger has unknown panel member {member_id!r}")
        label = member_info[member_id]
        digest = str(row.get("source_sha256", ""))
        if digest != label["source_sha256"] or int(row["fold"]) != int(label["fold"]):
            raise ValueError(f"E4 hash/fold disagrees with E2 member labels for {member_id}")
        _assert_hash(digest, source="E4 coverage ledger")
        info = hash_info[digest]
        if not _float_equal(row.get("target_seconds"), info["target_seconds"], field="E4 target_seconds"):
            raise ValueError(f"E4 hash target disagrees with alias-median Aer label for {digest}")
        by_method_hash[(method_id, digest)].append(row)
    if actual_keys != expected_keys:
        missing = len(expected_keys - actual_keys)
        extra = len(actual_keys - expected_keys)
        raise ValueError(f"E4 method/member terminal envelope mismatch (missing={missing}, extra={extra})")

    method_records: dict[str, dict[str, dict[str, Any]]] = {method: {} for method in methods}
    coverage_rows: list[dict[str, Any]] = []
    for (method_id, digest), aliases in sorted(by_method_hash.items()):
        info = hash_info[digest]
        statuses = {str(row.get("status", "")) for row in aliases}
        reasons = {str(row.get("terminal_reason", "")) for row in aliases}
        folds = {int(row["fold"]) for row in aliases}
        targets = {str(row.get("target_seconds", "")) for row in aliases}
        predictions = {str(row.get("prediction_seconds", "")) for row in aliases}
        if any(len(values) != 1 for values in (statuses, folds, targets, predictions)):
            raise ValueError(f"E4 aliases disagree on terminal outcome/prediction for {method_id}:{digest}")
        terminal_status = statuses.pop()
        prediction_text = predictions.pop()
        prediction = _number(prediction_text, field="E4 prediction_seconds", allow_blank=True)
        if terminal_status == "predicted":
            if prediction is None or prediction < 0:
                raise ValueError(f"E4 status predicted without finite nonnegative prediction for {method_id}:{digest}")
        elif prediction is not None:
            raise ValueError(f"E4 non-predicted status carries a prediction for {method_id}:{digest}")
        reason = ";".join(sorted(reason for reason in reasons if reason))
        record = {
            "source_sha256": digest,
            "fold": info["fold"],
            "source_artifact": "azizov_common_core_adaptation",
            "source_method_column": method_id,
            "target_seconds": info["target_seconds"],
            "target_status": info["target_status"],
            "quality_status": info["quality_status"],
            "prediction_seconds": prediction,
            "method_terminal_status": terminal_status,
            "terminal_reason": reason,
            "panel_member_count": len(info["member_ids"]),
            "member_ids": info["member_ids"],
        }
        method_records[method_id][digest] = record
        coverage_rows.append(
            {
                "domain_id": "aer_azizov_core_q9",
                "artifact_id": ARTIFACT_ID,
                "evaluation_target_clock": info["target_clock"],
                "method_output_clock": EXPECTED_AER_METHOD_OUTPUT_CLOCK,
                "source_artifact": "azizov_common_core_adaptation",
                "method_id": method_id,
                "source_method_column": method_id,
                "source_sha256": digest,
                "fold": info["fold"],
                "panel_member_count": len(info["member_ids"]),
                "member_ids_json": json.dumps(info["member_ids"], separators=(",", ":")),
                "target_status": info["target_status"],
                "quality_status": info["quality_status"],
                "target_seconds": info["target_seconds"],
                "prediction_seconds": "" if prediction is None else prediction,
                "method_terminal_status": terminal_status,
                "score_status": "predicted" if prediction is not None else terminal_status,
                "terminal_reason": reason,
            }
        )
    if set(hash_info) != set(next(iter(method_records.values()))):
        raise ValueError("E4 collapsed coverage does not include the complete exact-hash panel")
    folds = Counter(int(row["fold"]) for row in hash_info.values())
    if validate_frozen_panel and len(hash_info) != 150:
        raise ValueError(f"Aer exact-hash panel has {len(hash_info)} hashes, expected 150")
    if validate_frozen_panel and dict(sorted(folds.items())) != EXPECTED_FOLDS:
        raise ValueError(f"Aer C44 fold counts differ from frozen envelope: {dict(folds)}")
    return method_records, coverage_rows, str(next(iter(hash_info.values()))["target_clock"])


def _method_metrics_from_e4(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    by_method: dict[str, dict[str, str]] = {}
    for row in rows:
        method = str(row.get("method_id", ""))
        if not method or method in by_method:
            raise ValueError(f"empty or duplicate E4 method metric row {method!r}")
        by_method[method] = row
    if not by_method:
        raise ValueError("E4 method_metrics.csv contains no method rows")
    return by_method


def _validate_aer_fold_audit(path: Path, labels: Sequence[Mapping[str, str]]) -> None:
    rows = read_csv(path)
    if len(rows) != 162:
        raise ValueError(f"E2 fold/hash audit has {len(rows)} member rows, expected 162")
    label_by_member = {str(row["panel_member_id"]): row for row in labels}
    seen: set[str] = set()
    by_hash: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        member = str(row.get("panel_member_id", ""))
        if not member or member in seen or member not in label_by_member:
            raise ValueError(f"E2 fold/hash audit has an empty, duplicate or unknown member: {member!r}")
        seen.add(member)
        label = label_by_member[member]
        digest = str(row.get("source_sha256", ""))
        fold = int(row.get("c44_fold", row.get("fold", "-1")))
        if digest != label["source_sha256"] or fold != int(label["fold"]):
            raise ValueError(f"E2 fold/hash audit does not match label ledger for {member}")
        computed_fold = row.get("computed_fold", "")
        if computed_fold and int(computed_fold) != fold:
            raise ValueError(f"computed SHA-256 fold differs from frozen C44 fold for {member}")
        if row.get("fold_match", "true").lower() not in {"true", "1"}:
            raise ValueError(f"E2 fold/hash audit marks a fold mismatch for {member}")
        by_hash[digest].add(fold)
    if seen != set(label_by_member) or len(by_hash) != 150:
        raise ValueError("E2 fold/hash audit does not cover the exact 162-member/150-hash panel")
    if any(len(folds) != 1 for folds in by_hash.values()):
        raise ValueError("aliases of an exact QASM hash have different folds")
    counts = Counter(next(iter(folds)) for folds in by_hash.values())
    if dict(sorted(counts.items())) != EXPECTED_FOLDS:
        raise ValueError(f"E2 fold/hash audit fold counts differ from frozen C44: {dict(counts)}")


def _validate_e4_metrics(
    methods: Mapping[str, Mapping[str, Mapping[str, Any]]],
    supplied: Mapping[str, Mapping[str, str]],
) -> None:
    if set(methods) != set(supplied):
        raise ValueError("E4 method_metrics.csv and coverage ledger have different method IDs")
    for method_id, records in methods.items():
        calculated = summarize_method(
            domain_id="aer_azizov_core_q9",
            source_artifact="azizov_common_core_adaptation",
            source_method_column=method_id,
            method_id=method_id,
            records=records,
            target_clock="warm_execution",
            method_output_clock=EXPECTED_AER_METHOD_OUTPUT_CLOCK,
            population="all_finite_targets",
            quality_available=False,
        )
        supplied_row = supplied[method_id]
        if int(supplied_row.get("assigned_hashes", "-1")) != 150:
            raise ValueError(f"E4 method metric assigned denominator is not 150 for {method_id}")
        if int(supplied_row.get("predicted_hashes", "-1")) != int(calculated["scored_hashes"]):
            raise ValueError(f"E4 hash prediction count disagrees with collapsed ledger for {method_id}")
        for field in METRIC_NAMES:
            source_field = "log1p_mae" if field == "log1p_mae_seconds" else field
            if not _float_equal(supplied_row.get(source_field, ""), calculated[field], field=f"E4 {field}"):
                raise ValueError(f"E4 metric {field} does not reproduce from hash ledger for {method_id}")


def _validate_e4_pairs(
    path: Path,
    methods: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    rows = read_csv(path)
    expected: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    method_ids = sorted(methods)
    for index, method_a in enumerate(method_ids):
        for method_b in method_ids[index + 1 :]:
            common = sorted(set(methods[method_a]) & set(methods[method_b]))
            for digest in common:
                left, right = methods[method_a][digest], methods[method_b][digest]
                if left["prediction_seconds"] is None or right["prediction_seconds"] is None:
                    continue
                target = float(left["target_seconds"])
                expected[(method_a, method_b, digest)] = (
                    target,
                    float(left["prediction_seconds"]),
                    float(right["prediction_seconds"]),
                )
    observed: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for row in rows:
        method_a = str(row.get("method_a", ""))
        method_b = str(row.get("method_b", ""))
        digest = str(row.get("source_sha256", ""))
        key = (method_a, method_b, digest)
        if key in observed:
            raise ValueError(f"duplicate E4 pair/hash record: {key}")
        observed[key] = row
    if set(observed) != set(expected):
        raise ValueError(f"E4 shared-row pair envelope mismatch (expected={len(expected)}, got={len(observed)})")
    for key, values in expected.items():
        row = observed[key]
        if row.get("comparison_id") != f"{key[0]}__vs__{key[1]}":
            raise ValueError(f"E4 pair comparison ID mismatch for {key}")
        target, left, right = values
        for field, expected_value in (
            ("target_seconds", target),
            ("prediction_a_seconds", left),
            ("prediction_b_seconds", right),
            ("absolute_error_a_seconds", abs(left - target)),
            ("absolute_error_b_seconds", abs(right - target)),
            ("absolute_error_difference_a_minus_b_seconds", abs(left - target) - abs(right - target)),
        ):
            if not _float_equal(row.get(field), expected_value, field=f"E4 pair {field}"):
                raise ValueError(f"E4 shared-row pair value mismatch for {key}:{field}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def validate_e4_completion(run_manifest: Mapping[str, Any]) -> None:
    status = run_manifest.get("status")
    if status != "complete_five_fold_oof":
        raise ValueError(f"E6 requires complete five-fold E4 OOF; got status={status!r}")


def validate_e4_fold_audits(
    run_manifest: Mapping[str, Any],
    expected_member_counts: Mapping[int, int],
) -> None:
    folds = run_manifest.get("folds")
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"E4 run manifest does not enumerate all frozen folds: {folds!r}")
    audits = run_manifest.get("fold_audits")
    expected_keys = {str(fold) for fold in EXPECTED_FOLDS}
    if not isinstance(audits, Mapping) or set(audits) != expected_keys:
        raise ValueError("E4 run manifest fold_audits must contain exactly folds 0 through 4")
    if int(run_manifest.get("method_family_count", -1)) != 24:
        raise ValueError("E4 run manifest does not declare all 24 method families")
    if set(expected_member_counts) != set(EXPECTED_FOLDS) or sum(expected_member_counts.values()) != 162:
        raise ValueError("E2 labels do not supply the frozen per-fold 162-member envelope")
    for fold, expected_hashes in EXPECTED_FOLDS.items():
        audit = audits[str(fold)]
        if int(audit.get("fold", -1)) != fold:
            raise ValueError(f"E4 fold audit key/value mismatch for fold {fold}")
        if int(audit.get("assigned_hashes", -1)) != expected_hashes:
            raise ValueError(f"E4 fold {fold} assigned_hashes does not match frozen C44 count {expected_hashes}")
        if int(audit.get("eligible_hashes", -1)) != expected_hashes or int(audit.get("unavailable_representation_hashes", -1)) != 0:
            raise ValueError(f"E4 fold {fold} does not show all assigned hashes representation-eligible")
        expected_members = int(expected_member_counts[fold])
        if int(audit.get("assigned_members", -1)) != expected_members:
            raise ValueError(f"E4 fold {fold} assigned member count differs from E2 labels ({expected_members})")
        if int(audit.get("method_families", -1)) != 24:
            raise ValueError(f"E4 fold {fold} does not audit all 24 method families")
        method_audit = audit.get("method_audit")
        if not isinstance(method_audit, Mapping) or set(method_audit) != E4_BASE_METHOD_IDS:
            raise ValueError(f"E4 fold {fold} method_audit keys do not cover all 24 frozen method families")
        if int(audit.get("method_member_terminal_rows", -1)) != 24 * int(audit.get("assigned_members", -1)):
            raise ValueError(f"E4 fold {fold} method/member terminal count is inconsistent")
        for method_id, record in method_audit.items():
            status = record.get("status") if isinstance(record, Mapping) else None
            if status not in {"PASS", "FAIL", "BLOCKED"}:
                raise ValueError(f"E4 fold {fold} has an unknown terminal method-audit state: {method_id}={status!r}")
    fold0 = audits["0"]
    fold0_qa = run_manifest.get("fold0_qa")
    if not isinstance(fold0_qa, Mapping):
        raise ValueError("E4 run manifest is missing the fold0_qa record")
    if fold0_qa != fold0:
        raise ValueError("E4 fold0_qa and fold_audits['0'] differ")
    if fold0.get("technical_status") != "PASS":
        raise ValueError("E4 fold 0 technical QA did not pass")
    if fold0.get("continuation_decision") != "automatically_continue_folds_1_through_4":
        raise ValueError("E4 fold 0 did not authorize the frozen automatic continuation")
    if fold0.get("continuation_reasons") != []:
        raise ValueError("E4 fold 0 records continuation blockers")
    method_audit = fold0.get("method_audit")
    if not isinstance(method_audit, Mapping) or len(method_audit) != 24:
        raise ValueError("E4 fold 0 does not contain all 24 method-family audit rows")
    neural_audits = [value for method, value in method_audit.items() if str(method).startswith("azizov_gnn_")]
    if len(neural_audits) != 9 or any(value.get("status") != "PASS" for value in neural_audits):
        raise ValueError("E4 fold 0 does not show PASS for all nine CUDA-inference GNN families")
    if any(value.get("status") != "PASS" for value in method_audit.values()):
        raise ValueError("E4 fold 0 contains a non-passing method-family technical audit")


def validate_e4_terminal_ledger(
    audits: Mapping[str, Any],
    ledger: Sequence[Mapping[str, str]],
    expected_member_counts: Mapping[int, int],
) -> None:
    status_to_terminal = {
        "PASS": {"predicted"},
        "FAIL": {"technical_failure"},
        "BLOCKED": {"blocked_compute_lock_or_cuda"},
    }
    for fold in range(5):
        fold_audit = audits[str(fold)]
        expected_members = int(expected_member_counts[fold])
        for method_id, method_audit in fold_audit["method_audit"].items():
            method_rows = [
                row for row in ledger
                if int(row["fold"]) == fold and str(row["method_id"]) == method_id
            ]
            if len(method_rows) != expected_members:
                raise ValueError(f"E4 terminal ledger row count disagrees with fold {fold} audit for {method_id}")
            expected_statuses = status_to_terminal[str(method_audit["status"])]
            actual_statuses = {str(row.get("status", "")) for row in method_rows}
            if actual_statuses != expected_statuses:
                raise ValueError(
                    f"E4 fold {fold} audit/terminal-ledger status mismatch for {method_id}: "
                    f"audit={method_audit['status']!r}, terminal={sorted(actual_statuses)!r}"
                )


def load_aer_panel(directory: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]], str, str, dict[str, dict[str, str]]]:
    _require_files(directory, E4_REQUIRED, role="E4 Aer")
    run_manifest = _read_json(directory / "run_manifest.json")
    validate_e4_completion(run_manifest)
    labels = read_csv(directory / "labels_core.csv")
    if len(labels) != 162:
        raise ValueError(f"E4/E2 Aer label member count is {len(labels)}, expected 162")
    expected_member_counts = Counter(int(row["fold"]) for row in labels)
    validate_e4_fold_audits(run_manifest, expected_member_counts)
    _validate_aer_fold_audit(directory / "fold_assignments_and_hash_audit.csv", labels)
    representation_manifest = _read_json(directory / "representation_manifest.json")
    if representation_manifest.get("status") != "complete_static_materialization":
        raise ValueError("E4 source representation manifest is not a complete static materialization")
    if int(representation_manifest.get("available_hash_count", -1)) != 150 or int(representation_manifest.get("unavailable_hash_count", -1)) != 0:
        raise ValueError("E4 source representation manifest does not cover all 150 hashes")
    if sha256_file(directory / "source_hashes.json") != str(run_manifest.get("source_hashes_sha256", "")):
        raise ValueError("E4 run manifest does not pin the E2 source_hashes.json it consumed")
    if sha256_file(directory / "representation_manifest.json") != str(run_manifest.get("representation_manifest_sha256", "")):
        raise ValueError("E4 run manifest does not pin the E2 representation manifest it consumed")
    ledger = read_csv(directory / "coverage_and_terminal_statuses.csv")
    validate_e4_terminal_ledger(run_manifest["fold_audits"], ledger, expected_member_counts)
    methods, coverage, target_clock = collapse_aer_member_ledger(labels, ledger)
    supplied_metrics = _method_metrics_from_e4(directory / "method_metrics.csv")
    _validate_e4_metrics(methods, supplied_metrics)
    _validate_e4_pairs(directory / "same_hash_shared_row_pairs.csv", methods)
    if target_clock != "warm_execution":
        raise ValueError(f"E4 target clock is not frozen Aer warm_execution: {target_clock!r}")
    protocol = _read_json(ROOT / "benchmark_v1/protocol/azizov_common_core_gnn_v1.json")
    method_output_clock = str(protocol.get("local_cell", {}).get("method_output_clock", ""))
    if method_output_clock != EXPECTED_AER_METHOD_OUTPUT_CLOCK:
        raise ValueError(f"Azizov protocol method output clock differs from frozen mapping: {method_output_clock!r}")
    return methods, coverage, target_clock, method_output_clock, supplied_metrics


def _target_table(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(path)
    target_by_hash: dict[str, dict[str, Any]] = {}
    for row in rows:
        digest = str(row.get("source_qasm_sha256", ""))
        _assert_hash(digest, source=str(path))
        if digest in target_by_hash:
            raise ValueError(f"duplicate target hash in {path}: {digest}")
        target = _number(row.get("target_seconds"), field=f"{path.name}:target_seconds", allow_blank=True)
        target_status = str(row.get("target_status", ""))
        quality = str(row.get("quality_status_separate", row.get("quality_status_separate_audit_only", "")))
        fold = int(row["fold"])
        if target_status == "runtime_observed" and target is None:
            raise ValueError(f"runtime_observed hash lacks target in {path}: {digest}")
        if target_status != "runtime_observed" and target is not None:
            raise ValueError(f"unavailable-target hash has numeric target in {path}: {digest}")
        if target_status == "runtime_observed" and quality not in {"quality_pass", "quality_failed"}:
            raise ValueError(f"finite MPS target has invalid separate quality state in {path}: {digest}:{quality}")
        if target_status != "runtime_observed" and quality != "quality_unavailable":
            raise ValueError(f"unavailable MPS target has non-unavailable quality state in {path}: {digest}:{quality}")
        target_by_hash[digest] = {
            "fold": fold,
            "target_seconds": target,
            "target_status": target_status,
            "quality_status": quality,
            "panel_member_count": int(row.get("panel_member_count", "1")),
        }
    return target_by_hash


def _mps_oof_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    by_hash: dict[str, dict[str, str]] = {}
    for row in rows:
        digest = str(row.get("source_qasm_sha256", ""))
        _assert_hash(digest, source=str(path))
        if digest in by_hash:
            raise ValueError(f"duplicate exact hash in MPS OOF table {path}: {digest}")
        by_hash[digest] = row
    return by_hash


def _make_mps_method_records(
    *,
    domain_id: str,
    source_artifact: str,
    target_clock: str,
    method_output_clock: str,
    method_columns: Mapping[str, str],
    oof_by_hash: Mapping[str, Mapping[str, str]],
    target_by_hash: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    method_records: dict[str, dict[str, Any]] = {method: {} for method in method_columns}
    coverage_rows: list[dict[str, Any]] = []
    for digest, target in sorted(target_by_hash.items()):
        source = oof_by_hash[digest]
        if int(source["fold"]) != int(target["fold"]):
            raise ValueError(f"MPS OOF fold mismatch for {digest}")
        if source.get("target_status") != target["target_status"]:
            raise ValueError(f"MPS OOF target status mismatch for {digest}")
        if not _float_equal(source.get("target_seconds", ""), target["target_seconds"], field="MPS target_seconds"):
            raise ValueError(f"MPS OOF target mismatch for {digest}")
        source_quality = source.get("quality_status_separate_audit_only", source.get("quality_status_separate", ""))
        if source_quality != target["quality_status"]:
            raise ValueError(f"MPS OOF quality status mismatch for {digest}")
        for method_id, method_column in method_columns.items():
            prediction = _number(source.get(method_column, ""), field=f"{method_column}:{digest}", allow_blank=True)
            if prediction is not None and prediction < 0:
                raise ValueError(f"negative runtime prediction in {method_column}:{digest}")
            terminal_status = "predicted" if prediction is not None else "missing_prediction"
            record = {
                "source_sha256": digest,
                "fold": int(target["fold"]),
                "source_artifact": source_artifact,
                "source_method_column": method_column,
                "target_seconds": target["target_seconds"],
                "target_status": target["target_status"],
                "quality_status": target["quality_status"],
                "prediction_seconds": prediction,
                "method_terminal_status": terminal_status,
                "terminal_reason": "" if prediction is not None else "OOF prediction field missing/non-finite",
                "panel_member_count": target["panel_member_count"],
                "member_ids": [],
            }
            method_records[method_id][digest] = record
            score_status = "target_unavailable" if target["target_seconds"] is None else terminal_status
            coverage_rows.append(
                {
                "domain_id": domain_id,
                "artifact_id": ARTIFACT_ID,
                "evaluation_target_clock": target_clock,
                "method_output_clock": method_output_clock,
                    "source_artifact": source_artifact,
                    "method_id": method_id,
                    "source_method_column": method_column,
                    "source_sha256": digest,
                    "fold": target["fold"],
                    "panel_member_count": target["panel_member_count"],
                    "member_ids_json": "[]",
                    "target_status": target["target_status"],
                    "quality_status": target["quality_status"],
                    "target_seconds": "" if target["target_seconds"] is None else target["target_seconds"],
                    "prediction_seconds": "" if prediction is None else prediction,
                    "method_terminal_status": terminal_status,
                    "score_status": score_status,
                    "terminal_reason": record["terminal_reason"] or ("target unavailable; prediction retained but not scored" if score_status == "target_unavailable" else ""),
                }
            )
    if set(oof_by_hash) != set(target_by_hash):
        raise ValueError(f"MPS OOF exact-hash envelope mismatch ({len(oof_by_hash)} predictions vs {len(target_by_hash)} target rows)")
    return method_records, coverage_rows


def load_mps_panel(e3_dir: Path, e5_dir: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]], str]:
    _require_files(e3_dir, E3_REQUIRED, role="E3 fixed-MPS chi16")
    _require_files(e5_dir, E5_REQUIRED, role="E5 fixed-MPS family residual")
    for name, expected in EXPECTED_E5_SHA256.items():
        actual = sha256_file(e5_dir / name)
        if actual != expected:
            raise ValueError(f"pinned E5 input digest mismatch for {name}: {actual} != {expected}")

    e3_manifest = _read_json(e3_dir / "preflight_manifest.json")
    e5_manifest = _read_json(e5_dir / "materialization_manifest.json")
    target_clock = str(e3_manifest.get("target", ""))
    if target_clock != "cudaq_mps_fp64_bond16_warm_state_execution_seconds":
        raise ValueError(f"unexpected E3 MPS target clock: {target_clock!r}")
    if str(e5_manifest.get("target_clock_id", "")) != target_clock:
        raise ValueError("E3 and E5 MPS target clocks differ")
    e3_targets = _target_table(e3_dir / "hash_targets_and_static_features.csv")
    e5_targets = _target_table(e5_dir / "hash_targets_and_static_features.csv")
    if set(e3_targets) != set(e5_targets):
        raise ValueError("E3/E5 MPS hash target sets differ")
    for digest in e3_targets:
        left, right = e3_targets[digest], e5_targets[digest]
        if left["fold"] != right["fold"] or left["target_status"] != right["target_status"] or left["quality_status"] != right["quality_status"]:
            raise ValueError(f"E3/E5 MPS target status or fold mismatch for {digest}")
        if not _float_equal(left["target_seconds"], right["target_seconds"], field="E3/E5 MPS target"):
            raise ValueError(f"E3/E5 MPS label differs for {digest}")
    if len(e3_targets) != 150:
        raise ValueError(f"MPS exact-hash panel has {len(e3_targets)} assigned hashes, expected 150")
    counts = Counter(row["target_status"] for row in e3_targets.values())
    quality = Counter(row["quality_status"] for row in e3_targets.values())
    if counts != Counter({"runtime_observed": 144, "unavailable_adapter_error": 6}):
        raise ValueError(f"MPS target coverage changed from frozen 144/6 envelope: {dict(counts)}")
    if quality != Counter({"quality_pass": 142, "quality_failed": 2, "quality_unavailable": 6}):
        raise ValueError(f"MPS quality audit counts changed from frozen 142/2/6: {dict(quality)}")
    folds = Counter(int(row["fold"]) for row in e3_targets.values())
    if dict(sorted(folds.items())) != EXPECTED_FOLDS:
        raise ValueError(f"MPS C44 fold counts differ from frozen envelope: {dict(folds)}")

    e3_oof = _mps_oof_rows(e3_dir / "five_fold_oof_predictions.csv")
    e5_oof = _mps_oof_rows(e5_dir / "five_fold_oof_predictions.csv")
    method_columns = {
        "e3_fixed_mps_train_fold_median": "pred_train_fold_median_seconds",
        "e3_fixed_mps_ridge_alpha_1": "pred_ridge_alpha_1_seconds",
        "e3_fixed_mps_graph_median_three_seeds": "pred_graph_median_three_seeds_seconds",
        "e5_family_residual_median_three_seeds": "pred_family_residual_median_three_seeds_seconds",
        "e5_family_agnostic_median_three_seeds": "pred_family_agnostic_median_three_seeds_seconds",
    }
    e3_cols = {key: value for key, value in method_columns.items() if key.startswith("e3_")}
    e5_cols = {key: value for key, value in method_columns.items() if key.startswith("e5_")}
    methods_e3, coverage_e3 = _make_mps_method_records(
        domain_id="mps_fixed_chi16",
        source_artifact="mps_fixed_chi16_runtime_oof",
        target_clock=target_clock,
        method_output_clock=EXPECTED_MPS_METHOD_OUTPUT_CLOCK,
        method_columns=e3_cols,
        oof_by_hash=e3_oof,
        target_by_hash=e3_targets,
    )
    methods_e5, coverage_e5 = _make_mps_method_records(
        domain_id="mps_fixed_chi16",
        source_artifact="family_residual_fixed_mps_runtime",
        target_clock=target_clock,
        method_output_clock=EXPECTED_MPS_METHOD_OUTPUT_CLOCK,
        method_columns=e5_cols,
        oof_by_hash=e5_oof,
        target_by_hash=e3_targets,
    )
    # E5 embeds audited S85 comparators; require exact hash-by-hash identity with E3.
    comparator_columns = {
        "pred_s85_train_fold_median_seconds": "e3_fixed_mps_train_fold_median",
        "pred_s85_ridge_alpha_1_seconds": "e3_fixed_mps_ridge_alpha_1",
        "pred_s85_graph_median_three_seeds_seconds": "e3_fixed_mps_graph_median_three_seeds",
    }
    for digest in sorted(e3_oof):
        for e5_column, e3_method in comparator_columns.items():
            if not _float_equal(e5_oof[digest].get(e5_column, ""), e3_oof[digest].get(method_columns[e3_method], ""), field=e5_column):
                raise ValueError(f"E5's S85 comparator differs from E3 exact-hash output for {digest}:{e5_column}")
    methods = {**methods_e3, **methods_e5}
    coverage = coverage_e3 + coverage_e5
    return methods, coverage, target_clock


def _quality_population(record: Mapping[str, Any], population: str) -> bool:
    if record["target_seconds"] is None:
        return False
    if population == "all_finite_targets":
        return True
    if population == "quality_pass_only":
        return record["quality_status"] == "quality_pass"
    if population == "quality_failed_finite_only":
        return record["quality_status"] == "quality_failed"
    raise ValueError(f"unknown metric population: {population}")


def summarize_method(
    *,
    domain_id: str,
    source_artifact: str,
    source_method_column: str,
    method_id: str,
    records: Mapping[str, Mapping[str, Any]],
    target_clock: str,
    method_output_clock: str,
    population: str,
    quality_available: bool,
) -> dict[str, Any]:
    assigned = list(records.values())
    finite_targets = [record for record in assigned if record["target_seconds"] is not None]
    quality_pass = [record for record in assigned if record["quality_status"] == "quality_pass"]
    quality_failed = [record for record in assigned if record["quality_status"] == "quality_failed"]
    scored = [record for record in assigned if _quality_population(record, population) and record["prediction_seconds"] is not None]
    target = np.asarray([float(record["target_seconds"]) for record in scored], dtype=np.float64)
    prediction = np.asarray([float(record["prediction_seconds"]) for record in scored], dtype=np.float64)
    values: dict[str, Any] = {field: "" for field in METRIC_NAMES}
    if len(scored):
        if np.any(target < 0) or np.any(prediction < 0):
            raise ValueError(f"log1p metric requires nonnegative values in {method_id}")
        error = prediction - target
        absolute = np.abs(error)
        values.update(
            {
                "mae_seconds": float(np.mean(absolute)),
                "medae_seconds": float(np.median(absolute)),
                "rmse_seconds": float(np.sqrt(np.mean(error ** 2))),
                "log1p_mae_seconds": float(np.mean(np.abs(np.log1p(prediction) - np.log1p(target)))),
                "r2_seconds": "",
                "p90_absolute_error_seconds": float(np.quantile(absolute, 0.90, method="linear")),
                "p99_absolute_error_seconds": float(np.quantile(absolute, 0.99, method="linear")),
                "max_absolute_error_seconds": float(np.max(absolute)),
            }
        )
        variance = float(np.sum((target - np.mean(target)) ** 2))
        if len(scored) > 1 and variance > 0:
            values["r2_seconds"] = float(1.0 - np.sum(error ** 2) / variance)
    summary = {
        "domain_id": domain_id,
        "artifact_id": ARTIFACT_ID,
        "evaluation_target_clock": target_clock,
        "method_output_clock": method_output_clock,
        "source_artifact": source_artifact,
        "source_method_column": source_method_column,
        "method_id": method_id,
        "population": population,
        "assigned_hashes": len(assigned),
        "finite_target_hashes": len(finite_targets),
        "quality_pass_hashes": len(quality_pass) if quality_available else "",
        "quality_failed_finite_hashes": len(quality_failed) if quality_available else "",
        "target_unavailable_hashes": len(assigned) - len(finite_targets),
        "prediction_hashes_assigned": sum(record["prediction_seconds"] is not None for record in assigned),
        "scored_hashes": len(scored),
        "coverage_of_assigned": len(scored) / len(assigned) if assigned else 0.0,
        "coverage_of_finite_targets": len(scored) / len(finite_targets) if population == "all_finite_targets" and finite_targets else (len(scored) / max(1, len(quality_pass)) if population == "quality_pass_only" else (len(scored) / max(1, len(quality_failed)) if population == "quality_failed_finite_only" else 0.0)),
        **values,
        "status": "evaluated" if scored else "unavailable_no_scored_predictions",
    }
    return summary


def summarize_panel(
    *,
    domain_id: str,
    methods: Mapping[str, Mapping[str, Mapping[str, Any]]],
    target_clock: str,
    method_output_clock: str,
    quality_available: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id, records in sorted(methods.items()):
        sample = next(iter(records.values())) if records else {}
        source_artifact = str(sample.get("source_artifact", ""))
        source_column = str(sample.get("source_method_column", method_id))
        if quality_available:
            populations = ("all_finite_targets", "quality_pass_only", "quality_failed_finite_only")
        else:
            populations = ("all_finite_targets",)
        for population in populations:
            rows.append(
                summarize_method(
                    domain_id=domain_id,
                    source_artifact=source_artifact,
                    source_method_column=source_column,
                    method_id=method_id,
                    records=records,
                    target_clock=target_clock,
                    method_output_clock=method_output_clock,
                    population=population,
                    quality_available=quality_available,
                )
            )
    return rows


def registered_seed(registry: Mapping[str, Any], *, comparison_label: str) -> int:
    root_seed = int(registry["root_seed"])
    stream = str(registry["streams"]["bootstrap"])
    payload = f"{root_seed}|{stream}|{ARTIFACT_ID}|{comparison_label}|0".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def paired_bootstrap_row(
    *,
    domain_id: str,
    target_clock: str,
    method_output_clock: str,
    population: str,
    comparison_label: str,
    candidate_method_id: str,
    reference_method_id: str,
    candidate: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
    assigned_hashes: int,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    stream = str(registry["streams"]["bootstrap"])
    seed = registered_seed(registry, comparison_label=comparison_label)
    shared = sorted(
        digest
        for digest in set(candidate) & set(reference)
        if _quality_population(candidate[digest], population)
        and _quality_population(reference[digest], population)
        and candidate[digest]["prediction_seconds"] is not None
        and reference[digest]["prediction_seconds"] is not None
    )
    row: dict[str, Any] = {
        "domain_id": domain_id,
        "artifact_id": ARTIFACT_ID,
        "evaluation_target_clock": target_clock,
        "method_output_clock": method_output_clock,
        "population": population,
        "comparison_label": comparison_label,
        "candidate_method_id": candidate_method_id,
        "reference_method_id": reference_method_id,
        "assigned_hashes": assigned_hashes,
        "candidate_predicted_hashes": sum(_quality_population(value, population) and value["prediction_seconds"] is not None for value in candidate.values()),
        "reference_predicted_hashes": sum(_quality_population(value, population) and value["prediction_seconds"] is not None for value in reference.values()),
        "shared_hashes": len(shared),
        "shared_hash_set_sha256": _digest_sorted_hashes(shared),
        "observed_mae_difference_seconds": "",
        "bootstrap_mean_mae_difference_seconds": "",
        "bootstrap_ci_low_seconds": "",
        "bootstrap_ci_high_seconds": "",
        "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
        "bootstrap_replicates_completed": 0,
        "bootstrap_seed": seed,
        "bootstrap_stream": stream,
        "bootstrap_group_column": "source_sha256",
        "percentile_rule": BOOTSTRAP_QUANTILE_RULE,
        "coverage_asymmetric": row_coverage_asymmetric(candidate, reference, population),
        "status": "unavailable_no_shared_hashes",
    }
    if not shared:
        return row
    target = np.asarray([float(candidate[digest]["target_seconds"]) for digest in shared], dtype=np.float64)
    candidate_pred = np.asarray([float(candidate[digest]["prediction_seconds"]) for digest in shared], dtype=np.float64)
    reference_pred = np.asarray([float(reference[digest]["prediction_seconds"]) for digest in shared], dtype=np.float64)
    candidate_abs_error = np.abs(candidate_pred - target)
    reference_abs_error = np.abs(reference_pred - target)
    observed = float(np.mean(candidate_abs_error) - np.mean(reference_abs_error))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(shared), size=(BOOTSTRAP_REPLICATES, len(shared)))
    deltas = np.mean(candidate_abs_error[indices], axis=1) - np.mean(reference_abs_error[indices], axis=1)
    low, high = np.quantile(deltas, [0.025, 0.975], method="linear")
    row.update(
        {
            "observed_mae_difference_seconds": observed,
            "bootstrap_mean_mae_difference_seconds": float(np.mean(deltas)),
            "bootstrap_ci_low_seconds": float(low),
            "bootstrap_ci_high_seconds": float(high),
            "bootstrap_replicates_completed": BOOTSTRAP_REPLICATES,
            "status": "evaluated",
        }
    )
    return row


def row_coverage_asymmetric(
    candidate: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
    population: str,
) -> bool:
    def predicted(method: Mapping[str, Mapping[str, Any]]) -> set[str]:
        return {
            digest for digest, record in method.items()
            if _quality_population(record, population) and record["prediction_seconds"] is not None
        }
    return predicted(candidate) != predicted(reference)


def predefined_aer_comparisons(methods: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    medians = {view: f"azizov_gnn_{view}_median_three_seeds" for view in ("source", "hybrid", "transpiled")}
    for candidate_view, reference_view in (("source", "hybrid"), ("source", "transpiled"), ("hybrid", "transpiled")):
        specs.append(
            {
                "domain_id": "aer_azizov_core_q9",
                "comparison_label": f"azizov_{candidate_view}_vs_{reference_view}",
                "candidate_method_id": medians[candidate_view],
                "reference_method_id": medians[reference_view],
            }
        )
    for view, graph_method in medians.items():
        for classical in ("linear_regression", "ridge", "svr_rbf", "random_forest", "xgboost"):
            specs.append(
                {
                    "domain_id": "aer_azizov_core_q9",
                    "comparison_label": f"azizov_{view}_vs_classical_{classical}_{view}",
                    "candidate_method_id": graph_method,
                    "reference_method_id": f"classical_{classical}_{view}",
                }
            )
    return specs


def predefined_mps_comparisons() -> list[dict[str, str]]:
    residual = "e5_family_residual_median_three_seeds"
    agnostic = "e5_family_agnostic_median_three_seeds"
    graph = "e3_fixed_mps_graph_median_three_seeds"
    ridge = "e3_fixed_mps_ridge_alpha_1"
    specs = [
        ("family_residual_vs_family_agnostic", residual, agnostic),
        ("family_residual_vs_e3_s85_graph", residual, graph),
        ("family_residual_vs_e3_s85_ridge", residual, ridge),
        ("family_agnostic_vs_e3_s85_graph", agnostic, graph),
        ("family_agnostic_vs_e3_s85_ridge", agnostic, ridge),
    ]
    return [
        {
            "domain_id": "mps_fixed_chi16",
            "comparison_label": label,
            "candidate_method_id": candidate,
            "reference_method_id": reference,
        }
        for label, candidate, reference in specs
    ]


def build_method_cards(
    *,
    aer_methods: Mapping[str, Any],
    mps_methods: Mapping[str, Any],
    aer_output_clock: str,
    mps_output_clock: str,
) -> dict[str, dict[str, str]]:
    cards: dict[str, dict[str, str]] = {}
    for method_id in sorted(aer_methods):
        if method_id.startswith("azizov_gnn_"):
            prefix = "azizov_gnn_"
            suffix = method_id.removeprefix(prefix)
            view = next((name for name in ("source", "hybrid", "transpiled") if suffix.startswith(f"{name}_")), "")
            if not view:
                raise ValueError(f"cannot parse Azizov view from method ID {method_id!r}")
            detail = suffix.removeprefix(f"{view}_")
            view_label = {"source": "source circuit", "hybrid": "hybrid", "transpiled": "transpiled circuit"}[view]
            if detail == "median_three_seeds":
                seed_label = "median of seeds 42, 1234, and 31415"
            elif detail.startswith("seed_") and detail.removeprefix("seed_").isdigit():
                seed_label = f"seed {detail.removeprefix('seed_')} individual output"
            else:
                raise ValueError(f"cannot parse Azizov seed reduction from method ID {method_id!r}")
            role = "; individual-seed diagnostic, not a selected seed" if detail.startswith("seed_") else ""
            reader_label = f"Azizov-style {view_label} graph/global adaptation ({seed_label}{role})"
            fidelity = "adaptation"
            stage = view
            boundary = "Local Azizov-style graph/global adaptation on frozen Aer warm labels; not paper-exact or published-accuracy reproduction."
        else:
            prefix = "classical_"
            if not method_id.startswith(prefix):
                raise ValueError(f"cannot classify Aer method ID {method_id!r}")
            suffix = method_id.removeprefix(prefix)
            view = next((name for name in ("source", "hybrid", "transpiled") if suffix.endswith(f"_{name}")), "")
            if not view:
                raise ValueError(f"cannot parse classical representation view from method ID {method_id!r}")
            model = suffix[: -(len(view) + 1)]
            reader_label = f"Local {model.replace('_', ' ')} global-feature baseline ({view} view)"
            fidelity = "baseline"
            stage = f"{view}_global_features"
            boundary = "Local classical baseline on the frozen Aer warm target; not an Azizov-paper method reproduction."
        cards[method_id] = {
            "method_id": method_id,
            "reader_label": reader_label,
            "fidelity_class": fidelity,
            "representation_stage": stage,
            "method_output_clock": aer_output_clock,
            "claim_boundary": boundary,
        }
    for method_id in sorted(mps_methods):
        if method_id.startswith("e3_fixed_mps_"):
            suffix = method_id.removeprefix("e3_fixed_mps_")
            if suffix == "train_fold_median":
                reader_label = "S85 training-fold median baseline, fixed MPS chi16"
            elif suffix == "ridge_alpha_1":
                reader_label = "S85 six-feature Ridge baseline, fixed MPS chi16"
            elif suffix == "graph_median_three_seeds":
                reader_label = "S85 source-DAG graph median-of-seeds adaptation, fixed MPS chi16"
            else:
                reader_label = f"Fixed-MPS chi16 local {suffix.replace('_', ' ')} baseline"
            if suffix == "graph_median_three_seeds":
                fidelity = "adaptation"
                stage = "source_dag_graph_runtime_only_adaptation"
                boundary = "Local source-DAG graph runtime adaptation on fixed-MPS labels; not a simulator timing result or paper-exact reproduction."
            else:
                fidelity = "baseline"
                stage = "exact_hash_runtime_only_baseline"
                boundary = "S85 fixed-configuration MPS runtime-only baseline; not a quantum simulator timing or paper-exact reproduction."
        elif method_id.startswith("e5_"):
            suffix = method_id.removeprefix("e5_")
            reader_label = f"Family-Aware-inspired fixed-MPS {suffix.replace('_', ' ')}"
            fidelity = "adaptation"
            stage = "family_residual_runtime_only_adaptation" if "residual" in method_id else "family_agnostic_runtime_only_ablation"
            boundary = "Family-Aware-inspired runtime-only adaptation on fixed-MPS labels; no threshold output, family-held-out transfer, or paper-exact reproduction claim."
        else:
            raise ValueError(f"cannot classify fixed-MPS method ID {method_id!r}")
        cards[method_id] = {
            "method_id": method_id,
            "reader_label": reader_label,
            "fidelity_class": fidelity,
            "representation_stage": stage,
            "method_output_clock": mps_output_clock,
            "claim_boundary": boundary,
        }
    return cards


def build_comparison_rows(
    *,
    domain_id: str,
    methods: Mapping[str, Mapping[str, Mapping[str, Any]]],
    target_clock: str,
    method_output_clock: str,
    comparisons: Sequence[Mapping[str, str]],
    registry: Mapping[str, Any],
    quality_available: bool,
) -> list[dict[str, Any]]:
    populations = ("all_finite_targets", "quality_pass_only") if quality_available else ("all_finite_targets",)
    output: list[dict[str, Any]] = []
    for spec in comparisons:
        if spec["domain_id"] != domain_id:
            raise ValueError(f"cross-configuration comparison is forbidden: {spec}")
        candidate_id = spec["candidate_method_id"]
        reference_id = spec["reference_method_id"]
        for method_id in (candidate_id, reference_id):
            if method_id not in methods:
                raise ValueError(f"predeclared comparison references absent method {method_id!r} in {domain_id}")
        for population in populations:
            label = f"{domain_id}|{spec['comparison_label']}|{population}"
            output.append(
                paired_bootstrap_row(
                    domain_id=domain_id,
                    target_clock=target_clock,
                    method_output_clock=method_output_clock,
                    population=population,
                    comparison_label=label,
                    candidate_method_id=candidate_id,
                    reference_method_id=reference_id,
                    candidate=methods[candidate_id],
                    reference=methods[reference_id],
                    assigned_hashes=len(methods[candidate_id]),
                    registry=registry,
                )
            )
    return output


def _pin_files(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(set(paths)):
        if not path.is_file():
            raise FileNotFoundError(f"required provenance input is missing: {path}")
        relative = str(path.resolve().relative_to(ROOT))
        inventory[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return inventory


def _source_inventory(aer_dir: Path, e3_dir: Path, e5_dir: Path) -> dict[str, Any]:
    aer_paths = [aer_dir / name for name in E4_REQUIRED]
    e3_paths = [e3_dir / name for name in E3_REQUIRED]
    e5_paths = [e5_dir / name for name in E5_REQUIRED]
    # Pin fold/environment QA outputs for every outer fold, not only the combined OOF CSV.
    for fold in range(5):
        e3_paths.extend(
            e3_dir / f"fold_{fold}" / name
            for name in (f"fold{fold}_predictions.csv", f"fold{fold}_qa.json", "environment.json")
        )
    extra = [
        ROOT / "benchmark_v1/registry/seed_registry.json",
        ROOT / "benchmark_v1/protocol/azizov_common_core_gnn_v1.json",
        ROOT / "benchmark_v1/protocol/azizov_local_feature_dictionary_v1.json",
        ROOT / "benchmark_v1/protocol/family_residual_runtime_only_v1.json",
        ROOT / "benchmark_v1/protocol/unified_split_contract_v2.json",
        ROOT / "benchmark_v1/protocol/benchmark_result_table_schema_v1.json",
        ROOT / "benchmark_v1/protocol/simulator_completion_gates_v1.json",
        ROOT / "benchmark_v1/decisions/S82_UNIFIED_BENCHMARK_CONTRACT_20261002.md",
        ROOT / "benchmark_v1/decisions/S85_MPS_FOLD0_DIAGNOSTIC_AUTHORIZATION_20261002.md",
        ROOT / "benchmark_v1/execution/manifests/cudaq_verified_bridge_v1.json",
        ROOT / "benchmark_v1/execution/manifests/simulator_environment_lock_v1.json",
        ROOT / "benchmark_v1/scripts/aggregate_predictive_oof_v1.py",
        ROOT / "benchmark_v1/scripts/materialize_azizov_common_core_v1.py",
        ROOT / "benchmark_v1/scripts/run_azizov_common_core_adaptation_v1.py",
        ROOT / "benchmark_v1/scripts/run_mps_fixed_chi16_runtime_adaptation_v1.py",
        ROOT / "benchmark_v1/scripts/run_family_residual_fixed_mps_runtime_v1.py",
        ROOT / "benchmark_v1/requirements-azizov-fit.txt",
        ROOT / "artifacts/benchmark_v1/sim_common_q16_manifest_20260927/sim_common_q16_manifest.csv",
        ROOT / "artifacts/benchmark_v1/sim_common_q16_manifest_20260927/manifest.json",
        ROOT / "artifacts/benchmark_v1/c44_aer_q16_full_panel_evaluation_20260927/aer_q16_reduced_warm.csv",
        ROOT / "artifacts/benchmark_v1/sim_aer_q9_full_20260927/aer_q9_full.environment.json",
        ROOT / "artifacts/benchmark_v1/sim_aer_q9_full_20260927/aer_q9_full.pip-freeze.txt",
        ROOT / "artifacts/benchmark_v1/cudaq_mps_common_q16_bridge_full_v1_20260928/attempt_records.jsonl",
        ROOT / "artifacts/benchmark_v1/cudaq_mps_common_q16_bridge_full_v1_20260928/cudaq_mps_common_raw.csv",
        ROOT / "artifacts/benchmark_v1/cudaq_mps_common_q16_bridge_full_v1_20260928/environment.json",
        ROOT / "artifacts/benchmark_v1/cudaq_mps_common_q16_bridge_full_v1_20260928/run_manifest.json",
    ]
    # Pin the exact checked E5 inputs and retained execution/materialization context.
    inputs = [*aer_paths, *e3_paths, *e5_paths, *extra]
    return {
        "artifact_id": ARTIFACT_ID,
        "inputs": _pin_files(inputs),
        "source_roles": {
            "E4_aer_azizov": [str((aer_dir / name).resolve().relative_to(ROOT)) for name in E4_REQUIRED],
            "E3_fixed_mps_chi16": [str((e3_dir / name).resolve().relative_to(ROOT)) for name in E3_REQUIRED],
            "E5_family_residual_fixed_mps": [str((e5_dir / name).resolve().relative_to(ROOT)) for name in E5_REQUIRED],
        },
        "pinned_e5_input_sha256": dict(EXPECTED_E5_SHA256),
        "bootstrap": {
            "seed_registry_path": "benchmark_v1/registry/seed_registry.json",
            "stream": "20260925-bootstrap",
            "seed_derivation": "first 4 SHA256 bytes big-endian of str(root_seed)|bootstrap_stream|artifact_id|comparison_label|0",
            "replicates": BOOTSTRAP_REPLICATES,
            "percentile_rule": BOOTSTRAP_QUANTILE_RULE,
        },
    }


def _write_new_artifact(output_dir: Path, files: Mapping[str, bytes]) -> None:
    """Publish atomically to an additive path; never replace a non-identical directory."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"aggregate output path is not a directory: {output_dir}")
        existing = {str(path.relative_to(output_dir)): path.read_bytes() for path in output_dir.rglob("*") if path.is_file()}
        expected = dict(files)
        if existing == expected:
            return
        raise FileExistsError(f"refusing to overwrite a non-identical aggregate artifact: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent) as temp_name:
        temp_dir = Path(temp_name)
        for relative, payload in files.items():
            path = temp_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        os.replace(temp_dir, output_dir)


def _jsonable_float_columns(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    # csv.DictWriter serializes None as a blank cell and keeps terminal rows intact.
    return [dict(row) for row in rows]


def run_aggregate(
    *,
    aer_dir: Path = DEFAULT_AER_DIR,
    mps_e3_dir: Path = DEFAULT_MPS_E3_DIR,
    mps_e5_dir: Path = DEFAULT_MPS_E5_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    aer_dir, mps_e3_dir, mps_e5_dir = aer_dir.resolve(), mps_e3_dir.resolve(), mps_e5_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"aggregate output path is not a directory: {output_dir}")
    if not aer_dir.is_dir():
        raise FileNotFoundError(f"E4 Aer adaptation artifact is absent: {aer_dir}")

    aer_methods, aer_coverage, aer_clock, aer_output_clock, _ = load_aer_panel(aer_dir)
    mps_methods, mps_coverage, mps_clock = load_mps_panel(mps_e3_dir, mps_e5_dir)
    registry_path = ROOT / "benchmark_v1/registry/seed_registry.json"
    registry = _read_json(registry_path)
    if str(registry.get("streams", {}).get("bootstrap", "")) != "20260925-bootstrap":
        raise ValueError("seed registry bootstrap stream differs from the frozen value")

    aer_metrics = summarize_panel(
        domain_id="aer_azizov_core_q9",
        methods=aer_methods,
        target_clock=aer_clock,
        method_output_clock=aer_output_clock,
        quality_available=False,
    )
    mps_metrics = summarize_panel(
        domain_id="mps_fixed_chi16",
        methods=mps_methods,
        target_clock=mps_clock,
        method_output_clock=EXPECTED_MPS_METHOD_OUTPUT_CLOCK,
        quality_available=True,
    )
    aer_bootstrap = build_comparison_rows(
        domain_id="aer_azizov_core_q9",
        methods=aer_methods,
        target_clock=aer_clock,
        method_output_clock=aer_output_clock,
        comparisons=predefined_aer_comparisons(aer_methods),
        registry=registry,
        quality_available=False,
    )
    mps_bootstrap = build_comparison_rows(
        domain_id="mps_fixed_chi16",
        methods=mps_methods,
        target_clock=mps_clock,
        method_output_clock=EXPECTED_MPS_METHOD_OUTPUT_CLOCK,
        comparisons=predefined_mps_comparisons(),
        registry=registry,
        quality_available=True,
    )

    source_hashes = _source_inventory(aer_dir, mps_e3_dir, mps_e5_dir)
    source_hashes_bytes = canonical_json_bytes(source_hashes)
    source_hashes_sha256 = hashlib.sha256(source_hashes_bytes).hexdigest()
    method_cards = build_method_cards(
        aer_methods=aer_methods,
        mps_methods=mps_methods,
        aer_output_clock=aer_output_clock,
        mps_output_clock=EXPECTED_MPS_METHOD_OUTPUT_CLOCK,
    )
    manifest_base: dict[str, Any] = {
        "artifact_id": ARTIFACT_ID,
        "schema_version": "1.0",
        "status": "complete_aggregate" if all(row["status"] == "evaluated" for row in [*aer_metrics, *mps_metrics]) else "complete_with_unavailable_method_rows",
        "scope": "Separate exact-hash OOF summaries for Aer Azizov adaptation and fixed-MPS chi16; no cross-engine, cross-clock or cross-target pooling.",
        "panels": {
            "aer_azizov_core_q9": {
                "evaluation_target_clock": aer_clock,
                "method_output_clock": aer_output_clock,
                "assigned_hashes": len(next(iter(aer_methods.values()))),
                "method_count": len(aer_methods),
                "comparison_count": len(aer_bootstrap),
                "comparison_policy": "Three Azizov view contrasts plus each view against the five same-panel classical baselines; all on shared exact hashes.",
            },
            "mps_fixed_chi16": {
                "evaluation_target_clock": mps_clock,
                "method_output_clock": EXPECTED_MPS_METHOD_OUTPUT_CLOCK,
                "method_output_clock_mapping": "Prediction fields are evaluated against the frozen fixed-MPS target; the method_output_clock label distinguishes predicted output from measured target clock.",
                "assigned_hashes": len(next(iter(mps_methods.values()))),
                "finite_runtime_labels": 144,
                "quality_pass_finite": 142,
                "quality_failed_finite_retained": 2,
                "target_unavailable_unimputed": 6,
                "method_count": len(mps_methods),
                "comparison_count": len(mps_bootstrap),
                "comparison_policy": "Family residual vs no-family, each against matched S85/E3 graph and ridge; separate method outputs retained, no label imputation.",
            },
        },
        "metric_names": list(METRIC_NAMES),
        "method_cards": method_cards,
        "population_policy": {
            "aer": "One hash target is the median of existing alias-level warm_execution labels; metric population is finite hash targets.",
            "mps": "Primary finite target population retains 2 quality failures; 142 quality-pass metrics and the quality-failed subset are separately emitted; six unavailable targets remain unscored.",
        },
        "bootstrap_policy": {
            "replicates": BOOTSTRAP_REPLICATES,
            "group_column": "source_sha256",
            "paired": True,
            "interval": "95% percentile interval using numpy.quantile(method='linear')",
            "point_estimate": "observed MAE(candidate)-MAE(reference) on original shared hashes; separately recorded from bootstrap mean",
            "seed_registry": "benchmark_v1/registry/seed_registry.json",
            "cross_domain_comparisons": "forbidden",
        },
        "source_hashes_path": "source_hashes.json",
        "source_hashes_sha256": source_hashes_sha256,
        "runtime_environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "no_refitting_or_imputation": True,
    }
    output_rows = {
        "aer_azizov_hash_metrics.csv": aer_metrics,
        "aer_azizov_hash_coverage.csv": aer_coverage,
        "aer_azizov_paired_bootstrap.csv": aer_bootstrap,
        "mps_fixed_chi16_hash_metrics.csv": mps_metrics,
        "mps_fixed_chi16_hash_coverage.csv": mps_coverage,
        "mps_fixed_chi16_paired_bootstrap.csv": mps_bootstrap,
    }
    field_map = {
        "aer_azizov_hash_metrics.csv": METRICS_FIELDS,
        "aer_azizov_hash_coverage.csv": COVERAGE_FIELDS,
        "aer_azizov_paired_bootstrap.csv": BOOTSTRAP_FIELDS,
        "mps_fixed_chi16_hash_metrics.csv": METRICS_FIELDS,
        "mps_fixed_chi16_hash_coverage.csv": COVERAGE_FIELDS,
        "mps_fixed_chi16_paired_bootstrap.csv": BOOTSTRAP_FIELDS,
    }
    output_files = {name: csv_bytes(rows, field_map[name]) for name, rows in output_rows.items()}
    output_files["source_hashes.json"] = source_hashes_bytes
    manifest_base["files"] = {
        name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        for name, payload in sorted(output_files.items())
    }
    output_files["aggregate_manifest.json"] = canonical_json_bytes(manifest_base)
    _write_new_artifact(output_dir, output_files)
    return {
        "status": manifest_base["status"],
        "output_dir": str(output_dir),
        "aer_methods": len(aer_methods),
        "mps_methods": len(mps_methods),
        "aer_hashes": manifest_base["panels"]["aer_azizov_core_q9"]["assigned_hashes"],
        "mps_hashes": manifest_base["panels"]["mps_fixed_chi16"]["assigned_hashes"],
        "mps_finite_targets": 144,
        "mps_quality_pass": 142,
        "mps_unavailable": 6,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aer-dir", type=Path, default=DEFAULT_AER_DIR)
    parser.add_argument("--mps-e3-dir", type=Path, default=DEFAULT_MPS_E3_DIR)
    parser.add_argument("--mps-e5-dir", type=Path, default=DEFAULT_MPS_E5_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run_aggregate(aer_dir=args.aer_dir, mps_e3_dir=args.mps_e3_dir, mps_e5_dir=args.mps_e5_dir, output_dir=args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
