#!/usr/bin/env python3
"""Build the common 7,350-observation real-QPU result tables.

Reads only pinned OOF outputs. Does not fit models, change labels or overwrite
source attempts. Metrics are emitted with full assigned coverage and exact
row-set identities; method output clocks remain explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
PANEL_ROOT = ROOT / "artifacts/real_qpu/common_panel"
PANEL = PANEL_ROOT / "panel.csv"
TARGETS = PANEL_ROOT / "targets.csv"
OUTER = PANEL_ROOT / "outer_splits.csv"
ANALYTICAL = PANEL_ROOT / "analytical"
QONDUCTOR = PANEL_ROOT / "qonductor_polynomial"
HYB = PANEL_ROOT / "hyb"
MALI = PANEL_ROOT / "mali"
REPRESENTATIONS = ROOT / "artifacts/benchmark_v3/real_qpu/compiled_unified/representation_rows.csv"
COMPONENTS = ROOT / "artifacts/benchmark_v3/real_qpu/analytical_extension/components.csv"
OUTPUT = PANEL_ROOT / "summary"
CONTRACT = ROOT / "benchmark_v1/protocol/common_panel_completion.json"
COMMON_CONTRACT = ROOT / "benchmark_v1/protocol/real_qpu_common_panel.json"
ARCHIVE_ANALYTICAL = ROOT / "artifacts/benchmark_v3/real_qpu/archived_analytical_wave_v3_20260930/attempts.csv"
SCHOLTEN = ROOT / "artifacts/real_qpu/scholten_nominal/attempts.csv"
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 42
EXPECTED_SOURCES = {"mali_real_qpu": 340, "qonductor_single_circuit_ibm": 3065, "qpack_mcp": 3945}
TARGET_CLOCK = "archived_observed_service_execution_time"
SUMMARY_NAMES = ("coverage.csv", "method_metrics.csv", "paired_metrics.csv",
                 "source_and_reconstruction_metrics.csv", "unavailable_reasons.csv")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ids_sha(ids: list[str] | set[str]) -> str:
    return sha_bytes("\n".join(sorted(ids)).encode("utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(100_000_000)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def verify_manifest(folder: Path, *, required_inputs: dict[str, str] | None = None) -> dict[str, Any]:
    manifest_path = folder / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"run_manifest_missing:{manifest_path.relative_to(ROOT)}")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError(f"source_run_not_complete:{manifest_path.relative_to(ROOT)}")
    for name, expected in manifest.get("output_hashes", {}).items():
        path = folder / name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"source_output_hash_mismatch:{path.relative_to(ROOT)}")
    for relative, expected in manifest.get("input_hashes", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"source_input_hash_mismatch:{relative}")
    if required_inputs:
        input_hashes = manifest.get("input_hashes", {})
        for relative, expected in required_inputs.items():
            if input_hashes.get(relative) != expected:
                raise RuntimeError(f"source_common_panel_pin_mismatch:{relative}")
    return manifest


def load_common_population() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    panel_rows, target_rows, outer_rows = read_csv(PANEL), read_csv(TARGETS), read_csv(OUTER)
    def keyed(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
        result = {row["canonical_observation_id"]: row for row in rows}
        if len(result) != len(rows):
            raise RuntimeError(f"duplicate_{name}_id")
        return result
    panel = keyed(panel_rows, "panel")
    targets = keyed(target_rows, "target")
    outer = keyed(outer_rows, "outer")
    if len(panel) != 7350 or set(panel) != set(targets) or set(panel) != set(outer):
        raise RuntimeError("common_panel_identity_not_exact_7350")
    source_counts = Counter(row["source_id"] for row in panel.values())
    if dict(source_counts) != EXPECTED_SOURCES:
        raise RuntimeError(f"common_panel_source_counts_changed:{dict(source_counts)}")
    for identity, row in panel.items():
        target = targets[identity]
        if row["source_id"] != target["source_id"] or row["logical_input_tier"] != target["logical_input_tier"]:
            raise RuntimeError(f"panel_target_metadata_mismatch:{identity}")
        if int(outer[identity]["outer_fold"]) not in range(5):
            raise RuntimeError(f"invalid_outer_fold:{identity}")
    return panel, targets, outer


def row_record(identity: str, method_id: str, *, panel: dict, targets: dict, outer: dict,
               reader_label: str, fidelity: str, role: str, output_clock: str,
               status: str, prediction: Any = "", reason: str = "", variant: str = "",
               seed: str = "", family_id: str = "") -> dict[str, Any]:
    target = targets[identity]
    actual = float(target["target_seconds"])
    predicted = ""
    if status == "predicted":
        predicted = float(prediction)
        if not math.isfinite(predicted):
            raise RuntimeError(f"nonfinite_prediction_marked_predicted:{method_id}:{identity}")
    return {"canonical_observation_id": identity, "source_id": panel[identity]["source_id"],
        "backend": target["backend"], "group_id": outer[identity]["group_id"],
        "logical_input_tier": panel[identity]["logical_input_tier"], "outer_fold": int(outer[identity]["outer_fold"]),
        "method_id": method_id, "method_family_id": family_id or method_id, "variant": variant,
        "seed": seed, "reader_label": reader_label, "fidelity_class": fidelity,
        "comparison_role": role, "evaluation_target_clock": TARGET_CLOCK,
        "method_output_clock": output_clock, "actual_seconds": actual,
        "predicted_seconds": predicted, "status": status, "terminal_reason": reason}


def check_prediction_identity(row: dict[str, str], identity: str, target: dict[str, dict[str, str]],
                              *, actual_key: str) -> None:
    if identity not in target:
        raise RuntimeError(f"prediction_identity_outside_panel:{identity}")
    actual = float(row[actual_key])
    expected = float(target[identity]["target_seconds"])
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError(f"prediction_target_mismatch:{identity}:{actual}:{expected}")


def load_analytical(panel: dict, targets: dict, outer: dict, expected_pins: dict[str, str]) -> list[dict]:
    manifest = verify_manifest(ANALYTICAL, required_inputs=expected_pins)
    rows = read_csv(ANALYTICAL / "attempts.csv")
    if len(rows) != 21 * 7350:
        raise RuntimeError(f"analytical_attempt_rows_mismatch:{len(rows)}")
    source_archive = read_csv(ARCHIVE_ANALYTICAL)
    source_scholten = read_csv(SCHOLTEN)
    original: dict[tuple[str, str], dict[str, str]] = {}
    for row in source_archive:
        key = (row["method_id"], row["canonical_row_id"])
        if key in original:
            raise RuntimeError(f"duplicate_archived_analytical_attempt:{key}")
        original[key] = row
    scholten_by_id = {row["canonical_row_id"]: row for row in source_scholten}
    if len(scholten_by_id) != len(source_scholten):
        raise RuntimeError("duplicate_scholten_nominal_attempt")

    output = []
    keys = set()
    for row in rows:
        identity, method_id = row["canonical_observation_id"], row["method_id"]
        if identity not in panel or (method_id, identity) in keys:
            raise RuntimeError(f"analytical_attempt_id_invalid_or_duplicate:{method_id}:{identity}")
        keys.add((method_id, identity))
        if int(row["outer_fold"]) != int(outer[identity]["outer_fold"]):
            raise RuntimeError(f"analytical_outer_fold_mismatch:{method_id}:{identity}")
        check_prediction_identity(row, identity, targets, actual_key="actual_seconds")
        base_id = row["base_method_id"]
        source_row = scholten_by_id.get(identity) if base_id.startswith("scholten_") else original.get((base_id, identity))
        if source_row is None:
            raise RuntimeError(f"analytical_source_attempt_missing:{base_id}:{identity}")
        if base_id.startswith("scholten_"):
            source_status = source_row["status"]
            source_reason = source_row["terminal_reason"]
            source_prediction = source_row["prediction"]
            source_actual = source_row["target_seconds"]
        else:
            source_status = source_row["status"]
            source_reason = source_row["terminal_reason"]
            source_prediction = source_row["prediction_seconds"]
            source_actual = source_row["observed_target_seconds"]
        if not math.isclose(float(source_actual), float(targets[identity]["target_seconds"]), rel_tol=0, abs_tol=1e-10):
            raise RuntimeError(f"analytical_source_label_mismatch:{base_id}:{identity}")
        if (source_status in {"predicted", "prediction_produced"}) != (row["status"] == "predicted"):
            raise RuntimeError(f"analytical_source_terminal_mismatch:{method_id}:{identity}:{source_status}:{row['status']}")
        status = row["status"]
        reason = row["terminal_reason"]
        if status != "predicted":
            # The existing panel projection flattened original terminal causes.
            # Restore them in this derived table without rewriting its receipt.
            status, reason = source_status, source_reason or "source_terminal_reason_not_recorded"
        elif row["variant"] == "raw":
            expected = float(source_prediction)
            actual_prediction = float(row["predicted_seconds"])
            if not math.isclose(expected, actual_prediction, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f"analytical_raw_projection_value_mismatch:{method_id}:{identity}")
        base_name = row["base_method_id"]
        if base_name.startswith("qcre_"):
            family = "QCRE timing-aware adaptation"
            fidelity, role = "analytical_proxy", "timing_baseline"
        elif base_name.startswith("qiskit_"):
            family = "Qiskit scheduled-duration reference"
            fidelity, role = "analytical_proxy", "scheduled_reference"
        elif base_name.startswith("hyb_"):
            family = "Hyb-HANAS legacy effective-cost diagnostic"
            fidelity, role = "analytical_proxy", "legacy_diagnostic"
        else:
            family = "Scholten nominal-throughput adaptation"
            fidelity, role = "adaptation", "throughput_baseline"
        suffix = row["variant"].replace("outer_train_", "outer-train ").replace("_", " ")
        label = f"{family} — {suffix}"
        output.append(row_record(identity, method_id, panel=panel, targets=targets, outer=outer,
            reader_label=label, fidelity=fidelity, role=role, output_clock=row["method_output_clock"],
            status=status, prediction=row["predicted_seconds"], reason=reason, variant=row["variant"], family_id=base_name))
    if len(keys) != 21 * 7350:
        raise RuntimeError("analytical_method_row_coverage_mismatch")
    calibration_rows = read_csv(ANALYTICAL / "calibration_fits.csv")
    base_methods = sorted({row["base_method_id"] for row in rows})
    calibration_keys = set()
    for fit in calibration_rows:
        base_id, variant, fold = fit["base_method_id"], fit["variant"], int(fit["outer_fold"])
        key = (base_id, variant, fold)
        if key in calibration_keys or base_id not in base_methods or variant not in {"raw", "outer_train_affine", "outer_train_log_affine"}:
            raise RuntimeError(f"analytical_calibration_record_identity_invalid:{key}")
        calibration_keys.add(key)
        eligible = []
        for identity in panel:
            if int(outer[identity]["outer_fold"]) == fold:
                continue
            source_row = scholten_by_id.get(identity) if base_id.startswith("scholten_") else original.get((base_id, identity))
            if source_row is None:
                continue
            source_status = source_row["status"]
            source_prediction = source_row["prediction"] if base_id.startswith("scholten_") else source_row["prediction_seconds"]
            if source_status in {"predicted", "prediction_produced"} and source_prediction != "":
                prediction = float(source_prediction)
                if math.isfinite(prediction) and prediction >= 0:
                    eligible.append(identity)
        eligible.sort()
        if (int(fit["fit_rows"]) != len(eligible) or fit["fit_id_sha256"] != ids_sha(eligible)
                or fit["fit_rows_from_outer_train_only"].lower() != "true"):
            raise RuntimeError(f"analytical_calibration_fit_membership_invalid:{base_id}:{variant}:{fold}")
    expected_calibrations = len(base_methods) * 3 * 5
    if len(calibration_keys) != expected_calibrations:
        raise RuntimeError(f"analytical_calibration_row_count_invalid:{len(calibration_keys)}")
    return output


def load_qonductor(panel: dict, targets: dict, outer: dict, expected_pins: dict[str, str]) -> list[dict]:
    manifest = verify_manifest(QONDUCTOR, required_inputs=expected_pins)
    rows = read_csv(QONDUCTOR / "predictions.csv")
    method_id = manifest["method_id"]
    if len(rows) != 7350 or {row["canonical_observation_id"] for row in rows} != set(panel):
        raise RuntimeError("qonductor_prediction_panel_not_exact")
    result = []
    for row in rows:
        identity = row["canonical_observation_id"]
        check_prediction_identity(row, identity, targets, actual_key="actual_seconds")
        if int(row["outer_fold"]) != int(outer[identity]["outer_fold"]):
            raise RuntimeError(f"qonductor_fold_mismatch:{identity}")
        result.append(row_record(identity, method_id, panel=panel, targets=targets, outer=outer,
            reader_label=manifest["reader_label"], fidelity=manifest["fidelity_class"], role="direct_regression_adaptation",
            output_clock=TARGET_CLOCK + "_predicted", status=row["status"], prediction=row["predicted_seconds"],
            reason=row.get("terminal_reason", ""), family_id=method_id))
    return result


def load_hyb(panel: dict, targets: dict, outer: dict, expected_pins: dict[str, str]) -> list[dict]:
    manifest = verify_manifest(HYB, required_inputs=expected_pins)
    rows = read_csv(HYB / "attempts.csv")
    descriptions = {row["method_id"]: row for row in read_csv(HYB / "metrics.csv")}
    components_rows = read_csv(COMPONENTS)
    components = {row["canonical_row_id"]: row for row in components_rows}
    representations_rows = read_csv(REPRESENTATIONS)
    representations = {row["canonical_row_id"]: row for row in representations_rows}
    representation_mismatches = []
    if not set(panel) <= set(components) or not set(panel) <= set(representations):
        raise RuntimeError("hyb_component_representation_join_not_total")
    for identity in panel:
        target_shots = int(float(targets[identity]["shots"]))
        component, representation = components[identity], representations[identity]
        if int(float(component["shots"])) != target_shots:
            raise RuntimeError(f"hyb_component_shots_do_not_match_target:{identity}")
        if int(float(representation["shots"])) != target_shots:
            representation_mismatches.append(identity)
        if (component["source_id"] != panel[identity]["source_id"]
                or component["graph_digest"] != representation["graph_input_digest"]
                or component["graph_file"] != representation["graph_file"]):
            raise RuntimeError(f"hyb_component_representation_identity_mismatch:{identity}")
    shot_receipt = manifest.get("compiled_representation_shot_metadata", {})
    if (shot_receipt.get("mismatch_rows") != len(representation_mismatches)
            or shot_receipt.get("mismatch_ids_sha256") != ids_sha(representation_mismatches)
            or shot_receipt.get("component_shots_match_common_targets") is not True
            or shot_receipt.get("cost_features_use_component_shots") is not True):
        raise RuntimeError("hyb_shot_metadata_receipt_not_reproduced")
    fold_fits = read_csv(HYB / "fold_fits.csv")
    fit_keys = set()
    for fit in fold_fits:
        method, fold = fit["method_id"], int(fit["outer_fold"])
        key = (method, fold)
        if key in fit_keys or fit["status"] != "fit":
            raise RuntimeError(f"hyb_fit_record_identity_invalid:{key}")
        fit_keys.add(key)
        if method == "hyb_nominal_r2_log_cost_ridge_v1":
            feature_field = "nominal_log_cost"
        elif method == "hyb_kyoto_composite_r2_log_cost_ridge_v1":
            feature_field = "composite_log_cost"
        elif method == "hyb_kyoto_composite_r2_gate_time_ridge_v1":
            feature_field = "composite_gate_seconds"
        else:
            raise RuntimeError(f"unexpected_hyb_fit_method:{method}")
        eligible = []
        test_feature_rows = 0
        for identity in panel:
            component = components[identity]
            value = component.get(feature_field, "")
            available = value != "" and math.isfinite(float(value))
            if feature_field == "composite_gate_seconds":
                available = available and float(value) > 0
            if available and int(outer[identity]["outer_fold"]) != fold:
                eligible.append(identity)
            elif available and int(outer[identity]["outer_fold"]) == fold:
                test_feature_rows += 1
        eligible.sort()
        scores = json.loads(fit["inner_source_balanced_mae_seconds"])
        selected = min(scores, key=lambda alpha: (float(scores[alpha]), float(alpha)))
        if (int(fit["eligible_fit_rows"]) != len(eligible)
                or fit["fit_ids_sha256"] != ids_sha(eligible)
                or int(fit["test_feature_rows"]) != test_feature_rows
                or not math.isclose(float(fit["selected_alpha"]), float(selected), rel_tol=0, abs_tol=1e-15)
                or set(scores) != {"0.1", "1.0", "10.0", "100.0"}):
            raise RuntimeError(f"hyb_fit_receipt_mismatch:{method}:{fold}")
    if len(fit_keys) != 15:
        raise RuntimeError(f"hyb_fit_count_not_15:{len(fit_keys)}")
    if len(rows) != 7 * 7350:
        raise RuntimeError(f"hyb_attempt_rows_mismatch:{len(rows)}")
    result = []
    seen = set()
    for row in rows:
        identity, method_id = row["canonical_observation_id"], row["method_id"]
        if identity not in panel or (method_id, identity) in seen:
            raise RuntimeError(f"hyb_attempt_duplicate_or_outside_panel:{method_id}:{identity}")
        seen.add((method_id, identity))
        if int(row["outer_fold"]) != int(outer[identity]["outer_fold"]):
            raise RuntimeError(f"hyb_outer_fold_mismatch:{identity}")
        check_prediction_identity(row, identity, targets, actual_key="actual_seconds")
        feature_digest = sha_bytes(json.dumps({key: components[identity].get(key) for key in
            ("nominal_log_cost", "composite_log_cost", "nominal_gate_seconds", "composite_gate_seconds", "shots")},
            sort_keys=True, separators=(",", ":")).encode())
        if row["input_feature_digest"] != feature_digest:
            raise RuntimeError(f"hyb_attempt_feature_digest_mismatch:{method_id}:{identity}")
        info = descriptions.get(method_id)
        if info is None:
            raise RuntimeError(f"hyb_method_description_missing:{method_id}")
        result.append(row_record(identity, method_id, panel=panel, targets=targets, outer=outer,
            reader_label=info["reader_label"], fidelity=info["fidelity_class"],
            role="primary_hyb" if "nominal" in method_id else "hyb_sensitivity_or_control",
            output_clock=row["method_output_clock"], status=row["status"], prediction=row["predicted_seconds"],
            reason=row["terminal_reason"], variant=method_id, family_id="hyb_hanas_r2"))
    if len(seen) != 7 * 7350:
        raise RuntimeError("hyb_method_identity_coverage_mismatch")
    observed_terminals = {method: dict(Counter(row["status"] for row in rows if row["method_id"] == method))
                          for method in descriptions}
    if observed_terminals != manifest.get("terminal_counts_by_method"):
        raise RuntimeError("hyb_terminal_count_receipt_mismatch")
    return result


def load_mali(panel: dict, targets: dict, outer: dict) -> tuple[list[dict], dict[str, dict[str, list[float]]]]:
    receipt_path = MALI / "final_validation.json"
    if not receipt_path.is_file():
        raise RuntimeError("mali_final_validation_missing_run_supervisor_first")
    receipt = read_json(receipt_path)
    validator_path = ROOT / "benchmark_v1/scripts/validate_common_panel_execution.py"
    if (receipt.get("status") != "PASS" or receipt.get("cell_count") != 30
            or receipt.get("validator_sha256") != sha(validator_path)
            or receipt.get("contract_sha256") != sha(CONTRACT)):
        raise RuntimeError("mali_final_validation_receipt_invalid")

    output = []
    by_method_seed: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    cell_inputs = []
    for method in ("mali_logical_graph", "mali_global_mlp"):
        fidelity = "architecture_adaptation" if method == "mali_logical_graph" else "matched_feature_baseline"
        label = "Ma–Li-style logical graph + 51 globals (common-panel adaptation)" if method == "mali_logical_graph" else "Matched 51-global MLP baseline"
        for fold in range(5):
            fold_ids = {identity for identity, row in outer.items() if int(row["outer_fold"]) == fold}
            for seed in (42, 1234, 31415):
                folder = MALI / f"fold_{fold}" / method / f"seed_{seed}"
                manifest_path = folder / "cell_manifest.json"
                if not manifest_path.is_file():
                    raise RuntimeError(f"mali_cell_manifest_missing:{method}:{fold}:{seed}")
                manifest = read_json(manifest_path)
                if (manifest.get("status") != "complete" or manifest.get("epochs") != 500
                        or manifest.get("method") != method or manifest.get("outer_fold") != fold
                        or manifest.get("seed") != seed):
                    raise RuntimeError(f"mali_cell_manifest_identity_invalid:{method}:{fold}:{seed}")
                for name, expected in manifest.get("output_hashes", {}).items():
                    if not (folder / name).is_file() or sha(folder / name) != expected:
                        raise RuntimeError(f"mali_cell_artifact_hash_mismatch:{method}:{fold}:{seed}:{name}")
                if manifest.get("n_test_assigned") != len(fold_ids):
                    raise RuntimeError(f"mali_cell_fold_denominator_mismatch:{method}:{fold}:{seed}")
                cell_inputs.append(manifest_path.relative_to(ROOT).as_posix())
                pred_rows = read_csv(folder / "predictions.csv")
                failure_rows = read_csv(folder / "test_failures.csv")
                cell = []
                for row in pred_rows:
                    identity = row["canonical_observation_id"]
                    if (identity not in fold_ids or row["method_id"] != method or int(row["seed"]) != seed
                            or row["status"] != "predicted"):
                        raise RuntimeError(f"mali_prediction_cell_identity_invalid:{method}:{fold}:{seed}:{identity}")
                    cell.append(row_record(identity, f"{method}__seed_{seed}", panel=panel, targets=targets, outer=outer,
                        reader_label=label + f" (seed {seed})", fidelity=fidelity, role="seed_diagnostic",
                        output_clock=TARGET_CLOCK + "_predicted", status="predicted",
                        prediction=row["predicted_seconds"], seed=str(seed), family_id=method))
                for row in failure_rows:
                    identity = row["canonical_observation_id"]
                    if (identity not in fold_ids or row["method_id"] != method or int(row["seed"]) != seed
                            or row["status"] == "predicted"):
                        raise RuntimeError(f"mali_failure_cell_identity_invalid:{method}:{fold}:{seed}:{identity}")
                    cell.append(row_record(identity, f"{method}__seed_{seed}", panel=panel, targets=targets, outer=outer,
                        reader_label=label + f" (seed {seed})", fidelity=fidelity, role="seed_diagnostic",
                        output_clock=TARGET_CLOCK + "_predicted", status=row["status"],
                        reason=row["terminal_reason"], seed=str(seed), family_id=method))
                if len(cell) != len(fold_ids) or {row["canonical_observation_id"] for row in cell} != fold_ids:
                    raise RuntimeError(f"mali_cell_test_coverage_invalid:{method}:{fold}:{seed}")
                by_method_seed[method][str(seed)].extend(cell)

    dispersion: dict[str, dict[str, list[float]]] = {}
    for method in ("mali_logical_graph", "mali_global_mlp"):
        seed_values = by_method_seed[method]
        expected_ids = set(panel)
        for seed in ("42", "1234", "31415"):
            by_id = {row["canonical_observation_id"]: row for row in seed_values[seed]}
            if len(by_id) != 7350 or set(by_id) != expected_ids:
                raise RuntimeError(f"mali_seed_oof_coverage_not_7350:{method}:{seed}")
            output.extend(by_id.values())
        seed_maps = {seed: {row["canonical_observation_id"]: row for row in seed_values[seed]}
                     for seed in ("42", "1234", "31415")}
        medians = []
        spreads: dict[str, list[float]] = {"range": [], "median_absolute_deviation": []}
        for identity in sorted(expected_ids):
            records = [seed_maps[seed][identity] for seed in ("42", "1234", "31415")]
            good = [r for r in records if r["status"] == "predicted"]
            if len(good) == 3:
                values = [float(r["predicted_seconds"]) for r in good]
                median = float(np.median(values))
                medians.append(row_record(identity, method, panel=panel, targets=targets, outer=outer,
                    reader_label=("Ma–Li-style logical graph + 51 globals, 3-seed median" if method == "mali_logical_graph"
                                  else "Matched 51-global MLP, 3-seed median"),
                    fidelity="architecture_adaptation" if method == "mali_logical_graph" else "matched_feature_baseline",
                    role="primary_learned", output_clock=TARGET_CLOCK + "_predicted", status="predicted",
                    prediction=median, seed="median_42_1234_31415", family_id=method))
                spreads["range"].append(max(values) - min(values))
                spreads["median_absolute_deviation"].append(float(np.median(np.abs(np.asarray(values) - median))))
            else:
                reasons = sorted({f"seed_{r['seed']}:{r['status']}:{r['terminal_reason']}" for r in records
                                  if r["status"] != "predicted"})
                medians.append(row_record(identity, method, panel=panel, targets=targets, outer=outer,
                    reader_label=("Ma–Li-style logical graph + 51 globals, 3-seed median" if method == "mali_logical_graph"
                                  else "Matched 51-global MLP, 3-seed median"),
                    fidelity="architecture_adaptation" if method == "mali_logical_graph" else "matched_feature_baseline",
                    role="primary_learned", output_clock=TARGET_CLOCK + "_predicted",
                    status="incomplete_seed_predictions", reason=";".join(reasons),
                    seed="median_42_1234_31415", family_id=method))
        output.extend(medians)
        dispersion[method] = spreads
    return output, dispersion


def metric_summary(rows: list[dict]) -> dict[str, Any]:
    good = [row for row in rows if row["status"] == "predicted"]
    if not good:
        return {"scored_rows": 0, "scored_id_set_sha256": ids_sha([]), "coverage": 0.0,
            "mae_seconds": "", "medae_seconds": "", "r2_seconds": "", "log1p_mae": "",
            "p90_absolute_error_seconds": "", "p99_absolute_error_seconds": "",
            "max_absolute_error_seconds": "", "negative_prediction_count": 0,
            "source_balanced_mae_seconds": "", "source_balanced_status": "no_scored_rows"}
    ids = [row["canonical_observation_id"] for row in good]
    y = np.asarray([float(row["actual_seconds"]) for row in good], dtype=np.float64)
    p = np.asarray([float(row["predicted_seconds"]) for row in good], dtype=np.float64)
    errors = np.abs(y - p)
    if not np.isfinite(errors).all():
        raise RuntimeError("absolute_error_nonfinite")
    # Long-double accumulation avoids intermediate float64 overflow for very
    # poor but finite analytical proxies. If the true R² is outside float64,
    # retain that fact as -inf instead of silently clipping the prediction.
    yl, pl = y.astype(np.longdouble), p.astype(np.longdouble)
    residual = yl - pl
    centered = yl - np.mean(yl, dtype=np.longdouble)
    total_ss = np.sum(centered * centered, dtype=np.longdouble)
    residual_ss = np.sum(residual * residual, dtype=np.longdouble)
    if len(y) < 2 or total_ss == 0:
        r2: Any = ""
    else:
        value = np.longdouble(1.0) - residual_ss / total_ss
        r2 = float(value) if abs(value) <= np.finfo(np.float64).max else ("-inf" if value < 0 else "inf")
    log_errors = np.abs(np.log1p(y) - np.log1p(np.maximum(p, 0.0)))
    source_groups: dict[str, list[float]] = defaultdict(list)
    for row, error in zip(good, errors):
        source_groups[row["source_id"]].append(float(error))
    source_maes = [statistics.fmean(values) for values in source_groups.values()]
    return {"scored_rows": len(good), "scored_id_set_sha256": ids_sha(ids),
        "coverage": len(good) / len(rows) if rows else 0.0,
        "mae_seconds": statistics.fmean(float(value) for value in errors),
        "medae_seconds": float(np.median(errors)), "r2_seconds": r2,
        "log1p_mae": float(np.mean(log_errors)),
        "p90_absolute_error_seconds": float(np.quantile(errors, .90)),
        "p99_absolute_error_seconds": float(np.quantile(errors, .99)),
        "max_absolute_error_seconds": float(np.max(errors)),
        "negative_prediction_count": int((p < 0).sum()),
        "source_balanced_mae_seconds": statistics.fmean(source_maes) if len(source_maes) == len(EXPECTED_SOURCES) else "",
        "source_balanced_status": "all_sources_present" if len(source_maes) == len(EXPECTED_SOURCES)
            else "unavailable_source_without_scored_rows"}


def grouped_bootstrap_delta(rows_a: list[dict], rows_b: list[dict], common_ids: list[str],
                            all_groups: list[str], draws: np.ndarray) -> dict[str, Any]:
    by_a = {row["canonical_observation_id"]: row for row in rows_a if row["status"] == "predicted"}
    by_b = {row["canonical_observation_id"]: row for row in rows_b if row["status"] == "predicted"}
    source_by_id = {identity: by_a[identity]["source_id"] for identity in common_ids}
    group_index = {group: index for index, group in enumerate(all_groups)}
    group_delta = np.zeros(len(all_groups), dtype=np.float64)
    group_n = np.zeros(len(all_groups), dtype=np.float64)
    src_delta = {source: np.zeros(len(all_groups), dtype=np.float64) for source in EXPECTED_SOURCES}
    src_n = {source: np.zeros(len(all_groups), dtype=np.float64) for source in EXPECTED_SOURCES}
    observed_a, observed_b = [], []
    for identity in common_ids:
        a, b = by_a[identity], by_b[identity]
        err_a = abs(float(a["actual_seconds"]) - float(a["predicted_seconds"]))
        err_b = abs(float(b["actual_seconds"]) - float(b["predicted_seconds"]))
        delta = err_a - err_b
        group = a["group_id"]
        gi = group_index[group]
        group_delta[gi] += delta
        group_n[gi] += 1
        source = source_by_id[identity]
        src_delta[source][gi] += delta
        src_n[source][gi] += 1
        observed_a.append(err_a)
        observed_b.append(err_b)
    observed_delta = statistics.fmean(observed_a) - statistics.fmean(observed_b)
    boot_numer = draws @ group_delta
    boot_denom = draws @ group_n
    valid = boot_denom > 0
    pooled = boot_numer[valid] / boot_denom[valid]
    pooled_valid = int(pooled.size)
    result: dict[str, Any] = {"observed_delta_mae_a_minus_b_seconds": observed_delta,
        "bootstrap_mean_delta_mae_a_minus_b_seconds": float(np.mean(pooled)) if pooled_valid else "",
        "bootstrap_ci95_low_seconds": float(np.quantile(pooled, .025)) if pooled_valid >= 9500 else "",
        "bootstrap_ci95_high_seconds": float(np.quantile(pooled, .975)) if pooled_valid >= 9500 else "",
        "bootstrap_valid_replicates": pooled_valid}
    source_means = []
    source_replicates = None
    for source in EXPECTED_SOURCES:
        denom = draws @ src_n[source]
        vals = np.full(draws.shape[0], np.nan, dtype=np.float64)
        keep = denom > 0
        vals[keep] = (draws @ src_delta[source])[keep] / denom[keep]
        source_means.append(vals)
    matrix = np.vstack(source_means).T
    src_valid = np.isfinite(matrix).all(axis=1)
    source_replicates = matrix[src_valid].mean(axis=1)
    observed_source = []
    for source in EXPECTED_SOURCES:
        source_ids = [identity for identity in common_ids if source_by_id[identity] == source]
        if not source_ids:
            observed_source = []
            break
        observed_source.append(statistics.fmean(
            abs(float(by_a[i]["actual_seconds"]) - float(by_a[i]["predicted_seconds"]))
            - abs(float(by_b[i]["actual_seconds"]) - float(by_b[i]["predicted_seconds"])) for i in source_ids))
    result["observed_source_balanced_delta_a_minus_b_seconds"] = statistics.fmean(observed_source) if len(observed_source) == len(EXPECTED_SOURCES) else ""
    result["bootstrap_source_balanced_mean_delta_seconds"] = float(np.mean(source_replicates)) if source_replicates.size else ""
    result["bootstrap_source_balanced_ci95_low_seconds"] = float(np.quantile(source_replicates, .025)) if source_replicates.size >= 9500 else ""
    result["bootstrap_source_balanced_ci95_high_seconds"] = float(np.quantile(source_replicates, .975)) if source_replicates.size >= 9500 else ""
    result["bootstrap_source_balanced_valid_replicates"] = int(source_replicates.size)
    return result


def build_metrics(attempts: list[dict], panel: dict, targets: dict, outer: dict,
                  dispersion: dict[str, dict[str, list[float]]]) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    methods: dict[str, list[dict]] = defaultdict(list)
    for row in attempts:
        methods[row["method_id"]].append(row)
    for method, rows in methods.items():
        keyed = {row["canonical_observation_id"]: row for row in rows}
        if len(keyed) != len(rows) or set(keyed) != set(panel):
            raise RuntimeError(f"method_assigned_rows_not_exact:{method}:{len(keyed)}")
    if not methods:
        raise RuntimeError("no_methods_to_aggregate")

    coverage = []
    method_metrics = []
    reasons = []
    for method in sorted(methods):
        rows = methods[method]
        info = rows[0]
        status_counts = Counter(row["status"] for row in rows)
        reason_counts: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in rows:
            if row["status"] != "predicted":
                reason_counts[(row["status"], row["terminal_reason"] or "unspecified")].append(row["canonical_observation_id"])
        coverage.append({"method_id": method, "method_family_id": info["method_family_id"],
            "reader_label": info["reader_label"], "fidelity_class": info["fidelity_class"],
            "comparison_role": info["comparison_role"], "evaluation_target_clock": info["evaluation_target_clock"],
            "method_output_clock": info["method_output_clock"], "assigned_rows": 7350,
            "predicted_rows": status_counts.get("predicted", 0), "coverage": status_counts.get("predicted", 0) / 7350,
            "status_counts_json": json.dumps(dict(sorted(status_counts.items())), sort_keys=True),
            "assigned_id_set_sha256": ids_sha(set(panel))})
        summary = metric_summary(rows)
        method_metrics.append({"method_id": method, "method_family_id": info["method_family_id"],
            "variant": info["variant"], "seed": info["seed"], "reader_label": info["reader_label"],
            "fidelity_class": info["fidelity_class"], "comparison_role": info["comparison_role"],
            "evaluation_target_clock": info["evaluation_target_clock"], "method_output_clock": info["method_output_clock"],
            "assigned_rows": 7350, **summary,
            "seed_prediction_range_median_seconds": float(np.median(dispersion[info["method_family_id"]]["range"]))
                if info["method_id"] == info["method_family_id"] and info["method_family_id"] in dispersion and dispersion[info["method_family_id"]]["range"] else "",
            "seed_prediction_range_p90_seconds": float(np.quantile(dispersion[info["method_family_id"]]["range"], .90))
                if info["method_id"] == info["method_family_id"] and info["method_family_id"] in dispersion and dispersion[info["method_family_id"]]["range"] else "",
            "seed_prediction_mad_median_seconds": float(np.median(dispersion[info["method_family_id"]]["median_absolute_deviation"]))
                if info["method_id"] == info["method_family_id"] and info["method_family_id"] in dispersion and dispersion[info["method_family_id"]]["median_absolute_deviation"] else ""})
        for (status, reason), ids in sorted(reason_counts.items()):
            by_source = Counter(panel[identity]["source_id"] for identity in ids)
            reasons.append({"method_id": method, "status": status, "terminal_reason": reason,
                "count": len(ids), "id_set_sha256": ids_sha(ids),
                "source_counts_json": json.dumps(dict(sorted(by_source.items())), sort_keys=True)})

    slices = []
    for method in sorted(methods):
        by_id = {row["canonical_observation_id"]: row for row in methods[method]}
        partitions: dict[tuple[str, str], set[str]] = defaultdict(set)
        for identity, meta in panel.items():
            source, tier = meta["source_id"], meta["logical_input_tier"]
            partitions[("source", source)].add(identity)
            partitions[("reconstruction_tier", tier)].add(identity)
            partitions[("source_x_reconstruction_tier", f"{source}|{tier}")].add(identity)
        for (slice_type, slice_value), ids in sorted(partitions.items()):
            rows = [by_id[identity] for identity in sorted(ids)]
            metric = metric_summary(rows)
            slices.append({"method_id": method, "slice_type": slice_type, "slice_value": slice_value,
                "assigned_rows": len(ids), **metric})

    main_methods = {method: rows for method, rows in methods.items()
                    if not (rows[0]["seed"].isdigit())}
    main_ids = sorted(main_methods)
    all_intersection = set(panel)
    for rows in main_methods.values():
        all_intersection &= {row["canonical_observation_id"] for row in rows if row["status"] == "predicted"}
    paired = []
    for method in main_ids:
        rows = main_methods[method]
        subset = [row for row in rows if row["canonical_observation_id"] in all_intersection]
        summary = metric_summary(subset)
        paired.append({"comparison_type": "all_methods_intersection", "method_a": method, "method_b": "",
            "assigned_rows": 7350, "common_predicted_rows": len(all_intersection),
            "common_id_set_sha256": ids_sha(all_intersection),
            "mae_a_seconds": summary["mae_seconds"], "r2_a_seconds": summary["r2_seconds"],
            "mae_b_seconds": "", "r2_b_seconds": "", "observed_delta_mae_a_minus_b_seconds": "",
            "bootstrap_mean_delta_mae_a_minus_b_seconds": "", "bootstrap_ci95_low_seconds": "",
            "bootstrap_ci95_high_seconds": "", "bootstrap_valid_replicates": 0,
            "observed_source_balanced_delta_a_minus_b_seconds": "",
            "bootstrap_source_balanced_mean_delta_seconds": "", "bootstrap_source_balanced_ci95_low_seconds": "",
            "bootstrap_source_balanced_ci95_high_seconds": "", "bootstrap_source_balanced_valid_replicates": 0})

    all_groups = sorted({row["group_id"] for row in panel.values()})
    draws = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        len(all_groups), [1 / len(all_groups)] * len(all_groups), size=BOOTSTRAP_REPS).astype(np.float64)
    for index, method_a in enumerate(main_ids):
        rows_a = main_methods[method_a]
        ids_a = {row["canonical_observation_id"] for row in rows_a if row["status"] == "predicted"}
        for method_b in main_ids[index + 1:]:
            rows_b = main_methods[method_b]
            ids_b = {row["canonical_observation_id"] for row in rows_b if row["status"] == "predicted"}
            common = sorted(ids_a & ids_b)
            if not common:
                paired.append({"comparison_type": "pairwise", "method_a": method_a, "method_b": method_b,
                    "assigned_rows": 7350, "common_predicted_rows": 0, "common_id_set_sha256": ids_sha([])})
                continue
            common_set = set(common)
            score_a = metric_summary([row for row in rows_a if row["canonical_observation_id"] in common_set])
            score_b = metric_summary([row for row in rows_b if row["canonical_observation_id"] in common_set])
            boot = grouped_bootstrap_delta(rows_a, rows_b, common, all_groups, draws)
            paired.append({"comparison_type": "pairwise", "method_a": method_a, "method_b": method_b,
                "assigned_rows": 7350, "common_predicted_rows": len(common), "common_id_set_sha256": ids_sha(common),
                "common_group_count": len({outer[identity]["group_id"] for identity in common}),
                "mae_a_seconds": score_a["mae_seconds"], "r2_a_seconds": score_a["r2_seconds"],
                "mae_b_seconds": score_b["mae_seconds"], "r2_b_seconds": score_b["r2_seconds"], **boot})
    return coverage, method_metrics, paired, slices, reasons


def existing_summary_is_current() -> bool:
    manifest_path = OUTPUT / "run_manifest.json"
    if not OUTPUT.exists() or not any(OUTPUT.iterdir()):
        return False
    if not manifest_path.is_file():
        raise RuntimeError("summary_output_exists_without_manifest_refuse_to_overwrite")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError("existing_summary_manifest_not_complete_refuse_to_overwrite")
    refreshed_validation_receipt = "artifacts/real_qpu/common_panel/mali/final_validation.json"
    rebuildable_builder_pin = "benchmark_v1/scripts/aggregate_common_panel.py"
    refresh_required = False
    for relative, expected in manifest.get("input_hashes", {}).items():
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"existing_summary_stale_input_refuse_to_overwrite:{relative}")
        if sha(path) != expected:
            if relative == rebuildable_builder_pin:
                # A builder update requires recomputing derived outputs. Keep
                # the previous outputs protected by their hashes below.
                refresh_required = True
                continue
            if relative != refreshed_validation_receipt:
                raise RuntimeError(f"existing_summary_stale_input_refuse_to_overwrite:{relative}")
            receipt = read_json(path)
            validator_path = ROOT / "benchmark_v1/scripts/validate_common_panel_execution.py"
            if (receipt.get("status") != "PASS" or receipt.get("cell_count") != 30
                    or receipt.get("contract_sha256") != sha(CONTRACT)
                    or receipt.get("validator_sha256") != sha(validator_path)):
                raise RuntimeError("refreshed_neural_validation_receipt_not_current_PASS")
            # The independent validator records its run timestamp in this
            # receipt. If that is the only changed input, verify every
            # existing table hash and rebuild so the summary pins the new
            # receipt instead of silently retaining a stale provenance hash.
            refresh_required = True
    for name, expected in manifest.get("output_hashes", {}).items():
        path = OUTPUT / name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"existing_summary_output_hash_mismatch_refuse_to_overwrite:{name}")
    return not refresh_required


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    args = parser.parse_args()
    del args
    if existing_summary_is_current():
        print(json.dumps({"status": "complete", "summary": str(OUTPUT.relative_to(ROOT)),
                          "action": "validated_existing_noop"}, indent=2), flush=True)
        return
    pins = {"artifacts/real_qpu/common_panel/manifest.json": sha(PANEL_ROOT / "manifest.json"),
            "artifacts/real_qpu/common_panel/panel.csv": sha(PANEL),
            "artifacts/real_qpu/common_panel/outer_splits.csv": sha(OUTER),
            "artifacts/real_qpu/common_panel/targets.csv": sha(TARGETS)}
    for root in (ANALYTICAL, QONDUCTOR, HYB):
        if (root / "run_manifest.json").exists():
            old = read_json(root / "run_manifest.json")
            for relative in pins:
                if old.get("input_hashes", {}).get(relative) != pins[relative]:
                    raise RuntimeError(f"source_common_panel_input_pin_mismatch:{root.name}:{relative}")
    panel, targets, outer = load_common_population()
    required = {key: value for key, value in pins.items() if key != "artifacts/real_qpu/common_panel/manifest.json"}
    attempts = []
    attempts.extend(load_analytical(panel, targets, outer, required))
    attempts.extend(load_qonductor(panel, targets, outer, required))
    attempts.extend(load_hyb(panel, targets, outer, required))
    mali_attempts, dispersion = load_mali(panel, targets, outer)
    attempts.extend(mali_attempts)
    # Each reader-facing method, including seed diagnostics, must retain every
    # assigned identity exactly once. Seed medians are separately constrained
    # to rows where all three fixed seeds produced finite predictions.
    methods = {method_id for method_id in (row["method_id"] for row in attempts)}
    if len(methods) != 37:
        raise RuntimeError(f"expected_37_reader_methods_including_seed_diagnostics_got:{len(methods)}")
    # MALI seed diagnostics have one attempt per identity; each main method's
    # canonical table also has one attempt per identity.
    coverage, metrics, paired, slices, reasons = build_metrics(attempts, panel, targets, outer, dispersion)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payloads = {
        "coverage.csv": (coverage, list(coverage[0])),
        "method_metrics.csv": (metrics, list(metrics[0])),
        "paired_metrics.csv": (paired, list(paired[0])),
        "source_and_reconstruction_metrics.csv": (slices, list(slices[0])),
        "unavailable_reasons.csv": (reasons, list(reasons[0]) if reasons else
            ["method_id", "status", "terminal_reason", "count", "id_set_sha256", "source_counts_json"]),
    }
    for name, (rows, fields) in payloads.items():
        atomic_csv(OUTPUT / name, rows, fields)
    manifest = {"artifact_id": "real-qpu-common-panel-summary", "status": "complete",
        "assigned_rows_per_method": 7350, "source_counts": EXPECTED_SOURCES,
        "reader_method_count_including_seed_diagnostics": len(methods), "reader_main_method_count": len(metrics) - 6,
        "pairwise_comparison_count": sum(row["comparison_type"] == "pairwise" for row in paired),
        "bootstrap": {"replicates": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
            "unit": "frozen common-panel group_id", "groups": len({r["group_id"] for r in panel.values()}),
            "paired_delta_sign": "method_a minus method_b", "CI": "2.5th and 97.5th percentiles",
            "source_balanced_empty_source": "replicate marked invalid; valid replicate count is reported"},
        "reconstruction_sensitivity": {"excluded_tier": "candidate_recipe_sensitivity_only",
            "expected_rows": 4515, "no_retraining": True},
        "analytical_terminal_reasons": "derived rows restored from pinned archived source attempts; original analytical projection unchanged",
        "raw_extreme_predictions": "preserved; no clipping or row removal; R2 outside float64 range emitted as -inf",
        "common_panel_identity_sha256": ids_sha(set(panel)), "input_hashes": {
            "benchmark_v1/protocol/common_panel_completion.json": sha(CONTRACT),
            "benchmark_v1/protocol/real_qpu_common_panel.json": sha(COMMON_CONTRACT),
            "artifacts/real_qpu/common_panel/panel.csv": sha(PANEL),
            "artifacts/real_qpu/common_panel/targets.csv": sha(TARGETS),
            "artifacts/real_qpu/common_panel/outer_splits.csv": sha(OUTER),
            "artifacts/real_qpu/common_panel/analytical/run_manifest.json": sha(ANALYTICAL / "run_manifest.json"),
            "artifacts/real_qpu/common_panel/qonductor_polynomial/run_manifest.json": sha(QONDUCTOR / "run_manifest.json"),
            "artifacts/real_qpu/common_panel/hyb/run_manifest.json": sha(HYB / "run_manifest.json"),
            "artifacts/real_qpu/common_panel/mali/final_validation.json": sha(MALI / "final_validation.json"),
            "benchmark_v1/scripts/aggregate_common_panel.py": sha(Path(__file__))},
        "output_hashes": {name: sha(OUTPUT / name) for name in SUMMARY_NAMES}}
    tmp_manifest = OUTPUT / ".run_manifest.json.tmp"
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, OUTPUT / "run_manifest.json")
    print(json.dumps({"status": "complete", "reader_methods": len(methods),
        "main_metrics": len(metrics) - 6, "pairwise_comparisons": manifest["pairwise_comparison_count"],
        "all_method_intersection_rows": sum(r["comparison_type"] == "all_methods_intersection" for r in paired),
        "tables": {name: len(rows) for name, (rows, _fields) in payloads.items()},
        "summary": str(OUTPUT.relative_to(ROOT))}, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
