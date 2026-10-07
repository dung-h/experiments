"""Fail-closed input resolution for cached real-QPU refits.

This module is intentionally stdlib-only at import time.  It never guesses a
development checkout from the machine layout.  Neural training and QPU/simulator
execution are outside this adapter; the one optional numerical route is the
outer-train-only recalibration of already saved analytical predictions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


_FAMILIES = {
    "mali_graph_mlp": ("mali_real_qpu_graph_and_mlp", "frozen_panel_refit"),
    "qonductor_polynomial": ("qonductor_style_polynomial", "frozen_panel_refit"),
    "analytical_saved_raw": ("qcre_and_qiskit_duration", "saved_raw_calibration"),
    "hyb_components": ("hyb_hanas_components", "base_method_refit"),
}
_PANEL_INPUTS = {
    "data/real_qpu/inputs/panel.csv": "9ace84d3ec33ea050594ae57afdb76a4370f9717f361d78707f9d4ad4cd6238f",
    "data/real_qpu/inputs/targets.csv": "ed274962448b0d269dafd84db3eee121b5f42228b767b27153dd1ecf64ffcbfc",
    "data/real_qpu/inputs/outer_splits.csv": "62d6158411375795a89d6ce386f8c0f96f3124798ac2466b7fadb09b25698211",
    "data/real_qpu/inputs/inner_splits.csv": "67f3a77b1a76a046e8152060f3a2db544a6df78d7bd81b92baf5a78f1e67972e",
}
_MALI_INPUTS = {
    "data/real_qpu/inputs/feature_attempts.csv": "c1c5f4bf549b1c5f35c712d6148319da61a46110f3590b3a82563ac3a9e62bd5",
    "data/real_qpu/inputs/feature_manifest.json": "172affe8fd9bf1538e3e472abd450b7dcb0651222efbefbd61f1f28d70013a08",
    "data/real_qpu/inputs/mali_feature_ledger.csv": "322e7941434ed4a93fd98ce75fc2bbf0a0d93d8bab842261456bfd1b7d6bedb2",
    "data/real_qpu/inputs/input_index.csv": "5f0906d8c4108955389d321a705a43bb9095f67e19c53978af22060a6325512b",
    "data/real_qpu/inputs/input_index_manifest.json": "8df89f852c4fce0bfa3291665f12a11669371d67669be07104fede18eac72e4a",
}
_MALI_PACKAGED_INPUTS = (
    ("data/real_qpu/inputs/mali_batch_profile.json",
     "b601ba607188104b5cd4729789f253593301441b4b8809bd5872f7492d81da2b"),
    ("data/real_qpu/inputs/mali_neural_dependency_lock.txt",
     "766781c785791a57899c4ed98c2be6cdb48b840ce4c47c552f4fe46cdf7bf013"),
)
_ANALYTICAL_ATTEMPTS = "data/real_qpu/predictions/analytical/attempts.csv"
_HYB_COMPONENTS = "data/real_qpu/analytical_components/components.csv"
_HYB_RECEIPT = "data/real_qpu/analytical_components/run_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("input_path_must_be_nonempty_posix_relative_path")
    path = PurePosixPath(value.rstrip("/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe_relative_path:{value}")
    return path


def _resolve_under(root: Path, relative: str) -> Path:
    rel = _relative_path(relative)
    base = root.expanduser().resolve(strict=True)
    if not base.is_dir():
        raise ValueError(f"input_root_not_directory:{base}")
    candidate = base
    for part in rel.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"symlink_input_rejected:{relative}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(base):
        raise ValueError(f"input_path_escaped_root:{relative}")
    return resolved


def _directory_inventory(path: Path) -> str:
    pairs: list[list[str]] = []
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"symlink_in_directory_inventory:{item}")
        if item.is_file():
            pairs.append([item.relative_to(path).as_posix(), _sha256(item)])
        elif not item.is_dir():
            raise ValueError(f"non_regular_file_in_directory_inventory:{item}")
    pairs.sort(key=lambda pair: pair[0])
    payload = json.dumps(pairs, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest(package_root: Path) -> dict[str, Any]:
    path = _resolve_under(package_root, "protocol/reproduction.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _capability(manifest: dict[str, Any], family_id: str, capability_id: str) -> dict[str, Any]:
    family = next((x for x in manifest.get("families", []) if x.get("family_id") == family_id), None)
    if family is None:
        raise ValueError(f"unknown_method_family:{family_id}")
    capability = next((x for x in family.get("capabilities", [])
                       if x.get("capability_id") == capability_id), None)
    if capability is None:
        raise ValueError(f"unknown_capability:{family_id}:{capability_id}")
    return capability


def _resolve_entry(package_root: Path, entry: dict[str, Any],
                   external_roots: Mapping[str, str | os.PathLike[str]]) -> dict[str, Any]:
    declared = entry.get("status")
    result = {"path": entry.get("path"), "declared_status": declared,
              "root_kind": entry.get("root_kind"), "source": entry.get("source", ""),
              "expected_sha256": entry.get("sha256"), "status": "not_ready"}
    if declared == "scientifically_unavailable":
        result.update(status="unavailable", reason=entry.get("reason", "scientifically_unavailable"))
        return result
    if declared == "missing":
        result.update(status="not_ready", reason="declared_input_missing")
        return result
    root_kind = entry.get("root_kind")
    if root_kind == "package":
        root = package_root
    elif root_kind == "external":
        root_id = entry.get("external_root_id")
        result["external_root_id"] = root_id
        if not root_id or root_id not in external_roots:
            result.update(status="not_ready", reason="explicit_external_root_not_supplied")
            return result
        root = Path(external_roots[root_id])
    else:
        result.update(status="not_ready", reason="missing_or_invalid_root_kind")
        return result
    try:
        path = _resolve_under(root, str(entry.get("path", "")))
        expected = entry.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("missing_sha256_pin")
        if path.is_dir():
            kind = entry.get("inventory_digest_kind")
            if kind != "sha256_sorted_posix_relative_path_file_sha256_pairs_compact_json_utf8_v1":
                raise ValueError("directory_input_missing_supported_inventory_digest_kind")
            observed = _directory_inventory(path)
        elif path.is_file():
            observed = _sha256(path)
        else:
            raise ValueError("input_is_not_regular_file_or_directory")
        result.update(resolved_path=str(path), observed_sha256=observed,
                      status="ready" if observed == expected else "not_ready",
                      reason="" if observed == expected else "sha256_mismatch")
    except (OSError, ValueError) as exc:
        result.update(status="not_ready", reason=str(exc))
    return result


def _package_pin(package_root: Path, relative: str, expected: str,
                 source: str = "frozen publication package pin") -> dict[str, Any]:
    entry = {"path": relative, "sha256": expected, "status": "packaged", "root_kind": "package",
             "source": source}
    return _resolve_entry(package_root, entry, {})


def preflight_cached_refit(package_root: str | os.PathLike[str], method_family: str,
                           external_roots: Mapping[str, str | os.PathLike[str]] | None = None) -> dict[str, Any]:
    """Resolve a refit route's declared inputs; never guesses local dev paths."""
    if method_family not in _FAMILIES:
        raise ValueError(f"unsupported_method_family:{method_family}")
    root = Path(package_root).expanduser().resolve(strict=True)
    ext = external_roots or {}
    manifest = _manifest(root)
    family_id, capability_id = _FAMILIES[method_family]
    capability = _capability(manifest, family_id, capability_id)
    resolved = [_resolve_entry(root, item, ext) for item in capability.get("prerequisites", [])]
    for relative, expected in _PANEL_INPUTS.items():
        if method_family == "analytical_saved_raw" and relative.endswith("inner_splits.csv"):
            continue
        resolved.append(_package_pin(root, relative, expected))
    if method_family == "mali_graph_mlp":
        for relative, expected in _MALI_INPUTS.items():
            resolved.append(_package_pin(root, relative, expected))
        for relative, expected in _MALI_PACKAGED_INPUTS:
            source_path = {
                "data/real_qpu/inputs/mali_batch_profile.json":
                    "copied from development artifacts/real_qpu/common_panel/mali/batch_profile.json",
                "data/real_qpu/inputs/mali_neural_dependency_lock.txt":
                    "copied from development artifacts/real_qpu/mali_full_features_unified/dependency_lock_neural.txt",
            }[relative]
            resolved.append(_package_pin(root, relative, expected, source=source_path))
    elif method_family == "analytical_saved_raw":
        expected = next((item["sha256"] for item in capability.get("prerequisites", [])
                         if item.get("path") == _ANALYTICAL_ATTEMPTS), None)
        if expected:
            resolved.append(_package_pin(root, _ANALYTICAL_ATTEMPTS, expected))
    elif method_family == "hyb_components":
        resolved.extend([
            _package_pin(root, _HYB_COMPONENTS, "c98fe9babbbc8ed8e50a653bbd1d345ba96ab5eea3723f037e96bbd7e9da7444"),
            _package_pin(root, _HYB_RECEIPT, "077d44a972e90d4ee4c0e788fa42351a401623162fe246bcca41bba6e4becd26"),
        ])
    declared_blockers: list[str] = []
    if capability.get("status") != "ready":
        declared_blockers.append(
            f"capability_declared_{capability.get('status', 'not_ready')}; "
            "its frozen environment/prerequisite gate is not cleared"
        )
    if method_family == "mali_graph_mlp":
        declared_blockers.extend([
            "portable 7,350-row graph/MLP refit runner is not included in this adapter",
            "CUDA environment must match the pinned neural lock and the batch profile must be present and validated",
        ])
    elif method_family == "qonductor_polynomial":
        declared_blockers.append("Qonductor feature_rows sidecar remains external and its data-distribution rights are unresolved")
    elif method_family == "hyb_components":
        declared_blockers.append("cached component recalibration adapter is not included; fixed formula output is not a learned base-method refit")
    bad = [row for row in resolved if row["status"] != "ready"]
    status = "ready" if not bad and not declared_blockers else "not_ready"
    if capability.get("action") == "measure" or capability.get("status") == "unavailable":
        status = "unavailable"
    contract: dict[str, Any] = {}
    try:
        if method_family == "mali_graph_mlp":
            obj = json.loads(_resolve_under(root, "protocol/mali_full_features.json").read_text(encoding="utf-8"))
            contract = {"global_feature_order": obj["global_features"]["ordered_fields"],
                        "global_feature_count": len(obj["global_features"]["ordered_fields"]),
                        "node_feature_width": obj["graph"]["node_width"],
                        "representation": "51 globals + 178-node logical DAG; nominal-index T1/T2"}
        elif method_family == "qonductor_polynomial":
            obj = json.loads(_resolve_under(root, "protocol/qonductor_native_features.json").read_text(encoding="utf-8"))
            contract = {"feature_order": obj["features"]["actual_matrix_order"],
                        "preprocessing": obj["features"]["preprocessing"],
                        "degrees": obj["estimator"]["degree_candidates"]}
        elif method_family == "hyb_components":
            obj = json.loads(_resolve_under(root, "protocol/analytical_extension.json").read_text(encoding="utf-8"))
            contract = {"ridge_features": obj["ridge"]["hyb_features"],
                        "gate_time_features": obj["ridge"]["gate_time_control_features"],
                        "alphas": obj["ridge"]["alphas"]}
    except (OSError, KeyError, ValueError) as exc:
        declared_blockers.append(f"method_contract_unavailable:{exc}")
        status = "not_ready"
    return {"schema_version": 1, "method_family": method_family,
            "family_id": family_id, "capability_id": capability_id,
            "declared_capability_status": capability.get("status"),
            "status": status, "inputs": resolved, "feature_contract": contract,
            "blockers": declared_blockers + [f"{row['path']}:{row['reason']}" for row in bad],
            "execution_performed": False, "model_libraries_imported": False}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _line_fit(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    if len(pairs) < 2:
        raise ValueError("insufficient_outer_train_predictions")
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    if not all(math.isfinite(value) for value in (*xs, *ys)):
        raise ValueError("nonfinite_affine_fit_input")
    try:
        mx = math.fsum(xs) / len(xs)
        my = math.fsum(ys) / len(ys)
        denom = math.fsum((x - mx) ** 2 for x in xs)
        slope = (0.0 if denom == 0.0 else
                 math.fsum((x - mx) * (y - my) for x, y in pairs) / denom)
        intercept = my - slope * mx
    except (OverflowError, ValueError) as exc:
        raise ValueError("nonfinite_affine_fit_intermediate") from exc
    if not all(math.isfinite(value) for value in (mx, my, denom, slope, intercept)):
        raise ValueError("nonfinite_affine_fit_coefficients")
    return intercept, slope


def _apply_affine(coefficients: tuple[float, float], value: float) -> tuple[float | None, str, str]:
    intercept, slope = coefficients
    if not math.isfinite(intercept) or not math.isfinite(slope) or not math.isfinite(value):
        return None, "failed", "nonfinite_affine_coefficient_or_input"
    transformed = intercept + slope * value
    if not math.isfinite(transformed):
        return None, "overflow", "affine_prediction_nonfinite"
    return max(0.0, transformed), "predicted", ""


def _apply_log_affine(coefficients: tuple[float, float], value: float) -> tuple[float | None, str, str]:
    intercept, slope = coefficients
    if not math.isfinite(intercept) or not math.isfinite(slope) or not math.isfinite(value):
        return None, "failed", "nonfinite_log_affine_coefficient_or_input"
    log_prediction = intercept + slope * math.log1p(value)
    if not math.isfinite(log_prediction):
        return None, "overflow", "log_affine_prediction_nonfinite"
    try:
        transformed = math.expm1(log_prediction)
    except OverflowError:
        return None, "overflow", "log_affine_inverse_overflow"
    if not math.isfinite(transformed):
        return None, "overflow", "log_affine_inverse_nonfinite"
    return max(0.0, transformed), "predicted", ""


def refit_cached_inputs(method_family: str, package_root: str | os.PathLike[str],
                        output_dir: str | os.PathLike[str], *,
                        external_roots: Mapping[str, str | os.PathLike[str]] | None = None,
                        environment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run only the explicitly safe analytical recalibration route.

    Other families currently return a fail-closed not-ready record. This
    function must be invoked explicitly; importing this module never fits.
    """
    readiness = preflight_cached_refit(package_root, method_family, external_roots)
    if method_family != "analytical_saved_raw":
        return {**readiness, "execution_performed": False,
                "reason": "no_path_based_refit_implementation_for_this_method_family"}
    if readiness["status"] != "ready":
        return {**readiness, "execution_performed": False,
                "reason": "required_pinned_inputs_not_ready"}
    root = Path(package_root).expanduser().resolve(strict=True)
    requested_out = Path(os.path.abspath(os.fspath(output_dir)))
    out_parent = requested_out.parent.resolve(strict=True)
    out = out_parent / requested_out.name
    if out == root or out.is_relative_to(root):
        return {**readiness, "execution_performed": False,
                "reason": "output_directory_must_not_mutate_publication_package"}
    if out.exists() or out.is_symlink():
        return {**readiness, "execution_performed": False,
                "reason": "output_directory_must_not_exist"}
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out_parent))
    try:
        attempt_rows = _build_analytical_recalibration(root, staging, environment)
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"output_directory_created_during_refit:{out}")
        os.rename(staging, out)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "complete", "output_dir": str(out), "attempt_rows": attempt_rows,
            "execution_performed": True}


def _build_analytical_recalibration(root: Path, out: Path,
                                    environment: Mapping[str, Any] | None) -> int:
    paths = {"attempts": _resolve_under(root, _ANALYTICAL_ATTEMPTS),
             "targets": _resolve_under(root, "data/real_qpu/inputs/targets.csv"),
             "outer": _resolve_under(root, "data/real_qpu/inputs/outer_splits.csv")}
    base_rows = [r for r in _read_csv(paths["attempts"]) if r.get("variant") == "raw"]
    targets = {r["canonical_observation_id"]: r for r in _read_csv(paths["targets"])}
    split_rows = _read_csv(paths["outer"])
    folds = {r["canonical_observation_id"]: int(r["outer_fold"]) for r in split_rows}
    groups = {r["canonical_observation_id"]: r.get("group_id", "") for r in split_rows}
    if (not targets or set(targets) != set(folds) or len(folds) != len(split_rows)
            or any(fold not in range(5) for fold in folds.values()) or any(not group for group in groups.values())):
        raise ValueError("target_outer_split_identity_mismatch")
    for fold in range(5):
        train_groups = {groups[i] for i in folds if folds[i] != fold}
        test_groups = {groups[i] for i in folds if folds[i] == fold}
        if train_groups & test_groups:
            raise ValueError(f"outer_fold_group_leakage:{fold}")
    for identity, target in targets.items():
        actual = float(target["target_seconds"])
        if not math.isfinite(actual) or actual <= 0:
            raise ValueError(f"invalid_archived_target_seconds:{identity}")
    by_method: dict[str, dict[str, dict[str, str]]] = {}
    for row in base_rows:
        base_id = row.get("base_method_id", "")
        identity = row.get("canonical_observation_id", "")
        if not base_id or identity not in targets or identity in by_method.setdefault(base_id, {}):
            raise ValueError("duplicate_or_unknown_raw_analytical_attempt")
        if (int(row.get("outer_fold", -1)) != folds[identity]
                or row.get("source_id") != targets[identity]["source_id"]
                or row.get("evaluation_target_clock") != "archived_observed_service_execution_time"):
            raise ValueError(f"raw_attempt_identity_or_clock_mismatch:{base_id}:{identity}")
        raw_actual = float(row.get("actual_seconds", "nan"))
        if not math.isfinite(raw_actual) or raw_actual != float(targets[identity]["target_seconds"]):
            raise ValueError(f"raw_attempt_target_mismatch:{base_id}:{identity}")
        by_method[base_id][identity] = row
    output_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for method_id, rows in sorted(by_method.items()):
        for fold in range(5):
            train_ids = sorted(identity for identity in targets if folds[identity] != fold
                               and rows.get(identity, {}).get("status") == "predicted")
            test_ids = sorted(identity for identity in targets if folds[identity] == fold)
            raw_values: dict[str, float] = {}
            for identity, row in rows.items():
                if row.get("status") == "predicted":
                    value = float(row.get("predicted_seconds", "nan"))
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"invalid_raw_saved_prediction:{method_id}:{identity}")
                    raw_values[identity] = value
            affine_pairs = [(raw_values[i], float(targets[i]["target_seconds"])) for i in train_ids]
            log_pairs = [(math.log1p(raw_values[i]), math.log1p(float(targets[i]["target_seconds"])))
                         for i in train_ids]
            affine = _line_fit(affine_pairs)
            log_affine = _line_fit(log_pairs)
            fit_rows.extend([
                {"base_method_id": method_id, "outer_fold": fold, "variant": "outer_train_affine",
                 "fit_rows": len(train_ids), "fit_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
                 "intercept": affine[0], "slope": affine[1], "fit_scope": "successful outer-train raw predictions only"},
                {"base_method_id": method_id, "outer_fold": fold, "variant": "outer_train_log_affine",
                 "fit_rows": len(train_ids), "fit_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
                 "intercept": log_affine[0], "slope": log_affine[1], "fit_scope": "successful outer-train raw predictions only"},
            ])
            for identity in test_ids:
                target = targets[identity]
                source_row = rows.get(identity, {})
                raw = raw_values.get(identity)
                status = source_row.get("status", "unavailable")
                reason = source_row.get("terminal_reason", "raw_prediction_not_available")
                for variant in ("raw", "outer_train_affine", "outer_train_log_affine"):
                    prediction: float | str = ""
                    out_status, out_reason = status, reason
                    clock = source_row.get("method_output_clock", "") if variant == "raw" else "calibrated_observed_service_seconds"
                    if raw is not None:
                        if variant == "raw":
                            prediction, out_status, out_reason = raw, "predicted", ""
                        elif variant == "outer_train_affine":
                            prediction, out_status, out_reason = _apply_affine(affine, raw)
                            if prediction is None:
                                prediction = ""
                        else:
                            prediction, out_status, out_reason = _apply_log_affine(log_affine, raw)
                            if prediction is None:
                                prediction = ""
                    if out_status == "predicted" and (
                            not isinstance(prediction, (int, float))
                            or not math.isfinite(float(prediction))):
                        prediction, out_status, out_reason = "", "overflow", "nonfinite_prediction_guard"
                    output_rows.append({"canonical_observation_id": identity,
                        "base_method_id": method_id,
                        "method_id": f"{method_id}__{variant}",
                        "variant": variant, "outer_fold": fold, "source_id": target["source_id"],
                        "evaluation_target_clock": "archived_observed_service_execution_time",
                        "method_output_clock": clock, "actual_seconds": target["target_seconds"],
                        "predicted_seconds": prediction, "status": out_status, "terminal_reason": out_reason})
    if not output_rows:
        raise ValueError("no_raw_analytical_attempts_found")
    _write_csv(out / "attempts.csv", output_rows)
    _write_csv(out / "fold_fits.csv", fit_rows)
    runtime_environment = {
        "environment_id": "qpu_saved_raw_calibration_stdlib",
        "python_version": platform.python_version(),
        "python_implementation": sys.implementation.name,
        "third_party_dependencies": [],
        "caller_metadata": dict(environment or {}),
    }
    receipt = {"status": "complete", "method_family": method_family,
        "assigned_rows": len(targets), "attempt_rows": len(output_rows),
        "input_sha256": {key: _sha256(path) for key, path in paths.items()},
        "adapter_sha256": _sha256(Path(__file__)), "fit_policy": "raw saved predictions and actual labels from outer-train only",
        "prediction_policy": "outer-test raw values are transformed only after fold fit; raw rows retained",
        "environment": runtime_environment, "execution_performed": True}
    (out / "run_manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(output_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refuse_empty_output_csv")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
