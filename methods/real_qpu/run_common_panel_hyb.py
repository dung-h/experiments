#!/usr/bin/env python3
"""Run the frozen Hyb-HANAS component and calibration variants on 7,350 rows."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "artifacts/real_qpu/common_panel"
OUT = PANEL / "hyb"
CONTRACT_PATH = ROOT / "benchmark_v1/protocol/common_panel_completion.json"
HYB_CONTRACT_PATH = ROOT / "benchmark_v1/protocol/analytical_extension.json"
EXT_ROOT = ROOT / "artifacts/benchmark_v3/real_qpu/analytical_extension"
COMPONENTS = EXT_ROOT / "components.csv"
COMPONENT_MANIFEST = EXT_ROOT / "run_manifest.json"
REP_ROOT = ROOT / "artifacts/benchmark_v3/real_qpu/compiled_unified"
REPRESENTATIONS = REP_ROOT / "representation_rows.csv"
GRAPH_HASHES = REP_ROOT / "graph_file_hashes.json"
GRAPH_ROOT = REP_ROOT / "graphs"
TARGETS = PANEL / "targets.csv"
SPLITS = PANEL / "outer_splits.csv"
INNER = PANEL / "inner_splits.csv"
MAX_LOG = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
MIN_LOG = math.log(float.fromhex("0x0.0000000000001p-1022"))
ATTEMPT_FIELDS = ["canonical_observation_id", "source_id", "backend", "outer_fold", "group_id",
    "method_id", "evaluation_target_clock", "method_output_clock", "actual_seconds", "shots",
    "predicted_seconds", "status", "terminal_reason", "selected_alpha", "input_feature_digest",
    "compiled_graph_digest", "logical_input_tier"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(100_000_000)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
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


def digest_ids(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def canonical_backend(value: str) -> str:
    return "ibm_" + value[len("ibmq_"):] if value.startswith("ibmq_") else value


def feature(method_id: str, row: dict) -> tuple[float, ...] | None:
    shots = int(row["shots"])
    if shots <= 0:
        return None
    if method_id == "hyb_nominal_r2_log_cost_ridge_v1":
        value = row["nominal_log_cost"]
        return None if value == "" else (math.asinh(float(value)), math.log1p(shots))
    if method_id == "hyb_kyoto_composite_r2_log_cost_ridge_v1":
        value = row["composite_log_cost"]
        return None if value == "" else (math.asinh(float(value)), math.log1p(shots))
    if method_id == "hyb_kyoto_composite_r2_gate_time_ridge_v1":
        value = row["composite_gate_seconds"]
        if value == "" or not math.isfinite(float(value)) or float(value) <= 0:
            return None
        return (math.asinh(math.log(float(value))), math.log1p(shots))
    raise KeyError(method_id)


def balanced_mae(ids: list[str], predictions: dict[str, float], target: dict[str, float],
                 sources: dict[str, str]) -> float:
    errors: dict[str, list[float]] = defaultdict(list)
    for identity in ids:
        errors[sources[identity]].append(abs(target[identity] - predictions[identity]))
    if not errors:
        raise RuntimeError("empty_source_balanced_score")
    return statistics.fmean(statistics.fmean(values) for values in errors.values())


def metric(rows: list[dict]) -> dict:
    good = [row for row in rows if row["status"] == "predicted"]
    y = np.asarray([float(row["actual_seconds"]) for row in good], dtype=float)
    p = np.asarray([float(row["predicted_seconds"]) for row in good], dtype=float)
    errors = np.abs(y - p)
    r2 = None if len(y) < 2 or np.ptp(y) == 0 else float(1 - np.square(y - p).sum() / np.square(y - y.mean()).sum())
    return {"assigned_rows": len(rows), "predicted_rows": len(good),
        "coverage": len(good) / len(rows) if rows else None,
        "mae_seconds": float(errors.mean()) if len(good) else None,
        "medae_seconds": float(np.median(errors)) if len(good) else None,
        "r2_seconds": r2,
        "log1p_mae": float(np.mean(np.abs(np.log1p(y) - np.log1p(np.maximum(p, 0))))) if len(good) else None,
        "p90_absolute_error_seconds": float(np.quantile(errors, .90)) if len(good) else None,
        "p99_absolute_error_seconds": float(np.quantile(errors, .99)) if len(good) else None,
        "max_absolute_error_seconds": float(errors.max()) if len(good) else None,
        "negative_prediction_count": int((p < 0).sum())}


def load_and_audit_inputs():
    completion = json.loads(CONTRACT_PATH.read_text())
    for name, pin in completion["input_pins"].items():
        path = ROOT / pin["path"]
        if not path.is_file() or sha(path) != pin["sha256"]:
            raise RuntimeError(f"common_panel_hyb_frozen_input_pin_mismatch:{name}")
    if completion["input_pins"]["hyb_components"]["sha256"] != sha(COMPONENTS):
        raise RuntimeError("common_panel_hyb_component_pin_mismatch")
    if completion["input_pins"]["hyb_component_receipt"]["sha256"] != sha(COMPONENT_MANIFEST):
        raise RuntimeError("common_panel_hyb_component_manifest_pin_mismatch")
    if completion["input_pins"]["hyb_contract"]["sha256"] != sha(HYB_CONTRACT_PATH):
        raise RuntimeError("common_panel_hyb_formula_contract_pin_mismatch")
    parent = json.loads(HYB_CONTRACT_PATH.read_text())
    receipt = json.loads(COMPONENT_MANIFEST.read_text())
    if (receipt.get("status") != "complete" or receipt.get("components_sha256") != sha(COMPONENTS)
            or receipt.get("protocol_sha256") != sha(HYB_CONTRACT_PATH)
            or parent.get("input_pins").get("representation_rows", {}).get("sha256") != sha(REPRESENTATIONS)
            or parent.get("input_pins").get("graph_hashes", {}).get("sha256") != sha(GRAPH_HASHES)):
        raise RuntimeError("archived_analytical_component_authority_invalid")

    targets = read(TARGETS)
    target = {row["canonical_observation_id"]: row for row in targets}
    panel_rows = read(PANEL / "panel.csv")
    panel = {row["canonical_observation_id"]: row for row in panel_rows}
    components_all = read(COMPONENTS)
    components = {row["canonical_row_id"]: row for row in components_all}
    representation_rows = read(REPRESENTATIONS)
    representations = {row["canonical_row_id"]: row for row in representation_rows}
    outer_rows = read(SPLITS)
    outer = {row["canonical_observation_id"]: row for row in outer_rows}
    inner_rows = read(INNER)
    if any(len(m) != len(rows) for m, rows in ((target, targets), (panel, panel_rows),
            (components, components_all), (representations, representation_rows), (outer, outer_rows))):
        raise RuntimeError("duplicate_identity_in_common_panel_hyb_inputs")
    ids = set(panel)
    if len(ids) != 7350 or any(not ids <= set(table) for table in (target, components, representations, outer)):
        raise RuntimeError("common_panel_hyb_join_denominator_mismatch")
    if {row["canonical_observation_id"] for row in targets} != ids or set(outer) != ids:
        raise RuntimeError("common_panel_hyb_target_or_split_identity_mismatch")

    graph_hashes = json.loads(GRAPH_HASHES.read_text())
    verified_graph_files: set[str] = set()
    representation_shot_mismatches: list[str] = []
    for identity in sorted(ids):
        t, p, c, r = target[identity], panel[identity], components[identity], representations[identity]
        if (t["source_id"] != p["source_id"] or canonical_backend(t["backend"]) != c["backend"]
                or canonical_backend(t["backend"]) != r["backend_canonical"]
                or t["source_id"] != c["source_id"] or int(float(t["shots"])) != int(c["shots"])
                or c["graph_digest"] != r["graph_input_digest"]
                or c["graph_file"] != r["graph_file"]
                or c["graph_status"] != r["graph_status"]):
            raise RuntimeError(f"compiled_component_target_identity_mismatch:{identity}")
        if int(float(t["shots"])) != int(r["shots"]):
            representation_shot_mismatches.append(identity)
        if c["graph_status"] == "available":
            graph_name = c["graph_file"]
            expected_graph_sha = graph_hashes.get(graph_name)
            graph_path = (GRAPH_ROOT / graph_name).resolve()
            if (not expected_graph_sha or c["graph_sha256"] != expected_graph_sha
                    or graph_root_escape(graph_path) or not graph_path.is_file()):
                raise RuntimeError(f"compiled_graph_hash_manifest_mismatch:{identity}")
            if graph_name not in verified_graph_files:
                if sha(graph_path) != expected_graph_sha:
                    raise RuntimeError(f"compiled_graph_file_hash_mismatch:{identity}")
                verified_graph_files.add(graph_name)
        for key in ("nominal_log_cost", "composite_log_cost", "nominal_gate_seconds", "composite_gate_seconds"):
            if c[key] != "" and not math.isfinite(float(c[key])):
                raise RuntimeError(f"nonfinite_hyb_component:{key}:{identity}")
        seconds = float(t["target_seconds"])
        if not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError(f"invalid_archived_target:{identity}")

    folds = {identity: int(row["outer_fold"]) for identity, row in outer.items()}
    groups = {identity: row["group_id"] for identity, row in outer.items()}
    inner = {}
    for row in inner_rows:
        key = (int(row["outer_fold"]), row["canonical_observation_id"])
        if key in inner:
            raise RuntimeError(f"duplicate_inner_split_assignment:{key}")
        inner[key] = row
    for fold in range(5):
        train_ids = {identity for identity in ids if folds[identity] != fold}
        test_ids = {identity for identity in ids if folds[identity] == fold}
        assignments = {identity: inner[(fold, identity)] for identity in train_ids if (fold, identity) in inner}
        if set(assignments) != train_ids:
            raise RuntimeError(f"inner_split_identity_incomplete:{fold}")
        if {groups[i] for i in train_ids} & {groups[i] for i in test_ids}:
            raise RuntimeError(f"outer_group_overlap:{fold}")
        for held in range(4):
            val_ids = {i for i, row in assignments.items() if int(row["inner_fold"]) == held}
            fit_ids = train_ids - val_ids
            if not val_ids or not fit_ids or {groups[i] for i in val_ids} & {groups[i] for i in fit_ids}:
                raise RuntimeError(f"inner_group_or_empty_partition:{fold}:{held}")
    source_counts = {source: sum(row["source_id"] == source for row in panel_rows)
                     for source in sorted({row["source_id"] for row in panel_rows})}
    if source_counts != {"mali_real_qpu": 340, "qonductor_single_circuit_ibm": 3065, "qpack_mcp": 3945}:
        raise RuntimeError("common_panel_hyb_source_denominator_mismatch")
    return {"completion": completion, "parent_contract": parent, "target": target, "panel": panel,
            "components": components, "representations": representations, "outer": outer,
            "inner_rows": inner_rows, "folds": folds, "groups": groups,
            "graph_files_verified": len(verified_graph_files), "ids": sorted(ids),
            "representation_shot_mismatches": representation_shot_mismatches}


def graph_root_escape(path: Path) -> bool:
    return GRAPH_ROOT.resolve() not in path.parents


def base_attempt(identity: str, method: dict, inputs: dict, status: str, pred=None, reason="", alpha="") -> dict:
    target, panel, component = inputs["target"][identity], inputs["panel"][identity], inputs["components"][identity]
    return {"canonical_observation_id": identity, "source_id": target["source_id"],
        "backend": target["backend"], "outer_fold": inputs["folds"][identity],
        "group_id": inputs["groups"][identity], "method_id": method["method_id"],
        "evaluation_target_clock": "archived_observed_service_execution_time",
        "method_output_clock": method["method_output_clock"],
        "actual_seconds": float(target["target_seconds"]), "shots": int(float(target["shots"])),
        "predicted_seconds": "" if pred is None else float(pred), "status": status,
        "terminal_reason": reason, "selected_alpha": alpha,
        "input_feature_digest": hashlib.sha256(json.dumps({key: component.get(key) for key in
            ("nominal_log_cost", "composite_log_cost", "nominal_gate_seconds", "composite_gate_seconds", "shots")},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "compiled_graph_digest": component["graph_digest"], "logical_input_tier": target["logical_input_tier"]}


def ridge_fold(method: dict, fold: int, inputs: dict, rows: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    ids = inputs["ids"]
    target = {identity: float(inputs["target"][identity]["target_seconds"]) for identity in ids}
    source = {identity: inputs["target"][identity]["source_id"] for identity in ids}
    outer_train = sorted(identity for identity in ids if inputs["folds"][identity] != fold)
    test = sorted(identity for identity in ids if inputs["folds"][identity] == fold)
    assignments = {row["canonical_observation_id"]: row for row in inputs["inner_rows"]
                   if int(row["outer_fold"]) == fold}
    if set(assignments) != set(outer_train):
        raise RuntimeError(f"inner_assignment_does_not_match_outer_train:{fold}")
    per_alpha: dict[float, list[float]] = {float(a): [] for a in inputs["parent_contract"]["ridge"]["alphas"]}
    valid_inner = True
    for held in range(4):
        val = sorted(identity for identity in outer_train if int(assignments[identity]["inner_fold"]) == held
                     and feature(method["method_id"], rows[identity]) is not None)
        fit = sorted(identity for identity in outer_train if int(assignments[identity]["inner_fold"]) != held
                     and feature(method["method_id"], rows[identity]) is not None)
        if not fit or not val or ({inputs["groups"][i] for i in fit} & {inputs["groups"][i] for i in val}):
            valid_inner = False
            break
        x_fit = np.asarray([feature(method["method_id"], rows[i]) for i in fit], dtype=float)
        y_fit = np.log1p([target[i] for i in fit])
        x_val = np.asarray([feature(method["method_id"], rows[i]) for i in val], dtype=float)
        y_val = {identity: target[identity] for identity in val}
        for alpha in per_alpha:
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            estimator.fit(x_fit, y_fit)
            estimates = np.maximum(0.0, np.expm1(estimator.predict(x_val)))
            by_source: dict[str, list[float]] = defaultdict(list)
            for identity, predicted in zip(val, estimates):
                if not math.isfinite(float(predicted)):
                    raise RuntimeError(f"nonfinite_inner_prediction:{method['method_id']}:{fold}:{held}")
                by_source[source[identity]].append(abs(y_val[identity] - float(predicted)))
            per_alpha[alpha].append(statistics.fmean(statistics.fmean(x) for x in by_source.values()))

    if not valid_inner or any(len(scores) != 4 for scores in per_alpha.values()):
        return ([base_attempt(identity, method, inputs, "unavailable", reason="inner_fold_partition_unusable")
                 for identity in test], [{"outer_fold": fold, "method_id": method["method_id"],
                 "status": "unavailable", "selected_alpha": "", "fit_rows": 0,
                 "inner_source_balanced_mae_seconds": ""}])
    alpha = min(per_alpha, key=lambda a: (statistics.fmean(per_alpha[a]), a))
    eligible = sorted(identity for identity in outer_train if feature(method["method_id"], rows[identity]) is not None)
    if not eligible:
        return ([base_attempt(identity, method, inputs, "unavailable", reason="no_eligible_outer_train_rows")
                 for identity in test], [])
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    estimator.fit(np.asarray([feature(method["method_id"], rows[i]) for i in eligible], dtype=float),
                  np.log1p([target[i] for i in eligible]))
    attempts, test_feature_ids = [], []
    for identity in test:
        inputs_for_row = feature(method["method_id"], rows[identity])
        if inputs_for_row is None:
            component = rows[identity]
            reason = (component.get("nominal_log_reason") or component.get("composite_log_reason")
                      or component.get("composite_gate_reason") or "feature_unavailable")
            attempts.append(base_attempt(identity, method, inputs, "unavailable", reason=reason, alpha=alpha))
            continue
        logpred = float(estimator.predict(np.asarray(inputs_for_row).reshape(1, -1))[0])
        if not math.isfinite(logpred):
            attempts.append(base_attempt(identity, method, inputs, "failed", reason="nonfinite_inverse_log1p_prediction", alpha=alpha))
        elif logpred > MAX_LOG:
            attempts.append(base_attempt(identity, method, inputs, "overflow", reason="inverse_log1p_overflow", alpha=alpha))
        else:
            prediction = max(0.0, math.expm1(logpred))
            attempts.append(base_attempt(identity, method, inputs, "predicted", pred=prediction, alpha=alpha))
        test_feature_ids.append(identity)
    fit_receipt = {"outer_fold": fold, "method_id": method["method_id"], "status": "fit",
        "selected_alpha": alpha, "eligible_fit_rows": len(eligible), "fit_ids_sha256": digest_ids(eligible),
        "test_feature_rows": len(test_feature_ids),
        "inner_source_balanced_mae_seconds": json.dumps(
            {str(a): statistics.fmean(per_alpha[a]) for a in sorted(per_alpha)}, sort_keys=True)}
    return attempts, [fit_receipt]


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        manifest_path = OUT / "run_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("refuse_to_overwrite_unreceipted_hyb_output")
        previous = json.loads(manifest_path.read_text())
        for name, expected in previous.get("output_hashes", {}).items():
            if sha(OUT / name) != expected:
                raise RuntimeError(f"existing_hyb_output_hash_mismatch:{name}")
        raise RuntimeError("hyb_run_already_complete_use_validated_existing_outputs")
    inputs = load_and_audit_inputs()
    hyb_methods = [row for row in inputs["parent_contract"]["methods"] if row["implementation"] == "hyb"]
    if len(hyb_methods) != 7:
        raise RuntimeError("hyb_method_list_does_not_match_frozen_contract")
    direct = [method for method in hyb_methods if method["transform"] in {"raw_exp", "shot_exp"}]
    ridge_methods = [method for method in hyb_methods if method["transform"].endswith("ridge")]
    attempts, fits = [], []
    rows = inputs["components"]
    for method in direct:
        for identity in inputs["ids"]:
            row = rows[identity]
            kind = "nominal" if method["calibration"] == "nominal" else "composite"
            value = row[f"{kind}_log_cost"]
            reason = row[f"{kind}_log_reason"] or row[f"{kind}_gate_reason"]
            if value == "":
                attempt = base_attempt(identity, method, inputs, "unavailable", reason=reason or "log_cost_unavailable")
            else:
                log_value = float(value)
                if method["transform"] == "shot_exp":
                    log_value += math.log(int(float(inputs["target"][identity]["shots"])))
                if log_value > MAX_LOG:
                    attempt = base_attempt(identity, method, inputs, "overflow", reason="float64_exponential_overflow")
                elif log_value < MIN_LOG:
                    attempt = base_attempt(identity, method, inputs, "underflow", reason="float64_exponential_underflow")
                else:
                    attempt = base_attempt(identity, method, inputs, "predicted", pred=math.exp(log_value))
            attempts.append(attempt)
    for method in ridge_methods:
        for fold in range(5):
            batch, fold_fit = ridge_fold(method, fold, inputs, rows)
            attempts.extend(batch)
            fits.extend(fold_fit)
            print(json.dumps({"method": method["method_id"], "fold": fold,
                              "assigned": len(batch), "terminal": dict(__import__("collections").Counter(r["status"] for r in batch))}),
                  flush=True)

    expected = 7350 * len(hyb_methods)
    ids_by_method = defaultdict(set)
    for row in attempts:
        ids_by_method[row["method_id"]].add(row["canonical_observation_id"])
    if len(attempts) != expected or any(values != set(inputs["ids"]) for values in ids_by_method.values()):
        raise RuntimeError("hyb_attempt_envelope_not_7350_per_method")
    attempt_by_method = defaultdict(list)
    for row in attempts:
        attempt_by_method[row["method_id"]].append(row)
    metrics = []
    for method in hyb_methods:
        subset = attempt_by_method[method["method_id"]]
        metric_row = {"method_id": method["method_id"], "reader_label": method["reader_label"],
            "fidelity_class": method["fidelity_class"], "evaluation_target_clock": "archived_observed_service_execution_time",
            "method_output_clock": method["method_output_clock"], **metric(subset)}
        by_source = {}
        for source in sorted({row["source_id"] for row in subset}):
            by_source[source] = metric([row for row in subset if row["source_id"] == source])
        valid_source_mae = [row["mae_seconds"] for row in by_source.values() if row["mae_seconds"] is not None]
        metric_row["source_balanced_mae_seconds"] = statistics.fmean(valid_source_mae) if valid_source_mae else None
        metrics.append(metric_row)

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(OUT / "attempts.csv", attempts, ATTEMPT_FIELDS)
    atomic_csv(OUT / "fold_fits.csv", fits, ["outer_fold", "method_id", "status", "selected_alpha", "eligible_fit_rows",
        "fit_ids_sha256", "test_feature_rows", "inner_source_balanced_mae_seconds"])
    atomic_csv(OUT / "metrics.csv", metrics, list(metrics[0]))
    input_paths = [CONTRACT_PATH, HYB_CONTRACT_PATH, PANEL / "manifest.json", PANEL / "panel.csv", TARGETS,
        SPLITS, INNER, COMPONENTS, COMPONENT_MANIFEST, REPRESENTATIONS, GRAPH_HASHES,
        ROOT / "benchmark_v1/scripts/run_common_panel_hyb.py"]
    output_hashes = {name: sha(OUT / name) for name in ("attempts.csv", "fold_fits.csv", "metrics.csv")}
    terminal_counts = {method["method_id"]: dict(__import__("collections").Counter(
        row["status"] for row in attempt_by_method[method["method_id"]])) for method in hyb_methods}
    manifest = {"status": "complete", "method_family": "Hyb-HANAS time-cost components",
        "assigned_rows_per_method": 7350, "method_count": 7, "attempt_rows": len(attempts),
        "graph_files_rehashed": inputs["graph_files_verified"],
        "terminal_counts_by_method": terminal_counts,
        "compiled_representation_shot_metadata": {"mismatch_rows": len(inputs["representation_shot_mismatches"]),
            "mismatch_ids_sha256": digest_ids(inputs["representation_shot_mismatches"]),
            "component_shots_match_common_targets": True, "cost_features_use_component_shots": True},
        "selection": "lowest mean source-balanced MAE on four frozen inner folds; tie smaller Ridge alpha",
        "fit_scope": "inner-fit for alpha selection; complete eligible outer-train for final fit",
        "backend_alias_rule": "normalize archived ibmq_* labels to pinned Qiskit canonical ibm_* labels; no hardware is substituted",
        "source_id_or_backend_features": False, "test_labels_used_for_fit_or_selection": False,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "scikit_learn": __import__("sklearn").__version__,
        "thread_limit": 1, "input_hashes": {path.relative_to(ROOT).as_posix(): sha(path) for path in input_paths},
        "output_hashes": output_hashes, "input_graph_component_join": "PASS",
        "panel_identity_sha256": digest_ids(inputs["ids"]), "created_unix": __import__("time").time()}
    atomic_json(OUT / "run_manifest.json", manifest)
    return {"status": "complete", "attempt_rows": len(attempts), "metrics": metrics,
            "manifest": str(OUT / "run_manifest.json")}


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        print(json.dumps(run(), indent=2), flush=True)
