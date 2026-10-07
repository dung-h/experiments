#!/usr/bin/env python3
"""Replay frozen common-panel evidence and build reader-facing tables.

This CPU-only reporting tool reads only files inside the selected package. It
does not fit, infer, time a simulator, or access the development checkout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "artifacts/real_qpu/common_panel"
INPUT = PANEL / "inputs"
PRED = PANEL / "predictions"
OUT = PANEL / "analysis"
PROTOCOL = ROOT / "benchmark/protocol/common_panel_reporting.json"
SOURCE_SIM = ROOT / "artifacts/benchmark_v3/results/two_domain_recovery_reader_v1/simulator/method_comparison.csv"
REAL_TABLE = ROOT / "artifacts/results/real_qpu/method_comparison.csv"
SIM_TABLE = ROOT / "artifacts/results/simulator/method_comparison.csv"
SEEDS = (42, 1234, 31415)
GLOBAL_FIELDS = [f"global_{i:02d}" for i in range(51)]
EXPECTED_SOURCES = {"mali_real_qpu": 340, "qonductor_single_circuit_ibm": 3065, "qpack_mcp": 3945}
FOCUS = (
    "mali_logical_graph", "mali_global_mlp", "qonductor_feature_polynomial_common_panel",
    "qcre_shot_scaled_schedule_seconds__outer_train_affine",
    "hyb_nominal_r2_log_cost_ridge_v1", "hyb_kyoto_composite_r2_log_cost_ridge_v1",
    "hyb_kyoto_composite_r2_gate_time_ridge_v1",
    "scholten_unified_nominal_throughput__outer_train_affine",
)
PAIRS = (
    ("mali_logical_graph", "mali_global_mlp"),
    ("mali_logical_graph", "qonductor_feature_polynomial_common_panel"),
    ("mali_global_mlp", "qonductor_feature_polynomial_common_panel"),
    ("hyb_kyoto_composite_r2_log_cost_ridge_v1", "hyb_kyoto_composite_r2_gate_time_ridge_v1"),
    ("hyb_nominal_r2_log_cost_ridge_v1", "hyb_kyoto_composite_r2_log_cost_ridge_v1"),
    ("qcre_shot_scaled_schedule_seconds__outer_train_affine", "mali_logical_graph"),
    ("scholten_unified_nominal_throughput__outer_train_affine", "mali_logical_graph"),
)
ANALYSIS_FILES = (
    "input_support.csv", "analysis_metrics.csv", "analysis_pairs.csv", "dataset_profile.csv",
    "tail_cases.csv", "fold_feature_masks.csv", "run_manifest.json",
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ids_sha(ids: list[str] | set[str]) -> str:
    return sha_bytes("\n".join(sorted(ids)).encode("utf-8"))


def stable_hash(value: Any) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"required_input_missing:{path.relative_to(ROOT)}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


PINNED_SOURCE_PATHS = {
    "artifacts/real_qpu/common_panel/manifest.json": INPUT / "panel_manifest.json",
    "artifacts/real_qpu/common_panel/panel.csv": INPUT / "panel.csv",
    "artifacts/real_qpu/common_panel/outer_splits.csv": INPUT / "outer_splits.csv",
    "artifacts/real_qpu/common_panel/inner_splits.csv": INPUT / "inner_splits.csv",
    "artifacts/real_qpu/common_panel/targets.csv": INPUT / "targets.csv",
    "artifacts/real_qpu/common_panel/mali_feature_ledger.csv": INPUT / "mali_feature_ledger.csv",
    "artifacts/real_qpu/logical_inputs/feature_manifest.json": INPUT / "feature_manifest.json",
    "artifacts/real_qpu/logical_inputs/feature_attempts.csv": INPUT / "feature_attempts.csv",
    "artifacts/real_qpu/common_panel/qonductor_polynomial/run_manifest.json": PRED / "polynomial/run_manifest.json",
    "artifacts/real_qpu/common_panel/analytical/run_manifest.json": PRED / "analytical/run_manifest.json",
    "artifacts/real_qpu/common_panel/hyb/run_manifest.json": PRED / "hyb/run_manifest.json",
    "artifacts/real_qpu/common_panel/mali/final_validation.json": PRED / "neural/final_validation.json",
    "artifacts/real_qpu/common_panel/mali/fold_0/transform.json": PRED / "neural/fold_0_transform.json",
    "artifacts/real_qpu/common_panel/mali/fold_1/transform.json": PRED / "neural/fold_1_transform.json",
    "artifacts/real_qpu/common_panel/mali/fold_2/transform.json": PRED / "neural/fold_2_transform.json",
    "artifacts/real_qpu/common_panel/mali/fold_3/transform.json": PRED / "neural/fold_3_transform.json",
    "artifacts/real_qpu/common_panel/mali/fold_4/transform.json": PRED / "neural/fold_4_transform.json",
    "artifacts/real_qpu/common_panel/mali/final_validation.json": PRED / "neural/final_validation.json",
    "artifacts/real_qpu/common_panel/summary/run_manifest.json": PANEL / "summary/run_manifest.json",
    "artifacts/real_qpu/common_panel/mali/input_index.csv": INPUT / "input_index.csv",
    "artifacts/real_qpu/common_panel/mali/input_index_manifest.json": INPUT / "input_index_manifest.json",
    "artifacts/benchmark_v3/real_qpu/analytical_extension/components.csv": ROOT / "artifacts/benchmark_v3/real_qpu/analytical_extension/components.csv",
    "artifacts/benchmark_v3/real_qpu/analytical_extension/run_manifest.json": ROOT / "artifacts/benchmark_v3/real_qpu/analytical_extension/run_manifest.json",
    "benchmark_v1/protocol/analytical_extension.json": ROOT / "benchmark_v1/protocol/analytical_extension.json",
    "artifacts/real_qpu/mali_full_features_large_batch/stop_record.json": ROOT / "artifacts/real_qpu/mali_full_features_large_batch/stop_record.json",
    "artifacts/real_qpu/logical_inputs/logical_outer_splits.csv": ROOT / "artifacts/real_qpu/logical_inputs/logical_outer_splits.csv",
    "artifacts/real_qpu/logical_inputs/model_input_groups.csv": ROOT / "artifacts/real_qpu/logical_inputs/model_input_groups.csv",
    "benchmark_v1/scripts/freeze_real_qpu_panel.py": ROOT / "benchmark_v1/scripts/freeze_real_qpu_panel.py",
    "benchmark_v1/scripts/aggregate_common_panel.py": ROOT / "benchmark_v1/scripts/aggregate_common_panel.py",
    "benchmark_v1/protocol/common_panel_completion.json": ROOT / "benchmark_v1/protocol/common_panel_completion.json",
    "benchmark_v1/protocol/real_qpu_common_panel.json": ROOT / "benchmark_v1/protocol/real_qpu_common_panel.json",
    "benchmark_v1/scripts/run_mali_full_features.py": ROOT / "benchmark_v1/scripts/run_mali_full_features.py",
    "benchmark_v1/scripts/run_common_panel_mali.py": ROOT / "benchmark_v1/scripts/run_common_panel_mali.py",
    "work/repo_finalization/selected_export/artifacts/benchmark_v3/results/reader_method_dataset_table.csv": ROOT / "artifacts/benchmark_v3/results/reader_method_dataset_table.csv",
    "work/repo_finalization/selected_export/artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/aggregate_manifest.json": ROOT / "artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/aggregate_manifest.json",
    "work/repo_finalization/selected_export/artifacts/benchmark_v3/simulator/maestro_cpu_component_optimizer_off/evaluation_001/method_metrics.csv": ROOT / "artifacts/benchmark_v3/simulator/maestro_cpu_component_optimizer_off/evaluation_001/method_metrics.csv",
    "work/repo_finalization/selected_export/artifacts/benchmark_v3/results/two_domain_recovery_reader_v1/simulator/method_comparison.csv": ROOT / "artifacts/benchmark_v3/results/two_domain_recovery_reader_v1/simulator/method_comparison.csv",
}


def _assert_sha(path: Path, expected: str, authority: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"pinned_source_missing:{authority}:{path.relative_to(ROOT)}")
    actual = sha_file(path)
    if actual != expected:
        raise RuntimeError(f"pinned_source_hash_mismatch:{authority}:{path.relative_to(ROOT)}:{actual}:{expected}")


def verify_authority_pins() -> None:
    """Validate the frozen panel, component, summary and final-cell receipts."""
    completion_path = ROOT / "benchmark_v1/protocol/common_panel_completion.json"
    panel_protocol_path = ROOT / "benchmark_v1/protocol/real_qpu_common_panel.json"
    completion = read_json(completion_path)
    panel_protocol = read_json(panel_protocol_path)
    reporting_contract = read_json(PROTOCOL)
    panel_manifest_path = INPUT / "panel_manifest.json"
    if sha_file(panel_manifest_path) != completion["input_pins"]["panel_manifest"]["sha256"]:
        raise RuntimeError("panel_manifest_protocol_pin_mismatch")
    if panel_protocol.get("panel_manifest_sha256") != sha_file(panel_manifest_path):
        raise RuntimeError("panel_manifest_design_protocol_pin_mismatch")
    for rel, expected in reporting_contract.get("input_hashes", {}).items():
        path = PINNED_SOURCE_PATHS.get(rel)
        if path is None:
            raise RuntimeError(f"unmapped_reporting_contract_input:{rel}")
        _assert_sha(path, expected, "reporting_contract_input")
    for key, pin in completion["input_pins"].items():
        path = PINNED_SOURCE_PATHS.get(pin["path"])
        if path is None:
            raise RuntimeError(f"unmapped_completion_input_pin:{key}:{pin['path']}")
        _assert_sha(path, pin["sha256"], f"completion:{key}")
    panel_manifest = read_json(panel_manifest_path)
    for rel, expected in panel_manifest.get("input_hashes", {}).items():
        path = PINNED_SOURCE_PATHS.get(rel)
        if path is None:
            raise RuntimeError(f"unmapped_panel_manifest_input:{rel}")
        _assert_sha(path, expected, "panel_manifest_input")
    for name, expected in panel_manifest.get("output_hashes", {}).items():
        _assert_sha(INPUT / name, expected, "panel_manifest_output")
    input_index_manifest = read_json(INPUT / "input_index_manifest.json")
    index_path = INPUT / "input_index.csv"
    _assert_sha(index_path, input_index_manifest["input_index_sha256"], "input_index_manifest")
    index_rows = read_csv(index_path)
    if len(index_rows) != 7350 or len({r["canonical_observation_id"] for r in index_rows}) != 7350:
        raise RuntimeError("input_index_identity_count_mismatch")
    for key, expected in input_index_manifest.get("input_pins", {}).items():
        pin = completion["input_pins"].get(key)
        if pin is None or pin["sha256"] != expected:
            raise RuntimeError(f"input_index_completion_pin_mismatch:{key}")
        path = PINNED_SOURCE_PATHS.get(pin["path"])
        if path is None:
            raise RuntimeError(f"input_index_source_unmapped:{key}:{pin['path']}")
        _assert_sha(path, expected, f"input_index:{key}")
    summary_path = PANEL / "summary/run_manifest.json"
    summary_manifest = read_json(summary_path)
    for rel, expected in summary_manifest.get("input_hashes", {}).items():
        path = PINNED_SOURCE_PATHS.get(rel)
        if path is None:
            raise RuntimeError(f"unmapped_summary_input:{rel}")
        _assert_sha(path, expected, "summary_input")
    for name, expected in summary_manifest.get("output_hashes", {}).items():
        _assert_sha(PANEL / "summary" / name, expected, "summary_output")
    status_path = PANEL / "run_status.json"
    status = read_json(status_path)
    evidence = status.get("completion_evidence", {})
    if status.get("status") != "complete" or status.get("stage") != "final_validation":
        raise RuntimeError("derived_run_status_not_terminal")
    for path_key, sha_key in (("final_validation_receipt", "final_validation_receipt_sha256"),
                              ("summary_manifest", "summary_manifest_sha256")):
        rel = evidence.get(path_key, "")
        actual_path = PINNED_SOURCE_PATHS.get(rel)
        if actual_path is None:
            raise RuntimeError(f"run_status_receipt_path_unmapped:{path_key}:{rel}")
        _assert_sha(actual_path, evidence.get(sha_key, ""), f"run_status:{path_key}")
    receipt = read_json(PRED / "neural/final_validation.json")
    if receipt.get("status") != "PASS" or receipt.get("cell_count") != 30 or evidence.get("validated_cells") != 30 or evidence.get("failed_cells") != 0:
        raise RuntimeError("neural_terminal_receipt_not_30_pass")


def verify_summary_metrics(rows_by_method: dict[str, dict[str, dict[str, Any]]]) -> None:
    """Independently recompute reader-panel metrics from frozen OOF rows."""
    summary = {r["method_id"]: r for r in read_csv(PANEL / "summary/method_metrics.csv")}
    numeric_fields = {
        "MAE_seconds": "mae_seconds",
        "MedAE_seconds": "medae_seconds",
        "R2_seconds": "r2_seconds",
        "log1p_MAE": "log1p_mae",
        "p90_absolute_error_seconds": "p90_absolute_error_seconds",
        "p99_absolute_error_seconds": "p99_absolute_error_seconds",
        "max_absolute_error_seconds": "max_absolute_error_seconds",
    }
    for method in FOCUS:
        stored = summary.get(method)
        if stored is None:
            raise RuntimeError(f"summary_method_missing:{method}")
        attempts = list(rows_by_method[method].values())
        metric = _metric(attempts, 7350)
        if int(stored["assigned_rows"]) != 7350 or int(stored["scored_rows"]) != metric["scored_rows"]:
            raise RuntimeError(f"summary_denominator_mismatch:{method}")
        if stored["scored_id_set_sha256"] != metric["scored_id_set_sha256"]:
            raise RuntimeError(f"summary_scored_identity_mismatch:{method}")
        for actual_key, stored_key in numeric_fields.items():
            left, right = metric[actual_key], stored[stored_key]
            if left == "" or right == "":
                if left != right:
                    raise RuntimeError(f"summary_metric_availability_mismatch:{method}:{actual_key}")
            elif not math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-10):
                raise RuntimeError(f"summary_metric_value_mismatch:{method}:{actual_key}:{left}:{right}")


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    import io
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _csv_value(row.get(k, "")) for k in fields})
    return out.getvalue().encode("utf-8")


def _csv_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    return value


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def hash_ids_list(ids: list[str]) -> str:
    return stable_hash(sorted(ids))


def global_tensor(raw: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    meta = transform["global"]
    selected = raw.astype(np.float32, copy=False)[np.asarray(meta["retained_indices"], dtype=np.int64)]
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    variable = np.asarray(meta["variable_mask"], dtype=bool)
    result = np.zeros_like(selected, dtype=np.float32)
    result[variable] = (selected[variable] - mean[variable]) / std[variable]
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite_global_tensor")
    result[result == 0] = np.float32(0.0)
    return result


def _header(h: Any, name: str, dtype: str, shape: tuple[int, ...]) -> None:
    body = json.dumps({"name": name, "dtype": dtype, "shape": list(shape), "order": "C", "byte_order": "little"},
                      sort_keys=True, separators=(",", ":")).encode("ascii")
    h.update(len(body).to_bytes(8, "little"))
    h.update(body)


def signature_one(name: str, arr: np.ndarray) -> str:
    value = np.ascontiguousarray(arr)
    if value.dtype.kind == "f":
        if not np.isfinite(value).all():
            raise RuntimeError("nonfinite_tensor_signature_input")
        value = value.copy()
        value[value == 0] = 0.0
    dtype = value.dtype.newbyteorder("<")
    value = value.astype(dtype, copy=False)
    h = hashlib.sha256(b"qre-model-input-signature-v1\0")
    _header(h, name, dtype.str, tuple(value.shape))
    h.update(value.tobytes(order="C"))
    return h.hexdigest()


def _dense_node_chunk(z: Any, start: int, stop: int, transform: dict[str, Any]) -> np.ndarray:
    node_type = z["node_type"][start:stop]
    rows = len(node_type)
    x = np.zeros((rows, 178), dtype=np.float32)
    x[np.arange(rows), node_type] = 1.0
    wire0, wire1 = z["wire0"][start:stop], z["wire1"][start:stop]
    ix = np.flatnonzero(wire0 != 255)
    x[ix, 46 + wire0[ix]] = 1.0
    ix = np.flatnonzero(wire1 != 255)
    x[ix, 46 + wire1[ix]] = 1.0
    x[:, 173] = z["t1_0"][start:stop]
    x[:, 174] = z["t2_0"][start:stop]
    x[:, 175] = z["t1_1"][start:stop]
    x[:, 176] = z["t2_1"][start:stop]
    x[:, 177] = z["node_index"][start:stop]
    meta = transform["node"]
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    variable = np.asarray(meta["variable_mask"], dtype=bool)
    x[:, variable] = (x[:, variable] - mean[variable]) / std[variable]
    x[:, ~variable] = 0.0
    x[x == 0] = 0.0
    if not np.isfinite(x).all():
        raise RuntimeError("nonfinite_graph_tensor")
    return x


def graph_signature(path: Path, transform: dict[str, Any]) -> str:
    h = hashlib.sha256(b"qre-model-input-signature-v1\0")
    with np.load(path, allow_pickle=False) as z:
        nodes = int(len(z["node_type"]))
        edge = z["edge_index"].astype("<i8", copy=False)
        _header(h, "x", "<f4", (nodes, 178))
        for start in range(0, nodes, 65_536):
            chunk = np.ascontiguousarray(_dense_node_chunk(z, start, min(nodes, start + 65_536), transform), dtype="<f4")
            h.update(chunk.tobytes(order="C"))
        _header(h, "edge_index", "<i8", tuple(edge.shape))
        h.update(np.ascontiguousarray(edge).tobytes(order="C"))
    return h.hexdigest()


def _metric(rows: list[dict[str, Any]], assigned: int) -> dict[str, Any]:
    good = [r for r in rows if r["status"] == "predicted" and r["predicted_seconds"] != ""]
    if not good:
        return {"scored_rows": 0, "coverage": 0.0, "scored_id_set_sha256": ids_sha(set()),
                "MAE_seconds": "", "MedAE_seconds": "", "R2_seconds": "", "log1p_MAE": "",
                "p90_absolute_error_seconds": "", "p99_absolute_error_seconds": "",
                "max_absolute_error_seconds": "", "negative_prediction_count": 0,
                "source_balanced_MAE_seconds": "", "source_balanced_status": "no_scored_rows"}
    actual = np.asarray([float(r["actual_seconds"]) for r in good], dtype=np.float64)
    pred = np.asarray([float(r["predicted_seconds"]) for r in good], dtype=np.float64)
    error = np.abs(actual - pred)
    total = float(np.sum((actual - actual.mean()) ** 2))
    r2 = "" if len(actual) < 2 or total == 0 else float(1 - np.sum((actual - pred) ** 2) / total)
    by_source: dict[str, list[float]] = defaultdict(list)
    for r, e in zip(good, error):
        by_source[r["source_id"]].append(float(e))
    sb = [statistics.fmean(v) for s, v in sorted(by_source.items()) if s in EXPECTED_SOURCES]
    return {"scored_rows": len(good), "coverage": len(good) / assigned if assigned else 0.0,
            "scored_id_set_sha256": ids_sha({r["canonical_observation_id"] for r in good}),
            "MAE_seconds": statistics.fmean(map(float, error)), "MedAE_seconds": float(np.median(error)),
            "R2_seconds": r2, "log1p_MAE": float(np.mean(np.abs(np.log1p(actual) - np.log1p(np.maximum(pred, 0))))),
            "p90_absolute_error_seconds": float(np.quantile(error, .90)),
            "p99_absolute_error_seconds": float(np.quantile(error, .99)),
            "max_absolute_error_seconds": float(np.max(error)), "negative_prediction_count": int((pred < 0).sum()),
            "source_balanced_MAE_seconds": statistics.fmean(sb) if len(sb) == 3 else "",
            "source_balanced_status": "all_sources_present" if len(sb) == 3 else "unavailable_source_without_scored_rows"}


def _population(ids: set[str], label: str, rows_by_method: dict[str, dict[str, dict[str, Any]]], base: dict[str, dict[str, str]],
                scope_type: str = "all", scope_value: str = "all") -> list[dict[str, Any]]:
    result = []
    for method in FOCUS:
        attempts = [rows_by_method[method][i] for i in sorted(ids) if i in rows_by_method[method]]
        metric = _metric(attempts, len(ids))
        groups = {base[i]["group_id"] for i in ids}
        scored_ids = {r["canonical_observation_id"] for r in attempts if r["status"] == "predicted" and r["predicted_seconds"] != ""}
        scored_groups = {base[i]["group_id"] for i in scored_ids}
        status_counts = Counter(r["status"] for r in attempts)
        result.append({"population": label, "slice_type": scope_type, "slice_value": scope_value,
                       "method_id": method, "assigned_rows": len(ids), "groups_assigned": len(groups),
                       "groups_scored": len(scored_groups), "status_counts_json": json.dumps(dict(sorted(status_counts.items())), separators=(",", ":")), **metric})
    return result


def _bootstrap_pair(a: list[dict[str, Any]], b: list[dict[str, Any]], common: list[str],
                    base: dict[str, dict[str, str]], rng: np.random.Generator) -> dict[str, Any]:
    by_a = {r["canonical_observation_id"]: r for r in a}
    by_b = {r["canonical_observation_id"]: r for r in b}
    groups = sorted({base[i]["group_id"] for i in common})
    gi = {g: n for n, g in enumerate(groups)}
    delta = np.zeros(len(groups), dtype=np.float64)
    count = np.zeros(len(groups), dtype=np.float64)
    sdelta = {s: np.zeros(len(groups), dtype=np.float64) for s in EXPECTED_SOURCES}
    scount = {s: np.zeros(len(groups), dtype=np.float64) for s in EXPECTED_SOURCES}
    observed = []
    for identity in common:
        ra, rb = by_a[identity], by_b[identity]
        e = abs(float(ra["actual_seconds"]) - float(ra["predicted_seconds"])) - abs(float(rb["actual_seconds"]) - float(rb["predicted_seconds"]))
        idx = gi[base[identity]["group_id"]]
        src = base[identity]["source_id"]
        delta[idx] += e; count[idx] += 1
        sdelta[src][idx] += e; scount[src][idx] += 1
        observed.append(e)
    draw = rng.integers(0, len(groups), size=(10_000, len(groups)))
    weights = np.zeros((10_000, len(groups)), dtype=np.float64)
    np.add.at(weights, (np.repeat(np.arange(10_000), len(groups)), draw.reshape(-1)), 1.0)
    denom = weights @ count
    pooled = (weights @ delta)[denom > 0] / denom[denom > 0]
    source_reps = []
    for src in EXPECTED_SOURCES:
        n = weights @ scount[src]
        numer = weights @ sdelta[src]
        source_reps.append(np.divide(numer, n, out=np.full_like(numer, np.nan), where=n > 0))
    source_matrix = np.vstack(source_reps).T
    valid_source = np.isfinite(source_matrix).all(axis=1)
    sb = source_matrix[valid_source].mean(axis=1)
    observed_sources = []
    for src in EXPECTED_SOURCES:
        vals = [v for identity, v in zip(common, observed) if base[identity]["source_id"] == src]
        if vals:
            observed_sources.append(statistics.fmean(vals))
    return {"observed_delta_a_minus_b_seconds": statistics.fmean(observed),
            "bootstrap_mean_delta_a_minus_b_seconds": float(np.mean(pooled)),
            "bootstrap_ci95_low_seconds": float(np.quantile(pooled, .025)) if len(pooled) >= 9500 else "",
            "bootstrap_ci95_high_seconds": float(np.quantile(pooled, .975)) if len(pooled) >= 9500 else "",
            "bootstrap_valid_replicates": int(len(pooled)),
            "observed_source_balanced_delta_seconds": statistics.fmean(observed_sources) if len(observed_sources) == 3 else "",
            "bootstrap_source_balanced_mean_delta_seconds": float(np.mean(sb)) if len(sb) else "",
            "bootstrap_source_balanced_ci95_low_seconds": float(np.quantile(sb, .025)) if len(sb) >= 9500 else "",
            "bootstrap_source_balanced_ci95_high_seconds": float(np.quantile(sb, .975)) if len(sb) >= 9500 else "",
            "bootstrap_source_balanced_valid_replicates": int(len(sb)), "distinct_groups": len(groups)}


def _rows_by_method() -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    panel_rows = read_csv(PANEL / "inputs/panel.csv")
    target_rows = read_csv(PANEL / "inputs/targets.csv")
    outer_rows = read_csv(PANEL / "inputs/outer_splits.csv")
    panel = {r["canonical_observation_id"]: r for r in panel_rows}
    targets = {r["canonical_observation_id"]: r for r in target_rows}
    outer = {r["canonical_observation_id"]: r for r in outer_rows}
    if len(panel) != 7350 or set(panel) != set(targets) or set(panel) != set(outer):
        raise RuntimeError("panel_target_split_identity_mismatch")
    rows: dict[str, dict[str, dict[str, Any]]] = {m: {} for m in FOCUS}
    base = {}
    for identity, p in panel.items():
        t, o = targets[identity], outer[identity]
        if p["source_id"] != t["source_id"] or p["logical_input_tier"] != t["logical_input_tier"] or p["group_id"] != o["group_id"]:
            raise RuntimeError(f"panel_join_disagreement:{identity}")
        base[identity] = {**p, **t, "outer_fold": o["outer_fold"], "group_id": o["group_id"]}
    neural_manifest = read_json(PRED / "neural/final_validation.json")
    if neural_manifest.get("cell_count") != 30:
        raise RuntimeError("neural_final_receipt_cell_count_not_30")
    receipt_cells = {(int(c["fold"]), c["method"], int(c["seed"])): c for c in neural_manifest["cells"]}
    seed_rows: dict[tuple[str, str], dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for fold in range(5):
        transform_path = PRED / f"neural/fold_{fold}_transform.json"
        transform = read_json(transform_path)
        if len(transform["global"]["retained_indices"]) != 40 or sum(transform["global"]["variable_mask"]) != 40 or sum(transform["node"]["variable_mask"]) != 167:
            raise RuntimeError(f"fold_transform_mask_changed:{fold}")
        test_ids = sorted(i for i in panel if int(base[i]["outer_fold"]) == fold)
        fit_val = [r for r in read_csv(PANEL / "inputs/inner_splits.csv") if int(r["outer_fold"]) == fold]
        fit_ids = sorted(r["canonical_observation_id"] for r in fit_val if int(r["inner_fold"]) != 0)
        val_ids = sorted(r["canonical_observation_id"] for r in fit_val if int(r["inner_fold"]) == 0)
        if hash_ids_list(fit_ids) != transform["fit_ids_sha256"] or hash_ids_list(fit_ids) != transform["global"]["fit_ids_sha256"]:
            raise RuntimeError(f"transform_fit_id_hash_mismatch:{fold}")
        for method in ("mali_logical_graph", "mali_global_mlp"):
            for seed in SEEDS:
                rel = f"fold_{fold}/{method}/seed_{seed}"
                folder = PRED / "neural" / rel
                cm = read_json(folder / "cell_manifest.json")
                rc = receipt_cells[(fold, method, seed)]
                if cm.get("status") != "complete" or cm.get("output_hashes", {}).get("predictions.csv") != sha_file(folder / "predictions.csv") or cm.get("output_hashes", {}).get("test_failures.csv") != sha_file(folder / "test_failures.csv"):
                    raise RuntimeError(f"neural_cell_output_hash_mismatch:{rel}")
                if cm.get("identity", {}).get("transform_sha256") != stable_hash(transform) or rc.get("sha256") != sha_file(folder / "cell_manifest.json"):
                    raise RuntimeError(f"neural_cell_receipt_or_transform_mismatch:{rel}")
                if (cm.get("fit_ids_sha256"), cm.get("validation_ids_sha256"), cm.get("test_ids_sha256")) != (hash_ids_list(fit_ids), hash_ids_list(val_ids), hash_ids_list(test_ids)):
                    raise RuntimeError(f"neural_cell_partition_hash_mismatch:{rel}")
                predictions = read_csv(folder / "predictions.csv")
                failures = read_csv(folder / "test_failures.csv")
                if len(predictions) + len(failures) != len(test_ids) or int(cm.get("n_test_assigned", -1)) != len(test_ids):
                    raise RuntimeError(f"neural_cell_test_denominator_mismatch:{rel}")
                seen = set()
                for r in predictions:
                    i = r["canonical_observation_id"]
                    if i in seen or i not in test_ids or int(r["outer_fold"]) != fold or r["method_id"] != method or int(r["seed"]) != seed:
                        raise RuntimeError(f"neural_prediction_identity_mismatch:{rel}:{i}")
                    seen.add(i)
                    if r["status"] == "predicted":
                        seed_rows[(method, i)][seed] = float(r["predicted_seconds"])
                for r in failures:
                    i = r["canonical_observation_id"]
                    if i in seen or i not in test_ids:
                        raise RuntimeError(f"neural_failure_identity_mismatch:{rel}:{i}")
                    seen.add(i)
                if seen != set(test_ids):
                    raise RuntimeError(f"neural_test_ids_incomplete:{rel}")
    for method in ("mali_logical_graph", "mali_global_mlp"):
        for i in panel:
            vals = seed_rows[(method, i)]
            if set(vals) == set(SEEDS):
                rows[method][i] = _attempt(i, panel, targets, outer, float(np.median([vals[s] for s in SEEDS])), "predicted", "predicted_observed_service_seconds")
            else:
                rows[method][i] = _attempt(i, panel, targets, outer, "", "incomplete_seed_predictions", "predicted_observed_service_seconds")
    for r in read_csv(PRED / "polynomial/predictions.csv"):
        i = r["canonical_observation_id"]
        rows["qonductor_feature_polynomial_common_panel"][i] = _attempt(i, panel, targets, outer, r["predicted_seconds"] if r["status"] == "predicted" else "", r["status"], "predicted_observed_service_seconds")
    for r in read_csv(PRED / "analytical/attempts.csv"):
        if r["method_id"] == "qcre_shot_scaled_schedule_seconds__outer_train_affine" or r["method_id"] == "scholten_unified_nominal_throughput__outer_train_affine":
            rows[r["method_id"]][r["canonical_observation_id"]] = _attempt(r["canonical_observation_id"], panel, targets, outer,
                r["predicted_seconds"], r["status"], r["method_output_clock"])
    for r in read_csv(PRED / "hyb/attempts.csv"):
        if r["method_id"] in FOCUS:
            rows[r["method_id"]][r["canonical_observation_id"]] = _attempt(r["canonical_observation_id"], panel, targets, outer,
                r["predicted_seconds"], r["status"], r["method_output_clock"])
    for method in FOCUS:
        if set(rows[method]) != set(panel):
            raise RuntimeError(f"focus_method_assigned_identity_mismatch:{method}:{len(rows[method])}")
    return rows, base, panel, targets


def _attempt(identity: str, panel: dict[str, dict[str, str]], targets: dict[str, dict[str, str]], outer: dict[str, dict[str, str]],
             prediction: Any, status: str, clock: str) -> dict[str, Any]:
    value = ""
    if status == "predicted" and prediction not in (None, ""):
        num = float(prediction)
        if math.isfinite(num):
            value = num
        else:
            status = "failed_nonfinite_prediction"
    elif status == "predicted":
        status = "failed_missing_prediction"
    return {"canonical_observation_id": identity, "source_id": panel[identity]["source_id"],
            "group_id": outer[identity]["group_id"], "actual_seconds": float(targets[identity]["target_seconds"]),
            "predicted_seconds": value, "status": status, "method_output_clock": clock}


def build_support(base: dict[str, dict[str, str]], panel: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], list[dict[str, Any]]]:
    inner = read_csv(PANEL / "inputs/inner_splits.csv")
    outer_fold = {r["canonical_observation_id"]: int(r["outer_fold"]) for r in read_csv(PANEL / "inputs/outer_splits.csv")}
    inner_by_fold: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in inner:
        inner_by_fold[int(row["outer_fold"])].append(row)
    ledger = {r["canonical_row_id"]: r for r in read_csv(PANEL / "inputs/mali_feature_ledger.csv")}
    if set(ledger) != set(panel):
        raise RuntimeError("feature_ledger_identity_mismatch")
    support_rows = []
    sig_by_id: dict[str, dict[str, str]] = {}
    masks = []
    graph_cache: dict[tuple[str, str], str] = {}
    for fold in range(5):
        transform_path = PRED / f"neural/fold_{fold}_transform.json"
        transform = read_json(transform_path)
        transform_sha = sha_file(transform_path)
        for group, name, width, retained, variable in (
            ("global", "global", 51, set(transform["global"]["retained_indices"]), set(i for i,v in zip(transform["global"]["retained_indices"], transform["global"]["variable_mask"]) if v)),
            ("node", "node", 178, set(range(178)), set(i for i,v in enumerate(transform["node"]["variable_mask"]) if v)),
        ):
            for idx in range(width):
                masks.append({"outer_fold": fold, "feature_group": group, "feature_index": idx,
                              "retained": int(idx in retained), "variable": int(idx in variable),
                              "transform_sha256": transform_sha})
        refs_fit = {r["canonical_observation_id"] for r in inner_by_fold[fold] if int(r["inner_fold"]) != 0}
        refs_val = {r["canonical_observation_id"] for r in inner_by_fold[fold] if int(r["inner_fold"]) == 0}
        refs_test = {i for i in panel if outer_fold[i] == fold}
        if refs_fit & refs_val or refs_fit & refs_test or refs_val & refs_test:
            raise RuntimeError(f"fold_partition_overlap:{fold}")
        refs = {"fit": refs_fit, "fit_plus_validation": refs_fit | refs_val}
        sigs: dict[str, dict[str,str]] = {kind:{} for kind in ("global_tensor", "graph_tensor", "graph_plus_global_tensor")}
        for identity, row in panel.items():
            raw = np.asarray([float(ledger[identity][name]) for name in GLOBAL_FIELDS], dtype=np.float32)
            gs = signature_one("global_features", global_tensor(raw, transform))
            p = row["graph_file"]
            graph_path = INPUT / p
            graph_digest = row["graph_sha256"]
            if sha_file(graph_path) != graph_digest or ledger[identity]["graph_artifact_sha256"] != graph_digest:
                raise RuntimeError(f"graph_payload_hash_mismatch:{identity}")
            key = (graph_digest, transform_sha)
            if key not in graph_cache:
                graph_cache[key] = graph_signature(graph_path, transform)
            ds = graph_cache[key]
            combo = sha_bytes(b"qre-combined-signature-v1\0" + bytes.fromhex(gs) + bytes.fromhex(ds))
            sigs["global_tensor"][identity] = gs
            sigs["graph_tensor"][identity] = ds
            sigs["graph_plus_global_tensor"][identity] = combo
        reference_counts = {
            kind: {refname: Counter(idmap[x] for x in ref_ids) for refname, ref_ids in refs.items()}
            for kind, idmap in sigs.items()
        }
        for identity in sorted(refs_test):
            signature_values = {}
            counts = {}
            for kind, idmap in sigs.items():
                value = idmap[identity]
                signature_values[kind] = value
                for refname, ref_ids in refs.items():
                    nmatch = reference_counts[kind][refname].get(value, 0)
                    counts[f"{kind}_matches_{refname}_rows"] = nmatch
                    counts[f"{kind}_seen_{refname}"] = int(nmatch > 0)
            sig_by_id[identity] = {**signature_values, **counts}
            support_rows.append({"canonical_observation_id": identity, "outer_fold": fold,
                                 "source_id": base[identity]["source_id"], "logical_input_tier": base[identity]["logical_input_tier"],
                                 "group_id": base[identity]["group_id"], "transform_sha256": transform_sha, **signature_values, **counts})
    if len(support_rows) != 7350:
        raise RuntimeError(f"support_replay_incomplete:{len(support_rows)}")
    return support_rows, sig_by_id, masks


def _dataset_profile(base: dict[str, dict[str, str]], ledger: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str,str], list[str]] = defaultdict(list)
    for i, r in base.items(): groups[(r["source_id"],r["logical_input_tier"])].append(i)
    scopes: list[tuple[str,str,list[str]]] = [("all","all",list(base))]
    scopes += [("source",s,[i for i,r in base.items() if r["source_id"]==s]) for s in sorted(EXPECTED_SOURCES)]
    scopes += [("logical_input_tier",t,[i for i,r in base.items() if r["logical_input_tier"]==t]) for t in sorted({r["logical_input_tier"] for r in base.values()})]
    scopes += [("source_x_logical_input_tier",f"{s}|{t}",ids) for (s,t),ids in sorted(groups.items())]
    out=[]
    for typ,val,ids in scopes:
        widths=np.asarray([float(ledger[i]["global_44"]) for i in ids],dtype=float)
        depths=np.asarray([float(ledger[i]["global_45"]) for i in ids],dtype=float)
        shots=np.asarray([float(base[i]["shots"]) for i in ids],dtype=float)
        y=np.asarray([float(base[i]["target_seconds"]) for i in ids],dtype=float)
        out.append({"slice_type":typ,"slice_value":val,"assigned_rows":len(ids),"groups":len({base[i]["group_id"] for i in ids}),
                    "logical_width_median":float(np.median(widths)),"logical_width_min":float(widths.min()),"logical_width_max":float(widths.max()),
                    "logical_depth_median":float(np.median(depths)),"logical_depth_min":float(depths.min()),"logical_depth_max":float(depths.max()),
                    "shots_median":float(np.median(shots)),"shots_min":float(shots.min()),"shots_max":float(shots.max()),
                    "target_seconds_median":float(np.median(y)),"target_seconds_p90":float(np.quantile(y,.9)),"target_seconds_min":float(y.min()),"target_seconds_max":float(y.max())})
    return out


def build_analysis() -> dict[str, bytes]:
    verify_authority_pins()
    protocol = read_json(PROTOCOL)
    rows_by_method, base, panel, targets = _rows_by_method()
    verify_summary_metrics(rows_by_method)
    source_counts = Counter(r["source_id"] for r in panel.values())
    if dict(source_counts) != EXPECTED_SOURCES:
        raise RuntimeError(f"source_counts_changed:{dict(source_counts)}")
    candidate = {i for i,r in panel.items() if r["logical_input_tier"] == "candidate_recipe_sensitivity_only"}
    qualified = set(panel) - candidate
    if len(qualified) != 4515 or len({panel[i]["group_id"] for i in qualified}) != 165:
        raise RuntimeError("qualified_input_sensitivity_denominator_changed")
    support_rows, support_by_id, masks = build_support(base, panel)
    ledger = {r["canonical_row_id"]: r for r in read_csv(PANEL / "inputs/mali_feature_ledger.csv")}
    populations = [("common_panel", set(panel)), ("qualified_input_evaluation_subset_models_fit_on_7350", qualified)]
    metric_rows=[]
    for label, ids in populations:
        metric_rows += _population(ids,label,rows_by_method,base)
        slicers={
            "source":lambda i:base[i]["source_id"],
            "logical_input_tier":lambda i:base[i]["logical_input_tier"],
            "source_x_logical_input_tier":lambda i:f"{base[i]['source_id']}|{base[i]['logical_input_tier']}",
            "backend":lambda i:base[i]["backend"],
            "exact_integer_shots":lambda i:str(int(float(base[i]["shots"])) if float(base[i]["shots"]).is_integer() else (_ for _ in ()).throw(RuntimeError("shots_not_integer"))),
            "outer_fold":lambda i:base[i]["outer_fold"],
        }
        for slice_type, fn in slicers.items():
            vals=sorted({fn(i) for i in ids})
            for val in vals:
                slice_ids={i for i in ids if fn(i)==val}
                metric_rows += _population(slice_ids,label,rows_by_method,base,slice_type,val)
        for kind in ("global_tensor","graph_tensor","graph_plus_global_tensor"):
            for ref in ("fit","fit_plus_validation"):
                for status in (0,1):
                    for source in ["all",*sorted(EXPECTED_SOURCES)]:
                        slice_ids={i for i in ids if support_by_id[i][f"{kind}_seen_{ref}"]==status and (source=="all" or base[i]["source_id"]==source)}
                        if slice_ids:
                            metric_rows += _population(slice_ids,label,rows_by_method,base,
                                f"signature_{kind}_{ref}_{'seen' if status else 'unseen'}"+("_by_source" if source!="all" else ""),source)
    # Paired comparisons use only pair-successful IDs and keep the sign A minus B.
    pair_rows=[]; rng=np.random.Generator(np.random.PCG64(42))
    for label, ids in populations:
        for a_id,b_id in PAIRS:
            aa=rows_by_method[a_id]; bb=rows_by_method[b_id]
            common=sorted(i for i in ids if aa[i]["status"]=="predicted" and bb[i]["status"]=="predicted" and aa[i]["predicted_seconds"]!="" and bb[i]["predicted_seconds"]!="")
            ma=_metric([aa[i] for i in common],len(common)); mb=_metric([bb[i] for i in common],len(common))
            distinct_groups=len({base[i]["group_id"] for i in common})
            if distinct_groups >= 20:
                boot=_bootstrap_pair([aa[i] for i in common],[bb[i] for i in common],common,base,rng)
                boot["interval_status"]="exploratory_group_bootstrap_conditional_on_oof"
            else:
                boot={"distinct_groups":distinct_groups,"interval_status":"insufficient_groups_for_interval"}
            pair_rows.append({"population":label,"comparison":"focused_pair","method_a":a_id,"method_b":b_id,
                "sign":"method_a_minus_method_b","assigned_rows":len(ids),"common_predicted_rows":len(common),
                "intersection_coverage":len(common)/len(ids) if ids else 0,"common_id_set_sha256":ids_sha(set(common)),
                "mae_a_seconds":ma["MAE_seconds"],"r2_a_seconds":ma["R2_seconds"],"mae_b_seconds":mb["MAE_seconds"],"r2_b_seconds":mb["R2_seconds"],**boot})
        # Input-support paired rows: compare graph and MLP on exactly the same signature strata IDs.
        for kind in ("global_tensor","graph_tensor","graph_plus_global_tensor"):
            for ref in ("fit","fit_plus_validation"):
                for status in (0,1):
                    for source in ["all",*sorted(EXPECTED_SOURCES)]:
                        common=sorted(i for i in ids if support_by_id[i][f"{kind}_seen_{ref}"]==status and (source=="all" or base[i]["source_id"]==source)
                                      and rows_by_method["mali_logical_graph"][i]["status"]=="predicted" and rows_by_method["mali_global_mlp"][i]["status"]=="predicted")
                        if not common: continue
                        ma=_metric([rows_by_method["mali_logical_graph"][i] for i in common],len(common)); mb=_metric([rows_by_method["mali_global_mlp"][i] for i in common],len(common))
                        pair_rows.append({"population":label,"comparison":"input_support_same_ids","support_kind":kind,"reference_set":ref,
                            "support_status":"seen" if status else "unseen","source_scope":source,"method_a":"mali_logical_graph","method_b":"mali_global_mlp",
                            "sign":"method_a_minus_method_b","assigned_rows":len(ids),"common_predicted_rows":len(common),
                            "intersection_coverage":len(common)/len(ids),"common_id_set_sha256":ids_sha(set(common)),
                            "mae_a_seconds":ma["MAE_seconds"],"r2_a_seconds":ma["R2_seconds"],"mae_b_seconds":mb["MAE_seconds"],"r2_b_seconds":mb["R2_seconds"],
                            "observed_delta_a_minus_b_seconds":ma["MAE_seconds"]-mb["MAE_seconds"],"distinct_groups":len({base[i]["group_id"] for i in common}),
                            "interval_status":"descriptive_point_estimates_only"})
    # Top-20 error records remain diagnostics and never change assigned IDs.
    tails=[]
    seed_predictions: dict[tuple[int,str,int],dict[str,float]] = {}
    for method in ("mali_logical_graph","mali_global_mlp"):
        for fold in range(5):
            for seed in SEEDS:
                path=PRED/f"neural/fold_{fold}/{method}/seed_{seed}/predictions.csv"
                seed_predictions[(fold,method,seed)]={r["canonical_observation_id"]:float(r["predicted_seconds"])
                    for r in read_csv(path) if r["status"]=="predicted"}
    for label, ids in populations:
        for method in ("mali_logical_graph","mali_global_mlp","qonductor_feature_polynomial_common_panel"):
            scored=[rows_by_method[method][i] for i in ids if rows_by_method[method][i]["status"]=="predicted"]
            scored.sort(key=lambda r:(-abs(float(r["actual_seconds"])-float(r["predicted_seconds"])),r["canonical_observation_id"]))
            for rank,r in enumerate(scored[:20],1):
                i=r["canonical_observation_id"]
                spreads=[]
                if method in ("mali_logical_graph","mali_global_mlp"):
                    for seed in SEEDS:
                        fold=int(base[i]["outer_fold"]); vals=seed_predictions[(fold,method,seed)]
                        if i in vals: spreads.append(vals[i])
                tails.append({"population":label,"method_id":method,"rank_by_absolute_error":rank,
                    "canonical_observation_id":i,"source_id":base[i]["source_id"],"backend":base[i]["backend"],"shots":int(float(base[i]["shots"])),
                    "outer_fold":base[i]["outer_fold"],"logical_input_tier":base[i]["logical_input_tier"],"group_id":base[i]["group_id"],
                    "actual_seconds":r["actual_seconds"],"predicted_seconds":r["predicted_seconds"],
                    "absolute_error_seconds":abs(float(r["actual_seconds"])-float(r["predicted_seconds"])),
                    "seed_prediction_range_seconds":(max(spreads)-min(spreads)) if len(spreads)==3 else "",
                    "seed_prediction_mad_seconds":float(np.median(np.abs(np.asarray(spreads)-np.median(spreads)))) if len(spreads)==3 else "",
                    "diagnostic_only":1})
    profile=_dataset_profile(base,ledger)
    # Support file is a per-observation proof of what inputs were compared.
    support_fields=list(support_rows[0])
    metric_fields=list(metric_rows[0])
    pair_fields=sorted({k for r in pair_rows for k in r})
    tail_fields=list(tails[0])
    profile_fields=list(profile[0])
    mask_fields=list(masks[0])
    result={
      "input_support.csv":csv_bytes(support_rows,support_fields),
      "analysis_metrics.csv":csv_bytes(metric_rows,metric_fields),
      "analysis_pairs.csv":csv_bytes(pair_rows,pair_fields),
      "dataset_profile.csv":csv_bytes(profile,profile_fields),
      "tail_cases.csv":csv_bytes(tails,tail_fields),
      "fold_feature_masks.csv":csv_bytes(masks,mask_fields),
    }
    return result


def pasqal_quality_summary() -> dict[str, Any]:
    """Count analog cells once, preserving their separate timing-stage records."""
    folder = ROOT / "artifacts/benchmark_v1/pasqal_emu_mps_formula_crossed_pilot_v1r1_20260926"
    raw_path, qa_path = folder / "raw_records.jsonl", folder / "qa_final/qa.json"
    protocol_path = ROOT / "benchmark_v1/execution/manifests/pasqal_emu_mps_formula_crossed_pilot_v1.json"
    manifest, qa, protocol = read_json(folder / "run_manifest.json"), read_json(qa_path), read_json(protocol_path)
    _assert_sha(protocol_path, manifest["input_manifests"][0]["sha256"], "pasqal_input_contract")
    _assert_sha(raw_path, qa["raw_sha256"], "pasqal_terminal_qa")
    quality = protocol["quality"]
    reference_rule = re.search(r"Fidelity >=([0-9.]+)", quality["reference_validation"])
    if reference_rule is None:
        raise RuntimeError("pasqal_reference_threshold_not_declared")
    reference_min, candidate_min = float(reference_rule.group(1)), float(quality["candidate_threshold"])
    expected_ids = {f"{family}|{r}x{c}|chi{chi}"
                    for family in protocol["source_programs"]["families"]
                    for r, c in protocol["source_programs"]["layouts"]
                    for chi in protocol["simulator"]["max_bond_dimensions"]}
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    identities = set()
    for r in records:
        key = (r["row_id"], r["context_id"], r["session_id"])
        identity = (*key, r["stage"], r["repetition"])
        if identity in identities:
            raise RuntimeError("pasqal_duplicate_stage_record")
        identities.add(identity)
        groups[key].append(r)
    expected_count = len(expected_ids) * int(protocol["measurement"]["sessions"])
    if ({key[0] for key in groups} != expected_ids or len(groups) != expected_count
            or qa["status"] != "complete" or int(qa["complete_cells"]) != expected_count
            or int(qa["raw_rows"]) != len(records)):
        raise RuntimeError("pasqal_cell_identity_or_qa_mismatch")
    failed_cells = []
    for key, cell in sorted(groups.items()):
        expected_stages = {"network_build": 1, "first_execute": 1,
                           "warm_execute": int(protocol["measurement"]["timed_warm_repetitions_per_session"])}
        if dict(Counter(r["stage"] for r in cell)) != expected_stages:
            raise RuntimeError("pasqal_incomplete_timing_stages")
        passing = []
        for r in cell:
            candidate = float(r["quality"]["candidate_fidelity"])
            reference = float(r["quality"]["reference_fidelity_256_vs_512"])
            if (not math.isfinite(candidate) or not math.isfinite(reference)
                    or float(r["metadata"]["quality_threshold"]) != candidate_min):
                raise RuntimeError("pasqal_invalid_quality_evidence")
            passed = reference >= reference_min and candidate >= candidate_min
            if r["status"] != ("ok" if passed else "quality_failed"):
                raise RuntimeError("pasqal_quality_status_disagreement")
            passing.append(passed)
        if len(set(passing)) != 1:
            raise RuntimeError("pasqal_within_cell_quality_disagreement")
        if not all(passing):
            failed_cells.append(key[0])
    failed_records = sum(r["status"] == "quality_failed" for r in records)
    if failed_records != int(qa["quality_failed_rows"]):
        raise RuntimeError("pasqal_failed_record_count_disagreement")
    return {"assigned_cells": expected_count, "quality_pass_cells": expected_count - len(failed_cells),
            "quality_fail_cells": len(failed_cells), "quality_failed_cell_ids": failed_cells,
            "timing_stage_records": len(records), "quality_failed_stage_records": failed_records,
            "candidate_fidelity_threshold": candidate_min, "reference_fidelity_threshold": reference_min,
            "raw_path": str(raw_path.relative_to(ROOT)), "raw_sha256": sha_file(raw_path),
            "qa_path": str(qa_path.relative_to(ROOT)), "qa_sha256": sha_file(qa_path)}


def build_reader_tables(analysis_outputs: dict[str, bytes]) -> dict[str, bytes]:
    """Build a focused QPU comparison plus a context-qualified simulator catalogue."""
    metric_rows=list(csv.DictReader(analysis_outputs["analysis_metrics.csv"].decode().splitlines()))
    full={r["method_id"]:r for r in metric_rows if r["population"]=="common_panel" and r["slice_type"]=="all"}
    base_path=PANEL/"inputs/panel.csv"; panel={r["canonical_observation_id"]:r for r in read_csv(base_path)}
    target={r["canonical_observation_id"]:r for r in read_csv(PANEL/"inputs/targets.csv")}
    # These are the actual output clocks of the eight rows in the common panel.
    clocks={"mali_logical_graph":"predicted_observed_service_seconds","mali_global_mlp":"predicted_observed_service_seconds",
      "qonductor_feature_polynomial_common_panel":"predicted_observed_service_seconds",
      "qcre_shot_scaled_schedule_seconds__outer_train_affine":"calibrated_observed_service_seconds",
      "hyb_nominal_r2_log_cost_ridge_v1":"calibrated_observed_service_seconds",
      "hyb_kyoto_composite_r2_log_cost_ridge_v1":"calibrated_observed_service_seconds",
      "hyb_kyoto_composite_r2_gate_time_ridge_v1":"calibrated_observed_service_seconds",
      "scholten_unified_nominal_throughput__outer_train_affine":"calibrated_observed_service_seconds"}
    labels={"mali_logical_graph":"Ma–Li-style full-feature graph adaptation",
      "mali_global_mlp":"Matched Ma–Li-style global MLP (51 raw / 40 retained)",
      "qonductor_feature_polynomial_common_panel":"Qonductor-style unified polynomial adaptation",
      "qcre_shot_scaled_schedule_seconds__outer_train_affine":"QCRE shot-scaled schedule, outer-train affine",
      "hyb_nominal_r2_log_cost_ridge_v1":"Hyb-HANAS-style nominal log-cost ridge adaptation",
      "hyb_kyoto_composite_r2_log_cost_ridge_v1":"Hyb-HANAS-style Kyoto composite log-cost ridge adaptation",
      "hyb_kyoto_composite_r2_gate_time_ridge_v1":"Hyb-HANAS-style Kyoto composite gate-time ridge adaptation",
      "scholten_unified_nominal_throughput__outer_train_affine":"Scholten-style nominal throughput adaptation, outer-train affine"}
    fidelity={"mali_logical_graph":"architecture_adaptation","mali_global_mlp":"architecture_adaptation",
      "qonductor_feature_polynomial_common_panel":"adaptation","qcre_shot_scaled_schedule_seconds__outer_train_affine":"adaptation",
      "hyb_nominal_r2_log_cost_ridge_v1":"adaptation","hyb_kyoto_composite_r2_log_cost_ridge_v1":"adaptation",
      "hyb_kyoto_composite_r2_gate_time_ridge_v1":"adaptation","scholten_unified_nominal_throughput__outer_train_affine":"nominal_sensitivity_adaptation"}
    representations={
      "mali_logical_graph":"Logical-recipe directed DAG with 178 node fields and nominal-index T1/T2, plus 40 transformed globals from the 51-position raw schema",
      "mali_global_mlp":"51 raw logical-circuit global fields; 40 retained by the fit-only mask",
      "qonductor_feature_polynomial_common_panel":"Compiled/submitted CX/CZ/ECR counter (upstream swap), operand-stack depth, touched width, shots and circuit_count=1",
      "qcre_shot_scaled_schedule_seconds__outer_train_affine":"Compiled/submitted circuit, nominal instruction durations and shots; outer-train affine calibration",
      "hyb_nominal_r2_log_cost_ridge_v1":"Nominal r2 log-effective-cost and shots; outer-train log1p-target Ridge",
      "hyb_kyoto_composite_r2_log_cost_ridge_v1":"Kyoto composite r2 log-effective-cost and shots; outer-train log1p-target Ridge",
      "hyb_kyoto_composite_r2_gate_time_ridge_v1":"Kyoto composite log gate-time and shots; outer-train log1p-target Ridge",
      "scholten_unified_nominal_throughput__outer_train_affine":"Compiled quantum-wire depth, shots and nominal throughput; outer-train affine calibration"}
    out=[]
    for method in FOCUS:
      m=full[method]; out.append({"domain":"real_qpu","method_id":method,"reader_label":labels[method],"fidelity_class":fidelity[method],
        "context":"archived real-QPU observations; common grouped five-fold OOF panel","input_representation":representations[method],
        "evaluation_target_clock":"archived_observed_service_execution_time","method_output_clock":clocks[method],"population":"unified common panel",
        "split":"grouped five-fold OOF","assigned":m["assigned_rows"],"observed":len(target),"predicted":m["scored_rows"],"quality_pass":"","quality_fail":"",
        "failed":"","unavailable":int(m["assigned_rows"])-int(m["scored_rows"]),"coverage":m["coverage"],"MAE_seconds":m["MAE_seconds"],"R2_seconds":m["R2_seconds"],
        "tail_error_p99_seconds":m["p99_absolute_error_seconds"],"scored_id_set_sha256":m["scored_id_set_sha256"],"scope_note":"Common-panel adaptation; not necessarily paper-faithful reproduction."})
    # Preserve the deterministic timing references' raw outputs separately from
    # their outer-train calibrated predictions. They target a different clock.
    outer={r["canonical_observation_id"]:r for r in read_csv(PANEL/"inputs/outer_splits.csv")}
    analytic=read_csv(PRED/"analytical/attempts.csv")
    reference_ids=("qcre_snapshot_critical_path__raw","qcre_shot_scaled_schedule_seconds__raw",
                   "qiskit_estimate_duration_snapshot__raw","qiskit_estimate_duration_snapshot__outer_train_affine",
                   "scholten_unified_nominal_throughput__raw")
    ref_labels={"qcre_snapshot_critical_path__raw":"QCRE snapshot weighted critical-path schedule (raw)",
      "qcre_shot_scaled_schedule_seconds__raw":"QCRE snapshot schedule scaled by requested shots (raw)",
      "qiskit_estimate_duration_snapshot__raw":"Qiskit estimate_duration snapshot schedule (raw)",
      "qiskit_estimate_duration_snapshot__outer_train_affine":"Qiskit scheduled duration, outer-train affine diagnostic",
      "scholten_unified_nominal_throughput__raw":"Scholten-style nominal throughput, single-circuit adaptation (raw)"}
    ref_inputs={
      "qcre_snapshot_critical_path__raw":"Compiled/submitted quantum-wire critical path weighted by nominal instruction durations",
      "qcre_shot_scaled_schedule_seconds__raw":"Nominal duration-weighted critical path scaled by requested shots",
      "qiskit_estimate_duration_snapshot__raw":"Compiled/submitted circuit evaluated with the nominal Qiskit Target instruction-duration map",
      "qiskit_estimate_duration_snapshot__outer_train_affine":"Nominal Qiskit scheduled duration with outer-train affine service-time calibration",
      "scholten_unified_nominal_throughput__raw":"Compiled quantum-wire depth, requested shots and nominal throughput with its recorded definition retained"}
    refs=defaultdict(dict)
    for r in analytic:
        if r["method_id"] in reference_ids:
            i=r["canonical_observation_id"]
            refs[r["method_id"]][i]=_attempt(i,panel,target,outer,r["predicted_seconds"],r["status"],r["method_output_clock"])
    for mid in reference_ids:
        if set(refs[mid])!=set(panel): raise RuntimeError(f"reference_method_incomplete:{mid}")
        m=_metric(list(refs[mid].values()),len(panel))
        if "__outer_train_" in mid:
            ref_fidelity="adaptation"
        elif mid.startswith("qiskit_"):
            ref_fidelity="deterministic_scheduler_reference"
        elif mid.startswith("qcre_"):
            ref_fidelity="timing_aware_adaptation"
        else:
            ref_fidelity="nominal_sensitivity_adaptation"
        out.append({"domain":"real_qpu","method_id":mid,"reader_label":ref_labels[mid],"fidelity_class":ref_fidelity,
          "context":"archived real-QPU observations; common grouped five-fold OOF panel","input_representation":ref_inputs[mid],
          "evaluation_target_clock":"archived_observed_service_execution_time","method_output_clock":next(iter(refs[mid].values()))["method_output_clock"],
          "population":"unified common panel","split":"grouped five-fold OOF","assigned":len(panel),"observed":len(target),"predicted":m["scored_rows"],
          "quality_pass":"","quality_fail":"","failed":"","unavailable":len(panel)-m["scored_rows"],"coverage":m["coverage"],
          "MAE_seconds":m["MAE_seconds"],"R2_seconds":m["R2_seconds"],"tail_error_p99_seconds":m["p99_absolute_error_seconds"],
          "scored_id_set_sha256":m["scored_id_set_sha256"],"scope_note":"Raw schedule/cost is not observed service time; calibrated rows are a separate adaptation."})
    qpu_fields=list(out[0])
    # Retain existing simulator rows and correct two stale scope labels by creating a new semantic table.
    sim_rows=read_csv(SOURCE_SIM)
    for r in sim_rows:
      if r.get("method_id")=="family_conditioned" and r.get("population_id")=="sim_common_q16_v1_core_q2_q9_150_hashes":
        r["claim_boundary"]="Local CUDA-Q MPS joint runtime/quality adaptation; not the original approximation-threshold target."
      if r.get("method_id")=="family_residual_median_three_seeds":
        r["claim_boundary"]="Local fixed-chi runtime adaptation inspired by Family-Aware; original approximation-threshold route remains unavailable."
    # The earlier reader table claimed 900/900 scored rows for the joint MPS
    # ladder. The immutable C44 OOF artifact shows 900 assigned, 840 measured
    # targets and 60 technical target failures. Preserve those old rows as
    # historical, but build the current entries from the OOF/target artifact
    # and verify its published metrics before presenting them.
    joint_target = "mps_joint_rung_runtime_r9_v1"
    joint_methods = {"family_conditioned", "family_agnostic_ablation"}
    for row in sim_rows:
      if row.get("target_authority_id") == joint_target and row.get("method_id") in joint_methods:
        row["claim_boundary"] = (
          "Historical reader row superseded: it reported all 900 assigned hash-rung rows as scored. "
          "The current immutable OOF artifact has 840 measured target rows and 60 technical target failures."
        )
        row["comparison_role"] = "historical_superseded_not_ranked"
        row["result_status"] = "historical_metric_denominator_superseded"
        row["status_reason_counts_json"] = json.dumps({"superseded_by": "family_aware_joint_mps_c44_and_family_holdout"}, separators=(",", ":"))
    sim_fields=list(sim_rows[0])
    def simrow(**values: Any) -> dict[str, Any]:
        row={field:"" for field in sim_fields}
        row.update({"domain":"simulator","comparison_role":"diagnostic_only","result_kind":"prediction",
                    "expected_assigned_n":"","attempted_n":"","unattempted_n":"","predicted_n":"",
                    "unavailable_n":"","failed_n":"","pending_n":"","coverage":"",
                    "metric_n":"","mae_seconds":"","r2_seconds":"","quality_assessed_n":"",
                    "quality_pass_n":"","quality_fail_n":""})
        row.update(values)
        return row
    # Replace the old coverage-only Azizov entry with the scored common-core
    # adaptation. Keep its original coverage receipt as a historical route,
    # but make that distinction explicit in the new reader table.
    azizov_root=ROOT/"artifacts/benchmark_v3/simulator/azizov_common_core_adaptation"
    azizov_protocol=ROOT/"benchmark_v1/protocol/azizov_common_core_gnn_v1.json"
    azizov_metrics=read_csv(azizov_root/"method_metrics.csv")
    azizov_labels=read_csv(azizov_root/"labels_core.csv")
    azizov_hashes={r["source_sha256"] for r in azizov_labels}
    if len(azizov_hashes)!=150 or len(azizov_labels)!=162:
        raise RuntimeError("azizov_common_core_population_mismatch")
    azizov_rows=[r for r in azizov_metrics if r["method_family"]=="azizov_gnn_median_three_seeds" or r["method_family"].startswith("classical_")]
    if len(azizov_rows)!=18 or any(int(r["assigned_hashes"])!=150 or int(r["predicted_hashes"])!=150 for r in azizov_rows):
        raise RuntimeError("azizov_common_core_metric_coverage_mismatch")
    azizov_row_set=ids_sha(azizov_hashes)
    azizov_views={"source":"source-view","hybrid":"hybrid-view","transpiled":"transpiled-view"}
    for r in azizov_rows:
      family=r["method_family"]
      if family=="azizov_gnn_median_three_seeds":
        fidelity="adaptation"
        label=f"Azizov-style three-seed GNN median ({azizov_views[r['view']]})"
        role="primary_comparable"
      else:
        fidelity="common_baseline"
        algorithm={"classical_linear_regression":"Linear regression","classical_ridge":"Ridge",
          "classical_svr_rbf":"RBF-SVR","classical_random_forest":"Random forest",
          "classical_xgboost":"XGBoost"}[family]
        label=f"{algorithm} baseline ({azizov_views[r['view']]})"
        role="baseline"
      sim_rows.append(simrow(result_kind="prediction",task_id="azizov_common_core_adaptation",
        method_family=family,method_id=r["method_id"],reader_label=label,fidelity_class=fidelity,
        claim_boundary="Local Aer adaptation on 150 exact-QASM hashes (162 archive members), FakeSherbrooke noisy context and optimization level 1. This is not a paper-scale reproduction or a cross-engine comparison.",
        comparison_role=role,context_id="qiskit_aer_fake_sherbrooke_noisy_opt1",
        population_id="common_core_q2_q9_150_hashes",target_id="azizov_aer_warm_execution",
        evaluation_target_clock="warm_execution",method_output_clock="predicted_warm_execution",
        split_id="frozen_hash_grouped_five_fold_oof",result_status="completed_validated",
        expected_assigned_n="150",attempted_n="150",unattempted_n="0",predicted_n="150",
        unavailable_n="0",failed_n="0",pending_n="0",coverage="1",metric_n="150",
        row_set_sha256=azizov_row_set,mae_seconds=r["mae_seconds"],medae_seconds=r["medae_seconds"],
        log1p_mae_seconds=r["log1p_mae"],r2_seconds=r["r2_seconds"],
        p90_absolute_error_seconds=r["p90_absolute_error_seconds"],
        p99_absolute_error_seconds=r["p99_absolute_error_seconds"],
        max_absolute_error_seconds=r["max_absolute_error_seconds"],
        method_card_source_path=str(azizov_protocol.relative_to(ROOT)),
        method_card_source_sha256=sha_file(azizov_protocol),
        evidence_details_json=json.dumps({"view":r["view"],"seed_reduction":"median of seeds 42, 1234, 31415 for GNN; one frozen seed 1234 for classical baselines",
          "archive_members":162,"unique_qasm_hashes":150,"backend":"FakeSherbrooke","optimization_level":1,
          "snapshot_sha256":"509ab97f3bf4cd0c2ddbabe2cdc790e2e440726cb2b65b468c05fc54846ec9c1",
          "noise_model_sha256":"64d58378f06c47a1bd3d115a5ebea38748c2a1b9702937d497f99d0691b43aa5",
          "metrics_sha256":sha_file(azizov_root/"method_metrics.csv"),
          "labels_sha256":sha_file(azizov_root/"labels_core.csv")},sort_keys=True,separators=(",",":"))))
    for r in sim_rows:
      if r.get("method_id")=="azizov_aer_independent" and r.get("task_id")=="pre_recovery_simulator_reader":
        r["claim_boundary"]="Historical coverage-only 162-member route; it is not the scored common-core adaptation below."
        r["comparison_role"]="historical_coverage_only_not_ranked"
    joint_root = ROOT / "artifacts/benchmark_v3/simulator/family_aware_joint_mps_ladder_v1"
    joint_metric_rows = read_csv(joint_root / "aggregate/method_metrics.csv")
    joint_manifest = read_json(joint_root / "aggregate/manifest.json")
    if joint_manifest.get("measurement_campaign_status") != "PARTIAL" or joint_manifest.get("measurement_attempt_count") != 2700:
      raise RuntimeError("joint_mps_measurement_campaign_receipt_mismatch")
    split_files = {
      "c44": joint_root / "aggregate/c44_oof_predictions.csv",
      "family_holdout": joint_root / "aggregate/family_holdout_diagnostic_oof_predictions.csv",
    }
    split_labels = {"c44": "C44 hash-grouped five-fold OOF", "family_holdout": "family-component holdout diagnostic"}
    joint_card_path = joint_root / "aggregate/method_metrics.csv"
    joint_card_sha = sha_file(joint_card_path)
    joint_oof_hashes = {name: sha_file(path) for name, path in split_files.items()}
    joint_manifest_sha = sha_file(joint_root / "aggregate/manifest.json")
    for split, oof_path in split_files.items():
      oof_rows = read_csv(oof_path)
      if len(oof_rows) != 1800:
        raise RuntimeError(f"joint_mps_oof_population_mismatch:{split}:{len(oof_rows)}")
      for metric in joint_metric_rows:
        if metric["split_id"] != split or metric["scope"] != "all_rungs":
          continue
        method = metric["method_id"]
        if method not in joint_methods:
          continue
        assigned = [r for r in oof_rows if r["split_id"] == split and r["method_id"] == method]
        if len(assigned) != 900 or len({(r["source_qasm_sha256"], r["max_bond"]) for r in assigned}) != 900:
          raise RuntimeError(f"joint_mps_assigned_id_mismatch:{split}:{method}")
        if len({r["source_qasm_sha256"] for r in assigned}) != 150 or len({r["max_bond"] for r in assigned}) != 6:
          raise RuntimeError(f"joint_mps_hash_or_rung_count_mismatch:{split}:{method}")
        measured = [r for r in assigned if r["target_status"] == "runtime_observed"]
        if len(measured) != int(metric["runtime_evaluated_rows"]) or len(measured) != 840:
          raise RuntimeError(f"joint_mps_scored_denominator_mismatch:{split}:{method}")
        if any(r["target_runtime_seconds"] == "" or r["runtime_median_seed_seconds"] == "" for r in measured):
          raise RuntimeError(f"joint_mps_measured_target_or_prediction_missing:{split}:{method}")
        actual = np.asarray([float(r["target_runtime_seconds"]) for r in measured], dtype=np.float64)
        predicted = np.asarray([float(r["runtime_median_seed_seconds"]) for r in measured], dtype=np.float64)
        mae = float(np.mean(np.abs(actual - predicted)))
        r2 = float(1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2))
        quality = np.asarray([r["target_quality_pass"].lower() == "true" for r in measured], dtype=np.float64)
        quality_probability = np.asarray([float(r["quality_mean_seed_probability"]) for r in measured], dtype=np.float64)
        brier = float(np.mean((quality - quality_probability) ** 2))
        for name, observed, recorded in (("mae", mae, float(metric["runtime_mae_seconds"])),
                                         ("r2", r2, float(metric["runtime_r2"])),
                                         ("quality_brier", brier, float(metric["quality_brier_score"]))):
          if not math.isclose(observed, recorded, rel_tol=1e-10, abs_tol=1e-10):
            raise RuntimeError(f"joint_mps_metric_replay_mismatch:{split}:{method}:{name}:{observed}:{recorded}")
        scored_keys = sorted(f"{r['source_qasm_sha256']}|chi={int(r['max_bond'])}" for r in measured)
        row_set_sha = sha_bytes("\n".join(scored_keys).encode("utf-8"))
        quality_pass = int(quality.sum())
        quality_fail = len(quality) - quality_pass
        family_method = method == "family_conditioned"
        reader_label = ("Family-conditioned joint MPS runtime/quality adaptation" if family_method
                        else "Matched family-agnostic joint MPS ablation")
        split_suffix = "c44" if split == "c44" else "family_holdout"
        sim_rows.append(simrow(result_kind="prediction", task_id="family_aware_joint_mps_c44",
          method_family="family_aware_joint_runtime_quality", method_id=f"{method}_joint_rung_{split_suffix}",
          reader_label=reader_label + (" (C44)" if split == "c44" else " (family-holdout diagnostic)"),
          fidelity_class="adaptation" if family_method else "matched_ablation",
          claim_boundary=("Local CUDA-Q MPS FP64 joint runtime/quality rung adaptation on 150 exact-QASM hashes × six χ rungs. "
            "Metrics are recomputed on 840 measured targets; 60 technical target failures remain unavailable. "
            "This is not the paper's original approximation-threshold target."),
          comparison_role="primary_comparable" if family_method else "matched_baseline",
          comparison_group_id=f"family_aware_joint_mps_{split_suffix}",
          context_id="cudaq_mps_fp64_chi_2_4_8_16_32_64", population_id="150_qasm_hashes_x_6_chi_rungs",
          target_id="cudaq_mps_fp64_warm_get_state_seconds_per_hash_and_chi",
          target_authority_id=joint_target, evaluation_target_clock="warm_get_state_seconds_per_hash_and_chi",
          method_output_clock="predicted_warm_get_state_seconds_per_hash_and_chi",
          split_id=split_labels[split], result_status="completed_with_60_unavailable_targets",
          expected_assigned_n="900", attempted_n="900", unattempted_n="0", predicted_n="900",
          unavailable_n="60", failed_n="0", pending_n="0", coverage=str(len(measured) / 900), metric_n=str(len(measured)),
          row_set_sha256=row_set_sha, mae_seconds=mae, medae_seconds=metric["runtime_median_absolute_error_seconds"],
          log1p_mae_seconds=metric["runtime_log1p_mae"], r2_seconds=r2,
          p90_absolute_error_seconds=metric["runtime_p90_absolute_error_seconds"],
          p99_absolute_error_seconds=metric["runtime_p99_absolute_error_seconds"],
          max_absolute_error_seconds=metric["runtime_max_absolute_error_seconds"],
          quality_assessed_n=str(len(measured)), quality_unassessed_n="60", quality_pass_n=str(quality_pass),
          quality_fail_n=str(quality_fail), quality_metric_n=str(len(measured)),
          evidence_details_json=json.dumps({"quality_brier_score": brier,
            "technical_attempt_status_counts": joint_manifest["measurement_attempt_status_counts"],
            "measurement_campaign_status": joint_manifest["measurement_campaign_status"],
            "measurement_attempt_count": joint_manifest["measurement_attempt_count"],
            "family_holdout_is_diagnostic": split == "family_holdout",
            "supersedes_historical_reader_rows_with_900_scored": True,
            "source_qasm_hashes": 150, "chi_rungs": [2, 4, 8, 16, 32, 64],
            "quality_threshold": 0.99, "oof_sha256": joint_oof_hashes[split],
            "aggregate_manifest_sha256": joint_manifest_sha}, sort_keys=True, separators=(",", ":")),
          method_card_source_path=str((ROOT/"benchmark_v1/protocol/family_aware_joint_runtime_quality_mps_v1.json").relative_to(ROOT)),
          method_card_source_sha256=sha_file(ROOT/"benchmark_v1/protocol/family_aware_joint_runtime_quality_mps_v1.json")))
    # The current Maestro component-wise local adaptation is distinct from its
    # closed-source Composer/selector and from the failed optimizer-on pilot.
    maestro_path=ROOT/"artifacts/benchmark_v3/simulator/maestro_cpu_component_optimizer_off/evaluation_001/method_metrics.csv"
    maestro_protocol_path=maestro_path.parents[1]/"resolved_protocol.json"
    maestro_protocol=read_json(maestro_protocol_path)
    maestro_rows=read_csv(maestro_path)
    maestro_map={r["method_id"]:r for r in maestro_rows if r["comparison_scope"]=="method_available_rows"}
    maestro_labels={"maestro_style_cpu_sv_component_optimizer_off_adaptation":"Maestro-style Statevector CPU component-cost adaptation",
      "nested_grouped_ridge":"Nested grouped Ridge control","outer_train_median":"Outer-training median control",
      "source_dag_graph_adaptation_cuda":"Source-DAG graph control"}
    for mid,m in maestro_map.items():
      sim_rows.append(simrow(result_kind="prediction",task_id="maestro_component_optimizer_off",method_family="maestro_component_cost",
        method_id=mid,reader_label=maestro_labels.get(mid,mid),fidelity_class="adaptation" if mid.startswith("maestro_style") else "common_baseline",
        claim_boundary="Local QCSim CPU Statevector FP64, optimizer off; the exact closed Composer/selector remains unavailable.",
        comparison_role="primary_comparable" if mid.startswith("maestro_style") else "control",
        context_id="qcsim_statevector_cpu_fp64_1000_shots",population_id="maestro_cpu_optimizer_off_150_assigned",
        evaluation_target_clock=maestro_protocol["evaluation"]["evaluation_target_clock"],method_output_clock=maestro_protocol["evaluation"]["method_output_clock"],
        split_id="nested_grouped_cross_validation",result_status="completed_validated",expected_assigned_n=m["assigned_hashes"],
        attempted_n=m["n_evaluated"],unattempted_n=str(int(m["assigned_hashes"])-int(m["n_evaluated"])),predicted_n=m["n_evaluated"],
        unavailable_n=str(int(m["assigned_hashes"])-int(m["n_evaluated"])),failed_n="0",coverage=m["coverage_of_assigned"],metric_n=m["n_evaluated"],
        mae_seconds=m["mae_seconds"],r2_seconds=m["r2_seconds"],p99_absolute_error_seconds=m["p99_abs_error_seconds"],
        method_card_source_path=str(maestro_path.relative_to(ROOT)),method_card_source_sha256=sha_file(maestro_path),
        evidence_details_json=json.dumps({"dataset":"QCSim circuits","shots":1000,"quality_policy":"not_applicable","scope":"local component prediction; not full selector",
          "process_policy":maestro_protocol["measurement"]["process_policy"],"persistent_process_warm_claim":False,
          "target_reduction":"median of all 15 finite engine-reported samples per hash",
          "clock_contract_path":str(maestro_protocol_path.relative_to(ROOT)),"clock_contract_sha256":sha_file(maestro_protocol_path)},sort_keys=True,separators=(",",":"))))
    sim_rows.append(simrow(result_kind="method_availability_or_oof_coverage",task_id="maestro_original_route",method_family="maestro",
      method_id="maestro_closed_composer_selector_original",reader_label="Maestro original automatic selector / closed Composer route",
      fidelity_class="unavailable",claim_boundary="Original full selector route is not callable from the public repository; the local component adaptation above is a distinct method.",
      comparison_role="unavailable_original_route",context_id="not_available",population_id="not_available",
      evaluation_target_clock="simulator_runtime_target_not_available",method_output_clock="not_available",
      result_status="unavailable_original_route_closed_component",status_reason_counts_json="{\"closed_source_component\":1}"))
    # cuTensorNet stays a same-selected-plan diagnostic, not a cross-method MAE/R2 score.
    native=read_csv(ROOT/"artifacts/benchmark_v3/results/two_domain_recovery_reader_v1/simulator/native_method_appendix.csv")
    for r in native:
      if r["method_id"]=="cutensornet_runtime_est" and r["metric_id"]=="runtime_est_to_matching_warm_contraction_ratio":
        sim_rows.append(simrow(result_kind="paired_native_estimate_diagnostic",task_id="cutensornet_same_selected_plan",
          method_family="tensor_network_cost_estimator",method_id="cutensornet_runtime_est",reader_label="cuTensorNet RUNTIME_EST / matching warm contraction ratio",
          fidelity_class="native_method_specific_diagnostic",claim_boundary=r["claim_boundary"],comparison_role="paired_native_diagnostic",
          context_id=r["configuration_id"],population_id=r["stratum"],evaluation_target_clock=r["evaluation_target_clock"],
          method_output_clock=r["method_output_clock"],result_status=r["status"],attempted_n=r["n_successful_pairs"],metric_n=r["n_successful_pairs"],
          method_card_source_path=r["source_path"],method_card_source_sha256=r["source_sha256"],
          evidence_details_json=json.dumps({"metric_id":r["metric_id"],"median_ratio_estimate_to_actual":float(r["median_value"]),"note":r["note"]},sort_keys=True,separators=(",",":"))))
    # Pasqal is retained as a distinct analog MPS quality companion; its fitted
    # formula was not promoted to a scored predictor.
    pasqal_path=ROOT/"artifacts/benchmark_v1/pasqal_emu_mps_formula_crossed_pilot_v1r1_20260926/run_manifest.json"
    pasqal=pasqal_quality_summary()
    sim_rows.append(simrow(result_kind="analog_quality_companion",task_id="pasqal_emu_mps_formula_crossed_pilot",
      method_family="analog_mps_runtime_model",method_id="pasqal_emu_mps_formula_unpromoted",reader_label="Pasqal EMU-MPS analog companion (formula unpromoted)",
      fidelity_class="method_specific_analog_companion",claim_boundary=f"Analog neutral-atom simulation; {pasqal['assigned_cells']} cells, {pasqal['quality_pass_cells']} quality-pass and {pasqal['quality_fail_cells']} quality-fail; no promoted common runtime score.",
      comparison_role="not_comparable_to_digital_qasm",context_id="pasqal_emu_mps_analog",population_id="crossed_18_cells",
      evaluation_target_clock="analog_simulator_wall_clock",method_output_clock="unpromoted_formula_estimate",result_status="completed_quality_companion_formula_unpromoted",
      expected_assigned_n="18",attempted_n="18",unattempted_n="0",predicted_n="0",unavailable_n="0",failed_n="0",coverage="1",
      quality_assessed_n=str(pasqal["assigned_cells"]),quality_pass_n=str(pasqal["quality_pass_cells"]),quality_fail_n=str(pasqal["quality_fail_cells"]),method_card_source_path=str(pasqal_path.relative_to(ROOT)),
      method_card_source_sha256=sha_file(pasqal_path),evidence_details_json=json.dumps({"ranking_eligibility":"not_rankable","prediction_metrics":"not_promoted",**pasqal},sort_keys=True,separators=(",",":"))))
    sim_rows.append(simrow(result_kind="method_availability_or_oof_coverage",task_id="family_aware_original_route",method_family="family_aware_residual",
      method_id="family_aware_approximation_threshold_original",reader_label="Family-Aware original runtime + approximation-threshold route",
      fidelity_class="unavailable",claim_boundary="Original approximation-threshold target lacks compatible quality labels; local joint runtime/quality adaptation is reported separately.",
      comparison_role="unavailable_original_route",context_id="not_available",population_id="not_available",
      evaluation_target_clock="runtime_plus_approximation_threshold_not_available",method_output_clock="not_available",
      result_status="unavailable_original_quality_target_incompatible",status_reason_counts_json="{\"no_compatible_approximation_threshold_label\":1}"))
    return {"artifacts/results/real_qpu/method_comparison.csv":csv_bytes(out,qpu_fields),
            "artifacts/results/simulator/method_comparison.csv":csv_bytes(sim_rows,sim_fields)}


def input_hashes() -> dict[str,str]:
    azizov_root = ROOT / "artifacts/benchmark_v3/simulator/azizov_common_core_adaptation"
    joint_root = ROOT / "artifacts/benchmark_v3/simulator/family_aware_joint_mps_ladder_v1"
    paths=[PROTOCOL,ROOT/"benchmark_v1/protocol/common_panel_completion.json",ROOT/"benchmark_v1/protocol/real_qpu_common_panel.json",
      PANEL/"run_status.json",PANEL/"inputs/panel_manifest.json",PANEL/"inputs/panel.csv",PANEL/"inputs/targets.csv",PANEL/"inputs/outer_splits.csv",PANEL/"inputs/inner_splits.csv",
      PANEL/"inputs/mali_feature_ledger.csv",PANEL/"inputs/feature_manifest.json",PANEL/"inputs/feature_attempts.csv",PANEL/"inputs/input_index.csv",PANEL/"inputs/input_index_manifest.json",
      ROOT/"benchmark_v1/scripts/run_mali_full_features.py",ROOT/"benchmark_v1/scripts/run_common_panel_mali.py",
      PRED/"neural/final_validation.json",PRED/"polynomial/predictions.csv",PRED/"analytical/attempts.csv",PRED/"analytical/calibration_fits.csv",
      PRED/"hyb/attempts.csv",PRED/"hyb/fold_fits.csv",PANEL/"summary/run_manifest.json",PANEL/"summary/method_metrics.csv",
      PANEL/"summary/paired_metrics.csv",PANEL/"summary/coverage.csv",PANEL/"summary/source_and_reconstruction_metrics.csv",
      PANEL/"summary/unavailable_reasons.csv",SOURCE_SIM,
      ROOT/"artifacts/benchmark_v3/results/two_domain_recovery_reader_v1/simulator/native_method_appendix.csv",
      ROOT/"artifacts/benchmark_v3/simulator/maestro_cpu_component_optimizer_off/evaluation_001/method_metrics.csv",
      ROOT/"artifacts/benchmark_v3/simulator/maestro_cpu_component_optimizer_off/resolved_protocol.json",
      ROOT/"benchmark_v1/protocol/azizov_common_core_gnn_v1.json",
      azizov_root/"method_metrics.csv",azizov_root/"labels_core.csv",azizov_root/"run_manifest.json",
      azizov_root/"environment_and_context_lock.json",azizov_root/"representation_manifest.json",
      ROOT/"benchmark_v1/protocol/family_aware_joint_runtime_quality_mps_v1.json",
      joint_root/"aggregate/method_metrics.csv",joint_root/"aggregate/manifest.json",
      joint_root/"aggregate/c44_oof_predictions.csv",joint_root/"aggregate/family_holdout_diagnostic_oof_predictions.csv",
      joint_root/"aggregate/paired_comparisons.csv",joint_root/"aggregate/selection_metrics.csv",
      joint_root/"full_attempt_002/run_manifest.json",
      ROOT/"artifacts/benchmark_v1/pasqal_emu_mps_formula_crossed_pilot_v1r1_20260926/run_manifest.json",
      ROOT/"artifacts/benchmark_v1/pasqal_emu_mps_formula_crossed_pilot_v1r1_20260926/raw_records.jsonl",
      ROOT/"artifacts/benchmark_v1/pasqal_emu_mps_formula_crossed_pilot_v1r1_20260926/qa_final/qa.json",
      ROOT/"benchmark_v1/execution/manifests/pasqal_emu_mps_formula_crossed_pilot_v1.json"]
    paths += [PRED/f"neural/fold_{f}_transform.json" for f in range(5)]
    paths += sorted(PANEL.glob("inputs/graphs/*.npz"))
    for fold in range(5):
      for method in ("mali_logical_graph","mali_global_mlp"):
       for seed in SEEDS:
        cell=PRED/f"neural/fold_{fold}/{method}/seed_{seed}"
        paths += [cell/"cell_manifest.json",cell/"predictions.csv",cell/"test_failures.csv"]
    result={str(p.relative_to(ROOT)):sha_file(p) for p in paths}
    result["benchmark/scripts/analyze_common_panel.py"]=sha_file(Path(__file__).resolve())
    result["benchmark/requirements.txt"]=sha_file(ROOT/"benchmark/requirements.txt")
    return result


def all_outputs() -> dict[str,bytes]:
    outputs=build_analysis()
    outputs.update(build_reader_tables(outputs))
    # Manifest is deterministic: no wall-clock timestamps and no machine paths.
    pins=input_hashes()
    manifest={"artifact_id":"common-panel-reporting","status":"complete","scope":"OOF replay and post-fit analysis; no retraining or remeasurement",
      "assigned_rows":7350,"qualified_input_evaluation_rows":4515,"sensitivity_reuses_full_panel_oof":True,
      "seed_reduction":{"seeds":list(SEEDS),"rule":"median of all three; missing seed remains incomplete"},
      "input_signature":{"schema":"qre-model-input-signature-v1","global_width_raw":51,"global_width_used":40,"node_width":178,"node_variable_width":167,
        "node_chunk_rows":65536,"edge_index":"int64 directed ordered with duplicate edges retained","checkpoint_verification":"not_in_replay_scope"},
      "bootstrap":{"resamples":10000,"seed":42,"generator":"NumPy PCG64","group_unit":"group_id","source_balanced_empty_source":"invalid replicate","interval_min_valid":9500,
        "interval_scope":"conditional on frozen OOF predictions; exploratory pointwise, not multiplicity-adjusted"},
      "source_hashes":pins,"output_hashes":{k:sha_bytes(v) for k,v in sorted(outputs.items())}}
    outputs["run_manifest.json"]=(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode("utf-8")
    return outputs


def build(validate: bool) -> None:
    outputs=all_outputs()
    destinations={**{f"artifacts/real_qpu/common_panel/analysis/{k}":v for k,v in outputs.items() if k in ANALYSIS_FILES and k!="run_manifest.json"},
      **{k:v for k,v in outputs.items() if k.startswith("artifacts/results/")}}
    manifest=outputs["run_manifest.json"]
    destinations["artifacts/real_qpu/common_panel/analysis/run_manifest.json"]=manifest
    if validate:
      bad=[]
      for rel,data in destinations.items():
        path=ROOT/rel
        if not path.is_file() or path.read_bytes()!=data: bad.append(rel)
      if bad: raise SystemExit("VALIDATION_FAIL mismatched="+",".join(bad))
      print(f"VALIDATION_PASS files={len(destinations)} checkpoint_verification=not_in_replay_scope")
      return
    for rel,data in destinations.items(): atomic_write(ROOT/rel,data)
    print(f"BUILD_PASS files={len(destinations)} analysis_rows={len(list(csv.DictReader(outputs['analysis_metrics.csv'].decode().splitlines())))}")


def main() -> None:
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build",action="store_true")
    group.add_argument("--validate",action="store_true")
    args=parser.parse_args()
    if not PROTOCOL.is_file() or not PANEL.is_dir():
      raise SystemExit("selected-package inputs absent; development-workspace fallback is disabled")
    build(args.validate)


if __name__ == "__main__":
    main()
