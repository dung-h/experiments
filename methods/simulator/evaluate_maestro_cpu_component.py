#!/usr/bin/env python3
"""Evaluate predictors against the new Maestro/QCSim CPU-SV labels.

This program consumes one row per exact QASM hash from ``panel_targets.csv``.
It never reads Aer, CUDA-Q, historical Maestro, or QPU runtime labels. The
source-DAG arm is trained on CUDA only. Importing this module performs no
training, CUDA initialization, measurement, or artifact writes.

Optional ``calibration_model.json`` format (new CPU-SV calibration only)::

    {"schema_id":"maestro-cpu-component-calibration-v1", "artifact_status":"PASS",
     "status":"complete|complete_with_unavailable_knots",
     "shot_count":1000, "coefficient_formula":
       "2^n*(C1q(n)*normalized_1q_count + Ccx(n)*normalized_cx_count) + b(n) + m(n)*shots",
     "attempts_sha256":"<sha256 of attempts ledger>",
     "panel_targets_sha256":"<sha256 of supplied panel_targets.csv>",
     "calibration_source_qasm_sha256":[...],
     "width_coefficients":[
       {"width":2,"status":"valid","C1q":...,"Ccx":...,"b":...,"m":...}, ...]}

The corresponding prediction is ``2**n*(C1q*n1q + Ccx*nCX) + b + m*shots``.
No coefficients are inferred from panel targets. Invalid/missing width knots
remain unavailable, without interpolation across them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import run_mps_fixed_chi16_runtime_adaptation_v1 as s85


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "benchmark_v1/protocol/maestro_cpu_component_benchmark.json"
PANEL = ROOT / "artifacts/benchmark_v1/sim_common_q16_manifest_20260927/sim_common_q16_manifest.csv"
FOLDS = ROOT / "artifacts/benchmark_v1/c44_aer_q16_full_panel_evaluation_20260927/aer_q16_reduced_warm.csv"
TOPOLOGY = ROOT / "artifacts/benchmark_v3/simulator/azizov_common_core_gnn_v1_materialization/source_dag_topology.jsonl"
SOURCE_HASHES = TOPOLOGY.parent / "source_hashes.json"
SEED_REGISTRY = ROOT / "benchmark_v1/registry/seed_registry.json"
EXPECTED_FOLD_COUNTS = {0: 37, 1: 28, 2: 26, 3: 25, 4: 34}
SEEDS = (42, 1234, 31415)
METHOD_IDS = (
    "maestro_style_cpu_sv_component_adaptation",
    "outer_train_median",
    "nested_grouped_ridge",
    "source_dag_graph_adaptation_cuda",
)
COMPONENT_METHOD_ID = METHOD_IDS[0]
COMPONENT_PREDICTION_COLUMN = f"pred_{COMPONENT_METHOD_ID}_seconds"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
TARGET_COLUMNS = ("source_qasm_sha256", "outer_fold", "runtime_seconds", "target_status",
                  "normalized_qasm_sha256", "normalized_width",
                  "normalized_1q_gate_count", "normalized_cx_gate_count")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(TARGET_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"panel target table must contain columns {TARGET_COLUMNS}")
        return list(reader)


def load_frozen_context() -> tuple[dict[str, int], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    compare = protocol["comparison"]
    if compare["outer_folds"] != [0, 1, 2, 3, 4] or compare["neural_seeds"] != list(SEEDS):
        raise ValueError("Maestro comparator folds/seeds differ from the locked protocol")
    seed_context = protocol["context"]
    if sha256_file(SEED_REGISTRY) != seed_context["seed_registry_sha256"]:
        raise ValueError("registered seed registry pin changed")
    seed_registry = json.loads(SEED_REGISTRY.read_text(encoding="utf-8"))
    if seed_registry.get("streams", {}).get("simulator") != seed_context["seed_stream"]:
        raise ValueError("Maestro context seed stream differs from the registered simulator stream")
    if sha256_file(TOPOLOGY) != compare["source_topology_sha256"]:
        raise ValueError("pinned source-DAG topology hash changed")
    fold_rows = s85.read_csv(FOLDS)
    fold_by_hash = s85.frozen_fold_map(fold_rows)
    graphs = s85.load_topology()
    if set(graphs) != set(fold_by_hash):
        raise ValueError("source-DAG topology hash set differs from frozen C44")
    source_hashes = json.loads(SOURCE_HASHES.read_text(encoding="utf-8"))
    if sha256_file(TOPOLOGY) != source_hashes.get("source_dag_jsonl_sha256"):
        raise ValueError("source topology hash differs from its source_hashes.json pin")
    features: dict[str, dict[str, Any]] = {}
    for digest, record in graphs.items():
        globals_, graph = s85.graph_features(record)
        features[digest] = {"globals": globals_, "graph": graph}
    return fold_by_hash, graphs, features


def validate_panel_targets(rows: list[dict[str, str]], fold_by_hash: dict[str, int]) -> dict[str, dict[str, Any]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unavailable_statuses = {"unavailable", *(f"unavailable_{status}" for status in
        protocol["measurement_artifact_contract"]["terminal_statuses"])}
    allowed_statuses = {"runtime_observed", *unavailable_statuses}
    targets: dict[str, dict[str, Any]] = {}
    for row in rows:
        digest = row["source_qasm_sha256"].strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid exact-QASM SHA256: {digest!r}")
        if digest in targets:
            raise ValueError(f"duplicate target row for exact-QASM hash: {digest}")
        fold = int(row["outer_fold"])
        if fold not in EXPECTED_FOLD_COUNTS or fold_by_hash.get(digest) != fold:
            raise ValueError(f"target fold does not match frozen C44 for {digest}")
        status = row["target_status"].strip()
        if status not in allowed_statuses:
            raise ValueError(f"unknown target_status {status!r} for {digest}")
        raw_runtime = row["runtime_seconds"].strip()
        runtime: float | None = None
        if status == "runtime_observed":
            if not raw_runtime:
                raise ValueError(f"runtime_observed target has no runtime: {digest}")
            runtime = float(raw_runtime)
            if not math.isfinite(runtime) or runtime < 0:
                raise ValueError(f"runtime_observed target is non-finite or negative: {digest}")
        elif raw_runtime:
            raise ValueError(f"unavailable target must not carry runtime_seconds: {digest}")
        norm_fields = ("normalized_qasm_sha256", "normalized_width",
                       "normalized_1q_gate_count", "normalized_cx_gate_count")
        normalized_raw = {name: row.get(name, "").strip() for name in norm_fields}
        present = [bool(value) for value in normalized_raw.values()]
        if any(present) and not all(present):
            raise ValueError(f"partial normalized-QASM metadata for {digest}; provide all four fields or leave all blank")
        normalized: dict[str, Any] = {name: None for name in norm_fields}
        if all(present):
            normalized_hash = normalized_raw["normalized_qasm_sha256"].lower()
            if len(normalized_hash) != 64 or any(char not in "0123456789abcdef" for char in normalized_hash):
                raise ValueError(f"invalid normalized-QASM SHA256 for {digest}")
            try:
                normalized_width = int(normalized_raw["normalized_width"])
                normalized_1q = int(normalized_raw["normalized_1q_gate_count"])
                normalized_cx = int(normalized_raw["normalized_cx_gate_count"])
            except ValueError as error:
                raise ValueError(f"noninteger normalized execution metadata for {digest}") from error
            if normalized_width < 1 or normalized_1q < 0 or normalized_cx < 0:
                raise ValueError(f"invalid normalized execution width/count for {digest}")
            normalized = {"normalized_qasm_sha256": normalized_hash,
                          "normalized_width": normalized_width,
                          "normalized_1q_gate_count": normalized_1q,
                          "normalized_cx_gate_count": normalized_cx}
        targets[digest] = {"source_qasm_sha256": digest, "fold": fold,
                           "runtime_seconds": runtime, "target_status": status, **normalized}
    if set(targets) != set(fold_by_hash):
        missing = sorted(set(fold_by_hash) - set(targets))
        extra = sorted(set(targets) - set(fold_by_hash))
        raise ValueError(f"panel target hashes differ from frozen 150-hash envelope; missing={len(missing)} extra={len(extra)}")
    if dict(sorted(Counter(row["fold"] for row in targets.values()).items())) != EXPECTED_FOLD_COUNTS:
        raise ValueError("panel target fold counts differ from frozen C44 counts")
    return targets


def _six_features(feature_by_hash: dict[str, dict[str, Any]], hashes: list[str]):
    import numpy as np
    return np.asarray([[feature_by_hash[digest]["globals"][name] for name in s85.GLOBAL_NAMES]
                       for digest in hashes], dtype=np.float64)


def nested_ridge_fold(
    fold: int, targets: dict[str, dict[str, Any]], fold_by_hash: dict[str, int],
    feature_by_hash: dict[str, dict[str, Any]],
) -> tuple[dict[str, float | None], float | None, dict[str, float]]:
    """Fit log1p Ridge, selecting alpha by four-way grouped inner CV seconds-MAE."""
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    train = sorted(d for d, value in targets.items() if value["fold"] != fold and value["runtime_seconds"] is not None)
    test = sorted(d for d, value in targets.items() if value["fold"] == fold)
    predictions: dict[str, float | None] = {digest: None for digest in test}
    if len(train) < 4:
        return predictions, None, {}
    X = _six_features(feature_by_hash, train)
    y = np.asarray([targets[d]["runtime_seconds"] for d in train], dtype=np.float64)
    groups = np.asarray(train)
    splitter = GroupKFold(n_splits=4)
    scores: dict[str, float] = {}
    for alpha in ALPHAS:
        errors = []
        for inner_train, inner_valid in splitter.split(X, np.log1p(y), groups):
            scaler = StandardScaler().fit(X[inner_train])
            model = Ridge(alpha=alpha).fit(scaler.transform(X[inner_train]), np.log1p(y[inner_train]))
            pred = np.maximum(0.0, np.expm1(model.predict(scaler.transform(X[inner_valid]))))
            errors.extend(np.abs(y[inner_valid] - pred).tolist())
        if not errors:
            raise ValueError("four-way exact-hash inner CV produced no validation predictions")
        scores[f"{alpha:g}"] = float(np.mean(errors))
    chosen = min(ALPHAS, key=lambda alpha: (scores[f"{alpha:g}"], alpha))
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=chosen).fit(scaler.transform(X), np.log1p(y))
    X_test = _six_features(feature_by_hash, test)
    values = np.maximum(0.0, np.expm1(model.predict(scaler.transform(X_test))))
    for digest, pred in zip(test, values):
        predictions[digest] = float(pred)
    return predictions, float(chosen), scores


def train_graph_fold(
    fold: int, targets: dict[str, dict[str, Any]], fold_by_hash: dict[str, int],
    feature_by_hash: dict[str, dict[str, Any]],
) -> dict[int, dict[str, float]]:
    """S85-compatible source-DAG adaptation; a CUDA device is mandatory."""
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("source-DAG graph arm requires CUDA; CPU fallback is forbidden")
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    torch.set_num_threads(2)
    train = sorted(d for d, value in targets.items() if value["fold"] != fold and value["runtime_seconds"] is not None)
    test = sorted(d for d, value in targets.items() if value["fold"] == fold)
    if not train or not test or set(train) & set(test):
        raise ValueError(f"invalid exact-hash graph split for fold {fold}")
    vocabulary = sorted({node["operation"] for digest in train for node in feature_by_hash[digest]["graph"]["nodes"]})
    vocab_map = {name: index for index, name in enumerate(vocabulary)}
    unk = len(vocabulary)
    node_width = len(vocabulary) + 1 + 11
    log_global = {digest: np.log1p(np.asarray([feature_by_hash[digest]["globals"][name]
                                               for name in s85.GLOBAL_NAMES], dtype=np.float32))
                  for digest in targets}
    scaler = StandardScaler().fit(np.stack([log_global[digest] for digest in train]))
    mean = scaler.mean_.astype(np.float32)
    std = scaler.scale_.astype(np.float32)
    std[std < 1e-6] = 1.0

    def make_data(digest: str, include_target: bool):
        graph = feature_by_hash[digest]["graph"]
        nodes = graph["nodes"]
        x = torch.tensor([s85.encode_node_features(node, vocab_map, unk, int(graph["num_qubits"]))
                          for node in nodes], dtype=torch.float32)
        src, dst = graph["edge_src"], graph["edge_dst"]
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
        g = (log_global[digest] - mean) / std
        item = Data(x=x, edge_index=edge_index,
                    global_features=torch.as_tensor(g.reshape(1, -1), dtype=torch.float32))
        if include_target:
            item.y = torch.tensor(math.log1p(float(targets[digest]["runtime_seconds"])), dtype=torch.float32)
        return item

    model_class = s85._torch_graph_model(node_width, len(s85.GLOBAL_NAMES))
    device = torch.device("cuda:0")
    results: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        loader = DataLoader([make_data(digest, True) for digest in train], batch_size=32, shuffle=True)
        for epoch in range(500):
            model.train()
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(batch), batch.y.reshape(-1))
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"non-finite source-DAG loss in fold {fold}, seed {seed}, epoch {epoch}")
                loss.backward(); optimizer.step()
        model.eval()
        test_loader = DataLoader([make_data(digest, False) for digest in test], batch_size=32, shuffle=False)
        predictions: list[float] = []
        with torch.no_grad():
            for batch in test_loader:
                predictions.extend(torch.expm1(model(batch.to(device))).clamp_min(0).cpu().tolist())
        if len(predictions) != len(test) or any(not math.isfinite(value) or value < 0 for value in predictions):
            raise RuntimeError(f"invalid graph predictions in fold {fold}, seed {seed}")
        results[seed] = {digest: float(value) for digest, value in zip(test, predictions)}
        del model, optimizer, loader, test_loader
        torch.cuda.empty_cache()
    return results


def expected_calibration_hashes() -> list[str]:
    import prepare_maestro_cpu_component as preparer
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cells = preparer.calibration_cells(protocol)
    return sorted(row["qasm_sha256"] for row in cells)


def load_component_model(
    path: Path, target_hashes: set[str], expected_shots: int = 1000,
    expected_synthetic_hashes: list[str] | None = None,
    expected_panel_targets_sha256: str | None = None,
) -> dict[int, dict[str, Any]]:
    model = json.loads(path.read_text(encoding="utf-8"))
    active_protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if model.get("schema_id") != "maestro-cpu-component-calibration-v1":
        raise ValueError("calibration model has an unknown schema_id")
    if model.get("protocol_sha256") != sha256_file(PROTOCOL):
        raise ValueError("calibration model protocol hash differs from the selected runtime context")
    if model.get("protocol_id") != active_protocol["protocol_id"]:
        raise ValueError("calibration model protocol_id differs from the selected runtime context")
    if model.get("optimizer_enabled_requested") != active_protocol["context"].get("optimize_circuit"):
        raise ValueError("calibration model optimizer setting differs from the selected runtime context")
    if model.get("artifact_status") != "PASS":
        raise ValueError("calibration model artifact_status must be PASS")
    if model.get("status") not in {"complete", "complete_with_unavailable_knots"}:
        raise ValueError("calibration model status must report a completed calibration")
    panel_pin = model.get("panel_targets_sha256")
    if (expected_panel_targets_sha256 is None or not isinstance(panel_pin, str)
            or panel_pin.lower() != expected_panel_targets_sha256.lower()):
        raise ValueError("calibration model panel_targets_sha256 does not match supplied panel_targets.csv")
    if int(model.get("shot_count", -1)) != expected_shots:
        raise ValueError("calibration model shot_count differs from the frozen panel context")
    if model.get("coefficient_formula") != (
            "2^n*(C1q(n)*normalized_1q_count + Ccx(n)*normalized_cx_count) + b(n) + m(n)*shots"):
        raise ValueError("calibration model coefficient formula differs from the frozen Maestro component contract")
    # Pin the exact attempts ledger that produced the calibration summary.
    # Some producers additionally retain per-attempt hashes; the ledger pin is
    # mandatory because it proves the full ordered input record set.
    attempt_hashes = model.get("input_attempt_sha256")
    if not isinstance(attempt_hashes, list) or not attempt_hashes or any(
            len(str(value)) != 64 or any(char not in "0123456789abcdef" for char in str(value).lower())
            for value in attempt_hashes):
        raise ValueError("calibration model must pin valid input attempt SHA256 values")
    scalar_ledger_hash = model.get("attempts_sha256")
    if scalar_ledger_hash is not None:
        if (not isinstance(scalar_ledger_hash, str) or len(scalar_ledger_hash) != 64
                or any(char not in "0123456789abcdef" for char in scalar_ledger_hash.lower())):
            raise ValueError("calibration model contains a malformed attempts-ledger SHA256")
        if scalar_ledger_hash.lower() not in {str(value).lower() for value in attempt_hashes}:
            raise ValueError("attempts_sha256 scalar is not pinned by input_attempt_sha256")
    source_hash_list = model.get("calibration_source_qasm_sha256")
    if not isinstance(source_hash_list, list) or not source_hash_list:
        raise ValueError("calibration model must retain the exact synthetic calibration source-QASM hashes")
    source_hashes = set(source_hash_list)
    if source_hashes & target_hashes:
        raise ValueError("calibration model source QASM hashes overlap the panel targets")
    if expected_synthetic_hashes is None:
        expected_synthetic_hashes = expected_calibration_hashes()
    if sorted(str(value).lower() for value in source_hash_list) != expected_synthetic_hashes:
        raise ValueError("calibration model does not pin the exact 112 synthetic calibration-cell QASM hashes")
    knots: dict[int, dict[str, Any]] = {}
    for row in model.get("width_coefficients", []):
        width = int(row["width"])
        if width in knots:
            raise ValueError(f"duplicate calibration width knot: {width}")
        knots[width] = row
        if row.get("status") not in {"valid", "unavailable"}:
            raise ValueError(f"unknown calibration width status at width {width}")
        if row.get("status") == "valid":
            for key in ("C1q", "Ccx", "b", "m"):
                value = float(row[key])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid calibration coefficient {key} at width {width}")
    if set(knots) != set(range(2, 10)):
        raise ValueError("calibration model must retain explicit knots or unavailable reasons for widths 2 through 9")
    has_unavailable = any(row["status"] == "unavailable" for row in knots.values())
    if model["status"] == "complete" and has_unavailable:
        raise ValueError("complete calibration status conflicts with unavailable width knots")
    if model["status"] == "complete_with_unavailable_knots" and not has_unavailable:
        raise ValueError("complete_with_unavailable_knots status requires an unavailable width knot")
    return knots


def component_predictions(
    targets: dict[str, dict[str, Any]], feature_by_hash: dict[str, dict[str, Any]],
    calibration_model: dict[int, dict[str, Any]] | None, shots: int,
) -> dict[str, tuple[float | None, str]]:
    result: dict[str, tuple[float | None, str]] = {}
    for digest, target in targets.items():
        if calibration_model is None:
            result[digest] = (None, "calibration_model_not_provided")
            continue
        norm_hash = target.get("normalized_qasm_sha256")
        norm_width = target.get("normalized_width")
        norm_1q = target.get("normalized_1q_gate_count")
        norm_cx = target.get("normalized_cx_gate_count")
        if any(value is None for value in (norm_hash, norm_width, norm_1q, norm_cx)):
            result[digest] = (None, "normalized_execution_metadata_unavailable")
            continue
        source_width = int(feature_by_hash[digest]["graph"]["num_qubits"])
        width = int(norm_width)
        if width != source_width:
            raise ValueError(f"normalized width {width} disagrees with source-DAG width {source_width} for {digest}")
        knot = calibration_model.get(width)
        if knot is None or knot.get("status") != "valid":
            result[digest] = (None, "invalid_or_missing_width_knot")
            continue
        prediction = (2**width * (float(knot["C1q"]) * int(norm_1q)
                                  + float(knot["Ccx"]) * int(norm_cx))
                      + float(knot["b"]) + float(knot["m"]) * shots)
        if not math.isfinite(prediction) or prediction < 0:
            result[digest] = (None, "nonfinite_or_negative_component_prediction")
        else:
            result[digest] = (prediction, "available")
    return result


def metric_bundle(targets: list[float], predictions: list[float], assigned: int) -> dict[str, Any]:
    import numpy as np
    if len(targets) != len(predictions):
        raise ValueError("metric target/prediction length mismatch")
    if not targets:
        return {"n_evaluated": 0, "coverage_of_assigned": 0.0, "mae_seconds": None,
                "medae_seconds": None, "mae_log1p_seconds": None, "r2_seconds": None,
                "p90_abs_error_seconds": None, "p99_abs_error_seconds": None, "max_abs_error_seconds": None}
    y = np.asarray(targets, dtype=np.float64); p = np.asarray(predictions, dtype=np.float64)
    err = np.abs(y-p); denom = float(np.sum((y-y.mean())**2))
    return {"n_evaluated": int(len(y)), "coverage_of_assigned": float(len(y)/assigned),
            "mae_seconds": float(err.mean()), "medae_seconds": float(np.median(err)),
            "mae_log1p_seconds": float(np.mean(np.abs(np.log1p(y)-np.log1p(p)))),
            "r2_seconds": float(1-np.square(y-p).sum()/denom) if denom > 0 else None,
            "p90_abs_error_seconds": float(np.quantile(err, .90)),
            "p99_abs_error_seconds": float(np.quantile(err, .99)),
            "max_abs_error_seconds": float(err.max())}


def build_oof_predictions(
    targets: dict[str, dict[str, Any]], fold_by_hash: dict[str, int],
    feature_by_hash: dict[str, dict[str, Any]],
    graph_by_fold: dict[int, dict[int, dict[str, float]]],
    component_by_hash: dict[str, tuple[float | None, str]], shots: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    alpha_by_fold: dict[str, Any] = {}
    ridge_by_hash: dict[str, float | None] = {}
    median_by_hash: dict[str, float | None] = {}
    for fold in range(5):
        ridge_preds, alpha, alpha_scores = nested_ridge_fold(fold, targets, fold_by_hash, feature_by_hash)
        alpha_by_fold[str(fold)] = {"selected_alpha": alpha, "inner_seconds_mae_by_alpha": alpha_scores}
        train_values = [value["runtime_seconds"] for digest, value in targets.items()
                        if value["fold"] != fold and value["runtime_seconds"] is not None]
        median = float(statistics.median(train_values)) if train_values else None
        for digest, value in targets.items():
            if value["fold"] != fold:
                continue
            median_by_hash[digest] = median
            ridge_by_hash[digest] = ridge_preds[digest]
            graph_seed_predictions = graph_by_fold.get(fold, {})
            per_seed = {seed: graph_seed_predictions.get(seed, {}).get(digest) for seed in SEEDS}
            graph_median = (float(statistics.median(float(per_seed[s]) for s in SEEDS))
                            if all(per_seed[s] is not None for s in SEEDS) else None)
            component_value, component_status = component_by_hash[digest]
            row: dict[str, Any] = {
                "source_qasm_sha256": digest, "fold": fold,
                "target_status": value["target_status"],
                "runtime_seconds": "" if value["runtime_seconds"] is None else value["runtime_seconds"],
                "normalized_qasm_sha256": value["normalized_qasm_sha256"] or "",
                "normalized_width": "" if value["normalized_width"] is None else value["normalized_width"],
                "normalized_1q_gate_count": "" if value["normalized_1q_gate_count"] is None else value["normalized_1q_gate_count"],
                "normalized_cx_gate_count": "" if value["normalized_cx_gate_count"] is None else value["normalized_cx_gate_count"],
                "ridge_selected_alpha": "" if alpha is None else alpha,
                "component_prediction_status": component_status,
                COMPONENT_PREDICTION_COLUMN: "" if component_value is None else component_value,
                "outer_train_median_status": "available" if median is not None else "unavailable_no_finite_outer_train_labels",
                "pred_outer_train_median_seconds": "" if median is None else median,
                "nested_grouped_ridge_status": "available" if ridge_preds[digest] is not None else "unavailable_insufficient_outer_train_groups",
                "pred_nested_grouped_ridge_seconds": "" if ridge_preds[digest] is None else ridge_preds[digest],
                "source_dag_graph_status": "available" if graph_median is not None else "unavailable_graph_seed_prediction",
                "pred_source_dag_graph_adaptation_cuda_seconds": "" if graph_median is None else graph_median,
            }
            for seed in SEEDS:
                row[f"pred_graph_seed_{seed}_seconds"] = "" if per_seed[seed] is None else per_seed[seed]
            rows.append(row)
    rows.sort(key=lambda row: row["source_qasm_sha256"])
    component_rows = []
    for row in rows:
        component_rows.append({"source_qasm_sha256": row["source_qasm_sha256"], "fold": row["fold"],
                              "target_status": row["target_status"], "runtime_seconds": row["runtime_seconds"],
                              "normalized_qasm_sha256": row["normalized_qasm_sha256"],
                              "normalized_width": row["normalized_width"],
                              "normalized_1q_gate_count": row["normalized_1q_gate_count"],
                              "normalized_cx_gate_count": row["normalized_cx_gate_count"],
                              "predicted_seconds": row[COMPONENT_PREDICTION_COLUMN],
                              "prediction_status": row["component_prediction_status"], "shots": shots})
    return rows, component_rows, alpha_by_fold


def _bootstrap_seed(label: str) -> tuple[int, str]:
    registry = json.loads(SEED_REGISTRY.read_text(encoding="utf-8"))
    stream = str(registry["streams"]["bootstrap"])
    payload = f"{registry['root_seed']}|{stream}|maestro-cpu-component-oof-evaluation-v1|{label}|0".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big"), stream


def _paired_bootstrap_row(
    candidate: str, reference: str, candidate_col: str, reference_col: str,
    rows: list[dict[str, Any]], assigned: int,
) -> dict[str, Any]:
    import numpy as np
    shared = [row for row in rows if row["target_status"] == "runtime_observed"
              and row[candidate_col] not in ("", None) and row[reference_col] not in ("", None)]
    label = f"{candidate}_vs_{reference}_same_qcsim_oof"
    seed, stream = _bootstrap_seed(label)
    base: dict[str, Any] = {
        "method_id": candidate, "comparison_scope": "paired_bootstrap_delta",
        "assigned_hashes": assigned, "target_observed_hashes": sum(r["target_status"] == "runtime_observed" for r in rows),
        "prediction_coverage_on_observed": len(shared) / max(1, sum(r["target_status"] == "runtime_observed" for r in rows)),
        "n_evaluated": len(shared), "coverage_of_assigned": len(shared) / assigned,
        "reference_method_id": reference, "candidate_method_id": candidate,
        "shared_hashes": len(shared), "shared_hash_set_sha256": hashlib.sha256(
            "\n".join(sorted(row["source_qasm_sha256"] for row in shared)).encode()).hexdigest(),
        "observed_mae_difference_seconds": None, "bootstrap_mean_mae_difference_seconds": None,
        "bootstrap_ci_low_seconds": None, "bootstrap_ci_high_seconds": None,
        "bootstrap_replicates_requested": 10000, "bootstrap_replicates_completed": 0,
        "bootstrap_seed": seed, "bootstrap_stream": stream,
        "bootstrap_group_column": "source_qasm_sha256", "bootstrap_quantile_method": "linear",
        "bootstrap_rng_bit_generator": "PCG64", "status": "unavailable_no_shared_hashes",
    }
    if not shared:
        return base
    actual = np.asarray([float(row["runtime_seconds"]) for row in shared], dtype=np.float64)
    cand = np.asarray([float(row[candidate_col]) for row in shared], dtype=np.float64)
    ref = np.asarray([float(row[reference_col]) for row in shared], dtype=np.float64)
    cand_error = np.abs(cand - actual); ref_error = np.abs(ref - actual)
    observed = float(cand_error.mean() - ref_error.mean())
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(shared), size=(10000, len(shared)))
    deltas = cand_error[indices].mean(axis=1) - ref_error[indices].mean(axis=1)
    low, high = np.quantile(deltas, [0.025, 0.975], method="linear")
    base.update({"observed_mae_difference_seconds": observed,
                 "bootstrap_mean_mae_difference_seconds": float(deltas.mean()),
                 "bootstrap_ci_low_seconds": float(low), "bootstrap_ci_high_seconds": float(high),
                 "bootstrap_replicates_completed": 10000, "status": "evaluated"})
    return base


def score_oof(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = []
    pred_column_by_method = {
        METHOD_IDS[0]: COMPONENT_PREDICTION_COLUMN,
        METHOD_IDS[1]: "pred_outer_train_median_seconds",
        METHOD_IDS[2]: "pred_nested_grouped_ridge_seconds",
        METHOD_IDS[3]: "pred_source_dag_graph_adaptation_cuda_seconds",
    }
    assigned = len(rows)
    observed = [row for row in rows if row["target_status"] == "runtime_observed"]
    learned_methods = (METHOD_IDS[1], METHOD_IDS[2], METHOD_IDS[3])
    common_learned = [row for row in observed if all(
        row[pred_column_by_method[method]] not in ("", None) for method in learned_methods)]
    common_all = [row for row in observed if all(
        row[column] not in ("", None) for column in pred_column_by_method.values())]
    for method, column in pred_column_by_method.items():
        paired = [row for row in observed if row[column] not in ("", None)]
        for scope, scope_rows in (("method_available_rows", paired),
                                  ("common_learned_methods_intersection", common_learned),
                                  ("all_methods_common_successful_intersection", common_all)):
            # The component comparator is not part of the learned-method
            # intersection. Keep its row explicit with n=0 instead of trying
            # to parse its blank predictions on the learned-only shared set.
            if scope == "common_learned_methods_intersection" and method == METHOD_IDS[0]:
                scope_rows = []
            bundle = metric_bundle([float(row["runtime_seconds"]) for row in scope_rows],
                                   [float(row[column]) for row in scope_rows], assigned)
            metrics.append({"method_id": method, "comparison_scope": scope,
                            "assigned_hashes": assigned, "target_observed_hashes": len(observed),
                            "prediction_coverage_on_observed": len(scope_rows) / max(1, len(observed)), **bundle})
    method_columns = [(method, pred_column_by_method[method]) for method in METHOD_IDS]
    for index, (candidate, candidate_col) in enumerate(method_columns):
        for reference, reference_col in method_columns[index + 1:]:
            metrics.append(_paired_bootstrap_row(candidate, reference, candidate_col,
                                                 reference_col, rows, assigned))
    return metrics


def evaluate(
    panel_rows: list[dict[str, str]], fold_by_hash: dict[str, int],
    feature_by_hash: dict[str, dict[str, Any]], graph_runner: Callable[..., dict[int, dict[str, float]]] | None = None,
    calibration_model: dict[int, dict[str, Any]] | None = None, shots: int = 1000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    active_protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    targets = validate_panel_targets(panel_rows, fold_by_hash)
    graph_by_fold: dict[int, dict[int, dict[str, float]]] = {}
    if graph_runner is None:
        graph_runner = train_graph_fold
    for fold in range(5):
        graph_by_fold[fold] = graph_runner(fold, targets, fold_by_hash, feature_by_hash)
    component = component_predictions(targets, feature_by_hash, calibration_model, shots)
    oof_rows, component_rows, alpha_by_fold = build_oof_predictions(
        targets, fold_by_hash, feature_by_hash, graph_by_fold, component, shots)
    metric_rows = score_oof(oof_rows)
    import numpy as np
    manifest = {
        "artifact_id": f"{active_protocol['protocol_id']}-oof-evaluation",
        "status": "evaluation_complete",
        "protocol_id": active_protocol["protocol_id"],
        "execution_context_id": active_protocol.get("execution_context_id"),
        "evaluation_target_clock": active_protocol.get("evaluation", {}).get(
            "evaluation_target_clock", active_protocol["measurement"]["clock"]),
        "assigned_hashes": len(targets),
        "outer_fold_counts": {str(k): sum(value["fold"] == k for value in targets.values()) for k in range(5)},
        "target_observed_hashes": sum(value["runtime_seconds"] is not None for value in targets.values()),
        "target_runtime_source": "panel_targets.csv (new QCSim CPU-FP64 label only)",
        "historical_Aer_CUDAQ_QPU_labels_read": False,
        "outer_train_alpha_selection": alpha_by_fold,
        "graph_device": "CUDA required; no CPU fallback",
        "seeds": list(SEEDS), "methods": list(METHOD_IDS),
        "accuracy_gated_continuation": False,
        "calibration_model_imported": calibration_model is not None,
        "paired_bootstrap_implementation": {
            "replicates": 10000,
            "group_column": "source_qasm_sha256",
            "rng_bit_generator": "PCG64",
            "numpy_version": np.__version__,
            "confidence_interval_quantile_method": "linear",
            "confidence_interval_quantiles": [0.025, 0.975],
        },
    }
    return oof_rows, component_rows, metric_rows, manifest


def cuda_execution_identity() -> dict[str, Any]:
    """Initialize and record the required GPU context (caller holds the lease)."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("source-DAG graph arm requires CUDA; CPU fallback is forbidden")
    torch.set_num_threads(2)
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip().splitlines()
    if not driver:
        raise RuntimeError("could not record NVIDIA driver version")
    return {
        "device": "cuda:0",
        "gpu_name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": int(properties.total_memory),
        "driver_version": driver[0].strip(),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda,
        "torch_geometric_version": importlib.metadata.version("torch-geometric"),
        "python_version": sys.version.split()[0],
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "fp32_training": True,
        "cuda_initialized_after_compute_lease": True,
    }


def evaluate_with_exclusive_compute(
    panel_rows: list[dict[str, str]], fold_by_hash: dict[str, int],
    feature_by_hash: dict[str, dict[str, Any]],
    calibration_model: dict[int, dict[str, Any]] | None = None, shots: int = 1000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run every CUDA graph fold under the shared timing lock and GPU lease."""
    with s85.exclusive_compute_lock():
        cuda_identity = cuda_execution_identity()
        outputs = evaluate(panel_rows, fold_by_hash, feature_by_hash,
                           calibration_model=calibration_model, shots=shots)
    return (*outputs, cuda_identity)


def main() -> None:
    global PROTOCOL, METHOD_IDS, COMPONENT_METHOD_ID, COMPONENT_PREDICTION_COLUMN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-model", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL,
                        help="Full frozen or resolved runtime protocol; defaults to the historical optimizer-on protocol.")
    args = parser.parse_args()
    PROTOCOL = args.protocol.resolve()
    selected_protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if selected_protocol["protocol_id"] == "maestro-cpu-component-optimizer-off-v1":
        if selected_protocol.get("context", {}).get("optimize_circuit") is not False:
            raise ValueError("optimizer-off evaluator requires a resolved protocol with optimize_circuit=false")
        COMPONENT_METHOD_ID = selected_protocol["method_id"]
        METHOD_IDS = (COMPONENT_METHOD_ID, *METHOD_IDS[1:])
        COMPONENT_PREDICTION_COLUMN = f"pred_{COMPONENT_METHOD_ID}_seconds"
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    panel_rows = read_csv(args.panel_targets)
    fold_by_hash, _graphs, features = load_frozen_context()
    targets = validate_panel_targets(panel_rows, fold_by_hash)
    shots = int(json.loads(PROTOCOL.read_text())["context"]["shots"])
    calibration = load_component_model(args.calibration_model, set(targets), shots,
                                       expected_calibration_hashes(),
                                       sha256_file(args.panel_targets)) if args.calibration_model else None
    predictions, component_rows, metrics, manifest, cuda_identity = evaluate_with_exclusive_compute(
        panel_rows, fold_by_hash, features, calibration_model=calibration,
        shots=shots)
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "oof_predictions.csv", predictions)
    write_csv(args.output_dir / "component_predictions.csv", component_rows)
    write_csv(args.output_dir / "method_metrics.csv", metrics)
    manifest.update({
        "protocol_sha256": sha256_file(PROTOCOL),
        "protocol_id": selected_protocol["protocol_id"],
        "execution_context_id": selected_protocol.get("execution_context_id"),
        "resolved_from": selected_protocol.get("resolved_from"),
        "optimizer_enabled_requested": selected_protocol.get("context", {}).get("optimize_circuit"),
        "evaluation_target_clock": selected_protocol.get("evaluation", {}).get(
            "evaluation_target_clock", selected_protocol["measurement"]["clock"]),
        "panel_targets_sha256": sha256_file(args.panel_targets),
        "source_topology_sha256": sha256_file(TOPOLOGY),
        "c44_fold_source_sha256": sha256_file(FOLDS),
        "seed_registry_sha256": sha256_file(SEED_REGISTRY),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "cuda_execution": cuda_identity,
        "compute_exclusivity": {
            "shared_timing_lock": str(s85.SHARED_TIMING_LOCK),
            "host_gpu_lease": str(s85.GPU_LEASE_PATH),
            "held_during_cuda_initialization_and_all_five_folds": True,
        },
        "calibration_model_sha256": sha256_file(args.calibration_model) if args.calibration_model else None,
        "outputs": {name: sha256_file(args.output_dir / name) for name in (
            "oof_predictions.csv", "component_predictions.csv", "method_metrics.csv")},
    })
    (args.output_dir / "result_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "assigned_hashes": manifest["assigned_hashes"],
                      "target_observed_hashes": manifest["target_observed_hashes"],
                      "output_dir": str(args.output_dir.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
