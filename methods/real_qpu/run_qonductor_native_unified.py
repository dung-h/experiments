#!/usr/bin/env python3
"""Run the frozen joint Qonductor polynomial adaptation on archived QPU rows.

Consumes only a validated native-feature ledger. Fits one model per outer
fold, selects polynomial degree on the four frozen inner grouped folds, and
writes atomic checkpoints so interrupted execution can resume without leakage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "benchmark_v1/protocol/qonductor_native_features.json"
EXECUTION = ROOT / "benchmark_v1/protocol/estimator_execution.json"
AUTH = ROOT / "artifacts/unified_runtime_estimation/execution_authorization.json"
CANONICAL = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
OUTER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv"
INNER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_inner_split_v2.csv"
FEATURES = ROOT / "artifacts/real_qpu/qonductor_native_features/feature_rows.csv"
OUTPUT = ROOT / "artifacts/real_qpu/qonductor_native_features"
INITIAL_QA_RECEIPT = ROOT / "artifacts/unified_runtime_estimation/initial_cells_qa_receipt.json"
TECHNICAL_GATE_RECEIPT = ROOT / "artifacts/unified_runtime_estimation/technical_gate_receipt.json"
FEATURE_NAMES = ["swap", "depth", "num_qubits", "shots", "circuit_count"]
SOURCES = {"mali_real_qpu": 340, "qonductor_single_circuit_ibm": 4482, "qpack_mcp": 3945}
ATTEMPT_FIELDS = [
    "canonical_observation_id", "source_id", "outer_fold", "unified_leakage_group_id",
    "availability_status", "terminal_reason", "attempt_status", "selected_degree", "actual_seconds",
    "predicted_seconds", "absolute_error_seconds", "negative_prediction",
]
COMPARATORS = [
    ROOT / "artifacts/benchmark_v3/real_qpu/unified_learned_methods_oof_v4r2/method_attempts.csv",
    ROOT / "artifacts/benchmark_v3/real_qpu/unified_graph_v3_large_oof_20260930/method_attempts.csv",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_root_technical_gate(authorization_sha256: str) -> tuple[dict[str, Any], str]:
    """Fail closed unless the root-issued C1–C4 gate matches active pins."""
    if not TECHNICAL_GATE_RECEIPT.is_file():
        raise RuntimeError("root technical-gate receipt is required before Qonductor fitting")
    technical = json.loads(TECHNICAL_GATE_RECEIPT.read_text(encoding="utf-8"))
    gates = technical.get("gates", {})
    if technical.get("overall_status") != "PASS" or any(gates.get(name) != "PASS" for name in ("C1", "C2", "C3", "C4")):
        raise RuntimeError("technical gate receipt must show overall PASS and C1–C4 PASS")
    technical_sha = digest(TECHNICAL_GATE_RECEIPT)
    current_protocol_sha = digest(PROTOCOL)
    if technical.get("execution_authorization_sha256") != authorization_sha256:
        raise RuntimeError("technical gate receipt is not bound to the current execution authorization")
    contracts = technical.get("contract_hashes", {})
    if contracts.get("qonductor_native_features") != current_protocol_sha:
        raise RuntimeError("technical gate receipt does not bind the active Qonductor method contract")
    return technical, technical_sha


def require_root_initial_qa(signature: dict[str, Any], authorization_sha256: str) -> dict[str, Any]:
    """Fail closed unless the shared root-issued initial-cell QA is current."""
    if not INITIAL_QA_RECEIPT.is_file():
        raise RuntimeError("root initial-cell QA receipt is required before folds 1–4")
    _technical, technical_sha = require_root_technical_gate(authorization_sha256)

    receipt = json.loads(INITIAL_QA_RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("overall_status") != "PASS":
        raise RuntimeError("shared initial-cell QA receipt is not overall PASS")
    if receipt.get("technical_gate_receipt_sha256") != technical_sha:
        raise RuntimeError("initial-cell QA receipt is bound to a different technical-gate receipt")
    if receipt.get("execution_authorization_sha256") != authorization_sha256:
        raise RuntimeError("initial-cell QA receipt is bound to a different execution authorization")
    cell_id = "qonductor_native_feature_unified_polynomial/fold_0"
    cell = receipt.get("initial_cells", {}).get(cell_id)
    if not isinstance(cell, dict) or cell.get("qa_status") != "PASS":
        raise RuntimeError(f"root QA receipt lacks PASS for {cell_id}")
    if str(cell.get("terminal_status", "")).upper() not in {"PASS", "COMPLETE", "COMPLETED"}:
        raise RuntimeError(f"root QA receipt has nonterminal fold0 status for {cell_id}")
    expected = {
        "artifacts/real_qpu/qonductor_native_features/fold0_receipt.json": digest(OUTPUT / "fold0_receipt.json"),
        "artifacts/real_qpu/qonductor_native_features/fold0_attempts.csv": digest(OUTPUT / "fold0_attempts.csv"),
        "artifacts/real_qpu/qonductor_native_features/fold0_predictions.csv": digest(OUTPUT / "fold0_predictions.csv"),
    }
    listed = {item.get("path"): item.get("sha256") for item in cell.get("files", []) if isinstance(item, dict)}
    for relative_path, observed_sha in expected.items():
        if listed.get(relative_path) != observed_sha:
            raise RuntimeError(f"root fold0 QA receipt does not pin expected artifact {relative_path}")
    fold0 = json.loads((OUTPUT / "fold0_receipt.json").read_text(encoding="utf-8"))
    if fold0.get("status") != "complete_pending_root_qa" or fold0.get("input_signature") != signature:
        raise RuntimeError("fold0 artifact receipt does not bind the current model input signature")
    if fold0.get("files") != {
        "fold0_attempts.csv": expected["artifacts/real_qpu/qonductor_native_features/fold0_attempts.csv"],
        "fold0_predictions.csv": expected["artifacts/real_qpu/qonductor_native_features/fold0_predictions.csv"],
    }:
        raise RuntimeError("fold0 receipt's attempts/predictions hashes do not match their files")
    return receipt


def require_phase_receipts(phase: str, signature: dict[str, Any], authorization_sha256: str) -> None:
    """Require C1–C4 PASS for every phase and root QA for continuation."""
    require_root_technical_gate(authorization_sha256)
    if phase == "continue_after_qa":
        require_root_initial_qa(signature, authorization_sha256)
    elif phase != "initial_fold0":
        raise ValueError(f"unknown execution phase: {phase}")


def materialize_fold0_receipt(output: Path, signature: dict[str, Any], fold_record: dict[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Write immutable fold0-only predictions/attempts and their content receipt."""
    fold0 = [row for row in attempts if int(row["outer_fold"]) == 0]
    predicted = [row for row in fold0 if row["attempt_status"] == "prediction_produced"]
    expected_test_hash = fold_record["test_ids_sha256"]
    observed_test_hash = hashlib.sha256("\n".join(sorted(row["canonical_observation_id"] for row in predicted)).encode()).hexdigest()
    if observed_test_hash != expected_test_hash:
        raise RuntimeError("fold0 attempt IDs differ from the selected model's frozen held-out IDs")
    attempts_path = output / "fold0_attempts.csv"
    predictions_path = output / "fold0_predictions.csv"
    receipt_path = output / "fold0_receipt.json"
    payloads = (
        (attempts_path, csv_payload(fold0, ATTEMPT_FIELDS)),
        (predictions_path, csv_payload(predicted, ATTEMPT_FIELDS)),
    )
    for path, payload in payloads:
        if path.exists():
            if path.read_bytes() != payload:
                raise RuntimeError(f"refuse to overwrite changed immutable fold0 artifact: {path.name}")
        else:
            atomic_write(path, payload)
    receipt = {
        "artifact_id": "qonductor-native-unified-fold0-receipt",
        "status": "complete_pending_root_qa",
        "terminal_status": "completed",
        "fold": 0,
        "input_signature": signature,
        "fit_count": int(fold_record["fit_count"]),
        "selected_degree": int(fold_record["selected_degree"]),
        "eligible_rows": len(predicted),
        "assigned_rows": len(fold0),
        "test_ids_sha256": expected_test_hash,
        "files": {
            "fold0_attempts.csv": digest(attempts_path),
            "fold0_predictions.csv": digest(predictions_path),
        },
    }
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text(encoding="utf-8"))
        if prior != receipt:
            raise RuntimeError("existing fold0 receipt differs; preserve it and investigate instead of overwriting")
    else:
        atomic_write(receipt_path, canonical_json(receipt))
    return receipt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_c3_input_hashes(protocol: dict[str, Any]) -> dict[str, str]:
    """Reconstruct C3's complete pinned input-hash map from active contracts."""
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    authorization = json.loads(AUTH.read_text(encoding="utf-8"))
    if digest(EXECUTION) != authorization.get("governing_contract", {}).get("sha256"):
        raise RuntimeError("execution contract differs from authorized receipt")

    result: dict[str, str] = {}
    for contract_name in ("canonical", "outer", "inner", "representation_rows", "materializer_reference"):
        pin = protocol["input_pins"][contract_name]
        observed = digest(ROOT / pin["path"])
        if observed != pin["sha256"]:
            raise RuntimeError(f"pinned {contract_name} input changed")
        result[contract_name] = observed

    execution_name_map = {
        "seed_registry": "seed_registry",
        "CURRENT": "current_guard",
        "fidelity_registry": "fidelity_guard",
    }
    for manifest_name, execution_pin_name in execution_name_map.items():
        pin = execution["input_pins"][execution_pin_name]
        observed = digest(ROOT / pin["path"])
        if observed != pin["sha256"]:
            raise RuntimeError(f"pinned execution input changed: {execution_pin_name}")
        result[manifest_name] = observed

    for contract_name, pin in execution["method_contracts"].items():
        observed = digest(ROOT / pin["path"])
        if observed != pin["sha256"]:
            raise RuntimeError(f"pinned shared method contract changed: {contract_name}")
        result[f"contract:{contract_name}"] = observed
    archive_pin = protocol["external_assets"]["qonductor_archive"]["sha256"]
    archive_path = ROOT.parent / "Qonductor-SC25/data/database/circuits.zip"
    if digest(archive_path) != archive_pin:
        raise RuntimeError("pinned Qonductor source archive changed or is unavailable")
    result["qonductor_archive"] = archive_pin
    return result


def validate_c3_feature_artifacts(output: Path, protocol: dict[str, Any], feature_rows: list[dict[str, str]]) -> None:
    """Validate the C3 manifest → sidecars → feature-ledger provenance chain."""
    manifest_path = output / "run_manifest.json"
    parity_path = output / "source_parity.json"
    card_path = output / "method_card.json"
    ledger_path = output / "feature_rows.csv"
    for path in (manifest_path, parity_path, card_path, ledger_path):
        if not path.is_file():
            raise RuntimeError(f"complete C3 artifact is required before fitting: missing {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_id") != "qonductor-native-features-unified" or manifest.get("stage") != "feature_materialization_only":
        raise RuntimeError("C3 run manifest has the wrong artifact identity or stage")
    if manifest.get("status") != "pass" or manifest.get("assigned_rows") != 8767 or manifest.get("target_seconds_accessed") is not False:
        raise RuntimeError("C3 run manifest must PASS for all 8,767 rows without target access")
    if manifest.get("feature_order") != FEATURE_NAMES:
        raise RuntimeError("C3 run manifest feature order differs from the active contract")

    expected_inputs = expected_c3_input_hashes(protocol)
    if manifest.get("input_sha256") != expected_inputs:
        raise RuntimeError("C3 run manifest input hashes do not match current protocol/source pins")
    materializer_reference_sha = protocol["input_pins"]["materializer_reference"]["sha256"]
    expected_implementation = {
        "native_feature_materializer": digest(ROOT / "benchmark_v1/scripts/materialize_qonductor_native_features.py"),
        "pinned_source_adapter": digest(ROOT / "benchmark_v1/scripts/qonductor_native_adapter.py"),
        "hardware_wire_adapter": digest(ROOT / "benchmark_v1/scripts/qasm3_hardware_wire_adapter.py"),
        "graph_materializer_reference": materializer_reference_sha,
        "feature_tasks": digest(ROOT / "benchmark_v1/scripts/qonductor_feature_tasks.py"),
        "bounded_feature_pool": digest(ROOT / "benchmark_v1/scripts/bounded_feature_pool.py"),
    }
    if manifest.get("implementation_sha256") != expected_implementation:
        raise RuntimeError("C3 run manifest implementation hashes differ from current code")
    source_pin = protocol["source_pin"]
    source_adapter = manifest.get("source_adapter", {})
    if source_adapter.get("source_revision") != source_pin["revision"] or source_adapter.get("pinned_blobs") != {
        path: pin for path, pin in source_pin["files"].items()
    }:
        raise RuntimeError("C3 run manifest Qonductor source revision/blob pins differ from the contract")

    ledger_sha = digest(ledger_path)
    if manifest.get("feature_rows_sha256") != ledger_sha or parity.get("feature_rows_sha256") != ledger_sha:
        raise RuntimeError("C3 feature-ledger hash does not match its run-manifest/source-parity pins")
    if manifest.get("source_parity_sha256") != digest(parity_path):
        raise RuntimeError("C3 source-parity sidecar hash does not match the run manifest")
    if manifest.get("method_card_sha256") != digest(card_path):
        raise RuntimeError("C3 method-card hash does not match the run manifest")
    checkpoint = manifest.get("checkpoint")
    checkpoint_keys = {"rows_path", "rows_sha256", "manifest_path", "manifest_sha256", "progress_path", "progress_sha256"}
    if not isinstance(checkpoint, dict) or not checkpoint_keys.issubset(checkpoint) or checkpoint != parity.get("checkpoint"):
        raise RuntimeError("C3 run manifest and source-parity checkpoint pins differ")
    if parity.get("status") != "pass" or parity.get("failures") != []:
        raise RuntimeError("C3 source-parity sidecar is not a clean PASS")
    if parity.get("row4477", {}).get("observed_vector") != [133.0, 197.0, 12.0, 8192.0, 1.0]:
        raise RuntimeError("C3 row4477 exact-QASM feature canary differs from the pinned vector")
    expected_source_status = {
        source: {"assigned": count, "available": count, "unavailable": 0}
        for source, count in SOURCES.items()
    }
    if manifest.get("source_status") != expected_source_status or parity.get("source_status") != expected_source_status:
        raise RuntimeError("C3 run manifest/source-parity source coverage is incomplete or inconsistent")
    if manifest.get("split_validation") != {
        "outer_rows": 8767, "inner_rows": 43835,
        "outer_group_leakage": 0, "inner_group_leakage": 0,
    }:
        raise RuntimeError("C3 run manifest does not certify the frozen split identity/leakage checks")
    if len(feature_rows) != 8767 or len({row["canonical_row_id"] for row in feature_rows}) != 8767:
        raise RuntimeError("C3 feature ledger does not contain one unique row per canonical observation")
    if "target_seconds" in feature_rows[0]:
        raise RuntimeError("C3 feature ledger unexpectedly contains target_seconds")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def csv_payload(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    from io import StringIO
    s = StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=fields, extrasaction="raise")
    w.writeheader()
    w.writerows(rows)
    return s.getvalue().encode()


def write_unified_fit_manifest(output: Path, fit_manifest: dict[str, Any]) -> None:
    """Keep the validated feature-materialization receipt immutable."""
    fit_manifest_path = output / "unified_fit_manifest.json"
    payload = canonical_json(fit_manifest)
    if fit_manifest_path.exists() and fit_manifest_path.read_bytes() != payload:
        raise RuntimeError("existing unified-fit manifest differs; preserve it and investigate")
    if not fit_manifest_path.exists():
        atomic_write(fit_manifest_path, payload)


def verify_inputs(output: Path) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, dict[str, str]], dict[tuple[int, str], dict[str, str]], list[dict[str, str]]]:
    protocol_sha = digest(PROTOCOL)
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("method_id") != "qonductor_native_feature_unified_polynomial":
        raise RuntimeError("unexpected Qonductor method contract")
    auth = json.loads(AUTH.read_text())
    auth_pin = auth["method_contracts"]["qonductor_native_features"]
    if auth_pin["path"] != "benchmark_v1/protocol/qonductor_native_features.json" or auth_pin["sha256"] != protocol_sha:
        raise RuntimeError("execution authorization does not pin the current method contract")
    if auth.get("status") != "authorized_implementation_and_execution_subject_to_technical_gates":
        raise RuntimeError("execution authorization is not active")
    for key, path in (
        ("canonical", CANONICAL), ("outer", OUTER), ("inner", INNER),
        ("representation_rows", ROOT / "artifacts/benchmark_v3/real_qpu/compiled_unified/representation_rows.csv"),
        ("materializer_reference", ROOT / "benchmark_v1/scripts/materialize_compiled_unified_qpu_v1.py"),
    ):
        if digest(path) != protocol["input_pins"][key]["sha256"]:
            raise RuntimeError(f"pinned {key} input changed")
    canonical_rows, outer_rows, inner_rows, feature_rows = map(read_csv, (CANONICAL, OUTER, INNER, FEATURES))
    if any(len(x) != 8767 for x in (canonical_rows, outer_rows, feature_rows)) or len(inner_rows) != 43835:
        raise RuntimeError("unexpected canonical/feature/split row count")
    validate_c3_feature_artifacts(output, protocol, feature_rows)
    canon = {r["canonical_row_id"]: r for r in canonical_rows}
    outer = {r["canonical_observation_id"]: r for r in outer_rows}
    features = {r["canonical_row_id"]: r for r in feature_rows}
    if len(canon) != 8767 or set(canon) != set(outer) or set(canon) != set(features):
        raise RuntimeError("canonical, outer-split and feature identities differ")
    counts: dict[str, int] = {}
    for row in canonical_rows:
        counts[row["source_id"]] = counts.get(row["source_id"], 0) + 1
    if counts != SOURCES:
        raise RuntimeError(f"source counts changed: {counts}")
    inner = {}
    for row in inner_rows:
        key = (int(row["outer_fold"]), row["canonical_observation_id"])
        if key in inner:
            raise RuntimeError(f"duplicate inner split key {key}")
        inner[key] = row
    if len(inner) != 43835:
        raise RuntimeError("inner split duplicate identity")
    for row_id, feature in features.items():
        if feature["source_id"] != canon[row_id]["source_id"]:
            raise RuntimeError(f"source mismatch at {row_id}")
        if feature["unified_leakage_group_id"] != outer[row_id]["unified_leakage_group_id"]:
            raise RuntimeError(f"group mismatch at {row_id}")
        if int(feature["outer_fold"]) != int(outer[row_id]["outer_fold"]):
            raise RuntimeError(f"outer-fold mismatch at {row_id}")
        if not math.isfinite(float(canon[row_id]["target_seconds"])):
            raise RuntimeError(f"non-finite target at {row_id}")
        if feature["availability_status"].startswith("available"):
            if any(not math.isfinite(float(feature[n])) for n in FEATURE_NAMES):
                raise RuntimeError(f"non-finite feature at {row_id}")
    for fold in range(5):
        test_groups = {outer[rid]["unified_leakage_group_id"] for rid in canon if int(outer[rid]["outer_fold"]) == fold}
        train_groups = {outer[rid]["unified_leakage_group_id"] for rid in canon if int(outer[rid]["outer_fold"]) != fold}
        if test_groups & train_groups:
            raise RuntimeError(f"outer-fold group leakage in fold {fold}")
    for fold in range(5):
        group_to_inner: dict[str, set[str]] = {}
        for row_id in canon:
            entry = inner.get((fold, row_id))
            if entry is None:
                raise RuntimeError(f"invalid/missing frozen inner assignment {(fold, row_id)}")
            assignment = entry["inner_fold"]
            is_outer_test = int(outer[row_id]["outer_fold"]) == fold
            if is_outer_test:
                if assignment != "":
                    raise RuntimeError(f"outer-test row must have a blank inner assignment: {(fold, row_id)}")
            else:
                if assignment not in {"0", "1", "2", "3"}:
                    raise RuntimeError(f"outer-train row must have inner assignment 0..3: {(fold, row_id, assignment)}")
                group_to_inner.setdefault(outer[row_id]["unified_leakage_group_id"], set()).add(assignment)
        if any(len(assignments) != 1 for assignments in group_to_inner.values()):
            raise RuntimeError(f"frozen inner split leaks groups for outer fold {fold}")
        if {next(iter(assignments)) for assignments in group_to_inner.values()} != {"0", "1", "2", "3"}:
            raise RuntimeError(f"frozen inner split lacks one or more train/validation folds for outer fold {fold}")
    if output.exists() and any((output / name).exists() for name in ("canonical_observations.csv", "unified_outer_split_v2.csv", "unified_inner_split_v2.csv")):
        raise RuntimeError("output folder must not contain mutable copies of frozen input")
    return protocol, canon, outer, inner, feature_rows


def fit_outer_fold(fold: int, rows: list[dict[str, Any]], inner: dict[tuple[int, str], dict[str, str]], threads: int) -> tuple[int, dict[str, float], dict[str, Any]]:
    train = [r for r in rows if r["outer_fold"] != fold]
    test = [r for r in rows if r["outer_fold"] == fold]
    if len(train) < 2 or not test:
        raise RuntimeError(f"empty/too-small outer fold {fold}")
    X = np.asarray([[r[n] for n in FEATURE_NAMES] for r in train], dtype=np.float64)
    y = np.asarray([r["target_seconds"] for r in train], dtype=np.float64)
    candidates = []
    for degree in (2, 3, 4):
        scores = []
        per_inner = []
        for inner_fold in range(4):
            tr = np.asarray([int(inner[(fold, r["canonical_observation_id"])]["inner_fold"]) != inner_fold for r in train])
            va = ~tr
            train_groups = {train[i]["unified_leakage_group_id"] for i in np.flatnonzero(tr)}
            valid_groups = {train[i]["unified_leakage_group_id"] for i in np.flatnonzero(va)}
            if train_groups & valid_groups:
                raise RuntimeError(f"inner group leakage: outer={fold} inner={inner_fold}")
            if np.count_nonzero(va) < 2 or np.ptp(y[va]) == 0:
                raise RuntimeError(f"unavailable_invalid_inner_validation_partition: {fold}/{inner_fold}")
            model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression(fit_intercept=True, positive=False))
            with threadpool_limits(limits=threads):
                model.fit(X[tr], y[tr])
                pred = np.asarray(model.predict(X[va]), dtype=float)
            if not np.all(np.isfinite(pred)):
                raise RuntimeError(f"non-finite inner predictions: {fold}/{inner_fold}/{degree}")
            score = float(r2_score(y[va], pred))
            scores.append(score)
            per_inner.append({"inner_fold": inner_fold, "n_train": int(np.count_nonzero(tr)), "n_valid": int(np.count_nonzero(va)), "r2_seconds": score, "group_overlap": 0})
        candidates.append({"degree": degree, "mean_inner_r2_seconds": float(np.mean(scores)), "folds": per_inner})
    selected = select_degree(candidates)
    degree = int(selected["degree"])
    final = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression(fit_intercept=True, positive=False))
    Xt = np.asarray([[r[n] for n in FEATURE_NAMES] for r in test], dtype=np.float64)
    with threadpool_limits(limits=threads):
        final.fit(X, y)
        pred = np.asarray(final.predict(Xt), dtype=float)
    if not np.all(np.isfinite(pred)):
        raise RuntimeError(f"non-finite outer predictions in fold {fold}")
    linear = final.named_steps["linearregression"]
    singular = np.asarray(linear.singular_, dtype=float)
    record = {
        "outer_fold": fold, "status": "complete", "selected_degree": degree,
        "eligible_train_rows": len(train), "eligible_test_rows": len(test),
        "selection_metric": "unweighted arithmetic mean of four frozen inner-fold raw-seconds R2 scores",
        "candidates": candidates, "fit_count": 13,
        "expanded_feature_names": final.named_steps["polynomialfeatures"].get_feature_names_out(FEATURE_NAMES).tolist(),
        "coefficients": np.asarray(linear.coef_).reshape(-1).tolist(), "intercept": float(linear.intercept_),
        "coefficient_l2_norm": float(np.linalg.norm(linear.coef_)), "design_rank": int(linear.rank_),
        "design_columns": len(linear.coef_), "singular_values": singular.tolist(),
        "condition_number": float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else None,
        "test_ids_sha256": hashlib.sha256("\n".join(sorted(r["canonical_observation_id"] for r in test)).encode()).hexdigest(),
    }
    return degree, {r["canonical_observation_id"]: float(p) for r, p in zip(test, pred, strict=True)}, record


def select_degree(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Highest mean inner R² wins; an exact tie selects the lower degree."""
    if not candidates or any(not math.isfinite(float(r["mean_inner_r2_seconds"])) for r in candidates):
        raise ValueError("candidate scores must be nonempty and finite")
    return sorted(candidates, key=lambda r: (-float(r["mean_inner_r2_seconds"]), int(r["degree"])))[0]


def folds_for_phase(phase: str) -> list[int]:
    if phase == "initial_fold0":
        return [0]
    if phase == "continue_after_qa":
        return [1, 2, 3, 4]
    raise ValueError(f"unknown execution phase: {phase}")


def metric_row(method: str, scope: str, assigned: int, subset: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([float(r["actual_seconds"]) for r in subset])
    p = np.asarray([float(r["predicted_seconds"]) for r in subset])
    absolute = np.abs(y - p)
    return {
        "method_id": method, "scope": scope, "assigned_rows": assigned, "predicted_rows": len(subset),
        "coverage": len(subset) / assigned if assigned else None,
        "mae_seconds": float(mean_absolute_error(y, p)) if len(y) else None,
        "medae_seconds": float(median_absolute_error(y, p)) if len(y) else None,
        "r2_seconds": float(r2_score(y, p)) if len(y) > 1 and np.ptp(y) > 0 else None,
        "log1p_mae": float(np.mean(np.abs(np.log1p(y) - np.log1p(p)))) if len(y) and np.all(y >= 0) and np.all(p >= 0) else None,
        "p90_absolute_error_seconds": float(np.quantile(absolute, .90)) if len(y) else None,
        "p99_absolute_error_seconds": float(np.quantile(absolute, .99)) if len(y) else None,
        "max_absolute_error_seconds": float(np.max(absolute)) if len(y) else None,
        "negative_prediction_count": int(np.count_nonzero(p < 0)),
    }


def aggregate(attempts: list[dict[str, Any]], comparators: list[Path]) -> dict[str, list[dict[str, Any]]]:
    method = "qonductor_native_feature_unified_polynomial"
    predicted = [r for r in attempts if r["predicted_seconds"] != ""]
    metrics = [metric_row(method, "all_sources", len(attempts), predicted)]
    sources = sorted({r["source_id"] for r in attempts})
    source_metrics = [metric_row(method, source, sum(r["source_id"] == source for r in attempts), [r for r in predicted if r["source_id"] == source]) for source in sources]
    coverage: dict[tuple[str, str, str], int] = {}
    for row in attempts:
        key = (row["source_id"], row["availability_status"], row["terminal_reason"])
        coverage[key] = coverage.get(key, 0) + 1
    coverage_rows = [{"source_id": k[0], "availability_status": k[1], "terminal_reason": k[2], "row_count": n} for k, n in sorted(coverage.items())]
    intersections = []
    new_by_id = {r["canonical_observation_id"]: r for r in predicted}
    for path in comparators:
        if not path.exists():
            continue
        old = read_csv(path)
        id_col = "canonical_observation_id" if old and "canonical_observation_id" in old[0] else "canonical_row_id"
        status_col = "attempt_status" if old and "attempt_status" in old[0] else "status"
        by_method: dict[str, dict[str, dict[str, str]]] = {}
        for r in old:
            if r.get(status_col) not in {"prediction_produced", "predicted"} or not r.get("predicted_seconds"):
                continue
            if math.isfinite(float(r["predicted_seconds"])):
                method_id = r.get("method_id", "unknown_method")
                by_method.setdefault(method_id, {})[r[id_col]] = r
        for method_id, old_by_id in sorted(by_method.items()):
            common = sorted(set(new_by_id) & set(old_by_id))
            for scope in ["all_sources", *sources]:
                ids = [i for i in common if scope == "all_sources" or new_by_id[i]["source_id"] == scope]
                if not ids:
                    continue
                new_y = np.asarray([float(new_by_id[i]["actual_seconds"]) for i in ids])
                old_y = np.asarray([float(old_by_id[i]["actual_seconds"]) for i in ids])
                if not np.array_equal(new_y, old_y):
                    raise RuntimeError(f"paired-target mismatch against {path}/{method_id}")
                npred = np.asarray([float(new_by_id[i]["predicted_seconds"]) for i in ids])
                opred = np.asarray([float(old_by_id[i]["predicted_seconds"]) for i in ids])
                nm = metric_row(method, scope, len(ids), [{"actual_seconds": a, "predicted_seconds": p} for a, p in zip(new_y, npred)])
                om = metric_row("comparator", scope, len(ids), [{"actual_seconds": a, "predicted_seconds": p} for a, p in zip(old_y, opred)])
                intersections.append({
                    "comparator_artifact": path.relative_to(ROOT).as_posix(), "comparator_sha256": digest(path),
                    "comparator_method_ids": method_id, "scope": scope, "common_rows": len(ids),
                    "qonductor_mae_seconds": nm["mae_seconds"], "comparator_mae_seconds": om["mae_seconds"],
                    "mae_delta_qonductor_minus_comparator_seconds": nm["mae_seconds"] - om["mae_seconds"],
                    "qonductor_r2_seconds": nm["r2_seconds"], "comparator_r2_seconds": om["r2_seconds"],
                })
    return {"metrics": metrics, "source_metrics": source_metrics, "coverage": coverage_rows, "common_intersections": intersections}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--phase", choices=("initial_fold0", "continue_after_qa"), default="initial_fold0",
        help="Default is fold0 only. Folds1–4 require a root-issued initial-cell QA receipt.",
    )
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        raise SystemExit("--threads must be 1..4")
    if args.output_dir.resolve() != OUTPUT.resolve():
        raise RuntimeError("Qonductor unified experiment output path is protocol-pinned")
    protocol, canonical, outer, inner, feature_rows = verify_inputs(args.output_dir)
    auth_sha = digest(AUTH)
    inputs = {
        "protocol_sha256": digest(PROTOCOL), "authorization_sha256": auth_sha,
        "canonical_sha256": digest(CANONICAL), "outer_sha256": digest(OUTER),
        "inner_sha256": digest(INNER), "features_sha256": digest(FEATURES),
    }
    comparators = [p for p in COMPARATORS if p.is_file()]
    signature = {"method_id": protocol["method_id"], "inputs": inputs, "feature_order": FEATURE_NAMES,
                 "threads": args.threads, "fit_script_sha256": digest(Path(__file__)),
                 "comparators": {p.relative_to(ROOT).as_posix(): digest(p) for p in comparators}}
    # No fold may start from execution authorization alone: root must have
    # independently signed the technical C1–C4 gates against current pins.
    require_phase_receipts(args.phase, signature, auth_sha)
    fold_path, attempts_path = args.output_dir / "fold_manifests.json", args.output_dir / "attempts.csv"
    prior = json.loads(fold_path.read_text()) if fold_path.exists() else {
        "schema": "qonductor-native-unified-fold-checkpoint-v1", "input_signature": signature,
        "completed_folds": [], "folds": [], "expected_total_fits": 65,
    }
    if prior.get("input_signature") != signature:
        raise RuntimeError("resume checkpoint has different pinned inputs")
    done = set(int(x) for x in prior["completed_folds"])
    attempts = read_csv(attempts_path) if attempts_path.exists() else []
    attempts = [r for r in attempts if int(r["outer_fold"]) in done]
    expected_attempts = sum(sum(int(row["outer_fold"]) == fold for row in outer.values()) for fold in done)
    if len(attempts) != expected_attempts:
        raise RuntimeError("resume checkpoint has incomplete/duplicate row attempts")
    if args.phase == "initial_fold0" and any(fold != 0 for fold in done):
        raise RuntimeError("initial_fold0 mode refuses a checkpoint containing folds1–4")
    if args.phase == "continue_after_qa":
        if 0 not in done or not prior.get("folds"):
            raise RuntimeError("continue_after_qa requires a completed fold0 checkpoint")
        fold0_rows = [row for row in attempts if int(row["outer_fold"]) == 0]
        fold0_predictions = [row for row in fold0_rows if row["attempt_status"] == "prediction_produced"]
        if csv_payload(fold0_rows, ATTEMPT_FIELDS) != (OUTPUT / "fold0_attempts.csv").read_bytes():
            raise RuntimeError("mutable attempts.csv fold0 rows differ from root-reviewed immutable fold0 attempts")
        if csv_payload(fold0_predictions, ATTEMPT_FIELDS) != (OUTPUT / "fold0_predictions.csv").read_bytes():
            raise RuntimeError("mutable attempts.csv fold0 predictions differ from root-reviewed immutable predictions")

    canonical_rows = []
    for row in feature_rows:
        row_id = row["canonical_row_id"]
        status = row["availability_status"]
        canonical_rows.append({
            "canonical_observation_id": row_id, "source_id": row["source_id"],
            "outer_fold": int(outer[row_id]["outer_fold"]),
            "unified_leakage_group_id": outer[row_id]["unified_leakage_group_id"],
            "availability_status": status,
            "terminal_reason": row["terminal_reason"],
            "target_seconds": float(canonical[row_id]["target_seconds"]),
            **{name: float(row[name]) if status.startswith("available") else math.nan for name in FEATURE_NAMES},
        })
    eligible = [r for r in canonical_rows if r["availability_status"].startswith("available")]
    fold_records = list(prior["folds"])

    requested_folds = folds_for_phase(args.phase)
    for fold in requested_folds:
        if fold in done:
            continue
        degree, predictions, record = fit_outer_fold(fold, eligible, inner, args.threads)
        rows_for_fold = []
        for row in canonical_rows:
            if row["outer_fold"] != fold:
                continue
            pred = predictions.get(row["canonical_observation_id"])
            actual = row["target_seconds"]
            rows_for_fold.append({
                "canonical_observation_id": row["canonical_observation_id"], "source_id": row["source_id"],
                "outer_fold": fold, "unified_leakage_group_id": row["unified_leakage_group_id"],
                "availability_status": row["availability_status"],
                "terminal_reason": row["terminal_reason"],
                "attempt_status": "prediction_produced" if pred is not None else "unavailable",
                "selected_degree": degree if pred is not None else "",
                "actual_seconds": actual, "predicted_seconds": "" if pred is None else pred,
                "absolute_error_seconds": "" if pred is None else abs(actual - pred),
                "negative_prediction": "" if pred is None else str(pred < 0).lower(),
            })
        attempts = [r for r in attempts if int(r["outer_fold"]) != fold] + rows_for_fold
        attempts.sort(key=lambda r: r["canonical_observation_id"])
        atomic_write(attempts_path, csv_payload(attempts, ATTEMPT_FIELDS))
        fold_records = [r for r in fold_records if int(r["outer_fold"]) != fold] + [record]
        fold_records.sort(key=lambda r: int(r["outer_fold"]))
        done.add(fold)
        prior.update({"completed_folds": sorted(done), "folds": fold_records, "completed_fit_count": sum(int(r["fit_count"]) for r in fold_records),
                      "status": "complete" if len(done) == 5 else "partial_resume_safe", "threads": args.threads})
        atomic_write(fold_path, canonical_json(prior))
        print(json.dumps({"completed_folds": sorted(done), "completed_fit_count": prior["completed_fit_count"]}, sort_keys=True), flush=True)

    if args.phase == "initial_fold0":
        if 0 not in done:
            raise RuntimeError("initial fold0 phase did not complete fold0")
        fold0_record = next(row for row in fold_records if int(row["outer_fold"]) == 0)
        fold0_receipt = materialize_fold0_receipt(args.output_dir, signature, fold0_record, attempts)
        print(json.dumps({
            "status": "fold0_complete_pending_root_qa",
            "completed_folds": sorted(done),
            "completed_fit_count": prior.get("completed_fit_count"),
            "fold0_receipt_sha256": digest(args.output_dir / "fold0_receipt.json"),
            "fold0_attempts_sha256": fold0_receipt["files"]["fold0_attempts.csv"],
            "fold0_predictions_sha256": fold0_receipt["files"]["fold0_predictions.csv"],
        }, sort_keys=True))
        return 0

    if len(done) != 5 or len(attempts) != 8767:
        raise RuntimeError("final run requires five folds and 8,767 row attempts")
    summaries = aggregate(attempts, comparators)
    for name, rows in summaries.items():
        fields = list(rows[0]) if rows else {
            "coverage": ["source_id", "availability_status", "terminal_reason", "row_count"],
            "common_intersections": ["comparator_artifact", "comparator_sha256", "comparator_method_ids", "scope", "common_rows", "qonductor_mae_seconds", "comparator_mae_seconds", "mae_delta_qonductor_minus_comparator_seconds", "qonductor_r2_seconds", "comparator_r2_seconds"],
        }[name]
        atomic_write(args.output_dir / f"{name}.csv", csv_payload(rows, fields))
    prior.update({
        "status": "complete", "assigned_rows": 8767,
        "predicted_rows": sum(r["predicted_seconds"] != "" for r in attempts),
        "completed_fit_count": 65, "evaluation_target_clock": protocol["evaluation_target_clock"],
        "method_output_clock": protocol["method_output_clock"], "negative_predictions_preserved": True,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "output_hashes": {name: digest(args.output_dir / name) for name in ("attempts.csv", "metrics.csv", "source_metrics.csv", "coverage.csv", "common_intersections.csv")},
    })
    atomic_write(fold_path, canonical_json(prior))
    run_manifest_path = args.output_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise RuntimeError("C3 run_manifest.json is required before training completion")
    fit_manifest = {
        "artifact_id": "qonductor-native-unified-fit",
        "feature_run_manifest_sha256": digest(run_manifest_path),
        "method_id": protocol["method_id"],
        "status": "complete",
        "fit_script": "benchmark_v1/scripts/run_qonductor_native_unified.py",
        "fit_script_sha256": digest(Path(__file__)),
        "input_signature": signature,
        "selected_degrees_by_outer_fold": {str(r["outer_fold"]): r["selected_degree"] for r in fold_records},
        "fit_count": 65,
        "attempts_rows": len(attempts),
        "predicted_rows": prior["predicted_rows"],
        "output_hashes": prior["output_hashes"],
    }
    write_unified_fit_manifest(args.output_dir, fit_manifest)
    print(json.dumps({"status": "complete", "predicted_rows": prior["predicted_rows"], "assigned_rows": 8767}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
