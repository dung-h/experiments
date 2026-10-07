#!/usr/bin/env python3
"""Run the frozen CPU classical predictors for CUDA-Q dense R8.

This runner emits one out-of-fold attempt row per method and held-out exact
QASM hash for one precision/fold.  It never measures a simulator and never
uses the held-out target to fit a model.  The full-data path is deliberately
refused unless the exact pinned R8 CPU environment is active.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "benchmark_v1/protocol/cudaq_dense_statevector_predictor_comparison_v1.json"
PROTOCOL_SHA256 = "7598f38996c9e872245be503ac21f237d4aae0c95182127fa3ef3f3b03520258"
PANEL_PATH = ROOT / "artifacts/benchmark_v1/sim_common_q16_manifest_20260927/sim_common_q16_manifest.csv"
TARGET_DIR = ROOT / "work/benchmark_recovery/r5_dense/targets_recovered_v2"
TARGET_MANIFEST_PATH = TARGET_DIR / "manifest.json"
TARGET_PATH = TARGET_DIR / "dense_hash_targets.csv"
TARGET_COVERAGE_PATH = TARGET_DIR / "dense_member_coverage.csv"
FEATURE_DIR = ROOT / "work/benchmark_recovery/r3_dense/ir_features_v1"
FEATURE_MANIFEST_PATH = FEATURE_DIR / "manifest.json"
FEATURE_JSONL_PATH = FEATURE_DIR / "dense_features_by_hash.jsonl"
FEATURE_MEMBERS_PATH = FEATURE_DIR / "dense_feature_members.csv"
IR_PATH = FEATURE_DIR / "dense_ir_by_hash.csv"
FOLDS_PATH = ROOT / "artifacts/benchmark_v1/c44_aer_q16_full_panel_evaluation_20260927/aer_q16_reduced_warm.csv"
DEFAULT_ATTEMPT_ROOT = ROOT / "work/benchmark_recovery/r8_dense_predictors/attempt_001"

EXPECTED_INPUTS = {
    "r3_ir_manifest": "d9a35d022df246c381c098a11ff405d74b5cbbd416ac9b62c126f48616e3bad2",
    "r3_feature_jsonl": "cb8d92a255c218fd804a4a960dda78979699ac94ff1653b39c24fd6dc6f66def",
    "r3_ir_csv": "68c7cd713d854814e258701b973c43e8802c24a09b7cf6908ec3c324e6e5609c",
    "r5_manifest": "bfc22149ecc05a85a23598f20fb46cff0e0eb17479a4a5b7f8d35ab8707ca200",
    "r5_hash_targets": "1b88fef5015d5403f89956d8f6d6f92baced9aaecc149728e641cee10588c170",
    "r5_member_coverage": "93bef4133cc8043b7aa59ca540861d61370b2fa49501e5b40e4be94fb757c901",
    "c44_hash_assignment": "a25e2e740fd3058e487f713cd0fc074dd8761120a5aface275a212693ac05437",
    "panel_manifest_sha256": "9b33added9a2d94f9f621021cec16362561fe58454e233c495e25568a666fb8e",
    "r3_feature_members": "92756a674809cb6a0085b5cd1e6aa0118ff25d0140c248eaa79efdf94a130056",
}
PRECISIONS = ("fp32", "fp64")
FOLDS = (0, 1, 2, 3, 4)
FOLD_COUNTS = {0: 37, 1: 28, 2: 26, 3: 25, 4: 34}
GLOBAL_FEATURES = (
    "active_width", "structural_depth", "one_qubit_count", "two_qubit_count",
    "swap_like_count", "measurement_count", "shots",
)
CLASSICAL_METHOD_IDS = (
    "outer_train_median", "linear_regression", "ridge_alpha1", "rbf_svr",
    "random_forest", "xgboost", "work_scaled_ridge",
)
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)
THREAD_CAP = 2
NON_GATE_NODES = {"measure", "barrier", "snapshot"}
ATTEMPT_FIELDS = [
    "context_id", "precision", "method_id", "fold", "qasm_sha256",
    "assigned_core_oof", "target_status", "target_seconds", "prediction_seconds",
    "status", "terminal_reason", "train_hash_count", "test_hash_count",
    "train_hashes_sha256", "test_hashes_sha256", "model_params_json",
    "transform_contract", "seed", "prediction_aggregation", "attempt_terminal",
]


class ContractError(ValueError):
    """An input or execution contract differs from the frozen R8 protocol."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def hash_sorted_ids(values: Sequence[str]) -> str:
    return sha256_text("".join(f"{value}\n" for value in sorted(values)))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ContractError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite_number(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ContractError(f"{name} is not finite: {value!r}")
    return number


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != PROTOCOL_SHA256:
        raise ContractError(f"frozen R8 protocol missing/hash mismatch: {path}")
    protocol = read_json(path)
    if protocol.get("protocol_id") != "cudaq-dense-statevector-nine-predictor-comparison-v1":
        raise ContractError("unexpected R8 protocol identity")
    if protocol.get("status") != "frozen_for_preflight_and_execution":
        raise ContractError("R8 protocol is not frozen")
    return protocol


def _expected_files() -> dict[str, tuple[Path, str]]:
    return {
        "protocol": (PROTOCOL_PATH, PROTOCOL_SHA256),
        "panel_manifest_sha256": (PANEL_PATH, EXPECTED_INPUTS["panel_manifest_sha256"]),
        "r3_ir_manifest": (FEATURE_MANIFEST_PATH, EXPECTED_INPUTS["r3_ir_manifest"]),
        "r3_feature_jsonl": (FEATURE_JSONL_PATH, EXPECTED_INPUTS["r3_feature_jsonl"]),
        "r3_ir_csv": (IR_PATH, EXPECTED_INPUTS["r3_ir_csv"]),
        "r3_feature_members": (FEATURE_MEMBERS_PATH, EXPECTED_INPUTS["r3_feature_members"]),
        "r5_manifest": (TARGET_MANIFEST_PATH, EXPECTED_INPUTS["r5_manifest"]),
        "r5_hash_targets": (TARGET_PATH, EXPECTED_INPUTS["r5_hash_targets"]),
        "r5_member_coverage": (TARGET_COVERAGE_PATH, EXPECTED_INPUTS["r5_member_coverage"]),
        "c44_hash_assignment": (FOLDS_PATH, EXPECTED_INPUTS["c44_hash_assignment"]),
    }


def _count_work_scaled_multiqubit_nodes(feature: dict[str, Any]) -> int:
    dag = feature.get("dag") or {}
    names = dag.get("names") or []
    arities = dag.get("arity") or []
    if len(names) != len(arities) or int(dag.get("operation_count", -1)) != len(names):
        raise ContractError(f"malformed operation DAG for {feature.get('qasm_sha256')}")
    return sum(
        1 for name, arity in zip(names, arities)
        if str(name) not in NON_GATE_NODES and int(arity) >= 3
    )


def _feature_values(feature: dict[str, Any]) -> dict[str, float]:
    raw = feature.get("global_features")
    if not isinstance(raw, dict):
        raise ContractError(f"missing global feature object for {feature.get('qasm_sha256')}")
    values = {name: _finite_number(raw.get(name), name=f"{name}:{feature.get('qasm_sha256')}") for name in GLOBAL_FEATURES}
    if any(value < 0 for value in values.values()):
        raise ContractError(f"negative global feature for {feature.get('qasm_sha256')}")
    if int(values["shots"]) != 32:
        raise ContractError(f"dense feature shots must equal 32 for {feature.get('qasm_sha256')}")
    if int(values["active_width"]) < 1:
        raise ContractError(f"active width must be positive for {feature.get('qasm_sha256')}")
    return values


def validate_inputs() -> dict[str, Any]:
    """Load and verify the complete pinned 204/191 panel and 150-hash folds."""
    protocol = load_protocol()
    actual_hashes: dict[str, str] = {}
    for name, (path, expected) in _expected_files().items():
        if not path.is_file():
            raise ContractError(f"required R8 input is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ContractError(f"R8 input hash mismatch for {name}: expected={expected} actual={actual}")
        actual_hashes[name] = actual

    r3_manifest = read_json(FEATURE_MANIFEST_PATH)
    r5_manifest = read_json(TARGET_MANIFEST_PATH)
    expected_revisions = protocol["population"]["target_revisions"]
    if r3_manifest.get("status") != "PASS" or r3_manifest.get("artifact_id") != "cudaq-dense-normalized-ir-features-v1":
        raise ContractError("R3 feature materialization is not the expected PASS artifact")
    if r3_manifest.get("timing_performed") is not False or r3_manifest.get("target_read") is not False:
        raise ContractError("R3 features are not certified target-free CPU materialization")
    for key, expected in (
        ("dense_features_by_hash.jsonl", expected_revisions["r3_feature_jsonl"]),
        ("dense_ir_by_hash.csv", expected_revisions["r3_ir_csv"]),
        ("dense_feature_members.csv", EXPECTED_INPUTS["r3_feature_members"]),
    ):
        if (r3_manifest.get("outputs") or {}).get(key) != expected:
            raise ContractError(f"R3 manifest output pin mismatch: {key}")
    if int(r3_manifest.get("panel_members", -1)) != 204 or int(r3_manifest.get("unique_qasm_hashes", -1)) != 191:
        raise ContractError("R3 feature manifest does not identify the frozen 204/191 panel")

    if r5_manifest.get("status") != "PASS" or r5_manifest.get("artifact_id") != "cudaq-dense-hash-targets-v1":
        raise ContractError("R5 target materialization is not the expected PASS artifact")
    if r5_manifest.get("timing_performed") is not False or r5_manifest.get("model_fitting_performed") is not False:
        raise ContractError("R5 target materialization has unexpected timing/model-fitting state")
    if r5_manifest.get("measurement_context_sha256") != protocol["population"]["measurement_context_sha256"]:
        raise ContractError("R5 dense target context differs from the frozen protocol")
    for key, expected in (
        ("dense_hash_targets.csv", expected_revisions["r5_hash_targets"]),
        ("dense_member_coverage.csv", expected_revisions["r5_member_coverage"]),
    ):
        if (r5_manifest.get("outputs") or {}).get(key) != expected:
            raise ContractError(f"R5 manifest output pin mismatch: {key}")
    r5_inputs = r5_manifest.get("inputs") or {}
    if (r5_inputs.get("ir_artifact_manifest") or {}).get("sha256") != expected_revisions["r3_ir_manifest"]:
        raise ContractError("R5 target manifest does not pin the frozen R3 IR-feature manifest")
    if (r5_inputs.get("ir_manifest") or {}).get("sha256") != expected_revisions["r3_ir_csv"]:
        raise ContractError("R5 target manifest does not pin the frozen R3 IR table")

    panel = read_csv(PANEL_PATH)
    if len(panel) != 204 or len({row["panel_member_id"] for row in panel}) != 204:
        raise ContractError("panel must contain 204 distinct members")
    if len({row["qasm_sha256"] for row in panel}) != 191:
        raise ContractError("panel must contain 191 exact-QASM hashes")
    if any(row["panel_id"] != protocol["population"]["panel_id"] for row in panel):
        raise ContractError("unexpected simulator panel id")
    panel_by_member = {row["panel_member_id"]: row for row in panel}
    panel_by_hash = {row["qasm_sha256"]: row for row in panel}

    feature_rows = read_jsonl(FEATURE_JSONL_PATH)
    features: dict[str, dict[str, Any]] = {}
    for feature in feature_rows:
        digest = str(feature.get("qasm_sha256", ""))
        if digest in features or digest not in panel_by_hash:
            raise ContractError(f"duplicate or out-of-panel R3 feature identity: {digest}")
        _feature_values(feature)
        _count_work_scaled_multiqubit_nodes(feature)
        features[digest] = feature
    if set(features) != set(panel_by_hash):
        raise ContractError("R3 feature hashes do not exactly cover the 191-hash panel")
    multiq_total = sum(_count_work_scaled_multiqubit_nodes(feature) for feature in features.values())
    if multiq_total != 124:
        raise ContractError(f"frozen work-scaled arity>=3 node count changed: {multiq_total} != 124")

    ir_rows = read_csv(IR_PATH)
    ir_by_hash = {row["qasm_sha256"]: row for row in ir_rows}
    if len(ir_rows) != 191 or len(ir_by_hash) != 191 or set(ir_by_hash) != set(features):
        raise ContractError("R3 IR CSV does not exactly cover the 191 feature hashes")
    for digest, feature in features.items():
        ir = ir_by_hash[digest]
        if ir.get("normalized_ir_sha256") != feature.get("normalized_ir_sha256"):
            raise ContractError(f"R3 normalized IR join mismatch: {digest}")
        if ir.get("unitary_sha256") != feature.get("unitary_sha256"):
            raise ContractError(f"R3 unitary join mismatch: {digest}")

    feature_members = read_csv(FEATURE_MEMBERS_PATH)
    if len(feature_members) != 204 or {row["panel_member_id"] for row in feature_members} != set(panel_by_member):
        raise ContractError("R3 feature member coverage does not exactly cover all 204 panel members")
    for row in feature_members:
        panel_row = panel_by_member[row["panel_member_id"]]
        digest = row["qasm_sha256"]
        feature = features.get(digest)
        if not feature or digest != panel_row["qasm_sha256"]:
            raise ContractError(f"R3 feature member hash mismatch: {row['panel_member_id']}")
        if row.get("normalized_ir_sha256") != feature.get("normalized_ir_sha256") or row.get("unitary_sha256") != feature.get("unitary_sha256"):
            raise ContractError(f"R3 alias member IR/unitary mismatch: {row['panel_member_id']}")

    coverage = read_csv(TARGET_COVERAGE_PATH)
    coverage_keys = {(row["panel_member_id"], row["precision"]) for row in coverage}
    expected_coverage = {(row["panel_member_id"], precision) for row in panel for precision in PRECISIONS}
    if len(coverage) != 408 or coverage_keys != expected_coverage:
        raise ContractError("R5 member coverage must retain each of 204 members in both precisions")

    c44_rows = read_csv(FOLDS_PATH)
    fold_by_hash: dict[str, int] = {}
    for row in c44_rows:
        if row.get("stratum") != "core_q2_q9":
            continue
        digest = row.get("source_sha256", "")
        fold = int(row["fold"])
        if digest in fold_by_hash and fold_by_hash[digest] != fold:
            raise ContractError(f"C44 exact-QASM hash has conflicting folds: {digest}")
        fold_by_hash[digest] = fold
    if len(fold_by_hash) != 150 or dict(sorted(__import__("collections").Counter(fold_by_hash.values()).items())) != FOLD_COUNTS:
        raise ContractError("C44 split must cover 150 exact hashes with frozen fold counts")

    targets = read_csv(TARGET_PATH)
    target_keys = {(row["qasm_sha256"], row["precision"]) for row in targets}
    if len(targets) != 382 or len(target_keys) != 382:
        raise ContractError("R5 targets must have exactly one row for each hash and precision")
    per_precision: dict[str, list[dict[str, Any]]] = {precision: [] for precision in PRECISIONS}
    for row in targets:
        if row["precision"] not in PRECISIONS:
            raise ContractError(f"unexpected dense target precision {row['precision']}")
        if row["context_sha256"] != protocol["population"]["measurement_context_sha256"]:
            raise ContractError(f"R5 target measurement context mismatch: {row['qasm_sha256']} {row['precision']}")
        if row["stratum"] != protocol["population"]["primary_oof_stratum"]:
            continue
        digest = row["qasm_sha256"]
        if row.get("target_status") != "ok" or not _parse_bool(row.get("assigned_core_oof")):
            raise ContractError(f"primary R8 target is not eligible without imputation: {digest}/{row['precision']}")
        target = _finite_number(row.get("target_seconds"), name=f"target_seconds:{digest}/{row['precision']}")
        if target <= 0:
            raise ContractError(f"primary R8 target must be strictly positive: {digest}/{row['precision']}")
        feature = features.get(digest)
        if not feature or row.get("normalized_ir_sha256") != feature.get("normalized_ir_sha256"):
            raise ContractError(f"R5/R3 normalized IR mismatch: {digest}/{row['precision']}")
        if row.get("unitary_sha256") != feature.get("unitary_sha256"):
            raise ContractError(f"R5/R3 unitary mismatch: {digest}/{row['precision']}")
        if int(row["width_qubits"]) != int(_feature_values(feature)["active_width"]):
            raise ContractError(f"core active width differs from frozen circuit width: {digest}")
        if digest not in fold_by_hash or int(row["outer_fold"]) != fold_by_hash[digest]:
            raise ContractError(f"R5 target fold differs from C44 exact-hash fold: {digest}")
        target_record = {
            "qasm_sha256": digest,
            "precision": row["precision"],
            "fold": fold_by_hash[digest],
            "target_seconds": target,
            "target_status": row["target_status"],
            "features": _feature_values(feature),
            "multiq_count": _count_work_scaled_multiqubit_nodes(feature),
            "normalized_ir_sha256": row["normalized_ir_sha256"],
            "unitary_sha256": row["unitary_sha256"],
        }
        per_precision[row["precision"]].append(target_record)

    hash_sets: dict[str, set[str]] = {}
    for precision in PRECISIONS:
        rows = per_precision[precision]
        hashes = {row["qasm_sha256"] for row in rows}
        hash_sets[precision] = hashes
        counts = {fold: sum(row["fold"] == fold for row in rows) for fold in FOLDS}
        if len(rows) != 150 or len(hashes) != 150 or counts != FOLD_COUNTS:
            raise ContractError(f"{precision} primary OOF rows do not match the 150-hash frozen assignment: {counts}")
    if hash_sets["fp32"] != hash_sets["fp64"]:
        raise ContractError("FP32 and FP64 do not use identical exact-QASM hash identities")

    return {
        "protocol": protocol,
        "protocol_sha256": PROTOCOL_SHA256,
        "input_hashes": actual_hashes,
        "multiq_total": multiq_total,
        "panel_member_count": len(panel),
        "panel_unique_hash_count": len(features),
        "fold_counts": dict(FOLD_COUNTS),
        "rows_by_precision": {
            precision: sorted(rows, key=lambda row: row["qasm_sha256"])
            for precision, rows in per_precision.items()
        },
    }


def expected_versions() -> dict[str, str]:
    env = load_protocol()["execution"]["environment"]
    return {
        "python": str(env["python"]),
        "numpy": str(env["numpy"]),
        "scikit-learn": str(env["scikit_learn"]),
        "xgboost": str(env["xgboost"]),
    }


def inspect_environment() -> dict[str, str]:
    actual = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "xgboost": importlib.metadata.version("xgboost"),
        "threadpoolctl": importlib.metadata.version("threadpoolctl"),
    }
    return actual


def validate_execution_environment(actual: dict[str, str] | None = None) -> dict[str, str]:
    actual = actual or inspect_environment()
    expected = expected_versions()
    if not actual["python"].startswith("3.10."):
        raise ContractError(f"R8 requires Python 3.10.x; found {actual['python']}")
    for package in ("numpy", "scikit-learn", "xgboost"):
        if actual.get(package) != expected[package]:
            raise ContractError(f"R8 requires {package}=={expected[package]}; found {actual.get(package, 'missing')}")
    return actual


def set_native_thread_environment() -> None:
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(THREAD_CAP)


def protocol_method_parameters(protocol: dict[str, Any], method_id: str) -> dict[str, Any]:
    try:
        method = protocol["methods"][method_id]
    except KeyError as exc:
        raise ContractError(f"method missing from frozen R8 protocol: {method_id}") from exc
    if method.get("id") != method_id:
        raise ContractError(f"protocol method id mismatch: {method_id}")
    return method


def _import_ml_dependencies() -> dict[str, Any]:
    """Import CPU ML dependencies only after the pinned environment passed."""
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from threadpoolctl import threadpool_info, threadpool_limits
    from xgboost import XGBRegressor

    return {
        "np": np,
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "SVR": SVR,
        "RandomForestRegressor": RandomForestRegressor,
        "XGBRegressor": XGBRegressor,
        "StandardScaler": StandardScaler,
        "threadpool_info": threadpool_info,
        "threadpool_limits": threadpool_limits,
    }


def make_estimator(method_id: str, protocol: dict[str, Any], deps: dict[str, Any]) -> Any:
    method = protocol_method_parameters(protocol, method_id)
    estimator_spec = method.get("estimator")
    if not isinstance(estimator_spec, dict):
        raise ContractError(f"method has no estimator specification: {method_id}")
    params = dict(estimator_spec)
    params.pop("class", None)
    classes = {
        "linear_regression": deps["LinearRegression"],
        "ridge_alpha1": deps["Ridge"],
        "rbf_svr": deps["SVR"],
        "random_forest": deps["RandomForestRegressor"],
        "xgboost": deps["XGBRegressor"],
        "work_scaled_ridge": deps["Ridge"],
    }
    try:
        estimator = classes[method_id](**params)
    except KeyError as exc:
        raise ContractError(f"no classical estimator implementation for {method_id}") from exc
    return estimator


def _rows_to_matrix(rows: Sequence[dict[str, Any]], feature_name: str, np: Any) -> Any:
    return np.asarray([[float(row["features"][name]) for name in GLOBAL_FEATURES] for row in rows], dtype=np.float64)


def _scaled_train_test(
    train_values: Any, test_values: Any, *, scaler_class: Any, np: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    scaler = scaler_class(copy=True, with_mean=True, with_std=True)
    scaler.fit(train_values)
    fitted_std = np.std(train_values, axis=0, ddof=0)
    scale = np.asarray(scaler.scale_, dtype=np.float64).copy()
    scale[fitted_std < 1e-6] = 1.0
    train_scaled = (train_values - np.asarray(scaler.mean_, dtype=np.float64)) / scale
    test_scaled = (test_values - np.asarray(scaler.mean_, dtype=np.float64)) / scale
    state = {
        "mean": [float(value) for value in scaler.mean_],
        "std_before_constant_guard": [float(value) for value in fitted_std],
        "scale_after_constant_guard": [float(value) for value in scale],
        "constant_std_threshold": 1e-6,
        "constant_std_replacement": 1.0,
    }
    return train_scaled, test_scaled, state


def fit_global_transform(
    train_rows: Sequence[dict[str, Any]], test_rows: Sequence[dict[str, Any]], deps: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    """Apply log1p globals and a StandardScaler fit on outer-train only."""
    np = deps["np"]
    x_train = _rows_to_matrix(train_rows, "features", np)
    x_test = _rows_to_matrix(test_rows, "features", np)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all() or (x_train < 0).any() or (x_test < 0).any():
        raise ContractError("global features must be finite and nonnegative before log1p")
    logged_train = np.log1p(x_train)
    logged_test = np.log1p(x_test)
    train_scaled, test_scaled, state = _scaled_train_test(
        logged_train, logged_test, scaler_class=deps["StandardScaler"], np=np,
    )
    state["transform"] = "log1p_nonnegative_globals_then_outer_train_standard_scaler"
    return train_scaled, test_scaled, state


def _work_scaled_matrix(rows: Sequence[dict[str, Any]], np: Any) -> Any:
    values = []
    for row in rows:
        features = row["features"]
        n = int(features["active_width"])
        state_count = float(2 ** n)
        one_q = float(features["one_qubit_count"])
        two_q = float(features["two_qubit_count"])
        multi_q = float(row["multiq_count"])
        shots = float(features["shots"])
        values.append([state_count, state_count * one_q, state_count * two_q, state_count * multi_q, shots])
    matrix = np.asarray(values, dtype=np.float64)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ContractError("work-scaled features must be finite and nonnegative")
    return matrix


def fit_work_transform(
    train_rows: Sequence[dict[str, Any]], test_rows: Sequence[dict[str, Any]], deps: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    np = deps["np"]
    train_raw = _work_scaled_matrix(train_rows, np)
    test_raw = _work_scaled_matrix(test_rows, np)
    train_scaled, test_scaled, state = _scaled_train_test(
        train_raw, test_raw, scaler_class=deps["StandardScaler"], np=np,
    )
    state["feature_order"] = ["2^n", "2^n*one_qubit_count", "2^n*two_qubit_count", "2^n*work_scaled_multiqubit_count", "shots"]
    state["n_definition"] = "active_width"
    state["transform"] = "outer_train_standard_scaler_on_raw_work_features"
    return train_scaled, test_scaled, state


def _prediction_values(raw_prediction: Any, *, log_target: bool, np: Any) -> tuple[list[float | None], list[str]]:
    values = np.asarray(raw_prediction, dtype=np.float64).reshape(-1)
    predicted: list[float | None] = []
    reasons: list[str] = []
    for value in values:
        if not np.isfinite(value):
            predicted.append(None)
            reasons.append("nonfinite_model_output")
            continue
        if log_target:
            with np.errstate(over="ignore", invalid="ignore"):
                decoded = float(np.expm1(value))
        else:
            decoded = float(value)
        if not math.isfinite(decoded):
            predicted.append(None)
            reasons.append("nonfinite_inverse_target_transform")
            continue
        predicted.append(max(0.0, decoded))
        reasons.append("")
    return predicted, reasons


def fit_predict_method(
    method_id: str,
    train_rows: Sequence[dict[str, Any]],
    train_targets: Sequence[float],
    test_rows: Sequence[dict[str, Any]],
    protocol: dict[str, Any],
    deps: dict[str, Any],
) -> tuple[list[float | None], list[str], dict[str, Any]]:
    """Fit using training labels only; test records contain features, not labels."""
    if len(train_rows) != len(train_targets) or not train_rows or not test_rows:
        raise ContractError("fit input has empty or inconsistent outer train/test arrays")
    np = deps["np"]
    targets = np.asarray(train_targets, dtype=np.float64)
    if not np.isfinite(targets).all() or (targets <= 0).any():
        raise ContractError("outer-training targets must be finite and strictly positive")
    method = protocol_method_parameters(protocol, method_id)
    params = method.get("estimator") or {"prediction": method.get("prediction")}
    if method_id == "outer_train_median":
        guess = float(statistics.median(float(value) for value in targets))
        return [guess] * len(test_rows), [""] * len(test_rows), {
            "model_params": {"prediction": "median of outer-training raw target_seconds"},
            "transform": {"target": "raw_seconds_median_outer_train_only"},
        }

    if method_id == "work_scaled_ridge":
        x_train, x_test, transform = fit_work_transform(train_rows, test_rows, deps)
        y_train = targets
        log_target = False
    else:
        x_train, x_test, transform = fit_global_transform(train_rows, test_rows, deps)
        y_train = np.log1p(targets)
        log_target = True

    estimator = make_estimator(method_id, protocol, deps)
    estimator.fit(x_train, y_train)
    raw_prediction = estimator.predict(x_test)
    predictions, reasons = _prediction_values(raw_prediction, log_target=log_target, np=np)
    return predictions, reasons, {
        "model_params": params,
        "transform": {
            **transform,
            "target": "raw_seconds" if method_id == "work_scaled_ridge" else "log1p_seconds_with_expm1_inverse_and_lower_clip_zero",
        },
    }


def _fold_data(rows: Sequence[dict[str, Any]], fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [row for row in rows if int(row["fold"]) != fold]
    test = [row for row in rows if int(row["fold"]) == fold]
    train_hashes = [row["qasm_sha256"] for row in train]
    test_hashes = [row["qasm_sha256"] for row in test]
    if set(train_hashes) & set(test_hashes):
        raise ContractError("exact-QASM hash leakage across frozen outer fold")
    if len(set(train_hashes)) != len(train_hashes) or len(set(test_hashes)) != len(test_hashes):
        raise ContractError("duplicate hash in one precision/fold assignment")
    if len(test) != FOLD_COUNTS[fold] or len(train) + len(test) != 150:
        raise ContractError(f"fold {fold} has unexpected train/test sizes: {len(train)}/{len(test)}")
    return train, test


def run_fold(
    bundle: dict[str, Any], precision: str, fold: int, attempt_root: Path,
    environment: dict[str, str] | None = None, argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    if precision not in PRECISIONS or fold not in FOLDS:
        raise ContractError("precision/fold is outside the frozen R8 context")
    protocol = bundle["protocol"]
    environment = validate_execution_environment(environment)
    deps = _import_ml_dependencies()
    train, test = _fold_data(bundle["rows_by_precision"][precision], fold)
    train_hashes = [row["qasm_sha256"] for row in train]
    test_hashes = [row["qasm_sha256"] for row in test]
    train_hashes_sha = hash_sorted_ids(train_hashes)
    test_hashes_sha = hash_sorted_ids(test_hashes)
    train_by_hash = {row["qasm_sha256"]: row for row in train}
    test_by_hash = {row["qasm_sha256"]: row for row in test}
    train_features = [
        {"qasm_sha256": digest, "features": train_by_hash[digest]["features"], "multiq_count": train_by_hash[digest]["multiq_count"]}
        for digest in sorted(train_hashes)
    ]
    test_features = [
        {"qasm_sha256": digest, "features": test_by_hash[digest]["features"], "multiq_count": test_by_hash[digest]["multiq_count"]}
        for digest in sorted(test_hashes)
    ]
    train_targets = [float(train_by_hash[digest]["target_seconds"]) for digest in sorted(train_hashes)]

    precision_root = attempt_root / precision / "classical"
    fold_dir = precision_root / f"fold_{fold}"
    context_manifest_path = precision_root / "context_manifest.json"
    precision_root.mkdir(parents=True, exist_ok=True)
    if fold_dir.exists() and any(fold_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing fold attempt: {fold_dir}")

    context_manifest = {
        "artifact_id": "cudaq-dense-classical-predictors-context-v1",
        "protocol_sha256": bundle["protocol_sha256"],
        "precision": precision,
        "context_id": f"cudaq_dense_statevector_{precision}",
        "engine_id": "cudaq_dense",
        "target_id": "cudaq_dense_sample_wall_clock",
        "evaluation_target_clock": protocol["target"]["evaluation_target_clock"],
        "method_output_clock": protocol["target"]["method_output_clock"],
        "input_hashes": bundle["input_hashes"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "execution_environment": environment,
        "native_thread_cap": THREAD_CAP,
        "thread_environment": {name: str(THREAD_CAP) for name in THREAD_ENV_VARS},
        "method_ids": list(CLASSICAL_METHOD_IDS),
        "method_parameters": {
            method_id: protocol_method_parameters(protocol, method_id)
            for method_id in CLASSICAL_METHOD_IDS
        },
        "assigned_oof_hashes_per_method": 150,
        "precision_contexts_are_fit_separately": True,
        "held_out_targets_passed_to_fit": False,
    }
    context_manifest["context_manifest_sha256"] = sha256_json(context_manifest)
    if context_manifest_path.exists():
        if read_json(context_manifest_path) != context_manifest:
            raise ContractError(f"existing context manifest differs; use a new attempt root: {context_manifest_path}")
    else:
        _atomic_write_json(context_manifest_path, context_manifest)

    started = time.monotonic()
    predictions_by_method: dict[str, list[float | None]] = {}
    reasons_by_method: dict[str, list[str]] = {}
    method_fit_info: dict[str, dict[str, Any]] = {}
    for method_id in CLASSICAL_METHOD_IDS:
        try:
            predictions, reasons, info = fit_predict_method(
                method_id, train_features, train_targets, test_features, protocol, deps,
            )
            predictions_by_method[method_id] = predictions
            reasons_by_method[method_id] = reasons
            method_fit_info[method_id] = {"status": "fit_and_predict_returned", **info}
        except Exception as exc:
            predictions_by_method[method_id] = [None] * len(test_features)
            reasons_by_method[method_id] = [f"{type(exc).__name__}:{str(exc)}"[:600]] * len(test_features)
            method_fit_info[method_id] = {
                "status": "fit_failed",
                "error": f"{type(exc).__name__}:{str(exc)}"[:1000],
                "model_params": (protocol_method_parameters(protocol, method_id).get("estimator") or {}),
            }

    attempts: list[dict[str, Any]] = []
    for method_id in CLASSICAL_METHOD_IDS:
        info = method_fit_info[method_id]
        transform_contract = info.get("transform", {})
        params = info.get("model_params", {})
        for index, test_record in enumerate(sorted(test, key=lambda row: row["qasm_sha256"])):
            prediction = predictions_by_method[method_id][index]
            reason = reasons_by_method[method_id][index]
            status = "ok" if prediction is not None and not reason else "prediction_failed"
            if info["status"] == "fit_failed":
                status = "fit_failed"
            attempts.append({
                "context_id": context_manifest["context_id"],
                "precision": precision,
                "method_id": method_id,
                "fold": fold,
                "qasm_sha256": test_record["qasm_sha256"],
                "assigned_core_oof": True,
                "target_status": test_record["target_status"],
                "target_seconds": float(test_record["target_seconds"]),
                "prediction_seconds": "" if prediction is None else float(prediction),
                "status": status,
                "terminal_reason": reason,
                "train_hash_count": len(train_hashes),
                "test_hash_count": len(test_hashes),
                "train_hashes_sha256": train_hashes_sha,
                "test_hashes_sha256": test_hashes_sha,
                "model_params_json": json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False),
                "transform_contract": json.dumps(transform_contract, sort_keys=True, separators=(",", ":"), allow_nan=False),
                "seed": "",
                "prediction_aggregation": "single",
                "attempt_terminal": True,
            })

    expected_rows = len(CLASSICAL_METHOD_IDS) * len(test_hashes)
    if len(attempts) != expected_rows:
        raise ContractError(f"wrong attempt row count: {len(attempts)} != {expected_rows}")
    attempt_keys = {(row["method_id"], row["qasm_sha256"]) for row in attempts}
    if len(attempt_keys) != expected_rows:
        raise ContractError("duplicate method/hash attempt row")
    fold_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = fold_dir / "predictions.csv"
    _atomic_write_csv(predictions_path, attempts)
    method_status_counts = {
        method_id: {
            status: sum(row["method_id"] == method_id and row["status"] == status for row in attempts)
            for status in sorted({row["status"] for row in attempts if row["method_id"] == method_id})
        }
        for method_id in CLASSICAL_METHOD_IDS
    }
    fold_manifest = {
        "artifact_id": "cudaq-dense-classical-predictors-fold-v1",
        "context_id": context_manifest["context_id"],
        "precision": precision,
        "fold": fold,
        "protocol_sha256": bundle["protocol_sha256"],
        "context_manifest_sha256": context_manifest["context_manifest_sha256"],
        "input_hashes": bundle["input_hashes"],
        "runner_sha256": context_manifest["runner_sha256"],
        "execution_environment": environment,
        "train_hash_count": len(train_hashes),
        "test_hash_count": len(test_hashes),
        "train_hashes_sha256": train_hashes_sha,
        "test_hashes_sha256": test_hashes_sha,
        "train_hashes": sorted(train_hashes),
        "test_hashes": sorted(test_hashes),
        "method_ids": list(CLASSICAL_METHOD_IDS),
        "method_fit_info": method_fit_info,
        "method_status_counts": method_status_counts,
        "attempt_row_count": len(attempts),
        "predictions_csv_sha256": sha256_file(predictions_path),
        "fit_wall_seconds": time.monotonic() - started,
        "native_thread_cap": THREAD_CAP,
        "held_out_targets_passed_to_fit": False,
        "timing_performed": False,
        "gpu_work_performed": False,
    }
    fold_manifest["fold_manifest_sha256"] = sha256_json(fold_manifest)
    _atomic_write_json(fold_dir / "fold_manifest.json", fold_manifest)
    return fold_manifest


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ATTEMPT_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def exclusive_classical_cpu_lock(attempt_root: Path):
    lock_path = attempt_root / ".r8_classical_fit.lock"
    attempt_root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another R8 classical fit is active; fits must be sequential") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=PRECISIONS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--attempt-root", type=Path, default=DEFAULT_ATTEMPT_ROOT)
    parser.add_argument("--validate-inputs-only", action="store_true",
                        help="check pinned R3/R5/C44 inputs and splits without checking ML versions or fitting")
    args = parser.parse_args(argv)
    bundle = validate_inputs()
    if args.validate_inputs_only:
        print(json.dumps({
            "status": "PASS",
            "protocol_sha256": bundle["protocol_sha256"],
            "input_hashes": bundle["input_hashes"],
            "panel_members": bundle["panel_member_count"],
            "panel_unique_hashes": bundle["panel_unique_hash_count"],
            "primary_hashes_per_precision": {precision: len(rows) for precision, rows in bundle["rows_by_precision"].items()},
            "fold_counts": bundle["fold_counts"],
            "work_scaled_multiqubit_nodes_arity_ge_3": bundle["multiq_total"],
            "fitting_performed": False,
            "gpu_work_performed": False,
        }, sort_keys=True))
        return 0
    if args.precision is None or args.fold is None:
        parser.error("--precision and --fold are required unless --validate-inputs-only is set")
    set_native_thread_environment()
    environment = validate_execution_environment()
    deps = _import_ml_dependencies()
    # Configure and inspect the runtime pools before the estimator constructors
    # or any fit; the runner is one process and fits methods sequentially.
    with deps["threadpool_limits"](limits=THREAD_CAP):
        pool_info = deps["threadpool_info"]()
        over_cap = [pool for pool in pool_info if int(pool.get("num_threads", THREAD_CAP)) > THREAD_CAP]
        if over_cap:
            raise ContractError(f"native thread pool exceeds the frozen cap of {THREAD_CAP}: {over_cap}")
        environment["threadpool_info"] = pool_info
        with exclusive_classical_cpu_lock(args.attempt_root):
            result = run_fold(bundle, args.precision, args.fold, args.attempt_root, environment, argv or sys.argv[1:])
    print(json.dumps({
        "status": "completed_validated",
        "precision": result["precision"],
        "fold": result["fold"],
        "attempt_row_count": result["attempt_row_count"],
        "fold_manifest_sha256": result["fold_manifest_sha256"],
        "prediction_file_sha256": result["predictions_csv_sha256"],
        "fitting_performed": True,
        "gpu_work_performed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
