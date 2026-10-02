#!/usr/bin/env python3
"""Run the fixed-MPS Family-Aware-inspired runtime-only adaptation.

The implementation reuses S85's audited hash-level loader and frozen C44
folds.  Materialization and preflight are CPU/read-only operations against
historical inputs; CUDA work is restricted to ``fit-five-fold-oof`` and holds
both S85 timing-exclusion locks for the complete probe, smoke test, fit, and
inference interval.  This is a local adaptation, not a reproduction of the
original joint threshold/runtime method.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import run_mps_fixed_chi16_runtime_adaptation_v1 as s85


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "benchmark_v1/protocol/family_residual_runtime_only_v1.json"
SEED_REGISTRY = ROOT / "benchmark_v1/registry/seed_registry.json"
DEFAULT_OUTPUT = ROOT / "artifacts/benchmark_v3/simulator/family_residual_fixed_mps_runtime"
RUNNER_ID = "family-residual-fixed-mps-runtime-v1"
TARGET_ID = s85.TARGET_ID
FOLDS = s85.FROZEN_FOLDS
SEEDS = s85.SEEDS
FEATURES = s85.GLOBAL_NAMES
UNKNOWN_LABEL = "__UNKNOWN__"
UNKNOWN_INDEX_OFFSET = 0  # The actual index is the outer-training vocabulary size.
EXPECTED_FAMILY_METHODS = (
    "family_residual_median_three_seeds",
    "family_agnostic_median_three_seeds",
)
S85_METHODS = ("train_fold_median", "ridge_alpha_1", "graph_median_three_seeds")
EXPECTED_ACCOUNTING = {
    "assigned_hashes": 150,
    "finite_labels": 144,
    "quality_pass_finite": 142,
    "quality_failed_finite": 2,
    "unavailable_labels": 6,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def input_hashes() -> dict[str, Any]:
    """Pin the S85 loader source, its validated source files, and E5 contracts."""
    required = (PROTOCOL, SEED_REGISTRY, Path(s85.__file__).resolve())
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError("required E5 contract/source missing: " + ", ".join(absent))
    return {
        "s85_pinned_inputs": s85.input_hashes(),
        "s85_loader_source_sha256": sha256_file(Path(s85.__file__).resolve()),
        "e5_protocol_sha256": sha256_file(PROTOCOL),
        "seed_registry_sha256": sha256_file(SEED_REGISTRY),
    }


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "family-residual-fixed-mps-runtime-v1":
        raise ValueError("unexpected Family Residual protocol ID")
    if protocol.get("method_id") != "family_residual_fixed_mps_runtime_adaptation":
        raise ValueError("unexpected Family Residual method ID")
    data = protocol.get("data", {})
    if data.get("target_clock") != TARGET_ID:
        raise ValueError("protocol target clock differs from audited S85 target")
    if data.get("assigned_hashes") != EXPECTED_ACCOUNTING["assigned_hashes"]:
        raise ValueError("protocol assigned-hash count differs from the audited MPS panel")
    if data.get("finite_labels") != EXPECTED_ACCOUNTING["finite_labels"]:
        raise ValueError("protocol finite-label count differs from the audited MPS panel")
    if data.get("quality_pass_finite") != EXPECTED_ACCOUNTING["quality_pass_finite"] or data.get("quality_failed_finite") != EXPECTED_ACCOUNTING["quality_failed_finite"]:
        raise ValueError("protocol quality accounting differs from the audited MPS panel")
    if data.get("unavailable_labels") != EXPECTED_ACCOUNTING["unavailable_labels"]:
        raise ValueError("protocol unavailable-label count differs from the audited MPS panel")
    feature_names = protocol.get("features", {}).get("names")
    if feature_names != list(FEATURES):
        raise ValueError("protocol features must be the exact six S85 structural features in order")
    if protocol.get("runtime_model", {}).get("seeds") != list(SEEDS):
        raise ValueError("protocol seed set differs from S85 registered seeds")
    if protocol.get("data", {}).get("outer_folds") != "Unchanged C44 exact-QASM folds, counts 37/28/26/25/34. All aliases share a fold.":
        raise ValueError("protocol outer split contract changed")


def family_alias_components(panel_rows: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return exact-hash labels and family-name connected-component labels.

    Family aliases are joined whenever they co-occur for one exact-QASM hash.
    The resulting sorted member list, serialized as compact JSON, is the stable
    component label. No label is selected based on model accuracy.
    """
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            parent[second] = first

    families_by_hash: dict[str, set[str]] = defaultdict(set)
    for row in panel_rows:
        if row.get("stratum") != "core_q2_q9":
            continue
        digest = row.get("qasm_sha256", "")
        family = row.get("family", "").strip()
        if not digest or not family:
            raise ValueError("core panel row lacks exact-QASM hash or family name")
        families_by_hash[digest].add(family)
        find(family)
    if not families_by_hash:
        raise ValueError("no core family metadata found")
    for families in families_by_hash.values():
        ordered = sorted(families)
        for family in ordered[1:]:
            union(ordered[0], family)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for family in sorted(parent):
        members_by_root[find(family)].append(family)
    label_by_family = {
        family: json.dumps(members, ensure_ascii=False, separators=(",", ":"))
        for members in members_by_root.values()
        for family in members
    }
    component_by_hash: dict[str, str] = {}
    aliases_by_hash: dict[str, list[str]] = {}
    for digest, families in families_by_hash.items():
        components = {label_by_family[family] for family in families}
        if len(components) != 1:
            raise ValueError(f"family aliases for exact hash {digest} did not resolve to one connected component")
        component_by_hash[digest] = next(iter(components))
        aliases_by_hash[digest] = sorted(families)
    return component_by_hash, aliases_by_hash


def family_alias_component_members(panel_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """Map serialized component label to its sorted family-name members."""
    component_by_hash, aliases_by_hash = family_alias_components(panel_rows)
    members: dict[str, set[str]] = defaultdict(set)
    for digest, aliases in aliases_by_hash.items():
        members[component_by_hash[digest]].update(aliases)
    return {component: sorted(names) for component, names in sorted(members.items())}


def load_e5_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str], dict[str, list[str]]]:
    protocol = read_json(PROTOCOL)
    validate_protocol(protocol)
    hashes_before = input_hashes()
    metadata, targets, _graph_records, feature_by_hash = s85.load_inputs()
    hashes_after = input_hashes()
    if hashes_before != hashes_after or metadata.get("hashes") != hashes_before["s85_pinned_inputs"]:
        raise ValueError("audited S85 inputs changed while E5 loaded them")
    fold_by_hash = metadata["fold_by_hash"]
    component_by_hash, aliases_by_hash = family_alias_components(metadata["panel"])
    if len(targets) != EXPECTED_ACCOUNTING["assigned_hashes"] or set(fold_by_hash) != {row["source_qasm_sha256"] for row in targets}:
        raise ValueError("E5 target rows do not exactly join the frozen 150-hash C44 split")
    if set(feature_by_hash) != set(fold_by_hash) or set(component_by_hash) != set(fold_by_hash):
        raise ValueError("S85 feature/family metadata hash set differs from the frozen split")
    status_counts = Counter(row["target_status"] for row in targets)
    quality_counts = Counter(row["quality_status_separate"] for row in targets if row["target_status"] == "runtime_observed")
    if status_counts != Counter({"runtime_observed": 144, "unavailable_adapter_error": 6}):
        raise ValueError(f"unexpected E5 target status accounting: {status_counts}")
    if quality_counts != Counter({"quality_pass": 142, "quality_failed": 2}):
        raise ValueError(f"unexpected E5 finite quality accounting: {quality_counts}")
    for digest, entry in feature_by_hash.items():
        values = entry.get("globals", {})
        if set(values) != set(FEATURES):
            raise ValueError(f"six-feature dictionary mismatch for hash {digest}")
        if any(not math.isfinite(float(values[name])) or float(values[name]) < 0 for name in FEATURES):
            raise ValueError(f"invalid structural feature value for hash {digest}")
    for row in targets:
        value = row["target_seconds"]
        if row["target_status"] == "runtime_observed":
            if value == "" or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"invalid finite target for hash {row['source_qasm_sha256']}")
        elif value != "":
            raise ValueError("unavailable S85 target was imputed")
    expected_fold_counts = {int(k): int(v) for k, v in s85.EXPECTED_FOLD_HASH_COUNTS.items()}
    fold_counts = Counter(fold_by_hash.values())
    if dict(sorted(fold_counts.items())) != expected_fold_counts:
        raise ValueError("E5 fold counts do not match the frozen S85/C44 assignment")
    return metadata, targets, feature_by_hash, component_by_hash, aliases_by_hash


def split_assignment_sha256(fold_by_hash: dict[str, int]) -> str:
    canonical = [{"source_qasm_sha256": digest, "fold": int(fold_by_hash[digest])} for digest in sorted(fold_by_hash)]
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def _preflight_rows(
    targets: list[dict[str, Any]],
    feature_by_hash: dict[str, dict[str, Any]],
    component_by_hash: dict[str, str],
    aliases_by_hash: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in sorted(targets, key=lambda item: item["source_qasm_sha256"]):
        digest = target["source_qasm_sha256"]
        rows.append({
            **target,
            "family_component_audit_only": component_by_hash[digest],
            "family_alias_names_audit_only": json.dumps(aliases_by_hash[digest], ensure_ascii=False, separators=(",", ":")),
            **{name: feature_by_hash[digest]["globals"][name] for name in FEATURES},
        })
    return rows


def materialize(output_dir: Path) -> None:
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing E5 materialization: {output_dir}")
    metadata, targets, features, component_by_hash, aliases_by_hash = load_e5_inputs()
    hashes = input_hashes()
    rows = _preflight_rows(targets, features, component_by_hash, aliases_by_hash)
    output_dir.mkdir(parents=True)
    table = output_dir / "hash_targets_and_static_features.csv"
    write_csv(table, list(rows[0]), rows)
    fold_counts = Counter(metadata["fold_by_hash"].values())
    manifest = {
        "artifact_id": RUNNER_ID,
        "status": "static_materialization_complete_cuda_not_started",
        "method_claim": "Family-Aware-inspired fixed-MPS runtime-only adaptation",
        "not_claimed": ["original joint threshold/runtime method", "family-held-out transfer", "paper reproduction", "leaderboard promotion"],
        "target_clock_id": TARGET_ID,
        "target_reduction": "Audited S85 exact-hash median of session/member warm medians; no label recomputation",
        "assigned_hashes": len(rows),
        "finite_labels": sum(row["target_status"] == "runtime_observed" for row in rows),
        "quality_pass_finite": sum(row["quality_status_separate"] == "quality_pass" and row["target_status"] == "runtime_observed" for row in rows),
        "quality_failed_finite": sum(row["quality_status_separate"] == "quality_failed" and row["target_status"] == "runtime_observed" for row in rows),
        "unavailable_unimputed": sum(row["target_status"] != "runtime_observed" for row in rows),
        "quality_or_status_as_predictor": False,
        "family_component_count": len(family_alias_component_members(metadata["panel"])),
        "family_aliases": "Connected components of family names co-occurring on one exact-QASM hash; sorted component names are labels.",
        "feature_names": list(FEATURES),
        "split_authority": "unchanged C44 exact-QASM folds from the audited S85 loader",
        "split_assignment_sha256": split_assignment_sha256(metadata["fold_by_hash"]),
        "unique_hashes_by_fold": {str(fold): fold_counts[fold] for fold in sorted(fold_counts)},
        "input_sha256": hashes,
        "s85_loader_source_sha256": hashes["s85_loader_source_sha256"],
        "protocol_sha256": hashes["e5_protocol_sha256"],
        "seed_registry_sha256": hashes["seed_registry_sha256"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "target_table_sha256": sha256_file(table),
        "cuda_probe_status": "not_run_by_static_materialize_or_preflight",
        "training_performed": False,
        "simulator_timing_performed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "materialization_manifest.json", manifest)


def validate_materialization(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "materialization_manifest.json"
    table = output_dir / "hash_targets_and_static_features.csv"
    if not manifest_path.is_file() or not table.is_file():
        raise ValueError("E5 materialization is incomplete; run --action materialize first")
    manifest = read_json(manifest_path)
    qa_path = output_dir / "preflight_qa.json"
    if manifest.get("input_sha256") != input_hashes():
        raise ValueError("a pinned E5/S85 input changed after materialization")
    if manifest.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("E5 runner changed after materialization; materialize into a fresh output directory")
    if manifest.get("target_table_sha256") != sha256_file(table):
        raise ValueError("E5 hash-level target/feature table changed after materialization")
    metadata, targets, features, components, aliases = load_e5_inputs()
    rows = read_csv(table)
    expected_rows = _preflight_rows(targets, features, components, aliases)
    if len(rows) != EXPECTED_ACCOUNTING["assigned_hashes"] or [r["source_qasm_sha256"] for r in rows] != [r["source_qasm_sha256"] for r in expected_rows]:
        raise ValueError("E5 static table does not cover every exact hash exactly once")
    expected_by_hash = {row["source_qasm_sha256"]: row for row in expected_rows}
    for row in rows:
        expected = expected_by_hash[row["source_qasm_sha256"]]
        if int(row["fold"]) != int(metadata["fold_by_hash"][row["source_qasm_sha256"]]):
            raise ValueError("E5 static table contains an incorrect frozen fold")
        if row["target_status"] != expected["target_status"] or row["quality_status_separate"] != expected["quality_status_separate"]:
            raise ValueError("E5 static table target/quality status changed")
        value = row["target_seconds"]
        expected_value = expected["target_seconds"]
        if expected_value == "":
            if value != "":
                raise ValueError("E5 static table imputed an unavailable label")
        elif not value or not math.isclose(float(value), float(expected_value), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("E5 static table hash-level target changed")
        for feature in FEATURES:
            if not math.isclose(float(row[feature]), float(expected[feature]), rel_tol=0, abs_tol=0):
                raise ValueError(f"E5 static feature changed for {row['source_qasm_sha256']}: {feature}")
        if row["family_component_audit_only"] != expected["family_component_audit_only"]:
            raise ValueError("E5 alias component audit label changed")
    qa = {
        "artifact_id": f"{RUNNER_ID}-static-preflight",
        "status": "PASS",
        "static_preflight_only": True,
        "cuda_started": False,
        "training_performed": False,
        "simulator_timing_performed": False,
        "unique_hashes": len(rows),
        "assigned_hashes": len(metadata["fold_by_hash"]),
        "finite_labels": 144,
        "quality_pass_finite": 142,
        "quality_failed_finite_retained": 2,
        "unavailable_unimputed": 6,
        "fold_counts": {str(k): int(v) for k, v in sorted(Counter(metadata["fold_by_hash"].values()).items())},
        "hash_fold_join_exact": True,
        "family_component_aliases_connected": True,
        "feature_names": list(FEATURES),
        "quality_status_used_as_predictor": False,
        "target_table_sha256": sha256_file(table),
        "split_assignment_sha256": manifest["split_assignment_sha256"],
        "input_sha256": input_hashes(),
    }
    if qa_path.exists():
        existing_qa = read_json(qa_path)
        if existing_qa.get("status") != "PASS" or existing_qa.get("target_table_sha256") != sha256_file(table) or existing_qa.get("input_sha256") != input_hashes():
            raise ValueError("existing E5 preflight QA is stale or invalid")
        return existing_qa
    write_json(qa_path, qa)
    return qa


def ensure_static_preflight(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = output_dir / "materialization_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("run --action materialize and --action preflight first")
    qa_path = output_dir / "preflight_qa.json"
    if not qa_path.is_file():
        raise SystemExit("run --action preflight first")
    manifest, qa = read_json(manifest_path), read_json(qa_path)
    if qa.get("status") != "PASS" or qa.get("cuda_started") is not False:
        raise SystemExit("static preflight did not pass or unexpectedly performed CUDA work")
    if manifest.get("input_sha256") != input_hashes():
        raise SystemExit("a frozen input changed after E5 preflight; materialize into a fresh directory")
    if manifest.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise SystemExit("E5 runner changed after preflight; materialize into a fresh directory")
    table = output_dir / "hash_targets_and_static_features.csv"
    if not table.is_file() or sha256_file(table) != manifest.get("target_table_sha256") or qa.get("target_table_sha256") != manifest.get("target_table_sha256"):
        raise SystemExit("static E5 target/features table is missing or changed")
    return manifest, qa


def groupkfold_hash_splits(hashes: list[str], n_splits: int = 4) -> list[dict[str, list[str]]]:
    """Return deterministic GroupKFold hash lists, rejecting duplicate rows."""
    if hashes != sorted(hashes) or len(hashes) != len(set(hashes)):
        raise ValueError("GroupKFold input must be unique hashes sorted lexicographically")
    if len(hashes) < n_splits:
        raise ValueError(f"GroupKFold requires at least {n_splits} unique hashes")
    import numpy as np
    from sklearn.model_selection import GroupKFold

    index = np.arange(len(hashes))
    groups = np.asarray(hashes, dtype=object)
    splits: list[dict[str, list[str]]] = []
    for train_indices, predicted_indices in GroupKFold(n_splits=n_splits).split(index, groups=groups):
        splits.append({
            "train_hashes": [hashes[int(i)] for i in train_indices],
            "predicted_hashes": [hashes[int(i)] for i in predicted_indices],
        })
    predicted = [digest for split in splits for digest in split["predicted_hashes"]]
    if len(predicted) != len(hashes) or set(predicted) != set(hashes):
        raise ValueError("GroupKFold did not predict every finite training hash exactly once")
    if any(set(split["train_hashes"]) & set(split["predicted_hashes"]) for split in splits):
        raise ValueError("GroupKFold inner classifier hash leakage")
    return splits


def logged_features(feature_by_hash: dict[str, dict[str, Any]], hashes: Iterable[str]):
    import numpy as np

    ordered = list(hashes)
    matrix = np.asarray([
        [float(feature_by_hash[digest]["globals"][name]) for name in FEATURES]
        for digest in ordered
    ], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURES) or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("E5 structural features must be six finite non-negative values")
    return np.log1p(matrix)


def fit_feature_transform(training_logged_features):
    """Fit the protocol's population mean/std transform on training rows only."""
    import numpy as np

    values = np.asarray(training_logged_features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(FEATURES) or not np.isfinite(values).all():
        raise ValueError("feature scaler requires a non-empty finite N by six training matrix")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    return mean, scale


def transform_features(logged, mean, scale):
    import numpy as np

    output = (np.asarray(logged, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / np.asarray(scale, dtype=np.float64)
    if not np.isfinite(output).all():
        raise ValueError("non-finite standardized feature")
    return output.astype(np.float32)


def seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def classifier_predictions(
    train_hashes: list[str],
    predict_hashes: list[str],
    label_by_hash: dict[str, str],
    log_feature_by_hash: dict[str, Any],
    seed: int,
    torch,
    sklearn_version: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Fit the fixed two-layer SiLU family classifier on structural x only."""
    import numpy as np

    if not train_hashes or not predict_hashes:
        raise ValueError("family classifier needs non-empty train and prediction hashes")
    if set(train_hashes) & set(predict_hashes):
        raise ValueError("family classifier training/prediction exact hashes overlap")
    classes = sorted({label_by_hash[digest] for digest in train_hashes})
    if not classes:
        raise ValueError("family classifier has no outer-training family components")
    class_indices = {label: index for index, label in enumerate(classes)}
    mean, scale = fit_feature_transform(np.stack([log_feature_by_hash[h] for h in train_hashes]))
    x_train = transform_features(np.stack([log_feature_by_hash[h] for h in train_hashes]), mean, scale)
    x_predict = transform_features(np.stack([log_feature_by_hash[h] for h in predict_hashes]), mean, scale)
    y_train = np.asarray([class_indices[label_by_hash[h]] for h in train_hashes], dtype=np.int64)
    if len(classes) == 1:
        return ({digest: classes[0] for digest in predict_hashes}, {
            "training_hashes": list(train_hashes),
            "predicted_hashes": list(predict_hashes),
            "class_vocabulary": classes,
            "single_class_deterministic": True,
            "feature_names": list(FEATURES),
            "feature_input_policy": "six structural features only",
            "sklearn_version": sklearn_version,
        })

    seed_everything(seed, torch)
    model = torch.nn.Sequential(
        torch.nn.Linear(len(FEATURES), 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, len(classes)),
    ).to(torch.device("cuda:0"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    x_device = torch.as_tensor(x_train, dtype=torch.float32, device="cuda:0")
    y_device = torch.as_tensor(y_train, dtype=torch.long, device="cuda:0")
    for epoch in range(200):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_device)
        loss = torch.nn.functional.cross_entropy(logits, y_device)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"non-finite family-classifier loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x_predict, dtype=torch.float32, device="cuda:0"))
        indices = logits.argmax(dim=1).detach().cpu().tolist()
    predictions = {digest: classes[int(index)] for digest, index in zip(predict_hashes, indices)}
    if len(predictions) != len(predict_hashes):
        raise RuntimeError("family classifier prediction coverage mismatch")
    audit = {
        "training_hashes": list(train_hashes),
        "predicted_hashes": list(predict_hashes),
        "class_vocabulary": classes,
        "single_class_deterministic": False,
        "feature_names": list(FEATURES),
        "feature_input_policy": "six structural features only",
        "epochs": 200,
        "optimizer": "Adam(lr=0.001,weight_decay=0.0001)",
        "sklearn_version": sklearn_version,
    }
    return predictions, audit


def cross_fitted_family_predictions(
    outer_train_hashes: list[str],
    label_by_hash: dict[str, str],
    log_feature_by_hash: dict[str, Any],
    seed: int,
    torch,
    sklearn_version: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Obtain one GroupKFold OOF predicted family for each runtime-train hash."""
    predictions: dict[str, str] = {}
    audits: list[dict[str, Any]] = []
    for inner_index, split in enumerate(groupkfold_hash_splits(outer_train_hashes, n_splits=4)):
        inner_predictions, audit = classifier_predictions(
            split["train_hashes"], split["predicted_hashes"], label_by_hash,
            log_feature_by_hash, seed, torch, sklearn_version,
        )
        if set(predictions) & set(inner_predictions):
            raise ValueError("inner family predictions duplicate an outer-training hash")
        predictions.update(inner_predictions)
        audits.append({"inner_fold": inner_index, **audit})
    if set(predictions) != set(outer_train_hashes):
        raise ValueError("cross-fitted family predictions do not cover the finite outer-training hashes")
    return predictions, audits


def shared_state_digest(state: dict[str, Any]) -> str:
    """Hash tensor names, dtype, shape, and exact contiguous CPU bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _runtime_model_classes(torch):
    class SharedRuntimeCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(6, 128),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.2),
            )
            self.runtime_head = torch.nn.Linear(64, 1)
            self.shortcut = torch.nn.Linear(6, 1)

        def forward(self, features):
            return self.runtime_head(self.backbone(features)) + self.shortcut(features)

        def shared_state(self):
            return {
                **{f"backbone.{key}": value for key, value in self.backbone.state_dict().items()},
                **{f"runtime_head.{key}": value for key, value in self.runtime_head.state_dict().items()},
                **{f"shortcut.{key}": value for key, value in self.shortcut.state_dict().items()},
            }

    class FamilyResidualRuntime(torch.nn.Module):
        def __init__(self, common: SharedRuntimeCore, class_count: int):
            super().__init__()
            self.backbone = copy.deepcopy(common.backbone)
            self.runtime_head = copy.deepcopy(common.runtime_head)
            self.shortcut = copy.deepcopy(common.shortcut)
            self.class_count = class_count
            self.family_embedding = torch.nn.Embedding(class_count + 1, 64)
            self.family_mlp = torch.nn.Sequential(
                torch.nn.Linear(64, 64),
                torch.nn.SiLU(),
                torch.nn.Linear(64, 64),
                torch.nn.SiLU(),
            )
            self.film_gamma = torch.nn.Linear(64, 64)
            self.film_beta = torch.nn.Linear(64, 64)
            self.family_residual = torch.nn.Linear(64, 1)
            torch.nn.init.zeros_(self.family_residual.weight)
            torch.nn.init.zeros_(self.family_residual.bias)

        def shared_state(self):
            return {
                **{f"backbone.{key}": value for key, value in self.backbone.state_dict().items()},
                **{f"runtime_head.{key}": value for key, value in self.runtime_head.state_dict().items()},
                **{f"shortcut.{key}": value for key, value in self.shortcut.state_dict().items()},
            }

        def forward(self, features, family_indices):
            known = family_indices < self.class_count
            safe_indices = torch.where(known, family_indices, torch.zeros_like(family_indices))
            family = self.family_mlp(self.family_embedding(safe_indices))
            active = known.to(features.dtype).unsqueeze(1)
            gamma = self.film_gamma(family) * active
            beta = self.film_beta(family) * active
            correction = self.family_residual(family) * active
            hidden = self.backbone(features)
            modulated = hidden * (1.0 + gamma) + beta
            return self.runtime_head(modulated) + self.shortcut(features) + correction

    class FamilyAgnosticRuntime(torch.nn.Module):
        def __init__(self, common: SharedRuntimeCore):
            super().__init__()
            self.backbone = copy.deepcopy(common.backbone)
            self.runtime_head = copy.deepcopy(common.runtime_head)
            self.shortcut = copy.deepcopy(common.shortcut)

        def shared_state(self):
            return {
                **{f"backbone.{key}": value for key, value in self.backbone.state_dict().items()},
                **{f"runtime_head.{key}": value for key, value in self.runtime_head.state_dict().items()},
                **{f"shortcut.{key}": value for key, value in self.shortcut.state_dict().items()},
            }

        def forward(self, features):
            return self.runtime_head(self.backbone(features)) + self.shortcut(features)

    return SharedRuntimeCore, FamilyResidualRuntime, FamilyAgnosticRuntime


def _predict_runtime(model, x, family_indices, torch) -> list[float]:
    import numpy as np

    model.eval()
    with torch.no_grad():
        if family_indices is None:
            log_prediction = model(x).reshape(-1).detach().cpu().double().numpy()
        else:
            log_prediction = model(x, family_indices).reshape(-1).detach().cpu().double().numpy()
    if not np.isfinite(log_prediction).all():
        raise RuntimeError("non-finite predicted log1p runtime")
    with np.errstate(over="ignore", invalid="ignore"):
        seconds = np.expm1(log_prediction)
    if not np.isfinite(seconds).all():
        raise RuntimeError("predicted runtime overflow/non-finite; no fallback is permitted")
    seconds = np.maximum(0.0, seconds)
    if not np.isfinite(seconds).all():
        raise RuntimeError("runtime predictions are not finite after non-negative projection")
    return [float(value) for value in seconds]


def train_runtime_model(model, x_train, y_train, x_test, test_family_indices, seed: int, torch) -> list[float]:
    seed_everything(seed, torch)
    model.to(torch.device("cuda:0"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160)
    y = torch.as_tensor(y_train, dtype=torch.float32, device="cuda:0")
    for epoch in range(160):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if test_family_indices is None:
            predicted = model(x_train).reshape(-1)
        else:
            # Training inputs receive cross-fitted family classes, never true labels.
            train_family_indices = test_family_indices[0]
            predicted = model(x_train, train_family_indices).reshape(-1)
        loss = torch.nn.functional.mse_loss(predicted, y)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"non-finite runtime loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    if test_family_indices is None:
        return _predict_runtime(model, x_test, None, torch)
    train_indices, heldout_indices = test_family_indices
    del train_indices
    return _predict_runtime(model, x_test, heldout_indices, torch)


def cuda_smoke_test(torch) -> dict[str, Any]:
    """Small CUDA forward/backward check for classifier and both runtime paths."""
    from torch_geometric.data import Batch, Data

    SharedCore, FamilyModel, AblationModel = _runtime_model_classes(torch)
    common = SharedCore()
    family = FamilyModel(common, class_count=2).to(torch.device("cuda:0"))
    ablation = AblationModel(common).to(torch.device("cuda:0"))
    x = torch.zeros((4, 6), dtype=torch.float32, device="cuda:0", requires_grad=True)
    ids = torch.tensor([0, 1, 2, 2], dtype=torch.long, device="cuda:0")
    family_out = family(x, ids)
    ablation_out = ablation(x)
    classifier = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.SiLU(), torch.nn.Linear(64, 2)).to(torch.device("cuda:0"))
    class_out = classifier(x)
    (family_out.sum() + ablation_out.sum() + class_out.sum()).backward()
    pyg_example = Data(
        x=torch.ones((2, 3), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    pyg_batch = Batch.from_data_list([pyg_example]).to(torch.device("cuda:0"))
    pyg_batch.x.requires_grad_(True)
    pyg_scalar = pyg_batch.x.sum()
    pyg_scalar.backward()
    torch.cuda.synchronize(torch.device("cuda:0"))
    finite = all(torch.isfinite(value).all().item() for value in (family_out, ablation_out, class_out, pyg_scalar))
    if not finite:
        raise RuntimeError("Family Residual CUDA smoke produced non-finite values")
    family_digest, ablation_digest = shared_state_digest(family.shared_state()), shared_state_digest(ablation.shared_state())
    if family_digest != ablation_digest:
        raise RuntimeError("Family Residual CUDA smoke shared initial-state mismatch")
    del common, family, ablation, x, ids, family_out, ablation_out, classifier, class_out, pyg_example, pyg_batch, pyg_scalar
    torch.cuda.empty_cache()
    return {
        "status": "PASS",
        "cuda_device": torch.cuda.get_device_name(0),
        "classifier_forward_backward": "PASS",
        "family_runtime_forward_backward": "PASS",
        "family_agnostic_runtime_forward_backward": "PASS",
        "pyg_data_batch_cuda_forward_backward": "PASS",
        "shared_initial_state_digest_equal": True,
        "smoke_shared_initial_state_sha256": family_digest,
    }


def _training_fold(
    output_dir: Path,
    fold: int,
    metadata: dict[str, Any],
    targets: list[dict[str, Any]],
    feature_by_hash: dict[str, dict[str, Any]],
    family_component_by_hash: dict[str, str],
    expected_cuda_environment: dict[str, Any],
    cuda_smoke: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    import numpy as np
    import sklearn
    import torch

    targets_by_hash = {row["source_qasm_sha256"]: row for row in targets}
    fold_by_hash = metadata["fold_by_hash"]
    all_hashes = sorted(fold_by_hash)
    train_hashes = [digest for digest in all_hashes if fold_by_hash[digest] != fold and targets_by_hash[digest]["target_status"] == "runtime_observed"]
    test_hashes = [digest for digest in all_hashes if fold_by_hash[digest] == fold]
    if set(train_hashes) & set(test_hashes) or not train_hashes or not test_hashes:
        raise ValueError(f"outer fold {fold} exact-hash split is invalid")
    if train_hashes != sorted(train_hashes) or test_hashes != sorted(test_hashes):
        raise ValueError("outer fold hash rows must be sorted for deterministic fitting")

    train_labels = {digest: family_component_by_hash[digest] for digest in train_hashes}
    outer_vocab = sorted(set(train_labels.values()))
    if not outer_vocab:
        raise ValueError(f"outer fold {fold} has no finite family training labels")
    unknown_index = len(outer_vocab)
    outer_class_index = {label: index for index, label in enumerate(outer_vocab)}
    log_values = logged_features(feature_by_hash, all_hashes)
    log_feature_by_hash = {digest: log_values[index] for index, digest in enumerate(all_hashes)}

    runtime_mean, runtime_scale = fit_feature_transform(np.stack([log_feature_by_hash[digest] for digest in train_hashes]))
    x_train_np = transform_features(np.stack([log_feature_by_hash[digest] for digest in train_hashes]), runtime_mean, runtime_scale)
    x_test_np = transform_features(np.stack([log_feature_by_hash[digest] for digest in test_hashes]), runtime_mean, runtime_scale)
    x_train = torch.as_tensor(x_train_np, dtype=torch.float32, device="cuda:0")
    x_test = torch.as_tensor(x_test_np, dtype=torch.float32, device="cuda:0")
    y_train = np.asarray([math.log1p(float(targets_by_hash[digest]["target_seconds"])) for digest in train_hashes], dtype=np.float32)

    SharedCore, FamilyModel, AblationModel = _runtime_model_classes(torch)
    seed_predictions: dict[str, dict[str, dict[str, float]]] = {
        "family_residual": {}, "family_agnostic": {},
    }
    seed_audits: list[dict[str, Any]] = []
    test_family_pred_by_seed: dict[str, dict[str, str]] = {}
    shared_state_digests: dict[str, dict[str, str]] = {}

    for seed in SEEDS:
        train_family_predictions, inner_audit = cross_fitted_family_predictions(
            train_hashes, train_labels, log_feature_by_hash, seed, torch, sklearn.__version__,
        )
        outer_family_predictions, outer_fit_audit = classifier_predictions(
            train_hashes, test_hashes, train_labels, log_feature_by_hash, seed, torch, sklearn.__version__,
        )
        family_train_ids = [outer_class_index.get(train_family_predictions[digest], unknown_index) for digest in train_hashes]
        family_test_ids = [outer_class_index.get(outer_family_predictions[digest], unknown_index) for digest in test_hashes]
        train_family_tensor = torch.as_tensor(family_train_ids, dtype=torch.long, device="cuda:0")
        test_family_tensor = torch.as_tensor(family_test_ids, dtype=torch.long, device="cuda:0")

        # Initialize one shared core, clone it byte-for-byte into both variants,
        # then initialize only the family branch from the continuing RNG stream.
        seed_everything(seed, torch)
        common = SharedCore()
        common_state = common.shared_state()
        common_digest = shared_state_digest(common_state)
        family_model = FamilyModel(common, len(outer_vocab))
        ablation_model = AblationModel(common)
        family_digest = shared_state_digest(family_model.shared_state())
        ablation_digest = shared_state_digest(ablation_model.shared_state())
        if common_digest != family_digest or common_digest != ablation_digest:
            raise RuntimeError(f"fold {fold} seed {seed} shared initial weights differ across model variants")
        shared_state_digests[str(seed)] = {
            "shared_core_initial_sha256": common_digest,
            "family_runtime_initial_sha256": family_digest,
            "family_agnostic_initial_sha256": ablation_digest,
        }

        family_seconds = train_runtime_model(
            family_model, x_train, y_train, x_test, (train_family_tensor, test_family_tensor), seed, torch,
        )
        if len(family_seconds) != len(test_hashes):
            raise RuntimeError("family residual model prediction count differs from assigned outer-test hashes")
        seed_predictions["family_residual"][str(seed)] = dict(zip(test_hashes, family_seconds))
        del family_model
        torch.cuda.empty_cache()

        # Reset to the same seed before the matched ablation's training path.
        ablation_seconds = train_runtime_model(ablation_model, x_train, y_train, x_test, None, seed, torch)
        if len(ablation_seconds) != len(test_hashes):
            raise RuntimeError("family-agnostic model prediction count differs from assigned outer-test hashes")
        seed_predictions["family_agnostic"][str(seed)] = dict(zip(test_hashes, ablation_seconds))
        del ablation_model, common
        torch.cuda.empty_cache()

        test_family_pred_by_seed[str(seed)] = outer_family_predictions
        seed_audits.append({
            "seed": seed,
            "outer_family_classifier": {
                **outer_fit_audit,
                "family_predictions_by_hash": outer_family_predictions,
                "vocabulary_policy": "fixed finite outer-training component vocabulary plus UNKNOWN",
            },
            "inner_crossfit": inner_audit,
            "crossfit_train_family_predictions_by_hash": train_family_predictions,
            "class_vocabulary": outer_vocab,
            "unknown_class_index": unknown_index,
            "shared_initial_state": shared_state_digests[str(seed)],
        })

    def seed_median(kind: str, digest: str) -> float:
        return float(statistics.median(seed_predictions[kind][str(seed)][digest] for seed in SEEDS))

    prediction_rows: list[dict[str, Any]] = []
    for digest in test_hashes:
        target = targets_by_hash[digest]
        row: dict[str, Any] = {
            "source_qasm_sha256": digest,
            "fold": fold,
            "target_status": target["target_status"],
            "target_seconds": target["target_seconds"],
            "quality_status_separate_audit_only": target["quality_status_separate"],
        }
        for seed in SEEDS:
            row[f"pred_family_residual_seed_{seed}_seconds"] = seed_predictions["family_residual"][str(seed)][digest]
            row[f"pred_family_agnostic_seed_{seed}_seconds"] = seed_predictions["family_agnostic"][str(seed)][digest]
        row["pred_family_residual_median_three_seeds_seconds"] = seed_median("family_residual", digest)
        row["pred_family_agnostic_median_three_seeds_seconds"] = seed_median("family_agnostic", digest)
        prediction_rows.append(row)

    test_true = {digest: family_component_by_hash[digest] for digest in test_hashes}
    train_components = set(train_labels.values())
    absent_components = sorted(set(test_true.values()) - train_components)
    diagnostics = []
    for seed in SEEDS:
        predicted = test_family_pred_by_seed[str(seed)]
        accuracy = sum(predicted[digest] == test_true[digest] for digest in test_hashes) / len(test_hashes)
        diagnostics.append({
            "seed": seed,
            "test_family_classification_accuracy": accuracy,
            "assigned_test_hashes": len(test_hashes),
            "true_test_components_absent_from_finite_outer_training": absent_components,
            "diagnostic_after_structural_only_inference": True,
        })

    fold_dir = output_dir / f"fold_{fold}"
    if fold_dir.exists():
        raise SystemExit(f"refusing to overwrite existing E5 fold directory: {fold_dir}")
    fold_dir.mkdir()
    prediction_path = fold_dir / f"fold{fold}_predictions.csv"
    write_csv(prediction_path, list(prediction_rows[0]), prediction_rows)
    audit = {
        "fold": fold,
        "protocol_sha256": manifest["protocol_sha256"],
        "input_sha256": manifest["input_sha256"],
        "split_assignment_sha256": manifest["split_assignment_sha256"],
        "sklearn_version": sklearn.__version__,
        "runtime_outer_train_hashes": train_hashes,
        "runtime_outer_test_hashes": test_hashes,
        "runtime_outer_train_targets": "finite exact-hash runtime labels, including finite quality failures",
        "outer_test_family_metadata_passed_to_classifier": False,
        "family_classifier_feature_names": list(FEATURES),
        "family_classifier_inputs": "six structural features only",
        "quality_status_or_runtime_target_used_by_family_classifier": False,
        "outer_train_family_vocabulary": outer_vocab,
        "unknown_label": UNKNOWN_LABEL,
        "unknown_class_index": unknown_index,
        "seed_runs": seed_audits,
        "test_predicted_family_by_seed": test_family_pred_by_seed,
        "shared_initial_state_sha256_by_seed": shared_state_digests,
        "family_diagnostics_path": "family_diagnostics.json",
        "inner_crossfit_n_splits": 4,
        "inner_crossfit_group_column": "source_qasm_sha256",
    }
    audit_path = fold_dir / "family_prediction_audit.json"
    write_json(audit_path, audit)
    write_json(fold_dir / "family_diagnostics.json", {
        "fold": fold,
        "seed_diagnostics": diagnostics,
        "true_test_family_components": test_true,
        "metrics_quality_used_for_continuation": False,
    })
    quality_counts = Counter(targets_by_hash[digest]["quality_status_separate"] for digest in test_hashes if targets_by_hash[digest]["target_status"] == "runtime_observed")
    train_quality_counts = Counter(targets_by_hash[digest]["quality_status_separate"] for digest in train_hashes)
    environment_path = output_dir / "execution_environment.json"
    qa = {
        "status": "PASS",
        "fold": fold,
        "assigned_test_hashes": len(test_hashes),
        "train_finite_hashes": len(train_hashes),
        "train_test_exact_hash_overlap": 0,
        "test_hashes_unique": len(set(test_hashes)) == len(test_hashes),
        "predictions_cover_all_assigned_test_hashes": {row["source_qasm_sha256"] for row in prediction_rows} == set(test_hashes),
        "test_finite_label_hashes": sum(targets_by_hash[digest]["target_status"] == "runtime_observed" for digest in test_hashes),
        "test_unavailable_hashes": sum(targets_by_hash[digest]["target_status"] != "runtime_observed" for digest in test_hashes),
        "finite_quality_status_counts_test": dict(quality_counts),
        "finite_quality_status_counts_train": dict(train_quality_counts),
        "target_quality_status_used_as_feature": False,
        "true_family_used_for_runtime_training_or_test_prediction": False,
        "family_classifier_uses_only_structural_features": True,
        "cross_fitted_family_prediction_for_every_finite_outer_train_hash": True,
        "inner_fold_count_per_seed": 4,
        "no_inner_train_predicted_hash_overlap": True,
        "shared_initial_state_match_for_all_seeds": True,
        "shared_initial_state_sha256_by_seed": shared_state_digests,
        "seed_set": list(SEEDS),
        "runtime_models": ["family_residual", "family_agnostic_ablation"],
        "prediction_sha256": sha256_file(prediction_path),
        "family_prediction_audit_sha256": sha256_file(audit_path),
        "execution_environment_sha256": sha256_file(environment_path),
        "cuda_smoke_test": cuda_smoke,
        "target_table_sha256": manifest["target_table_sha256"],
        "fold0_checkpoint_rule": "technical_QA_only_no_metric_quality_gate" if fold == 0 else "not_applicable",
        "metrics_quality_consulted": False,
    }
    write_json(fold_dir / f"fold{fold}_qa.json", qa)
    validate_fold_integrity(output_dir, fold, metadata["fold_by_hash"], targets, manifest, expected_cuda_environment)


def validate_fold_integrity(
    output_dir: Path,
    fold: int,
    fold_by_hash: dict[str, int],
    targets: list[dict[str, Any]],
    manifest: dict[str, Any],
    expected_cuda_environment: dict[str, Any],
) -> dict[str, Any]:
    """Technical-only fold validation; does not open family diagnostics or metrics."""
    fold_dir = output_dir / f"fold_{fold}"
    prediction_path = fold_dir / f"fold{fold}_predictions.csv"
    qa_path = fold_dir / f"fold{fold}_qa.json"
    audit_path = fold_dir / "family_prediction_audit.json"
    for path in (prediction_path, qa_path, audit_path):
        if not path.is_file():
            raise ValueError(f"E5 fold {fold} technical artifact missing: {path.name}")
    qa = read_json(qa_path)
    audit = read_json(audit_path)
    if qa.get("status") != "PASS" or qa.get("fold") != fold:
        raise ValueError(f"E5 fold {fold} technical QA did not pass")
    if qa.get("metrics_quality_consulted") is not False or qa.get("fold0_checkpoint_rule") not in {"technical_QA_only_no_metric_quality_gate", "not_applicable"}:
        raise ValueError(f"E5 fold {fold} continuation boundary changed")
    if qa.get("seed_set") != list(SEEDS):
        raise ValueError(f"E5 fold {fold} seed set changed")
    if qa.get("cuda_smoke_test", {}).get("status") != "PASS":
        raise ValueError(f"E5 fold {fold} CUDA smoke failed")
    if qa.get("cuda_smoke_test", {}).get("pyg_data_batch_cuda_forward_backward") != "PASS":
        raise ValueError(f"E5 fold {fold} PyG CUDA smoke failed")
    if qa.get("prediction_sha256") != sha256_file(prediction_path) or qa.get("family_prediction_audit_sha256") != sha256_file(audit_path):
        raise ValueError(f"E5 fold {fold} prediction/audit checksum mismatch")
    if qa.get("execution_environment_sha256") != sha256_file(output_dir / "execution_environment.json"):
        raise ValueError(f"E5 fold {fold} CUDA environment checksum mismatch")
    if qa.get("train_test_exact_hash_overlap") != 0 or qa.get("target_quality_status_used_as_feature") is not False or qa.get("true_family_used_for_runtime_training_or_test_prediction") is not False:
        raise ValueError(f"E5 fold {fold} leakage/quality technical gate failed")
    if audit.get("outer_test_family_metadata_passed_to_classifier") is not False or audit.get("quality_status_or_runtime_target_used_by_family_classifier") is not False:
        raise ValueError(f"E5 fold {fold} family classifier input boundary changed")
    if audit.get("family_classifier_feature_names") != list(FEATURES) or audit.get("family_classifier_inputs") != "six structural features only":
        raise ValueError(f"E5 fold {fold} classifier feature boundary changed")
    if audit.get("split_assignment_sha256") != manifest["split_assignment_sha256"] or audit.get("protocol_sha256") != manifest["protocol_sha256"]:
        raise ValueError(f"E5 fold {fold} protocol/split pin mismatch")
    expected_test = {digest for digest, assigned in fold_by_hash.items() if assigned == fold}
    actual_train = set(audit.get("runtime_outer_train_hashes", []))
    actual_test = set(audit.get("runtime_outer_test_hashes", []))
    target_by_hash = {row["source_qasm_sha256"]: row for row in targets}
    expected_train = {digest for digest, assigned in fold_by_hash.items() if assigned != fold and target_by_hash[digest]["target_status"] == "runtime_observed"}
    if actual_train != expected_train or actual_test != expected_test or actual_train & actual_test:
        raise ValueError(f"E5 fold {fold} classifier/runtime outer hash accounting mismatch")
    rows = read_csv(prediction_path)
    hashes = [row.get("source_qasm_sha256", "") for row in rows]
    if len(hashes) != len(set(hashes)) or set(hashes) != expected_test or len(rows) != len(expected_test):
        raise ValueError(f"E5 fold {fold} predictions do not cover assigned test hashes exactly")
    if any(int(row["fold"]) != fold for row in rows):
        raise ValueError(f"E5 fold {fold} predictions contain a wrong fold ID")
    if audit.get("inner_crossfit_n_splits") != 4 or audit.get("inner_crossfit_group_column") != "source_qasm_sha256":
        raise ValueError(f"E5 fold {fold} inner cross-fit contract changed")
    seed_audits = {str(item.get("seed")): item for item in audit.get("seed_runs", [])}
    if set(seed_audits) != {str(seed) for seed in SEEDS}:
        raise ValueError(f"E5 fold {fold} family audit does not cover every seed")
    for seed in SEEDS:
        item = seed_audits[str(seed)]
        crossfit = item.get("inner_crossfit", [])
        if len(crossfit) != 4:
            raise ValueError(f"E5 fold {fold} seed {seed} is missing inner classifier fits")
        expected_inner = groupkfold_hash_splits(sorted(expected_train), n_splits=4)
        predicted_inner: list[str] = []
        for fit, expected_split in zip(crossfit, expected_inner):
            train = set(fit.get("training_hashes", []))
            predicted = set(fit.get("predicted_hashes", []))
            if not train or not predicted or train & predicted or not train <= expected_train or not predicted <= expected_train:
                raise ValueError(f"E5 fold {fold} seed {seed} inner classifier exact-hash leakage")
            if train != set(expected_split["train_hashes"]) or predicted != set(expected_split["predicted_hashes"]):
                raise ValueError(f"E5 fold {fold} seed {seed} GroupKFold assignment changed")
            if not isinstance(fit.get("sklearn_version"), str) or not fit["sklearn_version"]:
                raise ValueError(f"E5 fold {fold} seed {seed} inner classifier library version missing")
            predicted_inner.extend(fit["predicted_hashes"])
        if len(predicted_inner) != len(expected_train) or set(predicted_inner) != expected_train:
            raise ValueError(f"E5 fold {fold} seed {seed} cross-fit does not predict every finite outer-training hash exactly once")
        crossfit_family_predictions = item.get("crossfit_train_family_predictions_by_hash", {})
        if set(crossfit_family_predictions) != expected_train or not set(crossfit_family_predictions.values()) <= set(audit.get("outer_train_family_vocabulary", [])):
            raise ValueError(f"E5 fold {fold} seed {seed} cross-fitted family labels are incomplete/outside vocabulary")
        outer_cls = item.get("outer_family_classifier", {})
        if set(outer_cls.get("training_hashes", [])) != expected_train or set(outer_cls.get("predicted_hashes", [])) != expected_test:
            raise ValueError(f"E5 fold {fold} seed {seed} outer classifier hash accounting mismatch")
        if set(outer_cls.get("family_predictions_by_hash", {})) != expected_test:
            raise ValueError(f"E5 fold {fold} seed {seed} held-out structural predictions incomplete")
        if not set(outer_cls.get("family_predictions_by_hash", {}).values()) <= set(audit.get("outer_train_family_vocabulary", [])):
            raise ValueError(f"E5 fold {fold} seed {seed} held-out family inference expanded vocabulary")
        if not isinstance(outer_cls.get("sklearn_version"), str) or not outer_cls["sklearn_version"]:
            raise ValueError(f"E5 fold {fold} seed {seed} outer classifier library version missing")
        shared = item.get("shared_initial_state", {})
        if not shared.get("shared_core_initial_sha256") or shared.get("shared_core_initial_sha256") != shared.get("family_runtime_initial_sha256") or shared.get("shared_core_initial_sha256") != shared.get("family_agnostic_initial_sha256"):
            raise ValueError(f"E5 fold {fold} seed {seed} shared initialization mismatch")
    methods = [
        *[f"pred_family_residual_seed_{seed}_seconds" for seed in SEEDS],
        *[f"pred_family_agnostic_seed_{seed}_seconds" for seed in SEEDS],
        "pred_family_residual_median_three_seeds_seconds",
        "pred_family_agnostic_median_three_seeds_seconds",
    ]
    for row in rows:
        digest = row["source_qasm_sha256"]
        target = target_by_hash[digest]
        if row.get("target_status") != target["target_status"] or row.get("quality_status_separate_audit_only") != target["quality_status_separate"]:
            raise ValueError(f"E5 fold {fold} target/status changed for {digest}")
        observed = row.get("target_seconds", "")
        expected = target["target_seconds"]
        if expected == "":
            if observed != "":
                raise ValueError(f"E5 fold {fold} unavailable target imputed for {digest}")
        elif not observed or not math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"E5 fold {fold} finite target changed for {digest}")
        for column in methods:
            try:
                prediction = float(row[column])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"E5 fold {fold} missing/non-numeric prediction {column}") from exc
            if not math.isfinite(prediction) or prediction < 0:
                raise ValueError(f"E5 fold {fold} invalid prediction {column} for {digest}")
    environment = read_json(output_dir / "execution_environment.json")
    actual_cuda_environment = environment.get("cuda_training_environment")
    if actual_cuda_environment != expected_cuda_environment:
        raise ValueError(f"E5 fold {fold} CUDA environment differs from the locked execution fingerprint")
    return {
        "fold": fold,
        "assigned_test_hashes": len(expected_test),
        "finite_train_hashes": len(expected_train),
        "prediction_rows": len(rows),
        "technical_qa": "PASS",
        "metrics_quality_consulted": False,
    }


def _locked_cuda_setup(output_dir: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Probe CUDA/PyG and smoke-test under S85's two nonblocking locks."""
    import torch

    environment = s85.cuda_training_environment()
    freeze_bytes, freeze_tool = s85.package_freeze_snapshot()
    live_environment = {
        "python_executable": sys.executable,
        "torch": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda),
        "torch_geometric": environment["torch_geometric"],
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "pip_freeze_sha256": sha256_bytes(freeze_bytes),
        "pip_freeze_tool": freeze_tool,
    }
    s85.validate_cuda_environment_match(environment, live_environment)
    torch.set_num_threads(2)
    smoke = cuda_smoke_test(torch)
    output_path = output_dir / "execution_environment.json"
    evidence = {
        "shared_maestro_timing_lock_path": str(s85.SHARED_TIMING_LOCK.relative_to(ROOT)),
        "host_gpu_lease_path": str(s85.GPU_LEASE_PATH),
        "both_locks_acquired_nonblocking": True,
        "both_locks_held_for_entire_cuda_interval": True,
        "no_live_timing_worker_at_lock_acquisition": True,
        "live_timing_processes_at_lock_acquisition": [],
    }
    document = {
        "runner_id": RUNNER_ID,
        "cuda_training_environment": environment,
        "live_environment_match": True,
        "pyg_version_recorded": environment["torch_geometric"],
        "cuda_smoke_test": smoke,
        "timing_exclusion_evidence": evidence,
        "cpu_threads": torch.get_num_threads(),
        "environment_sha256": None,
        "protocol_sha256": manifest["protocol_sha256"],
        "input_sha256": manifest["input_sha256"],
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }
    document["environment_sha256"] = sha256_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if output_path.exists():
        existing = read_json(output_path)
        for key in ("cuda_training_environment", "cuda_smoke_test", "protocol_sha256", "input_sha256"):
            if existing.get(key) != document.get(key):
                raise ValueError("previous E5 CUDA preflight evidence differs; use a fresh output directory")
        return existing["cuda_training_environment"], existing["cuda_smoke_test"]
    write_json(output_path, document)
    return environment, smoke


def fit_five_fold_oof(output_dir: Path) -> None:
    manifest, _qa = ensure_static_preflight(output_dir)
    if input_hashes() != manifest["input_sha256"] or manifest.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise SystemExit("E5 frozen source/protocol/input changed after static preflight")
    for fold in FOLDS:
        if (output_dir / f"fold_{fold}").exists():
            raise SystemExit("E5 refuses partial-fold resume/overwrite; use a fresh output directory")
    try:
        metadata, targets, features, family_components, _aliases = load_e5_inputs()
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"E5 input validation failed: {exc}") from exc

    fold_reports = []
    with s85.exclusive_compute_lock():
        expected_cuda_environment, smoke = _locked_cuda_setup(output_dir, manifest)
        for fold in FOLDS:
            _training_fold(
                output_dir, fold, metadata, targets, features, family_components,
                expected_cuda_environment, smoke, manifest,
            )
            report = validate_fold_integrity(
                output_dir, fold, metadata["fold_by_hash"], targets,
                manifest, expected_cuda_environment,
            )
            fold_reports.append(report)
            if fold == 0:
                # The integrity validator above has no access to scores, metrics,
                # or family-classification diagnostics. Continuation is automatic.
                print(json.dumps({
                    "fold0_checkpoint": "TECHNICAL_QA_PASS",
                    "continuation": "automatically_start_folds_1_through_4_unchanged",
                    "metrics_quality_consulted": False,
                    "fold0_technical_qa": report,
                }, sort_keys=True))
    receipt = {
        "status": "five_fold_fit_technical_qa_complete_aggregation_pending",
        "folds": fold_reports,
        "method_claim": "Family-Aware-inspired fixed-MPS runtime-only adaptation; not original joint method",
        "timing_exclusion_evidence": {
            "shared_lock_path": str(s85.SHARED_TIMING_LOCK.relative_to(ROOT)),
            "host_gpu_lease_path": str(s85.GPU_LEASE_PATH),
            "both_locks_held_for_entire_five_fold_cuda_interval": True,
            "nonblocking_flock": True,
            "no_live_timing_worker_at_acquisition": True,
        },
        "fold0_continuation_used_metrics": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "five_fold_execution_receipt.json", receipt)


def metric_rows_for_population(
    rows: list[dict[str, str]],
    targets_by_hash: dict[str, dict[str, Any]],
    methods: dict[str, str],
    population: str,
) -> list[dict[str, Any]]:
    import numpy as np

    if population == "finite_144":
        selected = [row for row in rows if targets_by_hash[row["source_qasm_sha256"]]["target_status"] == "runtime_observed"]
    elif population == "quality_pass_142":
        selected = [row for row in rows if targets_by_hash[row["source_qasm_sha256"]]["target_status"] == "runtime_observed" and targets_by_hash[row["source_qasm_sha256"]]["quality_status_separate"] == "quality_pass"]
    else:
        raise ValueError(f"unknown E5 metrics population {population}")
    actual = [float(row["target_seconds"]) for row in selected]
    output = []
    for method_id, column in methods.items():
        predicted = [float(row[column]) for row in selected]
        if not all(math.isfinite(value) and value >= 0 for value in predicted):
            raise ValueError(f"non-finite E5 prediction in aggregate method {method_id}")
        output.append({"population": population, "method_id": method_id, **s85.metric_bundle(actual, predicted)})
    return output


def bootstrap_seed(comparison_label: str, registry: dict[str, Any], artifact_id: str = f"{RUNNER_ID}-five-fold-oof") -> int:
    stream = registry["streams"]["bootstrap"]
    material = f"{registry['root_seed']}|{stream}|{artifact_id}|{comparison_label}|0"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def paired_hash_bootstrap(
    hashes: list[str],
    deltas: list[float],
    comparison_label: str,
    registry: dict[str, Any],
    replicates: int = 10_000,
) -> dict[str, Any]:
    import numpy as np

    if not hashes or len(hashes) != len(deltas) or len(hashes) != len(set(hashes)):
        raise ValueError("paired exact-QASM bootstrap needs one finite error delta per unique hash")
    if replicates != 10_000:
        raise ValueError("Family Residual bootstrap is frozen at 10,000 replicates")
    delta = np.asarray(deltas, dtype=np.float64)
    if not np.isfinite(delta).all():
        raise ValueError("paired hash bootstrap contains non-finite error deltas")
    seed = bootstrap_seed(comparison_label, registry)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(hashes), size=len(hashes))
        values[index] = float(delta[selected].mean())
    return {
        "comparison_label": comparison_label,
        "bootstrap_stream": registry["streams"]["bootstrap"],
        "bootstrap_seed": seed,
        "replicates": replicates,
        "bootstrap_groups": len(hashes),
        "bootstrap_group_column": "source_qasm_sha256",
        "observed_mae_difference_seconds": float(delta.mean()),
        "bootstrap_mean_mae_difference_seconds": float(values.mean()),
        "bootstrap_ci_low_seconds": float(np.quantile(values, 0.025, method="linear")),
        "bootstrap_ci_high_seconds": float(np.quantile(values, 0.975, method="linear")),
        "percentile_rule": "numpy.quantile(method='linear', q=[0.025,0.975])",
    }


def aggregate_five_fold_oof(output_dir: Path, mps_oof_dir: Path) -> dict[str, Any]:
    if (output_dir / "five_fold_execution_receipt.json").is_file() is False:
        raise SystemExit("run --action fit-five-fold-oof before aggregate")
    manifest, _qa = ensure_static_preflight(output_dir)
    metadata, targets, _features, _families, _aliases = load_e5_inputs()
    targets_by_hash = {row["source_qasm_sha256"]: row for row in targets}
    fold_reports = []
    family_rows: list[dict[str, str]] = []
    for fold in FOLDS:
        report = validate_fold_integrity(output_dir, fold, metadata["fold_by_hash"], targets, manifest, read_json(output_dir / "execution_environment.json")["cuda_training_environment"])
        fold_reports.append(report)
        family_rows.extend(read_csv(output_dir / f"fold_{fold}" / f"fold{fold}_predictions.csv"))
    hashes = [row["source_qasm_sha256"] for row in family_rows]
    if len(hashes) != 150 or len(set(hashes)) != 150 or set(hashes) != set(metadata["fold_by_hash"]):
        raise ValueError("E5 aggregate predictions do not cover each frozen exact hash exactly once")
    if any(int(row["fold"]) != metadata["fold_by_hash"][row["source_qasm_sha256"]] for row in family_rows):
        raise ValueError("E5 aggregate predictions contain a wrong C44 fold")
    family_by_hash = {row["source_qasm_sha256"]: row for row in family_rows}

    baseline_path = mps_oof_dir / "five_fold_oof_predictions.csv"
    if not baseline_path.is_file():
        raise ValueError(f"required S85 common-comparator OOF table is missing: {baseline_path}")
    baseline_rows = read_csv(baseline_path)
    baseline_hashes = [row["source_qasm_sha256"] for row in baseline_rows]
    if len(baseline_hashes) != 150 or len(set(baseline_hashes)) != 150 or set(baseline_hashes) != set(family_by_hash):
        raise ValueError("S85 baseline OOF does not match E5's 150 exact hashes")
    baseline_by_hash = {row["source_qasm_sha256"]: row for row in baseline_rows}
    for digest in sorted(family_by_hash):
        fam_row, base_row = family_by_hash[digest], baseline_by_hash[digest]
        if int(base_row["fold"]) != metadata["fold_by_hash"][digest] or int(fam_row["fold"]) != int(base_row["fold"]):
            raise ValueError(f"E5/S85 fold mismatch for exact hash {digest}")
        if base_row.get("target_status") != fam_row.get("target_status") or base_row.get("quality_status_separate_audit_only") != fam_row.get("quality_status_separate_audit_only"):
            raise ValueError(f"E5/S85 target status mismatch for exact hash {digest}")
        expected = targets_by_hash[digest]
        observed = base_row.get("target_seconds", "")
        if expected["target_seconds"] == "":
            if observed != "":
                raise ValueError(f"S85 OOF imputed unavailable target for {digest}")
        elif not observed or not math.isclose(float(observed), float(expected["target_seconds"]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"S85 OOF target mismatch for exact hash {digest}")

    all_rows: list[dict[str, str]] = []
    for digest in sorted(family_by_hash):
        row = dict(family_by_hash[digest])
        for method in S85_METHODS:
            column = f"pred_{method}_seconds"
            if column not in baseline_by_hash[digest]:
                raise ValueError(f"S85 OOF missing required E5 comparator {column}")
            row[f"pred_s85_{method}_seconds"] = baseline_by_hash[digest][column]
        all_rows.append(row)
    methods = {
        "family_residual_median_three_seeds": "pred_family_residual_median_three_seeds_seconds",
        "family_agnostic_median_three_seeds": "pred_family_agnostic_median_three_seeds_seconds",
        **{f"family_residual_seed_{seed}": f"pred_family_residual_seed_{seed}_seconds" for seed in SEEDS},
        **{f"family_agnostic_seed_{seed}": f"pred_family_agnostic_seed_{seed}_seconds" for seed in SEEDS},
        "s85_train_fold_median": "pred_s85_train_fold_median_seconds",
        "s85_ridge_alpha_1": "pred_s85_ridge_alpha_1_seconds",
        "s85_source_dag_graph_median_three_seeds": "pred_s85_graph_median_three_seeds_seconds",
    }
    for row in all_rows:
        for column in methods.values():
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"E5/S85 comparator prediction missing or invalid: {column}") from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"E5/S85 comparator prediction is non-finite/negative: {column}")

    metrics = []
    for population in ("finite_144", "quality_pass_142"):
        metrics.extend(metric_rows_for_population(all_rows, targets_by_hash, methods, population))
    metrics_path = output_dir / "method_metrics.json"
    if metrics_path.exists():
        raise SystemExit("refusing to overwrite existing E5 aggregate files")
    write_json(metrics_path, {"metrics": metrics, "populations": ["finite_144", "quality_pass_142"]})

    registry = read_json(SEED_REGISTRY)
    paired_bootstrap_rows: list[dict[str, Any]] = []
    paired_detail_rows: list[dict[str, Any]] = []
    candidate = "family_residual_median_three_seeds"
    references = ["family_agnostic_median_three_seeds", "s85_train_fold_median", "s85_ridge_alpha_1", "s85_source_dag_graph_median_three_seeds"]
    for population in ("finite_144", "quality_pass_142"):
        selected = [row for row in all_rows if targets_by_hash[row["source_qasm_sha256"]]["target_status"] == "runtime_observed" and (population == "finite_144" or targets_by_hash[row["source_qasm_sha256"]]["quality_status_separate"] == "quality_pass")]
        actual = {row["source_qasm_sha256"]: float(row["target_seconds"]) for row in selected}
        for reference in references:
            candidate_column, reference_column = methods[candidate], methods[reference]
            comparison_label = f"{population}|{candidate}|{reference}|mae_seconds"
            pair_hashes, deltas = [], []
            for row in selected:
                digest = row["source_qasm_sha256"]
                cand_error = abs(actual[digest] - float(row[candidate_column]))
                ref_error = abs(actual[digest] - float(row[reference_column]))
                delta = cand_error - ref_error
                pair_hashes.append(digest)
                deltas.append(delta)
                paired_detail_rows.append({
                    "population": population,
                    "comparison_label": comparison_label,
                    "source_qasm_sha256": digest,
                    "target_seconds": actual[digest],
                    "candidate_method_id": candidate,
                    "candidate_predicted_seconds": row[candidate_column],
                    "candidate_absolute_error_seconds": cand_error,
                    "reference_method_id": reference,
                    "reference_predicted_seconds": row[reference_column],
                    "reference_absolute_error_seconds": ref_error,
                    "paired_absolute_error_delta_seconds": delta,
                })
            paired_bootstrap_rows.append({
                "population": population,
                "candidate_method_id": candidate,
                "reference_method_id": reference,
                **paired_hash_bootstrap(pair_hashes, deltas, comparison_label, registry),
            })

    comparison_path = output_dir / "paired_hash_comparisons.csv"
    bootstrap_path = output_dir / "paired_bootstrap.csv"
    write_csv(comparison_path, list(paired_detail_rows[0]), paired_detail_rows)
    write_csv(bootstrap_path, list(paired_bootstrap_rows[0]), paired_bootstrap_rows)

    dispersion_rows: list[dict[str, Any]] = []
    for digest in sorted(family_by_hash):
        row = family_by_hash[digest]
        family_values = [float(row[f"pred_family_residual_seed_{seed}_seconds"]) for seed in SEEDS]
        ablation_values = [float(row[f"pred_family_agnostic_seed_{seed}_seconds"]) for seed in SEEDS]
        dispersion_rows.append({
            "source_qasm_sha256": digest,
            "fold": row["fold"],
            "family_residual_seed_std_seconds": statistics.pstdev(family_values),
            "family_agnostic_seed_std_seconds": statistics.pstdev(ablation_values),
            "target_status": row["target_status"],
            "quality_status_separate_audit_only": row["quality_status_separate_audit_only"],
        })
    write_csv(output_dir / "seed_dispersion_by_hash.csv", list(dispersion_rows[0]), dispersion_rows)
    import numpy as np
    finite_dispersion = [row for row in dispersion_rows if targets_by_hash[row["source_qasm_sha256"]]["target_status"] == "runtime_observed"]
    dispersion_summary = {}
    for key in ("family_residual_seed_std_seconds", "family_agnostic_seed_std_seconds"):
        values = np.asarray([row[key] for row in finite_dispersion], dtype=np.float64)
        dispersion_summary[key] = {
            "n": int(len(values)),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90, method="linear")),
            "max": float(values.max()),
        }
    write_json(output_dir / "seed_dispersion_summary.json", dispersion_summary)

    finite_rows = [row for row in all_rows if targets_by_hash[row["source_qasm_sha256"]]["target_status"] == "runtime_observed"]
    coverage = {
        "assigned_hashes": len(all_rows),
        "prediction_coverage_hashes": len({row["source_qasm_sha256"] for row in all_rows}),
        "finite_target_hashes": len(finite_rows),
        "quality_pass_finite": sum(targets_by_hash[row["source_qasm_sha256"]]["quality_status_separate"] == "quality_pass" for row in finite_rows),
        "quality_failed_finite_retained": sum(targets_by_hash[row["source_qasm_sha256"]]["quality_status_separate"] == "quality_failed" for row in finite_rows),
        "unavailable_target_hashes_retained_unimputed": sum(row["target_status"] != "runtime_observed" for row in all_rows),
    }
    summary = {
        "artifact_id": f"{RUNNER_ID}-five-fold-oof",
        "status": "five_fold_oof_complete_not_promoted",
        "method_claim": "Family-Aware-inspired fixed-MPS runtime-only residual adaptation",
        "not_claimed": ["original threshold/runtime joint method", "family-OOD transfer", "paper reproduction", "leaderboard promotion"],
        "target_clock_id": TARGET_ID,
        "split_assignment_sha256": manifest["split_assignment_sha256"],
        "input_sha256": manifest["input_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "runner_sha256": manifest["runner_sha256"],
        "execution_environment_sha256": sha256_file(output_dir / "execution_environment.json"),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "coverage": coverage,
        "metrics_by_population": "method_metrics.json",
        "paired_comparisons": "paired_hash_comparisons.csv",
        "paired_bootstrap": {
            "path": "paired_bootstrap.csv",
            "replicates": 10_000,
            "group_column": "source_qasm_sha256",
            "seed_registry_sha256": sha256_file(SEED_REGISTRY),
            "observed_point_estimate_reported_separately": True,
            "conditions_on_fitted_oof_models": True,
        },
        "seed_dispersion": "seed_dispersion_summary.json",
        "quality_status_used_as_predictor": False,
        "unavailable_target_imputation": False,
        "fold0_checkpoint": "technical integrity only; folds 1-4 automatically continued without metric review",
        "leaderboard_promotion": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output_dir / "five_fold_oof_summary.json"
    write_json(summary_path, summary)
    write_csv(output_dir / "five_fold_oof_predictions.csv", list(all_rows[0]), all_rows)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("materialize", "preflight", "fit-five-fold-oof", "aggregate"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mps-oof-dir", type=Path, default=ROOT / "artifacts/benchmark_v3/simulator/mps_fixed_chi16_runtime_oof")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.action == "materialize":
        materialize(output)
    elif args.action == "preflight":
        if not (output / "materialization_manifest.json").is_file():
            materialize(output)
        qa = validate_materialization(output)
        print(json.dumps(qa, indent=2, sort_keys=True))
    elif args.action == "fit-five-fold-oof":
        fit_five_fold_oof(output)
    else:
        result = aggregate_five_fold_oof(output, args.mps_oof_dir.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
