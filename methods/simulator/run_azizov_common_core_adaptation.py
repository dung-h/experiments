#!/usr/bin/env python3
"""Run the frozen five-fold Azizov-style simulator adaptation.

This runner consumes E2's static artifacts, never calls Aer or transpiles a
circuit, and writes only the fit/report outputs into the same artifact root.
The single fit command runs the fold-0 technical checkpoint and automatically
continues folds 1--4 only when fold 0 passes contract QA.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _bootstrap_overlay_from_argv() -> None:
    """Put a target-installed overlay ahead of base packages before imports."""
    arguments = sys.argv[1:]
    for index, argument in enumerate(arguments):
        if argument == "--overlay-path" and index + 1 < len(arguments):
            overlay = Path(arguments[index + 1]).expanduser().resolve()
            if overlay.is_dir():
                sys.path.insert(0, str(overlay))
            return
        if argument.startswith("--overlay-path="):
            overlay = Path(argument.split("=", 1)[1]).expanduser().resolve()
            if overlay.is_dir():
                sys.path.insert(0, str(overlay))
            return


_bootstrap_overlay_from_argv()

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "benchmark_v1/protocol/azizov_common_core_gnn_v1.json"
DICTIONARY_PATH = ROOT / "benchmark_v1/protocol/azizov_local_feature_dictionary_v1.json"
FIT_REQUIREMENTS_PATH = ROOT / "benchmark_v1/requirements-azizov-fit.txt"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts/benchmark_v3/simulator/azizov_common_core_adaptation"
DEFAULT_CUDA_BASE_LOCK = ROOT / "artifacts/benchmark_v3/simulator/mps_fixed_chi16_runtime_oof/fold_0/environment.json"
FROZEN_FOLDS = (0, 1, 2, 3, 4)
NEURAL_SEEDS = (42, 1234, 31415)
CLASSICAL_SEED = 1234
MEMBER_COUNT = 162
HASH_COUNT = 150
UNKNOWN_GATE = "__UNKNOWN__"
MAX_WIRES = 127

INPUT_FILENAMES = (
    "labels_core.csv",
    "fold_assignments_and_hash_audit.csv",
    "global_features.csv",
    "source_graphs.jsonl",
    "transpiled_graphs.jsonl",
    "compiled_replay_audit.csv",
    "representation_coverage_and_terminal_statuses.csv",
    "representation_manifest.json",
    "source_hashes.json",
)

VIEW_NAMES = ("source", "hybrid", "transpiled")
CLASSICAL_NAMES = (
    "linear_regression",
    "ridge",
    "svr_rbf",
    "random_forest",
    "xgboost",
)
METHOD_IDS = tuple(
    [f"azizov_gnn_{view}_seed_{seed}" for view in VIEW_NAMES for seed in NEURAL_SEEDS]
    + [f"classical_{model}_{view}" for model in CLASSICAL_NAMES for view in VIEW_NAMES]
)

PREDICTION_FIELDS = [
    "panel_member_id", "source_sha256", "fold", "method_id", "method_family",
    "view", "seed", "target_seconds", "member_observed_seconds",
    "prediction_seconds", "status", "terminal_reason",
]
MEDIAN_FIELDS = [
    "panel_member_id", "source_sha256", "fold", "method_id", "method_family",
    "view", "target_seconds", "member_observed_seconds", "prediction_seconds",
    "successful_seed_count", "status", "terminal_reason",
]
STATUS_FIELDS = PREDICTION_FIELDS.copy()
STATUS_FIELDS.insert(STATUS_FIELDS.index("status"), "successful_seed_count")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def activate_overlay(path: Path | None) -> None:
    if path is None:
        return
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"fit package overlay is not a directory: {path}")
    overlay_text = str(path)
    if overlay_text in sys.path:
        sys.path.remove(overlay_text)
    sys.path.insert(0, overlay_text)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL record is not an object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} is non-finite: {value!r}")
    return number


def inner_validation_hashes(outer_train_hashes: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return deterministic inner-train and validation IDs from outer train."""
    ordered = sorted(set(outer_train_hashes), key=lambda digest: hashlib.sha256(
        ("azizov-inner-1234|" + digest).encode("ascii")
    ).hexdigest())
    if len(ordered) < 2:
        raise ValueError("inner epoch selection needs at least two outer-training hashes")
    validation_count = max(1, math.ceil(0.2 * len(ordered)))
    validation = ordered[:validation_count]
    inner_train = ordered[validation_count:]
    if not inner_train:
        raise ValueError("inner validation would leave no inner-training hashes")
    if set(inner_train) & set(validation):
        raise AssertionError("inner train and validation overlap")
    return inner_train, validation


@dataclass(frozen=True)
class GlobalScaler:
    input_names: tuple[str, ...]
    retained_names: tuple[str, ...]
    removed_zero_variance_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    all_constant: bool

    def transform(self, values: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.input_names):
            raise ValueError("global matrix shape differs from fitted scaler inputs")
        if not np.isfinite(matrix).all():
            raise ValueError("global feature matrix contains non-finite values")
        if self.all_constant:
            return np.zeros((matrix.shape[0], 1), dtype=np.float64)
        kept = [self.input_names.index(name) for name in self.retained_names]
        return (matrix[:, kept] - np.asarray(self.means)) / np.asarray(self.scales)


def fit_global_scaler(values: Sequence[Sequence[float]], names: Sequence[str]) -> GlobalScaler:
    """Fit zero-variance removal and population StandardScaler on train rows."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or matrix.shape[0] == 0:
        raise ValueError("cannot fit global scaler on an empty or malformed matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("global feature matrix contains non-finite values")
    constant = np.ptp(matrix, axis=0) == 0.0
    removed = tuple(name for name, is_constant in zip(names, constant) if bool(is_constant))
    retained = tuple(name for name, is_constant in zip(names, constant) if not bool(is_constant))
    if not retained:
        return GlobalScaler(tuple(names), ("__all_constant_zero__",), removed, (0.0,), (1.0,), True)
    indexes = [index for index, is_constant in enumerate(constant) if not bool(is_constant)]
    reduced = matrix[:, indexes]
    means = reduced.mean(axis=0)
    scales = reduced.std(axis=0, ddof=0)
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("invalid train-only StandardScaler statistics")
    return GlobalScaler(
        tuple(names), retained, removed,
        tuple(float(value) for value in means),
        tuple(float(value) for value in scales), False,
    )


@dataclass(frozen=True)
class NormalizedGraph:
    source_sha256: str
    num_qubits: int
    num_clbits: int
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[tuple[int, int], ...]


def _record_hash(record: Mapping[str, Any]) -> str:
    for key in ("source_sha256", "source_qasm_sha256"):
        if record.get(key):
            return str(record[key])
    raise ValueError("graph record is missing source SHA-256")


def _expected_edges(nodes: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, int], ...]:
    previous: dict[tuple[str, int], int] = {}
    edges: set[tuple[int, int]] = set()
    for index, node in enumerate(nodes):
        for kind, field in (("q", "qargs"), ("c", "cargs")):
            for wire in node[field]:
                key = (kind, int(wire))
                if key in previous:
                    edges.add((previous[key], index))
                previous[key] = index
    return tuple(sorted(edges))


def normalize_graph_record(record: Mapping[str, Any]) -> NormalizedGraph:
    """Accept E2's Terra source-DAG rows and raw-graph-v1 replay rows."""
    digest = _record_hash(record)
    num_qubits = int(record.get("num_qubits", -1))
    num_clbits = int(record.get("num_clbits", -1))
    if not (0 <= num_qubits <= MAX_WIRES and 0 <= num_clbits <= MAX_WIRES):
        raise ValueError(f"invalid declared wire widths for {digest}")
    raw_nodes = record.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError(f"empty or missing operation DAG for {digest}")
    dynamic_ops = {"if_else", "while_loop", "for_loop", "switch_case"}
    normalized = []
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            raise ValueError(f"malformed node {index} for {digest}")
        node_index = int(raw.get("node_index", raw.get("node_id", -1)))
        if node_index != index:
            raise ValueError(f"non-insertion-order node IDs for {digest}")
        operation = str(raw.get("operation", ""))
        if not operation or operation in dynamic_ops or raw.get("condition") is not None:
            raise ValueError(f"unsupported operation/control flow in {digest}: {operation!r}")
        qargs = [int(value) for value in raw.get("qargs", [])]
        cargs = [int(value) for value in raw.get("cargs", [])]
        if any(value < 0 or value >= MAX_WIRES for value in qargs + cargs):
            raise ValueError(f"wire index outside [0,126] for {digest} node {index}")
        if any(value >= num_qubits for value in qargs) or any(value >= num_clbits for value in cargs):
            raise ValueError(f"wire index exceeds declared width for {digest} node {index}")

        if "parameter_values_over_pi" in raw:
            params = [finite_float(v, field=f"{digest}.node{index}.parameter_values_over_pi") for v in raw["parameter_values_over_pi"]]
            masks = [finite_float(v, field=f"{digest}.node{index}.parameter_presence_masks") for v in raw.get("parameter_presence_masks", [])]
        else:
            raw_params = raw.get("params_text", [])
            if len(raw_params) > 4:
                raise ValueError(f"more than four parameters for {digest} node {index}")
            params = [finite_float(v, field=f"{digest}.node{index}.params_text") / math.pi for v in raw_params]
            masks = [1.0] * len(params)
        if len(params) > 4 or len(masks) > 4 or len(params) != len(masks):
            raise ValueError(f"invalid parameter layout for {digest} node {index}")
        params += [0.0] * (4 - len(params))
        masks += [0.0] * (4 - len(masks))
        if any(mask not in (0.0, 1.0) for mask in masks):
            raise ValueError(f"parameter masks must be binary for {digest} node {index}")
        if any(mask == 0.0 and value != 0.0 for value, mask in zip(params, masks)):
            raise ValueError(f"absent parameter value must be zero for {digest} node {index}")

        expected_norms = (
            len(qargs) / MAX_WIRES,
            len(cargs) / MAX_WIRES,
            index / max(1, len(raw_nodes) - 1),
        )
        supplied_norms = (
            float(raw.get("quantum_arity_normalized", expected_norms[0])),
            float(raw.get("classical_arity_normalized", expected_norms[1])),
            float(raw.get("instruction_index_normalized", expected_norms[2])),
        )
        if any(abs(a - b) > 1e-12 for a, b in zip(expected_norms, supplied_norms)):
            raise ValueError(f"fixed graph normalization differs from protocol for {digest} node {index}")
        normalized.append({
            "operation": operation,
            "qargs": qargs,
            "cargs": cargs,
            "parameter_values_over_pi": params,
            "parameter_presence_masks": masks,
            "quantum_arity_normalized": expected_norms[0],
            "classical_arity_normalized": expected_norms[1],
            "instruction_index_normalized": expected_norms[2],
        })

    expected_edges = _expected_edges(normalized)
    raw_edges = record.get("edges")
    if not isinstance(raw_edges, list):
        raise ValueError(f"missing DAG edges for {digest}")
    supplied_edges = []
    for edge in raw_edges:
        if isinstance(edge, dict):
            pair = (int(edge.get("source_node", edge.get("source", -1))), int(edge.get("target_node", edge.get("target", -1))))
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            pair = (int(edge[0]), int(edge[1]))
        else:
            raise ValueError(f"malformed graph edge for {digest}: {edge!r}")
        if pair[0] < 0 or pair[1] >= len(normalized) or pair[0] >= pair[1]:
            raise ValueError(f"invalid directed graph edge for {digest}: {pair}")
        supplied_edges.append(pair)
    if tuple(sorted(set(supplied_edges))) != expected_edges:
        raise ValueError(f"graph edge list differs from wire-touch DAG contract for {digest}")
    if int(record.get("node_count", len(normalized))) != len(normalized):
        raise ValueError(f"node_count differs from graph for {digest}")
    if int(record.get("edge_count", len(expected_edges))) != len(expected_edges):
        raise ValueError(f"edge_count differs from graph for {digest}")
    return NormalizedGraph(digest, num_qubits, num_clbits, tuple(normalized), expected_edges)


def fit_gate_vocabulary(graphs_by_view: Mapping[str, Mapping[str, NormalizedGraph]], train_hashes: Iterable[str]) -> dict[str, int]:
    names: set[str] = set()
    for digest in train_hashes:
        for view in ("source", "transpiled"):
            graph = graphs_by_view[view].get(digest)
            if graph is not None:
                names.update(node["operation"] for node in graph.nodes)
    return {UNKNOWN_GATE: 0, **{name: index for index, name in enumerate(sorted(names), start=1)}}


def encode_graph(graph: NormalizedGraph, vocabulary: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Encode K + 265 fixed node fields and directed edges; unknown gates map to 0."""
    if not vocabulary or vocabulary.get(UNKNOWN_GATE) != 0:
        raise ValueError("graph vocabulary must reserve __UNKNOWN__ at index 0")
    gate_width = len(vocabulary)
    rows = []
    unknown_count = 0
    for node in graph.nodes:
        row = [0.0] * gate_width
        operation_index = vocabulary.get(node["operation"], 0)
        if operation_index == 0:
            unknown_count += 1
        row[operation_index] = 1.0
        quantum_mask = [0.0] * MAX_WIRES
        classical_mask = [0.0] * MAX_WIRES
        for wire in node["qargs"]:
            quantum_mask[wire] = 1.0
        for wire in node["cargs"]:
            classical_mask[wire] = 1.0
        row.extend(quantum_mask)
        row.extend(classical_mask)
        row.extend(node["parameter_values_over_pi"])
        row.extend(node["parameter_presence_masks"])
        row.extend([
            node["quantum_arity_normalized"],
            node["classical_arity_normalized"],
            node["instruction_index_normalized"],
        ])
        rows.append(row)
    features = np.asarray(rows, dtype=np.float32)
    edges = np.asarray(graph.edges, dtype=np.int64)
    if edges.size == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        edge_index = edges.T
    if features.shape[1] != gate_width + 265 or not np.isfinite(features).all():
        raise ValueError("encoded graph feature width/non-finite check failed")
    return features, edge_index, unknown_count


@dataclass
class InputBundle:
    artifact_dir: Path
    protocol: dict[str, Any]
    dictionary: dict[str, Any]
    members: list[dict[str, Any]]
    hashes: dict[str, dict[str, Any]]
    global_names: dict[str, tuple[str, ...]]
    global_values: dict[str, dict[str, tuple[float, ...]]]
    graphs: dict[str, dict[str, NormalizedGraph]]
    representation_status: dict[str, str]
    input_hashes: dict[str, str]
    source_hashes_sha256: str
    representation_manifest_sha256: str


def _mapping_or_list_hash_rows(value: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, dict):
        if all(isinstance(record, dict) for record in value.values()):
            return {str(digest): record for digest, record in value.items()}
    if isinstance(value, list):
        result = {}
        for record in value:
            if isinstance(record, dict) and record.get("source_sha256"):
                result[str(record["source_sha256"])] = record
        return result
    return {}


def _verify_provenance_outputs(artifact_dir: Path, source_hashes: Mapping[str, Any]) -> None:
    outputs = source_hashes.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("source_hashes.outputs must be an object")
    artifact_root = artifact_dir.resolve()
    for raw_path, record in outputs.items():
        path = Path(raw_path)
        if path.name == "source_hashes.json":
            continue
        candidate = path.resolve() if path.is_absolute() else (artifact_root / path).resolve()
        if not candidate.is_relative_to(artifact_root):
            raise ValueError(f"E2 output path escapes its artifact root: {raw_path}")
        if not candidate.is_file():
            raise FileNotFoundError(f"E2 output pinned in source_hashes.json is missing: {raw_path}")
        expected_hash = record.get("sha256") if isinstance(record, dict) else record
        if expected_hash and sha256_file(candidate) != expected_hash:
            raise ValueError(f"E2 output hash mismatch for {raw_path}")
        if isinstance(record, dict) and record.get("bytes") is not None and candidate.stat().st_size != int(record["bytes"]):
            raise ValueError(f"E2 output byte count mismatch for {raw_path}")


def load_inputs(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> InputBundle:
    """Strictly join E2 rows, validate the frozen folds and decode both graph schemas."""
    missing = [name for name in INPUT_FILENAMES if not (artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("E2 materialization is incomplete; missing: " + ", ".join(missing))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    dictionary = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))
    local_cell = protocol["local_cell"]
    split_fit = protocol["split_and_fit"]
    if protocol.get("artifact_directory") != str(artifact_dir.relative_to(ROOT)):
        raise ValueError("artifact directory differs from the locked Azizov protocol")
    if tuple(split_fit["outer_folds"]) != FROZEN_FOLDS or tuple(split_fit["neural_seeds"]) != NEURAL_SEEDS:
        raise ValueError("frozen folds/seeds differ from this runner contract")
    if (int(local_cell["assigned_members"]), int(local_cell["unique_exact_qasm_hashes"])) != (MEMBER_COUNT, HASH_COUNT):
        raise ValueError("protocol assigned envelope differs from 162 members / 150 exact hashes")

    source_hashes_path = artifact_dir / "source_hashes.json"
    source_hashes = json.loads(source_hashes_path.read_text(encoding="utf-8"))
    _verify_provenance_outputs(artifact_dir, source_hashes)
    representation_path = artifact_dir / "representation_manifest.json"
    representation = json.loads(representation_path.read_text(encoding="utf-8"))
    if representation.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("representation manifest protocol_id differs from locked protocol")
    if int(representation.get("assigned_member_count", -1)) != MEMBER_COUNT or int(representation.get("unique_hash_count", -1)) != HASH_COUNT:
        raise ValueError("representation manifest does not cover 162 members / 150 hashes")

    labels_header, label_rows = read_csv(artifact_dir / "labels_core.csv")
    required_label_columns = {"panel_member_id", "source_sha256", "fold", "observed_seconds", "target_clock"}
    if not required_label_columns.issubset(labels_header):
        raise ValueError(f"labels_core.csv missing columns: {sorted(required_label_columns - set(labels_header))}")
    if len(label_rows) != MEMBER_COUNT or len({row["panel_member_id"] for row in label_rows}) != MEMBER_COUNT:
        raise ValueError("labels_core.csv must contain exactly 162 unique panel members")
    members = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in label_rows:
        if row["target_clock"] != local_cell["evaluation_target_clock"]:
            raise ValueError(f"target clock mismatch for {row['panel_member_id']}")
        digest = row["source_sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid source SHA-256 for {row['panel_member_id']}")
        fold = int(row["fold"])
        if fold not in FROZEN_FOLDS:
            raise ValueError(f"invalid outer fold for {row['panel_member_id']}")
        target = finite_float(row["observed_seconds"], field=f"{row['panel_member_id']}.observed_seconds")
        if target < 0:
            raise ValueError(f"negative runtime label for {row['panel_member_id']}")
        member = {
            "panel_member_id": row["panel_member_id"],
            "circuit_id": row.get("circuit_id", ""),
            "source_sha256": digest,
            "fold": fold,
            "observed_seconds": target,
        }
        members.append(member)
        grouped[digest].append(member)
    if len(grouped) != HASH_COUNT:
        raise ValueError(f"expected 150 exact hashes, got {len(grouped)}")
    hashes = {}
    for digest, rows in grouped.items():
        folds = {row["fold"] for row in rows}
        if len(folds) != 1:
            raise ValueError(f"exact hash crosses outer folds: {digest}")
        hashes[digest] = {
            "source_sha256": digest,
            "fold": next(iter(folds)),
            "members": rows,
            "member_ids": sorted(row["panel_member_id"] for row in rows),
            "target_seconds": float(statistics.median(row["observed_seconds"] for row in rows)),
        }

    audit_header, audit_rows = read_csv(artifact_dir / "fold_assignments_and_hash_audit.csv")
    if len(audit_rows) != MEMBER_COUNT or {row.get("panel_member_id") for row in audit_rows} != {row["panel_member_id"] for row in members}:
        raise ValueError("fold/hash audit does not exactly join all 162 members")
    label_by_member = {row["panel_member_id"]: row for row in members}
    for row in audit_rows:
        member = label_by_member[row["panel_member_id"]]
        if row.get("source_sha256") != member["source_sha256"]:
            raise ValueError(f"fold/hash audit hash mismatch for {row['panel_member_id']}")
        expected = int.from_bytes(hashlib.sha256(("20260925-split|" + member["source_sha256"]).encode("ascii")).digest()[:8], "big") % 5
        for field in ("c44_fold", "computed_fold"):
            if int(row.get(field, -1)) != expected:
                raise ValueError(f"frozen {field} mismatch for {row['panel_member_id']}")
        if str(row.get("fold_match", "")).lower() not in ("true", "1") or member["fold"] != expected:
            raise ValueError(f"fold audit failed for {row['panel_member_id']}")

    dictionary_globals = tuple(dictionary["ordered_global_fields"])
    dictionary_compiled = tuple(dictionary["ordered_compiled_extension_fields"])
    if len(dictionary_globals) != 51 or len(dictionary_compiled) != 13:
        raise ValueError("frozen local feature dictionary must define exactly 51 + 13 fields")
    coverage_header, coverage_rows = read_csv(artifact_dir / "representation_coverage_and_terminal_statuses.csv")
    required_coverage_columns = {
        "source_sha256", "fold", "member_ids_json", "source_gnn_status",
        "hybrid_gnn_status", "transpiled_gnn_status", "terminal_reason",
    }
    if not required_coverage_columns.issubset(coverage_header) or len(coverage_rows) != HASH_COUNT:
        raise ValueError("representation coverage does not match E2 controlled-trio schema")
    representation_status = {}
    for row in coverage_rows:
        digest = row["source_sha256"]
        if digest not in hashes or digest in representation_status or int(row["fold"]) != hashes[digest]["fold"]:
            raise ValueError(f"coverage/hash/fold mismatch for {digest}")
        member_ids = sorted(json.loads(row["member_ids_json"]))
        if member_ids != hashes[digest]["member_ids"]:
            raise ValueError(f"coverage member IDs mismatch for {digest}")
        states = [row[name] for name in ("source_gnn_status", "hybrid_gnn_status", "transpiled_gnn_status")]
        if len(set(states)) != 1 or states[0] not in ("available", "unavailable"):
            raise ValueError(f"controlled trio status differs or is invalid for {digest}")
        representation_status[digest] = states[0]
    if set(representation_status) != set(hashes):
        raise ValueError("representation coverage omits one or more exact hashes")

    expected_header = [
        "source_sha256",
        *[f"source__{name}" for name in dictionary_globals],
        *[f"compiled_ext__{name}" for name in dictionary_compiled],
        *[f"transpiled__{name}" for name in dictionary_globals],
    ]
    globals_header, global_rows = read_csv(artifact_dir / "global_features.csv")
    if globals_header != expected_header or len(global_rows) != HASH_COUNT:
        raise ValueError("global_features.csv columns/order or 150-hash envelope differs from E2 contract")
    global_names = {
        "source": tuple(expected_header[1:52]),
        "hybrid": tuple(expected_header[1:65]),
        "transpiled": tuple(expected_header[65:]),
    }
    global_values: dict[str, dict[str, tuple[float, ...]]] = {view: {} for view in VIEW_NAMES}
    for row in global_rows:
        digest = row["source_sha256"]
        if digest not in hashes or digest in global_values["source"]:
            raise ValueError(f"duplicate or unexpected hash in global_features.csv: {digest}")
        if representation_status[digest] == "available":
            source = tuple(finite_float(row[field], field=field) for field in global_names["source"])
            hybrid = tuple(finite_float(row[field], field=field) for field in global_names["hybrid"])
            transpiled = tuple(finite_float(row[field], field=field) for field in global_names["transpiled"])
        else:
            # Unavailable representations are retained in the coverage ledger;
            # their blank/non-finite inputs never enter a model or get imputed.
            source = hybrid = transpiled = ()
        global_values["source"][digest] = source
        global_values["hybrid"][digest] = hybrid
        global_values["transpiled"][digest] = transpiled
    if set(global_values["source"]) != set(hashes):
        raise ValueError("global feature hashes differ from the 150-label hash envelope")

    replay_header, replay_rows = read_csv(artifact_dir / "compiled_replay_audit.csv")
    if len(replay_rows) != HASH_COUNT or {row.get("source_sha256") for row in replay_rows} != set(hashes):
        raise ValueError("compiled replay audit does not cover the exact hash envelope")
    replay_by_hash = {row["source_sha256"]: row for row in replay_rows}
    match_fields = ("physical_width_match", "physical_depth_match", "physical_ops_match", "physical_two_qubit_ops_match", "swap_count_match", "dag_node_count_match")
    for digest, status in representation_status.items():
        audit = replay_by_hash[digest]
        if status == "available" and (audit.get("replay_status") != "PASS" or any(str(audit.get(field, "")).lower() not in ("true", "1") for field in match_fields)):
            raise ValueError(f"available representation lacks a passing compiled replay audit: {digest}")

    raw_graphs = {
        "source": read_jsonl(artifact_dir / "source_graphs.jsonl"),
        "transpiled": read_jsonl(artifact_dir / "transpiled_graphs.jsonl"),
    }
    normalized_graphs: dict[str, dict[str, NormalizedGraph]] = {"source": {}, "transpiled": {}}
    for view, rows in raw_graphs.items():
        if len(rows) != HASH_COUNT:
            raise ValueError(f"{view} graph JSONL must contain one row per exact hash")
        for raw in rows:
            digest = _record_hash(raw)
            if digest not in hashes or digest in normalized_graphs[view]:
                raise ValueError(f"duplicate or unexpected {view} graph hash: {digest}")
            if representation_status[digest] == "available":
                normalized_graphs[view][digest] = normalize_graph_record(raw)
        expected_available = {digest for digest, state in representation_status.items() if state == "available"}
        if set(normalized_graphs[view]) != expected_available:
            raise ValueError(f"{view} graph rows do not exactly cover representation-available hashes")

    required_outputs = {
        name: sha256_file(artifact_dir / name)
        for name in INPUT_FILENAMES
    }
    required_outputs["protocol"] = sha256_file(PROTOCOL_PATH)
    required_outputs["feature_dictionary"] = sha256_file(DICTIONARY_PATH)
    required_outputs["runner"] = sha256_file(Path(__file__).resolve())
    required_outputs["fit_requirements"] = sha256_file(FIT_REQUIREMENTS_PATH)
    return InputBundle(
        artifact_dir=artifact_dir,
        protocol=protocol,
        dictionary=dictionary,
        members=members,
        hashes=hashes,
        global_names=global_names,
        global_values=global_values,
        graphs=normalized_graphs,
        representation_status=representation_status,
        input_hashes=required_outputs,
        source_hashes_sha256=sha256_file(source_hashes_path),
        representation_manifest_sha256=sha256_file(representation_path),
    )


def method_metadata() -> dict[str, dict[str, Any]]:
    result = {}
    for view in VIEW_NAMES:
        for seed in NEURAL_SEEDS:
            method_id = f"azizov_gnn_{view}_seed_{seed}"
            result[method_id] = {"method_id": method_id, "method_family": "azizov_gnn", "view": view, "seed": seed, "model": "transformer_conv"}
    for model in CLASSICAL_NAMES:
        for view in VIEW_NAMES:
            method_id = f"classical_{model}_{view}"
            result[method_id] = {"method_id": method_id, "method_family": f"classical_{model}", "view": view, "seed": CLASSICAL_SEED, "model": model}
    if set(result) != set(METHOD_IDS):
        raise AssertionError("method-family definition drift")
    return result


def terminal_rows_for_hashes(
    bundle: InputBundle,
    method: Mapping[str, Any],
    hashes: Sequence[str],
    status: str,
    reason: str,
    predictions: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    predictions = predictions or {}
    output = []
    for digest in hashes:
        record = bundle.hashes[digest]
        for member in record["members"]:
            predicted = predictions.get(digest)
            if predicted is not None and (not math.isfinite(float(predicted)) or float(predicted) < 0):
                raise ValueError(f"invalid predicted seconds for {method['method_id']} {digest}")
            row_status = "predicted" if predicted is not None else status
            output.append({
                "panel_member_id": member["panel_member_id"],
                "source_sha256": digest,
                "fold": record["fold"],
                "method_id": method["method_id"],
                "method_family": method["method_family"],
                "view": method["view"],
                "seed": method["seed"],
                "target_seconds": record["target_seconds"],
                "member_observed_seconds": member["observed_seconds"],
                "prediction_seconds": "" if predicted is None else float(predicted),
                "status": row_status,
                "terminal_reason": "" if predicted is not None else reason,
            })
    return output


def build_terminal_ledger(
    bundle: InputBundle,
    completed_rows: Sequence[Mapping[str, Any]],
    *,
    stop_after_fold: int | None = None,
    stop_reason: str = "",
) -> list[dict[str, Any]]:
    """Ensure exactly one terminal member row for every method/assignment."""
    by_key = {(row["method_id"], row["panel_member_id"]): dict(row) for row in completed_rows}
    if len(by_key) != len(completed_rows):
        raise ValueError("duplicate member/method result in terminal ledger")
    methods = method_metadata()
    ledger = []
    for method_id, method in methods.items():
        for digest, hash_record in bundle.hashes.items():
            missing_members = [
                member for member in hash_record["members"]
                if (method_id, member["panel_member_id"]) not in by_key
            ]
            for member in hash_record["members"]:
                key = (method_id, member["panel_member_id"])
                if key in by_key:
                    ledger.append(by_key[key])
            if not missing_members:
                continue
            if stop_after_fold is not None and hash_record["fold"] > stop_after_fold:
                status = "not_run_fold0_technical_checkpoint" if stop_after_fold == 0 else "not_run_after_stop"
                reason = stop_reason
            elif bundle.representation_status[digest] == "unavailable":
                status, reason = "unavailable_representation", "E2 controlled-trio gate unavailable"
            else:
                status, reason = "missing_terminal_record", "runner did not record a result"
            for member in missing_members:
                ledger.append({
                    "panel_member_id": member["panel_member_id"],
                    "source_sha256": digest,
                    "fold": hash_record["fold"],
                    "method_id": method["method_id"],
                    "method_family": method["method_family"],
                    "view": method["view"],
                    "seed": method["seed"],
                    "target_seconds": hash_record["target_seconds"],
                    "member_observed_seconds": member["observed_seconds"],
                    "prediction_seconds": "",
                    "status": status,
                    "terminal_reason": reason,
                })
    seen = {(row["method_id"], row["panel_member_id"]) for row in ledger}
    expected = len(methods) * len(bundle.members)
    if len(ledger) != expected or len(seen) != expected:
        raise ValueError("terminal ledger does not have exactly one row per method/member")
    return ledger


def _eligible_hashes(bundle: InputBundle, view: str, fold: int) -> list[str]:
    return sorted(
        digest for digest, record in bundle.hashes.items()
        if record["fold"] == fold and bundle.representation_status[digest] == "available"
        and (view in ("source", "hybrid", "transpiled"))
    )


def _global_matrix(bundle: InputBundle, view: str, hashes: Sequence[str]) -> np.ndarray:
    return np.asarray([bundle.global_values[view][digest] for digest in hashes], dtype=np.float64)


def _classical_estimator(name: str) -> Any:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    if name == "linear_regression":
        return LinearRegression(fit_intercept=True, n_jobs=1)
    if name == "ridge":
        return Ridge(alpha=1.0, fit_intercept=True, solver="svd")
    if name == "svr_rbf":
        return SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale", tol=0.001, max_iter=-1)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_split=2,
            min_samples_leaf=2, max_features=1.0, bootstrap=True,
            random_state=CLASSICAL_SEED, n_jobs=2,
        )
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            objective="reg:squarederror", n_estimators=300, max_depth=3,
            learning_rate=0.05, subsample=1.0, colsample_bytree=1.0,
            reg_lambda=1.0, reg_alpha=0.0, tree_method="hist", device="cpu",
            random_state=CLASSICAL_SEED, n_jobs=2,
        )
    raise ValueError(f"unknown classical baseline: {name}")


def _inverse_log_predictions(values: Sequence[float], *, method: str) -> np.ndarray:
    logged = np.asarray(values, dtype=np.float64)
    if not np.isfinite(logged).all():
        raise ValueError(f"non-finite log predictions for {method}")
    with np.errstate(over="ignore", invalid="ignore"):
        predictions = np.expm1(logged)
    predictions = np.maximum(0.0, predictions)
    if not np.isfinite(predictions).all():
        raise ValueError(f"log prediction overflows expm1 for {method}")
    return predictions


def run_classical_method(bundle: InputBundle, view: str, model_name: str, fold: int) -> tuple[dict[str, float], dict[str, Any]]:
    method_id = f"classical_{model_name}_{view}"
    train_hashes = _eligible_hashes(bundle, view, fold)
    test_hashes = _eligible_hashes(bundle, view, fold)
    train_hashes = sorted(digest for digest, rec in bundle.hashes.items() if rec["fold"] != fold and bundle.representation_status[digest] == "available")
    test_hashes = sorted(digest for digest, rec in bundle.hashes.items() if rec["fold"] == fold and bundle.representation_status[digest] == "available")
    if len(train_hashes) < 2 or not test_hashes:
        raise ValueError(f"no adequate eligible hash train/test set for {method_id}, fold={fold}")
    names = bundle.global_names[view]
    scaler = fit_global_scaler(_global_matrix(bundle, view, train_hashes), names)
    x_train = scaler.transform(_global_matrix(bundle, view, train_hashes))
    x_test = scaler.transform(_global_matrix(bundle, view, test_hashes))
    y_train = np.log1p(np.asarray([bundle.hashes[digest]["target_seconds"] for digest in train_hashes], dtype=np.float64))
    if not np.isfinite(y_train).all():
        raise ValueError(f"non-finite transformed target for {method_id}")
    estimator = _classical_estimator(model_name)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        estimator.fit(x_train, y_train)
        predicted_log = estimator.predict(x_test)
    predicted_seconds = _inverse_log_predictions(predicted_log, method=method_id)
    return dict(zip(test_hashes, (float(value) for value in predicted_seconds))), {
        "train_hashes": train_hashes,
        "test_hashes": test_hashes,
        "scaler": scaler_record(scaler),
    }


def scaler_record(scaler: GlobalScaler) -> dict[str, Any]:
    return {
        "input_names": list(scaler.input_names),
        "retained_names": list(scaler.retained_names),
        "removed_zero_variance_names": list(scaler.removed_zero_variance_names),
        "means": list(scaler.means),
        "population_scales": list(scaler.scales),
        "all_constant_replaced_by_explicit_zero_column": scaler.all_constant,
    }


def _torch_modules() -> tuple[Any, Any, Any, Any]:
    import torch
    from torch import nn
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import TransformerConv, global_mean_pool
    return torch, nn, (Data, DataLoader), (TransformerConv, global_mean_pool)


def cuda_environment_and_smoke() -> dict[str, Any]:
    """Capture required CUDA facts and run a tiny PyG operation under the shared lock."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch, nn, _, geom = _torch_modules()
    TransformerConv, _ = geom
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is mandatory for Azizov GNN training and inference")
    device = torch.device("cuda:0")
    smoke = TransformerConv(3, 4, heads=1, concat=True, beta=False, dropout=0.0, root_weight=True).to(device)
    x = torch.ones((2, 3), dtype=torch.float32, device=device)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    result = smoke(x, edge_index)
    torch.cuda.synchronize(device)
    if result.shape != (2, 4) or not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("CUDA/PyG TransformerConv smoke operation failed")
    properties = torch.cuda.get_device_properties(0)
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda),
        "torch_geometric": str(metadata.version("torch-geometric")),
        "cuda_available": True,
        "device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "device_total_memory_bytes": int(properties.total_memory),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_operation_smoke": "PASS",
        "cuda_smoke_operation": "TransformerConv 3->4 forward on two nodes/one edge; float32",
        "precision": "float32_no_amp_tf32_disabled",
        "deterministic_algorithms": "required_during_fit",
        "shared_timing_lock_path": str(ROOT / "work/locks/maestro_qcsim_v2_cpu_timing.lock"),
        "host_gpu_lease_path": "/tmp/qre-benchmark-gpu-lease.lock",
        "both_locks_held_for_smoke_and_cuda_fit": True,
        "live_timing_workers_checked_by_shared_lock_helper": True,
    }
    return environment


FIT_PACKAGE_PINS = {
    "numpy": "1.26.4",
    "scikit-learn": "1.7.2",
    "torch": "2.7.1+cu128",
    "torch-geometric": "2.6.1",
    "xgboost": "2.1.4",
}
RUNTIME_LOCK_SCHEMA = "azizov-fit-runtime-lock-v1"
REQUIRED_RUNTIME_LOCK_VERSIONS = frozenset({
    "python_version", "numpy", "scikit-learn", "xgboost", "torch", "torch-geometric", "cuda_runtime",
})


def verify_fit_packages() -> dict[str, str]:
    """Fail before fold 0 unless the exact implementation pins are installed."""
    found = {}
    errors = []
    for package, expected in FIT_PACKAGE_PINS.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            errors.append(f"{package} is missing (required {expected})")
            continue
        found[package] = actual
        if actual != expected:
            errors.append(f"{package}={actual}, required={expected}")
    if errors:
        raise RuntimeError("fitting package pins do not match requirements-azizov-fit.txt: " + "; ".join(errors))
    return found


def runtime_version_snapshot() -> dict[str, str]:
    """Read the fit interpreter's full dynamic versions without touching CUDA."""
    import torch
    return {
        "python_version": platform.python_version(),
        "numpy": str(np.__version__),
        "scikit-learn": metadata.version("scikit-learn"),
        "xgboost": metadata.version("xgboost"),
        "torch": str(torch.__version__),
        "torch-geometric": metadata.version("torch-geometric"),
        "cuda_runtime": str(torch.version.cuda),
    }


def runtime_version_mismatches(expected: Mapping[str, Any], actual: Mapping[str, str]) -> list[str]:
    """Compare a frozen runtime lock's package map with actual imported metadata."""
    expected_versions = expected.get("expected_versions", expected.get("versions", {}))
    if not isinstance(expected_versions, dict) or not expected_versions:
        raise ValueError("runtime lock must declare non-empty expected_versions")
    return [
        f"{name}: expected {value}, actual {actual.get(name, '<missing>')}"
        for name, value in sorted(expected_versions.items())
        if str(actual.get(name, "<missing>")) != str(value)
    ]


def load_cuda_base_lock(path: Path) -> dict[str, Any]:
    """Read a reusable CUDA/PyG base lock without requiring its old interpreter path."""
    document = json.loads(path.read_text(encoding="utf-8"))
    environment = (
        document.get("cuda_training_environment")
        or document.get("cuda_training_environment_preflight")
        or document
    )
    python_text = str(document.get("python", environment.get("python_version", "")))
    python_version = python_text.split(" ", 1)[0] if python_text else ""
    expected_versions = {
        "torch": str(environment.get("torch", "")),
        "torch-geometric": str(environment.get("torch_geometric", environment.get("torch-geometric", ""))),
        "cuda_runtime": str(environment.get("torch_cuda_runtime", environment.get("cuda_runtime", ""))),
    }
    if python_version:
        expected_versions["python_version"] = python_version
    expected_versions = {key: value for key, value in expected_versions.items() if value}
    required = {"torch", "torch-geometric", "cuda_runtime"}
    if not required.issubset(expected_versions):
        raise ValueError(f"CUDA base lock is missing required runtime fields: {sorted(required - set(expected_versions))}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "expected_versions": expected_versions,
        "environment": environment,
        "base_pip_freeze_sha256": environment.get("pip_freeze_sha256"),
    }


def validate_cuda_base_versions(base_lock: Mapping[str, Any], actual_versions: Mapping[str, str]) -> list[str]:
    return runtime_version_mismatches(base_lock, actual_versions)


def validate_cuda_device_lock(base_lock: Mapping[str, Any], actual_cuda: Mapping[str, Any]) -> list[str]:
    expected = base_lock.get("environment", {})
    mismatches = []
    for key in ("cuda_available", "device_count", "device_name", "device_capability", "cuda_visible_devices"):
        if key in expected and expected[key] is not None and actual_cuda.get(key) != expected[key]:
            mismatches.append(f"{key}: expected {expected[key]!r}, actual {actual_cuda.get(key)!r}")
    return mismatches


def validate_runtime_lock_document(
    document: Mapping[str, Any],
    *,
    actual_versions: Mapping[str, str],
    overlay_sha256: str | None,
    overlay_freeze_sha256: str | None,
    base_cuda_lock_sha256: str,
    environment_freeze_sha256: str | None = None,
) -> list[str]:
    """Pure exact lock check; suitable for tests without a CUDA device."""
    mismatches = []
    if document.get("schema_version") != RUNTIME_LOCK_SCHEMA:
        mismatches.append(f"schema_version: expected {RUNTIME_LOCK_SCHEMA}, actual {document.get('schema_version', '<missing>')}")
    expected_versions = document.get("expected_versions", document.get("versions", {}))
    if not isinstance(expected_versions, dict):
        expected_versions = {}
    missing_versions = sorted(REQUIRED_RUNTIME_LOCK_VERSIONS - set(expected_versions))
    if missing_versions:
        mismatches.append("expected_versions missing required fields: " + ", ".join(missing_versions))
    if expected_versions:
        mismatches.extend(runtime_version_mismatches(document, actual_versions))
    expected_overlay = document.get("overlay_sha256")
    if overlay_sha256 is not None and expected_overlay is None:
        mismatches.append("runtime lock must pin overlay_sha256 when --overlay-path is supplied")
    if expected_overlay is not None and expected_overlay != overlay_sha256:
        mismatches.append(f"overlay_sha256: expected {expected_overlay}, actual {overlay_sha256}")
    expected_overlay_freeze = document.get("overlay_pip_freeze_sha256", document.get("uv_pip_freeze_sha256"))
    if overlay_freeze_sha256 is not None and expected_overlay_freeze is None:
        mismatches.append("runtime lock must pin overlay_pip_freeze_sha256 when --overlay-path is supplied")
    if expected_overlay_freeze is not None and expected_overlay_freeze != overlay_freeze_sha256:
        mismatches.append(f"overlay_pip_freeze_sha256: expected {expected_overlay_freeze}, actual {overlay_freeze_sha256}")
    expected_environment_freeze = document.get("pip_freeze_sha256", document.get("environment_pip_freeze_sha256"))
    if expected_environment_freeze is not None and expected_environment_freeze != environment_freeze_sha256:
        mismatches.append(f"environment_pip_freeze_sha256: expected {expected_environment_freeze}, actual {environment_freeze_sha256}")
    expected_base = document.get("base_cuda_lock_sha256")
    if expected_base is None:
        mismatches.append("runtime lock must pin base_cuda_lock_sha256")
    if expected_base is not None and expected_base != base_cuda_lock_sha256:
        mismatches.append(f"base_cuda_lock_sha256: expected {expected_base}, actual {base_cuda_lock_sha256}")
    return mismatches


def sha256_tree(path: Path) -> str:
    """Hash a site-packages overlay deterministically, excluding bytecode caches."""
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"runtime overlay does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(
        candidate for candidate in path.rglob("*")
        if candidate.is_file() and "__pycache__" not in candidate.parts and candidate.suffix != ".pyc"
    )
    if not files:
        raise ValueError(f"runtime overlay contains no files: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def package_freeze_snapshot(overlay_path: Path | None = None, uv_executable: str = "uv") -> tuple[bytes, str, bytes | None, str | None]:
    """Mirror E3's pip-to-uv fallback and capture an overlay-only uv lock."""
    pip_result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
    )
    if pip_result.returncode == 0:
        environment_bytes = pip_result.stdout
        environment_tool = "python-m-pip-freeze-all"
    else:
        uv_path = shutil.which(uv_executable)
        if uv_path is None:
            detail = pip_result.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["unknown pip error"]
            raise RuntimeError("cannot record package lock: pip freeze failed and uv was not found: " + "; ".join(detail))
        uv_command = [uv_path, "pip", "freeze", "--python", sys.executable]
        uv_result = subprocess.run(uv_command, capture_output=True)
        if uv_result.returncode != 0:
            detail = uv_result.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["unknown uv error"]
            raise RuntimeError("cannot record package lock: pip freeze failed and uv fallback failed: " + "; ".join(detail))
        environment_bytes = uv_result.stdout
        environment_tool = "uv-pip-freeze-python"

    overlay_bytes = None
    overlay_tool = None
    if overlay_path is not None:
        uv_path = shutil.which(uv_executable)
        if uv_path is None:
            raise RuntimeError("an overlay was supplied but uv is not available for `uv pip freeze --target`")
        overlay_result = subprocess.run(
            [uv_path, "pip", "freeze", "--target", str(overlay_path.resolve())],
            capture_output=True,
        )
        if overlay_result.returncode != 0:
            detail = overlay_result.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["unknown uv error"]
            raise RuntimeError("cannot freeze package overlay with `uv pip freeze --target`: " + "; ".join(detail))
        overlay_bytes = overlay_result.stdout
        overlay_tool = "uv-pip-freeze-target"
    return environment_bytes, environment_tool, overlay_bytes, overlay_tool


def capture_pip_freeze(
    output_dir: Path,
    *,
    overlay_path: Path | None = None,
    uv_executable: str = "uv",
) -> dict[str, Any]:
    environment_bytes, environment_tool, overlay_bytes, overlay_tool = package_freeze_snapshot(overlay_path, uv_executable)
    freeze_path = output_dir / "environment_pip_freeze.txt"
    if freeze_path.exists():
        raise FileExistsError(f"refusing to overwrite fitting lock: {freeze_path}")
    freeze_path.write_bytes(environment_bytes)
    result = {
        "pip_freeze_path": freeze_path.name,
        "pip_freeze_sha256": sha256_bytes(environment_bytes),
        "pip_freeze_line_count": len(environment_bytes.decode("utf-8").splitlines()),
        "pip_freeze_tool": environment_tool,
        "overlay_pip_freeze_path": None,
        "overlay_pip_freeze_sha256": None,
        "overlay_pip_freeze_line_count": None,
        "overlay_pip_freeze_tool": overlay_tool,
    }
    if overlay_bytes is not None:
        overlay_freeze_path = output_dir / "overlay_pip_freeze.txt"
        if overlay_freeze_path.exists():
            raise FileExistsError(f"refusing to overwrite overlay lock: {overlay_freeze_path}")
        overlay_freeze_path.write_bytes(overlay_bytes)
        result.update({
            "overlay_pip_freeze_path": overlay_freeze_path.name,
            "overlay_pip_freeze_sha256": sha256_bytes(overlay_bytes),
            "overlay_pip_freeze_line_count": len(overlay_bytes.decode("utf-8").splitlines()),
        })
    return result


def _seed_everything(seed: int) -> Any:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch, _, _, _ = _torch_modules()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    return torch


def _make_data(
    graph: NormalizedGraph,
    vocabulary: Mapping[str, int],
    globals_vector: Sequence[float],
    scaler: GlobalScaler,
    digest: str,
    target: float | None = None,
) -> tuple[Any, int]:
    torch, _, (Data, _), _ = _torch_modules()
    graph_x, edge_index, unknown_count = encode_graph(graph, vocabulary)
    global_x = scaler.transform([globals_vector]).astype(np.float32)
    data = Data(
        x=torch.tensor(graph_x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        global_x=torch.tensor(global_x, dtype=torch.float32),
        source_sha256=digest,
    )
    if target is not None:
        data.y = torch.tensor([math.log1p(target)], dtype=torch.float32)
    return data, unknown_count


def _build_azizov_model(input_node_dim: int, global_dim: int) -> Any:
    torch, nn, _, geom = _torch_modules()
    TransformerConv, global_mean_pool = geom

    class AzizovRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = TransformerConv(input_node_dim, 178, heads=1, concat=True, beta=False, dropout=0.0, root_weight=True)
            self.conv2 = TransformerConv(178, 178, heads=1, concat=True, beta=False, dropout=0.0, root_weight=True)
            self.conv3 = TransformerConv(178, 178, heads=1, concat=True, beta=False, dropout=0.0, root_weight=True)
            self.global_mlp = nn.Sequential(nn.Linear(global_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            self.head = nn.Sequential(
                nn.Linear(242, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 128), nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, batch: Any) -> Any:
            x = torch.relu(self.conv1(batch.x, batch.edge_index))
            x = torch.relu(self.conv2(x, batch.edge_index))
            x = torch.relu(self.conv3(x, batch.edge_index))
            graph_x = global_mean_pool(x, batch.batch)
            global_x = self.global_mlp(batch.global_x.reshape(int(batch.num_graphs), -1))
            return self.head(torch.cat((graph_x, global_x), dim=1)).reshape(-1)

    return AzizovRegressor()


def _data_loader(items: Sequence[Any], *, shuffle: bool, seed: int, batch_size: int = 8) -> Any:
    torch, _, (_, DataLoader), _ = _torch_modules()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(list(items), batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator if shuffle else None)


def _train_epoch(model: Any, loader: Any, optimizer: Any, device: Any, seed: int) -> float:
    torch, nn, _, _ = _torch_modules()
    model.train()
    loss_sum = 0.0
    count = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch)
        loss = nn.functional.mse_loss(predictions, batch.y.reshape(-1), reduction="mean")
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite training loss at seed {seed}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_count = int(batch.num_graphs)
        loss_sum += float(loss.detach().item()) * batch_count
        count += batch_count
    if count == 0:
        raise RuntimeError("empty training loader")
    return loss_sum / count


def _validation_loss(model: Any, loader: Any, device: Any) -> float:
    torch, nn, _, _ = _torch_modules()
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch)
            loss = nn.functional.mse_loss(predictions, batch.y.reshape(-1), reduction="sum")
            loss_sum += float(loss.item())
            count += int(batch.num_graphs)
    if count == 0:
        raise RuntimeError("empty epoch-selection validation loader")
    value = loss_sum / count
    if not math.isfinite(value):
        raise RuntimeError("non-finite validation loss")
    return value


def run_neural_method(
    bundle: InputBundle,
    view: str,
    seed: int,
    fold: int,
    *,
    smoke_passed: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Select epoch on inner hashes, then refit from scratch on outer train."""
    if not smoke_passed:
        raise RuntimeError("CUDA operation smoke must pass before GNN fitting")
    method_id = f"azizov_gnn_{view}_seed_{seed}"
    all_available = sorted(digest for digest, state in bundle.representation_status.items() if state == "available")
    train_hashes = sorted(digest for digest in all_available if bundle.hashes[digest]["fold"] != fold)
    test_hashes = sorted(digest for digest in all_available if bundle.hashes[digest]["fold"] == fold)
    if len(train_hashes) < 2 or not test_hashes:
        raise ValueError(f"no adequate eligible hash train/test set for {method_id}, fold={fold}")
    inner_train_hashes, validation_hashes = inner_validation_hashes(train_hashes)
    inner_scaler = fit_global_scaler(_global_matrix(bundle, view, inner_train_hashes), bundle.global_names[view])
    inner_vocabulary = fit_gate_vocabulary(bundle.graphs, inner_train_hashes)
    outer_scaler = fit_global_scaler(_global_matrix(bundle, view, train_hashes), bundle.global_names[view])
    outer_vocabulary = fit_gate_vocabulary(bundle.graphs, train_hashes)
    torch = _seed_everything(seed)
    device = torch.device("cuda:0")
    max_epochs = int(bundle.protocol["neural_model"]["max_epochs"])
    patience = int(bundle.protocol["neural_model"]["early_stopping_patience"])
    params = bundle.protocol["neural_model"]["optimizer"]

    def make_items(hashes: Sequence[str], vocab: Mapping[str, int], scaler: GlobalScaler, with_target: bool) -> tuple[list[Any], dict[str, int]]:
        items = []
        unknown_counts = {}
        for digest in hashes:
            target = bundle.hashes[digest]["target_seconds"] if with_target else None
            item, unknown = _make_data(bundle.graphs["source" if view in ("source", "hybrid") else "transpiled"][digest], vocab, bundle.global_values[view][digest], scaler, digest, target)
            items.append(item)
            unknown_counts[digest] = unknown
        return items, unknown_counts

    inner_train_items, _ = make_items(inner_train_hashes, inner_vocabulary, inner_scaler, True)
    validation_items, validation_unknown = make_items(validation_hashes, inner_vocabulary, inner_scaler, True)
    selection_model = _build_azizov_model(len(inner_vocabulary) + 265, len(inner_scaler.retained_names))
    selection_model = selection_model.to(device)
    optimizer = torch.optim.Adam(selection_model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
    train_loader = _data_loader(inner_train_items, shuffle=True, seed=seed, batch_size=int(bundle.protocol["neural_model"]["batch_size"]))
    validation_loader = _data_loader(validation_items, shuffle=False, seed=seed)
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        _train_epoch(selection_model, train_loader, optimizer, device, seed)
        validation_loss = _validation_loss(selection_model, validation_loader, device)
        if validation_loss < best_loss:  # ties select the earliest epoch
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_epoch < 1:
        raise RuntimeError(f"epoch selection failed for {method_id}")

    final_train_items, _ = make_items(train_hashes, outer_vocabulary, outer_scaler, True)
    test_items, test_unknown = make_items(test_hashes, outer_vocabulary, outer_scaler, False)
    final_model = _build_azizov_model(len(outer_vocabulary) + 265, len(outer_scaler.retained_names)).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
    final_loader = _data_loader(final_train_items, shuffle=True, seed=seed, batch_size=int(bundle.protocol["neural_model"]["batch_size"]))
    for _ in range(best_epoch):
        _train_epoch(final_model, final_loader, final_optimizer, device, seed)
    torch.cuda.synchronize(device)
    final_model.eval()
    prediction_by_hash: dict[str, float] = {}
    _, _, (_, DataLoader), _ = _torch_modules()
    test_loader = DataLoader(test_items, batch_size=8, shuffle=False, num_workers=0)
    log_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            values = final_model(batch).detach().cpu().numpy().astype(np.float64)
            log_predictions.extend(values.tolist())
    seconds = _inverse_log_predictions(log_predictions, method=method_id)
    if len(seconds) != len(test_hashes):
        raise RuntimeError(f"prediction cardinality mismatch for {method_id}")
    prediction_by_hash.update({digest: float(value) for digest, value in zip(test_hashes, seconds)})
    torch.cuda.synchronize(device)
    return prediction_by_hash, {
        "train_hashes": train_hashes,
        "inner_train_hashes": inner_train_hashes,
        "inner_validation_hashes": validation_hashes,
        "inner_train_hashes_sha256": sha256_bytes(("\n".join(inner_train_hashes) + "\n").encode()),
        "inner_validation_hashes_sha256": sha256_bytes(("\n".join(validation_hashes) + "\n").encode()),
        "validation_hashes": validation_hashes,
        "selected_epoch": best_epoch,
        "best_inner_validation_log1p_mse": best_loss,
        "seed": seed,
        "view": view,
        "global_scaler_inner": scaler_record(inner_scaler),
        "global_scaler_outer": scaler_record(outer_scaler),
        "inner_train_gate_vocabulary_size_including_unknown": len(inner_vocabulary),
        "outer_train_gate_vocabulary_size_including_unknown": len(outer_vocabulary),
        "validation_unknown_operation_nodes": sum(validation_unknown.values()),
        "test_unknown_operation_nodes": sum(test_unknown.values()),
        "batch_size": 8,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "precision": "float32_no_amp",
        "cuda_inference": True,
    }


def _empty_method_rows(bundle: InputBundle, fold: int, method: Mapping[str, Any], *, status: str, reason: str) -> list[dict[str, Any]]:
    test_hashes = sorted(digest for digest, row in bundle.hashes.items() if row["fold"] == fold)
    available = [digest for digest in test_hashes if bundle.representation_status[digest] == "available"]
    unavailable = [digest for digest in test_hashes if bundle.representation_status[digest] == "unavailable"]
    rows = terminal_rows_for_hashes(bundle, method, unavailable, "unavailable_representation", "E2 controlled-trio gate unavailable")
    rows += terminal_rows_for_hashes(bundle, method, available, status, reason)
    return rows


def _method_predictions(bundle: InputBundle, method: Mapping[str, Any], fold: int, predicted: Mapping[str, float]) -> list[dict[str, Any]]:
    test_hashes = sorted(digest for digest, row in bundle.hashes.items() if row["fold"] == fold)
    rows = []
    available = [digest for digest in test_hashes if bundle.representation_status[digest] == "available"]
    unavailable = [digest for digest in test_hashes if bundle.representation_status[digest] == "unavailable"]
    rows.extend(terminal_rows_for_hashes(bundle, method, unavailable, "unavailable_representation", "E2 controlled-trio gate unavailable"))
    rows.extend(terminal_rows_for_hashes(bundle, method, available, "technical_failure", "prediction absent", predicted))
    return rows


def _method_is_neural(method: Mapping[str, Any]) -> bool:
    return method["method_family"] == "azizov_gnn"


def _run_fold(
    bundle: InputBundle,
    fold: int,
    *,
    smoke: dict[str, Any] | None,
    cuda_base_lock: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    methods = method_metadata()
    rows: list[dict[str, Any]] = []
    audit: dict[str, dict[str, Any]] = {}
    available_test_hashes = [digest for digest, item in bundle.hashes.items() if item["fold"] == fold and bundle.representation_status[digest] == "available"]

    # CPU baselines do not hold the CUDA/timing exclusion locks.
    for method_id, method in methods.items():
        if _method_is_neural(method):
            continue
        try:
            predictions, record = run_classical_method(bundle, method["view"], method["model"], fold)
            rows.extend(_method_predictions(bundle, method, fold, predictions))
            audit[method_id] = {"status": "PASS", **record}
        except Exception as exc:  # terminal status is retained; no estimator substitution
            reason = f"{type(exc).__name__}: {exc}"
            rows.extend(_empty_method_rows(bundle, fold, method, status="technical_failure", reason=reason))
            audit[method_id] = {"status": "FAIL", "error_class": type(exc).__name__, "error": str(exc)}

    # All actual CUDA smoke, fitting, and inference happen under both shared locks.
    if available_test_hashes:
        try:
            from run_mps_fixed_chi16_runtime_adaptation_v1 import exclusive_compute_lock
            with exclusive_compute_lock():
                if smoke is None:
                    smoke = cuda_environment_and_smoke()
                device_mismatches = validate_cuda_device_lock(cuda_base_lock, smoke)
                if device_mismatches:
                    raise RuntimeError("live CUDA device differs from the frozen base lock: " + "; ".join(device_mismatches))
                for method_id, method in methods.items():
                    if not _method_is_neural(method):
                        continue
                    try:
                        predictions, record = run_neural_method(bundle, method["view"], int(method["seed"]), fold, smoke_passed=True)
                        rows.extend(_method_predictions(bundle, method, fold, predictions))
                        audit[method_id] = {"status": "PASS", **record}
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        rows.extend(_empty_method_rows(bundle, fold, method, status="technical_failure", reason=reason))
                        audit[method_id] = {"status": "FAIL", "error_class": type(exc).__name__, "error": str(exc)}
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            for method_id, method in methods.items():
                if _method_is_neural(method) and method_id not in audit:
                    rows.extend(_empty_method_rows(bundle, fold, method, status="blocked_compute_lock_or_cuda", reason=reason))
                    audit[method_id] = {"status": "BLOCKED", "error_class": type(exc).__name__, "error": str(exc)}
    else:
        for method_id, method in methods.items():
            if _method_is_neural(method):
                rows.extend(_empty_method_rows(bundle, fold, method, status="unavailable_representation", reason="no E2-available hash in this fold"))
                audit[method_id] = {"status": "NO_ELIGIBLE_HASHES"}

    expected_member_rows = sum(len(record["members"]) for record in bundle.hashes.values() if record["fold"] == fold)
    if len(rows) != expected_member_rows * len(methods):
        raise ValueError(f"fold {fold} emitted {len(rows)} method/member rows; expected {expected_member_rows * len(methods)}")
    keys = [(row["method_id"], row["panel_member_id"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"fold {fold} contains duplicate method/member outputs")
    qa = {
        "fold": fold,
        "role": "metric_blind_technical_integrity_checkpoint" if fold == 0 else "outer_oof_fold",
        "assigned_members": expected_member_rows,
        "assigned_hashes": sum(item["fold"] == fold for item in bundle.hashes.values()),
        "eligible_hashes": len(available_test_hashes),
        "unavailable_representation_hashes": sum(item["fold"] == fold and bundle.representation_status[digest] == "unavailable" for digest, item in bundle.hashes.items()),
        "method_families": len(methods),
        "method_member_terminal_rows": len(rows),
        "method_audit": audit,
        "status_counts": dict(Counter(row["status"] for row in rows)),
    }
    return rows, qa, {"cuda_environment": smoke} if smoke is not None else {}


def fold0_technical_gate(bundle: InputBundle, rows: Sequence[Mapping[str, Any]], qa: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    assigned_hashes = {digest for digest, item in bundle.hashes.items() if item["fold"] == 0}
    available = {digest for digest in assigned_hashes if bundle.representation_status[digest] == "available"}
    if not available:
        reasons.append("fold0 has no representation-eligible hashes")
    test_member_ids = {member["panel_member_id"] for digest in assigned_hashes for member in bundle.hashes[digest]["members"]}
    expected = test_member_ids and {(method_id, member_id) for method_id in METHOD_IDS for member_id in test_member_ids}
    actual = {(row["method_id"], row["panel_member_id"]) for row in rows}
    if actual != expected:
        reasons.append("fold0 method/member terminal envelope mismatch")
    for method_id in METHOD_IDS:
        method_rows = [row for row in rows if row["method_id"] == method_id]
        for digest in available:
            selected = [row for row in method_rows if row["source_sha256"] == digest]
            if not selected or any(row["status"] != "predicted" or row["prediction_seconds"] == "" for row in selected):
                reasons.append(f"fold0 missing finite prediction: {method_id}:{digest}")
                break
        if qa.get("method_audit", {}).get(method_id, {}).get("status") != "PASS":
            reasons.append(f"fold0 technical method failed: {method_id}")
    return not reasons, reasons


def _append_not_run_rows(bundle: InputBundle, completed_rows: list[dict[str, Any]], *, from_fold: int, reason: str) -> None:
    methods = method_metadata()
    for fold in FROZEN_FOLDS:
        if fold < from_fold:
            continue
        for method in methods.values():
            hashes = sorted(digest for digest, item in bundle.hashes.items() if item["fold"] == fold)
            unavailable = [digest for digest in hashes if bundle.representation_status[digest] == "unavailable"]
            eligible = [digest for digest in hashes if bundle.representation_status[digest] == "available"]
            completed_rows.extend(terminal_rows_for_hashes(bundle, method, unavailable, "unavailable_representation", "E2 controlled-trio gate unavailable"))
            completed_rows.extend(terminal_rows_for_hashes(bundle, method, eligible, "not_run_fold0_technical_checkpoint", reason))


def _method_hash_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    predictions: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["status"] != "predicted" or row["prediction_seconds"] == "":
            continue
        digest = str(row["source_sha256"])
        value = float(row["prediction_seconds"])
        old = predictions[str(row["method_id"])].setdefault(digest, value)
        if not math.isclose(old, value, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"member aliases have inconsistent hash prediction for {row['method_id']}:{digest}")
    return predictions


def _median_prediction_rows(bundle: InputBundle, seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["panel_member_id"]), str(row["view"]))].append(row)
    output = []
    methods = method_metadata()
    for (member_id, view), rows in sorted(grouped.items()):
        by_seed = {int(row["seed"]): row for row in rows}
        member = next(row for row in bundle.members if row["panel_member_id"] == member_id)
        method_id = f"azizov_gnn_{view}_median_three_seeds"
        good = [by_seed[seed] for seed in NEURAL_SEEDS if seed in by_seed and by_seed[seed]["status"] == "predicted"]
        if len(good) == len(NEURAL_SEEDS):
            prediction = float(statistics.median(float(row["prediction_seconds"]) for row in good))
            status, reason = "predicted", ""
        else:
            prediction = ""
            statuses = {row["status"] for row in by_seed.values()}
            status = "unavailable_representation" if statuses == {"unavailable_representation"} else "incomplete_seed_set"
            reason = ";".join(sorted({str(row["terminal_reason"]) for row in by_seed.values() if row["terminal_reason"]}))
        output.append({
            "panel_member_id": member_id,
            "source_sha256": member["source_sha256"],
            "fold": member["fold"],
            "method_id": method_id,
            "method_family": "azizov_gnn_median_three_seeds",
            "view": view,
            "target_seconds": bundle.hashes[member["source_sha256"]]["target_seconds"],
            "member_observed_seconds": member["observed_seconds"],
            "prediction_seconds": prediction,
            "successful_seed_count": len(good),
            "status": status,
            "terminal_reason": reason,
        })
    return output


def _compute_metrics(bundle: InputBundle, ledger: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_by_method = _method_hash_predictions(ledger)
    methods = {}
    for row in ledger:
        methods[str(row["method_id"])] = {
            "method_id": str(row["method_id"]),
            "method_family": str(row["method_family"]),
            "view": str(row["view"]),
            "seed": row["seed"],
        }
    seed_by_view: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for view in VIEW_NAMES:
        for seed in NEURAL_SEEDS:
            method_id = f"azizov_gnn_{view}_seed_{seed}"
            seed_by_view[view][method_id] = prediction_by_method.get(method_id, {})

    metrics = []
    for method_id, method in methods.items():
        prediction = prediction_by_method.get(method_id, {})
        eligible = [digest for digest, state in bundle.representation_status.items() if state == "available"]
        shared = sorted(digest for digest in prediction if digest in bundle.hashes and math.isfinite(bundle.hashes[digest]["target_seconds"]))
        actual = np.asarray([bundle.hashes[digest]["target_seconds"] for digest in shared], dtype=np.float64)
        predicted = np.asarray([prediction[digest] for digest in shared], dtype=np.float64)
        errors = predicted - actual
        absolute = np.abs(errors)
        if len(shared):
            mae = float(np.mean(absolute))
            medae = float(np.median(absolute))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            log_mae = float(np.mean(np.abs(np.log1p(predicted) - np.log1p(actual))))
            sst = float(np.sum((actual - np.mean(actual)) ** 2))
            r2 = float(1.0 - np.sum(errors ** 2) / sst) if len(shared) > 1 and sst > 0 else ""
            p90, p99, max_error = (float(value) for value in np.quantile(absolute, [0.90, 0.99, 1.0], method="linear"))
        else:
            mae = medae = rmse = log_mae = p90 = p99 = max_error = r2 = ""
        seed_dispersion_values = []
        if str(method["method_family"]).startswith("azizov_gnn"):
            all_seed_predictions = [seed_by_view[method["view"]][f"azizov_gnn_{method['view']}_seed_{seed}"] for seed in NEURAL_SEEDS]
            common = set.intersection(*(set(item) for item in all_seed_predictions)) if all_seed_predictions else set()
            seed_dispersion_values = [float(np.std([item[digest] for item in all_seed_predictions], ddof=0)) for digest in common]
        member_rows_by_id = {row["panel_member_id"]: row for row in ledger if row["method_id"] == method_id}
        member_actual = []
        member_predicted = []
        for member in bundle.members:
            row = member_rows_by_id.get(member["panel_member_id"])
            if row and row["status"] == "predicted":
                member_actual.append(float(member["observed_seconds"]))
                member_predicted.append(float(row["prediction_seconds"]))
        member_errors = np.asarray(member_predicted, dtype=np.float64) - np.asarray(member_actual, dtype=np.float64)
        metrics.append({
            "method_id": method_id,
            "method_family": method["method_family"],
            "view": method["view"],
            "seed": method["seed"],
            "assigned_hashes": HASH_COUNT,
            "representation_eligible_hashes": len(eligible),
            "predicted_hashes": len(shared),
            "hash_prediction_coverage": len(shared) / HASH_COUNT,
            "mae_seconds": mae,
            "medae_seconds": medae,
            "rmse_seconds": rmse,
            "log1p_mae": log_mae,
            "r2_seconds": r2,
            "p90_absolute_error_seconds": p90,
            "p99_absolute_error_seconds": p99,
            "max_absolute_error_seconds": max_error,
            "mean_per_hash_seed_std_seconds": float(np.mean(seed_dispersion_values)) if seed_dispersion_values else "",
            "member_prediction_count": len(member_actual),
            "member_mae_seconds": float(np.mean(np.abs(member_errors))) if len(member_errors) else "",
            "member_rmse_seconds": float(np.sqrt(np.mean(member_errors ** 2))) if len(member_errors) else "",
            "quantile_method": "numpy.quantile(method='linear')",
        })

    pair_rows = []
    method_ids = sorted(methods)
    for method_a, method_b in itertools.combinations(method_ids, 2):
        common = sorted(set(prediction_by_method.get(method_a, {})) & set(prediction_by_method.get(method_b, {})))
        comparison_id = f"{method_a}__vs__{method_b}"
        for digest in common:
            target = bundle.hashes[digest]["target_seconds"]
            pred_a = prediction_by_method[method_a][digest]
            pred_b = prediction_by_method[method_b][digest]
            error_a = abs(pred_a - target)
            error_b = abs(pred_b - target)
            pair_rows.append({
                "comparison_id": comparison_id,
                "method_a": method_a,
                "method_b": method_b,
                "source_sha256": digest,
                "target_seconds": target,
                "prediction_a_seconds": pred_a,
                "prediction_b_seconds": pred_b,
                "absolute_error_a_seconds": error_a,
                "absolute_error_b_seconds": error_b,
                "absolute_error_difference_a_minus_b_seconds": error_a - error_b,
            })
    return metrics, pair_rows


def _prepare_environment_lock(
    output_dir: Path,
    input_hashes: Mapping[str, str],
    *,
    expected_runtime_lock: Path | None = None,
    overlay_path: Path | None = None,
    cuda_base_lock_path: Path = DEFAULT_CUDA_BASE_LOCK,
    uv_executable: str = "uv",
) -> tuple[dict[str, Any], dict[str, Any]]:
    activate_overlay(overlay_path)
    packages = verify_fit_packages()
    actual_versions = runtime_version_snapshot()
    base_lock = load_cuda_base_lock(cuda_base_lock_path)
    base_version_mismatches = validate_cuda_base_versions(base_lock, actual_versions)
    if base_version_mismatches:
        raise RuntimeError("fit runtime differs from the frozen CUDA/PyG base: " + "; ".join(base_version_mismatches))
    expected_lock_document = None
    expected_lock_sha256 = None
    if expected_runtime_lock is not None:
        expected_lock_document = json.loads(expected_runtime_lock.read_text(encoding="utf-8"))
        expected_lock_sha256 = sha256_file(expected_runtime_lock)
    overlay_sha256 = sha256_tree(overlay_path) if overlay_path is not None else None
    freeze = capture_pip_freeze(output_dir, overlay_path=overlay_path, uv_executable=uv_executable)
    if expected_lock_document is not None:
        expected_lock_mismatches = validate_runtime_lock_document(
            expected_lock_document,
            actual_versions=actual_versions,
            overlay_sha256=overlay_sha256,
            overlay_freeze_sha256=freeze["overlay_pip_freeze_sha256"],
            base_cuda_lock_sha256=base_lock["sha256"],
            environment_freeze_sha256=freeze["pip_freeze_sha256"],
        )
        if expected_lock_mismatches:
            raise RuntimeError("fit runtime differs from the supplied exact environment lock: " + "; ".join(expected_lock_mismatches))
    environment = {
        "status": "fitting_environment_frozen_before_fold0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "versions": actual_versions,
        "platform": platform.platform(),
        "fitting_packages": packages,
        "requirements_path": str(FIT_REQUIREMENTS_PATH.relative_to(ROOT)),
        "requirements_sha256": sha256_file(FIT_REQUIREMENTS_PATH),
        "input_sha256": dict(input_hashes),
        "runtime_lock_path": str(expected_runtime_lock) if expected_runtime_lock else None,
        "runtime_lock_sha256": expected_lock_sha256,
        "overlay_path": str(overlay_path) if overlay_path else None,
        "overlay_sha256": overlay_sha256,
        "cuda_base_lock_path": base_lock["path"],
        "cuda_base_lock_sha256": base_lock["sha256"],
        "cuda_base_lock_versions": base_lock["expected_versions"],
        "cuda_base_lock_pip_freeze_sha256": base_lock["base_pip_freeze_sha256"],
        "uv_executable": shutil.which(uv_executable),
        **freeze,
        "cuda": None,
    }
    write_json(output_dir / "environment_and_cuda_lock.json", environment)
    return environment, base_lock


def _ensure_no_fit_outputs(output_dir: Path) -> None:
    outputs = (
        "oof_predictions_by_seed.csv", "oof_predictions_median.csv", "classical_oof_predictions.csv",
        "method_metrics.csv", "same_hash_shared_row_pairs.csv", "coverage_and_terminal_statuses.csv",
        "fold_0_qa.json", "environment_and_cuda_lock.json", "environment_pip_freeze.txt",
        "overlay_pip_freeze.txt", "run_manifest.json",
    )
    present = [name for name in outputs if (output_dir / name).exists()]
    if present:
        raise FileExistsError("refusing to overwrite existing E4 outputs; use a fresh artifact copy: " + ", ".join(present))


def fit_five_fold_oof(
    bundle: InputBundle,
    *,
    expected_runtime_lock: Path | None = None,
    overlay_path: Path | None = None,
    cuda_base_lock_path: Path = DEFAULT_CUDA_BASE_LOCK,
    uv_executable: str = "uv",
) -> int:
    output_dir = bundle.artifact_dir
    _ensure_no_fit_outputs(output_dir)
    environment, base_cuda_lock = _prepare_environment_lock(
        output_dir, bundle.input_hashes,
        expected_runtime_lock=expected_runtime_lock,
        overlay_path=overlay_path,
        cuda_base_lock_path=cuda_base_lock_path,
        uv_executable=uv_executable,
    )
    all_rows: list[dict[str, Any]] = []
    fold_audits: dict[str, Any] = {}
    cuda_smoke = None
    stopped_at_fold0 = False
    stop_reason = ""
    for fold in FROZEN_FOLDS:
        fold_rows, qa, runtime = _run_fold(bundle, fold, smoke=cuda_smoke, cuda_base_lock=base_cuda_lock)
        all_rows.extend(fold_rows)
        fold_audits[str(fold)] = qa
        if runtime.get("cuda_environment") is not None:
            cuda_smoke = runtime["cuda_environment"]
            environment["cuda"] = cuda_smoke
            write_json(output_dir / "environment_and_cuda_lock.json", environment)
        if fold == 0:
            passed, reasons = fold0_technical_gate(bundle, fold_rows, qa)
            qa["technical_status"] = "PASS" if passed else "FAIL"
            qa["continuation_decision"] = "automatically_continue_folds_1_through_4" if passed else "stop_before_fold1"
            qa["continuation_reasons"] = reasons
            write_json(output_dir / "fold_0_qa.json", qa)
            if not passed:
                stopped_at_fold0 = True
                stop_reason = "; ".join(reasons)
                _append_not_run_rows(bundle, all_rows, from_fold=1, reason=stop_reason)
                break
    if stopped_at_fold0:
        overall_status = "stopped_fold0_technical_failure"
    else:
        overall_status = "complete_five_fold_oof"

    ledger = build_terminal_ledger(
        bundle, all_rows,
        stop_after_fold=0 if stopped_at_fold0 else None,
        stop_reason=stop_reason,
    )
    neural_rows = [row for row in ledger if row["method_family"] == "azizov_gnn"]
    classical_rows = [row for row in ledger if row["method_family"].startswith("classical_")]
    median_rows = _median_prediction_rows(bundle, neural_rows)
    median_status_rows = [{**row, "seed": "median"} for row in median_rows]
    complete_status_rows = ledger + median_status_rows
    metrics, pairs = _compute_metrics(bundle, complete_status_rows)
    write_csv(output_dir / "oof_predictions_by_seed.csv", PREDICTION_FIELDS, neural_rows)
    write_csv(output_dir / "classical_oof_predictions.csv", PREDICTION_FIELDS, classical_rows)
    write_csv(output_dir / "oof_predictions_median.csv", MEDIAN_FIELDS, median_rows)
    write_csv(output_dir / "coverage_and_terminal_statuses.csv", STATUS_FIELDS, complete_status_rows)
    write_csv(output_dir / "method_metrics.csv", list(metrics[0]) if metrics else [], metrics)
    pair_fields = list(pairs[0]) if pairs else [
        "comparison_id", "method_a", "method_b", "source_sha256", "target_seconds",
        "prediction_a_seconds", "prediction_b_seconds", "absolute_error_a_seconds",
        "absolute_error_b_seconds", "absolute_error_difference_a_minus_b_seconds",
    ]
    write_csv(output_dir / "same_hash_shared_row_pairs.csv", pair_fields, pairs)
    run_manifest = {
        "artifact_id": "azizov-common-core-gnn-v1-e4-fit",
        "protocol_id": bundle.protocol["protocol_id"],
        "status": overall_status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": bundle.input_hashes,
        "source_hashes_sha256": bundle.source_hashes_sha256,
        "representation_manifest_sha256": bundle.representation_manifest_sha256,
        "runner_sha256": bundle.input_hashes["runner"],
        "fit_requirements_sha256": bundle.input_hashes["fit_requirements"],
        "method_ids": list(METHOD_IDS),
        "method_family_count": len(METHOD_IDS),
        "derived_three_view_seed_medians": 3,
        "member_method_terminal_rows": len(complete_status_rows),
        "folds": list(FROZEN_FOLDS),
        "neural_seeds": list(NEURAL_SEEDS),
        "classical_seed": CLASSICAL_SEED,
        "fold0_qa": fold_audits.get("0"),
        "fold_audits": fold_audits,
        "continuation_depends_on_metric_quality": False,
        "continuation_reason": "fold0 technical QA only",
        "cpu_baselines_run_outside_cuda_exclusion_locks": True,
        "cuda_smoke_and_gnn_fit_require_both_shared_locks": True,
        "no_simulator_timing_or_transpilation": True,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    return 1 if stopped_at_fold0 else 0


def validate_inputs_action(artifact_dir: Path) -> dict[str, Any]:
    bundle = load_inputs(artifact_dir)
    folds = Counter(record["fold"] for record in bundle.hashes.values())
    status_counts = Counter(bundle.representation_status.values())
    return {
        "status": "PASS_INPUT_CONTRACT",
        "protocol_id": bundle.protocol["protocol_id"],
        "members": len(bundle.members),
        "exact_hashes": len(bundle.hashes),
        "hashes_by_fold": {str(key): folds[key] for key in FROZEN_FOLDS},
        "representation_status_counts": dict(status_counts),
        "result_families": len(METHOD_IDS),
        "neural_families": len(VIEW_NAMES) * len(NEURAL_SEEDS),
        "classical_families": len(CLASSICAL_NAMES) * len(VIEW_NAMES),
        "input_sha256": bundle.input_hashes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("validate-inputs", "fit-five-fold-oof"), required=True)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--runtime-lock", type=Path,
        help=("optional exact JSON lock; keys: schema_version, expected_versions, overlay_sha256, "
              "overlay_pip_freeze_sha256, base_cuda_lock_sha256, and environment_pip_freeze_sha256"),
    )
    parser.add_argument("--overlay-path", type=Path, help="isolated fitting package overlay to hash and record")
    parser.add_argument("--cuda-base-lock", type=Path, default=DEFAULT_CUDA_BASE_LOCK)
    parser.add_argument("--uv-executable", default="uv")
    args = parser.parse_args(argv)
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else ROOT / args.artifact_dir
    try:
        if args.action == "validate-inputs":
            result = validate_inputs_action(artifact_dir)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        bundle = load_inputs(artifact_dir)
        runtime_lock = args.runtime_lock.resolve() if args.runtime_lock else None
        overlay_path = args.overlay_path.resolve() if args.overlay_path else None
        cuda_base_lock = args.cuda_base_lock.resolve()
        activate_overlay(overlay_path)
        return fit_five_fold_oof(
            bundle,
            expected_runtime_lock=runtime_lock,
            overlay_path=overlay_path,
            cuda_base_lock_path=cuda_base_lock,
            uv_executable=args.uv_executable,
        )
    except Exception as exc:
        print(f"Azizov E4 {args.action} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
