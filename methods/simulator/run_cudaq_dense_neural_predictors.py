#!/usr/bin/env python3
"""Run the two frozen CUDA-Q dense neural predictors for one precision/fold.

This is an architecture-adaptation experiment, not source-native Ma--Li.  It
uses repaired R3 normalized IR, R5 warm-target reductions, and the frozen C44
exact-QASM fold assignments.  The CLI is deliberately CUDA-only; importable
feature/preflight helpers remain CPU-testable on synthetic data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "benchmark_v1/protocol/cudaq_dense_statevector_predictor_comparison_v1.json"
PANEL = ROOT / "artifacts/benchmark_v1/sim_common_q16_manifest_20260927/sim_common_q16_manifest.csv"
R3_DIR = ROOT / "work/benchmark_recovery/r3_dense/ir_features_v1"
R5_DIR = ROOT / "work/benchmark_recovery/r5_dense/targets_recovered_v2"
C44_FOLDS = ROOT / "artifacts/benchmark_v1/c44_aer_q16_full_panel_evaluation_20260927/aer_q16_reduced_warm.csv"
DEFAULT_ATTEMPT_ROOT = ROOT / "work/benchmark_recovery/r8_dense_predictors/attempt_001"

PINNED_SHA256 = {
    "protocol": "7598f38996c9e872245be503ac21f237d4aae0c95182127fa3ef3f3b03520258",
    "panel": "9b33added9a2d94f9f621021cec16362561fe58454e233c495e25568a666fb8e",
    "r3_manifest": "d9a35d022df246c381c098a11ff405d74b5cbbd416ac9b62c126f48616e3bad2",
    "r3_features": "cb8d92a255c218fd804a4a960dda78979699ac94ff1653b39c24fd6dc6f66def",
    "r3_ir": "68c7cd713d854814e258701b973c43e8802c24a09b7cf6908ec3c324e6e5609c",
    "r5_manifest": "bfc22149ecc05a85a23598f20fb46cff0e0eb17479a4a5b7f8d35ab8707ca200",
    "r5_targets": "1b88fef5015d5403f89956d8f6d6f92baced9aaecc149728e641cee10588c170",
    "r5_coverage": "93bef4133cc8043b7aa59ca540861d61370b2fa49501e5b40e4be94fb757c901",
    "c44_folds": "a25e2e740fd3058e487f713cd0fc074dd8761120a5aface275a212693ac05437",
}
SEEDS = (42, 1234, 31415)
EPOCHS = 500
BATCH_SIZE = 32
GLOBAL_NAMES = (
    "active_width", "structural_depth", "one_qubit_count", "two_qubit_count",
    "swap_like_count", "measurement_count", "shots",
)
METHODS = ("mali_style_dense_graph", "matched_metadata_mlp")
ATTEMPT_STATUSES = {"ok", "fit_failed", "prediction_failed", "not_run_blocked"}
BASE_OUTPUT_FIELDS = (
    "context_id", "precision", "method_id", "fold", "qasm_sha256",
    "assigned_core_oof", "target_status", "target_seconds", "prediction_seconds",
    "status", "terminal_reason", "train_hash_count", "test_hash_count",
    "train_hashes_sha256", "test_hashes_sha256", "model_params_json",
    "transform_contract", "seed", "prediction_aggregation", "attempt_terminal",
)
AGGREGATE_EXTRA_FIELDS = (
    "seed_prediction_min_seconds", "seed_prediction_max_seconds", "seed_prediction_range_seconds",
)
EXPECTED_FOLD_COUNTS = {0: 37, 1: 28, 2: 26, 3: 25, 4: 34}
EXPECTED_CONTEXT_SHA256 = "daac3bf1dadfac952291c02ad50d0d3fb96520a242f6d80ebe260f09e68cba71"
EXCLUDED_MULTI_ARITY = {"measure", "barrier", "snapshot"}


class PredictionFailure(RuntimeError):
    """A failure in held-out prediction or validation of saved predictions."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(raw)


def hash_list_sha256(values: Iterable[str]) -> str:
    return sha256_bytes(("\n".join(sorted(values)) + "\n").encode("ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def verify_pins() -> dict[str, str]:
    paths = {
        "protocol": PROTOCOL,
        "panel": PANEL,
        "r3_manifest": R3_DIR / "manifest.json",
        "r3_features": R3_DIR / "dense_features_by_hash.jsonl",
        "r3_ir": R3_DIR / "dense_ir_by_hash.csv",
        "r5_manifest": R5_DIR / "manifest.json",
        "r5_targets": R5_DIR / "dense_hash_targets.csv",
        "r5_coverage": R5_DIR / "dense_member_coverage.csv",
        "c44_folds": C44_FOLDS,
    }
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"pinned input missing: {name}={path}")
        observed[name] = sha256_file(path)
        if observed[name] != PINNED_SHA256[name]:
            raise ValueError(f"immutable input hash mismatch for {name}: {observed[name]}")
    return observed


def verify_neural_software_versions() -> dict[str, str]:
    """Check method-relevant pinned software without probing/initializing CUDA."""
    import torch_geometric

    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_runtime_build": str(torch.version.cuda),
    }
    expected = {
        "python": ("3.10",),
        "numpy": ("1.26.4",),
        "torch": ("2.7.1+cu128",),
        "torch_geometric": ("2.6.1",),
        "cuda_runtime_build": ("12.8",),
    }
    if not observed["python"].startswith(expected["python"][0] + "."):
        raise ValueError(f"Python version differs from frozen neural environment: {observed['python']}")
    for name in ("numpy", "torch", "torch_geometric", "cuda_runtime_build"):
        if observed[name] not in expected[name]:
            raise ValueError(f"{name} version differs from frozen neural environment: {observed[name]}")
    return observed


def verify_neural_protocol(protocol: Mapping[str, Any]) -> None:
    expected_architectures = {
        "mali_style_dense_graph": "Use the existing UnifiedGraphModel: three TransformerConv(64) layers with ReLU, global mean pool, seven-to-64-to-64 global branch, concatenate, 128-to-512-to-512-to-128-to-1 ReLU head.",
        "matched_metadata_mlp": "Seven-to-64-to-64-to-512-to-512-to-128-to-1 fully connected ReLU MLP, as in the frozen R7 matched baseline.",
    }
    methods = protocol.get("methods", {})
    for method_id, expected_architecture in expected_architectures.items():
        method = methods.get(method_id)
        if not isinstance(method, dict):
            raise ValueError(f"frozen protocol lacks required neural method: {method_id}")
        expected = {
            "architecture": expected_architecture,
            "optimizer": "Adam lr=0.0005, weight_decay=0.0001",
            "loss": "unweighted MSE on log1p(seconds)",
            "precision": "float32 without AMP",
            "epochs": EPOCHS,
            "early_stopping": False,
            "effective_batch_size": BATCH_SIZE,
            "seeds": list(SEEDS),
            "device": "CUDA required; no CPU fallback",
        }
        mismatches = {key: (method.get(key), value) for key, value in expected.items() if method.get(key) != value}
        if mismatches:
            raise ValueError(f"runner/model contract differs from frozen protocol for {method_id}: {mismatches}")


def load_feature_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qhash = str(row["qasm_sha256"])
            if qhash in records:
                raise ValueError(f"duplicate R3 feature hash at line {line_number}: {qhash}")
            if row.get("format") != "cudaq-dense-normalized-ir-features-v1":
                raise ValueError(f"unexpected R3 feature format at line {line_number}")
            dag = row.get("dag", {})
            if dag.get("format") != "s71-operation-dag-v1":
                raise ValueError(f"unexpected DAG format at line {line_number}")
            records[qhash] = row
    return records


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"true", "1", "yes"}


def load_frozen_fold_map(path: Path) -> dict[str, int]:
    by_hash: dict[str, int] = {}
    for row in read_csv(path):
        if row.get("stratum") != "core_q2_q9":
            continue
        qhash = row["source_sha256"]
        fold = int(row["fold"])
        previous = by_hash.setdefault(qhash, fold)
        if previous != fold:
            raise ValueError(f"C44 assigns one exact-QASM hash to multiple folds: {qhash}")
    counts = {fold: sum(value == fold for value in by_hash.values()) for fold in range(5)}
    if counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"C44 exact-hash fold counts mismatch: {counts}")
    if len(by_hash) != 150:
        raise ValueError(f"C44 core must contain 150 unique hashes, got {len(by_hash)}")
    return by_hash


def count_work_scaled_multiqubit_nodes(features: Mapping[str, Mapping[str, Any]]) -> int:
    count = 0
    for row in features.values():
        dag = row["dag"]
        for name, arity in zip(dag["names"], dag["arity"]):
            if int(arity) >= 3 and str(name).lower() not in EXCLUDED_MULTI_ARITY:
                count += 1
    return count


def prepare_fold_data(precision: str) -> dict[str, Any]:
    """Read and validate the exact pinned core identities without fitting."""
    if precision not in {"fp32", "fp64"}:
        raise ValueError(f"invalid precision context: {precision}")
    input_hashes = verify_pins()
    protocol = read_json(PROTOCOL)
    verify_neural_protocol(protocol)
    software_versions = verify_neural_software_versions()
    if protocol["population"]["measurement_context_sha256"] != EXPECTED_CONTEXT_SHA256:
        raise ValueError("protocol measurement context differs from the frozen R5 context")
    r3_manifest = read_json(R3_DIR / "manifest.json")
    r5_manifest = read_json(R5_DIR / "manifest.json")
    if r3_manifest.get("status") != "PASS" or r5_manifest.get("status") != "PASS":
        raise ValueError("R3/R5 source manifests are not PASS")
    if (r3_manifest.get("panel_members"), r3_manifest.get("unique_qasm_hashes"),
            r3_manifest.get("core_members"), r3_manifest.get("core_unique_hashes")) != (204, 191, 162, 150):
        raise ValueError("R3 manifest population counts differ from frozen dense panel")
    if (r5_manifest.get("panel_members"), r5_manifest.get("unique_hashes"),
            r5_manifest.get("target_rows"), r5_manifest.get("core_unique_hashes")) != (204, 191, 382, 150):
        raise ValueError("R5 manifest population counts differ from frozen dense targets")
    for name, expected in (("dense_features_by_hash.jsonl", PINNED_SHA256["r3_features"]),
                           ("dense_ir_by_hash.csv", PINNED_SHA256["r3_ir"])):
        if r3_manifest.get("outputs", {}).get(name) != expected:
            raise ValueError(f"R3 manifest does not pin {name}")
    for name, expected in (("dense_hash_targets.csv", PINNED_SHA256["r5_targets"]),
                           ("dense_member_coverage.csv", PINNED_SHA256["r5_coverage"])):
        if r5_manifest.get("outputs", {}).get(name) != expected:
            raise ValueError(f"R5 manifest does not pin {name}")

    features = load_feature_records(R3_DIR / "dense_features_by_hash.jsonl")
    ir_rows = {row["qasm_sha256"]: row for row in read_csv(R3_DIR / "dense_ir_by_hash.csv")}
    if set(features) != set(ir_rows) or len(features) != 191:
        raise ValueError("R3 feature/IR exact-hash populations disagree")
    for qhash, feature in features.items():
        ir = ir_rows[qhash]
        if feature.get("normalized_ir_sha256") != ir.get("normalized_ir_sha256"):
            raise ValueError(f"R3 normalized IR digest mismatch for {qhash}")
        if feature.get("unitary_sha256") != ir.get("unitary_sha256"):
            raise ValueError(f"R3 unitary digest mismatch for {qhash}")
    multi_nodes = count_work_scaled_multiqubit_nodes(features)
    if multi_nodes != 124:
        raise ValueError(f"R3 work-scaled multi-qubit node count mismatch: {multi_nodes}")

    fold_map = load_frozen_fold_map(C44_FOLDS)
    targets = [row for row in read_csv(R5_DIR / "dense_hash_targets.csv") if row["precision"] == precision]
    by_hash: dict[str, dict[str, Any]] = {}
    for row in targets:
        if row["stratum"] != "core_q2_q9" or not _parse_bool(row["assigned_core_oof"]):
            continue
        qhash = row["qasm_sha256"]
        if qhash in by_hash:
            raise ValueError(f"R5 duplicate exact hash for precision {precision}: {qhash}")
        if row["target_status"] != "ok":
            raise ValueError(f"R5 core target is not successful: {qhash} status={row['target_status']}")
        target = float(row["target_seconds"])
        if not math.isfinite(target) or target <= 0:
            raise ValueError(f"R5 target is not finite positive seconds: {qhash}")
        if row["context_sha256"] != EXPECTED_CONTEXT_SHA256:
            raise ValueError(f"R5 target context mismatch: {qhash}")
        if qhash not in features:
            raise ValueError(f"R5 target missing exact R3 feature: {qhash}")
        feature = features[qhash]
        ir = ir_rows[qhash]
        if row["normalized_ir_sha256"] != ir["normalized_ir_sha256"] or row["unitary_sha256"] != ir["unitary_sha256"]:
            raise ValueError(f"R5-to-R3 representation identity mismatch: {qhash}")
        if row["normalized_ir_sha256"] != feature["normalized_ir_sha256"] or row["unitary_sha256"] != feature["unitary_sha256"]:
            raise ValueError(f"R5-to-feature representation identity mismatch: {qhash}")
        if int(row["outer_fold"]) != fold_map.get(qhash, -1):
            raise ValueError(f"R5 fold is not the frozen C44 exact-hash assignment: {qhash}")
        by_hash[qhash] = {
            "qasm_sha256": qhash,
            "fold": int(row["outer_fold"]),
            "target_seconds": target,
            "target_status": row["target_status"],
            "assigned_core_oof": True,
            "feature": feature,
        }
    if set(by_hash) != set(fold_map):
        raise ValueError(f"R5/C44 core exact-hash mismatch: missing={len(set(fold_map)-set(by_hash))}, extra={len(set(by_hash)-set(fold_map))}")
    if len(by_hash) != 150:
        raise ValueError(f"R5 core expected 150 hashes for {precision}, got {len(by_hash)}")
    actual_counts = {fold: sum(row["fold"] == fold for row in by_hash.values()) for fold in range(5)}
    if actual_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"R5/C44 fold counts differ for {precision}: {actual_counts}")
    all_hashes = set(by_hash)
    for fold, expected_test_count in EXPECTED_FOLD_COUNTS.items():
        test_hashes = {qhash for qhash, row in by_hash.items() if row["fold"] == fold}
        train_hashes = all_hashes - test_hashes
        if len(test_hashes) != expected_test_count or train_hashes & test_hashes or train_hashes | test_hashes != all_hashes:
            raise ValueError(f"invalid exact-hash train/test partition for {precision} fold {fold}")

    return {
        "precision": precision,
        "context_id": f"cudaq_dense_statevector_{precision}",
        "protocol": protocol,
        "input_hashes": input_hashes,
        "software_versions": software_versions,
        "feature_population_count": len(features),
        "multiqubit_node_count": multi_nodes,
        "fold_map": fold_map,
        "rows": by_hash,
        "expected_fold_counts": actual_counts,
    }


def raw_global_values(feature: Mapping[str, Any]) -> np.ndarray:
    values = feature["global_features"]
    result = np.asarray([float(values[name]) for name in GLOBAL_NAMES], dtype=np.float64)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("global features must be finite and nonnegative")
    return np.log1p(result)


def fit_global_transform(train_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    if not train_rows:
        raise ValueError("cannot fit global transform on empty outer-training data")
    train = np.stack([raw_global_values(row["feature"]) for row in train_rows])
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-6] = 1.0
    return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}


def fit_gate_vocabulary(train_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    names: set[str] = set()
    for row in train_rows:
        names.update(str(name) for name in row["feature"]["dag"]["names"])
    names.discard("__UNK_GATE__")
    return {name: index for index, name in enumerate(sorted(names))} | {"__UNK_GATE__": len(names)}


def normalized_globals(feature: Mapping[str, Any], transform: Mapping[str, Sequence[float]]) -> np.ndarray:
    values = raw_global_values(feature)
    mean = np.asarray(transform["mean"], dtype=np.float64)
    std = np.asarray(transform["std"], dtype=np.float64)
    if mean.shape != (7,) or std.shape != (7,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("invalid train-only seven-global transform")
    result = (values - mean) / std
    if not np.isfinite(result).all():
        raise ValueError("non-finite transformed global features")
    return result.astype(np.float32)


def encode_dag(dag: Mapping[str, Any], vocab: Mapping[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    names = list(dag["names"])
    arities = list(dag["arity"])
    n = len(names)
    if not (len(arities) == len(dag["parameter"]) == len(dag["q0"]) == n):
        raise ValueError("malformed normalized DAG node arrays")
    gate_width = len(vocab)
    node_width = gate_width + 4 + 1 + 4
    x = torch.zeros((n, node_width), dtype=torch.float32)
    denominator = max(1, int(dag["num_qubits"]) - 1)
    for index, name in enumerate(names):
        x[index, vocab.get(str(name), vocab["__UNK_GATE__"])] = 1.0
        arity = min(max(int(arities[index]), 0), 3)
        x[index, gate_width + arity] = 1.0
        x[index, gate_width + 4] = float(bool(dag["parameter"][index]))
        positions = [int(dag[key][index]) for key in ("q0", "q1", "q2", "q3") if int(dag[key][index]) >= 0]
        if positions:
            normalized = np.asarray(positions, dtype=np.float32) / float(denominator)
            x[index, gate_width + 5:gate_width + 8] = torch.tensor(
                [float(normalized.min()), float(normalized.mean()), float(normalized.max())], dtype=torch.float32
            )
            x[index, gate_width + 8] = 1.0
    src = torch.as_tensor(dag["edge_src"], dtype=torch.long)
    dst = torch.as_tensor(dag["edge_dst"], dtype=torch.long)
    if src.numel() != dst.numel() or (src.numel() and (int(src.max()) >= n or int(dst.max()) >= n)):
        raise ValueError("malformed normalized DAG edges")
    edge_index = torch.stack([src, dst], dim=0) if src.numel() else torch.empty((2, 0), dtype=torch.long)
    return x, edge_index


def record_to_graph(row: Mapping[str, Any], vocab: Mapping[str, int], transform: Mapping[str, Sequence[float]],
                    *, include_target: bool = True) -> Data:
    feature = row["feature"]
    x, edge_index = encode_dag(feature["dag"], vocab)
    data = Data(x=x, edge_index=edge_index)
    data.global_features = torch.as_tensor(normalized_globals(feature, transform), dtype=torch.float32).reshape(1, 7)
    if include_target:
        data.target_log_seconds = torch.tensor([math.log1p(float(row["target_seconds"]))], dtype=torch.float32)
    return data


class DenseGraphModel(nn.Module):
    """Frozen S71-style graph architecture adapted to local dense IR."""

    def __init__(self, node_width: int):
        super().__init__()
        self.conv0 = TransformerConv(node_width, 64)
        self.conv1 = TransformerConv(64, 64)
        self.conv2 = TransformerConv(64, 64)
        self.gf1 = nn.Linear(7, 64)
        self.gf2 = nn.Linear(64, 64)
        self.head1 = nn.Linear(128, 512)
        self.head2 = nn.Linear(512, 512)
        self.head3 = nn.Linear(512, 128)
        self.head4 = nn.Linear(128, 1)

    def forward(self, batch: Data) -> torch.Tensor:
        x = torch.relu(self.conv0(batch.x, batch.edge_index))
        x = torch.relu(self.conv1(x, batch.edge_index))
        x = torch.relu(self.conv2(x, batch.edge_index))
        x = global_mean_pool(x, batch.batch)
        global_features = torch.relu(self.gf1(batch.global_features))
        global_features = torch.relu(self.gf2(global_features))
        x = torch.cat([x, global_features], dim=1)
        x = torch.relu(self.head1(x))
        x = torch.relu(self.head2(x))
        x = torch.relu(self.head3(x))
        return self.head4(x).reshape(-1)


class MatchedMetadataMLP(nn.Module):
    """Seven-global matched-feature baseline from the frozen protocol."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, batch: Data) -> torch.Tensor:
        return self.network(batch.global_features).reshape(-1)


def aggregate_seed_predictions(seed_predictions: Mapping[int, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    if not seed_predictions:
        raise ValueError("cannot aggregate zero seed predictions")
    hash_sets = {tuple(sorted(values)) for values in seed_predictions.values()}
    if len(hash_sets) != 1:
        raise ValueError("seed prediction hash sets differ")
    output: dict[str, dict[str, float]] = {}
    for qhash in next(iter(hash_sets)):
        values = [float(seed_predictions[seed][qhash]) for seed in sorted(seed_predictions)]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"seed predictions must be finite nonnegative seconds: {qhash}")
        output[qhash] = {
            "median": float(statistics.median(values)),
            "minimum": float(min(values)),
            "maximum": float(max(values)),
            "range": float(max(values) - min(values)),
        }
    return output


def fold_rows(rows: Mapping[str, Mapping[str, Any]], fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if fold not in range(5):
        raise ValueError(f"invalid outer fold: {fold}")
    train = sorted((row for row in rows.values() if row["fold"] != fold), key=lambda row: row["qasm_sha256"])
    test = sorted((row for row in rows.values() if row["fold"] == fold), key=lambda row: row["qasm_sha256"])
    train_hashes = {row["qasm_sha256"] for row in train}
    test_hashes = {row["qasm_sha256"] for row in test}
    if not train or not test or train_hashes & test_hashes:
        raise ValueError("empty or leaking frozen outer fold")
    if len(test) != EXPECTED_FOLD_COUNTS[fold]:
        raise ValueError(f"frozen test fold {fold} size mismatch: {len(test)}")
    return train, test


def model_params(method_id: str) -> dict[str, Any]:
    common = {"optimizer": "Adam", "lr": 0.0005, "weight_decay": 0.0001, "loss": "unweighted_mse_log1p_seconds", "epochs": EPOCHS, "batch_size": BATCH_SIZE, "early_stopping": False, "seeds": list(SEEDS), "model_precision": "float32_no_amp"}
    if method_id == "mali_style_dense_graph":
        return {**common, "architecture": "three_TransformerConv_64_relu_global_mean_pool_global_7_to_64_to_64_concat_128_to_512_to_512_to_128_to_1", "representation": "s71-operation-dag-v1 + seven globals", "fidelity_class": "architecture_adaptation"}
    if method_id == "matched_metadata_mlp":
        return {**common, "architecture": "7_to_64_to_64_to_512_to_512_to_128_to_1_fully_connected_relu", "representation": "seven globals only", "fidelity_class": "matched_feature_baseline"}
    raise ValueError(f"unknown method: {method_id}")


def _atomic_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def run_identity(prepared: Mapping[str, Any], fold: int, train_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "protocol_sha256": prepared["input_hashes"]["protocol"],
        "input_hashes": dict(prepared["input_hashes"]),
        "software_versions": dict(prepared["software_versions"]),
        "precision": prepared["precision"],
        "context_id": prepared["context_id"],
        "context_sha256": EXPECTED_CONTEXT_SHA256,
        "fold": fold,
        "seed_registry": list(SEEDS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "train_hashes_sha256": hash_list_sha256(row["qasm_sha256"] for row in train_rows),
        "test_hashes_sha256": hash_list_sha256(row["qasm_sha256"] for row in test_rows),
        "train_hash_count": len(train_rows),
        "test_hash_count": len(test_rows),
        "method_ids": list(METHODS),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }


def _restore_rng(payload: Mapping[str, Any], device: torch.device, loader_generator: torch.Generator) -> None:
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    loader_generator.set_state(payload["loader_generator_state"].cpu())


def _checkpoint_payload(
    *, identity: Mapping[str, Any], method_id: str, seed: int, epoch_completed: int,
    model: nn.Module, optimizer: torch.optim.Optimizer, loader_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "identity": dict(identity),
        "method_id": method_id,
        "seed": seed,
        "epoch_completed": epoch_completed,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
        "loader_generator_state": loader_generator.get_state(),
    }


def validate_resume_identity(payload: Mapping[str, Any], expected: Mapping[str, Any], method_id: str, seed: int) -> None:
    if payload.get("schema_version") != 1 or payload.get("identity") != dict(expected):
        raise ValueError("checkpoint immutable run identity mismatch")
    if payload.get("method_id") != method_id or int(payload.get("seed", -1)) != seed:
        raise ValueError("checkpoint method/seed mismatch")
    epoch = int(payload.get("epoch_completed", -1))
    if epoch < 0 or epoch > EPOCHS or (epoch != EPOCHS and epoch % 10 != 0):
        raise ValueError(f"checkpoint is not at a valid 10-epoch boundary: {epoch}")


def _make_model(method_id: str, node_width: int) -> nn.Module:
    if method_id == "mali_style_dense_graph":
        return DenseGraphModel(node_width)
    if method_id == "matched_metadata_mlp":
        return MatchedMetadataMLP()
    raise ValueError(f"unknown method: {method_id}")


def _set_frozen_thread_caps() -> None:
    """Set both PyTorch CPU thread pools before constructing models/loaders."""
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)


def _seed_torch(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _canary_train_step_and_inference(
    method_id: str, train_data: Data, test_data: Data, node_width: int, device: torch.device, seed: int = 42
) -> dict[str, Any]:
    if hasattr(test_data, "target_log_seconds"):
        raise ValueError("held-out inference Data must not contain target_log_seconds")
    _seed_torch(seed)
    model = _make_model(method_id, node_width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    train_batch = next(iter(DataLoader([train_data], batch_size=1, shuffle=False, num_workers=0))).to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction_log = model(train_batch)
    loss = nn.functional.mse_loss(prediction_log, train_batch.target_log_seconds.reshape(-1))
    if not torch.isfinite(prediction_log).all() or not torch.isfinite(loss):
        raise RuntimeError(f"non-finite canary forward/loss for {method_id}")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError(f"non-finite or missing canary gradients for {method_id}")
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError(f"non-finite model parameter after canary step for {method_id}")

    test_batch = next(iter(DataLoader([test_data], batch_size=1, shuffle=False, num_workers=0))).to(device)
    if hasattr(test_batch, "target_log_seconds"):
        raise ValueError("held-out inference batch unexpectedly contains target_log_seconds")
    model.eval()
    with torch.no_grad():
        test_log = model(test_batch)
        prediction_seconds = torch.expm1(test_log).clamp_min(0)
    if not torch.isfinite(test_log).all() or not torch.isfinite(prediction_seconds).all() or (prediction_seconds < 0).any():
        raise RuntimeError(f"invalid held-out canary inference output for {method_id}")
    return {
        "status": "PASS",
        "seed": seed,
        "optimizer_steps": 1,
        "training_loss_log1p_mse": float(loss.detach().cpu()),
        "train_forward_finite": True,
        "gradient_finite": True,
        "post_step_parameters_finite": True,
        "test_inference_finite_nonnegative": True,
        "test_prediction_seconds": float(prediction_seconds.detach().cpu().reshape(-1)[0]),
        "test_target_attached": False,
    }


def run_cuda_canary(prepared: Mapping[str, Any], fold: int, receipt_dir: Path) -> dict[str, Any]:
    """One-step technical smoke test using real R8 fold representations only.

    This is not a fold fit and never writes to a fold output directory.
    """
    if receipt_dir.exists():
        raise FileExistsError(f"refusing to overwrite canary scratch receipt: {receipt_dir}")
    train_rows, test_rows = fold_rows(prepared["rows"], fold)
    if not torch.cuda.is_available():
        raise RuntimeError("R8 CUDA canary requires CUDA; no CPU fallback")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "RTX 5070 Ti" not in device_name:
        raise RuntimeError(f"frozen R8 GPU is RTX 5070 Ti, detected {device_name!r}")
    _set_frozen_thread_caps()

    transform = fit_global_transform(train_rows)
    vocabulary = fit_gate_vocabulary(train_rows)
    train_row = train_rows[0]
    test_row = test_rows[0]
    graph_train = record_to_graph(train_row, vocabulary, transform, include_target=True)
    graph_test = record_to_graph(test_row, vocabulary, transform, include_target=False)
    if not hasattr(graph_train, "target_log_seconds") or hasattr(graph_test, "target_log_seconds"):
        raise ValueError("canary train/test label firewall failed")
    mlp_train = Data(global_features=graph_train.global_features,
                     target_log_seconds=graph_train.target_log_seconds)
    mlp_test = Data(global_features=graph_test.global_features)
    if hasattr(mlp_test, "target_log_seconds"):
        raise ValueError("canary matched-MLP held-out Data contains the target")

    node_width = int(graph_train.x.shape[1])
    results = {
        "mali_style_dense_graph": _canary_train_step_and_inference(
            "mali_style_dense_graph", graph_train, graph_test, node_width, device
        ),
        "matched_metadata_mlp": _canary_train_step_and_inference(
            "matched_metadata_mlp", mlp_train, mlp_test, node_width, device
        ),
    }
    train_hashes = [row["qasm_sha256"] for row in train_rows]
    test_hashes = [row["qasm_sha256"] for row in test_rows]
    receipt = {
        "artifact_id": "cudaq-dense-neural-technical-canary-v1",
        "status": "PASS" if all(value["status"] == "PASS" for value in results.values()) else "FAIL",
        "mode": "one_step_forward_backward_update_and_held_out_inference_only",
        "precision_context": prepared["precision"],
        "context_id": prepared["context_id"],
        "fold": fold,
        "method_ids": list(METHODS),
        "input_hashes": dict(prepared["input_hashes"]),
        "software_versions": dict(prepared["software_versions"]),
        "train_hashes_sha256": hash_list_sha256(train_hashes),
        "test_hashes_sha256": hash_list_sha256(test_hashes),
        "representative_train_hash": train_row["qasm_sha256"],
        "representative_test_hash": test_row["qasm_sha256"],
        "target_firewall": "test target_seconds is not attached to graph or metadata inference Data",
        "train_transform_sha256": canonical_json_sha256(transform),
        "train_vocabulary_sha256": canonical_json_sha256(dict(sorted(vocabulary.items()))),
        "results": results,
        "device": device_name,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "full_fold_training_started": False,
        "timing_experiment_started": False,
        "receipt_scope": "scratch-only; separate from any fold output",
    }
    _atomic_json(receipt_dir / "canary_receipt.json", receipt)
    return receipt


def _predict(model: nn.Module, test_data: Sequence[Data], device: torch.device) -> list[float]:
    model.eval()
    loader = DataLoader(list(test_data), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    predictions: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            values = torch.expm1(model(batch)).clamp_min(0).detach().cpu().tolist()
            predictions.extend(float(value) for value in values)
    if len(predictions) != len(test_data) or not all(math.isfinite(value) and value >= 0 for value in predictions):
        raise RuntimeError("model returned non-finite, negative, or incomplete test predictions")
    return predictions


def _train_seed(
    *, method_id: str, seed: int, train_data: Sequence[Data], test_data: Sequence[Data],
    node_width: int, device: torch.device, checkpoint_path: Path, seed_output_dir: Path,
    identity: Mapping[str, Any], resume: bool, test_hashes: Sequence[str],
) -> list[float]:
    if device.type != "cuda":
        raise RuntimeError("R8 neural fitting/inference is CUDA-required; CPU fallback is forbidden")
    if seed_output_dir.joinpath("complete.json").is_file():
        complete = read_json(seed_output_dir / "complete.json")
        if complete.get("identity") != dict(identity) or complete.get("method_id") != method_id or complete.get("seed") != seed:
            raise PredictionFailure("completed seed output identity mismatch")
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != complete.get("checkpoint_sha256"):
            raise PredictionFailure("completed seed checkpoint hash mismatch")
        pred_rows = read_csv(seed_output_dir / "predictions.csv")
        if sha256_file(seed_output_dir / "predictions.csv") != complete.get("predictions_sha256"):
            raise PredictionFailure("completed seed prediction hash mismatch")
        if [row["qasm_sha256"] for row in pred_rows] != list(test_hashes):
            raise PredictionFailure("completed seed predictions do not match frozen test-hash order")
        predictions = [float(row["prediction_seconds"]) for row in pred_rows]
        if len(predictions) != len(test_hashes) or not all(math.isfinite(value) and value >= 0 for value in predictions):
            raise PredictionFailure("completed seed predictions contain invalid seconds or coverage")
        return predictions

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    model = _make_model(method_id, node_width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    start_epoch = 0
    if resume and checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        validate_resume_identity(payload, identity, method_id, seed)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch_completed"])
        _restore_rng(payload, device, generator)
    elif resume and not checkpoint_path.exists():
        # No checkpoint means a never-started cell, so deterministic fresh fit.
        pass
    elif not resume and checkpoint_path.exists():
        raise FileExistsError(f"checkpoint exists; pass --resume to continue: {checkpoint_path}")

    loader = DataLoader(list(train_data), batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)
    model.train()
    for epoch in range(start_epoch, EPOCHS):
        losses: list[float] = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction_log = model(batch)
            loss = nn.functional.mse_loss(prediction_log, batch.target_log_seconds.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss method={method_id}, seed={seed}, epoch={epoch + 1}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not losses or not math.isfinite(float(np.mean(losses))):
            raise RuntimeError(f"invalid epoch losses method={method_id}, seed={seed}, epoch={epoch + 1}")
        completed = epoch + 1
        if completed % 10 == 0:
            payload = _checkpoint_payload(
                identity=identity, method_id=method_id, seed=seed, epoch_completed=completed,
                model=model, optimizer=optimizer, loader_generator=generator,
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temp = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
            torch.save(payload, temp)
            os.replace(temp, checkpoint_path)
    try:
        predictions = _predict(model, test_data, device)
    except Exception as exc:
        raise PredictionFailure(f"held-out prediction failed: {type(exc).__name__}: {str(exc).splitlines()[0][:400]}") from exc
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"qasm_sha256": qhash, "prediction_seconds": prediction}
            for qhash, prediction in zip(test_hashes, predictions)]
    _atomic_csv(seed_output_dir / "predictions.csv", ["qasm_sha256", "prediction_seconds"], rows)
    _atomic_json(seed_output_dir / "complete.json", {
        "schema_version": 1, "identity": dict(identity), "method_id": method_id, "seed": seed,
        "predictions_sha256": sha256_file(seed_output_dir / "predictions.csv"),
        "checkpoint_sha256": sha256_file(checkpoint_path), "epoch_completed": EPOCHS,
    })
    return predictions


def _feature_contract(vocab: Mapping[str, int], transform: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    return {
        "graph": "outer-train vocabulary; unseen operation maps to __UNK_GATE__; node features gate one-hot + arity-0..3 + parameter-present + normalized wire min/mean/max + wire-present",
        "globals": list(GLOBAL_NAMES),
        "global_transform": "log1p then mean/std fitted on outer-training hashes only; std<1e-6 replaced with 1.0",
        "global_transform_sha256": canonical_json_sha256(transform),
        "global_transform_values": {"mean": list(transform["mean"]), "std": list(transform["std"])},
        "vocabulary_sha256": canonical_json_sha256(dict(sorted(vocab.items()))),
        "gate_vocabulary": dict(sorted(vocab.items())),
        "unknown_token": "__UNK_GATE__",
        "target": "log1p(target_seconds), inverse=expm1 then clamp at zero",
    }


def _base_output_row(prepared: Mapping[str, Any], fold: int, method: str, row: Mapping[str, Any],
                     prediction: float, train_hash_count: int, test_hash_count: int,
                     train_hash_sha: str, test_hash_sha: str, params_json: str,
                     transform_contract_json: str, seed: int | None, aggregation: str,
                     status: str = "ok", terminal_reason: str = "") -> dict[str, Any]:
    if status not in ATTEMPT_STATUSES:
        raise ValueError(f"invalid neural attempt status: {status}")
    if status == "ok":
        if prediction is None or not math.isfinite(float(prediction)) or float(prediction) < 0:
            raise ValueError("successful attempt must have a finite nonnegative prediction")
    elif prediction is not None:
        raise ValueError("failed/not-run attempt must not carry an imputed prediction")
    contract = json.loads(transform_contract_json)
    if json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False) != transform_contract_json:
        raise ValueError("transform_contract must be canonical JSON")
    return {
        "context_id": prepared["context_id"], "precision": prepared["precision"], "method_id": method,
        "fold": fold, "qasm_sha256": row["qasm_sha256"], "assigned_core_oof": True,
        "target_status": row["target_status"], "target_seconds": row["target_seconds"],
        "prediction_seconds": prediction, "status": status, "terminal_reason": terminal_reason,
        "train_hash_count": train_hash_count, "test_hash_count": test_hash_count,
        "train_hashes_sha256": train_hash_sha, "test_hashes_sha256": test_hash_sha,
        "model_params_json": params_json, "transform_contract": transform_contract_json,
        "seed": "" if seed is None else seed, "prediction_aggregation": aggregation,
        "attempt_terminal": True,
    }


def _write_fold_results(output_dir: Path, prepared: Mapping[str, Any], fold: int,
                        train_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]],
                        seed_rows_by_method: Mapping[str, Mapping[int, Sequence[float]]],
                        seed_errors_by_method: Mapping[str, Mapping[int, Mapping[str, str]]],
                        feature_contracts: Mapping[str, Mapping[str, Any]], identity: Mapping[str, Any]) -> None:
    train_hashes = [row["qasm_sha256"] for row in train_rows]
    test_hashes = [row["qasm_sha256"] for row in test_rows]
    train_hash_sha = hash_list_sha256(train_hashes)
    test_hash_sha = hash_list_sha256(test_hashes)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "seed_predictions.csv"
    aggregate_path = output_dir / "predictions.csv"
    all_seed_rows: list[dict[str, Any]] = []
    all_aggregate_rows: list[dict[str, Any]] = []
    any_failure = False
    for method in METHODS:
        seed_predictions: dict[int, dict[str, float]] = {}
        params_json = json.dumps(model_params(method), sort_keys=True, separators=(",", ":"))
        transform_contract_json = json.dumps(feature_contracts[method], sort_keys=True, separators=(",", ":"), allow_nan=False)
        for seed, predictions in sorted(seed_rows_by_method[method].items()):
            if len(predictions) != len(test_rows):
                raise ValueError(f"prediction count mismatch for {method} seed={seed}")
            by_hash = {row["qasm_sha256"]: float(value) for row, value in zip(test_rows, predictions)}
            seed_predictions[seed] = by_hash
            for row, prediction in zip(test_rows, predictions):
                all_seed_rows.append(_base_output_row(
                    prepared, fold, method, row, float(prediction), len(train_rows), len(test_rows),
                    train_hash_sha, test_hash_sha, params_json, transform_contract_json, seed, "single_seed",
                ))
        for seed in SEEDS:
            if seed in seed_predictions:
                continue
            any_failure = True
            failure = seed_errors_by_method[method].get(seed, {
                "status": "not_run_blocked", "reason": "seed_not_run_after_prior_seed_failure",
            })
            status = failure["status"]
            reason = failure["reason"]
            for row in test_rows:
                all_seed_rows.append(_base_output_row(
                    prepared, fold, method, row, None, len(train_rows), len(test_rows),
                    train_hash_sha, test_hash_sha, params_json, transform_contract_json, seed, "single_seed",
                    status=status,
                    terminal_reason=reason,
                ))
        complete = len(seed_predictions) == len(SEEDS)
        reduced = aggregate_seed_predictions(seed_predictions) if complete else {}
        for row in test_rows:
            if complete:
                pred = reduced[row["qasm_sha256"]]
                result = _base_output_row(
                    prepared, fold, method, row, pred["median"], len(train_rows), len(test_rows),
                    train_hash_sha, test_hash_sha, params_json, transform_contract_json, None, "rowwise_median",
                )
                result.update({
                    "seed_prediction_min_seconds": pred["minimum"],
                    "seed_prediction_max_seconds": pred["maximum"],
                    "seed_prediction_range_seconds": pred["range"],
                })
            else:
                missing_errors = [
                    (seed, seed_errors_by_method[method].get(seed, {
                        "status": "not_run_blocked", "reason": "not run after earlier seed failure",
                    }))
                    for seed in SEEDS if seed not in seed_predictions
                ]
                if any(item[1]["status"] == "fit_failed" for item in missing_errors):
                    aggregate_status = "fit_failed"
                elif any(item[1]["status"] == "prediction_failed" for item in missing_errors):
                    aggregate_status = "prediction_failed"
                else:
                    aggregate_status = "not_run_blocked"
                error_detail = "; ".join(f"seed {seed} {failure['status']}: {failure['reason']}" for seed, failure in missing_errors)
                result = _base_output_row(
                    prepared, fold, method, row, None, len(train_rows), len(test_rows),
                    train_hash_sha, test_hash_sha, params_json, transform_contract_json, None, "rowwise_median",
                    status=aggregate_status, terminal_reason=error_detail,
                )
                result.update({"seed_prediction_min_seconds": "", "seed_prediction_max_seconds": "", "seed_prediction_range_seconds": ""})
            all_aggregate_rows.append(result)
    _atomic_csv(seed_path, BASE_OUTPUT_FIELDS, all_seed_rows)
    _atomic_csv(aggregate_path, BASE_OUTPUT_FIELDS + AGGREGATE_EXTRA_FIELDS, all_aggregate_rows)
    _atomic_json(output_dir / "fold_manifest.json", {
        "artifact_id": "cudaq-dense-neural-predictors-v1-fold",
        "status": "PARTIAL_FAILURE" if any_failure else "PASS",
        "identity": dict(identity),
        "assigned_attempts": len(test_rows) * len(METHODS),
        "seed_level_predictions": len(all_seed_rows),
        "aggregate_predictions": len(all_aggregate_rows),
        "seed_prediction_sha256": sha256_file(seed_path),
        "aggregate_prediction_sha256": sha256_file(aggregate_path),
        "test_hashes_sha256": hash_list_sha256(test_hashes),
        "method_ids": list(METHODS),
        "seed_registry": list(SEEDS),
        "aggregation": "per exact hash and method, median across seeds; retain min/max/range",
        "model_precision": "float32 without AMP",
        "device": "CUDA required; one GPU worker; no CPU fallback",
    })


def run_fold(prepared: Mapping[str, Any], fold: int, output_dir: Path, resume: bool) -> dict[str, Any]:
    train_rows, test_rows = fold_rows(prepared["rows"], fold)
    identity = run_identity(prepared, fold, train_rows, test_rows)
    manifest_path = output_dir / "run_manifest.json"
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite attempt output; pass --resume: {output_dir}")
        if not manifest_path.is_file():
            raise ValueError("--resume requires a run_manifest.json in the existing output directory")
        existing = read_json(manifest_path)
        if existing.get("identity") != identity:
            raise ValueError("existing output manifest does not match current immutable run identity")
    elif resume:
        # The output root does not exist, so this is a new deterministic attempt.
        resume = False

    # Enforce CUDA only after all immutable CPU-side preflight validation.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R8 neural fitting/inference; no CPU fallback")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "RTX 5070 Ti" not in device_name:
        raise RuntimeError(f"frozen R8 GPU is RTX 5070 Ti, detected {device_name!r}")
    _set_frozen_thread_caps()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(manifest_path, {"artifact_id": "cudaq-dense-neural-predictors-v1-attempt", "status": "RUNNING", "identity": identity})

    train_transform = fit_global_transform(train_rows)
    vocabulary = fit_gate_vocabulary(train_rows)
    graph_feature_contract = _feature_contract(vocabulary, train_transform)
    mlp_feature_contract = {
        "globals": list(GLOBAL_NAMES),
        "global_transform": "log1p then mean/std fitted on outer-training hashes only; std<1e-6 replaced with 1.0",
        "global_transform_sha256": canonical_json_sha256(train_transform),
        "global_transform_values": {"mean": list(train_transform["mean"]), "std": list(train_transform["std"])},
        "representation": "seven global features only; no operation vocabulary or graph structure",
        "target": "log1p(target_seconds), inverse=expm1 then clamp at zero",
    }
    graph_train = [record_to_graph(row, vocabulary, train_transform) for row in train_rows]
    graph_test = [record_to_graph(row, vocabulary, train_transform, include_target=False) for row in test_rows]
    # The matched MLP receives only the same seven standardized globals.
    mlp_train = [Data(global_features=row.global_features, target_log_seconds=row.target_log_seconds) for row in graph_train]
    mlp_test = [Data(global_features=row.global_features) for row in graph_test]
    node_width = int(graph_train[0].x.shape[1])
    if any(int(row.x.shape[1]) != node_width for row in graph_train + graph_test):
        raise ValueError("node feature width changed within frozen fold")
    feature_contracts = {"mali_style_dense_graph": graph_feature_contract,
                         "matched_metadata_mlp": mlp_feature_contract}
    fold_inputs = {
        "identity": identity,
        "train_only_vocabulary": vocabulary,
        "global_transform": train_transform,
        "feature_contracts": feature_contracts,
        "feature_contract_sha256": {method: canonical_json_sha256(contract) for method, contract in feature_contracts.items()},
        "node_width": node_width,
        "representation": "R3 normalized operation DAG including terminal measurement nodes + seven globals",
        "target": "R5 target_seconds, warm sample(32) seconds",
    }
    fold_inputs_path = output_dir / "fold_inputs.json"
    if fold_inputs_path.exists():
        if read_json(fold_inputs_path) != fold_inputs:
            raise ValueError("existing fold input transforms/vocabulary differ from deterministic reconstruction")
    else:
        _atomic_json(fold_inputs_path, fold_inputs)

    seed_rows_by_method: dict[str, dict[int, Sequence[float]]] = {method: {} for method in METHODS}
    seed_errors_by_method: dict[str, dict[int, dict[str, str]]] = {method: {} for method in METHODS}
    for method in METHODS:
        method_data = graph_train if method == "mali_style_dense_graph" else mlp_train
        test_data = graph_test if method == "mali_style_dense_graph" else mlp_test
        method_dir = output_dir / method
        for seed_index, seed in enumerate(SEEDS):
            seed_dir = method_dir / f"seed_{seed}"
            checkpoint = method_dir / "checkpoints" / f"seed_{seed}.pt"
            try:
                predictions = _train_seed(
                    method_id=method, seed=seed, train_data=method_data, test_data=test_data,
                    node_width=node_width, device=device, checkpoint_path=checkpoint,
                    seed_output_dir=seed_dir, identity=identity, resume=resume,
                    test_hashes=[row["qasm_sha256"] for row in test_rows],
                )
                seed_rows_by_method[method][seed] = predictions
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:500]}"
                failure_status = "prediction_failed" if isinstance(exc, PredictionFailure) else "fit_failed"
                seed_errors_by_method[method][seed] = {"status": failure_status, "reason": error}
                for later_seed in SEEDS[seed_index + 1:]:
                    seed_errors_by_method[method][later_seed] = {
                        "status": "not_run_blocked", "reason": "seed_not_run_after_prior_seed_failure",
                    }
                break
    _write_fold_results(output_dir, prepared, fold, train_rows, test_rows, seed_rows_by_method,
                        seed_errors_by_method, feature_contracts, identity)
    final = read_json(output_dir / "fold_manifest.json")
    _atomic_json(manifest_path, {
        "artifact_id": "cudaq-dense-neural-predictors-v1-attempt",
        "status": final["status"],
        "identity": identity,
        "fold_manifest_sha256": sha256_file(output_dir / "fold_manifest.json"),
        "aggregate_predictions_sha256": final["aggregate_prediction_sha256"],
        "seed_predictions_sha256": final["seed_prediction_sha256"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": __import__("torch_geometric").__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device_name,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    })
    return read_json(manifest_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("fp32", "fp64"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--canary-output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true", help="CPU-only input/hash/split validation; no output files")
    parser.add_argument("--canary-only", action="store_true", help="one CUDA technical step per neural method; no fold fit")
    parser.add_argument("--resume", action="store_true", help="resume only exact-identity 10-epoch checkpoints")
    args = parser.parse_args(argv)

    if args.preflight_only and args.canary_only:
        parser.error("--preflight-only and --canary-only are mutually exclusive")
    if args.canary_only and (args.output_dir is not None or args.resume):
        parser.error("--canary-only uses its separate scratch receipt path; do not pass --output-dir or --resume")
    if args.canary_output_dir is not None and not args.canary_only:
        parser.error("--canary-output-dir is valid only with --canary-only")

    prepared = prepare_fold_data(args.precision)
    train_rows, test_rows = fold_rows(prepared["rows"], args.fold)
    identity = run_identity(prepared, args.fold, train_rows, test_rows)
    result = {
        "status": "PASS",
        "mode": "preflight_only" if args.preflight_only else "ready_for_cuda_execution",
        "precision": args.precision,
        "fold": args.fold,
        "context_id": prepared["context_id"],
        "train_hash_count": len(train_rows),
        "test_hash_count": len(test_rows),
        "train_test_overlap": 0,
        "fold_counts": prepared["expected_fold_counts"],
        "feature_population_count": prepared["feature_population_count"],
        "work_scaled_multiqubit_node_count": prepared["multiqubit_node_count"],
        "input_hashes": prepared["input_hashes"],
        "identity": identity,
        "cuda_training_started": False,
    }
    if args.preflight_only:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.canary_only:
        receipt_dir = args.canary_output_dir or (
            DEFAULT_ATTEMPT_ROOT / "canary_scratch" / args.precision / f"fold_{args.fold}"
        )
        receipt = run_cuda_canary(prepared, args.fold, receipt_dir)
        print(json.dumps({
            "status": receipt["status"], "mode": receipt["mode"],
            "precision": args.precision, "fold": args.fold,
            "receipt_path": str(receipt_dir / "canary_receipt.json"),
            "receipt_sha256": sha256_file(receipt_dir / "canary_receipt.json"),
            "full_fold_training_started": False, "timing_experiment_started": False,
        }, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 2
    output_dir = args.output_dir or DEFAULT_ATTEMPT_ROOT / args.precision / "neural" / f"fold_{args.fold}"
    run_result = run_fold(prepared, args.fold, output_dir, args.resume)
    result.update({"status": run_result["status"], "output_dir": str(output_dir), "run_manifest_sha256": sha256_file(output_dir / "run_manifest.json"), "cuda_training_started": True})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if run_result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
