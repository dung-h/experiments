#!/usr/bin/env python3
"""Pilot a physics-guided, order-aware DAG residual on Ma--Li graphs.

The full native DAGs are too large to batch as ordinary PyG objects (about
67.4m nodes in the 300-graph corpus).  This pilot preserves topological order
and timing state by aggregating native nodes into at most 256 ordered bins, then
uses a GRU to predict a residual around the QCRE weighted critical path.  It is
an explicit coarsened-DAG experiment, not a claim that a full GNN has been
trained.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_sequence(path: Path, max_bins: int) -> np.ndarray:
    """Return ordered layer/bin features with fixed dimension 21."""
    with np.load(path, allow_pickle=False) as archive:
        opcode = archive["node_opcode"].astype(np.int64)
        duration = archive["node_duration_dt"].astype(np.float64)
        layer = archive["node_topo_layer"].astype(np.int64)
        forward = archive["node_forward_path_dt"].astype(np.float64)
        reverse = archive["node_reverse_path_dt"].astype(np.float64)
        criticality = archive["node_criticality_dt"].astype(np.float64)
        edge_src = archive["edge_src"].astype(np.int64)
        edge_dst = archive["edge_dst"].astype(np.int64)

    unique_layers = np.unique(layer)
    if len(unique_layers) == 0:
        return np.zeros((1, 21), dtype=np.float32)
    bins = min(max_bins, len(unique_layers))
    # Ordered coarse bins preserve the complete circuit order while bounding
    # recurrent sequence length for the long QWalk/native circuits.
    layer_to_bin = np.floor(np.arange(len(unique_layers)) * bins / len(unique_layers)).astype(np.int64)
    node_bins = layer_to_bin[np.searchsorted(unique_layers, layer)]
    counts = np.bincount(node_bins, minlength=bins).astype(np.float64)
    output = np.zeros((bins, 21), dtype=np.float64)
    output[:, 0] = np.log1p(counts)
    for opcode_id in range(14):
        output[:, 1 + opcode_id] = np.log1p(
            np.bincount(node_bins, weights=(opcode == opcode_id).astype(float), minlength=bins)
        )

    def sum_feature(values: np.ndarray) -> np.ndarray:
        return np.log1p(np.bincount(node_bins, weights=np.maximum(values, 0), minlength=bins))

    def max_feature(values: np.ndarray) -> np.ndarray:
        result = np.zeros(bins, dtype=np.float64)
        np.maximum.at(result, node_bins, np.maximum(values, 0))
        return np.log1p(result)

    output[:, 15] = sum_feature(duration)
    output[:, 16] = max_feature(duration)
    output[:, 17] = max_feature(forward)
    output[:, 18] = max_feature(reverse)
    output[:, 19] = max_feature(criticality)
    if len(edge_src):
        edge_bins = node_bins[edge_src]
        output[:, 20] = np.log1p(np.bincount(edge_bins, minlength=bins))
    return output.astype(np.float32)


class LayerResidualGRU(nn.Module):
    def __init__(self, input_dim: int = 21, hidden: int = 32) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(self.input(sequence[None, ...]))
        pooled = torch.cat((hidden[:, -1, :], hidden.mean(dim=1)), dim=-1)
        return self.head(pooled).squeeze()


def group_inner_split(rows: list[dict[str, str]], train_indices: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted({rows[index]["qasm_sha256"] for index in train_indices})
    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(groups)[rng.permutation(len(groups))])
    validation_groups = set(shuffled[: max(1, len(shuffled) // 5)])
    inner_val = np.asarray([index for index in train_indices if rows[index]["qasm_sha256"] in validation_groups], dtype=int)
    inner_train = np.asarray([index for index in train_indices if rows[index]["qasm_sha256"] not in validation_groups], dtype=int)
    return inner_train, inner_val


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.maximum(prediction, 0.0)
    error = prediction - target
    target_log = np.log1p(target)
    prediction_log = np.log1p(prediction)
    return {
        "mae_seconds": float(np.mean(np.abs(error))),
        "medae_seconds": float(np.median(np.abs(error))),
        "rmse_seconds": float(np.sqrt(np.mean(error**2))),
        "r2_seconds": float(1.0 - np.sum(error**2) / np.sum((target - target.mean()) ** 2)),
        "mae_log1p_seconds": float(np.mean(np.abs(prediction_log - target_log))),
        "r2_log1p_seconds": float(1.0 - np.sum((prediction_log - target_log) ** 2) / np.sum((target_log - target_log.mean()) ** 2)),
    }


def split_sets(rows: list[dict[str, str]], folds: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    assignments = np.asarray([int(row["group_fold"]) for row in rows])
    backend = np.asarray([row["backend"] for row in rows])
    output = []
    for fold in range(1, folds + 1):
        output.append((f"grouped_qasm_fold_{fold}", np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)))
    for held in ("osaka", "kyoto"):
        for fold in range(1, folds + 1):
            test = np.flatnonzero((backend == held) & (assignments == fold))
            train = np.flatnonzero((backend != held) & (assignments != fold))
            if len(test) and len(train):
                output.append((f"strict_{held}_fold_{fold}", train, test))
    return output


def train_one(
    rows: list[dict[str, str]], sequences: dict[str, np.ndarray], train_indices: np.ndarray,
    test_indices: np.ndarray, device: torch.device, seed: int, epochs: int, patience: int,
) -> np.ndarray:
    seed_all(seed)
    inner_train, inner_val = group_inner_split(rows, train_indices, seed + 17)
    model = LayerResidualGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    target_log = np.asarray([np.log1p(float(row["target_seconds"])) for row in rows])
    qcre_log = np.asarray([np.log1p(float(row["qcre_critical_path_seconds"])) for row in rows])
    best_state = copy.deepcopy(model.state_dict())
    best_value = float("inf")
    stale = 0
    for _epoch in range(epochs):
        model.train()
        order = np.array(inner_train, copy=True)
        np.random.default_rng(seed + _epoch).shuffle(order)
        for index in order:
            sequence = torch.from_numpy(sequences[rows[index]["graph_id"]]).to(device)
            residual = model(sequence)
            target_residual = torch.tensor(target_log[index] - qcre_log[index], device=device, dtype=torch.float32)
            loss = (residual - target_residual) ** 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            values = []
            for index in inner_val:
                sequence = torch.from_numpy(sequences[rows[index]["graph_id"]]).to(device)
                values.append(float((qcre_log[index] + model(sequence)).cpu()))
        validation_loss = float(np.mean((np.asarray(values) - target_log[inner_val]) ** 2))
        if validation_loss < best_value:
            best_value = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for index in test_indices:
            sequence = torch.from_numpy(sequences[rows[index]["graph_id"]]).to(device)
            predictions.append(float(torch.expm1(qcre_log[index] + model(sequence)).cpu()))
    return np.asarray(predictions, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "validation" / "mali_physical_dag_v1")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "artifacts" / "validation" / "mali_deep_protocol_v1" / "split_audit" / "row_manifest.csv")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "work" / "mali_physical_dag_v1" / "graphs")
    parser.add_argument("--max-bins", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    manifest = read_csv(args.artifact_dir / "manifest.csv")
    split_rows = read_csv(args.split_manifest)
    split_by_key = {(row["qasm_sha256"], row["backend"]): row for row in split_rows}
    rows = []
    for row in manifest:
        split = split_by_key[(row["qasm_sha256"], row["backend"])]
        enriched = dict(row)
        enriched["group_fold"] = split["group_fold"]
        enriched["family_component"] = split["family_component"]
        rows.append(enriched)

    graph_ids = sorted({row["graph_id"] for row in rows})
    sequences = {
        graph_id: aggregate_sequence(args.graph_dir / f"{graph_id}.npz", args.max_bins)
        for graph_id in graph_ids
    }
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    split_rows_out: list[dict[str, object]] = []
    predictions_out: list[dict[str, object]] = []
    target = np.asarray([float(row["target_seconds"]) for row in rows])
    for split_name, train_indices, test_indices in split_sets(rows, args.folds):
        prediction = train_one(rows, sequences, train_indices, test_indices, device, args.seed, args.epochs, args.patience)
        result = metrics(target[test_indices], prediction)
        result.update({"split": split_name, "n_train": len(train_indices), "n_test": len(test_indices), "device": str(device)})
        split_rows_out.append(result)
        for index, estimate in zip(test_indices, prediction):
            predictions_out.append(
                {
                    "split": split_name,
                    "row_id": rows[index]["row_id"],
                    "circuit": rows[index]["circuit"],
                    "backend": rows[index]["backend"],
                    "family_component": rows[index]["family_component"],
                    "target_seconds": target[index],
                    "prediction_seconds": estimate,
                    "absolute_error_seconds": abs(float(estimate) - target[index]),
                }
            )
        print(f"layer-dag-rnn {split_name}: logR2={result['r2_log1p_seconds']:.4f} secR2={result['r2_seconds']:.4f} MAE={result['mae_seconds']:.4f}", flush=True)

    output_dir = args.artifact_dir / "layer_dag_rnn"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("metrics.csv", split_rows_out), ("oof_predictions.csv", predictions_out)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    summary = {
        "model": "physics_guided_coarsened_layer_DAG_GRU_residual",
        "n_rows": len(rows),
        "n_unique_graphs": len(graph_ids),
        "max_bins": args.max_bins,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "target": "log1p(Ma-Li observed result.time_taken seconds)",
        "residual": "log1p(target) - log1p(QCRE weighted critical path seconds)",
        "feature_semantics": "ordered native-DAG topology layers coarsened into at most max_bins bins",
        "provenance_class": "CURRENT_FAKE_SNAPSHOT_PROXY",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Ma--Li coarsened layer-DAG residual pilot",
        "",
        "This pilot keeps ordered compiled-DAG timing structure while avoiding a",
        "67-million-node PyG batch. Native nodes are aggregated into at most",
        f"**{args.max_bins}** ordered bins and a small GRU predicts the residual",
        "around the QCRE weighted critical path. It is not a full native-operation",
        "GNN and uses the current FakeBackend reconstruction.",
        "",
        f"Device: **{device}**; seed: **{args.seed}**; early stopping uses a",
        "group-disjoint inner validation split inside each outer training set.",
        "",
        "| Split | MAE (s) | MedAE (s) | log-R² | seconds-R² |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in split_rows_out:
        report.append(
            f"| {row['split']} | {row['mae_seconds']:.4f} | {row['medae_seconds']:.4f} | "
            f"{row['r2_log1p_seconds']:.4f} | {row['r2_seconds']:.4f} |"
        )
    report.extend(
        [
            "",
            "This is a pilot architecture result. It must be compared with the",
            "static physical-summary baseline in the same artifact, and it must",
            "pass the QWalk/tail guard and multiple-seed stability check before a",
            "claim about DAG topology is made.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
