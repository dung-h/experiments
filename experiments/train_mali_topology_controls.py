#!/usr/bin/env python3
"""Retrained topology controls on the coarsened Ma--Li physical DAG corpus.

Every variant receives identical 256-bin node descriptors, target, split,
optimizer and parameterization.  Only the bin-level adjacency is changed:

* ``node_only``: no messages;
* ``true_dag``: source-to-destination dependency bins;
* ``reversed_dag``: the same edges reversed;
* ``shuffled_dag``: destination bins deterministically permuted per graph.

The test is deliberately modest: it determines whether coarse dependency
topology contributes beyond node/timing summaries.  It is not a full 67M-node
GNN claim.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from train_mali_layer_dag_rnn import (
    ROOT,
    group_inner_split,
    metrics,
    read_csv,
    seed_all,
    split_sets,
)


VARIANTS = ("node_only", "true_dag", "reversed_dag", "shuffled_dag")


def aggregate_graph(path: Path, max_bins: int, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bin descriptors and weighted directed coarse edges.

    Edge weights are log1p(native edge multiplicity). Shuffling preserves each
    coarse source edge count and the number of edges, but breaks source-to-
    destination dependency meaning with a graph-local deterministic permutation.
    """
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
    if not len(unique_layers):
        return np.zeros((1, 20), dtype=np.float32), np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.float32)
    bins = min(max_bins, len(unique_layers))
    layer_to_bin = np.floor(np.arange(len(unique_layers)) * bins / len(unique_layers)).astype(np.int64)
    node_bins = layer_to_bin[np.searchsorted(unique_layers, layer)]
    counts = np.bincount(node_bins, minlength=bins).astype(np.float64)
    output = np.zeros((bins, 20), dtype=np.float64)
    output[:, 0] = np.log1p(counts)
    for opcode_id in range(14):
        output[:, 1 + opcode_id] = np.log1p(np.bincount(node_bins, weights=(opcode == opcode_id).astype(float), minlength=bins))

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

    if variant == "node_only" or not len(edge_src):
        return output.astype(np.float32), np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.float32)
    source = node_bins[edge_src]
    destination = node_bins[edge_dst]
    packed = source.astype(np.int64) * bins + destination.astype(np.int64)
    unique, multiplicity = np.unique(packed, return_counts=True)
    source = unique // bins
    destination = unique % bins
    if variant == "reversed_dag":
        source, destination = destination, source
    elif variant == "shuffled_dag":
        # Stable graph-local shuffle: repeatable, preserves each source's
        # outgoing coarse-edge count, and removes the true target relation.
        token = hashlib.sha256(path.stem.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(token[:8], "little"))
        destination = rng.permutation(destination)
    elif variant != "true_dag":
        raise ValueError(f"unknown variant: {variant}")
    return output.astype(np.float32), np.vstack((source, destination)), np.log1p(multiplicity.astype(np.float32))


class CoarsenedMessageResidual(nn.Module):
    def __init__(self, input_dim: int = 20, hidden: int = 32) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.message_1 = nn.Linear(hidden, hidden, bias=False)
        self.update_1 = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.message_2 = nn.Linear(hidden, hidden, bias=False)
        self.update_2 = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1))

    @staticmethod
    def _propagate(hidden: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                   transform: nn.Linear, update: nn.Module) -> torch.Tensor:
        if edge_index.numel() == 0:
            aggregate = torch.zeros_like(hidden)
        else:
            source, destination = edge_index
            message = transform(hidden[source]) * edge_weight[:, None]
            aggregate = torch.zeros_like(hidden)
            aggregate.index_add_(0, destination, message)
            normalizer = torch.zeros((len(hidden),), dtype=hidden.dtype, device=hidden.device)
            normalizer.index_add_(0, destination, edge_weight)
            aggregate = aggregate / normalizer.clamp_min(1.0)[:, None]
        return update(torch.cat((hidden, aggregate), dim=-1))

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        hidden = self.input(nodes)
        hidden = self._propagate(hidden, edge_index, edge_weight, self.message_1, self.update_1)
        hidden = self._propagate(hidden, edge_index, edge_weight, self.message_2, self.update_2)
        pooled = torch.cat((hidden.mean(dim=0), hidden.max(dim=0).values, hidden[-1]), dim=-1)
        return self.head(pooled).squeeze()


def train_one(rows: list[dict[str, str]], graphs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
              train_indices: np.ndarray, test_indices: np.ndarray, device: torch.device,
              seed: int, epochs: int, patience: int) -> np.ndarray:
    seed_all(seed)
    inner_train, inner_val = group_inner_split(rows, train_indices, seed + 17)
    model = CoarsenedMessageResidual().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    target_log = np.asarray([np.log1p(float(row["target_seconds"])) for row in rows])
    qcre_log = np.asarray([np.log1p(float(row["qcre_critical_path_seconds"])) for row in rows])

    def tensor_graph(index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes, edge_index, edge_weight = graphs[rows[index]["graph_id"]]
        return (torch.from_numpy(nodes).to(device), torch.from_numpy(edge_index).to(device),
                torch.from_numpy(edge_weight).to(device))

    best_state = copy.deepcopy(model.state_dict()); best_value = float("inf"); stale = 0
    for epoch in range(epochs):
        model.train()
        order = np.array(inner_train, copy=True)
        np.random.default_rng(seed + epoch).shuffle(order)
        for index in order:
            prediction = model(*tensor_graph(index))
            residual = torch.tensor(target_log[index] - qcre_log[index], device=device, dtype=torch.float32)
            loss = (prediction - residual) ** 2
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            predicted = np.asarray([float((qcre_log[index] + model(*tensor_graph(index))).cpu()) for index in inner_val])
        value = float(np.mean((predicted - target_log[inner_val]) ** 2))
        if value < best_value:
            best_value = value; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        return np.asarray([float(torch.expm1(torch.tensor(qcre_log[index], device=device) + model(*tensor_graph(index))).cpu()) for index in test_indices])


def pooled(predictions: list[dict[str, object]], prefix: str, variant: str) -> dict[str, float]:
    selected = [row for row in predictions if row["variant"] == variant and str(row["split"]).startswith(prefix)]
    return metrics(np.asarray([float(row["target_seconds"]) for row in selected]),
                   np.asarray([float(row["prediction_seconds"]) for row in selected]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/validation/mali_physical_dag_v1")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "work/mali_physical_dag_v1/graphs")
    parser.add_argument("--max-bins", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--evaluation", choices=("all", "grouped", "strict"), default="all",
                        help="Run all controls, or a clearly-labelled grouped/strict screening pass.")
    parser.add_argument("--output-name", default="topology_controls",
                        help="Subdirectory below artifact-dir; use a distinct name for screening runs.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    manifest = read_csv(args.artifact_dir / "manifest.csv")
    split_map = {(row["qasm_sha256"], row["backend"]): row for row in read_csv(args.split_manifest)}
    rows = []
    for row in manifest:
        item = dict(row); split = split_map[(row["qasm_sha256"], row["backend"])]
        item.update({key: split[key] for key in ("group_fold", "family_component")}); rows.append(item)
    graph_ids = sorted({row["graph_id"] for row in rows})
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    all_graphs: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for variant in VARIANTS:
        all_graphs[variant] = {graph_id: aggregate_graph(args.graph_dir / f"{graph_id}.npz", args.max_bins, variant) for graph_id in graph_ids}
        edge_total = sum(graph[1].shape[1] for graph in all_graphs[variant].values())
        print(f"prepared {variant}: {len(graph_ids)} graphs, {edge_total} coarse edges", flush=True)

    selected_splits = split_sets(rows, args.folds)
    if args.evaluation != "all":
        selected_splits = [item for item in selected_splits if item[0].startswith(args.evaluation)]
    if not selected_splits:
        raise ValueError(f"no splits selected for evaluation={args.evaluation}")
    target = np.asarray([float(row["target_seconds"]) for row in rows])
    metric_rows: list[dict[str, object]] = []; prediction_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for variant in VARIANTS:
            for split_name, train, test in selected_splits:
                prediction = train_one(rows, all_graphs[variant], train, test, device, seed, args.epochs, args.patience)
                value = metrics(target[test], prediction)
                metric_rows.append({"seed": seed, "variant": variant, "split": split_name, "n_train": len(train), "n_test": len(test), **value})
                for index, estimate in zip(test, prediction):
                    prediction_rows.append({"seed": seed, "variant": variant, "split": split_name, "row_id": rows[index]["row_id"], "circuit": rows[index]["circuit"], "backend": rows[index]["backend"], "target_seconds": target[index], "prediction_seconds": estimate, "absolute_error_seconds": abs(float(estimate) - target[index])})
                print(f"topology-control seed={seed} {variant} {split_name}: logR2={value['r2_log1p_seconds']:.4f} MAE={value['mae_seconds']:.4f}", flush=True)

    output = args.artifact_dir / args.output_name; output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0])); writer.writeheader(); writer.writerows(metric_rows)
    with (output / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    summary = {"model": "two-layer_coarsened_message_passing_residual", "variants": list(VARIANTS), "seeds": args.seeds, "max_bins": args.max_bins, "epochs": args.epochs, "patience": args.patience, "device": str(device), "target": "log1p(Ma-Li observed result.time_taken seconds)", "provenance_class": "CURRENT_FAKE_SNAPSHOT_PROXY", "pooled_metrics": {}}
    for prefix, label in (("grouped", "grouped_qasm"), ("strict", "strict_all")):
        if not any(str(row["split"]).startswith(prefix) for row in prediction_rows):
            continue
        summary["pooled_metrics"][label] = {variant: {str(seed): pooled([row for row in prediction_rows if int(row["seed"]) == seed], prefix, variant) for seed in args.seeds} for variant in VARIANTS}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = ["# Ma--Li coarsened topology controls", "", "All variants use identical binned node features, splits, residual target and optimizer. Only adjacency differs. This is a retrained topology diagnostic, not a full native-DAG GNN result.", "", "| Evaluation | Variant | Seed | MAE (s) | log-R² | seconds-R² |", "|---|---|---:|---:|---:|---:|"]
    for label, key in (("grouped-QASM", "grouped_qasm"), ("strict backend+QASM", "strict_all")):
        if key not in summary["pooled_metrics"]:
            continue
        for variant in VARIANTS:
            for seed in args.seeds:
                value = summary["pooled_metrics"][key][variant][str(seed)]
                report.append(f"| {label} | `{variant}` | {seed} | {value['mae_seconds']:.4f} | {value['r2_log1p_seconds']:.4f} | {value['r2_seconds']:.4f} |")
    report += ["", "Topology is supported only if true DAG improves over both node-only and shuffled controls consistently across seeds and split families. Reversed-vs-true is a directional-signal diagnostic."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
