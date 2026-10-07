#!/usr/bin/env python3
"""Run or dry-run one frozen outer fold of the unified Ma--Li graph model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
OUTER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv"
FEATURES = ROOT / "artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c134_feature_sidecar_v3.csv"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def global_values(row: dict[str, str]) -> np.ndarray:
    return np.asarray([np.log1p(max(0.0, float(row[name]))) for name in ("active_width", "structural_depth", "one_qubit_count", "two_qubit_count", "swap_like_count", "measurement_count", "shots")], dtype=np.float32)


def record_to_data(record: dict, row: dict[str, str], vocab: dict[str, int], mean: np.ndarray, std: np.ndarray) -> Data:
    gate_width = len(vocab) + 1
    node_width = gate_width + 4 + 1 + 4
    names = record["names"]
    n = len(names)
    x = torch.zeros((n, node_width), dtype=torch.float32)
    max_q = max(1, int(record["num_qubits"]) - 1)
    for index, name in enumerate(names):
        x[index, vocab.get(name, vocab["__UNK_GATE__"])] = 1.0
        arity = min(int(record["arity"][index]), 3)
        x[index, gate_width + arity] = 1.0
        x[index, gate_width + 4] = float(record["parameter"][index])
        positions = [int(record[key][index]) for key in ("q0", "q1", "q2", "q3") if int(record[key][index]) >= 0]
        if positions:
            normalized = np.asarray(positions, dtype=np.float32) / float(max_q)
            x[index, gate_width + 5:gate_width + 8] = torch.tensor([normalized.min(), normalized.mean(), normalized.max()])
            x[index, gate_width + 8] = 1.0
    edge_index = torch.empty((2, len(record["edge_src"])), dtype=torch.long)
    if len(record["edge_src"]):
        edge_index[0] = torch.as_tensor(record["edge_src"], dtype=torch.long)
        edge_index[1] = torch.as_tensor(record["edge_dst"], dtype=torch.long)
    values = (global_values(row) - mean) / std
    data = Data(x=x, edge_index=edge_index, global_features=torch.as_tensor(values, dtype=torch.float32).reshape(1, -1))
    data.y = torch.tensor(float(row["target_seconds"]), dtype=torch.float32)
    return data


class UnifiedGraphModel(torch.nn.Module):
    def __init__(self, node_width: int, global_width: int = 7):
        super().__init__()
        self.conv0 = TransformerConv(node_width, 64)
        self.conv1 = TransformerConv(64, 64)
        self.conv2 = TransformerConv(64, 64)
        self.gf1 = torch.nn.Linear(global_width, 64)
        self.gf2 = torch.nn.Linear(64, 64)
        self.head1 = torch.nn.Linear(128, 512)
        self.head2 = torch.nn.Linear(512, 512)
        self.head3 = torch.nn.Linear(512, 128)
        self.head4 = torch.nn.Linear(128, 1)

    def forward(self, batch):
        x = torch.relu(self.conv0(batch.x, batch.edge_index))
        x = torch.relu(self.conv1(x, batch.edge_index))
        x = torch.relu(self.conv2(x, batch.edge_index))
        x = global_mean_pool(x, batch.batch)
        gf = torch.relu(self.gf1(batch.global_features))
        gf = torch.relu(self.gf2(gf))
        x = torch.cat([x, gf], dim=1)
        x = torch.relu(self.head1(x)); x = torch.relu(self.head2(x)); x = torch.relu(self.head3(x))
        return self.head4(x).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,1234,31415")
    parser.add_argument("--fold", type=int, choices=range(5), default=0,
                        help="Frozen outer fold to predict (default: 0).")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; CPU fallback is forbidden")
    panel = {row["canonical_observation_id"]: row for row in read(args.panel_dir / "graph_row_manifest.csv")}
    canonical = {row["canonical_row_id"]: row for row in read(CANONICAL)}
    outer = {row["canonical_observation_id"]: row for row in read(OUTER)}
    feature = {row["canonical_row_id"]: row for row in read(FEATURES)}
    ids = set(canonical)
    if set(panel) != ids or set(outer) != ids or set(feature) != ids:
        raise ValueError("graph panel/canonical/outer/feature identity mismatch")
    eligible = sorted(cid for cid, row in panel.items() if row["status"] == "eligible")
    train_ids = [cid for cid in eligible if int(outer[cid]["outer_fold"]) != args.fold]
    test_ids = [cid for cid in eligible if int(outer[cid]["outer_fold"]) == args.fold]
    train_groups = {outer[cid]["unified_leakage_group_id"] for cid in train_ids}
    test_groups = {outer[cid]["unified_leakage_group_id"] for cid in test_ids}
    if train_groups & test_groups:
        raise ValueError("graph outer leakage")
    vocab_names: set[str] = set()
    records: dict[str, dict] = {}
    for cid in train_ids + test_ids:
        path = args.panel_dir / "graphs" / panel[cid]["graph_file"]
        with path.open("rb") as handle:
            record = pickle.load(handle)
        records.setdefault(panel[cid]["representation_digest"], record)
        if cid in train_ids:
            vocab_names.update(record["names"])
    vocab = {name: index for index, name in enumerate(sorted(vocab_names))}
    vocab["__UNK_GATE__"] = len(vocab)
    train_values = np.stack([global_values(feature[cid]) for cid in train_ids])
    mean = train_values.mean(axis=0); std = train_values.std(axis=0); std[std < 1e-6] = 1.0
    # Build one Data object per observation.  A representation digest may be
    # shared by rows from different backends/shots, so global features and y
    # must never be cached with the topology.  Reusing a Data object here
    # would silently train on the first row's target for every duplicate.
    train_data = [
        record_to_data(
            records[panel[cid]["representation_digest"]],
            {**feature[cid], "target_seconds": canonical[cid]["target_seconds"]},
            vocab,
            mean,
            std,
        )
        for cid in train_ids
    ]
    test_data = [
        record_to_data(
            records[panel[cid]["representation_digest"]],
            {**feature[cid], "target_seconds": canonical[cid]["target_seconds"]},
            vocab,
            mean,
            std,
        )
        for cid in test_ids
    ]
    args.output_dir.mkdir(parents=True)
    dry = {"artifact_id": f"s71-graph-fold{args.fold}-dry-run-v3", "status": "PASS", "outer_fold": args.fold, "device": torch.cuda.get_device_name(0), "train_rows": len(train_data), "test_rows": len(test_data), "train_groups": len(train_groups), "test_groups": len(test_groups), "group_overlap": len(train_groups & test_groups), "node_width": int(train_data[0].x.shape[1]), "unique_graphs": len(records), "vocab_size": len(vocab), "seeds": [int(x) for x in args.seeds.split(",")], "forbidden_backend_feature": True, "per_observation_targets": True}
    if args.dry_run:
        (args.output_dir / "dry_run.json").write_text(json.dumps(dry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.output_dir / "run_manifest.json").write_text(json.dumps({
            "artifact_id": dry["artifact_id"],
            "status": "PASS",
            "mode": "dry_run",
            "inputs": {"canonical_sha256": sha(CANONICAL), "outer_split_sha256": sha(OUTER), "feature_sidecar_sha256": sha(FEATURES), "graph_row_manifest_sha256": sha(args.panel_dir / "graph_row_manifest.csv")},
            "device": dry["device"],
            "outer_fold": args.fold,
            "forbidden_backend_feature": True,
            "contract": "benchmark_v1/protocol/mali_unified_graph_outer_cv_v4.json",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(dry, indent=2, sort_keys=True)); return 0
    torch.set_num_threads(args.cpu_threads)
    all_results = []
    device = torch.device("cuda:0")
    for seed in [int(x) for x in args.seeds.split(",")]:
        seed_everything(seed)
        model = UnifiedGraphModel(int(train_data[0].x.shape[1])).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        model.train()
        started = time.monotonic()
        for epoch in range(args.epochs):
            losses = []
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch)
                loss = torch.nn.functional.mse_loss(prediction, torch.log1p(batch.y.reshape(-1)))
                loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
            if not math.isfinite(float(np.mean(losses))):
                raise RuntimeError(f"non-finite training loss at seed={seed}, epoch={epoch}")
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        model.eval(); predicted=[]; actual=[]
        with torch.no_grad():
            for batch in test_loader:
                batch=batch.to(device); predicted.extend(torch.expm1(model(batch)).clamp_min(0).detach().cpu().tolist()); actual.extend(batch.y.reshape(-1).detach().cpu().tolist())
        result={"seed":seed,"status":"PASS","n_test":len(actual),"mae_seconds":float(mean_absolute_error(actual,predicted)),"r2_seconds":float(r2_score(actual,predicted)),"elapsed_seconds":time.monotonic()-started,"predicted_seconds":predicted,"actual_seconds":actual}
        all_results.append(result)
    pred = np.median(np.asarray([result["predicted_seconds"] for result in all_results]), axis=0)
    actual = np.asarray(all_results[0]["actual_seconds"])
    prediction_path = args.output_dir / "predictions.csv"
    prediction_fields = ["canonical_observation_id", "source_id", "outer_fold", "actual_seconds", "predicted_seconds", "absolute_error_seconds"] + [f"predicted_seed_{seed}_seconds" for seed in [int(x) for x in args.seeds.split(",")]]
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for index, cid in enumerate(test_ids):
            row = {
                "canonical_observation_id": cid,
                "source_id": outer[cid]["source_id"],
                "outer_fold": args.fold,
                "actual_seconds": float(actual[index]),
                "predicted_seconds": float(pred[index]),
                "absolute_error_seconds": abs(float(pred[index]) - float(actual[index])),
            }
            for result in all_results:
                row[f"predicted_seed_{result['seed']}_seconds"] = float(result["predicted_seconds"][index])
            writer.writerow(row)
    summary = {"artifact_id":f"s71-graph-fold{args.fold}-checkpoint-v3","status":"PASS","outer_fold":args.fold,"seeds":[result["seed"] for result in all_results],"n_train":len(train_data),"n_test":len(test_data),"group_overlap":0,"mae_seconds":float(mean_absolute_error(actual,pred)),"r2_seconds":float(r2_score(actual,pred)),"device":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda":torch.version.cuda,"node_width":int(train_data[0].x.shape[1]),"batch_size":args.batch_size,"epochs":args.epochs,"forbidden_backend_feature":True,"per_observation_targets":True,"seed_results":[{k:v for k,v in result.items() if k not in {"predicted_seconds","actual_seconds"}} for result in all_results]}
    summary["prediction_sha256"] = sha(prediction_path)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (args.output_dir / "run_manifest.json").write_text(json.dumps({
        "artifact_id": summary["artifact_id"],
        "status": "PASS",
        "mode": "full_checkpoint",
        "command": sys.argv,
        "inputs": {"canonical_sha256": sha(CANONICAL), "outer_split_sha256": sha(OUTER), "feature_sidecar_sha256": sha(FEATURES), "graph_row_manifest_sha256": sha(args.panel_dir / "graph_row_manifest.csv")},
        "outputs": {"predictions_sha256": summary["prediction_sha256"], "summary_sha256": sha(summary_path)},
        "contract": "benchmark_v1/protocol/mali_unified_graph_outer_cv_v4.json",
        "device": summary["device"],
        "cuda": summary["cuda"],
        "outer_fold": args.fold,
        "forbidden_backend_feature": True,
        "per_observation_targets": True,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary,indent=2,sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
