#!/usr/bin/env python3
"""Strict no-real-QASM simulator-to-QPU transfer screen for Ma--Li.

This is deliberately a lightweight common-representation experiment, not a
replacement for the original Graph Transformer: raw logical circuit features
are available with identical semantics for the simulator and observed-QPU
domains.  Every one of the 170 real-QPU QASM hashes is removed from simulator
pretraining.  The comparison is therefore circuit-inductive across domains.
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

from evaluate_mali_physical_baselines import ROOT, metrics, read_csv


GATES = ("h", "x", "sx", "rz", "rx", "ry", "cx", "cz", "ecr", "swap", "measure", "barrier")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def qasm_features(path: Path) -> np.ndarray:
    from qiskit import QuantumCircuit
    circuit = QuantumCircuit.from_qasm_file(path)
    counts = circuit.count_ops()
    two_q = sum(count for operation, count in counts.items() if circuit.num_qubits and operation in {"cx", "cz", "ecr", "swap", "rzz", "iswap"})
    raw = [circuit.num_qubits, circuit.depth(), circuit.size(), two_q, *[counts.get(gate, 0) for gate in GATES]]
    return np.log1p(np.asarray(raw, dtype=np.float64))


class SharedEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(width, 48), nn.LayerNorm(48), nn.GELU(), nn.Linear(48, 24), nn.GELU())
        self.head = nn.Linear(24, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(value)).squeeze(-1)


def group_inner(groups: np.ndarray, candidate: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted(set(groups[candidate])), dtype=object)
    rng = np.random.default_rng(seed); rng.shuffle(unique)
    valid = set(unique[:max(1, len(unique) // 5)])
    val = np.asarray([index for index in candidate if groups[index] in valid], dtype=int)
    train = np.asarray([index for index in candidate if groups[index] not in valid], dtype=int)
    return train, val


def train_model(model: nn.Module, x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray,
                seed: int, epochs: int, patience: int, freeze_encoder: bool = False) -> nn.Module:
    if freeze_encoder:
        for parameter in model.encoder.parameters(): parameter.requires_grad = False
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
    best = copy.deepcopy(model.state_dict()); best_value = float("inf"); stale = 0
    device = next(model.parameters()).device
    for epoch in range(epochs):
        order = np.array(train, copy=True); np.random.default_rng(seed + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), 64):
            index = order[start:start + 64]
            pred = model(torch.as_tensor(x[index], dtype=torch.float32, device=device))
            target = torch.as_tensor(y[index], dtype=torch.float32, device=device)
            loss = torch.mean((pred - target) ** 2)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            estimate = model(torch.as_tensor(x[val], dtype=torch.float32, device=device)).cpu().numpy()
        value = float(np.mean((estimate - y[val]) ** 2))
        if value < best_value:
            best = copy.deepcopy(model.state_dict()); best_value = value; stale = 0
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mali-root", type=Path, default=ROOT.parent / "Quantum-Execution-Time-Prediction")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/validation/mali_strict_transfer_logical_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 2025, 31415])
    parser.add_argument("--pretrain-epochs", type=int, default=150)
    parser.add_argument("--finetune-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()
    root = args.mali_root.resolve()
    import pandas as pd
    source = pd.read_csv(root / "data/training_data_manifest.csv")
    real = pd.read_csv(root / "data/osaka_kyoto_fake_manifest.csv")
    source["qasm_sha256"] = source["qasm_path"].map(lambda value: sha256(root / value))
    real["qasm_sha256"] = real["qasm_path"].map(lambda value: sha256(root / value))
    real_hashes = set(real["qasm_sha256"])
    source = source.loc[~source["qasm_sha256"].isin(real_hashes)].reset_index(drop=True)
    if source["qasm_sha256"].isin(real_hashes).any(): raise RuntimeError("real-QASM leakage remains in simulator pretraining")
    audit = read_csv(args.split_manifest)
    fold_by_key = {(row["qasm_sha256"], row["backend"]): int(row["group_fold"]) for row in audit}
    real["group_fold"] = [fold_by_key[(row.qasm_sha256, row.source_device)] for row in real.itertuples()]

    feature_cache: dict[str, np.ndarray] = {}
    def features(frame) -> np.ndarray:
        result = []
        for row in frame.itertuples():
            token = row.qasm_sha256
            if token not in feature_cache: feature_cache[token] = qasm_features(root / row.qasm_path)
            result.append(feature_cache[token])
        return np.asarray(result, dtype=np.float64)
    x_source_raw, x_real_raw = features(source), features(real)
    mean = x_source_raw.mean(axis=0); scale = x_source_raw.std(axis=0); scale[scale == 0] = 1.0
    x_source = ((x_source_raw - mean) / scale).astype(np.float32)
    x_real = ((x_real_raw - mean) / scale).astype(np.float32)
    y_source = np.log1p(source["target_time_taken"].to_numpy(dtype=float)).astype(np.float32)
    y_real = np.log1p(real["target_time_taken"].to_numpy(dtype=float)).astype(np.float32)
    groups_source = source["qasm_sha256"].to_numpy(); groups_real = real["qasm_sha256"].to_numpy()
    source_train, source_val = group_inner(groups_source, np.arange(len(source)), 727)
    output_rows: list[dict[str, object]] = []; metric_rows: list[dict[str, object]] = []
    source_records = []
    for seed in args.seeds:
        seed_all(seed)
        pretrained = SharedEncoder(x_source.shape[1])
        pretrained = train_model(pretrained, x_source, y_source, source_train, source_val, seed, args.pretrain_epochs, args.patience)
        pretrained_state = copy.deepcopy(pretrained.state_dict())
        with torch.no_grad():
            pred = pretrained(torch.as_tensor(x_source, dtype=torch.float32)).numpy()
        source_records.append({"seed": seed, **metrics(np.expm1(y_source), np.expm1(pred)), "source_rows": len(source), "source_qasm_hashes": len(set(groups_source))})
        for mode in ("scratch", "frozen_transfer", "finetune_transfer"):
            for fold in range(1, 6):
                test = np.flatnonzero(real["group_fold"].to_numpy() == fold)
                outer = np.flatnonzero(real["group_fold"].to_numpy() != fold)
                train, val = group_inner(groups_real, outer, seed + fold * 100)
                seed_all(seed + fold * 1000)
                model = SharedEncoder(x_real.shape[1])
                if mode != "scratch": model.load_state_dict(pretrained_state)
                model.head = nn.Linear(24, 1)
                model = train_model(model, x_real, y_real, train, val, seed + fold * 1000, args.finetune_epochs, args.patience, freeze_encoder=mode == "frozen_transfer")
                with torch.no_grad(): estimate = model(torch.as_tensor(x_real[test], dtype=torch.float32)).numpy()
                seconds = np.expm1(estimate); actual = np.expm1(y_real[test])
                value = metrics(actual, seconds)
                metric_rows.append({"seed": seed, "mode": mode, "fold": fold, "n_train": len(train), "n_validation": len(val), "n_test": len(test), **value})
                for index, prediction in zip(test, seconds):
                    output_rows.append({"seed": seed, "mode": mode, "fold": fold, "row_id": f"{real.iloc[index].source_device}:{real.iloc[index].sample_index}", "qasm_sha256": groups_real[index], "backend": real.iloc[index].source_device, "target_seconds": float(actual[list(test).index(index)]), "prediction_seconds": float(prediction), "absolute_error_seconds": abs(float(prediction) - float(actual[list(test).index(index)]))})
                print(f"transfer seed={seed} mode={mode} fold={fold}: logR2={value['r2_log1p_seconds']:.4f} MAE={value['mae_seconds']:.4f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("source_pretraining_metrics.csv", source_records), ("real_transfer_metrics.csv", metric_rows), ("real_transfer_oof_predictions.csv", output_rows)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    pooled = {}
    for mode in ("scratch", "frozen_transfer", "finetune_transfer"):
        pooled[mode] = {}
        for seed in args.seeds:
            chosen = [row for row in output_rows if row["mode"] == mode and row["seed"] == seed]
            pooled[mode][str(seed)] = metrics(np.asarray([row["target_seconds"] for row in chosen]), np.asarray([row["prediction_seconds"] for row in chosen]))
    summary = {"experiment": "strict_no_real_qasm_logical_feature_transfer_screen", "target_semantics": "Ma-Li observed Osaka/Kyoto result.time_taken; 1024 shots; queue excluded", "source_semantics": "Ma-Li Washington/Sherbrooke simulator/device-labelled runtime; auxiliary pretraining only", "source_rows_before_filter": 3020, "source_rows_after_filter": len(source), "source_qasm_hashes_after_filter": len(set(groups_source)), "removed_real_qasm_hashes": len(real_hashes), "removed_source_rows_matching_real_qasm": 3020 - len(source), "real_rows": len(real), "real_qasm_hashes": len(real_hashes), "representation": "logical QASM structural features shared exactly across source and target; not a Graph Transformer", "seeds": args.seeds, "pooled_real_oof": pooled}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = ["# Ma--Li strict no-real-QASM transfer screen", "", "All 170 real-QPU QASM hashes are removed from simulator pretraining. The model is a lightweight shared logical-feature MLP, not the original Graph Transformer; it tests strict circuit-inductive transfer in a common source/target representation.", "", f"Simulator rows: 3,020 before filtering, {len(source):,} after filtering. Removed rows: {3020-len(source):,}.", "", "| Mode | Seed | MAE (s) | log-R² | seconds-R² |", "|---|---:|---:|---:|---:|"]
    for mode in ("scratch", "frozen_transfer", "finetune_transfer"):
        for seed in args.seeds:
            value = pooled[mode][str(seed)]
            report.append(f"| `{mode}` | {seed} | {value['mae_seconds']:.4f} | {value['r2_log1p_seconds']:.4f} | {value['r2_seconds']:.4f} |")
    report += ["", "A strict transfer benefit requires frozen or finetuned transfer to beat scratch consistently. This screen cannot establish that the full original DAG transformer transfers, but it does remove exact circuit identity leakage from the common logical representation."]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
