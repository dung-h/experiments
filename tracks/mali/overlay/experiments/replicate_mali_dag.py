#!/usr/bin/env python3
"""Reproduce Ma--Li's graph-transformer runtime experiment locally.

The paper trains a three-layer Graph Transformer on 3,020 FakeWashington /
FakeSherbrooke samples, combines 178-dimensional graph nodes with 41 global
features, and reports raw-seconds MSE, NMSE and R-squared on a fixed 8:1:1
split.  This runner keeps graphs on CPU and transfers one batch at a time so
the 12 GB serialized extract can be used with a 16 GB GPU.

Example (main simulator experiment)::

    CUDA_VISIBLE_DEVICES=0 .venv-mali-gpu/bin/python \
      experiments/replicate_mali_dag.py --epochs 500 --batch-size 32

The paper's batch size 128 is exposed, but it exceeds this GPU's memory for
the largest graph batches; 32 is the largest tested safe setting here.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from transformer_model import Simple_Model  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path, limit: int | None = None) -> list:
    with path.open("rb") as handle:
        dataset = pickle.load(handle)
    if limit is not None:
        dataset = dataset[:limit]
    # CircDataset in the original package adds this batch dimension before
    # using PyG DataLoader.  The serialized artifact intentionally stays 1-D.
    for graph in dataset:
        if graph.global_features.ndim == 1:
            graph.global_features = graph.global_features.unsqueeze(0)
        graph.y = graph.y.reshape(-1).to(torch.float32)
    return dataset


def split_dataset(dataset: list, seed: int) -> tuple[list, list, list]:
    # Match the repository's seeded random.shuffle + 8:1:1 partition.
    shuffled = list(dataset)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(0.8 * n)
    valid_end = int(0.9 * n)
    return shuffled[:train_end], shuffled[train_end:valid_end], shuffled[valid_end:]


def model_args(mode: str, layers: int) -> SimpleNamespace:
    return SimpleNamespace(
        use_graph_features=mode in ("full", "graph"),
        use_global_features=mode in ("full", "global"),
        use_gate_type=True,
        use_qubit_index=True,
        use_T1T2=True,
        use_gate_index=True,
        num_layers=layers,
    )


def evaluate(model, loader, device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).reshape(-1).float()
            predictions.append(out.cpu().numpy())
            targets.append(batch.y.reshape(-1).float().cpu().numpy())
    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    mse = float(mean_squared_error(target, pred))
    variance = float(np.mean((target - target.mean()) ** 2))
    metrics = {
        "mse_seconds": mse,
        "rmse_seconds": float(np.sqrt(mse)),
        "mae_seconds": float(mean_absolute_error(target, pred)),
        "r2_seconds": float(r2_score(target, pred)),
        "nmse_seconds": float(mse / variance) if variance else float("nan"),
        "log_mae": float(mean_absolute_error(np.log1p(target), np.log1p(np.maximum(pred, 0)))),
        "log_rmse": float(
            np.sqrt(mean_squared_error(np.log1p(target), np.log1p(np.maximum(pred, 0))))
        ),
    }
    return metrics, target, pred


def run(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dataset = load_dataset(args.data, args.limit)
    gf_sizes = {int(graph.global_features.shape[-1]) for graph in dataset}
    if len(gf_sizes) != 1:
        raise ValueError(f"Inconsistent global feature widths: {sorted(gf_sizes)}")
    global_feature_width = gf_sizes.pop()
    train, valid, test = split_dataset(dataset, args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    # Fake/backend feature masks can retain 40 columns while the simulator
    # union has 41. Infer the serialized width instead of silently assuming
    # the simulator-only default.
    model = Simple_Model(
        model_args(args.mode, args.layers), length_of_gf=global_feature_width
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{args.mode}_checkpoint.pt"
    start_epoch = 0
    best_valid = float("inf")
    best_state = None
    history = []
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_valid = float(checkpoint["best_valid"])
        best_state = checkpoint["best_state"]
        history = checkpoint.get("history", [])
        print(f"resumed {checkpoint_path} at epoch {start_epoch}", flush=True)

    print(
        f"mode={args.mode} device={device} samples={len(dataset)} "
        f"split={len(train)}/{len(valid)}/{len(test)} batch={args.batch_size} "
        f"epochs={args.epochs}",
        flush=True,
    )
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch).reshape(-1).float()
            target = batch.y.reshape(-1).float()
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(target)
            seen += len(target)
        scheduler.step()
        train_loss /= max(seen, 1)
        valid_metrics, _, _ = evaluate(model, valid_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_mse_seconds": train_loss,
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }
        history.append(row)
        if valid_metrics["mse_seconds"] < best_valid:
            best_valid = valid_metrics["mse_seconds"]
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % args.log_every == 0 or epoch == start_epoch:
            print(
                f"epoch {epoch + 1}/{args.epochs} train_mse={train_loss:.5f} "
                f"valid_mse={valid_metrics['mse_seconds']:.5f} "
                f"valid_r2={valid_metrics['r2_seconds']:.4f} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "best_state": best_state,
                    "optimizer": optimizer.state_dict(),
                    "best_valid": best_valid,
                    "history": history,
                },
                checkpoint_path,
            )

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    test_metrics, target, pred = evaluate(model, test_loader, device)
    result = {
        "paper_protocol": {
            "dataset": str(args.data),
            "mode": args.mode,
            "split": [len(train), len(valid), len(test)],
            "seed": args.seed,
            "layers": args.layers,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "device": str(device),
        },
        "best_valid_mse_seconds": best_valid,
        "test": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    if device.type == "cuda":
        result["peak_cuda_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    result_path = output_dir / f"{args.mode}_result.json"
    result_path.write_text(json.dumps(result, indent=2))
    torch.save({"model": best_state, "result": result}, output_dir / f"{args.mode}_best.pt")
    np.save(output_dir / f"{args.mode}_test_targets.npy", target)
    np.save(output_dir / f"{args.mode}_test_predictions.npy", pred)
    print("test", json.dumps(test_metrics, sort_keys=True), flush=True)
    print(f"saved {result_path}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "training_data_standardization.npy",
    )
    parser.add_argument("--mode", choices=["full", "graph", "global"], default="full")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Small smoke subset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "results")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
