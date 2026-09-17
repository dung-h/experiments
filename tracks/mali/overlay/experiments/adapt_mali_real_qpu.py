#!/usr/bin/env python3
"""Fine-tune the simulator checkpoint on the committed Osaka/Kyoto labels.

This is the second stage of Ma--Li's protocol: graph inputs are built for the
Osaka/Kyoto backend snapshots and the target is the already-recorded
``time_taken`` in ``data/{osaka,kyoto}_time_taken.csv``.  No QPU submission is
performed by this script.

The input graph artifact can be generated offline with, for example::

    .venv-mali/bin/python data_preparation/build_training_data_from_logs.py \
      --devices osaka kyoto --backend-properties-dir data/fake_backend_properties \
      --workers 8 --output data/osaka_kyoto_fake_standardization.npy \
      --manifest data/osaka_kyoto_fake_manifest.csv

The resulting experiment is labelled ``fake_snapshot_adaptation`` because a
fake backend snapshot is a proxy for the historical IBM properties used when
the measurements were collected.  Supplying a cache made from the actual
historical properties gives the exact-properties variant without changing
the training code.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from transformer_model import Simple_Model  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_args(layers: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        use_graph_features=True,
        use_global_features=True,
        use_gate_type=True,
        use_qubit_index=True,
        use_T1T2=True,
        use_gate_index=True,
        num_layers=layers,
    )


def load_dataset(path: Path) -> tuple[list, int]:
    with path.open("rb") as handle:
        dataset = pickle.load(handle)
    for graph in dataset:
        if graph.global_features.ndim == 1:
            graph.global_features = graph.global_features.unsqueeze(0)
        graph.y = graph.y.reshape(-1).to(torch.float32)
    gf_sizes = {int(graph.global_features.shape[-1]) for graph in dataset}
    if len(gf_sizes) != 1:
        raise ValueError(f"Inconsistent global feature widths: {sorted(gf_sizes)}")
    return dataset, gf_sizes.pop()


def evaluate(model, loader, device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch).reshape(-1).float()
            predictions.append(output.cpu().numpy())
            targets.append(batch.y.reshape(-1).float().cpu().numpy())
    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    mse = float(mean_squared_error(target, pred))
    variance = float(np.mean((target - target.mean()) ** 2))
    return (
        {
            "mse_seconds": mse,
            "rmse_seconds": float(np.sqrt(mse)),
            "mae_seconds": float(mean_absolute_error(target, pred)),
            "r2_seconds": float(r2_score(target, pred)),
            "nmse_seconds": float(mse / variance) if variance else float("nan"),
            "log_mae": float(
                mean_absolute_error(np.log1p(target), np.log1p(np.maximum(pred, 0)))
            ),
            "log_rmse": float(
                np.sqrt(
                    mean_squared_error(
                        np.log1p(target), np.log1p(np.maximum(pred, 0))
                    )
                )
            ),
        },
        target,
        pred,
    )


def fold_indices(n: int, fold: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Return an 8:1:1 split with a deterministic held-out tenth."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    fold_size = n // 10
    test_start = fold * fold_size
    test_end = (fold + 1) * fold_size if fold < 9 else n
    test = order[test_start:test_end]
    remaining = order[:test_start] + order[test_end:]

    # Shuffle only the non-test rows so every fold has a fresh fine-tuning
    # train/validation partition while the held-out fold remains fixed.
    random.Random(seed + 10000 + fold).shuffle(remaining)
    train_size = int(0.8 * n)
    valid_size = int(0.9 * n) - train_size
    train = remaining[:train_size]
    valid = remaining[train_size : train_size + valid_size]
    return train, valid, test


def _state_dict(checkpoint: Path, device: torch.device) -> dict:
    payload = torch.load(checkpoint, map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint}")
    return payload


def _adapt_global_feature_width(state: dict, target_width: int) -> dict:
    """Adapt the first global-feature layer when refinement masks differ.

    The committed real-QPU circuit subset has 40 retained global columns,
    while the simulator union has 41.  The extra simulator column is the
    ``rccx`` gate count (raw feature index 37); it is absent from every
    Osaka/Kyoto row.  Removing that input weight preserves the ordering of all
    shared features and is preferable to silently shifting columns or padding
    an arbitrary zero at the end.
    """
    current_width = int(state["gf_linear1.weight"].shape[1])
    if current_width == target_width:
        return state
    simulator_raw = [
        i
        for i in range(51)
        if i not in {4, 5, 33, 34, 38, 39, 40, 41, 42, 43}
    ]
    real_raw = [
        i
        for i in range(51)
        if i not in {4, 5, 33, 34, 37, 38, 39, 40, 41, 42, 43}
    ]
    if current_width == len(simulator_raw) and target_width == len(real_raw):
        keep = [simulator_raw.index(raw_index) for raw_index in real_raw]
        adapted = dict(state)
        adapted["gf_linear1.weight"] = state["gf_linear1.weight"][:, keep]
        return adapted
    raise ValueError(
        "Cannot align pretrained global-feature width "
        f"{current_width} -> {target_width}; provide an artifact with matching masks."
    )


def run(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset, global_feature_width = load_dataset(args.data)
    if len(dataset) < 10:
        raise ValueError("At least 10 graphs are required for 10-fold adaptation")
    manifest = None
    if args.manifest:
        import pandas as pd

        manifest = pd.read_csv(args.manifest)
        if len(manifest) != len(dataset):
            raise ValueError(
                f"Manifest rows ({len(manifest)}) do not match graphs ({len(dataset)})"
            )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_state = (
        _adapt_global_feature_width(_state_dict(args.pretrained, device), global_feature_width)
        if args.init == "pretrained"
        else None
    )
    fold_results = []
    started = time.perf_counter()

    for fold in range(args.folds):
        train_idx, valid_idx, test_idx = fold_indices(len(dataset), fold, args.seed)
        train_loader = DataLoader(
            [dataset[i] for i in train_idx], batch_size=args.batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            [dataset[i] for i in valid_idx], batch_size=args.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            [dataset[i] for i in test_idx], batch_size=args.batch_size, shuffle=False
        )

        model = Simple_Model(
            model_args(args.layers), length_of_gf=global_feature_width
        ).to(device)
        if pretrain_state is not None:
            model.load_state_dict(pretrain_state, strict=True)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        best_valid = float("inf")
        best_state = None
        history = []
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            seen = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch).reshape(-1).float()
                target = batch.y.reshape(-1).float()
                loss = torch.nn.functional.mse_loss(output, target)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(target)
                seen += len(target)
            valid_metrics, _, _ = evaluate(model, valid_loader, device)
            row = {
                "epoch": epoch + 1,
                "train_mse_seconds": total_loss / max(seen, 1),
                **{f"valid_{k}": v for k, v in valid_metrics.items()},
            }
            history.append(row)
            if valid_metrics["mse_seconds"] < best_valid:
                best_valid = valid_metrics["mse_seconds"]
                best_state = copy.deepcopy(model.state_dict())
            if (epoch + 1) % args.log_every == 0 or epoch == 0:
                print(
                    f"fold {fold + 1}/{args.folds} epoch {epoch + 1}/{args.epochs} "
                    f"train_mse={row['train_mse_seconds']:.5f} "
                    f"valid_r2={valid_metrics['r2_seconds']:.4f}",
                    flush=True,
                )

        if best_state is None:
            best_state = model.state_dict()
        model.load_state_dict(best_state)
        test_metrics, target, pred = evaluate(model, test_loader, device)
        fold_result = {
            "fold": fold,
            "split_sizes": [len(train_idx), len(valid_idx), len(test_idx)],
            "test_metrics": test_metrics,
            "best_valid_mse_seconds": best_valid,
            "history": history,
        }
        fold_results.append(fold_result)
        torch.save(
            {"model": best_state, "fold_result": fold_result},
            output_dir / f"fold_{fold:02d}_best.pt",
        )
        np.save(output_dir / f"fold_{fold:02d}_test_targets.npy", target)
        np.save(output_dir / f"fold_{fold:02d}_test_predictions.npy", pred)
        print(f"fold {fold + 1} test {json.dumps(test_metrics, sort_keys=True)}", flush=True)

    metric_names = list(fold_results[0]["test_metrics"])
    aggregate = {
        name: {
            "mean": float(np.mean([r["test_metrics"][name] for r in fold_results])),
            "std": float(np.std([r["test_metrics"][name] for r in fold_results])),
        }
        for name in metric_names
    }
    result = {
        "experiment": "fake_snapshot_adaptation"
        if args.properties_kind == "fake"
        else "real_qpu_adaptation",
        "properties_kind": args.properties_kind,
        "data": str(args.data),
        "manifest": str(args.manifest) if args.manifest else None,
        "pretrained": str(args.pretrained) if args.init == "pretrained" else None,
        "init": args.init,
        "protocol": {
            "samples": len(dataset),
            "folds": args.folds,
            "split": "8:1:1 inside each 10-fold held-out split",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "layers": args.layers,
            "seed": args.seed,
            "device": str(device),
        },
        "source_counts": (
            manifest["source_device"].value_counts().to_dict()
            if manifest is not None and "source_device" in manifest
            else None
        ),
        "target_semantics": (
            sorted(manifest["target_semantics"].dropna().unique().tolist())
            if manifest is not None and "target_semantics" in manifest
            else ["Ma-Li upstream time_taken; device-labelled run"]
        ),
        "aggregate_test": aggregate,
        "folds_detail": fold_results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    path = output_dir / f"{args.properties_kind}_{args.init}_result.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"saved {path}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--init", choices=["pretrained", "random"], default="pretrained")
    parser.add_argument("--properties-kind", choices=["fake", "historical"], default="fake")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments" / "results" / "adaptation"
    )
    args = parser.parse_args()
    if args.init == "pretrained" and args.pretrained is None:
        parser.error("--pretrained is required when --init pretrained")
    run(args)


if __name__ == "__main__":
    main()
