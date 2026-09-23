#!/usr/bin/env python3
"""Audit what the saved Ma--Li graph models use at inference time.

This experiment evaluates the already-trained out-of-fold checkpoints under
controlled test-time interventions.  It does *not* retrain a model and must not
be described as a feature-ablation benchmark.  Setting a standardized feature
block to zero replaces it with the artifact-wide mean; edge interventions are
deliberately out-of-distribution diagnostics.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader


VARIANTS = (
    "full",
    "zero_node_type",
    "zero_qubit_index",
    "zero_T1T2",
    "zero_node_index",
    "no_edges",
    "reverse_edges",
    "rewired_edges",
)

FEATURE_SLICES = {
    "zero_node_type": slice(0, 46),
    "zero_qubit_index": slice(46, 173),
    "zero_T1T2": slice(173, 177),
    "zero_node_index": slice(177, 178),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mali-root",
        type=Path,
        default=root.parent / "Quantum-Execution-Time-Prediction",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts" / "mali" / "graph_intervention_audit_20260921",
    )
    return parser.parse_args()


def fold_indices(n: int, fold: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Reproduce the held-out folds used by the local Ma--Li adaptation."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    fold_size = n // 10
    test_start = fold * fold_size
    test_end = (fold + 1) * fold_size if fold < 9 else n
    test = order[test_start:test_end]
    remaining = order[:test_start] + order[test_end:]
    random.Random(seed + 10000 + fold).shuffle(remaining)
    train_size = int(0.8 * n)
    valid_size = int(0.9 * n) - train_size
    return (
        remaining[:train_size],
        remaining[train_size : train_size + valid_size],
        test,
    )


def model_args() -> SimpleNamespace:
    return SimpleNamespace(
        use_graph_features=True,
        use_global_features=True,
        use_gate_type=True,
        use_qubit_index=True,
        use_T1T2=True,
        use_gate_index=True,
        num_layers=3,
    )


def mutate_graph(graph, variant: str, seed: int):
    result = graph.clone()
    if result.global_features.ndim == 1:
        result.global_features = result.global_features.unsqueeze(0)
    result.y = result.y.reshape(-1).to(torch.float32)

    if variant in FEATURE_SLICES:
        # The stored artifact is standardized.  Zero therefore represents the
        # artifact-wide mean, not a raw physical value of zero.
        result.x[:, FEATURE_SLICES[variant]] = 0
    elif variant == "no_edges":
        result.edge_index = torch.empty((2, 0), dtype=torch.long)
    elif variant == "reverse_edges":
        result.edge_index = torch.flip(result.edge_index, dims=[0])
    elif variant == "rewired_edges":
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(result.edge_index.shape[1], generator=generator)
        result.edge_index = torch.stack(
            [result.edge_index[0], result.edge_index[1, permutation]], dim=0
        )
    elif variant != "full":
        raise ValueError(f"Unknown intervention: {variant}")
    return result


def metric_row(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(target, prediction))
    target_log = np.log1p(target)
    prediction_log = np.log1p(np.maximum(prediction, 0))
    return {
        "rows": int(len(target)),
        "mae_seconds": float(mean_absolute_error(target, prediction)),
        "rmse_seconds": float(np.sqrt(mse)),
        "r2_seconds": float(r2_score(target, prediction)),
        "log_mae": float(mean_absolute_error(target_log, prediction_log)),
        "log_rmse": float(np.sqrt(mean_squared_error(target_log, prediction_log))),
        "log_r2": float(r2_score(target_log, prediction_log)),
    }


def main() -> None:
    args = parse_args()
    mali_root = args.mali_root.resolve()
    sys.path.insert(0, str(mali_root / "model"))
    from transformer_model import Simple_Model  # noqa: PLC0415

    data_path = mali_root / "data" / "osaka_kyoto_fake_standardization.npy"
    manifest_path = mali_root / "data" / "osaka_kyoto_fake_manifest.csv"
    run_dirs = {
        "pretrained": mali_root / "experiments" / "results" / "adaptation_fake_full",
        "scratch": mali_root
        / "experiments"
        / "results"
        / "adaptation_fake_full_from_scratch",
    }

    with data_path.open("rb") as handle:
        dataset = pickle.load(handle)
    manifest = pd.read_csv(manifest_path)
    if len(dataset) != len(manifest):
        raise ValueError(f"graph/manifest mismatch: {len(dataset)} != {len(manifest)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    global_widths = {int(graph.global_features.shape[-1]) for graph in dataset}
    if len(global_widths) != 1:
        raise ValueError(f"inconsistent global widths: {sorted(global_widths)}")
    global_width = global_widths.pop()

    started = time.perf_counter()
    prediction_rows: list[dict] = []
    parameter_count = None
    for initialization, run_dir in run_dirs.items():
        for fold in range(10):
            _, _, test_indices = fold_indices(len(dataset), fold, args.seed)
            payload = torch.load(
                run_dir / f"fold_{fold:02d}_best.pt",
                map_location=device,
                weights_only=False,
            )
            model = Simple_Model(model_args(), length_of_gf=global_width).to(device)
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
            if parameter_count is None:
                parameter_count = int(sum(p.numel() for p in model.parameters()))

            with torch.inference_mode():
                for variant_index, variant in enumerate(VARIANTS):
                    mutated = [
                        mutate_graph(
                            dataset[index],
                            variant,
                            args.seed + fold * 100 + variant_index * 1000 + index,
                        )
                        for index in test_indices
                    ]
                    offset = 0
                    loader = DataLoader(
                        mutated, batch_size=args.batch_size, shuffle=False
                    )
                    for batch in loader:
                        batch = batch.to(device)
                        outputs = model(batch).reshape(-1).detach().cpu().numpy()
                        targets = batch.y.reshape(-1).detach().cpu().numpy()
                        batch_indices = test_indices[offset : offset + len(outputs)]
                        offset += len(outputs)
                        for sample_index, target, prediction in zip(
                            batch_indices, targets, outputs
                        ):
                            source = manifest.iloc[sample_index]
                            prediction_rows.append(
                                {
                                    "initialization": initialization,
                                    "fold": fold,
                                    "variant": variant,
                                    "sample_index": int(sample_index),
                                    "source_device": source["source_device"],
                                    "circuit_name": source["circuit_name"],
                                    "target_seconds": float(target),
                                    "prediction_seconds": float(prediction),
                                    "absolute_error_seconds": float(
                                        abs(float(target) - float(prediction))
                                    ),
                                }
                            )

    predictions = pd.DataFrame(prediction_rows)
    metrics_rows = []
    for (initialization, variant), frame in predictions.groupby(
        ["initialization", "variant"], sort=False
    ):
        row = {
            "initialization": initialization,
            "variant": variant,
            **metric_row(
                frame["target_seconds"].to_numpy(),
                frame["prediction_seconds"].to_numpy(),
            ),
        }
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "oof_intervention_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "oof_intervention_metrics.csv", index=False)
    metadata = {
        "experiment": "mali_saved_model_test_time_graph_intervention_audit",
        "interpretation": "diagnostic intervention; not a retrained feature ablation",
        "target_semantics": (
            "Ma--Li recorded IBM Osaka/Kyoto result.time_taken at 1024 shots; "
            "queue time excluded"
        ),
        "data": str(data_path),
        "manifest": str(manifest_path),
        "checkpoint_directories": {k: str(v) for k, v in run_dirs.items()},
        "rows": len(dataset),
        "folds": 10,
        "seed": args.seed,
        "device": str(device),
        "model_parameters": parameter_count,
        "variants": list(VARIANTS),
        "feature_zero_semantics": (
            "zero in the stored standardized tensor equals the artifact-wide mean"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics.to_dict(orient="records"),
    }
    (args.output_dir / "metadata_and_metrics.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(metrics.to_string(index=False))
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
