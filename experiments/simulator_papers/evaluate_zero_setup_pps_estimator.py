#!/usr/bin/env python3
"""Evaluate a pre-run local PPS propagation-time estimator.

Post-run representation statistics such as ``num_paulis`` and
``truncated_norm`` are deliberately excluded from model features. Only circuit
structure and declared approximation controls are used to predict
``propagation_seconds``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["log_trotter_steps", "neg_log10_delta", "circuit_depth", "circuit_gate_count", "topology_edges"]


def load(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" in row:
            continue
        rows.append(row)
    if len(rows) < 12:
        raise SystemExit("need at least twelve successful rows")
    frame = pd.DataFrame(rows)
    frame["propagation_seconds"] = frame["propagation_seconds"].astype(float)
    frame = frame[frame["propagation_seconds"] > 0].copy()
    frame["log_target"] = np.log(frame["propagation_seconds"])
    frame["log_trotter_steps"] = np.log(frame["num_trotter_steps"].astype(float))
    frame["neg_log10_delta"] = -np.log10(frame["delta"].astype(float))
    frame["configuration_group"] = frame["num_trotter_steps"].astype(str) + "::" + frame["delta"].astype(str)
    return frame


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    actual, predicted = np.exp(y), np.exp(p)
    return {
        "log_mae": float(mean_absolute_error(y, p)), "log_r2": float(r2_score(y, p)),
        "mae_seconds": float(mean_absolute_error(actual, predicted)),
        "r2_seconds": float(r2_score(actual, predicted)),
    }


def models(seed: int):
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, min_samples_leaf=3, l2_regularization=1.0, random_state=seed),
    }


def splits(frame: pd.DataFrame) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    ids = np.arange(len(frame))
    return {
        "configuration_group_5fold": list(GroupKFold(n_splits=5).split(ids, groups=frame["configuration_group"])),
        "delta_held_out": list(LeaveOneGroupOut().split(ids, groups=frame["delta"])),
        "trotter_steps_held_out": list(LeaveOneGroupOut().split(ids, groups=frame["num_trotter_steps"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    frame = load(args.input)
    x, y = frame[FEATURES], frame["log_target"].to_numpy(dtype=float)
    metric_rows, prediction_rows = [], []
    for split_name, fold_list in splits(frame).items():
        predictions = {"median_train_log": np.full(len(frame), np.nan)}
        predictions.update({name: np.full(len(frame), np.nan) for name in models(args.seed)})
        fold_number = np.full(len(frame), -1, dtype=int)
        for fold, (train, test) in enumerate(fold_list):
            predictions["median_train_log"][test] = float(np.median(y[train]))
            for name, factory in models(args.seed).items():
                model = factory()
                model.fit(x.iloc[train], y[train])
                predictions[name][test] = model.predict(x.iloc[test])
            fold_number[test] = fold
        for name, values in predictions.items():
            metric_rows.append({"split": split_name, "model": name, "n": len(frame), **score(y, values)})
            for index, value in enumerate(values):
                prediction_rows.append({
                    "split": split_name, "fold": int(fold_number[index]), "model": name,
                    "num_trotter_steps": int(frame.iloc[index]["num_trotter_steps"]), "delta": float(frame.iloc[index]["delta"]),
                    "actual_seconds": float(np.exp(y[index])), "predicted_seconds": float(np.exp(value)),
                })
    metrics = pd.DataFrame(metric_rows)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_canonical_rows.csv"), index=False)
    metrics.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv"), index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_prefix.with_name(args.output_prefix.name + "_oof_predictions.csv"), index=False)
    summary = {
        "target_semantics": "local CPU seconds inside propagate_through_rotation_gates only",
        "n_successful_rows": int(len(frame)), "features": FEATURES,
        "excluded_post_run_features": ["num_paulis", "truncated_norm", "expectation_value"],
        "splits": metrics.to_dict(orient="records"),
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    table = ["| Split | Model | Log MAE | Log R2 | MAE seconds | R2 seconds |", "|---|---|---:|---:|---:|---:|"]
    for row in metrics.itertuples(index=False):
        table.append(f"| {row.split} | {row.model} | {row.log_mae:.3f} | {row.log_r2:.3f} | {row.mae_seconds:.3f} | {row.r2_seconds:.3f} |")
    args.output_prefix.with_suffix(".md").write_text("\n".join([
        "# PPS propagation-time estimator", "",
        "The target is the local CPU propagation kernel only. Inputs are known pre-run circuit and approximation controls; post-run Pauli representation statistics are excluded.", "", *table,
        "", "Delta- and Trotter-step-held-out results are extrapolation diagnostics. They do not claim cloud MPS or generic simulator runtime prediction.", "",
    ]))
    print(args.output_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
