#!/usr/bin/env python3
"""Evaluate source-aware noisy-Aer runtime predictions with conformal bands.

The input must contain successful local Aer executions after fake-backend
transpilation. The model target is ``log1p(t_exec_s)``. A train-fold-only
calibration partition supplies a 90% absolute-residual interval; no future
resource-limit row is treated as a completed runtime label.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


FEATURES = [
    "logical_width", "logical_depth", "logical_ops", "logical_two_qubit_ops",
    "physical_width", "physical_depth", "physical_ops", "physical_two_qubit_ops",
    "swap_count", "dag_node_count", "optimization_level", "backend_is_sherbrooke",
]


def load(path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]
    if len(rows) < 100:
        raise SystemExit("expected a complete successful Aer matrix")
    x = np.asarray([[
        float(row.get(feature.replace("backend_is_sherbrooke", "") or 0.0)
              if feature != "backend_is_sherbrooke" else row["backend_class"] == "FakeSherbrooke")
        for feature in FEATURES
    ] for row in rows], dtype=float)
    y = np.log1p(np.asarray([float(row["t_exec_s"]) for row in rows], dtype=float))
    return rows, x, y


def outer_splits(rows: list[dict[str, str]]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    index = np.arange(len(rows))
    return {
        "grouped_circuit_5fold": list(GroupKFold(n_splits=5).split(index, groups=np.asarray([row["circuit_id"] for row in rows]))),
        "family_held_out": list(GroupKFold(n_splits=5).split(index, groups=np.asarray([row["family"] for row in rows]))),
        "backend_held_out": [
            (np.asarray([i for i, row in enumerate(rows) if row["backend_class"] != backend]),
             np.asarray([i for i, row in enumerate(rows) if row["backend_class"] == backend]))
            for backend in sorted({row["backend_class"] for row in rows})
        ],
    }


def model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=12, min_samples_leaf=5,
        l2_regularization=1.0, random_state=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if not 0.5 < args.coverage < 1:
        raise SystemExit("coverage must be between 0.5 and 1")
    rows, x, y = load(args.input)
    metric_rows, predictions = [], []
    for split_name, folds in outer_splits(rows).items():
        all_y, all_p, all_low, all_high = [], [], [], []
        for fold, (train_outer, test) in enumerate(folds):
            groups = np.asarray([rows[index]["circuit_id"] for index in train_outer])
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed + fold)
            fit_local, calibration_local = next(splitter.split(train_outer, groups=groups))
            fit_index, calibration_index = train_outer[fit_local], train_outer[calibration_local]
            fitted = model(args.seed + fold)
            fitted.fit(x[fit_index], y[fit_index])
            calibration_error = np.abs(y[calibration_index] - fitted.predict(x[calibration_index]))
            quantile = float(np.quantile(calibration_error, args.coverage, method="higher"))
            predicted = fitted.predict(x[test])
            all_y.extend(y[test]); all_p.extend(predicted); all_low.extend(predicted - quantile); all_high.extend(predicted + quantile)
            for index, value, low, high in zip(test, predicted, predicted - quantile, predicted + quantile):
                predictions.append({
                    "split": split_name, "fold": fold, "circuit_id": rows[index]["circuit_id"], "family": rows[index]["family"],
                    "backend_class": rows[index]["backend_class"], "actual_seconds": float(np.expm1(y[index])),
                    "predicted_seconds": float(np.expm1(value)), "lower_seconds": float(max(0.0, np.expm1(low))),
                    "upper_seconds": float(np.expm1(high)), "conformal_log_radius": quantile,
                })
        actual, predicted, lower, upper = np.asarray(all_y), np.asarray(all_p), np.asarray(all_low), np.asarray(all_high)
        actual_seconds, predicted_seconds = np.expm1(actual), np.expm1(predicted)
        metric_rows.append({
            "split": split_name, "n": len(actual), "log_mae": float(mean_absolute_error(actual, predicted)),
            "log_r2": float(r2_score(actual, predicted)), "mae_seconds": float(mean_absolute_error(actual_seconds, predicted_seconds)),
            "r2_seconds": float(r2_score(actual_seconds, predicted_seconds)),
            "interval_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
            "mean_interval_width_seconds": float(np.mean(np.expm1(upper) - np.maximum(0.0, np.expm1(lower)))),
        })
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader(); writer.writerows(metric_rows)
    with args.output_prefix.with_name(args.output_prefix.name + "_oof_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    summary = {"target_semantics": "local noisy-Aer execution seconds after fake-backend transpilation; transpilation excluded",
               "n_successful_rows": len(rows), "features": FEATURES, "coverage_target": args.coverage, "splits": metric_rows,
               "censoring_rule": "resource_limit/timeout rows are not imputed as completed runtime labels"}
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    table = ["| Split | N | Log MAE | Log R2 | MAE seconds | R2 seconds | Interval coverage | Mean width s |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metric_rows:
        table.append(f"| {row['split']} | {row['n']} | {row['log_mae']:.3f} | {row['log_r2']:.3f} | {row['mae_seconds']:.3f} | {row['r2_seconds']:.3f} | {100*row['interval_coverage']:.1f}% | {row['mean_interval_width_seconds']:.3f} |")
    args.output_prefix.with_suffix(".md").write_text("\n".join([
        "# Noisy-Aer source-aware runtime intervals", "",
        "All point estimates and conformal radii are fitted within each outer training fold. The target excludes transpilation; it is not QPU runtime or cloud turnaround.", "", *table,
        "", "Backend-held-out metrics are the relevant warning for fake-backend domain transfer. Censored resource rows remain a separate feasibility boundary.", "",
    ]))
    print(args.output_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
