#!/usr/bin/env python3
"""Evaluate source-aware baselines for the dense-statevector v1 matrix.

Only successful rows from one simulator contract are used.  Random-row scores
are reported as a leakage-prone diagnostic; circuit, family and execution-
context holdouts are the relevant stress tests.  The script does not fit or
serialize a deployment model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_NUMERIC = [
    "num_qubits",
    "depth_parameter",
    "logical_depth",
    "logical_total_gate_count",
    "logical_two_qubit_gate_count",
    "logical_single_qubit_gate_count",
]
FEATURE_CATEGORICAL = ["family", "execution_device", "precision"]
TARGET = "warm_execution_median_seconds"


def metrics(y_true: np.ndarray, y_log_pred: np.ndarray) -> dict[str, float]:
    y_log = np.log(y_true)
    y_pred = np.exp(y_log_pred)
    return {
        "mae_log": float(mean_absolute_error(y_log, y_log_pred)),
        "rmse_log": float(mean_squared_error(y_log, y_log_pred) ** 0.5),
        "r2_log": float(r2_score(y_log, y_log_pred)),
        "mae_seconds": float(mean_absolute_error(y_true, y_pred)),
        "rmse_seconds": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2_seconds": float(r2_score(y_true, y_pred)),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact table without depending on the optional tabulate package."""
    headers = [str(column) for column in frame.columns]
    rendered: list[list[str]] = []
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rendered.append(values)
    widths = [max(len(header), *(len(row[index]) for row in rendered)) for index, header in enumerate(headers)]
    render_row = lambda values: "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rendered)])


def ridge_model() -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline([("scale", StandardScaler())]), FEATURE_NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), FEATURE_CATEGORICAL),
        ]
    )
    return Pipeline([("features", preprocess), ("model", Ridge(alpha=1.0))])


def hgb_model() -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("numeric", "passthrough", FEATURE_NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURE_CATEGORICAL),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline([
        ("features", preprocess),
        ("model", HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, max_leaf_nodes=10, min_samples_leaf=5, random_state=42)),
    ])


def predict_median(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.full(len(test), float(np.log(train[TARGET]).median()))


def evaluate_partition(
    train: pd.DataFrame, test: pd.DataFrame, split: str, fold: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for name, model in (("median", None), ("ridge_log", ridge_model()), ("hist_gradient_log", hgb_model())):
        if model is None:
            prediction = predict_median(train, test)
        else:
            model.fit(train, np.log(train[TARGET].to_numpy()))
            prediction = model.predict(test)
        row: dict[str, object] = {
            "split": split,
            "fold": fold,
            "model": name,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        }
        row.update(metrics(test[TARGET].to_numpy(), prediction))
        rows.append(row)
        for (_, sample), log_prediction in zip(test.iterrows(), prediction):
            predictions.append({
                "split": split,
                "fold": fold,
                "model": name,
                "record_id": sample["record_id"],
                "circuit_id": sample["circuit_id"],
                "family": sample["family"],
                "context_id": sample["context_id"],
                "actual_seconds": float(sample[TARGET]),
                "prediction_log": float(log_prediction),
                "prediction_seconds": float(np.exp(log_prediction)),
            })
    return rows, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    data = raw[(raw["status"] == "ok") & raw[TARGET].notna()].copy()
    data = data[np.isfinite(data[TARGET]) & (data[TARGET] > 0)].copy()
    if len(data) < 20:
        raise ValueError(f"Need at least 20 successful rows; found {len(data)}")
    missing = [field for field in FEATURE_NUMERIC + FEATURE_CATEGORICAL + ["circuit_id"] if field not in data]
    if missing:
        raise ValueError(f"Input missing required fields: {missing}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.25, random_state=42)
    partition_rows, partition_predictions = evaluate_partition(
        data.iloc[train_idx], data.iloc[test_idx], "random_row", "42"
    )
    rows.extend(partition_rows)
    prediction_rows.extend(partition_predictions)

    circuit_folds = GroupKFold(n_splits=4)
    for fold_index, (train_idx, test_idx) in enumerate(circuit_folds.split(data, groups=data["circuit_id"]), start=1):
        partition_rows, partition_predictions = evaluate_partition(
            data.iloc[train_idx], data.iloc[test_idx], "circuit_group", str(fold_index)
        )
        rows.extend(partition_rows)
        prediction_rows.extend(partition_predictions)

    for split_name, column in (
        ("family_held_out", "family"),
        ("width_held_out", "num_qubits"),
        ("context_held_out", "context_id"),
    ):
        logo = LeaveOneGroupOut()
        for fold_index, (train_idx, test_idx) in enumerate(logo.split(data, groups=data[column]), start=1):
            partition_rows, partition_predictions = evaluate_partition(
                data.iloc[train_idx], data.iloc[test_idx], split_name, str(fold_index)
            )
            rows.extend(partition_rows)
            prediction_rows.extend(partition_predictions)

    metric_rows = pd.DataFrame(rows)
    metric_rows.to_csv(output_dir / "per_fold_metrics.csv", index=False)
    prediction_frame = pd.DataFrame(prediction_rows)
    prediction_frame.to_csv(output_dir / "oof_predictions.csv", index=False)
    aggregate_rows: list[dict[str, object]] = []
    for (split, model), group in prediction_frame.groupby(["split", "model"], sort=True):
        aggregate_row: dict[str, object] = {
            "split": split,
            "model": model,
            "folds": int(group["fold"].nunique()),
            "n_test": int(len(group)),
        }
        aggregate_row.update(metrics(group["actual_seconds"].to_numpy(), group["prediction_log"].to_numpy()))
        aggregate_rows.append(aggregate_row)
    aggregate = pd.DataFrame(aggregate_rows).sort_values(["split", "model"])
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    coverage = (
        data.groupby(["family", "execution_device", "precision"], as_index=False)
        .agg(rows=("record_id", "count"), median_seconds=(TARGET, "median"), max_seconds=(TARGET, "max"))
        .sort_values(["family", "execution_device", "precision"])
    )
    coverage.to_csv(output_dir / "coverage.csv", index=False)

    report = [
        "# Dense-statevector v1 baseline evaluation",
        "",
        "The target is prepared dense-statevector execution only. It excludes construction, gate materialization, compilation/transpilation and sampling.",
        "",
        f"Successful rows: {len(data)}; logical circuits: {data['circuit_id'].nunique()}; contexts: {data['context_id'].nunique()}.",
        "",
        "Random-row evaluation is descriptive only because the same circuit may appear under another context. Circuit-group, family-held-out, width-held-out and context-held-out results are retained as stress tests.",
        "",
        markdown_table(aggregate),
        "",
        "A context-held-out score cannot establish cross-device transfer: this matrix has one physical GPU and one CPU. It tests whether the declared device/precision features are sufficient to extrapolate to an unseen execution context.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = {
        "target": TARGET,
        "successful_rows": int(len(data)),
        "logical_circuits": int(data["circuit_id"].nunique()),
        "contexts": sorted(data["context_id"].unique().tolist()),
        "aggregate_metrics": aggregate.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print((output_dir / "REPORT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
