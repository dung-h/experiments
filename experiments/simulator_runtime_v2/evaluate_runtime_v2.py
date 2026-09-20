#!/usr/bin/env python3
"""Evaluate pre-execution runtime and context-selection baselines for v2.

The primary split holds *all contexts of a logical circuit* out together.
Thus a model cannot learn the CPU/GPU variant of a test circuit from training.
Random-row results remain only a diagnostic.  All targets are restricted to
successful prepared-execution rows; resource-limit rows are summarized
separately and never coerced into a duration.
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


TARGET = "warm_execution_median_seconds"
ANALYTICAL_NUMERIC = ["log2_statevector_bytes", "log2_analytical_gate_work"]
LOGICAL_NUMERIC = [
    "num_qubits",
    "depth_parameter",
    "logical_depth",
    "logical_total_gate_count",
    "logical_two_qubit_gate_count",
    "logical_single_qubit_gate_count",
    *ANALYTICAL_NUMERIC,
]
GRAPH_NUMERIC = [
    *LOGICAL_NUMERIC,
    "interaction_unique_edge_count",
    "interaction_density",
    "two_qubit_span_mean",
    "two_qubit_span_max",
    "qubit_activity_max",
    "qubit_activity_std",
    "two_qubit_degree_max",
    "two_qubit_degree_std",
]
CATEGORICAL = ["family", "context_id"]


def metric_dict(actual: np.ndarray, predicted_log: np.ndarray) -> dict[str, float]:
    actual_log = np.log(actual)
    predicted_seconds = np.exp(predicted_log)
    return {
        "mae_log": float(mean_absolute_error(actual_log, predicted_log)),
        "rmse_log": float(mean_squared_error(actual_log, predicted_log) ** 0.5),
        "r2_log": float(r2_score(actual_log, predicted_log)),
        "mae_seconds": float(mean_absolute_error(actual, predicted_seconds)),
        "rmse_seconds": float(mean_squared_error(actual, predicted_seconds) ** 0.5),
        "r2_seconds": float(r2_score(actual, predicted_seconds)),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    render = lambda row: "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
    return "\n".join([render(headers), render(["-" * width for width in widths]), *(render(row) for row in rows)])


def ridge_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("features", transformer), ("model", Ridge(alpha=1.0))])


def hgb_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformer = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], sparse_threshold=0.0)
    return Pipeline([
        ("features", transformer),
        ("model", HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.06, max_leaf_nodes=12,
            min_samples_leaf=6, random_state=42,
        )),
    ])


def models() -> tuple[tuple[str, Pipeline | None], ...]:
    return (
        ("median_log", None),
        ("analytical_calibrated_ridge", ridge_model(ANALYTICAL_NUMERIC, ["context_id"])),
        ("analytical_hgb", hgb_model(ANALYTICAL_NUMERIC, ["context_id"])),
        ("logical_hgb", hgb_model(LOGICAL_NUMERIC, ["context_id"])),
        ("graph_hgb", hgb_model(GRAPH_NUMERIC, ["context_id"])),
        ("graph_hgb_with_family", hgb_model(GRAPH_NUMERIC, CATEGORICAL)),
        ("structural_ridge", ridge_model(GRAPH_NUMERIC, CATEGORICAL)),
    )


def evaluate_partition(train: pd.DataFrame, test: pd.DataFrame, split: str, fold: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for name, model in models():
        if model is None:
            predicted_log = np.full(len(test), float(np.log(train[TARGET]).median()))
        else:
            model.fit(train, np.log(train[TARGET].to_numpy()))
            predicted_log = model.predict(test)
        row: dict[str, object] = {"split": split, "fold": fold, "model": name, "n_train": len(train), "n_test": len(test)}
        row.update(metric_dict(test[TARGET].to_numpy(), predicted_log))
        metric_rows.append(row)
        for (_, sample), prediction in zip(test.iterrows(), predicted_log):
            prediction_rows.append({
                "split": split,
                "fold": fold,
                "model": name,
                "corpus_record_id": sample["corpus_record_id"],
                "circuit_id": sample["circuit_id"],
                "family": sample["family"],
                "num_qubits": int(sample["num_qubits"]),
                "seed": int(sample["seed"]),
                "context_id": sample["context_id"],
                "actual_seconds": float(sample[TARGET]),
                "prediction_log": float(prediction),
                "prediction_seconds": float(np.exp(prediction)),
            })
    return metric_rows, prediction_rows


def selection_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure whether a model picks the shortest measured context for a test circuit."""
    rows: list[dict[str, object]] = []
    required_contexts = 4
    for (model, circuit_id), group in predictions.groupby(["model", "circuit_id"], sort=True):
        if group["context_id"].nunique() != required_contexts:
            continue
        oracle = group.loc[group["actual_seconds"].idxmin()]
        chosen = group.loc[group["prediction_seconds"].idxmin()]
        rows.append({
            "model": model,
            "circuit_id": circuit_id,
            "oracle_context": oracle["context_id"],
            "selected_context": chosen["context_id"],
            "oracle_seconds": float(oracle["actual_seconds"]),
            "selected_actual_seconds": float(chosen["actual_seconds"]),
            "selection_regret_seconds": float(chosen["actual_seconds"] - oracle["actual_seconds"]),
            "selection_regret_ratio": float(chosen["actual_seconds"] / oracle["actual_seconds"]),
            "exact_fastest": bool(chosen["context_id"] == oracle["context_id"]),
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return (
            pd.DataFrame(columns=["model", "complete_circuit_groups", "exact_fastest_rate", "median_regret_ratio", "mean_regret_seconds"]),
            detail,
        )
    aggregate = detail.groupby("model", as_index=False).agg(
        complete_circuit_groups=("circuit_id", "count"),
        exact_fastest_rate=("exact_fastest", "mean"),
        median_regret_ratio=("selection_regret_ratio", "median"),
        mean_regret_seconds=("selection_regret_seconds", "mean"),
    )
    return aggregate, detail


def resource_envelope(raw: pd.DataFrame) -> pd.DataFrame:
    gpu_ok = raw[(raw["execution_device"] == "cuda") & (raw["status"] == "ok")].copy()
    gpu_ok = gpu_ok[gpu_ok["torch_cuda_peak_reserved_bytes"].notna()]
    if gpu_ok.empty:
        return pd.DataFrame()
    gpu_ok["peak_reserved_over_statevector"] = (
        gpu_ok["torch_cuda_peak_reserved_bytes"].astype(float) / gpu_ok["statevector_bytes"].astype(float)
    )
    return gpu_ok.groupby(["context_id", "num_qubits"], as_index=False).agg(
        successful_rows=("record_id", "count"),
        max_peak_reserved_over_statevector=("peak_reserved_over_statevector", "max"),
        median_peak_reserved_over_statevector=("peak_reserved_over_statevector", "median"),
        q95_peak_reserved_over_statevector=("peak_reserved_over_statevector", lambda values: float(np.quantile(values, 0.95))),
        device_total_memory_bytes=("device_total_memory_bytes", "max"),
    )


def resource_frontier_prediction(raw: pd.DataFrame, calibration_max_width: int = 24) -> pd.DataFrame:
    """Test a peak-memory envelope calibrated only through a smaller width.

    This deliberately differs from the naive lower-bound check.  A final
    statevector that fits in VRAM can still fail when direct gate application
    creates a temporary tensor.  The output is an out-of-sample frontier table
    once q26/q28 observations are present.
    """
    gpu = raw[raw["execution_device"] == "cuda"].copy()
    if gpu.empty or "device_total_memory_bytes" not in gpu:
        return pd.DataFrame()
    successful = gpu[(gpu["status"] == "ok") & gpu["torch_cuda_peak_reserved_bytes"].notna()].copy()
    successful["peak_ratio"] = (
        successful["torch_cuda_peak_reserved_bytes"].astype(float) / successful["statevector_bytes"].astype(float)
    )
    calibration = successful[successful["num_qubits"] <= calibration_max_width]
    calibration_rows: list[dict[str, object]] = []
    for context_id, group in calibration.groupby("context_id"):
        largest = int(group["num_qubits"].max())
        at_largest = group[group["num_qubits"] == largest]
        calibration_rows.append({
            "context_id": context_id,
            "calibration_largest_width": largest,
            "calibration_peak_ratio": float(at_largest["peak_ratio"].max()),
        })
    calibration_frame = pd.DataFrame(calibration_rows)
    frontier = gpu[gpu["num_qubits"] > calibration_max_width].copy()
    if frontier.empty or calibration_frame.empty:
        return pd.DataFrame()
    grouped = frontier.groupby(["context_id", "num_qubits", "statevector_bytes", "device_total_memory_bytes"], as_index=False).agg(
        observed_rows=("status", "count"),
        observed_ok=("status", lambda status: int((status == "ok").sum())),
        observed_resource_limit=("status", lambda status: int((status == "resource_limit").sum())),
        observed_error=("status", lambda status: int((status == "error").sum())),
    )
    grouped = grouped.merge(calibration_frame, on="context_id", how="left", validate="many_to_one")
    grouped["raw_statevector_fits"] = grouped["statevector_bytes"] <= grouped["device_total_memory_bytes"]
    grouped["envelope_peak_bytes"] = grouped["statevector_bytes"] * grouped["calibration_peak_ratio"]
    grouped["envelope_fits"] = grouped["envelope_peak_bytes"] <= grouped["device_total_memory_bytes"]
    return grouped.sort_values(["context_id", "num_qubits"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    required = {TARGET, "status", "circuit_id", "corpus_record_id", *GRAPH_NUMERIC, *CATEGORICAL}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"input missing: {missing}")
    data = raw[(raw["status"] == "ok") & raw[TARGET].notna()].copy()
    data = data[np.isfinite(data[TARGET]) & (data[TARGET] > 0)].copy()
    if data["circuit_id"].nunique() < 8:
        raise ValueError("need at least eight successful logical circuits")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []

    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.25, random_state=42)
    rows, predictions = evaluate_partition(data.iloc[train_idx], data.iloc[test_idx], "random_row", "42")
    all_metrics.extend(rows); all_predictions.extend(predictions)

    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=min(5, data["circuit_id"].nunique())).split(data, groups=data["circuit_id"]), 1):
        rows, predictions = evaluate_partition(data.iloc[train_idx], data.iloc[test_idx], "circuit_group", str(fold))
        all_metrics.extend(rows); all_predictions.extend(predictions)

    for split, column in (("family_held_out", "family"), ("width_held_out", "num_qubits")):
        if data[column].nunique() > 1:
            for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(data, groups=data[column]), 1):
                rows, predictions = evaluate_partition(data.iloc[train_idx], data.iloc[test_idx], split, str(fold))
                all_metrics.extend(rows); all_predictions.extend(predictions)

    # Seeds 43/101 are deliberately GPU-only to contain experiment cost.
    # Restrict this split to GPU so every held seed sees the same contexts.
    structural = data[
        (data["source_run"] == "v2_structural")
        & (data["execution_device"] == "cuda")
        & (data["num_qubits"] <= 24)
    ].copy()
    if structural["seed"].nunique() > 1 and structural["circuit_id"].nunique() >= 8:
        for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(structural, groups=structural["seed"]), 1):
            rows, predictions = evaluate_partition(structural.iloc[train_idx], structural.iloc[test_idx], "seed_held_out_structural", str(fold))
            all_metrics.extend(rows); all_predictions.extend(predictions)

    per_fold = pd.DataFrame(all_metrics)
    predictions = pd.DataFrame(all_predictions)
    aggregate_rows: list[dict[str, object]] = []
    for (split, model), group in predictions.groupby(["split", "model"], sort=True):
        row: dict[str, object] = {"split": split, "model": model, "folds": group["fold"].nunique(), "n_test": len(group)}
        row.update(metric_dict(group["actual_seconds"].to_numpy(), group["prediction_log"].to_numpy()))
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows).sort_values(["split", "model"])
    circuit_predictions = predictions[predictions["split"] == "circuit_group"].copy()
    selection, selection_detail = selection_metrics(circuit_predictions)
    oracle_distribution = (
        selection_detail.drop_duplicates("circuit_id").groupby("oracle_context", as_index=False).size().rename(columns={"size": "circuit_groups"})
        if not selection_detail.empty else pd.DataFrame(columns=["oracle_context", "circuit_groups"])
    )
    envelope = resource_envelope(raw)
    frontier = resource_frontier_prediction(raw)
    per_fold.to_csv(out / "per_fold_metrics.csv", index=False)
    predictions.to_csv(out / "oof_predictions.csv", index=False)
    aggregate.to_csv(out / "aggregate_metrics.csv", index=False)
    selection.to_csv(out / "context_selection_metrics.csv", index=False)
    selection_detail.to_csv(out / "context_selection_detail.csv", index=False)
    oracle_distribution.to_csv(out / "context_selection_oracle_distribution.csv", index=False)
    envelope.to_csv(out / "resource_envelope.csv", index=False)
    frontier.to_csv(out / "resource_frontier_prediction.csv", index=False)
    report = [
        "# Dense-statevector v2 estimator evaluation",
        "",
        "The target is local prepared dense-statevector execution. Resource-limit observations are retained in the raw corpus but excluded from duration regression.",
        "",
        f"Successful duration rows: {len(data)}; logical circuits: {data['circuit_id'].nunique()}; source runs: {data['source_run'].value_counts().to_dict()}.",
        "",
        "Random-row scores are diagnostic only. The primary `circuit_group` split keeps every CPU/GPU/precision context of a logical test circuit outside training.",
        "The optional `seed_held_out_structural` split is GPU-only and restricted through q24 because seeds 43/101 were intentionally collected only for the two GPU contexts at those widths; it tests unseen random topologies under matched coverage.",
        "",
        "## Runtime prediction",
        "",
        markdown_table(aggregate),
        "",
        "The ablation is cumulative: `analytical_hgb` tests nonlinear capacity using only statevector bytes and gate-work; `logical_hgb` adds depth/gate counts; `graph_hgb` adds interaction-graph features; `graph_hgb_with_family` additionally exposes the coarse family label. These are calibrated estimators for this one PyTorch kernel, not universal simulator predictors.",
        "",
        "## Context-selection task",
        "",
        markdown_table(selection),
        "",
        "Oracle fastest-context distribution:",
        "",
        markdown_table(oracle_distribution),
        "",
        "This task considers only complete four-context circuit groups held out by `circuit_group`. A low regret is useful only if the fastest context varies; a single dominating oracle context makes this a hardware fact rather than a model contribution.",
        "",
        "## GPU memory envelope",
        "",
        markdown_table(envelope),
        "",
        "The envelope is empirical and context-specific. Raw statevector bytes are a lower bound, not a peak-allocation guarantee because the gate kernel creates temporary tensors.",
        "",
        "## Resource-frontier prediction",
        "",
        markdown_table(frontier),
        "",
        "This table calibrates a peak-allocation ratio only through q24, then applies it to wider actual observations. `raw_statevector_fits` is the naive lower-bound decision; `envelope_fits` is the context-calibrated decision.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = {
        "target": TARGET,
        "successful_duration_rows": int(len(data)),
        "resource_limit_rows": int((raw["status"] == "resource_limit").sum()),
        "aggregate_metrics": aggregate.to_dict(orient="records"),
        "context_selection": selection.to_dict(orient="records"),
        "oracle_context_distribution": oracle_distribution.to_dict(orient="records"),
        "resource_envelope": envelope.to_dict(orient="records"),
        "resource_frontier_prediction": frontier.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print((out / "REPORT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
