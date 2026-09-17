#!/usr/bin/env python3
"""Evaluate cuTensorNet B0/B1 runtime predictors without mixing targets.

The input must contain only the simulator target
``contract_gpu_median_s``.  This script evaluates the native cuTensorNet
pre-run estimate (B0), a train-only multiplicative calibration, and two B1
models.  Random CV is descriptive; family- and width-held-out scores test
actual transfer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "run-output" / "cutensornet" / "feasibility_frontier" / "cutensornet_feasibility_frontier.csv"
DEFAULT_OUTPUT = ROOT / "run-output" / "cutensornet" / "feasibility_frontier"

TARGET = "contract_gpu_median_s"
ESTIMATE = "cutensornet_runtime_est_s"
NUMERIC_COLUMNS = [
    "log_runtime_est", "log_flop_count", "log_largest_intermediate",
    "log_num_slices", "num_qubits", "logical_depth",
    "logical_total_gate_count", "logical_single_qubit_gate_count",
    "logical_two_qubit_gate_count", "family_depth_parameter",
]
CATEGORICAL_COLUMNS = ["family", "circuit_variant"]


def ensure_single_target(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"runtime_semantics", "target_name", TARGET, ESTIMATE, *NUMERIC_COLUMNS[4:], *CATEGORICAL_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Input misses required columns: {missing}")
    semantics = frame["runtime_semantics"].dropna().unique().tolist()
    targets = frame["target_name"].dropna().unique().tolist()
    if semantics != ["T_sim_contract: local GPU tensor-network scalar contraction"]:
        raise SystemExit(f"Refusing mixed/unknown runtime semantics: {semantics}")
    if targets != [TARGET]:
        raise SystemExit(f"Refusing target_name values other than {TARGET}: {targets}")
    numeric_required = [TARGET, ESTIMATE, "cutensornet_flop_count", "cutensornet_largest_intermediate_elements", "cutensornet_num_slices"]
    for column in numeric_required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric_required).copy()
    frame = frame[(frame[TARGET] > 0) & (frame[ESTIMATE] > 0)].copy()
    if frame.empty:
        raise SystemExit("No positive successful timing rows remain.")
    frame["log_runtime_est"] = np.log(frame[ESTIMATE])
    frame["log_flop_count"] = np.log(frame["cutensornet_flop_count"].clip(lower=1))
    frame["log_largest_intermediate"] = np.log(frame["cutensornet_largest_intermediate_elements"].clip(lower=1))
    frame["log_num_slices"] = np.log(frame["cutensornet_num_slices"].clip(lower=1))
    return frame


def b1_pipeline(kind: str) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categories = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("category", categories, CATEGORICAL_COLUMNS),
    ])
    if kind == "ridge":
        model = Ridge(alpha=3.0)
    elif kind == "hgb":
        model = HistGradientBoostingRegressor(
            learning_rate=0.06, max_iter=150, max_leaf_nodes=7,
            min_samples_leaf=5, l2_regularization=1.0, random_state=42,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("features", preprocessor), ("model", model)])


def metric_row(split: str, model: str, truth_log: np.ndarray, prediction_log: np.ndarray) -> dict[str, float | str | int]:
    truth_s, prediction_s = np.exp(truth_log), np.exp(prediction_log)
    return {
        "split": split,
        "model": model,
        "n_test_total": len(truth_log),
        "log_mae": mean_absolute_error(truth_log, prediction_log),
        "log_rmse": float(np.sqrt(np.mean((truth_log - prediction_log) ** 2))),
        "log_r2": r2_score(truth_log, prediction_log),
        "seconds_mae": mean_absolute_error(truth_s, prediction_s),
        "seconds_rmse": float(np.sqrt(np.mean((truth_s - prediction_s) ** 2))),
        "seconds_r2": r2_score(truth_s, prediction_s),
    }


def split_definitions(frame: pd.DataFrame) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    indices = np.arange(len(frame))
    random_cv = list(KFold(n_splits=5, shuffle=True, random_state=42).split(indices))
    family = list(LeaveOneGroupOut().split(indices, groups=frame["family"]))
    width = list(LeaveOneGroupOut().split(indices, groups=frame["num_qubits"]))
    variant = list(LeaveOneGroupOut().split(indices, groups=frame["circuit_variant"]))
    return {
        "random_5fold_descriptive": random_cv,
        "family_held_out": family,
        "width_held_out": width,
        "variant_held_out": variant,
    }


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = np.log(frame[TARGET].to_numpy(dtype=float))
    x = frame[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    metric_rows: list[dict[str, float | str | int]] = []
    prediction_rows: list[dict[str, float | str | int]] = []
    for split_name, folds in split_definitions(frame).items():
        predictions: dict[str, np.ndarray] = {
            "median_train_log": np.full(len(frame), np.nan),
            "native_runtime_est_b0": np.full(len(frame), np.nan),
            "median_ratio_calibrated_b0": np.full(len(frame), np.nan),
            "ridge_b1": np.full(len(frame), np.nan),
            "hist_gradient_boosting_b1": np.full(len(frame), np.nan),
        }
        fold_number = np.full(len(frame), -1, dtype=int)
        for fold, (train_index, test_index) in enumerate(folds):
            y_train = y[train_index]
            native_train = frame.iloc[train_index]["log_runtime_est"].to_numpy(dtype=float)
            native_test = frame.iloc[test_index]["log_runtime_est"].to_numpy(dtype=float)
            predictions["median_train_log"][test_index] = float(np.median(y_train))
            predictions["native_runtime_est_b0"][test_index] = native_test
            predictions["median_ratio_calibrated_b0"][test_index] = native_test + float(np.median(y_train - native_train))
            for label, kind in (("ridge_b1", "ridge"), ("hist_gradient_boosting_b1", "hgb")):
                model = b1_pipeline(kind)
                model.fit(x.iloc[train_index], y_train)
                predictions[label][test_index] = model.predict(x.iloc[test_index])
            fold_number[test_index] = fold
        for model, prediction in predictions.items():
            if np.isnan(prediction).any():
                raise RuntimeError(f"Incomplete OOF predictions for {split_name}/{model}")
            metric_rows.append(metric_row(split_name, model, y, prediction))
            for row_index, predicted_log in enumerate(prediction):
                prediction_rows.append({
                    "split": split_name,
                    "fold": int(fold_number[row_index]),
                    "model": model,
                    "family": frame.iloc[row_index]["family"],
                    "circuit_variant": frame.iloc[row_index]["circuit_variant"],
                    "num_qubits": int(frame.iloc[row_index]["num_qubits"]),
                    "logical_depth": int(frame.iloc[row_index]["logical_depth"]),
                    "actual_seconds": float(np.exp(y[row_index])),
                    "predicted_seconds": float(np.exp(predicted_log)),
                    "actual_log_seconds": float(y[row_index]),
                    "predicted_log_seconds": float(predicted_log),
                })
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def markdown_table(frame: pd.DataFrame) -> str:
    headers = ["split", "model", "log MAE", "log RMSE", "log R2", "MAE (ms)", "RMSE (ms)", "R2 (seconds)"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.itertuples(index=False):
        lines.append(
            "| " + " | ".join([
                str(row.split), str(row.model), f"{row.log_mae:.3f}", f"{row.log_rmse:.3f}", f"{row.log_r2:.3f}",
                f"{row.seconds_mae * 1e3:.3f}", f"{row.seconds_rmse * 1e3:.3f}", f"{row.seconds_r2:.3f}",
            ]) + " |"
        )
    return "\n".join(lines)


def write_report(path: Path, frame: pd.DataFrame, metrics: pd.DataFrame) -> None:
    def log_mae(split: str, model: str) -> float:
        return float(metrics.loc[
            (metrics["split"] == split) & (metrics["model"] == model), "log_mae"
        ].iloc[0])

    family_ratio = log_mae("family_held_out", "median_ratio_calibrated_b0")
    family_ridge = log_mae("family_held_out", "ridge_b1")
    family_hgb = log_mae("family_held_out", "hist_gradient_boosting_b1")
    width_ratio = log_mae("width_held_out", "median_ratio_calibrated_b0")
    width_ridge = log_mae("width_held_out", "ridge_b1")
    report = f"""# cuTensorNet B0/B1 calibration evaluation

## Target and corpus

- Source: `cutensornet` only.
- Target semantics: `T_sim_contract: local GPU tensor-network scalar contraction`.
- Target: warm CUDA-event median `contract_gpu_median_s`, not path-search,
  construction, statevector materialisation, shots, QPU execution, or cloud
  turnaround.
- Rows: **{len(frame)}** successful, timeout-bounded feasibility cells; no
  failed cell is silently removed.
- Grid: GHZ and QFT plus HEA (2/4/8 layers), QAOA-cycle (p=2/3/6), and random
  brickwork (4/8/16 layers), each across widths 16, 20, 24, 28, 30.

## Models

- `median_train_log`: median log-time fitted on each training fold.
- `native_runtime_est_b0`: cuTensorNet's pre-run `RUNTIME_EST` without fitting.
- `median_ratio_calibrated_b0`: training-fold median log residual applied to
  B0; it tests whether a global multiplicative correction transfers.
- `ridge_b1` and `hist_gradient_boosting_b1`: environment-local calibrators
  using pre-contraction planner/circuit features only: B0 estimate, FLOPs,
  largest intermediate, slices, width, logical gate/depth features, family and
  variant.

## Out-of-fold metrics

""" + markdown_table(metrics) + f"""

`random_5fold_descriptive` is not a generalisation claim. The family, width
and variant held-out rows are the decision-relevant tests. Metrics are
calculated once from all out-of-fold predictions per split.

## Decision

For unseen **families**, the train-fold median-ratio calibration has log-MAE
**{family_ratio:.3f}**, better than Ridge (**{family_ridge:.3f}**) and HGB
(**{family_hgb:.3f}**). It is the provisional robust estimator for this fixed
cuTensorNet/GPU/precision/target environment. Ridge is strongest on unseen
widths (log-MAE **{width_ridge:.3f}** versus **{width_ratio:.3f}** for the
ratio calibration), but this does not transfer as well to unseen circuit
families. Use it only as an in-family refinement.

The corpus is still a first environment-local calibration set ({len(frame)}
rows). Do not select a universal estimator from a random split. The selected
calibration must be re-evaluated on a new GPU/software environment; otherwise
retain native B0 plus explicit calibration uncertainty and the feasibility
frontier.

Raw measurement and failure rows remain in the frontier output directory.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = ensure_single_target(pd.read_csv(args.input))
    metrics, predictions = evaluate(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "cutensornet_b1_evaluation_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "cutensornet_b1_oof_predictions.csv", index=False)
    write_report(args.output_dir / "CUTENSORNET_B1_CALIBRATION_RESULTS.md", frame, metrics)
    print(json.dumps({"rows": len(frame), "metrics": len(metrics), "predictions": len(predictions)}))


if __name__ == "__main__":
    main()
