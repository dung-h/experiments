#!/usr/bin/env python3
"""Evaluate small, target-specific CUDA-Q runtime baselines.

The matrix is deliberately evaluated per simulator target.  A CPU QPP timing,
GPU FP32 timing and GPU FP64 timing are different labels; this evaluator does
not hide that difference behind a universal ``runtime_seconds`` column.
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
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = [
    "width",
    "logical_depth",
    "gate_count",
    "two_qubit_gates",
    "statevector_bytes",
]
CATEGORICAL = ["family"]


def score(y_log: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    y = np.expm1(y_log)
    pred = np.maximum(0.0, np.expm1(pred_log))
    return {
        "log_mae": float(mean_absolute_error(y_log, pred_log)),
        "log_rmse": float(mean_squared_error(y_log, pred_log) ** 0.5),
        "log_r2": float(r2_score(y_log, pred_log)) if len(y_log) > 1 else float("nan"),
        "seconds_mae": float(mean_absolute_error(y, pred)),
        "seconds_rmse": float(mean_squared_error(y, pred) ** 0.5),
        "seconds_r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
    }


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        remainder="drop",
    )


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, model_name: str) -> np.ndarray:
    y = np.log1p(train["target_seconds"].to_numpy(dtype=float))
    if model_name == "ridge":
        estimator = Ridge(alpha=1.0)
    elif model_name == "hgb":
        estimator = HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0,
            random_state=7,
        )
    else:
        raise ValueError(model_name)
    pipe = Pipeline([("features", preprocessor()), ("model", estimator)])
    pipe.fit(train[NUMERIC + CATEGORICAL], y)
    return pipe.predict(test[NUMERIC + CATEGORICAL])


def evaluate_split(df: pd.DataFrame, target_name: str, split_name: str, groups: np.ndarray, models: list[str]) -> list[dict]:
    rows: list[dict] = []
    y_all = np.log1p(df["target_seconds"].to_numpy(dtype=float))
    if split_name == "random_5fold":
        splitter = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=7)
        splits = splitter.split(df)
    else:
        splitter = LeaveOneGroupOut()
        splits = splitter.split(df, groups=groups)

    for fold, (train_idx, test_idx) in enumerate(splits):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        baseline = np.full(len(test), np.median(y_all[train_idx]))
        metric = score(y_all[test_idx], baseline)
        rows.append({"target_name": str(train["target_name"].iloc[0]), "split": split_name, "model": "median_log", "fold": fold, "n_train": len(train), "n_test": len(test), **metric})
        for model in models:
            try:
                pred = fit_predict(train, test, model)
                metric = score(y_all[test_idx], pred)
                rows.append({"target_name": str(train["target_name"].iloc[0]), "split": split_name, "model": model, "fold": fold, "n_train": len(train), "n_test": len(test), **metric})
            except Exception as exc:
                rows.append({"target_name": str(train["target_name"].iloc[0]), "split": split_name, "model": model, "fold": fold, "n_train": len(train), "n_test": len(test), "error": f"{type(exc).__name__}: {exc}"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="warm_sample_median_s", choices=["warm_sample_median_s", "first_sample_s"])
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    raw = raw.loc[raw["status"].eq("ok")].copy()
    raw["target_seconds"] = pd.to_numeric(raw[args.target], errors="coerce")
    raw = raw.loc[np.isfinite(raw["target_seconds"]) & (raw["target_seconds"] > 0)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[dict] = []
    for target_name, target_df in raw.groupby("target_name", sort=True):
        target_df = target_df.reset_index(drop=True)
        if len(target_df) < 4:
            continue
        target_df["family_group"] = target_df["family"].astype(str)
        target_df["width_group"] = target_df["width"].astype(str)
        metrics.extend(evaluate_split(target_df, target_name, "random_5fold", target_df.index.to_numpy(), ["ridge", "hgb"]))
        metrics.extend(evaluate_split(target_df, target_name, "family_held_out", target_df["family_group"].to_numpy(), ["ridge", "hgb"]))
        metrics.extend(evaluate_split(target_df, target_name, "width_held_out", target_df["width_group"].to_numpy(), ["ridge", "hgb"]))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(args.output_dir / f"metrics_{args.target}.csv", index=False)
    summary = (
        metrics_df.loc[metrics_df.get("error").isna() if "error" in metrics_df else np.ones(len(metrics_df), dtype=bool)]
        .groupby(["target_name", "split", "model"], as_index=False)
        .agg({"log_mae": "mean", "log_rmse": "mean", "log_r2": "mean", "seconds_mae": "mean", "seconds_rmse": "mean", "seconds_r2": "mean", "fold": "count"})
        .rename(columns={"fold": "folds"})
    )
    summary.to_csv(args.output_dir / f"summary_{args.target}.csv", index=False)
    manifest = {
        "target_column": args.target,
        "target_semantics": "CUDA-Q local simulator sample wall-clock; target-specific; first and warm calls are separate",
        "input": str(args.input),
        "rows_used": int(len(raw)),
        "targets": sorted(raw["target_name"].unique().tolist()),
        "splits": ["random_5fold", "family_held_out", "width_held_out"],
        "models": ["median_log", "ridge", "hgb"],
        "warning": "small pilot; width/family held-out metrics are diagnostics, not portable cross-framework claims",
    }
    (args.output_dir / f"evaluation_{args.target}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
