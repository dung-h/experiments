#!/usr/bin/env python3
"""Evaluate static CUDA-Q MPS runtime baselines with leakage-safe splits.

Only features known before execution are used.  Returned MPS tensor extents,
observed bond dimension, tensor bytes and dense-reference fidelity are kept
out of the runtime feature block; they are post-run diagnostics or labels.
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
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = [
    "width",
    "logical_depth",
    "gate_count",
    "two_qubit_gates",
    "statevector_bytes",
    "mps_max_bond_config",
]
CATEGORICAL = ["family"]


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def scores(y_log: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    actual = np.expm1(y_log)
    pred = np.maximum(0.0, np.expm1(pred_log))
    return {
        "log_mae": float(mean_absolute_error(y_log, pred_log)),
        "log_rmse": float(mean_squared_error(y_log, pred_log) ** 0.5),
        "log_r2": float(r2_score(y_log, pred_log)) if len(y_log) > 1 else float("nan"),
        "seconds_mae": float(mean_absolute_error(actual, pred)),
        "seconds_rmse": float(mean_squared_error(actual, pred) ** 0.5),
        "seconds_r2": float(r2_score(actual, pred)) if len(actual) > 1 else float("nan"),
    }


def splits(df: pd.DataFrame, name: str):
    if name == "grouped_circuit":
        groups = (df["family"].astype(str) + ":" + df["width"].astype(str)).to_numpy()
        return GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(df, groups=groups)
    if name == "family_held_out":
        groups = df["family"].astype(str).to_numpy()
    elif name == "cap_held_out":
        groups = df["mps_max_bond_config"].astype(str).to_numpy()
    elif name == "width_held_out":
        groups = df["width"].astype(str).to_numpy()
    else:
        raise ValueError(name)
    return LeaveOneGroupOut().split(df, groups=groups)


def predict(train: pd.DataFrame, test: pd.DataFrame, target: str, model_name: str) -> np.ndarray:
    y = np.log1p(train[target].to_numpy(dtype=float))
    if model_name == "ridge":
        model = Ridge(alpha=1.0)
    elif model_name == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.05, max_leaf_nodes=7,
            l2_regularization=1.0, random_state=7,
        )
    else:
        raise ValueError(model_name)
    pipe = Pipeline([("features", make_preprocessor()), ("model", model)])
    pipe.fit(train[NUMERIC + CATEGORICAL], y)
    return pipe.predict(test[NUMERIC + CATEGORICAL])


def evaluate(df: pd.DataFrame, target: str, split_name: str) -> list[dict]:
    y = np.log1p(df[target].to_numpy(dtype=float))
    out = []
    for fold, (train_idx, test_idx) in enumerate(splits(df, split_name)):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        baseline = np.full(len(test), np.median(y[train_idx]))
        out.append({
            "target": target, "split": split_name, "model": "median_log",
            "fold": fold, "n_train": len(train), "n_test": len(test),
            **scores(y[test_idx], baseline),
        })
        for model in ("ridge", "hgb"):
            try:
                pred = predict(train, test, target, model)
                out.append({
                    "target": target, "split": split_name, "model": model,
                    "fold": fold, "n_train": len(train), "n_test": len(test),
                    **scores(y[test_idx], pred),
                })
            except Exception as exc:
                out.append({
                    "target": target, "split": split_name, "model": model,
                    "fold": fold, "n_train": len(train), "n_test": len(test),
                    "error": f"{type(exc).__name__}: {exc}",
                })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    raw = raw.loc[raw["status"].eq("ok")].copy()
    for col in ("state_warm_median_s", "sample_warm_median_s"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.loc[np.isfinite(raw["state_warm_median_s"]) & np.isfinite(raw["sample_warm_median_s"])].reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    for target in ("state_warm_median_s", "sample_warm_median_s"):
        for split_name in ("grouped_circuit", "family_held_out", "cap_held_out", "width_held_out"):
            metrics.extend(evaluate(raw, target, split_name))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(args.output_dir / "mps_runtime_metrics.csv", index=False)
    valid = metrics_df.loc[metrics_df.get("error").isna() if "error" in metrics_df else np.ones(len(metrics_df), dtype=bool)]
    summary = (
        valid.groupby(["target", "split", "model"], as_index=False)
        .agg({"log_mae": "mean", "log_rmse": "mean", "log_r2": "mean", "seconds_mae": "mean", "seconds_rmse": "mean", "seconds_r2": "mean", "fold": "count"})
        .rename(columns={"fold": "folds"})
    )
    summary.to_csv(args.output_dir / "mps_runtime_summary.csv", index=False)
    manifest = {
        "rows_used": int(len(raw)),
        "targets": ["state_warm_median_s", "sample_warm_median_s"],
        "splits": ["grouped_circuit", "family_held_out", "cap_held_out", "width_held_out"],
        "models": ["median_log", "ridge", "hgb"],
        "pre_run_features": NUMERIC + CATEGORICAL,
        "excluded_post_run_fields": ["observed_max_bond", "mps_tensor_bytes", "mps_tensor_elements", "fidelity_vs_dense"],
        "warning": "48-row controlled MPS pilot; metrics are diagnostics, not a portable estimator claim",
    }
    (args.output_dir / "mps_runtime_evaluation.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
