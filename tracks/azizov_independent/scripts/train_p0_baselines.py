#!/usr/bin/env python3
"""Fit small grouped baselines on P0 to validate the modeling interface."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


NUMERIC = [
    "logical_width", "logical_depth", "logical_ops", "logical_two_qubit_ops",
    "physical_width", "physical_depth", "physical_ops", "physical_two_qubit_ops",
    "swap_count", "dag_node_count", "optimization_level",
]


def row_vector(row):
    values = [float(row.get(k) or 0.0) for k in NUMERIC]
    values.append(1.0 if row["backend_class"] == "FakeSherbrooke" else 0.0)
    return values


def metrics(y_log, pred_log, y_seconds):
    pred_seconds = np.maximum(0.0, np.expm1(pred_log))
    return {
        "r2_log": float(r2_score(y_log, pred_log)),
        "rmse_seconds": float(mean_squared_error(y_seconds, pred_seconds) ** 0.5),
        "mae_seconds": float(mean_absolute_error(y_seconds, pred_seconds)),
        "r2_seconds": float(r2_score(y_seconds, pred_seconds)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    with args.input.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("status") == "ok"]
    if len(rows) < 10:
        raise SystemExit("P0 baseline requires at least 10 successful rows")
    X = np.asarray([row_vector(r) for r in rows], dtype=float)
    y_seconds = np.asarray([float(r["t_exec_s"]) for r in rows], dtype=float)
    y_log = np.log1p(y_seconds)
    groups = np.asarray([r["circuit_id"] for r in rows])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y_log, groups=groups))
    models = {
        "median": None,
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "svr": make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05)),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=args.seed, n_jobs=1, min_samples_leaf=2),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=200, random_state=args.seed, l2_regularization=1.0),
    }
    result = {
        "protocol": "single grouped 80/20 split by circuit_id; target=log1p(T_exec_s)",
        "input_rows": len(rows),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_circuits": len(set(groups[train_idx])),
        "test_circuits": len(set(groups[test_idx])),
        "seed": args.seed,
        "models": {},
    }
    for name, model in models.items():
        if model is None:
            pred = np.full(len(test_idx), np.median(y_log[train_idx]))
        else:
            model.fit(X[train_idx], y_log[train_idx])
            pred = model.predict(X[test_idx])
        result["models"][name] = metrics(y_log[test_idx], pred, y_seconds[test_idx])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
