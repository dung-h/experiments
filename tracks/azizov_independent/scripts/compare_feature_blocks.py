#!/usr/bin/env python3
"""Compare source, transpiled and hybrid global feature blocks on P0."""

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


SOURCE = ["logical_width", "logical_depth", "logical_ops", "logical_two_qubit_ops"]
COMPILED = ["physical_width", "physical_depth", "physical_ops", "physical_two_qubit_ops", "swap_count", "dag_node_count"]


def vector(row, names):
    result = [float(row.get(name) or 0.0) for name in names]
    result += [float(row["optimization_level"]), 1.0 if row["backend_class"] == "FakeSherbrooke" else 0.0]
    return result


def score(y_log, pred_log, y_seconds):
    pred_seconds = np.maximum(0.0, np.expm1(pred_log))
    return {
        "r2_log": float(r2_score(y_log, pred_log)),
        "r2_seconds": float(r2_score(y_seconds, pred_seconds)),
        "rmse_seconds": float(mean_squared_error(y_seconds, pred_seconds) ** 0.5),
        "mae_seconds": float(mean_absolute_error(y_seconds, pred_seconds)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    with args.input.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("status") == "ok"]
    y_seconds = np.asarray([float(r["t_exec_s"]) for r in rows])
    y_log = np.log1p(y_seconds)
    groups = np.asarray([r["circuit_id"] for r in rows])
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train, test = next(split.split(rows, y_log, groups=groups))
    blocks = {
        "source": SOURCE,
        "compiled": COMPILED,
        "hybrid": SOURCE + COMPILED,
    }
    model_factories = {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": lambda: RandomForestRegressor(n_estimators=200, random_state=args.seed, n_jobs=1, min_samples_leaf=2),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(max_iter=200, random_state=args.seed, l2_regularization=1.0),
    }
    result = {
        "protocol": "same grouped 80/20 circuit-id split for all blocks",
        "input_rows": len(rows),
        "train_circuits": len(set(groups[train])),
        "test_circuits": len(set(groups[test])),
        "seed": args.seed,
        "blocks": {},
    }
    for block_name, names in blocks.items():
        X = np.asarray([vector(r, names) for r in rows], dtype=float)
        result["blocks"][block_name] = {"features": names + ["optimization_level", "backend_is_sherbrooke"], "models": {}}
        for model_name, factory in model_factories.items():
            model = factory()
            model.fit(X[train], y_log[train])
            result["blocks"][block_name]["models"][model_name] = score(y_log[test], model.predict(X[test]), y_seconds[test])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
