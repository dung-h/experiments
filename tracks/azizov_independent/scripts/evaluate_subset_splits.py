#!/usr/bin/env python3
"""Evaluate baseline models on the empirically feasible subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


SOURCE = ["logical_width", "logical_depth", "logical_ops", "logical_two_qubit_ops"]
COMPILED = ["physical_width", "physical_depth", "physical_ops", "physical_two_qubit_ops", "swap_count", "dag_node_count"]


def vector(row, names):
    return [*(float(row.get(name) or 0.0) for name in names), float(row["optimization_level"]), float(row["backend_class"] == "FakeSherbrooke")]


def metric(y_log, pred_log, y_seconds):
    pred_seconds = np.maximum(0.0, np.expm1(pred_log))
    return {
        "r2_log": float(r2_score(y_log, pred_log)),
        "r2_seconds": float(r2_score(y_seconds, pred_seconds)),
        "rmse_seconds": float(mean_squared_error(y_seconds, pred_seconds) ** 0.5),
        "mae_seconds": float(mean_absolute_error(y_seconds, pred_seconds)),
    }


def factories(seed):
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "svr": lambda: make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05)),
        "random_forest": lambda: RandomForestRegressor(n_estimators=250, random_state=seed, n_jobs=1, min_samples_leaf=2),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(max_iter=250, random_state=seed, l2_regularization=1.0),
    }


def evaluate(rows, train_idx, test_idx, feature_names, seed):
    y_seconds = np.asarray([float(r["t_exec_s"]) for r in rows])
    y_log = np.log1p(y_seconds)
    X = np.asarray([vector(r, feature_names) for r in rows], dtype=float)
    result = {}
    for name, factory in factories(seed).items():
        model = factory()
        model.fit(X[train_idx], y_log[train_idx])
        result[name] = metric(y_log[test_idx], model.predict(X[test_idx]), y_seconds[test_idx])
    median_pred = np.full(len(test_idx), np.median(y_log[train_idx]))
    result["median"] = metric(y_log[test_idx], median_pred, y_seconds[test_idx])
    return result


def add_split(result, split_name, rows, train_idx, test_idx, seed):
    result[split_name] = {
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_circuits": len({rows[i]["circuit_id"] for i in train_idx}),
        "test_circuits": len({rows[i]["circuit_id"] for i in test_idx}),
        "train_families": len({rows[i]["family"] for i in train_idx}),
        "test_families": len({rows[i]["family"] for i in test_idx}),
        "blocks": {
            "source": evaluate(rows, train_idx, test_idx, SOURCE, seed),
            "compiled": evaluate(rows, train_idx, test_idx, COMPILED, seed),
            "hybrid": evaluate(rows, train_idx, test_idx, SOURCE + COMPILED, seed),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    with args.input.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("status") == "ok"]
    if len(rows) < 100:
        raise SystemExit("Expected the complete feasible subset, not a smoke file")
    indices = np.arange(len(rows))
    y = np.asarray([float(r["t_exec_s"]) for r in rows])
    result = {
        "protocol": "target=log1p(T_exec_s), same feature blocks and model factories across splits",
        "input_rows": len(rows),
        "seed": args.seed,
        "feature_blocks": {"source": SOURCE, "compiled": COMPILED, "hybrid": SOURCE + COMPILED},
        "splits": {},
    }
    train, test = train_test_split(indices, test_size=0.2, random_state=args.seed)
    add_split(result["splits"], "random_row", rows, train, test, args.seed)
    grouped = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train, test = next(grouped.split(indices, y, groups=np.asarray([r["circuit_id"] for r in rows])))
    add_split(result["splits"], "grouped_circuit", rows, train, test, args.seed)
    family_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train, test = next(family_split.split(indices, y, groups=np.asarray([r["family"] for r in rows])))
    add_split(result["splits"], "family_held_out", rows, train, test, args.seed)

    for held_out in sorted({r["backend_class"] for r in rows}):
        train = np.asarray([i for i, row in enumerate(rows) if row["backend_class"] != held_out])
        test = np.asarray([i for i, row in enumerate(rows) if row["backend_class"] == held_out])
        add_split(result["splits"], f"backend_held_out_{held_out}", rows, train, test, args.seed)

    lines = [
        "# Feasible subset split evaluation",
        "",
        "This report uses the empirically feasible 149-circuit subset (1,192",
        "successful runtime rows). The target is `log1p(T_exec_s)`. All models",
        "use the same split within a section; the grouped circuit split is the",
        "primary estimate because each logical circuit has multiple backend/opt",
        "rows.",
        "",
        "| Split | Block | Best model by log-R² | R² log | R² seconds | RMSE seconds |",
        "|---|---|---|---:|---:|---:|",
    ]
    for split_name, split_data in result["splits"].items():
        for block_name, models in split_data["blocks"].items():
            best_name = max(models, key=lambda name: models[name]["r2_log"])
            best = models[best_name]
            lines.append(f"| {split_name} | {block_name} | {best_name} | {best['r2_log']:.4f} | {best['r2_seconds']:.4f} | {best['rmse_seconds']:.4f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- Random-row scores are optimistic because the same logical circuit can",
        "  appear in train and test under another backend or optimization level.",
        "- Grouped-circuit scores are the primary within-domain estimate.",
        "- Family-held-out scores test algorithm-family transfer, while the two",
        "  backend-held-out scores test transfer between current fake snapshots.",
        "- These are baseline results on a conservative local subset, not the",
        "  paper's full 1,402-circuit/GNN result.",
    ]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    args.output_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
