#!/usr/bin/env python3
"""Evaluate a source-specific VQCSim prepared-inference runtime estimator.

Only successful VQCSim inference rows at batch size one are accepted. The
target is ``mean_wall_ms`` from the upstream runner's GPU-synchronized timed
stage; preparation, parity and circuit generation are excluded. Input files
are explicitly labelled (for example ``base=records.csv`` and
``mirror=records.csv``), preserving structural-variant provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = [
    "num_qubits", "depth", "total_gate_count", "two_qubit_gate_count",
    "two_qubit_gate_density", "parameter_count", "measurement_count",
]
CATEGORICAL = ["family", "dataset_variant", "entanglement_class", "entanglement_pattern_class"]
TARGET = "mean_wall_ms"


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("input must be LABEL=PATH")
    return label.strip(), Path(path).expanduser()


def load_rows(inputs: list[tuple[str, Path]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, path in inputs:
        with path.open(newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        for row in source_rows:
            if row.get("framework") != "vqcsim" or row.get("mode") != "inference":
                continue
            if row.get("terminal_status") != "ok" or row.get("parity_passed") != "True":
                continue
            if int(float(row.get("batch_size") or 0)) != 1:
                continue
            try:
                target = float(row[TARGET])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(target) or target <= 0:
                continue
            parsed: dict[str, object] = dict(row)
            parsed["dataset_variant"] = label
            parsed["source_records_path"] = str(path)
            for column in NUMERIC:
                parsed[column] = float(row[column]) if row.get(column) not in (None, "") else np.nan
            rows.append(parsed)
    if not rows:
        raise SystemExit("No eligible VQCSim batch-1 inference rows")
    frame = pd.DataFrame(rows)
    for column in CATEGORICAL:
        frame[column] = frame[column].fillna("unknown").astype(str)
    frame["target_log_ms"] = np.log(frame[TARGET].astype(float))
    frame["circuit_group"] = frame["family"].astype(str) + "::" + frame["size"].astype(str)
    return frame


def make_model(name: str, seed: int) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    category = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformer = ColumnTransformer([("numeric", numeric, NUMERIC), ("category", category, CATEGORICAL)])
    if name == "ridge":
        predictor = Ridge(alpha=3.0)
    elif name == "random_forest":
        predictor = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=1)
    elif name == "hist_gradient_boosting":
        predictor = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=3, l2_regularization=1.0, random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([("features", transformer), ("model", predictor)])


def fold_sets(frame: pd.DataFrame, seed: int) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    index = np.arange(len(frame))
    return {
        "random_5fold_descriptive": list(KFold(n_splits=5, shuffle=True, random_state=seed).split(index)),
        "circuit_group_5fold": list(GroupKFold(n_splits=5).split(index, groups=frame["circuit_group"])),
        "family_held_out": list(LeaveOneGroupOut().split(index, groups=frame["family"])),
        "width_held_out": list(LeaveOneGroupOut().split(index, groups=frame["num_qubits"])),
    }


def score(y_log: np.ndarray, p_log: np.ndarray) -> dict[str, float]:
    y_ms, p_ms = np.exp(y_log), np.exp(p_log)
    return {
        "log_mae": float(mean_absolute_error(y_log, p_log)),
        "log_rmse": float(np.sqrt(np.mean((y_log - p_log) ** 2))),
        "log_r2": float(r2_score(y_log, p_log)),
        "mae_ms": float(mean_absolute_error(y_ms, p_ms)),
        "rmse_ms": float(np.sqrt(np.mean((y_ms - p_ms) ** 2))),
        "r2_ms": float(r2_score(y_ms, p_ms)),
    }


def evaluate(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = frame[NUMERIC + CATEGORICAL]
    y = frame["target_log_ms"].to_numpy(dtype=float)
    metrics: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for split_name, folds in fold_sets(frame, seed).items():
        predictions = {"median_train_log": np.full(len(frame), np.nan)}
        predictions.update({name: np.full(len(frame), np.nan) for name in ("ridge", "random_forest", "hist_gradient_boosting")})
        fold_number = np.full(len(frame), -1, dtype=int)
        for fold, (train, test) in enumerate(folds):
            predictions["median_train_log"][test] = float(np.median(y[train]))
            for name in ("ridge", "random_forest", "hist_gradient_boosting"):
                model = make_model(name, seed)
                model.fit(x.iloc[train], y[train])
                predictions[name][test] = model.predict(x.iloc[test])
            fold_number[test] = fold
        for model_name, prediction in predictions.items():
            if np.isnan(prediction).any():
                raise RuntimeError(f"incomplete OOF values: {split_name}/{model_name}")
            metrics.append({"split": split_name, "model": model_name, "n": len(frame), **score(y, prediction)})
            for position, predicted in enumerate(prediction):
                prediction_rows.append({
                    "split": split_name, "fold": int(fold_number[position]), "model": model_name,
                    "family": frame.iloc[position]["family"], "dataset_variant": frame.iloc[position]["dataset_variant"],
                    "num_qubits": int(frame.iloc[position]["num_qubits"]), "depth": int(frame.iloc[position]["depth"]),
                    "actual_ms": float(np.exp(y[position])), "predicted_ms": float(np.exp(predicted)),
                    "actual_log_ms": float(y[position]), "predicted_log_ms": float(predicted),
                })
    return pd.DataFrame(metrics), pd.DataFrame(prediction_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=parse_input, action="append", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    frame = load_rows(args.input)
    metrics, predictions = evaluate(frame, args.seed)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_canonical_rows.csv"), index=False)
    metrics.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv"), index=False)
    predictions.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_oof_predictions.csv"), index=False)
    summary = {
        "source": "VQCSim only",
        "target_semantics": "prepared-circuit, batch-1 local GPU inference mean_wall_ms; preparation and parity excluded",
        "n_rows": int(len(frame)),
        "families": sorted(frame["family"].unique().tolist()),
        "widths": sorted(int(value) for value in frame["num_qubits"].unique()),
        "dataset_variants": sorted(frame["dataset_variant"].unique().tolist()),
        "grouping": "family::size keeps paired base/mirror records out of the same circuit-group fold",
        "seed": args.seed,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = []
    for row in metrics.itertuples(index=False):
        rows.append(f"| {row.split} | {row.model} | {row.log_mae:.3f} | {row.log_r2:.3f} | {row.mae_ms:.3f} | {row.r2_ms:.3f} |")
    args.output_prefix.with_suffix(".md").write_text("\n".join([
        "# VQCSim prepared-inference runtime estimator", "",
        "This evaluation is VQCSim-only. The target is the GPU-synchronized prepared-circuit inference stage at batch size one.",
        f"Rows: **{len(frame)}**; variants: **{', '.join(summary['dataset_variants'])}**; widths: **{min(summary['widths'])}–{max(summary['widths'])}**.",
        "", "| Split | Model | Log MAE | Log R2 | MAE ms | R2 ms |", "|---|---|---:|---:|---:|---:|", *rows,
        "", "`circuit_group_5fold` is the primary split: base and mirror versions of the same family/size are not separated across train and test.",
        "Family- and width-held-out rows are transfer diagnostics, not evidence for an all-simulator estimator.", "",
    ]))
    print(args.output_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
