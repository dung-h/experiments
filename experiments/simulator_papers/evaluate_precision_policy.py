#!/usr/bin/env python3
"""Evaluate a train-fold-only complex-precision policy on matched TN pairs.

The policy predicts whether complex64 agrees with the matched complex128
scalar to a declared relative tolerance. If it predicts unsafe, it selects
complex128. This is a local strategy-selection study, not a universal
numerical-accuracy guarantee.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


KEYS = ("family", "circuit_variant", "family_depth_multiplier", "num_qubits", "circuit_seed")
NUMERIC = [
    "num_qubits", "logical_depth", "logical_total_gate_count", "logical_two_qubit_gate_count",
    "family_depth_multiplier", "family_depth_parameter", "pre_run_flop_count",
    "pre_run_largest_intermediate_elements", "pre_run_num_slices",
]
CATEGORICAL = ["family", "circuit_variant"]


def paired_frame(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    grouped: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        # Initial 20-pair pilot predates explicit depth metadata. Preserve
        # compatibility while making the expanded grid self-describing.
        row.setdefault("family_depth_multiplier", "1.0")
        row.setdefault("family_depth_parameter", "0")
        key = tuple(row[name] for name in KEYS)
        grouped.setdefault(key, {})[row["precision"]] = row
    pairs: list[dict[str, object]] = []
    for key, values in grouped.items():
        if set(values) != {"complex64", "complex128"}:
            continue
        low, high = values["complex64"], values["complex128"]
        result = dict(zip(KEYS, key))
        for name in NUMERIC:
            result[name] = float(low[name])
        low_value = complex(float(low["scalar_real"]), float(low["scalar_imag"]))
        high_value = complex(float(high["scalar_real"]), float(high["scalar_imag"]))
        result.update({
            "complex64_seconds": float(low["contract_gpu_median_s"]),
            "complex128_seconds": float(high["contract_gpu_median_s"]),
            "relative_scalar_difference": abs(low_value - high_value) / max(abs(high_value), 1e-15),
        })
        pairs.append(result)
    frame = pd.DataFrame(pairs)
    if len(frame) < 12:
        raise SystemExit("Need at least 12 complete matched precision pairs")
    return frame


def make_model(seed: int) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    category = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transform = ColumnTransformer([("numeric", numeric, NUMERIC), ("category", category, CATEGORICAL)])
    return Pipeline([("features", transform), ("classifier", RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=1,
    ))])


def folds(frame: pd.DataFrame, seed: int) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    indices = np.arange(len(frame))
    return {
        "random_5fold_descriptive": list(KFold(n_splits=5, shuffle=True, random_state=seed).split(indices)),
        "family_held_out": list(LeaveOneGroupOut().split(indices, groups=frame["family"])),
        "width_held_out": list(LeaveOneGroupOut().split(indices, groups=frame["num_qubits"])),
        "circuit_group_5fold": list(GroupKFold(n_splits=5).split(indices, groups=frame["family"].astype(str) + "::" + frame["num_qubits"].astype(str))),
    }


def metrics(actual_safe: np.ndarray, choose64: np.ndarray, low_time: np.ndarray, high_time: np.ndarray) -> dict[str, float]:
    selected = np.where(choose64, low_time, high_time)
    oracle = np.where(actual_safe, low_time, high_time)
    unsafe = choose64 & ~actual_safe
    return {
        "n": int(len(actual_safe)),
        "complex64_selection_rate": float(np.mean(choose64)),
        "unsafe_selection_rate": float(np.mean(unsafe)),
        "unsafe_given_complex64_rate": float(np.mean(unsafe[choose64])) if choose64.any() else 0.0,
        "policy_over_always_complex128_time_ratio": float(np.sum(selected) / np.sum(high_time)),
        "policy_over_safe_oracle_time_ratio": float(np.sum(selected) / np.sum(oracle)),
        "always_complex64_unsafe_rate": float(np.mean(~actual_safe)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="One or more homogeneous precision-run CSV files.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=5e-7)
    parser.add_argument("--safe-probability", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.relative_tolerance <= 0 or not 0 < args.safe_probability <= 1:
        raise SystemExit("tolerance must be positive and safe probability must be in (0, 1]")
    frame = paired_frame(args.inputs)
    frame["safe_complex64"] = frame["relative_scalar_difference"] <= args.relative_tolerance
    if frame["safe_complex64"].nunique() < 2:
        raise SystemExit("Tolerance yields one class; choose a tolerance that produces safe and unsafe paired labels")
    x = frame[NUMERIC + CATEGORICAL]
    y = frame["safe_complex64"].to_numpy(dtype=bool)
    low_time = frame["complex64_seconds"].to_numpy(dtype=float)
    high_time = frame["complex128_seconds"].to_numpy(dtype=float)
    summaries: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    for split_name, split_folds in folds(frame, args.seed).items():
        probability = np.full(len(frame), np.nan)
        fold_no = np.full(len(frame), -1, dtype=int)
        for fold, (train, test) in enumerate(split_folds):
            if len(np.unique(y[train])) < 2:
                probability[test] = float(np.mean(y[train]))
            else:
                model = make_model(args.seed)
                model.fit(x.iloc[train], y[train])
                positive = list(model.named_steps["classifier"].classes_).index(True)
                probability[test] = model.predict_proba(x.iloc[test])[:, positive]
            fold_no[test] = fold
        choose64 = probability >= args.safe_probability
        roc_auc = float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, probability))
        summaries.append({
            "split": split_name, "relative_tolerance": args.relative_tolerance,
            "safe_probability": args.safe_probability, "roc_auc": roc_auc,
            **metrics(y, choose64, low_time, high_time),
        })
        for position in range(len(frame)):
            output_rows.append({
                "split": split_name, "fold": int(fold_no[position]),
                **{key: frame.iloc[position][key] for key in KEYS},
                "actual_safe_complex64": bool(y[position]), "predicted_safe_probability": float(probability[position]),
                "selected_precision": "complex64" if choose64[position] else "complex128",
                "relative_scalar_difference": float(frame.iloc[position]["relative_scalar_difference"]),
                "selected_seconds": float(low_time[position] if choose64[position] else high_time[position]),
            })
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_canonical_pairs.csv"), index=False)
    pd.DataFrame(summaries).to_csv(args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv"), index=False)
    pd.DataFrame(output_rows).to_csv(args.output_prefix.with_name(args.output_prefix.name + "_oof_decisions.csv"), index=False)
    summary = {
        "target_semantics": "strategy label from local complex64/complex128 scalar agreement; runtime remains separate warm contraction target",
        "n_pairs": int(len(frame)), "relative_tolerance": args.relative_tolerance,
        "safe_probability": args.safe_probability, "safe_complex64_pairs": int(y.sum()),
        "unsafe_complex64_pairs": int((~y).sum()), "splits": summaries,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    table = ["| Split | AUC | 64 selection | Unsafe given 64 | Policy / 128 time | Policy / oracle time |", "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        table.append(f"| {row['split']} | {row['roc_auc']:.3f} | {100*row['complex64_selection_rate']:.1f}% | {100*row['unsafe_given_complex64_rate']:.1f}% | {row['policy_over_always_complex128_time_ratio']:.3f}× | {row['policy_over_safe_oracle_time_ratio']:.3f}× |")
    args.output_prefix.with_suffix(".md").write_text("\n".join([
        "# cuTensorNet precision-policy evaluation", "",
        f"A `complex64` decision is called safe only when its scalar relative difference against the matched `complex128` result is at most `{args.relative_tolerance:g}`.",
        "The policy is trained and evaluated out-of-fold. This does not establish statevector fidelity, hardware-execution fidelity or a universal precision policy.", "", *table, "",
    ]))
    print(args.output_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
