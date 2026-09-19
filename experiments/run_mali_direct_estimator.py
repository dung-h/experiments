#!/usr/bin/env python3
"""Evaluate source-specific pre-run Ma--Li runtime estimators.

The target is the recorded Osaka/Kyoto ``result.time_taken`` label.  This
script consumes the current-FakeBackend compiled-feature proxy created by
``mali_qcre_proxy.py``; it does not claim to recover the historical physical
circuits or calibration snapshots used by Ma--Li.

It deliberately evaluates direct models and a cross-fitted residual model
around the weighted-duration proxy.  All outer tests are group-aware: repeated
Osaka/Kyoto logical circuits share a split group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import sklearn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts" / "validation" / "mali_qcre_proxy" / "mali_qcre_proxy_features.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "validation" / "mali_direct_estimator"
RANDOM_SEED = 1234

LOGICAL_RAW = (
    "logical_width",
    "logical_depth",
    "logical_two_qubit_depth",
    "logical_total_gate_count",
)
PHYSICAL_RAW = (
    "physical_depth",
    "physical_two_qubit_depth",
    "physical_total_gate_count",
    "physical_two_qubit_gate_count",
    "physical_swap_count",
)
PROXY_FEATURE = "log10_weighted_path_seconds"
CATEGORICAL = ("backend_label",)


@dataclass(frozen=True)
class Split:
    name: str
    held_out: str
    train: np.ndarray
    test: np.ndarray


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    """Prefer a capsule-relative path in a versioned provenance record."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def family_from_circuit(name: str) -> str:
    """Recover the public Ma--Li family token without reading target fields."""
    token = str(name).lower()
    return token.split("_indep_")[0].replace("-", "_")


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "circuit", "backend_label", "target_time_taken_seconds", "qasm_sha256",
        *LOGICAL_RAW, *PHYSICAL_RAW, "qcre_weighted_critical_path_seconds",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"input is missing required fields: {missing}")
    if frame["qasm_sha256"].isna().any():
        raise ValueError("logical-QASM split group is missing")
    if (frame["target_time_taken_seconds"] <= 0).any():
        raise ValueError("Ma--Li target must be strictly positive")
    if (frame["qcre_weighted_critical_path_seconds"] <= 0).any():
        raise ValueError("weighted duration proxy must be strictly positive")

    for column in (*LOGICAL_RAW, *PHYSICAL_RAW, "qcre_weighted_critical_path_seconds"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["target_time_taken_seconds"] = pd.to_numeric(
        frame["target_time_taken_seconds"], errors="raise"
    )
    frame["circuit_family"] = frame["circuit"].map(family_from_circuit)
    for column in (*LOGICAL_RAW, *PHYSICAL_RAW):
        frame[f"log1p_{column}"] = np.log1p(frame[column].clip(lower=0))
    frame[PROXY_FEATURE] = np.log10(frame["qcre_weighted_critical_path_seconds"])
    return frame


def group_fold_assignments(frame: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
    """Assign every logical-QASM group to one reproducible outer fold."""
    groups = np.asarray(sorted(frame["qasm_sha256"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    groups = groups[rng.permutation(len(groups))]
    assignments = np.empty(len(frame), dtype=int)
    for fold, held_out_groups in enumerate(np.array_split(groups, folds)):
        mask = frame["qasm_sha256"].isin(held_out_groups).to_numpy()
        assignments[mask] = fold
    return assignments


def grouped_circuit_splits(frame: pd.DataFrame, folds: int, seed: int) -> list[Split]:
    assignments = group_fold_assignments(frame, folds, seed)
    return [
        Split(
            name="grouped_logical_circuit_5fold",
            held_out=f"fold_{fold + 1}",
            train=np.flatnonzero(assignments != fold),
            test=np.flatnonzero(assignments == fold),
        )
        for fold in range(folds)
    ]


def strict_backend_and_circuit_splits(
    frame: pd.DataFrame, folds: int, seed: int
) -> list[Split]:
    """Hold out a backend and its logical-QASM groups simultaneously.

    The ordinary paired-backend check sees the same logical circuit on the
    opposite backend during training.  That is useful diagnostically, but is
    not a new-backend/new-circuit transfer test.  This construction excludes
    both sources of overlap from the training set.
    """
    assignments = group_fold_assignments(frame, folds, seed)
    backend = frame["backend_label"].astype(str).to_numpy()
    output: list[Split] = []
    for held_backend in sorted(np.unique(backend)):
        for fold in range(folds):
            test = (backend == held_backend) & (assignments == fold)
            train = (backend != held_backend) & (assignments != fold)
            if not test.any() or not train.any():
                continue
            output.append(
                Split(
                    name="backend_and_logical_circuit_held_out",
                    held_out=f"{held_backend}:fold_{fold + 1}",
                    train=np.flatnonzero(train),
                    test=np.flatnonzero(test),
                )
            )
    return output


def leave_group_out_splits(frame: pd.DataFrame, column: str, name: str) -> list[Split]:
    output: list[Split] = []
    for value in sorted(frame[column].astype(str).unique()):
        test = frame[column].astype(str).eq(value).to_numpy()
        if test.all() or not test.any():
            continue
        output.append(
            Split(name=name, held_out=value, train=np.flatnonzero(~test), test=np.flatnonzero(test))
        )
    return output


def feature_columns(block: str) -> tuple[list[str], list[str]]:
    logical = [f"log1p_{column}" for column in LOGICAL_RAW]
    physical = [f"log1p_{column}" for column in PHYSICAL_RAW]
    if block == "physical_depth":
        return ["log1p_physical_depth"], list(CATEGORICAL)
    if block == "weighted_path":
        return [PROXY_FEATURE], list(CATEGORICAL)
    if block == "compiled":
        return [*logical, *physical, PROXY_FEATURE], list(CATEGORICAL)
    raise ValueError(f"unknown feature block: {block}")


def preprocessor(numeric: list[str], categorical: list[str], *, ordinal: bool, scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    if ordinal:
        categorical_encoder: object = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
    else:
        categorical_encoder = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    return ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("categorical", categorical_encoder, categorical)],
        sparse_threshold=0,
    )


def make_model(kind: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    if kind == "ridge":
        return Pipeline(
            [("features", preprocessor(numeric, categorical, ordinal=False, scale=True)),
             ("model", Ridge(alpha=1.0))]
        )
    if kind == "random_forest":
        return Pipeline(
            [("features", preprocessor(numeric, categorical, ordinal=False, scale=False)),
             ("model", RandomForestRegressor(
                 n_estimators=500, min_samples_leaf=2, max_features=0.8,
                 random_state=RANDOM_SEED, n_jobs=-1,
             ))]
        )
    if kind == "hist_gradient_boosting":
        return Pipeline(
            [("features", preprocessor(numeric, categorical, ordinal=True, scale=False)),
             ("model", HistGradientBoostingRegressor(
                 learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                 random_state=RANDOM_SEED,
             ))]
        )
    raise ValueError(f"unknown model kind: {kind}")


def metrics(y_log: np.ndarray, prediction_log: np.ndarray) -> dict[str, float]:
    actual = np.expm1(y_log)
    prediction = np.maximum(np.expm1(prediction_log), 0.0)
    return {
        "mae_log": float(mean_absolute_error(y_log, prediction_log)),
        "rmse_log": float(mean_squared_error(y_log, prediction_log) ** 0.5),
        "r2_log": float(r2_score(y_log, prediction_log)),
        "mae_seconds": float(mean_absolute_error(actual, prediction)),
        "medae_seconds": float(np.median(np.abs(actual - prediction))),
        "rmse_seconds": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2_seconds": float(r2_score(actual, prediction)),
        "median_relative_error": float(np.median(np.abs(actual - prediction) / actual)),
    }


def predict_median(y_train: np.ndarray, size: int) -> np.ndarray:
    return np.full(size, float(np.median(y_train)))


def cross_fitted_residual_prediction(
    frame: pd.DataFrame,
    train: np.ndarray,
    test: np.ndarray,
    y_log: np.ndarray,
) -> np.ndarray:
    """Fit a proxy baseline then learn cross-fitted residuals from structure."""
    proxy_numeric, proxy_categorical = feature_columns("weighted_path")
    full_numeric, full_categorical = feature_columns("compiled")
    train_frame = frame.iloc[train]
    test_frame = frame.iloc[test]
    groups = train_frame["qasm_sha256"].to_numpy()
    unique_groups = np.unique(groups)
    folds = min(5, len(unique_groups))
    if folds < 2:
        raise ValueError("residual model needs at least two logical-circuit groups")
    base_oof = np.empty(len(train), dtype=float)
    for fit_relative, validation_relative in GroupKFold(n_splits=folds).split(train_frame, groups=groups):
        base = make_model("ridge", proxy_numeric, proxy_categorical)
        base.fit(train_frame.iloc[fit_relative], y_log[train[fit_relative]])
        base_oof[validation_relative] = base.predict(train_frame.iloc[validation_relative])
    residual = y_log[train] - base_oof
    residual_model = make_model("random_forest", full_numeric, full_categorical)
    residual_model.fit(train_frame, residual)
    base_final = make_model("ridge", proxy_numeric, proxy_categorical)
    base_final.fit(train_frame, y_log[train])
    return base_final.predict(test_frame) + residual_model.predict(test_frame)


def evaluate_split(
    frame: pd.DataFrame, split: Split, y_log: np.ndarray
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_frame = frame.iloc[split.train]
    test_frame = frame.iloc[split.test]
    results: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []

    model_specs: list[tuple[str, str, str | None]] = [
        ("median_log", "median", None),
        ("physical_depth_ridge", "ridge", "physical_depth"),
        ("weighted_path_ridge", "ridge", "weighted_path"),
        ("compiled_ridge", "ridge", "compiled"),
        ("compiled_random_forest", "random_forest", "compiled"),
        ("compiled_hist_gradient_boosting", "hist_gradient_boosting", "compiled"),
        ("weighted_path_plus_compiled_residual_rf", "residual", "compiled"),
    ]
    for model_name, kind, block in model_specs:
        if kind == "median":
            prediction_log = predict_median(y_log[split.train], len(split.test))
        elif kind == "residual":
            prediction_log = cross_fitted_residual_prediction(frame, split.train, split.test, y_log)
        else:
            numeric, categorical = feature_columns(str(block))
            model = make_model(kind, numeric, categorical)
            model.fit(train_frame, y_log[split.train])
            prediction_log = model.predict(test_frame)
        score = metrics(y_log[split.test], prediction_log)
        results.append(
            {
                "split": split.name,
                "held_out": split.held_out,
                "model": model_name,
                "n_train": int(len(split.train)),
                "n_test": int(len(split.test)),
                **score,
            }
        )
        for absolute_index, predicted_log in zip(split.test, prediction_log):
            predictions.append(
                {
                    "split": split.name,
                    "held_out": split.held_out,
                    "model": model_name,
                    "row_index": int(absolute_index),
                    "circuit": frame.iloc[absolute_index]["circuit"],
                    "circuit_family": frame.iloc[absolute_index]["circuit_family"],
                    "backend_label": frame.iloc[absolute_index]["backend_label"],
                    "logical_circuit_hash": frame.iloc[absolute_index]["qasm_sha256"],
                    "actual_seconds": float(np.expm1(y_log[absolute_index])),
                    "predicted_seconds": float(max(np.expm1(predicted_log), 0.0)),
                    "actual_log": float(y_log[absolute_index]),
                    "predicted_log": float(predicted_log),
                }
            )
    return results, predictions


def aggregate_metrics(predictions: pd.DataFrame, per_fold: pd.DataFrame) -> pd.DataFrame:
    """Score concatenated out-of-fold predictions, never mean per-fold R².

    R² is unstable, and even undefined, for small held-out circuit families.
    Each split design here assigns every row to exactly one outer test set, so
    the scientifically meaningful aggregate is calculated across its complete
    out-of-fold prediction vector.
    """
    metric_columns = [
        "mae_log", "rmse_log", "r2_log", "mae_seconds", "medae_seconds",
        "rmse_seconds", "r2_seconds", "median_relative_error",
    ]
    records: list[dict[str, object]] = []
    for (split, model), group in predictions.groupby(["split", "model"], sort=True):
        if group["row_index"].duplicated().any():
            raise ValueError(f"{split}/{model} has a duplicate out-of-fold row")
        score = metrics(
            group["actual_log"].to_numpy(dtype=float),
            group["predicted_log"].to_numpy(dtype=float),
        )
        if set(metric_columns) != set(score):
            raise AssertionError("metric schema drift")
        records.append(
            {
                "split": split,
                "model": model,
                "n_oof_rows": int(len(group)),
                "n_outer_folds": int(
                    per_fold.loc[
                        (per_fold["split"] == split) & (per_fold["model"] == model),
                        "held_out",
                    ].nunique()
                ),
                **score,
            }
        )
    return pd.DataFrame(records).sort_values(["split", "mae_log", "model"])


def strict_backend_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report the two strict transfer directions separately, not as CV folds."""
    strict = predictions.loc[
        predictions["split"].eq("backend_and_logical_circuit_held_out")
    ]
    records: list[dict[str, object]] = []
    for (backend, model), group in strict.groupby(["backend_label", "model"], sort=True):
        score = metrics(
            group["actual_log"].to_numpy(dtype=float),
            group["predicted_log"].to_numpy(dtype=float),
        )
        records.append({"held_out_backend": backend, "model": model, "n_oof_rows": int(len(group)), **score})
    return pd.DataFrame(records).sort_values(["held_out_backend", "mae_log", "model"])


def markdown_report(
    source: Path,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    strict_directions: pd.DataFrame,
    provenance: dict[str, object],
) -> str:
    lines = [
        "# Ma--Li direct compiled-runtime estimator",
        "",
        "## Target and provenance",
        "",
        "Target: Ma--Li recorded Osaka/Kyoto `result.time_taken` at fixed 1,024 shots.",
        "Queue wait is not added. Compiled features and the weighted-duration path are",
        "reconstructed with current FakeOsaka/FakeKyoto targets, so they are pre-run",
        "proxy features rather than the authors' historical physical circuits.",
        "",
        f"- Input: `{portable_path(source)}`",
        f"- SHA-256: `{provenance['input_sha256']}`",
        f"- Rows: {len(frame)}; logical circuit groups: {frame['qasm_sha256'].nunique()}",
        f"- Backends: {', '.join(sorted(frame['backend_label'].unique()))}",
        f"- Families: {frame['circuit_family'].nunique()}",
        "",
        "## Models",
        "",
        "- `median_log`: train-fold log-median baseline.",
        "- `physical_depth_ridge`: physical depth plus backend.",
        "- `weighted_path_ridge`: current-target weighted critical path plus backend.",
        "- `compiled_*`: logical, compiled physical, weighted-path and backend features.",
        "- `weighted_path_plus_compiled_residual_rf`: cross-fitted duration-proxy Ridge",
        "  plus a Random-Forest residual model. It uses no target-derived input at inference.",
        "",
        "At inference, the input is a logical QASM circuit plus the intended target. The",
        "circuit is transpiled under the frozen FakeBackend protocol, then the model sees",
        "`log1p` logical width/depth/two-qubit-depth/gate-count; `log1p` physical depth,",
        "physical two-qubit depth, total/two-qubit gate counts and SWAP count; `log10`",
        "weighted critical-path seconds; and backend identity. The target is `log1p`",
        "observed seconds. This evaluation intentionally saves out-of-fold predictions,",
        "not a deployable final fitted model: model selection must be frozen first.",
        "",
        "## Split protocol",
        "",
        "- `grouped_logical_circuit_5fold`: primary within-domain test; each QASM SHA-256",
        "  group is entirely train or test.",
        "- `paired_backend_transfer_diagnostic`: hold out one backend, but permits that",
        "  QASM's other-backend copy in training. It diagnoses paired-device variation only.",
        "- `backend_and_logical_circuit_held_out`: for each of five QASM folds and each",
        "  backend, test that backend/fold and train only on the other backend/other four",
        "  folds. It removes the paired-circuit overlap.",
        "- `family_held_out`: leave an algorithm-family token out. The family is used only",
        "  to form the split, never as a model feature.",
        "",
        "## Aggregate out-of-fold metrics",
        "",
        "Every metric is calculated once over the complete vector of out-of-fold",
        "predictions for that split/model, rather than averaging fold-level R² values.",
        "This avoids unstable R² values from small circuit-family folds.",
        "",
        "| Split | Model | OOF rows | Folds | Log MAE | Log R² | MAE (s) | MedAE (s) | R² (s) | Median relative error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['split']} | {row['model']} | {row['n_oof_rows']:.0f} | {row['n_outer_folds']:.0f} | "
            f"{row['mae_log']:.4f} | {row['r2_log']:.4f} | "
            f"{row['mae_seconds']:.4f} | {row['medae_seconds']:.4f} | {row['r2_seconds']:.4f} | "
            f"{100 * row['median_relative_error']:.1f}% |"
        )
    lines += [
        "",
        "## Strict backend + unseen-circuit directions",
        "",
        "There are only two device directions, so they are reported separately. The full",
        "seven-model direction table is in `mali_direct_estimator_strict_backend_metrics.csv`.",
        "",
        "| Held-out backend | Model | OOF rows | Log R² | MAE (s) | Median relative error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    direction_models = {
        "median_log",
        "physical_depth_ridge",
        "compiled_ridge",
        "weighted_path_plus_compiled_residual_rf",
    }
    for _, row in strict_directions.loc[
        strict_directions["model"].isin(direction_models)
    ].iterrows():
        lines.append(
            f"| {row['held_out_backend']} | {row['model']} | {row['n_oof_rows']:.0f} | "
            f"{row['r2_log']:.4f} | {row['mae_seconds']:.4f} | "
            f"{100 * row['median_relative_error']:.1f}% |"
        )
    lines += [
        "",
        "## Reading this experiment",
        "",
        "The primary within-domain result is grouped-logical-circuit performance. The",
        "`paired_backend_transfer_diagnostic` is intentionally weaker: training sees",
        "the same logical QASM on the other backend, so it isolates backend variation",
        "but is not a new-circuit transfer result. `backend_and_logical_circuit_held_out`",
        "removes both the backend and paired logical circuit from training. Family-held-out",
        "is a separate algorithm-family transfer test. A model must beat the physical-depth",
        "baseline on those strict tests before it can support a stronger claim.",
        "This experiment is source-specific and must not be pooled with simulator,",
        "Qonductor job-time, QPack workflow-event, or IonQ service-time labels.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    source = args.input.resolve()
    frame = load_frame(source)
    y_log = np.log1p(frame["target_time_taken_seconds"].to_numpy(dtype=float))

    splits = [*grouped_circuit_splits(frame, args.folds, args.seed)]
    splits.extend(
        leave_group_out_splits(
            frame, "backend_label", "paired_backend_transfer_diagnostic"
        )
    )
    splits.extend(strict_backend_and_circuit_splits(frame, args.folds, args.seed))
    splits.extend(leave_group_out_splits(frame, "circuit_family", "family_held_out"))
    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for split in splits:
        result, prediction = evaluate_split(frame, split, y_log)
        result_rows.extend(result)
        prediction_rows.extend(prediction)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_fold = pd.DataFrame(result_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = aggregate_metrics(predictions, per_fold)
    strict_directions = strict_backend_metrics(predictions)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": portable_path(source),
        "input_sha256": sha256(source),
        "target": "Ma-Li Osaka/Kyoto observed result.time_taken; fixed 1024 shots",
        "target_semantics": frame["target_semantics"].iloc[0],
        "n_rows": int(len(frame)),
        "n_logical_circuit_groups": int(frame["qasm_sha256"].nunique()),
        "n_families": int(frame["circuit_family"].nunique()),
        "n_backends": int(frame["backend_label"].nunique()),
        "outer_splits": {name: int(sum(split.name == name for split in splits)) for name in sorted({split.name for split in splits})},
        "model_configuration": {
            "random_forest": {"n_estimators": 500, "min_samples_leaf": 2, "max_features": 0.8, "seed": RANDOM_SEED},
            "hist_gradient_boosting": {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0, "seed": RANDOM_SEED},
            "ridge": {"alpha": 1.0},
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "feature_blocks": {
            "logical_raw": list(LOGICAL_RAW),
            "physical_raw": list(PHYSICAL_RAW),
            "duration_proxy": "qcre_weighted_critical_path_seconds",
            "categorical": list(CATEGORICAL),
        },
        "boundary": "Current-FakeBackend compiled proxy, not historical Ma-Li physical circuit/calibration recovery.",
    }
    per_fold.to_csv(args.output_dir / "mali_direct_estimator_per_fold.csv", index=False)
    predictions.to_csv(args.output_dir / "mali_direct_estimator_oof_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "mali_direct_estimator_summary.csv", index=False)
    strict_directions.to_csv(
        args.output_dir / "mali_direct_estimator_strict_backend_metrics.csv", index=False
    )
    (args.output_dir / "mali_direct_estimator_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "MALI_DIRECT_COMPILED_ESTIMATOR_REPORT.md").write_text(
        markdown_report(source, frame, summary, strict_directions, provenance), encoding="utf-8"
    )
    print(args.output_dir / "MALI_DIRECT_COMPILED_ESTIMATOR_REPORT.md")


if __name__ == "__main__":
    main()
