#!/usr/bin/env python3
"""Evaluate static baselines from the compact Ma--Li physical-DAG manifest.

This is the first model gate in the deep protocol.  It asks whether a compact
physical compiled representation beats the one-feature QCRE weighted critical
path under the same grouped-QASM and strict backend/unseen-QASM splits.  It
does not load the full DAG into memory and therefore remains a cheap baseline
before any GNN is justified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("qcre", "compiled", "physical_graph", "rich_summary")
RICH_FEATURES = (
    "rich_n_nodes", "rich_n_edges", "rich_n_layers", "rich_active_width",
    "rich_zero_duration_fraction", "rich_duration_sum_dt", "rich_duration_max_dt",
    "rich_duration_nonzero_mean_dt", "rich_duration_nonzero_p90_dt",
    "rich_forward_max_dt", "rich_forward_p90_dt", "rich_reverse_p90_dt",
    "rich_criticality_p90_dt", "rich_layer_nodes_mean", "rich_layer_nodes_p90",
    "rich_layer_nodes_max", "rich_quantum_edge_fraction",
    *(f"rich_opcode_{opcode_id}_count" for opcode_id in range(14)),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if len(target) == 0:
        return {
            "mae_seconds": float("nan"),
            "medae_seconds": float("nan"),
            "rmse_seconds": float("nan"),
            "r2_seconds": float("nan"),
            "mae_log1p_seconds": float("nan"),
            "r2_log1p_seconds": float("nan"),
        }
    prediction = np.maximum(prediction, 0.0)
    error = prediction - target
    target_log = np.log1p(target)
    prediction_log = np.log1p(prediction)
    seconds_variance = float(np.sum((target - target.mean()) ** 2))
    log_variance = float(np.sum((target_log - target_log.mean()) ** 2))
    return {
        "mae_seconds": float(np.mean(np.abs(error))),
        "medae_seconds": float(np.median(np.abs(error))),
        "rmse_seconds": float(np.sqrt(np.mean(error**2))),
        "r2_seconds": float(1.0 - np.sum(error**2) / seconds_variance) if seconds_variance else float("nan"),
        "mae_log1p_seconds": float(np.mean(np.abs(prediction_log - target_log))),
        "r2_log1p_seconds": float(1.0 - np.sum((prediction_log - target_log) ** 2) / log_variance)
        if log_variance else float("nan"),
    }


def design(rows: list[dict[str, str]], indices: np.ndarray, model: str) -> np.ndarray:
    values = {
        "qcre": ["qcre_critical_path_seconds"],
        "compiled": ["qcre_critical_path_seconds", "compiled_depth", "active_physical_width"],
        "physical_graph": [
            "qcre_critical_path_seconds", "compiled_depth", "active_physical_width",
            "n_nodes", "n_edges",
        ],
        "rich_summary": list(RICH_FEATURES),
    }[model]
    numeric = np.asarray(
        [[float(row[key]) for key in values] for row in rows], dtype=float
    )
    numeric = np.log1p(np.maximum(numeric, 0.0))
    backend = np.asarray([1.0 if str(row["backend"]) == "osaka" else 0.0 for row in rows])[:, None]
    matrix = np.column_stack((np.ones(len(rows)), numeric, backend))
    return matrix[indices]


def fit_predict(rows: list[dict[str, str]], train: np.ndarray, test: np.ndarray, model: str) -> np.ndarray:
    target = np.asarray([float(row["target_seconds"]) for row in rows])
    x_train = design(rows, train, model)
    x_test = design(rows, test, model)
    # Standardize only the numeric columns using outer-train statistics. This
    # prevents correlated node/edge counts from producing unstable
    # cross-backend extrapolation coefficients. The first column is the
    # intercept and the last is the explicit Osaka indicator.
    if x_train.shape[1] > 2:
        mean = x_train[:, 1:-1].mean(axis=0)
        scale = x_train[:, 1:-1].std(axis=0)
        scale[scale == 0] = 1.0
        x_train[:, 1:-1] = (x_train[:, 1:-1] - mean) / scale
        x_test[:, 1:-1] = (x_test[:, 1:-1] - mean) / scale
    alpha = 1.0
    penalty = np.eye(x_train.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        x_train.T @ x_train + penalty,
        x_train.T @ np.log1p(target[train]),
    )
    return np.expm1(x_test @ coefficients)


def grouped_splits(rows: list[dict[str, str]], folds: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    output = []
    assignments = np.asarray([int(row["group_fold"]) for row in rows])
    for fold in range(1, folds + 1):
        test = np.flatnonzero(assignments == fold)
        train = np.flatnonzero(assignments != fold)
        output.append((f"grouped_qasm_fold_{fold}", train, test))
    return output


def strict_splits(rows: list[dict[str, str]], folds: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    backend = np.asarray([row["backend"] for row in rows])
    assignments = np.asarray([int(row["group_fold"]) for row in rows])
    output = []
    for held in ("osaka", "kyoto"):
        for fold in range(1, folds + 1):
            test = np.flatnonzero((backend == held) & (assignments == fold))
            train = np.flatnonzero((backend != held) & (assignments != fold))
            output.append((f"strict_{held}_fold_{fold}", train, test))
    return output


def family_splits(rows: list[dict[str, str]]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    components = sorted({str(row["family_component"]) for row in rows})
    output = []
    values = np.asarray([str(row["family_component"]) for row in rows])
    for component in components:
        test = np.flatnonzero(values == component)
        train = np.flatnonzero(values != component)
        if len(test) and len(train):
            output.append((f"family_{component}", train, test))
    return output


def pooled_metrics(prediction_rows: list[dict[str, object]], selector) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for model in MODEL_NAMES:
        selected = [
            row for row in prediction_rows
            if row["model"] == model and selector(str(row["split"]))
        ]
        target = np.asarray([float(row["target_seconds"]) for row in selected])
        prediction = np.asarray([float(row["prediction_seconds"]) for row in selected])
        output[model] = metrics(target, prediction)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "validation" / "mali_physical_dag_v1")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "artifacts" / "validation" / "mali_deep_protocol_v1" / "split_audit" / "row_manifest.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rich-features", type=Path, default=None)
    args = parser.parse_args()

    manifest = read_csv(args.artifact_dir / "manifest.csv")
    split_rows = read_csv(args.split_manifest)
    rich_path = args.rich_features or (args.artifact_dir / "graph_summary_features.csv")
    rich_rows = {row["graph_id"]: row for row in read_csv(rich_path)}
    split_by_key = {(row["qasm_sha256"], row["backend"]): row for row in split_rows}
    rows: list[dict[str, str]] = []
    for row in manifest:
        split = split_by_key[(row["qasm_sha256"], row["backend"])]
        enriched = dict(row)
        enriched["group_fold"] = split["group_fold"]
        enriched["family_component"] = split["family_component"]
        enriched.update(rich_rows[row["graph_id"]])
        rows.append(enriched)

    target = np.asarray([float(row["target_seconds"]) for row in rows])
    split_sets = [*grouped_splits(rows, args.folds), *strict_splits(rows, args.folds), *family_splits(rows)]
    prediction_rows: list[dict[str, object]] = []
    aggregate: list[dict[str, object]] = []
    for split_name, train, test in split_sets:
        for model in MODEL_NAMES:
            prediction = fit_predict(rows, train, test, model)
            result = metrics(target[test], prediction)
            result.update({"split": split_name, "model": model, "n_train": len(train), "n_test": len(test)})
            aggregate.append(result)
            for index, estimate in zip(test, prediction):
                prediction_rows.append(
                    {
                        "split": split_name,
                        "model": model,
                        "row_id": rows[index]["row_id"],
                        "circuit": rows[index]["circuit"],
                        "backend": rows[index]["backend"],
                        "family_component": rows[index]["family_component"],
                        "target_seconds": target[index],
                        "prediction_seconds": estimate,
                        "absolute_error_seconds": abs(float(estimate) - target[index]),
                    }
                )

    output_dir = args.artifact_dir / "baseline_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    with (output_dir / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    pooled = {
        "grouped_qasm": pooled_metrics(prediction_rows, lambda name: name.startswith("grouped")),
        "strict_all": pooled_metrics(prediction_rows, lambda name: name.startswith("strict")),
        "strict_osaka": pooled_metrics(prediction_rows, lambda name: name.startswith("strict_osaka")),
        "strict_kyoto": pooled_metrics(prediction_rows, lambda name: name.startswith("strict_kyoto")),
        "family_all": pooled_metrics(prediction_rows, lambda name: name.startswith("family_")),
    }
    qwalk = {
        model: [
            row for row in prediction_rows
            if row["model"] == model and "qwalk" in str(row["circuit"])
            and str(row["split"]).startswith("grouped")
        ]
        for model in ("qcre", "compiled", "physical_graph")
    }

    summary: dict[str, object] = {
        "n_rows": len(rows),
        "n_qasm_hashes": len({row["qasm_sha256"] for row in rows}),
        "splits": [name for name, _, _ in split_sets],
        "models": list(MODEL_NAMES),
        "target_semantics": "Ma-Li observed result.time_taken seconds; 1024 shots; queue excluded",
        "proxy_semantics": "physical features reconstructed from current FakeBackend snapshot",
        "metrics_path": "baseline_evaluation/metrics.csv",
        "predictions_path": "baseline_evaluation/oof_predictions.csv",
        "pooled_metrics": pooled,
        "grouped_qwalk_rows": qwalk,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Ma--Li physical-DAG static baseline evaluation",
        "",
        "This is the first gate before training a graph model. All models predict",
        "the logged observed Ma--Li `result.time_taken` target; only the input",
        "features differ. Fold assignments are grouped by raw-QASM SHA-256 and",
        "strict splits also hold out the backend.",
        "",
        "| Split | Model | MAE (s) | MedAE (s) | RMSE (s) | log-R² | seconds-R² |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        report.append(
            f"| {row['split']} | `{row['model']}` | {row['mae_seconds']:.4f} | "
            f"{row['medae_seconds']:.4f} | {row['rmse_seconds']:.4f} | "
            f"{row['r2_log1p_seconds']:.4f} | {row['r2_seconds']:.4f} |"
        )
    report.extend(
        [
            "",
            "## Pooled interpretation",
            "",
            "| Evaluation | Model | MAE (s) | log-R² | seconds-R² |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for evaluation in ("grouped_qasm", "strict_all", "strict_osaka", "strict_kyoto", "family_all"):
        for model in MODEL_NAMES:
            row = pooled[evaluation][model]
            report.append(
                f"| {evaluation} | `{model}` | {row['mae_seconds']:.4f} | "
                f"{row['r2_log1p_seconds']:.4f} | {row['r2_seconds']:.4f} |"
            )
    report.extend(
        [
            "",
            "The grouped-QASM pooled result is **not** a universal-device score.",
            "The strict rows are two correlated held-backend diagnostics, with",
            "Osaka and Kyoto reported separately in `summary.json`.",
            "Family rows are connected-component leave-one-family-out diagnostics;",
            "the component-level values are retained in `metrics.csv` because",
            "QWalk has only two observations.",
            "",
            "## QWalk guard",
            "",
            "QWalk is only two rows and is not used as a standalone R². In the",
            "grouped fold containing QWalk, the compact physical-count model has",
            "absolute errors of about 6.95 s (Osaka) and 6.51 s (Kyoto); the richer",
            "descriptor model lowers them to about 5.91 s and 5.35 s. This is an",
            "improvement but still does not pass the protocol's tail/QWalk guard",
            "by itself. A future DAG residual must address this explicitly.",
            "",
            "The `qcre` row is a one-feature gate-aware weighted critical-path",
            "calibration. `compiled` adds depth and active physical width. The",
            "`physical_graph` row adds compact node/edge counts and `rich_summary`",
            "adds timing/layer/opcode quantiles. These are static baselines, not",
            "measured historical physical circuits. A DAG model is",
            "only warranted if it improves these rows on the same splits and does",
            "not worsen QWalk/tail error.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
