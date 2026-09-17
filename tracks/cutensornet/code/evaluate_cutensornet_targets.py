#!/usr/bin/env python3
"""Evaluate one cuTensorNet estimate against several measured targets.

cuTensorNet returns one native ``RUNTIME_EST`` per optimized network.  This
report deliberately does not present that value as three predictions: the
same estimate is evaluated separately against warm contraction, first
contraction, and first-call end-to-end latency.  The comparison is
environment-local: one GPU, one CUDA/cuTensorNet stack, one precision and one
memory policy.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "run-output" / "cutensornet" / "current_grid" / "cutensornet_runtime_benchmark.csv"
DEFAULT_OUTPUT = ROOT / "run-output" / "cutensornet" / "current_grid" / "CUTENSORNET_TARGET_EVALUATION.md"

ESTIMATE = "cutensornet_runtime_est_s"
TARGETS = {
    "warm_contract": "contract_gpu_median_s",
    "first_contract": "first_contract_gpu_s",
    "first_end_to_end": "end_to_end_first_s",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values(rows: Iterable[dict[str, str]], column: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for row in rows:
        try:
            estimate = float(row[ESTIMATE])
            target = float(row[column])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(estimate) and math.isfinite(target) and estimate > 0 and target > 0:
            pairs.append((estimate, target))
    if not pairs:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.asarray(pairs, dtype=float).T


def metric_row(rows: Iterable[dict[str, str]], target_column: str) -> dict[str, float | int | str]:
    estimate, target = values(rows, target_column)
    error = estimate - target
    ratio = target / estimate
    log_error = np.log(estimate) - np.log(target)
    ss_res = float(np.sum((target - estimate) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "target": target_column,
        "n": int(target.size),
        "mae_s": float(np.mean(np.abs(error))),
        "rmse_s": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias_est_minus_actual_s": float(np.mean(error)),
        "median_actual_over_estimate": float(np.median(ratio)),
        "geomean_actual_over_estimate": float(np.exp(np.mean(np.log(ratio)))),
        "median_abs_log_error": float(np.median(np.abs(log_error))),
    }


def fmt_seconds(value: float) -> str:
    if abs(value) >= 1:
        return f"{value:.3f} s"
    return f"{value * 1e3:.3f} ms"


def write_report(path: Path, input_path: Path, rows: list[dict[str, str]]) -> None:
    metrics = [metric_row(rows, target) for target in TARGETS.values()]
    by_family: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_family.setdefault(row.get("family", "unknown"), []).append(row)

    lines = [
        "# cuTensorNet target-semantics evaluation v1",
        "",
        "This is a single-environment evaluation on the RTX 5070 Ti. cuTensorNet",
        "returns one native `RUNTIME_EST` per optimized network; it does not",
        "return separate estimates for warm, first-contraction, and end-to-end",
        "latency. The same estimate is evaluated against each measured target",
        "below. This is not a GPU-held-out or universal-runtime claim.",
        "",
        f"Input: `{input_path}` ({len(rows)} rows).",
        "",
        "## What is compared",
        "",
        "- `cutensornet_runtime_est_s`: the single estimate returned after path optimization, before contraction.",
        "- `contract_gpu_median_s`: warm CUDA-event contraction; build/path/allocation are excluded.",
        "- `first_contract_gpu_s`: first contraction after path setup.",
        "- `end_to_end_first_s`: tensor-network build + path optimization + first contraction.",
        "",
        "## Global metrics: one estimate, three measured targets",
        "",
        "Each row reuses the same `cutensornet_runtime_est_s` prediction and changes only the measured target used for evaluation.",
        "",
        "| Measured target (same estimate reused) | n | MAE | RMSE | R2 | Bias (estimate−actual) | Median actual/estimate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        lines.append(
            "| {target} | {n} | {mae} | {rmse} | {r2:.3f} | {bias} | {ratio:.3f} |".format(
                target=metric["target"],
                n=metric["n"],
                mae=fmt_seconds(float(metric["mae_s"])),
                rmse=fmt_seconds(float(metric["rmse_s"])),
                r2=float(metric["r2"]),
                bias=fmt_seconds(float(metric["bias_est_minus_actual_s"])),
                ratio=float(metric["median_actual_over_estimate"]),
            )
        )

    lines += [
        "",
        "## Warm-contract family ratios",
        "",
        "| Family | Rows | Median actual/estimate | Median absolute log error |",
        "|---|---:|---:|---:|",
    ]
    for family, family_rows in sorted(by_family.items()):
        metric = metric_row(family_rows, TARGETS["warm_contract"])
        lines.append(
            f"| {family} | {metric['n']} | {float(metric['median_actual_over_estimate']):.3f} | {float(metric['median_abs_log_error']):.3f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The warm row asks whether the single native estimate is useful for the GPU",
        "contraction segment. The first-contraction and end-to-end rows answer different",
        "questions about what a user experiences on the first call. It is therefore",
        "incorrect to interpret these rows as three cuTensorNet predictions or",
        "to use the warm-target error as the error of the complete simulator invocation.",
        "",
        "The native estimator is a useful B0 path/feasibility signal, but its",
        "heuristic throughput model is not a live benchmark. A production",
        "user-facing estimator should either (a) report warm contraction and",
        "planning latency separately, or (b) fit a local calibration for a",
        "declared end-to-end target. The latter must be retrained when the GPU,",
        "CUDA/cuTensorNet version, precision or memory policy changes.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = read_rows(args.input)
    write_report(args.output, args.input, rows)
    for target_name, target_column in TARGETS.items():
        metric = metric_row(rows, target_column)
        print(
            target_name,
            f"n={metric['n']}",
            f"MAE={fmt_seconds(float(metric['mae_s']))}",
            f"RMSE={fmt_seconds(float(metric['rmse_s']))}",
            f"R2={float(metric['r2']):.3f}",
            f"median_actual_over_estimate={float(metric['median_actual_over_estimate']):.3f}",
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
