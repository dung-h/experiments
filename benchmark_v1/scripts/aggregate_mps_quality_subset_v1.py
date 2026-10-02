#!/usr/bin/env python3
"""Derive quality-pass-only metrics and paired hash-bootstrap intervals for MPS OOF."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "artifacts/benchmark_v3/simulator/mps_fixed_chi16_runtime_oof"
SEED_REGISTRY = ROOT / "benchmark_v1/registry/seed_registry.json"
SUMMARY_NAME = "mps_quality_pass_metrics_v1.json"
METHODS = (
    "train_fold_median",
    "ridge_alpha_1",
    "graph_seed_42",
    "graph_seed_1234",
    "graph_seed_31415",
    "graph_median_three_seeds",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def derived_seed(root_seed: int, stream: str, artifact_id: str, comparison_label: str) -> int:
    material = f"{root_seed}|{stream}|{artifact_id}|{comparison_label}|0".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def metric_bundle(actual: list[float], predicted: list[float]) -> dict[str, float | int | None]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("metrics require equally sized, nonempty vectors")
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    residuals = [a - p for a, p in zip(actual, predicted)]
    mean_actual = statistics.fmean(actual)
    total = sum((value - mean_actual) ** 2 for value in actual)
    ordered = sorted(errors)

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "n": len(actual),
        "mae_seconds": statistics.fmean(errors),
        "medae_seconds": statistics.median(errors),
        "rmse_seconds": math.sqrt(statistics.fmean(value * value for value in residuals)),
        "mae_log1p_seconds": statistics.fmean(abs(math.log1p(a) - math.log1p(max(p, 0.0))) for a, p in zip(actual, predicted)),
        "r2_seconds": 1.0 - sum(value * value for value in residuals) / total if total else None,
        "p90_abs_error_seconds": quantile(0.90),
        "p99_abs_error_seconds": quantile(0.99),
        "max_abs_error_seconds": max(errors),
    }


def paired_mae_bootstrap(
    actual: list[float],
    left: list[float],
    right: list[float],
    *,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    if not (len(actual) == len(left) == len(right)) or not actual:
        raise ValueError("paired bootstrap requires aligned nonempty vectors")
    observed = statistics.fmean(abs(y - p) - abs(y - q) for y, p, q in zip(actual, left, right))
    rng = random.Random(seed)
    n = len(actual)
    samples: list[float] = []
    for _ in range(replicates):
        indices = [rng.randrange(n) for _ in range(n)]
        samples.append(statistics.fmean(abs(actual[i] - left[i]) - abs(actual[i] - right[i]) for i in indices))
    samples.sort()

    def quantile(probability: float) -> float:
        position = (len(samples) - 1) * probability
        low = math.floor(position)
        high = math.ceil(position)
        return samples[low] + (samples[high] - samples[low]) * (position - low)

    return {
        "observed_mae_difference_seconds_left_minus_right": observed,
        "bootstrap_mean_difference_seconds": statistics.fmean(samples),
        "percentile_95_ci_seconds": [quantile(0.025), quantile(0.975)],
        "replicates": replicates,
        "seed_uint32": seed,
        "quantile_method": "linear interpolation at position (n-1)*p",
    }


def build(run_dir: Path) -> dict[str, Any]:
    prediction_path = run_dir / "five_fold_oof_predictions.csv"
    target_path = run_dir / "hash_targets_and_static_features.csv"
    seed_document = json.loads(SEED_REGISTRY.read_text(encoding="utf-8"))
    predictions = read_csv(prediction_path)
    targets = read_csv(target_path)
    if len(predictions) != 150 or len(targets) != 150:
        raise ValueError("MPS output must include all 150 assigned hashes in both files")
    pred_by_hash = {row["source_qasm_sha256"]: row for row in predictions}
    target_by_hash = {row["source_qasm_sha256"]: row for row in targets}
    if len(pred_by_hash) != 150 or len(target_by_hash) != 150 or set(pred_by_hash) != set(target_by_hash):
        raise ValueError("prediction/target hash identities are not one-to-one")
    for digest, pred in pred_by_hash.items():
        target = target_by_hash[digest]
        if pred["fold"] != target["fold"]:
            raise ValueError(f"fold mismatch for {digest}")
        if (pred["target_seconds"], pred["target_status"]) != (target["target_seconds"], target["target_status"]):
            raise ValueError(f"target mismatch for {digest}")
        if pred["quality_status_separate_audit_only"] != target["quality_status_separate"]:
            raise ValueError(f"quality status mismatch for {digest}")
    finite = [d for d, row in target_by_hash.items() if row["target_status"] == "runtime_observed"]
    quality = [d for d in finite if target_by_hash[d]["quality_status_separate"] == "quality_pass"]
    quality_failed = [d for d in finite if target_by_hash[d]["quality_status_separate"] == "quality_failed"]
    unavailable = [d for d, row in target_by_hash.items() if row["target_status"] != "runtime_observed"]
    if (len(finite), len(quality), len(quality_failed), len(unavailable)) != (144, 142, 2, 6):
        raise ValueError("expected 144 finite (142 quality-pass, 2 quality-failed) and six unavailable hashes")
    actual = [float(target_by_hash[d]["target_seconds"]) for d in quality]
    method_predictions = {
        method: [float(pred_by_hash[d][f"pred_{method}_seconds"]) for d in quality]
        for method in METHODS
    }
    if any(not math.isfinite(value) or value < 0 for values in method_predictions.values() for value in values):
        raise ValueError("quality-pass predictions contain a nonfinite or negative value")
    artifact_id = "mps-fixed-chi16-source-dag-runtime-adaptation-v1-quality-pass"
    metrics = {method: metric_bundle(actual, values) for method, values in method_predictions.items()}
    paired = {}
    baseline = method_predictions["ridge_alpha_1"]
    for method, values in method_predictions.items():
        if method == "ridge_alpha_1":
            continue
        label = f"quality-pass:{method}_minus_ridge_alpha_1"
        seed = derived_seed(int(seed_document["root_seed"]), seed_document["streams"]["bootstrap"], artifact_id, label)
        paired[label] = paired_mae_bootstrap(actual, values, baseline, seed=seed)
    return {
        "artifact_id": artifact_id,
        "status": "quality_pass_subset_derived_from_frozen_oof",
        "method_claim": "fixed-configuration local MPS runtime-only adaptation; not Family-Aware reproduction",
        "target_clock_id": "cudaq_mps_fp64_bond16_warm_state_execution_seconds",
        "prediction_unit": "exact-QASM SHA-256 hash",
        "input_files_sha256": {prediction_path.name: sha256(prediction_path), target_path.name: sha256(target_path), SEED_REGISTRY.name: sha256(SEED_REGISTRY)},
        "coverage": {"assigned_hashes": 150, "finite_runtime_labels": 144, "quality_pass_finite": len(quality), "quality_failed_finite_retained_in_primary": len(quality_failed), "unavailable_unimputed": len(unavailable)},
        "quality_pass_metrics_only": metrics,
        "paired_mae_difference_vs_ridge": paired,
        "bootstrap_specification": {"replicates": 10_000, "interval": "95% percentile", "resampling_unit": "paired exact-QASM hash", "quantile_method": "linear interpolation at position (n-1)*p", "observed_delta_separate_from_bootstrap_mean": True},
        "quality_status_used_as_model_input": False,
        "interpretation": "This subset answers performance among quality-pass labels only. Primary runtime-only metrics retain all 144 finite labels, including two quality-failed labels; unavailable targets remain unscored and unimputed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    result = build(args.run_dir)
    output = args.run_dir / SUMMARY_NAME
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing derived output: {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output), "n_quality_pass": result["coverage"]["quality_pass_finite"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
