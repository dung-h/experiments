#!/usr/bin/env python3
"""Read-only verification of Study-R coarsened-DAG OOF metrics.

This script deliberately consumes only the saved three-seed mean OOF table.
It creates a new post-freeze audit directory and never trains or overwrites a
Study-R artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


VARIANTS = ("node_only", "true_dag", "reversed_dag", "shuffled_dag")
CONTRASTS = (
    ("true_dag_minus_node_only", "true_dag", "node_only"),
    ("true_dag_minus_shuffled_dag", "true_dag", "shuffled_dag"),
    ("true_dag_minus_reversed_dag", "true_dag", "reversed_dag"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("non-finite target or prediction")
    if (y < 0).any() or (p < 0).any():
        raise ValueError("log1p metric requires nonnegative target and prediction")
    err = p - y
    ss_raw = float(np.sum((y - y.mean()) ** 2))
    ly, lp = np.log1p(y), np.log1p(p)
    ss_log = float(np.sum((ly - ly.mean()) ** 2))
    if ss_raw <= 0 or ss_log <= 0:
        raise ValueError("zero target variance in bootstrap replicate")
    return {
        "mae_seconds": float(np.mean(np.abs(err))),
        "rmse_seconds": float(np.sqrt(np.mean(err ** 2))),
        "r2_raw_seconds": float(1.0 - np.sum(err ** 2) / ss_raw),
        "r2_log1p_seconds": float(1.0 - np.sum((lp - ly) ** 2) / ss_log),
        "mean_signed_error_seconds": float(np.mean(err)),
    }


def pct(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--canonical-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.input)
    split = {r["row_id"]: r for r in read_csv(args.split_manifest)}
    by_variant: dict[str, dict[str, dict[str, str]]] = {v: {} for v in VARIANTS}
    for row in rows:
        variant, row_id = row["variant"], row["row_id"]
        if variant not in by_variant:
            raise ValueError(f"unexpected variant {variant}")
        if row_id in by_variant[variant]:
            raise ValueError(f"duplicate {variant}/{row_id}")
        by_variant[variant][row_id] = row
    row_ids = sorted(by_variant["node_only"])
    if len(row_ids) != 340 or set(split) != set(row_ids):
        raise ValueError("saved OOF rows do not exactly match 340-row split manifest")
    for variant in VARIANTS:
        if set(by_variant[variant]) != set(row_ids):
            raise ValueError(f"incomplete/misaligned OOF coverage: {variant}")
    target = np.asarray([float(by_variant["node_only"][r]["target_seconds"]) for r in row_ids])
    hashes = np.asarray([split[r]["qasm_sha256"] for r in row_ids])
    backends = np.asarray([split[r]["backend"] for r in row_ids])
    for variant in VARIANTS:
        candidate = np.asarray([float(by_variant[variant][r]["target_seconds"]) for r in row_ids])
        if not np.array_equal(target, candidate):
            raise ValueError(f"targets differ for {variant}")
    prediction = {
        variant: np.asarray([float(by_variant[variant][r]["prediction_seconds"]) for r in row_ids])
        for variant in VARIANTS
    }
    result_rows = []
    for variant in VARIANTS:
        result_rows.append({"scope": "pooled", "variant": variant, "n_rows": len(target),
                            "n_raw_qasm_groups": len(set(hashes)), **metrics(target, prediction[variant])})
    for backend in sorted(set(backends)):
        index = np.flatnonzero(backends == backend)
        for variant in VARIANTS:
            result_rows.append({"scope": f"backend_{backend}_descriptive", "variant": variant,
                                "n_rows": len(index), "n_raw_qasm_groups": len(set(hashes[index])),
                                **metrics(target[index], prediction[variant][index])})
    fields = list(result_rows[0])
    with (args.output / "dag_raw_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(result_rows)

    canonical = json.loads(args.canonical_summary.read_text(encoding="utf-8"))["ensemble_metrics"]
    canonical_differences = {
        variant: max(abs(base - float(canonical[variant][old])) for base, old in (
            (next(r for r in result_rows if r["scope"] == "pooled" and r["variant"] == variant)["mae_seconds"], "mae_seconds"),
            (next(r for r in result_rows if r["scope"] == "pooled" and r["variant"] == variant)["rmse_seconds"], "rmse_seconds"),
            (next(r for r in result_rows if r["scope"] == "pooled" and r["variant"] == variant)["r2_raw_seconds"], "r2_seconds"),
            (next(r for r in result_rows if r["scope"] == "pooled" and r["variant"] == variant)["r2_log1p_seconds"], "r2_log1p_seconds"),
        )) for variant in VARIANTS
    }
    if any(value > 1e-12 for value in canonical_differences.values()):
        raise ValueError(f"recomputed metrics disagree with canonical summary: {canonical_differences}")

    groups = np.asarray(sorted(set(hashes)))
    group_indices = {g: np.flatnonzero(hashes == g) for g in groups}
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(groups), size=(args.replicates, len(groups)), endpoint=False)
    contrast_rows, bootstrap_payload = [], {}
    base_metrics = {r["variant"]: r for r in result_rows if r["scope"] == "pooled"}
    for name, treatment, control in CONTRASTS:
        d_mae, d_r2 = [], []
        discarded = 0
        for draw in draws:
            index = np.concatenate([group_indices[groups[i]] for i in draw])
            try:
                mt = metrics(target[index], prediction[treatment][index])
                mc = metrics(target[index], prediction[control][index])
            except ValueError:
                discarded += 1
                continue
            d_mae.append(mt["mae_seconds"] - mc["mae_seconds"])
            d_r2.append(mt["r2_raw_seconds"] - mc["r2_raw_seconds"])
        d_mae_a, d_r2_a = np.asarray(d_mae), np.asarray(d_r2)
        mae_ci, r2_ci = pct(d_mae_a), pct(d_r2_a)
        row = {
            "contrast": name, "treatment": treatment, "control": control,
            "bootstrap_unit": "raw_qasm_sha256", "bootstrap_replicates_requested": args.replicates,
            "bootstrap_replicates_valid": len(d_mae_a), "bootstrap_replicates_discarded": discarded,
            "bootstrap_seed": args.seed,
            "delta_mae_seconds": base_metrics[treatment]["mae_seconds"] - base_metrics[control]["mae_seconds"],
            "delta_raw_r2": base_metrics[treatment]["r2_raw_seconds"] - base_metrics[control]["r2_raw_seconds"],
            "delta_mae_ci_low": mae_ci[0], "delta_mae_ci_high": mae_ci[1],
            "delta_raw_r2_ci_low": r2_ci[0], "delta_raw_r2_ci_high": r2_ci[1],
            "empirical_bootstrap_fraction_mae_true_dag_improves": float(np.mean(d_mae_a < 0)),
            "empirical_bootstrap_fraction_raw_r2_true_dag_improves": float(np.mean(d_r2_a > 0)),
        }
        contrast_rows.append(row)
        bootstrap_payload[name] = {"delta_mae": d_mae_a.tolist(), "delta_raw_r2": d_r2_a.tolist()}
    with (args.output / "dag_raw_paired_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrast_rows[0])); writer.writeheader(); writer.writerows(contrast_rows)
    np.savez_compressed(args.output / "bootstrap_draws_and_deltas.npz", draws=draws,
                        **{f"{key}_mae": np.asarray(value["delta_mae"]) for key, value in bootstrap_payload.items()},
                        **{f"{key}_r2": np.asarray(value["delta_raw_r2"]) for key, value in bootstrap_payload.items()})
    receipt = {
        "input": str(args.input), "split_manifest": str(args.split_manifest),
        "canonical_summary": str(args.canonical_summary),
        "n_rows": len(row_ids), "n_raw_qasm_groups": len(groups),
        "row_ids_identical_across_variants": True, "targets_identical_across_variants": True,
        "predictions_nonnegative": {v: bool((prediction[v] >= 0).all()) for v in VARIANTS},
        "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
        "canonical_metric_max_abs_difference": canonical_differences,
        "note": "Raw R2 is recomputed after every group-resample on the pooled resampled rows; it is not an average of group R2 values.",
    }
    (args.output / "verification_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
