#!/usr/bin/env python3
"""Group-bootstrap evaluation for strict no-real-QASM transfer screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_mali_physical_baselines import ROOT, metrics, read_csv


def ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "artifacts/validation/mali_strict_transfer_logical_v1")
    parser.add_argument("--mali-root", type=Path, default=ROOT.parent / "Quantum-Execution-Time-Prediction")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    rows = read_csv(args.input_dir / "real_transfer_oof_predictions.csv")
    import pandas as pd
    manifest = pd.read_csv(args.mali_root / "data/osaka_kyoto_fake_manifest.csv")
    audit = {
        f"{row.source_device}:{row.sample_index}": hashlib.sha256((args.mali_root / row.qasm_path).read_bytes()).hexdigest()
        for row in manifest.itertuples()
    }
    for row in rows:
        row["qasm_sha256"] = audit[row["row_id"]]
    modes = sorted({row["mode"] for row in rows}); seeds = sorted({int(row["seed"]) for row in rows}); ids = sorted({row["row_id"] for row in rows})
    index = {(int(row["seed"]), row["mode"], row["row_id"]): row for row in rows}
    for seed in seeds:
        for mode in modes:
            if any((seed, mode, row_id) not in index for row_id in ids): raise ValueError("incomplete OOF coverage")
    ensemble = {}
    for mode in modes:
        values = []
        for row_id in ids:
            source = [index[(seed, mode, row_id)] for seed in seeds]
            values.append({"row_id": row_id, "qasm_sha256": source[0]["qasm_sha256"], "target_seconds": float(source[0]["target_seconds"]), "prediction_seconds": float(np.mean([float(item["prediction_seconds"]) for item in source]))})
        ensemble[mode] = values
    aggregate = {mode: metrics(np.asarray([r["target_seconds"] for r in values]), np.asarray([r["prediction_seconds"] for r in values])) for mode, values in ensemble.items()}
    groups = sorted({row["qasm_sha256"] for row in ensemble["scratch"]})
    group_indices = {group: np.asarray([i for i, row in enumerate(ensemble["scratch"]) if row["qasm_sha256"] == group]) for group in groups}
    rng = np.random.default_rng(args.seed); comparisons = {}
    for mode in ("frozen_transfer", "finetune_transfer"):
        deltas_log, deltas_mae = [], []
        for _ in range(args.bootstrap_replicates):
            selected = np.concatenate([group_indices[group] for group in rng.choice(groups, len(groups), replace=True)])
            target = np.asarray([ensemble["scratch"][i]["target_seconds"] for i in selected])
            scratch = np.asarray([ensemble["scratch"][i]["prediction_seconds"] for i in selected])
            transfer = np.asarray([ensemble[mode][i]["prediction_seconds"] for i in selected])
            a, b = metrics(target, scratch), metrics(target, transfer)
            deltas_log.append(b["r2_log1p_seconds"] - a["r2_log1p_seconds"]); deltas_mae.append(b["mae_seconds"] - a["mae_seconds"])
        log, mae = np.asarray(deltas_log), np.asarray(deltas_mae)
        comparisons[f"{mode}_minus_scratch"] = {"delta_log_r2": aggregate[mode]["r2_log1p_seconds"] - aggregate["scratch"]["r2_log1p_seconds"], "delta_mae_seconds": aggregate[mode]["mae_seconds"] - aggregate["scratch"]["mae_seconds"], "log_r2_95pct": ci(log), "mae_seconds_95pct": ci(mae), "probability_log_r2_better": float(np.mean(log > 0)), "probability_mae_better": float(np.mean(mae < 0))}
    output = args.input_dir / "evaluation"; output.mkdir(parents=True, exist_ok=True)
    out_rows = [{"mode": mode, **row} for mode, values in ensemble.items() for row in values]
    with (output / "seed_mean_oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0])); writer.writeheader(); writer.writerows(out_rows)
    summary = {"seeds": seeds, "n_rows": len(ids), "n_qasm_hashes": len(groups), "bootstrap_replicates": args.bootstrap_replicates, "aggregate_seed_mean_predictions": aggregate, "paired_group_bootstrap": comparisons, "interpretation": "logical-feature strict no-real-QASM screen only; mean predictions summarize seed variance and are not a selected deployment model"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = ["# Strict no-real-QASM transfer evaluation", "", "All 170 real-QPU hashes were excluded from source pretraining. Evaluation groups rows by raw QASM hash and compares three-seed mean predictions; bootstrap resamples the 170 QASM groups.", "", "| Mode | MAE (s) | log-R² |", "|---|---:|---:|"]
    for mode in modes: report.append(f"| `{mode}` | {aggregate[mode]['mae_seconds']:.4f} | {aggregate[mode]['r2_log1p_seconds']:.4f} |")
    report += ["", "| Comparison | Δ log-R² 95% CI | Δ MAE 95% CI | P(log-R² improves) | P(MAE improves) |", "|---|---:|---:|---:|---:|"]
    for name, value in comparisons.items():
        report.append(f"| `{name}` | [{value['log_r2_95pct'][0]:.4f}, {value['log_r2_95pct'][1]:.4f}] | [{value['mae_seconds_95pct'][0]:.4f}, {value['mae_seconds_95pct'][1]:.4f}] | {value['probability_log_r2_better']:.3f} | {value['probability_mae_better']:.3f} |")
    report += ["", "Frozen transfer failing establishes source/target representation mismatch in this strict screen. Fine-tuning would need to beat scratch consistently before it can be described as useful transfer; this does not adjudicate the unavailable full Graph Transformer strict experiment."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
