#!/usr/bin/env python3
"""Fit the pinned Qonductor polynomial on the frozen 7,350-row panel."""
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark_v1/scripts"))
import run_qonductor_native_unified as upstream_runner

PANEL = ROOT / "artifacts/real_qpu/common_panel"
FEATURES = ROOT / "artifacts/real_qpu/qonductor_native_features/feature_rows.csv"
TARGETS = PANEL / "targets.csv"
OUTER = PANEL / "outer_splits.csv"
INNER = PANEL / "inner_splits.csv"
OUT = PANEL / "qonductor_polynomial"
FEATURE_NAMES = upstream_runner.FEATURE_NAMES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"refuse to overwrite {OUT}")
    panel = read(PANEL / "panel.csv")
    targets = {r["canonical_observation_id"]: r for r in read(TARGETS)}
    feature_rows = {r["canonical_row_id"]: r for r in read(FEATURES)}
    outer = {r["canonical_observation_id"]: r for r in read(OUTER)}
    inner_rows = read(INNER)
    ids = {r["canonical_observation_id"] for r in panel}
    if len(panel) != 7350 or ids != set(targets) or ids != set(outer):
        raise ValueError("frozen_panel_id_mismatch")
    if not ids <= set(feature_rows):
        raise ValueError("native_feature_input_missing_panel_ids")
    inner = {(int(r["outer_fold"]), r["canonical_observation_id"]):
             {"inner_fold": r["inner_fold"]} for r in inner_rows}
    eligible = []
    for rid in sorted(ids):
        f, t, s = feature_rows[rid], targets[rid], outer[rid]
        if not f["availability_status"].startswith("available"):
            continue
        row = {"canonical_observation_id": rid, "source_id": t["source_id"],
               "outer_fold": int(s["outer_fold"]), "unified_leakage_group_id": s["group_id"],
               "target_seconds": float(t["target_seconds"]),
               **{name: float(f[name]) for name in FEATURE_NAMES}}
        eligible.append(row)
    if len(eligible) != len(ids):
        raise ValueError(f"native_features_not_complete_for_panel:{len(eligible)}/{len(ids)}")
    attempts, folds = [], []
    for fold in range(5):
        degree, predictions, record = upstream_runner.fit_outer_fold(fold, eligible, inner, 2)
        test = sorted((r for r in eligible if r["outer_fold"] == fold), key=lambda r: r["canonical_observation_id"])
        if set(predictions) != {r["canonical_observation_id"] for r in test}:
            raise ValueError(f"fold_prediction_id_mismatch:{fold}")
        for row in test:
            rid, pred = row["canonical_observation_id"], predictions[row["canonical_observation_id"]]
            attempts.append({"canonical_observation_id": rid, "source_id": row["source_id"],
                             "outer_fold": fold, "group_id": row["unified_leakage_group_id"],
                             "logical_input_tier": targets[rid]["logical_input_tier"],
                             "actual_seconds": row["target_seconds"], "predicted_seconds": pred,
                             "absolute_error_seconds": abs(row["target_seconds"] - pred),
                             "selected_degree": degree, "status": "predicted"})
        folds.append(record)
        print(json.dumps({"fold": fold, "n_test": len(test), "degree": degree}), flush=True)
    if len(attempts) != 7350 or len({r["canonical_observation_id"] for r in attempts}) != 7350:
        raise ValueError("oof_denominator_mismatch")
    write(OUT / "predictions.csv", attempts)
    metrics = []
    for scope in ["all"] + sorted({r["source_id"] for r in attempts}) + sorted({r["logical_input_tier"] for r in attempts}):
        subset = attempts if scope == "all" else [r for r in attempts if scope in (r["source_id"], r["logical_input_tier"])]
        y = np.asarray([r["actual_seconds"] for r in subset], dtype=float)
        p = np.asarray([r["predicted_seconds"] for r in subset], dtype=float)
        metrics.append({"scope": scope, "n": len(subset), "mae_seconds": float(np.mean(np.abs(y-p))),
                        "medae_seconds": float(np.median(np.abs(y-p))),
                        "r2_seconds": float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.ptp(y)>0 else "",
                        "log1p_mae": float(np.mean(np.abs(np.log1p(y)-np.log1p(np.maximum(p,0)))))})
    write(OUT / "metrics.csv", metrics)
    (OUT / "fold_selection.json").write_text(json.dumps(folds, indent=2, sort_keys=True) + "\n")
    manifest = {"status": "complete", "method_id": "qonductor_feature_polynomial_common_panel",
                "reader_label": "Qonductor code-faithful feature polynomial, common-panel adaptation",
                "fidelity_class": "adaptation", "assigned_rows": 7350, "predicted_rows": len(attempts),
                "feature_order": FEATURE_NAMES, "selection": "highest mean raw-seconds R2 over four inner grouped folds; exact tie lower degree",
                "target_transform": "raw seconds, unchanged", "outer_folds": 5, "inner_folds": 4,
                "threads": 2, "device": "CPU", "python": sys.version, "platform": platform.platform(),
                "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in
                                 (PANEL / "manifest.json", PANEL / "panel.csv", TARGETS, OUTER, INNER,
                                  FEATURES, Path(__file__), ROOT / "benchmark_v1/scripts/run_qonductor_native_unified.py",
                                  ROOT / "benchmark_v1/protocol/qonductor_native_features.json")},
                "output_hashes": {"predictions.csv": sha(OUT / "predictions.csv"),
                                  "metrics.csv": sha(OUT / "metrics.csv"),
                                  "fold_selection.json": sha(OUT / "fold_selection.json")}}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "assigned": 7350, "predicted": len(attempts),
                      "metrics": metrics[0]}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
