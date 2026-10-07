#!/usr/bin/env python3
"""Project pinned analytical outputs to the panel and refit fold-safe calibration."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "artifacts/real_qpu/common_panel"
TARGETS = PANEL / "targets.csv"
SPLIT = PANEL / "outer_splits.csv"
ARCHIVE = ROOT / "artifacts/benchmark_v3/real_qpu/archived_analytical_wave_v3_20260930/attempts.csv"
SCHOLTEN = ROOT / "artifacts/real_qpu/scholten_nominal/attempts.csv"
OUT = PANEL / "analytical"
BASES = {
    "qcre_snapshot_critical_path": "scheduled_critical_path_seconds",
    "qcre_shot_scaled_schedule_seconds": "shot_scaled_scheduled_seconds",
    "qiskit_estimate_duration_snapshot": "scheduled_single_shot_seconds",
    "qiskit_shot_scaled_schedule_seconds": "shot_scaled_scheduled_seconds",
    "hyb_hanas_one_circuit_effective_cost_adaptation": "one_circuit_effective_cost_seconds",
    "hyb_hanas_shot_linear_effective_seconds": "shot_linear_effective_cost_seconds",
    "scholten_unified_nominal_throughput": "nominal_throughput_proxy_seconds",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, values):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        prior_path = OUT / "run_manifest.json"
        if not prior_path.is_file():
            raise SystemExit(f"refuse to overwrite unreceipted output {OUT}")
    target = {r["canonical_observation_id"]: r for r in read(TARGETS)}
    panel_ids = set(target)
    folds = {r["canonical_observation_id"]: int(r["outer_fold"]) for r in read(SPLIT)}
    archive = read(ARCHIVE)
    nominal = read(SCHOLTEN)
    by_method = defaultdict(dict)
    for row in archive:
        method = row["method_id"]
        if method in BASES and row["canonical_row_id"] in panel_ids:
            by_method[method][row["canonical_row_id"]] = row
    for row in nominal:
        rid = row["canonical_row_id"]
        if rid in panel_ids:
            by_method["scholten_unified_nominal_throughput"][rid] = row
    raw = {}
    for method, output_clock in BASES.items():
        if set(by_method[method]) - panel_ids:
            raise ValueError(f"unexpected_analytical_identity:{method}")
        for rid, row in by_method[method].items():
            status = row.get("status", "")
            pred_field = "prediction" if method.startswith("scholten_") else "prediction_seconds"
            value = row.get(pred_field, "")
            if status in {"predicted", "prediction_produced"} and value:
                value = float(value)
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"invalid_raw_prediction:{method}:{rid}")
                raw[(method, rid)] = value
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    calibration_records = []
    for method, output_clock in BASES.items():
        for fold in range(5):
            train = sorted(rid for rid in panel_ids if folds[rid] != fold and (method, rid) in raw)
            test = sorted(rid for rid in panel_ids if folds[rid] == fold)
            if len(train) < 2:
                continue
            y_train = np.asarray([float(target[rid]["target_seconds"]) for rid in train])
            x_train = np.asarray([raw[(method, rid)] for rid in train])
            models = {}
            for variant in ("raw", "outer_train_affine", "outer_train_log_affine"):
                if variant == "outer_train_affine":
                    model = LinearRegression().fit(x_train.reshape(-1, 1), y_train)
                    models[variant] = (model, "calibrated_observed_service_seconds")
                elif variant == "outer_train_log_affine":
                    model = LinearRegression().fit(np.log1p(x_train).reshape(-1, 1), np.log1p(y_train))
                    models[variant] = (model, "calibrated_observed_service_seconds")
                else:
                    models[variant] = (None, output_clock)
                model = models[variant][0]
                calibration_records.append({"base_method_id": method, "variant": variant,
                                            "outer_fold": fold, "fit_rows": len(train),
                                            "fit_id_sha256": hashlib.sha256("\n".join(train).encode()).hexdigest(),
                                            "slope": "" if model is None else float(model.coef_[0]),
                                            "intercept": "" if model is None else float(model.intercept_),
                                            "transformed_fit": variant == "outer_train_log_affine",
                                            "fit_rows_from_outer_train_only": True})
            for variant, (model, clock) in models.items():
                for rid in test:
                    x = raw.get((method, rid))
                    status, reason, pred = "unavailable", "source_method_unavailable", ""
                    if x is not None:
                        if variant == "raw":
                            pred, status, reason = x, "predicted", ""
                        elif variant == "outer_train_affine":
                            pred = max(0.0, float(model.predict([[x]])[0]))
                            status, reason = "predicted", ""
                        else:
                            pred = max(0.0, float(np.expm1(model.predict([[np.log1p(x)]])[0])))
                            status, reason = "predicted", ""
                    results.append({"canonical_observation_id": rid, "method_id": f"{method}__{variant}",
                                    "base_method_id": method, "variant": variant, "source_id": target[rid]["source_id"],
                                    "outer_fold": fold, "evaluation_target_clock": "archived_observed_service_execution_time",
                                    "method_output_clock": clock, "actual_seconds": target[rid]["target_seconds"],
                                    "predicted_seconds": pred, "status": status, "terminal_reason": reason})
    if len(results) != 7350 * len(BASES) * 3:
        raise ValueError(f"attempt_count_mismatch:{len(results)}")
    write(OUT / "attempts.csv", results)
    write(OUT / "calibration_fits.csv", calibration_records)
    metrics = []
    for method in sorted({r["method_id"] for r in results}):
        subset = [r for r in results if r["method_id"] == method]
        ok = [r for r in subset if r["status"] == "predicted"]
        y = np.asarray([float(r["actual_seconds"]) for r in ok])
        p = np.asarray([float(r["predicted_seconds"]) for r in ok])
        metrics.append({"method_id": method, "assigned_rows": 7350, "predicted_rows": len(ok),
                        "coverage": len(ok) / 7350, "mae_seconds": float(np.mean(np.abs(y-p))) if len(y) else "",
                        "medae_seconds": float(np.median(np.abs(y-p))) if len(y) else "",
                        "r2_seconds": float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.ptp(y)>0 else ""})
    write(OUT / "metrics.csv", metrics)
    manifest = {"status": "complete", "panel_rows": 7350, "method_variants": len(BASES)*3,
                "attempt_rows": len(results), "all_variants_use_same_assigned_rows": True,
                "calibration_fit_rows": "successful base-method predictions in each outer-train only",
                "prediction_floor_seconds": 0, "threads": 1,
                "input_hashes": {str(p.relative_to(ROOT)): digest(p) for p in
                                 (PANEL / "manifest.json", PANEL / "panel.csv", TARGETS, SPLIT, ARCHIVE, SCHOLTEN, Path(__file__))},
                "output_hashes": {"attempts.csv": digest(OUT / "attempts.csv"),
                                  "calibration_fits.csv": digest(OUT / "calibration_fits.csv"),
                                  "metrics.csv": digest(OUT / "metrics.csv")}}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "attempts": len(results), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
