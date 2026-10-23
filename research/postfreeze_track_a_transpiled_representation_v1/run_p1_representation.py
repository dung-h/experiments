#!/usr/bin/env python3
"""Nested, group-safe frozen-P1 representation comparison.

This is deliberately a bounded tabular comparison: same frozen rows/splits and
same learner within every representation view.  It does not construct or fit a
GNN and it never opens a circuit/QASM file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/postfreeze_track_a_transpiled_representation_v1"
FEATURES = ART / "features"
PANEL = ROOT / "artifacts/real_qpu/qualified_logical_panel/panel.csv"
OUTER = ROOT / "artifacts/real_qpu/qualified_logical_panel/outer_splits.csv"
INNER = ROOT / "artifacts/real_qpu/qualified_logical_panel/inner_splits.csv"
PRIMARY = ART / "p1_manifest.json"
HGB = ART / "p1_secondary_hgb_manifest.json"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, values: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader(); writer.writerows(values)


def metric(rows_: list[dict[str, Any]]) -> dict[str, float | int]:
    y = np.asarray([float(r["target_seconds"]) for r in rows_], dtype=float)
    p = np.asarray([float(r["prediction_seconds"]) for r in rows_], dtype=float)
    e = p - y
    absolute = np.abs(e)
    group_values = defaultdict(list)
    source_values = defaultdict(list)
    for record, error in zip(rows_, absolute):
        group_values[record["group_id"]].append(float(error)); source_values[record["source_id"]].append(float(error))
    ylog, plog = np.log1p(y), np.log1p(np.maximum(p, -0.999999999))
    denom = float(np.sum((y - y.mean()) ** 2)); log_denom = float(np.sum((ylog-ylog.mean())**2))
    return {"rows": len(rows_), "groups": len(group_values), "pooled_mae_seconds": float(absolute.mean()),
            "group_balanced_mae_seconds": float(np.mean([np.mean(v) for v in group_values.values()])),
            "source_balanced_mae_seconds": float(np.mean([np.mean(v) for v in source_values.values()])),
            "medae_seconds": float(np.median(absolute)), "rmse_seconds": float(np.sqrt(np.mean(e**2))),
            "raw_r2": float(1-np.sum(e**2)/denom) if denom else float("nan"),
            "log1p_r2": float(1-np.sum((plog-ylog)**2)/log_denom) if log_denom else float("nan"),
            "p90_absolute_error_seconds": float(np.quantile(absolute,.9)), "p95_absolute_error_seconds": float(np.quantile(absolute,.95)),
            "p99_absolute_error_seconds": float(np.quantile(absolute,.99)), "max_absolute_error_seconds": float(absolute.max()),
            "mean_signed_error_seconds": float(e.mean()), "negative_prediction_count": int(np.sum(p<0))}


def transform_columns(view: str, source: dict[str, str], static: dict[str, str], timing: dict[str, str]) -> tuple[list[str], list[float]]:
    source_keys = [f"g{i:02d}" for i in range(51)]
    static_keys = [k for k in static if k not in {"canonical_observation_id", "log1p_shots"}]
    values: list[tuple[str, float, bool]] = []
    if view in {"S0", "H0", "H1"}:
        for index, key in enumerate(source_keys):
            values.append(("source_"+key, float(source[key]), index <= 45))
    if view in {"P0", "P1", "H0", "H1"}:
        values.extend(("physical_"+key, float(static[key]), True) for key in static_keys)
    if view in {"P1", "H1"}:
        values.append(("timing_nominal_single_shot_duration_seconds", float(timing["nominal_single_shot_duration_seconds"]), True))
    values.append(("context_log1p_shots", float(source["log1p_shots"]), False))
    names = [v[0] for v in values]
    numeric = [math.log1p(v) if is_count_or_duration else v for _, v, is_count_or_duration in values]
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"nonfinite_feature:{view}")
    return names, numeric


def model_for(learner: str, candidate: Any):
    if learner == "ridge":
        return Ridge(alpha=float(candidate))
    return HistGradientBoostingRegressor(**candidate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner", choices=("ridge", "hgb"), required=True)
    args = parser.parse_args()
    primary = json.loads(PRIMARY.read_text())
    if args.learner == "ridge":
        candidates = primary["primary_learner"]["alpha_grid"]
    else:
        candidates = json.loads(HGB.read_text())["candidates"]
    feature_hashes = json.loads((FEATURES / "feature_hashes.json").read_text())
    for relative, expected in feature_hashes.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("feature_hash_mismatch:" + relative)

    panel = {r["canonical_observation_id"]: r for r in rows(PANEL)}
    outer = {r["canonical_observation_id"]: int(r["outer_fold"]) for r in rows(OUTER)}
    inner = {(r["canonical_observation_id"], int(r["outer_fold"])): int(r["inner_fold"]) for r in rows(INNER)}
    source = {r["canonical_observation_id"]: r for r in rows(FEATURES / "source_static.csv")}
    static = {r["canonical_observation_id"]: r for r in rows(FEATURES / "transpiled_static.csv")}
    timing = {r["canonical_observation_id"]: r for r in rows(FEATURES / "timing_features.csv")}
    ids = sorted(panel)
    if not (len(ids)==4515 and set(ids)==set(outer)==set(source)==set(static)==set(timing)):
        raise RuntimeError("input_identity_mismatch")

    views = ("S0", "P0", "P1", "H0", "H1")
    output = ART / "p1_representation" / args.learner
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    feature_receipt: dict[str, list[str]] = {}
    for view in views:
        names, _ = transform_columns(view, source[ids[0]], static[ids[0]], timing[ids[0]])
        feature_receipt[view] = names
        all_x = np.asarray([transform_columns(view, source[cid], static[cid], timing[cid])[1] for cid in ids], dtype=float)
        id_pos = {cid: index for index, cid in enumerate(ids)}
        for fold in range(5):
            test_ids = [cid for cid in ids if outer[cid] == fold]
            train_ids = [cid for cid in ids if outer[cid] != fold]
            inner_values = {cid: inner[(cid, fold)] for cid in train_ids}
            if set(inner_values.values()) != {0,1,2,3}:
                raise RuntimeError("inner_fold_support_failure")
            candidate_scores: list[tuple[Any,float]] = []
            for candidate in candidates:
                validation_rows: list[dict[str, Any]] = []
                for validation_fold in range(4):
                    fit = [cid for cid in train_ids if inner_values[cid] != validation_fold]
                    validation = [cid for cid in train_ids if inner_values[cid] == validation_fold]
                    fit_ix, val_ix = [id_pos[c] for c in fit], [id_pos[c] for c in validation]
                    scaler = StandardScaler().fit(all_x[fit_ix])
                    yfit = np.log1p(np.asarray([float(panel[c]["target_seconds"]) for c in fit]))
                    fitted = model_for(args.learner, candidate).fit(scaler.transform(all_x[fit_ix]), yfit)
                    prediction = np.expm1(fitted.predict(scaler.transform(all_x[val_ix])))
                    validation_rows.extend({"group_id": panel[c]["group_id"], "source_id": panel[c]["source_id"], "target_seconds": panel[c]["target_seconds"], "prediction_seconds": p} for c,p in zip(validation,prediction))
                candidate_scores.append((candidate, metric(validation_rows)["group_balanced_mae_seconds"]))
            best_candidate, best_score = min(candidate_scores, key=lambda pair: pair[1])
            selection_rows.extend({"learner": args.learner, "view": view, "outer_fold": fold, "candidate": json.dumps(candidate, sort_keys=True), "inner_group_balanced_mae_seconds": score, "selected": int(candidate == best_candidate)} for candidate, score in candidate_scores)
            fit_ix, test_ix = [id_pos[c] for c in train_ids], [id_pos[c] for c in test_ids]
            scaler = StandardScaler().fit(all_x[fit_ix])
            yfit = np.log1p(np.asarray([float(panel[c]["target_seconds"]) for c in train_ids]))
            fitted = model_for(args.learner, best_candidate).fit(scaler.transform(all_x[fit_ix]), yfit)
            prediction = np.expm1(fitted.predict(scaler.transform(all_x[test_ix])))
            prediction_rows.extend({"canonical_observation_id": c, "learner": args.learner, "view": view, "outer_fold": fold,
                                    "group_id": panel[c]["group_id"], "source_id": panel[c]["source_id"], "backend": panel[c]["backend"],
                                    "shots": panel[c]["shots"], "target_seconds": panel[c]["target_seconds"], "prediction_seconds": p,
                                    "selected_candidate": json.dumps(best_candidate, sort_keys=True)} for c,p in zip(test_ids,prediction))
            print(f"learner={args.learner} view={view} fold={fold} selected={best_candidate} inner_gb_mae={best_score:.6f}", flush=True)
    write_csv(output / "outer_predictions.csv", prediction_rows, list(prediction_rows[0]))
    write_csv(output / "hyperparameter_receipts.csv", selection_rows, list(selection_rows[0]))
    (output / "feature_names.json").write_text(json.dumps(feature_receipt, indent=2, sort_keys=True) + "\n")
    receipt = {"status": "COMPLETE_UNAGGREGATED", "learner": args.learner, "rows": len(prediction_rows), "views": list(views),
               "per_view_expected_rows": 4515, "panel_sha256": sha(PANEL), "p1_manifest_sha256": sha(PRIMARY),
               "python": sys.version, "platform": platform.platform()}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
