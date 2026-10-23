#!/usr/bin/env python3
"""Aggregate nested group-subsampled transfer predictions without refitting."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts/postfreeze_track_a_transfer_sample_efficiency_v1"
OLD = REPO / "artifacts/postfreeze_track_a_transfer_v1"
SEEDS = (42, 1234, 31415)
TREATMENTS = ("M-S", "M-P", "G-S", "G-P")
FRACTION_KEYS = {10: 0.10, 25: 0.25, 50: 0.50}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_predictions():
    rows = []
    for path in sorted((OUT / "qpu_factorial").glob("fraction_*/subset_*/**/predictions.csv")):
        fraction = int(next(p for p in path.parts if p.startswith("fraction_")).split("_")[1]) / 100.0
        subset_seed = int(next(p for p in path.parts if p.startswith("subset_")).split("_")[1])
        for r in read_csv(path):
            r.update({"fraction": f"{fraction:.2f}", "subset_seed": str(subset_seed), "prediction_path": str(path.relative_to(REPO))})
            rows.append(r)
    # Frozen 100% A1 prediction rows are reused exactly; make the provenance explicit.
    for r in read_csv(OLD / "per_seed_predictions.csv"):
        r.update({"fraction": "1.00", "subset_seed": "FROZEN_A1", "prediction_path": str((OLD / "per_seed_predictions.csv").relative_to(REPO))})
        rows.append(r)
    return rows


def median_rows(rows):
    keys = {}
    for r in rows:
        key = (r["fraction"], r["subset_seed"], r["canonical_observation_id"], r["treatment"])
        keys.setdefault(key, []).append(r)
    out = []
    for key, vals in sorted(keys.items()):
        if len(vals) != 3:
            raise RuntimeError(f"seed_coverage:{key}:{len(vals)}")
        target = float(vals[0]["target_seconds"])
        preds = [float(v["prediction_seconds"]) for v in vals]
        if any(abs(float(v["target_seconds"]) - target) > 1e-9 for v in vals):
            raise RuntimeError(f"target_mismatch:{key}")
        out.append({
            "fraction": key[0], "subset_seed": key[1], "canonical_observation_id": key[2],
            "treatment": key[3], "architecture": vals[0]["architecture"],
            "initialization": vals[0]["initialization"], "source_id": vals[0]["source_id"],
            "group_id": vals[0]["group_id"], "outer_fold": vals[0]["outer_fold"],
            "target_seconds": target, "prediction_seconds": float(np.median(preds)),
            "seed_count": 3,
        })
    return out


def group_mae(rows):
    by = {}
    for r in rows:
        by.setdefault(r["group_id"], []).append(abs(float(r["prediction_seconds"]) - float(r["target_seconds"])))
    return float(np.mean([np.mean(v) for v in by.values()])), {g: float(np.mean(v)) for g, v in by.items()}


def bootstrap(deltas, group_ids, reps=10000, seed=42):
    rng = np.random.default_rng(seed)
    group_ids = np.asarray(sorted(group_ids))
    matrix = np.asarray([[deltas[g] for g in group_ids]], dtype=float)
    draws = rng.integers(0, len(group_ids), size=(reps, len(group_ids)))
    return matrix[:, draws].mean(axis=2).ravel()


def aggregate():
    seed_rows = load_predictions()
    if not seed_rows:
        raise RuntimeError("no_prediction_rows")
    write_csv(OUT / "all_seed_predictions.csv", seed_rows)
    subset_rows = []
    for path in sorted((OUT / "qpu_factorial").glob("fraction_*/subset_*/**/subset_receipt.json")):
        payload = json.loads(path.read_text())
        s = payload["subset"]
        subset_rows.append({"fraction": s["fraction"], "subset_seed": s["subset_seed"], "outer_fold": s["outer_fold"], "full_fit_group_count": s["full_fit_group_count"], "selected_fit_group_count": s["selected_fit_group_count"], "fit_observation_count": s["fit_observation_count"], "selected_group_sha256": s["selected_group_sha256"], "fit_id_sha256": s["fit_id_sha256"], "validation_id_sha256": s["validation_id_sha256"], "test_id_sha256": s["test_id_sha256"], "cell_path": str(path.parent.relative_to(REPO))})
    write_csv(OUT / "subset_manifest.csv", subset_rows)
    med = median_rows(seed_rows)
    write_csv(OUT / "median_predictions.csv", med)
    results = []
    paired = []
    intervals = []
    for fraction in sorted({r["fraction"] for r in med}, key=float):
        for treatment in TREATMENTS:
            subset = [r for r in med if r["fraction"] == fraction and r["treatment"] == treatment]
            mae, _ = group_mae(subset)
            pooled = float(np.mean([abs(float(r["prediction_seconds"]) - float(r["target_seconds"])) for r in subset]))
            results.append({"fraction": fraction, "subset_seed": "ALL", "treatment": treatment, "rows": len(subset), "groups": len({r["group_id"] for r in subset}), "mae_group_balanced": mae, "mae_pooled": pooled})
        for arch, left, right in (("MLP", "M-P", "M-S"), ("GRAPH_MLP", "G-P", "G-S")):
            per_subset = []
            for subset_seed in sorted({r["subset_seed"] for r in med if r["fraction"] == fraction}):
                a = [r for r in med if r["fraction"] == fraction and r["subset_seed"] == subset_seed and r["treatment"] == left]
                b = [r for r in med if r["fraction"] == fraction and r["subset_seed"] == subset_seed and r["treatment"] == right]
                aa = {r["canonical_observation_id"]: r for r in a}; bb = {r["canonical_observation_id"]: r for r in b}
                if set(aa) != set(bb): raise RuntimeError(f"paired_id_mismatch:{fraction}:{subset_seed}:{arch}")
                group_values = {}
                for oid, x in aa.items():
                    y = bb[oid]
                    delta = abs(float(x["prediction_seconds"]) - float(x["target_seconds"])) - abs(float(y["prediction_seconds"]) - float(y["target_seconds"]))
                    group_values.setdefault(x["group_id"], []).append(delta)
                per_subset.append({"subset_seed": subset_seed, "groups": {g: float(np.mean(v)) for g, v in group_values.items()}})
                for g, val in per_subset[-1]["groups"].items(): paired.append({"fraction": fraction, "subset_seed": subset_seed, "architecture": arch, "group_id": g, "delta_group_mae": val})
            common = set.intersection(*(set(v["groups"]) for v in per_subset))
            point = float(np.mean([np.mean([v["groups"][g] for g in common]) for v in per_subset]))
            rng = np.random.default_rng(42 + int(float(fraction) * 100) + (0 if arch == "MLP" else 1000))
            draws = []
            common = sorted(common)
            for _ in range(10000):
                sampled = rng.choice(common, size=len(common), replace=True)
                draws.append(float(np.mean([np.mean([v["groups"][g] for g in sampled]) for v in per_subset])))
            draws = np.asarray(draws)
            p = min(1.0, 2.0 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0))))
            intervals.append({"fraction": fraction, "architecture": arch, "left": left, "right": right, "point_group_balanced_mae_delta": point, "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975)), "bootstrap_replicates": 10000, "bootstrap_seed": 42 + int(float(fraction) * 100) + (0 if arch == "MLP" else 1000), "subset_replicates": len(per_subset), "two_sided_p": p})
    write_csv(OUT / "model_results.csv", results)
    write_csv(OUT / "paired_group_deltas.csv", paired)
    write_csv(OUT / "bootstrap_intervals.csv", intervals)
    manifest = {"status": "AGGREGATED", "median_prediction_rows": len(med), "seed_prediction_rows": len(seed_rows), "fractions": sorted({r["fraction"] for r in med}), "sha256_median_predictions": hashlib.sha256((OUT / "median_predictions.csv").read_bytes()).hexdigest()}
    (OUT / "aggregation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(aggregate(), indent=2, sort_keys=True))
