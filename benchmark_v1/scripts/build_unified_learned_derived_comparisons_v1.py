#!/usr/bin/env python3
"""Materialize immutable, paired derived comparisons for V3-large, V3, and polynomial OOF predictions.

This script only reads frozen OOF predictions; it performs no training or inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
OUTER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv"
LARGE = ROOT / "artifacts/benchmark_v3/real_qpu/unified_graph_v3_large_oof_20260930/predictions.csv"
LEARNED_REPORT = ROOT / "artifacts/benchmark_v3/real_qpu/unified_learned_methods_oof_v4r2/report.json"
SEED_REGISTRY = ROOT / "benchmark_v1/registry/seed_registry.json"
DEFAULT_OUT = ROOT / "artifacts/benchmark_v3/real_qpu/unified_learned_derived_comparisons_v1"
TARGET_CLOCK = "archived_observed_service_execution_time"
METHODS = {
    "v3_large": "mali_graph_architecture_v3_large",
    "v3": "mali_graph_architecture_v3",
    "polynomial": "unified_polynomial_v3",
}
PAIRINGS = (("large_vs_poly", "v3_large", "polynomial"), ("large_vs_v3", "v3_large", "v3"))
EXPECTED = {
    "canonical_n": 8767,
    "canonical_hash": "dff36af67d89253c933e4dbcc877d57b0fca2d29eac711c5dfdaadbfcafda322",
    "large_vs_poly_n": 8766,
    "large_vs_poly_hash": "e8c02117bfc0c0c4294312783678de0203aa56950b53ef5687aa38876c0dd0b4",
    "large_vs_v3_n": 8763,
    "large_vs_v3_hash": "a11cd7e1cbcbf07eef1ee5d88869bef4e232f2485f17635fe6f5de28ba091514",
    "v3_large_predictions_sha256": "3c179ae464fc25167ae0f1aca028cf5ff1117409dce9d15f6945e4f621420d2c",
}
TAIL_IDS = (
    "qonductor_single_circuit_ibm|row39",
    "qonductor_single_circuit_ibm|row40",
    "qonductor_single_circuit_ibm|row41",
)
UNAVAILABLE = "qonductor_single_circuit_ibm|row4477"
V3_UNAVAILABLE_REASON = "unavailable_resource_limit_graph_operation_count"
ROW4477_REASON = "dynamic_control_not_representable_by_declared_flat_dag_v3"
SOURCES = ("mali_real_qpu", "qonductor_single_circuit_ibm", "qpack_mcp")
CLOCK_FIELDS = {
    "evaluation_target_clock": TARGET_CLOCK,
    "method_output_clock": TARGET_CLOCK,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity_hash(ids: list[str]) -> str:
    material = "\n".join(sorted(ids)) + f"\ncount={len(ids)}\n"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if not rows and not fields:
        raise ValueError(f"refusing empty CSV without fields: {path}")
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def load_prediction_files(
    paths: list[Path],
    label: str,
    canonical: dict[str, dict[str, str]],
    outer: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in read_csv(path):
            cid = row["canonical_observation_id"]
            if cid in result:
                raise ValueError(f"duplicate {label} prediction: {cid}")
            if cid not in canonical or cid not in outer:
                raise ValueError(f"non-canonical {label} prediction: {cid}")
            if int(row["outer_fold"]) != int(outer[cid]["outer_fold"]):
                raise ValueError(f"outer-fold drift in {label}: {cid}")
            if not math.isclose(
                float(row["actual_seconds"]),
                float(canonical[cid]["target_seconds"]),
                rel_tol=1e-6,
                abs_tol=1e-5,
            ):
                raise ValueError(f"target drift in {label}: {cid}")
            if not math.isfinite(float(row["predicted_seconds"])):
                raise ValueError(f"non-finite {label} prediction: {cid}")
            result[cid] = row
    return result


def metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    y = np.array([float(r["actual_seconds"]) for r in rows], dtype=float)
    p = np.array([float(r["predicted_seconds"]) for r in rows], dtype=float)
    e = np.abs(y - p)
    return {
        "n": len(rows),
        "mae_seconds": float(e.mean()),
        "medae_seconds": float(np.median(e)),
        "log1p_mae": float(np.abs(np.log1p(np.maximum(y, 0)) - np.log1p(np.maximum(p, 0))).mean()),
        "r2_seconds": float(r2_score(y, p)),
        "spearman_rho": float(spearmanr(y, p).statistic),
        "p90_abs_error_seconds": float(np.quantile(e, 0.90)),
        "p99_abs_error_seconds": float(np.quantile(e, 0.99)),
        "max_abs_error_seconds": float(e.max()),
    }


def cluster_key(row: dict[str, str]) -> str:
    source = row["source_id"]
    if source == "mali_real_qpu":
        value = row["circuit_group_hash"]
        kind = "exact_logical_qasm"
    elif source == "qonductor_single_circuit_ibm":
        value = row["qasm_bytes_sha256"]
        kind = "exact_submitted_qasm"
    elif source == "qpack_mcp":
        value = row["workflow_group_hash"]
        kind = "workflow_id"
    else:
        raise ValueError(f"unknown source {source}")
    if not value:
        raise ValueError(f"missing {kind} cluster identity for {row['canonical_row_id']}")
    return f"{source}|{kind}|{value}"


def seed_for(pair_name: str, registry: dict) -> int:
    stream = registry["streams"]["bootstrap"]
    material = f"{registry['root_seed']}|{stream}|unified-learned-derived-comparisons-v1|{pair_name}|0"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def bootstrap(rows: list[dict[str, object]], pair_name: str, seed: int, builder_sha256: str) -> dict[str, float | int | str]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["bootstrap_cluster_key"])].append(row)
    keys = sorted(groups)
    if not keys:
        raise ValueError(f"{pair_name}: no bootstrap clusters")
    count = np.array([len(groups[k]) for k in keys], dtype=np.int64)
    delta_sum = np.array([sum(float(r["paired_error_delta_seconds"]) for r in groups[k]) for k in keys])
    win_sum = np.array([sum(int(r["candidate_win"]) for r in groups[k]) for k in keys])
    rng = np.random.default_rng(seed)
    mae_delta, wins = np.empty(1000), np.empty(1000)
    for i in range(1000):
        picked = rng.integers(0, len(keys), len(keys))
        denom = int(count[picked].sum())
        if denom == 0:
            raise ValueError(f"{pair_name}: empty bootstrap replicate")
        mae_delta[i] = delta_sum[picked].sum() / denom
        wins[i] = win_sum[picked].sum() / denom
    source_clusters = {source: sum(1 for k in keys if k.startswith(source + "|")) for source in SOURCES}
    return {
        "pair_name": pair_name,
        "stream": "20260925-bootstrap",
        "seed": seed,
        "replicates": 1000,
        "clusters": len(keys),
        "mali_real_qpu_clusters": source_clusters["mali_real_qpu"],
        "qonductor_single_circuit_ibm_clusters": source_clusters["qonductor_single_circuit_ibm"],
        "qpack_mcp_clusters": source_clusters["qpack_mcp"],
        "percentile_rule": "numpy.quantile_linear_0.025_0.975",
        "mae_delta_95pct_low_seconds": float(np.quantile(mae_delta, 0.025)),
        "mae_delta_95pct_high_seconds": float(np.quantile(mae_delta, 0.975)),
        "win_rate_95pct_low": float(np.quantile(wins, 0.025)),
        "win_rate_95pct_high": float(np.quantile(wins, 0.975)),
        "builder_sha256": builder_sha256,
        "macro_conclusion_authorized": False,
        "note": "S49 grouped bootstrap on complete source-specific clusters; source-balanced macro MAE is diagnostic only",
    }


def own_success_rows(short: str, pred: dict[str, dict[str, str]], canonical: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for cid, row in pred[short].items():
        rows.append(
            {
                "canonical_observation_id": cid,
                "source_id": canonical[cid]["source_id"],
                "actual_seconds": float(canonical[cid]["target_seconds"]),
                "predicted_seconds": float(row["predicted_seconds"]),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    learned = json.loads(LEARNED_REPORT.read_text(encoding="utf-8"))
    canonical_rows = read_csv(CANONICAL)
    canonical = {r["canonical_row_id"]: r for r in canonical_rows}
    outer = {r["canonical_observation_id"]: r for r in read_csv(OUTER)}
    if len(canonical) != EXPECTED["canonical_n"] or set(canonical) != set(outer):
        raise ValueError("canonical and outer split identity mismatch")
    if identity_hash(list(canonical)) != EXPECTED["canonical_hash"]:
        raise ValueError("canonical identity hash mismatch")
    if sha(LARGE) != EXPECTED["v3_large_predictions_sha256"]:
        raise ValueError("V3-large predictions SHA-256 drifted")
    poly_paths = [ROOT / p for p in learned["input_prediction_files"][METHODS["polynomial"]]]
    v3_paths = [ROOT / p for p in learned["input_prediction_files"][METHODS["v3"]]]
    pred = {
        "v3_large": load_prediction_files([LARGE], "v3_large", canonical, outer),
        "polynomial": load_prediction_files(poly_paths, "polynomial", canonical, outer),
        "v3": load_prediction_files(v3_paths, "v3", canonical, outer),
    }
    if set(pred["v3_large"]) != set(canonical) - {UNAVAILABLE}:
        raise ValueError("V3-large coverage is not canonical minus row4477")
    if set(pred["v3"]) != set(canonical) - set(TAIL_IDS) - {UNAVAILABLE}:
        raise ValueError("V3 coverage is not expected eligible universe")
    if set(pred["polynomial"]) != set(canonical):
        raise ValueError("polynomial coverage is not canonical universe")
    for cid in TAIL_IDS:
        if cid not in pred["v3_large"]:
            raise ValueError(f"V3-large missing tail diagnostic prediction: {cid}")
        if cid in pred["v3"]:
            raise ValueError(f"V3 unexpectedly predicted tail ID {cid}")
    if UNAVAILABLE in pred["v3_large"] or UNAVAILABLE in pred["v3"]:
        raise ValueError("row4477 must not be spliced into V3 or V3-large")

    cluster_by_id = {cid: cluster_key(row) for cid, row in canonical.items()}
    cluster_folds: dict[str, set[int]] = defaultdict(set)
    for cid, key in cluster_by_id.items():
        cluster_folds[key].add(int(outer[cid]["outer_fold"]))
    leakage = {key: folds for key, folds in cluster_folds.items() if len(folds) != 1}
    if leakage:
        raise ValueError(f"bootstrap clusters leak across outer folds: {sorted(leakage)[:3]}")
    if len(cluster_folds) != 2320:
        raise ValueError(f"canonical cluster count {len(cluster_folds)} != 2320")

    out.mkdir(parents=True)
    coverage_rows = []
    for short, method in METHODS.items():
        coverage_rows.append(
            {
                "method_key": short,
                "method_id": method,
                **CLOCK_FIELDS,
                "attempted_observations": 8767,
                "successful_predictions": len(pred[short]),
                "terminal_unavailable": 8767 - len(pred[short]),
                "coverage_fraction": len(pred[short]) / 8767,
            }
        )
    write_csv(out / "coverage.csv", coverage_rows)

    unavailable_rows: list[dict[str, object]] = []
    unavailable_rows.append(
        {
            "canonical_observation_id": UNAVAILABLE,
            "method_key": "v3_large",
            "method_id": METHODS["v3_large"],
            "source_id": canonical[UNAVAILABLE]["source_id"],
            "outer_fold": outer[UNAVAILABLE]["outer_fold"],
            **CLOCK_FIELDS,
            "reason": ROW4477_REASON,
        }
    )
    for cid in (*TAIL_IDS, UNAVAILABLE):
        unavailable_rows.append(
            {
                "canonical_observation_id": cid,
                "method_key": "v3",
                "method_id": METHODS["v3"],
                "source_id": canonical[cid]["source_id"],
                "outer_fold": outer[cid]["outer_fold"],
                **CLOCK_FIELDS,
                "reason": ROW4477_REASON if cid == UNAVAILABLE else V3_UNAVAILABLE_REASON,
            }
        )
    write_csv(out / "unavailable.csv", unavailable_rows)

    own_rows: list[dict[str, object]] = []
    for short, method in METHODS.items():
        rows = own_success_rows(short, pred, canonical)
        overall = metrics(rows)
        own_rows.append(
            {
                "method_key": short,
                "method_id": method,
                "source_id": "all_successful",
                "role": "own_success",
                **CLOCK_FIELDS,
                **overall,
                "metric_role": "own_success_not_a_paired_ranking",
            }
        )
        source_maes = []
        for source in SOURCES:
            sr = [r for r in rows if r["source_id"] == source]
            m = metrics(sr)
            source_maes.append(m["mae_seconds"])
            own_rows.append(
                {
                    "method_key": short,
                    "method_id": method,
                    "source_id": source,
                    "role": "own_success",
                    **CLOCK_FIELDS,
                    **m,
                    "metric_role": "own_success_source_stratified",
                }
            )
        own_rows.append(
            {
                "method_key": short,
                "method_id": method,
                "source_id": "equal_weight_three_source_macro_diagnostic",
                "role": "own_success",
                **CLOCK_FIELDS,
                "n": 3,
                "mae_seconds": float(np.mean(source_maes)),
                "medae_seconds": "",
                "log1p_mae": "",
                "r2_seconds": "",
                "spearman_rho": "",
                "p90_abs_error_seconds": "",
                "p99_abs_error_seconds": "",
                "max_abs_error_seconds": "",
                "metric_role": "source_balanced_macro_diagnostic_not_ranking",
            }
        )
    write_csv(out / "own_success_metrics.csv", own_rows)

    pair_metric_rows, source_rows, delta_rows, bootstraps = [], [], [], []
    builder_sha256 = sha(Path(__file__))
    registry = json.loads(SEED_REGISTRY.read_text(encoding="utf-8"))
    for pair_name, candidate, reference in PAIRINGS:
        ids = sorted(set(pred[candidate]) & set(pred[reference]))
        expected_n, expected_hash = (
            (EXPECTED["large_vs_poly_n"], EXPECTED["large_vs_poly_hash"])
            if pair_name == "large_vs_poly"
            else (EXPECTED["large_vs_v3_n"], EXPECTED["large_vs_v3_hash"])
        )
        if len(ids) != expected_n or identity_hash(ids) != expected_hash:
            raise ValueError(f"{pair_name} identity mismatch")
        pair_rows = []
        for cid in ids:
            actual = float(canonical[cid]["target_seconds"])
            cp = float(pred[candidate][cid]["predicted_seconds"])
            rp = float(pred[reference][cid]["predicted_seconds"])
            ce, re = abs(actual - cp), abs(actual - rp)
            row = {
                "pair_name": pair_name,
                "canonical_observation_id": cid,
                "source_id": canonical[cid]["source_id"],
                "outer_fold": int(outer[cid]["outer_fold"]),
                "bootstrap_cluster_key": cluster_by_id[cid],
                **CLOCK_FIELDS,
                "actual_seconds": actual,
                "candidate_predicted_seconds": cp,
                "reference_predicted_seconds": rp,
                "candidate_absolute_error_seconds": ce,
                "reference_absolute_error_seconds": re,
                "paired_error_delta_seconds": ce - re,
                "candidate_win": int(ce < re),
                "tie": int(ce == re),
                "candidate_loss": int(ce > re),
            }
            pair_rows.append(row)
            delta_rows.append(row)
        cand_m = metrics([{**r, "predicted_seconds": r["candidate_predicted_seconds"]} for r in pair_rows])
        ref_m = metrics([{**r, "predicted_seconds": r["reference_predicted_seconds"]} for r in pair_rows])
        paired_mae_delta = float(np.mean([r["paired_error_delta_seconds"] for r in pair_rows]))
        paired_medae_delta = float(np.median([r["paired_error_delta_seconds"] for r in pair_rows]))
        win_rate = float(np.mean([r["candidate_win"] for r in pair_rows]))
        tie_rate = float(np.mean([r["tie"] for r in pair_rows]))
        loss_rate = float(np.mean([r["candidate_loss"] for r in pair_rows]))
        for role, short, m in (("candidate", candidate, cand_m), ("reference", reference, ref_m)):
            pair_metric_rows.append(
                {
                    "pair_name": pair_name,
                    "role": role,
                    "method_key": short,
                    "method_id": METHODS[short],
                    **CLOCK_FIELDS,
                    **m,
                    "paired_mae_delta_seconds": paired_mae_delta,
                    "paired_medae_delta_seconds": paired_medae_delta,
                    "win_rate": win_rate,
                    "tie_rate": tie_rate,
                    "loss_rate": loss_rate,
                    "row_set_n": expected_n,
                    "row_set_hash": expected_hash,
                }
            )
        for source in SOURCES:
            sr = [r for r in pair_rows if r["source_id"] == source]
            source_delta = float(np.mean([r["paired_error_delta_seconds"] for r in sr]))
            for role, short, field in (
                ("candidate", candidate, "candidate_predicted_seconds"),
                ("reference", reference, "reference_predicted_seconds"),
            ):
                source_rows.append(
                    {
                        "pair_name": pair_name,
                        "source_id": source,
                        "role": role,
                        "method_key": short,
                        "method_id": METHODS[short],
                        **CLOCK_FIELDS,
                        **metrics([{**r, "predicted_seconds": r[field]} for r in sr]),
                        "paired_mae_delta_seconds": source_delta,
                        "metric_role": "source_stratified_paired",
                    }
                )
        for role, short in (("candidate", candidate), ("reference", reference)):
            vals = [
                x["mae_seconds"]
                for x in source_rows
                if x["pair_name"] == pair_name and x["role"] == role and x["source_id"] in SOURCES
            ]
            source_rows.append(
                {
                    "pair_name": pair_name,
                    "source_id": "equal_weight_three_source_macro_diagnostic",
                    "role": role,
                    "method_key": short,
                    "method_id": METHODS[short],
                    **CLOCK_FIELDS,
                    "n": 3,
                    "mae_seconds": float(np.mean(vals)),
                    "medae_seconds": "",
                    "log1p_mae": "",
                    "r2_seconds": "",
                    "spearman_rho": "",
                    "p90_abs_error_seconds": "",
                    "p99_abs_error_seconds": "",
                    "max_abs_error_seconds": "",
                    "paired_mae_delta_seconds": "",
                    "metric_role": "source_balanced_macro_diagnostic_not_ranking",
                }
            )
        bootstraps.append(bootstrap(pair_rows, pair_name, seed_for(pair_name, registry), builder_sha256))
    write_csv(out / "pair_metrics.csv", pair_metric_rows)
    write_csv(out / "source_stratified.csv", source_rows)
    write_csv(out / "paired_deltas.csv", delta_rows)
    write_csv(out / "bootstrap.csv", bootstraps)

    tail_rows = []
    for cid in TAIL_IDS:
        r = pred["v3_large"][cid]
        tail_rows.append(
            {
                "canonical_observation_id": cid,
                "source_id": canonical[cid]["source_id"],
                "method_id": METHODS["v3_large"],
                **CLOCK_FIELDS,
                "actual_seconds": r["actual_seconds"],
                "predicted_seconds": r["predicted_seconds"],
                "absolute_error_seconds": r["absolute_error_seconds"],
                "diagnostic_label": "representable but inaccurate extrapolation",
                "not_unavailable": True,
            }
        )
    write_csv(out / "tail_diagnostics.csv", tail_rows)

    source_paths = {
        "canonical": CANONICAL,
        "outer_split": OUTER,
        "v3_large_predictions": LARGE,
        "learned_methods_report": LEARNED_REPORT,
        "seed_registry": SEED_REGISTRY,
        **{f"polynomial_fold_{i}": p for i, p in enumerate(poly_paths)},
        **{f"v3_fold_{i}": p for i, p in enumerate(v3_paths)},
    }
    source_hashes = {name: {"path": rel(path), "sha256": sha(path)} for name, path in source_paths.items()}
    source_hashes["builder"] = {"path": rel(Path(__file__)), "sha256": builder_sha256}
    (out / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "artifact_id": "unified_learned_derived_comparisons_v1",
        "status": "PASS",
        "no_training_or_inference": True,
        "git_mutated": False,
        "five_fold_training_authorized": False,
        "evaluation_target_clock": TARGET_CLOCK,
        "method_output_clock": TARGET_CLOCK,
        "canonical": EXPECTED,
        "pairings": [{"pair_name": n, "candidate": METHODS[c], "reference": METHODS[r]} for n, c, r in PAIRINGS],
        "bootstrap": bootstraps,
        "tail_diagnostic_label": "representable but inaccurate extrapolation",
        "unavailable_row": UNAVAILABLE,
        "s49": "1,000 grouped bootstrap replicates on stream 20260925-bootstrap",
        "source_macro": "diagnostic, not a ranking claim",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md = """# Unified learned derived comparisons v1

This stable derived artifact compares frozen out-of-fold predictions only; it performs no training or inference. The two comparisons are V3-large versus polynomial on 8,766 common rows and V3-large versus V3 on 8,763 common rows. Coverage remains on the 8,767-row canonical envelope. Both clocks are `archived_observed_service_execution_time`.

Paired deltas are candidate absolute error minus reference absolute error, so negative favours the candidate (always V3-large). S49 grouped bootstrap uses 1,000 complete source-specific clusters and the registered `20260925-bootstrap` stream. Source-balanced three-source macro MAE values are diagnostics, not cross-source ranking claims.

Rows `qonductor_single_circuit_ibm|row39`, `row40`, and `row41` are representable but inaccurate extrapolation for V3-large; they are predicted, not unavailable. Row `qonductor_single_circuit_ibm|row4477` is unavailable for V3-large and V3 because dynamic control is not representable by the declared flat DAG. V3 is also unavailable on the three tail IDs due to the graph operation-count resource limit. Unavailable rows are never scored as zero error and are not spliced into V3 or V3-large.
"""
    (out / "REPORT.md").write_text(report_md, encoding="utf-8")
    outputs = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    manifest = {
        "artifact_id": report["artifact_id"],
        "status": "PASS",
        "no_training_or_inference": True,
        "git_mutated": False,
        "five_fold_training_authorized": False,
        "evaluation_target_clock": TARGET_CLOCK,
        "method_output_clock": TARGET_CLOCK,
        "contract": {
            "S49": "1,000 grouped bootstrap replicates",
            "S49_stream": "20260925-bootstrap",
            "S77": "Spearman authorized",
        },
        "inputs": source_hashes,
        "identity_contract": EXPECTED,
        "cluster_contract": {
            "mali_real_qpu": "exact logical-QASM (circuit_group_hash)",
            "qonductor_single_circuit_ibm": "exact submitted-QASM (qasm_bytes_sha256)",
            "qpack_mcp": "workflow ID (workflow_group_hash)",
            "canonical_clusters": 2320,
        },
        "bootstrap": bootstraps,
        "outputs": outputs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": rel(out), "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
