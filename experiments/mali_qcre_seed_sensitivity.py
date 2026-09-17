#!/usr/bin/env python3
"""Run a small transpiler-seed sensitivity check for the Ma--Li proxy.

Seed 1234 is read from the committed full feature artifact.  Additional seeds
retranspile the same 340 QASM rows with the same FakeBackend and optimization
level, then use the fixed QASM-hash grouped folds from the final validation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np

try:
    from mali_qcre_proxy import backend_metadata, build_rows, load_labels, resolve_mali_root
    from mali_qcre_validation import (
        add_logical_hashes,
        grouped_log_calibration,
        make_group_folds,
    )
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mali_qcre_proxy import backend_metadata, build_rows, load_labels, resolve_mali_root
    from mali_qcre_validation import (
        add_logical_hashes,
        grouped_log_calibration,
        make_group_folds,
    )


ROOT = Path(__file__).resolve().parents[1]
FEATURES = (
    "physical_depth",
    "physical_two_qubit_depth",
    "qcre_weighted_critical_path_seconds",
)


def read_baseline(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty baseline CSV: {path}")
    return rows


def metrics_for_seed(
    rows: list[dict[str, object]], split_seed: int, folds: int
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, object]]:
    row_folds, split = make_group_folds(rows, split_seed, folds)
    result: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    for feature in FEATURES:
        prediction, details = grouped_log_calibration(rows, row_folds, (feature,), folds)
        result[feature] = details
        predictions[feature] = prediction
    return result, predictions, split


def write_rows(path: Path, rows_by_seed: dict[int, list[dict[str, object]]]) -> None:
    fields = [
        "transpile_seed",
        "circuit",
        "backend_label",
        "logical_circuit_hash",
        "target_time_taken_seconds",
        "physical_depth",
        "physical_two_qubit_depth",
        "qcre_weighted_critical_path_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed, rows in rows_by_seed.items():
            for row in rows:
                writer.writerow(
                    {
                        "transpile_seed": seed,
                        "circuit": row["circuit"],
                        "backend_label": row["backend_label"],
                        "logical_circuit_hash": row["logical_circuit_hash"],
                        "target_time_taken_seconds": row["target_time_taken_seconds"],
                        "physical_depth": row["physical_depth"],
                        "physical_two_qubit_depth": row["physical_two_qubit_depth"],
                        "qcre_weighted_critical_path_seconds": row[
                            "qcre_weighted_critical_path_seconds"
                        ],
                    }
                )


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Ma–Li / QCRE transpiler-seed sensitivity",
        "",
        "The same 340 QASM rows were transpiled with a fixed Qiskit/FakeBackend",
        "configuration and three `seed_transpiler` values. All scores use the",
        "same logical-QASM-hash grouped five-fold split (split seed 1234).",
        "",
        f"Seeds: **{', '.join(str(seed) for seed in summary['seeds'])}**; "
        f"optimization level: **{summary['optimization_level']}**.",
        "",
        "## Grouped out-of-sample R² by transpiler seed",
        "",
        "| Seed | Physical depth | Physical 2Q depth | Weighted path |",
        "|---:|---:|---:|---:|",
    ]
    for seed in summary["seeds"]:
        metrics = summary["per_seed"][str(seed)]["features"]
        lines.append(
            f"| {seed} | {metrics['physical_depth']['aggregate']['r2_seconds']:.4f} | "
            f"{metrics['physical_two_qubit_depth']['aggregate']['r2_seconds']:.4f} | "
            f"{metrics['qcre_weighted_critical_path_seconds']['aggregate']['r2_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Stability summary",
            "",
            "| Feature | Mean R² | Std R² | Min | Max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for feature, values in summary["stability"].items():
        lines.append(
            f"| `{feature}` | {values['mean_r2']:.4f} | {values['std_r2']:.4f} | "
            f"{values['min_r2']:.4f} | {values['max_r2']:.4f} |"
        )
    lines.extend(
        [
            "",
            "A narrow range supports reporting a central estimate; a wide range",
            "would require reporting the seed range and treating one seed as",
            "exploratory. This check changes only transpiler seed, not labels,",
            "target semantics, grouped split or calibration protocol.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mali-root", default=None)
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=ROOT / "artifacts" / "validation" / "mali_qcre_proxy" / "mali_qcre_proxy_features.csv",
    )
    parser.add_argument(
        "--rows-csv",
        type=Path,
        default=None,
        help="Reuse a previously generated seed-sensitivity row CSV without retranspiling.",
    )
    parser.add_argument("--seeds", default="1234,2025,31415")
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "validation" / "mali_qcre_seed_sensitivity"
    )
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 3:
        raise ValueError("provide at least three transpiler seeds")
    mali_root = resolve_mali_root(args.mali_root)
    if args.rows_csv:
        rows_by_seed: dict[int, list[dict[str, object]]] = {seed: [] for seed in seeds}
        with args.rows_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                seed = int(row["transpile_seed"])
                if seed in rows_by_seed:
                    rows_by_seed[seed].append(row)
        if any(not rows for rows in rows_by_seed.values()):
            raise ValueError("rows CSV does not contain every requested seed")
    else:
        baseline = read_baseline(args.baseline_csv)
        add_logical_hashes(baseline, mali_root)
        rows_by_seed = {seeds[0]: baseline}
        for seed in seeds[1:]:
            print(f"building seed {seed}", flush=True)
            rows = build_rows(load_labels(mali_root), mali_root, seed, args.optimization_level)
            add_logical_hashes(rows, mali_root)
            rows_by_seed[seed] = rows

    per_seed: dict[str, object] = {}
    common_split: dict[str, object] | None = None
    for seed in seeds:
        features, _, split = metrics_for_seed(rows_by_seed[seed], args.split_seed, args.folds)
        if common_split is None:
            common_split = split
        per_seed[str(seed)] = {
            "n": len(rows_by_seed[seed]),
            "features": features,
            "physical_depth_median": float(
                np.median([float(row["physical_depth"]) for row in rows_by_seed[seed]])
            ),
            "weighted_path_median_seconds": float(
                np.median(
                    [float(row["qcre_weighted_critical_path_seconds"]) for row in rows_by_seed[seed]]
                )
            ),
        }
    stability: dict[str, object] = {}
    for feature in FEATURES:
        values = [
            float(per_seed[str(seed)]["features"][feature]["aggregate"]["r2_seconds"])
            for seed in seeds
        ]
        stability[feature] = {
            "mean_r2": float(statistics.mean(values)),
            "std_r2": float(statistics.pstdev(values)),
            "min_r2": min(values),
            "max_r2": max(values),
        }
    summary = {
        "provenance_class": "OUR_VALIDATION",
        "seeds": seeds,
        "optimization_level": args.optimization_level,
        "qiskit_protocol": "same FakeOsaka/FakeKyoto targets, Qiskit 1.4.1, seed_transpiler varied",
        "software_and_backend_metadata": backend_metadata(),
        "split_seed": args.split_seed,
        "split": common_split,
        "per_seed": per_seed,
        "stability": stability,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mali_qcre_seed_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_rows(args.output_dir / "mali_qcre_seed_sensitivity_rows.csv", rows_by_seed)
    write_report(args.output_dir / "MALI_QCRE_SEED_SENSITIVITY_REPORT.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
