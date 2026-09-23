#!/usr/bin/env python3
"""Experiment C: fully unseen-family transfer on the 10 Ma--Li hardware families.

For each family connected component in the 340 observed Osaka/Kyoto rows:

- QPU train uses the other components only;
- simulator pretraining uses the 3,020 FakeWashington/Sherbrooke rows after
  removing that component's family name(s);
- test is the held component's observed-QPU rows.

Simulator pretraining stays on logical T0 features. Washington/Sherbrooke
compilation is not treated as Osaka/Kyoto physical structure. This experiment
cannot score the 12 MQT families that have no hardware labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_mali_physical_baselines import RICH_FEATURES
from mali_family_ood_common import (
    LOGICAL_FEATURE_KEYS,
    affine_map,
    component_families,
    filename_family,
    logical_features,
    matrix_from_rows,
    metrics,
    ridge_log1p,
)

T1_KEYS = ("compiled_depth", "active_physical_width", "native_two_qubit_gate_count")
T2_KEYS = T1_KEYS + ("qcre_critical_path_seconds",)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_logical_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    return {row["circuit"]: row for row in read_csv(path) if row.get("status", "ok") == "ok"}


def attach_logical(
    rows: list[dict[str, str]],
    mali_root: Path,
    cache: dict[str, dict[str, str]],
    circuit_key: str,
    qasm_key: str,
) -> None:
    from qiskit import QuantumCircuit

    missing = 0
    for row in rows:
        circuit = Path(row[circuit_key]).stem
        cached = cache.get(circuit)
        if cached:
            for key in LOGICAL_FEATURE_KEYS:
                row[key] = cached[key]
            continue
        qasm_path = mali_root / row[qasm_key]
        circuit_obj = QuantumCircuit.from_qasm_file(str(qasm_path))
        features = logical_features(circuit_obj)
        for key, value in features.items():
            row[key] = value
        cache[circuit] = {"circuit": circuit, "status": "ok", **{key: str(value) for key, value in features.items()}}
        missing += 1
        if missing % 25 == 0:
            print(f"extracted {missing} uncached logical circuits", flush=True)


def predict_block(train_rows, test_rows, keys, y_train, backend_key=None):
    x_train = matrix_from_rows(train_rows, keys, backend_key=backend_key)
    x_test = matrix_from_rows(test_rows, keys, backend_key=backend_key)
    return ridge_log1p(x_train, y_train, x_test)


def fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if np.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mali-root", type=Path, default=ROOT.parent / "Quantum-Execution-Time-Prediction")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv")
    parser.add_argument("--physical-manifest", type=Path, default=ROOT / "artifacts/validation/mali_physical_dag_v1/manifest.csv")
    parser.add_argument("--graph-records", type=Path, default=ROOT / "artifacts/validation/mali_physical_dag_v1/graph_records.csv")
    parser.add_argument("--rich-features", type=Path, default=ROOT / "artifacts/validation/mali_physical_dag_v1/graph_summary_features.csv")
    parser.add_argument("--logical-cache", type=Path, default=ROOT / "artifacts/validation/mali_family_ood_v1/feature_space_audit/logical_features.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/validation/mali_family_ood_v1/unseen_family_transfer")
    parser.add_argument("--finding-date", default="2026-09-22")
    args = parser.parse_args()

    mali_root = args.mali_root.resolve()
    split_rows = read_csv(args.split_manifest)
    physical_rows = read_csv(args.physical_manifest)
    graph_rows = {row["graph_id"]: row for row in read_csv(args.graph_records)}
    rich_rows = {row["graph_id"]: row for row in read_csv(args.rich_features)}
    sim_rows = read_csv(mali_root / "data" / "training_data_manifest.csv")
    cache = load_logical_cache(args.logical_cache)

    split_by_key = {(row["qasm_sha256"], row["backend"]): row for row in split_rows}
    qpu_rows: list[dict[str, object]] = []
    for row in physical_rows:
        split = split_by_key[(row["qasm_sha256"], row["backend"])]
        graph = graph_rows[row["graph_id"]]
        rich = rich_rows[row["graph_id"]]
        item = dict(row)
        item["family_component"] = split["family_component"]
        item["filename_family"] = filename_family(row["circuit"])
        item["native_two_qubit_gate_count"] = graph["native_two_qubit_gate_count"]
        item["qasm_path"] = split["qasm_path"]
        item.update(rich)
        qpu_rows.append(item)

    print(f"attaching logical T0 to {len(qpu_rows)} QPU rows and {len(sim_rows)} simulator rows", flush=True)
    attach_logical(qpu_rows, mali_root, cache, "circuit", "qasm_path")
    attach_logical(sim_rows, mali_root, cache, "circuit_name", "qasm_path")
    for row in sim_rows:
        row["filename_family"] = filename_family(row["circuit_name"])
        row["target_seconds"] = float(row["target_time_taken"])
    for row in qpu_rows:
        row["target_seconds"] = float(row["target_seconds"])

    components = sorted({str(row["family_component"]) for row in qpu_rows})
    prediction_rows: list[dict[str, object]] = []
    aggregate: list[dict[str, object]] = []

    ladders = {
        "scratch_T0": LOGICAL_FEATURE_KEYS,
        "scratch_T1": T1_KEYS,
        "scratch_T2": T2_KEYS,
        "scratch_T3": RICH_FEATURES,
        "scratch_qcre": ("qcre_critical_path_seconds",),
        "scratch_compiled": ("qcre_critical_path_seconds", "compiled_depth", "active_physical_width"),
        "scratch_physical_graph": (
            "qcre_critical_path_seconds",
            "compiled_depth",
            "active_physical_width",
            "n_nodes",
            "n_edges",
        ),
    }

    for component in components:
        held_families = set(component_families(component))
        test_rows = [row for row in qpu_rows if row["family_component"] == component]
        train_rows = [row for row in qpu_rows if row["family_component"] != component]
        sim_train = [row for row in sim_rows if row["filename_family"] not in held_families]
        y_train = np.asarray([float(row["target_seconds"]) for row in train_rows], dtype=float)
        y_test = np.asarray([float(row["target_seconds"]) for row in test_rows], dtype=float)
        y_sim = np.asarray([float(row["target_seconds"]) for row in sim_train], dtype=float)
        print(
            f"component={component} n_test={len(test_rows)} n_qpu_train={len(train_rows)} "
            f"n_sim_train={len(sim_train)} held_families={sorted(held_families)}",
            flush=True,
        )

        for model, keys in ladders.items():
            prediction = predict_block(train_rows, test_rows, keys, y_train, backend_key="backend")
            result = metrics(y_test, prediction)
            result.update(
                {
                    "split": f"family_{component}",
                    "model": model,
                    "n_train": len(train_rows),
                    "n_test": len(test_rows),
                    "n_sim_train": 0,
                }
            )
            aggregate.append(result)
            for row, estimate in zip(test_rows, prediction):
                prediction_rows.append(
                    {
                        "split": f"family_{component}",
                        "model": model,
                        "row_id": row["row_id"],
                        "circuit": row["circuit"],
                        "backend": row["backend"],
                        "family_component": component,
                        "target_seconds": float(row["target_seconds"]),
                        "prediction_seconds": float(estimate),
                        "absolute_error_seconds": abs(float(estimate) - float(row["target_seconds"])),
                    }
                )

        sim_pred_train = predict_block(sim_train, train_rows, LOGICAL_FEATURE_KEYS, y_sim, backend_key=None)
        sim_pred_test = predict_block(sim_train, test_rows, LOGICAL_FEATURE_KEYS, y_sim, backend_key=None)
        calibrated = affine_map(sim_pred_train, y_train, sim_pred_test)
        result = metrics(y_test, calibrated)
        result.update(
            {
                "split": f"family_{component}",
                "model": "sim_T0_affine_qpu",
                "n_train": len(train_rows),
                "n_test": len(test_rows),
                "n_sim_train": len(sim_train),
            }
        )
        aggregate.append(result)
        for row, estimate in zip(test_rows, calibrated):
            prediction_rows.append(
                {
                    "split": f"family_{component}",
                    "model": "sim_T0_affine_qpu",
                    "row_id": row["row_id"],
                    "circuit": row["circuit"],
                    "backend": row["backend"],
                    "family_component": component,
                    "target_seconds": float(row["target_seconds"]),
                    "prediction_seconds": float(estimate),
                    "absolute_error_seconds": abs(float(estimate) - float(row["target_seconds"])),
                }
            )

    def pooled(model: str, exclude_qwalk: bool = False) -> dict[str, float]:
        selected = [
            row
            for row in prediction_rows
            if row["model"] == model and (not exclude_qwalk or "qwalk" not in str(row["family_component"]))
        ]
        return metrics(
            np.asarray([float(row["target_seconds"]) for row in selected]),
            np.asarray([float(row["prediction_seconds"]) for row in selected]),
        )

    models = [*ladders, "sim_T0_affine_qpu"]
    pooled_all = {model: pooled(model, False) for model in models}
    pooled_no_qwalk = {model: pooled(model, True) for model in models}

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics.csv", aggregate)
    write_csv(output_dir / "oof_predictions.csv", prediction_rows)
    summary = {
        "finding_date": args.finding_date,
        "n_qpu_rows": len(qpu_rows),
        "n_sim_rows": len(sim_rows),
        "n_components": len(components),
        "components": components,
        "target_semantics_qpu": "Ma-Li observed result.time_taken seconds; 1024 shots; queue excluded",
        "target_semantics_sim": "Ma-Li FakeWashington/Sherbrooke time_taken; not Osaka/Kyoto QPU",
        "proxy_semantics": "T1-T3 physical features are CURRENT_FAKE_SNAPSHOT_PROXY; simulator pretrain uses logical T0 only",
        "models": models,
        "pooled_all_components": pooled_all,
        "pooled_excluding_qwalk": pooled_no_qwalk,
        "qwalk_note": "qwalk-noancilla is 2 rows; headline pooled numbers exclude it",
        "experiment_b_status": "blocked_no_qpu_labels_for_12_missing_families",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Unseen-family transfer on the 10 Ma--Li hardware families",
        "",
        f"**Finding recorded:** {args.finding_date}.",
        "",
        "This is Experiment C. A family connected component is removed from",
        "both simulator pretraining and QPU training, then scored on that",
        "component's observed Osaka/Kyoto `result.time_taken` rows. The 12 MQT",
        "families without hardware labels are not in this table.",
        "",
        "`realamprandom` and `twolocalrandom` share 22 QASM hashes, so they are",
        "held out as one connected component. `qwalk-noancilla` has two rows",
        "and is reported, not used as a headline R².",
        "",
        "Simulator pretraining is logical T0 only. FakeWashington/Sherbrooke",
        "compilation is a different device snapshot from FakeOsaka/FakeKyoto,",
        "so compiled features are not transferred from the simulator table.",
        "",
        "## Pooled out-of-family predictions",
        "",
        "| Model | All 9 components log-R² / MAE | Excluding QWalk log-R² / MAE |",
        "|---|---:|---:|",
    ]
    for model in models:
        all_m = pooled_all[model]
        no_q = pooled_no_qwalk[model]
        report.append(
            f"| `{model}` | {fmt(all_m['r2_log1p_seconds'])} / {fmt(all_m['mae_seconds'], 4)} s | "
            f"{fmt(no_q['r2_log1p_seconds'])} / {fmt(no_q['mae_seconds'], 4)} s |"
        )
    report.extend(
        [
            "",
            "## Per-component scores",
            "",
            "| Component | n_test | scratch T0 | scratch T1 | scratch T3 | sim T0 + affine QPU |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    by_key = {(row["split"], row["model"]): row for row in aggregate}
    for component in components:
        split = f"family_{component}"
        t0 = by_key[(split, "scratch_T0")]
        t3 = by_key[(split, "scratch_T3")]
        sim = by_key[(split, "sim_T0_affine_qpu")]
        report.append(
            f"| `{component}` | {t0['n_test']} | {fmt(t0['r2_log1p_seconds'])} / {fmt(t0['mae_seconds'], 3)} s | "
            f"{fmt(by_key[(split, 'scratch_T1')]['r2_log1p_seconds'])} / {fmt(by_key[(split, 'scratch_T1')]['mae_seconds'], 3)} s | "
            f"{fmt(t3['r2_log1p_seconds'])} / {fmt(t3['mae_seconds'], 3)} s | "
            f"{fmt(sim['r2_log1p_seconds'])} / {fmt(sim['mae_seconds'], 3)} s |"
        )
    report.extend(
        [
            "",
            "## Reading the comparison",
            "",
            "- `scratch_T*` trains only on other hardware families. That is the",
            "  fully unseen-family QPU baseline.",
            "- `sim_T0_affine_qpu` first fits a logical estimator on simulator",
            "  rows whose family names are not the held component, then fits an",
            "  affine map on the remaining QPU families. If this does not beat",
            "  `scratch_T0`, simulator pretraining is not helping family-OOD",
            "  transfer under this representation.",
            "- T1 compiled depth/width/2q, T2 adds the duration-weighted path,",
            "  and T3 is the rich static summary. These blocks are alternative",
            "  QPU-scratch representations, not nested on top of T0, and they",
            "  are not simulator-to-QPU compiled transfer.",
            "",
            "Experiment B, simulator-to-QPU transfer onto the 12 missing",
            "families, remains blocked: those families have no observed QPU",
            "labels, and the original Osaka/Kyoto machines are retired.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"pooled_excluding_qwalk": pooled_no_qwalk, "components": components}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
