#!/usr/bin/env python3
"""Experiment B2: hardware-unseen family transfer with simulator-seen families.

This is not historical Experiment B (labels for the 12 missing families on
Osaka/Kyoto). It keeps Experiment C's QPU splits and test rows, then changes
only the simulator pretraining condition:

- Experiment C: target family is absent from simulator pretraining and QPU train.
- Experiment B2: target family is present in simulator pretraining, still
  absent from QPU train, and scored on the same QPU test rows.

Scratch T0–T3 are recomputed on the same splits so the comparison is paired.
Simulator compiled-proxy features are CURRENT_FAKE_SNAPSHOT_PROXY descriptors
averaged over FakeOsaka/FakeKyoto. They are not Washington/Sherbrooke
compilation, and simulator labels remain FakeWashington/Sherbrooke time_taken.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
PROXY_T1_KEYS = ("proxy_compiled_depth", "proxy_width", "proxy_two_qubit_count")
PROXY_T2_KEYS = PROXY_T1_KEYS + ("proxy_critical_path_seconds",)
PROXY_T0T1_KEYS = LOGICAL_FEATURE_KEYS + PROXY_T1_KEYS
PROXY_SOURCE_KEYS = {
    "proxy_compiled_depth": "physical_depth",
    "proxy_width": "logical_width",
    "proxy_two_qubit_count": "physical_two_qubit_gate_count",
    "proxy_critical_path_seconds": "qcre_weighted_critical_path_seconds",
    "proxy_routing_depth_ratio": "routing_depth_ratio",
}


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
        cache[circuit] = {
            "circuit": circuit,
            "status": "ok",
            **{key: str(value) for key, value in features.items()},
        }
        missing += 1
        if missing % 25 == 0:
            print(f"extracted {missing} uncached logical circuits", flush=True)


def average_proxy_by_circuit(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        grouped[Path(row["circuit"]).stem].append(row)
    averaged: dict[str, dict[str, float]] = {}
    for circuit, items in grouped.items():
        averaged[circuit] = {
            proxy_key: float(np.mean([float(item[source_key]) for item in items]))
            for proxy_key, source_key in PROXY_SOURCE_KEYS.items()
        }
        averaged[circuit]["n_proxy_backends"] = float(len(items))
    return averaged


def attach_proxy(rows: list[dict[str, object]], proxy: dict[str, dict[str, float]], circuit_key: str) -> None:
    missing = []
    for row in rows:
        circuit = Path(str(row[circuit_key])).stem
        features = proxy.get(circuit)
        if features is None:
            missing.append(circuit)
            continue
        row.update(features)
    if missing:
        raise KeyError(f"missing FakeOsaka/FakeKyoto proxy for {len(missing)} circuits, e.g. {missing[:5]}")


def predict_block(train_rows, test_rows, keys, y_train, backend_key=None):
    x_train = matrix_from_rows(train_rows, keys, backend_key=backend_key)
    x_test = matrix_from_rows(test_rows, keys, backend_key=backend_key)
    return ridge_log1p(x_train, y_train, x_test)


def sim_affine_block(sim_train, qpu_train, qpu_test, keys):
    y_sim = np.asarray([float(row["target_seconds"]) for row in sim_train], dtype=float)
    y_train = np.asarray([float(row["target_seconds"]) for row in qpu_train], dtype=float)
    sim_pred_train = predict_block(sim_train, qpu_train, keys, y_sim, backend_key=None)
    sim_pred_test = predict_block(sim_train, qpu_test, keys, y_sim, backend_key=None)
    return affine_map(sim_pred_train, y_train, sim_pred_test)


def fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if np.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def pooled_from_predictions(prediction_rows, model: str, exclude_qwalk: bool = False) -> dict[str, float]:
    selected = [
        row
        for row in prediction_rows
        if row["model"] == model and (not exclude_qwalk or "qwalk" not in str(row["family_component"]))
    ]
    return metrics(
        np.asarray([float(row["target_seconds"]) for row in selected]),
        np.asarray([float(row["prediction_seconds"]) for row in selected]),
    )


def append_predictions(prediction_rows, test_rows, estimates, split, model):
    for row, estimate in zip(test_rows, estimates):
        prediction_rows.append(
            {
                "split": split,
                "model": model,
                "row_id": row["row_id"],
                "circuit": row["circuit"],
                "backend": row["backend"],
                "family_component": row["family_component"],
                "target_seconds": float(row["target_seconds"]),
                "prediction_seconds": float(estimate),
                "absolute_error_seconds": abs(float(estimate) - float(row["target_seconds"])),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mali-root", type=Path, default=ROOT.parent / "Quantum-Execution-Time-Prediction")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv",
    )
    parser.add_argument(
        "--physical-manifest",
        type=Path,
        default=ROOT / "artifacts/validation/mali_physical_dag_v1/manifest.csv",
    )
    parser.add_argument(
        "--graph-records",
        type=Path,
        default=ROOT / "artifacts/validation/mali_physical_dag_v1/graph_records.csv",
    )
    parser.add_argument(
        "--rich-features",
        type=Path,
        default=ROOT / "artifacts/validation/mali_physical_dag_v1/graph_summary_features.csv",
    )
    parser.add_argument(
        "--logical-cache",
        type=Path,
        default=ROOT / "artifacts/validation/mali_family_ood_v1/feature_space_audit/logical_features.csv",
    )
    parser.add_argument(
        "--physical-proxy",
        type=Path,
        default=ROOT / "artifacts/validation/mali_family_ood_v1/feature_space_audit/physical_proxy_features.csv",
    )
    parser.add_argument(
        "--experiment-c-metrics",
        type=Path,
        default=ROOT / "artifacts/validation/mali_family_ood_v1/unseen_family_transfer/metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/validation/mali_family_ood_v1/simulator_assisted_transfer",
    )
    parser.add_argument("--finding-date", default="2026-09-22")
    args = parser.parse_args()

    mali_root = args.mali_root.resolve()
    split_rows = read_csv(args.split_manifest)
    physical_rows = read_csv(args.physical_manifest)
    graph_rows = {row["graph_id"]: row for row in read_csv(args.graph_records)}
    rich_rows = {row["graph_id"]: row for row in read_csv(args.rich_features)}
    sim_rows = read_csv(mali_root / "data" / "training_data_manifest.csv")
    cache = load_logical_cache(args.logical_cache)
    proxy = average_proxy_by_circuit(read_csv(args.physical_proxy))

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

    print(f"attaching logical T0 and compiled proxy to {len(qpu_rows)} QPU rows and {len(sim_rows)} simulator rows", flush=True)
    attach_logical(qpu_rows, mali_root, cache, "circuit", "qasm_path")
    attach_logical(sim_rows, mali_root, cache, "circuit_name", "qasm_path")
    attach_proxy(qpu_rows, proxy, "circuit")
    attach_proxy(sim_rows, proxy, "circuit_name")
    for row in sim_rows:
        row["filename_family"] = filename_family(row["circuit_name"])
        row["target_seconds"] = float(row["target_time_taken"])
    for row in qpu_rows:
        row["target_seconds"] = float(row["target_seconds"])

    components = sorted({str(row["family_component"]) for row in qpu_rows})
    prediction_rows: list[dict[str, object]] = []
    aggregate: list[dict[str, object]] = []

    scratch_ladders = {
        "scratch_T0": (LOGICAL_FEATURE_KEYS, "backend"),
        "scratch_T1": (T1_KEYS, "backend"),
        "scratch_T2": (T2_KEYS, "backend"),
        "scratch_T3": (RICH_FEATURES, "backend"),
    }
    sim_ladders = {
        "sim_T0": LOGICAL_FEATURE_KEYS,
        "sim_T1proxy": PROXY_T1_KEYS,
        "sim_T2proxy": PROXY_T2_KEYS,
        "sim_T0T1proxy": PROXY_T0T1_KEYS,
    }

    for component in components:
        held_families = set(component_families(component))
        test_rows = [row for row in qpu_rows if row["family_component"] == component]
        train_rows = [row for row in qpu_rows if row["family_component"] != component]
        sim_excl = [row for row in sim_rows if row["filename_family"] not in held_families]
        sim_incl = list(sim_rows)
        y_train = np.asarray([float(row["target_seconds"]) for row in train_rows], dtype=float)
        y_test = np.asarray([float(row["target_seconds"]) for row in test_rows], dtype=float)
        print(
            f"component={component} n_test={len(test_rows)} n_qpu_train={len(train_rows)} "
            f"n_sim_excl={len(sim_excl)} n_sim_incl={len(sim_incl)} held={sorted(held_families)}",
            flush=True,
        )

        for model, (keys, backend_key) in scratch_ladders.items():
            prediction = predict_block(train_rows, test_rows, keys, y_train, backend_key=backend_key)
            result = metrics(y_test, prediction)
            result.update(
                {
                    "split": f"family_{component}",
                    "model": model,
                    "n_train": len(train_rows),
                    "n_test": len(test_rows),
                    "n_sim_train": 0,
                    "sim_target_family": "absent_from_qpu_and_unused",
                }
            )
            aggregate.append(result)
            append_predictions(prediction_rows, test_rows, prediction, f"family_{component}", model)

        for stem, keys in sim_ladders.items():
            for tag, sim_train in (("excl", sim_excl), ("incl", sim_incl)):
                model = f"{stem}_{tag}_affine_qpu"
                prediction = sim_affine_block(sim_train, train_rows, test_rows, keys)
                result = metrics(y_test, prediction)
                result.update(
                    {
                        "split": f"family_{component}",
                        "model": model,
                        "n_train": len(train_rows),
                        "n_test": len(test_rows),
                        "n_sim_train": len(sim_train),
                        "sim_target_family": "excluded" if tag == "excl" else "included",
                    }
                )
                aggregate.append(result)
                append_predictions(prediction_rows, test_rows, prediction, f"family_{component}", model)

    models = [row["model"] for row in aggregate]
    models = list(dict.fromkeys(models))
    pooled_all = {model: pooled_from_predictions(prediction_rows, model, False) for model in models}
    pooled_no_qwalk = {model: pooled_from_predictions(prediction_rows, model, True) for model in models}

    c_metrics = []
    if args.experiment_c_metrics.is_file():
        c_metrics = read_csv(args.experiment_c_metrics)
    c_by_key = {(row["split"], row["model"]): row for row in c_metrics}
    scratch_match = []
    for row in aggregate:
        if not row["model"].startswith("scratch_"):
            continue
        previous = c_by_key.get((row["split"], row["model"]))
        if previous is None:
            continue
        delta = abs(float(row["r2_log1p_seconds"]) - float(previous["r2_log1p_seconds"]))
        scratch_match.append(
            {
                "split": row["split"],
                "model": row["model"],
                "abs_delta_log_r2": delta,
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics.csv", aggregate)
    write_csv(output_dir / "oof_predictions.csv", prediction_rows)
    if scratch_match:
        write_csv(output_dir / "scratch_match_to_experiment_c.csv", scratch_match)

    def contrast(incl_model: str, excl_model: str) -> dict[str, float]:
        incl = pooled_no_qwalk[incl_model]
        excl = pooled_no_qwalk[excl_model]
        return {
            "incl_log_r2": incl["r2_log1p_seconds"],
            "excl_log_r2": excl["r2_log1p_seconds"],
            "delta_log_r2_incl_minus_excl": incl["r2_log1p_seconds"] - excl["r2_log1p_seconds"],
            "incl_mae": incl["mae_seconds"],
            "excl_mae": excl["mae_seconds"],
            "delta_mae_incl_minus_excl": incl["mae_seconds"] - excl["mae_seconds"],
        }

    summary = {
        "finding_date": args.finding_date,
        "experiment": "B2_simulator_assisted_hardware_unseen_family",
        "n_qpu_rows": len(qpu_rows),
        "n_sim_rows": len(sim_rows),
        "n_components": len(components),
        "components": components,
        "target_semantics_qpu": "Ma-Li observed result.time_taken seconds; 1024 shots; queue excluded",
        "target_semantics_sim": "Ma-Li FakeWashington/Sherbrooke time_taken; not Osaka/Kyoto QPU",
        "proxy_semantics": (
            "proxy_T1/T2 features are CURRENT_FAKE_SNAPSHOT_PROXY FakeOsaka/FakeKyoto "
            "compiled descriptors averaged over the two backends; they are not "
            "Washington/Sherbrooke compilation and are not observed QPU runtime"
        ),
        "models": models,
        "pooled_all_components": pooled_all,
        "pooled_excluding_qwalk": pooled_no_qwalk,
        "seeing_family_on_simulator": {
            "T0": contrast("sim_T0_incl_affine_qpu", "sim_T0_excl_affine_qpu"),
            "T1proxy": contrast("sim_T1proxy_incl_affine_qpu", "sim_T1proxy_excl_affine_qpu"),
            "T2proxy": contrast("sim_T2proxy_incl_affine_qpu", "sim_T2proxy_excl_affine_qpu"),
            "T0T1proxy": contrast("sim_T0T1proxy_incl_affine_qpu", "sim_T0T1proxy_excl_affine_qpu"),
        },
        "max_abs_scratch_log_r2_delta_vs_experiment_c": (
            max(row["abs_delta_log_r2"] for row in scratch_match) if scratch_match else None
        ),
        "historical_b_status": "blocked_no_osaka_kyoto_labels_for_12_missing_families",
        "qwalk_note": "qwalk-noancilla is 2 rows; headline pooled numbers exclude it",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "REPORT.md").write_text(render_report(summary, aggregate, components), encoding="utf-8")
    print(
        json.dumps(
            {
                "pooled_excluding_qwalk": {
                    key: pooled_no_qwalk[key]
                    for key in [
                        "scratch_T0",
                        "scratch_T1",
                        "scratch_T3",
                        "sim_T0_excl_affine_qpu",
                        "sim_T0_incl_affine_qpu",
                        "sim_T1proxy_excl_affine_qpu",
                        "sim_T1proxy_incl_affine_qpu",
                        "sim_T0T1proxy_incl_affine_qpu",
                    ]
                },
                "seeing_family_on_simulator": summary["seeing_family_on_simulator"],
                "max_abs_scratch_log_r2_delta_vs_experiment_c": summary[
                    "max_abs_scratch_log_r2_delta_vs_experiment_c"
                ],
            },
            indent=2,
        )
    )
    return 0


def render_report(summary: dict, aggregate: list[dict], components: list[str]) -> str:
    pooled = summary["pooled_excluding_qwalk"]
    all_p = summary["pooled_all_components"]
    seeing = summary["seeing_family_on_simulator"]
    order = [
        "scratch_T0",
        "scratch_T1",
        "scratch_T2",
        "scratch_T3",
        "sim_T0_excl_affine_qpu",
        "sim_T0_incl_affine_qpu",
        "sim_T1proxy_excl_affine_qpu",
        "sim_T1proxy_incl_affine_qpu",
        "sim_T2proxy_excl_affine_qpu",
        "sim_T2proxy_incl_affine_qpu",
        "sim_T0T1proxy_excl_affine_qpu",
        "sim_T0T1proxy_incl_affine_qpu",
    ]
    lines = [
        "# Simulator-assisted transfer onto hardware-unseen families",
        "",
        f"**Finding recorded:** {summary['finding_date']}.",
        "",
        "This is Experiment B2. It uses the 340 Osaka/Kyoto `result.time_taken`",
        "rows and the same 9 family connected components as Experiment C.",
        "The target family stays out of QPU training. The only protocol change",
        "is whether that family remains in FakeWashington/Sherbrooke",
        "pretraining.",
        "",
        "Historical Experiment B, observed runtime for the 12 MQT families",
        "missing from the hardware table, remains blocked. This table does not",
        "score GHZ, Grover, QAOA or the other missing families on a QPU.",
        "",
        "`excl` matches Experiment C: the held family is removed from simulator",
        "pretraining. `incl` is B2: the held family stays in simulator",
        "pretraining. Scratch T0–T3 never use simulator labels.",
        "",
        "Compiled-proxy simulator models predict Washington/Sherbrooke",
        "`time_taken` from CURRENT_FAKE_SNAPSHOT_PROXY FakeOsaka/FakeKyoto",
        "descriptors averaged over the two backends. That X is a circuit",
        "descriptor, not a matched-device compilation of the simulator runs.",
        "",
        "## Pooled out-of-family predictions",
        "",
        "| Model | Target family in sim? | All 9 components log-R² / MAE | Excluding QWalk log-R² / MAE |",
        "|---|---|---:|---:|",
    ]
    in_sim = {
        "scratch_T0": "unused",
        "scratch_T1": "unused",
        "scratch_T2": "unused",
        "scratch_T3": "unused",
        "sim_T0_excl_affine_qpu": "no (Experiment C)",
        "sim_T0_incl_affine_qpu": "yes (B2)",
        "sim_T1proxy_excl_affine_qpu": "no",
        "sim_T1proxy_incl_affine_qpu": "yes (B2)",
        "sim_T2proxy_excl_affine_qpu": "no",
        "sim_T2proxy_incl_affine_qpu": "yes (B2)",
        "sim_T0T1proxy_excl_affine_qpu": "no",
        "sim_T0T1proxy_incl_affine_qpu": "yes (B2)",
    }
    for model in order:
        lines.append(
            f"| `{model}` | {in_sim[model]} | {fmt(all_p[model]['r2_log1p_seconds'])} / {fmt(all_p[model]['mae_seconds'], 3)} s | "
            f"{fmt(pooled[model]['r2_log1p_seconds'])} / {fmt(pooled[model]['mae_seconds'], 3)} s |"
        )
    lines.extend(
        [
            "",
            "## Does seeing the family on the simulator help?",
            "",
            "Delta is B2 (`incl`) minus Experiment C (`excl`), excluding QWalk.",
            "Positive log-R² delta, or negative MAE delta, means keeping the",
            "family in simulator pretraining improved hardware-unseen transfer.",
            "",
            "| Representation | excl log-R² | incl log-R² | Δ log-R² | excl MAE | incl MAE | Δ MAE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "T0": "logical T0",
        "T1proxy": "compiled proxy T1",
        "T2proxy": "compiled proxy T2",
        "T0T1proxy": "logical T0 + compiled proxy T1",
    }
    for key, label in labels.items():
        row = seeing[key]
        lines.append(
            f"| {label} | {fmt(row['excl_log_r2'])} | {fmt(row['incl_log_r2'])} | "
            f"{fmt(row['delta_log_r2_incl_minus_excl'])} | {fmt(row['excl_mae'], 3)} s | "
            f"{fmt(row['incl_mae'], 3)} s | {fmt(row['delta_mae_incl_minus_excl'], 3)} s |"
        )
    by_key = {(row["split"], row["model"]): row for row in aggregate}
    lines.extend(
        [
            "",
            "## Per-component log-R² / MAE",
            "",
            "| Component | n | scratch T1 | sim T0 incl | sim T2proxy excl | sim T2proxy incl |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for component in components:
        split = f"family_{component}"
        t1 = by_key[(split, "scratch_T1")]
        t0_incl = by_key[(split, "sim_T0_incl_affine_qpu")]
        t2_excl = by_key[(split, "sim_T2proxy_excl_affine_qpu")]
        t2_incl = by_key[(split, "sim_T2proxy_incl_affine_qpu")]
        lines.append(
            f"| `{component}` | {t1['n_test']} | {fmt(t1['r2_log1p_seconds'])} / {fmt(t1['mae_seconds'], 3)} s | "
            f"{fmt(t0_incl['r2_log1p_seconds'])} / {fmt(t0_incl['mae_seconds'], 3)} s | "
            f"{fmt(t2_excl['r2_log1p_seconds'])} / {fmt(t2_excl['mae_seconds'], 3)} s | "
            f"{fmt(t2_incl['r2_log1p_seconds'])} / {fmt(t2_incl['mae_seconds'], 3)} s |"
        )
    match = summary.get("max_abs_scratch_log_r2_delta_vs_experiment_c")
    match_text = fmt(match, 8) if match is not None else "NA"
    lines.extend(
        [
            "",
            "## What this can and cannot claim",
            "",
            f"- Scratch T0–T3 were recomputed on Experiment C's splits. The largest absolute log-R² difference versus the frozen C metrics file is {match_text}.",
            "- A positive `incl − excl` delta would mean simulator-seen family structure helps after affine calibration on other QPU families.",
            "- Beating scratch T0 is not enough. The relevant ceiling on this split is compiled scratch T1/T3.",
            "- Pooled compiled-proxy T2 log-R² of about 0.25 is family-heterogeneous. Ansatz components can be weakly positive; QFT/QPE/QNN remain large-negative. Do not read the pooled number as a universal simulator-to-QPU estimator.",
            "- Compiled-proxy simulator models are not device-matched pretraining. Washington/Sherbrooke `time_taken` is the label; FakeOsaka/FakeKyoto structure is only a circuit descriptor.",
            "- This still does not score the 12 missing families on real QPU. That remains historical B, blocked, or prospective B1 if new hardware labels are collected.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
