#!/usr/bin/env python3
"""Azizov-style compiled simulator pretraining, then transfer to Ma--Li QPU labels.

Stage 1 trains on FakeWashington/FakeSherbrooke compiled features against the
public simulator ``time_taken`` CSVs. Stage 2 maps those predictions onto the
340 Osaka/Kyoto ``result.time_taken`` rows, whose inputs are FakeOsaka/FakeKyoto
compiled features. Clocks are never pooled.

This is not the Ma--Li source-DAG Graph Transformer, and it is not a claim that
current fake snapshots recover historical calibration.
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

from mali_family_ood_common import (
    LOGICAL_FEATURE_KEYS,
    affine_map,
    component_families,
    filename_family,
    matrix_from_rows,
    metrics,
    ridge_log1p,
)

T0_KEYS = (
    "logical_width",
    "logical_depth",
    "logical_two_qubit_depth",
    "logical_two_qubit_count",
    "logical_gate_count",
)
T1_KEYS = (
    "physical_depth",
    "physical_two_qubit_depth",
    "physical_two_qubit_gate_count",
    "physical_gate_count",
    "logical_width",
)
T2_KEYS = T1_KEYS + (
    "qcre_weighted_critical_path_seconds",
    "routing_depth_ratio",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if np.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def load_sim_labels(mali_root: Path) -> dict[tuple[str, str], float]:
    labels = {}
    for device in ("washington", "sherbrooke"):
        path = mali_root / "data" / f"{device}_time_taken.csv"
        for row in read_csv(path):
            name = row.get("circuit_name") or row.get("quantum_circuit")
            labels[(name, device)] = float(row["time_taken"])
    return labels


def attach_sim(
    compiled_rows: list[dict[str, str]],
    labels: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    attached = []
    for row in compiled_rows:
        if row.get("status") != "ok":
            continue
        key = (row["circuit"], row["backend"])
        if key not in labels:
            continue
        item = dict(row)
        item["target_seconds"] = labels[key]
        item["filename_family"] = filename_family(row["circuit"])
        attached.append(item)
    return attached


def attach_qpu(
    split_rows: list[dict[str, str]],
    physical_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    physical = {(row["circuit"], row["backend"]): row for row in physical_rows if row.get("status", "ok") == "ok"}
    attached = []
    missing = []
    for split in split_rows:
        key = (split["circuit"], split["backend"])
        compiled = physical.get(key)
        if compiled is None:
            missing.append(key)
            continue
        item = dict(compiled)
        item["target_seconds"] = float(split["target_seconds"])
        item["family_component"] = split["family_component"]
        item["group_fold"] = int(split["group_fold"])
        item["qasm_sha256"] = split["qasm_sha256"]
        item["filename_family"] = split["filename_family"]
        attached.append(item)
    if missing:
        raise KeyError(f"missing QPU compiled features for {len(missing)} rows, e.g. {missing[:3]}")
    return attached


def hashed_folds(keys: list[str], folds: int, seed: int = 1234) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    unique = sorted(set(keys))
    rng.shuffle(unique)
    return {key: (index % folds) + 1 for index, key in enumerate(unique)}


def design(rows: list[dict[str, object]], keys: tuple[str, ...], positive_backend: str | None = None) -> np.ndarray:
    matrix = matrix_from_rows(rows, keys)
    if not positive_backend:
        return matrix
    dummy = np.asarray(
        [[1.0 if str(row.get("backend")) == positive_backend else 0.0] for row in rows],
        dtype=float,
    )
    return np.hstack((matrix, dummy))


def predict_scratch(
    train_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    keys: tuple[str, ...],
    positive_backend: str | None = None,
) -> np.ndarray:
    x_train = design(train_rows, keys, positive_backend)
    x_test = design(test_rows, keys, positive_backend)
    y_train = np.asarray([float(row["target_seconds"]) for row in train_rows], dtype=float)
    return ridge_log1p(x_train, y_train, x_test)


def transfer_affine(
    sim_train: list[dict[str, object]],
    qpu_train: list[dict[str, object]],
    qpu_test: list[dict[str, object]],
    keys: tuple[str, ...],
) -> np.ndarray:
    x_sim = matrix_from_rows(sim_train, keys)
    y_sim = np.asarray([float(row["target_seconds"]) for row in sim_train], dtype=float)
    qpu_train_pred = ridge_log1p(x_sim, y_sim, matrix_from_rows(qpu_train, keys))
    qpu_test_pred = ridge_log1p(x_sim, y_sim, matrix_from_rows(qpu_test, keys))
    y_qpu_train = np.asarray([float(row["target_seconds"]) for row in qpu_train], dtype=float)
    return affine_map(qpu_train_pred, y_qpu_train, qpu_test_pred)


def pooled_metrics(rows: list[dict[str, object]], model: str, exclude_qwalk: bool = False) -> dict[str, float]:
    selected = [
        row
        for row in rows
        if row["model"] == model and (not exclude_qwalk or "qwalk" not in str(row.get("family_component", row.get("filename_family", ""))))
    ]
    if not selected:
        return metrics(np.asarray([]), np.asarray([]))
    target = np.asarray([float(row["target_seconds"]) for row in selected], dtype=float)
    prediction = np.asarray([float(row["prediction_seconds"]) for row in selected], dtype=float)
    out = metrics(target, prediction)
    out["n"] = len(selected)
    return out


def record_predictions(
    split: str,
    model: str,
    rows: list[dict[str, object]],
    prediction: np.ndarray,
    extra: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    output = []
    for row, value in zip(rows, prediction):
        item = {
            "split": split,
            "model": model,
            "circuit": row["circuit"],
            "backend": row.get("backend"),
            "filename_family": row.get("filename_family"),
            "family_component": row.get("family_component", ""),
            "qasm_sha256": row.get("qasm_sha256"),
            "target_seconds": float(row["target_seconds"]),
            "prediction_seconds": float(value),
        }
        if extra:
            item.update(extra)
        output.append(item)
    return output


def evaluate_grouped_scratch(rows: list[dict[str, object]], name: str, folds: dict[str, int], positive_backend: str | None = None) -> list[dict[str, object]]:
    predictions = []
    for fold in range(1, 6):
        train = [row for row in rows if folds[row["qasm_sha256"]] != fold]
        test = [row for row in rows if folds[row["qasm_sha256"]] == fold]
        if not train or not test:
            continue
        split = f"{name}_grouped_fold_{fold}"
        for model, keys in (("T0", T0_KEYS), ("T1", T1_KEYS), ("T2", T2_KEYS)):
            pred = predict_scratch(train, test, keys, positive_backend=positive_backend)
            predictions.extend(record_predictions(split, f"{name}_{model}", test, pred, {"fold": fold}))
    return predictions


def evaluate_same_circuit_transfer(
    sim_rows: list[dict[str, object]],
    qpu_rows: list[dict[str, object]],
    include_test_qasm: bool,
    tag: str,
) -> list[dict[str, object]]:
    predictions = []
    for fold in range(1, 6):
        qpu_train = [row for row in qpu_rows if int(row["group_fold"]) != fold]
        qpu_test = [row for row in qpu_rows if int(row["group_fold"]) == fold]
        test_hashes = {row["qasm_sha256"] for row in qpu_test}
        if include_test_qasm:
            sim_train = sim_rows
        else:
            sim_train = [row for row in sim_rows if row["qasm_sha256"] not in test_hashes]
        split = f"{tag}_qpu_fold_{fold}"
        for model, keys in (("T0", T0_KEYS), ("T1", T1_KEYS), ("T2", T2_KEYS)):
            pred = transfer_affine(sim_train, qpu_train, qpu_test, keys)
            predictions.extend(
                record_predictions(
                    split,
                    f"{tag}_{model}",
                    qpu_test,
                    pred,
                    {"fold": fold, "n_sim_train": len(sim_train)},
                )
            )
            scratch = predict_scratch(qpu_train, qpu_test, keys, positive_backend="osaka")
            predictions.extend(record_predictions(split, f"qpu_scratch_{model}", qpu_test, scratch, {"fold": fold}))
    return predictions


def evaluate_family_transfer(
    sim_rows: list[dict[str, object]],
    qpu_rows: list[dict[str, object]],
    tag: str,
) -> list[dict[str, object]]:
    predictions = []
    components = sorted({str(row["family_component"]) for row in qpu_rows})
    for component in components:
        held = set(component_families(component))
        qpu_test = [row for row in qpu_rows if row["family_component"] == component]
        qpu_train = [row for row in qpu_rows if row["family_component"] != component]
        sim_incl = sim_rows
        sim_excl = [row for row in sim_rows if filename_family(row["circuit"]) not in held]
        split = f"family_{component}"
        for model, keys in (("T1", T1_KEYS), ("T2", T2_KEYS)):
            pred_incl = transfer_affine(sim_incl, qpu_train, qpu_test, keys)
            pred_excl = transfer_affine(sim_excl, qpu_train, qpu_test, keys)
            scratch = predict_scratch(qpu_train, qpu_test, keys, positive_backend="osaka")
            predictions.extend(record_predictions(split, f"{tag}_{model}_incl", qpu_test, pred_incl, {"family_component": component}))
            predictions.extend(record_predictions(split, f"{tag}_{model}_excl", qpu_test, pred_excl, {"family_component": component}))
            predictions.extend(record_predictions(split, f"qpu_scratch_{model}_family", qpu_test, scratch, {"family_component": component}))
    return predictions


def summarize(name: str, rows: list[dict[str, object]], models: list[str]) -> dict[str, dict[str, float]]:
    return {model: pooled_metrics(rows, model, exclude_qwalk=False) for model in models}


def write_report(
    path: Path,
    finding_date: str,
    sim_ws: list[dict[str, object]],
    qpu_rows: list[dict[str, object]],
    sim_grouped: list[dict[str, object]],
    transfer_incl: list[dict[str, object]],
    transfer_excl: list[dict[str, object]],
    family_rows: list[dict[str, object]],
    coverage: dict[str, object],
) -> None:
    def line(model: str, rows: list[dict[str, object]], exclude_qwalk: bool = False) -> str:
        score = pooled_metrics(rows, model, exclude_qwalk=exclude_qwalk)
        return f"{fmt(score['r2_log1p_seconds'])} / {fmt(score['mae_seconds'], 3)} s (n={score['n']})"

    models_sim = ["sim_ws_T0", "sim_ws_T1", "sim_ws_T2"]
    models_transfer = [
        "matched_incl_T0",
        "matched_incl_T1",
        "matched_incl_T2",
        "qpu_scratch_T0",
        "qpu_scratch_T1",
        "qpu_scratch_T2",
    ]
    lines = [
        "# Azizov-style compiled simulator pretraining → Ma–Li QPU transfer",
        "",
        f"**Finding recorded:** {finding_date}.",
        "",
        "Stage 1 predicts FakeWashington/FakeSherbrooke `time_taken` from current",
        "FakeWashingtonV2/FakeSherbrooke compiled features. Stage 2 keeps that",
        "Ridge mapping, then fits an affine head on Osaka/Kyoto compiled features",
        "against the 340 hardware `result.time_taken` labels. The two clocks stay",
        "separate.",
        "",
        "## Coverage",
        "",
        f"- Simulator compiled rows used: {coverage['n_sim_ws']}",
        f"- Unique simulator QASMs: {coverage['n_sim_qasm']}",
        f"- QPU rows: {len(qpu_rows)}",
        f"- Priority QPU QASMs present on both W/S: {coverage['n_priority_complete']}",
        "",
        "## Simulator in-domain (grouped QASM, Washington/Sherbrooke labels)",
        "",
        "| Model | log-R² / MAE |",
        "|---|---|",
        f"| logical T0 | {line('sim_ws_T0', sim_grouped)} |",
        f"| compiled T1 | {line('sim_ws_T1', sim_grouped)} |",
        f"| compiled T2 | {line('sim_ws_T2', sim_grouped)} |",
        "",
        "## Same-circuit transfer onto 340 QPU rows",
        "",
        "`incl` keeps the test QASM in simulator pretraining (Ma–Li analogue).",
        "`excl` removes it. Scratch never uses simulator labels.",
        "",
        "| Model | All families log-R² / MAE | Excluding QWalk |",
        "|---|---|---|",
        f"| QPU scratch T0 | {line('qpu_scratch_T0', transfer_incl)} | {line('qpu_scratch_T0', transfer_incl, True)} |",
        f"| QPU scratch T1 | {line('qpu_scratch_T1', transfer_incl)} | {line('qpu_scratch_T1', transfer_incl, True)} |",
        f"| QPU scratch T2 | {line('qpu_scratch_T2', transfer_incl)} | {line('qpu_scratch_T2', transfer_incl, True)} |",
        f"| Sim→QPU T0 incl | {line('matched_incl_T0', transfer_incl)} | {line('matched_incl_T0', transfer_incl, True)} |",
        f"| Sim→QPU T1 incl | {line('matched_incl_T1', transfer_incl)} | {line('matched_incl_T1', transfer_incl, True)} |",
        f"| Sim→QPU T2 incl | {line('matched_incl_T2', transfer_incl)} | {line('matched_incl_T2', transfer_incl, True)} |",
        f"| Sim→QPU T1 excl | {line('matched_excl_T1', transfer_excl)} | {line('matched_excl_T1', transfer_excl, True)} |",
        f"| Sim→QPU T2 excl | {line('matched_excl_T2', transfer_excl)} | {line('matched_excl_T2', transfer_excl, True)} |",
        "",
        "## Family-unseen QPU, family-seen simulator",
        "",
        "| Model | All 9 components | Excluding QWalk |",
        "|---|---|---|",
        f"| QPU scratch T1 | {line('qpu_scratch_T1_family', family_rows)} | {line('qpu_scratch_T1_family', family_rows, True)} |",
        f"| Sim T1 incl | {line('matched_T1_incl', family_rows)} | {line('matched_T1_incl', family_rows, True)} |",
        f"| Sim T2 incl | {line('matched_T2_incl', family_rows)} | {line('matched_T2_incl', family_rows, True)} |",
        f"| Sim T2 excl | {line('matched_T2_excl', family_rows)} | {line('matched_T2_excl', family_rows, True)} |",
        "",
        "## What this can and cannot claim",
        "",
        "- A win for `incl` over scratch would mean compiled simulator pretraining helps QPU estimation.",
        "- A win for compiled T1/T2 over T0 on simulator labels is the Azizov-style representation result, on Ma–Li's simulator clock, not Aer `T_exec` re-measured here.",
        "- Current FakeWashingtonV2/FakeSherbrooke/FakeOsaka/FakeKyoto snapshots are not historical job-day calibrations.",
        "- This does not train the paper GNN, and it does not collect new QPU jobs.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mali-root", type=Path, default=ROOT.parent / "Quantum-Execution-Time-Prediction")
    parser.add_argument(
        "--ws-compiled",
        type=Path,
        default=ROOT / "artifacts/validation/mali_azizov_compiled_transfer_v1/washington_sherbrooke_compiled_features.csv",
    )
    parser.add_argument(
        "--qpu-physical",
        type=Path,
        default=ROOT / "artifacts/validation/mali_family_ood_v1/feature_space_audit/physical_proxy_features.csv",
    )
    parser.add_argument(
        "--qpu-manifest",
        type=Path,
        default=ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/validation/mali_azizov_compiled_transfer_v1",
    )
    parser.add_argument("--finding-date", default="2026-09-22")
    parser.add_argument("--min-sim-rows", type=int, default=200)
    args = parser.parse_args()

    if not args.ws_compiled.is_file():
        raise FileNotFoundError(f"compiled simulator features missing: {args.ws_compiled}")

    mali_root = args.mali_root.resolve()
    sim_labels = load_sim_labels(mali_root)
    sim_ws = attach_sim(read_csv(args.ws_compiled), sim_labels)
    qpu_rows = attach_qpu(read_csv(args.qpu_manifest), read_csv(args.qpu_physical))
    if len(sim_ws) < args.min_sim_rows:
        raise RuntimeError(f"only {len(sim_ws)} simulator compiled rows; wait for transpile")

    qpu_hashes = {row["qasm_sha256"] for row in qpu_rows}
    priority_by_qasm = defaultdict(set)
    for row in sim_ws:
        if row["qasm_sha256"] in qpu_hashes:
            priority_by_qasm[row["qasm_sha256"]].add(row["backend"])
    coverage = {
        "n_sim_ws": len(sim_ws),
        "n_sim_qasm": len({row["qasm_sha256"] for row in sim_ws}),
        "n_priority_complete": sum(1 for backends in priority_by_qasm.values() if backends == {"washington", "sherbrooke"}),
        "n_qpu": len(qpu_rows),
    }

    sim_folds = hashed_folds([row["qasm_sha256"] for row in sim_ws], 5)
    sim_grouped = evaluate_grouped_scratch(sim_ws, "sim_ws", sim_folds, positive_backend="sherbrooke")
    for row in sim_grouped:
        row["model"] = row["model"]

    transfer_incl = evaluate_same_circuit_transfer(sim_ws, qpu_rows, True, "matched_incl")
    transfer_excl = evaluate_same_circuit_transfer(sim_ws, qpu_rows, False, "matched_excl")
    # drop duplicated scratch from excl file; keep scratch from incl
    transfer_excl = [row for row in transfer_excl if not str(row["model"]).startswith("qpu_scratch_")]
    family_rows = evaluate_family_transfer(sim_ws, qpu_rows, "matched")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = sim_grouped + transfer_incl + transfer_excl + family_rows
    write_csv(output_dir / "predictions.csv", all_rows)
    summary = {
        "coverage": coverage,
        "sim_grouped": summarize("sim", sim_grouped, ["sim_ws_T0", "sim_ws_T1", "sim_ws_T2"]),
        "transfer_incl": summarize(
            "incl",
            transfer_incl,
            ["matched_incl_T0", "matched_incl_T1", "matched_incl_T2", "qpu_scratch_T0", "qpu_scratch_T1", "qpu_scratch_T2"],
        ),
        "transfer_excl": summarize("excl", transfer_excl, ["matched_excl_T0", "matched_excl_T1", "matched_excl_T2"]),
        "family": summarize(
            "family",
            family_rows,
            ["matched_T1_incl", "matched_T2_incl", "matched_T1_excl", "matched_T2_excl", "qpu_scratch_T1_family", "qpu_scratch_T2_family"],
        ),
        "family_no_qwalk": {
            model: pooled_metrics(family_rows, model, True)
            for model in ["matched_T1_incl", "matched_T2_incl", "qpu_scratch_T1_family"]
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(
        output_dir / "REPORT.md",
        args.finding_date,
        sim_ws,
        qpu_rows,
        sim_grouped,
        transfer_incl,
        transfer_excl,
        family_rows,
        coverage,
    )
    print(json.dumps({"coverage": coverage, "sim_grouped": summary["sim_grouped"], "transfer_incl": summary["transfer_incl"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
