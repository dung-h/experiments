#!/usr/bin/env python3
"""Compare Ma--Li observed labels with QCRE-style timing proxies.

This is deliberately an ``OUR_PROXY`` experiment.  Ma--Li supplies the
logical QASM and an observed ``time_taken`` label, but it does not expose the
historical physical transpilation and calibration snapshot for every row.  We
therefore transpile each QASM with the current FakeOsaka/FakeKyoto target and
use a target-duration weighted critical path as a gate-aware proxy.  The output keeps
logical and physical features separate and never calls the proxy a measured
QPU runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
import qiskit_ibm_runtime
from qiskit_ibm_runtime.fake_provider import FakeKyoto, FakeOsaka


ROOT = Path(__file__).resolve().parents[1]
FEATURES = (
    "logical_depth",
    "logical_two_qubit_depth",
    "physical_depth",
    "physical_two_qubit_depth",
    "qcre_weighted_critical_path_seconds",
)


def backend_metadata() -> dict[str, object]:
    """Record the exact FakeBackend target used by the proxy."""
    metadata: dict[str, object] = {
        "qiskit_version": qiskit.__version__,
        "qiskit_ibm_runtime_version": qiskit_ibm_runtime.__version__,
        "backends": {},
    }
    for name, backend in (("osaka", FakeOsaka()), ("kyoto", FakeKyoto())):
        edges = [list(edge) for edge in backend.coupling_map.get_edges()]
        metadata["backends"][name] = {
            "class": type(backend).__name__,
            "backend_version": getattr(backend, "backend_version", None),
            "fake_backend_api_version": getattr(backend, "version", None),
            "num_qubits": int(backend.num_qubits),
            "dt_seconds": float(backend.dt),
            "basis_gates": list(backend.configuration().basis_gates),
            "target_operation_names": sorted(backend.operation_names),
            "coupling_map_edges": edges,
        }
    return metadata


def resolve_mali_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    candidates = (
        ROOT / "work" / "mali",
        ROOT.parent / "Quantum-Execution-Time-Prediction",
    )
    for candidate in candidates:
        if (candidate / "data" / "quantum_circuits").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Ma-Li checkout not found; run scripts/bootstrap_upstreams.sh work "
        "or pass --mali-root"
    )


def qarg_indices(circuit: QuantumCircuit, qargs: Iterable) -> tuple[int, ...]:
    return tuple(circuit.find_bit(qubit).index for qubit in qargs)


def iter_instructions(circuit: QuantumCircuit):
    """Yield ``(operation, qargs, cargs)`` across Qiskit 1.x tuple APIs."""
    for entry in circuit.data:
        operation = getattr(entry, "operation", None)
        if operation is None:
            operation, qargs, cargs = entry
        else:
            qargs, cargs = entry.qubits, entry.clbits
        yield operation, qargs, cargs


def two_qubit_depth(circuit: QuantumCircuit) -> int:
    """Critical-path depth counting only multi-qubit operations."""
    clocks = [0] * circuit.num_qubits
    for instruction, qargs, _ in iter_instructions(circuit):
        if len(qargs) < 2 or getattr(instruction, "_directive", False):
            continue
        locations = qarg_indices(circuit, qargs)
        start = max((clocks[index] for index in locations), default=0)
        finish = start + 1
        for index in locations:
            clocks[index] = finish
    return max(clocks, default=0)


def weighted_critical_path(
    circuit: QuantumCircuit, backend
) -> tuple[int, list[str]]:
    """Return a target-duration weighted critical path in backend ``dt``.

    The transpiler output is topologically ordered.  For each instruction we
    advance the clocks of its qubits by the duration published in the current
    FakeBackend target.  This avoids Qiskit's ALAP scheduling pass (which is
    known to fail on some of these large QASM inputs under Qiskit 1.4), while
    retaining the hardware-aware gate duration information.  Unsupported
    operations are reported instead of silently being presented as hardware
    time.
    """
    dt_seconds = float(backend.dt)
    clocks = [0] * circuit.num_qubits
    unsupported: list[str] = []
    target = backend.target
    for instruction, qargs, _ in iter_instructions(circuit):
        if getattr(instruction, "_directive", False) or instruction.name == "barrier":
            continue
        locations = qarg_indices(circuit, qargs)
        if not locations:
            continue
        properties = None
        try:
            properties = target[instruction.name].get(locations)
            if properties is None and len(locations) == 2:
                properties = target[instruction.name].get(tuple(reversed(locations)))
        except KeyError:
            properties = None
        duration_seconds = getattr(properties, "duration", None)
        if duration_seconds is None:
            # Delay is already a duration-bearing instruction.  It is not
            # normally present before scheduling, but handling it makes the
            # helper deterministic for future inputs.
            duration_value = getattr(instruction, "duration", None)
            duration_unit = getattr(instruction, "unit", "dt")
            if duration_value is not None:
                duration_seconds = (
                    float(duration_value) * dt_seconds
                    if duration_unit == "dt"
                    else float(duration_value)
                )
            else:
                unsupported.append(instruction.name)
                duration_seconds = 0.0
        duration_dt = max(0, int(round(float(duration_seconds) / dt_seconds)))
        start = max((clocks[index] for index in locations), default=0)
        finish = start + duration_dt
        for index in locations:
            clocks[index] = finish
    return max(clocks, default=0), unsupported


def row_from_circuits(
    original: QuantumCircuit,
    compiled: QuantumCircuit,
    backend_name: str,
    circuit_name: str,
    target_seconds: float,
    backend_dt: float,
    weighted_duration_dt: int,
    unsupported_duration_ops: list[str],
    seed: int,
    optimization_level: int,
) -> dict[str, object]:
    counts = compiled.count_ops()
    physical_two_qubit = sum(
        count for name, count in counts.items() if name in {"cx", "cz", "ecr", "swap"}
    )
    physical_two_depth = two_qubit_depth(compiled)
    return {
        "circuit": circuit_name,
        "backend_label": backend_name,
        "target_time_taken_seconds": target_seconds,
        "target_semantics": (
            "Ma-Li upstream Osaka/Kyoto observed result.time_taken; execute shots=1024; "
            "averaged upstream runs; queue semantics per upstream"
        ),
        "shots": 1024,
        "logical_width": original.num_qubits,
        "logical_depth": original.depth(),
        "logical_two_qubit_depth": two_qubit_depth(original),
        "logical_total_gate_count": len(original.data),
        "physical_width": compiled.num_qubits,
        "physical_depth": compiled.depth(),
        "physical_two_qubit_depth": physical_two_depth,
        "physical_total_gate_count": len(compiled.data),
        "physical_two_qubit_gate_count": physical_two_qubit,
        "physical_swap_count": int(counts.get("swap", 0)),
        "qcre_weighted_critical_path_dt": int(weighted_duration_dt),
        "qcre_weighted_critical_path_seconds": float(weighted_duration_dt * backend_dt),
        "qcre_unsupported_duration_ops": ",".join(sorted(set(unsupported_duration_ops))),
        "fake_backend_dt_seconds": float(backend_dt),
        "transpile_optimization_level": optimization_level,
        "transpile_seed": seed,
        "status": "ok",
    }


def load_labels(mali_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for device in ("osaka", "kyoto"):
        path = mali_root / "data" / f"{device}_time_taken.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for source_row in reader:
                name = source_row.get("quantum_circuit") or source_row.get("circuit_name")
                if not name:
                    raise ValueError(f"missing circuit name in {path}")
                rows.append(
                    {
                        "device": device,
                        "circuit": name,
                        "target": float(source_row["time_taken"]),
                    }
                )
    return rows


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2.0 + 1.0
        position = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return pearson, spearman


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.maximum(prediction, 0.0)
    error = prediction - target
    mse = float(np.mean(error**2))
    target_log = np.log1p(target)
    prediction_log = np.log1p(prediction)
    log_error = prediction_log - target_log
    target_mean = float(np.mean(target))
    target_log_mean = float(np.mean(target_log))
    return {
        "mae_seconds": float(np.mean(np.abs(error))),
        "rmse_seconds": float(math.sqrt(mse)),
        "r2_seconds": float(1 - np.sum(error**2) / np.sum((target - target_mean) ** 2)),
        "mae_log1p_seconds": float(np.mean(np.abs(log_error))),
        "rmse_log1p_seconds": float(math.sqrt(np.mean(log_error**2))),
        "r2_log1p_seconds": float(
            1 - np.sum(log_error**2) / np.sum((target_log - target_log_mean) ** 2)
        ),
    }


def cross_validated_calibration(
    values: np.ndarray, target: np.ndarray, seed: int, folds: int = 5
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit log1p(target) ~ 1 + log1p(feature) on held-out folds."""
    indices = list(range(len(target)))
    random.Random(seed).shuffle(indices)
    predictions = np.empty(len(target), dtype=float)
    for fold_indices in np.array_split(np.asarray(indices), folds):
        test = fold_indices.astype(int)
        train = np.asarray([index for index in indices if index not in set(test)], dtype=int)
        x_train = np.column_stack((np.ones(len(train)), np.log1p(np.maximum(values[train], 0))))
        y_train = np.log1p(target[train])
        coefficients, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
        x_test = np.column_stack((np.ones(len(test)), np.log1p(np.maximum(values[test], 0))))
        predictions[test] = np.expm1(x_test @ coefficients)
    return predictions, regression_metrics(target, predictions)


def held_out_group_calibration(
    rows: list[dict[str, object]], feature: str, group_key: str
) -> dict[str, object]:
    """Fit on all but one backend and score the held-out backend."""
    groups = sorted({str(row[group_key]) for row in rows})
    result: dict[str, object] = {}
    for held_out in groups:
        train_rows = [row for row in rows if str(row[group_key]) != held_out]
        test_rows = [row for row in rows if str(row[group_key]) == held_out]
        train_values = np.asarray([float(row[feature]) for row in train_rows])
        train_target = np.asarray(
            [float(row["target_time_taken_seconds"]) for row in train_rows]
        )
        test_values = np.asarray([float(row[feature]) for row in test_rows])
        test_target = np.asarray(
            [float(row["target_time_taken_seconds"]) for row in test_rows]
        )
        x_train = np.column_stack(
            (np.ones(len(train_rows)), np.log1p(np.maximum(train_values, 0)))
        )
        coefficients, *_ = np.linalg.lstsq(x_train, np.log1p(train_target), rcond=None)
        x_test = np.column_stack(
            (np.ones(len(test_rows)), np.log1p(np.maximum(test_values, 0)))
        )
        prediction = np.expm1(x_test @ coefficients)
        result[held_out] = {
            "n": len(test_rows),
            "train_groups": [group for group in groups if group != held_out],
            "metrics": regression_metrics(test_target, prediction),
        }
    return result


def build_rows(
    labels: list[dict[str, object]], mali_root: Path, seed: int, optimization_level: int
) -> list[dict[str, object]]:
    backends = {"osaka": FakeOsaka(), "kyoto": FakeKyoto()}
    rows: list[dict[str, object]] = []
    for position, label in enumerate(labels):
        if position % 25 == 0:
            print(f"transpiling {position}/{len(labels)}", flush=True)
        device = str(label["device"])
        name = str(label["circuit"])
        qasm_path = mali_root / "data" / "quantum_circuits" / f"{name}.qasm"
        original = QuantumCircuit.from_qasm_file(str(qasm_path))
        backend = backends[device]
        # Scheduling inside ``transpile(..., scheduling_method="alap")`` is
        # not robust for this Qiskit 1.4/FakeBackend combination (the
        # TimeUnitConversion pass can see stale bit references on otherwise
        # valid large QASM circuits).  We therefore transpile first and
        # reconstruct a deterministic weighted critical path directly from
        # the backend target durations.  This is still a gate-aware proxy,
        # but is explicitly not called an exact ALAP schedule.
        compiled = transpile(
            original,
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=seed,
        )
        weighted_duration_dt, unsupported_duration_ops = weighted_critical_path(
            compiled, backend
        )
        row = row_from_circuits(
            original,
            compiled,
            device,
            name,
            float(label["target"]),
            float(backend.dt),
            weighted_duration_dt,
            unsupported_duration_ops,
            seed,
            optimization_level,
        )
        # Keep the artifact portable: the checkout root is supplied by the
        # reviewer, so only the path relative to that root is recorded.
        row["qasm_path"] = str(qasm_path.relative_to(mali_root))
        row["qasm_sha256"] = hashlib.sha256(qasm_path.read_bytes()).hexdigest()
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], seed: int) -> dict[str, object]:
    for row in rows:
        row.setdefault("shots", 1024)
    target = np.asarray([float(row["target_time_taken_seconds"]) for row in rows])
    feature_summary: dict[str, object] = {}
    for feature in FEATURES:
        values = np.asarray([float(row[feature]) for row in rows])
        pearson, spearman = correlation(np.log1p(values), np.log1p(target))
        predictions, metrics = cross_validated_calibration(values, target, seed)
        prediction_name = f"calibrated_{feature}_seconds"
        for row, prediction in zip(rows, predictions):
            row[prediction_name] = float(prediction)
        feature_summary[feature] = {
            "log1p_pearson": pearson,
            "log1p_spearman": spearman,
            "five_fold_log_calibration": metrics,
        }

    per_backend: dict[str, object] = {}
    for backend in sorted({str(row["backend_label"]) for row in rows}):
        subset = [row for row in rows if row["backend_label"] == backend]
        backend_target = np.asarray([float(row["target_time_taken_seconds"]) for row in subset])
        per_backend[backend] = {
            "n": len(subset),
            "features": {},
        }
        for feature in FEATURES:
            values = np.asarray([float(row[feature]) for row in subset])
            pearson, spearman = correlation(np.log1p(values), np.log1p(backend_target))
            per_backend[backend]["features"][feature] = {
                "log1p_pearson": pearson,
                "log1p_spearman": spearman,
            }

    held_out_backend = {
        feature: held_out_group_calibration(rows, feature, "backend_label")
        for feature in FEATURES
    }

    return {
        "provenance_class": "OUR_PROXY",
        "target_semantics": (
            "Ma-Li upstream Osaka/Kyoto observed result.time_taken from execute(shots=1024), "
            "averaged upstream runs; queue semantics inherited from upstream labels"
        ),
        "proxy_semantics": "current FakeOsaka/FakeKyoto transpile + backend-target weighted critical path; not historical physical circuit truth",
        "n": len(rows),
        "seed": seed,
        "features": feature_summary,
        "per_backend": per_backend,
        "held_out_backend_log_calibration": held_out_backend,
        "software_and_backend_metadata": backend_metadata(),
    }


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Ma–Li / QCRE gate-aware proxy validation",
        "",
        "This is an `OUR_PROXY` analysis, not exact historical hardware validation.",
        "The observed target is Ma–Li's device-labelled `time_taken`; labels come",
        "from the upstream `result.time_taken` path with 1024 shots (averaged over",
        "the upstream repeated runs). The circuit is transpiled with the current",
        "FakeOsaka/FakeKyoto target and a",
        "backend-target weighted critical path is used as the gate-aware proxy.",
        "",
        f"Rows: **{summary['n']}**; seed: **{summary['seed']}**; "
        f"transpile level: **{summary.get('transpile_optimization_level', 'see CSV')}**.",
        "",
        "## Overall log-scale association and five-fold calibration",
        "",
        "| Feature | Pearson | Spearman | MAE (s) | RMSE (s) | R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature, values in summary["features"].items():
        metrics = values["five_fold_log_calibration"]
        lines.append(
            f"| `{feature}` | {values['log1p_pearson']:.4f} | "
            f"{values['log1p_spearman']:.4f} | {metrics['mae_seconds']:.4f} | "
            f"{metrics['rmse_seconds']:.4f} | {metrics['r2_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Leave-one-backend-out calibration",
            "",
            "Each row below trains the one-feature log calibration on the other",
            "backend and scores the held-out backend. This is a domain-transfer",
            "diagnostic, not a claim that two backends are iid.",
            "",
            "| Feature | Held-out backend | n | MAE (s) | RMSE (s) | R² |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for feature, groups in summary["held_out_backend_log_calibration"].items():
        for backend, values in groups.items():
            metrics = values["metrics"]
            lines.append(
                f"| `{feature}` | {backend} | {values['n']} | "
                f"{metrics['mae_seconds']:.4f} | {metrics['rmse_seconds']:.4f} | "
                f"{metrics['r2_seconds']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The table compares association and a held-out one-dimensional log",
            "calibration. It does not establish that the reconstructed physical",
            "circuit or calibration snapshot equals the one used for the original",
            "Ma–Li runtime label. A positive result supports transfer of a timing",
            "proxy; it does not support a universal runtime estimator or compiler",
            "selection claim.",
            "",
            "Per-backend coefficients are in `mali_qcre_proxy_summary.json`; every",
            "row-level feature and calibrated prediction is in the CSV beside this",
            "report. The same JSON records leave-one-backend-out calibration; that",
            "is the relevant domain-transfer check and is not an iid random split.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mali-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=None,
        help="Reuse an existing feature CSV and recompute reports without transpiling.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "validation" / "mali_qcre_proxy"
    )
    args = parser.parse_args()
    mali_root = resolve_mali_root(args.mali_root)
    if args.features_csv:
        with args.features_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["target_semantics"] = (
                "Ma-Li upstream Osaka/Kyoto observed result.time_taken; execute shots=1024; "
                "averaged upstream runs; queue semantics per upstream"
            )
            row.setdefault("shots", 1024)
            qasm_path = mali_root / str(row["qasm_path"])
            if qasm_path.is_file():
                row["qasm_sha256"] = hashlib.sha256(qasm_path.read_bytes()).hexdigest()
    else:
        labels = load_labels(mali_root)
        if args.limit is not None:
            labels = labels[: args.limit]
        rows = build_rows(labels, mali_root, args.seed, args.optimization_level)
    summary = summarize(rows, args.seed)
    levels = {str(row.get("transpile_optimization_level", "unknown")) for row in rows}
    summary["transpile_optimization_level"] = (
        next(iter(levels)) if len(levels) == 1 else sorted(levels)
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "mali_qcre_proxy_features.csv", rows)
    (output_dir / "mali_qcre_proxy_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output_dir / "MALI_QCRE_PROXY_REPORT.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
