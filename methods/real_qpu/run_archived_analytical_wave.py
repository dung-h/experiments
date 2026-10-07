#!/usr/bin/env python3
"""CPU analytical-method wave on the canonical 8,767 archived-real-QPU ledger.

Reuses C133 FakeBackend Targets and C135 transpile/QCRE/Qiskit helpers.
Does not train, does not start CUDA, does not submit QPU jobs, does not
mutate git, and does not overwrite C121/C133/C135/C4 artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from zipfile import ZipFile

import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["QISKIT_IN_PARALLEL"] = "FALSE"

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "benchmark_v1/execution/manifests/archived_real_qpu_analytical_wave_v3_20260930.json"
CANON = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
SPLIT = ROOT / "artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv"
QPACK_STRUCT = ROOT / "artifacts/benchmark_v1/qpack_mcp_structural_reconstruction_20260927/qpack_mcp_structural_rows.csv"
C133_CSV = ROOT / "artifacts/benchmark_v3/recovery_real_qpu_20260929/c133_snapshot_registry_v3.csv"
DEFAULT_OUT = ROOT / "artifacts/benchmark_v3/real_qpu/archived_analytical_wave_v3_20260930"
ROW4477 = "qonductor_single_circuit_ibm|row4477"
ROW4477_REASON = "dynamic_control_no_admissible_snapshot_or_frozen_duration_semantics"
EVAL_CLOCK = "archived_observed_service_execution_time"
WORKERS = 16

CLOPS_V_LIKE = {
    "ibm_jakarta": 2438.0,
    "ibm_nairobi": 2645.0,
    "ibm_perth": 2891.0,
    "ibm_manila": 2819.0,
    "ibm_lagos": 2741.0,
    "ibm_kolkata": 2029.0,
}
DURATION_BEARING_SKIP = {"barrier", "snapshot"}
HYB_SKIP = {"barrier", "snapshot", "delay", "reset"}
RAW_VARIANTS = (
    ("qcre_snapshot_critical_path", "scheduled_critical_path_seconds", "qcre_tau"),
    ("qcre_shot_scaled_schedule_seconds", "shot_scaled_scheduled_seconds", "qcre_shot"),
    ("qiskit_estimate_duration_snapshot", "scheduled_single_shot_seconds", "qiskit_tau"),
    ("qiskit_shot_scaled_schedule_seconds", "shot_scaled_scheduled_seconds", "qiskit_shot"),
    ("hyb_hanas_one_circuit_effective_cost_adaptation", "one_circuit_effective_cost_seconds", "hyb_tau"),
    ("hyb_hanas_shot_linear_effective_seconds", "shot_linear_effective_cost_seconds", "hyb_shot"),
    ("scholten_nominal_single_circuit_adaptation", "nominal_clops_v_like_seconds", "scholten"),
)
ATTEMPT_FIELDS = [
    "canonical_row_id",
    "source_id",
    "source_row_index",
    "backend_observed",
    "backend_canonical",
    "shots",
    "outer_fold",
    "representation_id",
    "reconstruction_qualified",
    "snapshot_id",
    "snapshot_tier",
    "compiled_digest",
    "method_id",
    "evaluation_target_clock",
    "method_output_clock",
    "prediction_seconds",
    "status",
    "terminal_reason",
    "observed_target_seconds",
    "qpack_flag",
    "routing_mode",
]


def load_recovery():
    path = ROOT / "benchmark_v1/scripts/run_recovery_c133_c135_v3.py"
    spec = importlib.util.spec_from_file_location("c133c135_recovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REC = None
_STATE: dict[str, object] = {}


def rec():
    global REC
    if REC is None:
        REC = load_recovery()
    return REC


def qpack_seed_digest(spec: dict[str, str]) -> str:
    recovery = rec()
    return recovery.stable_digest(
        {
            "qpack_revision": recovery.QPACK_REVISION,
            "problem": spec["problem"],
            "size": int(spec["size"]),
            "p": int(spec["p"]),
        }
    )


def cached_backend(name: str):
    cache = _STATE.setdefault("backends", {})
    if name not in cache:
        recovery = rec()
        cache[name] = getattr(recovery.fake_provider, recovery.BACKENDS[name])()
    return cache[name]


def qasm_bytes_for(row: dict[str, str]) -> bytes:
    recovery = rec()
    digest = row["qasm_bytes_sha256"]
    cache = _STATE.setdefault("qasm_bytes", {})
    if digest in cache:
        return cache[digest]
    if row["source_id"] == "mali_real_qpu":
        data = (recovery.MALI_QASM_ROOT / row["qasm_path_or_member"]).read_bytes()
    else:
        archive = _STATE.get("qond_zip")
        if archive is None:
            archive = ZipFile(recovery.QONDUCTOR_ZIP)
            _STATE["qond_zip"] = archive
        data = archive.read(row["qasm_path_or_member"])
    actual = recovery.sha256_bytes(data)
    if actual != digest:
        raise RuntimeError(f"qasm_hash_mismatch:{row['canonical_row_id']}:{actual}")
    cache[digest] = data
    return data


def delay_duration_dt(op: object, dt: float, original_exc: Exception) -> float:
    duration = getattr(op, "duration", None)
    unit = getattr(op, "unit", None) or "dt"
    if duration is None:
        raise ValueError(f"missing_delay_duration:{original_exc}") from original_exc
    duration = float(duration)
    if unit == "dt":
        return duration
    if unit == "s":
        return duration / dt
    if unit == "ns":
        return duration * 1e-9 / dt
    if unit == "us":
        return duration * 1e-6 / dt
    if unit == "ms":
        return duration * 1e-3 / dt
    raise ValueError(f"unsupported_delay_unit:{unit}") from original_exc


def qcre_path_seconds(circuit, backend) -> float:
    """C135 critical-path seconds, using an explicit Delay duration when the Target table omits it."""
    recovery = rec()
    target = backend.target
    durations = target.durations()
    dag = recovery.circuit_to_dag(circuit)
    finish: dict[object, float] = {}
    total = 0.0
    dt = float(target.dt or 0.0)
    if dt <= 0:
        raise ValueError("target_dt_missing")
    for node in dag.topological_op_nodes():
        qargs = tuple(recovery.qubit_index(circuit, q) for q in node.qargs)
        name = str(node.op.name)
        if not recovery.target_supported(target, name, qargs):
            raise ValueError(f"unsupported_operation:{name}:{qargs}")
        start = max((finish.get(pred, 0.0) for pred in dag.predecessors(node)), default=0.0)
        if name in DURATION_BEARING_SKIP:
            duration_dt_value = 0.0
        else:
            try:
                duration_dt_value = float(durations.get(name, qargs))
            except Exception as exc:
                if name == "delay":
                    duration_dt_value = delay_duration_dt(node.op, dt, exc)
                else:
                    raise ValueError(f"missing_duration:{name}:{qargs}:{exc}") from exc
        finish[node] = start + duration_dt_value
        total = max(total, finish[node])
    return total * dt


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha(path)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})
    return sha(path)


def identity_hash(ids: list[str]) -> str:
    material = "\n".join(sorted(ids)) + f"\ncount={len(ids)}\n"
    return hashlib.sha256(material.encode()).hexdigest()


def load_contract() -> dict:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if payload.get("wave4_five_fold_training_authorized") is not False:
        raise SystemExit("contract authorizes V4 training")
    if payload.get("gpu_process_started") is not False:
        raise SystemExit("contract claims GPU")
    return payload


def verify_input_hashes(contract: dict) -> list[str]:
    errors: list[str] = []
    for key, meta in contract["immutable_inputs"].items():
        path = meta.get("path")
        expected = meta.get("sha256")
        if not path or not expected:
            continue
        observed = sha(ROOT / path)
        if observed != expected:
            errors.append(f"{key} hash {observed} != {expected}")
    return errors


def load_tables() -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    with CANON.open(newline="", encoding="utf-8") as handle:
        canon = list(csv.DictReader(handle))
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        split_rows = {row["canonical_observation_id"]: row for row in csv.DictReader(handle)}
    with QPACK_STRUCT.open(newline="", encoding="utf-8") as handle:
        qpack = {row["source_row_index"]: row for row in csv.DictReader(handle)}
    with C133_CSV.open(newline="", encoding="utf-8") as handle:
        snapshots = {row["backend_canonical"]: row for row in csv.DictReader(handle)}
    return canon, split_rows, qpack, snapshots


def materialize_circuit(row: dict[str, str], qpack_rows: dict[str, dict[str, str]], qasm_cache: dict[str, object]):
    recovery = rec()
    source = row["source_id"]
    if source == "qpack_mcp":
        spec = qpack_rows[row["source_row_index"]]
        circuit = recovery.qpack_circuit(int(spec["size"]), int(spec["p"]))
        representation = "pinned_angle_insensitive_to_target_v3"
        digest = row.get("workflow_group_hash") or spec["size"] + "|" + spec["p"]
        return circuit, representation, str(digest), True
    digest = row["qasm_bytes_sha256"]
    circuit = qasm_cache[digest]
    if source == "mali_real_qpu":
        return circuit, "exact_logical_to_pinned_target_v3", digest, False
    return circuit, "exact_submitted_physical_v3", digest, False


def compile_circuit(circuit, source: str, digest: str, backend_name: str, backend):
    recovery = rec()
    if source in {"mali_real_qpu", "qpack_mcp"}:
        seed = recovery.seed_for(digest, source, backend_name)
        compiled = recovery.transpile(
            circuit,
            backend=backend,
            optimization_level=1,
            seed_transpiler=seed,
            layout_method="trivial",
            routing_method="sabre",
            translation_method="translator",
        )
        return compiled, seed, "transpiled"
    if circuit.num_qubits > backend.num_qubits:
        raise RuntimeError(f"width_exceeds_target:{circuit.num_qubits}>{backend.num_qubits}")
    return circuit, "", "exact_submitted_physical"


def unsupported_ops(compiled, backend) -> list[str]:
    recovery = rec()
    found = []
    for inst in compiled.data:
        name = str(inst.operation.name)
        qargs = tuple(recovery.qubit_index(compiled, q) for q in inst.qubits)
        if not recovery.target_supported(backend.target, name, qargs):
            found.append(f"{name}:{qargs}")
    return found


def duration_dt(backend, name: str, qargs: tuple[int, ...]) -> float:
    if name in DURATION_BEARING_SKIP:
        return 0.0
    return float(backend.target.durations().get(name, qargs))


def scholten_depth(compiled) -> int:
    return int(
        compiled.depth(
            filter_function=lambda inst: str(inst.operation.name) not in DURATION_BEARING_SKIP
        )
    )


def hyb_effective_seconds(compiled, backend) -> float:
    recovery = rec()
    props = backend.properties()
    durations = backend.target.durations()
    dt = float(backend.target.dt or 0.0)
    if dt <= 0:
        raise ValueError("target_dt_missing")
    groups: dict[str, list[tuple[float, float]]] = {"one": [], "two": [], "measure": []}
    active: set[int] = set()
    for inst in compiled.data:
        name = str(inst.operation.name)
        qargs = tuple(recovery.qubit_index(compiled, q) for q in inst.qubits)
        if name in DURATION_BEARING_SKIP:
            continue
        active.update(qargs)
        if name in HYB_SKIP:
            continue
        try:
            dur = float(durations.get(name, qargs)) * dt
        except Exception as exc:
            raise ValueError(f"missing_hyb_duration:{name}:{qargs}:{exc}") from exc
        if name == "measure":
            err = float(props.readout_error(qargs[0]))
            groups["measure"].append((dur, err))
        else:
            try:
                err = float(props.gate_error(name, qargs))
            except Exception as exc:
                raise ValueError(f"missing_hyb_error:{name}:{qargs}:{exc}") from exc
            bucket = "two" if len(qargs) == 2 else "one"
            groups[bucket].append((dur, err))
        if not math.isfinite(dur) or not math.isfinite(err):
            raise ValueError(f"non_finite_hyb_calibration:{name}:{qargs}")
    if not active:
        raise ValueError("no_active_qubits")
    t2s = []
    for qubit in sorted(active):
        value = float(props.t2(qubit))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"missing_hyb_T2_active_qubit:{qubit}")
        t2s.append(value)
    t_gate = 0.0
    p_keep = 1.0
    for bucket, items in groups.items():
        n_ops = len(items)
        if n_ops == 0:
            continue
        t_gate += sum(item[0] for item in items)
        mean_err = float(np.mean([item[1] for item in items]))
        p_keep *= (1.0 - mean_err) ** n_ops
    if t_gate < 0 or not math.isfinite(t_gate):
        raise ValueError("hyb_t_gate_invalid")
    p_gate = 1.0 - p_keep
    p_decoh = 1.0 - math.exp(-t_gate / min(t2s))
    p_fail = 1.0 - (1.0 - p_gate) * (1.0 - p_decoh)
    if p_fail >= 1.0 or not math.isfinite(p_fail):
        raise ValueError("hyb_failure_probability_ge_one")
    t_eff = t_gate / (1.0 - p_fail)
    if not math.isfinite(t_eff) or t_eff < 0:
        raise ValueError("hyb_t_eff_invalid")
    return float(t_eff)


def attempt_row(base: dict[str, object], method_id: str, clock: str, value, status: str, reason: str) -> dict[str, object]:
    row = dict(base)
    row.update(
        {
            "method_id": method_id,
            "evaluation_target_clock": EVAL_CLOCK,
            "method_output_clock": clock,
            "prediction_seconds": "" if value is None or value == "" else f"{float(value):.17g}",
            "status": status,
            "terminal_reason": reason,
        }
    )
    return row


def evaluate_compiled(base: dict[str, object], compiled, backend, backend_name: str, shots: float) -> list[dict[str, object]]:
    recovery = rec()
    rows: list[dict[str, object]] = []
    unsupported = unsupported_ops(compiled, backend)
    qcre_value = None
    qiskit_value = None
    qcre_reason = ""
    qiskit_reason = ""
    if unsupported:
        qcre_reason = qiskit_reason = "unsupported_operations:" + ",".join(unsupported[:12])
        if base["canonical_row_id"] == ROW4477:
            qcre_reason = qiskit_reason = ROW4477_REASON
    else:
        try:
            qcre_value = float(qcre_path_seconds(compiled, backend))
            if not math.isfinite(qcre_value):
                raise ValueError("qcre_non_finite")
        except Exception as exc:
            qcre_reason = f"{type(exc).__name__}:{exc}"
            if base["canonical_row_id"] == ROW4477:
                qcre_reason = ROW4477_REASON
        try:
            qiskit_value = float(compiled.estimate_duration(target=backend.target, unit="s"))
            if not math.isfinite(qiskit_value):
                raise ValueError("qiskit_non_finite")
        except Exception as exc:
            qiskit_reason = f"{type(exc).__name__}:{exc}"
            if base["canonical_row_id"] == ROW4477:
                qiskit_reason = ROW4477_REASON

    if qcre_value is None:
        rows.append(attempt_row(base, "qcre_snapshot_critical_path", "scheduled_critical_path_seconds", None, "unavailable", qcre_reason))
        rows.append(attempt_row(base, "qcre_shot_scaled_schedule_seconds", "shot_scaled_scheduled_seconds", None, "unavailable", qcre_reason))
    else:
        rows.append(attempt_row(base, "qcre_snapshot_critical_path", "scheduled_critical_path_seconds", qcre_value, "predicted", ""))
        rows.append(attempt_row(base, "qcre_shot_scaled_schedule_seconds", "shot_scaled_scheduled_seconds", shots * qcre_value, "predicted", ""))

    if qiskit_value is None:
        rows.append(attempt_row(base, "qiskit_estimate_duration_snapshot", "scheduled_single_shot_seconds", None, "unavailable", qiskit_reason))
        rows.append(attempt_row(base, "qiskit_shot_scaled_schedule_seconds", "shot_scaled_scheduled_seconds", None, "unavailable", qiskit_reason))
    else:
        rows.append(attempt_row(base, "qiskit_estimate_duration_snapshot", "scheduled_single_shot_seconds", qiskit_value, "predicted", ""))
        rows.append(attempt_row(base, "qiskit_shot_scaled_schedule_seconds", "shot_scaled_scheduled_seconds", shots * qiskit_value, "predicted", ""))

    hyb_reason = qcre_reason if unsupported else ""
    hyb_value = None
    if not unsupported:
        try:
            hyb_value = hyb_effective_seconds(compiled, backend)
        except Exception as exc:
            hyb_reason = f"{type(exc).__name__}:{exc}"
            if base["canonical_row_id"] == ROW4477:
                hyb_reason = ROW4477_REASON
    elif base["canonical_row_id"] == ROW4477:
        hyb_reason = ROW4477_REASON
    if hyb_value is None:
        rows.append(attempt_row(base, "hyb_hanas_one_circuit_effective_cost_adaptation", "one_circuit_effective_cost_seconds", None, "unavailable", hyb_reason))
        rows.append(attempt_row(base, "hyb_hanas_shot_linear_effective_seconds", "shot_linear_effective_cost_seconds", None, "unavailable", hyb_reason))
    else:
        rows.append(attempt_row(base, "hyb_hanas_one_circuit_effective_cost_adaptation", "one_circuit_effective_cost_seconds", hyb_value, "predicted", ""))
        rows.append(attempt_row(base, "hyb_hanas_shot_linear_effective_seconds", "shot_linear_effective_cost_seconds", shots * hyb_value, "predicted", ""))

    clops = CLOPS_V_LIKE.get(backend_name)
    if clops is None:
        sch_reason = "no_same_backend_nominal_clops_v_like"
        if backend_name in {"ibm_osaka", "ibm_kyoto"}:
            sch_reason = "clops_h_only_never_converted_to_clops_v"
        rows.append(attempt_row(base, "scholten_nominal_single_circuit_adaptation", "nominal_clops_v_like_seconds", None, "unavailable", sch_reason))
    elif unsupported:
        reason = ROW4477_REASON if base["canonical_row_id"] == ROW4477 else "unsupported_operations_for_d_proxy"
        rows.append(attempt_row(base, "scholten_nominal_single_circuit_adaptation", "nominal_clops_v_like_seconds", None, "unavailable", reason))
    else:
        depth = scholten_depth(compiled)
        t_hat = shots * float(depth) / float(clops)
        rows.append(attempt_row(base, "scholten_nominal_single_circuit_adaptation", "nominal_clops_v_like_seconds", t_hat, "predicted", ""))
    return rows


def process_one(payload: dict[str, object]) -> list[dict[str, object]]:
    recovery = rec()
    row = payload["row"]
    qpack_rows = payload["qpack"]
    snapshots = payload["snapshots"]
    backend_name = recovery.canonical_backend(row["backend"])
    snapshot = snapshots[backend_name]
    backend = cached_backend(backend_name)
    shots = float(row["shots"])
    base = {
        "canonical_row_id": row["canonical_row_id"],
        "source_id": row["source_id"],
        "source_row_index": row["source_row_index"],
        "backend_observed": row["backend"],
        "backend_canonical": backend_name,
        "shots": row["shots"],
        "outer_fold": payload["outer_fold"],
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_tier": snapshot["snapshot_tier"],
        "observed_target_seconds": row["target_seconds"],
        "compiled_digest": "",
        "qpack_flag": "",
        "routing_mode": "absorbed_physical" if row["source_id"] == "qonductor_single_circuit_ibm" else "logical_to_target",
    }
    try:
        if row["source_id"] == "qpack_mcp":
            spec = qpack_rows[row["source_row_index"]]
            circuit = recovery.qpack_circuit(int(spec["size"]), int(spec["p"]))
            representation = "pinned_angle_insensitive_to_target_v3"
            digest = qpack_seed_digest(spec)
            reconstruction = True
        else:
            data = payload.get("qasm_bytes") or qasm_bytes_for(row)
            actual = recovery.sha256_bytes(data)
            if actual != row["qasm_bytes_sha256"]:
                raise RuntimeError(f"qasm_hash_mismatch:{actual}")
            circuit = recovery.parse_qasm_bytes(data)
            digest = row["qasm_bytes_sha256"]
            representation = (
                "exact_logical_to_pinned_target_v3"
                if row["source_id"] == "mali_real_qpu"
                else "exact_submitted_physical_v3"
            )
            reconstruction = False
        compiled, seed, _mode = compile_circuit(circuit, row["source_id"], digest, backend_name, backend)
        base["representation_id"] = representation
        base["reconstruction_qualified"] = "true" if reconstruction else "false"
        base["qpack_flag"] = "reconstruction_qualified" if reconstruction else ""
        base["compiled_digest"] = recovery.compiled_ir_digest(compiled)
        base["transpiler_seed"] = seed
        return evaluate_compiled(base, compiled, backend, backend_name, shots)
    except Exception as exc:
        reason = f"{type(exc).__name__}:{exc}"
        if row["canonical_row_id"] == ROW4477:
            reason = ROW4477_REASON
        base.setdefault("representation_id", "")
        base.setdefault("reconstruction_qualified", "false")
        return [
            attempt_row(base, method_id, clock, None, "unavailable", reason)
            for method_id, clock, _key in RAW_VARIANTS
        ]


def representation_digest_for_selection(row: dict[str, str], qpack: dict[str, dict[str, str]]) -> str:
    if row["source_id"] == "qpack_mcp":
        return qpack_seed_digest(qpack[row["source_row_index"]])
    return row["qasm_bytes_sha256"] or row["canonical_row_id"]


def select_preflight(canon: list[dict[str, str]], qpack: dict[str, dict[str, str]]) -> list[str]:
    recovery = rec()
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in canon:
        backend = recovery.canonical_backend(row["backend"])
        grouped[(row["source_id"], backend)].append(row)
    selected: list[str] = []
    for (source, backend), rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda item, source=source, backend=backend: (
                recovery.stable_digest(
                    {
                        "seed": recovery.ROOT_SEED,
                        "stream": recovery.TRANSPILE_STREAM,
                        "context": recovery.CONTEXT_ID,
                        "source": source,
                        "backend": backend,
                        "representation": representation_digest_for_selection(item, qpack),
                    }
                ),
                item["canonical_row_id"],
            ),
        )
        selected.extend(item["canonical_row_id"] for item in ranked[:10])
    if ROW4477 not in selected:
        selected.append(ROW4477)
    return selected


def run_preflight(out: Path, contract: dict) -> dict:
    errors = verify_input_hashes(contract)
    canon, split_rows, qpack, snapshots = load_tables()
    ids = [row["canonical_row_id"] for row in canon]
    ident = identity_hash(ids)
    if len(canon) != 8767:
        errors.append(f"canonical rows {len(canon)} != 8767")
    if len(split_rows) != 8767:
        errors.append(f"split rows {len(split_rows)} != 8767")
    if ROW4477 not in ids:
        errors.append("row4477 missing from canonical ledger")
    selected_ids = select_preflight(canon, qpack)
    by_id = {row["canonical_row_id"]: row for row in canon}
    results: list[dict[str, object]] = []
    cells: dict[str, list[dict[str, object]]] = defaultdict(list)
    for canonical_id in selected_ids:
        row = by_id[canonical_id]
        payload = {
            "row": row,
            "qpack": qpack,
            "snapshots": snapshots,
            "outer_fold": split_rows[canonical_id]["outer_fold"],
        }
        attempts = process_one(payload)
        qiskit_tau = next(item for item in attempts if item["method_id"] == "qiskit_estimate_duration_snapshot")
        qcre_tau = next(item for item in attempts if item["method_id"] == "qcre_snapshot_critical_path")
        qiskit_shot = next(item for item in attempts if item["method_id"] == "qiskit_shot_scaled_schedule_seconds")
        expected = canonical_id == ROW4477
        rel = ""
        check = "PASS"
        reason = ""
        if expected:
            if qiskit_tau["status"] != "unavailable" or qcre_tau["status"] != "unavailable":
                check = "FAIL"
                reason = "row4477_expected_unavailable_but_predicted"
            elif qiskit_tau["terminal_reason"] != ROW4477_REASON:
                check = "FAIL"
                reason = f"row4477_reason_drift:{qiskit_tau['terminal_reason']}"
            else:
                reason = ROW4477_REASON
        else:
            if qiskit_tau["status"] != "predicted" or qcre_tau["status"] != "predicted":
                check = "FAIL"
                reason = qiskit_tau["terminal_reason"] or qcre_tau["terminal_reason"] or "eligible_row_unavailable"
            else:
                qv = float(qiskit_tau["prediction_seconds"])
                cv = float(qcre_tau["prediction_seconds"])
                rel = abs(cv - qv) / max(abs(qv), 1e-30)
                if rel > 0.05:
                    check = "FAIL"
                    reason = f"qcre_qiskit_relative_error:{rel}"
                shot_expected = float(row["shots"]) * qv
                if abs(float(qiskit_shot["prediction_seconds"]) - shot_expected) / max(abs(shot_expected), 1e-30) > 1e-12:
                    check = "FAIL"
                    reason = "shot_scaling_mismatch"
                if row["source_id"] in {"mali_real_qpu", "qpack_mcp"}:
                    recovery = rec()
                    backend_name = recovery.canonical_backend(row["backend"])
                    backend = getattr(recovery.fake_provider, recovery.BACKENDS[backend_name])()
                    if row["source_id"] == "qpack_mcp":
                        spec = qpack[row["source_row_index"]]
                        circuit = recovery.qpack_circuit(int(spec["size"]), int(spec["p"]))
                        digest = qpack_seed_digest(spec)
                    else:
                        circuit = recovery.parse_qasm_bytes(qasm_bytes_for(row))
                        digest = row["qasm_bytes_sha256"]
                    first = compile_circuit(circuit, row["source_id"], digest, backend_name, backend)[0]
                    second = compile_circuit(circuit, row["source_id"], digest, backend_name, backend)[0]
                    if recovery.compiled_ir_digest(first) != recovery.compiled_ir_digest(second):
                        check = "FAIL"
                        reason = "transpile_digest_unstable"
        record = {
            "canonical_row_id": canonical_id,
            "source_id": row["source_id"],
            "backend_canonical": rec().canonical_backend(row["backend"]),
            "expected_unavailable": expected,
            "qiskit_status": qiskit_tau["status"],
            "qcre_status": qcre_tau["status"],
            "qiskit_seconds": qiskit_tau["prediction_seconds"],
            "qcre_seconds": qcre_tau["prediction_seconds"],
            "relative_error": rel,
            "checks_status": check,
            "failure_reason": reason,
            "compiled_digest": qiskit_tau.get("compiled_digest", ""),
        }
        results.append(record)
        cells[f"{record['source_id']}|{record['backend_canonical']}"].append(record)

    unexpected = [row for row in results if row["checks_status"] != "PASS"]
    cell_summary = []
    for cell, members in sorted(cells.items()):
        cell_ok = all(item["checks_status"] == "PASS" for item in members)
        cell_summary.append(
            {
                "preflight_cell": cell,
                "attempts": len(members),
                "successes": sum(item["checks_status"] == "PASS" for item in members),
                "status": "PASS" if cell_ok else "FAIL",
            }
        )
    status = "PASS" if not errors and not unexpected else "FAIL"
    write_csv(out / "preflight_results.csv", results, list(results[0]))
    write_csv(out / "preflight_cell_summary.csv", cell_summary, list(cell_summary[0]))
    manifest = {
        "artifact_id": "archived-real-qpu-analytical-wave-v3-preflight",
        "status": status,
        "canonical_identity_hash": ident,
        "canonical_n": len(canon),
        "selected_n": len(selected_ids),
        "row4477_included": ROW4477 in selected_ids,
        "row4477_expected_unavailable": True,
        "errors": errors,
        "failures": [row["canonical_row_id"] + ":" + str(row["failure_reason"]) for row in unexpected],
        "cells": cell_summary,
        "gpu_process_started": False,
        "wave4_five_fold_training_authorized": False,
        "c121_reused": False,
    }
    write_json(out / "preflight_manifest.json", manifest)
    print(json.dumps({"stage": "preflight", "status": status, "selected_n": len(selected_ids), "failures": manifest["failures"]}, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit("preflight failed")
    return manifest


def shard_key(row: dict[str, str]) -> str:
    recovery = rec()
    return f"{row['source_id']}__{recovery.canonical_backend(row['backend'])}"


def execute_shard(item: tuple[str, list[dict[str, str]], str, dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]) -> str:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["QISKIT_IN_PARALLEL"] = "FALSE"
    name, rows, path_str, split_rows, qpack, snapshots = item
    path = Path(path_str)
    tmp = path.with_suffix(".partial.csv")
    attempts: list[dict[str, object]] = []
    for row in rows:
        payload = {
            "row": row,
            "qpack": qpack,
            "snapshots": snapshots,
            "outer_fold": split_rows[row["canonical_row_id"]]["outer_fold"],
        }
        attempts.extend(process_one(payload))
    write_csv(tmp, attempts, ATTEMPT_FIELDS)
    tmp.replace(path)
    return name


def run_execute(out: Path, contract: dict) -> None:
    preflight = json.loads((out / "preflight_manifest.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise SystemExit("refusing full execution; preflight is not PASS")
    canon, split_rows, qpack, snapshots = load_tables()
    shards: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in canon:
        shards[shard_key(row)].append(row)
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for name, rows in sorted(shards.items()):
        shard_path = shard_dir / f"{name}.csv"
        if shard_path.exists() and shard_path.stat().st_size > 0:
            continue
        tasks.append((name, rows, str(shard_path), split_rows, qpack, snapshots))
    print(json.dumps({"stage": "execute", "pending_shards": len(tasks), "total_shards": len(shards)}, indent=2))
    if not tasks:
        return
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(execute_shard, item): item[0] for item in tasks}
        for future in as_completed(futures):
            name = futures[future]
            exc = future.exception()
            if exc:
                raise SystemExit(f"shard {name} failed: {exc}")
            print(json.dumps({"shard_done": name}, sort_keys=True), flush=True)


def fit_ols(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    if len(xs) < 2:
        return None
    matrix = np.column_stack([np.ones(len(xs)), np.asarray(xs, dtype=float)])
    try:
        coef, _, _, _ = np.linalg.lstsq(matrix, np.asarray(ys, dtype=float), rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b = float(coef[0]), float(coef[1])
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return a, b


def apply_affine(a: float, b: float, x: float) -> float:
    return max(0.0, a + b * x)


def apply_log_affine(a: float, b: float, x: float) -> float:
    return max(0.0, math.expm1(a + b * math.log1p(x)))


def load_all_attempts(out: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((out / "shards").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def mae_r2(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, r2


def run_aggregate(out: Path) -> dict:
    attempts = load_all_attempts(out)
    if not attempts:
        raise SystemExit("no shard attempts to aggregate")
    n_obs = len({row["canonical_row_id"] for row in attempts})
    if n_obs != 8767:
        raise SystemExit(f"attempt envelope covers {n_obs} observations, not 8767")
    calibrated: list[dict[str, str]] = []
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in attempts:
        by_method[row["method_id"]].append(row)
    for method_id, rows in by_method.items():
        clock = rows[0]["method_output_clock"]
        for fold in sorted({row["outer_fold"] for row in rows}):
            train = [
                row
                for row in rows
                if row["outer_fold"] != fold
                and row["status"] == "predicted"
                and row["prediction_seconds"]
                and row["observed_target_seconds"]
            ]
            xs = [float(row["prediction_seconds"]) for row in train]
            ys = [float(row["observed_target_seconds"]) for row in train]
            affine = fit_ols(xs, ys)
            log_xs = [math.log1p(x) for x in xs]
            log_ys = [math.log1p(y) for y in ys]
            log_fit = fit_ols(log_xs, log_ys) if xs else None
            test_rows = [row for row in rows if row["outer_fold"] == fold]
            for row in test_rows:
                base = dict(row)
                if row["status"] != "predicted" or not row["prediction_seconds"]:
                    calibrated.append(
                        attempt_row(
                            base,
                            f"{method_id}_outer_train_affine_seconds",
                            "calibrated_observed_service_seconds",
                            None,
                            "unavailable",
                            row["terminal_reason"] or "raw_unavailable",
                        )
                    )
                    calibrated.append(
                        attempt_row(
                            base,
                            f"{method_id}_outer_train_log_affine_seconds",
                            "calibrated_observed_service_seconds",
                            None,
                            "unavailable",
                            row["terminal_reason"] or "raw_unavailable",
                        )
                    )
                    continue
                x = float(row["prediction_seconds"])
                if affine is None:
                    calibrated.append(
                        attempt_row(base, f"{method_id}_outer_train_affine_seconds", "calibrated_observed_service_seconds", None, "unavailable", "insufficient_outer_train_points")
                    )
                else:
                    calibrated.append(
                        attempt_row(base, f"{method_id}_outer_train_affine_seconds", "calibrated_observed_service_seconds", apply_affine(*affine, x), "predicted", "")
                    )
                if log_fit is None:
                    calibrated.append(
                        attempt_row(base, f"{method_id}_outer_train_log_affine_seconds", "calibrated_observed_service_seconds", None, "unavailable", "insufficient_outer_train_points")
                    )
                else:
                    calibrated.append(
                        attempt_row(base, f"{method_id}_outer_train_log_affine_seconds", "calibrated_observed_service_seconds", apply_log_affine(*log_fit, x), "predicted", "")
                    )
    all_rows = attempts + calibrated
    write_csv(out / "attempts.csv", all_rows, ATTEMPT_FIELDS)

    metrics = []
    for method_id, rows in sorted(defaultdict(list, {row["method_id"]: [] for row in all_rows}).items()):
        pass
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        grouped[row["method_id"]].append(row)
    for method_id, rows in sorted(grouped.items()):
        predicted = [row for row in rows if row["status"] == "predicted" and row["prediction_seconds"]]
        unavailable = [row for row in rows if row["status"] != "predicted"]
        y = np.array([float(row["observed_target_seconds"]) for row in predicted], dtype=float) if predicted else np.array([])
        yhat = np.array([float(row["prediction_seconds"]) for row in predicted], dtype=float) if predicted else np.array([])
        mae = r2 = ""
        if len(predicted) >= 1:
            mae_v, r2_v = mae_r2(y, yhat)
            mae, r2 = f"{mae_v:.17g}", f"{r2_v:.17g}"
        source_counts = Counter(row["source_id"] for row in predicted)
        unavail_reasons = Counter(row["terminal_reason"] or "unspecified" for row in unavailable)
        row_ids = sorted(row["canonical_row_id"] for row in predicted)
        metrics.append(
            {
                "method_id": method_id,
                "evaluation_target_clock": EVAL_CLOCK,
                "method_output_clock": rows[0]["method_output_clock"],
                "assigned_n": 8767,
                "predicted_n": len(predicted),
                "unavailable_n": len(unavailable),
                "n_metric": len(predicted),
                "mae_seconds": mae,
                "r2_seconds": r2,
                "coverage_fraction": f"{len(predicted)/8767:.17g}",
                "row_set_hash": identity_hash(row_ids) if row_ids else "",
                "mali_predicted_n": source_counts.get("mali_real_qpu", 0),
                "qonductor_predicted_n": source_counts.get("qonductor_single_circuit_ibm", 0),
                "qpack_predicted_n": source_counts.get("qpack_mcp", 0),
                "unavailable_reason_json": json.dumps(unavail_reasons, sort_keys=True),
                "row4477_status": next((row["status"] for row in rows if row["canonical_row_id"] == ROW4477), ""),
                "row4477_reason": next((row["terminal_reason"] for row in rows if row["canonical_row_id"] == ROW4477), ""),
            }
        )
    write_csv(out / "method_metrics.csv", metrics, list(metrics[0]))

    source_rows = []
    for method_id, rows in sorted(grouped.items()):
        for source in ("mali_real_qpu", "qonductor_single_circuit_ibm", "qpack_mcp"):
            subset = [row for row in rows if row["source_id"] == source]
            predicted = [row for row in subset if row["status"] == "predicted" and row["prediction_seconds"]]
            assigned = len(subset)
            mae = r2 = ""
            if predicted:
                y = np.array([float(row["observed_target_seconds"]) for row in predicted], dtype=float)
                yhat = np.array([float(row["prediction_seconds"]) for row in predicted], dtype=float)
                mae_v, r2_v = mae_r2(y, yhat)
                mae, r2 = f"{mae_v:.17g}", f"{r2_v:.17g}"
            source_rows.append(
                {
                    "method_id": method_id,
                    "source_id": source,
                    "evaluation_target_clock": EVAL_CLOCK,
                    "method_output_clock": subset[0]["method_output_clock"] if subset else "",
                    "assigned_n": assigned,
                    "predicted_n": len(predicted),
                    "unavailable_n": assigned - len(predicted),
                    "n_metric": len(predicted),
                    "mae_seconds": mae,
                    "r2_seconds": r2,
                    "reconstruction_qualified": "true" if source == "qpack_mcp" else "false",
                    "row_set_hash": identity_hash([row["canonical_row_id"] for row in predicted]) if predicted else "",
                }
            )
    write_csv(out / "source_stratified_metrics.csv", source_rows, list(source_rows[0]))

    raw_ids = [method_id for method_id, _clock, _key in RAW_VARIANTS]
    successful_sets = {
        method_id: {row["canonical_row_id"] for row in grouped[method_id] if row["status"] == "predicted"}
        for method_id in raw_ids
        if method_id in grouped
    }
    intersections = []
    methods = list(successful_sets)
    for i, left in enumerate(methods):
        for right in methods[i + 1 :]:
            shared = sorted(successful_sets[left] & successful_sets[right])
            intersections.append(
                {
                    "method_id": left,
                    "paired_method_id": right,
                    "n_shared": len(shared),
                    "row_set_hash": identity_hash(shared) if shared else "",
                }
            )
            intersections.append(
                {
                    "method_id": right,
                    "paired_method_id": left,
                    "n_shared": len(shared),
                    "row_set_hash": identity_hash(shared) if shared else "",
                }
            )
            for method_id in (left, right):
                subset = [
                    row
                    for row in grouped[method_id]
                    if row["canonical_row_id"] in set(shared) and row["status"] == "predicted"
                ]
                if not subset:
                    continue
                y = np.array([float(row["observed_target_seconds"]) for row in subset], dtype=float)
                yhat = np.array([float(row["prediction_seconds"]) for row in subset], dtype=float)
                mae_v, r2_v = mae_r2(y, yhat)
                intersections[-2 if method_id == left else -1].update(
                    {
                        "mae_seconds": f"{mae_v:.17g}",
                        "r2_seconds": f"{r2_v:.17g}",
                        "evaluation_target_clock": EVAL_CLOCK,
                        "method_output_clock": subset[0]["method_output_clock"],
                    }
                )
    if intersections:
        write_csv(out / "common_successful_intersections.csv", intersections, sorted({key for row in intersections for key in row}))

    unavailable = [row for row in attempts if row["status"] != "predicted"]
    write_csv(out / "unavailable_appendix.csv", unavailable, ATTEMPT_FIELDS)
    row4477_rows = [row for row in attempts if row["canonical_row_id"] == ROW4477]
    write_csv(out / "row4477_attempts.csv", row4477_rows, ATTEMPT_FIELDS)

    report = {
        "artifact_id": "archived-real-qpu-analytical-wave-v3",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_n": 8767,
        "canonical_identity_hash": identity_hash(sorted({row["canonical_row_id"] for row in attempts})),
        "attempt_rows_raw": len(attempts),
        "attempt_rows_with_calibration": len(all_rows),
        "methods": metrics,
        "row4477": {
            "canonical_row_id": ROW4477,
            "expected_reason": ROW4477_REASON,
            "attempts": [
                {"method_id": row["method_id"], "status": row["status"], "reason": row["terminal_reason"]}
                for row in row4477_rows
            ],
        },
        "not_evaluated": [{"method_id": "qcre_gate_aware_depth_rank", "reason": "not_evaluated_no_covering_architecture_map"}],
        "c121_reused": False,
        "gpu_process_started": False,
        "wave4_five_fold_training_authorized": False,
        "git_mutated": False,
        "evaluation_target_clock": EVAL_CLOCK,
        "workers": WORKERS,
        "python": sys.executable,
    }
    write_json(out / "result_manifest.json", report)
    lines = [
        "# Archived real-QPU analytical wave v3",
        "",
        f"Status: {report['status']}. Canonical n=8767. CPU-only. C121 not reused. V4 not trained.",
        "",
        f"row4477 remains in the denominator. Schedule-dependent methods are unavailable with `{ROW4477_REASON}`.",
        "",
        "| method_id | method_output_clock | predicted_n | unavailable_n | MAE | R2 |",
        "|---|---|---|---|---|---|",
    ]
    for row in metrics:
        if row["method_id"].endswith("_seconds") and "outer_train" in row["method_id"]:
            continue
        if row["method_id"] in raw_ids or row["method_id"].endswith("_seconds") and "outer_train" not in row["method_id"]:
            lines.append(
                f"| {row['method_id']} | {row['method_output_clock']} | {row['predicted_n']} | {row['unavailable_n']} | {row['mae_seconds']} | {row['r2_seconds']} |"
            )
    lines += [
        "",
        "Raw scheduled/effective/nominal clocks are not observed service time. Calibrated variants use outer-train OLS only.",
        "QPack rows are reconstruction_qualified. GAD was not evaluated (no covering architecture map).",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"stage": "aggregate", "status": "PASS", "raw_attempts": len(attempts), "all_attempts": len(all_rows)}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stage", choices=("preflight", "execute", "aggregate", "all"), default="all")
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    contract = load_contract()
    source = Path(__file__).read_text(encoding="utf-8")
    for name in ("add", "commit", "reset", "clean", "rm", "checkout"):
        if f'git(["{name}"' in source:
            raise SystemExit(f"forbidden git invocation {name}")
    if "Does not train" not in source:
        raise SystemExit("lost the no-training boundary")
    if args.stage in {"preflight", "all"}:
        out.mkdir(parents=True, exist_ok=True)
        existing_path = out / "preflight_manifest.json"
        if existing_path.exists() and args.stage == "all":
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if existing.get("status") == "PASS":
                print(json.dumps({"stage": "preflight", "status": "PASS", "reused": True}, indent=2))
            else:
                run_preflight(out, contract)
        else:
            run_preflight(out, contract)
    if args.stage in {"execute", "all"}:
        run_execute(out, contract)
    if args.stage in {"aggregate", "all"}:
        run_aggregate(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
