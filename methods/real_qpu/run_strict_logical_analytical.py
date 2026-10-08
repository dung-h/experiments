#!/usr/bin/env python3
"""Evaluate frozen analytical adapters on the strict 340-row Ma-Li cohort.

Run with the pinned BQSKit/Qiskit adapter environment. All physical inputs are
the already frozen nominal compiled QASM views of the same source-logical rows.
"""
from __future__ import annotations

import argparse
import csv
import concurrent.futures
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit_ibm_runtime import fake_provider

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "artifacts/real_qpu/strict_logical_panel"
PARENT = ROOT / "benchmark_v1/protocol/strict_logical_benchmark.json"
COMPLETION = ROOT / "benchmark_v1/protocol/strict_method_completion.json"
AUTH = PANEL / "execution_authorization.json"
REPRESENTATION = ROOT / "artifacts/benchmark_v3/real_qpu/compiled_unified/representation_rows.csv"
REGISTRY = ROOT / "artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv"
SCHOLTEN_PROTOCOL = ROOT / "benchmark_v1/protocol/scholten_unified_nominal.json"
QCRE_ROOT = ROOT.parent / "qcre"
QCRE_REVISION = "b3505da6bfa6d1184c21eb8410708cb759eadc48"
QCRE_FILE_SHA256 = "e85873a810519c2741df281de5f9a8c6a89cedf80c17da65b74c7df39fa497c0"
SCHOLTEN = PANEL / "scholten"
OUTPUT = PANEL / "analytical"
BACKENDS = {"ibm_osaka": "FakeOsaka", "ibm_kyoto": "FakeKyoto"}
QCRE_ARITY = {"id": 1, "rx": 1, "rz": 1, "sx": 1, "x": 1, "cz": 2, "ecr": 2}
OMIT_UNITARY = {"measure", "barrier", "snapshot"}
UNSUPPORTED_UNITARY = {"reset", "delay", "if_else", "while_loop", "for_loop", "switch_case"}
TARGET_CLOCK = "archived_mean_result_time_taken_seconds"
FIELDS = ["canonical_observation_id", "outer_fold", "backend", "source_logical_qasm_sha256",
          "compiled_qasm_sha256", "snapshot_id", "snapshot_tier", "evaluation_target_clock",
          "method_id", "method_output_clock", "target_seconds", "prediction_seconds", "status",
          "terminal_reason", "omitted_measurement_count", "compiled_wire_depth",
          "auxiliary_log_cost", "shots"]
CALIBRATABLE_METHODS = ["strict_qiskit_schedule_raw", "strict_qiskit_schedule_shot_scaled",
    "strict_qcre_source_estimator_raw", "strict_qcre_source_estimator_shot_scaled",
    "strict_hyb_effective_cost_raw", "strict_hyb_effective_cost_shot_scaled",
    "strict_scholten_depth_throughput_proxy", "strict_scholten_same_width_qv_extension"]
_ANALYTICAL_WORKER_STATE: dict[str, Any] | None = None


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def standardized_ridge_fit_predict(x_fit, y_fit, x_predict, alpha):
    """Closed-form dense equivalent of StandardScaler + Ridge(fit_intercept=True)."""
    x_fit = np.asarray(x_fit, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    x_predict = np.asarray(x_predict, dtype=np.float64)
    center = x_fit.mean(axis=0)
    scale = x_fit.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    z_fit = (x_fit - center) / scale
    z_predict = (x_predict - center) / scale
    y_center = float(y_fit.mean())
    gram = z_fit.T @ z_fit + float(alpha) * np.eye(z_fit.shape[1])
    coef = np.linalg.solve(gram, z_fit.T @ (y_fit - y_center))
    return y_center + z_predict @ coef


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] = FIELDS) -> None:
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode())


def module_at(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module_load_failed:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_sources():
    if sha_file(PARENT) != json.loads(AUTH.read_text())["bound_inputs"]["parent_protocol_sha256"]:
        raise RuntimeError("strict_parent_protocol_pin_mismatch")
    if sha_file(COMPLETION) != json.loads(AUTH.read_text())["bound_inputs"]["completion_protocol_sha256"]:
        raise RuntimeError("strict_completion_protocol_pin_mismatch")
    if sha_file(PANEL / "manifest.json") != json.loads(AUTH.read_text())["bound_inputs"]["panel_manifest_sha256"]:
        raise RuntimeError("strict_panel_manifest_pin_mismatch")
    source_git = __import__("subprocess").check_output(["git", "-C", str(QCRE_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if source_git != QCRE_REVISION or sha_file(QCRE_ROOT / "estimate.py") != QCRE_FILE_SHA256:
        raise RuntimeError("qcre_source_pin_mismatch")
    if importlib.util.find_spec("bqskit") is None:
        site = os.environ.get("STRICT_BQSKIT_SITE", "")
        if not site or not Path(site, "bqskit").is_dir():
            raise RuntimeError("install_bqskit_1_2_in_active_environment_or_set_STRICT_BQSKIT_SITE")
        # The active Qiskit version has already been imported at module load;
        # this path supplies only the pinned BQSKit adapter and its dependencies.
        sys.path.append(site)
        importlib.invalidate_caches()
    sys.path.insert(0, str(QCRE_ROOT))
    from estimate import estimate_runtime
    from bqskit.ext import qiskit as bq_qiskit
    archived = module_at(ROOT / "benchmark_v1/scripts/run_archived_analytical_wave_v3.py", "strict_archived_helpers")
    return estimate_runtime, bq_qiskit, archived


def load_targets():
    registry = {r["backend_canonical"]: r for r in read_csv(REGISTRY)}
    targets, duration_maps = {}, {}
    for backend, class_name in BACKENDS.items():
        row = registry[backend]
        target = getattr(fake_provider, class_name)()
        package_dir = Path(fake_provider.__file__).parent / "backends" / backend.removeprefix("ibm_")
        assets = ((package_dir / f"conf_{backend.removeprefix('ibm_')}.json", row["configuration_sha256"]),
                  (package_dir / f"props_{backend.removeprefix('ibm_')}.json", row["properties_sha256"]))
        for asset, expected in assets:
            if not asset.is_file() or sha_file(asset) != expected:
                raise RuntimeError(f"strict_snapshot_asset_hash_mismatch:{backend}:{asset.name}")
        dmap = {}
        durations = target.target.durations()
        for name, arity in QCRE_ARITY.items():
            if name not in target.target.operation_names:
                continue
            for location in target.target[name]:
                loc = tuple(int(q) for q in location)
                value = float(durations.get(name, loc, unit="s"))
                if not math.isfinite(value) or value < 0:
                    raise RuntimeError(f"invalid_snapshot_duration:{backend}:{name}:{loc}")
                dmap.setdefault(arity, {}).setdefault(loc, {})[name] = value
        targets[backend] = target
        duration_maps[backend] = dmap
    return targets, duration_maps, registry


def source_and_representation():
    cohort = read_csv(PANEL / "logical_features.csv")
    reps = [r for r in read_csv(REPRESENTATION) if r["source_id"] == "mali_real_qpu"]
    by_rep = {r["canonical_row_id"]: r for r in reps}
    if len(cohort) != 340 or len({r["canonical_observation_id"] for r in cohort}) != 340 or set(by_rep) != {r["canonical_observation_id"] for r in cohort}:
        raise RuntimeError("strict_340_representation_identity_mismatch")
    outer = {r["canonical_observation_id"]: int(r["outer_fold"]) for r in read_csv(PANEL / "outer_splits.csv")}
    inner = {(r["canonical_observation_id"], int(r["outer_fold"])): int(r["inner_fold"])
             for r in read_csv(PANEL / "inner_splits.csv")}
    return cohort, by_rep, outer, inner


def qiskit_wire_depth(circuit: QuantumCircuit) -> int:
    depth = [0] * circuit.num_qubits
    seen_measurement = False
    for item in circuit.data:
        name = str(item.operation.name)
        if name == "measure":
            seen_measurement = True
            continue
        if name in {"barrier", "snapshot"}:
            continue
        if name in UNSUPPORTED_UNITARY:
            raise ValueError(f"unsupported_scholten_unitary_instruction:{name}")
        if seen_measurement:
            raise ValueError("scholten_mid_circuit_measurement_unsupported")
        wires = [circuit.find_bit(q).index for q in item.qubits]
        end = max((depth[q] for q in wires), default=0) + (1 if wires else 0)
        for q in wires:
            depth[q] = end
    return max(depth, default=0)


def qcre_unitary_input(circuit: QuantumCircuit) -> tuple[QuantumCircuit, int]:
    unitary = QuantumCircuit(circuit.num_qubits)
    unitary.global_phase = circuit.global_phase
    seen_measurement = False
    omitted_measurements = 0
    for item in circuit.data:
        name = str(item.operation.name)
        if name == "measure":
            seen_measurement = True
            omitted_measurements += 1
            continue
        if name in {"barrier", "snapshot"}:
            continue
        if name in UNSUPPORTED_UNITARY:
            raise ValueError(f"qcre_unsupported_nonunitary_operation:{name}")
        if seen_measurement:
            raise ValueError("qcre_nonterminal_measurement")
        if name not in QCRE_ARITY:
            raise ValueError(f"qcre_gate_not_in_source_duration_map:{name}")
        qubits = [unitary.qubits[circuit.find_bit(q).index] for q in item.qubits]
        unitary.append(item.operation, qubits)
    return unitary, omitted_measurements


def qcre_custom_path(circuit: QuantumCircuit, duration_map: dict) -> float:
    clock = [0.0] * circuit.num_qubits
    for item in circuit.data:
        name = str(item.operation.name)
        if name in OMIT_UNITARY:
            continue
        qargs = tuple(circuit.find_bit(q).index for q in item.qubits)
        duration = float(duration_map[len(qargs)][qargs][name])
        end = max((clock[q] for q in qargs), default=0.0) + duration
        for q in qargs:
            clock[q] = end
    return max(clock, default=0.0)


def hyb_log_cost(circuit: QuantumCircuit, backend) -> float:
    """Paper-derived log cost from the frozen analytical-extension contract."""
    props = backend.properties()
    durations = backend.target.durations()
    dt = float(backend.target.dt or 0.0)
    if dt <= 0:
        raise ValueError("target_dt_missing")
    buckets: dict[str, list[float]] = {"one": [], "two": [], "readout": []}
    active: set[int] = set()
    t_gate = 0.0
    for item in circuit.data:
        name = str(item.operation.name)
        qargs = tuple(circuit.find_bit(q).index for q in item.qubits)
        if name in {"barrier", "snapshot", "delay", "reset"}:
            continue
        if name in UNSUPPORTED_UNITARY:
            raise ValueError(f"hyb_unsupported_operation:{name}")
        active.update(qargs)
        try:
            duration = float(durations.get(name, qargs, unit="s"))
        except Exception as exc:
            raise ValueError(f"hyb_missing_duration:{name}:{qargs}:{exc}") from exc
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f"hyb_invalid_duration:{name}:{qargs}")
        t_gate += duration
        if name == "measure":
            err = float(props.readout_error(qargs[0]))
            bucket = "readout"
        else:
            err = float(props.gate_error(name, qargs))
            bucket = "two" if len(qargs) == 2 else "one"
        if not math.isfinite(err) or not 0 <= err <= 1:
            raise ValueError(f"hyb_invalid_error:{name}:{qargs}:{err}")
        buckets[bucket].append(err)
    if t_gate <= 0 or not active:
        raise ValueError("hyb_gate_time_or_active_wire_missing")
    t2 = [float(props.t2(q)) for q in sorted(active)]
    if not t2 or any(not math.isfinite(x) or x <= 0 for x in t2):
        raise ValueError("hyb_invalid_t2_on_active_wire")
    penalty = t_gate / min(t2)
    for errors in buckets.values():
        if errors:
            mean_error = float(np.mean(errors))
            if mean_error >= 1:
                raise ValueError("hyb_zero_survival_bucket")
            penalty -= len(errors) * math.log1p(-mean_error)
    return float(math.log(t_gate) + penalty)


def attempt(base, method, output_clock, value=None, reason="", omitted=0, depth="", log_cost=""):
    return {"canonical_observation_id": base["canonical_observation_id"],
        "outer_fold": base["outer_fold"], "backend": base["backend"],
        "source_logical_qasm_sha256": base["logical_qasm_sha256"],
        "compiled_qasm_sha256": base["compiled_qasm3_sha256"],
        "snapshot_id": base["snapshot_id"], "snapshot_tier": base["snapshot_tier"],
        "evaluation_target_clock": TARGET_CLOCK, "method_id": method,
        "method_output_clock": output_clock,
        "target_seconds": base["target_seconds"],
        "prediction_seconds": "" if value is None else format(float(value), ".17g"),
        "status": "unavailable" if value is None else "predicted",
        "terminal_reason": reason, "omitted_measurement_count": omitted,
        "compiled_wire_depth": depth, "auxiliary_log_cost": log_cost,
        "shots": base["shots"]}


def parity_preflight(estimate_runtime, converter, targets, duration_maps):
    results = {}
    for backend_name, backend in targets.items():
        dmap = duration_maps[backend_name]
        ecr_locations = sorted(dmap.get(2, {}))
        if not ecr_locations:
            raise RuntimeError(f"qcre_fixture_target_missing_two_qubit_gate:{backend_name}")
        pair = ecr_locations[0]
        width = max(max(pair) + 1, 3)
        qc = QuantumCircuit(width)
        qc.rz(0.5, 0)
        qc.sx(0)
        qc.sx(2)
        qc.ecr(*pair)
        bqc = converter.qiskit_to_bqskit(qc)
        direct = float(estimate_runtime(bqc, dmap))
        oracle = qcre_custom_path(qc, dmap)
        if not math.isclose(direct, oracle, rel_tol=1e-12, abs_tol=1e-15):
            raise RuntimeError(f"qcre_custom_path_parity_failed:{backend_name}:{direct}:{oracle}")
        x_seconds = float(backend.target.durations().get("x", (0,), unit="s"))
        x_dt = float(backend.target.durations().get("x", (0,), unit="dt"))
        if not math.isclose(x_seconds, x_dt * float(backend.target.dt), rel_tol=1e-12, abs_tol=1e-15):
            raise RuntimeError(f"qcre_duration_unit_conversion_failed:{backend_name}")
        rz = float(backend.target.durations().get("rz", (0,), unit="s"))
        if rz != 0.0:
            raise RuntimeError(f"qcre_virtual_rz_snapshot_not_zero:{backend_name}:{rz}")
        expected_ecr = float(backend.target.durations().get("ecr", pair, unit="s"))
        if not math.isclose(dmap[2][pair]["ecr"], expected_ecr, rel_tol=1e-12, abs_tol=1e-15):
            raise RuntimeError(f"qcre_placement_duration_lookup_failed:{backend_name}")
        results[backend_name] = {"status": "PASS", "fixture_qubits": width,
            "fixture_ecr_location": list(pair), "source_estimator_seconds": direct,
            "independent_wire_path_seconds": oracle, "x_seconds": x_seconds,
            "x_dt": x_dt, "dt_seconds": float(backend.target.dt), "rz_seconds": rz,
            "placement_ecr_seconds": expected_ecr}
    return results


def preflight():
    estimate_runtime, converter, archived = load_sources()
    targets, duration_maps, registry = load_targets()
    parity = parity_preflight(estimate_runtime, converter, targets, duration_maps)
    result = {"artifact_id": "strict-logical-analytical-adapter-preflight", "status": "PASS",
        "source_qcre_revision": QCRE_REVISION, "qcre_estimate_sha256": sha_file(QCRE_ROOT / "estimate.py"),
        "qiskit": importlib.metadata.version("qiskit"),
        "qiskit_ibm_runtime": importlib.metadata.version("qiskit-ibm-runtime"),
        "bqskit": importlib.metadata.version("bqskit"),
        "bqskit_import_path": str(Path(__import__("bqskit").__file__).resolve()),
        "numpy": importlib.metadata.version("numpy"),
        "ridge_implementation": "numpy_closed_form_equivalent_to_standardscaler_ridge_fit_intercept",
        "registry_sha256": sha_file(REGISTRY),
        "parent_protocol_sha256": sha_file(PARENT), "completion_protocol_sha256": sha_file(COMPLETION),
        "authorization_sha256": sha_file(AUTH), "snapshot_assets": {
            b: {"configuration_sha256": registry[b]["configuration_sha256"],
                "properties_sha256": registry[b]["properties_sha256"], "snapshot_id": registry[b]["snapshot_id"]}
            for b in BACKENDS}, "fixtures": parity,
        "checks": ["pinned QCRE estimator source", "serial/parallel unitary path parity",
                   "placement-specific ECR duration", "seconds/dt conversion", "zero-duration virtual RZ",
                   "snapshot configuration/property byte hashes"]}
    target = OUTPUT / "adapter_preflight.json"
    if target.exists() and json.loads(target.read_text()) != result:
        previous = json.loads(target.read_text())
        stable = (previous.get("source_qcre_revision") == result["source_qcre_revision"]
            and previous.get("qcre_estimate_sha256") == result["qcre_estimate_sha256"]
            and previous.get("fixtures") == result["fixtures"]
            and previous.get("snapshot_assets") == result["snapshot_assets"]
            and previous.get("registry_sha256") == result["registry_sha256"])
        if not stable:
            raise RuntimeError("analytical_preflight_existing_mismatch_not_environment_only")
        result["supersedes_preflight_sha256"] = sha_file(target)
        result["previous_qiskit_version"] = previous.get("qiskit")
    if not target.exists(): atomic_json(target, result)
    return result


def load_qv_means(width_by_id, backend_by_id):
    path = SCHOLTEN / "qv_references.csv"
    manifest = SCHOLTEN / "run_manifest.json"
    if not path.is_file() or not manifest.is_file():
        raise RuntimeError("scholten_qv_reference_cache_incomplete")
    m = json.loads(manifest.read_text())
    rows = read_csv(path)
    planned_widths = sorted(set(width_by_id.values()))
    expected = {(w, b, s) for w in planned_widths for b in BACKENDS for s in range(20)}
    got = {(int(r["logical_width"]), r["backend"], int(r["reference_seed"])) for r in rows}
    if m.get("status") != "complete" or got != expected or any(r["status"] != "PASS" for r in rows):
        raise RuntimeError("scholten_qv_reference_cache_not_complete_pass")
    grouped = {}
    for row in rows:
        grouped.setdefault((int(row["logical_width"]), row["backend"]), []).append(int(row["compiled_depth_unitary"]))
    means = {key: float(np.mean(values)) for key, values in grouped.items()}
    if any(not math.isfinite(v) or v <= 0 for v in means.values()):
        raise RuntimeError("scholten_qv_reference_mean_invalid")
    return means, sha_file(path), sha_file(manifest)


def load_analytical_context():
    """Load one read-only, pin-checked context for a serial or worker run."""
    estimate_runtime, converter, archived = load_sources()
    targets, duration_maps, _registry = load_targets()
    cohort, reps, outer, inner = source_and_representation()
    width_by_id = {r["canonical_observation_id"]: int(float(r["g44"])) for r in cohort}
    backend_by_id = {r["canonical_observation_id"]: r["backend"] for r in cohort}
    qv_means, qv_hash, qv_manifest_hash = load_qv_means(width_by_id, backend_by_id)
    entries = json.loads(SCHOLTEN_PROTOCOL.read_text())["entries"]
    for backend in BACKENDS:
        if (float(entries[backend]["value"]) != 5000
                or entries[backend]["definition"] != "reported_CLOPS_definition_unspecified"):
            raise RuntimeError(f"scholten_nominal_throughput_protocol_drift:{backend}")
    return {"estimate_runtime": estimate_runtime, "converter": converter, "archived": archived,
        "targets": targets, "duration_maps": duration_maps, "cohort": cohort, "reps": reps,
        "outer": outer, "inner": inner, "width_by_id": width_by_id,
        "backend_by_id": backend_by_id, "qv_means": qv_means, "qv_hash": qv_hash,
        "qv_manifest_hash": qv_manifest_hash, "entries": entries,
        "source_root": ROOT.parent / "Quantum-Execution-Time-Prediction",
        "cohort_by_id": {r["canonical_observation_id"]: r for r in cohort}}


def initialize_analytical_worker():
    """Initialize an isolated worker; no worker writes benchmark artifacts."""
    global _ANALYTICAL_WORKER_STATE
    _ANALYTICAL_WORKER_STATE = load_analytical_context()


def compute_analytical_observation(identity: str):
    """Compute the nine unchanged analytical outputs for one frozen row."""
    state = _ANALYTICAL_WORKER_STATE
    if state is None:
        raise RuntimeError("analytical_worker_not_initialized")
    base = state["cohort_by_id"][identity]
    representation = state["reps"][identity]
    if (representation["source_qasm_sha256"] != base["logical_qasm_sha256"]
            or representation["backend_canonical"] != base["backend"]
            or representation["target_snapshot_id"] != base["snapshot_id"]):
        raise RuntimeError(f"analytical_representation_identity_mismatch:{identity}")
    compiled_path = ROOT / "artifacts/benchmark_v3/real_qpu/compiled_unified" / representation["compiled_qasm3_file"]
    if sha_file(compiled_path) != representation["compiled_qasm3_sha256"]:
        raise RuntimeError(f"analytical_compiled_qasm_hash_mismatch:{identity}")
    source_path = state["source_root"] / base["logical_qasm_source_path"]
    if sha_file(source_path) != base["logical_qasm_sha256"]:
        raise RuntimeError(f"analytical_logical_qasm_hash_mismatch:{identity}")
    qiskit_circuit = qasm3.load(str(compiled_path))
    backend = state["targets"][base["backend"]]
    dmap = state["duration_maps"][base["backend"]]
    attempt_base = {**base, **representation, "outer_fold": state["outer"][identity],
                    "snapshot_id": base["snapshot_id"], "snapshot_tier": representation["snapshot_tier"]}
    depth = ""
    try:
        depth = qiskit_wire_depth(qiskit_circuit)
    except Exception:
        pass
    rows = []
    omitted_measurements = 0
    # Qiskit Target schedule and the QCRE source estimator are separate clocks.
    try:
        qiskit_seconds = float(qiskit_circuit.estimate_duration(target=backend.target, unit="s"))
        if not math.isfinite(qiskit_seconds) or qiskit_seconds < 0:
            raise ValueError("qiskit_duration_nonfinite")
        rows.append(attempt(attempt_base, "strict_qiskit_schedule_raw", "scheduled_single_shot_seconds", qiskit_seconds, depth=depth))
        rows.append(attempt(attempt_base, "strict_qiskit_schedule_shot_scaled", "shot_scaled_scheduled_seconds",
                            qiskit_seconds * int(base["shots"]), depth=depth))
    except Exception as exc:
        reason = f"{type(exc).__name__}:{str(exc)[:220]}"
        rows.append(attempt(attempt_base, "strict_qiskit_schedule_raw", "scheduled_single_shot_seconds", reason=reason, depth=depth))
        rows.append(attempt(attempt_base, "strict_qiskit_schedule_shot_scaled", "shot_scaled_scheduled_seconds", reason=reason, depth=depth))
    try:
        unitary, omitted = qcre_unitary_input(qiskit_circuit)
        omitted_measurements = omitted
        bqskit_circuit = state["converter"].qiskit_to_bqskit(unitary)
        qcre_seconds = float(state["estimate_runtime"](bqskit_circuit, dmap))
        if not math.isfinite(qcre_seconds) or qcre_seconds < 0:
            raise ValueError("qcre_duration_nonfinite")
        rows.append(attempt(attempt_base, "strict_qcre_source_estimator_raw", "scheduled_unitary_critical_path_seconds",
                            qcre_seconds, omitted=omitted, depth=depth))
        rows.append(attempt(attempt_base, "strict_qcre_source_estimator_shot_scaled", "shot_scaled_unitary_seconds",
                            qcre_seconds * int(base["shots"]), omitted=omitted, depth=depth))
    except Exception as exc:
        reason = f"{type(exc).__name__}:{str(exc)[:220]}"
        rows.append(attempt(attempt_base, "strict_qcre_source_estimator_raw", "scheduled_unitary_critical_path_seconds",
                            reason=reason, depth=depth))
        rows.append(attempt(attempt_base, "strict_qcre_source_estimator_shot_scaled", "shot_scaled_unitary_seconds",
                            reason=reason, depth=depth))
    try:
        hyb_value = float(state["archived"].hyb_effective_seconds(qiskit_circuit, backend))
        log_cost = float(hyb_log_cost(qiskit_circuit, backend))
        if not math.isfinite(hyb_value) or hyb_value < 0:
            raise ValueError("hyb_raw_cost_nonfinite")
        rows.append(attempt(attempt_base, "strict_hyb_effective_cost_raw", "one_circuit_effective_cost_seconds",
                            hyb_value, depth=depth, log_cost=log_cost))
        rows.append(attempt(attempt_base, "strict_hyb_effective_cost_shot_scaled", "shot_linear_effective_cost_seconds",
                            hyb_value * int(base["shots"]), depth=depth, log_cost=log_cost))
    except Exception as exc:
        try:
            log_cost = hyb_log_cost(qiskit_circuit, backend)
        except Exception:
            log_cost = ""
        reason = f"{type(exc).__name__}:{str(exc)[:220]}"
        rows.append(attempt(attempt_base, "strict_hyb_effective_cost_raw", "one_circuit_effective_cost_seconds",
                            reason=reason, depth=depth, log_cost=log_cost))
        rows.append(attempt(attempt_base, "strict_hyb_effective_cost_shot_scaled", "shot_linear_effective_cost_seconds",
                            reason=reason, depth=depth, log_cost=log_cost))
    throughput = float(state["entries"][base["backend"]]["value"])
    if depth == "":
        proxy = None
        proxy_reason = "unsupported_instruction_for_unitary_depth"
        qv = None
        qv_reason = proxy_reason
    else:
        proxy = int(base["shots"]) * int(depth) / throughput
        proxy_reason = ""
        mean_depth = state["qv_means"][(state["width_by_id"][identity], base["backend"])]
        effective = int(depth) / mean_depth * state["width_by_id"][identity]
        qv = int(base["shots"]) * effective / throughput
        qv_reason = ""
    rows.append(attempt(attempt_base, "strict_scholten_depth_throughput_proxy", "nominal_clops_depth_proxy_seconds",
                        proxy, reason=proxy_reason, depth=depth))
    rows.append(attempt(attempt_base, "strict_scholten_same_width_qv_extension", "same_width_qv_normalized_throughput_seconds",
                        qv, reason=qv_reason, depth=depth))
    rows.append(attempt(attempt_base, "strict_scholten_original_paper_eligible", "scholten_original_paper_formula_seconds",
                        reason="paper_definition_not_applicable_missing_verified_kernel_n_and_template_repetitions",
                        depth=depth))
    if len(rows) != 9:
        raise RuntimeError(f"analytical_method_attempt_count_mismatch:{identity}:{len(rows)}")
    return rows, omitted_measurements


def run_raw(max_circuits=None, workers=1):
    if workers > 1:
        return run_raw_parallel(max_circuits, workers)
    preflight_result = preflight()
    estimate_runtime, converter, archived = load_sources()
    targets, duration_maps, registry = load_targets()
    cohort, reps, outer, inner = source_and_representation()
    width_by_id = {r["canonical_observation_id"]: int(float(r["g44"])) for r in cohort}
    backend_by_id = {r["canonical_observation_id"]: r["backend"] for r in cohort}
    qv_means, qv_hash, qv_manifest_hash = load_qv_means(width_by_id, backend_by_id)
    scholten_contract = json.loads(SCHOLTEN_PROTOCOL.read_text())
    entries = scholten_contract["entries"]
    for backend in BACKENDS:
        if float(entries[backend]["value"]) != 5000 or entries[backend]["definition"] != "reported_CLOPS_definition_unspecified":
            raise RuntimeError(f"scholten_nominal_throughput_protocol_drift:{backend}")
    output = OUTPUT / "raw_attempts.csv"
    prior = read_csv(output) if output.exists() else []
    by_id: dict[str, list[dict[str, str]]] = {}
    for row in prior:
        by_id.setdefault(row["canonical_observation_id"], []).append(row)
    if any(len(rows) != 9 or len({r["method_id"] for r in rows}) != 9 for rows in by_id.values()):
        raise RuntimeError("analytical_resume_partial_row_identity_mismatch")
    source_root = ROOT.parent / "Quantum-Execution-Time-Prediction"
    measurements_omitted = 0
    if max_circuits is not None and max_circuits < 1:
        raise ValueError("max_circuits_must_be_positive")
    run_order = sorted(cohort, key=lambda r: (int(reps[r["canonical_observation_id"]]["operation_count"]),
                                               r["canonical_observation_id"]))
    selected_order = run_order if max_circuits is None else run_order[:max_circuits]
    started = time.monotonic()
    for index, base in enumerate(selected_order, 1):
        identity = base["canonical_observation_id"]
        if identity in by_id:
            continue
        representation = reps[identity]
        if (representation["source_qasm_sha256"] != base["logical_qasm_sha256"]
                or representation["backend_canonical"] != base["backend"]
                or representation["target_snapshot_id"] != base["snapshot_id"]):
            raise RuntimeError(f"analytical_representation_identity_mismatch:{identity}")
        compiled_path = ROOT / "artifacts/benchmark_v3/real_qpu/compiled_unified" / representation["compiled_qasm3_file"]
        if sha_file(compiled_path) != representation["compiled_qasm3_sha256"]:
            raise RuntimeError(f"analytical_compiled_qasm_hash_mismatch:{identity}")
        source_path = source_root / base["logical_qasm_source_path"]
        if sha_file(source_path) != base["logical_qasm_sha256"]:
            raise RuntimeError(f"analytical_logical_qasm_hash_mismatch:{identity}")
        qiskit_circuit = qasm3.load(str(compiled_path))
        backend = targets[base["backend"]]
        dmap = duration_maps[base["backend"]]
        attempt_base = {**base, **representation, "outer_fold": outer[identity],
                        "snapshot_id": base["snapshot_id"], "snapshot_tier": representation["snapshot_tier"]}
        depth = ""
        try:
            depth = qiskit_wire_depth(qiskit_circuit)
        except Exception:
            pass
        rows = []
        # Qiskit Target schedule and the QCRE source estimator are separate clocks.
        try:
            qiskit_seconds = float(qiskit_circuit.estimate_duration(target=backend.target, unit="s"))
            if not math.isfinite(qiskit_seconds) or qiskit_seconds < 0: raise ValueError("qiskit_duration_nonfinite")
            rows.append(attempt(attempt_base, "strict_qiskit_schedule_raw", "scheduled_single_shot_seconds", qiskit_seconds, depth=depth))
            rows.append(attempt(attempt_base, "strict_qiskit_schedule_shot_scaled", "shot_scaled_scheduled_seconds",
                                qiskit_seconds * int(base["shots"]), depth=depth))
        except Exception as exc:
            reason = f"{type(exc).__name__}:{str(exc)[:220]}"
            rows.append(attempt(attempt_base, "strict_qiskit_schedule_raw", "scheduled_single_shot_seconds", reason=reason, depth=depth))
            rows.append(attempt(attempt_base, "strict_qiskit_schedule_shot_scaled", "shot_scaled_scheduled_seconds", reason=reason, depth=depth))
        try:
            unitary, omitted = qcre_unitary_input(qiskit_circuit)
            measurements_omitted += omitted
            bqskit_circuit = converter.qiskit_to_bqskit(unitary)
            qcre_seconds = float(estimate_runtime(bqskit_circuit, dmap))
            if not math.isfinite(qcre_seconds) or qcre_seconds < 0: raise ValueError("qcre_duration_nonfinite")
            rows.append(attempt(attempt_base, "strict_qcre_source_estimator_raw", "scheduled_unitary_critical_path_seconds",
                                qcre_seconds, omitted=omitted, depth=depth))
            rows.append(attempt(attempt_base, "strict_qcre_source_estimator_shot_scaled", "shot_scaled_unitary_seconds",
                                qcre_seconds * int(base["shots"]), omitted=omitted, depth=depth))
        except Exception as exc:
            reason = f"{type(exc).__name__}:{str(exc)[:220]}"
            rows.append(attempt(attempt_base, "strict_qcre_source_estimator_raw", "scheduled_unitary_critical_path_seconds",
                                reason=reason, depth=depth))
            rows.append(attempt(attempt_base, "strict_qcre_source_estimator_shot_scaled", "shot_scaled_unitary_seconds",
                                reason=reason, depth=depth))
        try:
            hyb_value = float(archived.hyb_effective_seconds(qiskit_circuit, backend))
            log_cost = float(hyb_log_cost(qiskit_circuit, backend))
            if not math.isfinite(hyb_value) or hyb_value < 0: raise ValueError("hyb_raw_cost_nonfinite")
            rows.append(attempt(attempt_base, "strict_hyb_effective_cost_raw", "one_circuit_effective_cost_seconds",
                                hyb_value, depth=depth, log_cost=log_cost))
            rows.append(attempt(attempt_base, "strict_hyb_effective_cost_shot_scaled", "shot_linear_effective_cost_seconds",
                                hyb_value * int(base["shots"]), depth=depth, log_cost=log_cost))
        except Exception as exc:
            try: log_cost = hyb_log_cost(qiskit_circuit, backend)
            except Exception: log_cost = ""
            reason = f"{type(exc).__name__}:{str(exc)[:220]}"
            rows.append(attempt(attempt_base, "strict_hyb_effective_cost_raw", "one_circuit_effective_cost_seconds",
                                reason=reason, depth=depth, log_cost=log_cost))
            rows.append(attempt(attempt_base, "strict_hyb_effective_cost_shot_scaled", "shot_linear_effective_cost_seconds",
                                reason=reason, depth=depth, log_cost=log_cost))
        throughput = float(entries[base["backend"]]["value"])
        if depth == "":
            proxy = None; proxy_reason = "unsupported_instruction_for_unitary_depth"
            qv = None; qv_reason = proxy_reason
        else:
            proxy = int(base["shots"]) * int(depth) / throughput; proxy_reason = ""
            mean_depth = qv_means[(width_by_id[identity], base["backend"])]
            effective = int(depth) / mean_depth * width_by_id[identity]
            qv = int(base["shots"]) * effective / throughput; qv_reason = ""
        rows.append(attempt(attempt_base, "strict_scholten_depth_throughput_proxy", "nominal_clops_depth_proxy_seconds",
                            proxy, reason=proxy_reason, depth=depth))
        rows.append(attempt(attempt_base, "strict_scholten_same_width_qv_extension", "same_width_qv_normalized_throughput_seconds",
                            qv, reason=qv_reason, depth=depth))
        rows.append(attempt(attempt_base, "strict_scholten_original_paper_eligible", "scholten_original_paper_formula_seconds",
                            reason="paper_definition_not_applicable_missing_verified_kernel_n_and_template_repetitions",
                            depth=depth))
        if len(rows) != 9:
            raise RuntimeError(f"analytical_method_attempt_count_mismatch:{identity}:{len(rows)}")
        by_id[identity] = rows
        if index % 5 == 0 or index == len(cohort):
            current = sorted((r for values in by_id.values() for r in values),
                             key=lambda r: (r["canonical_observation_id"], r["method_id"]))
            atomic_csv(output, current)
            print(json.dumps({"completed_circuits": len(by_id), "assigned": len(cohort),
                              "elapsed_seconds": round(time.monotonic() - started, 2),
                              "last_id": identity}), flush=True)
    raw_rows = sorted((r for values in by_id.values() for r in values),
                      key=lambda r: (r["canonical_observation_id"], r["method_id"]))
    if len(by_id) != len(cohort):
        checkpoint = {"artifact_id": "strict-logical-analytical-raw-checkpoint", "status": "partial",
            "completed_observations": len(by_id), "assigned_observations": len(cohort),
            "raw_attempt_rows": len(raw_rows), "execution_order": "ascending_compiled_operation_count_then_id",
            "qcre_revision": QCRE_REVISION, "qcre_estimate_sha256": QCRE_FILE_SHA256,
            "qv_references_sha256": qv_hash, "qv_run_manifest_sha256": qv_manifest_hash,
            "raw_attempts_sha256": sha_file(output), "runner_sha256": sha_file(Path(__file__))}
        atomic_json(OUTPUT / "raw_checkpoint.json", checkpoint)
        return checkpoint
    calibrations, calibration_manifest = calibrate(raw_rows, cohort, outer, inner)
    atomic_csv(OUTPUT / "calibrated_attempts.csv", calibrations)
    atomic_json(OUTPUT / "calibration_selection.json", calibration_manifest)
    manifest = {"artifact_id": "strict-logical-analytical-oof", "status": "complete",
        "assigned_observations": 340, "raw_attempt_rows": len(raw_rows), "calibrated_attempt_rows": len(calibrations),
        "evaluation_target_clock": TARGET_CLOCK, "qcre_revision": QCRE_REVISION,
        "qcre_estimate_sha256": QCRE_FILE_SHA256, "adapter_preflight_sha256": sha_file(OUTPUT / "adapter_preflight.json"),
        "scholten_qv_references_sha256": qv_hash, "scholten_qv_run_manifest_sha256": qv_manifest_hash,
        "qiskit": importlib.metadata.version("qiskit"), "qiskit_ibm_runtime": importlib.metadata.version("qiskit-ibm-runtime"),
        "bqskit": importlib.metadata.version("bqskit"),
        "bqskit_import_path": str(Path(__import__("bqskit").__file__).resolve()),
        "ridge_implementation": "numpy_closed_form_equivalent_to_standardscaler_ridge_fit_intercept",
        "logical_features_sha256": sha_file(PANEL / "logical_features.csv"),
        "compiled_representation_sha256": sha_file(REPRESENTATION), "outer_splits_sha256": sha_file(PANEL / "outer_splits.csv"),
        "inner_splits_sha256": sha_file(PANEL / "inner_splits.csv"),
        "omitted_terminal_measurements_for_qcre": measurements_omitted,
        "outputs": {name: sha_file(OUTPUT / name) for name in
                    ("raw_attempts.csv", "calibrated_attempts.csv", "calibration_selection.json", "adapter_preflight.json")}}
    atomic_json(OUTPUT / "run_manifest.json", manifest)
    return manifest


def run_raw_parallel(max_circuits=None, workers=4):
    """Resume-safe per-observation calculation on a bounded spawn pool."""
    if workers < 2 or workers > 4:
        raise ValueError("analytical_workers_must_be_between_2_and_4")
    preflight()
    state = load_analytical_context()
    cohort, reps, outer, inner = state["cohort"], state["reps"], state["outer"], state["inner"]
    qv_hash, qv_manifest_hash = state["qv_hash"], state["qv_manifest_hash"]
    output = OUTPUT / "raw_attempts.csv"
    prior = read_csv(output) if output.exists() else []
    by_id: dict[str, list[dict[str, str]]] = {}
    for row in prior:
        by_id.setdefault(row["canonical_observation_id"], []).append(row)
    if any(len(rows) != 9 or len({r["method_id"] for r in rows}) != 9 for rows in by_id.values()):
        raise RuntimeError("analytical_resume_partial_row_identity_mismatch")
    if max_circuits is not None and max_circuits < 1:
        raise ValueError("max_circuits_must_be_positive")
    run_order = sorted(cohort, key=lambda r: (int(reps[r["canonical_observation_id"]]["operation_count"]),
                                               r["canonical_observation_id"]))
    selected_order = run_order if max_circuits is None else run_order[:max_circuits]
    todo = [row["canonical_observation_id"] for row in selected_order
            if row["canonical_observation_id"] not in by_id]
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_analytical_worker) as executor:
        future_ids = {executor.submit(compute_analytical_observation, identity): identity for identity in todo}
        for completed, future in enumerate(concurrent.futures.as_completed(future_ids), 1):
            identity = future_ids[future]
            rows, _omitted = future.result()
            if identity in by_id or len(rows) != 9:
                raise RuntimeError(f"analytical_duplicate_or_incomplete_observation:{identity}")
            by_id[identity] = rows
            if completed % 5 == 0 or completed == len(todo):
                current = sorted((row for values in by_id.values() for row in values),
                                 key=lambda row: (row["canonical_observation_id"], row["method_id"]))
                atomic_csv(output, current)
                print(json.dumps({"completed_circuits": len(by_id), "new_circuits": completed,
                    "assigned": len(cohort), "workers": workers, "execution_engine": "spawn_process_pool",
                    "elapsed_seconds": round(time.monotonic() - started, 2), "last_id": identity}), flush=True)
    raw_rows = sorted((row for values in by_id.values() for row in values),
                      key=lambda row: (row["canonical_observation_id"], row["method_id"]))
    if len(by_id) != len(cohort):
        checkpoint = {"artifact_id": "strict-logical-analytical-raw-checkpoint", "status": "partial",
            "completed_observations": len(by_id), "assigned_observations": len(cohort),
            "raw_attempt_rows": len(raw_rows), "execution_order": "ascending_compiled_operation_count_then_id",
            "qcre_revision": QCRE_REVISION, "qcre_estimate_sha256": QCRE_FILE_SHA256,
            "qv_references_sha256": qv_hash, "qv_run_manifest_sha256": qv_manifest_hash,
            "raw_attempts_sha256": sha_file(output), "runner_sha256": sha_file(Path(__file__)),
            "worker_count": workers, "execution_engine": "spawn_process_pool"}
        atomic_json(OUTPUT / "raw_checkpoint.json", checkpoint)
        return checkpoint
    calibrations, calibration_manifest = calibrate(raw_rows, cohort, outer, inner)
    atomic_csv(OUTPUT / "calibrated_attempts.csv", calibrations)
    atomic_json(OUTPUT / "calibration_selection.json", calibration_manifest)
    manifest = {"artifact_id": "strict-logical-analytical-oof", "status": "complete",
        "assigned_observations": 340, "raw_attempt_rows": len(raw_rows), "calibrated_attempt_rows": len(calibrations),
        "evaluation_target_clock": TARGET_CLOCK, "qcre_revision": QCRE_REVISION,
        "qcre_estimate_sha256": QCRE_FILE_SHA256, "adapter_preflight_sha256": sha_file(OUTPUT / "adapter_preflight.json"),
        "scholten_qv_references_sha256": qv_hash, "scholten_qv_run_manifest_sha256": qv_manifest_hash,
        "qiskit": importlib.metadata.version("qiskit"), "qiskit_ibm_runtime": importlib.metadata.version("qiskit-ibm-runtime"),
        "bqskit": importlib.metadata.version("bqskit"),
        "bqskit_import_path": str(Path(__import__("bqskit").__file__).resolve()),
        "ridge_implementation": "numpy_closed_form_equivalent_to_standardscaler_ridge_fit_intercept",
        "logical_features_sha256": sha_file(PANEL / "logical_features.csv"),
        "compiled_representation_sha256": sha_file(REPRESENTATION), "outer_splits_sha256": sha_file(PANEL / "outer_splits.csv"),
        "inner_splits_sha256": sha_file(PANEL / "inner_splits.csv"),
        "omitted_terminal_measurements_for_qcre": sum(
            int(row["omitted_measurement_count"] or 0) for row in raw_rows
            if row["method_id"] == "strict_qcre_source_estimator_raw"),
        "worker_count": workers, "execution_engine": "spawn_process_pool",
        "outputs": {name: sha_file(OUTPUT / name) for name in
                    ("raw_attempts.csv", "calibrated_attempts.csv", "calibration_selection.json", "adapter_preflight.json")}}
    atomic_json(OUTPUT / "run_manifest.json", manifest)
    return manifest


def calibrate(raw_rows, cohort, outer, inner):
    y = {r["canonical_observation_id"]: float(r["target_seconds"]) for r in cohort}
    by_method_id = {(r["method_id"], r["canonical_observation_id"]): r for r in raw_rows}
    output_rows, records = [], {}
    for method in CALIBRATABLE_METHODS:
        for fold in range(5):
            fit_ids = [i for i, f in outer.items() if f != fold and (method, i) in by_method_id
                       and by_method_id[(method, i)]["status"] == "predicted"]
            test_ids = sorted(i for i, f in outer.items() if f == fold)
            for transform in ("affine", "log_affine"):
                method_id = f"{method}_{transform}"
                eligible = [i for i in fit_ids if math.isfinite(float(by_method_id[(method, i)]["prediction_seconds"]))
                            and (transform == "affine" or float(by_method_id[(method, i)]["prediction_seconds"]) >= 0)]
                if len(eligible) < 2:
                    a = b = None
                else:
                    x = np.asarray([float(by_method_id[(method, i)]["prediction_seconds"]) for i in eligible])
                    yy = np.asarray([y[i] for i in eligible])
                    if transform == "log_affine": x = np.log1p(x); yy = np.log1p(yy)
                    a, b = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), yy, rcond=None)[0]
                for identity in test_ids:
                    source = by_method_id.get((method, identity))
                    if source is None or source["status"] != "predicted":
                        if source is None:
                            source = {"canonical_observation_id": identity, "outer_fold": fold,
                                "backend": next(r["backend"] for r in cohort if r["canonical_observation_id"] == identity),
                                "evaluation_target_clock": TARGET_CLOCK, "target_seconds": y[identity],
                                "shots": next(r["shots"] for r in cohort if r["canonical_observation_id"] == identity)}
                        output_rows.append({**source, "method_id": method_id,
                            "method_output_clock": "calibrated_archived_mean_result_time_taken_seconds",
                            "prediction_seconds": "", "status": "unavailable",
                            "terminal_reason": "raw_method_unavailable" if source.get("status") != "unavailable" else source.get("terminal_reason", "raw_method_unavailable")})
                    elif a is None:
                        output_rows.append({**source, "method_id": method_id,
                            "method_output_clock": "calibrated_archived_mean_result_time_taken_seconds",
                            "prediction_seconds": "", "status": "unavailable", "terminal_reason": "insufficient_outer_train_calibration_rows"})
                    else:
                        xval = float(source["prediction_seconds"])
                        if transform == "log_affine": xval = math.log1p(max(xval, 0.0))
                        pred = float(a + b*xval)
                        if transform == "log_affine": pred = float(np.expm1(pred))
                        if not math.isfinite(pred):
                            output_rows.append({**source, "method_id": method_id,
                                "method_output_clock": "calibrated_archived_mean_result_time_taken_seconds",
                                "prediction_seconds": "", "status": "unavailable", "terminal_reason": "nonfinite_outer_train_calibration"})
                        else:
                            output_rows.append({**source, "method_id": method_id,
                                "method_output_clock": "calibrated_archived_mean_result_time_taken_seconds",
                                "prediction_seconds": format(max(0.0, pred), ".17g"), "status": "predicted", "terminal_reason": ""})
                records[f"{method}|fold{fold}|{transform}"] = {"fit_rows": len(eligible),
                    "fit_ids_sha256": sha_bytes("\n".join(sorted(eligible)).encode()),
                    "intercept": None if a is None else float(a), "slope": None if b is None else float(b)}
    # Explicit Hyb-HANAS log-cost Ridge variant, with every scaler/fit inside
    # the predeclared inner/outer train IDs.
    for fold in range(5):
        test_ids = sorted(i for i, f in outer.items() if f == fold)
        fold_train = {i for i, f in outer.items() if f != fold}
        inner_ids = {i: inner[(i, fold)] for i in fold_train if (i, fold) in inner}
        raw_hyb = {(r["canonical_observation_id"]): r for r in raw_rows if r["method_id"] == "strict_hyb_effective_cost_raw"}
        feature_ids = {i for i in fold_train if i in raw_hyb
                       and raw_hyb[i]["auxiliary_log_cost"] != ""
                       and math.isfinite(float(raw_hyb[i]["auxiliary_log_cost"]))}
        alpha_scores = []
        for alpha in (0.1, 1.0, 10.0, 100.0):
            fold_maes = []
            for inner_fold in range(4):
                valid = sorted(i for i in feature_ids if inner_ids.get(i) == inner_fold)
                fit = sorted(i for i in feature_ids if inner_ids.get(i) != inner_fold)
                if len(valid) < 2 or len(fit) < 2:
                    continue
                xfit = np.asarray([[np.arcsinh(float(raw_hyb[i]["auxiliary_log_cost"])), np.log1p(int(raw_hyb[i]["shots"]))] for i in fit])
                xval = np.asarray([[np.arcsinh(float(raw_hyb[i]["auxiliary_log_cost"])), np.log1p(int(raw_hyb[i]["shots"]))] for i in valid])
                pred_log = standardized_ridge_fit_predict(xfit, np.log1p([y[i] for i in fit]), xval, alpha)
                pred = np.maximum(np.expm1(pred_log), 0.0)
                fold_maes.append(float(np.mean(np.abs(np.asarray([y[i] for i in valid]) - pred))))
            alpha_scores.append({"alpha": alpha, "inner_fold_mae_seconds": fold_maes,
                                 "mean_inner_mae_seconds": float(np.mean(fold_maes)) if len(fold_maes) == 4 else None,
                                 "valid_inner_folds": len(fold_maes)})
        eligible_alpha = [r for r in alpha_scores if r["valid_inner_folds"] == 4]
        method_id = "strict_hyb_log_cost_ridge"
        if not eligible_alpha:
            chosen = None
        else:
            chosen = min(eligible_alpha, key=lambda r: (r["mean_inner_mae_seconds"], r["alpha"]))["alpha"]
        train_ids = sorted(feature_ids)
        if chosen is not None and len(train_ids) >= 2:
            xtrain = np.asarray([[np.arcsinh(float(raw_hyb[i]["auxiliary_log_cost"])), np.log1p(int(raw_hyb[i]["shots"]))] for i in train_ids])
            eligible_tests = [i for i in test_ids if i in raw_hyb
                              and raw_hyb[i]["auxiliary_log_cost"] != ""
                              and math.isfinite(float(raw_hyb[i]["auxiliary_log_cost"]))]
            xval = np.asarray([[np.arcsinh(float(raw_hyb[i]["auxiliary_log_cost"])), np.log1p(int(raw_hyb[i]["shots"]))] for i in eligible_tests])
            pred_log = standardized_ridge_fit_predict(xtrain, np.log1p([y[i] for i in train_ids]), xval, chosen)
            predictions = {i: max(0.0, float(np.expm1(p))) for i, p in zip(eligible_tests, pred_log)}
        else:
            predictions = {}
        for identity in test_ids:
            source = raw_hyb.get(identity)
            if identity in predictions:
                output_rows.append({**source, "method_id": method_id,
                    "method_output_clock": "calibrated_archived_mean_result_time_taken_seconds",
                    "prediction_seconds": format(predictions[identity], ".17g"), "status": "predicted", "terminal_reason": ""})
            else:
                reason = "hyb_log_cost_or_model_unavailable" if chosen is not None else "no_complete_inner_alpha_selection"
                output_rows.append({**(source or {}), "canonical_observation_id": identity,
                    "outer_fold": fold, "backend": next(r["backend"] for r in cohort if r["canonical_observation_id"] == identity),
                    "evaluation_target_clock": TARGET_CLOCK, "method_id": method_id,
                    "method_output_clock": "calibrated_archived_mean_result_time_taken_seconds",
                    "target_seconds": y[identity], "prediction_seconds": "", "status": "unavailable", "terminal_reason": reason})
        records[f"{method_id}|fold{fold}"] = {"chosen_alpha": chosen, "alpha_scores": alpha_scores,
            "fit_rows": len(train_ids), "fit_ids_sha256": sha_bytes("\n".join(train_ids).encode())}
    return sorted(output_rows, key=lambda r: (r.get("canonical_observation_id", ""), r["method_id"], r.get("outer_fold", -1))), {
        "fit_scope": "outer-train only; alpha chosen on four frozen grouped inner folds",
        "calibration_floor_seconds": 0, "ridge_features": ["asinh(log_cost)", "log1p(shots)"],
        "ridge_target": "log1p(target_seconds)", "ridge_alpha_candidates": [0.1, 1, 10, 100], "fold_records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "run"), required=True)
    parser.add_argument("--max-circuits", type=int, default=None,
                        help="Resume-safe raw adapter canary; calibration waits until all 340 rows exist.")
    parser.add_argument("--workers", type=int, default=1,
                        help="CPU process workers (1–4); outputs are checkpointed by the parent process.")
    args = parser.parse_args()
    result = preflight() if args.stage == "preflight" else run_raw(args.max_circuits, args.workers)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
