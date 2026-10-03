#!/usr/bin/env python3
"""Process-isolated QCSim CPU component measurements (implementation only).

This worker consumes the immutable plan written by prepare_maestro_cpu_component.py.
It intentionally has no Maestro/QASM measurement side effects at import time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    # Make the documented `python benchmark_v1/scripts/...` invocation work
    # from the repository root; script execution otherwise exposes only this
    # file's directory, breaking the package import used by plan validation.
    sys.path.insert(0, str(ROOT))
PROTOCOL_PATH = ROOT / "benchmark_v1/protocol/maestro_cpu_component_benchmark.json"
SEED_REGISTRY_PATH = ROOT / "benchmark_v1/registry/seed_registry.json"
PREPARER_PATH = ROOT / "benchmark_v1/scripts/prepare_maestro_cpu_component.py"
S85_HELPER = ROOT / "benchmark_v1/scripts/run_mps_fixed_chi16_runtime_adaptation_v1.py"
THREAD_VARS = ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
               "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
ATTEMPT_COLUMNS = [
    "phase", "record_type", "cell_id", "qasm_sha256", "normalized_qasm_sha256",
    "normalized_width", "normalized_1q_gate_count", "normalized_cx_gate_count",
    "measurement_map", "measurement_map_sha256", "measured_qubit_count",
    "measured_classical_bit_count", "measured_classical_extent",
    "declared_classical_width", "measurement_terminal", "measurement_map_injective",
    "outer_fold", "thread_arm", "session", "repetition", "attempt_ordinal", "seed_uint32", "status",
    "engine_reported_seconds", "host_wall_seconds", "outer_wall_seconds",
    "simulator_id", "method_id", "optimizer_enabled_requested", "optimizer_enabled_resolved",
    "protocol_sha256", "plan_sha256", "worker_sha256",
    "environment_sha256", "error",
]
TERMINAL_STATUSES = {"ok", "unsupported", "timeout", "resource_limit", "worker_error",
                     "invalid_engine_identity", "nonfinite_runtime"}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical_sha(value: Any) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False).encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def registry_seed(qasm_hash: str, context_id: str, repetition: int,
                  registry: dict[str, Any] | None = None) -> int:
    registry = registry or read_json(SEED_REGISTRY_PATH)
    material = (f"{registry['root_seed']}|{registry['streams']['simulator']}|"
                f"{qasm_hash}|{context_id}|{repetition}").encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big", signed=False)


def order_seed(protocol: dict[str, Any], registry: dict[str, Any]) -> int:
    material = (f"{registry['root_seed']}|{registry['streams']['execution_order']}|"
                f"{protocol['protocol_id']}|pilot|0").encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big", signed=False)


def validate_calibration_cell_hashes(calibration: list[dict[str, Any]]) -> set[str]:
    cal_ids, cal_hashes = set(), set()
    calibration_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration:
        if not isinstance(row.get("qasm"), str) or sha_bytes(row["qasm"].encode()) != row.get("qasm_sha256"):
            raise ValueError(f"calibration QASM/hash mismatch: {row.get('cell_id')}")
        if row["cell_id"] in cal_ids:
            raise ValueError("duplicate calibration cell ID")
        cal_ids.add(row["cell_id"])
        cal_hashes.add(row["qasm_sha256"])
        calibration_by_hash[row["qasm_sha256"]].append(row)
        if canonical_sha({k: v for k, v in row.items() if k not in {"qasm", "cell_id"}}) != row["cell_id"]:
            raise ValueError(f"calibration cell ID mismatch: {row.get('cell_id')}")
    for rows in calibration_by_hash.values():
        if len(rows) == 1:
            continue
        descriptors = {(r.get("kind"), r.get("width"), r.get("operation"), r.get("repeats"), r.get("qasm"))
                       for r in rows}
        shots = [r.get("shots") for r in rows]
        if (len(descriptors) != 1 or len(set(shots)) != len(shots) or
                rows[0].get("kind") != "sampling" or rows[0].get("operation") != "zero_gate_sample"):
            raise ValueError("repeated calibration QASM is allowed only for distinct zero-state sampling shot knots")
    return cal_hashes


def expected_plan_identity(plan_path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    protocol = read_json(PROTOCOL_PATH)
    plan = read_json(plan_path)
    plan_hash = sha_file(plan_path)
    if plan.get("protocol_id") not in (None, protocol["protocol_id"]):
        raise ValueError("plan protocol_id differs from the selected protocol")
    if plan.get("protocol_sha256") != sha_file(PROTOCOL_PATH):
        raise ValueError("plan protocol_sha256 does not match current frozen protocol")
    if plan.get("preparer_sha256") != sha_file(PREPARER_PATH):
        raise ValueError("plan preparer_sha256 does not match immutable preparer")
    calibration = plan.get("calibration_cells")
    panel = plan.get("panel")
    if not isinstance(calibration, list) or len(calibration) != 112:
        raise ValueError("plan must contain exactly 112 calibration cells")
    if not isinstance(panel, list) or len(panel) != 150:
        raise ValueError("plan must contain exactly 150 unique panel QASM hashes")
    cal_hashes = validate_calibration_cell_hashes(calibration)
    panel_hashes = set()
    for row in panel:
        h = row.get("source_qasm_sha256")
        if not isinstance(h, str) or len(h) != 64 or h in panel_hashes:
            raise ValueError("invalid/duplicate panel exact-QASM hash")
        source = Path(row["source_file"])
        if not source.is_file() or sha_file(source) != h:
            raise ValueError(f"panel source QASM missing or hash mismatch: {source}")
        panel_hashes.add(h)
    manifest_path = ROOT / protocol["panel"]["manifest"]
    fold_path = ROOT / protocol["panel"]["fold_source"]
    if not manifest_path.is_file() or sha_file(manifest_path) != protocol["panel"]["manifest_sha256"]:
        raise ValueError("pinned simulator-panel manifest missing or changed")
    if not fold_path.is_file() or sha_file(fold_path) != protocol["panel"]["fold_source_sha256"]:
        raise ValueError("pinned C44 fold source missing or changed")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        aliases = [r for r in csv.DictReader(handle) if r["stratum"] == "core_q2_q9"]
    with fold_path.open(newline="", encoding="utf-8") as handle:
        fold_rows = [r for r in csv.DictReader(handle) if r["stratum"] == "core_q2_q9"]
    fold_map = {}
    for row in fold_rows:
        old = fold_map.setdefault(row["source_sha256"], int(row["fold"]))
        if old != int(row["fold"]):
            raise ValueError("pinned source hash crosses frozen outer folds")
    alias_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in aliases:
        alias_groups[row["qasm_sha256"]].append(row)
    if len(aliases) != 162 or len(alias_groups) != 150 or set(alias_groups) != panel_hashes or set(fold_map) != panel_hashes:
        raise ValueError("pinned panel manifest/fold-source coverage mismatch")
    for row in panel:
        group = alias_groups[row["source_qasm_sha256"]]
        expected_aliases = sorted(x["panel_member_id"] for x in group)
        actual_aliases = sorted(row["aliases"])
        canonical = min(group, key=lambda x: x["basename"])
        from benchmark_v1.scripts.prepare_maestro_cpu_component import measurement_profile
        profile = measurement_profile(Path(row["source_file"]).read_text(encoding="utf-8"))
        if any(row.get(key) != value for key, value in profile.items()):
            raise ValueError(f"panel measurement profile differs from exact source QASM: {h}")
        if (int(row["width"]) != int(canonical["width_qubits"]) or
                int(row["outer_fold"]) != fold_map[row["source_qasm_sha256"]] or
                actual_aliases != expected_aliases or int(row["shots"]) != protocol["context"]["shots"]):
            raise ValueError(f"panel plan metadata/fold differs from pinned source rows: {row['source_qasm_sha256']}")
    if cal_hashes & panel_hashes:
        raise ValueError("synthetic calibration hashes overlap panel source hashes")
    from benchmark_v1.scripts.prepare_maestro_cpu_component import calibration_cells
    frozen_calibration = calibration_cells(protocol)
    if [{k: v for k, v in row.items() if k != "qasm"} for row in calibration] != [
            {k: v for k, v in row.items() if k != "qasm"} for row in frozen_calibration]:
        raise ValueError("calibration descriptors/QASM differ from immutable preparer output")
    if plan.get("assigned_hashes") != 150 or plan.get("calibration_cell_count") != 112:
        raise ValueError("plan assigned denominator/count mismatch")
    pins = plan.get("source_pins", {})
    for key, protocol_key in (("panel_sha256", "manifest_sha256"), ("fold_source_sha256", "fold_source_sha256")):
        expected = protocol["panel"][protocol_key]
        if pins.get(key) != expected:
            raise ValueError(f"plan source pin mismatch: {key}")
    if sha_file(SEED_REGISTRY_PATH) != protocol["context"]["seed_registry_sha256"]:
        raise ValueError("seed registry SHA differs from protocol pin")
    return protocol, plan, plan_hash


def _native_pins(protocol: dict[str, Any]) -> dict[str, Any]:
    env = protocol["local_execution_environment"]
    module = Path(env["native_module_path"])
    boost = Path(env["runtime_link_dependency"]["library_path"])
    if not module.is_file() or sha_file(module) != env["native_module_sha256"]:
        raise ValueError("pinned Maestro native module absent or SHA mismatch")
    if not boost.is_file() or sha_file(boost) != env["runtime_link_dependency"]["library_sha256"]:
        raise ValueError("pinned Boost library absent or SHA mismatch")
    omp = Path(env["openmp_runtime"]["library_path"])
    if not omp.is_file() or sha_file(omp) != env["openmp_runtime"]["library_sha256"]:
        raise ValueError("pinned OpenMP runtime absent or SHA mismatch")
    return {"native_module": str(module), "native_module_sha256": sha_file(module),
            "boost_library": str(boost), "boost_library_sha256": sha_file(boost),
            "openmp_library": str(omp), "openmp_library_sha256": sha_file(omp)}


def cpu_identity() -> dict[str, Any]:
    model = platform.processor() or None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware")) and ":" in line:
                model = line.split(":", 1)[1].strip() or model
                break
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {"model": model, "logical_cpu_count": os.cpu_count(), "affinity_cpu_ids": affinity,
            "affinity_cpu_count": len(affinity) if affinity is not None else None,
            "total_os_threads_are_not_native_library_thread_limit": True}


def threadpool_fingerprint() -> dict[str, Any]:
    try:
        import importlib.metadata as metadata
        import threadpoolctl
    except Exception as exc:
        return {"status": "unavailable", "reason": f"threadpoolctl import unavailable: {type(exc).__name__}: {exc}"}
    try:
        entries = threadpoolctl.threadpool_info()
        normalized = [{key: item.get(key) for key in
                       ("user_api", "internal_api", "prefix", "version", "num_threads", "threading_layer", "architecture")}
                      for item in entries]
        return {"status": "available" if normalized else "unavailable_no_loaded_pools",
                "threadpoolctl_version": metadata.version("threadpoolctl"),
                "libraries": normalized}
    except Exception as exc:
        return {"status": "unavailable", "reason": f"threadpool_info failed: {type(exc).__name__}: {exc}"}


def environment_identity(protocol: dict[str, Any], thread_arm: str = "primary_native_threads_1") -> dict[str, Any]:
    native = _native_pins(protocol)
    pinned_env = protocol["local_execution_environment"]
    if platform.python_version() != pinned_env["python"] or str(Path(sys.executable).resolve()) != str(Path(pinned_env["python_executable"]).resolve()):
        raise ValueError(f"interpreter mismatch: Python={platform.python_version()} executable={sys.executable}")
    try:
        import importlib.metadata as metadata
        packages = {p: metadata.version(p) for p in ("qiskit", "numpy", "scipy", "qoro-maestro")}
    except Exception as exc:
        raise ValueError(f"cannot inspect pinned Python package versions: {exc}") from exc
    if packages != {"qiskit": "2.5.2", "numpy": "2.2.6", "scipy": "1.15.3", "qoro-maestro": "0.3.1"}:
        raise ValueError(f"Python package version mismatch: {packages}")
    threads = ({v: "1" for v in THREAD_VARS} if thread_arm == "primary_native_threads_1"
               else {v: None for v in THREAD_VARS})
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu": cpu_identity(), "threadpool_fingerprint": threadpool_fingerprint(),
            "native_library_thread_policy": {"thread_arm": thread_arm,
                                               "requested_threads_per_native_library": 1 if thread_arm == "primary_native_threads_1" else None,
                                               "os_logical_cpu_count": os.cpu_count()},
            "packages": packages, "native": native,
            "thread_arm": thread_arm, "thread_environment": threads,
            "runner_sha256": sha_file(Path(__file__)),
            "seed_registry_sha256": sha_file(SEED_REGISTRY_PATH)}


def worker_context(protocol: dict[str, Any], plan_hash: str, arm: str) -> dict[str, Any]:
    environment = environment_identity(protocol, arm)
    return {"protocol_sha256": sha_file(PROTOCOL_PATH), "plan_sha256": plan_hash,
            "environment": environment, "environment_sha256": canonical_sha(environment),
            "worker_sha256": sha_file(Path(__file__)), "simulator_id": "QCSim",
            "method_id": "Statevector", "precision": "FP64/std::complex<double>",
            "cpu_use_double_precision_flag": False,
            "optimizer_enabled_requested": protocol["context"].get("optimize_circuit")}


def make_cell_specs(plan: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    if phase == "pilot":
        selected = set(plan["pilot_cell_ids"])
        cells = [dict(r, scope="calibration") for r in plan["calibration_cells"] if r["cell_id"] in selected]
        if len(cells) != 10:
            raise ValueError("pilot cell selection must contain ten calibration cells")
        return cells
    if phase == "full":
        calibration = [dict(r, scope="calibration") for r in plan["calibration_cells"]]
        panel = [dict(r, cell_id=r["source_qasm_sha256"], qasm_sha256=r["source_qasm_sha256"],
                      qasm=None, kind="panel", outer_fold=r["outer_fold"], width=r["width"],
                      operation="panel_qasm", repeats=None, scope="panel") for r in plan["panel"]]
        return calibration + panel
    raise ValueError(f"unknown phase: {phase}")


def qasm_for(spec: dict[str, Any]) -> str:
    if spec.get("scope") == "calibration":
        return spec["qasm"]
    path = Path(spec["source_file"])
    raw = path.read_text(encoding="utf-8")
    if sha_bytes(raw.encode()) != spec["qasm_sha256"]:
        raise ValueError(f"panel source changed after plan validation: {path}")
    # Normalization is deliberately outside both selected timing clocks.
    from qiskit import qasm2, transpile
    circuit = load_source_qasm(raw, qasm2)
    measured = [(i, inst) for i, inst in enumerate(circuit.data) if inst.operation.name == "measure"]
    if not measured:
        raise UnsupportedCircuit("source QASM has no terminal measurement")
    first_measure = min(i for i, _ in measured)
    if any(inst.operation.name not in {"measure", "barrier"} for inst in circuit.data[first_measure:]):
        raise UnsupportedCircuit("nonterminal measurement in source QASM")
    mapping = [(inst.qubits[0], inst.clbits[0]) for _, inst in measured]
    if (len({q for q, _ in mapping}) != len(mapping) or
            len({c for _, c in mapping}) != len(mapping)):
        raise UnsupportedCircuit("terminal measurement mapping reuses a qubit or classical bit")
    def indexed_mapping(value):
        return [(value.find_bit(inst.qubits[0]).index, value.find_bit(inst.clbits[0]).index)
                for inst in value.data if inst.operation.name == "measure"]
    source_mapping = indexed_mapping(circuit)
    norm = transpile(circuit, basis_gates=["rx", "rz", "cx"], optimization_level=0,
                     seed_transpiler=12345, approximation_degree=1.0)
    if norm.num_qubits != circuit.num_qubits or norm.num_clbits != circuit.num_clbits:
        raise UnsupportedCircuit("transpilation changed circuit width/classical width")
    # Independent terminal measurements commute; transpile may reorder them.
    # Check the actual map first, then restore source order without adding bits.
    if sorted(indexed_mapping(norm)) != sorted(source_mapping):
        raise UnsupportedCircuit("transpilation changed terminal measurement mapping")
    norm.data = [inst for inst in norm.data if inst.operation.name != "measure"]
    for qubit, clbit in source_mapping:
        norm.measure(qubit, clbit)
    if indexed_mapping(norm) != source_mapping:
        raise UnsupportedCircuit("transpilation changed ordered terminal measurement mapping")
    # Verify unitary equivalence on zero and two seeded product states outside timing.
    from qiskit.quantum_info import Statevector
    import numpy as np
    source_u = circuit.remove_final_measurements(inplace=False)
    norm_u = norm.remove_final_measurements(inplace=False)
    rng = np.random.default_rng(12345)
    probes = [np.zeros(circuit.num_qubits), *[rng.uniform(0, 2 * np.pi, circuit.num_qubits) for _ in range(2)]]
    for angles in probes:
        prefix = circuit.copy_empty_like()
        for q, angle in enumerate(angles):
            if angle:
                prefix.rx(float(angle), q)
        v1 = Statevector.from_instruction(prefix.compose(source_u)).data
        v2 = Statevector.from_instruction(prefix.compose(norm_u)).data
        overlap = abs(np.vdot(v1, v2)) ** 2
        if not math.isfinite(float(overlap)) or float(overlap) < 0.9999999999:
            raise UnsupportedCircuit("normalized QASM failed frozen statevector equivalence check")
    return qasm2.dumps(norm)


def normalized_execution_identity(qasm: str) -> dict[str, Any]:
    """Hash and count the exact rx/rz/cx OpenQASM string sent to simple_execute."""
    from qiskit import qasm2
    circuit = qasm2.loads(qasm, strict=True,
                          custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    counts = {name: int(count) for name, count in circuit.count_ops().items()}
    allowed = {"rx", "rz", "cx", "measure", "barrier"}
    unexpected = sorted(set(counts) - allowed)
    if unexpected:
        raise UnsupportedCircuit(f"normalized QASM contains operations outside rx/rz/cx: {unexpected}")
    return {"normalized_qasm_sha256": sha_bytes(qasm.encode()),
            "normalized_width": int(circuit.num_qubits),
            "normalized_1q_gate_count": counts.get("rx", 0) + counts.get("rz", 0),
            "normalized_cx_gate_count": counts.get("cx", 0)}


def load_source_qasm(raw: str, qasm2_module: Any | None = None) -> Any:
    """Parse legacy OpenQASM 2 gate spellings with Qiskit's frozen legacy map."""
    if qasm2_module is None:
        from qiskit import qasm2 as qasm2_module
    return qasm2_module.loads(raw, strict=True,
                              custom_instructions=qasm2_module.LEGACY_CUSTOM_INSTRUCTIONS)


class UnsupportedCircuit(ValueError):
    pass


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Hidden spawned-process entrypoint. Import Maestro only inside the worker."""
    started = time.perf_counter()
    optimizer_requested = payload.get("optimize_circuit")
    optimizer_resolved = None
    try:
        os.environ["LD_LIBRARY_PATH"] = str(Path(payload["boost_library"]).parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        sys.path.insert(0, payload["native_build_dir"])
        import maestro
        if Path(maestro.__file__).resolve() != Path(payload["native_module"]).resolve():
            return {"status": "invalid_engine_identity", "error": "imported native module path differs from pin"}
        config = maestro.SimulatorConfig(
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.Statevector,
            seed=int(payload["seed"]),
            use_double_precision=False,
        )
        if optimizer_requested is not None:
            if not hasattr(config, "optimize_circuit") or not hasattr(maestro, "_optimizer_profile"):
                return {"status": "invalid_engine_identity", "error": "optimizer override binding is unavailable",
                        "optimizer_enabled_requested": bool(optimizer_requested),
                        "optimizer_enabled_resolved": None}
            config.optimize_circuit = bool(optimizer_requested)
            resolved_value = config.optimize_circuit
            if resolved_value is None or bool(resolved_value) != bool(optimizer_requested):
                return {"status": "invalid_engine_identity", "error": "resolved optimizer flag differs from request",
                        "optimizer_enabled_requested": bool(optimizer_requested),
                        "optimizer_enabled_resolved": None if resolved_value is None else bool(resolved_value)}
            optimizer_resolved = bool(resolved_value)
        t0 = time.perf_counter()
        result = maestro.simple_execute(payload["qasm"], config, shots=int(payload["shots"]))
        host_wall = time.perf_counter() - t0
        simulator = result.get("simulator") if isinstance(result, dict) else None
        method = result.get("method") if isinstance(result, dict) else None
        if simulator != maestro.SimulatorType.QCSim.value or method != maestro.SimulationType.Statevector.value:
            return {"status": "invalid_engine_identity", "simulator": simulator, "method": method,
                    "host_wall_seconds": host_wall,
                    "optimizer_enabled_requested": optimizer_requested,
                    "optimizer_enabled_resolved": optimizer_resolved,
                    "error": "result simulator/method IDs do not match explicit config"}
        selected = result.get("time_taken")
        try:
            selected = float(selected)
        except (TypeError, ValueError):
            selected = math.nan
        if not math.isfinite(selected) or selected < 0:
            return {"status": "nonfinite_runtime", "simulator": simulator, "method": method,
                    "host_wall_seconds": host_wall, "engine_reported_seconds": selected,
                    "optimizer_enabled_requested": optimizer_requested,
                    "optimizer_enabled_resolved": optimizer_resolved,
                    "error": f"invalid result.time_taken: {selected}"}
        return {"status": "ok", "simulator": simulator, "method": method,
                "host_wall_seconds": host_wall, "engine_reported_seconds": selected,
                "outer_worker_seconds": time.perf_counter() - started,
                "optimizer_enabled_requested": optimizer_requested,
                "optimizer_enabled_resolved": optimizer_resolved}
    except MemoryError as exc:
        return {"status": "resource_limit", "error": f"{type(exc).__name__}: {exc}",
                "optimizer_enabled_requested": optimizer_requested,
                "optimizer_enabled_resolved": optimizer_resolved}
    except BaseException as exc:
        return {"status": "worker_error", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=4),
                "optimizer_enabled_requested": optimizer_requested,
                "optimizer_enabled_resolved": optimizer_resolved}


def subprocess_attempt(payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    env = dict(os.environ)
    arm = payload["thread_arm"]
    if arm == "primary_native_threads_1":
        env.update({name: "1" for name in THREAD_VARS})
    elif arm == "unset_native_thread_environment":
        for name in THREAD_VARS:
            env.pop(name, None)
    else:
        return {"status": "worker_error", "error": f"unknown thread arm: {arm}"}
    native = payload["native_module"]
    env["PYTHONPATH"] = str(Path(native).parent) + os.pathsep + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = str(Path(payload["boost_library"]).parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    command = [sys.executable, str(Path(__file__).resolve()), "--_worker"]
    start = time.monotonic()
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, env=env)
    try:
        out, err = proc.communicate(json.dumps(payload), timeout=timeout_seconds)
        outer = time.monotonic() - start
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return {"status": "timeout", "outer_wall_seconds": time.monotonic() - start,
                "error": f"worker exceeded {timeout_seconds}s deadline"}
    if proc.returncode != 0:
        status = "resource_limit" if proc.returncode in {-signal.SIGKILL, -signal.SIGXCPU} else "worker_error"
        return {"status": status, "outer_wall_seconds": outer,
                "error": f"worker exit {proc.returncode}; stderr={err[-1000:]}"}
    try:
        result = json.loads(out)
        if result.get("status") not in TERMINAL_STATUSES:
            raise ValueError("worker returned nonterminal/unknown status")
        result["outer_wall_seconds"] = outer
        if result.get("status") == "ok":
            result["engine_reported_seconds"] = float(result["engine_reported_seconds"])
            result["host_wall_seconds"] = float(result["host_wall_seconds"])
        return result
    except Exception as exc:
        return {"status": "worker_error", "outer_wall_seconds": outer,
                "error": f"invalid worker response {type(exc).__name__}: {exc}; stdout={out[-1000:]}; stderr={err[-1000:]}"}


def ensure_scratch(path: Path) -> None:
    p = path.resolve()
    work = (ROOT / "work").resolve()
    if work not in p.parents:
        raise ValueError(f"scratch output must be below ignored work/: {p}")


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def manifest_for(phase: str, plan_hash: str, protocol: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return {"artifact_id": "maestro-cpu-component-v1", "phase": phase,
            "status": "in_progress", "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_id": protocol["protocol_id"], "predictor_method_id": protocol.get("method_id"),
            "execution_context_id": protocol.get("execution_context_id"),
            "protocol_sha256": sha_file(PROTOCOL_PATH), "plan_sha256": plan_hash,
            "runner_sha256": sha_file(Path(__file__)), "seed_registry_sha256": sha_file(SEED_REGISTRY_PATH),
            "worker_context": context, "simulator_id": "QCSim", "method_id": "Statevector",
            "precision": "FP64/std::complex<double>", "attempt_ledger": "attempts.jsonl",
            "terminal_failure_retry": False, "training_performed": False}


def read_ledger(path: Path) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]], dict[str, dict[str, Any]]]:
    records, started, terminal = [], set(), {}
    if not path.exists():
        return records, started, terminal
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            if set(row) != set(ATTEMPT_COLUMNS):
                raise ValueError(f"attempt ledger schema mismatch at line {line_no}")
            records.append(row)
            ident = attempt_key(row)
            if row.get("record_type") == "attempt_started":
                if ident in started:
                    raise ValueError(f"duplicate attempt-start identity in ledger at line {line_no}")
                started.add(ident)
            elif row.get("record_type") == "terminal":
                key = canonical_sha(list(ident))
                if key in terminal:
                    raise ValueError(f"duplicate terminal identity in ledger at line {line_no}")
                terminal[key] = row
    return records, started, terminal


def attempt_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(k) for k in ("phase", "cell_id", "thread_arm", "session", "repetition"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    if set(row) != set(ATTEMPT_COLUMNS):
        raise ValueError("attempt ledger row does not match frozen attempt_columns")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def stable_identity(row: dict[str, Any], context: dict[str, Any], plan_hash: str) -> None:
    if row.get("plan_sha256") != plan_hash or row.get("protocol_sha256") != context["protocol_sha256"]:
        raise ValueError("ledger belongs to another plan/protocol")
    if row.get("worker_sha256") != context["worker_sha256"] or row.get("environment_sha256") != context["environment_sha256"]:
        raise ValueError("ledger worker/environment identity differs; refusing resume")


def call_specs(specs: list[dict[str, Any]], phase: str, arms: list[str], out_dir: Path,
               protocol: dict[str, Any], plan_hash: str, contexts: dict[str, dict[str, Any]],
               pilot_reuse: list[dict[str, Any]] | None = None,
               call=subprocess_attempt) -> None:
    ledger = out_dir / "attempts.jsonl"
    records, started, terminal_map = read_ledger(ledger)
    for row in records:
        if row.get("record_type") in {"attempt_started", "terminal"}:
            arm = row.get("thread_arm")
            if arm not in contexts:
                raise ValueError("existing attempt has unexpected thread arm")
            stable_identity(row, contexts[arm], plan_hash)
    terminal_keys = {attempt_key(r) for r in records if r.get("record_type") == "terminal"}
    started_keys = {attempt_key(r) for r in records if r.get("record_type") == "attempt_started"}
    if terminal_keys - started_keys:
        raise ValueError("terminal row without immutable start marker")
    if started_keys - terminal_keys:
        # A prior controller died after declaring an attempt. Mark it terminal; never rerun it.
        for row in records:
            key = attempt_key(row)
            if row.get("record_type") == "attempt_started" and key not in terminal_keys:
                terminal = dict(row, record_type="terminal", status="worker_error",
                                error="controller_interrupted_after_start; no retry permitted",
                                engine_reported_seconds=None, host_wall_seconds=None, outer_wall_seconds=None)
                append_jsonl(ledger, terminal); terminal_keys.add(key)

    # Resume checks are hash/cell/arm/session/repetition exact; completed terminal failures remain final.
    ordered: list[tuple[dict[str, Any], str, int, int]] = []
    if phase == "pilot":
        registry = read_json(SEED_REGISTRY_PATH)
        rng = random.Random(order_seed(protocol, registry))
        for spec in sorted(specs, key=lambda x: x["cell_id"]):
            for session in range(1, 4):
                for repetition in range(1, 6):
                    pair = list(arms); rng.shuffle(pair)
                    ordered.extend((spec, arm, session, repetition) for arm in pair)
    else:
        ordered = [(spec, arms[0], s, r) for spec in specs for s in range(1, 4) for r in range(1, 6)]

    reuse_by_key = {}
    for row in pilot_reuse or []:
        if row.get("record_type") != "terminal" or row.get("status") != "ok":
            continue
        reuse_by_key[(row["cell_id"], row["session"], row["repetition"], row["qasm_sha256"])] = row
    for ordinal, (spec, arm, session, repetition) in enumerate(ordered, 1):
        cell_id = spec["cell_id"]
        qhash = spec["qasm_sha256"]
        identity = (phase, cell_id, arm, session, repetition)
        if identity in terminal_keys:
            continue
        if identity in started_keys:
            raise ValueError("unclosed started attempt should have been terminalized")
        ctx = contexts[arm]
        seed_context = protocol["context"]["seed_context_id"].format(cell_shots=spec["shots"])
        seed = registry_seed(qhash, seed_context, (session - 1) * 5 + repetition)
        base = {"phase": phase, "record_type": "attempt_started", "cell_id": cell_id,
                "qasm_sha256": qhash,
                "normalized_qasm_sha256": spec.get("normalized_qasm_sha256"),
                "normalized_width": spec.get("normalized_width"),
                "normalized_1q_gate_count": spec.get("normalized_1q_gate_count"),
                "normalized_cx_gate_count": spec.get("normalized_cx_gate_count"),
                "measurement_map": spec.get("measurement_map"),
                "measurement_map_sha256": spec.get("measurement_map_sha256"),
                "measured_qubit_count": spec.get("measured_qubit_count"),
                "measured_classical_bit_count": spec.get("measured_classical_bit_count"),
                "measured_classical_extent": spec.get("measured_classical_extent"),
                "declared_classical_width": spec.get("declared_classical_width"),
                "measurement_terminal": spec.get("measurement_terminal"),
                "measurement_map_injective": spec.get("measurement_map_injective"),
                "outer_fold": spec.get("outer_fold"), "thread_arm": arm,
                "session": session, "repetition": repetition, "attempt_ordinal": ordinal,
                "seed_uint32": seed, "status": "started", "engine_reported_seconds": None,
                "host_wall_seconds": None, "outer_wall_seconds": None, "simulator_id": "QCSim",
                "method_id": "Statevector",
                "optimizer_enabled_requested": protocol["context"].get("optimize_circuit"),
                "optimizer_enabled_resolved": None,
                "protocol_sha256": ctx["protocol_sha256"],
                "plan_sha256": plan_hash, "worker_sha256": ctx["worker_sha256"],
                "environment_sha256": ctx["environment_sha256"], "error": None}
        append_jsonl(ledger, base); started_keys.add(identity)
        reused = (phase == "full" and arm == "primary_native_threads_1" and
                  reuse_by_key.get((cell_id, session, repetition, qhash)))
        if reused:
            # Reference, not a second execution. Exact identity fingerprints are checked by caller.
            terminal = dict(base, record_type="terminal", status="ok",
                            optimizer_enabled_requested=reused.get("optimizer_enabled_requested"),
                            optimizer_enabled_resolved=reused.get("optimizer_enabled_resolved"),
                            engine_reported_seconds=reused.get("engine_reported_seconds"),
                            host_wall_seconds=reused.get("host_wall_seconds"),
                            outer_wall_seconds=reused.get("outer_wall_seconds"),
                            error=f"pilot_reuse_reference:pilot/attempts.jsonl:ordinal={reused.get('attempt_ordinal')}")
        else:
            try:
                if spec.get("_prep_error"):
                    raise UnsupportedCircuit(spec["_prep_error"])
                qasm = spec.get("_prepared_qasm", spec.get("qasm"))
                if not isinstance(qasm, str):
                    raise UnsupportedCircuit("no normalized QASM available in immutable plan")
                payload = {"qasm": qasm, "shots": int(spec["shots"]), "seed": seed,
                           "thread_arm": arm, "native_module": ctx["environment"]["native"]["native_module"],
                           "native_build_dir": str(Path(ctx["environment"]["native"]["native_module"]).parent),
                           "boost_library": ctx["environment"]["native"]["boost_library"],
                           "optimize_circuit": protocol["context"].get("optimize_circuit")}
                result = call(payload, int(protocol["measurement"]["timeout_seconds"]))
            except UnsupportedCircuit as exc:
                result = {"status": "unsupported", "error": str(exc)}
            except BaseException as exc:
                result = {"status": "worker_error", "error": f"preparation:{type(exc).__name__}:{exc}"}
            if result.get("status") not in TERMINAL_STATUSES:
                result = {"status": "worker_error", "error": "runner returned unknown status"}
            terminal = dict(base, record_type="terminal", status=result["status"],
                            engine_reported_seconds=result.get("engine_reported_seconds"),
                            host_wall_seconds=result.get("host_wall_seconds"),
                            outer_wall_seconds=result.get("outer_wall_seconds"),
                            simulator_id=result.get("simulator", "QCSim"),
                            method_id=result.get("method", "Statevector"),
                            optimizer_enabled_requested=result.get(
                                "optimizer_enabled_requested", protocol["context"].get("optimize_circuit")),
                            optimizer_enabled_resolved=result.get("optimizer_enabled_resolved"),
                            error=result.get("error"))
        append_jsonl(ledger, terminal); terminal_keys.add(identity)


def pilot_gate(plan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    terminal = [r for r in rows if r.get("record_type") == "terminal"]
    expected = 300
    primary = [r for r in terminal if r["thread_arm"] == "primary_native_threads_1"]
    per_cell: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    failures = []
    for row in primary:
        if row["status"] != "ok" or not isinstance(row.get("engine_reported_seconds"), (int, float)):
            failures.append({"cell_id": row["cell_id"], "session": row["session"],
                             "repetition": row["repetition"], "status": row["status"]})
        else:
            per_cell[row["cell_id"]][int(row["session"])].append(float(row["engine_reported_seconds"]))
    abs_tol = float(read_json(PROTOCOL_PATH)["pilot"]["stability"]["absolute_seconds"])
    rel_tol = float(read_json(PROTOCOL_PATH)["pilot"]["stability"]["relative"])
    stability = []
    for cell_id in plan["pilot_cell_ids"]:
        sessions = per_cell.get(cell_id, {})
        medians, ok = [], True
        for s in range(1, 4):
            values = sessions.get(s, [])
            if len(values) != 5 or not all(math.isfinite(x) and x >= 0 for x in values):
                ok = False; continue
            med = statistics.median(values); mad = statistics.median(abs(x - med) for x in values)
            medians.append(med)
            if mad > max(abs_tol, rel_tol * med): ok = False
        if len(medians) == 3 and (max(medians) - min(medians) > max(abs_tol, rel_tol * statistics.median(medians))):
            ok = False
        stability.append({"cell_id": cell_id, "sessions_complete": len(medians) == 3, "stable": ok,
                          "session_medians_seconds": medians})
    wanted = {(cell, arm, session, repetition)
              for cell in plan["pilot_cell_ids"]
              for arm in ("unset_native_thread_environment", "primary_native_threads_1")
              for session in (1, 2, 3) for repetition in (1, 2, 3, 4, 5)}
    actual = {(r.get("cell_id"), r.get("thread_arm"), r.get("session"), r.get("repetition"))
              for r in terminal}
    complete = len(terminal) == expected and actual == wanted
    primary_complete = len(primary) == 150 and not failures and all(x["stable"] for x in stability)
    primary_outer = [r["outer_wall_seconds"] for r in primary
                     if isinstance(r.get("outer_wall_seconds"), (int, float)) and math.isfinite(r["outer_wall_seconds"])]
    estimate = statistics.median(primary_outer) if primary_outer else None
    projected_from_scratch = estimate * 3930 if estimate is not None else None
    projected_remaining = estimate * 3780 if estimate is not None else None
    return {"status": "pass" if complete and primary_complete else "fail",
            "technical_complete_300_terminal_rows": complete,
            "primary_arm_acceptance": primary_complete,
            "terminal_rows": len(terminal), "primary_terminal_rows": len(primary),
            "primary_failures": failures, "primary_stability": stability,
            "median_outer_wall_per_call_seconds_diagnostic": estimate,
            "projected_3930_assigned_calls_from_scratch_seconds": projected_from_scratch,
            "projected_3780_new_calls_after_150_pilot_primary_reuse_seconds": projected_remaining,
            "projected_time_assumptions": "linear extrapolation from median primary-arm outer wall per call; informational only, no protocol cutoff",
            "accuracy_gate_applied": False,
            "sampling_fit_gate": "not_evaluable_from_pilot_design; only the 1000-shot n=9 sampling cell is selected"}


def create_or_resume_run(out_dir: Path, phase: str, plan_hash: str, protocol: dict[str, Any], context: dict[str, Any]) -> None:
    manifest_path = out_dir / "run_manifest.json"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        atomic_json(manifest_path, manifest_for(phase, plan_hash, protocol, context))
    else:
        if not manifest_path.is_file():
            raise ValueError("existing output dir has no run_manifest.json; refusing ambiguous resume")
        manifest = read_json(manifest_path)
        if manifest.get("phase") != phase or manifest.get("plan_sha256") != plan_hash:
            raise ValueError("existing run manifest differs from requested phase/plan")
        if manifest.get("protocol_sha256") != context["protocol_sha256"] or manifest.get("worker_context") != context:
            raise ValueError("existing run context differs; refusing resume")


def validate_pilot_report(path: Path, plan_hash: str, primary_context: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = read_json(path)
    if report.get("status") != "pass" or report.get("plan_sha256") != plan_hash:
        raise ValueError("full stage requires a PASS pilot report for the exact plan")
    if report.get("primary_environment_sha256") != primary_context["environment_sha256"] or report.get("runner_sha256") != primary_context["worker_sha256"]:
        raise ValueError("pilot environment/runner does not exactly match full primary context")
    ledger = path.parent / "attempts.jsonl"
    if not ledger.is_file() or report.get("terminal_attempts_sha256") != sha_file(ledger):
        raise ValueError("pilot attempt ledger changed since pilot report was written")
    rows = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
    terminals = [r for r in rows if r.get("record_type") == "terminal"]
    primary = [r for r in terminals if r.get("thread_arm") == "primary_native_threads_1"]
    cells = report.get("pilot_cell_ids", [])
    expected = {(cell, arm, s, r) for cell in cells
                for arm in ("unset_native_thread_environment", "primary_native_threads_1")
                for s in (1, 2, 3) for r in (1, 2, 3, 4, 5)}
    actual = {(r.get("cell_id"), r.get("thread_arm"), r.get("session"), r.get("repetition")) for r in terminals}
    if len(cells) != 10 or len(terminals) != 300 or actual != expected:
        raise ValueError("pilot terminal-attempt identity coverage is incomplete or changed")
    if len(primary) != 150 or any(r.get("status") != "ok" for r in primary):
        raise ValueError("pilot ledger no longer matches its passing report")
    return report, primary


def execute_stage(phase: str, plan_path: Path, out_dir: Path, pilot_report_path: Path | None = None) -> dict[str, Any]:
    ensure_scratch(out_dir)
    protocol, plan, plan_hash = expected_plan_identity(plan_path)
    arms = (["unset_native_thread_environment", "primary_native_threads_1"] if phase == "pilot"
            else ["primary_native_threads_1"])
    contexts = {arm: worker_context(protocol, plan_hash, arm) for arm in arms}
    pilot_rows = None
    if phase == "full":
        if pilot_report_path is None:
            raise ValueError("full requires --pilot-report from a passing pilot10")
        _, pilot_rows = validate_pilot_report(pilot_report_path, plan_hash, contexts["primary_native_threads_1"])
    create_or_resume_run(out_dir, phase, plan_hash, protocol, contexts[arms[-1]])
    specs = make_cell_specs(plan, phase)
    # Exact input validation/normalization is performed before timing lock acquisition.
    if phase == "full":
        for spec in specs:
            if spec["scope"] == "panel":
                try:
                    spec["source_file"] = next(r["source_file"] for r in plan["panel"]
                                                if r["source_qasm_sha256"] == spec["qasm_sha256"])
                    spec["_prepared_qasm"] = qasm_for(spec)
                    spec.update(normalized_execution_identity(spec["_prepared_qasm"]))
                except UnsupportedCircuit as exc:
                    spec["_prep_error"] = str(exc)
                except Exception as exc:
                    spec["_prep_error"] = f"normalization_failed:{type(exc).__name__}:{exc}"
                    spec.update(normalized_qasm_sha256=None, normalized_width=None,
                                normalized_1q_gate_count=None, normalized_cx_gate_count=None)
    for spec in specs:
        if spec["scope"] == "calibration":
            try:
                spec["_prepared_qasm"] = spec["qasm"]
                spec.update(normalized_execution_identity(spec["_prepared_qasm"]))
            except Exception as exc:
                spec["_prep_error"] = f"calibration_normalization_failed:{type(exc).__name__}:{exc}"
                spec.update(normalized_qasm_sha256=None, normalized_width=None,
                            normalized_1q_gate_count=None, normalized_cx_gate_count=None)
    from benchmark_v1.scripts.run_mps_fixed_chi16_runtime_adaptation_v1 import exclusive_compute_lock
    with exclusive_compute_lock():
        call_specs(specs, phase, arms, out_dir, protocol, plan_hash, contexts, pilot_rows)
    records = [json.loads(l) for l in (out_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if phase == "pilot":
        gate = pilot_gate(plan, records)
        report = {"artifact_id": "maestro-cpu-component-pilot-v1", "status": gate["status"],
                  "plan_sha256": plan_hash, "protocol_sha256": sha_file(PROTOCOL_PATH),
                  "pilot_cell_ids": list(plan["pilot_cell_ids"]),
                  "runner_sha256": sha_file(Path(__file__)),
                  "primary_environment_sha256": contexts["primary_native_threads_1"]["environment_sha256"],
                  "terminal_attempts_sha256": sha_file(out_dir / "attempts.jsonl"), "gate": gate}
        atomic_json(out_dir / "pilot_report.json", report)
    manifest = read_json(out_dir / "run_manifest.json")
    manifest.update(status="completed" if (phase != "pilot" or gate["status"] == "pass") else "pilot_gate_failed",
                    completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    attempts_sha256=sha_file(out_dir / "attempts.jsonl"))
    atomic_json(out_dir / "run_manifest.json", manifest)
    return {"phase": phase, "output_dir": str(out_dir), "status": manifest["status"],
            "terminal_attempts": sum(r.get("record_type") == "terminal" for r in records)}


def _session_medians(rows: list[dict[str, Any]], cell_id: str) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("cell_id") == cell_id and row.get("record_type") == "terminal" and row.get("status") == "ok":
            v = row.get("engine_reported_seconds")
            if isinstance(v, (int, float)) and math.isfinite(v): values[int(row["session"])].append(float(v))
    return {s: statistics.median(xs) for s, xs in values.items() if len(xs) == 5}


def _cell_stability(rows: list[dict[str, Any]], cell_id: str, absolute: float = 5e-5,
                    relative: float = 0.2) -> tuple[bool, dict[str, Any]]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if (row.get("record_type") == "terminal" and row.get("cell_id") == cell_id and
                row.get("status") == "ok" and isinstance(row.get("engine_reported_seconds"), (int, float))):
            v = float(row["engine_reported_seconds"])
            if math.isfinite(v) and v >= 0:
                values[int(row["session"])].append(v)
    medians, details = [], []
    stable = True
    for session in (1, 2, 3):
        xs = values.get(session, [])
        if len(xs) != 5:
            stable = False
            details.append({"session": session, "n": len(xs), "pass": False})
            continue
        median = statistics.median(xs)
        mad = statistics.median(abs(x - median) for x in xs)
        tol = max(absolute, relative * median)
        ok = mad <= tol
        stable &= ok
        medians.append(median)
        details.append({"session": session, "n": 5, "median_seconds": median,
                        "mad_seconds": mad, "tolerance_seconds": tol, "pass": ok})
    if len(medians) == 3:
        spread = max(medians) - min(medians)
        tol = max(absolute, relative * statistics.median(medians))
        session_ok = spread <= tol
        stable &= session_ok
    else:
        spread = tol = None; session_ok = False
    return stable, {"sessions": details, "session_median_range_seconds": spread,
                    "session_median_range_tolerance_seconds": tol,
                    "session_medians_pass": session_ok}


def _identified_increment(values: list[float], absolute: float = 5e-5,
                          relative: float = 0.2) -> dict[str, Any]:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if len(finite) != 3:
        return {"values_seconds": finite, "pass": False, "reason": "not_three_finite_sessions"}
    median = statistics.median(finite)
    spread = max(finite) - min(finite)
    tolerance = max(absolute, relative * median)
    reasons = []
    if any(v <= 0 for v in finite): reasons.append("nonpositive_session_increment")
    if median <= absolute: reasons.append("median_increment_not_resolved_above_absolute_floor")
    if spread > tolerance: reasons.append("session_increment_unstable")
    return {"values_seconds": finite, "median_seconds": median, "range_seconds": spread,
            "range_tolerance_seconds": tolerance, "pass": not reasons, "reasons": reasons}


def reduce_calibration(plan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = plan["calibration_cells"]
    by_key = {(c["width"], c["operation"], c["repeats"], c["shots"]): c for c in cells if c["kind"] == "operation"}
    sampling = {(c["width"], c["shots"]): c for c in cells if c["kind"] == "sampling"}
    cfg = read_json(PROTOCOL_PATH)["calibration"]
    knots = []
    for n in cfg["widths"]:
        word_session: dict[tuple[str, int], dict[int, float]] = {}
        reasons = []
        stability_results = {}
        for cell in (c for c in cells if c["width"] == n):
            stable, report = _cell_stability(rows, cell["cell_id"])
            stability_results[cell["cell_id"]] = report
            if not stable:
                reasons.append(f"unstable_or_incomplete_cell_{cell['cell_id']}")
        for word in cfg["operation_classes"]:
            for sess in range(1, 4):
                vals = {}
                for rep in (256, 512, 1024):
                    cid = by_key[(n, word, rep, cfg["operation_shots"])]["cell_id"]
                    sm = _session_medians(rows, cid)
                    if sess in sm: vals[rep] = sm[sess]
                word_session[(word, sess)] = vals
                if set(vals) != {256, 512, 1024}: reasons.append(f"missing_{word}_session_{sess}")
        increments = {"one_qubit_noncommuting": [], "two_qubit_wrapper_control": [], "two_qubit_cx_interleaved": []}
        linearity = []
        for word in cfg["operation_classes"]:
            for sess in range(1, 4):
                v = word_session[(word, sess)]
                if set(v) != {256, 512, 1024}: continue
                slope = (v[1024] - v[256]) / (1024 - 256)
                increments[word].append(slope)
                pred512 = v[256] + (v[1024] - v[256]) / 3
                tol = max(5e-5, 0.2 * v[512])
                linearity.append({"operation": word, "session": sess, "observed_512": v[512],
                                  "predicted_512": pred512, "absolute_error": abs(v[512] - pred512),
                                  "tolerance": tol, "pass": abs(v[512] - pred512) <= tol})
        inc1 = increments["one_qubit_noncommuting"]
        wrap = increments["two_qubit_wrapper_control"]
        cx = increments["two_qubit_cx_interleaved"]
        cx_extra = [a-b for a, b in zip(cx, wrap)] if len(cx) == len(wrap) == 3 else []
        id1q = _identified_increment([value * (1024 - 256) for value in inc1])
        idcx = _identified_increment([value * (1024 - 256) for value in cx_extra])
        c1q = statistics.median(inc1) / (2 * (2 ** n)) if len(inc1) == 3 else None
        ccx = statistics.median(cx_extra) / (2 ** n) if len(cx_extra) == 3 else None
        if not id1q["pass"]: reasons.extend(f"C1q_increment:{x}" for x in id1q.get("reasons", [id1q.get("reason", "failed")]))
        if not idcx["pass"]: reasons.extend(f"Ccx_increment:{x}" for x in idcx.get("reasons", [idcx.get("reason", "failed")]))
        if len(linearity) != 9 or not all(x["pass"] for x in linearity): reasons.append("heldout_R512_linearity_failed")
        sample_medians = []
        for shots in cfg["sampling_shots"]:
            cid = sampling[(n, shots)]["cell_id"]
            sess = _session_medians(rows, cid)
            sample_medians.append((shots, statistics.median(sess.values()) if len(sess) == 3 else None))
        valid_sample = all(v is not None for _, v in sample_medians)
        if valid_sample:
            xs = [float(x) for x, _ in sample_medians]; ys = [float(y) for _, y in sample_medians]
            xbar, ybar = statistics.mean(xs), statistics.mean(ys)
            slope = sum((x-xbar)*(y-ybar) for x,y in zip(xs,ys)) / sum((x-xbar)**2 for x in xs)
            intercept = ybar - slope*xbar
            sample_ok = intercept >= 0 and slope >= 0 and all(abs(y-(intercept+slope*x)) <= max(5e-5, .2*y) for x,y in zip(xs,ys))
        else:
            intercept = slope = None; sample_ok = False
        if not sample_ok: reasons.append("sampling_affine_fit_unavailable_or_unstable")
        knots.append({"width": n, "C1q_seconds_per_gate_amplitude_scale": c1q,
                      "Ccx_seconds_per_CX_amplitude_scale": ccx,
                      "per_session_operation_slopes": {k: v for k, v in increments.items()},
                      "per_session_CX_minus_wrapper_slopes": cx_extra,
                      "identified_increment_1q": id1q, "identified_increment_cx_differential": idcx,
                      "R512_checks": linearity,
                      "sampling_shot_medians": [{"shots": x, "median_seconds": y} for x,y in sample_medians],
                      "sampling_intercept_seconds": intercept, "sampling_slope_seconds_per_shot": slope,
                      "stability_by_cell": stability_results,
                      "valid": not reasons, "unavailable_reasons": reasons})
    return {"artifact_id": "maestro-cpu-component-calibration-v1",
            "status": "complete_with_unavailable_knots" if any(not k["valid"] for k in knots) else "complete",
            "coefficient_formula": cfg["coefficient_formula"], "sampling_formula": cfg["sampling_formula"],
            "prediction_formula": cfg["prediction"], "interpolation": cfg["interpolation"],
            "interpolation_validity": {"width_knots": [k["width"] for k in knots],
                                       "valid_widths": [k["width"] for k in knots if k["valid"]],
                                       "invalid_widths": [k["width"] for k in knots if not k["valid"]],
                                       "no_extrapolation": True, "no_bridging_invalid_knots": True},
            "knots": knots}


def aggregate(out_dir: Path, plan_path: Path) -> dict[str, Any]:
    protocol, plan, plan_hash = expected_plan_identity(plan_path)
    manifest = read_json(out_dir / "run_manifest.json")
    if manifest.get("phase") != "full" or manifest.get("plan_sha256") != plan_hash:
        raise ValueError("aggregate requires a completed full run for this exact plan")
    if manifest.get("status") != "completed":
        raise ValueError("aggregate requires full execution to be marked completed")
    ledger = out_dir / "attempts.jsonl"
    if not ledger.is_file() or manifest.get("attempts_sha256") != sha_file(ledger):
        raise ValueError("aggregate refuses an attempt ledger changed since run completion")
    records = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
    starts = [r for r in records if r.get("phase") == "full" and r.get("record_type") == "attempt_started"]
    terminals = [r for r in records if r.get("record_type") == "terminal"]
    expected_optimizer = protocol["context"].get("optimize_circuit")
    if expected_optimizer is False:
        if any(row.get("optimizer_enabled_requested") is not False for row in starts + terminals):
            raise ValueError("optimizer-off ledger has an attempt without an explicit false request")
        if any(row.get("status") == "ok" and row.get("optimizer_enabled_resolved") is not False
               for row in terminals):
            raise ValueError("optimizer-off ledger contains a successful call without a resolved false flag")
    cell_hash = {c["cell_id"]: c["qasm_sha256"] for c in plan["calibration_cells"]}
    cell_hash.update({p["source_qasm_sha256"]: p["source_qasm_sha256"] for p in plan["panel"]})
    expected_execution_identity = {}
    for cell in plan["calibration_cells"]:
        expected_execution_identity[cell["cell_id"]] = normalized_execution_identity(cell["qasm"])
    for panel_row in plan["panel"]:
        spec = dict(panel_row, qasm_sha256=panel_row["source_qasm_sha256"], scope="panel")
        try:
            expected_execution_identity[panel_row["source_qasm_sha256"]] = normalized_execution_identity(qasm_for(spec))
        except Exception:
            expected_execution_identity[panel_row["source_qasm_sha256"]] = {
                "normalized_qasm_sha256": None, "normalized_width": None,
                "normalized_1q_gate_count": None, "normalized_cx_gate_count": None}
    expected = {(cell, session, repetition) for cell in cell_hash
                for session in (1, 2, 3) for repetition in (1, 2, 3, 4, 5)}
    def identity(row: dict[str, Any]) -> tuple[Any, ...]:
        return row.get("cell_id"), row.get("session"), row.get("repetition")
    start_ids = [identity(r) for r in starts]
    terminal_ids = [identity(r) for r in terminals]
    if (len(starts) != len(expected) or len(terminals) != len(expected) or
            set(start_ids) != expected or set(terminal_ids) != expected or
            len(set(start_ids)) != len(start_ids) or len(set(terminal_ids)) != len(terminal_ids)):
        raise ValueError("aggregate refuses incomplete, duplicated, or orphaned full attempt identities")
    starts_by_id = {identity(r): r for r in starts}
    for row in terminals:
        ident = identity(row)
        start = starts_by_id[ident]
        if (row.get("phase") != "full" or row.get("thread_arm") != "primary_native_threads_1" or
                start.get("thread_arm") != row.get("thread_arm") or row.get("qasm_sha256") != cell_hash[row["cell_id"]] or
                any(row.get(k) != start.get(k) for k in ("plan_sha256", "protocol_sha256", "worker_sha256", "environment_sha256", "attempt_ordinal",
                                                          "normalized_qasm_sha256", "normalized_width", "normalized_1q_gate_count", "normalized_cx_gate_count")) or
                any(row.get(k) != expected_execution_identity[row["cell_id"]].get(k)
                    for k in ("normalized_qasm_sha256", "normalized_width", "normalized_1q_gate_count", "normalized_cx_gate_count")) or
                row.get("status") not in TERMINAL_STATUSES):
            raise ValueError(f"aggregate detects mismatched start/terminal identity: {ident}")
        if row.get("status") == "ok":
            runtime = row.get("engine_reported_seconds")
            if not isinstance(runtime, (int, float)) or not math.isfinite(runtime) or runtime < 0:
                raise ValueError(f"aggregate detects invalid successful runtime: {ident}")
    targets = []
    for spec in plan["panel"]:
        cell_rows = [r for r in terminals if r.get("cell_id") == spec["source_qasm_sha256"]]
        xs = [float(r["engine_reported_seconds"]) for r in cell_rows
              if r.get("status") == "ok"
              and isinstance(r.get("engine_reported_seconds"), (int,float)) and math.isfinite(r["engine_reported_seconds"])]
        status = "runtime_observed" if len(xs) == 15 and len(cell_rows) == 15 else "unavailable"
        if status == "unavailable":
            causes = sorted({str(r.get("status")) for r in cell_rows if r.get("status") != "ok"})
            if len(causes) == 1 and causes[0] in TERMINAL_STATUSES:
                status = f"unavailable_{causes[0]}"
        targets.append({"source_qasm_sha256": spec["source_qasm_sha256"], "outer_fold": spec["outer_fold"],
                        "runtime_seconds": statistics.median(xs) if status == "runtime_observed" else "",
                        "target_status": status,
                        "measurement_map": json.dumps(spec["measurement_map"], separators=(",", ":")),
                        "measurement_map_sha256": spec["measurement_map_sha256"],
                        "measured_qubit_count": spec["measured_qubit_count"],
                        "measured_classical_bit_count": spec["measured_classical_bit_count"],
                        "measured_classical_extent": spec["measured_classical_extent"],
                        "declared_classical_width": spec["declared_classical_width"],
                        "measurement_terminal": spec["measurement_terminal"],
                        "measurement_map_injective": spec["measurement_map_injective"],
                        **expected_execution_identity[spec["source_qasm_sha256"]]})
    with (out_dir / "panel_targets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_qasm_sha256", "outer_fold", "runtime_seconds", "target_status",
                                                    "measurement_map", "measurement_map_sha256",
                                                    "measured_qubit_count", "measured_classical_bit_count",
                                                    "measured_classical_extent", "declared_classical_width",
                                                    "measurement_terminal", "measurement_map_injective",
                                                    "normalized_qasm_sha256", "normalized_width",
                                                    "normalized_1q_gate_count", "normalized_cx_gate_count"])
        writer.writeheader(); writer.writerows(targets)
    model = reduce_calibration(plan, terminals)
    attempt_hash = sha_file(ledger)
    detailed_knots = model["knots"]
    model["width_coefficients"] = [
        {"width": row["width"], "status": "valid" if row["valid"] else "unavailable",
         "C1q": row["C1q_seconds_per_gate_amplitude_scale"],
         "Ccx": row["Ccx_seconds_per_CX_amplitude_scale"],
         "b": row["sampling_intercept_seconds"],
         "m": row["sampling_slope_seconds_per_shot"],
         "unavailable_reason": ";".join(row["unavailable_reasons"]) if not row["valid"] else None}
        for row in detailed_knots
    ]
    model.update({"schema_id": "maestro-cpu-component-calibration-v1",
                  "artifact_status": "PASS", "shot_count": int(protocol["context"]["shots"]),
                  "protocol_id": protocol["protocol_id"], "method_id": protocol.get("method_id"),
                  "execution_context_id": protocol.get("execution_context_id"),
                  "optimizer_enabled_requested": protocol["context"].get("optimize_circuit"),
                  "resolved_from": protocol.get("resolved_from"),
                  "coefficient_formula": "2^n*(C1q(n)*normalized_1q_count + Ccx(n)*normalized_cx_count) + b(n) + m(n)*shots",
                  "calibration_source_qasm_sha256": sorted(c["qasm_sha256"] for c in plan["calibration_cells"]),
                  "input_attempt_sha256": [attempt_hash], "plan_sha256": plan_hash,
                  "protocol_sha256": sha_file(PROTOCOL_PATH), "attempts_sha256": attempt_hash,
                  "panel_targets_sha256": sha_file(out_dir / "panel_targets.csv"),
                  "assigned_calibration_cells": 112,
                  "complete_panel_hashes": sum(x["target_status"] == "runtime_observed" for x in targets),
                  "assigned_panel_hashes": len(targets), "precision": "FP64/std::complex<double>"})
    atomic_json(out_dir / "calibration_model.json", model)
    return {"status": model["status"], "panel_targets": str(out_dir / "panel_targets.csv"),
            "calibration_model": str(out_dir / "calibration_model.json"),
            "panel_complete": model["complete_panel_hashes"], "panel_assigned": len(targets)}


def preflight(plan_path: Path) -> dict[str, Any]:
    protocol, plan, plan_hash = expected_plan_identity(plan_path)
    native = _native_pins(protocol)
    # Static/API environment inspection only; no Maestro import or circuit execution.
    primary = environment_identity(protocol)
    return {"status": "pass", "plan_sha256": plan_hash, "protocol_sha256": sha_file(PROTOCOL_PATH),
            "calibration_cells": len(plan["calibration_cells"]), "panel_hashes": len(plan["panel"]),
            "pilot_cells": len(plan["pilot_cell_ids"]), "native_pins": native,
            "environment_sha256": canonical_sha(primary), "timing_performed": False,
            "maestro_imported": False, "training_performed": False}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--_worker"]:
        try:
            result = _worker(json.load(sys.stdin))
            print(json.dumps(result, allow_nan=False))
            return 0
        except BaseException as exc:
            print(json.dumps({"status": "worker_error", "error": f"{type(exc).__name__}: {exc}"}))
            return 0
    global PROTOCOL_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["preflight", "pilot10", "full", "aggregate"], required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH,
                        help="Full frozen or resolved runtime protocol; defaults to the historical optimizer-on protocol.")
    args = parser.parse_args(argv)
    try:
        PROTOCOL_PATH = args.protocol.resolve()
        if args.action == "preflight":
            ensure_scratch(args.output_dir)
            report = preflight(args.plan)
            if args.output_dir.exists(): raise FileExistsError("preflight output exists; choose a new scratch path")
            args.output_dir.mkdir(parents=True)
            atomic_json(args.output_dir / "run_manifest.json", {"phase":"preflight", **report})
            print(json.dumps(report, indent=2)); return 0
        if args.action in {"pilot10", "full"}:
            report = execute_stage("pilot" if args.action == "pilot10" else "full", args.plan,
                                   args.output_dir, args.pilot_report)
            print(json.dumps(report, indent=2)); return 0 if report["status"] in {"completed", "pilot_gate_failed"} else 1
        ensure_scratch(args.output_dir)
        report = aggregate(args.output_dir, args.plan)
        print(json.dumps(report, indent=2)); return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
