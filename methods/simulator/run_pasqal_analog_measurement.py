#!/usr/bin/env python3
"""Portable runner for the selected Pasqal analog crossed-grid protocol.

The runner is intentionally separate from the retained historical pilot runner.
It requires ``--execute`` before any simulator call and writes only to a new or
matching resumable output directory outside the publication package.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = "protocol/measurement/pasqal_emu_mps_formula_crossed_pilot.json"
REPRODUCTION_PATH = "protocol/reproduction.json"
CONSTRUCTORS_PATH = "methods/simulator/analog_programs.py"
RUNNER_PATH = "methods/simulator/run_pasqal_analog_measurement.py"
MEASUREMENT_REQUIREMENTS_PATH = "requirements-pasqal-measurement.txt"
EXPECTED_PROTOCOL_ID = "pasqal-emu-mps-formula-crossed-pilot-v1"
# The frozen source contract declares this value in prose, not a numeric key.
REFERENCE_THRESHOLD = 0.9999
REFERENCE_VALIDATION = (
    "For each (family, layout, session), fixed chi=256 must have Fidelity "
    ">=0.9999 against fixed chi=512 before candidate cells are accepted."
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(payload.encode("utf-8"))


def indexed_file(root: Path, relative: str) -> Path:
    index_path = root / "provenance/files.json"
    if index_path.is_symlink() or index_path.parent.is_symlink():
        raise ValueError("package_index_must_not_be_symlink")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = next((row for row in index.get("files", [])
                  if isinstance(row, dict) and row.get("path") == relative), None)
    if entry is None:
        raise ValueError(f"required_file_not_indexed:{relative}")
    path = root / relative
    if (not path.is_relative_to(root) or any(part.is_symlink() for part in
            [path, *path.parents] if part != root and part.is_relative_to(root))
            or not path.is_file()):
        raise ValueError(f"required_file_missing_or_symlink:{relative}")
    if path.stat().st_size != entry.get("bytes") or sha256_file(path) != entry.get("sha256"):
        raise ValueError(f"package_file_pin_mismatch:{relative}")
    return path


def load_protocol(root: Path) -> tuple[dict[str, Any], str, str, str]:
    path = indexed_file(root, PROTOCOL_PATH)
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("manifest_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError("unexpected_pasqal_measurement_protocol")
    reproduction = json.loads(indexed_file(root, REPRODUCTION_PATH).read_text(encoding="utf-8"))
    environment = next((item for item in reproduction.get("environments", [])
                        if item.get("environment_id") == "pasqal_analog_measurement"), None)
    if not isinstance(environment, dict) or environment.get("status") not in {"ready", "not_ready"}:
        raise ValueError("pasqal_measurement_environment_status_invalid")
    requirements = indexed_file(root, MEASUREMENT_REQUIREMENTS_PATH)
    constructor = indexed_file(root, CONSTRUCTORS_PATH)
    runner = indexed_file(root, RUNNER_PATH)
    direct_pins = environment.get("pins")
    if (not isinstance(direct_pins, dict)
            or direct_pins.get("python") != "3.10.21"
            or not isinstance(direct_pins.get("packages"), dict)):
        raise ValueError("pasqal_measurement_direct_pins_missing")
    requirements_pins = parse_requirements(requirements, root)
    if requirements_pins != direct_pins["packages"]:
        raise ValueError("pasqal_measurement_requirements_do_not_match_protocol_pins")
    simulator = source.get("simulator", {})
    measurement = source.get("measurement", {})
    quality = source.get("quality", {})
    if (source.get("source_programs", {}).get("families") != ["adiabatic_afm", "quench"]
            or source.get("source_programs", {}).get("layouts") != [[3, 3], [3, 4], [4, 4]]
            or simulator.get("max_bond_dimensions") != [16, 32, 64]
            or simulator.get("dt_ns") != 10.0
            or simulator.get("precision") != 1e-6
            or simulator.get("max_krylov_dim") != 40
            or simulator.get("interaction_cutoff") != 0.0
            or simulator.get("optimize_qubit_ordering") is not False
            or simulator.get("gpu_concurrency") != 1
            or quality.get("reference_rungs") != [256, 512]
            or quality.get("reference_validation") != REFERENCE_VALIDATION
            or quality.get("candidate_threshold") != 0.99
            or measurement.get("sessions") != 1
            or measurement.get("untimed_warmups_per_session") != 2
            or measurement.get("timed_warm_repetitions_per_session") != 2
            or measurement.get("cell_timeout_seconds") != 1800):
        raise ValueError("pasqal_measurement_contract_does_not_match_selected_crossed_grid")
    return source, sha256_file(path), sha256_file(constructor), sha256_file(runner)


def parse_requirements(path: Path, root: Path) -> dict[str, str]:
    """Read exact pins from the small packaged requirements include chain."""
    result: dict[str, str] = {}
    pending = [path]
    seen: set[Path] = set()
    while pending:
        current = pending.pop()
        current = indexed_file(root, current.relative_to(root).as_posix())
        if current in seen:
            continue
        seen.add(current)
        for raw in current.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("--"):
                continue
            if line.startswith("-r "):
                include = current.parent / line[3:].strip()
                if ".." in include.relative_to(root).parts:
                    raise ValueError("requirements_include_must_stay_inside_package")
                pending.append(include)
            elif "==" in line and not line.startswith("-"):
                name, version = line.split("==", 1)
                package = name.strip().lower().replace("_", "-").split("[", 1)[0]
                result[package] = version.strip()
            else:
                raise ValueError(f"unlocked_requirement:{line}")
    return result


def direct_versions() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "pulser-core", "emu-mps", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}


def transitive_pins(root: Path, environment: dict[str, Any]) -> dict[str, str]:
    lock = environment.get("verified_transitive_lock")
    if not isinstance(lock, dict) or lock.get("status") != "packaged":
        raise ValueError("verified_transitive_measurement_lock_missing")
    path = indexed_file(root, lock["path"])
    if sha256_file(path) != lock.get("sha256"):
        raise ValueError("transitive_measurement_lock_pin_mismatch")
    pins = parse_requirements(path, root)
    if any(pins.get(name) != version for name, version in environment["pins"]["packages"].items()):
        raise ValueError("transitive_measurement_lock_disagrees_with_direct_pins")
    return pins


def check_direct_environment(root: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    reproduction = json.loads(indexed_file(root, REPRODUCTION_PATH).read_text(encoding="utf-8"))
    environment = next(item for item in reproduction["environments"]
                       if item.get("environment_id") == "pasqal_analog_measurement")
    expected = environment["pins"]
    observed = direct_versions()
    locked = transitive_pins(root, environment)
    for name in locked:
        try:
            observed["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            observed["packages"][name] = None
    mismatches = []
    if observed["python"] != expected["python"]:
        mismatches.append({"package": "python", "expected": expected["python"], "observed": observed["python"]})
    for name, version in locked.items():
        actual = observed["packages"].get(name)
        if actual != version:
            mismatches.append({"package": name, "expected": version, "observed": actual})
    check = {"expected": {"python": expected["python"], "packages": locked}, "observed": observed,
             "matches_direct_pins": not mismatches, "mismatches": mismatches}
    if mismatches:
        return check, {}, None
    import numpy as np
    import torch
    import emu_mps
    import pulser
    import pulser.devices
    modules = {"numpy": np, "torch": torch, "emu_mps": emu_mps, "pulser": pulser}
    for name, module in modules.items():
        imported_version = getattr(module, "__version__", None)
        if imported_version is not None and str(imported_version) != expected["packages"].get(
                "pulser-core" if name == "pulser" else "emu-mps" if name == "emu_mps" else name):
            check["matches_direct_pins"] = False
            check["mismatches"].append({"package": f"imported {name}",
                "expected": observed["packages"].get("pulser-core" if name == "pulser"
                    else "emu-mps" if name == "emu_mps" else name),
                "observed": str(imported_version)})
    if not torch.cuda.is_available():
        check["matches_direct_pins"] = False
        check["mismatches"].append({"package": "CUDA runtime", "expected": "available", "observed": "unavailable"})
    return check, modules, np


def measurement_capability_ready(root: Path) -> tuple[bool, str]:
    reproduction = json.loads(indexed_file(root, REPRODUCTION_PATH).read_text(encoding="utf-8"))
    family = next((item for item in reproduction.get("families", [])
                   if item.get("family_id") == "pasqal_emu_mps_analog"), None)
    capability = next((item for item in (family or {}).get("capabilities", [])
                       if item.get("capability_id") == "new_analog_measurements"), None)
    if not isinstance(capability, dict):
        return False, "measurement_capability_missing_from_protocol"
    if capability.get("status") != "ready":
        return False, str(capability.get("reason", "measurement_capability_not_ready"))
    environment = next((item for item in reproduction.get("environments", [])
                        if item.get("environment_id") == "pasqal_analog_measurement"), {})
    if environment.get("status") != "ready":
        return False, "measurement_environment_not_ready"
    try:
        transitive_pins(root, environment)
    except (KeyError, OSError, ValueError) as exc:
        return False, f"measurement_dependency_lock_not_ready:{exc}"
    return True, "measurement_capability_ready"


def load_constructor(path: Path):
    spec = importlib.util.spec_from_file_location("_published_pasqal_analog_programs", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot_load_pinned_pasqal_source_constructors")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def program_descriptor(protocol: dict[str, Any], constructor_sha: str,
                        family: str, rows: int, columns: int) -> tuple[Any, dict[str, Any]]:
    source = protocol["source_programs"]
    sequence = _PROGRAMS.source_sequence(family, rows, columns)
    duration_ns = int(sequence.get_duration())
    if duration_ns <= 0:
        raise ValueError(f"invalid_sequence_duration:{family}:{rows}x{columns}")
    dt_ns = float(protocol["simulator"]["dt_ns"])
    descriptor = {
        "source_repository": source["repository"],
        "source_commit": source["commit"],
        "source_document_sha256": source["document_sha256"],
        "reconstruction_module_sha256": constructor_sha,
        "family": family,
        "rows": rows,
        "columns": columns,
        "sequence_duration_ns": duration_ns,
        "num_steps_estimate": int(round(duration_ns / dt_ns)),
        "step_count_definition": "round(Pulser Sequence.get_duration() / configured EMU-MPS dt_ns); used only for per-step normalization",
        "canonical_tensor_ordering": "Pulser Register.rectangle order; optimize_qubit_ordering=false",
    }
    return sequence, descriptor


def make_config(protocol: dict[str, Any], max_bond: int, observables: list[Any]) -> Any:
    from emu_mps import MPSConfig
    sim = protocol["simulator"]
    return MPSConfig(
        dt=sim["dt_ns"], precision=sim["precision"], max_bond_dim=max_bond,
        max_krylov_dim=sim["max_krylov_dim"], num_gpus_to_use=1,
        interaction_cutoff=sim["interaction_cutoff"],
        optimize_qubit_ordering=sim["optimize_qubit_ordering"],
        observables=observables, log_level=40,
    )


def checked_fidelity(value: Any) -> float:
    """EMU-MPS 2.9.1 overlap is |inner|²; do not square it a second time."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_fidelity:{value}") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0 + 1e-6:
        raise ValueError(f"invalid_fidelity:{result}")
    return result


def run_cell(protocol: dict[str, Any], family: str, rows: int, columns: int,
             max_bond: int, output_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "ok", "error": None,
                               "warm_execute_seconds": [], "warm_network_build_seconds": []}
    try:
        import torch
        from emu_mps import MPSBackend, StateResult
        sequence, descriptor = program_descriptor(
            protocol, _CONSTRUCTOR_SHA, family, rows, columns
        )
        # Both source constructors evolve over normalized interval [0, 1].
        end_time = 1.0
        quality = protocol["quality"]
        chi256, chi512 = quality["reference_rungs"]
        started = time.perf_counter()
        ref512_result = MPSBackend(sequence, config=make_config(
            protocol, chi512, [StateResult(evaluation_times=[end_time])])).run()
        payload["reference_512_seconds"] = time.perf_counter() - started
        ref512 = ref512_result.get_result("state", end_time)
        started = time.perf_counter()
        ref256_result = MPSBackend(sequence, config=make_config(
            protocol, chi256, [StateResult(evaluation_times=[end_time])])).run()
        payload["reference_256_seconds"] = time.perf_counter() - started
        ref256 = ref256_result.get_result("state", end_time)
        started = time.perf_counter()
        reference_fidelity = checked_fidelity(ref512.overlap(ref256))
        payload["reference_overlap_seconds"] = time.perf_counter() - started
        payload["reference_fidelity_256_vs_512"] = reference_fidelity
        payload["quality_reference_hash"] = canonical_digest({
            "descriptor": descriptor, "reference_rungs": [chi256, chi512], "metric": "Fidelity"
        })
        if quality.get("reference_validation") != REFERENCE_VALIDATION:
            raise ValueError("reference_threshold_not_bound_to_frozen_contract")
        if reference_fidelity < REFERENCE_THRESHOLD:
            payload.update(status="quality_failed", error=f"reference fidelity below threshold: {reference_fidelity}")
        else:
            started = time.perf_counter()
            backend = MPSBackend(sequence, config=make_config(protocol, max_bond, []))
            payload["network_build_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            backend.run()
            payload["first_execute_seconds"] = time.perf_counter() - started
            for _ in range(protocol["measurement"]["untimed_warmups_per_session"]):
                MPSBackend(sequence, config=make_config(protocol, max_bond, [])).run()
            for _ in range(protocol["measurement"]["timed_warm_repetitions_per_session"]):
                started = time.perf_counter()
                warm_backend = MPSBackend(sequence, config=make_config(protocol, max_bond, []))
                payload["warm_network_build_seconds"].append(time.perf_counter() - started)
                started = time.perf_counter()
                warm_backend.run()
                payload["warm_execute_seconds"].append(time.perf_counter() - started)
            started = time.perf_counter()
            candidate_result = MPSBackend(sequence, config=make_config(
                protocol, max_bond, [StateResult(evaluation_times=[end_time])])).run()
            payload["candidate_quality_state_capture_seconds"] = time.perf_counter() - started
            candidate = candidate_result.get_result("state", end_time)
            started = time.perf_counter()
            payload["candidate_fidelity"] = checked_fidelity(ref256.overlap(candidate))
            payload["candidate_overlap_seconds"] = time.perf_counter() - started
            payload["quality_status"] = (
                "ok" if payload["candidate_fidelity"] >= quality["candidate_threshold"]
                else "quality_failed"
            )
        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
    except MemoryError as exc:
        payload.update(status="resource_limit", error=f"MemoryError: {exc}")
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}".splitlines()[0][:1000]
        status = ("unsupported" if "cuda unavailable" in message.lower()
                  else "resource_limit" if "out of memory" in message.lower() else "error")
        payload.update(status=status, error=message, error_trace=traceback.format_exc(limit=8))
    output_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return payload


def gpu_inventory() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT, timeout=10,
        ).strip().splitlines()
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    if len(output) != 1:
        return {"status": "unexpected_gpu_count", "visible_devices": output}
    fields = [item.strip() for item in output[0].split(",")]
    return {"status": "observed", "name": fields[0], "uuid": fields[1],
            "memory_total": fields[2], "driver_version": fields[3]}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(raw_path, path)
    except BaseException:
        try:
            os.unlink(raw_path)
        except FileNotFoundError:
            pass
        raise


def ensure_output_directory(root: Path, output_dir: Path) -> Path:
    requested = output_dir.expanduser().absolute()
    parent = requested.parent.resolve(strict=True)
    output = parent / requested.name
    if output == root or output.is_relative_to(root):
        raise ValueError("measurement_output_must_be_outside_publication_package")
    if requested.is_symlink() or output.is_symlink():
        raise ValueError("measurement_output_must_not_be_symlink")
    if output.exists():
        if not output.is_dir():
            raise FileExistsError("measurement_output_must_be_directory")
    else:
        output.mkdir()
    return output.resolve(strict=True)


def base_record(run_id: str, protocol: dict[str, Any], descriptor: dict[str, Any],
                program_hash: str, family: str, rows: int, columns: int, bond: int) -> dict[str, Any]:
    pulse_hash = canonical_digest(descriptor)
    layout_hash = canonical_digest({"rows": rows, "columns": columns})
    return {
        "protocol_id": "qre-benchmark-v1",
        "run_id": run_id,
        "row_id": f"{family}|{rows}x{columns}|chi{bond}",
        "circuit_hash": pulse_hash,
        "context_id": f"pasqal-emu-mps-gpu-chi{bond}-precision{protocol['simulator']['precision']:g}",
        "session_id": "session-1",
        "seed_record": {"seed_not_required": "fixed source-pinned deterministic pulse program"},
        "input_hashes": {"pulse_hash": pulse_hash, "atom_layout_hash": layout_hash},
        "metadata": {
            **descriptor,
            "reconstruction_module_sha256": program_hash,
            "pulse_hash": pulse_hash,
            "atom_layout_hash": layout_hash,
            "max_bond_dimension": bond,
            "quality_threshold": protocol["quality"]["candidate_threshold"],
            "reference_validation_threshold": REFERENCE_THRESHOLD,
            "first_call_scope": "first candidate run after two reference runs; not a cold-process or first-CUDA-call measurement",
            "execution_clock": "host perf_counter around MPSBackend.run with observables=[]; constructor clock separate; state capture and overlap are post-timer quality work",
        },
    }


def run_parent(args: argparse.Namespace) -> int:
    root = args.package_root.expanduser().resolve(strict=True)
    ready, reason = measurement_capability_ready(root)
    if not ready:
        raise RuntimeError(f"measurement_capability_not_ready:{reason}; no simulation or timing was started")
    protocol, protocol_hash, constructor_hash, runner_hash = load_protocol(root)
    environment_check, modules, numpy_module = check_direct_environment(root)
    if not environment_check["matches_direct_pins"]:
        raise RuntimeError(f"direct_environment_mismatch:{json.dumps(environment_check,sort_keys=True)}")
    torch = modules["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_runtime_unavailable")
    constructor = load_constructor(indexed_file(root, CONSTRUCTORS_PATH))
    global _PROGRAMS, _CONSTRUCTOR_SHA
    _PROGRAMS, _CONSTRUCTOR_SHA = constructor, constructor_hash
    gpu = gpu_inventory()
    if gpu.get("status") != "observed":
        raise RuntimeError(f"one_visible_gpu_required:{gpu}")

    program_descriptors: dict[tuple[str, int, int], dict[str, Any]] = {}
    for family in protocol["source_programs"]["families"]:
        for rows, columns in protocol["source_programs"]["layouts"]:
            _, descriptor = program_descriptor(protocol, constructor_hash, family, rows, columns)
            program_descriptors[(family, rows, columns)] = descriptor

    output = ensure_output_directory(root, args.output_dir)
    run_id = args.run_id or "pasqal_analog_crossed_grid"
    environment = {
        "python": platform.python_version(),
        "numpy_module_version": numpy_module.__version__,
        "pulser_core": importlib.metadata.version("pulser-core"),
        "emu_mps": importlib.metadata.version("emu-mps"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_concurrency": 1,
        "thread_policy": "one sequential GPU worker; no concurrent timing workload",
        "requirements_sha256": sha256_file(indexed_file(root, MEASUREMENT_REQUIREMENTS_PATH)),
        "direct_environment_check": environment_check,
    }
    run_identity = {
        "run_id": run_id,
        "protocol_sha256": protocol_hash,
        "constructor_sha256": constructor_hash,
        "runner_sha256": runner_hash,
        "environment": environment,
    }
    run_manifest_path = output / "run_manifest.json"
    raw_path = output / "raw_records.jsonl"
    if run_manifest_path.is_symlink() or raw_path.is_symlink():
        raise ValueError("measurement_resume_files_must_not_be_symlinks")
    if run_manifest_path.exists() or raw_path.exists():
        if not run_manifest_path.is_file() or not raw_path.is_file():
            raise ValueError("incomplete_measurement_output_cannot_resume")
        existing = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_identity") != run_identity:
            raise ValueError("resume_identity_does_not_match_protocol_code_or_environment")
    else:
        if any(output.iterdir()):
            raise FileExistsError("nonempty_measurement_output_has_no_matching_run_manifest")
        raw_path.write_text("", encoding="utf-8")
        atomic_json(run_manifest_path, {
            "run_id": run_id,
            "method_id": "pasqal_emu_mps",
            "target_id": "local_analog_mps",
            "protocol_id": protocol["manifest_id"],
            "status": "running",
            "protocol_sha256": protocol_hash,
            "constructor_sha256": constructor_hash,
            "runner_sha256": runner_hash,
            "run_identity": run_identity,
            "environment": environment,
            "assigned_cells": len(protocol["source_programs"]["families"])
            * len(protocol["source_programs"]["layouts"])
            * len(protocol["simulator"]["max_bond_dimensions"]),
            "timing_performed": True,
            "simulation_performed": True,
            "thread_policy": "one sequential GPU worker",
        })

    raw_records = []
    for line_number, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"raw_record_not_object_at_line:{line_number}")
        raw_records.append(record)
    settings = protocol["measurement"]
    planned_cells = []
    for family in protocol["source_programs"]["families"]:
        for rows, columns in protocol["source_programs"]["layouts"]:
            for bond in protocol["simulator"]["max_bond_dimensions"]:
                descriptor = program_descriptors[(family, rows, columns)]
                record = base_record(run_id, protocol, descriptor, constructor_hash,
                                     family, rows, columns, bond)
                key = (record["row_id"], record["context_id"], record["session_id"])
                planned_cells.append((key, record, family, rows, columns, bond))
    expected_by_key = {key: record for key, record, *_ in planned_cells}
    done = validate_resume_records(raw_records, expected_by_key, settings)
    for key, record, family, rows, columns, bond in planned_cells:
        if key in done:
            continue
        with tempfile.TemporaryDirectory(prefix="pasqal-cell-") as temp_dir:
            payload_path = Path(temp_dir) / "payload.json"
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--worker", "--package-root", str(root),
                "--family", family, "--rows", str(rows), "--columns", str(columns),
                "--max-bond", str(bond), "--worker-output", str(payload_path),
            ]
            print(f"[pasqal-analog] {record['row_id']}", flush=True)
            try:
                process = subprocess.run(command, text=True, capture_output=True,
                                         timeout=settings["cell_timeout_seconds"], check=False)
                payload = (json.loads(payload_path.read_text(encoding="utf-8"))
                           if payload_path.is_file() else {
                               "status": "error",
                               "error": f"worker exit={process.returncode}; {process.stderr[-600:]}"
                           })
            except subprocess.TimeoutExpired:
                payload = {"status": "timeout",
                           "error": f"cell exceeded protocol timeout {settings['cell_timeout_seconds']}s"}
        status = payload.get("status", "error")
        if status == "ok":
            status = payload.get("quality_status", "error")
        quality = {key: payload.get(key) for key in (
            "reference_fidelity_256_vs_512", "candidate_fidelity", "quality_reference_hash",
            "reference_512_seconds", "reference_256_seconds", "reference_overlap_seconds",
            "candidate_quality_state_capture_seconds", "candidate_overlap_seconds",
        )}
        rows_to_write = [
            {**record, "stage": "network_build", "repetition": 0, "status": status,
             "elapsed_seconds": payload.get("network_build_seconds"), "quality": quality,
             "error": payload.get("error")},
            {**record, "stage": "first_execute", "repetition": 0, "status": status,
             "elapsed_seconds": payload.get("first_execute_seconds"), "quality": quality,
             "error": payload.get("error")},
        ]
        warm_values = payload.get("warm_execute_seconds", [])
        for repetition in range(settings["timed_warm_repetitions_per_session"]):
            value = warm_values[repetition] if isinstance(warm_values, list) and repetition < len(warm_values) else None
            rows_to_write.append({
                **record, "stage": "warm_execute", "repetition": repetition,
                "observation_kind": "raw_repetition", "status": status,
                "elapsed_seconds": value, "quality": quality, "error": payload.get("error"),
            })
        # Whole-cell replacement: interruption leaves either the old complete
        # cells or all four new records, never a first-only resume marker.
        validate_resume_records(raw_records + rows_to_write, expected_by_key, settings)
        atomic_records(raw_path, raw_records + rows_to_write)
        raw_records.extend(rows_to_write)
        done.add(key)

    done = validate_resume_records(raw_records, expected_by_key, settings)

    final = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    final["status"] = "complete" if len(done) == len(planned_cells) else "running"
    final["completed_cells"] = len(done)
    final["completion_definition"] = "terminal attempted cells, including quality/resource/timeout failures; not quality-pass cells"
    final["raw_records_sha256"] = sha256_file(raw_path)
    atomic_json(run_manifest_path, final)
    return 0 if final["status"] == "complete" else 2


def validate_resume_records(records: list[dict[str, Any]], expected: dict,
                            settings: dict[str, Any]) -> set:
    """Fail before launching work if saved cells are duplicate, partial or foreign."""
    stages = {("network_build", 0), ("first_execute", 0)} | {
        ("warm_execute", repetition)
        for repetition in range(settings["timed_warm_repetitions_per_session"])
    }
    grouped: dict[tuple, set] = {}
    for row in records:
        key = (row.get("row_id"), row.get("context_id"), row.get("session_id"))
        template = expected.get(key)
        if template is None or any(row.get(field) != template[field] for field in (
                "run_id", "protocol_id", "circuit_hash", "input_hashes", "metadata")):
            raise ValueError("raw_records_contain_rows_outside_frozen_measurement_grid")
        stage = (row.get("stage"), row.get("repetition"))
        seen = grouped.setdefault(key, set())
        if stage not in stages or stage in seen:
            raise ValueError("duplicate_or_unexpected_measurement_record")
        seen.add(stage)
        status, elapsed = row.get("status"), row.get("elapsed_seconds")
        if status not in {"ok", "quality_failed", "resource_limit", "unsupported", "error", "timeout"}:
            raise ValueError("invalid_measurement_terminal_status")
        if (elapsed is not None and (isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed) or elapsed < 0)) or (status == "ok" and elapsed is None):
            raise ValueError("invalid_measurement_elapsed_seconds")
        if status == "ok":
            quality = row.get("quality") or {}
            if (checked_fidelity(quality.get("reference_fidelity_256_vs_512")) < REFERENCE_THRESHOLD
                    or checked_fidelity(quality.get("candidate_fidelity")) < template["metadata"]["quality_threshold"]):
                raise ValueError("successful_record_does_not_pass_quality")
    if any(seen != stages for seen in grouped.values()):
        raise ValueError("incomplete_measurement_cell_cannot_resume")
    return set(grouped)


def atomic_records(path: Path, records: list[dict[str, Any]]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def worker_main(args: argparse.Namespace) -> int:
    root = args.package_root.expanduser().resolve(strict=True)
    ready, reason = measurement_capability_ready(root)
    if not ready:
        args.worker_output.write_text(json.dumps({
            "status": "not_ready", "error": f"measurement_capability_not_ready:{reason}"
        }, sort_keys=True), encoding="utf-8")
        return 2
    protocol, _, constructor_hash, _ = load_protocol(root)
    check, modules, _ = check_direct_environment(root)
    if not check["matches_direct_pins"]:
        args.worker_output.write_text(json.dumps({"status": "unsupported",
            "error": f"direct_environment_mismatch:{check}"}, sort_keys=True), encoding="utf-8")
        return 2
    if not modules["torch"].cuda.is_available():
        args.worker_output.write_text(json.dumps({"status": "unsupported",
            "error": "cuda_runtime_unavailable"}, sort_keys=True), encoding="utf-8")
        return 2
    global _PROGRAMS, _CONSTRUCTOR_SHA
    _PROGRAMS = load_constructor(indexed_file(root, CONSTRUCTORS_PATH))
    _CONSTRUCTOR_SHA = constructor_hash
    if args.family not in protocol["source_programs"]["families"]:
        raise ValueError("worker_family_not_in_frozen_protocol")
    if [args.rows, args.columns] not in protocol["source_programs"]["layouts"]:
        raise ValueError("worker_layout_not_in_frozen_protocol")
    if args.max_bond not in protocol["simulator"]["max_bond_dimensions"]:
        raise ValueError("worker_bond_not_in_frozen_protocol")
    run_cell(protocol, args.family, args.rows, args.columns, args.max_bond, args.worker_output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path,
                        help="new external directory for raw records and run manifest")
    parser.add_argument("--run-id", default="pasqal_analog_crossed_grid")
    parser.add_argument("--execute", action="store_true",
                        help="required to run EMU-MPS and record timing")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--family", help=argparse.SUPPRESS)
    parser.add_argument("--rows", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--columns", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-bond", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


_PROGRAMS = None
_CONSTRUCTOR_SHA = ""


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker:
        if any(value is None for value in (args.family, args.rows, args.columns,
                                           args.max_bond, args.worker_output)):
            parser.error("internal worker arguments are incomplete")
        try:
            return worker_main(args)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"measurement not run: {exc}", file=sys.stderr)
            return 2
    if not args.execute:
        parser.error("measurement is not started implicitly; pass --execute only after reviewing the measurement contract")
    if args.output_dir is None:
        parser.error("--output-dir is required with --execute")
    if not args.run_id or "/" in args.run_id or "\\" in args.run_id:
        parser.error("--run-id must be a non-empty single path component")
    try:
        return run_parent(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"measurement not run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
