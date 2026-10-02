#!/usr/bin/env python3
"""Preflight/predict the frozen Maestro QCSim common panel; optionally measure.

Default execution is parser/API preflight plus component predictions only.
QCSim execution requires --measure and both explicit C3a release flags.  No
selector or oracle metrics are computed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing as mp
import platform
import queue as queue_module
import re
import sys
import time
import os
import traceback
import contextlib
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_v1.qre_benchmark.maestro_component_predictor import (  # noqa: E402
    CandidateRuntimePrediction,
    CandidateRuntimePredictionV2,
    CircuitOperationFeatures,
    FittedCalibration,
    PredictionUnavailable,
    fit_frozen_calibration,
    parse_qasm_operation_features,
    predict_candidate_runtime,
    predict_candidate_runtime_v2,
    predict_candidate_runtime_v3,
    predict_candidate_runtime_partial,
)
from benchmark_v1.qre_benchmark.maestro_component_v2 import (  # noqa: E402
    V2_CLOCK_FIELD,
    V2_CLOCK_ID,
    V2_MANIFEST_ID,
    V3_MANIFEST_ID,
    V3_TIMED_STAGE,
    V3_UNTIMED_STAGE,
    PARTIAL_FIT_VERSION,
    V3Unavailable,
    V3_PANEL_ATTEMPT_STATUS_SCHEMA,
    build_v3_run_identity,
    resolve_v3_contract,
    validate_v3_resume_identity,
    V2Unavailable,
    canonical_hash as v2_canonical_hash,
    canonical_json as v2_canonical_json,
    environment_pins as v2_environment_pins,
    normalize_qasm as v2_normalize_qasm,
    package_tree_hash as v2_package_tree_hash,
    quality_gate_status as v2_quality_gate_status,
    resolve_v2_config as v2_resolve_config,
    sha256_file as v2_sha256_file,
    statevector_reference_fidelity as v2_statevector_reference_fidelity,
    validate_resume_identity as v2_validate_resume_identity,
    fit_v2_calibration_rows as v2_fit_calibration_rows,
    fit_v3_calibration_rows,
)


DEFAULT_MANIFEST = ROOT / "benchmark_v1/execution/manifests/maestro_common_q16_componentwise_v1.json"
DEFAULT_CALIBRATION_MANIFEST = ROOT / "benchmark_v1/execution/manifests/maestro_componentwise_core_v3.json"
DEFAULT_CALIBRATION_RAW = ROOT / "artifacts/benchmark_v1/maestro_componentwise_core_v3_20260926/raw_records.csv"
V2_SPLIT_ID = "maestro_common_q16_exact_qasm_group_5fold_v1"
V2_CANDIDATE_IDS = ("statevector", "mps_fixed_chi")
PARTIAL_MANIFEST_ID = "maestro-common-panel-completion-v3-partial-resume"
PARTIAL_MANIFEST_SHA256 = "6c16608fee8ea664be606d8605da13821be2e07bdcc7874dee44d4700d560db4"
PARTIAL_BASE_MANIFEST_SHA256 = "bc2a3bc150421698ff09e08a2055a40979f96d2965fe2dcb3e20402c1016574d"
MAESTRO_TIMING_LOCK_PATH = ROOT / "work/locks/maestro_qcsim_v2_cpu_timing.lock"


@contextlib.contextmanager
def maestro_timing_lock(protocol_id: str, run_identity_sha256: str,
                        root: Path = ROOT):
    """Serialize all Maestro timing protocols and bind the lock to run identity."""
    import fcntl

    # Retain the historical filename as the shared cross-protocol flock path;
    # its persistent existence is normal and is not evidence of a live owner.
    lock_path = root / "work/locks/maestro_qcsim_v2_cpu_timing.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another exclusive Maestro timing worker owns the shared lock") from exc
        metadata = {
            "schema": "maestro_timing_lock_v1",
            "protocol_id": str(protocol_id),
            "run_identity_sha256": str(run_identity_sha256),
            "pid": os.getpid(),
        }
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n")
        lock_handle.flush()
        os.fsync(lock_handle.fileno())
        try:
            yield {"path": str(lock_path.resolve()), **metadata}
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def resolve_partial_contract(manifest_path: Path, base_path: Path | None = None) -> dict[str, Any]:
    """Purely resolve the locked S6 amendment over its immutable v3 parent.

    The parent contract is returned unchanged so existing panel/cell algebra
    continues to use the exact frozen v3 descriptors and IDs.  Amendment
    policy and its identity are carried separately and must be included in
    every partial-run identity; this resolver does not import Maestro or call
    native APIs.
    """
    manifest_path = Path(manifest_path).resolve()
    try:
        amendment_bytes = manifest_path.read_bytes()
        amendment = json.loads(amendment_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V3Unavailable(f"partial_manifest_unreadable:{type(exc).__name__}") from exc
    amendment_sha256 = sha256_bytes(amendment_bytes)
    if amendment_sha256 != PARTIAL_MANIFEST_SHA256:
        raise V3Unavailable(f"partial_manifest_hash_mismatch:{amendment_sha256}")
    if (amendment.get("schema_version") != "maestro_partial_completion_policy_v1"
            or amendment.get("manifest_id") != PARTIAL_MANIFEST_ID):
        raise V3Unavailable("partial_manifest_identity_mismatch")
    width_policy = amendment.get("zero_state_control_policy", {})
    if (width_policy.get("policy_id") != "untimed_zero_state_all_wire_measurement_v1"
            or width_policy.get("operation") != "measure q -> c;"):
        raise V3Unavailable("partial_zero_state_width_policy_mismatch")
    previous = amendment["evidence"]["failed_partial_bootstrap"]
    previous_path = ROOT / previous["path"]
    if sha256_file(previous_path) != previous["sha256"]:
        raise V3Unavailable("partial_failed_bootstrap_evidence_hash_mismatch")
    width_failure = amendment["evidence"]["failed_width_bootstrap"]
    if sha256_file(ROOT / width_failure["path"]) != width_failure["sha256"]:
        raise V3Unavailable("partial_failed_width_bootstrap_evidence_hash_mismatch")
    identity_policy = amendment.get("execution_identity_policy", {})
    if (identity_policy.get("policy_id") != "canonical_identity_body_v1"
            or identity_policy.get("old_preflight_reuse_allowed") is not False):
        raise V3Unavailable("partial_execution_identity_policy_mismatch")
    for key in ("superseded_identity_bootstrap", "superseded_identity_acceptance"):
        evidence = amendment["evidence"][key]
        if sha256_file(ROOT / evidence["path"]) != evidence["sha256"]:
            raise V3Unavailable(f"partial_identity_predecessor_hash_mismatch:{key}")

    base_spec = amendment.get("base_contract")
    if not isinstance(base_spec, dict):
        raise V3Unavailable("partial_base_contract_missing")
    expected_base_rel = "benchmark_v1/execution/manifests/maestro_common_panel_completion_v3.json"
    if base_spec.get("path") != expected_base_rel:
        raise V3Unavailable("partial_base_contract_path_mismatch")
    if base_spec.get("sha256") != PARTIAL_BASE_MANIFEST_SHA256:
        raise V3Unavailable("partial_base_contract_declared_hash_mismatch")
    selected_base = (Path(base_path).resolve() if base_path is not None
                     else ROOT / expected_base_rel)
    actual_base_sha256 = sha256_file(selected_base) if selected_base.is_file() else ""
    if actual_base_sha256 != PARTIAL_BASE_MANIFEST_SHA256:
        raise V3Unavailable(f"partial_base_contract_hash_mismatch:{actual_base_sha256}")
    try:
        parent_contract = resolve_v3_contract(selected_base)
    except Exception as exc:
        reason = str(exc) if isinstance(exc, V3Unavailable) else f"{type(exc).__name__}:{exc}"
        raise V3Unavailable(f"partial_parent_v3_resolution_failed:{reason}") from exc
    if parent_contract.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("partial_parent_is_not_v3")

    calibration_policy = amendment.get("calibration_policy")
    panel_policy = amendment.get("panel_policy")
    preparation_policy = amendment.get("preparation_policy")
    if not all(isinstance(value, dict) for value in (
        calibration_policy, panel_policy, preparation_policy,
    )):
        raise V3Unavailable("partial_policy_sections_missing")
    quarantine = calibration_policy.get("quarantine_selector")
    if not isinstance(quarantine, dict):
        raise V3Unavailable("partial_quarantine_selector_missing")
    expected_counts = {
        "assigned_cells": 432,
        "quarantined_operation_cells": 27,
        "quarantined_diagnostic_cells": 6,
        "quarantined_total_cells": 33,
        "eligible_primary_cells": 393,
        "eligible_diagnostic_cells": 6,
        "eligible_total_cells": 399,
        "planned_timed_calls_if_complete": 5985,
        "planned_untimed_calls_if_complete": 3591,
        "planned_native_calls_if_complete": 9576,
    }
    for key, expected in expected_counts.items():
        if calibration_policy.get(key) != expected:
            raise V3Unavailable(f"partial_calibration_allocation_mismatch:{key}")
    if (int(panel_policy.get("assigned_members", -1)) != 204
            or int(panel_policy.get("assigned_candidate_attempts", -1)) != 408
            or int(parent_contract.get("panel", {}).get("assigned_members", -1)) != 204
            or int(parent_contract.get("panel", {}).get("assigned_candidate_attempts", -1)) != 408):
        raise V3Unavailable("partial_panel_denominator_mismatch")
    if (quarantine.get("candidate") != "mps_fixed_chi"
            or int(quarantine.get("width", -1)) != 16
            or [int(value) for value in quarantine.get("chis", [])] != [4, 8, 32]
            or set(quarantine.get("affected_kinds", [])) != {
                "operation", "entangled_sampling_transfer_diagnostic"
            }):
        raise V3Unavailable("partial_quarantine_selector_mismatch")

    amendment_identity = {
        "manifest_id": PARTIAL_MANIFEST_ID,
        "schema_version": amendment["schema_version"],
        "path": str(manifest_path.relative_to(ROOT)),
        "sha256": amendment_sha256,
    }
    parent_contract_sha256 = v2_canonical_hash(parent_contract)
    quarantine_policy = {
        "assigned_cells": calibration_policy["assigned_cells"],
        "selector": dict(quarantine),
        "selector_rule": calibration_policy.get("quarantine_selector_rule", ""),
        "quarantined_operation_cells": calibration_policy["quarantined_operation_cells"],
        "quarantined_diagnostic_cells": calibration_policy["quarantined_diagnostic_cells"],
        "quarantined_total_cells": calibration_policy["quarantined_total_cells"],
        "eligible_primary_cells": calibration_policy["eligible_primary_cells"],
        "eligible_diagnostic_cells": calibration_policy["eligible_diagnostic_cells"],
        "eligible_total_cells": calibration_policy["eligible_total_cells"],
        "skip_encoding": calibration_policy.get("skip_encoding", ""),
        "coefficients": calibration_policy.get("coefficients", ""),
        "zero_state_sampling_is_independent": True,
    }
    resolved_contract = {
        "schema_version": "maestro_partial_resolved_contract_v1",
        "amendment_identity": amendment_identity,
        "parent_manifest_id": V3_MANIFEST_ID,
        "parent_contract_sha256": parent_contract_sha256,
        "base_resolution_provenance": parent_contract.get("resolution_provenance", {}),
        "preparation_policy": preparation_policy,
        "zero_state_control_policy": width_policy,
        "calibration_policy": calibration_policy,
        "panel_policy": panel_policy,
        "deadline_policy": amendment.get("deadline_policy", {}),
        "execution_authority": amendment.get("execution_authority", {}),
        "outputs": amendment.get("outputs", {}),
        "claim_boundary": amendment.get("claim_boundary", ""),
    }
    return {
        "contract": parent_contract,
        "amendment": amendment,
        "amendment_identity": amendment_identity,
        "parent_contract_sha256": parent_contract_sha256,
        "resolved_contract": resolved_contract,
        "resolved_contract_sha256": v2_canonical_hash(resolved_contract),
        "quarantine_policy": quarantine_policy,
    }


def partial_resume_identity(resolution: dict[str, Any], manifest_sha256: str,
                            input_pins: dict[str, Any], environment: dict[str, Any],
                            binary_pins: dict[str, Any], configs: dict[str, dict[str, Any]],
                            calibration_run_sha256: str, calibration_ledger_sha256: str,
                            calibration_run_identity_sha256: str,
                            panel_pilot_selection: dict[str, Any]) -> dict[str, Any]:
    """Build a partial-only identity; v3/v2 runs and ledgers cannot resume it."""
    amendment_identity = resolution.get("amendment_identity")
    if not isinstance(amendment_identity, dict) or amendment_identity.get("manifest_id") != PARTIAL_MANIFEST_ID:
        raise V3Unavailable("partial_identity_requires_pinned_amendment")
    if not isinstance(manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
        raise V3Unavailable("partial_identity_hash_invalid:manifest")
    calibration_hashes = (calibration_run_sha256, calibration_ledger_sha256,
                          calibration_run_identity_sha256)
    if any(calibration_hashes) and any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in calibration_hashes
    ):
        raise V3Unavailable("partial_identity_calibration_hash_invalid_or_incomplete")
    contract = resolution["contract"]
    pins = {
        "amendment_identity": amendment_identity,
        "parent_contract_sha256": resolution["parent_contract_sha256"],
        "resolved_contract_sha256": resolution["resolved_contract_sha256"],
        "overlay_manifest_sha256": manifest_sha256,
        "input_pins": input_pins,
        "environment_pins": environment,
        "binary_pins": binary_pins,
        "candidate_config_sha256": {
            name: str(value.get("sha256", "")) for name, value in sorted(configs.items())
        },
        "resolved_config_sha256": v2_resolved_config_sha256(configs),
        "calibration_run_manifest_sha256": calibration_run_sha256,
        "calibration_ledger_sha256": calibration_ledger_sha256,
        "calibration_run_identity_sha256": calibration_run_identity_sha256,
        "selected_clock_id": str(contract["measurement"]["selected_clock_id"]),
        "selected_clock_field": str(contract["measurement"]["selected_clock_field"]),
        "split_id": V2_SPLIT_ID,
        "panel_pilot_selection": panel_pilot_selection,
    }
    body = {
        "identity_schema": "maestro_partial_panel_run_identity_v1",
        "manifest_id": PARTIAL_MANIFEST_ID,
        "parent_manifest_id": V3_MANIFEST_ID,
        "resolved_contract_sha256": resolution["resolved_contract_sha256"],
        "pins": pins,
    }
    return {
        "manifest_id": PARTIAL_MANIFEST_ID,
        "run_identity": body,
        "run_identity_sha256": v2_canonical_hash(body),
    }


def partial_plan_identity(resolution: dict[str, Any], manifest_sha256: str,
                          input_pins: dict[str, Any], environment: dict[str, Any],
                          binary_pins: dict[str, Any], configs: dict[str, dict[str, Any]],
                          panel_pilot_selection: dict[str, Any]) -> dict[str, Any]:
    """Dry-run identity explicitly records that no accepted calibration exists yet."""
    return partial_resume_identity(
        resolution, manifest_sha256, input_pins, environment, binary_pins, configs,
        "", "", "", panel_pilot_selection,
    )


def validate_partial_resume_identity(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    """Refuse v2/v3 identity reuse or any changed S6 input/context/calibration pin."""
    if existing.get("manifest_id") != PARTIAL_MANIFEST_ID or expected.get("manifest_id") != PARTIAL_MANIFEST_ID:
        raise V3Unavailable("partial_resume_rejects_non_partial_records")
    old_body = existing.get("run_identity")
    new_body = expected.get("run_identity")
    if not isinstance(old_body, dict) or not isinstance(new_body, dict):
        raise V3Unavailable("partial_resume_identity_missing")
    if v2_canonical_hash(old_body) != existing.get("run_identity_sha256"):
        raise V3Unavailable("partial_existing_resume_identity_hash_mismatch")
    if v2_canonical_hash(new_body) != expected.get("run_identity_sha256"):
        raise V3Unavailable("partial_expected_resume_identity_hash_mismatch")
    if existing.get("run_identity_sha256") != expected.get("run_identity_sha256"):
        raise V3Unavailable("partial_resume_identity_mismatch")


CANONICAL_WORKER_IMPORTS = ("maestro", "numpy", "scipy.linalg")


def _canonical_native_imports() -> dict[str, Any]:
    """Load the pinned native-facing modules in one order in every worker."""
    return {name: importlib.import_module(name) for name in CANONICAL_WORKER_IMPORTS}


def _context_value(value: Any) -> Any:
    """Convert resolved config data to canonical JSON without repr fallbacks."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, dict):
        return {str(key): _context_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_context_value(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return _context_value(value.value)
    if hasattr(value, "name") and isinstance(value.name, str):
        return {"enum_type": type(value).__name__, "name": value.name}
    if hasattr(value, "__dict__"):
        return _context_value(vars(value))
    raise TypeError(f"worker context value is not canonically serializable: {type(value).__name__}")


def _path_fingerprint(path: str | Path | None) -> dict[str, str] | None:
    if not path:
        return None
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "sha256": "unavailable:not_a_file"}
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def collect_worker_context(config: Any = None,
                           backend_threading_policy: Any = None) -> dict[str, Any]:
    """Fingerprint the actual worker after canonical imports and before its API call.

    Identity excludes PID/time and includes only execution-relevant, reproducible
    state.  Observation metadata is retained separately so parent telemetry is
    never mistaken for the worker's execution identity.
    """
    modules = _canonical_native_imports()
    maestro = modules["maestro"]
    numpy = modules["numpy"]
    scipy = importlib.import_module("scipy")
    scipy_linalg = modules["scipy.linalg"]

    libraries: dict[str, Any] = {}
    for key, module, version, package_root in (
        ("maestro", maestro, _installed_version("maestro", maestro),
         Path(maestro.__file__).resolve().parent),
        ("numpy", numpy, str(getattr(numpy, "__version__", "unknown")),
         Path(numpy.__file__).resolve().parent),
        ("scipy.linalg", scipy_linalg, str(getattr(scipy, "__version__", "unknown")),
         Path(scipy.__file__).resolve().parent),
    ):
        package_sha256, package_file_count = v2_package_tree_hash(package_root)
        libraries[key] = {
            "version": version,
            "module": _path_fingerprint(getattr(module, "__file__", None)),
            "package_root": str(package_root),
            "package_tree_sha256": package_sha256,
            "package_file_count": package_file_count,
        }

    # Include all loaded native modules under the canonical package trees plus
    # the two Maestro binaries, without hashing unrelated imported packages.
    native_files: dict[str, dict[str, str] | None] = {}
    native_suffixes = (".so", ".pyd", ".dll", ".dylib")
    for module_name, module in sorted(sys.modules.items()):
        if not (module_name == "maestro" or module_name.startswith(("maestro.", "numpy.", "scipy."))):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path and str(module_path).lower().endswith(native_suffixes):
            native_files[str(Path(module_path).resolve())] = _path_fingerprint(module_path)
    maestro_root = Path(maestro.__file__).resolve().parent
    for binary in (maestro_root / "libmaestro.so",):
        native_files[str(binary.resolve())] = _path_fingerprint(binary)
    try:
        extension_spec = importlib.util.find_spec("maestro.maestro")
        if extension_spec is not None and extension_spec.origin:
            extension_path = Path(extension_spec.origin).resolve()
            native_files[str(extension_path)] = _path_fingerprint(extension_path)
    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    try:
        from threadpoolctl import threadpool_info
        threadpools: Any = []
        for pool in threadpool_info():
            item = {key: pool.get(key) for key in (
                "user_api", "internal_api", "num_threads", "prefix", "version",
                "threading_layer", "architecture", "filepath",
            ) if pool.get(key) is not None}
            if item.get("filepath"):
                item["library"] = _path_fingerprint(item["filepath"])
            threadpools.append(item)
        threadpools.sort(key=canonical_hash)
    except ImportError:
        threadpools = "unavailable:threadpoolctl_not_installed"
    except Exception as exc:  # pragma: no cover - platform/library-specific
        threadpools = f"unavailable:{type(exc).__name__}"

    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    identity = {
        "schema": "maestro_worker_execution_context_v1",
        "canonical_imports": list(CANONICAL_WORKER_IMPORTS),
        "libraries": libraries,
        "loaded_native_files": native_files,
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "executable_sha256": sha256_file(Path(sys.executable).resolve()),
            "version": sys.version,
        },
        "platform": platform.platform(),
        "resolved_config": _context_value(config),
        "resolved_config_sha256": canonical_hash(_context_value(config)),
        "backend_threading_policy": _context_value(backend_threading_policy),
        "backend_threading_policy_sha256": canonical_hash(_context_value(backend_threading_policy)),
        "cpu": {"machine": platform.machine(), "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(), "affinity": affinity},
        "native_thread_environment": {name: os.environ.get(name, "") for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
        )},
        "native_threadpools": threadpools,
    }
    environment_sha256 = canonical_hash(worker_environment_identity({"identity": identity}))
    return {
        "context_owner": "worker",
        "identity": identity,
        "identity_sha256": canonical_hash(identity),
        "resolved_config_sha256": identity["resolved_config_sha256"],
        "backend_threading_policy_sha256": identity["backend_threading_policy_sha256"],
        "worker_environment_sha256": environment_sha256,
        "observation": {"pid": os.getpid(), "captured_monotonic_ns": time.monotonic_ns()},
    }


def _installed_version(distribution: str, module: Any) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unreported"))


def collect_parent_context(config: Any = None,
                           backend_threading_policy: Any = None) -> dict[str, Any]:
    """Capture parent-only imports as separate telemetry, never worker identity."""
    context = collect_worker_context(config, backend_threading_policy)
    context["context_owner"] = "parent"
    context["observation"] = {"pid": os.getpid(), "captured_monotonic_ns": time.monotonic_ns()}
    return context


def _context_differences(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in actual:
                differences.append(path)
            else:
                differences.extend(_context_differences(expected[key], actual[key], path))
        return differences
    if expected != actual:
        return [prefix]
    return []


def worker_environment_identity(context: dict[str, Any]) -> dict[str, Any]:
    """Return the common static worker environment, excluding config policy."""
    identity = context.get("identity")
    if not isinstance(identity, dict):
        raise V2Unavailable("worker_context_identity_missing")
    fields = (
        "schema", "canonical_imports", "libraries", "loaded_native_files",
        "python", "platform", "cpu", "native_thread_environment",
    )
    missing = [field for field in fields if field not in identity]
    if missing:
        raise V2Unavailable(f"worker_environment_identity_missing_fields:{missing}")
    return {field: identity[field] for field in fields}


def validate_worker_environment_consistency(
    contexts: dict[str, dict[str, Any]], actual: dict[str, Any],
) -> dict[str, Any]:
    """Require one static package/platform/CPU environment across configs."""
    if actual.get("context_owner") != "worker":
        raise V2Unavailable("worker_context_owner_mismatch")
    actual_hash = canonical_hash(worker_environment_identity(actual))
    if actual.get("worker_environment_sha256") != actual_hash:
        raise V2Unavailable("worker_environment_hash_invalid")
    for previous in contexts.values():
        if previous.get("context_owner") != "worker":
            raise V2Unavailable("worker_context_owner_mismatch")
        previous_hash = canonical_hash(worker_environment_identity(previous))
        if previous.get("worker_environment_sha256") != previous_hash:
            raise V2Unavailable("stored_worker_environment_hash_invalid")
        if previous_hash != actual_hash:
            raise V2Unavailable("worker_environment_drift_across_configurations")
    return {"status": "pass", "worker_environment_sha256": actual_hash}


def validate_worker_context(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """Reject real worker identity drift; never compare parent telemetry as identity."""
    if expected.get("context_owner") != "worker" or actual.get("context_owner") != "worker":
        raise V2Unavailable("worker_context_owner_mismatch:parent_telemetry_is_not_execution_identity")
    expected_identity = expected.get("identity")
    actual_identity = actual.get("identity")
    if not isinstance(expected_identity, dict) or not isinstance(actual_identity, dict):
        raise V2Unavailable("worker_context_identity_missing")
    if expected.get("identity_sha256") != canonical_hash(expected_identity):
        raise V2Unavailable("expected_worker_context_hash_invalid")
    if actual.get("identity_sha256") != canonical_hash(actual_identity):
        raise V2Unavailable("actual_worker_context_hash_invalid")
    for context, identity, label in (
        (expected, expected_identity, "expected"), (actual, actual_identity, "actual"),
    ):
        if context.get("resolved_config_sha256") != identity.get("resolved_config_sha256"):
            raise V2Unavailable(f"{label}_resolved_config_hash_invalid")
        if context.get("backend_threading_policy_sha256") != identity.get("backend_threading_policy_sha256"):
            raise V2Unavailable(f"{label}_backend_threading_policy_hash_invalid")
        if context.get("worker_environment_sha256") != canonical_hash(worker_environment_identity(context)):
            raise V2Unavailable(f"{label}_worker_environment_hash_invalid")
    differences = _context_differences(expected_identity, actual_identity)
    if differences:
        raise V2Unavailable(f"worker_context_drift:{differences[:20]}")
    return {"status": "pass", "identity_sha256": expected["identity_sha256"], "differences": []}


def worker_context_config_key(resolved_config: Any, backend_threading_policy: Any) -> str:
    return canonical_hash({
        "resolved_config_sha256": canonical_hash(_context_value(resolved_config)),
        "backend_threading_policy_sha256": canonical_hash(_context_value(backend_threading_policy)),
    })


def _worker_context_config_key(context: dict[str, Any]) -> str:
    return canonical_hash({
        "resolved_config_sha256": context.get("resolved_config_sha256", ""),
        "backend_threading_policy_sha256": context.get("backend_threading_policy_sha256", ""),
    })


def _restore_worker_contexts(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for row in records:
        if row.get("stage") != "worker_context":
            continue
        context = row.get("worker_context")
        if not isinstance(context, dict):
            raise V2Unavailable("worker_context_ledger_entry_missing_context")
        validate_worker_environment_consistency(contexts, context)
        key = _worker_context_config_key(context)
        previous = contexts.get(key)
        if previous is not None:
            validate_worker_context(previous, context)
        else:
            contexts[key] = context
    return contexts


def _record_worker_context(context: Any, contexts: dict[str, dict[str, Any]],
                            ledger_path: Path, run_identity_sha256: str,
                            by_id: dict[str, dict[str, Any]],
                            related_record_id: str = "") -> dict[str, str]:
    if not isinstance(context, dict):
        return {
            "worker_context_sha256": "", "worker_pid": "",
            "worker_environment_sha256": "", "resolved_config_sha256": "",
            "backend_threading_policy_sha256": "",
        }
    if context.get("context_owner") != "worker":
        raise V2Unavailable("worker_context_owner_mismatch")
    validate_worker_environment_consistency(contexts, context)
    key = _worker_context_config_key(context)
    previous = contexts.get(key)
    if previous is not None:
        validate_worker_context(previous, context)
    else:
        contexts[key] = context
        record_id = v2_canonical_hash({
            "stage": "worker_context", "run_identity_sha256": run_identity_sha256,
            "resolved_config_key": key,
        })
        if record_id not in by_id:
            record = {
                "record_id": record_id, "stage": "worker_context",
                "resolved_config_key": key, "worker_context": context,
                "related_record_id": related_record_id,
                "run_identity_sha256": run_identity_sha256,
            }
            _v2_append_jsonl(ledger_path, record)
            by_id[record_id] = record
    return {
        "worker_context_sha256": str(context.get("identity_sha256", "")),
        "worker_pid": str(context.get("observation", {}).get("pid", "")),
        "worker_environment_sha256": str(context.get("worker_environment_sha256", "")),
        "resolved_config_sha256": str(context.get("resolved_config_sha256", "")),
        "backend_threading_policy_sha256": str(context.get("backend_threading_policy_sha256", "")),
    }


def _expected_worker_context(configs: dict[str, dict[str, Any]] | None,
                             candidate: str,
                             contexts: dict[str, dict[str, Any]],
                             backend_threading_policy: Any = None) -> dict[str, Any] | None:
    if not configs:
        return None
    resolved = configs.get(candidate, {}).get("resolved")
    if not isinstance(resolved, dict) or not resolved:
        return None
    return contexts.get(worker_context_config_key(resolved, backend_threading_policy))


def exact_qasm_fold(qasm_sha256: str, split_id: str) -> int:
    material = f"maestro-common-q16-v1|{qasm_sha256}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 5


def ordered_member_digest(rows: list[dict[str, str]]) -> str:
    material = "".join(
        f"{row['qasm_sha256']}  {row['basename']}\n"
        for row in sorted(rows, key=lambda item: item["basename"])
    ).encode("utf-8")
    return sha256_bytes(material)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_maestro_parser():
    try:
        import maestro
    except Exception as exc:  # pragma: no cover - environment-specific diagnostic
        raise RuntimeError(f"Maestro parser unavailable in this Python: {type(exc).__name__}: {exc}") from exc
    return maestro


def maestro_version(maestro: Any) -> str:
    try:
        return importlib.metadata.version("maestro")
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(maestro, "__version__", "package_version_unreported"))


def verify_frozen_inputs(manifest: dict[str, Any], calibration_manifest_path: Path,
                         calibration_raw_path: Path, panel_csv_path: Path,
                         fixture_manifest_path: Path) -> None:
    calibration = manifest["calibration"]
    pins = (
        (calibration_manifest_path, calibration["manifest_sha256"]),
        (calibration_raw_path, calibration["raw_records_sha256"]),
        (panel_csv_path, manifest["panel"]["manifest_csv_sha256"]),
        (fixture_manifest_path, manifest["panel"]["qasm_fixture_manifest_sha256"]),
    )
    for path, expected in pins:
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input hash mismatch for {path}: {actual} != {expected}")


def materialize_rows(manifest: dict[str, Any], panel_csv_path: Path, qasm_root: Path,
                     fixture_manifest_path: Path, maestro: Any) -> tuple[list[dict[str, Any]], dict[str, CircuitOperationFeatures]]:
    expected = manifest["panel"]
    panel_rows = _csv_rows(panel_csv_path)
    if len(panel_rows) != int(expected["rows"]):
        raise ValueError(f"panel row count {len(panel_rows)} != frozen {expected['rows']}")
    if len({row["panel_member_id"] for row in panel_rows}) != len(panel_rows):
        raise ValueError("duplicate panel_member_id in frozen panel")
    if len({row["qasm_sha256"] for row in panel_rows}) != int(expected["unique_qasm_hashes"]):
        raise ValueError("unique QASM hash count does not match the frozen panel")
    actual_digest = ordered_member_digest(panel_rows)
    if actual_digest != expected["ordered_member_digest"]:
        raise ValueError(f"ordered member digest mismatch: {actual_digest}")
    fixture_payload = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
    fixture_members = {item["basename"]: item["sha256"] for item in fixture_payload["members"]}
    if len(fixture_members) != len(fixture_payload["members"]):
        raise ValueError("duplicate basename in local QASM fixture manifest")
    if len(fixture_members) != len(panel_rows):
        raise ValueError("local QASM fixture does not contain all 204 panel members")
    panel_hashes = {row["basename"]: row["qasm_sha256"] for row in panel_rows}
    if fixture_members != panel_hashes:
        raise ValueError("local QASM fixture identities differ from the frozen panel manifest")

    output: list[dict[str, Any]] = []
    features_by_member: dict[str, CircuitOperationFeatures] = {}
    for row in panel_rows:
        path = qasm_root / row["basename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        qasm_bytes = path.read_bytes()
        qasm_hash = sha256_bytes(qasm_bytes)
        if qasm_hash != row["qasm_sha256"]:
            raise ValueError(f"QASM SHA-256 mismatch for {row['panel_member_id']}")
        qasm = qasm_bytes.decode("utf-8")
        member_features = parse_qasm_operation_features(qasm, int(row["width_qubits"]))
        parser_status = "pass"
        try:
            circuit = maestro.QasmToCirc().parse_and_translate(qasm)
            parser_n = int(circuit.num_qubits)
            if parser_n != int(row["width_qubits"]):
                parser_status = "fail_width_mismatch"
                member_features = CircuitOperationFeatures(
                    n_qubits=member_features.n_qubits,
                    n_one_qubit_operations=member_features.n_one_qubit_operations,
                    n_cx_operations=member_features.n_cx_operations,
                    measured_qubits=member_features.measured_qubits,
                    unsupported_operations={**member_features.unsupported_operations,
                                           "maestro_parser_width_mismatch": 1},
                    gate_counts=member_features.gate_counts,
                )
        except Exception as exc:
            parser_status = f"fail:{type(exc).__name__}:{exc}"
            member_features = CircuitOperationFeatures(
                n_qubits=member_features.n_qubits,
                n_one_qubit_operations=member_features.n_one_qubit_operations,
                n_cx_operations=member_features.n_cx_operations,
                measured_qubits=member_features.measured_qubits,
                unsupported_operations={**member_features.unsupported_operations,
                                       "maestro_parser_rejected": 1},
                gate_counts=member_features.gate_counts,
            )
        features_by_member[row["panel_member_id"]] = member_features
        fold = exact_qasm_fold(qasm_hash, manifest["outer_split"]["split_id"])
        feature_payload = {
            "parser": "maestro.QasmToCirc.parse_and_translate",
            "maestro_num_qubits": int(row["width_qubits"]),
            "native_qasm_one_qubit_count": member_features.n_one_qubit_operations,
            "native_qasm_cx_count": member_features.n_cx_operations,
            "measurement_qubit_count": member_features.measured_qubits,
            "unsupported_operations": member_features.unsupported_operations,
            "gate_counts": member_features.gate_counts,
        }
        output.append({
            "panel_id": row["panel_id"],
            "panel_member_id": row["panel_member_id"],
            "basename": row["basename"],
            "source_path": row["source_path"],
            "qasm_sha256": qasm_hash,
            "fold_id": fold,
            "split_id": manifest["outer_split"]["split_id"],
            "family": row["family"],
            "width_qubits": int(row["width_qubits"]),
            "stratum": row["stratum"],
            "hash_group_id": row["hash_group_id"],
            "maestro_parser_status": parser_status,
            "maestro_parser_num_qubits": int(row["width_qubits"]),
            "n_one_qubit_operations": member_features.n_one_qubit_operations,
            "n_cx_operations": member_features.n_cx_operations,
            "n_measured_qubits": member_features.measured_qubits,
            "unsupported_operations_json": json.dumps(member_features.unsupported_operations, sort_keys=True),
            "gate_counts_json": json.dumps(member_features.gate_counts, sort_keys=True),
            "candidate_ir_feature_hash": canonical_hash(feature_payload),
            "candidate_ir_feature_source": "Maestro-parser-accepted native QASM instruction counts; binding exposes width but not an operation-list API",
        })
    return output, features_by_member


def _find_prediction_domain_error(n: int) -> str | None:
    if n < 8:
        return "out_of_grid:width_below_frozen_minimum_8"
    if n > 16:
        return "out_of_grid:width_above_common_panel_core_prediction_ceiling_16"
    return None


def materialize_predictions(manifest: dict[str, Any], row_manifest: list[dict[str, Any]],
                            features: dict[str, CircuitOperationFeatures],
                            calibration_by_fold: dict[int, FittedCalibration]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    measurement = manifest["measurement"]
    quality = manifest["quality_policy"]
    for row in row_manifest:
        member_id = row["panel_member_id"]
        feature = features[member_id]
        fold = int(row["fold_id"])
        calibration = calibration_by_fold[fold]
        for candidate_spec in manifest["candidates"]:
            candidate_id = candidate_spec["candidate_id"]
            candidate_model_id = candidate_spec["calibration_candidate"]
            chi = candidate_spec.get("configured_chi")
            config_hash = canonical_hash(candidate_spec)
            pred_id = canonical_hash({
                "panel_member_id": member_id,
                "qasm_sha256": row["qasm_sha256"],
                "fold_id": fold,
                "candidate_id": candidate_id,
                "candidate_config_sha256": config_hash,
                "calibration_raw_sha256": calibration.calibration_raw_sha256,
                "clock_id": manifest["measurement"]["selected_clock_id"],
            })
            gate_id = candidate_spec.get("quality_gate_id", "not_required_exact_statevector")
            quality_ref_id = canonical_hash({
                "panel_member_id": member_id,
                "qasm_sha256": row["qasm_sha256"],
                "candidate_config_sha256": config_hash,
                "quality_gate_id": gate_id,
                "reference_shots": quality.get("reference_shots", 0),
            })
            predicted: CandidateRuntimePrediction | None = None
            prediction_status = "predicted"
            reason = ""
            domain_error = _find_prediction_domain_error(int(row["width_qubits"]))
            if domain_error:
                prediction_status = "out_of_grid"
                reason = domain_error
            else:
                try:
                    predicted = predict_candidate_runtime(
                        calibration=calibration,
                        features=feature,
                        candidate=candidate_model_id,
                        shots=int(candidate_spec["shots"]),
                        configured_chi=None if chi is None else int(chi),
                    )
                except PredictionUnavailable as exc:
                    prediction_status = "unavailable"
                    reason = str(exc)
            quality_status = "not_required" if candidate_id == "qcsim_statevector_cpu" else "quality_pending_not_eligible"
            eligible = candidate_id == "qcsim_statevector_cpu" and prediction_status == "predicted"
            prediction_rows.append({
                "prediction_id": pred_id,
                "panel_id": row["panel_id"],
                "panel_member_id": member_id,
                "qasm_sha256": row["qasm_sha256"],
                "fold_id": fold,
                "split_id": row["split_id"],
                "family": row["family"],
                "stratum": row["stratum"],
                "width_qubits": row["width_qubits"],
                "candidate_id": candidate_id,
                "candidate_config_sha256": config_hash,
                "configured_chi": "" if chi is None else chi,
                "shots": candidate_spec["shots"],
                "selected_clock_id": manifest["measurement"]["selected_clock_id"],
                "selected_clock_field": manifest["measurement"]["selected_clock_field"],
                "calibration_clock_id": calibration.clock_id,
                "calibration_raw_sha256": calibration.calibration_raw_sha256,
                "calibration_manifest_id": calibration.calibration_manifest_id,
                "calibration_fit_fold": fold,
                "quality_gate_id": gate_id,
                "quality_reference_id": quality_ref_id,
                "quality_gate_status": quality_status,
                "eligible_for_selector": eligible,
                "prediction_status": prediction_status,
                "unavailable_reason": reason,
                "predicted_one_qubit_seconds": "" if predicted is None else predicted.one_qubit_seconds,
                "predicted_cx_seconds": "" if predicted is None else predicted.cx_seconds,
                "predicted_sampling_seconds": "" if predicted is None else predicted.sampling_seconds,
                "predicted_warm_runtime_seconds": "" if predicted is None else predicted.predicted_warm_seconds,
                "prediction_role": "candidate_runtime_prediction_only_no_selection_or_oracle",
            })
            if prediction_status == "predicted":
                if candidate_id == "qcsim_statevector_cpu":
                    attempt_status = "eligible_for_warm_measurement"
                else:
                    attempt_status = "quality_gate_required_before_warm_measurement"
            else:
                attempt_status = f"not_attempted_{prediction_status}"
            plan_rows.append({
                "attempt_id": canonical_hash({"prediction_id": pred_id, "attempt": "warm_label"}),
                "panel_id": row["panel_id"],
                "panel_member_id": member_id,
                "qasm_sha256": row["qasm_sha256"],
                "candidate_ir_feature_hash": row["candidate_ir_feature_hash"],
                "fold_id": fold,
                "split_id": row["split_id"],
                "family": row["family"],
                "stratum": row["stratum"],
                "width_qubits": row["width_qubits"],
                "candidate_id": candidate_id,
                "candidate_config_sha256": config_hash,
                "simulator_type": candidate_spec["simulator_type"],
                "simulation_type": candidate_spec["simulation_type"],
                "configured_chi": "" if chi is None else chi,
                "shots": candidate_spec["shots"],
                "seed": candidate_spec["seed"],
                "selected_clock_id": measurement["selected_clock_id"],
                "selected_clock_field": measurement["selected_clock_field"],
                "quality_gate_id": gate_id,
                "quality_reference_id": quality_ref_id,
                "quality_threshold": quality.get("threshold", ""),
                "quality_reference_shots": quality.get("reference_shots", ""),
                "prediction_id": pred_id,
                "prediction_status": prediction_status,
                "attempt_plan_status": attempt_status,
                "sessions": measurement["sessions"],
                "untimed_warmups_per_session": measurement["untimed_warmups_per_session"],
                "warm_repetitions_per_session": measurement["warm_repetitions_per_session"],
                "cell_timeout_seconds": measurement["cell_timeout_seconds"],
            })
    return prediction_rows, plan_rows


def v2_candidate_specs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the locked v2 candidate pair without changing the v1 manifest."""
    implementation = manifest["implementation"]
    settings = manifest["panel_candidate_configuration"]
    result: list[dict[str, Any]] = []
    for candidate in implementation["candidates"]:
        if candidate not in V2_CANDIDATE_IDS:
            raise ValueError(f"unexpected v2 candidate: {candidate}")
        spec = {
            "candidate_id": candidate,
            "candidate": candidate,
            "seed": int(settings["seed"]),
            "shots": int(settings["shots"]),
            "configured_chi": (None if candidate == "statevector"
                                else int(settings["mps_configured_chi"])),
            "singular_value_threshold": (None if candidate == "statevector"
                                         else float(settings["mps_singular_value_threshold"])),
            "quality_gate_id": ("not_required_exact_statevector" if candidate == "statevector"
                                else str(manifest["quality_policy"]["quality_gate_id"])),
        }
        result.append(spec)
    if {row["candidate"] for row in result} != set(V2_CANDIDATE_IDS):
        raise ValueError("v2 candidate pair must include statevector and mps_fixed_chi")
    return result


def v2_verify_inputs(manifest: dict[str, Any], manifest_path: Path,
                     panel_csv_path: Path | None = None) -> tuple[Path, Path, list[dict[str, str]], dict[str, Any]]:
    """Verify the immutable source panel and fixture before any derived work."""
    if manifest.get("manifest_id") != V2_MANIFEST_ID:
        raise ValueError("v2 runner requires the locked completion manifest")
    panel = manifest["panel"]
    panel_path = panel_csv_path or ROOT / panel["manifest_csv"]
    fixture_path = ROOT / panel["source_fixture_manifest"]
    for path, expected, label in (
        (manifest_path, None, "v2 manifest"),
        (panel_path, panel["manifest_csv_sha256"], "panel csv"),
        (fixture_path, panel["source_fixture_manifest_sha256"], "fixture manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise ValueError(f"{label} sha256 mismatch: {actual} != {expected}")
    rows = _csv_rows(panel_path)
    if len(rows) != int(panel["assigned_members"]):
        raise ValueError(f"panel rows {len(rows)} != frozen {panel['assigned_members']}")
    if len({row["panel_member_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate panel_member_id in frozen v2 panel")
    if len({row["qasm_sha256"] for row in rows}) != int(panel["source_qasm_hashes"]):
        raise ValueError("unique source QASM hashes do not match v2 contract")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture_members = {item["basename"]: item["sha256"] for item in fixture["members"]}
    if len(fixture_members) != len(fixture["members"]):
        raise ValueError("duplicate fixture basename")
    panel_members = {row["basename"]: row["qasm_sha256"] for row in rows}
    if fixture_members != panel_members:
        raise ValueError("fixture QASM identities differ from the frozen v2 panel")
    return panel_path, fixture_path, rows, fixture


def materialize_v2_panel(manifest: dict[str, Any], panel_rows: list[dict[str, str]],
                         qasm_root: Path, maestro: Any,
                         normalizer=None) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                  dict[str, CircuitOperationFeatures]]:
    """Make a deterministic normalized-QASM sidecar, retaining every member."""
    if normalizer is None:
        normalizer = v2_normalize_qasm
    normalization = manifest["normalization"]
    row_manifest: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []
    features: dict[str, CircuitOperationFeatures] = {}
    for source_row in panel_rows:
        member_id = source_row["panel_member_id"]
        basename = source_row["basename"]
        source_path = qasm_root / basename
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_bytes = source_path.read_bytes()
        source_hash = sha256_bytes(source_bytes)
        if source_hash != source_row["qasm_sha256"]:
            raise ValueError(f"source QASM hash mismatch for {member_id}")
        width = int(source_row["width_qubits"])
        fold = exact_qasm_fold(source_hash, V2_SPLIT_ID)
        try:
            normalized = normalizer(source_bytes.decode("utf-8"), width, normalization, maestro)
            if normalized.status != "ok":
                raise V2Unavailable(f"normalizer_status:{normalized.status}:{normalized.reason}")
            feature_counts = {str(key): int(value) for key, value in normalized.features.items()}
            feature = CircuitOperationFeatures(
                n_qubits=width,
                n_one_qubit_operations=feature_counts.get("rx", 0) + feature_counts.get("rz", 0),
                n_cx_operations=feature_counts.get("cx", 0),
                measured_qubits=width,
                unsupported_operations={},
                gate_counts=feature_counts,
            )
            features[member_id] = feature
            norm_status, reason = "ok", ""
            normalized_qasm = normalized.normalized_qasm
            normalized_hash = normalized.normalized_qasm_sha256
            unitary_prefix_qasm = normalized.unitary_prefix_qasm
            source_unitary_prefix_qasm = normalized.source_unitary_prefix_qasm
            provenance = dict(normalized.provenance)
            provenance_hash = canonical_hash(provenance)
        except Exception as exc:
            # A semantic/parser failure is an unavailable member, not a reason
            # to shrink the frozen denominator or drop its candidate attempts.
            norm_status = "unavailable"
            reason = f"{type(exc).__name__}:{exc}"
            normalized_qasm = ""
            normalized_hash = ""
            unitary_prefix_qasm = ""
            source_unitary_prefix_qasm = ""
            feature_counts = {}
            provenance = {}
            provenance_hash = ""
        row = {
            "panel_id": source_row["panel_id"],
            "panel_member_id": member_id,
            "basename": basename,
            "source_path": source_row.get("source_path", ""),
            "qasm_sha256": source_hash,
            "hash_group_id": source_row.get("hash_group_id", ""),
            "fold_id": fold,
            "split_id": V2_SPLIT_ID,
            "family": source_row.get("family", ""),
            "width_qubits": width,
            "stratum": source_row.get("stratum", ""),
            "normalization_status": norm_status,
            "unavailable_reason": reason,
            "normalized_qasm_sha256": normalized_hash,
            "normalization_provenance_sha256": provenance_hash,
            "candidate_ir_feature_hash": canonical_hash(feature_counts),
        }
        row_manifest.append(row)
        sidecar.append({
            **row,
            "normalized_qasm": normalized_qasm,
            "unitary_prefix_qasm": unitary_prefix_qasm,
            "source_unitary_prefix_qasm": source_unitary_prefix_qasm,
            "normalized_features": feature_counts,
            "normalization_provenance": provenance,
        })
    if len(row_manifest) != int(manifest["panel"]["assigned_members"]):
        raise ValueError("normalization changed the frozen member denominator")
    folds_by_hash: dict[str, set[int]] = {}
    for row in row_manifest:
        folds_by_hash.setdefault(row["qasm_sha256"], set()).add(int(row["fold_id"]))
    if any(len(folds) != 1 for folds in folds_by_hash.values()):
        raise ValueError("identical source QASM hashes do not share a fold")
    return row_manifest, sidecar, features


def _v2_checkpoint_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    return str(value)


def _v2_artifact_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _v2_load_calibration(
    manifest: dict[str, Any], manifest_path: Path,
    configs: dict[str, dict[str, Any]], environment: dict[str, Any],
) -> tuple[FittedCalibration | None, str, str, dict[str, Any] | None, str]:
    """Load only a complete, row-identical, code/config/environment-pinned R3 fit."""
    cal_dir = ROOT / manifest["outputs"]["calibration"]
    run_path = cal_dir / "run_manifest.json"
    if not run_path.is_file():
        return None, "calibration_not_available", "", None, ""
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run_hash = sha256_file(run_path)
        expected_manifest_hash = sha256_file(manifest_path)
        if run.get("manifest_id") != V2_MANIFEST_ID:
            raise V2Unavailable("calibration_manifest_id_mismatch")
        if run.get("status") not in {"complete", "complete_with_unavailable_knots"}:
            raise V2Unavailable(f"calibration_status_not_complete:{run.get('status', 'missing')}")
        if int(run.get("completed_cells", -1)) != int(run.get("planned_cells", -2)):
            raise V2Unavailable("calibration_completed_cell_count_mismatch")
        if int(run.get("planned_cells", -1)) != int(manifest["synthetic_corpus"]["planned_cells"]):
            raise V2Unavailable("calibration_planned_cell_count_mismatch")

        identity = run.get("run_identity", {})
        if identity.get("manifest_sha256") != expected_manifest_hash:
            raise V2Unavailable("calibration_manifest_sha256_mismatch")
        if (identity.get("selected_clock_id") != V2_CLOCK_ID
                or identity.get("selected_clock_field") != V2_CLOCK_FIELD
                or run.get("clock_id") != V2_CLOCK_ID
                or run.get("selected_clock_field") != V2_CLOCK_FIELD):
            raise V2Unavailable("calibration_selected_clock_mismatch")

        # The calibration and panel must use the same Python/native simulator,
        # Qiskit, CPU affinity, native thread settings, and resource snapshot.
        for key in (
            "python_executable", "python_executable_sha256", "python_version", "platform",
            "maestro_module", "maestro_package_tree_sha256", "maestro_package_file_count",
            "qiskit_module", "qiskit_version", "qiskit_package_tree_sha256",
            "qiskit_package_file_count", "cpu_affinity", "native_thread_environment",
            "resource_profile",
        ):
            if identity.get(key) != environment.get(key):
                raise V2Unavailable(f"calibration_environment_pin_mismatch:{key}")

        calibration_runner = ROOT / "benchmark_v1/scripts/run_maestro_component_calibration.py"
        helper = ROOT / "benchmark_v1/qre_benchmark/maestro_component_v2.py"
        predictor = ROOT / "benchmark_v1/qre_benchmark/maestro_component_predictor.py"
        current_calibration_code = {
            str(path.resolve().relative_to(ROOT)): sha256_file(path)
            for path in (calibration_runner, helper, predictor)
        }
        for path, digest in current_calibration_code.items():
            if identity.get("code_hashes", {}).get(path) != digest:
                raise V2Unavailable(f"calibration_code_hash_mismatch:{path}")

        resolved_calibration_configs = identity.get("configs", {})
        expected_config_keys = {"statevector:None": "statevector", "mps_fixed_chi:32": "mps_fixed_chi"}
        for calibration_key, candidate in expected_config_keys.items():
            expected_config_sha = str(configs.get(candidate, {}).get("sha256", ""))
            actual_config_sha = str(resolved_calibration_configs.get(calibration_key, {}).get("sha256", ""))
            if not expected_config_sha or actual_config_sha != expected_config_sha:
                raise V2Unavailable(f"calibration_resolved_config_mismatch:{calibration_key}")

        raw_meta = run.get("raw_records", {})
        raw_path = _v2_artifact_path(raw_meta.get("path", ""), cal_dir)
        checkpoint_path = _v2_artifact_path(run.get("checkpoint_path", ""), cal_dir)
        fit_path = cal_dir / "fitted_calibration.json"
        if not raw_path.is_file() or not checkpoint_path.is_file() or not fit_path.is_file():
            raise V2Unavailable("calibration_raw_checkpoint_or_fit_missing")
        raw_hash = sha256_file(raw_path)
        if raw_hash != raw_meta.get("sha256"):
            raise V2Unavailable("calibration_raw_records_sha256_mismatch")
        if run.get("fit_summary", {}).get("calibration_raw_sha256") != raw_hash:
            raise V2Unavailable("calibration_fit_raw_sha256_mismatch")
        if raw_meta.get("rows") != run.get("checkpoint_attempt_rows"):
            raise V2Unavailable("calibration_raw_checkpoint_row_count_mismatch")

        csv_rows = _csv_rows(raw_path)
        if len(csv_rows) != int(raw_meta.get("rows", -1)):
            raise V2Unavailable("calibration_raw_csv_row_count_mismatch")
        checkpoint_hash = sha256_file(checkpoint_path)
        if checkpoint_hash != run.get("checkpoint_sha256"):
            raise V2Unavailable("calibration_checkpoint_sha256_mismatch")
        if run.get("checkpoint_format") != "maestro_calibration_checkpoint_jsonl_v1":
            raise V2Unavailable("calibration_checkpoint_format_mismatch")
        checkpoint_rows: list[dict[str, str]] = []
        with checkpoint_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.endswith("\n"):
                    raise V2Unavailable(f"calibration_checkpoint_incomplete_line:{line_number}")
                cell = json.loads(line)
                if cell.get("run_identity_sha256") != run.get("run_identity_sha256"):
                    raise V2Unavailable(f"calibration_checkpoint_identity_mismatch:{line_number}")
                for row in cell.get("rows", []):
                    checkpoint_rows.append({key: _v2_checkpoint_value(value) for key, value in row.items()})
        if checkpoint_rows != csv_rows:
            raise V2Unavailable("calibration_checkpoint_raw_row_mismatch")
        attempt_ids = [str(row.get("attempt_id", "")) for row in csv_rows]
        if not all(attempt_ids) or len(set(attempt_ids)) != len(attempt_ids):
            raise V2Unavailable("calibration_duplicate_or_missing_attempt_id")
        attempt_ids_hash = v2_canonical_hash(sorted(attempt_ids))
        if (attempt_ids_hash != run.get("checkpoint_attempt_ids_sha256")
                or attempt_ids_hash != raw_meta.get("attempt_ids_sha256")):
            raise V2Unavailable("calibration_attempt_id_hash_mismatch")

        fit_payload = json.loads(fit_path.read_text(encoding="utf-8"))
        if (fit_payload.get("calibration_manifest_id") != V2_MANIFEST_ID
                or fit_payload.get("calibration_raw_sha256") != raw_hash
                or fit_payload.get("fit_version") != "process_isolated_affine_v2"):
            raise V2Unavailable("calibration_fitted_artifact_identity_mismatch")
        fitted = v2_fit_calibration_rows(csv_rows, manifest, raw_hash)
        return fitted, "", raw_hash, run, run_hash
    except Exception as exc:
        reason = str(exc) if isinstance(exc, V2Unavailable) else f"{type(exc).__name__}:{exc}"
        return None, f"calibration_unavailable:{reason}", "", None, ""


def v2_materialize_predictions(manifest: dict[str, Any], row_manifest: list[dict[str, Any]],
                               features: dict[str, CircuitOperationFeatures],
                               configs: dict[str, dict[str, Any]],
                               calibration: FittedCalibration | None,
                               calibration_reason: str,
                               manifest_sha256: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return all assigned attempts with prediction/execution states independent."""
    predictions: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    specs = v2_candidate_specs(manifest)
    max_width = max(int(v) for v in manifest["synthetic_corpus"]["statevector_widths"])
    for row in row_manifest:
        member_id = row["panel_member_id"]
        width = int(row["width_qubits"])
        member_features = features.get(member_id)
        n_one_qubit_operations = 0 if member_features is None else int(
            member_features.n_one_qubit_operations
        )
        n_cx_operations = 0 if member_features is None else int(member_features.n_cx_operations)
        for spec in specs:
            candidate = spec["candidate"]
            config = configs.get(candidate, {})
            config_sha = str(config.get("sha256", ""))
            prediction_reason = ""
            prediction_status = "available"
            predicted: CandidateRuntimePredictionV2 | None = None
            execution_reason = ""
            if row["normalization_status"] != "ok":
                execution_eligibility = "unavailable"
                execution_reason = f"normalization_unavailable:{row['unavailable_reason']}"
            elif config.get("status") != "ok":
                execution_eligibility = "unavailable"
                execution_reason = str(config.get("reason", "candidate_config_unavailable"))
            else:
                execution_eligibility = "eligible"

            if row["normalization_status"] != "ok":
                prediction_status = "unavailable"
                prediction_reason = f"normalization_unavailable:{row['unavailable_reason']}"
            elif width > max_width:
                prediction_status = "out_of_grid"
                prediction_reason = f"out_of_grid:width_above_{max_width}"
            elif config.get("status") != "ok":
                prediction_status = "unavailable"
                prediction_reason = str(config.get("reason", "candidate_config_unavailable"))
            elif calibration is None:
                prediction_status = "unavailable"
                prediction_reason = calibration_reason or "calibration_not_available"
            else:
                try:
                    predicted = predict_candidate_runtime_v2(
                        calibration=calibration,
                        features=features[member_id],
                        candidate=candidate,
                        shots=int(spec["shots"]),
                        configured_chi=spec["configured_chi"],
                    )
                except PredictionUnavailable as exc:
                    prediction_status = "unavailable"
                    prediction_reason = str(exc)
            attempt_id = canonical_hash({
                "manifest_sha256": manifest_sha256,
                "panel_member_id": member_id,
                "qasm_sha256": row["qasm_sha256"],
                "fold_id": int(row["fold_id"]),
                "candidate": candidate,
                "candidate_config_sha256": config_sha or canonical_hash(spec),
                "calibration_raw_sha256": "" if calibration is None else calibration.calibration_raw_sha256,
                "selected_clock_id": V2_CLOCK_ID,
                "selected_clock_field": V2_CLOCK_FIELD,
            })
            gate_id = str(spec["quality_gate_id"])
            quality_status = (
                "not_required" if candidate == "statevector"
                else ("not_run" if execution_eligibility == "eligible" else "quality_unavailable")
            )
            timing_status = "not_started"
            timing_reason = execution_reason
            metric_eligibility = False
            metric_ineligibility_reason = "timing_not_yet_observed"
            if execution_eligibility == "unavailable":
                attempt_status = "not_executable"
            elif prediction_status != "available":
                attempt_status = "executable_prediction_unavailable"
            elif candidate == "mps_fixed_chi":
                attempt_status = "executable_quality_pending"
            else:
                attempt_status = "executable"
            prediction = {
                "prediction_id": attempt_id,
                "attempt_id": attempt_id,
                "panel_id": row["panel_id"],
                "panel_member_id": member_id,
                "qasm_sha256": row["qasm_sha256"],
                "normalized_qasm_sha256": row["normalized_qasm_sha256"],
                "fold_id": int(row["fold_id"]),
                "split_id": V2_SPLIT_ID,
                "family": row["family"],
                "stratum": row["stratum"],
                "width_qubits": width,
                "n_one_qubit_operations": n_one_qubit_operations,
                "n_cx_operations": n_cx_operations,
                "candidate_id": candidate,
                "candidate_config_sha256": config_sha,
                "candidate_config_status": config.get("status", "unavailable"),
                "configured_chi": "" if spec["configured_chi"] is None else spec["configured_chi"],
                "shots": spec["shots"],
                "seed": spec["seed"],
                "selected_clock_id": V2_CLOCK_ID,
                "selected_clock_field": V2_CLOCK_FIELD,
                "calibration_clock_id": "" if calibration is None else calibration.clock_id,
                "calibration_raw_sha256": "" if calibration is None else calibration.calibration_raw_sha256,
                "quality_gate_id": gate_id,
                "quality_gate_status": quality_status,
                "normalization_status": row["normalization_status"],
                "execution_eligibility": execution_eligibility,
                "execution_reason": execution_reason,
                "prediction_available": prediction_status == "available",
                "prediction_status": prediction_status,
                "prediction_reason": prediction_reason,
                "timing_status": timing_status,
                "timing_reason": timing_reason,
                "quality_status": quality_status,
                "quality_reason": "" if quality_status in {"not_required", "not_run"} else execution_reason,
                "metric_eligibility": metric_eligibility,
                "metric_ineligibility_reason": metric_ineligibility_reason,
                "predicted_one_qubit_seconds": "" if predicted is None else predicted.one_qubit_seconds,
                "predicted_cx_seconds": "" if predicted is None else predicted.cx_seconds,
                "predicted_sampling_seconds": "" if predicted is None else predicted.sampling_seconds,
                "predicted_process_isolated_seconds": (
                    "" if predicted is None else predicted.predicted_process_isolated_seconds
                ),
                "prediction_role": "candidate_runtime_prediction_only_no_selector_or_oracle",
            }
            plan = {
                **{key: prediction[key] for key in (
                    "attempt_id", "panel_id", "panel_member_id", "qasm_sha256", "normalized_qasm_sha256",
                    "fold_id", "split_id", "family", "stratum", "width_qubits",
                    "n_one_qubit_operations", "n_cx_operations", "candidate_id",
                    "candidate_config_sha256", "configured_chi", "shots", "seed", "selected_clock_id",
                    "selected_clock_field", "quality_gate_id", "normalization_status",
                    "execution_eligibility", "execution_reason", "prediction_status", "prediction_reason",
                    "timing_status", "timing_reason", "quality_status", "quality_reason",
                    "metric_eligibility", "metric_ineligibility_reason",
                )},
                "attempt_plan_status": attempt_status,
                "quality_threshold": ("" if candidate == "statevector"
                                      else manifest["quality_policy"]["threshold"]),
                "sessions": manifest["measurement"]["sessions"],
                "untimed_warmups_per_session": manifest["measurement"]["untimed_warmups_per_session"],
                "timed_repetitions_per_session": manifest["measurement"]["timed_warm_repetitions_per_session"],
                "cell_timeout_seconds": manifest["measurement"]["cell_timeout_seconds"],
            }
            predictions.append(prediction)
            plans.append(plan)
    expected_attempts = int(manifest["panel"]["assigned_candidate_attempts"])
    if len(predictions) != expected_attempts or len(plans) != expected_attempts:
        raise ValueError(f"v2 plan must preserve {expected_attempts} attempts")
    if len({row["attempt_id"] for row in plans}) != expected_attempts:
        raise ValueError("duplicate v2 attempt identity")
    return predictions, plans


def v3_materialize_predictions(contract: dict[str, Any], row_manifest: list[dict[str, Any]],
                               features: dict[str, CircuitOperationFeatures],
                               configs: dict[str, dict[str, Any]],
                               calibration: FittedCalibration | None,
                               calibration_reason: str,
                               manifest_sha256: str,
                               calibration_raw_sha256: str = "") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Plan every assigned v3 attempt; execution never depends on prediction."""
    if contract.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("v3_panel_plan_requires_resolved_v3_contract")
    if "resolution_provenance" not in contract:
        raise V3Unavailable("v3_panel_plan_requires_resolved_contract")
    panel_protocol_id = str(contract.get("panel_run_manifest_id", V3_MANIFEST_ID))
    if panel_protocol_id not in {V3_MANIFEST_ID, PARTIAL_MANIFEST_ID}:
        raise V3Unavailable("unsupported_panel_protocol_for_v3_contract")
    if (panel_protocol_id == PARTIAL_MANIFEST_ID
            and not re.fullmatch(r"[0-9a-f]{64}", str(contract.get("partial_resolved_contract_sha256", "")))):
        raise V3Unavailable("partial_panel_requires_resolved_amendment_identity")
    measurement = contract["measurement"]
    eligibility = contract["eligibility"]
    selected_clock_id = str(measurement["selected_clock_id"])
    selected_clock_field = str(measurement["selected_clock_field"])
    specs = v2_candidate_specs(contract)
    predictions: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    maximum_width = max(int(value) for value in contract["synthetic_corpus"]["statevector_widths"])

    for row in row_manifest:
        member_id = str(row["panel_member_id"])
        width = int(row["width_qubits"])
        feature = features.get(member_id)
        n_one = 0 if feature is None else int(feature.n_one_qubit_operations)
        n_cx = 0 if feature is None else int(feature.n_cx_operations)
        for spec in specs:
            candidate = str(spec["candidate"])
            config = configs.get(candidate, {})
            config_sha = str(config.get("sha256", ""))
            if row.get("normalization_status") != "ok":
                execution_status = "unavailable"
                execution_reason = f"normalization_unavailable:{row.get('unavailable_reason', 'unknown')}"
            elif config.get("status") != "ok" or not config_sha:
                execution_status = "unavailable"
                execution_reason = str(config.get("reason") or "candidate_config_unavailable")
            else:
                execution_status, execution_reason = "eligible", ""

            prediction = None
            if row.get("normalization_status") != "ok":
                prediction_status = "unavailable"
                prediction_reason = f"normalization_unavailable:{row.get('unavailable_reason', 'unknown')}"
            elif width > maximum_width:
                prediction_status, prediction_reason = "out_of_grid", f"out_of_grid:width_above_{maximum_width}"
            elif config.get("status") != "ok":
                prediction_status = "unavailable"
                prediction_reason = str(config.get("reason") or "candidate_config_unavailable")
            elif calibration is None:
                prediction_status = "unavailable"
                prediction_reason = calibration_reason or "v3_calibration_not_available"
            elif feature is None:
                prediction_status, prediction_reason = "unavailable", "normalized_features_missing"
            else:
                try:
                    predictor = (predict_candidate_runtime_partial
                                 if panel_protocol_id == PARTIAL_MANIFEST_ID
                                 else predict_candidate_runtime_v3)
                    prediction = predictor(
                        calibration=calibration, features=feature, candidate=candidate,
                        shots=int(spec["shots"]), configured_chi=spec["configured_chi"],
                    )
                    prediction_status, prediction_reason = "available", ""
                except PredictionUnavailable as exc:
                    prediction_reason = str(exc)
                    prediction_status = (
                        "out_of_grid" if any(part in prediction_reason.lower()
                                             for part in ("outside_grid", "out_of_grid", "extrapolat"))
                        else "unavailable"
                    )

            attempt_id = canonical_hash({
                "protocol": panel_protocol_id,
                "manifest_sha256": manifest_sha256,
                "partial_resolved_contract_sha256": str(
                    contract.get("partial_resolved_contract_sha256", "")
                ),
                "panel_member_id": member_id,
                "source_qasm_sha256": str(row["qasm_sha256"]),
                "fold_id": int(row["fold_id"]),
                "candidate_id": candidate,
                "candidate_config_sha256": config_sha or canonical_hash(spec),
                "calibration_raw_sha256": calibration_raw_sha256,
                "selected_clock_id": selected_clock_id,
                "selected_clock_field": selected_clock_field,
            })
            quality_status = (
                "not_required" if candidate == "statevector"
                else ("not_run" if execution_status == "eligible" else "quality_unavailable")
            )
            prediction_row = {
                "attempt_id": attempt_id, "prediction_id": attempt_id,
                "manifest_id": panel_protocol_id,
                "panel_id": row.get("panel_id", ""), "panel_member_id": member_id,
                "basename": row.get("basename", ""), "qasm_sha256": row.get("qasm_sha256", ""),
                "normalized_qasm_sha256": row.get("normalized_qasm_sha256", ""),
                "candidate_ir_feature_hash": row.get("candidate_ir_feature_hash", ""),
                "fold_id": int(row["fold_id"]), "split_id": V2_SPLIT_ID,
                "family": row.get("family", ""), "stratum": row.get("stratum", ""),
                "width_qubits": width, "n_one_qubit_operations": n_one, "n_cx_operations": n_cx,
                "gate_counts": {} if feature is None else dict(feature.gate_counts),
                "candidate_id": candidate, "candidate_config_sha256": config_sha,
                "candidate_config_status": config.get("status", "unavailable"),
                "configured_chi": "" if spec["configured_chi"] is None else spec["configured_chi"],
                "shots": int(spec["shots"]), "seed": int(spec["seed"]),
                "selected_clock_id": selected_clock_id, "selected_clock_field": selected_clock_field,
                "calibration_manifest_id": "" if calibration is None else calibration.calibration_manifest_id,
                "calibration_raw_sha256": calibration_raw_sha256,
                "normalization_status": row.get("normalization_status", "unknown"),
                "execution_eligibility": execution_status, "execution_reason": execution_reason,
                "prediction_status": prediction_status, "prediction_reason": prediction_reason,
                "timing_status": "not_started", "timing_reason": execution_reason,
                "quality_status": quality_status,
                "metric_eligibility": False,
                "metric_ineligibility_reason": "timing_not_yet_observed",
                "predicted_one_qubit_seconds": "" if prediction is None else prediction.one_qubit_seconds,
                "predicted_cx_seconds": "" if prediction is None else prediction.cx_seconds,
                "predicted_sampling_seconds": "" if prediction is None else prediction.sampling_seconds,
                "predicted_process_isolated_seconds": (
                    "" if prediction is None else prediction.predicted_process_isolated_seconds
                ),
                "prediction_role": "candidate_runtime_prediction_only_no_selector_or_oracle",
            }
            prediction_row["prediction_available"] = prediction_status == "available"
            prediction_row["quality_gate_id"] = str(spec["quality_gate_id"])
            prediction_row["quality_reason"] = execution_reason if quality_status == "quality_unavailable" else ""
            prediction_row["cell_timeout_seconds"] = float(measurement["cell_timeout_seconds"])
            plans.append({
                **prediction_row,
                "attempt_plan_status": (
                    "not_executable" if execution_status == "unavailable"
                    else ("executable" if prediction_status == "available"
                          else f"executable_{prediction_status}")
                ),
                "quality_threshold": ("" if candidate == "statevector"
                                      else float(contract["quality_policy"]["threshold"])),
                "sessions": int(measurement["sessions"]),
                "untimed_calls_per_session": int(measurement["untimed_calls_per_cell_per_session"]),
                "timed_repetitions_per_session": int(measurement["timed_repetitions_per_cell_per_session"]),
            })
            predictions.append(prediction_row)

    expected = int(eligibility["assigned_candidate_attempts"])
    if expected != 408 or len(plans) != expected or len(predictions) != expected:
        raise V3Unavailable(f"v3_panel_denominator_mismatch:{len(plans)}!={expected}")
    ids = [str(row["attempt_id"]) for row in plans]
    if len(set(ids)) != expected:
        raise V3Unavailable("v3_panel_attempt_identity_collision")
    expected_members = int(contract["panel"]["assigned_members"])
    if len(row_manifest) != expected_members or len({row["panel_member_id"] for row in row_manifest}) != expected_members:
        raise V3Unavailable("v3_panel_member_denominator_or_identity_mismatch")
    folds_by_hash: dict[str, set[int]] = {}
    for row in row_manifest:
        source_hash = str(row["qasm_sha256"])
        expected_fold = exact_qasm_fold(source_hash, V2_SPLIT_ID)
        if int(row["fold_id"]) != expected_fold:
            raise V3Unavailable(f"v3_exact_qasm_fold_assignment_mismatch:{row['panel_member_id']}")
        folds_by_hash.setdefault(source_hash, set()).add(int(row["fold_id"]))
    if any(len(folds) != 1 for folds in folds_by_hash.values()):
        raise V3Unavailable("v3_exact_qasm_fold_grouping_mismatch")
    if sum(row["execution_eligibility"] == "eligible" for row in plans) != sum(
        row["execution_eligibility"] == "eligible" for row in predictions
    ):
        raise V3Unavailable("v3_execution_eligibility_materialization_mismatch")
    return predictions, plans


def v3_resume_identity(contract: dict[str, Any], manifest_sha256: str,
                       input_pins: dict[str, Any], environment: dict[str, Any],
                       binary_pins: dict[str, Any], configs: dict[str, dict[str, Any]],
                       calibration_run_sha256: str, calibration_raw_sha256: str,
                       panel_pilot_selection: dict[str, Any], implementation_test_sha256: str) -> dict[str, Any]:
    """Pin immutable panel/split, current code/environment, and exact pilot order."""
    pins = {
        "overlay_manifest_sha256": manifest_sha256,
        "input_pins": input_pins,
        "environment_pins": environment,
        "binary_pins": binary_pins,
        "candidate_config_sha256": {
            name: str(value.get("sha256", "")) for name, value in sorted(configs.items())
        },
        "resolved_config_sha256": v2_resolved_config_sha256(configs),
        "calibration_run_manifest_sha256": calibration_run_sha256,
        "calibration_raw_sha256": calibration_raw_sha256,
        "selected_clock_id": str(contract["measurement"]["selected_clock_id"]),
        "selected_clock_field": str(contract["measurement"]["selected_clock_field"]),
        "split_id": V2_SPLIT_ID,
        "panel_pilot_selection": panel_pilot_selection,
        "implementation_test_sha256": implementation_test_sha256,
    }
    return build_v3_run_identity(contract, pins, {
        "manifest": manifest_sha256,
        "panel_csv": str(input_pins.get("panel_csv", {}).get("sha256", "")),
        "source_fixture_manifest": str(input_pins.get("source_fixture_manifest", {}).get("sha256", "")),
    })


def v3_validate_execution_acceptance(
    acceptance_path: Path | None, manifest_sha256: str, configs: dict[str, dict[str, Any]],
    environment: dict[str, Any], calibration_run: dict[str, Any] | None,
    calibration_run_sha256: str, calibration_raw_sha256: str,
    panel_pilot_selection: dict[str, Any], attempt_limit: int | None,
    output_dir: Path, contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless a v3-only calibration and panel-pilot acceptance matches."""
    if acceptance_path is None:
        raise V3Unavailable("v3_measure_requires_explicit_acceptance_json")
    if calibration_run is None:
        raise V3Unavailable("v3_measure_requires_completed_v3_calibration")
    acceptance = json.loads(Path(acceptance_path).read_text(encoding="utf-8"))
    run_status = str(calibration_run.get("status", ""))
    if calibration_run.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("v3_calibration_manifest_id_mismatch")
    if run_status not in {"complete", "complete_with_unavailable_knots"}:
        raise V3Unavailable(f"v3_calibration_status_not_complete:{run_status}")
    run_identity = calibration_run.get("run_identity", {})
    if run_identity.get("manifest_sha256") != manifest_sha256:
        raise V3Unavailable("v3_calibration_manifest_sha256_mismatch")
    expected = {
        "status": "pass", "manifest_id": V3_MANIFEST_ID,
        "manifest_sha256": manifest_sha256,
        "code_hashes": environment.get("code_hashes", {}),
        "calibration_run_manifest_sha256": calibration_run_sha256,
        "calibration_status": run_status,
        "calibration_raw_sha256": calibration_raw_sha256,
        "worker_environment_sha256": (
            calibration_run.get("fit_summary", {}).get("worker_environment_sha256", "")
        ),
        "worker_context_sha256_by_config_policy": (
            calibration_run.get("fit_summary", {}).get("worker_context_sha256_by_config_policy", {})
        ),
        "resolved_config_sha256": v2_resolved_config_sha256(configs),
        "selected_clock_id": str(calibration_run.get("selected_clock_id", "")),
        "selected_clock_field": str(calibration_run.get("selected_clock_field", "")),
        "panel_pilot_selection_sha256": canonical_hash(panel_pilot_selection),
    }
    if not expected["worker_environment_sha256"] or not expected["worker_context_sha256_by_config_policy"]:
        raise V3Unavailable("v3_calibration_worker_context_pins_missing")
    for key, value in expected.items():
        if acceptance.get(key) != value:
            raise V3Unavailable(f"v3_acceptance_{key}_mismatch_or_missing")
    if acceptance["selected_clock_id"] != str(
        calibration_run.get("selected_clock_id")
    ) or acceptance["selected_clock_field"] != str(calibration_run.get("selected_clock_field")):
        raise V3Unavailable("v3_acceptance_selected_clock_mismatch")

    if attempt_limit != 10:
        if contract is None:
            raise V3Unavailable("v3_full_panel_requires_resolved_contract")
        run_path = _v2_artifact_path(contract["outputs"]["measurement"], output_dir) / "run_manifest.json"
        if not run_path.is_file():
            raise V3Unavailable("v3_full_panel_requires_completed_ten_attempt_pilot")
        prior = json.loads(run_path.read_text(encoding="utf-8"))
        if prior.get("manifest_id") != V3_MANIFEST_ID:
            raise V3Unavailable("v3_resume_rejects_non_v3_panel_manifest")
        pilot = prior.get("measurement", {}).get("pilot_ledger", {})
        ledger_path = Path(str(pilot.get("path", "")))
        if not ledger_path.is_absolute():
            ledger_path = output_dir / ledger_path
        byte_count = int(pilot.get("byte_count", -1))
        _, prefix_hash = _v2_ledger_prefix(ledger_path, byte_count)
        if prefix_hash != pilot.get("sha256"):
            raise V3Unavailable("v3_pilot_ledger_prefix_hash_mismatch")
        if acceptance.get("pilot_ledger_sha256") != prefix_hash or acceptance.get("pilot_qa") != "pass":
            raise V3Unavailable("v3_full_panel_acceptance_requires_pilot_ledger_and_qa_pass")
        with ledger_path.open("rb") as handle:
            prefix_rows = [json.loads(line) for line in handle.read(byte_count).decode("utf-8").splitlines()
                           if line.strip()]
        measurement = (contract or {}).get("measurement", {
            "sessions": 3, "untimed_calls_per_cell_per_session": 3,
            "timed_repetitions_per_cell_per_session": 5,
        })
        computed_qa = v3_assess_panel_pilot(prefix_rows, panel_pilot_selection, measurement)
        if computed_qa.get("status") != "pass":
            raise V3Unavailable(f"v3_pilot_ledger_computed_qa_failed:{computed_qa.get('reasons', [])}")
        if prior.get("measurement", {}).get("pilot_qa") != computed_qa:
            raise V3Unavailable("v3_pilot_ledger_computed_qa_manifest_mismatch")
    return acceptance


def partial_validate_execution_acceptance(
    acceptance_path: Path | None, resolution: dict[str, Any], manifest_sha256: str,
    configs: dict[str, dict[str, Any]], environment: dict[str, Any],
    calibration_run: dict[str, Any] | None, calibration_run_sha256: str,
    calibration_ledger_sha256: str, panel_pilot_selection: dict[str, Any],
    attempt_limit: int | None, output_dir: Path,
) -> dict[str, Any]:
    """Require a partial-only accepted ledger/context and (for full run) pilot."""
    if acceptance_path is None:
        raise V3Unavailable("partial_measure_requires_explicit_acceptance_json")
    if calibration_run is None:
        raise V3Unavailable("partial_measure_requires_completed_partial_calibration")
    acceptance = json.loads(Path(acceptance_path).read_text(encoding="utf-8"))
    if acceptance.get("manifest_id") in {V2_MANIFEST_ID, V3_MANIFEST_ID}:
        raise V3Unavailable("partial_measure_rejects_v2_or_v3_acceptance")
    if acceptance.get("status") != "pass":
        raise V3Unavailable("partial_panel_acceptance_status_not_pass")
    if calibration_run.get("manifest_id") != PARTIAL_MANIFEST_ID:
        raise V3Unavailable("partial_calibration_manifest_id_mismatch")
    run_status = str(calibration_run.get("status", ""))
    if run_status not in {"complete", "complete_with_unavailable_knots"}:
        raise V3Unavailable(f"partial_calibration_status_not_complete:{run_status}")
    fit_summary = calibration_run.get("fit_summary", {})
    expected = {
        "manifest_id": PARTIAL_MANIFEST_ID,
        "manifest_sha256": manifest_sha256,
        "amendment_identity": resolution["amendment_identity"],
        "parent_contract_sha256": resolution["parent_contract_sha256"],
        "resolved_contract_sha256": resolution["resolved_contract_sha256"],
        "code_hashes": environment.get("code_hashes", {}),
        "calibration_run_manifest_sha256": calibration_run_sha256,
        "calibration_raw_sha256": calibration_ledger_sha256,
        "calibration_status": run_status,
        "resolved_config_sha256": v2_resolved_config_sha256(configs),
        "worker_environment_sha256": fit_summary.get("worker_environment_sha256", ""),
        "worker_context_sha256_by_config_policy": fit_summary.get(
            "worker_context_sha256_by_config_policy", {}
        ),
        "selected_clock_id": str(resolution["contract"]["measurement"]["selected_clock_id"]),
        "selected_clock_field": str(resolution["contract"]["measurement"]["selected_clock_field"]),
        "panel_pilot_selection_sha256": canonical_hash(panel_pilot_selection),
    }
    if not expected["worker_environment_sha256"] or not expected["worker_context_sha256_by_config_policy"]:
        raise V3Unavailable("partial_calibration_worker_context_pins_missing")
    for key, value in expected.items():
        if acceptance.get(key) != value:
            raise V3Unavailable(f"partial_panel_acceptance_{key}_mismatch_or_missing")
    if attempt_limit == 10:
        return acceptance

    run_path = _v2_artifact_path(
        resolution["amendment"]["outputs"]["measurement"], output_dir,
    ) / "run_manifest.json"
    if not run_path.is_file():
        raise V3Unavailable("partial_full_panel_requires_completed_ten_attempt_pilot")
    prior = json.loads(run_path.read_text(encoding="utf-8"))
    if prior.get("manifest_id") != PARTIAL_MANIFEST_ID:
        raise V3Unavailable("partial_resume_rejects_non_partial_panel_manifest")
    pilot = prior.get("measurement", {}).get("pilot_ledger", {})
    ledger_path = Path(str(pilot.get("path", "")))
    if not ledger_path.is_absolute():
        ledger_path = output_dir / ledger_path
    byte_count = int(pilot.get("byte_count", -1))
    prefix, prefix_sha = _v2_ledger_prefix(ledger_path, byte_count)
    if prefix_sha != pilot.get("sha256"):
        raise V3Unavailable("partial_panel_pilot_ledger_prefix_hash_mismatch")
    if acceptance.get("pilot_ledger_sha256") != prefix_sha or acceptance.get("pilot_qa") != "pass":
        raise V3Unavailable("partial_full_panel_acceptance_requires_pilot_ledger_and_qa_pass")
    rows = [json.loads(line) for line in prefix.decode("utf-8").splitlines() if line.strip()]
    computed = v3_assess_panel_pilot(
        rows, panel_pilot_selection, resolution["contract"]["measurement"],
    )
    if computed.get("status") != "pass" or prior.get("measurement", {}).get("pilot_qa") != computed:
        raise V3Unavailable("partial_panel_pilot_ledger_qa_mismatch")
    return acceptance


def v3_load_calibration(contract: dict[str, Any], manifest_path: Path,
                        configs: dict[str, dict[str, Any]],
                        environment: dict[str, Any]) -> tuple[FittedCalibration | None, str, str,
                                                               dict[str, Any] | None, str]:
    """Load only a complete, pinned v3 calibration bundle; missing means unavailable."""
    calibration_dir = ROOT / contract["outputs"]["calibration"]
    run_path = calibration_dir / "run_manifest.json"
    if not run_path.is_file():
        return None, "v3_calibration_not_available", "", None, ""
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run_sha = sha256_file(run_path)
        if run.get("manifest_id") != V3_MANIFEST_ID:
            raise V3Unavailable("v3_calibration_manifest_id_mismatch")
        if run.get("status") not in {"complete", "complete_with_unavailable_knots"}:
            raise V3Unavailable(f"v3_calibration_status_not_complete:{run.get('status', 'missing')}")
        if int(run.get("completed_cells", -1)) != int(run.get("planned_cells", -2)):
            raise V3Unavailable("v3_calibration_completed_cell_count_mismatch")
        if int(run.get("planned_cells", -1)) != int(contract["synthetic_corpus"]["planned_cells_including_diagnostics"]):
            raise V3Unavailable("v3_calibration_planned_cell_count_mismatch")
        clock_id = str(contract["measurement"]["selected_clock_id"])
        clock_field = str(contract["measurement"]["selected_clock_field"])
        if (run.get("selected_clock_id", run.get("clock_id")) != clock_id
                or run.get("selected_clock_field") != clock_field):
            raise V3Unavailable("v3_calibration_selected_clock_mismatch")

        wrapper = run.get("run_identity", {})
        identity = wrapper.get("run_identity", wrapper)
        identity_sha = run.get("run_identity_sha256", wrapper.get("run_identity_sha256", ""))
        if not isinstance(identity, dict) or not identity_sha or canonical_hash(identity) != identity_sha:
            raise V3Unavailable("v3_calibration_run_identity_hash_mismatch")
        if wrapper.get("manifest_id", run.get("manifest_id")) != V3_MANIFEST_ID:
            raise V3Unavailable("v3_calibration_identity_manifest_mismatch")
        pins = identity.get("pins", identity)
        expected_manifest_sha = sha256_file(manifest_path)
        manifest_pin = (pins.get("overlay_manifest_sha256") or pins.get("manifest_sha256")
                        or identity.get("manifest_sha256"))
        if manifest_pin != expected_manifest_sha:
            raise V3Unavailable("v3_calibration_overlay_manifest_sha256_mismatch")
        if identity.get("manifest_id", V3_MANIFEST_ID) != V3_MANIFEST_ID:
            raise V3Unavailable("v3_calibration_identity_manifest_mismatch")

        calibration_code = {
            "benchmark_v1/scripts/run_maestro_component_calibration.py": sha256_file(
                ROOT / "benchmark_v1/scripts/run_maestro_component_calibration.py"),
            "benchmark_v1/scripts/run_maestro_common_panel.py": sha256_file(
                ROOT / "benchmark_v1/scripts/run_maestro_common_panel.py"),
            "benchmark_v1/qre_benchmark/maestro_component_v2.py": sha256_file(
                ROOT / "benchmark_v1/qre_benchmark/maestro_component_v2.py"),
            "benchmark_v1/qre_benchmark/maestro_component_predictor.py": sha256_file(
                ROOT / "benchmark_v1/qre_benchmark/maestro_component_predictor.py"),
        }
        code_hashes = (pins.get("code_hashes") or identity.get("code_hashes")
                       or pins.get("environment_pins", {}).get("code_hashes", {}))
        for path, digest in calibration_code.items():
            if code_hashes.get(path) != digest:
                raise V3Unavailable(f"v3_calibration_code_hash_mismatch:{path}")
        configuration_records = (pins.get("configuration_records")
                                 or identity.get("configuration_records")
                                 or run.get("configuration_records", {}))
        for candidate, chi in (("statevector", None),
                               ("mps_fixed_chi", int(contract["panel_candidate_configuration"]["mps_configured_chi"]))):
            key = f"{candidate}:{chi}"
            actual = configuration_records.get(key, {})
            expected_sha = str(configs.get(candidate, {}).get("sha256", ""))
            if not expected_sha or actual.get("sha256") != expected_sha:
                raise V3Unavailable(f"v3_calibration_resolved_config_mismatch:{key}")
        for key in ("python_executable_sha256", "python_version", "platform",
                    "maestro_package_tree_sha256", "qiskit_package_tree_sha256",
                    "cpu_affinity", "native_thread_environment", "resource_profile"):
            expected_value = environment.get(key)
            actual_value = pins.get(key, pins.get("environment_pins", {}).get(key))
            if expected_value is not None and actual_value != expected_value:
                raise V3Unavailable(f"v3_calibration_environment_pin_mismatch:{key}")

        raw_meta = run.get("raw_records", {})
        raw_path = _v2_artifact_path(str(raw_meta.get("path", "")), calibration_dir)
        checkpoint_path = _v2_artifact_path(str(run.get("checkpoint_path", "")), calibration_dir)
        fit_path = calibration_dir / "fitted_calibration.json"
        if not raw_path.is_file() or not checkpoint_path.is_file() or not fit_path.is_file():
            raise V3Unavailable("v3_calibration_raw_checkpoint_or_fit_missing")
        raw_sha = sha256_file(raw_path)
        if raw_sha != raw_meta.get("sha256"):
            raise V3Unavailable("v3_calibration_raw_sha256_mismatch")
        if int(raw_meta.get("rows", -1)) != len(_csv_rows(raw_path)):
            raise V3Unavailable("v3_calibration_raw_row_count_mismatch")
        checkpoint_sha = sha256_file(checkpoint_path)
        if checkpoint_sha != run.get("checkpoint_sha256"):
            raise V3Unavailable("v3_calibration_checkpoint_sha256_mismatch")
        if run.get("checkpoint_format") != "maestro_v3_calibration_checkpoint_jsonl_v1":
            raise V3Unavailable("v3_calibration_checkpoint_format_mismatch")
        checkpoint_rows: list[dict[str, str]] = []
        with checkpoint_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.endswith("\n"):
                    raise V3Unavailable(f"v3_calibration_checkpoint_incomplete_line:{line_number}")
                record = json.loads(line)
                if (record.get("manifest_id") != V3_MANIFEST_ID
                        or record.get("run_identity_sha256") != run.get("run_identity_sha256")):
                    raise V3Unavailable(f"v3_calibration_checkpoint_identity_mismatch:{line_number}")
                if record.get("record_type") == "attempt_result":
                    checkpoint_rows.append({key: _v2_checkpoint_value(value)
                                            for key, value in record.get("row", {}).items()})
        csv_rows = _csv_rows(raw_path)
        if checkpoint_rows != csv_rows:
            raise V3Unavailable("v3_calibration_checkpoint_raw_record_mismatch")
        if int(run.get("checkpoint_attempt_rows", -1)) != len(checkpoint_rows):
            raise V3Unavailable("v3_calibration_checkpoint_attempt_count_mismatch")
        attempt_ids = [row.get("attempt_id", "") for row in csv_rows]
        if not all(attempt_ids) or len(set(attempt_ids)) != len(attempt_ids):
            raise V3Unavailable("v3_calibration_missing_or_duplicate_attempt_id")
        fit_payload = json.loads(fit_path.read_text(encoding="utf-8"))
        if (fit_payload.get("calibration_manifest_id") != V3_MANIFEST_ID
                or fit_payload.get("calibration_raw_sha256", fit_payload.get("raw_sha256")) != raw_sha
                or fit_payload.get("fit_version") != "paired_endpoint_heldout_512_v3"):
            raise V3Unavailable("v3_calibration_fitted_artifact_identity_mismatch")
        rank_records = run.get("rank_preflight_records", [])
        rank_path = run.get("rank_preflight_path")
        if rank_path:
            rank_file = _v2_artifact_path(str(rank_path), calibration_dir)
            if not rank_file.is_file():
                raise V3Unavailable("v3_calibration_rank_preflight_missing")
            rank_records = json.loads(rank_file.read_text(encoding="utf-8"))
        fitted = fit_v3_calibration_rows(
            csv_rows, contract, raw_sha,
            expected_run_identity_sha256=run.get("run_identity_sha256", identity_sha),
            rank_preflight_records=rank_records,
        )
        fit_summary = run.get("fit_summary", {})
        if fitted.fit_diagnostics.get("worker_environment_sha256") != fit_summary.get("worker_environment_sha256"):
            raise V3Unavailable("v3_calibration_worker_environment_fit_mismatch")
        if (fitted.fit_diagnostics.get("worker_context_sha256_by_config_policy")
                != fit_summary.get("worker_context_sha256_by_config_policy")):
            raise V3Unavailable("v3_calibration_worker_context_fit_mismatch")
        return fitted, "", raw_sha, run, run_sha
    except Exception as exc:
        reason = str(exc) if isinstance(exc, V3Unavailable) else f"{type(exc).__name__}:{exc}"
        return None, f"v3_calibration_unavailable:{reason}", "", None, ""


def v3_terminal_attempt_status(plan: dict[str, Any], timing_status: str,
                               quality_status: str) -> dict[str, Any]:
    """Pure terminal reducer: failed MPS quality retains timing but bars metrics."""
    if timing_status not in {"not_started", "ok", "timeout", "error"}:
        raise V3Unavailable(f"v3_invalid_timing_status:{timing_status}")
    if quality_status not in {"not_required", "not_run", "ok", "quality_failed", "quality_unavailable"}:
        raise V3Unavailable(f"v3_invalid_quality_status:{quality_status}")
    eligible = bool(
        plan.get("execution_eligibility") == "eligible"
        and plan.get("prediction_status") == "available"
        and timing_status == "ok"
        and quality_status in {"not_required", "ok"}
    )
    if plan.get("execution_eligibility") != "eligible":
        reason = str(plan.get("execution_reason") or "execution_unavailable")
    elif plan.get("prediction_status") != "available":
        reason = str(plan.get("prediction_reason") or "prediction_unavailable")
    elif timing_status != "ok":
        reason = str(plan.get("timing_reason") or f"timing_{timing_status}")
    elif quality_status not in {"not_required", "ok"}:
        reason = str(plan.get("quality_reason") or quality_status)
    else:
        reason = ""
    return {
        "normalization_status": plan.get("normalization_status", "unknown"),
        "execution_eligibility": plan.get("execution_eligibility", "unavailable"),
        "execution_reason": plan.get("execution_reason", ""),
        "prediction_status": plan.get("prediction_status", "unavailable"),
        "prediction_reason": plan.get("prediction_reason", ""),
        "timing_status": timing_status, "timing_reason": plan.get("timing_reason", ""),
        "quality_status": quality_status,
        "metric_eligibility": eligible,
        "metric_ineligibility_reason": reason,
        "selected_clock_id": plan.get("selected_clock_id", ""),
        "selected_clock_field": plan.get("selected_clock_field", ""),
    }


def v3_assess_panel_pilot(ledger_rows: list[dict[str, Any]],
                          panel_pilot_selection: dict[str, Any],
                          measurement: dict[str, Any]) -> dict[str, Any]:
    """Compute PASS only for the exact ordered, complete ten-attempt pilot."""
    selected = [str(value) for value in panel_pilot_selection.get("selected_attempt_ids", [])]
    reasons: list[str] = []
    if len(selected) != 10 or len(set(selected)) != 10:
        reasons.append("pilot_selection_not_exactly_ten_unique_ids")
    coverage = panel_pilot_selection.get("coverage", {})
    if not all(coverage.get(key) is True for key in (
        "both_engines", "lowest_width_covered", "q16_covered", "exact_entangled_input_covered",
    )):
        reasons.append("pilot_required_strata_missing")
    selection_record = next((row for row in ledger_rows if row.get("stage") == "panel_pilot_selection"), None)
    if not selection_record or selection_record.get("selection") != panel_pilot_selection:
        reasons.append("pilot_selection_record_mismatch")
    control = next((row for row in ledger_rows if row.get("stage") == "quality_controls"), None)
    if not control or control.get("status") != "pass":
        reasons.append("panel_quality_controls_not_passed")

    terminals = [row for row in ledger_rows if row.get("stage") == "attempt_complete"
                 and str(row.get("attempt_id", "")) in set(selected)]
    terminal_order = [str(row.get("attempt_id", "")) for row in terminals]
    if terminal_order != selected:
        reasons.append("pilot_terminal_order_mismatch")
    terminal_by_attempt = {str(row.get("attempt_id", "")): row for row in terminals}
    expected_observations = int(measurement["sessions"]) * (
        int(measurement["untimed_calls_per_cell_per_session"])
        + int(measurement["timed_repetitions_per_cell_per_session"])
    )
    expected_timed = int(measurement["sessions"]) * int(measurement["timed_repetitions_per_cell_per_session"])
    quality_counts: Counter[str] = Counter()
    for attempt_id in selected:
        terminal = terminal_by_attempt.get(attempt_id)
        if terminal is None:
            reasons.append(f"pilot_terminal_missing:{attempt_id}")
            continue
        if terminal.get("execution_eligibility") != "eligible":
            reasons.append(f"pilot_execution_unavailable:{attempt_id}")
        if terminal.get("timing_status") != "ok":
            reasons.append(f"pilot_timing_not_ok:{attempt_id}")
        quality_status = str(terminal.get("quality_status", ""))
        quality_counts[quality_status] += 1
        if quality_status not in {"not_required", "ok", "quality_failed", "quality_unavailable"}:
            reasons.append(f"pilot_quality_status_invalid:{attempt_id}")
        observations = [row for row in ledger_rows
                        if row.get("attempt_id") == attempt_id
                        and row.get("stage") in {V3_UNTIMED_STAGE, V3_TIMED_STAGE}]
        timed = [row for row in observations if row.get("stage") == V3_TIMED_STAGE]
        untimed = [row for row in observations if row.get("stage") == V3_UNTIMED_STAGE]
        if (len(observations) != expected_observations or len(timed) != expected_timed
                or len(untimed) != expected_observations - expected_timed):
            reasons.append(f"pilot_observation_count_mismatch:{attempt_id}")
        if any(row.get("status") != "ok" or row.get("timing_status") != "ok" for row in observations):
            reasons.append(f"pilot_observation_failure:{attempt_id}")
    return {
        "status": "pass" if not reasons else "fail",
        "attempt_count": len(selected), "attempt_ids": selected,
        "coverage": coverage, "timing_status_counts": dict(Counter(
            str(row.get("timing_status", "missing")) for row in terminals
        )),
        "quality_status_counts": dict(sorted(quality_counts.items())),
        "quality_controls_status": "pass" if control and control.get("status") == "pass" else "fail",
        "reasons": reasons,
    }


def v3_calibration_worker_context_check(context: dict[str, Any], calibration_run: dict[str, Any]) -> None:
    """Match an actual panel worker against W3's accepted context fingerprints."""
    if context.get("context_owner") != "worker":
        raise V3Unavailable("v3_panel_context_owner_mismatch")
    fit_summary = calibration_run.get("fit_summary", {})
    expected_environment = str(fit_summary.get("worker_environment_sha256", ""))
    expected_contexts = fit_summary.get("worker_context_sha256_by_config_policy", {})
    actual_environment = str(context.get("worker_environment_sha256", ""))
    if not expected_environment or actual_environment != expected_environment:
        raise V3Unavailable("v3_panel_worker_environment_drift_from_calibration")
    identity = context.get("identity", {})
    config_sha = str(context.get("resolved_config_sha256", ""))
    policy_sha = str(context.get("backend_threading_policy_sha256", ""))
    if (identity.get("resolved_config_sha256") != config_sha
            or identity.get("backend_threading_policy_sha256") != policy_sha
            or context.get("identity_sha256") != canonical_hash(identity)):
        raise V3Unavailable("v3_panel_worker_context_hash_invalid")
    if context.get("identity_sha256") != expected_contexts.get(f"{config_sha}:{policy_sha}"):
        raise V3Unavailable("v3_panel_worker_context_drift_from_calibration")


def v3_measure_plan(contract: dict[str, Any], plans: list[dict[str, Any]],
                    normalized_sidecar: list[dict[str, Any]], output_root: Path,
                    attempt_limit: int | None, run_identity: dict[str, Any],
                    configs: dict[str, dict[str, Any]], calibration_run: dict[str, Any],
                    panel_pilot_selection: dict[str, Any]) -> dict[str, Any]:
    """Run the accepted v3 pilot/full panel; never prune on prediction or quality."""
    if len(plans) != int(contract["eligibility"]["assigned_candidate_attempts"]) or len(plans) != 408:
        raise V3Unavailable("v3_measurement_denominator_mismatch")
    if attempt_limit not in (None, 10, 408):
        raise V3Unavailable("v3_attempt_limit_must_be_10_or_408")
    protocol_id = str(contract.get("panel_run_manifest_id", V3_MANIFEST_ID))
    if protocol_id not in {V3_MANIFEST_ID, PARTIAL_MANIFEST_ID}:
        raise V3Unavailable("unsupported_panel_protocol_for_measurement")
    selected_clock_id = str(contract["measurement"]["selected_clock_id"])
    selected_clock_field = str(contract["measurement"]["selected_clock_field"])
    if selected_clock_field != V2_CLOCK_FIELD:
        raise V3Unavailable("v3_shared_execute_worker_selected_clock_field_unsupported")
    identity_sha = str(run_identity["run_identity_sha256"])
    ledger_path = _v2_artifact_path(contract["outputs"]["measurement"], output_root) / "raw_records.jsonl"
    ledger_path = ledger_path.resolve()
    sidecar_by_member = {str(row["panel_member_id"]): row for row in normalized_sidecar}
    eligible_ids = [str(row["attempt_id"]) for row in plans
                    if row.get("execution_eligibility") == "eligible"]
    if attempt_limit == 10:
        selected_ids = list(panel_pilot_selection.get("selected_attempt_ids", []))
        if len(selected_ids) != 10 or len(set(selected_ids)) != 10 or not set(selected_ids).issubset(eligible_ids):
            raise V3Unavailable("v3_exact_stratified_pilot_ids_invalid")
    else:
        selected_ids = eligible_ids
    selected_id_set = set(selected_ids)
    measurement = contract["measurement"]
    policy = _manifest_thread_policy(contract)
    seed = int(contract["seed"])
    specs = {spec["candidate"]: spec for spec in v2_candidate_specs(contract)}
    new_complete = 0

    with maestro_timing_lock(protocol_id, identity_sha, root=ROOT):
        records = _v2_jsonl_records(ledger_path)
        if any(row.get("run_identity_sha256") != identity_sha for row in records):
            raise V3Unavailable("v3_resume_ledger_identity_mismatch")
        by_id = {str(row["record_id"]): row for row in records}
        selection_record_id = canonical_hash({
            "manifest_id": protocol_id, "stage": "panel_pilot_selection",
            "run_identity_sha256": identity_sha,
        })
        saved_selection = by_id.get(selection_record_id)
        if saved_selection is not None and saved_selection.get("selection") != panel_pilot_selection:
            raise V3Unavailable("v3_resume_pilot_selection_mismatch")
        if saved_selection is None:
            record = {"record_id": selection_record_id, "manifest_id": protocol_id,
                      "stage": "panel_pilot_selection", "selection": panel_pilot_selection,
                      "run_identity_sha256": identity_sha}
            _v2_append_jsonl(ledger_path, record)
            by_id[selection_record_id] = record

        control_id = canonical_hash({"manifest_id": protocol_id, "stage": "quality_controls",
                                     "run_identity_sha256": identity_sha})
        control_record = by_id.get(control_id)
        if control_record is None:
            # This call is only reachable after the v3 acceptance gate.  It is
            # deliberately before any panel timing and is checkpointed once.
            controls = v2_run_quality_controls(contract, load_maestro_parser())
            control_record = {"record_id": control_id, "manifest_id": protocol_id,
                              "stage": "quality_controls", "status": controls.get("status", "failed"),
                              "controls": controls, "run_identity_sha256": identity_sha}
            _v2_append_jsonl(ledger_path, control_record)
            by_id[control_id] = control_record
        if control_record.get("status") != "pass":
            raise SystemExit("v3 measurement blocked: zero/Bell/negative/asymmetric-order controls did not pass")
        expected_environment = str(calibration_run.get("fit_summary", {}).get("worker_environment_sha256", ""))
        control_results = control_record.get("controls", {}).get("results", {})
        if not expected_environment or any(
            detail.get("worker_environment_sha256") != expected_environment
            for detail in control_results.values()
        ):
            raise V3Unavailable("v3_quality_control_worker_environment_drift_from_calibration")

        saved_contexts = calibration_run.get("worker_contexts_by_config", calibration_run.get("worker_contexts", []))
        if isinstance(saved_contexts, dict):
            saved_contexts = list(saved_contexts.values())
        context_records = [{"stage": "worker_context", "worker_context": context}
                           for context in saved_contexts if isinstance(context, dict)]
        context_registry = _restore_worker_contexts(context_records) if context_records else {}
        # Also validate that every panel configuration maps to a frozen W3
        # context fingerprint before launching an attempt.
        calibration_context_ids = calibration_run.get("fit_summary", {}).get(
            "worker_context_sha256_by_config_policy", {})
        if not calibration_context_ids:
            raise V3Unavailable("v3_calibration_worker_context_hashes_missing")

        completed_attempts = {str(row.get("attempt_id")) for row in by_id.values()
                              if row.get("stage") == "attempt_complete"}
        started_calls = {str(row.get("record_id")) for row in by_id.values()
                         if row.get("stage") == "call_started"}
        plan_by_id = {str(row["attempt_id"]): row for row in plans}
        unavailable_plans = [row for row in plans if row.get("execution_eligibility") != "eligible"]
        selected_plans = [plan_by_id[attempt_id] for attempt_id in selected_ids]
        remaining_eligible = [row for row in plans
                              if row.get("execution_eligibility") == "eligible"
                              and row["attempt_id"] not in selected_id_set]
        ordered_plans = unavailable_plans + selected_plans + (remaining_eligible if attempt_limit != 10 else [])
        for plan in ordered_plans:
            attempt_id = str(plan["attempt_id"])
            if attempt_id in completed_attempts:
                continue
            if plan.get("execution_eligibility") != "eligible":
                terminal = {
                    "record_id": canonical_hash({"manifest_id": protocol_id,
                                                  "attempt_id": attempt_id, "stage": "attempt_complete"}),
                    "manifest_id": protocol_id, "attempt_id": attempt_id,
                    "stage": "attempt_complete", "status": "not_executable",
                    **v3_terminal_attempt_status(plan, "not_started",
                        "not_required" if plan["candidate_id"] == "statevector" else "quality_unavailable"),
                    "run_identity_sha256": identity_sha,
                }
                _v2_append_jsonl(ledger_path, terminal)
                by_id[terminal["record_id"]] = terminal
                completed_attempts.add(attempt_id)
                new_complete += 1
                continue
            if attempt_id not in selected_id_set:
                continue

            quality_result: dict[str, Any] = {"status": "not_required"}
            if plan["candidate_id"] == "mps_fixed_chi":
                quality_id = canonical_hash({"manifest_id": protocol_id, "attempt_id": attempt_id,
                                             "stage": "quality_reference"})
                saved_quality = by_id.get(quality_id)
                if saved_quality:
                    quality_result = saved_quality.get("quality_result", {"status": "quality_unavailable"})
                else:
                    call_id = canonical_hash({"record_id": quality_id, "stage": "call_started"})
                    if call_id in started_calls:
                        quality_result = {"status": "quality_unavailable", "reason": "resume_interrupted_quality_call"}
                    else:
                        marker = {"record_id": call_id, "manifest_id": protocol_id,
                                  "related_record_id": quality_id, "attempt_id": attempt_id,
                                  "stage": "call_started", "call_kind": "quality_reference",
                                  "run_identity_sha256": identity_sha}
                        _v2_append_jsonl(ledger_path, marker)
                        started_calls.add(call_id)
                        quality_result = _v2_quality_for_attempt(
                            sidecar_by_member[plan["panel_member_id"]], plan, contract,
                            expected_worker_context=_expected_worker_context(
                                configs, "mps_fixed_chi", context_registry, policy,
                            ),
                        )
                    context = quality_result.get("worker_context")
                    if isinstance(context, dict):
                        v3_calibration_worker_context_check(context, calibration_run)
                        meta = _record_worker_context(context, context_registry, ledger_path,
                                                      identity_sha, by_id, related_record_id=quality_id)
                    else:
                        meta = {}
                    quality_record = {
                        "record_id": quality_id, "manifest_id": protocol_id,
                        "attempt_id": attempt_id, "stage": "quality_reference",
                        "quality_status": quality_result.get("status", "quality_unavailable"),
                        "quality_result": {key: value for key, value in quality_result.items()
                                           if key != "worker_context"},
                        "quality_value": quality_result.get("fidelity", ""),
                        **meta, "run_identity_sha256": identity_sha,
                    }
                    _v2_append_jsonl(ledger_path, quality_record)
                    by_id[quality_id] = quality_record
            quality_status = (
                "not_required" if plan["candidate_id"] == "statevector"
                else ("ok" if quality_result.get("status") == "ok"
                      else ("quality_failed" if quality_result.get("status") == "quality_failed"
                            else "quality_unavailable"))
            )
            spec = specs[plan["candidate_id"]]
            sidecar = sidecar_by_member[plan["panel_member_id"]]
            for session in range(1, int(measurement["sessions"]) + 1):
                stages = [(V3_UNTIMED_STAGE, repetition) for repetition in range(
                    int(measurement["untimed_calls_per_cell_per_session"]))]
                stages += [(V3_TIMED_STAGE, repetition) for repetition in range(
                    int(measurement["timed_repetitions_per_cell_per_session"]))]
                for stage, repetition in stages:
                    result_id = canonical_hash({"manifest_id": protocol_id, "attempt_id": attempt_id,
                                                "session": session, "stage": stage,
                                                "repetition": repetition})
                    if result_id in by_id:
                        continue
                    call_id = canonical_hash({"record_id": result_id, "stage": "call_started"})
                    if call_id in started_calls:
                        result = {"ok": False, "error": "resume_interrupted_call", "transport_status": "hard_exit"}
                    else:
                        marker = {"record_id": call_id, "manifest_id": protocol_id,
                                  "related_record_id": result_id, "attempt_id": attempt_id,
                                  "stage": "call_started", "call_kind": stage,
                                  "run_identity_sha256": identity_sha}
                        _v2_append_jsonl(ledger_path, marker)
                        started_calls.add(call_id)
                        result = isolated_call(_v2_simple_execute_worker, (
                            sidecar["normalized_qasm"], spec["candidate"], spec["configured_chi"],
                            int(spec["shots"]), int(spec["seed"]),
                            float(spec["singular_value_threshold"] or 0.0), policy,
                            _expected_worker_context(configs, spec["candidate"], context_registry, policy),
                        ), float(measurement["cell_timeout_seconds"]))
                    ok = bool(result.get("ok"))
                    transport = result.get("transport_status", "unknown")
                    context = result.get("worker_context")
                    if isinstance(context, dict):
                        v3_calibration_worker_context_check(context, calibration_run)
                        meta = _record_worker_context(context, context_registry, ledger_path,
                                                      identity_sha, by_id, related_record_id=result_id)
                    else:
                        meta = {}
                    status = "ok" if ok else ("timeout" if result.get("timeout") else "error")
                    observation = {
                        "record_id": result_id, "manifest_id": protocol_id,
                        "attempt_id": attempt_id, "panel_member_id": plan["panel_member_id"],
                        "qasm_sha256": plan["qasm_sha256"],
                        "normalized_qasm_sha256": plan["normalized_qasm_sha256"],
                        "candidate_id": plan["candidate_id"],
                        "candidate_config_sha256": plan["candidate_config_sha256"],
                        "configured_chi": plan["configured_chi"], "shots": plan["shots"],
                        "session": session, "repetition": repetition, "stage": stage,
                        "status": status, "transport_status": transport,
                        "normalization_status": plan["normalization_status"],
                        "execution_eligibility": plan["execution_eligibility"],
                        "execution_reason": plan["execution_reason"],
                        "prediction_status": plan["prediction_status"],
                        "prediction_reason": plan["prediction_reason"],
                        "timing_status": status, "timing_reason": result.get("error", "") if not ok else "",
                        "quality_status": quality_status,
                        "quality_reason": quality_result.get("reason", ""),
                        "quality_value": quality_result.get("fidelity", ""),
                        "selected_clock_id": selected_clock_id,
                        "selected_clock_field": selected_clock_field,
                        selected_clock_field: result.get(selected_clock_field, "")
                        if stage == V3_TIMED_STAGE else "",
                        "host_wall_seconds": result.get("host_wall_seconds", ""),
                        "error": result.get("error", ""),
                        "run_identity_sha256": identity_sha, **meta,
                    }
                    _v2_append_jsonl(ledger_path, observation)
                    by_id[result_id] = observation
            attempt_observations = [row for row in by_id.values()
                                    if row.get("attempt_id") == attempt_id
                                    and row.get("stage") in {V3_UNTIMED_STAGE, V3_TIMED_STAGE}]
            expected_observations = int(measurement["sessions"]) * (
                int(measurement["untimed_calls_per_cell_per_session"])
                + int(measurement["timed_repetitions_per_cell_per_session"])
            )
            if len(attempt_observations) != expected_observations:
                raise V3Unavailable(f"v3_attempt_observation_count_incomplete:{attempt_id}")
            timed = [row for row in attempt_observations if row.get("stage") == V3_TIMED_STAGE]
            if any(row.get("status") == "timeout" for row in attempt_observations):
                timing_status, timing_reason = "timeout", "one_or_more_process_isolated_calls_timed_out"
            elif any(row.get("status") != "ok" for row in attempt_observations) or not timed:
                timing_status, timing_reason = "error", "one_or_more_process_isolated_calls_failed"
            else:
                timing_status, timing_reason = "ok", ""
            terminal_state = v3_terminal_attempt_status(plan, timing_status, quality_status)
            terminal_state["timing_reason"] = timing_reason
            terminal_id = canonical_hash({"manifest_id": protocol_id,
                                          "attempt_id": attempt_id, "stage": "attempt_complete"})
            terminal = {"record_id": terminal_id, "manifest_id": protocol_id,
                        "attempt_id": attempt_id, "stage": "attempt_complete", "status": "measured",
                        **terminal_state, "quality_reason": quality_result.get("reason", ""),
                        "timed_repetitions": len(timed), "run_identity_sha256": identity_sha}
            _v2_append_jsonl(ledger_path, terminal)
            by_id[terminal_id] = terminal
            completed_attempts.add(attempt_id)
            new_complete += 1

    records = _v2_jsonl_records(ledger_path)
    if any(row.get("manifest_id") != protocol_id for row in records):
        raise V3Unavailable("panel_ledger_contains_non_current_protocol_record")
    if any(row.get("run_identity_sha256") != identity_sha for row in records):
        raise V3Unavailable("v3_ledger_run_identity_mismatch")
    terminal_rows = [row for row in records if row.get("stage") == "attempt_complete"]
    completed_ids = [str(row["attempt_id"]) for row in terminal_rows]
    if len(set(completed_ids)) != len(completed_ids):
        raise V3Unavailable("v3_duplicate_completed_attempt")
    pending = set(plan_by_id) - set(completed_ids)
    pilot_complete = (attempt_limit == 10 and all(attempt_id in completed_ids for attempt_id in selected_ids))
    measurement_result: dict[str, Any] = {
        "status": "checkpointed_partial" if pending else "complete",
        "path": str(ledger_path), "sha256": sha256_file(ledger_path), "rows": len(records),
        "attempt_denominator": len(plans), "completed_attempts": len(completed_ids),
        "pending_attempts": len(pending), "new_attempts_completed_this_call": new_complete,
        "attempt_limit": attempt_limit, "attempt_limit_in_run_identity": False,
        "selected_attempt_ids": selected_ids, "panel_pilot_selection": panel_pilot_selection,
        "pilot_scope_complete": pilot_complete,
        "selected_clock_id": selected_clock_id, "selected_clock_field": selected_clock_field,
        "quality_controls": next((row.get("controls", {}) for row in records
                                   if row.get("stage") == "quality_controls"), {}),
        "worker_contexts": list(context_registry.values()),
        "qcsim_calls_started": any(row.get("stage") == "call_started" for row in records),
    }
    if pilot_complete:
        pilot_qa = v3_assess_panel_pilot(records, panel_pilot_selection, measurement)
        measurement_result["pilot_ledger"] = {
            "path": str(ledger_path), "byte_count": ledger_path.stat().st_size,
            "sha256": sha256_file(ledger_path), "attempt_ids": selected_ids,
        }
        measurement_result["pilot_qa"] = pilot_qa
    return measurement_result


def v3_verify_panel_inputs(contract: dict[str, Any], manifest_path: Path,
                           panel_csv_path: Path | None = None):
    """Verify the inherited exact-QASM panel without changing its fold policy."""
    if contract.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("v3_panel_verification_requires_v3_contract")
    base = dict(contract)
    base["manifest_id"] = V2_MANIFEST_ID
    return v2_verify_inputs(base, manifest_path, panel_csv_path)


def partial_load_calibration(resolution: dict[str, Any], manifest_path: Path,
                             configs: dict[str, dict[str, Any]],
                             environment: dict[str, Any]) -> tuple[
                                 FittedCalibration | None, str, str,
                                 dict[str, Any] | None, str
                             ]:
    """Refit only the S6 partial raw ledger and reject every v2/v3 bundle."""
    amendment = resolution["amendment"]
    calibration_dir = ROOT / amendment["outputs"]["calibration"]
    run_path = calibration_dir / "run_manifest.json"
    if not run_path.is_file():
        return None, "partial_calibration_not_available", "", None, ""
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run_sha256 = sha256_file(run_path)
        if run.get("manifest_id") != PARTIAL_MANIFEST_ID:
            raise V3Unavailable("partial_calibration_manifest_id_mismatch")
        if run.get("status") not in {"complete", "complete_with_unavailable_knots"}:
            raise V3Unavailable(f"partial_calibration_not_complete:{run.get('status', 'missing')}")
        if (int(run.get("planned_cells", -1)) != 432
                or int(run.get("completed_cells", -1)) != 399
                or int(run.get("planned_enabled_cells", 399)) != 399
                or int(run.get("planned_quarantined_cells", 33)) != 33):
            raise V3Unavailable("partial_calibration_allocation_or_completion_mismatch")
        for key, expected in (
            ("amendment_identity", resolution["amendment_identity"]),
            ("parent_contract_sha256", resolution["parent_contract_sha256"]),
            ("resolved_contract_sha256", resolution["resolved_contract_sha256"]),
        ):
            if run.get(key) != expected:
                raise V3Unavailable(f"partial_calibration_{key}_mismatch")
        if (run.get("selected_clock_id", run.get("clock_id"))
                != resolution["contract"]["measurement"]["selected_clock_id"]
                or run.get("selected_clock_field")
                != resolution["contract"]["measurement"]["selected_clock_field"]):
            raise V3Unavailable("partial_calibration_selected_clock_mismatch")

        wrapper = run.get("run_identity", {})
        identity = wrapper.get("run_identity", wrapper) if isinstance(wrapper, dict) else {}
        identity_sha = str(run.get("run_identity_sha256", wrapper.get("run_identity_sha256", "")))
        if not isinstance(identity, dict) or not identity_sha or canonical_hash(identity) != identity_sha:
            raise V3Unavailable("partial_calibration_run_identity_hash_mismatch")
        pins = identity.get("pins", identity)
        if identity.get("manifest_id", PARTIAL_MANIFEST_ID) != PARTIAL_MANIFEST_ID:
            raise V3Unavailable("partial_calibration_identity_manifest_mismatch")
        if pins.get("amendment_identity") != resolution["amendment_identity"]:
            raise V3Unavailable("partial_calibration_amendment_identity_mismatch")
        if (pins.get("parent_contract_sha256") != resolution["parent_contract_sha256"]
                or pins.get("resolved_contract_sha256") != resolution["resolved_contract_sha256"]):
            raise V3Unavailable("partial_calibration_resolved_contract_pin_mismatch")
        if pins.get("manifest_sha256") != sha256_file(manifest_path):
            raise V3Unavailable("partial_calibration_manifest_sha256_mismatch")

        expected_code_hashes = environment.get("code_hashes", {})
        actual_code_hashes = pins.get("code_hashes", {})
        required_code_paths = (
            "benchmark_v1/scripts/run_maestro_component_calibration.py",
            "benchmark_v1/scripts/run_maestro_common_panel.py",
            "benchmark_v1/qre_benchmark/maestro_component_v2.py",
            "benchmark_v1/qre_benchmark/maestro_component_predictor.py",
        )
        for code_path in required_code_paths:
            if (not expected_code_hashes.get(code_path)
                    or actual_code_hashes.get(code_path) != expected_code_hashes[code_path]):
                raise V3Unavailable(f"partial_calibration_code_hash_mismatch:{code_path}")
        for key in (
            "python_executable_sha256", "python_version", "platform",
            "maestro_package_tree_sha256", "qiskit_package_tree_sha256",
            "cpu_affinity", "native_thread_environment", "resource_profile",
        ):
            actual = pins.get(key, pins.get("runtime_pins", {}).get(key))
            if environment.get(key) is not None and actual != environment.get(key):
                raise V3Unavailable(f"partial_calibration_environment_pin_mismatch:{key}")

        configuration_records = (pins.get("configuration_records")
                                 or run.get("configuration_records", {}))
        for candidate, chi in (("statevector", None),
                               ("mps_fixed_chi", int(resolution["contract"][
                                   "panel_candidate_configuration"]["mps_configured_chi"]))):
            key = f"{candidate}:{chi}"
            actual = configuration_records.get(key, {})
            expected = str(configs.get(candidate, {}).get("sha256", ""))
            if not expected or actual.get("sha256") != expected:
                raise V3Unavailable(f"partial_calibration_resolved_config_mismatch:{key}")

        raw_meta = run.get("raw_records", {})
        raw_path = _v2_artifact_path(str(raw_meta.get("path", "")), calibration_dir)
        checkpoint_path = _v2_artifact_path(str(run.get("checkpoint_path", "")), calibration_dir)
        if not raw_path.is_file() or not checkpoint_path.is_file():
            raise V3Unavailable("partial_calibration_raw_ledger_or_checkpoint_missing")
        raw_sha = sha256_file(raw_path)
        if raw_sha != raw_meta.get("sha256"):
            raise V3Unavailable("partial_calibration_raw_ledger_hash_mismatch")
        raw_rows = _csv_rows(raw_path)
        if int(raw_meta.get("rows", -1)) != len(raw_rows):
            raise V3Unavailable("partial_calibration_raw_ledger_row_count_mismatch")
        if sha256_file(checkpoint_path) != run.get("checkpoint_sha256"):
            raise V3Unavailable("partial_calibration_checkpoint_hash_mismatch")
        from benchmark_v1.scripts.run_maestro_component_calibration import (
            _partial_read_checkpoint,
        )
        from benchmark_v1.qre_benchmark.maestro_component_v2 import (
            build_partial_v3_plan, build_partial_v3_execution_schedule,
        )
        cells = build_partial_v3_plan(resolution["contract"], resolution["quarantine_policy"])
        schedule = build_partial_v3_execution_schedule(resolution["contract"], cells)
        _, _, skips, budget_stops = _partial_read_checkpoint(checkpoint_path, schedule, identity_sha)
        if len(skips) != 792 or budget_stops:
            raise V3Unavailable("partial_calibration_terminal_ledger_incomplete")
        checkpoint_rows: list[dict[str, str]] = []
        checkpoint_attempt_ids: list[str] = []

        def add_checkpoint_row(row: Any, line_no: int) -> None:
            if not isinstance(row, dict):
                raise V3Unavailable(f"partial_calibration_checkpoint_row_missing:{line_no}")
            if (row.get("manifest_id") != PARTIAL_MANIFEST_ID
                    or row.get("run_identity_sha256") != identity_sha):
                raise V3Unavailable(f"partial_calibration_checkpoint_row_identity_mismatch:{line_no}")
            checkpoint_attempt_ids.append(str(row.get("attempt_id", "")))
            checkpoint_rows.append({
                key: (json.dumps(value, sort_keys=True, separators=(",", ":"))
                      if isinstance(value, (dict, list, tuple))
                      else _v2_checkpoint_value(value))
                for key, value in row.items()
            })

        with checkpoint_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.endswith("\n"):
                    raise V3Unavailable(f"partial_calibration_checkpoint_incomplete_line:{line_no}")
                record = json.loads(line)
                if (record.get("manifest_id") not in (None, PARTIAL_MANIFEST_ID)
                        or record.get("run_identity_sha256") != identity_sha):
                    raise V3Unavailable(f"partial_calibration_checkpoint_identity_mismatch:{line_no}")
                record_type = record.get("record_type")
                if record_type == "attempt_result":
                    add_checkpoint_row(record.get("row"), line_no)
                    continue
                if isinstance(record.get("rows"), list):
                    for row in record["rows"]:
                        add_checkpoint_row(row, line_no)
                    continue
                if record_type in {"slot_allocated", "terminal_skip", "call_started",
                                   "quality_control", "worker_context"}:
                    continue
                raise V3Unavailable(f"partial_calibration_checkpoint_record_unrecognized:{line_no}")
        if checkpoint_rows != raw_rows:
            raise V3Unavailable("partial_calibration_checkpoint_raw_row_mismatch")
        if (len(checkpoint_rows) != int(run.get("checkpoint_attempt_rows", -1))
                or len(raw_rows) != 9_576):
            raise V3Unavailable("partial_calibration_checkpoint_attempt_count_mismatch")
        if (not all(checkpoint_attempt_ids)
                or len(set(checkpoint_attempt_ids)) != len(checkpoint_attempt_ids)):
            raise V3Unavailable("partial_calibration_attempt_ids_missing_or_duplicate")
        attempt_ids_sha = canonical_hash(sorted(checkpoint_attempt_ids))
        if (attempt_ids_sha != run.get("checkpoint_attempt_ids_sha256")
                or attempt_ids_sha != raw_meta.get("attempt_ids_sha256")):
            raise V3Unavailable("partial_calibration_attempt_id_hash_mismatch")

        rank_records = run.get("rank_preflight_records", [])
        if not isinstance(rank_records, list) or not rank_records:
            raise V3Unavailable("partial_calibration_rank_preflight_records_missing")
        evidence_raw = run.get("quarantine_evidence_by_config", {})
        if not isinstance(evidence_raw, dict):
            raise V3Unavailable("partial_calibration_quarantine_evidence_missing")
        evidence: dict[tuple[str, int, int | None], dict[str, Any]] = {}
        for key, value in evidence_raw.items():
            parts = str(key).split(":")
            if len(parts) != 3 or not isinstance(value, dict):
                raise V3Unavailable(f"partial_calibration_quarantine_evidence_key_invalid:{key}")
            evidence[(parts[0], int(parts[1]), None if parts[2] == "None" else int(parts[2]))] = value
        fitted = fit_v3_calibration_rows(
            raw_rows, resolution["contract"], raw_sha,
            expected_run_identity_sha256=identity_sha,
            rank_preflight_records=rank_records,
            partial_resolution=resolution,
            partial_quarantine_evidence=evidence,
        )
        if (fitted.calibration_manifest_id != PARTIAL_MANIFEST_ID
                or fitted.fit_version != PARTIAL_FIT_VERSION):
            raise V3Unavailable("partial_calibration_fit_protocol_mismatch")
        summary = run.get("fit_summary", {})
        if (fitted.fit_diagnostics.get("worker_environment_sha256")
                != summary.get("worker_environment_sha256")
                or fitted.fit_diagnostics.get("worker_context_sha256_by_config_policy")
                != summary.get("worker_context_sha256_by_config_policy")):
            raise V3Unavailable("partial_calibration_worker_context_fit_mismatch")
        if (fitted.clock_id != resolution["contract"]["measurement"]["selected_clock_id"]):
            raise V3Unavailable("partial_calibration_fit_clock_mismatch")
        return fitted, "", raw_sha, run, run_sha256
    except Exception as exc:
        reason = str(exc) if isinstance(exc, V3Unavailable) else f"{type(exc).__name__}:{exc}"
        return None, f"partial_calibration_unavailable:{reason}", "", None, ""


def run_completion_panel(args: argparse.Namespace, overlay: dict[str, Any],
                         partial_resolution: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a v3 panel or its distinct S6 partial amendment under separate identity."""
    manifest_path = Path(args.manifest).resolve()
    partial = partial_resolution is not None
    protocol_id = PARTIAL_MANIFEST_ID if partial else V3_MANIFEST_ID
    if partial:
        if overlay.get("manifest_id") != PARTIAL_MANIFEST_ID:
            raise V3Unavailable("partial_dispatch_overlay_identity_mismatch")
        if (partial_resolution.get("amendment_identity", {}).get("sha256")
                != sha256_file(manifest_path)):
            raise V3Unavailable("partial_dispatch_resolution_manifest_hash_mismatch")
        contract = dict(partial_resolution["contract"])
        contract["outputs"] = dict(overlay["outputs"])
        contract["method_id"] = str(overlay.get("method_id", contract["method_id"]))
        contract["reader_label"] = str(overlay.get("reader_label", contract["reader_label"]))
        contract["variant_id"] = str(overlay.get("variant_id", ""))
        contract["fidelity_class"] = str(overlay.get("fidelity_class", ""))
        contract["claim_boundary"] = str(overlay.get("claim_boundary", ""))
        contract["partial_execution_policy"] = {
            key: overlay.get(key) for key in (
                "preparation_policy", "calibration_policy", "panel_policy", "deadline_policy",
            )
        }
        contract["panel_run_manifest_id"] = PARTIAL_MANIFEST_ID
        contract["partial_resolved_contract_sha256"] = partial_resolution["resolved_contract_sha256"]
        contract["partial_amendment_sha256"] = partial_resolution["amendment_identity"]["sha256"]
        contract["partial_quarantine_policy"] = partial_resolution["quarantine_policy"]
    else:
        contract = resolve_v3_contract(manifest_path)
        if overlay.get("manifest_id") != V3_MANIFEST_ID:
            raise V3Unavailable("v3_dispatch_overlay_identity_mismatch")
    if args.dry_run and args.measure:
        raise SystemExit("--dry-run and --measure are mutually exclusive")
    if not args.dry_run and not args.measure:
        raise SystemExit(f"{protocol_id} dispatch requires --dry-run or --measure")
    if args.attempt_limit not in (None, 10, 408):
        raise SystemExit(f"{protocol_id} measurement permits only the stratified ten-attempt pilot or full 408-attempt panel")

    output_root = Path(args.output_dir).resolve()
    expected_root = (ROOT / contract["outputs"]["root"]).resolve()
    if output_root != expected_root:
        raise SystemExit(f"{protocol_id} output must use the single semantic artifact root: {expected_root}")
    panel_path, fixture_path, source_rows, _ = v3_verify_panel_inputs(
        contract, manifest_path, args.panel_csv,
    )
    qasm_root = fixture_path.parent
    try:
        maestro = load_maestro_parser()
        maestro_load_error = ""
    except Exception as exc:
        maestro = None
        maestro_load_error = f"{type(exc).__name__}:{exc}"
    if maestro is None:
        def missing_normalizer(*_args):
            raise V3Unavailable(f"maestro_parser_unavailable:{maestro_load_error}")
        normalizer = missing_normalizer
        configs = {
            candidate: {"status": "unavailable", "requested": {}, "resolved": {}, "sha256": "",
                        "reason": f"maestro_parser_unavailable:{maestro_load_error}"}
            for candidate in V2_CANDIDATE_IDS
        }
    else:
        normalizer = None
        configs = v2_resolve_candidate_configs(contract, maestro)

    rows, normalized_sidecar, features = materialize_v2_panel(
        contract, source_rows, qasm_root, maestro, normalizer=normalizer,
    )
    code_paths = [
        Path(__file__).resolve(),
        ROOT / "benchmark_v1/qre_benchmark/maestro_component_v2.py",
        ROOT / "benchmark_v1/qre_benchmark/maestro_component_predictor.py",
        ROOT / "benchmark_v1/scripts/run_maestro_component_calibration.py",
    ]
    if maestro is None:
        environment = {"status": "unavailable", "reason": maestro_load_error, "code_hashes": {
            str(path.resolve().relative_to(ROOT)): sha256_file(path) for path in code_paths if path.is_file()
        }}
    else:
        try:
            environment = v2_environment_pins(manifest_path, maestro, code_paths)
            environment["status"] = "pass"
        except Exception as exc:
            environment = {"status": "unavailable", "reason": f"{type(exc).__name__}:{exc}"}
    base_for_pins = dict(contract)
    base_for_pins["manifest_id"] = V2_MANIFEST_ID
    input_pins = v2_input_pins(base_for_pins, manifest_path, panel_path, fixture_path, source_rows, qasm_root)
    binary_pins = v2_binary_pins(contract)
    if maestro is not None and environment.get("status") == "pass":
        calibration, calibration_reason, calibration_raw_sha, calibration_run, calibration_run_sha = (
            partial_load_calibration(partial_resolution, manifest_path, configs, environment)
            if partial else v3_load_calibration(contract, manifest_path, configs, environment)
        )
    else:
        calibration, calibration_reason, calibration_raw_sha, calibration_run, calibration_run_sha = (
            None, "panel_environment_unavailable", "", None, ""
        )
    predictions, plans = v3_materialize_predictions(
        contract, rows, features, configs, calibration, calibration_reason,
        sha256_file(manifest_path), calibration_raw_sha,
    )
    try:
        pilot = v2_panel_pilot_selection(plans, normalized_sidecar)
    except Exception as exc:
        pilot = {"status": "unavailable", "reason": f"{type(exc).__name__}:{exc}",
                 "selected_attempt_ids": [], "selected_attempts": [], "coverage": {}}
    test_path = ROOT / "benchmark_v1/tests/test_maestro_common_panel_v2.py"
    test_sha = sha256_file(test_path) if test_path.is_file() else ""
    if partial:
        calibration_run_identity_sha = (
            "" if calibration_run is None else str(calibration_run.get("run_identity_sha256", ""))
        )
        identity = partial_resume_identity(
            partial_resolution, sha256_file(manifest_path), input_pins, environment,
            binary_pins, configs, calibration_run_sha, calibration_raw_sha,
            calibration_run_identity_sha, pilot,
        ) if calibration_run is not None else partial_plan_identity(
            partial_resolution, sha256_file(manifest_path), input_pins, environment,
            binary_pins, configs, pilot,
        )
    else:
        identity = v3_resume_identity(
            contract, sha256_file(manifest_path), input_pins, environment, binary_pins, configs,
            calibration_run_sha, calibration_raw_sha, pilot, test_sha,
        )
    identity_sha = identity["run_identity_sha256"]
    measurement = contract["measurement"]
    payload: dict[str, Any] = {
        "manifest_id": protocol_id,
        "amendment_identity": None if not partial else partial_resolution["amendment_identity"],
        "resolved_contract_sha256": ("" if not partial else partial_resolution["resolved_contract_sha256"]),
        "partial_quarantine_policy": (None if not partial else partial_resolution["quarantine_policy"]),
        "partial_execution_policy": (None if not partial else contract["partial_execution_policy"]),
        "method_id": contract["method_id"], "reader_label": contract["reader_label"],
        "variant_id": contract["variant_id"],
        "fidelity_class": contract.get("fidelity_class", ""),
        "claim_boundary": contract.get("claim_boundary", ""),
        "status": "dry_run_no_files_written" if args.dry_run else f"{protocol_id}_panel_plan_ready_for_acceptance_gate",
        "panel": {
            "rows": len(rows), "unique_source_qasm_hashes": len({row["qasm_sha256"] for row in rows}),
            "candidate_attempt_denominator": int(contract["eligibility"]["assigned_candidate_attempts"]),
            "attempt_rows_planned": len(plans),
            "fold_counts": dict(sorted(Counter(int(row["fold_id"]) for row in rows).items())),
            "split_id": V2_SPLIT_ID, "group_key": contract["panel"]["group_key"],
            "fold_assignment": contract["panel"]["fold_assignment"],
            "panel_csv_sha256": input_pins["panel_csv"]["sha256"],
            "source_fixture_manifest_sha256": input_pins["source_fixture_manifest"]["sha256"],
        },
        "normalization": {
            "status_counts": dict(Counter(row["normalization_status"] for row in rows)),
            "unavailable_members": [
                {"panel_member_id": row["panel_member_id"], "qasm_sha256": row["qasm_sha256"],
                 "reason": row["unavailable_reason"]}
                for row in rows if row["normalization_status"] != "ok"
            ],
            "sidecar_member_count": len(normalized_sidecar),
        },
        "attempt_plan": {
            "rows": len(plans),
            "normalization_status_counts": dict(Counter(row["normalization_status"] for row in plans)),
            "execution_eligibility_counts": dict(Counter(row["execution_eligibility"] for row in plans)),
            "prediction_status_counts": dict(Counter(row["prediction_status"] for row in plans)),
            "timing_status_counts": dict(Counter(row["timing_status"] for row in plans)),
            "quality_status_counts": dict(Counter(row["quality_status"] for row in plans)),
            "metric_eligibility_counts": dict(Counter(str(row["metric_eligibility"]).lower() for row in plans)),
            "selected_clock_id": measurement["selected_clock_id"],
            "selected_clock_field": measurement["selected_clock_field"],
            "attempt_limit_not_in_run_identity": True,
            "pilot_attempt_limit": args.attempt_limit,
            "panel_pilot_selection": pilot,
            "status_schema": V3_PANEL_ATTEMPT_STATUS_SCHEMA,
        },
        "prediction": {
            "status_counts": dict(Counter(row["prediction_status"] for row in predictions)),
            "role": "candidate runtime prediction only; does not control execution eligibility",
        },
        "execution_context": {
            "parent_context": "captured separately; never substituted for worker identity",
            "worker_contexts": "recorded per actual isolated worker when measurement is accepted",
        },
        "inputs": input_pins, "binary_pins": binary_pins,
        "candidate_configs": {key: {k: v for k, v in value.items() if k != "object"}
                              for key, value in sorted(configs.items())},
        "calibration": {
            "status": "available" if calibration is not None else "unavailable",
            "reason": calibration_reason, "raw_records_sha256": calibration_raw_sha,
            "run_manifest_sha256": calibration_run_sha,
            "run_status": "unavailable" if calibration_run is None else calibration_run.get("status"),
            "selected_clock_id": measurement["selected_clock_id"],
            "selected_clock_field": measurement["selected_clock_field"],
            "fit_version": "" if calibration is None else calibration.fit_version,
        },
        "environment_pins": environment,
        "resolved_config_sha256": v2_resolved_config_sha256(configs),
        "implementation_test_sha256": test_sha,
        "resume_identity": identity, "resume_identity_sha256": identity_sha,
        "measurement_started": False, "qcsim_calls_started": False, "gpu_used": False,
    }
    if args.dry_run:
        return payload
    if len(plans) != 408 or len(rows) != 204:
        raise V3Unavailable(f"{protocol_id}_panel_plan_denominator_not_frozen")
    if calibration is None or calibration_run is None:
        raise SystemExit(f"{protocol_id} measurement blocked: {calibration_reason or 'completed calibration unavailable'}")
    try:
        if partial:
            partial_validate_execution_acceptance(
                args.acceptance, partial_resolution, sha256_file(manifest_path), configs,
                environment, calibration_run, calibration_run_sha, calibration_raw_sha,
                pilot, args.attempt_limit, output_root,
            )
        else:
            v3_validate_execution_acceptance(
                args.acceptance, sha256_file(manifest_path), configs, environment,
                calibration_run, calibration_run_sha, calibration_raw_sha, pilot,
                args.attempt_limit, output_root, contract,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{protocol_id} measurement blocked by acceptance/calibration gate: {exc}") from exc
    measurement_root = _v2_artifact_path(contract["outputs"]["measurement"], output_root)
    run_path = measurement_root / "run_manifest.json"
    old_run: dict[str, Any] | None = None
    if measurement_root.exists() and any(measurement_root.iterdir()) and not run_path.is_file():
        raise SystemExit(f"{protocol_id} resume refused: measurement artifacts exist without run_manifest.json")
    if run_path.is_file():
        if not args.resume:
            raise SystemExit(f"refusing existing {protocol_id} panel run without --resume")
        old_run = json.loads(run_path.read_text(encoding="utf-8"))
        try:
            if partial:
                validate_partial_resume_identity(old_run.get("resume_identity", {}), identity)
            else:
                validate_v3_resume_identity(old_run.get("resume_identity", {}), identity)
        except V3Unavailable as exc:
            raise SystemExit(f"{protocol_id} resume refused: {exc}") from exc
    elif args.resume:
        raise SystemExit(f"{protocol_id} resume requested but no measurement/run_manifest.json exists")

    # Enforce immutable derived panel artifacts on resume: identical content is
    # reused, while any unexpected prior bytes stop before timing.
    normalized_root = _v2_artifact_path(contract["outputs"]["normalized_panel"], output_root)
    normalized_root.mkdir(parents=True, exist_ok=True)
    derived_payloads = {
        "normalized_panel.json": normalized_sidecar,
        "row_manifest.csv": rows,
        "predictions.csv": predictions,
        "measurement_plan.csv": plans,
    }
    derived_pins: dict[str, dict[str, Any]] = {}
    for name, value in derived_payloads.items():
        path = normalized_root / name
        if name.endswith(".json"):
            raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            if path.is_file() and sha256_file(path) != sha256_bytes(raw):
                raise SystemExit(f"{protocol_id} immutable panel artifact differs on resume: {path}")
            if not path.is_file():
                _atomic_json_write(path, value)
        else:
            if path.is_file():
                if _csv_rows(path) != [{str(k): _v2_checkpoint_value(v) for k, v in row.items()}
                                       for row in value]:
                    raise SystemExit(f"{protocol_id} immutable panel artifact differs on resume: {path}")
            else:
                _write_csv(path, value)
        derived_pins[name] = {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": len(value)}
    payload["outputs"] = derived_pins
    if old_run is not None:
        old_measurement = old_run.get("measurement", {})
        old_pilot_pin = old_measurement.get("pilot_ledger")
        old_pilot_qa = old_measurement.get("pilot_qa")
        if old_pilot_pin is not None or old_pilot_qa is not None:
            payload["measurement"] = {
                **({"pilot_ledger": old_pilot_pin} if old_pilot_pin is not None else {}),
                **({"pilot_qa": old_pilot_qa} if old_pilot_qa is not None else {}),
            }
    measurement_root.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(run_path, payload)
    measurement_result = v3_measure_plan(
        contract, plans, normalized_sidecar, output_root, args.attempt_limit,
        identity, configs, calibration_run, pilot,
    )
    previous_measurement = (old_run or {}).get("measurement", {})
    if previous_measurement.get("pilot_ledger"):
        measurement_result["pilot_ledger"] = previous_measurement["pilot_ledger"]
    if previous_measurement.get("pilot_qa"):
        measurement_result["pilot_qa"] = previous_measurement["pilot_qa"]
    payload["measurement"] = measurement_result
    payload["measurement_started"] = bool(measurement_result.get("qcsim_calls_started"))
    payload["qcsim_calls_started"] = bool(measurement_result.get("qcsim_calls_started"))
    payload["status"] = measurement_result["status"]
    if measurement_result.get("pilot_scope_complete") and not payload["measurement"].get("pilot_ledger"):
        payload["measurement"]["pilot_ledger"] = measurement_result.get("pilot_ledger")
    _atomic_json_write(run_path, payload)
    return payload


def run_v3_panel(args: argparse.Namespace, overlay: dict[str, Any]) -> dict[str, Any]:
    return run_completion_panel(args, overlay)


def run_partial_panel(args: argparse.Namespace, overlay: dict[str, Any]) -> dict[str, Any]:
    resolution = resolve_partial_contract(Path(args.manifest).resolve())
    return run_completion_panel(args, overlay, resolution)


def v2_validate_quality_controls(results: dict[str, dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Fail closed unless positive, negative, and basis-order controls pass."""
    expected = {"zero_state", "bell_chi32", "bell_chi1", "basis_q0", "basis_q1"}
    missing = sorted(expected - set(results))
    if missing:
        return {"status": "failed", "reason": f"missing_quality_controls:{missing}", "results": results}
    statuses = {
        key: v2_quality_gate_status(float(results[key].get("fidelity", float("nan"))), threshold)
        for key in expected
    }
    positive_ok = statuses["zero_state"] == "ok" and statuses["bell_chi32"] == "ok"
    negative_ok = statuses["bell_chi1"] == "quality_failed" and math.isclose(
        float(results["bell_chi1"].get("fidelity", float("nan"))), 0.5, rel_tol=0.0, abs_tol=1e-6
    )
    basis_ok = (statuses["basis_q0"] == "ok" and statuses["basis_q1"] == "ok"
                and int(results["basis_q0"].get("candidate_argmax_index", -1)) == 1
                and int(results["basis_q1"].get("candidate_argmax_index", -1)) == 2)
    return {
        "status": "pass" if positive_ok and negative_ok and basis_ok else "failed",
        "positive_controls_pass": positive_ok,
        "negative_control_demonstrates_truncation": negative_ok,
        "basis_order_check_pass": basis_ok,
        "threshold": float(threshold),
        "statuses": statuses,
        "results": results,
    }


def v2_resume_identity(manifest_sha256: str, input_pins: dict[str, Any],
                       environment_pins: dict[str, Any], binary_pins: dict[str, Any],
                       configs: dict[str, dict[str, Any]], calibration_sha256: str,
                       calibration_run_sha256: str = "", implementation_test_sha256: str = "",
                       panel_pilot_selection: dict[str, Any] | None = None) -> dict[str, Any]:
    """Identity excludes pilot limits/progress so a gated run resumes in place."""
    return {
        "manifest_sha256": manifest_sha256,
        "input_pins": input_pins,
        "environment_pins": environment_pins,
        "binary_pins": binary_pins,
        "candidate_config_sha256": {name: value.get("sha256", "") for name, value in sorted(configs.items())},
        "calibration_raw_sha256": calibration_sha256,
        "calibration_run_manifest_sha256": calibration_run_sha256,
        "implementation_test_sha256": implementation_test_sha256,
        "selected_clock_id": V2_CLOCK_ID,
        "selected_clock_field": V2_CLOCK_FIELD,
        "split_id": V2_SPLIT_ID,
        "panel_pilot_selection": panel_pilot_selection or {},
    }


def v2_resolve_candidate_configs(manifest: dict[str, Any], maestro: Any) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for spec in v2_candidate_specs(manifest):
        candidate = spec["candidate"]
        requested = {key: spec[key] for key in (
            "candidate", "seed", "shots", "configured_chi", "singular_value_threshold"
        )}
        try:
            resolved = v2_resolve_config(
                maestro, candidate, spec["configured_chi"], spec["seed"],
                float(spec["singular_value_threshold"] or 0.0),
            )
            configs[candidate] = {
                "status": "ok", "requested": requested,
                "resolved": resolved["resolved"], "sha256": resolved["sha256"],
                "object": resolved["object"],
            }
        except Exception as exc:
            configs[candidate] = {
                "status": "unavailable", "requested": requested,
                "resolved": {}, "sha256": "",
                "reason": f"{type(exc).__name__}:{exc}",
            }
    return configs


def v2_binary_pins(manifest: dict[str, Any]) -> dict[str, Any]:
    expected = manifest["implementation"].get("observed_current_binary_baseline", {})
    package_root: Path | None = None
    extension_path: Path | None = None
    try:
        extension_spec = importlib.util.find_spec("maestro.maestro")
        if extension_spec is not None and extension_spec.origin:
            extension_path = Path(extension_spec.origin).resolve()
            package_root = extension_path.parent
    except (ImportError, ModuleNotFoundError, ValueError):
        pass
    paths = {
        "libmaestro.so": None if package_root is None else package_root / "libmaestro.so",
        "maestro.cpython-310-x86_64-linux-gnu.so": extension_path,
    }
    pins: dict[str, Any] = {}
    for name, path in paths.items():
        actual = sha256_file(path) if path is not None and path.is_file() else None
        expected_hash = expected.get(name)
        pins[name] = {
            "path": None if path is None else str(path),
            "expected_baseline_sha256": expected_hash,
            "actual_sha256": actual,
            "matches_baseline": bool(actual and expected_hash and actual == expected_hash),
        }
    pins["status"] = ("pass" if all(pins[name]["matches_baseline"] for name in paths)
                       else "mismatch_or_unavailable")
    pins["historical_binary_identity"] = manifest["implementation"].get("historical_binary_identity", "unknown")
    return pins


def v2_input_pins(manifest: dict[str, Any], manifest_path: Path, panel_path: Path,
                  fixture_path: Path, panel_rows: list[dict[str, str]],
                  qasm_root: Path) -> dict[str, Any]:
    panel = manifest["panel"]
    qasm_files = []
    for row in sorted(panel_rows, key=lambda item: item["basename"]):
        path = qasm_root / row["basename"]
        qasm_files.append({
            "basename": row["basename"],
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "expected_sha256": row["qasm_sha256"],
        })
    return {
        "completion_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "panel_csv": {"path": str(panel_path.resolve()), "sha256": sha256_file(panel_path),
                      "expected_sha256": panel["manifest_csv_sha256"]},
        "source_fixture_manifest": {"path": str(fixture_path.resolve()), "sha256": sha256_file(fixture_path),
                                     "expected_sha256": panel["source_fixture_manifest_sha256"]},
        "source_qasm_files": qasm_files,
    }


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def v2_resolved_config_sha256(configs: dict[str, dict[str, Any]]) -> str:
    return v2_canonical_hash({
        candidate: str(configs.get(candidate, {}).get("sha256", ""))
        for candidate in sorted(V2_CANDIDATE_IDS)
    })


def v2_environment_sha256(environment: dict[str, Any]) -> str:
    return v2_canonical_hash({key: value for key, value in environment.items() if key != "status"})


def v2_merge_pilot_pin(measurement: dict[str, Any], prior_measurement: dict[str, Any] | None) -> dict[str, Any]:
    """Carry the immutable first-ten ledger pin through every later resume."""
    merged = dict(measurement)
    pilot = (prior_measurement or {}).get("pilot_ledger")
    if pilot is not None:
        merged["pilot_ledger"] = pilot
    return merged


def _v2_ledger_prefix(path: Path, byte_count: int) -> tuple[bytes, str]:
    if byte_count < 0 or not path.is_file():
        raise V2Unavailable("pilot_ledger_missing")
    with path.open("rb") as handle:
        prefix = handle.read(byte_count)
    if len(prefix) != byte_count:
        raise V2Unavailable("pilot_ledger_prefix_truncated")
    return prefix, sha256_bytes(prefix)


def v2_validate_execution_acceptance(
    acceptance_path: Path | None, manifest: dict[str, Any], manifest_sha256: str,
    configs: dict[str, dict[str, Any]], environment: dict[str, Any],
    calibration_run: dict[str, Any] | None, calibration_run_sha256: str,
    attempt_limit: int | None, output_dir: Path,
    plans: list[dict[str, Any]],
    panel_pilot_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed on implementation, completed-calibration, and pilot pins."""
    if acceptance_path is None:
        raise V2Unavailable("v2_measure_requires_explicit_acceptance_json")
    if calibration_run is None:
        raise V2Unavailable("v2_measure_requires_completed_calibration")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    calibration_status = str(calibration_run.get("status", ""))
    required = {
        "status": "pass",
        "manifest_id": V2_MANIFEST_ID,
        "manifest_sha256": manifest_sha256,
        "code_hashes": environment.get("code_hashes", {}),
        "calibration_run_manifest_sha256": calibration_run_sha256,
        "calibration_status": calibration_status,
        "first_ten_cell_qa": "pass",
        "selected_clock_id": V2_CLOCK_ID,
        "selected_clock_field": V2_CLOCK_FIELD,
        "resolved_config_sha256": v2_resolved_config_sha256(configs),
        "environment_sha256": v2_environment_sha256(environment),
    }
    if calibration_status not in {"complete", "complete_with_unavailable_knots"}:
        raise V2Unavailable(f"v2_measure_calibration_status_not_complete:{calibration_status}")
    for key, expected in required.items():
        if acceptance.get(key) != expected:
            raise V2Unavailable(f"v2_acceptance_{key}_mismatch_or_missing")

    # The first invocation is exactly the ten-cell pilot.  Any broader or
    # resumed run requires a second coordinator acceptance over the immutable
    # pilot prefix and its independent QA verdict.
    if attempt_limit != 10:
        run_path = output_dir / "run_manifest.json"
        if not run_path.is_file():
            raise V2Unavailable("full_panel_requires_completed_pilot_manifest")
        prior = json.loads(run_path.read_text(encoding="utf-8"))
        pilot = prior.get("measurement", {}).get("pilot_ledger", {})
        pilot_path = Path(str(pilot.get("path", "")))
        if not pilot_path.is_absolute():
            pilot_path = output_dir / pilot_path
        byte_count = int(pilot.get("byte_count", -1))
        _, prefix_hash = _v2_ledger_prefix(pilot_path, byte_count)
        if prefix_hash != pilot.get("sha256"):
            raise V2Unavailable("stored_pilot_ledger_hash_mismatch")
        if acceptance.get("pilot_ledger_sha256") != prefix_hash or acceptance.get("pilot_qa") != "pass":
            raise V2Unavailable("full_panel_acceptance_requires_pilot_ledger_and_qa_pass")
        with pilot_path.open("rb") as handle:
            prefix = handle.read(byte_count).decode("utf-8")
        prefix_rows = [json.loads(line) for line in prefix.splitlines() if line.strip()]
        completed = {row.get("attempt_id") for row in prefix_rows
                     if row.get("stage") == "attempt_complete"}
        selected_pilot = (panel_pilot_selection or {}).get("selected_attempt_ids", [])
        if len(selected_pilot) != 10 or not set(selected_pilot).issubset(completed):
            raise V2Unavailable("pilot_ledger_does_not_complete_stratified_first_ten_executable_attempts")
        selection_record = next((row for row in prefix_rows
                                 if row.get("stage") == "panel_pilot_selection"), None)
        if not selection_record or selection_record.get("selection") != panel_pilot_selection:
            raise V2Unavailable("pilot_ledger_stratified_selection_identity_mismatch")
        selected_terminal_order = [str(row.get("attempt_id", "")) for row in prefix_rows
                                   if row.get("stage") == "attempt_complete"
                                   and row.get("attempt_id") in set(selected_pilot)]
        if selected_terminal_order != selected_pilot:
            raise V2Unavailable("pilot_ledger_stratified_attempt_order_mismatch")
        controls = next((row for row in prefix_rows if row.get("stage") == "quality_controls"), None)
        if not controls or controls.get("status") != "pass":
            raise V2Unavailable("pilot_ledger_missing_passing_quality_controls")
    return acceptance


def run_v2_panel(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    """Build the v2 plan or execute only under the explicit R4 acceptance gate."""
    manifest_path = args.manifest.resolve()
    panel_path, fixture_path, source_rows, fixture_payload = v2_verify_inputs(
        manifest, manifest_path, args.panel_csv
    )
    qasm_root = fixture_path.parent
    try:
        maestro = load_maestro_parser()
        maestro_load_error = ""
    except Exception as exc:
        maestro = None
        maestro_load_error = f"{type(exc).__name__}:{exc}"
    if maestro is None:
        def missing_maestro_normalizer(*_args):
            raise V2Unavailable(f"maestro_parser_unavailable:{maestro_load_error}")
        normalizer = missing_maestro_normalizer
    else:
        normalizer = None
    rows, normalized_sidecar, features = materialize_v2_panel(
        manifest, source_rows, qasm_root, maestro, normalizer=normalizer
    )
    configs = v2_resolve_candidate_configs(manifest, maestro) if maestro is not None else {
        candidate: {"status": "unavailable", "requested": {}, "resolved": {}, "sha256": "",
                    "reason": f"maestro_parser_unavailable:{maestro_load_error}"}
        for candidate in V2_CANDIDATE_IDS
    }
    binary_pins = v2_binary_pins(manifest)
    code_paths = [
        Path(__file__).resolve(),
        ROOT / "benchmark_v1/qre_benchmark/maestro_component_v2.py",
        ROOT / "benchmark_v1/qre_benchmark/maestro_component_predictor.py",
        ROOT / "benchmark_v1/scripts/run_maestro_component_calibration.py",
    ]
    if maestro is None:
        environment = {"status": "unavailable", "reason": maestro_load_error}
    else:
        try:
            environment = v2_environment_pins(manifest_path, maestro, code_paths)
            environment["status"] = "pass"
        except Exception as exc:
            environment = {"status": "unavailable", "reason": f"{type(exc).__name__}:{exc}"}
    input_pins = v2_input_pins(manifest, manifest_path, panel_path, fixture_path, source_rows, qasm_root)
    try:
        parent_execution_context: dict[str, Any] = collect_parent_context(
            backend_threading_policy=_manifest_thread_policy(manifest),
        )
    except Exception as exc:
        parent_execution_context = {
            "context_owner": "parent", "status": "unavailable",
            "reason": f"{type(exc).__name__}:{exc}",
        }
    calibration, calibration_reason, calibration_sha, calibration_run, calibration_run_sha = (
        _v2_load_calibration(manifest, manifest_path, configs, environment)
        if maestro is not None and environment.get("status") == "pass"
        else (None, "calibration_environment_unavailable", "", None, "")
    )
    predictions, plans = v2_materialize_predictions(
        manifest, rows, features, configs, calibration, calibration_reason,
        sha256_file(manifest_path),
    )
    panel_pilot_selection = v2_panel_pilot_selection(plans, normalized_sidecar)
    test_path = ROOT / "benchmark_v1/tests/test_maestro_common_panel_v2.py"
    test_sha = sha256_file(test_path) if test_path.is_file() else "missing"
    identity = v2_resume_identity(
        sha256_file(manifest_path), input_pins, environment, binary_pins, configs,
        calibration_sha, calibration_run_sha, test_sha, panel_pilot_selection,
    )
    identity_sha = v2_canonical_hash(identity)
    fold_counts = dict(sorted(Counter(int(row["fold_id"]) for row in rows).items()))
    candidate_statuses = {
        candidate: dict(Counter(row["prediction_status"] for row in predictions
                                if row["candidate_id"] == candidate))
        for candidate in V2_CANDIDATE_IDS
    }
    payload: dict[str, Any] = {
        "manifest_id": V2_MANIFEST_ID,
        "method_id": manifest["method_id"],
        "reader_label": manifest["reader_label"],
        "variant_id": manifest["variant_id"],
        "status": "dry_run_no_files_written" if args.dry_run else "prediction_plan_complete_no_timing",
        "panel": {
            "rows": len(rows), "unique_source_qasm_hashes": len({row["qasm_sha256"] for row in rows}),
            "candidate_attempt_denominator": int(manifest["panel"]["assigned_candidate_attempts"]),
            "attempt_rows_planned": len(plans), "fold_counts": fold_counts,
            "split_id": V2_SPLIT_ID, "group_key": manifest["panel"]["group_key"],
            "fold_assignment": manifest["panel"]["fold_assignment"],
        },
        "normalization": {
            "status_counts": dict(Counter(row["normalization_status"] for row in rows)),
            "unavailable_members": [
                {"panel_member_id": row["panel_member_id"], "qasm_sha256": row["qasm_sha256"],
                 "reason": row["unavailable_reason"]}
                for row in rows if row["normalization_status"] != "ok"
            ],
            "sidecar_member_count": len(normalized_sidecar),
        },
        "prediction": {
            "status_counts": dict(Counter(row["prediction_status"] for row in predictions)),
            "candidate_status_counts": candidate_statuses,
            "role": "candidate runtime prediction only; selector/oracle metrics not produced",
        },
        "attempt_plan": {
            "rows": len(plans), "status_counts": dict(Counter(row["attempt_plan_status"] for row in plans)),
            "execution_eligibility_counts": dict(Counter(row["execution_eligibility"] for row in plans)),
            "prediction_status_counts": dict(Counter(row["prediction_status"] for row in plans)),
            "quality_status_counts": dict(Counter(row["quality_status"] for row in plans)),
            "selected_clock_id": V2_CLOCK_ID, "selected_clock_field": V2_CLOCK_FIELD,
            "attempt_limit_not_in_run_identity": True,
            "pilot_attempt_limit": args.attempt_limit,
            "panel_pilot_selection": panel_pilot_selection,
        },
        "execution_context": {
            "parent_context": parent_execution_context,
            "worker_contexts": "recorded per actual spawned result; parent context is not execution identity",
        },
        "inputs": input_pins,
        "binary_pins": binary_pins,
        "candidate_configs": {
            candidate: {key: value for key, value in config.items() if key != "object"}
            for candidate, config in sorted(configs.items())
        },
        "calibration": {
            "status": "available" if calibration is not None else "unavailable",
            "reason": calibration_reason,
            "raw_records_sha256": calibration_sha,
            "run_manifest_sha256": calibration_run_sha,
            "run_status": "unavailable" if calibration_run is None else calibration_run.get("status"),
            "selected_clock_id": V2_CLOCK_ID,
            "fit_version": "process_isolated_affine_v2" if calibration is not None else "",
        },
        "environment_pins": environment,
        "resolved_config_sha256": v2_resolved_config_sha256(configs),
        "implementation_test_sha256": test_sha,
        "resume_identity": identity,
        "resume_identity_sha256": identity_sha,
        "measurement_started": False,
        "qcsim_calls_started": False,
        "gpu_used": False,
    }
    if args.dry_run:
        if args.measure:
            raise SystemExit("--dry-run and --measure are mutually exclusive")
        return payload

    output_dir = args.output_dir.resolve()
    existing_run_manifest = output_dir / "run_manifest.json"
    old_run: dict[str, Any] | None = None
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.resume:
            raise SystemExit(f"refusing to overwrite non-empty v2 output directory: {output_dir}")
        if not existing_run_manifest.is_file():
            raise SystemExit("resume refused: existing output has no run_manifest.json")
        old_run = json.loads(existing_run_manifest.read_text(encoding="utf-8"))
        try:
            v2_validate_resume_identity(old_run.get("resume_identity", {}), identity)
        except V2Unavailable as exc:
            raise SystemExit(f"resume refused: {exc}") from exc

    if args.measure:
        if args.attempt_limit not in (10, 408, None):
            raise SystemExit("v2 measurement permits only the ten-attempt QA pilot or the full 408-attempt panel")
        if args.attempt_limit != 10 and old_run is None:
            raise SystemExit("v2 full-panel measurement requires a completed ten-attempt pilot first")
        try:
            acceptance = v2_validate_execution_acceptance(
                args.acceptance, manifest, sha256_file(manifest_path), configs, environment,
                calibration_run, calibration_run_sha, args.attempt_limit, output_dir, plans,
                panel_pilot_selection,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"v2 measurement blocked by acceptance/calibration gate: {exc}") from exc
    elif args.acceptance is not None or args.attempt_limit is not None:
        if not args.dry_run:
            raise SystemExit("--acceptance and --attempt-limit are reserved for v2 --measure")
        acceptance = None
    else:
        acceptance = None

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "normalized_panel.json"
    row_path = output_dir / "row_manifest.csv"
    predictions_path = output_dir / "predictions.csv"
    plan_path = output_dir / "measurement_plan.csv"
    _atomic_json_write(normalized_path, normalized_sidecar)
    _write_csv(row_path, rows)
    _write_csv(predictions_path, predictions)
    _write_csv(plan_path, plans)
    payload["outputs"] = {
        "normalized_panel": {"path": str(normalized_path), "sha256": sha256_file(normalized_path),
                             "rows": len(normalized_sidecar)},
        "row_manifest": {"path": str(row_path), "sha256": sha256_file(row_path), "rows": len(rows)},
        "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path),
                        "rows": len(predictions)},
        "measurement_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path), "rows": len(plans)},
    }
    # A previous pilot prefix is immutable across full-panel resume.  Retain
    # its QA pin while the live ledger grows.
    if old_run is not None and old_run.get("measurement", {}).get("pilot_ledger"):
        payload["measurement"] = old_run["measurement"]
    _atomic_json_write(existing_run_manifest, payload)
    if args.measure:
        measurement_result = v2_measure_plan(
            manifest, plans, normalized_sidecar, output_dir, args.attempt_limit, identity_sha, configs,
            panel_pilot_selection,
        )
        prior_measurement = (old_run or {}).get("measurement", {})
        payload["measurement"] = v2_merge_pilot_pin(measurement_result, prior_measurement)
        payload["measurement_started"] = True
        payload["qcsim_calls_started"] = bool(measurement_result.get("qcsim_calls_started"))
        payload["status"] = measurement_result["status"]
        if not payload["measurement"].get("pilot_ledger") and args.attempt_limit == 10:
            ledger_path = Path(measurement_result["path"])
            if measurement_result.get("pilot_scope_complete"):
                byte_count = ledger_path.stat().st_size
                payload["measurement"]["pilot_ledger"] = {
                    "path": str(ledger_path.resolve()), "byte_count": byte_count,
                    "sha256": sha256_file(ledger_path),
                    "attempt_ids": measurement_result.get("pilot_attempt_ids", []),
                }
        _atomic_json_write(existing_run_manifest, payload)
    return payload


def _manifest_thread_policy(manifest: dict[str, Any]) -> dict[str, Any]:
    measurement = manifest.get("measurement", {})
    return {
        "declared_policy": measurement.get("native_thread_policy", measurement.get("thread_policy")),
        "affinity": measurement.get("cpu_affinity"),
        "native_thread_environment": measurement.get("native_thread_environment"),
        "timing_worker_count": measurement.get("timing_worker_count", 1),
    }


def _v2_mps_statevector_worker(qasm: str, chi: int, seed: int,
                               singular_value_threshold: float,
                               backend_threading_policy: Any,
                               expected_worker_context: dict[str, Any] | None, out: Any) -> None:
    """Return actual QCSim MPS amplitudes in a fresh process; quality only."""
    try:
        modules = _canonical_native_imports()
        maestro = modules["maestro"]
        circuit = maestro.QasmToCirc().parse_and_translate(qasm)
        resolved_config = v2_resolve_config(
            maestro, "mps_fixed_chi", int(chi), int(seed), float(singular_value_threshold)
        )
        config = resolved_config["object"]
        worker_context = collect_worker_context(
            config=resolved_config["resolved"], backend_threading_policy=backend_threading_policy,
        )
        if expected_worker_context is not None:
            validate_worker_context(expected_worker_context, worker_context)
        started = time.perf_counter()
        amplitudes = maestro.get_statevector(circuit, config)
        out.put({"ok": True, "amplitudes": list(amplitudes),
                 "host_wall_seconds": time.perf_counter() - started,
                 "worker_context": worker_context})
    except BaseException as exc:
        out.put({"ok": False, "worker_exception": True, "exception_type": type(exc).__name__,
                 "error": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc(),
                 "worker_context": locals().get("worker_context")})


def _v2_simple_execute_worker(qasm: str, candidate: str, chi: int | None, shots: int,
                              seed: int, singular_value_threshold: float,
                              backend_threading_policy: Any,
                              expected_worker_context: dict[str, Any] | None, out: Any) -> None:
    """One process-isolated QCSim API call, reporting the frozen clock only."""
    try:
        modules = _canonical_native_imports()
        maestro = modules["maestro"]
        resolved_config = v2_resolve_config(
            maestro, candidate, chi, int(seed), float(singular_value_threshold)
        )
        config = resolved_config["object"]
        worker_context = collect_worker_context(
            config=resolved_config["resolved"], backend_threading_policy=backend_threading_policy,
        )
        if expected_worker_context is not None:
            validate_worker_context(expected_worker_context, worker_context)
        started = time.perf_counter()
        result = maestro.simple_execute(qasm, config, shots=int(shots))
        host_wall = time.perf_counter() - started
        selected = result.get("time_taken") if isinstance(result, dict) else None
        value = float(selected)
        if not math.isfinite(value) or value < 0:
            raise V2Unavailable(f"invalid_selected_clock_value:{selected}")
        out.put({"ok": True, V2_CLOCK_FIELD: value,
                 "host_wall_seconds": host_wall, "reported_result_method": result.get("method", ""),
                 "worker_context": worker_context})
    except BaseException as exc:
        out.put({"ok": False, "worker_exception": True, "exception_type": type(exc).__name__,
                 "error": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc(),
                 "worker_context": locals().get("worker_context")})


def _v2_qiskit_statevector(qasm: str) -> tuple[list[complex], float]:
    """Exact FP64 source-reference vector and separate wall duration."""
    from qiskit import qasm2
    from qiskit.quantum_info import Statevector

    started = time.perf_counter()
    circuit = qasm2.loads(qasm, strict=True)
    amplitudes = Statevector.from_instruction(circuit).data
    return list(amplitudes), time.perf_counter() - started


def v2_exact_entanglement_rank(source_unitary_prefix_qasm: str, width: int) -> int:
    """Exact source-QASM Schmidt rank across the central logical-wire cut."""
    if width < 2:
        return 1
    amplitudes, _ = _v2_qiskit_statevector(source_unitary_prefix_qasm)
    import numpy as np

    vector = np.asarray(amplitudes, dtype=np.complex128)
    if vector.size != 2 ** int(width) or not np.isfinite(vector).all():
        raise V2Unavailable("pilot_entanglement_reference_vector_invalid")
    split = int(width) // 2
    # Qiskit statevector basis indices use little-endian logical-wire bits:
    # rows are higher wires, columns are q[0:split].
    matrix = vector.reshape((2 ** (int(width) - split), 2 ** split))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if not singular_values.size or singular_values[0] <= 0:
        raise V2Unavailable("pilot_entanglement_reference_norm_invalid")
    tolerance = max(float(singular_values[0]) * 1e-10, 1e-12)
    return int(np.count_nonzero(singular_values > tolerance))


def v2_panel_pilot_selection(plans: list[dict[str, Any]], normalized_sidecar: list[dict[str, Any]],
                              target: int = 10) -> dict[str, Any]:
    """Choose a deterministic, label-blind executable pilot with coverage pins."""
    eligible = [row for row in plans if row.get("execution_eligibility") == "eligible"]
    if target < 1 or len(eligible) < target:
        raise V2Unavailable(f"panel_pilot_has_fewer_than_{target}_executable_attempts")
    candidate_order = {candidate: index for index, candidate in enumerate(V2_CANDIDATE_IDS)}
    stable_rows = sorted(eligible, key=lambda row: (
        str(row.get("qasm_sha256", "")), str(row.get("panel_member_id", "")),
        candidate_order.get(str(row.get("candidate_id", "")), 99), str(row["attempt_id"]),
    ))
    selected: list[dict[str, Any]] = []
    reasons: dict[str, list[str]] = {}
    entanglement_by_member: dict[str, dict[str, Any]] = {}
    sidecar_by_member = {str(row["panel_member_id"]): row for row in normalized_sidecar}

    def add(row: dict[str, Any], reason: str) -> None:
        attempt_id = str(row["attempt_id"])
        if attempt_id not in reasons:
            selected.append(row)
            reasons[attempt_id] = []
        if reason not in reasons[attempt_id]:
            reasons[attempt_id].append(reason)

    # First ensure each engine is represented, then explicitly cover the
    # lowest supported width, the frozen q16 endpoint, and an exact entangled
    # source input.  Remaining positions are filled in stable hash order.
    for candidate in V2_CANDIDATE_IDS:
        candidate_row = next((row for row in stable_rows if row.get("candidate_id") == candidate), None)
        if candidate_row is not None:
            add(candidate_row, f"candidate:{candidate}")

    min_width = min(int(row.get("width_qubits", 0)) for row in eligible)
    low_row = next(row for row in stable_rows if int(row.get("width_qubits", 0)) == min_width)
    add(low_row, f"low_width:{min_width}")
    q16_row = next((row for row in stable_rows if int(row.get("width_qubits", 0)) == 16), None)
    if q16_row is not None:
        add(q16_row, "high_width:q16")

    entangled_row: dict[str, Any] | None = None
    for row in stable_rows:
        if int(row.get("n_cx_operations", 0) or 0) <= 0:
            continue
        member_id = str(row.get("panel_member_id", ""))
        if member_id not in entanglement_by_member:
            sidecar_row = sidecar_by_member.get(member_id)
            source_prefix = "" if sidecar_row is None else str(
                sidecar_row.get("source_unitary_prefix_qasm", "")
            )
            try:
                rank = v2_exact_entanglement_rank(source_prefix, int(row["width_qubits"]))
            except Exception as exc:
                entanglement_by_member[member_id] = {
                    "status": "unavailable", "reason": f"{type(exc).__name__}:{exc}",
                }
            else:
                entanglement_by_member[member_id] = {
                    "status": "ok", "central_schmidt_rank": rank,
                    "source_qasm_sha256": str(row.get("qasm_sha256", "")),
                }
        if entanglement_by_member[member_id].get("central_schmidt_rank", 1) > 1:
            same_member = [candidate for candidate in stable_rows
                           if candidate.get("panel_member_id") == member_id]
            entangled_row = next((candidate for candidate in same_member
                                  if candidate.get("candidate_id") == "mps_fixed_chi"), row)
            add(entangled_row, "exact_entangled_input")
            break

    for row in stable_rows:
        if len(selected) >= target:
            break
        add(row, "deterministic_fill")
    selected = selected[:target]
    selected_ids = {str(row["attempt_id"]) for row in selected}
    selected_candidates = sorted({str(row.get("candidate_id", "")) for row in selected})
    selected_widths = sorted({int(row.get("width_qubits", 0)) for row in selected})
    min_selected = min(selected_widths) if selected_widths else None
    entangled_selected = entangled_row is not None and str(entangled_row["attempt_id"]) in selected_ids
    coverage = {
        "candidate_ids": selected_candidates,
        "both_engines": set(V2_CANDIDATE_IDS).issubset(selected_candidates),
        "width_qubits": selected_widths,
        "lowest_executable_width": min_width,
        "lowest_width_covered": min_selected == min_width,
        "q16_covered": 16 in selected_widths,
        "exact_entangled_input_covered": entangled_selected,
    }
    if target == 10 and not all((coverage["both_engines"], coverage["lowest_width_covered"],
                                 coverage["q16_covered"], coverage["exact_entangled_input_covered"])):
        raise V2Unavailable(f"panel_pilot_required_coverage_missing:{coverage}")
    selected_attempts = []
    for row in selected:
        item = {
            "attempt_id": str(row["attempt_id"]),
            "panel_member_id": str(row.get("panel_member_id", "")),
            "qasm_sha256": str(row.get("qasm_sha256", "")),
            "candidate_id": str(row.get("candidate_id", "")),
            "width_qubits": int(row.get("width_qubits", 0)),
            "family": str(row.get("family", "")),
            "stratum": str(row.get("stratum", "")),
            "n_cx_operations": int(row.get("n_cx_operations", 0) or 0),
            "selection_reasons": reasons[str(row["attempt_id"])],
        }
        if str(row["attempt_id"]) == ("" if entangled_row is None else str(entangled_row["attempt_id"])):
            item["central_schmidt_rank"] = entanglement_by_member[
                str(row["panel_member_id"])
            ]["central_schmidt_rank"]
        selected_attempts.append(item)
    return {
        "policy_id": "maestro_panel_first_ten_blind_stratified_v1",
        "selected_attempt_ids": [str(row["attempt_id"]) for row in selected],
        "selected_attempts": selected_attempts,
        "coverage": coverage,
        "entanglement_checks": entanglement_by_member,
    }


def v2_run_quality_controls(manifest: dict[str, Any], maestro: Any) -> dict[str, Any]:
    """Run the required small QCSim MPS positive/negative/order controls.

    This function is intentionally called only from the explicit, later-gated
    measurement mode; the dry-run/preflight path never executes QCSim.
    """
    width = 2
    base = ["OPENQASM 2.0;", 'include "qelib1.inc";', "qreg q[2];", "creg c[2];"]
    cases = {
        "zero_state": ([], 32, None),
        "bell_chi32": (["h q[0];", "cx q[0],q[1];"], 32, None),
        "bell_chi1": (["h q[0];", "cx q[0],q[1];"], 1, None),
        "basis_q0": (["x q[0];"], 32, 1),
        "basis_q1": (["x q[1];"], 32, 2),
    }
    quality = manifest["quality_policy"]
    normalization = manifest["normalization"]
    details: dict[str, dict[str, Any]] = {}
    for label, (gates, chi, expected_index) in cases.items():
        source = "\n".join(base + list(gates) + ["measure q -> c;", ""])
        normalized = v2_normalize_qasm(source, width, normalization, maestro)
        exact, reference_wall = _v2_qiskit_statevector(normalized.source_unitary_prefix_qasm)
        result = isolated_call(_v2_mps_statevector_worker, (
            normalized.unitary_prefix_qasm, chi, int(manifest["seed"]),
            float(manifest["panel_candidate_configuration"]["mps_singular_value_threshold"]),
            _manifest_thread_policy(manifest),
            None,
        ), float(manifest["measurement"]["cell_timeout_seconds"]))
        if not result.get("ok"):
            details[label] = {"status": "quality_unavailable", "error": result.get("error", "timeout"),
                              "quality_reference_wall_seconds": reference_wall}
            continue
        candidate = result["amplitudes"]
        fidelity = v2_statevector_reference_fidelity(
            exact, candidate, width, float(quality["vector_norm_atol"])
        )
        import numpy as np
        argmax_index = int(np.argmax(np.abs(np.asarray(candidate, dtype=np.complex128))))
        details[label] = {
            "status": v2_quality_gate_status(fidelity, float(quality["threshold"])),
            "fidelity": fidelity,
            "candidate_argmax_index": argmax_index,
            "expected_argmax_index": expected_index,
            "configured_chi": chi,
            "quality_reference_wall_seconds": reference_wall,
            "candidate_statevector_wall_seconds": result.get("host_wall_seconds", ""),
            "worker_context_sha256": result.get("worker_context", {}).get("identity_sha256", ""),
            "worker_environment_sha256": result.get("worker_context", {}).get("worker_environment_sha256", ""),
            "transport_status": result.get("transport_status", "unknown"),
            "metric": quality["metric"],
        }
    controls = v2_validate_quality_controls(details, float(quality["threshold"]))
    controls["quality_gate_id"] = quality["quality_gate_id"]
    controls["basis_order_check"] = "source-derived exact vector vs QCSim MPS; X(q0)->index1, X(q1)->index2"
    return controls


def _v2_quality_for_attempt(sidecar_row: dict[str, Any], plan: dict[str, Any],
                            manifest: dict[str, Any],
                            expected_worker_context: dict[str, Any] | None = None) -> dict[str, Any]:
    quality = manifest["quality_policy"]
    try:
        exact, reference_wall = _v2_qiskit_statevector(sidecar_row["source_unitary_prefix_qasm"])
        chi = int(plan["configured_chi"])
        candidate = isolated_call(_v2_mps_statevector_worker, (
            sidecar_row["unitary_prefix_qasm"], chi, int(plan["seed"]),
            float(manifest["panel_candidate_configuration"]["mps_singular_value_threshold"]),
            _manifest_thread_policy(manifest),
            expected_worker_context,
        ), float(manifest["measurement"]["cell_timeout_seconds"]))
        if not candidate.get("ok"):
            return {"status": "quality_unavailable", "reason": candidate.get("error", "timeout"),
                    "quality_reference_wall_seconds": reference_wall,
                    "worker_context": candidate.get("worker_context"),
                    "transport_status": candidate.get("transport_status", "unknown")}
        fidelity = v2_statevector_reference_fidelity(
            exact, candidate["amplitudes"], int(plan["width_qubits"]),
            float(quality["vector_norm_atol"]),
        )
        return {
            "status": v2_quality_gate_status(fidelity, float(quality["threshold"])),
            "fidelity": fidelity,
            "quality_reference_wall_seconds": reference_wall,
            "candidate_statevector_wall_seconds": candidate.get("host_wall_seconds", ""),
            "worker_context": candidate.get("worker_context"),
            "transport_status": candidate.get("transport_status", "unknown"),
            "metric": quality["metric"],
        }
    except Exception as exc:
        return {"status": "quality_unavailable", "reason": f"{type(exc).__name__}:{exc}"}


def _v2_append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _v2_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row.get("record_id", ""))
            if not record_id or record_id in seen_ids:
                raise ValueError(f"missing or duplicate JSONL record_id at line {line_number}")
            seen_ids.add(record_id)
            rows.append(row)
    return rows


def v2_record_quality_controls(manifest: dict[str, Any], ledger_path: Path,
                               run_identity_sha256: str) -> dict[str, Any]:
    """Run each control at most once, journaling call starts before QCSim."""
    records = _v2_jsonl_records(ledger_path)
    by_id = {row["record_id"]: row for row in records}
    context_registry = _restore_worker_contexts(records)
    aggregate_id = v2_canonical_hash({"manifest_id": V2_MANIFEST_ID, "stage": "quality_controls"})
    if aggregate_id in by_id:
        return by_id[aggregate_id]["controls"]

    base = ["OPENQASM 2.0;", 'include "qelib1.inc";', "qreg q[2];", "creg c[2];"]
    cases = {
        "zero_state": ([], 32, None),
        "bell_chi32": (["h q[0];", "cx q[0],q[1];"], 32, None),
        "bell_chi1": (["h q[0];", "cx q[0],q[1];"], 1, None),
        "basis_q0": (["x q[0];"], 32, 1),
        "basis_q1": (["x q[1];"], 32, 2),
    }
    details: dict[str, dict[str, Any]] = {}
    for label, (gates, chi, expected_index) in cases.items():
        result_context: dict[str, Any] | None = None
        result_id = v2_canonical_hash({"manifest_id": V2_MANIFEST_ID,
                                       "stage": "quality_control_result", "label": label})
        saved = by_id.get(result_id)
        if saved is not None:
            details[label] = saved["detail"]
            continue
        source = "\n".join(base + list(gates) + ["measure q -> c;", ""])
        try:
            normalized = v2_normalize_qasm(source, 2, manifest["normalization"], load_maestro_parser())
            exact, reference_wall = _v2_qiskit_statevector(normalized.source_unitary_prefix_qasm)
        except Exception as exc:
            detail = {"status": "quality_unavailable", "reason": f"{type(exc).__name__}:{exc}"}
            row = {"record_id": result_id, "stage": "quality_control_result", "label": label,
                   "detail": detail, "status": detail["status"],
                   "run_identity_sha256": run_identity_sha256}
            _v2_append_jsonl(ledger_path, row)
            by_id[result_id] = row
            details[label] = detail
            continue

        call_id = v2_canonical_hash({"record_id": result_id, "stage": "call_started"})
        if call_id in by_id:
            detail = {"status": "quality_unavailable", "reason": "resume_interrupted_quality_control_call",
                      "quality_reference_wall_seconds": reference_wall}
        else:
            maestro = load_maestro_parser()
            resolved_config = v2_resolve_config(
                maestro, "mps_fixed_chi", chi, int(manifest["seed"]),
                float(manifest["panel_candidate_configuration"]["mps_singular_value_threshold"]),
            )["resolved"]
            thread_policy = _manifest_thread_policy(manifest)
            expected_context = context_registry.get(worker_context_config_key(resolved_config, thread_policy))
            marker = {"record_id": call_id, "related_record_id": result_id,
                      "stage": "call_started", "call_kind": "quality_control", "label": label,
                      "run_identity_sha256": run_identity_sha256}
            _v2_append_jsonl(ledger_path, marker)
            by_id[call_id] = marker
            result = isolated_call(_v2_mps_statevector_worker, (
                normalized.unitary_prefix_qasm, chi, int(manifest["seed"]),
                float(manifest["panel_candidate_configuration"]["mps_singular_value_threshold"]),
                thread_policy,
                expected_context,
            ), float(manifest["measurement"]["cell_timeout_seconds"]))
            result_context = result.get("worker_context")
            if not result.get("ok"):
                detail = {"status": "quality_unavailable", "reason": result.get("error", "timeout"),
                          "quality_reference_wall_seconds": reference_wall,
                          "candidate_statevector_wall_seconds": result.get("host_wall_seconds", ""),
                          "worker_context_sha256": result.get("worker_context", {}).get("identity_sha256", ""),
                          "transport_status": result.get("transport_status", "unknown")}
            else:
                fidelity = v2_statevector_reference_fidelity(
                    exact, result["amplitudes"], 2,
                    float(manifest["quality_policy"]["vector_norm_atol"]),
                )
                import numpy as np
                argmax_index = int(np.argmax(np.abs(np.asarray(result["amplitudes"], dtype=np.complex128))))
                detail = {
                    "status": v2_quality_gate_status(fidelity, float(manifest["quality_policy"]["threshold"])),
                    "fidelity": fidelity, "candidate_argmax_index": argmax_index,
                    "expected_argmax_index": expected_index, "configured_chi": chi,
                    "quality_reference_wall_seconds": reference_wall,
                    "candidate_statevector_wall_seconds": result.get("host_wall_seconds", ""),
                    "worker_context_sha256": result.get("worker_context", {}).get("identity_sha256", ""),
                    "transport_status": result.get("transport_status", "unknown"),
                    "metric": manifest["quality_policy"]["metric"],
                }
        row = {"record_id": result_id, "stage": "quality_control_result", "label": label,
               "detail": detail, "status": detail.get("status", "quality_unavailable"),
               "run_identity_sha256": run_identity_sha256}
        _v2_append_jsonl(ledger_path, row)
        by_id[result_id] = row
        details[label] = detail
        _record_worker_context(
            result_context, context_registry, ledger_path, run_identity_sha256, by_id,
            related_record_id=result_id,
        )

    controls = v2_validate_quality_controls(details, float(manifest["quality_policy"]["threshold"]))
    controls["quality_gate_id"] = manifest["quality_policy"]["quality_gate_id"]
    controls["basis_order_check"] = "source-derived exact vector vs QCSim MPS; X(q0)->index1, X(q1)->index2"
    _v2_append_jsonl(ledger_path, {
        "record_id": aggregate_id, "stage": "quality_controls", "status": controls["status"],
        "controls": controls, "selected_clock_id": V2_CLOCK_ID,
        "run_identity_sha256": run_identity_sha256,
    })
    return controls


def _v2_validate_attempt_records(plans: list[dict[str, Any]], records: list[dict[str, Any]],
                                 measurement: dict[str, Any]) -> None:
    by_id = {row["record_id"]: row for row in records}
    plan_ids = {row["attempt_id"] for row in plans}
    terminals = [row for row in records if row.get("stage") == "attempt_complete"]
    terminal_attempts = [str(row.get("attempt_id", "")) for row in terminals]
    if len(terminal_attempts) != len(set(terminal_attempts)):
        raise V2Unavailable("duplicate_attempt_terminal_record")
    for terminal in terminals:
        attempt_id = str(terminal["attempt_id"])
        expected_id = v2_canonical_hash({"attempt_id": attempt_id, "stage": "attempt_complete"})
        if attempt_id not in plan_ids or terminal.get("record_id") != expected_id:
            raise V2Unavailable("attempt_terminal_outside_frozen_plan")
        if terminal.get("status") == "measured":
            for session in range(1, int(measurement["sessions"]) + 1):
                for stage, count in (
                    ("untimed_warmup", int(measurement["untimed_warmups_per_session"])),
                    ("process_isolated_timed_repeat", int(measurement["timed_warm_repetitions_per_session"])),
                ):
                    for repetition in range(count):
                        record_id = v2_canonical_hash({"attempt_id": attempt_id, "session": session,
                                                       "stage": stage, "repetition": repetition})
                        if record_id not in by_id or by_id[record_id].get("stage") != stage:
                            raise V2Unavailable(f"attempt_marked_complete_with_missing_observation:{attempt_id}:{stage}")


def v2_measure_plan(manifest: dict[str, Any], plans: list[dict[str, Any]],
                    normalized_sidecar: list[dict[str, Any]], output_dir: Path,
                    attempt_limit: int | None,
                    run_identity_sha256: str | None = None,
                    resolved_configs: dict[str, dict[str, Any]] | None = None,
                    panel_pilot_selection: dict[str, Any] | None = None) -> dict[str, Any]:
    """Append observations; attempt limits stop only between complete attempts."""
    if len(plans) != int(manifest["panel"]["assigned_candidate_attempts"]):
        raise ValueError("measurement denominator differs from the frozen 408 attempts")
    if attempt_limit not in (None, 10, 408):
        raise ValueError("attempt_limit must be the stratified pilot 10 or full panel 408")
    run_identity_sha256 = run_identity_sha256 or v2_canonical_hash([p["attempt_id"] for p in plans])
    ledger_path = output_dir / "measurement/raw_records.jsonl"
    sidecar_by_member = {row["panel_member_id"]: row for row in normalized_sidecar}
    panel_pilot_selection = panel_pilot_selection or v2_panel_pilot_selection(plans, normalized_sidecar)
    eligible_ids = [p["attempt_id"] for p in plans
                    if p.get("execution_eligibility") == "eligible"]
    selected_ids = (eligible_ids if attempt_limit is None
                    else (eligible_ids if attempt_limit == 408
                          else list(panel_pilot_selection["selected_attempt_ids"])))
    if attempt_limit == 10 and len(selected_ids) != 10:
        raise V2Unavailable("panel_pilot_selection_count_mismatch")
    selected_id_set = set(selected_ids)
    new_complete = 0
    try:
        with maestro_timing_lock(V2_MANIFEST_ID, run_identity_sha256, root=ROOT):
            records = _v2_jsonl_records(ledger_path)
            if any(row.get("run_identity_sha256") != run_identity_sha256 for row in records):
                raise SystemExit("resume refused: ledger run identity mismatch")
            selection_record_id = v2_canonical_hash({
                "stage": "panel_pilot_selection", "run_identity_sha256": run_identity_sha256,
            })
            by_id = {row["record_id"]: row for row in records}
            saved_selection = by_id.get(selection_record_id)
            if saved_selection is not None:
                if saved_selection.get("selection") != panel_pilot_selection:
                    raise V2Unavailable("panel_pilot_selection_resume_mismatch")
            else:
                selection_record = {
                    "record_id": selection_record_id, "stage": "panel_pilot_selection",
                    "selection": panel_pilot_selection,
                    "run_identity_sha256": run_identity_sha256,
                }
                _v2_append_jsonl(ledger_path, selection_record)
            controls = v2_record_quality_controls(manifest, ledger_path, run_identity_sha256)
            if controls.get("status") != "pass":
                raise SystemExit("v2 measurement blocked: required zero/Bell/basis-order controls did not pass")

            records = _v2_jsonl_records(ledger_path)
            by_id = {row["record_id"]: row for row in records}
            context_registry = _restore_worker_contexts(records)
            started_call_ids = {row["record_id"] for row in records if row.get("stage") == "call_started"}
            completed_attempts = {row["attempt_id"] for row in records if row.get("stage") == "attempt_complete"}
            specs = {row["candidate"]: row for row in v2_candidate_specs(manifest)}
            if attempt_limit == 10:
                plan_by_id = {str(plan["attempt_id"]): plan for plan in plans}
                ordered_plans = [plan for plan in plans
                                 if plan.get("execution_eligibility") != "eligible"]
                ordered_plans.extend(plan_by_id[attempt_id] for attempt_id in selected_ids)
                ordered_plans.extend(plan for plan in plans
                                     if plan.get("execution_eligibility") == "eligible"
                                     and str(plan["attempt_id"]) not in selected_id_set)
            else:
                ordered_plans = plans
            for plan in ordered_plans:
                attempt_id = plan["attempt_id"]
                if attempt_id in completed_attempts:
                    continue
                if plan.get("execution_eligibility") != "eligible":
                    terminal = {
                        "record_id": v2_canonical_hash({"attempt_id": attempt_id, "stage": "attempt_complete"}),
                        "attempt_id": attempt_id, "stage": "attempt_complete",
                        "status": "not_executable",
                        "normalization_status": plan.get("normalization_status", "unknown"),
                        "execution_eligibility": "unavailable",
                        "execution_reason": plan.get("execution_reason", "execution_not_eligible"),
                        "prediction_status": plan.get("prediction_status", "unavailable"),
                        "prediction_reason": plan.get("prediction_reason", ""),
                        "timing_status": "not_started",
                        "timing_reason": plan.get("execution_reason", "execution_not_eligible"),
                        "quality_status": plan.get("quality_status", "quality_unavailable"),
                        "quality_reason": plan.get("quality_reason", "execution_not_eligible"),
                        "metric_eligibility": False,
                        "metric_ineligibility_reason": "execution_unavailable",
                        "selected_clock_id": V2_CLOCK_ID,
                        "run_identity_sha256": run_identity_sha256,
                    }
                    _v2_append_jsonl(ledger_path, terminal)
                    by_id[terminal["record_id"]] = terminal
                    completed_attempts.add(attempt_id)
                    new_complete += 1
                    continue
                if attempt_id not in selected_id_set:
                    continue

                quality_result: dict[str, Any] = {"status": "not_required"}
                if plan["candidate_id"] == "mps_fixed_chi":
                    quality_id = v2_canonical_hash({"attempt_id": attempt_id, "stage": "quality_reference"})
                    saved_quality = by_id.get(quality_id)
                    if saved_quality is not None:
                        quality_result = saved_quality["quality_result"]
                    else:
                        call_id = v2_canonical_hash({"record_id": quality_id, "stage": "call_started"})
                        if call_id in started_call_ids:
                            quality_result = {"status": "quality_unavailable", "reason": "resume_interrupted_call"}
                        else:
                            marker = {"record_id": call_id, "related_record_id": quality_id,
                                      "attempt_id": attempt_id, "stage": "call_started",
                                      "call_kind": "quality_reference", "run_identity_sha256": run_identity_sha256}
                            _v2_append_jsonl(ledger_path, marker)
                            started_call_ids.add(call_id)
                            quality_result = _v2_quality_for_attempt(
                                sidecar_by_member[plan["panel_member_id"]], plan, manifest,
                                expected_worker_context=_expected_worker_context(
                                    resolved_configs, "mps_fixed_chi", context_registry,
                                    _manifest_thread_policy(manifest),
                                ),
                            )
                        context_meta = _record_worker_context(
                            quality_result.get("worker_context"), context_registry, ledger_path,
                            run_identity_sha256, by_id, related_record_id=quality_id,
                        )
                        quality_record = {
                            "record_id": quality_id, "attempt_id": attempt_id, "stage": "quality_reference",
                            "panel_member_id": plan["panel_member_id"], "qasm_sha256": plan["qasm_sha256"],
                            "candidate_id": plan["candidate_id"],
                            "candidate_config_sha256": plan["candidate_config_sha256"],
                            "quality_gate_id": plan["quality_gate_id"],
                            "quality_metric": manifest["quality_policy"]["metric"],
                            "quality_threshold": manifest["quality_policy"]["threshold"],
                            "quality_value": quality_result.get("fidelity", ""),
                            "quality_result": {key: value for key, value in quality_result.items()
                                                if key != "worker_context"},
                            "status": quality_result.get("status", "quality_unavailable"),
                            "quality_status": quality_result.get("status", "quality_unavailable"),
                            "timing_status": "not_started",
                            "metric_eligibility": False,
                            "metric_ineligibility_reason": "timing_not_yet_observed",
                            **context_meta,
                            "selected_clock_id": V2_CLOCK_ID,
                            "run_identity_sha256": run_identity_sha256,
                        }
                        _v2_append_jsonl(ledger_path, quality_record)
                        by_id[quality_id] = quality_record

                quality_status = (
                    "not_required" if plan["candidate_id"] == "statevector"
                    else ("ok" if quality_result.get("status") == "ok"
                          else ("quality_failed" if quality_result.get("status") == "quality_failed"
                                else "quality_unavailable"))
                )

                spec = specs[plan["candidate_id"]]
                for session in range(1, int(manifest["measurement"]["sessions"]) + 1):
                    stages = [("untimed_warmup", rep) for rep in range(
                        int(manifest["measurement"]["untimed_warmups_per_session"]))]
                    stages += [("process_isolated_timed_repeat", rep) for rep in range(
                        int(manifest["measurement"]["timed_warm_repetitions_per_session"]))]
                    for stage, repetition in stages:
                        result_id = v2_canonical_hash({"attempt_id": attempt_id, "session": session,
                                                       "stage": stage, "repetition": repetition})
                        if result_id in by_id:
                            continue
                        call_id = v2_canonical_hash({"record_id": result_id, "stage": "call_started"})
                        if call_id in started_call_ids:
                            result = {"ok": False, "error": "resume_interrupted_call",
                                      "status": "interrupted_unknown"}
                        else:
                            marker = {"record_id": call_id, "related_record_id": result_id,
                                      "attempt_id": attempt_id, "stage": "call_started", "call_kind": stage,
                                      "run_identity_sha256": run_identity_sha256}
                            _v2_append_jsonl(ledger_path, marker)
                            started_call_ids.add(call_id)
                            result = isolated_call(_v2_simple_execute_worker, (
                                sidecar_by_member[plan["panel_member_id"]]["normalized_qasm"],
                                spec["candidate"], spec["configured_chi"], spec["shots"], spec["seed"],
                                float(spec["singular_value_threshold"] or 0.0),
                                _manifest_thread_policy(manifest),
                                _expected_worker_context(
                                    resolved_configs, spec["candidate"], context_registry,
                                    _manifest_thread_policy(manifest),
                                ),
                            ), float(plan["cell_timeout_seconds"]))
                        ok = bool(result.get("ok"))
                        observation = {
                            "record_id": result_id, "attempt_id": attempt_id,
                            "panel_member_id": plan["panel_member_id"], "qasm_sha256": plan["qasm_sha256"],
                            "normalized_qasm_sha256": plan["normalized_qasm_sha256"],
                            "candidate_id": plan["candidate_id"],
                            "candidate_config_sha256": plan["candidate_config_sha256"],
                            "configured_chi": plan["configured_chi"], "shots": plan["shots"],
                            "session": session, "repetition": repetition, "stage": stage,
                            "status": "ok" if ok else ("timeout" if result.get("timeout") else "error"),
                            "normalization_status": plan.get("normalization_status", "ok"),
                            "execution_eligibility": plan.get("execution_eligibility", "eligible"),
                            "execution_reason": plan.get("execution_reason", ""),
                            "prediction_status": plan.get("prediction_status", "unavailable"),
                            "prediction_reason": plan.get("prediction_reason", ""),
                            "timing_status": "ok" if ok else ("timeout" if result.get("timeout") else "error"),
                            "timing_reason": result.get("error", "") if not ok else "",
                            "selected_clock_id": V2_CLOCK_ID, "selected_clock_field": V2_CLOCK_FIELD,
                            V2_CLOCK_FIELD: result.get(V2_CLOCK_FIELD, "") if stage == "process_isolated_timed_repeat" else "",
                            "host_wall_seconds": result.get("host_wall_seconds", ""),
                            "quality_gate_id": plan["quality_gate_id"],
                            "quality_value": quality_result.get("fidelity", ""),
                            "quality_status": quality_status,
                            "quality_reason": quality_result.get("reason", ""),
                            "metric_eligibility": False,
                            "metric_ineligibility_reason": "attempt_timing_summary_pending",
                            "transport_status": result.get("transport_status", "unknown"),
                            "worker_context_sha256": result.get("worker_context", {}).get("identity_sha256", ""),
                            "worker_pid": result.get("worker_context", {}).get("observation", {}).get("pid", ""),
                            "worker_environment_sha256": result.get("worker_context", {}).get("worker_environment_sha256", ""),
                            "resolved_config_sha256": result.get("worker_context", {}).get("resolved_config_sha256", ""),
                            "backend_threading_policy_sha256": result.get("worker_context", {}).get("backend_threading_policy_sha256", ""),
                            "error": result.get("error", ""), "run_identity_sha256": run_identity_sha256,
                        }
                        _v2_append_jsonl(ledger_path, observation)
                        by_id[result_id] = observation
                        _record_worker_context(
                            result.get("worker_context"), context_registry, ledger_path,
                            run_identity_sha256, by_id, related_record_id=result_id,
                        )
                observations = [row for row in by_id.values()
                                if row.get("attempt_id") == attempt_id
                                and row.get("stage") in {"untimed_warmup", "process_isolated_timed_repeat"}]
                timed_observations = [row for row in observations
                                      if row.get("stage") == "process_isolated_timed_repeat"]
                timing_status = "ok"
                timing_reason = ""
                if not timed_observations or any(row.get("status") == "timeout" for row in observations):
                    timing_status = "timeout"
                    timing_reason = "one_or_more_process_isolated_calls_timed_out"
                elif any(row.get("status") != "ok" for row in observations):
                    timing_status = "error"
                    timing_reason = "one_or_more_process_isolated_calls_failed"
                metric_eligible = (
                    plan.get("prediction_status") == "available"
                    and timing_status == "ok"
                    and quality_status in {"not_required", "ok"}
                )
                metric_reason = "" if metric_eligible else (
                    plan.get("prediction_reason") if plan.get("prediction_status") != "available"
                    else ("timing_not_ok" if timing_status != "ok" else "quality_not_ok")
                )
                terminal = {
                    "record_id": v2_canonical_hash({"attempt_id": attempt_id, "stage": "attempt_complete"}),
                    "attempt_id": attempt_id, "stage": "attempt_complete", "status": "measured",
                    "normalization_status": plan.get("normalization_status", "ok"),
                    "execution_eligibility": "eligible", "execution_reason": "",
                    "prediction_status": plan.get("prediction_status", "unavailable"),
                    "prediction_reason": plan.get("prediction_reason", ""),
                    "timing_status": timing_status, "timing_reason": timing_reason,
                    "quality_status": quality_status,
                    "quality_reason": quality_result.get("reason", ""),
                    "metric_eligibility": metric_eligible,
                    "metric_ineligibility_reason": metric_reason,
                    "selected_clock_id": V2_CLOCK_ID, "run_identity_sha256": run_identity_sha256,
                }
                _v2_append_jsonl(ledger_path, terminal)
                completed_attempts.add(attempt_id)
                new_complete += 1
    except BaseException:
        raise

    ledger = _v2_jsonl_records(ledger_path)
    if any(row.get("run_identity_sha256") != run_identity_sha256 for row in ledger):
        raise V2Unavailable("ledger_run_identity_mismatch")
    _v2_validate_attempt_records(plans, ledger, manifest["measurement"])
    completed_ids = {row["attempt_id"] for row in ledger if row.get("stage") == "attempt_complete"}
    pending_ids = {row["attempt_id"] for row in plans} - completed_ids
    pilot_complete = attempt_limit == 10 and len(selected_ids) == 10 and set(selected_ids).issubset(completed_ids)
    control = next((row for row in ledger if row.get("stage") == "quality_controls"), {})
    return {
        "status": "checkpointed_partial" if pending_ids else "complete",
        "path": str(ledger_path.resolve()), "sha256": sha256_file(ledger_path), "rows": len(ledger),
        "attempt_denominator": len(plans), "completed_attempts": len(completed_ids),
        "pending_attempts": len(pending_ids), "new_attempts_completed_this_call": new_complete,
        "attempt_limit": attempt_limit, "attempt_limit_in_run_identity": False,
        "selected_attempt_ids": selected_ids, "pilot_scope_complete": pilot_complete,
        "pilot_attempt_ids": selected_ids if pilot_complete else [],
        "panel_pilot_selection": panel_pilot_selection,
        "quality_controls": control.get("controls", {}),
        "worker_contexts": list(context_registry.values()),
        "qcsim_calls_started": any(row.get("stage") == "call_started" for row in ledger),
        "selected_clock_id": V2_CLOCK_ID, "selected_clock_field": V2_CLOCK_FIELD,
        "process_isolated_per_api_call": True,
    }


def _execute_worker(qasm: str, candidate_id: str, chi: int | None, shots: int,
                    seed: int, backend_threading_policy: Any,
                    expected_worker_context: dict[str, Any] | None, out: Any) -> None:
    try:
        modules = _canonical_native_imports()
        maestro = modules["maestro"]
        kwargs: dict[str, Any] = {
            "simulator_type": maestro.SimulatorType.QCSim,
            "simulation_type": (maestro.SimulationType.Statevector if candidate_id == "qcsim_statevector_cpu"
                                else maestro.SimulationType.MatrixProductState),
            "seed": seed,
        }
        if candidate_id == "qcsim_mps_cpu_chi32":
            kwargs.update(max_bond_dimension=chi, singular_value_threshold=1e-8)
        config = maestro.SimulatorConfig(**kwargs)
        worker_context = collect_worker_context(
            config=kwargs, backend_threading_policy=backend_threading_policy,
        )
        if expected_worker_context is not None:
            validate_worker_context(expected_worker_context, worker_context)
        start = time.perf_counter()
        result = maestro.simple_execute(qasm, config, shots=shots)
        out.put({"ok": True, "result": dict(result), "wall_seconds": time.perf_counter() - start,
                 "worker_context": worker_context})
    except BaseException as exc:
        out.put({"ok": False, "worker_exception": True, "exception_type": type(exc).__name__,
                 "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                 "worker_context": locals().get("worker_context")})


def _quality_worker(qasm: str, chi: int, shots: int, seed: int,
                    backend_threading_policy: Any,
                    expected_worker_context: dict[str, Any] | None, out: Any) -> None:
    try:
        modules = _canonical_native_imports()
        maestro = modules["maestro"]
        circuit = maestro.QasmToCirc().parse_and_translate(qasm)
        config = maestro.SimulatorConfig(
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=chi,
            singular_value_threshold=1e-8,
            seed=seed,
        )
        config_values = {
            "simulator_type": "QCSim",
            "simulation_type": "MatrixProductState",
            "max_bond_dimension": chi,
            "singular_value_threshold": 1e-8,
            "seed": seed,
        }
        worker_context = collect_worker_context(
            config=config_values, backend_threading_policy=backend_threading_policy,
        )
        if expected_worker_context is not None:
            validate_worker_context(expected_worker_context, worker_context)
        start = time.perf_counter()
        fidelity = float(maestro.mirror_fidelity(circuit, config, shots=shots))
        out.put({"ok": True, "fidelity": fidelity, "wall_seconds": time.perf_counter() - start,
                 "worker_context": worker_context})
    except BaseException as exc:
        out.put({"ok": False, "worker_exception": True, "exception_type": type(exc).__name__,
                 "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                 "worker_context": locals().get("worker_context")})


def _isolated_worker_entry(worker: Any, args: tuple[Any, ...], result_queue: Any) -> None:
    """Turn uncaught worker exceptions into typed, drainable result payloads."""
    try:
        worker(*args, result_queue)
    except BaseException as exc:  # workers must not lose their exception class
        try:
            result_queue.put({
                "ok": False,
                "worker_exception": True,
                "exception_type": type(exc).__name__,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
        except BaseException:
            # The parent will classify a missing payload using the real exitcode.
            pass


def _reap_isolated_process(process: Any) -> dict[str, Any]:
    cleanup = {"terminated": False, "killed": False, "reaped": False}
    if process.is_alive():
        cleanup["terminated"] = True
        try:
            process.terminate()
        except (OSError, ValueError):
            pass
        process.join(0.5)
    if process.is_alive() and hasattr(process, "kill"):
        cleanup["killed"] = True
        try:
            process.kill()
        except (OSError, ValueError):
            pass
        process.join(1.0)
    cleanup["reaped"] = not process.is_alive()
    return cleanup


def _finish_isolated_call(result: dict[str, Any], process: Any, started: float,
                          transport_status: str,
                          cleanup: dict[str, Any] | None = None) -> dict[str, Any]:
    result["transport_status"] = transport_status
    result["child_exitcode"] = process.exitcode
    result["worker_pid"] = process.pid
    result["elapsed_seconds"] = time.monotonic() - started
    result["process_cleanup"] = cleanup or {"terminated": False, "killed": False,
                                             "reaped": not process.is_alive()}
    if transport_status != "ok":
        result["ok"] = False
    return result


def isolated_call(worker, args: tuple[Any, ...], timeout: float) -> dict[str, Any]:
    """Spawn one worker, drain its payload, then join under one deadline.

    Queue payloads are consumed while the child is still alive so large results
    cannot deadlock on the child queue-feeder thread during process shutdown.
    ``timeout`` covers startup, payload wait, and normal join; forced cleanup is
    bounded separately so a timed-out child is always terminated/reaped.
    """
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("isolated_call timeout must be a positive finite number")
    started = time.monotonic()
    deadline = started + timeout
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_isolated_worker_entry,
                              args=(worker, tuple(args), result_queue))
    try:
        process.start()
    except BaseException as exc:
        try:
            result_queue.close()
        except Exception:
            pass
        try:
            process.close()
        except (AttributeError, OSError, ValueError):
            pass
        return {
            "ok": False, "transport_status": "spawn_error",
            "exception_type": type(exc).__name__, "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started,
            "process_cleanup": {"terminated": False, "killed": False, "reaped": True},
        }

    payload: Any = None
    payload_received = False
    terminal_status = ""
    cleanup: dict[str, Any] | None = None
    try:
        while not payload_received:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminal_status = "timeout"
                cleanup = _reap_isolated_process(process)
                break
            try:
                payload = result_queue.get(timeout=min(remaining, 0.05))
                payload_received = True
            except queue_module.Empty:
                if not process.is_alive():
                    terminal_status = "hard_exit" if process.exitcode not in (0, None) else "missing_payload"
                    break
            except (EOFError, OSError, ValueError) as exc:
                terminal_status = "transport_error"
                payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                cleanup = _reap_isolated_process(process)
                break

        if payload_received:
            remaining = deadline - time.monotonic()
            process.join(max(0.0, remaining))
            if process.is_alive():
                terminal_status = "exit_timeout_after_payload"
                cleanup = _reap_isolated_process(process)
            elif process.exitcode not in (0, None):
                terminal_status = "nonzero_exit_after_payload"
            elif isinstance(payload, dict) and payload.get("worker_exception"):
                terminal_status = "worker_exception"
            elif isinstance(payload, dict) and not payload.get("ok", True):
                terminal_status = "worker_error"
            else:
                terminal_status = "ok"
    finally:
        if process.is_alive():
            cleanup = _reap_isolated_process(process)
        try:
            result_queue.close()
            result_queue.join_thread()
        except (AttributeError, OSError, ValueError):
            pass

    if isinstance(payload, dict):
        result = dict(payload)
    elif payload_received:
        result = {"ok": True, "value": payload}
    else:
        result = {}
    if terminal_status == "timeout":
        result.setdefault("error", f"timeout after {timeout:g}s")
        result["timeout"] = True
    elif terminal_status == "hard_exit":
        result.setdefault("error", f"child exited without payload (exitcode={process.exitcode})")
        result["hard_exit"] = True
    elif terminal_status == "missing_payload":
        result.setdefault("error", "child exited successfully without a result payload")
    elif terminal_status == "exit_timeout_after_payload":
        result.setdefault("error", f"child did not exit before {timeout:g}s deadline after sending payload")
        result["timeout"] = True
    elif terminal_status == "nonzero_exit_after_payload":
        result.setdefault("error", f"child exited with code {process.exitcode} after sending payload")
        result["hard_exit"] = True
    finished = _finish_isolated_call(result, process, started, terminal_status, cleanup)
    try:
        process.close()
    except (AttributeError, OSError, ValueError):
        pass
    return finished


def measure_plan(manifest: dict[str, Any], plan_rows: list[dict[str, Any]],
                 row_manifest: list[dict[str, Any]], qasm_root: Path, output_dir: Path,
                 maestro: Any) -> list[dict[str, Any]]:
    """Run optional local candidate label measurements; never called by preflight."""
    lookup = {row["panel_member_id"]: row for row in row_manifest}
    candidate_lookup = {row["candidate_id"]: row for row in manifest["candidates"]}
    mps_quality: dict[str, dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []
    timeout = float(manifest["measurement"]["cell_timeout_seconds"])
    quality = manifest["quality_policy"]
    mps_rows = [row for row in plan_rows
                if row["candidate_id"] == "qcsim_mps_cpu_chi32"
                and row["prediction_status"] == "predicted"]

    # Complete the independent fixed-chi quality screen before any candidate
    # warm timing; fidelity and its wall clock stay outside the selected target.
    for plan in mps_rows:
        row = lookup[plan["panel_member_id"]]
        qasm = (qasm_root / row["basename"]).read_text(encoding="utf-8")
        config = candidate_lookup[plan["candidate_id"]]
        result = isolated_call(_quality_worker, (
            qasm, int(config["configured_chi"]), int(quality["reference_shots"]),
            int(config["seed"]), _manifest_thread_policy(manifest), None,
        ), timeout)
        status = "error"
        fidelity = ""
        if result.get("ok"):
            fidelity = float(result["fidelity"])
            status = "ok" if fidelity >= float(quality["threshold"]) else "quality_failed"
        elif result.get("timeout"):
            status = "timeout"
        mps_quality[plan["panel_member_id"]] = {"status": status, "fidelity": fidelity, **result}
        raw_rows.append({
            "record_id": canonical_hash({"attempt_id": plan["attempt_id"], "stage": "quality_reference"}),
            "panel_id": plan["panel_id"],
            "panel_member_id": plan["panel_member_id"],
            "qasm_sha256": plan["qasm_sha256"],
            "fold_id": plan["fold_id"],
            "candidate_id": plan["candidate_id"],
            "candidate_config_sha256": plan["candidate_config_sha256"],
            "configured_chi": plan["configured_chi"],
            "stage": "quality_reference",
            "session": "",
            "repetition": 0,
            "status": status,
            "quality_gate_id": plan["quality_gate_id"],
            "quality_metric": "mirror_fidelity",
            "quality_value": fidelity,
            "quality_threshold": quality["threshold"],
            "quality_reference_shots": quality["reference_shots"],
            "quality_reference_wall_seconds": result.get("wall_seconds", ""),
            "selected_clock_id": manifest["measurement"]["selected_clock_id"],
            "reported_time_seconds": "",
            "host_wall_seconds": "",
            "error": result.get("error", ""),
        })

    for plan in plan_rows:
        candidate_id = plan["candidate_id"]
        member_id = plan["panel_member_id"]
        if plan["prediction_status"] != "predicted":
            raw_rows.append({
                "record_id": canonical_hash({"attempt_id": plan["attempt_id"], "stage": "warm_execute"}),
                "panel_id": plan["panel_id"], "panel_member_id": member_id,
                "qasm_sha256": plan["qasm_sha256"], "fold_id": plan["fold_id"],
                "candidate_id": candidate_id, "candidate_config_sha256": plan["candidate_config_sha256"],
                "configured_chi": plan["configured_chi"], "stage": "warm_execute",
                "session": "", "repetition": "", "status": plan["attempt_plan_status"],
                "quality_gate_id": plan["quality_gate_id"], "quality_metric": "",
                "quality_value": "", "quality_threshold": plan["quality_threshold"],
                "quality_reference_shots": plan["quality_reference_shots"],
                "quality_reference_wall_seconds": "", "selected_clock_id": plan["selected_clock_id"],
                "reported_time_seconds": "", "host_wall_seconds": "", "error": "",
            })
            continue
        quality_record = mps_quality.get(member_id)
        if candidate_id == "qcsim_mps_cpu_chi32" and (not quality_record or quality_record["status"] != "ok"):
            raw_rows.append({
                "record_id": canonical_hash({"attempt_id": plan["attempt_id"], "stage": "warm_execute"}),
                "panel_id": plan["panel_id"], "panel_member_id": member_id,
                "qasm_sha256": plan["qasm_sha256"], "fold_id": plan["fold_id"],
                "candidate_id": candidate_id, "candidate_config_sha256": plan["candidate_config_sha256"],
                "configured_chi": plan["configured_chi"], "stage": "warm_execute",
                "session": "", "repetition": "", "status": "quality_gate_unavailable_or_failed",
                "quality_gate_id": plan["quality_gate_id"], "quality_metric": "mirror_fidelity",
                "quality_value": "" if not quality_record else quality_record.get("fidelity", ""),
                "quality_threshold": plan["quality_threshold"],
                "quality_reference_shots": plan["quality_reference_shots"],
                "quality_reference_wall_seconds": "" if not quality_record else quality_record.get("wall_seconds", ""),
                "selected_clock_id": plan["selected_clock_id"], "reported_time_seconds": "",
                "host_wall_seconds": "", "error": "",
            })
            continue

        row = lookup[member_id]
        qasm = (qasm_root / row["basename"]).read_text(encoding="utf-8")
        config = candidate_lookup[candidate_id]
        for session in range(1, int(manifest["measurement"]["sessions"]) + 1):
            first = isolated_call(_execute_worker, (
                qasm, candidate_id, config.get("configured_chi"), int(config["shots"]), int(config["seed"]),
                _manifest_thread_policy(manifest), None,
            ), timeout)
            first_payload = first.get("result") or {}
            first_status = "ok" if first.get("ok") else ("timeout" if first.get("timeout") else "error")
            raw_rows.append({
                "record_id": canonical_hash({"attempt_id": plan["attempt_id"], "session": session,
                                              "stage": "first_execute", "repetition": 0}),
                "panel_id": plan["panel_id"], "panel_member_id": member_id,
                "qasm_sha256": plan["qasm_sha256"], "fold_id": plan["fold_id"],
                "candidate_id": candidate_id, "candidate_config_sha256": plan["candidate_config_sha256"],
                "configured_chi": plan["configured_chi"], "stage": "first_execute",
                "session": session, "repetition": 0, "status": first_status,
                "quality_gate_id": plan["quality_gate_id"],
                "quality_metric": "mirror_fidelity" if quality_record else "not_required",
                "quality_value": "" if not quality_record else quality_record.get("fidelity", ""),
                "quality_threshold": plan["quality_threshold"],
                "quality_reference_shots": plan["quality_reference_shots"],
                "quality_reference_wall_seconds": "" if not quality_record else quality_record.get("wall_seconds", ""),
                "selected_clock_id": plan["selected_clock_id"],
                "reported_time_seconds": first_payload.get("time_taken", ""),
                "host_wall_seconds": first.get("wall_seconds", ""), "error": first.get("error", ""),
            })
            for repetition in range(3):
                isolated_call(_execute_worker, (
                    qasm, candidate_id, config.get("configured_chi"), int(config["shots"]), int(config["seed"]),
                    _manifest_thread_policy(manifest), None,
                ), timeout)
            for repetition in range(int(manifest["measurement"]["warm_repetitions_per_session"])):
                result = isolated_call(_execute_worker, (
                    qasm, candidate_id, config.get("configured_chi"), int(config["shots"]), int(config["seed"]),
                    _manifest_thread_policy(manifest), None,
                ), timeout)
                payload = result.get("result") or {}
                status = "ok" if result.get("ok") else ("timeout" if result.get("timeout") else "error")
                raw_rows.append({
                    "record_id": canonical_hash({"attempt_id": plan["attempt_id"], "session": session,
                                                  "stage": "warm_execute", "repetition": repetition}),
                    "panel_id": plan["panel_id"], "panel_member_id": member_id,
                    "qasm_sha256": plan["qasm_sha256"], "fold_id": plan["fold_id"],
                    "candidate_id": candidate_id, "candidate_config_sha256": plan["candidate_config_sha256"],
                    "configured_chi": plan["configured_chi"], "stage": "warm_execute",
                    "session": session, "repetition": repetition, "status": status,
                    "quality_gate_id": plan["quality_gate_id"],
                    "quality_metric": "mirror_fidelity" if quality_record else "not_required",
                    "quality_value": "" if not quality_record else quality_record.get("fidelity", ""),
                    "quality_threshold": plan["quality_threshold"],
                    "quality_reference_shots": plan["quality_reference_shots"],
                    "quality_reference_wall_seconds": "" if not quality_record else quality_record.get("wall_seconds", ""),
                    "selected_clock_id": plan["selected_clock_id"],
                    "reported_time_seconds": payload.get("time_taken", ""),
                    "host_wall_seconds": result.get("wall_seconds", ""), "error": result.get("error", ""),
                })
    _write_csv(output_dir / "raw_records.csv", raw_rows)
    return raw_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--calibration-manifest", type=Path, default=DEFAULT_CALIBRATION_MANIFEST)
    parser.add_argument("--calibration-raw", type=Path, default=DEFAULT_CALIBRATION_RAW)
    parser.add_argument("--panel-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--measure", action="store_true", help="perform QCSim quality/timing calls after preflight")
    parser.add_argument("--dry-run", action="store_true", help="build and print a plan without writing files")
    parser.add_argument("--resume", action="store_true", help="resume only under an identical protocol run identity")
    parser.add_argument("--attempt-limit", type=int, help="QA pilot limit (10) or full panel (408)")
    parser.add_argument("--acceptance", type=Path, help="coordinator acceptance JSON required for --measure")
    parser.add_argument("--c3a-fold0-qa-passed", action="store_true")
    parser.add_argument("--gpu-training-released", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("manifest_id") in {"maestro-common-panel-completion-v3-partial",
                                       "maestro-common-panel-completion-v3-partial-width-control",
                                       "maestro-common-panel-completion-v3-partial-zero-measurement"}:
        raise SystemExit("Superseded Maestro bundle is terminal for execution; use the signed resume amendment")
    if manifest.get("manifest_id") == PARTIAL_MANIFEST_ID:
        payload = run_partial_panel(args, manifest)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if manifest.get("manifest_id") == V3_MANIFEST_ID:
        payload = run_v3_panel(args, manifest)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if manifest.get("manifest_id") == V2_MANIFEST_ID:
        payload = run_v2_panel(args, manifest)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.dry_run or args.resume or args.attempt_limit is not None or args.acceptance is not None:
        raise SystemExit("--dry-run/--resume/--attempt-limit/--acceptance are reserved for the v2 manifest")
    panel_csv = args.panel_csv or ROOT / manifest["panel"]["manifest_csv"]
    qasm_root = ROOT / manifest["panel"]["qasm_fixture_dir"]
    fixture_manifest_path = ROOT / manifest["panel"]["qasm_fixture_manifest"]
    calibration_manifest = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    verify_frozen_inputs(manifest, args.calibration_manifest, args.calibration_raw, panel_csv,
                         fixture_manifest_path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {args.output_dir}")

    maestro = load_maestro_parser()
    rows, features = materialize_rows(manifest, panel_csv, qasm_root, fixture_manifest_path, maestro)
    fold_rows = Counter(int(row["fold_id"]) for row in rows)
    if set(fold_rows) != set(range(5)):
        raise ValueError(f"outer split does not populate all five folds: {dict(fold_rows)}")
    calibration_by_fold = {
        fold: fit_frozen_calibration(args.calibration_raw, calibration_manifest,
                                     manifest["calibration"]["selected_calibration_clock"])
        for fold in range(5)
    }
    # The frozen fit CSVs are recovery pins.  Reconstructed knots are checked
    # against them before any prediction or timing artifact is emitted.
    frozen_coefficient_path = ROOT / manifest["calibration"]["frozen_coefficient_grid"]
    frozen_sampling_path = ROOT / manifest["calibration"]["frozen_sampling_grid"]
    frozen_coefs = _csv_rows(frozen_coefficient_path)
    frozen_samples = _csv_rows(frozen_sampling_path)
    if sha256_file(frozen_coefficient_path) != manifest["calibration"]["frozen_coefficient_grid_sha256"]:
        raise ValueError("frozen coefficient grid hash mismatch")
    if sha256_file(frozen_sampling_path) != manifest["calibration"]["frozen_sampling_grid_sha256"]:
        raise ValueError("frozen sampling grid hash mismatch")
    recovered = calibration_by_fold[0]
    for frozen in frozen_coefs:
        key = (frozen["candidate"], frozen["component"], int(frozen["width"]),
               None if frozen["configured_chi"] == "" else int(frozen["configured_chi"]))
        value = recovered.operation_knots.get(key)
        if value is None or not math.isclose(value, float(frozen["coefficient"]), rel_tol=1e-9, abs_tol=1e-15):
            raise ValueError(f"recovered operation knot differs from frozen fit: {key}")
    for frozen in frozen_samples:
        key = (frozen["candidate"], int(frozen["width"]),
               None if frozen["configured_chi"] == "" else int(frozen["configured_chi"]))
        intercept = recovered.sampling_intercept_knots.get(key)
        per_shot = recovered.sampling_per_shot_knots.get(key)
        if (intercept is None or per_shot is None
                or not math.isclose(intercept, float(frozen["sampling_intercept_seconds"]), rel_tol=1e-8, abs_tol=1e-12)
                or not math.isclose(per_shot, float(frozen["sampling_per_shot_seconds"]), rel_tol=1e-8, abs_tol=1e-12)):
            raise ValueError(f"recovered sampling knot differs from frozen fit: {key}")

    prediction_rows, plan_rows = materialize_predictions(manifest, rows, features, calibration_by_fold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "row_manifest.csv", rows)
    _write_csv(args.output_dir / "measurement_plan.csv", plan_rows)
    _write_csv(args.output_dir / "predictions.csv", prediction_rows)
    row_manifest_hash = sha256_file(args.output_dir / "row_manifest.csv")
    prediction_hash = sha256_file(args.output_dir / "predictions.csv")
    plan_hash = sha256_file(args.output_dir / "measurement_plan.csv")
    run_payload = {
        "manifest_id": manifest["manifest_id"],
        "source_manifest_path": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.manifest),
        "status": "prediction_preflight_complete_no_timing" if not args.measure else "measurement_complete",
        "parser_api_preflight": {
            "status": ("pass" if all(row["maestro_parser_status"] == "pass" for row in rows)
                       else "partial_unsupported_members_retained"),
            "api": "maestro.QasmToCirc.parse_and_translate",
            "members_parsed": sum(row["maestro_parser_status"] == "pass" for row in rows),
            "members_rejected_or_width_mismatched": sum(row["maestro_parser_status"] != "pass" for row in rows),
            "rejected_members": [
                {"panel_member_id": row["panel_member_id"],
                 "qasm_sha256": row["qasm_sha256"],
                 "status": row["maestro_parser_status"]}
                for row in rows if row["maestro_parser_status"] != "pass"
            ],
            "candidate_qasm_to_ir_count_limitation": "public binding exposes parser success and circuit.num_qubits, but not the internal operation list; hashes pin the parser-validated native QASM feature counts",
        },
        "panel": {"rows": len(rows), "unique_qasm_hashes": len({row["qasm_sha256"] for row in rows}),
                  "ordered_member_digest": manifest["panel"]["ordered_member_digest"]},
        "split": {"split_id": manifest["outer_split"]["split_id"], "fold_rows": dict(sorted(fold_rows.items())),
                  "row_manifest_sha256": row_manifest_hash},
        "calibration": {"manifest_id": calibration_manifest["manifest_id"],
                        "raw_records_sha256": recovered.calibration_raw_sha256,
                        "fold_local_refit_folds": sorted(calibration_by_fold),
                        "recovered_coefficient_knots": len(recovered.operation_knots),
                        "recovered_sampling_knots": len(recovered.sampling_intercept_knots),
                        "invalid_negative_sampling_knots": [
                            {"candidate": cand, "width": n, "chi": chi, "per_shot_seconds": val}
                            for (cand, n, chi), val in recovered.sampling_per_shot_knots.items() if val < 0
                        ]},
        "prediction": {"path": str((args.output_dir / "predictions.csv").resolve()),
                       "sha256": prediction_hash,
                       "statuses": dict(Counter(row["prediction_status"] for row in prediction_rows)),
                       "candidate_statuses": {
                           candidate: dict(Counter(row["prediction_status"] for row in prediction_rows
                                                   if row["candidate_id"] == candidate))
                           for candidate in (spec["candidate_id"] for spec in manifest["candidates"])
                       },
                       "role": "candidate runtime prediction only; selector/oracle metrics not produced"},
        "measurement_plan": {"path": str((args.output_dir / "measurement_plan.csv").resolve()),
                             "sha256": plan_hash, "rows": len(plan_rows)},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "maestro_version": maestro_version(maestro),
                        "maestro_module_path": str(Path(maestro.__file__).resolve()),
                        "thread_policy": manifest["measurement"]["worker_policy"]},
        "measurement_started": False,
        "live_qpu": False,
    }

    if args.measure:
        if not args.c3a_fold0_qa_passed or not args.gpu_training_released:
            raise SystemExit("measurement blocked: verify C3a fold-0 QA and GPU-training release first")
        raw_rows = measure_plan(manifest, plan_rows, rows, qasm_root, args.output_dir, maestro)
        run_payload["measurement_started"] = True
        run_payload["raw_records"] = {
            "path": str((args.output_dir / "raw_records.csv").resolve()),
            "sha256": sha256_file(args.output_dir / "raw_records.csv"),
            "rows": len(raw_rows),
        }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
