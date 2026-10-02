#!/usr/bin/env python3
"""Execute a frozen, disjoint component calibration grid for Maestro.

The worker invokes the public explicit QCSim configurations.  It never calls
the unavailable Composer auto-selector, and it records a fixed configured MPS
bond rather than an adaptive post-run bond.  Every microbenchmark is spawned
in a killable process because native simulator calls can ignore Python alarms.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PILOT = Path("/home/server/Documents/maestro/experiments/maestro_paper_pilot.py")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def qasm(width: int, op_class: str, repeats: int) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{width}];", f"creg c[{width}];"]
    if op_class == "one_qubit_noncommuting":
        # A different angle on each cycle prevents a compiler from reducing a
        # repeated Clifford word such as X^64 to identity.  There are exactly
        # 2*repeats one-qubit operations in this cell.
        for index in range(repeats):
            lines.extend((f"rz({0.013 + index * 0.0001:.8f}) q[0];", f"rx({0.021 + index * 0.0001:.8f}) q[0];"))
    elif op_class in {"two_qubit_wrapper_control", "two_qubit_cx_interleaved"}:
        if width < 2:
            raise ValueError("cx needs width >= 2")
        # This control carries exactly the same non-commuting 1Q wrapper as
        # the CX word.  Their cycle-time difference is the 2Q component;
        # neither word is assumed to be algebraically simplifiable.
        for index in range(repeats):
            lines.append(f"rx({0.017 + index * 0.0001:.8f}) q[1];")
            if op_class == "two_qubit_cx_interleaved":
                lines.append("cx q[0],q[1];")
            lines.append(f"rz({0.031 + index * 0.0001:.8f}) q[1];")
    elif op_class not in {"final_measurement", "zero_gate_sample"}:
        raise ValueError(f"unknown operation class: {op_class}")
    lines.append("measure q -> c;")
    return "\n".join(lines) + "\n"


def operation_counts(op_class: str, repeats: int) -> dict[str, int]:
    """Exact primitive counts exposed to the component differential fit."""
    if op_class in {"one_qubit_noncommuting", "two_qubit_wrapper_control"}:
        return {"one_qubit_operations": 2 * repeats, "two_qubit_operations": 0}
    if op_class == "two_qubit_cx_interleaved":
        return {"one_qubit_operations": 2 * repeats, "two_qubit_operations": repeats}
    return {"one_qubit_operations": 0, "two_qubit_operations": 0}


def import_pilot() -> Any:
    spec = importlib.util.spec_from_file_location("maestro_paper_pilot", PILOT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PILOT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def run_cell(pilot: Any, program: str, candidate: str, shots: int, chi: int | None, warmups: int, repetitions: int, timeout: float) -> dict[str, Any]:
    simulation = "statevector" if candidate == "statevector" else "mps"
    # The first call is recorded separately.  Untimed calls warm the same
    # public API/configuration; timed repetitions preserve both internal
    # reported time and external host wall-clock.
    first = pilot.isolated_call(pilot._execute_worker, (program, simulation, shots, chi), timeout)
    for _ in range(warmups):
        pilot.isolated_call(pilot._execute_worker, (program, simulation, shots, chi), timeout)
    rows = []
    for repetition in range(repetitions):
        result = pilot.isolated_call(pilot._execute_worker, (program, simulation, shots, chi), timeout)
        rows.append({
            "repetition": repetition,
            "status": "ok" if result.get("ok") else ("timeout" if result.get("timeout") else "error"),
            "reported_time_seconds": (result.get("result") or {}).get("time_taken"),
            "wall_seconds": result.get("wall_seconds"), "error": result.get("error"),
        })
    return {"first": first, "warm": rows}


def planned(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    corpus = manifest["synthetic_corpus"]
    cells = []
    # Do not evaluate the legacy fallback eagerly: v3 supplies the repeat
    # vector and intentionally has no singular legacy field.
    repeats = corpus.get("operation_repeat_counts")
    if repeats is None:
        repeats = [corpus["repetitions_per_operation_cell"]]
    operation_shots = int(corpus.get("operation_shots", 1000))
    for candidate in manifest["implementation"]["candidates"]:
        widths = corpus["statevector_widths"] if candidate == "statevector" else corpus["mps_widths"]
        bonds = [None] if candidate == "statevector" else corpus["mps_configured_bonds"]
        for width in widths:
            for bond in bonds:
                for op in corpus["operation_classes"]:
                    for repeat in repeats:
                        cells.append({"kind":"operation", "candidate":candidate,"width":width,"chi":bond,"operation_class":op,"shots":operation_shots,"operation_repeats":int(repeat)})
    sampling_candidates = corpus.get("sampling_candidates", ["statevector"])
    for candidate in sampling_candidates:
        widths = corpus["statevector_widths"] if candidate == "statevector" else corpus["mps_widths"]
        bonds = [None] if candidate == "statevector" else corpus["mps_configured_bonds"]
        for width in widths:
            for bond in bonds:
                for shots in corpus["sampling_shots"]:
                    cells.append({"kind":"sampling", "candidate":candidate,"width":width,"chi":bond,"operation_class":"zero_gate_sample","shots":shots,"operation_repeats":0})
    return cells


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def _v2_worker(program: str, candidate: str, shots: int, chi: int | None,
               seed: int, singular_value_threshold: float, out: Any) -> None:
    """One fresh-process simple_execute call; no call is treated as warm."""
    try:
        import maestro
        simulation = (maestro.SimulationType.Statevector if candidate == "statevector"
                      else maestro.SimulationType.MatrixProductState)
        kwargs: dict[str, Any] = {
            "simulator_type": maestro.SimulatorType.QCSim,
            "simulation_type": simulation,
            "seed": int(seed),
        }
        if candidate == "mps_fixed_chi":
            kwargs["max_bond_dimension"] = int(chi)
            kwargs["singular_value_threshold"] = float(singular_value_threshold)
        config = maestro.SimulatorConfig(**kwargs)
        start = time.perf_counter()
        result = maestro.simple_execute(program, config, shots=int(shots))
        host_wall = time.perf_counter() - start
        payload = dict(result)
        reported = payload.get("time_taken")
        out.put({"ok": True, "reported_time_seconds": reported,
                 "host_wall_seconds": host_wall})
    except BaseException as exc:
        out.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _v2_isolated_call(program: str, candidate: str, shots: int, chi: int | None,
                      seed: int, singular_value_threshold: float,
                      timeout: float) -> dict[str, Any]:
    """Run exactly one API call in a fresh child and report both clocks."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_v2_worker,
                              args=(program, candidate, shots, chi, seed,
                                    singular_value_threshold, queue))
    outer_start = time.perf_counter()
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        outer_wall = time.perf_counter() - outer_start
        return {"ok": False, "timeout": True,
                "error": f"timeout after {timeout:g}s", "outer_wall_seconds": outer_wall}
    try:
        result = queue.get(timeout=1)
    except Exception:
        result = {"ok": False, "error": f"child exited with code {process.exitcode}"}
    result["outer_wall_seconds"] = time.perf_counter() - outer_start
    return result


def _v2_cell_id(run_identity_sha256: str, cell: dict[str, Any], qasm_sha256: str,
                config_sha256: str) -> str:
    return _canonical_hash({"run_identity_sha256": run_identity_sha256, "cell": cell,
                            "qasm_sha256": qasm_sha256, "config_sha256": config_sha256})


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _v2_environment_pins(manifest_path: Path, maestro: Any) -> dict[str, Any]:
    from benchmark_v1.qre_benchmark.maestro_component_v2 import environment_pins

    code_files = [Path(__file__).resolve(),
                  ROOT / "benchmark_v1/qre_benchmark/maestro_component_v2.py",
                  ROOT / "benchmark_v1/qre_benchmark/maestro_component_predictor.py"]
    return environment_pins(manifest_path, maestro, code_files)


def _v3_code_hashes() -> dict[str, str]:
    """Hash the complete calibration/panel implementation used by S5."""
    paths = (
        ROOT / "benchmark_v1/scripts/run_maestro_component_calibration.py",
        ROOT / "benchmark_v1/scripts/run_maestro_common_panel.py",
        ROOT / "benchmark_v1/qre_benchmark/maestro_component_v2.py",
        ROOT / "benchmark_v1/qre_benchmark/maestro_component_predictor.py",
    )
    return {str(path.relative_to(ROOT)): _sha256_file(path) for path in paths}


def _v3_static_bundle(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        build_v3_execution_schedule, build_v3_plan, resolve_v3_contract, v3_pilot_cell_ids,
    )

    contract = resolve_v3_contract(manifest_path)
    cells = build_v3_plan(contract)
    pilot = v3_pilot_cell_ids(cells)
    ordered_ids = [*pilot["pre_expansion_cell_ids"]]
    included = set(ordered_ids)
    ordered_ids.extend(str(cell["cell_id"]) for cell in cells if str(cell["cell_id"]) not in included)
    by_id = {str(cell["cell_id"]): cell for cell in cells}
    ordered_cells = [by_id[cell_id] for cell_id in ordered_ids]
    schedule = build_v3_execution_schedule(contract, ordered_cells)
    static = {
        "manifest_sha256": _sha256_file(manifest_path),
        "base_contract_sha256": contract["resolution_provenance"]["base_contract_sha256"],
        "resolved_contract_sha256": hashlib.sha256(json.dumps(
            contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")).hexdigest(),
        "code_hashes": _v3_code_hashes(),
        "plan_sha256": _canonical_hash(ordered_cells),
        "schedule_sha256": _canonical_hash(schedule),
        "planned_cells": len(ordered_cells),
        "pilot_cell_ids": pilot["first_ten_cell_ids"],
        "pre_expansion_cell_ids": pilot["pre_expansion_cell_ids"],
    }
    return contract, ordered_cells, schedule, static


def _partial_quarantine_evidence(
    partial_resolution: dict[str, Any],
) -> dict[tuple[str, int, int | None], dict[str, Any]]:
    """Load and hash-check only the frozen S5 failure evidence; never rerun it."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        PARTIAL_QUARANTINE_REASON, V3Unavailable,
    )

    amendment = partial_resolution.get("amendment")
    if not isinstance(amendment, dict):
        raise V3Unavailable("partial_amendment_document_missing")
    pinned = amendment.get("evidence")
    if not isinstance(pinned, dict):
        raise V3Unavailable("partial_pinned_failure_evidence_missing")
    selected: dict[str, dict[str, Any]] = {}
    for name in ("failed_bootstrap", "configuration_diagnosis", "prefix_localization"):
        item = pinned.get(name)
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise V3Unavailable(f"partial_pinned_evidence_descriptor_missing:{name}")
        path = (ROOT / item["path"]).resolve()
        if ROOT not in path.parents or not path.is_file():
            raise V3Unavailable(f"partial_pinned_evidence_file_missing:{name}")
        digest = _sha256_file(path)
        if digest != item.get("sha256"):
            raise V3Unavailable(f"partial_pinned_evidence_hash_mismatch:{name}:{digest}")
        selected[name] = {"path": item["path"], "sha256": digest}
    diagnosis_path = ROOT / selected["configuration_diagnosis"]["path"]
    diagnosis = load(diagnosis_path)
    if diagnosis.get("failed_manifest_sha256") != selected["failed_bootstrap"]["sha256"]:
        raise V3Unavailable("partial_diagnosis_failed_manifest_pin_mismatch")
    expected_norms = {
        int(record["chi"]): float(record["squared_norm"])
        for record in amendment.get("preparation_policy", {}).get("quarantined_configurations", [])
        if record.get("candidate") == "mps_fixed_chi" and int(record.get("width", -1)) == 16
    }
    if set(expected_norms) != {4, 8, 32}:
        raise V3Unavailable("partial_manifest_quarantine_norms_mismatch")
    found: dict[tuple[str, int, int | None], dict[str, Any]] = {}
    for raw in diagnosis.get("controls", []):
        if (raw.get("candidate") != "mps_fixed_chi" or int(raw.get("width", -1)) != 16
                or raw.get("chi") in (None, "")):
            continue
        chi = int(raw["chi"])
        if chi not in expected_norms:
            continue
        key = ("mps_fixed_chi", 16, chi)
        context = raw.get("worker_context")
        context_sha = context.get("identity_sha256") if isinstance(context, dict) else None
        if (not raw.get("ok") or raw.get("transport_status") != "ok"
                or raw.get("norm_pass") is not False
                or float(raw.get("squared_norm", math.nan)) != expected_norms[chi]
                or not isinstance(raw.get("amplitude_sha256"), str)
                or not isinstance(raw.get("qasm_sha256"), str)
                or not isinstance(context_sha, str)):
            raise V3Unavailable(f"partial_quarantine_diagnosis_record_invalid:{key}")
        found[key] = {
            "squared_norm": float(raw["squared_norm"]),
            "qasm_sha256": str(raw["qasm_sha256"]),
            "amplitude_sha256": str(raw["amplitude_sha256"]),
            "worker_context_sha256": context_sha,
            "diagnosis_path": selected["configuration_diagnosis"]["path"],
            "diagnosis_sha256": selected["configuration_diagnosis"]["sha256"],
            "failed_run_manifest_sha256": selected["failed_bootstrap"]["sha256"],
            "localization_path": selected["prefix_localization"]["path"],
            "localization_sha256": selected["prefix_localization"]["sha256"],
            "status": "unavailable",
            "reason": PARTIAL_QUARANTINE_REASON,
            "rank_from_unnormalized_amplitudes": False,
        }
    if set(found) != {
        ("mps_fixed_chi", 16, 4), ("mps_fixed_chi", 16, 8),
        ("mps_fixed_chi", 16, 32),
    }:
        raise V3Unavailable("partial_quarantine_diagnosis_coverage_mismatch")
    return found


def _partial_evidence_document(evidence: dict) -> dict[str, Any]:
    """Stable JSON identity for the in-memory tuple-keyed configuration map."""
    return {f"{candidate}:{width}:{chi}": value
            for (candidate, width, chi), value in evidence.items()}


def _partial_static_bundle(manifest_path: Path) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    """Resolve S6, preserve all parent cell IDs, and pin its 10,368-slot plan."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        PARTIAL_MANIFEST_ID, PARTIAL_MANIFEST_SHA256, PARTIAL_PARENT_V3_SHA256,
        V3Unavailable, build_partial_v3_execution_schedule, build_partial_v3_plan,
        canonical_hash, partial_v3_pilot_cell_ids,
    )
    from benchmark_v1.scripts.run_maestro_common_panel import resolve_partial_contract

    resolution = resolve_partial_contract(manifest_path)
    if (resolution.get("amendment_identity", {}).get("manifest_id") != PARTIAL_MANIFEST_ID
            or resolution.get("amendment_identity", {}).get("sha256") != PARTIAL_MANIFEST_SHA256
            or resolution.get("amendment", {}).get("base_contract", {}).get("sha256")
            != PARTIAL_PARENT_V3_SHA256):
        raise V3Unavailable("partial_resolver_identity_or_hash_mismatch")
    contract = resolution["contract"]
    cells = build_partial_v3_plan(contract, resolution["quarantine_policy"])
    quarantine_evidence = _partial_quarantine_evidence(resolution)
    pilot = partial_v3_pilot_cell_ids(cells)
    schedule = build_partial_v3_execution_schedule(contract, cells)
    if (len(cells) != 432 or len(schedule) != 10_368
            or len(pilot["first_ten_cell_ids"]) != 10
            or len(pilot["pre_expansion_cell_ids"]) != 20):
        raise V3Unavailable("partial_frozen_plan_shape_mismatch")
    base_resolution = contract.get("resolution_provenance", {})
    static = {
        "manifest_id": PARTIAL_MANIFEST_ID,
        "manifest_sha256": str(resolution["amendment_identity"]["sha256"]),
        "amendment_identity": dict(resolution["amendment_identity"]),
        "parent_manifest_id": "maestro-common-panel-completion-v3",
        "parent_manifest_sha256": PARTIAL_PARENT_V3_SHA256,
        "parent_contract_sha256": str(resolution["parent_contract_sha256"]),
        "base_contract_sha256": str(base_resolution.get("base_contract_sha256", "")),
        "resolved_contract_sha256": str(resolution["resolved_contract_sha256"]),
        "code_hashes": _v3_code_hashes(),
        "plan_sha256": canonical_hash(cells),
        "schedule_sha256": canonical_hash(schedule),
        "planned_cells": len(cells),
        "planned_enabled_cells": 399,
        "planned_quarantined_cells": 33,
        "planned_timed_calls": 5_985,
        "planned_untimed_calls": 3_591,
        "planned_quarantined_slots": 792,
        "first_ten_cell_ids": pilot["first_ten_cell_ids"],
        "pre_expansion_cell_ids": pilot["pre_expansion_cell_ids"],
        "quarantine_evidence": quarantine_evidence,
        "quarantine_evidence_sha256": _canonical_hash(_partial_evidence_document(quarantine_evidence)),
        "resolved_contract": resolution["resolved_contract"],
        "quarantine_policy": resolution["quarantine_policy"],
    }
    if not static["base_contract_sha256"]:
        raise V3Unavailable("partial_parent_base_contract_hash_missing")
    return resolution, cells, schedule, static


def _v3_validate_acceptance(acceptance: Any, static: dict[str, Any]) -> dict[str, Any]:
    """Reject stale/v2/mismatched authorization before any Maestro import."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import V3_MANIFEST_ID, V3Unavailable

    if not isinstance(acceptance, dict):
        raise V3Unavailable("v3_acceptance_must_be_json_object")
    if acceptance.get("manifest_id") == "maestro-common-panel-completion-v2":
        raise V3Unavailable("v3_rejects_v2_acceptance")
    if acceptance.get("schema") != "maestro_v3_calibration_acceptance_v1":
        raise V3Unavailable("v3_acceptance_schema_mismatch")
    if acceptance.get("status") != "pass":
        raise V3Unavailable("v3_acceptance_status_not_pass")
    if acceptance.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("v3_acceptance_manifest_id_mismatch")
    accepted_prefix = acceptance.get("authorized_cell_prefix")
    if accepted_prefix != 432:
        raise V3Unavailable("v3_acceptance_must_authorize_frozen_full_prefix")
    for key in ("manifest_sha256", "base_contract_sha256", "resolved_contract_sha256",
                "code_hashes", "plan_sha256", "schedule_sha256", "planned_cells"):
        if acceptance.get(key) != static[key]:
            raise V3Unavailable(f"v3_acceptance_pin_mismatch:{key}")
    return acceptance


def _v3_backend_threading_policy(contract: dict[str, Any]) -> dict[str, Any]:
    measure = contract["measurement"]
    return {
        "declared_policy": measure.get("native_thread_policy"),
        "affinity": measure.get("cpu_affinity"),
        "native_thread_environment": measure.get("native_thread_environment"),
        "timing_worker_count": measure.get("timing_worker_count", 1),
    }


def _v3_configuration_records(contract: dict[str, Any], maestro: Any) -> dict[str, dict[str, Any]]:
    from benchmark_v1.qre_benchmark.maestro_component_v2 import resolve_v2_config

    result: dict[str, dict[str, Any]] = {}
    for candidate in contract["implementation"]["candidates"]:
        chis = ([None] if candidate == "statevector" else
                [int(value) for value in contract["synthetic_corpus"]["mps_configured_bonds"]])
        for chi in chis:
            resolved = resolve_v2_config(
                maestro, candidate, chi, int(contract["seed"]),
                float(contract["panel_candidate_configuration"]["mps_singular_value_threshold"]),
            )
            result[f"{candidate}:{chi}"] = {
                "sha256": resolved["sha256"], "resolved": resolved["resolved"],
            }
    return result


def _v3_runtime_pins(manifest_path: Path, maestro: Any) -> dict[str, Any]:
    """Capture the existing canonical environment pins for calibration and panel."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import environment_pins

    code_files = [ROOT / "benchmark_v1/scripts/run_maestro_component_calibration.py",
                  ROOT / "benchmark_v1/scripts/run_maestro_common_panel.py",
                  ROOT / "benchmark_v1/qre_benchmark/maestro_component_v2.py",
                  ROOT / "benchmark_v1/qre_benchmark/maestro_component_predictor.py"]
    return environment_pins(manifest_path, maestro, code_files)


def _v3_validate_context_acceptance(contract: dict[str, Any],
                                    configurations: dict[str, dict[str, Any]],
                                    acceptance: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str, dict[str, str]]:
    from benchmark_v1.qre_benchmark.maestro_component_v2 import V3Unavailable
    from benchmark_v1.scripts.run_maestro_common_panel import (
        validate_worker_context, validate_worker_environment_consistency,
        worker_context_config_key,
    )

    policy = _v3_backend_threading_policy(contract)
    accepted = acceptance.get("worker_contexts")
    if not isinstance(accepted, dict) or set(accepted) != set(configurations):
        raise V3Unavailable("v3_acceptance_worker_context_config_set_mismatch")
    worker_contexts: dict[str, dict[str, Any]] = {}
    environment_contexts: dict[str, dict[str, Any]] = {}
    expected_hashes: dict[str, str] = {}
    environment_hashes: set[str] = set()
    for config_key, config in configurations.items():
        context = accepted[config_key]
        if not isinstance(context, dict):
            raise V3Unavailable(f"v3_acceptance_worker_context_missing:{config_key}")
        validate_worker_context(context, context)
        expected_config_key = worker_context_config_key(config["resolved"], policy)
        actual_config_key = worker_context_config_key(
            context.get("identity", {}).get("resolved_config"),
            context.get("identity", {}).get("backend_threading_policy"),
        )
        if actual_config_key != expected_config_key:
            raise V3Unavailable(f"v3_acceptance_worker_context_config_mismatch:{config_key}")
        if (context.get("resolved_config_sha256") != config["sha256"]
                or context.get("backend_threading_policy_sha256") != _canonical_hash(policy)):
            raise V3Unavailable(f"v3_acceptance_worker_context_pin_mismatch:{config_key}")
        validate_worker_environment_consistency(environment_contexts, context)
        environment_contexts[config_key] = context
        worker_contexts[config_key] = context
        expected_hashes[config_key] = str(context["identity_sha256"])
        environment_hashes.add(str(context["worker_environment_sha256"]))
    if len(environment_hashes) != 1:
        raise V3Unavailable("v3_acceptance_worker_environment_drift")
    environment_hash = next(iter(environment_hashes))
    if acceptance.get("worker_environment_sha256") != environment_hash:
        raise V3Unavailable("v3_acceptance_worker_environment_pin_mismatch")
    if acceptance.get("worker_context_sha256_by_config") != expected_hashes:
        raise V3Unavailable("v3_acceptance_worker_context_hash_map_mismatch")
    return worker_contexts, environment_hash, expected_hashes


def _v3_build_identity(contract: dict[str, Any], static: dict[str, Any],
                       configurations: dict[str, dict[str, Any]],
                       worker_context_hashes: dict[str, str],
                       worker_environment_sha256: str,
                       runtime_pins: dict[str, Any]) -> dict[str, Any]:
    from benchmark_v1.qre_benchmark.maestro_component_v2 import build_v3_run_identity

    pins = {
        "manifest_sha256": static["manifest_sha256"],
        "base_contract_sha256": static["base_contract_sha256"],
        "code_hashes": static["code_hashes"],
        "plan_sha256": static["plan_sha256"],
        "schedule_sha256": static["schedule_sha256"],
        "configuration_records": configurations,
        "worker_context_sha256_by_config": worker_context_hashes,
        "worker_environment_sha256": worker_environment_sha256,
        **runtime_pins,
        "runtime_pins": runtime_pins,
    }
    input_hashes = {
        "plan_sha256": static["plan_sha256"],
        "schedule_sha256": static["schedule_sha256"],
    }
    return build_v3_run_identity(contract, pins, input_hashes)


def _partial_build_identity(
    resolution: dict[str, Any], static: dict[str, Any],
    configurations: dict[str, Any], worker_contexts: dict[str, Any],
    worker_environment_sha256: str, runtime_pins: dict[str, Any],
) -> dict[str, Any]:
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        build_partial_v3_run_identity,
    )

    context_hashes = {
        key: str(value.get("identity_sha256", ""))
        for key, value in worker_contexts.items()
    }
    pins = {
        "manifest_sha256": static["manifest_sha256"],
        "amendment_identity": static["amendment_identity"],
        "parent_manifest_sha256": static["parent_manifest_sha256"],
        "parent_contract_sha256": static["parent_contract_sha256"],
        "base_contract_sha256": static["base_contract_sha256"],
        "resolved_contract_sha256": static["resolved_contract_sha256"],
        "resolved_contract": static["resolved_contract"],
        "quarantine_policy": static["quarantine_policy"],
        "quarantine_evidence_sha256": static["quarantine_evidence_sha256"],
        "code_hashes": static["code_hashes"],
        "plan_sha256": static["plan_sha256"],
        "schedule_sha256": static["schedule_sha256"],
        "configuration_records": configurations,
        "worker_contexts": worker_contexts,
        "worker_context_sha256_by_config": context_hashes,
        "worker_environment_sha256": worker_environment_sha256,
        "runtime_pins": runtime_pins,
    }
    evidence_hash = _canonical_hash(_partial_evidence_document(static["quarantine_evidence"]))
    return build_partial_v3_run_identity(
        resolution, pins,
        {
            "plan_sha256": static["plan_sha256"],
            "schedule_sha256": static["schedule_sha256"],
            "quarantine_evidence_sha256": evidence_hash,
        },
    )


def _partial_preflight_evidence_sha256(
    static: dict[str, Any], runtime_pins: dict[str, Any],
    configurations: dict[str, Any], worker_contexts: dict[str, Any],
    rank_records: list[dict[str, Any]], worker_environment_sha256: str,
) -> str:
    return _canonical_hash({
        "manifest_sha256": static["manifest_sha256"],
        "parent_manifest_sha256": static["parent_manifest_sha256"],
        "parent_contract_sha256": static["parent_contract_sha256"],
        "resolved_contract_sha256": static["resolved_contract_sha256"],
        "code_hashes": static["code_hashes"],
        "plan_sha256": static["plan_sha256"],
        "schedule_sha256": static["schedule_sha256"],
        "quarantine_evidence": _partial_evidence_document(static["quarantine_evidence"]),
        "runtime_pins": runtime_pins,
        "configuration_records": configurations,
        "worker_contexts": worker_contexts,
        "worker_environment_sha256": worker_environment_sha256,
        "rank_preflight_records": rank_records,
    })


def _partial_acceptance_static_pins(
    acceptance: dict[str, Any], static: dict[str, Any],
) -> None:
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        PARTIAL_MANIFEST_ID, V3Unavailable,
    )

    if acceptance.get("manifest_id") in {
        "maestro-common-panel-completion-v2", "maestro-common-panel-completion-v3",
    }:
        raise V3Unavailable("partial_rejects_v2_or_failed_v3_acceptance")
    if (acceptance.get("schema") != "maestro_v3_partial_calibration_acceptance_v1"
            or acceptance.get("status") != "pass"
            or acceptance.get("manifest_id") != PARTIAL_MANIFEST_ID):
        raise V3Unavailable("partial_acceptance_schema_status_or_manifest_mismatch")
    if acceptance.get("authorized_cell_prefix") != 399:
        raise V3Unavailable("partial_acceptance_must_authorize_399_enabled_cells")
    for key in (
        "manifest_sha256", "amendment_identity", "parent_manifest_sha256",
        "parent_contract_sha256", "base_contract_sha256", "resolved_contract_sha256",
        "code_hashes", "plan_sha256", "schedule_sha256", "planned_cells",
        "planned_enabled_cells", "planned_quarantined_cells", "planned_timed_calls",
        "planned_untimed_calls", "planned_quarantined_slots",
        "quarantine_evidence_sha256",
    ):
        if acceptance.get(key) != static.get(key):
            raise V3Unavailable(f"partial_acceptance_pin_mismatch:{key}")


def _partial_validate_preflight_bundle(
    existing: dict[str, Any], acceptance: dict[str, Any], static: dict[str, Any],
    runtime_pins: dict[str, Any], configurations: dict[str, Any],
    worker_contexts: dict[str, Any], worker_environment_sha256: str,
    identity: dict[str, Any], preflight_run_manifest_sha256: str,
) -> str:
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        PARTIAL_MANIFEST_ID, V3Unavailable,
    )

    if existing.get("manifest_id") != PARTIAL_MANIFEST_ID:
        raise V3Unavailable("partial_preflight_bundle_manifest_mismatch")
    if existing.get("status") not in {
        "preflight_complete", "preflight_accepted", "checkpointed_partial",
    }:
        raise V3Unavailable(f"partial_preflight_bundle_status_invalid:{existing.get('status')}")
    if existing.get("measurement_started") not in (False, True):
        raise V3Unavailable("partial_preflight_measurement_status_missing")
    for key in (
        "manifest_sha256", "amendment_identity", "parent_manifest_sha256",
        "parent_contract_sha256", "base_contract_sha256", "resolved_contract_sha256",
        "code_hashes", "plan_sha256", "schedule_sha256", "planned_cells",
        "planned_enabled_cells", "planned_quarantined_cells", "planned_timed_calls",
        "planned_untimed_calls", "planned_quarantined_slots",
        "quarantine_evidence_sha256",
    ):
        if existing.get(key) != static.get(key):
            raise V3Unavailable(f"partial_preflight_static_pin_mismatch:{key}")
    if existing.get("runtime_pins") != runtime_pins:
        raise V3Unavailable("partial_preflight_runtime_pins_mismatch")
    if existing.get("configuration_records") != configurations:
        raise V3Unavailable("partial_preflight_configuration_records_mismatch")
    if existing.get("worker_contexts_by_config") != worker_contexts:
        raise V3Unavailable("partial_preflight_worker_contexts_mismatch")
    if existing.get("worker_environment_sha256") != worker_environment_sha256:
        raise V3Unavailable("partial_preflight_worker_environment_mismatch")
    from benchmark_v1.qre_benchmark.maestro_component_v2 import validate_partial_v3_resume_identity
    validate_partial_v3_resume_identity(existing, identity)
    if existing.get("run_identity") != identity["run_identity"]:
        raise V3Unavailable("partial_preflight_run_identity_mismatch")
    evidence_sha = _partial_preflight_evidence_sha256(
        static, runtime_pins, configurations, worker_contexts,
        existing.get("rank_preflight_records", []), worker_environment_sha256,
    )
    if existing.get("preflight_evidence_sha256") != evidence_sha:
        raise V3Unavailable("partial_preflight_evidence_hash_mismatch")
    if acceptance.get("runtime_pins") != runtime_pins:
        raise V3Unavailable("partial_acceptance_runtime_pins_mismatch")
    if acceptance.get("configuration_records") != configurations:
        raise V3Unavailable("partial_acceptance_configuration_records_mismatch")
    if acceptance.get("worker_contexts") != worker_contexts:
        raise V3Unavailable("partial_acceptance_worker_contexts_mismatch")
    if acceptance.get("worker_context_sha256_by_config") != identity["run_identity"]["pins"]["worker_context_sha256_by_config"]:
        raise V3Unavailable("partial_acceptance_worker_context_hash_map_mismatch")
    if acceptance.get("worker_environment_sha256") != worker_environment_sha256:
        raise V3Unavailable("partial_acceptance_worker_environment_mismatch")
    if acceptance.get("preflight_evidence_sha256") != evidence_sha:
        raise V3Unavailable("partial_acceptance_preflight_evidence_mismatch")
    if acceptance.get("preflight_run_manifest_sha256") != preflight_run_manifest_sha256:
        raise V3Unavailable("partial_acceptance_preflight_manifest_hash_mismatch")
    if acceptance.get("run_identity_sha256") != identity.get("run_identity_sha256"):
        raise V3Unavailable("partial_acceptance_run_identity_mismatch")
    return preflight_run_manifest_sha256


def _v3_append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _v3_read_checkpoint(path: Path, schedule: list[dict[str, Any]],
                        run_identity_sha256: str) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Validate the append-only attempt ledger; interrupted calls are never retried."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import V3Unavailable

    expected = {str(event["attempt_id"]): event for event in schedule}
    started: set[str] = set()
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results, started
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise V3Unavailable(f"v3_resume_incomplete_checkpoint_line:{line_no}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise V3Unavailable(f"v3_resume_invalid_checkpoint_json:{line_no}") from exc
            if (record.get("manifest_id") != "maestro-common-panel-completion-v3"
                    or record.get("run_identity_sha256") != run_identity_sha256):
                raise V3Unavailable(f"v3_resume_checkpoint_identity_mismatch:{line_no}")
            attempt_id = str(record.get("attempt_id", ""))
            if attempt_id not in expected:
                raise V3Unavailable(f"v3_resume_attempt_not_in_schedule:{line_no}")
            event = expected[attempt_id]
            kind = record.get("record_type")
            if kind == "call_started":
                if attempt_id in started or attempt_id in results:
                    raise V3Unavailable(f"v3_resume_duplicate_call_start:{attempt_id}")
                started.add(attempt_id)
            elif kind == "attempt_result":
                if attempt_id not in started or attempt_id in results:
                    raise V3Unavailable(f"v3_resume_result_without_unique_start:{attempt_id}")
                row = record.get("row")
                if not isinstance(row, dict) or row.get("attempt_id") != attempt_id:
                    raise V3Unavailable(f"v3_resume_result_row_mismatch:{attempt_id}")
                for field in ("cell_id", "stage", "session", "repetition", "whole_qasm_sha256"):
                    if row.get(field) != event.get(field):
                            raise V3Unavailable(f"v3_resume_result_schedule_mismatch:{attempt_id}:{field}")
                results[attempt_id] = row
            else:
                raise V3Unavailable(f"v3_resume_unknown_checkpoint_record:{line_no}")
    interrupted = started - set(results)
    if interrupted:
        raise V3Unavailable(f"v3_resume_interrupted_call_requires_review:{sorted(interrupted)[0]}")
    return results, started


def _partial_initialize_checkpoint(
    path: Path, schedule: list[dict[str, Any]], run_identity_sha256: str,
) -> None:
    """Write the complete slot allocation and quarantine terminal states once."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        PARTIAL_MANIFEST_ID, V3Unavailable,
    )

    if path.exists():
        return
    for event in schedule:
        _v3_append_jsonl(path, {
            "record_type": "slot_allocated", "manifest_id": PARTIAL_MANIFEST_ID,
            "attempt_id": str(event["attempt_id"]),
            "run_identity_sha256": run_identity_sha256, "event": event,
        })
    for event in schedule:
        if event.get("calibration_eligibility") != "quarantined":
            continue
        row = {
            **event, "record_type": "terminal_skip",
            "timing_status": "not_started",
            "timing_reason": str(event.get("timing_reason") or "synthetic_preparation_norm_failure"),
            "reported_time_seconds": "", "host_wall_seconds": "", "outer_wall_seconds": "",
        }
        _v3_append_jsonl(path, {
            "record_type": "terminal_skip", "manifest_id": PARTIAL_MANIFEST_ID,
            "attempt_id": str(event["attempt_id"]),
            "run_identity_sha256": run_identity_sha256, "row": row,
        })


def _partial_read_checkpoint(
    path: Path, schedule: list[dict[str, Any]], run_identity_sha256: str,
) -> tuple[dict[str, dict[str, Any]], set[str], set[str], set[str]]:
    """Validate the full allocation, explicit quarantine, and no-retry ledger."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        PARTIAL_MANIFEST_ID, V3Unavailable,
    )

    expected = {str(event["attempt_id"]): event for event in schedule}
    if len(expected) != len(schedule):
        raise V3Unavailable("partial_schedule_attempt_id_collision")
    allocated: dict[str, dict[str, Any]] = {}
    terminal_skips: dict[str, dict[str, Any]] = {}
    budget_stops: dict[str, dict[str, Any]] = {}
    started: set[str] = set()
    results: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return results, started, set(), set()

    def validate_event_payload(attempt_id: str, row: Any, event: dict[str, Any], label: str) -> None:
        if not isinstance(row, dict):
            raise V3Unavailable(f"partial_checkpoint_{label}_row_missing:{attempt_id}")
        for field, expected_value in event.items():
            if label != "allocation" and field in {"timing_status", "timing_reason"}:
                continue  # These are outcomes, not frozen scheduling identity.
            if row.get(field) != expected_value:
                raise V3Unavailable(f"partial_checkpoint_{label}_schedule_mismatch:{attempt_id}:{field}")

    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise V3Unavailable(f"partial_checkpoint_incomplete_line:{line_no}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise V3Unavailable(f"partial_checkpoint_invalid_json:{line_no}") from exc
            if (record.get("manifest_id") != PARTIAL_MANIFEST_ID
                    or record.get("run_identity_sha256") != run_identity_sha256):
                raise V3Unavailable(f"partial_checkpoint_identity_mismatch:{line_no}")
            attempt_id = str(record.get("attempt_id", ""))
            event = expected.get(attempt_id)
            if event is None:
                raise V3Unavailable(f"partial_checkpoint_attempt_outside_schedule:{line_no}")
            record_type = str(record.get("record_type", ""))
            if record_type == "slot_allocated":
                if attempt_id in allocated:
                    raise V3Unavailable(f"partial_checkpoint_duplicate_slot_allocation:{attempt_id}")
                validate_event_payload(attempt_id, record.get("event"), event, "allocation")
                allocated[attempt_id] = event
            elif record_type == "terminal_skip":
                if (event.get("calibration_eligibility") != "quarantined"
                        or attempt_id in terminal_skips or attempt_id in started or attempt_id in results):
                    raise V3Unavailable(f"partial_checkpoint_invalid_terminal_skip:{attempt_id}")
                row = record.get("row")
                validate_event_payload(attempt_id, row, event, "skip")
                if (row.get("timing_status") != "not_started"
                        or row.get("timing_reason") != "synthetic_preparation_norm_failure"
                        or any(row.get(name) not in (None, "") for name in (
                            "reported_time_seconds", "host_wall_seconds", "outer_wall_seconds"
                        ))):
                    raise V3Unavailable(f"partial_checkpoint_skip_status_mismatch:{attempt_id}")
                terminal_skips[attempt_id] = row
            elif record_type == "call_started":
                if (event.get("calibration_eligibility") != "eligible"
                        or attempt_id in terminal_skips or attempt_id in started or attempt_id in results):
                    raise V3Unavailable(f"partial_checkpoint_invalid_call_start:{attempt_id}")
                started.add(attempt_id)
            elif record_type == "attempt_result":
                if (event.get("calibration_eligibility") != "eligible"
                        or attempt_id not in started or attempt_id in results):
                    raise V3Unavailable(f"partial_checkpoint_result_without_unique_start:{attempt_id}")
                row = record.get("row")
                validate_event_payload(attempt_id, row, event, "result")
                if str(row.get("attempt_id", "")) != attempt_id:
                    raise V3Unavailable(f"partial_checkpoint_result_attempt_id_mismatch:{attempt_id}")
                results[attempt_id] = row
            elif record_type == "not_started_due_to_pilot_budget":
                if (event.get("calibration_eligibility") != "eligible"
                        or attempt_id in started or attempt_id in results or attempt_id in budget_stops):
                    raise V3Unavailable(f"partial_checkpoint_invalid_budget_stop:{attempt_id}")
                row = record.get("row")
                validate_event_payload(attempt_id, row, event, "budget")
                if (row.get("timing_status") != "not_started"
                        or row.get("timing_reason") not in {
                            "not_started_due_to_pilot_budget", "not_started_due_to_deadline",
                        }
                        or any(row.get(name) not in (None, "") for name in (
                            "reported_time_seconds", "host_wall_seconds", "outer_wall_seconds"
                        ))):
                    raise V3Unavailable(f"partial_checkpoint_budget_status_mismatch:{attempt_id}")
                budget_stops[attempt_id] = row
            else:
                raise V3Unavailable(f"partial_checkpoint_unknown_record:{line_no}:{record_type}")

    if set(allocated) != set(expected):
        raise V3Unavailable(
            f"partial_checkpoint_allocation_coverage_mismatch:{len(allocated)}:{len(expected)}"
        )
    expected_skips = {
        attempt_id for attempt_id, event in expected.items()
        if event.get("calibration_eligibility") == "quarantined"
    }
    if set(terminal_skips) != expected_skips:
        raise V3Unavailable(
            f"partial_checkpoint_quarantine_ledger_mismatch:{len(terminal_skips)}:{len(expected_skips)}"
        )
    interrupted = started - set(results)
    if interrupted:
        raise V3Unavailable(f"partial_resume_interrupted_call_requires_review:{sorted(interrupted)[0]}")
    if set(budget_stops) & set(results):
        raise V3Unavailable("partial_checkpoint_budget_stop_has_result")
    return results, started, set(terminal_skips), set(budget_stops)



def _v3_write_raw(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    serializable = []
    for row in rows:
        serializable.append({
            key: (json.dumps(value, sort_keys=True, separators=(",", ":"))
                  if isinstance(value, (dict, list, tuple)) else value)
            for key, value in row.items()
        })
    write_csv(temporary, serializable)
    os.replace(temporary, path)


def _v3_cell_order(cells: list[dict[str, Any]], static: dict[str, Any]) -> list[str]:
    ordered = [str(value) for value in static["pre_expansion_cell_ids"]]
    seen = set(ordered)
    ordered.extend(str(cell["cell_id"]) for cell in cells if str(cell["cell_id"]) not in seen)
    return ordered


def _v3_preflight_evidence_sha256(static: dict[str, Any], runtime_pins: dict[str, Any],
                                 configurations: dict[str, Any],
                                 worker_contexts: dict[str, Any],
                                 rank_records: list[dict[str, Any]],
                                 worker_environment_sha256: str) -> str:
    return _canonical_hash({
        "manifest_sha256": static["manifest_sha256"],
        "base_contract_sha256": static["base_contract_sha256"],
        "resolved_contract_sha256": static["resolved_contract_sha256"],
        "code_hashes": static["code_hashes"],
        "plan_sha256": static["plan_sha256"], "schedule_sha256": static["schedule_sha256"],
        "runtime_pins": runtime_pins, "configuration_records": configurations,
        "worker_contexts": worker_contexts,
        "worker_environment_sha256": worker_environment_sha256,
        "rank_preflight_records": rank_records,
    })


def _v3_validate_preflight_bundle(existing: dict[str, Any], acceptance: dict[str, Any],
                                 static: dict[str, Any], runtime_pins: dict[str, Any],
                                 configurations: dict[str, Any], worker_contexts: dict[str, Any],
                                 worker_environment_sha256: str,
                                 run_identity: dict[str, Any], run_path_sha256: str) -> str:
    """Bind coordinator authorization to the exact bootstrap evidence bundle."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import V3Unavailable

    if existing.get("manifest_id") != "maestro-common-panel-completion-v3":
        raise V3Unavailable("v3_preflight_bundle_manifest_mismatch")
    if existing.get("status") not in {"preflight_complete", "preflight_accepted", "checkpointed_partial"}:
        raise V3Unavailable(f"v3_preflight_bundle_status_invalid:{existing.get('status')}")
    if existing.get("measurement_started") not in (False, True):
        raise V3Unavailable("v3_preflight_bundle_measurement_status_missing")
    if existing.get("runtime_pins") != runtime_pins:
        raise V3Unavailable("v3_preflight_runtime_pins_mismatch")
    if existing.get("configuration_records") != configurations:
        raise V3Unavailable("v3_preflight_configuration_records_mismatch")
    if existing.get("worker_contexts_by_config") != worker_contexts:
        raise V3Unavailable("v3_preflight_worker_contexts_mismatch")
    if existing.get("worker_environment_sha256") != worker_environment_sha256:
        raise V3Unavailable("v3_preflight_worker_environment_mismatch")
    if existing.get("run_identity_sha256") != run_identity.get("run_identity_sha256"):
        raise V3Unavailable("v3_preflight_run_identity_mismatch")
    for key in ("manifest_sha256", "base_contract_sha256", "resolved_contract_sha256",
                "code_hashes", "plan_sha256", "schedule_sha256"):
        if existing.get(key) != static.get(key):
            raise V3Unavailable(f"v3_preflight_static_pin_mismatch:{key}")
    rank_records = existing.get("rank_preflight_records")
    if not isinstance(rank_records, list) or not rank_records:
        raise V3Unavailable("v3_preflight_rank_records_missing")
    evidence_sha = _v3_preflight_evidence_sha256(
        static, runtime_pins, configurations, worker_contexts, rank_records,
        worker_environment_sha256,
    )
    if existing.get("preflight_evidence_sha256") != evidence_sha:
        raise V3Unavailable("v3_preflight_evidence_hash_mismatch")
    accepted_preflight_sha = str(existing.get("accepted_preflight_run_manifest_sha256") or run_path_sha256)
    expected_acceptance = {
        "preflight_run_manifest_sha256": accepted_preflight_sha,
        "preflight_evidence_sha256": evidence_sha,
        "runtime_pins": runtime_pins,
        "configuration_records": configurations,
        "worker_contexts": worker_contexts,
        "worker_context_sha256_by_config": {
            key: value["identity_sha256"] for key, value in worker_contexts.items()
        },
        "worker_environment_sha256": worker_environment_sha256,
        "run_identity_sha256": run_identity["run_identity_sha256"],
    }
    for key, value in expected_acceptance.items():
        if acceptance.get(key) != value:
            raise V3Unavailable(f"v3_acceptance_preflight_pin_mismatch:{key}")
    return accepted_preflight_sha


def _v3_bootstrap_preflight(manifest_path: Path, artifact_dir: Path,
                            contract: dict[str, Any], cells: list[dict[str, Any]],
                            schedule: list[dict[str, Any]], static: dict[str, Any],
                            resume: bool) -> dict[str, Any]:
    """Create the context/rank evidence bundle only; never enters timed cells."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        V3_MANIFEST_ID, V3Unavailable, validate_v3_rank_preflight_records,
    )
    from benchmark_v1.scripts.run_maestro_common_panel import maestro_timing_lock

    run_path = artifact_dir / "run_manifest.json"
    if artifact_dir.exists():
        if not resume:
            raise V3Unavailable("v3_preflight_bundle_exists_use_resume_or_acceptance")
        if not run_path.is_file():
            raise V3Unavailable("v3_preflight_existing_bundle_missing_run_manifest")
        old = json.loads(run_path.read_text(encoding="utf-8"))
        if old.get("status") == "preflight_failed":
            raise V3Unavailable("v3_preflight_failure_is_terminal_no_retry")
        if old.get("status") == "preflight_complete":
            raise V3Unavailable("v3_preflight_already_complete_request_final_acceptance")
        if old.get("measurement_started"):
            raise V3Unavailable("v3_bootstrap_refuses_bundle_after_measurement_started")
    elif resume:
        raise V3Unavailable("v3_preflight_resume_bundle_missing")

    try:
        import maestro
    except Exception as exc:
        raise V3Unavailable(f"v3_maestro_api_unavailable:{type(exc).__name__}:{exc}") from exc
    configurations = _v3_configuration_records(contract, maestro)
    runtime_pins = _v3_runtime_pins(manifest_path, maestro)
    try:
        with maestro_timing_lock(V3_MANIFEST_ID, _canonical_hash(static), root=ROOT):
            rank_records, worker_contexts = _v3_reference_and_candidate_rank_preflight(
                contract, configurations, None
            )
        validated = validate_v3_rank_preflight_records(contract, rank_records)
        if validated.get("status") != "pass":
            raise V3Unavailable(f"v3_bootstrap_rank_preflight_failed:{validated.get('unavailable_reasons')}")
        rank_records = validated["records"]
        if set(worker_contexts) != set(configurations):
            raise V3Unavailable("v3_bootstrap_worker_context_config_set_mismatch")
        environment_hashes = {str(context.get("worker_environment_sha256", ""))
                              for context in worker_contexts.values()}
        if len(environment_hashes) != 1 or "" in environment_hashes:
            raise V3Unavailable("v3_bootstrap_worker_environment_drift")
        worker_environment_sha256 = next(iter(environment_hashes))
        context_hashes = {key: str(value["identity_sha256"])
                          for key, value in worker_contexts.items()}
        # Reuse the same canonical context validator used after acceptance.
        _v3_validate_context_acceptance(contract, configurations, {
            "worker_contexts": worker_contexts,
            "worker_environment_sha256": worker_environment_sha256,
            "worker_context_sha256_by_config": context_hashes,
        })
        identity = _v3_build_identity(
            contract, static, configurations, context_hashes,
            worker_environment_sha256, runtime_pins,
        )
    except BaseException as exc:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_path, {
            "manifest_id": V3_MANIFEST_ID, "status": "preflight_failed",
            "measurement_started": False,
            "manifest_sha256": static["manifest_sha256"],
            "base_contract_sha256": static["base_contract_sha256"],
            "resolved_contract_sha256": static["resolved_contract_sha256"],
            "code_hashes": static["code_hashes"], "planned_cells": len(cells),
            "error": f"{type(exc).__name__}:{exc}",
            "failed_control": getattr(exc, "failed_control", None),
        })
        raise

    evidence_sha = _v3_preflight_evidence_sha256(
        static, runtime_pins, configurations, worker_contexts, rank_records,
        worker_environment_sha256,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_id": V3_MANIFEST_ID, "status": "preflight_complete",
        "run_identity": identity, "run_identity_sha256": identity["run_identity_sha256"],
        "manifest_sha256": static["manifest_sha256"],
        "base_contract_sha256": static["base_contract_sha256"],
        "resolved_contract_sha256": static["resolved_contract_sha256"],
        "code_hashes": static["code_hashes"],
        "plan_sha256": static["plan_sha256"], "schedule_sha256": static["schedule_sha256"],
        "planned_cells": len(cells), "completed_cells": 0,
        "configuration_records": configurations, "runtime_pins": runtime_pins,
        "worker_contexts_by_config": worker_contexts,
        "worker_context_sha256_by_config": context_hashes,
        "worker_environment_sha256": worker_environment_sha256,
        "rank_preflight_records": rank_records,
        "preflight_evidence_sha256": evidence_sha,
        "selected_clock_id": str(contract["measurement"]["selected_clock_id"]),
        "selected_clock_field": str(contract["measurement"]["selected_clock_field"]),
        "measurement_started": False, "timing_run": False,
        "checkpoint_path": str(artifact_dir / "checkpoint.jsonl"),
        "checkpoint_sha256": "", "raw_records": {"path": str(artifact_dir / "raw_records.csv"), "rows": 0},
        "bootstrap_only": True,
    }
    _atomic_json(run_path, payload)
    return {
        "manifest_id": V3_MANIFEST_ID, "status": "preflight_complete",
        "run_identity_sha256": identity["run_identity_sha256"],
        "preflight_evidence_sha256": evidence_sha,
        "preflight_run_manifest_sha256": _sha256_file(run_path),
        "planned_cells": len(cells), "measurement_started": False,
        "timing_run": False,
    }


def _partial_bootstrap_preflight(
    manifest_path: Path, artifact_dir: Path, resolution: dict[str, Any],
    cells: list[dict[str, Any]], schedule: list[dict[str, Any]],
    static: dict[str, Any], resume: bool,
) -> dict[str, Any]:
    """Collect actual enabled rank/context controls, then stop before timing."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        PARTIAL_MANIFEST_ID, V3Unavailable, validate_partial_v3_rank_preflight_records,
    )
    from benchmark_v1.scripts.run_maestro_common_panel import maestro_timing_lock

    run_path = artifact_dir / "run_manifest.json"
    if artifact_dir.exists():
        if not resume:
            raise V3Unavailable("partial_preflight_bundle_exists_use_resume_or_acceptance")
        if not run_path.is_file():
            raise V3Unavailable("partial_preflight_existing_bundle_missing_run_manifest")
        old = load(run_path)
        if old.get("status") in {
            "preflight_failed", "failed_global_error", "pilot_gate_failed", "complete",
            "complete_with_unavailable_knots",
        }:
            raise V3Unavailable(f"partial_preflight_terminal_status:{old.get('status')}")
        if old.get("status") == "preflight_complete":
            raise V3Unavailable("partial_preflight_already_complete_request_final_acceptance")
        if old.get("measurement_started"):
            raise V3Unavailable("partial_bootstrap_refuses_bundle_after_measurement_started")
    elif resume:
        raise V3Unavailable("partial_preflight_resume_bundle_missing")

    try:
        with maestro_timing_lock(
            PARTIAL_MANIFEST_ID, _canonical_hash({
                **static,
                "quarantine_evidence": _partial_evidence_document(static["quarantine_evidence"]),
            }), root=ROOT,
        ):
            import maestro
            configurations = _v3_configuration_records(resolution["contract"], maestro)
            runtime_pins = _v3_runtime_pins(manifest_path, maestro)
            rank_records, worker_contexts = _v3_reference_and_candidate_rank_preflight(
                resolution["contract"], configurations, None,
                resolution["quarantine_policy"], static["quarantine_evidence"],
            )
        rank_qa = validate_partial_v3_rank_preflight_records(
            resolution["contract"], rank_records, resolution["quarantine_policy"],
            static["quarantine_evidence"],
        )
        if rank_qa.get("status") != "pass":
            raise V3Unavailable(f"partial_rank_preflight_failed:{rank_qa.get('unavailable_reasons')}")
        rank_records = rank_qa["records"]
        if set(worker_contexts) != set(configurations):
            raise V3Unavailable("partial_worker_context_config_set_mismatch")
        environment_hashes = {
            str(context.get("worker_environment_sha256", ""))
            for context in worker_contexts.values()
        }
        if len(environment_hashes) != 1 or "" in environment_hashes:
            raise V3Unavailable("partial_worker_environment_drift")
        worker_environment_sha256 = next(iter(environment_hashes))
        context_hashes = {
            key: str(value.get("identity_sha256", ""))
            for key, value in worker_contexts.items()
        }
        _v3_validate_context_acceptance(resolution["contract"], configurations, {
            "worker_contexts": worker_contexts,
            "worker_environment_sha256": worker_environment_sha256,
            "worker_context_sha256_by_config": context_hashes,
        })
        identity = _partial_build_identity(
            resolution, static, configurations, worker_contexts,
            worker_environment_sha256, runtime_pins,
        )
        evidence_sha = _partial_preflight_evidence_sha256(
            static, runtime_pins, configurations, worker_contexts,
            rank_records, worker_environment_sha256,
        )
    except BaseException as exc:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_path, {
            "manifest_id": PARTIAL_MANIFEST_ID, "status": "preflight_failed",
            "measurement_started": False, "timing_run": False,
            **{key: static[key] for key in (
                "manifest_sha256", "amendment_identity", "parent_manifest_sha256",
                "parent_contract_sha256", "base_contract_sha256", "resolved_contract_sha256",
                "code_hashes", "plan_sha256", "schedule_sha256", "planned_cells",
                "planned_enabled_cells", "planned_quarantined_cells", "planned_timed_calls",
                "planned_untimed_calls", "planned_quarantined_slots",
                "quarantine_evidence_sha256",
            )},
            "quarantine_evidence_by_config": {
                f"{key[0]}:{key[1]}:{key[2]}": value
                for key, value in static["quarantine_evidence"].items()
            },
            "error": f"{type(exc).__name__}:{exc}",
            "failed_control": getattr(exc, "failed_control", None),
        })
        raise

    preflight_evidence = {
        f"{key[0]}:{key[1]}:{key[2]}": value
        for key, value in static["quarantine_evidence"].items()
    }
    payload = {
        "manifest_id": PARTIAL_MANIFEST_ID, "status": "preflight_complete",
        "run_identity": identity["run_identity"], "run_identity_sha256": identity["run_identity_sha256"],
        **{key: static[key] for key in (
            "manifest_sha256", "amendment_identity", "parent_manifest_sha256",
            "parent_contract_sha256", "base_contract_sha256", "resolved_contract_sha256",
            "code_hashes", "plan_sha256", "schedule_sha256", "planned_cells",
            "planned_enabled_cells", "planned_quarantined_cells", "planned_timed_calls",
            "planned_untimed_calls", "planned_quarantined_slots",
            "first_ten_cell_ids", "pre_expansion_cell_ids",
            "quarantine_evidence_sha256",
        )},
        "configuration_records": configurations, "runtime_pins": runtime_pins,
        "worker_contexts_by_config": worker_contexts,
        "worker_context_sha256_by_config": context_hashes,
        "worker_environment_sha256": worker_environment_sha256,
        "rank_preflight_status": rank_qa["status"],
        "rank_preflight_records": rank_qa["records"],
        "rank_preflight_summary": rank_qa,
        "quarantine_evidence_by_config": preflight_evidence,
        "preflight_evidence_sha256": evidence_sha,
        "selected_clock_id": str(resolution["contract"]["measurement"]["selected_clock_id"]),
        "selected_clock_field": str(resolution["contract"]["measurement"]["selected_clock_field"]),
        "measurement_started": False, "timing_run": False, "score_created": False,
        "checkpoint_path": str(artifact_dir / "checkpoint.jsonl"),
        "checkpoint_format": "maestro_v3_partial_terminal_checkpoint_jsonl_v1",
        "checkpoint_sha256": "",
        "raw_records": {"path": str(artifact_dir / "raw_records.csv"), "rows": 0},
        "bootstrap_only": True,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(run_path, payload)
    return {
        "manifest_id": PARTIAL_MANIFEST_ID, "status": "preflight_complete",
        "run_identity_sha256": identity["run_identity_sha256"],
        "preflight_evidence_sha256": evidence_sha,
        "preflight_run_manifest_sha256": _sha256_file(run_path),
        "planned_cells": len(cells), "planned_enabled_cells": 399,
        "quarantined_cells": 33, "scheduled_slots": len(schedule),
        "measurement_started": False, "timing_run": False, "score_created": False,
    }


def run_v3_calibration(manifest_path: Path, artifact_dir: Path,
                       acceptance_path: Path | None, resume: bool,
                       max_cells: int | None) -> dict[str, Any] | None:
    """Bootstrap untimed controls first; measure only after exact acceptance."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        V3_MANIFEST_ID, V3_TIMED_STAGE, V3_UNTIMED_STAGE,
        V3Unavailable, build_v3_execution_schedule, build_v3_plan,
        canonical_hash, fit_v3_calibration_rows, generate_v3_probe,
        validate_v3_rank_preflight_records, validate_v3_resume_identity,
        v3_pilot_cell_ids,
    )

    if max_cells not in (None, 10, 20, 432):
        raise V3Unavailable("v3_max_cells_must_be_10_20_or_432")
    contract, cells, schedule, static = _v3_static_bundle(manifest_path)
    if len(cells) != 432 or contract.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("v3_frozen_plan_must_have_432_cells")
    artifact_dir = artifact_dir.resolve()
    semantic_dir = (ROOT / contract["outputs"]["calibration"]).resolve()
    if artifact_dir != semantic_dir:
        raise V3Unavailable("v3_artifact_dir_must_match_frozen_semantic_calibration_output")
    run_path = artifact_dir / "run_manifest.json"
    checkpoint_path = artifact_dir / "checkpoint.jsonl"
    raw_path = artifact_dir / "raw_records.csv"
    fit_path = artifact_dir / "fitted_calibration.json"
    if acceptance_path is None:
        return _v3_bootstrap_preflight(
            manifest_path, artifact_dir, contract, cells, schedule, static, resume
        )
    if not resume:
        raise V3Unavailable("v3_accepted_timing_requires_resume_of_preflight_bundle")
    try:
        acceptance_value = json.loads(acceptance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V3Unavailable(f"v3_acceptance_unreadable:{type(exc).__name__}") from exc
    acceptance = _v3_validate_acceptance(acceptance_value, static)
    if not run_path.is_file():
        raise V3Unavailable("v3_timing_requires_bootstrap_preflight_bundle")
    existing: dict[str, Any] = json.loads(run_path.read_text(encoding="utf-8"))
    if existing.get("status") in {"pilot_gate_failed", "failed_global_error", "preflight_failed"}:
        raise V3Unavailable(f"v3_resume_refused_terminal_status:{existing.get('status')}")

    # Static acceptance is checked before importing Maestro. Context/runtime
    # pins are then compared with the actual, previously saved worker preflight.
    try:
        import maestro
    except Exception as exc:
        raise V3Unavailable(f"v3_maestro_api_unavailable:{type(exc).__name__}:{exc}") from exc
    configurations = _v3_configuration_records(contract, maestro)
    runtime_pins = _v3_runtime_pins(manifest_path, maestro)
    if acceptance.get("runtime_pins") != runtime_pins:
        raise V3Unavailable("v3_acceptance_runtime_pins_mismatch")
    worker_contexts, worker_environment_sha256, context_hashes = (
        _v3_validate_context_acceptance(contract, configurations, acceptance)
    )
    identity = _v3_build_identity(
        contract, static, configurations, context_hashes,
        worker_environment_sha256, runtime_pins,
    )
    validate_v3_resume_identity(existing, identity)
    current_or_accepted_preflight_sha = (
        str(existing.get("accepted_preflight_run_manifest_sha256") or _sha256_file(run_path))
    )
    accepted_preflight_sha = _v3_validate_preflight_bundle(
        existing, acceptance, static, runtime_pins, configurations, worker_contexts,
        worker_environment_sha256, identity, current_or_accepted_preflight_sha,
    )
    acceptance_sha = _sha256_file(acceptance_path)
    if existing.get("acceptance_sha256") and existing.get("acceptance_sha256") != acceptance_sha:
        raise V3Unavailable("v3_resume_acceptance_file_hash_mismatch")
    if checkpoint_path.exists():
        if existing.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
            raise V3Unavailable("v3_resume_checkpoint_hash_mismatch")
    elif existing.get("checkpoint_sha256"):
        raise V3Unavailable("v3_resume_checkpoint_missing")
    if raw_path.exists():
        if existing.get("raw_records", {}).get("sha256") != _sha256_file(raw_path):
            raise V3Unavailable("v3_resume_raw_hash_mismatch")
    elif existing.get("raw_records", {}).get("sha256"):
        raise V3Unavailable("v3_resume_raw_records_missing")

    rank_records = existing.get("rank_preflight_records", [])
    rank_qa = validate_v3_rank_preflight_records(contract, rank_records)
    if rank_qa.get("status") != "pass":
        raise V3Unavailable("v3_accepted_rank_preflight_invalid")
    rank_records = rank_qa["records"]
    if not existing.get("accepted_preflight_run_manifest_sha256"):
        existing["accepted_preflight_run_manifest_sha256"] = accepted_preflight_sha
        existing["status"] = "preflight_accepted"
        existing["acceptance_sha256"] = acceptance_sha
        _atomic_json(run_path, existing)

    expected_context_by_key = worker_contexts
    policy = _v3_backend_threading_policy(contract)
    timeout = float(contract["measurement"]["cell_timeout_seconds"])
    results, started = _v3_read_checkpoint(checkpoint_path, schedule, identity["run_identity_sha256"])
    if raw_path.exists():
        with raw_path.open(newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
        if {str(row.get("attempt_id", "")) for row in raw_rows} != set(results):
            raise V3Unavailable("v3_resume_raw_checkpoint_attempt_set_mismatch")
    elif results:
        raise V3Unavailable("v3_resume_checkpoint_results_without_raw_records")
    all_rows = list(results.values())

    def write_bundle(status: str, completed_cells: int,
                     fit: Any | None = None, pilot_qa: dict[str, Any] | None = None,
                     error: str = "") -> None:
        raw_meta: dict[str, Any] = {"path": str(raw_path), "rows": len(all_rows)}
        if raw_path.exists():
            raw_meta["sha256"] = _sha256_file(raw_path)
        run_payload = {
            "manifest_id": V3_MANIFEST_ID, "status": status,
            "run_identity": identity, "run_identity_sha256": identity["run_identity_sha256"],
            "resolved_contract_sha256": static["resolved_contract_sha256"],
            "manifest_sha256": static["manifest_sha256"],
            "base_contract_sha256": static["base_contract_sha256"],
            "code_hashes": static["code_hashes"],
            "worker_contexts_by_config": expected_context_by_key,
            "worker_context_sha256_by_config": context_hashes,
            "worker_environment_sha256": worker_environment_sha256,
            "rank_preflight_records": rank_records,
            "preflight_evidence_sha256": existing["preflight_evidence_sha256"],
            "accepted_preflight_run_manifest_sha256": accepted_preflight_sha,
            "acceptance_sha256": acceptance_sha,
            "configuration_records": configurations,
            "runtime_pins": runtime_pins,
            "plan_sha256": static["plan_sha256"],
            "schedule_sha256": static["schedule_sha256"],
            "planned_cells": len(cells), "completed_cells": completed_cells,
            "max_cells_limit": max_cells,
            "max_cells_is_identity_pin": False,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_format": "maestro_v3_calibration_checkpoint_jsonl_v1",
            "checkpoint_sha256": _sha256_file(checkpoint_path) if checkpoint_path.exists() else "",
            "checkpoint_attempt_rows": len(all_rows), "raw_records": raw_meta,
            "selected_clock_id": str(contract["measurement"]["selected_clock_id"]),
            "selected_clock_field": str(contract["measurement"]["selected_clock_field"]),
            "timed_stage": V3_TIMED_STAGE, "untimed_stage": V3_UNTIMED_STAGE,
            "process_isolation": "one fresh spawned process per API call; no persistent-process warm claim",
            "measurement_started": bool(started), "pilot_qa": pilot_qa or {},
            "error": error,
        }
        if fit is not None:
            diagnostics = fit.fit_diagnostics
            fit_payload = {
                "fit_version": fit.fit_version,
                "calibration_manifest_id": fit.calibration_manifest_id,
                "calibration_raw_sha256": fit.calibration_raw_sha256,
                "run_identity_sha256": fit.run_identity_sha256,
                "clock_id": fit.clock_id,
                "calibration_timed_observations": fit.calibration_timed_observations,
                "valid_operation_knots": len(fit.operation_knots),
                "valid_sampling_intercept_knots": len(fit.sampling_intercept_knots),
                "valid_sampling_per_shot_knots": len(fit.sampling_per_shot_knots),
                "invalid_knots": [
                    {"key": list(key), "reason": value}
                    for key, value in sorted(fit.invalid_knot_states.items(), key=lambda pair: str(pair[0]))
                ],
                "operation_knots": [{"key": list(key), "coefficient": value}
                                    for key, value in sorted(fit.operation_knots.items(), key=lambda pair: str(pair[0]))],
                "sampling_intercept_knots": [{"key": list(key), "coefficient": value}
                                             for key, value in sorted(fit.sampling_intercept_knots.items(), key=lambda pair: str(pair[0]))],
                "sampling_per_shot_knots": [{"key": list(key), "coefficient": value}
                                            for key, value in sorted(fit.sampling_per_shot_knots.items(), key=lambda pair: str(pair[0]))],
                "interpolation_width_grid": {key: list(value) for key, value in fit.interpolation_width_grid.items()},
                "interpolation_chi_grid": list(fit.interpolation_chi_grid),
                "fit_diagnostics": diagnostics,
            }
            _atomic_json(fit_path, fit_payload)
            run_payload["fit_summary"] = diagnostics
            run_payload["fitted_calibration_path"] = str(fit_path)
            run_payload["fitted_calibration_sha256"] = _sha256_file(fit_path)
        _atomic_json(run_path, run_payload)

    if raw_path.exists():
        _v3_write_raw(raw_path, all_rows)
    pilot = v3_pilot_cell_ids(cells)
    ordered_ids = _v3_cell_order(cells, static)
    target_cells = 432 if max_cells is None else max_cells
    targets = [value for value in (10, 20, 432) if value <= target_cells]
    existing_cell_ids = {str(row.get("cell_id", "")) for row in all_rows}
    fit_result = None
    pilot_qa: dict[str, Any] = {}

    from benchmark_v1.scripts.run_maestro_common_panel import isolated_call, _v2_simple_execute_worker

    try:
        with maestro_timing_lock(V3_MANIFEST_ID, identity["run_identity_sha256"], root=ROOT):
            for target in targets:
                selected_ids = set(ordered_ids[:target])
                stage_events = [event for event in schedule if str(event["cell_id"]) in selected_ids]
                for event in stage_events:
                    attempt_id = str(event["attempt_id"])
                    if attempt_id in results:
                        continue
                    cell = next(item for item in cells if str(item["cell_id"]) == str(event["cell_id"]))
                    probe = generate_v3_probe(
                        contract, int(cell["width"]), str(cell["operation_class"]),
                        int(cell["operation_repeats"]),
                    )
                    for field in ("preparation_qasm_sha256", "variable_word_qasm_sha256", "whole_qasm_sha256"):
                        if probe[field] != event[field]:
                            raise V3Unavailable(f"v3_generated_probe_schedule_mismatch:{attempt_id}:{field}")
                    if attempt_id in started:
                        raise V3Unavailable(f"v3_resume_attempt_was_started_without_result:{attempt_id}")
                    _v3_append_jsonl(checkpoint_path, {
                        "record_type": "call_started", "manifest_id": V3_MANIFEST_ID,
                        "attempt_id": attempt_id, "run_identity_sha256": identity["run_identity_sha256"],
                    })
                    started.add(attempt_id)
                    config_key = f"{event['candidate']}:{event['chi']}"
                    call = isolated_call(_v2_simple_execute_worker, (
                        probe["program"], str(event["candidate"]),
                        None if event["chi"] is None else int(event["chi"]),
                        int(event["shots"]), int(contract["seed"]),
                        float(contract["panel_candidate_configuration"]["mps_singular_value_threshold"]),
                        policy, expected_context_by_key[config_key],
                    ), timeout)
                    transport = str(call.get("transport_status", "missing"))
                    context = call.get("worker_context") if isinstance(call.get("worker_context"), dict) else {}
                    row = {
                        **event,
                        "record_type": "attempt_result", "run_identity_sha256": identity["run_identity_sha256"],
                        "status": "ok" if call.get("ok") else ("timeout" if call.get("timeout") else "error"),
                        "transport_status": transport,
                        "clock_id": str(contract["measurement"]["selected_clock_id"]),
                        "selected_clock_id": str(contract["measurement"]["selected_clock_id"]),
                        "selected_clock_field": str(contract["measurement"]["selected_clock_field"]),
                        "reported_time_seconds": (call.get("reported_time_seconds", "")
                                                   if event["stage"] == V3_TIMED_STAGE else ""),
                        "host_wall_seconds": call.get("host_wall_seconds", ""),
                        "outer_wall_seconds": call.get("elapsed_seconds", ""),
                        "call_process_isolated": True,
                        "worker_context_sha256": context.get("identity_sha256", ""),
                        "worker_environment_sha256": context.get("worker_environment_sha256", ""),
                        "resolved_config_sha256": context.get("resolved_config_sha256", ""),
                        "backend_threading_policy_sha256": context.get("backend_threading_policy_sha256", ""),
                        "worker_pid": context.get("observation", {}).get("pid", ""),
                        "error": call.get("error", ""),
                    }
                    _v3_append_jsonl(checkpoint_path, {
                        "record_type": "attempt_result", "manifest_id": V3_MANIFEST_ID,
                        "attempt_id": attempt_id, "run_identity_sha256": identity["run_identity_sha256"],
                        "row": row,
                    })
                    results[attempt_id] = row
                    all_rows.append(row)
                    _v3_write_raw(raw_path, all_rows)
                    existing_cell_ids.add(str(event["cell_id"]))
                    completed_count = sum(
                        1 for cell_id in ordered_ids
                        if all(str(item["attempt_id"]) in results
                               for item in schedule if str(item["cell_id"]) == cell_id)
                    )
                    write_bundle("running", completed_count)
                    if transport != "ok" or not call.get("ok"):
                        write_bundle("failed_global_error", len(existing_cell_ids), error=f"{attempt_id}:{transport}:{call.get('error', '')}")
                        raise V3Unavailable(f"v3_global_transport_or_worker_error:{attempt_id}:{transport}")
                    actual_context = context
                    from benchmark_v1.scripts.run_maestro_common_panel import validate_worker_context
                    try:
                        validate_worker_context(expected_context_by_key[config_key], actual_context)
                    except Exception as exc:
                        write_bundle("failed_global_error", completed_count,
                                     error=f"worker_context_drift:{attempt_id}:{exc}")
                        raise

                # Fit after each mandatory prefix. The 512 values are QA only;
                # the fitter uses 256/1024 endpoints for coefficient slopes.
                _v3_write_raw(raw_path, all_rows)
                fit_result = fit_v3_calibration_rows(
                    all_rows, contract, _sha256_file(raw_path),
                    expected_run_identity_sha256=identity["run_identity_sha256"],
                    rank_preflight_records=rank_records,
                )
                current_qa = fit_result.fit_diagnostics["cell_qa"]
                gate_ids = (pilot["first_ten_cell_ids"] if target == 10
                            else pilot["pre_expansion_cell_ids"] if target == 20 else [])
                if gate_ids:
                    pilot_qa[str(target)] = {
                        "status": "pass" if all(current_qa.get(cell_id, {}).get("status") == "ok" for cell_id in gate_ids) else "fail",
                        "cell_status": {cell_id: current_qa.get(cell_id, {}).get("status", "missing")
                                        for cell_id in gate_ids},
                    }
                    if pilot_qa[str(target)]["status"] != "pass":
                        write_bundle("pilot_gate_failed", len(existing_cell_ids), fit_result, pilot_qa)
                        raise V3Unavailable(f"v3_mandatory_{target}_cell_pilot_gate_failed")

                completed_count = sum(
                    1 for cell_id in ordered_ids
                    if all(str(item["attempt_id"]) in results
                           for item in schedule if str(item["cell_id"]) == cell_id)
                )
                complete = target == 432 and completed_count == 432
                status = ("complete_with_unavailable_knots" if complete and fit_result.invalid_knot_states
                          else "complete" if complete else "checkpointed_partial")
                write_bundle(status, completed_count, fit_result, pilot_qa)
                if target == target_cells:
                    break
    except BaseException:
        raise


def run_partial_calibration(
    manifest_path: Path, artifact_dir: Path, acceptance_path: Path | None,
    resume: bool, max_cells: int | None,
) -> dict[str, Any] | None:
    """Run only the S6-amended partial allocation; never resume failed v3."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        PARTIAL_FIT_VERSION, PARTIAL_MANIFEST_ID, PARTIAL_QUARANTINE_REASON,
        V3_TIMED_STAGE, V3_UNTIMED_STAGE, V3Unavailable,
        fit_v3_calibration_rows,
        generate_v3_probe, partial_v3_pilot_cell_ids,
        validate_partial_v3_rank_preflight_records,
        validate_partial_v3_resume_identity,
    )
    from benchmark_v1.scripts.run_maestro_common_panel import (
        isolated_call, maestro_timing_lock, validate_worker_context,
        _v2_simple_execute_worker,
    )

    if max_cells not in (None, 10, 20, 399):
        raise V3Unavailable("partial_max_cells_must_be_10_20_or_399")
    resolution, cells, schedule, static = _partial_static_bundle(manifest_path)
    artifact_dir = artifact_dir.resolve()
    semantic_dir = (ROOT / resolution["amendment"]["outputs"]["calibration"]).resolve()
    if artifact_dir != semantic_dir:
        raise V3Unavailable("partial_artifact_dir_must_match_frozen_semantic_calibration_output")
    run_path = artifact_dir / "run_manifest.json"
    checkpoint_path = artifact_dir / "checkpoint.jsonl"
    raw_path = artifact_dir / "raw_records.csv"
    fit_path = artifact_dir / "fitted_calibration.json"

    if acceptance_path is None:
        return _partial_bootstrap_preflight(
            manifest_path, artifact_dir, resolution, cells, schedule, static, resume,
        )
    if not resume:
        raise V3Unavailable("partial_accepted_timing_requires_resume_of_preflight_bundle")
    try:
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V3Unavailable(f"partial_acceptance_unreadable:{type(exc).__name__}") from exc
    _partial_acceptance_static_pins(acceptance, static)
    if not run_path.is_file():
        raise V3Unavailable("partial_timing_requires_bootstrap_preflight_bundle")
    existing = load(run_path)
    if existing.get("status") in {
        "preflight_failed", "failed_global_error", "pilot_gate_failed",
        "complete", "complete_with_unavailable_knots",
    }:
        raise V3Unavailable(f"partial_resume_refused_terminal_status:{existing.get('status')}")
    if existing.get("manifest_id") != PARTIAL_MANIFEST_ID:
        raise V3Unavailable("partial_resume_rejects_non_partial_bundle")
    if existing.get("status") not in {
        "preflight_complete", "preflight_accepted", "checkpointed_partial",
    }:
        raise V3Unavailable(f"partial_resume_status_invalid:{existing.get('status')}")

    # Exact static acceptance is checked before importing Maestro. Actual
    # runtime/config/context identities are then recomputed and compared.
    try:
        import maestro
    except Exception as exc:
        raise V3Unavailable(f"partial_maestro_api_unavailable:{type(exc).__name__}:{exc}") from exc
    contract = resolution["contract"]
    configurations = _v3_configuration_records(contract, maestro)
    runtime_pins = _v3_runtime_pins(manifest_path, maestro)
    worker_contexts, worker_environment_sha256, context_hashes = (
        _v3_validate_context_acceptance(contract, configurations, acceptance)
    )
    identity = _partial_build_identity(
        resolution, static, configurations, worker_contexts,
        worker_environment_sha256, runtime_pins,
    )
    validate_partial_v3_resume_identity(existing, identity)
    preflight_manifest_sha256 = str(
        existing.get("accepted_preflight_run_manifest_sha256") or _sha256_file(run_path)
    )
    accepted_preflight_sha256 = _partial_validate_preflight_bundle(
        existing, acceptance, static, runtime_pins, configurations, worker_contexts,
        worker_environment_sha256, identity, preflight_manifest_sha256,
    )
    quarantine_evidence = static["quarantine_evidence"]
    rank_qa = validate_partial_v3_rank_preflight_records(
        contract, existing.get("rank_preflight_records", []),
        resolution["quarantine_policy"], quarantine_evidence,
    )
    if rank_qa.get("status") != "pass":
        raise V3Unavailable("partial_accepted_rank_preflight_invalid")
    acceptance_sha256 = _sha256_file(acceptance_path)
    if existing.get("acceptance_sha256") not in (None, "", acceptance_sha256):
        raise V3Unavailable("partial_resume_acceptance_file_hash_mismatch")
    if checkpoint_path.exists():
        if existing.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
            raise V3Unavailable("partial_resume_checkpoint_hash_mismatch")
    elif existing.get("checkpoint_sha256"):
        raise V3Unavailable("partial_resume_checkpoint_missing")
    if raw_path.exists():
        if existing.get("raw_records", {}).get("sha256") != _sha256_file(raw_path):
            raise V3Unavailable("partial_resume_raw_hash_mismatch")
    elif existing.get("raw_records", {}).get("sha256"):
        raise V3Unavailable("partial_resume_raw_records_missing")

    if not existing.get("accepted_preflight_run_manifest_sha256"):
        existing["accepted_preflight_run_manifest_sha256"] = accepted_preflight_sha256
        existing["status"] = "preflight_accepted"
        existing["acceptance_sha256"] = acceptance_sha256
        _atomic_json(run_path, existing)
    expected_context_by_key = worker_contexts
    threading_policy = _v3_backend_threading_policy(contract)
    timeout = float(contract["measurement"]["cell_timeout_seconds"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _partial_initialize_checkpoint(checkpoint_path, schedule, identity["run_identity_sha256"])
    results, started, skipped_ids, budget_stopped_ids = _partial_read_checkpoint(
        checkpoint_path, schedule, identity["run_identity_sha256"],
    )
    if raw_path.exists():
        with raw_path.open(newline="", encoding="utf-8") as handle:
            saved_raw = list(csv.DictReader(handle))
        if {str(row.get("attempt_id", "")) for row in saved_raw} != set(results):
            raise V3Unavailable("partial_resume_raw_checkpoint_attempt_set_mismatch")
    elif results:
        raise V3Unavailable("partial_resume_checkpoint_results_without_raw_records")
    all_rows = list(results.values())
    evidence_by_config = {
        f"{key[0]}:{key[1]}:{key[2]}": value
        for key, value in quarantine_evidence.items()
    }

    def completed_cell_ids() -> set[str]:
        by_cell: dict[str, list[dict[str, Any]]] = {}
        for event in schedule:
            by_cell.setdefault(str(event["cell_id"]), []).append(event)
        complete: set[str] = set()
        for cell_id, events in by_cell.items():
            if events[0]["calibration_eligibility"] != "eligible":
                continue
            if all(str(event["attempt_id"]) in results for event in events):
                complete.add(cell_id)
        return complete

    def fit_payload_for(fit: Any) -> dict[str, Any]:
        return {
            "fit_version": fit.fit_version,
            "calibration_manifest_id": fit.calibration_manifest_id,
            "calibration_raw_sha256": fit.calibration_raw_sha256,
            "run_identity_sha256": fit.run_identity_sha256,
            "resolved_contract_sha256": fit.resolved_contract_sha256,
            "clock_id": fit.clock_id,
            "calibration_warm_observations": 0,
            "calibration_timed_observations": fit.calibration_timed_observations,
            "valid_operation_knots": len(fit.operation_knots),
            "valid_sampling_intercept_knots": len(fit.sampling_intercept_knots),
            "valid_sampling_per_shot_knots": len(fit.sampling_per_shot_knots),
            "invalid_knots": [
                {"key": list(key), "reason": reason}
                for key, reason in sorted(fit.invalid_knot_states.items(), key=lambda item: str(item[0]))
            ],
            "operation_knots": [
                {"key": list(key), "coefficient": value}
                for key, value in sorted(fit.operation_knots.items(), key=lambda item: str(item[0]))
            ],
            "sampling_intercept_knots": [
                {"key": list(key), "coefficient": value}
                for key, value in sorted(fit.sampling_intercept_knots.items(), key=lambda item: str(item[0]))
            ],
            "sampling_per_shot_knots": [
                {"key": list(key), "coefficient": value}
                for key, value in sorted(fit.sampling_per_shot_knots.items(), key=lambda item: str(item[0]))
            ],
            "interpolation_width_grid": {
                key: list(value) for key, value in fit.interpolation_width_grid.items()
            },
            "interpolation_chi_grid": list(fit.interpolation_chi_grid),
            "fit_diagnostics": fit.fit_diagnostics,
            "amendment_identity": static["amendment_identity"],
            "parent_contract_sha256": static["parent_contract_sha256"],
            "quarantine_evidence_by_config": evidence_by_config,
            "code_hashes": static["code_hashes"],
            "configuration_records": configurations,
            "worker_context_sha256_by_config": context_hashes,
            "worker_environment_sha256": worker_environment_sha256,
        }

    def write_bundle(
        status: str, fit: Any | None = None, pilot_qa: dict[str, Any] | None = None,
        error: str = "", *, pilot_first_timed_monotonic: float | None = None,
        pilot_first_timed_wall_utc: str = "", budget_reason: str = "",
    ) -> None:
        if all_rows:
            _v3_write_raw(raw_path, all_rows)
        raw_meta: dict[str, Any] = {"path": str(raw_path), "rows": len(all_rows)}
        if raw_path.exists():
            raw_meta["sha256"] = _sha256_file(raw_path)
            raw_meta["attempt_ids_sha256"] = _canonical_hash(
                sorted(str(row["attempt_id"]) for row in all_rows)
            )
        fit_summary: dict[str, Any] | None = None
        if fit is not None:
            fit_summary = fit_payload_for(fit)
            _atomic_json(fit_path, fit_summary)
        completed = completed_cell_ids()
        timed_started = any(
            event.get("stage") == V3_TIMED_STAGE
            and str(event["attempt_id"]) in started
            for event in schedule
        )
        checkpoint_sha = _sha256_file(checkpoint_path) if checkpoint_path.is_file() else ""
        payload = {
            "manifest_id": PARTIAL_MANIFEST_ID, "status": status,
            "run_identity": identity["run_identity"], "run_identity_sha256": identity["run_identity_sha256"],
            **{key: static[key] for key in (
                "manifest_sha256", "amendment_identity", "parent_manifest_id",
                "parent_manifest_sha256", "parent_contract_sha256", "base_contract_sha256",
                "resolved_contract_sha256", "code_hashes", "plan_sha256", "schedule_sha256",
                "planned_cells", "planned_enabled_cells", "planned_quarantined_cells",
                "planned_timed_calls", "planned_untimed_calls", "planned_quarantined_slots",
                "first_ten_cell_ids", "pre_expansion_cell_ids",
                "quarantine_evidence_sha256",
            )},
            "configuration_records": configurations, "runtime_pins": runtime_pins,
            "worker_contexts_by_config": expected_context_by_key,
            "worker_context_sha256_by_config": context_hashes,
            "worker_environment_sha256": worker_environment_sha256,
            "rank_preflight_status": "pass",
            "rank_preflight_records": rank_qa["records"],
            "rank_preflight_summary": rank_qa,
            "quarantine_evidence_by_config": evidence_by_config,
            "preflight_evidence_sha256": existing["preflight_evidence_sha256"],
            "accepted_preflight_run_manifest_sha256": accepted_preflight_sha256,
            "acceptance_sha256": acceptance_sha256,
            "accepted_cell_prefix": 399,
            "planned_cells": 432, "completed_cells": len(completed),
            "completed_cell_ids": sorted(completed),
            "max_cells_limit": max_cells, "max_cells_is_identity_pin": False,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_format": "maestro_v3_partial_terminal_checkpoint_jsonl_v1",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_allocation_rows": 10_368,
            "checkpoint_attempt_rows": len(all_rows),
            "checkpoint_quarantined_terminal_rows": len(skipped_ids),
            "checkpoint_budget_terminal_rows": len(budget_stopped_ids),
            "checkpoint_terminal_ledger_complete": (
                len(skipped_ids) == 792 and not budget_reason
                and status in {"complete", "complete_with_unavailable_knots"}
            ),
            "raw_records": raw_meta,
            "selected_clock_id": str(contract["measurement"]["selected_clock_id"]),
            "selected_clock_field": str(contract["measurement"]["selected_clock_field"]),
            "timed_stage": V3_TIMED_STAGE, "untimed_stage": V3_UNTIMED_STAGE,
            "process_isolation": "one fresh spawned process per API call; no persistent-process warm claim",
            "measurement_started": bool(timed_started), "timing_run": bool(timed_started),
            "score_created": False, "pilot_qa": pilot_qa or {},
            "pilot_first_timed_monotonic": pilot_first_timed_monotonic,
            "pilot_first_timed_wall_utc": pilot_first_timed_wall_utc,
            "pilot_budget_seconds": 7200,
            "pilot_budget_stop_reason": budget_reason,
            "fit_summary": fit_summary,
            "fitted_calibration_path": str(fit_path) if fit is not None else "",
            "fitted_calibration_sha256": _sha256_file(fit_path) if fit is not None else "",
            "error": error,
        }
        _atomic_json(run_path, payload)

    fit_result = None
    pilot_qa: dict[str, Any] = dict(existing.get("pilot_qa", {}))
    pilot_first_timed_monotonic = existing.get("pilot_first_timed_monotonic")
    pilot_first_timed_wall_utc = str(existing.get("pilot_first_timed_wall_utc", ""))
    cell_order = list(static["pre_expansion_cell_ids"])
    seen_cells = set(cell_order)
    cell_order.extend(
        str(cell["cell_id"]) for cell in cells
        if cell.get("calibration_eligibility") == "eligible"
        and str(cell["cell_id"]) not in seen_cells
    )
    if len(cell_order) != 399 or len(set(cell_order)) != 399:
        raise V3Unavailable("partial_enabled_cell_order_mismatch")
    target_cells = 399 if max_cells is None else max_cells
    targets = [value for value in (10, 20, 399) if value <= target_cells]
    cell_by_id = {str(cell["cell_id"]): cell for cell in cells}
    pilot_ids = partial_v3_pilot_cell_ids(cells)
    budget = resolution["amendment"]["pilot_policy"]
    budget_seconds = float(budget["pilot_timing_budget_seconds"])
    cutoff = datetime(2026, 10, 2, 10, 0, tzinfo=timezone(timedelta(hours=7)))

    def stop_for_budget() -> str:
        now_local = datetime.now(timezone(timedelta(hours=7)))
        if now_local >= cutoff:
            return "not_started_due_to_deadline"
        if (pilot_first_timed_monotonic is not None
                and pilot_qa.get("20", {}).get("status") != "pass"):
            if time.monotonic() - float(pilot_first_timed_monotonic) >= budget_seconds:
                return "not_started_due_to_pilot_budget"
        return ""

    def mark_budget_stops(reason: str) -> None:
        for event in schedule:
            attempt_id = str(event["attempt_id"])
            if event.get("calibration_eligibility") != "eligible":
                continue
            if attempt_id in results or attempt_id in budget_stopped_ids:
                continue
            row = {
                **event, "record_type": reason,
                "timing_status": "not_started", "timing_reason": reason,
                "reported_time_seconds": "", "host_wall_seconds": "", "outer_wall_seconds": "",
            }
            _v3_append_jsonl(checkpoint_path, {
                "record_type": "not_started_due_to_pilot_budget",
                "manifest_id": PARTIAL_MANIFEST_ID, "attempt_id": attempt_id,
                "run_identity_sha256": identity["run_identity_sha256"], "row": row,
            })
            budget_stopped_ids.add(attempt_id)

    try:
        with maestro_timing_lock(PARTIAL_MANIFEST_ID, identity["run_identity_sha256"], root=ROOT):
            for target in targets:
                selected_ids = set(cell_order[:target])
                stage_events = [
                    event for event in schedule
                    if str(event["cell_id"]) in selected_ids
                    and event.get("calibration_eligibility") == "eligible"
                ]
                for event in stage_events:
                    attempt_id = str(event["attempt_id"])
                    if attempt_id in results:
                        continue
                    if attempt_id in budget_stopped_ids:
                        continue
                    cell = cell_by_id[str(event["cell_id"])]
                    probe = generate_v3_probe(
                        contract, int(cell["width"]), str(cell["operation_class"]),
                        int(cell["operation_repeats"]),
                    )
                    for field in (
                        "preparation_qasm_sha256", "variable_word_qasm_sha256",
                        "whole_qasm_sha256",
                    ):
                        if probe[field] != event[field]:
                            raise V3Unavailable(
                                f"partial_generated_probe_schedule_mismatch:{attempt_id}:{field}"
                            )
                    if attempt_id in started:
                        raise V3Unavailable(f"partial_resume_attempt_started_without_result:{attempt_id}")
                    if event["stage"] == V3_TIMED_STAGE:
                        stop_reason = stop_for_budget()
                        if stop_reason:
                            mark_budget_stops(stop_reason)
                            write_bundle(
                                "checkpointed_partial", fit_result, pilot_qa,
                                pilot_first_timed_monotonic=pilot_first_timed_monotonic,
                                pilot_first_timed_wall_utc=pilot_first_timed_wall_utc,
                                budget_reason=stop_reason,
                            )
                            return {
                                "manifest_id": PARTIAL_MANIFEST_ID,
                                "status": "checkpointed_partial",
                                "reason": stop_reason,
                                "completed_cells": len(completed_cell_ids()),
                                "measurement_started": bool(pilot_first_timed_monotonic),
                                "timing_run": bool(pilot_first_timed_monotonic),
                            }
                        if pilot_first_timed_monotonic is None:
                            pilot_first_timed_monotonic = time.monotonic()
                            pilot_first_timed_wall_utc = datetime.now(timezone.utc).isoformat()
                            write_bundle(
                                "running", fit_result, pilot_qa,
                                pilot_first_timed_monotonic=pilot_first_timed_monotonic,
                                pilot_first_timed_wall_utc=pilot_first_timed_wall_utc,
                            )
                    _v3_append_jsonl(checkpoint_path, {
                        "record_type": "call_started", "manifest_id": PARTIAL_MANIFEST_ID,
                        "attempt_id": attempt_id,
                        "run_identity_sha256": identity["run_identity_sha256"],
                    })
                    started.add(attempt_id)
                    config_key = f"{event['candidate']}:{event['chi']}"
                    call = isolated_call(_v2_simple_execute_worker, (
                        probe["program"], str(event["candidate"]),
                        None if event["chi"] is None else int(event["chi"]),
                        int(event["shots"]), int(contract["seed"]),
                        float(contract["panel_candidate_configuration"]["mps_singular_value_threshold"]),
                        threading_policy, expected_context_by_key[config_key],
                    ), timeout)
                    transport = str(call.get("transport_status", "missing"))
                    context = call.get("worker_context") if isinstance(call.get("worker_context"), dict) else {}
                    status = "ok" if call.get("ok") else (
                        "timeout" if call.get("timeout") else "error"
                    )
                    row = {
                        **event, "record_type": "attempt_result",
                        "run_identity_sha256": identity["run_identity_sha256"],
                        "status": status, "timing_status": status,
                        "timing_reason": str(call.get("error", "")),
                        "transport_status": transport,
                        "clock_id": str(contract["measurement"]["selected_clock_id"]),
                        "selected_clock_id": str(contract["measurement"]["selected_clock_id"]),
                        "selected_clock_field": str(contract["measurement"]["selected_clock_field"]),
                        "reported_time_seconds": (
                            call.get("reported_time_seconds", "")
                            if event["stage"] == V3_TIMED_STAGE else ""
                        ),
                        "host_wall_seconds": call.get("host_wall_seconds", ""),
                        "outer_wall_seconds": call.get("elapsed_seconds", ""),
                        "call_process_isolated": True,
                        "worker_context_sha256": context.get("identity_sha256", ""),
                        "worker_environment_sha256": context.get("worker_environment_sha256", ""),
                        "resolved_config_sha256": context.get("resolved_config_sha256", ""),
                        "backend_threading_policy_sha256": context.get("backend_threading_policy_sha256", ""),
                        "worker_pid": context.get("observation", {}).get("pid", ""),
                        "error": call.get("error", ""),
                    }
                    _v3_append_jsonl(checkpoint_path, {
                        "record_type": "attempt_result", "manifest_id": PARTIAL_MANIFEST_ID,
                        "attempt_id": attempt_id,
                        "run_identity_sha256": identity["run_identity_sha256"], "row": row,
                    })
                    results[attempt_id] = row
                    all_rows.append(row)
                    _v3_write_raw(raw_path, all_rows)
                    if transport != "ok" or not call.get("ok"):
                        write_bundle(
                            "failed_global_error", fit_result, pilot_qa,
                            error=f"{attempt_id}:{transport}:{call.get('error', '')}",
                            pilot_first_timed_monotonic=pilot_first_timed_monotonic,
                            pilot_first_timed_wall_utc=pilot_first_timed_wall_utc,
                        )
                        raise V3Unavailable(
                            f"partial_global_transport_or_worker_error:{attempt_id}:{transport}"
                        )
                    try:
                        validate_worker_context(expected_context_by_key[config_key], context)
                    except Exception as exc:
                        write_bundle(
                            "failed_global_error", fit_result, pilot_qa,
                            error=f"worker_context_drift:{attempt_id}:{exc}",
                            pilot_first_timed_monotonic=pilot_first_timed_monotonic,
                            pilot_first_timed_wall_utc=pilot_first_timed_wall_utc,
                        )
                        raise V3Unavailable(f"partial_worker_context_drift:{attempt_id}:{exc}") from exc
                    write_bundle(
                        "running", fit_result, pilot_qa,
                        pilot_first_timed_monotonic=pilot_first_timed_monotonic,
                        pilot_first_timed_wall_utc=pilot_first_timed_wall_utc,
                    )

                _v3_write_raw(raw_path, all_rows)
                try:
                    fit_result = fit_v3_calibration_rows(
                        all_rows, contract, _sha256_file(raw_path),
                        expected_run_identity_sha256=identity["run_identity_sha256"],
                        rank_preflight_records=rank_qa["records"],
                        partial_resolution=resolution,
                        partial_quarantine_evidence=quarantine_evidence,
                    )
                except BaseException as exc:
                    write_bundle(
                        "failed_global_error", fit_result,
                        error=f"fit_or_protocol_validation:{type(exc).__name__}:{exc}",
                        pilot_first_timed_monotonic=pilot_first_timed_monotonic,
                        pilot_first_timed_wall_utc=pilot_first_timed_wall_utc,
                    )
                    raise
                cell_qa = fit_result.fit_diagnostics["cell_qa"]
                required_ids = (pilot_ids["first_ten_cell_ids"] if target == 10
                                else pilot_ids["pre_expansion_cell_ids"] if target == 20 else [])
                if required_ids:
                    pilot_qa[str(target)] = {
                        "status": "pass" if all(
                            cell_qa.get(cell_id, {}).get("status") == "ok"
                            for cell_id in required_ids
                        ) else "fail",
                        "cell_status": {
                            cell_id: cell_qa.get(cell_id, {}).get("status", "missing")
                            for cell_id in required_ids
                        },
                        "required_cells": required_ids,
                        "reuse_first_ten_in_twenty": target == 20,
                    }
                    if pilot_qa[str(target)]["status"] != "pass":
                        write_bundle(
                            "pilot_gate_failed", fit_result, pilot_qa,
                            pilot_first_timed_monotonic=pilot_first_timed_monotonic,
                            pilot_first_timed_wall_utc=pilot_first_timed_wall_utc,
                        )
                        raise V3Unavailable(f"partial_mandatory_{target}_cell_pilot_gate_failed")

                completed = len(completed_cell_ids())
                complete = target == 399 and completed == 399
                status = (
                    "complete_with_unavailable_knots"
                    if complete and fit_result.invalid_knot_states else
                    "complete" if complete else "checkpointed_partial"
                )
                if target == 20:
                    observed_outer = [
                        float(row["outer_wall_seconds"]) for row in all_rows
                        if row.get("stage") in {V3_TIMED_STAGE, V3_UNTIMED_STAGE}
                        and row.get("outer_wall_seconds") not in (None, "")
                    ]
                    median_outer = float(statistics.median(observed_outer)) if observed_outer else None
                    remaining_calls = max(0, 9_576 - len(all_rows))
                    pilot_qa["remaining_cost_estimate"] = {
                        "observed_outer_wall_median_seconds": median_outer,
                        "remaining_api_calls": remaining_calls,
                        "estimated_remaining_seconds": (
                            median_outer * remaining_calls if median_outer is not None else None
                        ),
                        "includes_process_startup": True,
                        "uncertain": True,
                    }
                write_bundle(
                    status, fit_result, pilot_qa,
                    pilot_first_timed_monotonic=pilot_first_timed_monotonic,
                    pilot_first_timed_wall_utc=pilot_first_timed_wall_utc,
                )
                if target == target_cells:
                    break
    except BaseException:
        raise
    return {
        "manifest_id": PARTIAL_MANIFEST_ID,
        "status": load(run_path)["status"],
        "completed_cells": len(completed_cell_ids()),
        "planned_cells": 432,
        "eligible_cells": 399,
        "quarantined_cells": 33,
        "quarantined_slots": 792,
        "measurement_started": bool(pilot_first_timed_monotonic),
        "timing_run": bool(pilot_first_timed_monotonic),
        "score_created": False,
        "run_manifest": str(run_path),
        "checkpoint": str(checkpoint_path),
        "raw_records": str(raw_path),
        "fitted_calibration": str(fit_path),
    }


def _v3_checked_preparation_rank(amplitudes: Any, width: int, candidate: str,
                                 chi: int | None, qasm: str, rank_tol: float,
                                 norm_tol: float,
                                 native_max_bond_dim_reached: Any = None) -> dict[str, Any]:
    """Preserve the frozen check and attribute failures to the exact control."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        V3Unavailable, assess_v3_candidate_preparation, central_schmidt_rank_v3,
    )

    try:
        if candidate == "mps_fixed_chi":
            return assess_v3_candidate_preparation(
                amplitudes, width, int(chi), native_max_bond_dim_reached,
                rank_tol, norm_tol,
            )
        return central_schmidt_rank_v3(amplitudes, width, rank_tol, norm_tol)
    except V3Unavailable as exc:
        control = {
            "candidate": candidate, "width": int(width), "chi": chi,
            "qasm_sha256": hash_text(qasm), "norm_atol": float(norm_tol),
            "original_error": str(exc),
        }
        failure = V3Unavailable(
            f"{exc}:candidate={candidate}:n={width}:chi={chi}:"
            f"qasm_sha256={control['qasm_sha256']}"
        )
        failure.failed_control = control
        raise failure from exc


def zero_state_width_control_qasm(contract: dict[str, Any], width: int) -> str:
    """Preserve declared width for an untimed state control on operation-sized APIs."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import generate_v3_probe
    probe = generate_v3_probe(contract, width, "zero_gate_sample", 0, include_measurement=True)
    return probe["program"]


def assess_zero_state_width_control(result: dict[str, Any], program: str,
                                    width: int, candidate: str, chi: int,
                                    norm_tol: float) -> dict[str, Any]:
    """Retain shape, numerical and worker evidence even when the control fails."""
    from benchmark_v1.qre_benchmark.maestro_component_v2 import V3Unavailable, canonical_hash
    amplitudes = result.get("amplitudes")
    control = {
        "candidate": candidate, "width": int(width), "chi": int(chi),
        "qasm": program, "qasm_sha256": hash_text(program),
        "declared_width": int(width), "parsed_width": result.get("parsed_width"),
        "expected_amplitude_count": 2**int(width),
        "observed_amplitude_count": len(amplitudes) if isinstance(amplitudes, (list, tuple)) else None,
        "worker_context": result.get("worker_context"),
        "transport_status": result.get("transport_status"),
        "norm_atol": norm_tol, "status": "unavailable",
    }
    def fail(reason: str) -> None:
        control["reason"] = reason
        exc = V3Unavailable(f"{reason}:{candidate}:{chi}:n={width}")
        exc.failed_control = control
        raise exc
    if not result.get("ok") or result.get("transport_status") != "ok":
        control["error"] = result.get("error", "")
        fail("partial_zero_state_preflight_transport_failure")
    if not isinstance(amplitudes, (list, tuple)):
        fail("partial_zero_state_amplitude_shape_mismatch")
    vector = []
    norm = 0.0
    for raw in amplitudes:
        value = complex(raw)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            fail("partial_zero_state_nonfinite_amplitude")
        vector.append([float(value.real), float(value.imag)])
        norm += abs(value)**2
    control.update({"amplitude_sha256": canonical_hash(vector), "squared_norm": norm,
                    "zero_basis_probability": abs(complex(amplitudes[0]))**2 if amplitudes else None})
    if result.get("parsed_width") != int(width):
        fail("partial_zero_state_parsed_width_mismatch")
    if len(amplitudes) != 2**int(width):
        fail("partial_zero_state_amplitude_shape_mismatch")
    if abs(norm - 1.0) > norm_tol:
        fail("partial_zero_state_norm_out_of_tolerance")
    if abs(control["zero_basis_probability"] - 1.0) > norm_tol:
        fail("partial_zero_state_not_zero_basis")
    context = result.get("worker_context")
    if not isinstance(context, dict) or not context.get("identity_sha256"):
        fail("partial_zero_state_preflight_context_missing")
    control.update({"status": "ok", "reason": "",
                    "worker_context_sha256": context["identity_sha256"],
                    "control_role": "separate_zero_state_sampling_normalization_gate"})
    return control


def _v3_preparation_worker(qasm: str, candidate: str, chi: int | None,
                           seed: int, singular_value_threshold: float,
                           backend_threading_policy: dict[str, Any],
                           expected_worker_context: dict[str, Any] | None, out: Any) -> None:
    """Untimed configured-state control; transport is shared with W1."""
    try:
        from benchmark_v1.scripts.run_maestro_common_panel import (
            _canonical_native_imports, collect_worker_context, validate_worker_context,
        )
        from benchmark_v1.qre_benchmark.maestro_component_v2 import resolve_v2_config

        modules = _canonical_native_imports()
        maestro = modules["maestro"]
        config_record = resolve_v2_config(
            maestro, candidate, chi, int(seed), float(singular_value_threshold)
        )
        context = collect_worker_context(
            config=config_record["resolved"],
            backend_threading_policy=backend_threading_policy,
        )
        if expected_worker_context is not None:
            validate_worker_context(expected_worker_context, context)
        circuit = maestro.QasmToCirc().parse_and_translate(qasm)
        amplitudes = maestro.get_statevector(circuit, config_record["object"])
        out.put({"ok": True, "amplitudes": list(amplitudes), "worker_context": context,
                 "parsed_width": int(circuit.num_qubits),
                 "max_bond_dim_reached": None,
                 "telemetry_status": "unknown_not_returned_by_statevector_api"})
    except BaseException as exc:
        out.put({"ok": False, "worker_exception": True,
                 "error": f"{type(exc).__name__}:{exc}",
                 "worker_context": locals().get("context")})


def _v3_reference_and_candidate_rank_preflight(
    contract: dict[str, Any], configurations: dict[str, dict[str, Any]],
    expected_contexts: dict[str, dict[str, Any]] | None,
    partial_quarantine_policy: dict[str, Any] | None = None,
    partial_quarantine_evidence: dict[tuple[str, int, int | None], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    from benchmark_v1.scripts.run_maestro_common_panel import (
        _v2_qiskit_statevector, isolated_call, validate_worker_context,
        validate_worker_environment_consistency,
    )
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        V3Unavailable, assess_v3_candidate_preparation, canonical_hash,
        central_schmidt_rank_v3, generate_v3_probe, validate_v3_rank_preflight_records,
        validate_partial_v3_rank_preflight_records,
    )

    corpus = contract["synthetic_corpus"]
    rank_policy = corpus["background"]
    rank_tol = float(rank_policy["relative_singular_value_rank_tolerance"])
    norm_tol = float(contract["quality_policy"]["vector_norm_atol"])
    timeout = float(contract["measurement"]["cell_timeout_seconds"])
    policy = _v3_backend_threading_policy(contract)
    context_registry: dict[str, dict[str, Any]] = {}
    context_keys: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    quarantine_selector = ((partial_quarantine_policy or {}).get("selector")
                           if partial_quarantine_policy else None)
    quarantined_configs = set()
    if partial_quarantine_policy is not None:
        if not isinstance(quarantine_selector, dict):
            raise V3Unavailable("partial_rank_preflight_selector_missing")
        quarantined_configs = {
            (str(quarantine_selector["candidate"]), int(quarantine_selector["width"]), int(chi))
            for chi in quarantine_selector.get("chis", [])
        }
        if not isinstance(partial_quarantine_evidence, dict):
            raise V3Unavailable("partial_rank_preflight_evidence_missing")
    for width_value in corpus["statevector_widths"]:
        width = int(width_value)
        reference_probe = generate_v3_probe(
            contract, width, "entangled_sampling_diagnostic", 0, include_measurement=False
        )
        reference_amplitudes, reference_wall = _v2_qiskit_statevector(reference_probe["program"])
        reference_rank = _v3_checked_preparation_rank(
            reference_amplitudes, width, "qiskit_reference", None,
            reference_probe["program"], rank_tol, norm_tol,
        )
        expected_rank = int(rank_policy["reference_central_schmidt_rank_at_tolerance"][str(width)])
        if int(reference_rank["logical_central_schmidt_rank"]) != expected_rank:
            raise V3Unavailable(f"v3_reference_rank_mismatch:n={width}")
        if partial_quarantine_policy is not None:
            reference_vector = [
                [float(complex(value).real), float(complex(value).imag)]
                for value in reference_amplitudes
            ]
            records.append({
                "candidate": "qiskit_reference", "width": width, "chi": None,
                "reference_logical_central_schmidt_rank": expected_rank,
                "candidate_logical_central_schmidt_rank": int(
                    reference_rank["logical_central_schmidt_rank"]
                ),
                "candidate_norm": float(reference_rank["norm"]),
                "candidate_amplitudes_sha256": canonical_hash(reference_vector),
                "qasm_sha256": reference_probe["whole_qasm_sha256"],
                "reference_wall_seconds": reference_wall,
                "preflight_status": "ok", "preflight_reasons": [],
                "internal_bond_dimension_claim": False,
            })
        for candidate in contract["implementation"]["candidates"]:
            chis = ([None] if candidate == "statevector" else
                    [int(value) for value in corpus["mps_configured_bonds"]])
            for chi in chis:
                config_key = f"{candidate}:{chi}"
                config_tuple = (str(candidate), int(width), None if chi is None else int(chi))
                if config_tuple in quarantined_configs:
                    evidence = partial_quarantine_evidence.get(config_tuple)  # type: ignore[union-attr]
                    if not isinstance(evidence, dict):
                        raise V3Unavailable(f"partial_quarantine_evidence_missing:{config_tuple}")
                    if config_key not in context_registry:
                        raise V3Unavailable(f"partial_quarantine_worker_context_unavailable:{config_key}")
                    zero_program = zero_state_width_control_qasm(contract, int(width))
                    zero_result = isolated_call(_v3_preparation_worker, (
                        zero_program, str(candidate), chi, int(contract["seed"]),
                        float(contract["panel_candidate_configuration"]["mps_singular_value_threshold"]),
                        policy, context_registry[config_key],
                    ), timeout)
                    zero_state_control = assess_zero_state_width_control(
                        zero_result, zero_program, int(width), str(candidate), int(chi), norm_tol)
                    try:
                        validate_worker_context(context_registry[config_key], zero_result["worker_context"])
                    except Exception as cause:
                        failure = V3Unavailable(f"partial_zero_state_worker_context_mismatch:{config_key}")
                        failure.failed_control = {**zero_state_control, "status": "unavailable", "error": str(cause)}
                        raise failure from cause
                    records.append({
                        "candidate": str(candidate), "width": int(width), "chi": chi,
                        "preflight_status": "unavailable",
                        "preflight_reasons": ["synthetic_preparation_norm_failure"],
                        "quarantine_evidence": evidence,
                        "zero_state_sampling_control": zero_state_control,
                        "logical_rank_status": "not_computed_from_unnormalized_amplitudes",
                        "internal_bond_dimension_claim": False,
                    })
                    continue
                expected_context = None if expected_contexts is None else expected_contexts[config_key]
                probe = generate_v3_probe(
                    contract, width, "entangled_sampling_diagnostic", 0, include_measurement=False
                )
                result = isolated_call(_v3_preparation_worker, (
                    probe["program"], str(candidate), chi, int(contract["seed"]),
                    float(contract["panel_candidate_configuration"]["mps_singular_value_threshold"]),
                    policy, expected_context,
                ), timeout)
                if not result.get("ok") or result.get("transport_status") != "ok":
                    raise V3Unavailable(
                        f"v3_candidate_rank_preflight_transport_failure:{config_key}:n={width}:"
                        f"{result.get('transport_status', 'missing')}:{result.get('error', '')}"
                    )
                actual_context = result.get("worker_context")
                if not isinstance(actual_context, dict):
                    raise V3Unavailable(f"v3_candidate_rank_preflight_context_missing:{config_key}:n={width}")
                if expected_context is not None:
                    validate_worker_context(expected_context, actual_context)
                elif config_key in context_registry:
                    validate_worker_context(context_registry[config_key], actual_context)
                validate_worker_environment_consistency(context_registry, actual_context)
                context_registry[config_key] = actual_context
                context_keys[config_key] = actual_context
                amplitudes = result.get("amplitudes")
                if candidate == "statevector":
                    candidate_rank = _v3_checked_preparation_rank(
                        amplitudes, width, str(candidate), chi, probe["program"], rank_tol, norm_tol,
                    )
                    telemetry, telemetry_status = None, "not_applicable"
                else:
                    assessment = _v3_checked_preparation_rank(
                        amplitudes, width, str(candidate), chi, probe["program"],
                        rank_tol, norm_tol, result.get("max_bond_dim_reached"),
                    )
                    candidate_rank = assessment
                    telemetry = assessment["native_max_bond_dim_reached"]
                    telemetry_status = assessment["telemetry_status"]
                    if assessment["preparation_status"] != "ok":
                        raise V3Unavailable(
                            f"v3_candidate_preparation_unavailable:{config_key}:n={width}:"
                            f"{assessment['preparation_reason']}"
                        )
                candidate_vector = [[float(complex(value).real), float(complex(value).imag)]
                                    for value in amplitudes]
                records.append({
                    "candidate": str(candidate), "width": width, "chi": chi,
                    "reference_logical_central_schmidt_rank": expected_rank,
                    "candidate_logical_central_schmidt_rank": int(
                        candidate_rank["logical_central_schmidt_rank"]
                    ),
                    "candidate_norm": float(candidate_rank["norm"]),
                    "candidate_amplitudes_sha256": canonical_hash(candidate_vector),
                    "max_bond_dim_reached": telemetry,
                    "telemetry_status": telemetry_status,
                    "reference_wall_seconds": reference_wall,
                    "worker_context_sha256": actual_context["identity_sha256"],
                    "worker_environment_sha256": actual_context["worker_environment_sha256"],
                })
    if partial_quarantine_policy is not None:
        validated_partial = validate_partial_v3_rank_preflight_records(
            contract, records, partial_quarantine_policy, partial_quarantine_evidence or {},
        )
        if validated_partial["status"] != "pass":
            raise V3Unavailable(
                f"partial_rank_preflight_failed:{validated_partial.get('unavailable_reasons')}"
            )
        return validated_partial["records"], context_keys
    validated = validate_v3_rank_preflight_records(contract, records)
    if validated["status"] != "pass":
        raise V3Unavailable(f"v3_rank_preflight_failed:{validated['unavailable_reasons']}")
    return records, context_keys


def run_v2_calibration(manifest_path: Path, manifest: dict[str, Any],
                       artifact_dir: Path, cells: list[dict[str, Any]],
                       acceptance_path: Path, resume: bool,
                       max_cells: int | None) -> None:
    """Execute v2 only after a coordinator acceptance file pins this code.

    This function is intentionally not reached by the v2 dry-run path. Each
    untimed/timed call is a new child process and is represented as such.
    """
    from benchmark_v1.qre_benchmark.maestro_component_v2 import (
        V2_CLOCK_ID, V2_TIMED_STAGE, V2_UNTIMED_STAGE,
        fit_v2_calibration_rows, resolve_v2_config,
    )

    try:
        import maestro
    except Exception as exc:
        raise SystemExit(f"Maestro API unavailable: {type(exc).__name__}: {exc}") from exc
    pins = _v2_environment_pins(manifest_path, maestro)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "pass":
        raise SystemExit("v2 execution blocked: implementation acceptance status is not pass")
    if acceptance.get("manifest_id") != manifest["manifest_id"]:
        raise SystemExit("v2 execution blocked: acceptance manifest id mismatch")
    if acceptance.get("manifest_sha256") != pins["manifest_sha256"]:
        raise SystemExit("v2 execution blocked: acceptance manifest hash mismatch")
    if acceptance.get("code_hashes") != pins["code_hashes"]:
        raise SystemExit("v2 execution blocked: acceptance code hashes differ from current runner/helpers")
    if max_cells is not None and max_cells < 1:
        raise SystemExit("--max-cells must be positive")

    measure = manifest["measurement"]
    configs: dict[tuple[str, int | None], dict[str, Any]] = {}
    for candidate in manifest["implementation"]["candidates"]:
        bonds = [None] if candidate == "statevector" else [int(v) for v in manifest["synthetic_corpus"]["mps_configured_bonds"]]
        for chi in bonds:
            configs[(candidate, chi)] = resolve_v2_config(
                maestro, candidate, chi, int(manifest["seed"]),
                float(manifest["panel_candidate_configuration"]["mps_singular_value_threshold"]),
            )
    run_identity = {"manifest_id": manifest["manifest_id"], **pins,
                    "selected_clock_id": V2_CLOCK_ID,
                    "selected_clock_field": manifest["measurement"]["selected_clock_field"],
                    "selected_calibration_stage": manifest["measurement"]["selected_calibration_stage"],
                    "configs": {f"{candidate}:{chi}": {"sha256": value["sha256"],
                                                           "resolved": value["resolved"]}
                                for (candidate, chi), value in sorted(configs.items(), key=lambda item: str(item[0]))},
                    "thread_policy": measure["thread_policy"]}
    run_identity_sha256 = _canonical_hash(run_identity)
    expected_cell_ids: set[str] = set()
    for planned_cell in cells:
        repeats = int(planned_cell["operation_repeats"])
        program = qasm(int(planned_cell["width"]), str(planned_cell["operation_class"]), repeats)
        program_hash = hash_text(program)
        config = configs[(str(planned_cell["candidate"]),
                          None if planned_cell["chi"] is None else int(planned_cell["chi"]))]
        expected_cell_ids.add(_v2_cell_id(run_identity_sha256, planned_cell,
                                          program_hash, config["sha256"]))

    if artifact_dir.exists():
        if not resume:
            raise SystemExit(f"refusing existing v2 artifact directory without --resume: {artifact_dir}")
        run_path = artifact_dir / "run_manifest.json"
        if not run_path.is_file():
            raise SystemExit("resume refused: run_manifest.json is missing")
        existing = json.loads(run_path.read_text(encoding="utf-8"))
        if existing.get("run_identity_sha256") != run_identity_sha256:
            raise SystemExit("resume refused: run identity/code/config/package pins changed")
    elif resume:
        raise SystemExit("resume requested but v2 artifact directory does not exist")

    lock_path = ROOT / "work/locks/maestro_qcsim_v2_cpu_timing.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(lock_fd)
        raise SystemExit(f"v2 execution blocked: exclusive CPU timing lock is held at {lock_path}") from exc
    os.ftruncate(lock_fd, 0)
    os.write(lock_fd, f"pid={os.getpid()}\nidentity={run_identity_sha256}\n".encode())
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = artifact_dir / "checkpoint.jsonl"
        completed_cells: dict[str, dict[str, Any]] = {}
        all_rows: list[dict[str, Any]] = []
        seen_attempt_ids: set[str] = set()
        if checkpoint_path.is_file():
            with checkpoint_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.endswith("\n"):
                        raise SystemExit(f"resume refused: incomplete checkpoint ledger line {line_number}")
                    saved = json.loads(line)
                    cell_id = str(saved.get("cell_id", ""))
                    if not cell_id or saved.get("run_identity_sha256") != run_identity_sha256:
                        raise SystemExit(f"resume refused: checkpoint pin mismatch at line {line_number}")
                    if cell_id not in expected_cell_ids:
                        raise SystemExit(f"resume refused: checkpoint cell is outside the frozen plan at line {line_number}")
                    if cell_id in completed_cells:
                        raise SystemExit(f"resume refused: duplicate checkpoint cell id at line {line_number}")
                    rows_for_cell = saved.get("rows", [])
                    for row in rows_for_cell:
                        if (row.get("cell_id") != cell_id
                                or row.get("run_identity_sha256") != run_identity_sha256):
                            raise SystemExit(f"resume refused: row identity mismatch at line {line_number}")
                        attempt_id = str(row.get("attempt_id", ""))
                        if not attempt_id or attempt_id in seen_attempt_ids:
                            raise SystemExit(f"resume refused: duplicate/missing attempt id at line {line_number}")
                        seen_attempt_ids.add(attempt_id)
                    completed_cells[cell_id] = saved
                    all_rows.extend(rows_for_cell)
        _atomic_json(artifact_dir / "run_manifest.json", {
            "manifest_id": manifest["manifest_id"], "status": "running",
            "run_identity": run_identity, "run_identity_sha256": run_identity_sha256,
            "planned_cells": len(cells),
            "planned_timed_observations": int(manifest["synthetic_corpus"]["planned_timed_observations"]),
            "process_isolation": "one new spawn child per API call; no persistent-process warm claim",
            "clock_id": V2_CLOCK_ID,
            "completed_cells": len(completed_cells),
            "checkpoint_path": str(checkpoint_path),
        })
        for index, cell in enumerate(cells, 1):
            if max_cells is not None and index > max_cells:
                break
            repeats = int(cell["operation_repeats"])
            program = qasm(int(cell["width"]), str(cell["operation_class"]), repeats)
            qasm_sha256 = hash_text(program)
            config_record = configs[(str(cell["candidate"]),
                                     None if cell["chi"] is None else int(cell["chi"]))]
            cell_id = _v2_cell_id(run_identity_sha256, cell, qasm_sha256, config_record["sha256"])
            if cell_id in completed_cells:
                continue
            cell_rows: list[dict[str, Any]] = []
            common = {**cell, "operation_repeats": repeats,
                      **operation_counts(str(cell["operation_class"]), repeats),
                      "qasm_sha256": qasm_sha256, "config_sha256": config_record["sha256"],
                      "run_identity_sha256": run_identity_sha256, "cell_id": cell_id,
                      "clock_id": V2_CLOCK_ID, "selected_clock_id": V2_CLOCK_ID,
                      "selected_clock_field": manifest["measurement"]["selected_clock_field"],
                      "selected_calibration_stage": manifest["measurement"]["selected_calibration_stage"]}
            print(f"[maestro-calibration-v2] {index}/{len(cells)} {cell['candidate']} n={cell['width']} "
                  f"chi={cell['chi']} op={cell['operation_class']} shots={cell['shots']}", flush=True)
            for session in range(1, int(measure["sessions"]) + 1):
                for stage, count in ((V2_UNTIMED_STAGE, int(measure["untimed_warmups_per_session"])),
                                     (V2_TIMED_STAGE, int(measure["timed_warm_repetitions_per_session"]))):
                    for repetition in range(count):
                        result = _v2_isolated_call(
                            program, str(cell["candidate"]), int(cell["shots"]),
                            None if cell["chi"] is None else int(cell["chi"]),
                            int(manifest["seed"]),
                            float(manifest["panel_candidate_configuration"]["mps_singular_value_threshold"]),
                            float(measure["cell_timeout_seconds"]),
                        )
                        status = "ok" if result.get("ok") else ("timeout" if result.get("timeout") else "error")
                        cell_rows.append({
                            **common,
                            "attempt_id": _canonical_hash({"cell_id": cell_id, "session": session,
                                                           "stage": stage, "repetition": repetition}),
                            "stage": stage, "session": session, "repetition": repetition,
                            "status": status,
                            "reported_time_seconds": result.get("reported_time_seconds", ""),
                            "host_wall_seconds": result.get("host_wall_seconds", ""),
                            "outer_wall_seconds": result.get("outer_wall_seconds", ""),
                            "call_process_isolated": True,
                            "error": result.get("error", ""),
                        })
            checkpoint_record = {"cell_id": cell_id,
                                 "run_identity_sha256": run_identity_sha256,
                                 "rows": cell_rows}
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(checkpoint_record, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            completed_cells[cell_id] = checkpoint_record
            all_rows.extend(cell_rows)
        raw_path = artifact_dir / "raw_records.csv"
        write_csv(raw_path, all_rows)
        checkpoint_attempt_ids_sha256 = _canonical_hash(
            sorted(str(row["attempt_id"]) for row in all_rows)
        )
        fit = fit_v2_calibration_rows(all_rows, manifest, _sha256_file(raw_path))
        fit_payload = {
            "fit_version": fit.fit_version,
            "calibration_raw_sha256": fit.calibration_raw_sha256,
            "calibration_manifest_id": fit.calibration_manifest_id,
            "clock_id": fit.clock_id,
            "calibration_warm_observations": fit.calibration_warm_observations,
            "calibration_timed_observations": fit.calibration_timed_observations,
            "valid_operation_knots": len(fit.operation_knots),
            "valid_sampling_intercept_knots": len(fit.sampling_intercept_knots),
            "valid_sampling_per_shot_knots": len(fit.sampling_per_shot_knots),
            "invalid_knots": [{"candidate": key[0], "component": key[1], "width": key[2],
                               "chi": key[3], "reason": value}
                              for key, value in sorted(fit.invalid_knot_states.items(), key=lambda item: str(item[0]))],
        }
        _atomic_json(artifact_dir / "fitted_calibration.json", {
            **fit_payload,
            "interpolation_width_grid": {candidate: list(widths)
                                          for candidate, widths in fit.interpolation_width_grid.items()},
            "interpolation_chi_grid": list(fit.interpolation_chi_grid),
            "operation_knots": [
                {"candidate": key[0], "component": key[1], "width": key[2],
                 "chi": key[3], "coefficient": value}
                for key, value in sorted(fit.operation_knots.items(), key=lambda item: str(item[0]))
            ],
            "sampling_intercept_knots": [
                {"candidate": key[0], "width": key[1], "chi": key[2], "coefficient": value}
                for key, value in sorted(fit.sampling_intercept_knots.items(), key=lambda item: str(item[0]))
            ],
            "sampling_per_shot_knots": [
                {"candidate": key[0], "width": key[1], "chi": key[2], "coefficient": value}
                for key, value in sorted(fit.sampling_per_shot_knots.items(), key=lambda item: str(item[0]))
            ],
        })
        _atomic_json(artifact_dir / "fitted_calibration_summary.json", fit_payload)
        is_complete = len(completed_cells) == len(cells)
        run_payload = {
            "manifest_id": manifest["manifest_id"],
            "status": ("complete" if is_complete and not fit.invalid_knot_states else
                       "complete_with_unavailable_knots" if is_complete else "checkpointed_partial"),
            "run_identity": run_identity, "run_identity_sha256": run_identity_sha256,
            "planned_cells": len(cells), "completed_cells": len(completed_cells),
            "max_cells_limit": max_cells,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_format": "maestro_calibration_checkpoint_jsonl_v1",
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "checkpoint_attempt_rows": len(all_rows),
            "checkpoint_attempt_ids_sha256": checkpoint_attempt_ids_sha256,
            "raw_records": {"path": str(raw_path), "sha256": _sha256_file(raw_path), "rows": len(all_rows),
                            "attempt_ids_sha256": checkpoint_attempt_ids_sha256},
            "fit_summary": fit_payload,
            "clock_id": V2_CLOCK_ID,
            "selected_clock_field": manifest["measurement"]["selected_clock_field"],
            "selected_calibration_stage": manifest["measurement"]["selected_calibration_stage"],
            "timed_stage": V2_TIMED_STAGE, "untimed_stage": V2_UNTIMED_STAGE,
            "measurement_started": True,
        }
        _atomic_json(artifact_dir / "run_manifest.json", run_payload)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmark_v1/execution/manifests/maestro_componentwise_calibration_pilot_v1.json")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true", help="execute only with matching protocol-specific acceptance JSON")
    parser.add_argument("--acceptance", type=Path, help="coordinator-signed calibration acceptance JSON")
    parser.add_argument("--resume", action="store_true", help="resume only when the full run identity matches")
    parser.add_argument("--max-cells", type=int, help="stop after this many planned cells; limit does not change run identity")
    args = parser.parse_args()
    manifest = load(args.manifest)
    if manifest.get("manifest_id") in {"maestro-common-panel-completion-v3-partial",
                                       "maestro-common-panel-completion-v3-partial-width-control",
                                       "maestro-common-panel-completion-v3-partial-zero-measurement"}:
        raise SystemExit("Superseded Maestro bundle is terminal for execution; use the signed resume amendment")
    if manifest.get("manifest_id") == "maestro-common-panel-completion-v3-partial-resume":
        from benchmark_v1.qre_benchmark.maestro_component_v2 import (
            PARTIAL_MANIFEST_ID, V3Unavailable,
        )

        if PARTIAL_MANIFEST_ID != manifest.get("manifest_id"):
            raise SystemExit("partial calibration manifest identity mismatch")
        try:
            resolution, cells, schedule, static = _partial_static_bundle(args.manifest)
        except V3Unavailable as exc:
            raise SystemExit(str(exc)) from exc
        if args.dry_run:
            summary = {
                "manifest_id": PARTIAL_MANIFEST_ID,
                "manifest_sha256": static["manifest_sha256"],
                "amendment_identity": static["amendment_identity"],
                "parent_manifest_sha256": static["parent_manifest_sha256"],
                "parent_contract_sha256": static["parent_contract_sha256"],
                "resolved_contract_sha256": static["resolved_contract_sha256"],
                "plan_sha256": static["plan_sha256"],
                "schedule_sha256": static["schedule_sha256"],
                "planned_cells": 432, "eligible_cells": 399,
                "quarantined_cells": 33,
                "quarantined_operation_cells": 27,
                "quarantined_diagnostic_cells": 6,
                "assigned_schedule_slots": len(schedule),
                "quarantined_schedule_slots": static["planned_quarantined_slots"],
                "planned_timed_calls": static["planned_timed_calls"],
                "planned_untimed_calls": static["planned_untimed_calls"],
                "first_ten_cell_ids": static["first_ten_cell_ids"],
                "pre_expansion_cell_ids": static["pre_expansion_cell_ids"],
                "quarantine_evidence_by_config": {
                    f"{key[0]}:{key[1]}:{key[2]}": value
                    for key, value in static["quarantine_evidence"].items()
                },
                "cells": cells,
                "schedule": schedule,
                "measurement_started": False, "timing_run": False,
                "score_created": False, "dry_run_writes": False,
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            return
        if args.execute:
            try:
                result = run_partial_calibration(
                    args.manifest, args.artifact_dir, args.acceptance,
                    args.resume, args.max_cells,
                )
            except V3Unavailable as exc:
                raise SystemExit(str(exc)) from exc
            if result is not None:
                print(json.dumps(result, indent=2, sort_keys=True))
            return
        if args.acceptance is not None or args.resume or args.max_cells is not None:
            raise SystemExit("partial --acceptance/--resume/--max-cells require --execute")
        raise SystemExit("partial calibration requires --dry-run or accepted --execute")
    if manifest.get("manifest_id") == "maestro-common-panel-completion-v3":
        from benchmark_v1.qre_benchmark.maestro_component_v2 import (
            V3_MANIFEST_ID, V3_PANEL_ATTEMPT_STATUS_SCHEMA,
            V3Unavailable, build_v3_execution_schedule, build_v3_plan, resolve_v3_contract,
            v3_pilot_cell_ids,
        )

        if V3_MANIFEST_ID != manifest.get("manifest_id"):
            raise SystemExit("v3 calibration manifest identity mismatch")
        contract = resolve_v3_contract(args.manifest)
        cells = build_v3_plan(contract)
        pilot = v3_pilot_cell_ids(cells)
        pilot_schedule = build_v3_execution_schedule(
            contract, cells, pilot["first_ten_cell_ids"]
        )
        if args.dry_run:
            measurement = contract["measurement"]
            timed = (len(cells) * int(measurement["sessions"])
                     * int(measurement["timed_repetitions_per_cell_per_session"]))
            untimed = (len(cells) * int(measurement["sessions"])
                       * int(measurement["untimed_calls_per_cell_per_session"]))
            summary = {
                "manifest_id": V3_MANIFEST_ID,
                "resolved_contract_sha256": hashlib.sha256(json.dumps(
                    contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")).hexdigest(),
                "planned_cells": len(cells),
                "planned_primary_cells": int(contract["synthetic_corpus"]["planned_primary_cells"]),
                "planned_diagnostic_cells": int(
                    contract["synthetic_corpus"]["planned_cells_including_diagnostics"]
                    - contract["synthetic_corpus"]["planned_primary_cells"]
                ),
                "planned_timed_observations": timed,
                "planned_untimed_calls": untimed,
                "planned_api_calls": timed + untimed,
                "first_ten_cell_ids": pilot["first_ten_cell_ids"],
                "pre_expansion_cell_ids": pilot["pre_expansion_cell_ids"],
                "first_ten_process_isolated_schedule": pilot_schedule,
                "panel_attempt_status_schema": V3_PANEL_ATTEMPT_STATUS_SCHEMA,
                "measurement_started": False,
                "timing_run": False,
                "score_created": False,
                "dry_run_writes": False,
                "cells": cells,
            }
            print(json.dumps(summary, indent=2))
            return
        if args.execute:
            try:
                result = run_v3_calibration(args.manifest, args.artifact_dir, args.acceptance,
                                           args.resume, args.max_cells)
            except V3Unavailable as exc:
                raise SystemExit(str(exc)) from exc
            if result is not None:
                print(json.dumps(result, indent=2, sort_keys=True))
            return
        if args.acceptance is not None or args.resume or args.max_cells is not None:
            raise SystemExit("v3 --acceptance/--resume/--max-cells require --execute")
        raise SystemExit("v3 calibration requires --dry-run or explicit accepted --execute")
    cells = planned(manifest)
    if args.dry_run:
        summary = {"manifest_id": manifest["manifest_id"], "planned_cells": len(cells), "cells": cells,
                   "planned_timed_observations": manifest.get("synthetic_corpus", {}).get("planned_timed_observations"),
                   "measurement_started": False, "timing_run": False, "score_created": False}
        print(json.dumps(summary, indent=2)); return
    if manifest.get("manifest_id") == "maestro-common-panel-completion-v2":
        if not args.execute:
            raise SystemExit("v2 calibration is code-only by default; use --dry-run or wait for explicit acceptance and --execute")
        if args.acceptance is None:
            raise SystemExit("v2 calibration execution requires --acceptance")
        run_v2_calibration(args.manifest, manifest, args.artifact_dir, cells,
                           args.acceptance, args.resume, args.max_cells)
        return
    if args.execute or args.acceptance is not None or args.resume or args.max_cells is not None:
        raise SystemExit("--execute/--acceptance/--resume/--max-cells are reserved for the v2 calibration path")
    pilot = import_pilot(); measure = manifest["measurement"]; corpus = manifest["synthetic_corpus"]
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.artifact_dir / "environment.json").write_text(json.dumps({"python":sys.version,"platform":platform.platform(),"maestro_pilot_path":str(PILOT),"maestro_pilot_sha256":hashlib.sha256(PILOT.read_bytes()).hexdigest(),"thread_policy":measure["thread_policy"]}, indent=2, sort_keys=True)+"\n")
    rows = []
    for index, cell in enumerate(cells, 1):
        repeats_value = cell.get("operation_repeats")
        if repeats_value is None:
            repeats_value = corpus.get("repetitions_per_operation_cell", 0) if cell["kind"] == "operation" else 0
        repeats = int(repeats_value)
        program = qasm(cell["width"], cell["operation_class"], repeats)
        print(f"[maestro-calibration] {index}/{len(cells)} {cell['candidate']} n={cell['width']} chi={cell['chi']} op={cell['operation_class']} shots={cell['shots']}", flush=True)
        common = {**cell,"qasm_sha256":hash_text(program),"operation_repeats":repeats,**operation_counts(cell["operation_class"], repeats)}
        for session in range(1, measure["sessions"] + 1):
            result = run_cell(pilot, program, cell["candidate"], cell["shots"], cell["chi"], measure["untimed_warmups_per_session"], measure["timed_warm_repetitions_per_session"], measure["cell_timeout_seconds"])
            first = result["first"]
            rows.append({**common,"stage":"first_execute","session":session,"repetition":0,"status":"ok" if first.get("ok") else ("timeout" if first.get("timeout") else "error"),"reported_time_seconds":(first.get("result") or {}).get("time_taken"),"wall_seconds":first.get("wall_seconds"),"error":first.get("error")})
            rows += [{**common,"stage":"warm_execute","session":session,**warm} for warm in result["warm"]]
    write_csv(args.artifact_dir / "raw_records.csv", rows)
    print(args.artifact_dir / "raw_records.csv")


if __name__ == "__main__":
    main()
