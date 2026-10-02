#!/usr/bin/env python3
"""Paired S84 Maestro thread-policy diagnostic.

Fresh runs reserve a new output directory and fingerprint both arms in fresh
workers before any paired calls. A narrowly scoped append-only continuation
can resume only the explicitly pinned `attempt_003` host-blocked prefix.
Importing this module and running its tests never imports Maestro or calls its
execution API.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import importlib
import io
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL = ROOT / "benchmark_v1/decisions/S84_MAESTRO_THREADPOOL_DIAGNOSTIC_20261002.md"
PROTOCOL_SHA256 = "fe57c0ddac9ef609a4e88adcae661dafc7291d5d5e12ade73f507ecb31ef885e"
BASE_CONTRACT = ROOT / "benchmark_v1/execution/manifests/maestro_common_panel_completion_v3.json"
PILOT_RUN = ROOT / "artifacts/benchmark_v3/simulator/maestro_candidate_runtime_resume/calibration/run_manifest.json"
PILOT_RAW = ROOT / "artifacts/benchmark_v3/simulator/maestro_candidate_runtime_resume/calibration/raw_records.csv"
PILOT_ACCEPTANCE = ROOT / "artifacts/benchmark_v3/simulator/maestro_candidate_runtime_resume/calibration/acceptance.json"
PILOT_SUMMARY = ROOT / "artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/simulator/maestro_pilot_summary.csv"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts/benchmark_v3/simulator/maestro_threadpool_diagnostic_20261002"
RESUME_ARTIFACT_RELATIVE = Path(
    "artifacts/benchmark_v3/simulator/maestro_threadpool_diagnostic_20261002/attempt_003"
)
RESUME_ORIGINAL_PROTOCOL_SHA256 = "35b2e7a36148c80ee13b1095f9f2e474a2c0bfb925745974e9d75daf3801e0ae"
RESUME_ORIGINAL_RUNNER_SHA256 = "e0b52ef864c3621707432790c90e4a94f678d3615294b6527dce6999a2f0cf02"
RESUME_ORIGINAL_MANIFEST_SHA256 = "d24ebefc97301b8b217a404d28c85085fa8e7820928b40f081d1569f85befb3a"
RESUME_ORIGINAL_ATTEMPTS_SHA256 = "de1db999379ee6d36d0f6482c9807d81c0830e7af5caa3b9e3c62800f7141d0a"
RESUME_ORIGINAL_BUDGET_STARTED_UNIX = 1790920889.1173275
RESUME_ORIGINAL_ATTEMPT_COUNT = 3
SHARED_TIMING_LOCK = ROOT / "work/locks/maestro_qcsim_v2_cpu_timing.lock"

PINNED_INPUTS = {
    "benchmark_v1/decisions/S83_FAMILY_AWARE_AND_MAESTRO_ADJUDICATION_20261002.md":
        "58e2af1c879ad59dfda3c9ead99c6f14deb165eb0eb4f1afc77a1d25b553597d",
    "benchmark_v1/execution/manifests/maestro_common_panel_completion_v3.json":
        "bc2a3bc150421698ff09e08a2055a40979f96d2965fe2dcb3e20402c1016574d",
    "artifacts/benchmark_v3/simulator/maestro_candidate_runtime_resume/calibration/run_manifest.json":
        "8989735f4291621b2ee18c04ae7c326dd029183661bb510d1f0acc74147d92c2",
    "artifacts/benchmark_v3/simulator/maestro_candidate_runtime_resume/calibration/raw_records.csv":
        "9122f3b28047e52f33654eb7f83296255ff13b1f2ab285d44d1970cb27610709",
    "artifacts/benchmark_v3/simulator/maestro_candidate_runtime_resume/calibration/acceptance.json":
        "90bbc6b6f819d316e7fd4b40c77668856337de31d05a7d97eb4c4d8371019b44",
    "artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/simulator/maestro_pilot_summary.csv":
        "d6107c87721695537cee9c67f3e7c51ab916f620a665b532487d1374e7d02ad5",
}
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)
ARM_DEFAULT = "default"
ARM_LIMITED = "single_thread"
ARMS = (ARM_DEFAULT, ARM_LIMITED)
SCHEDULE_SEED = 20261002
SESSIONS = 3
REPETITIONS = 5
CALL_TIMEOUT_SECONDS = 180.0
TOTAL_BUDGET_SECONDS = 3600.0
CLEANUP_RESERVE_SECONDS = 15.0
EXPECTED_AFFINITY = list(range(28))
PINNED_DEFAULT_POLICY = (
    "retain pinned library defaults; verify worker threadpool configuration including "
    "28-thread OpenMP/BLAS defaults; record Maestro backend/shot-dependent threading; no cap tuning"
)
EXPECTED_OPERATIONS = {
    "one_qubit_noncommuting", "two_qubit_wrapper_control", "two_qubit_cx_interleaved",
}
EXPECTED_REPEATS = {256, 512, 1024}
APPROVED_GPU_BASELINE = {
    "device_uuid": "GPU-7c51e3b6-b160-fc40-c14f-292641e5c95e",
    "device_name": "NVIDIA GeForce RTX 5070 Ti",
    "utilization_baseline_percent": 2,
    "max_utilization_percent": 10,
    "device_memory_baseline_mib": 797,
    "max_device_memory_mib": 925,
    "max_process_memory_delta_mib": 64,
    "total_process_fb_memory_baseline_mib": 746,
    "max_total_process_fb_memory_mib": 874,
    "processes": [
        {
            "pid": 42623, "uid": 1000, "start_time_ticks": 37701,
            "exe_path": "/usr/bin/gnome-shell",
            "exe_sha256": "cf699357b6b6bbbacda5250261200ab92f6cccd491b126c6e30142c9fdafd619",
            "pmon_type": "G", "pmon_command": "gnome-shell", "baseline_fb_memory_mib": 178,
        },
        {
            "pid": 43388, "uid": 1000, "start_time_ticks": 37797,
            "exe_path": "/usr/bin/Xwayland",
            "exe_sha256": "700d3cba7f946c3ff3100ee36f15012c1c027bb67f29e84def1a1b67bb8f3587",
            "pmon_type": "G", "pmon_command": "Xwayland", "baseline_fb_memory_mib": 70,
        },
        {
            "pid": 181789, "uid": 1000, "start_time_ticks": 190789,
            "exe_path": "/usr/bin/ptyxis",
            "exe_sha256": "d896539ba35478c4fc307be5f0fc2f3b2c6997114b90189810233e450d11d1f3",
            "pmon_type": "C+G", "pmon_command": "ptyxis", "baseline_fb_memory_mib": 41,
        },
        {
            "pid": 1169761, "uid": 1000, "start_time_ticks": 6291646,
            "exe_path": "/snap/firefox/8969/usr/lib/firefox/firefox",
            "exe_sha256": "5e800815dacd4c2a80e17f69dcd46e4cba2c82871ce0867efc8e4a8eabcd1efe",
            "pmon_type": "G", "pmon_command": "firefox", "baseline_fb_memory_mib": 134,
        },
        {
            "pid": 2925123, "uid": 1000, "start_time_ticks": 3406758,
            "exe_path": "/usr/share/rustdesk/rustdesk",
            "exe_sha256": "58ef1e984727d827836c8ad84ad50a4971db4a3cb80ad7a646982594af155c52",
            "pmon_type": "C+G", "pmon_command": "rustdesk", "baseline_fb_memory_mib": 284,
        },
        {
            "pid": 3197852, "uid": 1000, "start_time_ticks": 3748766,
            "exe_path": "/usr/share/rustdesk/rustdesk",
            "exe_sha256": "58ef1e984727d827836c8ad84ad50a4971db4a3cb80ad7a646982594af155c52",
            "pmon_type": "G", "pmon_command": "rustdesk", "baseline_fb_memory_mib": 39,
        },
    ],
}
ATTEMPT_FIELDS = (
    "attempt_index", "pair_id", "cell_id", "arm", "session", "repetition",
    "candidate", "width", "operation_class", "operation_repeats", "shots", "seed",
    "whole_qasm_sha256",
    "host_activity_json",
    "api_called", "status", "reported_time_seconds", "host_wall_seconds",
    "outer_wall_seconds", "worker_context_sha256", "error",
)
TABLE_ROW_RE = re.compile(
    r"^\|\s*([a-z0-9_]+)\s*\|\s*(\d+)\s*\|\s*`([0-9a-f]{64})`\s*\|$"
)


class ProtocolError(ValueError):
    """Frozen S84 inputs or runtime context do not satisfy this diagnostic."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_protocol_cells(protocol_path: Path) -> list[dict[str, Any]]:
    text = protocol_path.read_text(encoding="utf-8")
    if "# S84 — Maestro native-thread-policy diagnostic" not in text:
        raise ProtocolError("unexpected_protocol_identity")
    if "The stable MPS row is excluded." not in text:
        raise ProtocolError("protocol_mps_exclusion_missing")
    try:
        table = text.split("| Operation | Repeats | Existing cell ID |", 1)[1]
        table = table.split("The stable MPS row is excluded.", 1)[0]
    except IndexError as exc:
        raise ProtocolError("protocol_cell_table_missing") from exc
    cells: list[dict[str, Any]] = []
    for line in table.splitlines():
        match = TABLE_ROW_RE.match(line.strip())
        if not match:
            continue
        operation, repeats, cell_id = match.groups()
        cells.append({
            "cell_id": cell_id,
            "candidate": "statevector",
            "width": 16,
            "operation_class": operation,
            "operation_repeats": int(repeats),
            "shots": 1,
            "seed": 12345,
        })
    if len(cells) != 9 or len({cell["cell_id"] for cell in cells}) != 9:
        raise ProtocolError("protocol_must_list_nine_unique_statevector_cells")
    if {cell["operation_class"] for cell in cells} != EXPECTED_OPERATIONS:
        raise ProtocolError("protocol_operation_set_mismatch")
    if {cell["operation_repeats"] for cell in cells} != EXPECTED_REPEATS:
        raise ProtocolError("protocol_repeat_set_mismatch")
    if {(cell["operation_class"], cell["operation_repeats"]) for cell in cells} != {
        (operation, repeats) for operation in EXPECTED_OPERATIONS for repeats in EXPECTED_REPEATS
    }:
        raise ProtocolError("protocol_cell_grid_incomplete")
    return cells


def verify_pinned_inputs(root: Path = ROOT) -> dict[str, str]:
    verified: dict[str, str] = {}
    for relative, expected in PINNED_INPUTS.items():
        path = root / relative
        if not path.is_file():
            raise ProtocolError(f"pinned_input_missing:{relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ProtocolError(f"pinned_input_hash_mismatch:{relative}:{actual}")
        verified[relative] = actual
    return verified


def load_frozen_cells(protocol_path: Path = PROTOCOL, root: Path = ROOT) -> list[dict[str, Any]]:
    protocol_hash = sha256_file(protocol_path)
    if protocol_hash != PROTOCOL_SHA256:
        raise ProtocolError(f"S84_protocol_hash_mismatch:{protocol_hash}")
    verify_pinned_inputs(root)
    cells = read_protocol_cells(protocol_path)
    contract = json.loads((root / "benchmark_v1/execution/manifests/maestro_common_panel_completion_v3.json").read_text())
    # Parent-only deterministic QASM construction. Keep this lazy: a spawned
    # timing worker imports this module before setting its arm environment.
    generate_v3_probe = importlib.import_module(
        "benchmark_v1.qre_benchmark.maestro_component_v2"
    ).generate_v3_probe
    run_manifest = json.loads(PILOT_RUN.read_text(encoding="utf-8"))
    if run_manifest.get("first_ten_cell_ids", [])[:9] != [cell["cell_id"] for cell in cells]:
        raise ProtocolError("S84_cells_not_the_first_nine_pinned_q16_cells")
    with (root / "artifacts/benchmark_v3/simulator/maestro_candidate_runtime_resume/calibration/raw_records.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        raw_rows = list(csv.DictReader(handle))
    timed: dict[str, list[dict[str, str]]] = {}
    for row in raw_rows:
        if row.get("record_type") != "attempt_result" or row.get("stage") != "v3_process_isolated_timed_repeat":
            continue
        timed.setdefault(row["cell_id"], []).append(row)
    for cell in cells:
        rows = timed.get(cell["cell_id"], [])
        if len(rows) != SESSIONS * REPETITIONS:
            raise ProtocolError(f"pilot_timed_row_count_mismatch:{cell['cell_id']}:{len(rows)}")
        if any(row.get("candidate") != "statevector" or row.get("width") != "16"
               or row.get("operation_class") != cell["operation_class"]
               or row.get("operation_repeats") != str(cell["operation_repeats"])
               or row.get("shots") != "1" for row in rows):
            raise ProtocolError(f"pilot_cell_descriptor_mismatch:{cell['cell_id']}")
        observed_pairs = {(int(row["session"]), int(row["repetition"])) for row in rows}
        expected_pairs = {(session, repetition) for session in range(1, SESSIONS + 1)
                          for repetition in range(REPETITIONS)}
        if observed_pairs != expected_pairs:
            raise ProtocolError(f"pilot_repetition_grid_mismatch:{cell['cell_id']}")
        probe = generate_v3_probe(contract, 16, cell["operation_class"], cell["operation_repeats"])
        if any(row.get("whole_qasm_sha256") != probe["whole_qasm_sha256"] for row in rows):
            raise ProtocolError(f"pilot_qasm_hash_mismatch:{cell['cell_id']}")
        cell["program"] = probe["program"]
        cell["whole_qasm_sha256"] = probe["whole_qasm_sha256"]
    summary_path = root / "artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/simulator/maestro_pilot_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary = {row["cell_id"]: row for row in csv.DictReader(handle)}
    if any(summary.get(cell["cell_id"], {}).get("stability_status") != "unavailable" for cell in cells):
        raise ProtocolError("S84_primary_cells_no_longer_match_unstable_pilot_scope")
    return cells


def build_paired_schedule(cells: list[dict[str, Any]], seed: int = SCHEDULE_SEED,
                          sessions: int = SESSIONS,
                          repetitions: int = REPETITIONS) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    schedule: list[dict[str, Any]] = []
    index = 0
    for cell in cells:
        for session in range(1, sessions + 1):
            for repetition in range(repetitions):
                pair_id = f"{cell['cell_id']}:s{session}:r{repetition}"
                order = list(ARMS)
                rng.shuffle(order)
                for arm in order:
                    index += 1
                    schedule.append({
                        "attempt_index": index,
                        "pair_id": pair_id,
                        "cell_id": cell["cell_id"],
                        "arm": arm,
                        "session": session,
                        "repetition": repetition,
                        "candidate": cell["candidate"],
                        "width": cell["width"],
                        "operation_class": cell["operation_class"],
                        "operation_repeats": cell["operation_repeats"],
                        "shots": cell["shots"],
                        "seed": cell["seed"],
                        "whole_qasm_sha256": cell.get("whole_qasm_sha256", ""),
                    })
    return schedule


def set_thread_environment(arm: str, environ: MutableMapping[str, str] | None = None) -> None:
    """Set the S84 arm policy before any native-facing imports in the child."""
    if arm not in ARMS:
        raise ProtocolError(f"unknown_thread_arm:{arm}")
    target = os.environ if environ is None else environ
    for key in THREAD_ENV_KEYS:
        target.pop(key, None)
    if arm == ARM_LIMITED:
        target["OMP_NUM_THREADS"] = "1"
        target["OPENBLAS_NUM_THREADS"] = "1"


def configure_then_capture(arm: str, capture: Callable[[], Any],
                           environ: MutableMapping[str, str] | None = None) -> Any:
    """Shared ordering primitive: apply env before the callback imports libraries."""
    set_thread_environment(arm, environ)
    return capture()


def _arm_policy(arm: str) -> dict[str, Any]:
    variables = {key: os.environ.get(key) for key in THREAD_ENV_KEYS}
    if arm == ARM_DEFAULT:
        declared = PINNED_DEFAULT_POLICY
    else:
        declared = "S84 treatment: OMP and OpenBLAS limited to one before native imports"
    return {
        "declared_policy": declared,
        "affinity": EXPECTED_AFFINITY,
        "native_thread_environment": variables,
        "timing_worker_count": 1,
    }


def _capture_context(arm: str) -> tuple[dict[str, Any], Any, Any]:
    """Import native packages only after set_thread_environment has run."""
    common = importlib.import_module("benchmark_v1.scripts.run_maestro_common_panel")
    modules = common._canonical_native_imports()
    maestro = modules["maestro"]
    v2 = importlib.import_module("benchmark_v1.qre_benchmark.maestro_component_v2")
    record = v2.resolve_v2_config(maestro, "statevector", None, 12345, 1e-8)
    context = common.collect_worker_context(
        config=record["resolved"], backend_threading_policy=_arm_policy(arm),
    )
    return context, maestro, record["object"]


def _pool_counts(context: Mapping[str, Any]) -> dict[str, list[int]]:
    identity = context.get("identity", {})
    pools = identity.get("native_threadpools")
    if not isinstance(pools, list):
        raise ProtocolError("threadpool_inventory_unavailable")
    observed: dict[str, list[int]] = {"openmp": [], "openblas": []}
    for pool in pools:
        internal = pool.get("internal_api")
        if internal in observed:
            observed[internal].append(int(pool["num_threads"]))
    return {key: sorted(values) for key, values in observed.items()}


def verify_arm_context(context: Mapping[str, Any], arm: str) -> None:
    identity = context.get("identity", {})
    if identity.get("cpu", {}).get("affinity") != EXPECTED_AFFINITY:
        raise ProtocolError(f"cpu_affinity_mismatch:{arm}")
    expected_env = {key: "" for key in THREAD_ENV_KEYS}
    if arm == ARM_LIMITED:
        expected_env["OMP_NUM_THREADS"] = "1"
        expected_env["OPENBLAS_NUM_THREADS"] = "1"
        expected_counts = {"openmp": [1], "openblas": [1, 1]}
    else:
        expected_counts = {"openmp": [28], "openblas": [28, 28]}
    if identity.get("native_thread_environment") != expected_env:
        raise ProtocolError(f"thread_environment_mismatch:{arm}")
    if _pool_counts(context) != expected_counts:
        raise ProtocolError(f"effective_threadpool_mismatch:{arm}:{_pool_counts(context)}")


def _normalized_context(context: Mapping[str, Any]) -> dict[str, Any]:
    identity = json.loads(json.dumps(context["identity"]))
    identity.pop("native_thread_environment", None)
    identity.pop("backend_threading_policy_sha256", None)
    policy = identity.get("backend_threading_policy", {})
    policy.pop("declared_policy", None)
    policy.pop("native_thread_environment", None)
    pools = identity.get("native_threadpools", [])
    if not isinstance(pools, list):
        raise ProtocolError("threadpool_inventory_unavailable")
    normalized_pools: list[dict[str, Any]] = []
    for pool in pools:
        if not isinstance(pool, dict):
            raise ProtocolError("invalid_threadpool_inventory_entry")
        # Threadpoolctl does not promise enumeration order. Remove only the
        # intentional treatment field, preserve every library identity field,
        # and compare the resulting inventory as a sorted multiset.
        normalized_pool = json.loads(json.dumps(pool))
        normalized_pool.pop("num_threads", None)
        normalized_pools.append(normalized_pool)
    identity["native_threadpools"] = sorted(
        normalized_pools,
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
    )
    return identity


def verify_pair_contexts(default_context: Mapping[str, Any], limited_context: Mapping[str, Any],
                         historical_default_identity_sha256: str) -> None:
    if default_context.get("identity_sha256") != historical_default_identity_sha256:
        raise ProtocolError("default_context_does_not_match_S83_worker_identity")
    if _normalized_context(default_context) != _normalized_context(limited_context):
        raise ProtocolError("non_thread_context_drift_between_arms")


def create_output_dir(path: Path) -> None:
    """Create once; refuse all existing paths, including empty directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)


def append_terminal_attempt(path: Path, row: Mapping[str, Any]) -> None:
    """Append and fsync one terminal attempt; never filter errors/timeouts."""
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ATTEMPT_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in ATTEMPT_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def _csv_prefix_sha256(path: Path, row_count: int) -> str:
    """Hash the exact CSV byte prefix through header plus `row_count` rows."""
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if len(lines) < row_count + 1:
        raise ProtocolError("resume_csv_prefix_shorter_than_row_count")
    return hashlib.sha256(b"".join(lines[:row_count + 1])).hexdigest()


def validate_resume_checkpoint(manifest_path: Path, attempts_path: Path,
                               expected_schedule: list[dict[str, Any]],
                               schedule_sha256: str,
                               expected_input_hashes: Mapping[str, str],
                               now_unix: float | None = None,
                               require_pinned_origin: bool = True) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate the exact append-only host-blocked prefix; no worker/API calls."""
    if set(path.name for path in manifest_path.parent.iterdir()) != {"run_manifest.json", "attempts.csv"}:
        raise ProtocolError("resume_artifact_directory_has_unexpected_or_temporary_files")
    if not manifest_path.is_file() or not attempts_path.is_file():
        raise ProtocolError("resume_checkpoint_files_missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "maestro_s84_threadpool_diagnostic_run_v1":
        raise ProtocolError("resume_manifest_schema_mismatch")
    if manifest.get("status") != "partial":
        raise ProtocolError(f"resume_manifest_not_partial:{manifest.get('status')}")
    if manifest.get("protocol_path") != str(PROTOCOL.relative_to(ROOT)):
        raise ProtocolError("resume_protocol_path_mismatch")
    if manifest.get("protocol_sha256") != RESUME_ORIGINAL_PROTOCOL_SHA256:
        raise ProtocolError("resume_original_protocol_pin_mismatch")
    if manifest.get("runner_sha256") != RESUME_ORIGINAL_RUNNER_SHA256:
        raise ProtocolError("resume_original_runner_pin_mismatch")
    if manifest.get("input_hashes") != dict(expected_input_hashes):
        raise ProtocolError("resume_original_input_hash_pins_mismatch")
    if manifest.get("schedule_seed") != SCHEDULE_SEED:
        raise ProtocolError("resume_schedule_seed_mismatch")
    if manifest.get("scheduled_attempts") != len(expected_schedule):
        raise ProtocolError("resume_scheduled_attempt_count_mismatch")
    if manifest.get("schedule_sha256") != schedule_sha256:
        raise ProtocolError("resume_schedule_hash_mismatch")
    if manifest.get("schedule") != expected_schedule:
        raise ProtocolError("resume_frozen_schedule_mismatch")
    if not attempts_path.is_file() or manifest.get("attempts_sha256") != sha256_file(attempts_path):
        raise ProtocolError("resume_attempt_ledger_hash_mismatch")

    with attempts_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(ATTEMPT_FIELDS):
            raise ProtocolError("resume_attempt_ledger_schema_mismatch")
        rows = list(reader)
    count = len(rows)
    if len(attempts_path.read_bytes().splitlines(keepends=True)) != count + 1:
        raise ProtocolError("resume_csv_physical_line_count_mismatch")
    if count < 3 or count >= len(expected_schedule) or manifest.get("attempt_count") != count:
        raise ProtocolError("resume_attempt_prefix_count_invalid")
    if require_pinned_origin:
        origin_manifest_sha256 = manifest.get("resume_origin_manifest_sha256")
        origin_attempts_sha256 = manifest.get("resume_origin_attempts_sha256")
        invocation_history = manifest.get("resume_invocations", [])
        if not invocation_history:
            if origin_manifest_sha256 is not None or origin_attempts_sha256 is not None:
                raise ProtocolError("resume_origin_pins_without_invocation_history")
            if sha256_file(manifest_path) != RESUME_ORIGINAL_MANIFEST_SHA256:
                raise ProtocolError("resume_original_manifest_sha256_mismatch")
            if sha256_file(attempts_path) != RESUME_ORIGINAL_ATTEMPTS_SHA256:
                raise ProtocolError("resume_original_attempts_sha256_mismatch")
        else:
            if origin_manifest_sha256 is None or origin_attempts_sha256 is None:
                raise ProtocolError("resume_origin_pins_missing_after_first_invocation")
            if (origin_manifest_sha256 != RESUME_ORIGINAL_MANIFEST_SHA256
                    or origin_attempts_sha256 != RESUME_ORIGINAL_ATTEMPTS_SHA256
                    or manifest.get("resume_origin_attempt_count") != RESUME_ORIGINAL_ATTEMPT_COUNT):
                raise ProtocolError("resume_origin_hash_pins_mismatch")
        try:
            origin_prefix_sha256 = _csv_prefix_sha256(attempts_path, RESUME_ORIGINAL_ATTEMPT_COUNT)
        except ProtocolError:
            raise
        if origin_prefix_sha256 != RESUME_ORIGINAL_ATTEMPTS_SHA256:
            raise ProtocolError("resume_original_attempts_byte_prefix_mismatch")
        if manifest.get("wall_budget_started_unix") != RESUME_ORIGINAL_BUDGET_STARTED_UNIX:
            raise ProtocolError("resume_original_budget_start_pin_mismatch")
    if manifest.get("terminal_counts") != dict(sorted({
        status: sum(row.get("status") == status for row in rows)
        for status in {row.get("status") for row in rows}
    }.items())):
        raise ProtocolError("resume_terminal_counts_mismatch")

    descriptor_fields = tuple(expected_schedule[0])
    for index, (row, expected) in enumerate(zip(rows, expected_schedule[:count]), start=1):
        if any(row.get(field) != str(expected[field]) for field in descriptor_fields):
            raise ProtocolError(f"resume_schedule_prefix_descriptor_mismatch:{index}")
        status = row.get("status")
        api_called = row.get("api_called", "").lower() == "true"
        if status == "ok":
            if not api_called or not row.get("reported_time_seconds"):
                raise ProtocolError(f"resume_invalid_success_row:{index}")
        elif status == "blocked_host_activity":
            if api_called or not row.get("error", "").startswith("host_activity_gate_"):
                raise ProtocolError(f"resume_invalid_host_block_row:{index}")
        else:
            raise ProtocolError(f"resume_disallowed_terminal_row:{index}:{status}")

    if [row["status"] for row in rows[:3]] != ["ok", "ok", "blocked_host_activity"]:
        raise ProtocolError("resume_initial_three_row_condition_mismatch")
    if rows[-1].get("status") != "blocked_host_activity" or rows[-1].get("api_called", "").lower() != "false":
        raise ProtocolError("resume_last_row_not_non_api_host_block")
    if manifest.get("terminal_reason") != rows[-1].get("error"):
        raise ProtocolError("resume_terminal_reason_mismatch")
    last = manifest.get("last_attempt")
    if not isinstance(last, dict) or any(
        str(last.get(field, "")) != rows[-1].get(field, "")
        for field in (*descriptor_fields, "status", "api_called", "error")
    ):
        raise ProtocolError("resume_manifest_last_attempt_mismatch")

    invocations = manifest.get("resume_invocations", [])
    if not isinstance(invocations, list):
        raise ProtocolError("resume_invocation_history_invalid")
    previous_terminal = 3
    for invocation_index, invocation in enumerate(invocations, start=1):
        if not isinstance(invocation, dict):
            raise ProtocolError("resume_prior_invocation_not_object")
        first_index = previous_terminal + 1
        terminal_index = invocation.get("terminal_attempt_index")
        if (invocation.get("invocation_index") != invocation_index
                or invocation.get("from_attempt_count") != previous_terminal
                or invocation.get("first_attempt_index") != first_index):
            raise ProtocolError("resume_invocation_range_mismatch")
        if invocation.get("status") == "blocked_resume_preflight":
            if (terminal_index != previous_terminal
                    or invocation.get("attempts_appended") != 0
                    or invocation.get("api_called") is not False
                    or invocation.get("terminal_reason") != "resume_host_activity_preflight_not_clear"
                    or not isinstance(invocation.get("host_activity_preflight"), dict)
                    or invocation["host_activity_preflight"].get("gate") not in {"busy", "unknown"}
                    or invocation.get("prior_attempts_sha256") !=
                        _csv_prefix_sha256(attempts_path, previous_terminal)):
                raise ProtocolError("resume_preflight_block_changed_attempt_count")
            continue
        if (invocation.get("status") != "blocked_host_activity"
                or not isinstance(terminal_index, int)
                or terminal_index < first_index or terminal_index > count
                or rows[terminal_index - 1].get("status") != "blocked_host_activity"):
            raise ProtocolError("resume_invocation_range_mismatch")
        appended = rows[previous_terminal:terminal_index]
        if (invocation.get("prior_attempts_sha256") !=
                _csv_prefix_sha256(attempts_path, previous_terminal)
                or invocation.get("attempts_appended") != len(appended)
                or invocation.get("api_called") is not any(
                    row.get("api_called", "").lower() == "true" for row in appended
                )):
            raise ProtocolError("resume_invocation_attempt_audit_mismatch")
        if (not appended
                or appended[-1].get("status") != "blocked_host_activity"
                or appended[-1].get("api_called", "").lower() != "false"
                or invocation.get("terminal_reason") != appended[-1].get("error")
                or any(row.get("status") != "ok" for row in appended[:-1])):
            raise ProtocolError("resume_invocation_suffix_order_mismatch")
        previous_terminal = terminal_index
    if previous_terminal != count:
        raise ProtocolError("resume_invocation_history_does_not_cover_suffix")
    if not invocations and count != 3:
        raise ProtocolError("resume_unrecorded_suffix_before_resume")

    started_unix = manifest.get("wall_budget_started_unix")
    if not isinstance(started_unix, (int, float)) or started_unix <= 0:
        raise ProtocolError("resume_original_budget_start_missing")
    if require_pinned_origin and started_unix != RESUME_ORIGINAL_BUDGET_STARTED_UNIX:
        raise ProtocolError("resume_original_budget_start_pin_mismatch")
    current_unix = time.time() if now_unix is None else now_unix
    remaining = started_unix + TOTAL_BUDGET_SECONDS - current_unix
    if remaining <= CLEANUP_RESERVE_SECONDS:
        raise ProtocolError("resume_original_wall_budget_expired_or_at_reserve")
    if manifest.get("wall_budget_seconds") != TOTAL_BUDGET_SECONDS:
        raise ProtocolError("resume_wall_budget_contract_mismatch")
    if float(manifest.get("wall_budget_overrun_seconds", 0.0)) > 0:
        raise ProtocolError("resume_original_wall_budget_overrun")
    return manifest, rows


def _resume_completion_fields(started_unix: float, finished_unix: float) -> dict[str, float]:
    elapsed = max(0.0, finished_unix - started_unix)
    return {
        "wall_budget_finished_unix": finished_unix,
        "wall_budget_elapsed_seconds": round(elapsed, 6),
        "wall_budget_overrun_seconds": round(max(0.0, elapsed - TOTAL_BUDGET_SECONDS), 6),
    }


def _update_resume_terminal_reason(manifest: MutableMapping[str, Any],
                                   status: str, reason: str) -> None:
    """Keep the prior ledger terminal on no-API preflight blocks only."""
    if status != "blocked_resume_preflight":
        manifest["terminal_reason"] = reason


def _record_resume_terminal_state(manifest: MutableMapping[str, Any],
                                  invocation: MutableMapping[str, Any],
                                  status: str, reason: str,
                                  prior_count: int) -> None:
    if status == "blocked_resume_preflight":
        # No scheduled row or API call was added, so preserve the ledger's
        # terminal reason while retaining the failed preflight in history.
        manifest["status"] = "partial"
        invocation["status"] = status
        invocation["terminal_attempt_index"] = prior_count
        invocation["attempts_appended"] = 0
        invocation["api_called"] = False
    else:
        manifest["status"] = status
        invocation["terminal_attempt_index"] = manifest["attempt_count"]
    invocation["terminal_reason"] = reason
    _update_resume_terminal_reason(manifest, status, reason)


def resume_attempt_003(protocol_path: Path, artifact_dir: Path,
                       root: Path = ROOT) -> dict[str, Any]:
    """Append only the unrun suffix of the one approved host-blocked attempt_003."""
    expected_dir = (root / RESUME_ARTIFACT_RELATIVE).resolve()
    if artifact_dir.resolve() != expected_dir:
        raise ProtocolError("resume_is_scoped_to_attempt_003_only")
    verified_pins = verify_pinned_inputs(root)
    cells = load_frozen_cells(protocol_path, root)
    schedule = build_paired_schedule(cells)
    if len(schedule) != 270:
        raise ProtocolError(f"unexpected_resume_attempt_count:{len(schedule)}")
    schedule_hash = canonical_hash(schedule)
    protocol_hash = sha256_file(protocol_path)
    runner_hash = sha256_file(Path(__file__).resolve())
    current_input_hashes = {
        **verified_pins,
        str(protocol_path.resolve().relative_to(root)): protocol_hash,
        "benchmark_v1/scripts/run_maestro_threadpool_intervention.py": runner_hash,
    }
    manifest_path = artifact_dir / "run_manifest.json"
    attempts_path = artifact_dir / "attempts.csv"

    with _exclusive_timing_lock(root):
        # Re-read and validate under the lock. Never rewrite or replace the
        # existing attempt prefix; all writes below append only fresh rows.
        expected_original_inputs = {
            **verified_pins,
            str(protocol_path.resolve().relative_to(root)): RESUME_ORIGINAL_PROTOCOL_SHA256,
            "benchmark_v1/scripts/run_maestro_threadpool_intervention.py": RESUME_ORIGINAL_RUNNER_SHA256,
        }
        manifest, prior_rows = validate_resume_checkpoint(
            manifest_path, attempts_path, schedule, schedule_hash,
            expected_original_inputs,
        )
        manifest["resume_origin_manifest_sha256"] = RESUME_ORIGINAL_MANIFEST_SHA256
        manifest["resume_origin_attempts_sha256"] = RESUME_ORIGINAL_ATTEMPTS_SHA256
        manifest["resume_origin_attempt_count"] = RESUME_ORIGINAL_ATTEMPT_COUNT
        prior_count = len(prior_rows)
        started_unix = float(manifest["wall_budget_started_unix"])
        remaining = started_unix + TOTAL_BUDGET_SECONDS - time.time()
        if remaining <= CLEANUP_RESERVE_SECONDS:
            raise ProtocolError("resume_original_wall_budget_expired_or_at_reserve")
        deadline = time.monotonic() + remaining
        invocation = {
            "invocation_index": len(manifest.get("resume_invocations", [])) + 1,
            "started_unix": time.time(),
            "protocol_sha256": protocol_hash,
            "runner_sha256": runner_hash,
            "input_hashes": current_input_hashes,
            "prior_attempts_sha256": manifest["attempts_sha256"],
            "from_attempt_count": prior_count,
            "first_attempt_index": prior_count + 1,
            "original_wall_budget_started_unix": started_unix,
            "original_wall_budget_deadline_unix": started_unix + TOTAL_BUDGET_SECONDS,
            "remaining_wall_budget_at_resume_start_seconds": round(remaining, 6),
            "attempts_appended": 0,
            "api_called": False,
            "terminal_attempt_index": prior_count,
            "status": "preflight_started",
        }
        manifest.setdefault("resume_invocations", []).append(invocation)
        manifest["status"] = "resuming"
        manifest["resume_runner_sha256"] = runner_hash
        _write_manifest(manifest_path, manifest)

        def finish(status: str, reason: str) -> dict[str, Any]:
            with attempts_path.open(newline="", encoding="utf-8") as handle:
                current_rows = list(csv.DictReader(handle))
            appended_rows = current_rows[prior_count:]
            invocation["attempts_appended"] = len(appended_rows)
            invocation["api_called"] = any(
                row.get("api_called", "").lower() == "true" for row in appended_rows
            )
            # A preflight block keeps the ledger checkpoint eligible; all
            # ledger-changing terminal states advance its top-level reason.
            _record_resume_terminal_state(manifest, invocation, status, reason, prior_count)
            manifest["attempts_sha256"] = sha256_file(attempts_path)
            invocation["completed_unix"] = time.time()
            if status == "complete":
                invocation["status"] = "completed"
            elif status == "blocked_resume_preflight":
                pass
            elif status == "partial" and manifest.get("last_attempt", {}).get("status") == "blocked_host_activity":
                invocation["status"] = "blocked_host_activity"
            elif "budget" in reason or "reserve" in reason:
                invocation["status"] = "budget_exhausted"
            else:
                invocation["status"] = "stopped_on_error"
            finished_unix = time.time()
            manifest.update(_resume_completion_fields(started_unix, finished_unix))
            _write_manifest(manifest_path, manifest)
            return manifest

        invocation["host_activity_preflight"] = capture_host_activity()
        if _remaining_wall_budget(deadline) <= CLEANUP_RESERVE_SECONDS:
            return finish("blocked_resume_budget", "wall_budget_cleanup_reserve_reached_during_resume_host_sample")
        if not host_activity_allows_run(invocation["host_activity_preflight"]):
            return finish("blocked_resume_preflight", "resume_host_activity_preflight_not_clear")

        preflight: dict[str, dict[str, Any]] = {}
        try:
            for arm in ARMS:
                timeout = _worker_timeout_seconds(deadline)
                if timeout is None:
                    raise ProtocolError(f"wall_budget_reserve_before_{arm}_resume_preflight")
                result = _isolated_worker(arm, None, None, timeout)
                if result.get("worker_reaped") is not True:
                    raise ProtocolError(f"resume_preflight_worker_reaping_failure:{arm}:{result}")
                if result.get("kind") != "preflight" or result.get("ok") is not True:
                    raise ProtocolError(f"resume_preflight_failed:{arm}:{result}")
                context = result["context"]
                verify_arm_context(context, arm)
                preflight[arm] = context
                if _remaining_wall_budget(deadline) <= CLEANUP_RESERVE_SECONDS:
                    raise ProtocolError(f"wall_budget_reserve_after_{arm}_resume_preflight")
            run_manifest = json.loads(PILOT_RUN.read_text(encoding="utf-8"))
            historical_identity = run_manifest["worker_contexts_by_config"]["statevector:None"]["identity_sha256"]
            verify_pair_contexts(preflight[ARM_DEFAULT], preflight[ARM_LIMITED], historical_identity)
        except Exception as exc:
            invocation["preflight"] = preflight
            reason = f"{type(exc).__name__}:{exc}"
            if "wall_budget" in reason or "reserve" in reason:
                return finish("blocked_resume_budget", reason)
            return finish("stopped_on_error", reason)

        invocation["preflight"] = {
            arm: {"identity_sha256": _sha_identity(ctx), "context": ctx}
            for arm, ctx in preflight.items()
        }
        invocation["status"] = "running"
        invocation["timing_started_unix"] = time.time()
        manifest["status"] = "resuming"
        _write_manifest(manifest_path, manifest)

        cell_by_id = {cell["cell_id"]: cell for cell in cells}
        terminal_counts = dict(manifest["terminal_counts"])
        stop_reason = ""
        for attempt in schedule[prior_count:]:
            snapshot: dict[str, Any] = {}
            remaining = _remaining_wall_budget(deadline)
            if remaining <= CLEANUP_RESERVE_SECONDS:
                row = _not_started_budget_row(attempt, snapshot)
                row["error"] = "wall_budget_cleanup_reserve_reached_before_resume_attempt"
            else:
                snapshot = capture_host_activity()
                row = _host_sample_budget_row(attempt, snapshot, deadline)
                remaining = _remaining_wall_budget(deadline)
                if row is None and remaining <= CLEANUP_RESERVE_SECONDS:
                    row = _not_started_budget_row(attempt, snapshot)
                    row["error"] = "wall_budget_cleanup_reserve_reached_after_resume_host_sample"
                if row is None and not host_activity_allows_run(snapshot):
                    row = {
                        **attempt,
                        "host_activity_json": json.dumps(snapshot, sort_keys=True, separators=(",", ":")),
                        "api_called": False,
                        "status": "blocked_host_activity",
                        "error": f"host_activity_gate_{snapshot.get('gate')}",
                    }
                if row is None:
                    timeout = _worker_timeout_seconds(deadline)
                    if timeout is None:
                        row = _not_started_budget_row(attempt, snapshot)
                        row["error"] = "wall_budget_cleanup_reserve_reached_at_resume_dispatch"
                    else:
                        cell = cell_by_id[attempt["cell_id"]]
                        result = _isolated_worker(
                            attempt["arm"], cell["program"], _sha_identity(preflight[attempt["arm"]]), timeout,
                        )
                        kind = str(result.get("kind", ""))
                        status = str(result.get("status") or (
                            "ok" if kind == "api_result" and result.get("ok", True) else
                            "timeout" if kind == "" and result.get("status") == "timeout" else
                            "blocked_context" if kind == "context_mismatch" else "error"
                        ))
                        worker_reaped = result.get("worker_reaped") is True
                        if not worker_reaped:
                            status = "worker_reaping_failure"
                        row = {
                            **attempt,
                            "host_activity_json": json.dumps(snapshot, sort_keys=True, separators=(",", ":")),
                            "api_called": bool(result.get("api_called", False)),
                            "status": status,
                            "reported_time_seconds": result.get("reported_time_seconds", ""),
                            "host_wall_seconds": result.get("host_wall_seconds", ""),
                            "outer_wall_seconds": result.get("outer_wall_seconds", ""),
                            "worker_context_sha256": result.get("worker_context_sha256", ""),
                            "error": (
                                "worker_not_reaped_after_bounded_cleanup"
                                if not worker_reaped
                                else result.get("error", result.get("reason", ""))
                            ),
                        }

            append_terminal_attempt(attempts_path, row)
            status = str(row["status"])
            terminal_counts[status] = terminal_counts.get(status, 0) + 1
            manifest["attempt_count"] += 1
            manifest["terminal_counts"] = dict(sorted(terminal_counts.items()))
            manifest["last_attempt"] = row
            _write_manifest(manifest_path, manifest)
            if status == "blocked_host_activity":
                stop_reason = str(row["error"])
                break
            if status == "not_started_wall_budget_exhausted":
                stop_reason = str(row["error"])
                break
            stop_reason = _timed_result_stop_reason(
                status, result.get("worker_reaped") is True, bool(row.get("api_called")),
            )
            if stop_reason:
                break

        final_status = "complete" if manifest["attempt_count"] == len(schedule) else "partial"
        if not stop_reason and final_status == "partial":
            stop_reason = "resume_suffix_stopped_without_terminal_row"
        return finish(final_status, stop_reason)


def _not_started_budget_row(attempt: Mapping[str, Any],
                            host_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **attempt,
        "host_activity_json": json.dumps(host_snapshot, sort_keys=True, separators=(",", ":")),
        "api_called": False,
        "status": "not_started_wall_budget_exhausted",
        "error": "wall_budget_cleanup_reserve_reached_during_host_activity_sample",
    }


def _remaining_wall_budget(deadline: float, now: float | None = None) -> float:
    """Return the budget left at a supplied monotonic instant (testable without waits)."""
    return deadline - (time.monotonic() if now is None else now)


def _wall_budget_completion_fields(started_monotonic: float,
                                   finished_monotonic: float,
                                   finished_unix: float) -> dict[str, float]:
    elapsed = max(0.0, finished_monotonic - started_monotonic)
    return {
        "wall_budget_finished_unix": finished_unix,
        "wall_budget_elapsed_seconds": round(elapsed, 6),
        "wall_budget_overrun_seconds": round(max(0.0, elapsed - TOTAL_BUDGET_SECONDS), 6),
    }


def _worker_timeout_seconds(deadline: float, now: float | None = None) -> float | None:
    """Bound a worker by the run budget while reserving time for cleanup and ledger writes."""
    remaining = _remaining_wall_budget(deadline, now)
    if remaining <= CLEANUP_RESERVE_SECONDS:
        return None
    return min(CALL_TIMEOUT_SECONDS, remaining - CLEANUP_RESERVE_SECONDS)


def _host_sample_budget_row(attempt: Mapping[str, Any],
                            host_snapshot: Mapping[str, Any],
                            deadline: float,
                            now: float | None = None) -> dict[str, Any] | None:
    """Return a durable no-API row only if the just-finished host sample used the budget."""
    if _remaining_wall_budget(deadline, now) > CLEANUP_RESERVE_SECONDS:
        return None
    return _not_started_budget_row(attempt, host_snapshot)


def _process_identity(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    stat_text = (proc / "stat").read_text(encoding="utf-8")
    closing_paren = stat_text.rfind(")")
    if closing_paren < 0:
        raise ValueError("proc_stat_missing_comm_delimiter")
    stat_fields_after_comm = stat_text[closing_paren + 2:].split()
    if len(stat_fields_after_comm) <= 19:
        raise ValueError("proc_stat_missing_start_time")
    uid_line = next(line for line in (proc / "status").read_text(encoding="utf-8").splitlines()
                    if line.startswith("Uid:"))
    uid = int(uid_line.split()[1])
    exe_path = str((proc / "exe").resolve(strict=True))
    return {
        "pid": pid,
        "uid": uid,
        "start_time_ticks": int(stat_fields_after_comm[19]),
        "exe_path": exe_path,
        "exe_sha256": sha256_file(Path(exe_path)),
    }


def _read_proc_cpu_snapshot() -> tuple[dict[int, dict[str, Any]], int, int]:
    processes: dict[int, dict[str, Any]] = {}
    raced = 0
    unreadable = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat_text = (proc / "stat").read_text(encoding="utf-8")
            opening_paren = stat_text.find("(")
            closing_paren = stat_text.rfind(")")
            fields = stat_text[closing_paren + 2:].split()
            if opening_paren < 0 or closing_paren < opening_paren or len(fields) <= 19:
                raise ValueError("malformed_proc_stat")
            pid = int(proc.name)
            processes[pid] = {
                "pid": pid,
                "command": stat_text[opening_paren + 1:closing_paren],
                "start_time_ticks": int(fields[19]),
                "cpu_ticks": int(fields[11]) + int(fields[12]),
            }
        except FileNotFoundError:
            raced += 1
        except (OSError, ValueError):
            unreadable += 1
    return processes, raced, unreadable


def _read_system_cpu_counters() -> tuple[int, int]:
    first_line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    fields = first_line.split()
    if not fields or fields[0] != "cpu" or len(fields) < 6:
        raise ValueError("malformed_system_cpu_counters")
    counters = [int(value) for value in fields[1:]]
    # guest/guest_nice are already included in user/nice; exclude them.
    total = sum(counters[:8])
    idle = counters[3] + counters[4]
    return total, idle


def _read_system_uptime_seconds() -> float:
    return float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])


def _build_cpu_activity(before: Mapping[int, Mapping[str, Any]],
                        after: Mapping[int, Mapping[str, Any]],
                        interval_seconds: float, runner_pid: int,
                        load_average: list[float],
                        system_before: tuple[int, int],
                        system_after: tuple[int, int],
                        uptime_after_seconds: float,
                        raced_process_count: int = 0,
                        unreadable_process_count: int = 0) -> dict[str, Any]:
    hz = int(os.sysconf("SC_CLK_TCK"))
    per_pid: list[dict[str, Any]] = []
    new_pids: list[int] = []
    for pid, current in after.items():
        previous = before.get(pid)
        if previous is None or previous.get("start_time_ticks") != current.get("start_time_ticks"):
            new_pids.append(pid)
            process_age = uptime_after_seconds - int(current["start_time_ticks"]) / hz
            if 0 <= process_age <= interval_seconds + 0.05:
                delta_ticks = int(current["cpu_ticks"])
                sample_kind = "born_during_interval_lower_bound"
            else:
                delta_ticks = None
                sample_kind = "new_or_unmatched_pid_unscored"
        else:
            delta_ticks = int(current["cpu_ticks"]) - int(previous["cpu_ticks"])
            sample_kind = "matched_pid_delta"
        if delta_ticks is not None and delta_ticks <= 0:
            continue
        per_pid.append({
            "pid": pid,
            "command": current["command"],
            "start_time_ticks": current["start_time_ticks"],
            "delta_cpu_ticks": delta_ticks,
            "cpu_percent_one_core_equivalent": (
                100.0 * delta_ticks / (hz * interval_seconds) if delta_ticks is not None else None
            ),
            "sample_kind": sample_kind,
        })
    per_pid.sort(key=lambda row: row["cpu_percent_one_core_equivalent"] or 0.0, reverse=True)
    external = [row for row in per_pid if row["pid"] != runner_pid]
    external_core_equivalent = sum(
        (row["cpu_percent_one_core_equivalent"] or 0.0) / 100.0 for row in external
    )
    max_external_pct = max(
        (row["cpu_percent_one_core_equivalent"] or 0.0 for row in external), default=0.0
    )
    total_delta = system_after[0] - system_before[0]
    idle_delta = system_after[1] - system_before[1]
    system_busy_pct = (
        100.0 * max(0, total_delta - idle_delta) / total_delta if total_delta > 0 else None
    )
    cpu: dict[str, Any] = {
        "status": "unavailable" if unreadable_process_count else "available",
        "sampling_method": "/proc/stat plus per-PID /proc/<pid>/stat utime+stime delta",
        "requested_interval_seconds": 0.25,
        "sample_interval_seconds": interval_seconds,
        "clock_ticks_per_second": hz,
        "load_average_1_5_15": load_average,
        "pinned_affinity_cpu_count": len(EXPECTED_AFFINITY),
        "system_cpu_busy_percent": system_busy_pct,
        "process_count_before": len(before),
        "process_count_after": len(after),
        "sampled_pid_count": sum(row["sample_kind"] == "matched_pid_delta" for row in per_pid),
        "uptime_at_sample_end_seconds": uptime_after_seconds,
        "external_cpu_core_equivalent": external_core_equivalent,
        "max_external_process_cpu_percent_one_core_equivalent": max_external_pct,
        "per_pid_cpu_delta": per_pid,
        "new_or_reused_pids_during_sample": sorted(new_pids),
        "raced_process_count": raced_process_count,
        "unreadable_process_count": unreadable_process_count,
        "gate_thresholds": {
            "max_external_process_percent": 100.0,
            "max_external_cpu_core_equivalent": 1.0,
            "max_one_minute_load_average": float(len(EXPECTED_AFFINITY)),
        },
    }
    cpu["busy"] = (
        max_external_pct >= 100.0
        or external_core_equivalent >= 1.0
        or load_average[0] >= len(EXPECTED_AFFINITY)
    )
    return cpu


def _assess_cpu_activity(cpu: dict[str, Any]) -> dict[str, Any]:
    deviations = []
    if cpu.get("status") != "available":
        deviations.append("cpu_process_or_system_sample_unavailable")
    if cpu.get("max_external_process_cpu_percent_one_core_equivalent", 0.0) >= 100.0:
        deviations.append("external_process_used_at_least_one_core_in_sample")
    if cpu.get("external_cpu_core_equivalent", 0.0) >= 1.0:
        deviations.append("aggregate_external_cpu_at_least_one_core_in_sample")
    load = cpu.get("load_average_1_5_15", [0.0])
    if load and load[0] >= len(EXPECTED_AFFINITY):
        deviations.append("one_minute_load_reached_28_cpu_affinity")
    cpu["deviations"] = deviations
    cpu["busy"] = bool(deviations)
    return cpu


def _assess_host_activity(snapshot: dict[str, Any]) -> dict[str, Any]:
    deviations: list[str] = []
    cpu = snapshot.get("cpu", {})
    gpu = snapshot.get("gpu", {})
    if cpu.get("status") != "available" or gpu.get("status") != "available":
        snapshot["gate"] = "unknown"
        snapshot["deviations"] = ["cpu_or_gpu_snapshot_unavailable"]
        return snapshot
    _assess_cpu_activity(cpu)
    if cpu.get("busy"):
        deviations.extend(cpu.get("deviations", []))

    devices = gpu.get("devices", [])
    if len(devices) != 1:
        deviations.append("gpu_device_count_changed")
    elif (devices[0].get("uuid") != APPROVED_GPU_BASELINE["device_uuid"]
          or devices[0].get("name") != APPROVED_GPU_BASELINE["device_name"]):
        deviations.append("gpu_device_identity_changed")
    if devices:
        device = devices[0]
        util = device.get("utilization_percent")
        memory = device.get("memory_used_mib")
        if util is None or memory is None:
            deviations.append("gpu_resource_measurement_unavailable")
        else:
            if util > APPROVED_GPU_BASELINE["max_utilization_percent"]:
                deviations.append("gpu_utilization_above_10_percent")
            if memory > APPROVED_GPU_BASELINE["max_device_memory_mib"]:
                deviations.append("gpu_device_memory_above_925_mib")

    observed = gpu.get("compute_processes", [])
    expected = APPROVED_GPU_BASELINE["processes"]
    observed_identities = [row.get("identity") for row in observed]
    expected_identities = [
        {key: row[key] for key in ("pid", "uid", "start_time_ticks", "exe_path", "exe_sha256")}
        for row in expected
    ]
    if any(identity is None for identity in observed_identities):
        deviations.append("gpu_process_identity_unavailable")
    elif sorted(observed_identities, key=lambda row: row["pid"]) != sorted(expected_identities, key=lambda row: row["pid"]):
        deviations.append("gpu_process_identity_set_changed")
    expected_by_pid = {row["pid"]: row for row in expected}
    for row in observed:
        identity = row.get("identity") or {}
        baseline = expected_by_pid.get(row.get("pid"), {})
        if (row.get("type_cg") != baseline.get("pmon_type")
                or row.get("command") != baseline.get("pmon_command")
                or row.get("command") != Path(identity.get("exe_path", "")).name):
            deviations.append(f"gpu_process_type_or_command_changed:{row.get('pid')}")

    expected_memory = {row["pid"]: row["baseline_fb_memory_mib"] for row in expected}
    observed_memory = {row.get("pid"): row.get("framebuffer_memory_mib") for row in observed}
    if set(observed_memory) == set(expected_memory):
        for pid, baseline in expected_memory.items():
            current = observed_memory[pid]
            if current is None:
                deviations.append(f"gpu_process_framebuffer_memory_unavailable:{pid}")
            elif current > baseline + APPROVED_GPU_BASELINE["max_process_memory_delta_mib"]:
                deviations.append(f"gpu_process_framebuffer_delta_above_64_mib:{pid}")
        total_memory = sum(value for value in observed_memory.values() if value is not None)
        if total_memory > APPROVED_GPU_BASELINE["max_total_process_fb_memory_mib"]:
            deviations.append("gpu_process_framebuffer_above_874_mib")
    else:
        deviations.append("gpu_process_memory_set_changed")

    snapshot["gate"] = "busy" if deviations else "clear"
    snapshot["deviations"] = deviations
    return snapshot


def capture_host_activity() -> dict[str, Any]:
    """Capture interval CPU deltas and the NVIDIA C/G process table."""
    snapshot: dict[str, Any] = {
        "captured_unix": time.time(), "runner_pid": os.getpid(), "cpu": {}, "gpu": {},
    }
    try:
        sample_started = time.monotonic()
        system_before = _read_system_cpu_counters()
        before, raced_before, unreadable_before = _read_proc_cpu_snapshot()
        time.sleep(0.25)
        after, raced_after, unreadable_after = _read_proc_cpu_snapshot()
        system_after = _read_system_cpu_counters()
        uptime_after = _read_system_uptime_seconds()
        interval = time.monotonic() - sample_started
        snapshot["cpu"] = _build_cpu_activity(
            before, after, interval, os.getpid(), list(os.getloadavg()),
            system_before, system_after, uptime_after,
            raced_process_count=raced_before + raced_after,
            unreadable_process_count=unreadable_before + unreadable_after,
        )
        _assess_cpu_activity(snapshot["cpu"])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        snapshot["cpu"] = {
            "status": "unavailable", "error": f"{type(exc).__name__}:{exc}", "busy": None,
        }

    try:
        pmon = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        devices = []
        for values in csv.reader(io.StringIO(gpu.stdout)):
            values = [part.strip() for part in values]
            if len(values) != 5:
                continue
            try:
                util, memory = int(values[3]), int(values[4])
            except ValueError:
                util, memory = None, None
            devices.append({"index": values[0], "uuid": values[1], "name": values[2],
                            "utilization_percent": util, "memory_used_mib": memory})
        gpu_processes = []
        parse_errors = []
        for line in pmon.stdout.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 12:
                parse_errors.append(line)
                continue
            try:
                pid = int(fields[1])
                fb_memory = int(fields[9])
                identity = _process_identity(pid)
            except (OSError, ValueError, RuntimeError, StopIteration) as exc:
                identity = None
                fb_memory = None
                try:
                    pid = int(fields[1])
                except ValueError:
                    pid = None
            gpu_processes.append({
                "gpu_index": fields[0], "pid": pid, "type_cg": fields[2],
                "sm_percent": fields[3], "mem_percent": fields[4],
                "framebuffer_memory_mib": fb_memory, "command": " ".join(fields[11:]),
                "identity": identity,
            })
        snapshot["gpu"] = {
            "status": "unavailable" if parse_errors else "available",
            "process_enumeration": "nvidia-smi pmon C/G table",
            "pmon_parse_errors": parse_errors,
            "devices": devices,
            "compute_processes": gpu_processes,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        snapshot["gpu"] = {
            "status": "unavailable", "error": f"{type(exc).__name__}:{exc}",
        }
    return _assess_host_activity(snapshot)


def host_activity_allows_run(snapshot: Mapping[str, Any]) -> bool:
    return snapshot.get("gate") == "clear"


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextlib.contextmanager
def _exclusive_timing_lock(root: Path = ROOT):
    lock_path = root / "work/locks/maestro_qcsim_v2_cpu_timing.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("another_Maestro_timing_worker_is_active") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _worker_entry(connection: Any, arm: str, qasm: str | None,
                  expected_identity_sha256: str | None) -> None:
    """Spawn target. Native env is set before the first native-facing import."""
    api_called = False
    try:
        context, maestro, config = configure_then_capture(
            arm, lambda: _capture_context(arm), os.environ,
        )
        verify_arm_context(context, arm)
        if qasm is None:
            connection.send({"kind": "preflight", "ok": True, "context": context})
            return
        if context.get("identity_sha256") != expected_identity_sha256:
            connection.send({"kind": "context_mismatch", "ok": False,
                             "reason": "timed_worker_identity_changed",
                             "actual_identity_sha256": context.get("identity_sha256")})
            return
        connection.send({"kind": "api_started"})
        api_called = True
        started = time.perf_counter()
        result = maestro.simple_execute(qasm, config, shots=1)
        host_wall = time.perf_counter() - started
        payload = dict(result)
        reported = payload.get("time_taken")
        if reported is None:
            connection.send({"kind": "api_error", "api_called": True,
                             "error": "missing_reported_time_seconds",
                             "host_wall_seconds": host_wall,
                             "worker_context_sha256": context.get("identity_sha256")})
        else:
            numeric = float(reported)
            status = "ok" if math.isfinite(numeric) and numeric > 0 else "error"
            connection.send({"kind": "api_result" if status == "ok" else "api_error",
                             "api_called": True, "status": status,
                             "reported_time_seconds": reported,
                             "host_wall_seconds": host_wall,
                             "worker_context_sha256": context.get("identity_sha256"),
                             "error": "" if status == "ok" else "nonpositive_or_nonfinite_reported_time"})
    except BaseException as exc:
        try:
            connection.send({"kind": "api_error" if api_called else "worker_error",
                             "api_called": api_called,
                             "error": f"{type(exc).__name__}:{exc}"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _isolated_worker(arm: str, qasm: str | None, expected_identity_sha256: str | None,
                     timeout_seconds: float) -> dict[str, Any]:
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker_entry,
                              args=(sender, arm, qasm, expected_identity_sha256))
    outer_start = time.perf_counter()
    process.start()
    sender.close()
    deadline = outer_start + timeout_seconds
    api_called = False
    final: dict[str, Any] | None = None
    while time.perf_counter() < deadline:
        remaining = max(0.0, deadline - time.perf_counter())
        if receiver.poll(min(0.1, remaining)):
            message = receiver.recv()
            if message.get("kind") == "api_started":
                api_called = True
                continue
            final = message
            break
        if not process.is_alive():
            break
    if final is None and process.is_alive():
        reaped = _bounded_reap_worker(process)
        receiver.close()
        return {"ok": False,
                "status": "timeout" if reaped else "worker_reaping_failure",
                "api_called": api_called,
                "error": "worker_timeout" if reaped else "worker_not_reaped_after_bounded_cleanup",
                "outer_wall_seconds": time.perf_counter() - outer_start,
                "worker_reaped": reaped}
    reaped = _bounded_reap_worker(process)
    receiver.close()
    if final is None:
        return {"ok": False,
                "status": "error" if reaped else "worker_reaping_failure",
                "api_called": api_called,
                "error": f"worker_exit_without_terminal_message:{process.exitcode}",
                "outer_wall_seconds": time.perf_counter() - outer_start,
                "worker_reaped": reaped}
    final["outer_wall_seconds"] = time.perf_counter() - outer_start
    final["worker_reaped"] = reaped
    if not reaped:
        final["ok"] = False
        final["status"] = "worker_reaping_failure"
        final["error"] = "worker_not_reaped_after_bounded_cleanup"
    return final


def _bounded_reap_worker(process: Any, grace_seconds: float = 1.0,
                         terminate_seconds: float = 3.0,
                         kill_seconds: float = 3.0) -> bool:
    """Join, then terminate/kill a lingering child within a bounded grace."""
    process.join(grace_seconds)
    if process.is_alive():
        process.terminate()
        process.join(terminate_seconds)
    if process.is_alive():
        process.kill()
        process.join(kill_seconds)
    return not process.is_alive()


def _timed_result_stop_reason(status: str, worker_reaped: bool,
                              api_called: bool) -> str:
    if not worker_reaped:
        return "timed_worker_reaping_failure"
    if status != "ok":
        return f"timed_worker_terminal_{status}"
    if not api_called:
        return "timed_worker_ok_without_api_call"
    return ""


def _sha_identity(context: Mapping[str, Any]) -> str:
    return str(context["identity_sha256"])


def execute(protocol_path: Path, artifact_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    verified_pins = verify_pinned_inputs(root)
    cells = load_frozen_cells(protocol_path, root)
    schedule = build_paired_schedule(cells)
    if len(schedule) != 270:
        raise ProtocolError(f"unexpected_attempt_count:{len(schedule)}")
    schedule_hash = canonical_hash(schedule)
    protocol_hash = sha256_file(protocol_path)
    runner_hash = sha256_file(Path(__file__).resolve())
    input_hashes = {
        **verified_pins,
        str(protocol_path.resolve().relative_to(root)): protocol_hash,
        "benchmark_v1/scripts/run_maestro_threadpool_intervention.py": runner_hash,
    }

    with _exclusive_timing_lock(root):
        create_output_dir(artifact_dir)
        # The operational no-new-worker budget starts when this run reserves
        # its unique output directory. OS scheduling and filesystem stalls can
        # still overrun it; completion metadata records the observed duration.
        budget_started_monotonic = time.monotonic()
        budget_started_unix = time.time()
        deadline = budget_started_monotonic + TOTAL_BUDGET_SECONDS
        manifest_path = artifact_dir / "run_manifest.json"
        attempts_path = artifact_dir / "attempts.csv"
        manifest: dict[str, Any] = {
            "schema": "maestro_s84_threadpool_diagnostic_run_v1",
            "protocol_path": str(protocol_path.resolve().relative_to(root)),
            "protocol_sha256": protocol_hash,
            "runner_sha256": runner_hash,
            "input_hashes": input_hashes,
            "status": "preflight_started",
            "schedule_seed": SCHEDULE_SEED,
            "schedule_sha256": schedule_hash,
            "scheduled_attempts": len(schedule),
            "wall_budget_seconds": TOTAL_BUDGET_SECONDS,
            "wall_budget_policy": "operational_no_new_worker_after_deadline_not_hard_realtime_cap",
            "worker_cleanup_and_ledger_reserve_seconds": CLEANUP_RESERVE_SECONDS,
            "wall_budget_started_unix": budget_started_unix,
            "wall_budget_scope": "from_output_directory_reservation_through_preflight_timing_worker_cleanup_and_terminal_persistence",
            "cells": [{key: value for key, value in cell.items() if key != "program"}
                      for cell in cells],
            "schedule": schedule,
            "preflight": {},
            "host_activity_preflight": None,
            "approved_gpu_baseline": APPROVED_GPU_BASELINE,
            "host_activity_gate": {
                "cpu_sample": "/proc/stat and per-PID /proc/<pid>/stat utime+stime deltas; requested 0.25 seconds, actual interval recorded",
                "cpu_busy_if": "a non-runner PID reaches 100% of one core, aggregate non-runner CPU reaches one core, or 1-minute load reaches 28",
                "gpu_process_source": "nvidia-smi pmon -c 1 -s um; includes C, G, and C+G clients",
                "gpu_busy_if": "GPU identity or C/G process set differs from pinned baseline, utilization exceeds 10%, device memory exceeds 925 MiB, any per-process framebuffer memory exceeds baseline by 64 MiB, or total exceeds 874 MiB",
                "unknown_policy": "fail closed if CPU or GPU activity cannot be sampled",
                "limitation": "point-in-time snapshots cannot rule out brief activity between polls; a changed desktop session requires a newly reviewed protocol",
            },
            "attempt_count": 0,
            "terminal_counts": {},
        }

        def finish_manifest() -> dict[str, Any]:
            # First commit the terminal result and all attempts, then snapshot
            # elapsed wall time. The final metadata rewrite itself is not
            # self-timed; a stalled/reaped-late worker or fsync can still be
            # visible as an overrun in this durable snapshot.
            _write_manifest(manifest_path, manifest)
            finished_monotonic = time.monotonic()
            manifest.update(_wall_budget_completion_fields(
                budget_started_monotonic, finished_monotonic, time.time()
            ))
            _write_manifest(manifest_path, manifest)
            return manifest

        _write_manifest(manifest_path, manifest)
        preflight: dict[str, dict[str, Any]] = {}
        host_snapshot = capture_host_activity()
        manifest["host_activity_preflight"] = host_snapshot
        if _remaining_wall_budget(deadline) <= CLEANUP_RESERVE_SECONDS:
            manifest["status"] = "blocked_preflight"
            manifest["terminal_reason"] = "wall_budget_cleanup_reserve_reached_during_initial_host_activity_sample"
            return finish_manifest()
        if not host_activity_allows_run(host_snapshot):
            manifest["status"] = "blocked_preflight"
            manifest["terminal_reason"] = f"host_activity_gate_{host_snapshot.get('gate')}"
            return finish_manifest()
        try:
            for arm in ARMS:
                preflight_timeout = _worker_timeout_seconds(deadline)
                if preflight_timeout is None:
                    manifest["status"] = "blocked_preflight"
                    manifest["preflight"] = {
                        key: {"identity_sha256": _sha_identity(value), "context": value}
                        for key, value in preflight.items()
                    }
                    manifest["terminal_reason"] = f"wall_budget_cleanup_reserve_reached_before_{arm}_preflight"
                    return finish_manifest()
                result = _isolated_worker(arm, None, None, preflight_timeout)
                if result.get("worker_reaped") is not True:
                    raise ProtocolError(f"preflight_worker_reaping_failure:{arm}:{result}")
                if result.get("kind") != "preflight" or result.get("ok") is not True:
                    raise ProtocolError(f"preflight_failed:{arm}:{result}")
                context_value = result["context"]
                verify_arm_context(context_value, arm)
                preflight[arm] = context_value
                if _remaining_wall_budget(deadline) <= CLEANUP_RESERVE_SECONDS:
                    manifest["status"] = "blocked_preflight"
                    manifest["preflight"] = {
                        key: {"identity_sha256": _sha_identity(value), "context": value}
                        for key, value in preflight.items()
                    }
                    manifest["terminal_reason"] = f"wall_budget_cleanup_reserve_reached_after_{arm}_preflight"
                    return finish_manifest()
            run_manifest = json.loads(PILOT_RUN.read_text(encoding="utf-8"))
            historical_identity = run_manifest["worker_contexts_by_config"]["statevector:None"]["identity_sha256"]
            verify_pair_contexts(preflight[ARM_DEFAULT], preflight[ARM_LIMITED], historical_identity)
        except Exception as exc:
            manifest["status"] = "blocked_preflight"
            manifest["preflight"] = preflight
            if _remaining_wall_budget(deadline) <= CLEANUP_RESERVE_SECONDS:
                manifest["terminal_reason"] = "wall_budget_cleanup_reserve_reached_during_preflight"
            else:
                manifest["terminal_reason"] = f"{type(exc).__name__}:{exc}"
            return finish_manifest()

        if _remaining_wall_budget(deadline) <= CLEANUP_RESERVE_SECONDS:
            manifest["status"] = "blocked_preflight"
            manifest["preflight"] = {
                arm: {"identity_sha256": _sha_identity(ctx), "context": ctx}
                for arm, ctx in preflight.items()
            }
            manifest["terminal_reason"] = "wall_budget_cleanup_reserve_reached_after_preflight_before_timing"
            return finish_manifest()

        manifest["preflight"] = {
            arm: {"identity_sha256": _sha_identity(ctx), "context": ctx}
            for arm, ctx in preflight.items()
        }
        manifest["status"] = "running"
        manifest["timing_started_unix"] = time.time()
        _write_manifest(manifest_path, manifest)
        cell_by_id = {cell["cell_id"]: cell for cell in cells}
        terminal_counts: dict[str, int] = {}
        stop_reason = ""
        for attempt in schedule:
            remaining = _remaining_wall_budget(deadline)
            if remaining <= CLEANUP_RESERVE_SECONDS:
                row = _not_started_budget_row(attempt, host_snapshot)
                row["error"] = "wall_budget_cleanup_reserve_reached_before_host_activity_sample"
                append_terminal_attempt(attempts_path, row)
                terminal_counts[row["status"]] = terminal_counts.get(row["status"], 0) + 1
                manifest["attempt_count"] += 1
                manifest["terminal_counts"] = dict(sorted(terminal_counts.items()))
                manifest["last_attempt"] = row
                stop_reason = row["error"]
                _write_manifest(manifest_path, manifest)
                break
            host_snapshot = capture_host_activity()
            row = _host_sample_budget_row(attempt, host_snapshot, deadline)
            remaining = _remaining_wall_budget(deadline)
            if row is None and remaining <= CLEANUP_RESERVE_SECONDS:
                row = _not_started_budget_row(attempt, host_snapshot)
                row["error"] = "wall_budget_cleanup_reserve_reached_after_host_activity_sample_before_worker"
            if row is not None:
                append_terminal_attempt(attempts_path, row)
                terminal_counts[row["status"]] = terminal_counts.get(row["status"], 0) + 1
                manifest["attempt_count"] += 1
                manifest["terminal_counts"] = dict(sorted(terminal_counts.items()))
                manifest["last_attempt"] = row
                stop_reason = row["error"]
                _write_manifest(manifest_path, manifest)
                break
            if not host_activity_allows_run(host_snapshot):
                row = {
                    **attempt,
                    "host_activity_json": json.dumps(host_snapshot, sort_keys=True, separators=(",", ":")),
                    "api_called": False,
                    "status": "blocked_host_activity",
                    "error": f"host_activity_gate_{host_snapshot.get('gate')}",
                }
                append_terminal_attempt(attempts_path, row)
                terminal_counts["blocked_host_activity"] = terminal_counts.get("blocked_host_activity", 0) + 1
                manifest["attempt_count"] += 1
                manifest["terminal_counts"] = dict(sorted(terminal_counts.items()))
                manifest["last_attempt"] = row
                stop_reason = row["error"]
                _write_manifest(manifest_path, manifest)
                break
            cell = cell_by_id[attempt["cell_id"]]
            # Account for gate evaluation and dispatch preparation too: do not
            # spawn a timed worker using a stale positive timeout.
            remaining = _remaining_wall_budget(deadline)
            if remaining <= CLEANUP_RESERVE_SECONDS:
                row = _not_started_budget_row(attempt, host_snapshot)
                row["error"] = "wall_budget_cleanup_reserve_reached_before_timed_worker"
                append_terminal_attempt(attempts_path, row)
                terminal_counts[row["status"]] = terminal_counts.get(row["status"], 0) + 1
                manifest["attempt_count"] += 1
                manifest["terminal_counts"] = dict(sorted(terminal_counts.items()))
                manifest["last_attempt"] = row
                stop_reason = row["error"]
                _write_manifest(manifest_path, manifest)
                break
            timed_timeout = _worker_timeout_seconds(deadline)
            if timed_timeout is None:
                row = _not_started_budget_row(attempt, host_snapshot)
                row["error"] = "wall_budget_cleanup_reserve_reached_at_timed_worker_dispatch"
                append_terminal_attempt(attempts_path, row)
                terminal_counts[row["status"]] = terminal_counts.get(row["status"], 0) + 1
                manifest["attempt_count"] += 1
                manifest["terminal_counts"] = dict(sorted(terminal_counts.items()))
                manifest["last_attempt"] = row
                stop_reason = row["error"]
                _write_manifest(manifest_path, manifest)
                break
            result = _isolated_worker(
                attempt["arm"], cell["program"], _sha_identity(preflight[attempt["arm"]]),
                timed_timeout,
            )
            kind = str(result.get("kind", ""))
            status = str(result.get("status") or (
                "ok" if kind == "api_result" and result.get("ok", True) else
                "timeout" if kind == "" and result.get("status") == "timeout" else
                "blocked_context" if kind == "context_mismatch" else
                "error"
            ))
            worker_reaped = result.get("worker_reaped") is True
            if not worker_reaped:
                status = "worker_reaping_failure"
            row = {
                **attempt,
                "host_activity_json": json.dumps(host_snapshot, sort_keys=True, separators=(",", ":")),
                "api_called": bool(result.get("api_called", False)),
                "status": status,
                "reported_time_seconds": result.get("reported_time_seconds", ""),
                "host_wall_seconds": result.get("host_wall_seconds", ""),
                "outer_wall_seconds": result.get("outer_wall_seconds", ""),
                "worker_context_sha256": result.get("worker_context_sha256", ""),
                "error": (
                    "worker_not_reaped_after_bounded_cleanup"
                    if not worker_reaped
                    else result.get("error", result.get("reason", ""))
                ),
            }
            append_terminal_attempt(attempts_path, row)
            terminal_counts[status] = terminal_counts.get(status, 0) + 1
            manifest["attempt_count"] += 1
            manifest["terminal_counts"] = dict(sorted(terminal_counts.items()))
            manifest["last_attempt"] = row
            stop_reason = _timed_result_stop_reason(status, worker_reaped, bool(row["api_called"]))
            if stop_reason:
                _write_manifest(manifest_path, manifest)
                break
            _write_manifest(manifest_path, manifest)
        manifest["status"] = "complete" if manifest["attempt_count"] == len(schedule) else "partial"
        manifest["terminal_reason"] = stop_reason
        manifest["timing_finished_unix"] = time.time()
        if attempts_path.exists():
            manifest["attempts_sha256"] = sha256_file(attempts_path)
        return finish_manifest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--execute", action="store_true",
                        help="perform the S84 API calls; requires an independently reviewed invocation")
    parser.add_argument("--resume", action="store_true",
                        help="append only to the specifically validated attempt_003 host-blocked prefix")
    args = parser.parse_args()
    if not args.execute:
        parser.error("explicit --execute is required; no timing is run without it")
    protocol = args.protocol.resolve()
    artifact = args.artifact_dir.resolve()
    try:
        result = resume_attempt_003(protocol, artifact) if args.resume else execute(protocol, artifact)
    except (OSError, ProtocolError, json.JSONDecodeError) as exc:
        parser.exit(2, f"S84 blocked before/while executing: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": result.get("status"),
        "attempt_count": result.get("attempt_count"),
        "terminal_counts": result.get("terminal_counts"),
        "artifact_dir": str(artifact),
        "protocol_sha256": result.get("protocol_sha256"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
