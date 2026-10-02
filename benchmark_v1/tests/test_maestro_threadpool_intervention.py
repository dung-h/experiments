"""Static safeguards for the S84 paired-thread-policy runner; no Maestro calls."""
from __future__ import annotations

import csv
import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmark_v1/scripts/run_maestro_threadpool_intervention.py"


def runner_module():
    spec = importlib.util.spec_from_file_location("maestro_s84_runner_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_synthetic_resume(module, *args, **kwargs):
    return module.validate_resume_checkpoint(
        *args, **kwargs, require_pinned_origin=False,
    )


def test_s84_cells_and_source_qasm_hashes_are_frozen_without_api_calls() -> None:
    module = runner_module()
    cells = module.load_frozen_cells()
    assert len(cells) == 9
    assert all(cell["candidate"] == "statevector" and cell["width"] == 16 for cell in cells)
    assert all(len(cell["program"]) > 0 and len(cell["whole_qasm_sha256"]) == 64 for cell in cells)


def test_schedule_is_deterministic_paired_and_randomized() -> None:
    module = runner_module()
    cells = [
        {"cell_id": f"cell-{idx}", "candidate": "statevector", "width": 16,
         "operation_class": "one_qubit_noncommuting", "operation_repeats": idx,
         "shots": 1, "seed": 12345, "whole_qasm_sha256": f"hash-{idx}"}
        for idx in (256, 512)
    ]
    first = module.build_paired_schedule(cells, seed=20261002)
    assert first == module.build_paired_schedule(cells, seed=20261002)
    assert len(first) == 2 * 3 * 5 * 2
    pairs: dict[str, list[dict[str, object]]] = {}
    for row in first:
        pairs.setdefault(str(row["pair_id"]), []).append(row)
    assert len(pairs) == 2 * 3 * 5
    assert all({str(row["arm"]) for row in rows} == set(module.ARMS) for rows in pairs.values())
    orientations = {tuple(str(row["arm"]) for row in rows) for rows in pairs.values()}
    assert orientations == {(module.ARM_DEFAULT, module.ARM_LIMITED),
                            (module.ARM_LIMITED, module.ARM_DEFAULT)}


def test_arm_environment_is_set_before_context_import_callback() -> None:
    module = runner_module()
    environment = {key: "99" for key in module.THREAD_ENV_KEYS}
    observed = module.configure_then_capture(
        module.ARM_LIMITED, lambda: dict(environment), environment,
    )
    assert observed["OMP_NUM_THREADS"] == "1"
    assert observed["OPENBLAS_NUM_THREADS"] == "1"
    assert "MKL_NUM_THREADS" not in observed
    assert "NUMEXPR_NUM_THREADS" not in observed
    control = {key: "99" for key in module.THREAD_ENV_KEYS}
    module.set_thread_environment(module.ARM_DEFAULT, control)
    assert control == {}


def test_spawn_worker_sets_environment_before_native_context_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    module = runner_module()
    assert not hasattr(module, "generate_v3_probe")  # no eager native-facing helper import
    for key in module.THREAD_ENV_KEYS:
        monkeypatch.setenv(key, "99")
    observed: dict[str, str] = {}
    context = {
        "identity_sha256": "test-context",
        "identity": {
            "cpu": {"affinity": list(range(28))},
            "native_thread_environment": {
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "", "NUMEXPR_NUM_THREADS": "",
            },
            "native_threadpools": [
                {"internal_api": "openmp", "num_threads": 1},
                {"internal_api": "openblas", "num_threads": 1},
                {"internal_api": "openblas", "num_threads": 1},
            ],
        },
    }

    def capture(arm: str):
        observed.update({key: os.environ.get(key, "") for key in module.THREAD_ENV_KEYS})
        return context, object(), object()

    class Connection:
        messages: list[dict[str, object]] = []

        def send(self, value: dict[str, object]) -> None:
            self.messages.append(value)

        def close(self) -> None:
            pass

    monkeypatch.setattr(module, "_capture_context", capture)
    connection = Connection()
    module._worker_entry(connection, module.ARM_LIMITED, None, None)
    assert observed == {
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "", "NUMEXPR_NUM_THREADS": "",
    }
    assert connection.messages == [{"kind": "preflight", "ok": True, "context": context}]


def _paired_threadpool_contexts(module):
    pools = [
        {
            "filepath": "/env/maestro.libs/libgomp.so",
            "internal_api": "openmp",
            "library": {"path": "/env/maestro.libs/libgomp.so", "sha256": "hash-gomp"},
            "num_threads": 28,
            "prefix": "libgomp",
            "user_api": "openmp",
        },
        {
            "architecture": "Haswell",
            "filepath": "/env/numpy.libs/libopenblas.so",
            "internal_api": "openblas",
            "library": {"path": "/env/numpy.libs/libopenblas.so", "sha256": "hash-numpy-blas"},
            "num_threads": 28,
            "prefix": "libscipy_openblas",
            "threading_layer": "pthreads",
            "user_api": "blas",
            "version": "0.3.29",
        },
        {
            "architecture": "Haswell",
            "filepath": "/env/scipy.libs/libopenblas.so",
            "internal_api": "openblas",
            "library": {"path": "/env/scipy.libs/libopenblas.so", "sha256": "hash-scipy-blas"},
            "num_threads": 28,
            "prefix": "libscipy_openblas",
            "threading_layer": "pthreads",
            "user_api": "blas",
            "version": "0.3.28",
        },
    ]
    default_environment = {key: "" for key in module.THREAD_ENV_KEYS}
    limited_environment = {
        **default_environment, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    }
    common_identity = {
        "cpu": {"affinity": list(range(28))},
        "backend_threading_policy": {
            "declared_policy": "arm policy varies",
            "affinity": list(range(28)),
            "native_thread_environment": default_environment,
            "timing_worker_count": 1,
        },
        "backend_threading_policy_sha256": "arm-policy-hash",
    }
    default_identity = {
        **copy.deepcopy(common_identity),
        "native_thread_environment": default_environment,
        "native_threadpools": copy.deepcopy(pools),
    }
    limited_identity = {
        **copy.deepcopy(common_identity),
        "native_thread_environment": limited_environment,
        "native_threadpools": [
            {**copy.deepcopy(pool), "num_threads": 1} for pool in reversed(pools)
        ],
    }
    limited_identity["backend_threading_policy"]["native_thread_environment"] = limited_environment
    return (
        {"identity_sha256": "historical-default-sha", "identity": default_identity},
        {"identity_sha256": "treatment-sha", "identity": limited_identity},
    )


def test_pair_context_accepts_reordered_native_pool_inventory_only() -> None:
    module = runner_module()
    default_context, limited_context = _paired_threadpool_contexts(module)

    module.verify_pair_contexts(
        default_context, limited_context, historical_default_identity_sha256="historical-default-sha",
    )
    normalized_pools = module._normalized_context(default_context)["native_threadpools"]
    identities = [json.dumps(pool, sort_keys=True, separators=(",", ":")) for pool in normalized_pools]
    assert identities == sorted(identities)
    assert len(normalized_pools) == 3
    assert all("num_threads" not in pool for pool in normalized_pools)
    assert all(pool["filepath"] and pool["library"]["sha256"] for pool in normalized_pools)
    assert all(pool["library"]["path"] and pool["internal_api"] and pool["user_api"]
               for pool in normalized_pools)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("filepath",), "/different/libopenblas.so"),
        (("library", "sha256"), "different-hash"),
        (("internal_api",), "different-api"),
        (("user_api",), "different-user-api"),
    ],
)
def test_pair_context_rejects_native_pool_identity_drift(path, replacement) -> None:
    module = runner_module()
    default_context, limited_context = _paired_threadpool_contexts(module)
    pool = limited_context["identity"]["native_threadpools"][0]
    target = pool
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(module.ProtocolError, match="non_thread_context_drift_between_arms"):
        module.verify_pair_contexts(
            default_context, limited_context, historical_default_identity_sha256="historical-default-sha",
        )


def test_host_activity_gate_fails_closed_on_busy_or_unknown() -> None:
    module = runner_module()
    assert module.host_activity_allows_run({"gate": "clear"})
    assert not module.host_activity_allows_run({"gate": "busy"})
    assert not module.host_activity_allows_run({"gate": "unknown"})


def _baseline_host_snapshot(module):
    baseline = module.APPROVED_GPU_BASELINE
    identity_keys = ("pid", "uid", "start_time_ticks", "exe_path", "exe_sha256")
    processes = []
    for item in baseline["processes"]:
        processes.append({
            "pid": item["pid"],
            "command": item["pmon_command"],
            "type_cg": item["pmon_type"],
            "framebuffer_memory_mib": item["baseline_fb_memory_mib"],
            "identity": {key: item[key] for key in identity_keys},
        })
    return {
        "cpu": {
            "status": "available", "busy": False,
            "load_average_1_5_15": [1.0, 1.0, 1.0],
            "external_cpu_core_equivalent": 0.0,
            "max_external_process_cpu_percent_one_core_equivalent": 0.0,
        },
        "gpu": {
            "status": "available",
            "devices": [{
                "uuid": baseline["device_uuid"], "name": baseline["device_name"],
                "utilization_percent": baseline["utilization_baseline_percent"],
                "memory_used_mib": baseline["device_memory_baseline_mib"],
            }],
            "compute_processes": processes,
        },
    }


def test_pinned_remote_desktop_gpu_baseline_is_allowed_with_small_rendering_jitter() -> None:
    module = runner_module()
    snapshot = _baseline_host_snapshot(module)
    snapshot["gpu"]["devices"][0]["utilization_percent"] = 10
    snapshot["gpu"]["devices"][0]["memory_used_mib"] = 925
    snapshot["gpu"]["compute_processes"][0]["framebuffer_memory_mib"] += 64
    snapshot["gpu"]["compute_processes"][1]["framebuffer_memory_mib"] += 64
    result = module._assess_host_activity(snapshot)
    assert result["gate"] == "clear"
    assert result["deviations"] == []


def test_new_graphics_client_blocks_even_if_it_reuses_an_approved_executable() -> None:
    module = runner_module()
    snapshot = _baseline_host_snapshot(module)
    extra = dict(snapshot["gpu"]["compute_processes"][0])
    extra["pid"] = 999999
    extra["identity"] = {**extra["identity"], "pid": 999999, "start_time_ticks": 123}
    snapshot["gpu"]["compute_processes"].append(extra)
    result = module._assess_host_activity(snapshot)
    assert result["gate"] == "busy"
    assert "gpu_process_identity_set_changed" in result["deviations"]


def test_gpu_graphics_type_is_part_of_the_frozen_process_identity() -> None:
    module = runner_module()
    snapshot = _baseline_host_snapshot(module)
    graphics = snapshot["gpu"]["compute_processes"][0]
    assert graphics["type_cg"] == "G"
    graphics["type_cg"] = "C+G"
    result = module._assess_host_activity(snapshot)
    assert result["gate"] == "busy"
    assert any(item.startswith("gpu_process_type_or_command_changed") for item in result["deviations"])


def test_gpu_activity_above_frozen_thresholds_blocks() -> None:
    module = runner_module()
    snapshot = _baseline_host_snapshot(module)
    snapshot["gpu"]["devices"][0]["utilization_percent"] = 11
    snapshot["gpu"]["devices"][0]["memory_used_mib"] = 926
    snapshot["gpu"]["compute_processes"][0]["framebuffer_memory_mib"] += 65
    result = module._assess_host_activity(snapshot)
    assert result["gate"] == "busy"
    assert "gpu_utilization_above_10_percent" in result["deviations"]
    assert "gpu_device_memory_above_925_mib" in result["deviations"]
    assert any(item.startswith("gpu_process_framebuffer_delta_above_64_mib") for item in result["deviations"])


def test_cpu_interval_delta_blocks_a_non_runner_using_one_full_core() -> None:
    module = runner_module()
    hz = int(module.os.sysconf("SC_CLK_TCK"))
    interval = 0.25
    one_core_ticks = int(module.math.ceil(hz * interval))
    before = {}
    after = {
        4242: {"pid": 4242, "command": "cpu-worker",
               "start_time_ticks": round((1000.0 - interval) * hz),
               "cpu_ticks": one_core_ticks},
    }
    cpu = module._build_cpu_activity(
        before, after, interval, runner_pid=9999,
        load_average=[0.2, 0.3, 0.4],
        system_before=(10000, 5000), system_after=(10000 + 4 * hz, 5000 + 2 * hz),
        uptime_after_seconds=1000.0,
    )
    result = module._assess_cpu_activity(cpu)
    assert result["busy"]
    assert "external_process_used_at_least_one_core_in_sample" in result["deviations"]
    process = next(item for item in result["per_pid_cpu_delta"] if item["pid"] == 4242)
    assert process["command"] == "cpu-worker"
    assert process["sample_kind"] == "born_during_interval_lower_bound"
    assert result["sample_interval_seconds"] == interval


def test_host_snapshot_reserve_threshold_records_no_api_call() -> None:
    module = runner_module()
    snapshot = {
        "gate": "clear",
        "cpu": {"sample_interval_seconds": 0.25},
    }
    assert module._host_sample_budget_row(
        {"attempt_index": 7, "cell_id": "test-cell"}, snapshot,
        deadline=100.0, now=84.99,
    ) is None
    row = module._host_sample_budget_row(
        {"attempt_index": 7, "cell_id": "test-cell"},
        snapshot, deadline=100.0, now=85.0,
    )
    assert row is not None
    assert row["status"] == "not_started_wall_budget_exhausted"
    assert row["api_called"] is False
    assert "sample_interval_seconds" in row["host_activity_json"]
    assert row["error"] == "wall_budget_cleanup_reserve_reached_during_host_activity_sample"


def test_workers_are_bounded_by_the_run_deadline_and_cleanup_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runner_module()
    assert module._worker_timeout_seconds(deadline=100.0, now=70.0) == 15.0
    assert module._worker_timeout_seconds(deadline=100.0, now=80.0) == 5.0
    assert module._worker_timeout_seconds(deadline=100.0, now=84.75) == 0.25
    assert module._worker_timeout_seconds(deadline=100.0, now=85.0) is None
    assert module._worker_timeout_seconds(deadline=1000.0, now=0.0) == 180.0
    monkeypatch.setattr(module.time, "monotonic", lambda: 85.0)
    assert module._worker_timeout_seconds(deadline=100.0) is None

    source = inspect.getsource(module.execute)
    assert source.index("create_output_dir(artifact_dir)") < source.index(
        "deadline = budget_started_monotonic + TOTAL_BUDGET_SECONDS"
    ) < source.index("host_snapshot = capture_host_activity()")
    assert "preflight_timeout = _worker_timeout_seconds(deadline)" in source
    assert "timed_timeout = _worker_timeout_seconds(deadline)" in source
    assert "wall_budget_cleanup_reserve_reached_after_preflight_before_timing" in source


def _write_resume_fixture(tmp_path: Path, module):
    directory = tmp_path / "attempt_003"
    directory.mkdir()
    schedule = [
        {
            "attempt_index": index,
            "pair_id": f"cell-a:s1:r{(index - 1) // 2}",
            "cell_id": "cell-a",
            "arm": "single_thread" if index in (1, 3) else "default",
            "session": 1,
            "repetition": (index - 1) // 2,
            "candidate": "statevector",
            "width": 16,
            "operation_class": "one_qubit_noncommuting",
            "operation_repeats": 256,
            "shots": 1,
            "seed": 12345,
            "whole_qasm_sha256": "qasm-hash",
        }
        for index in range(1, 9)
    ]
    rows = []
    for idx, item in enumerate(schedule[:3], start=1):
        row = {field: "" for field in module.ATTEMPT_FIELDS}
        row.update({key: str(value) for key, value in item.items()})
        row["host_activity_json"] = "{}"
        row["api_called"] = "True" if idx <= 2 else "False"
        row["status"] = "ok" if idx <= 2 else "blocked_host_activity"
        row["reported_time_seconds"] = "0.01" if idx <= 2 else ""
        row["error"] = "" if idx <= 2 else "host_activity_gate_busy"
        rows.append(row)
    ledger = directory / "attempts.csv"
    with ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=module.ATTEMPT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    now = module.time.time()
    manifest = {
        "schema": "maestro_s84_threadpool_diagnostic_run_v1",
        "status": "partial",
        "protocol_path": str(module.PROTOCOL.relative_to(module.ROOT)),
        "protocol_sha256": module.RESUME_ORIGINAL_PROTOCOL_SHA256,
        "runner_sha256": module.RESUME_ORIGINAL_RUNNER_SHA256,
        "input_hashes": {"test-pinned-input": "test-pin-sha"},
        "schedule_seed": module.SCHEDULE_SEED,
        "schedule_sha256": module.canonical_hash(schedule),
        "scheduled_attempts": len(schedule),
        "schedule": copy.deepcopy(schedule),
        "attempt_count": 3,
        "terminal_counts": {"blocked_host_activity": 1, "ok": 2},
        "terminal_reason": "host_activity_gate_busy",
        "last_attempt": rows[-1],
        "attempts_sha256": module.sha256_file(ledger),
        "wall_budget_started_unix": now - 60,
        "wall_budget_seconds": module.TOTAL_BUDGET_SECONDS,
        "wall_budget_elapsed_seconds": 60,
        "wall_budget_overrun_seconds": 0,
        "resume_invocations": [],
    }
    manifest_path = directory / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path, ledger, schedule, manifest


def _extend_resume_fixture_with_two_host_blocks(module, ledger: Path, schedule, manifest):
    with ledger.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    new_rows = []
    for descriptor, status, api_called, error in (
        (schedule[3], "ok", "True", ""),
        (schedule[4], "blocked_host_activity", "False", "host_activity_gate_gpu_busy"),
        (schedule[5], "blocked_host_activity", "False", "host_activity_gate_cpu_busy"),
    ):
        row = {field: "" for field in module.ATTEMPT_FIELDS}
        row.update({key: str(value) for key, value in descriptor.items()})
        row.update({
            "host_activity_json": "{}",
            "api_called": api_called,
            "status": status,
            "reported_time_seconds": "0.02" if status == "ok" else "",
            "error": error,
        })
        new_rows.append(row)
    with ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=module.ATTEMPT_FIELDS)
        writer.writeheader()
        writer.writerows([*rows, *new_rows])
    manifest["attempt_count"] = 6
    manifest["terminal_counts"] = {"blocked_host_activity": 3, "ok": 3}
    manifest["terminal_reason"] = new_rows[-1]["error"]
    manifest["last_attempt"] = new_rows[-1]
    manifest["attempts_sha256"] = module.sha256_file(ledger)
    manifest["resume_invocations"] = [
        {
            "invocation_index": 1,
            "from_attempt_count": 3,
            "first_attempt_index": 4,
            "terminal_attempt_index": 5,
            "attempts_appended": 2,
            "api_called": True,
            "prior_attempts_sha256": module._csv_prefix_sha256(ledger, 3),
            "status": "blocked_host_activity",
            "terminal_reason": new_rows[1]["error"],
        },
        {
            "invocation_index": 2,
            "from_attempt_count": 5,
            "first_attempt_index": 6,
            "terminal_attempt_index": 6,
            "attempts_appended": 1,
            "api_called": False,
            "prior_attempts_sha256": module._csv_prefix_sha256(ledger, 5),
            "status": "blocked_host_activity",
            "terminal_reason": new_rows[2]["error"],
        },
    ]
    return new_rows


def _pin_synthetic_resume_origin(module, ledger: Path, manifest: dict,
                                 manifest_pin: str) -> None:
    origin_attempts = module._csv_prefix_sha256(ledger, 3)
    manifest.update({
        "resume_origin_manifest_sha256": manifest_pin,
        "resume_origin_attempts_sha256": origin_attempts,
        "resume_origin_attempt_count": 3,
        "resume_invocations": [{
            "invocation_index": 1,
            "from_attempt_count": 3,
            "first_attempt_index": 4,
            "terminal_attempt_index": 3,
            "attempts_appended": 0,
            "api_called": False,
            "prior_attempts_sha256": module.sha256_file(ledger),
            "status": "blocked_resume_preflight",
            "terminal_reason": "resume_host_activity_preflight_not_clear",
            "host_activity_preflight": {"gate": "busy"},
        }],
    })


def test_append_only_resume_accepts_exact_host_blocked_prefix(tmp_path: Path) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    restored, rows = _validate_synthetic_resume(module,
        manifest_path, ledger, schedule, manifest["schedule_sha256"],
        expected_input_hashes={"test-pinned-input": "test-pin-sha"},
        now_unix=manifest["wall_budget_started_unix"] + 61,
    )
    assert restored["attempt_count"] == 3
    assert [row["status"] for row in rows] == ["ok", "ok", "blocked_host_activity"]
    assert rows[-1]["api_called"] == "False"


def test_append_only_resume_allows_no_api_resume_preflight_retry(tmp_path: Path) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    invocation = {
        "invocation_index": 1,
        "from_attempt_count": 3,
        "first_attempt_index": 4,
        "status": "preflight_started",
        "prior_attempts_sha256": module.sha256_file(ledger),
        "host_activity_preflight": {"gate": "busy"},
    }
    module._record_resume_terminal_state(
        manifest, invocation, "blocked_resume_preflight",
        "resume_host_activity_preflight_not_clear", prior_count=3,
    )
    manifest["resume_invocations"] = [invocation]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    original_ledger_sha256 = module.sha256_file(ledger)
    restored, rows = _validate_synthetic_resume(module,
        manifest_path, ledger, schedule, manifest["schedule_sha256"],
        expected_input_hashes={"test-pinned-input": "test-pin-sha"},
        now_unix=manifest["wall_budget_started_unix"] + 61,
    )
    assert restored["attempt_count"] == 3
    assert len(rows) == 3
    assert restored["terminal_reason"] == "host_activity_gate_busy"
    assert restored["resume_invocations"][0]["terminal_reason"] == "resume_host_activity_preflight_not_clear"
    assert module.sha256_file(ledger) == original_ledger_sha256


def test_resume_terminal_reason_advances_with_new_ledger_terminal() -> None:
    module = runner_module()
    manifest = {"status": "resuming", "attempt_count": 6,
                "terminal_reason": "host_activity_gate_busy"}
    invocation = {"status": "running"}
    module._record_resume_terminal_state(
        manifest, invocation, "partial", "host_activity_gate_gpu_busy", prior_count=5,
    )
    assert manifest["terminal_reason"] == "host_activity_gate_gpu_busy"
    assert invocation["terminal_reason"] == "host_activity_gate_gpu_busy"
    assert invocation["terminal_attempt_index"] == 6


def test_append_only_resume_validates_two_host_block_invocations(tmp_path: Path) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    _extend_resume_fixture_with_two_host_blocks(module, ledger, schedule, manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    restored, rows = _validate_synthetic_resume(module,
        manifest_path, ledger, schedule, manifest["schedule_sha256"],
        expected_input_hashes={"test-pinned-input": "test-pin-sha"},
        now_unix=manifest["wall_budget_started_unix"] + 61,
    )
    assert restored["attempt_count"] == 6
    assert [row["status"] for row in rows[3:]] == [
        "ok", "blocked_host_activity", "blocked_host_activity",
    ]


def test_append_only_resume_rejects_inconsistent_second_host_block_metadata(tmp_path: Path) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    _extend_resume_fixture_with_two_host_blocks(module, ledger, schedule, manifest)
    manifest["resume_invocations"][1]["attempts_appended"] = 2
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(module.ProtocolError, match="resume_invocation_attempt_audit_mismatch"):
        _validate_synthetic_resume(module,
            manifest_path, ledger, schedule, manifest["schedule_sha256"],
            expected_input_hashes={"test-pinned-input": "test-pin-sha"},
            now_unix=manifest["wall_budget_started_unix"] + 61,
        )


def test_append_only_resume_requires_host_block_to_end_each_appended_suffix(tmp_path: Path) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    _extend_resume_fixture_with_two_host_blocks(module, ledger, schedule, manifest)
    with ledger.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    all_rows[3]["status"] = "blocked_host_activity"
    all_rows[3]["api_called"] = "False"
    all_rows[3]["error"] = "host_activity_gate_busy"
    with ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=module.ATTEMPT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    manifest["terminal_counts"] = {"blocked_host_activity": 4, "ok": 2}
    manifest["resume_invocations"][0]["api_called"] = False
    manifest["attempts_sha256"] = module.sha256_file(ledger)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(module.ProtocolError, match="resume_invocation_suffix_order_mismatch"):
        _validate_synthetic_resume(module,
            manifest_path, ledger, schedule, manifest["schedule_sha256"],
            expected_input_hashes={"test-pinned-input": "test-pin-sha"},
            now_unix=manifest["wall_budget_started_unix"] + 61,
        )


def test_exact_csv_prefix_hash_survives_append(tmp_path: Path) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    initial_prefix = module._csv_prefix_sha256(ledger, 3)
    descriptor = schedule[3]
    row = {field: "" for field in module.ATTEMPT_FIELDS}
    row.update(descriptor)
    row.update({"api_called": True, "status": "ok", "reported_time_seconds": 0.02})
    module.append_terminal_attempt(ledger, row)
    assert module._csv_prefix_sha256(ledger, 3) == initial_prefix
    assert module.sha256_file(ledger) != initial_prefix


def test_first_resume_rejects_unpinned_or_tampered_original_manifest(tmp_path: Path) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    with pytest.raises(module.ProtocolError, match="resume_original_manifest_sha256_mismatch"):
        module.validate_resume_checkpoint(
            manifest_path, ledger, schedule, manifest["schedule_sha256"],
            expected_input_hashes={"test-pinned-input": "test-pin-sha"},
            now_unix=manifest["wall_budget_started_unix"] + 61,
        )


def test_first_resume_rejects_fabricated_origin_pins_without_history(tmp_path: Path) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    manifest.update({
        "resume_origin_manifest_sha256": module.RESUME_ORIGINAL_MANIFEST_SHA256,
        "resume_origin_attempts_sha256": module.RESUME_ORIGINAL_ATTEMPTS_SHA256,
        "resume_origin_attempt_count": 3,
    })
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(module.ProtocolError, match="resume_origin_pins_without_invocation_history"):
        module.validate_resume_checkpoint(
            manifest_path, ledger, schedule, manifest["schedule_sha256"],
            expected_input_hashes={"test-pinned-input": "test-pin-sha"},
            now_unix=manifest["wall_budget_started_unix"] + 61,
        )


def test_later_resume_rejects_reset_budget_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    test_origin_manifest = "a" * 64
    test_origin_attempts = module._csv_prefix_sha256(ledger, 3)
    monkeypatch.setattr(module, "RESUME_ORIGINAL_MANIFEST_SHA256", test_origin_manifest)
    monkeypatch.setattr(module, "RESUME_ORIGINAL_ATTEMPTS_SHA256", test_origin_attempts)
    monkeypatch.setattr(module, "RESUME_ORIGINAL_BUDGET_STARTED_UNIX", manifest["wall_budget_started_unix"])
    _pin_synthetic_resume_origin(module, ledger, manifest, test_origin_manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    # The fixture is accepted with the original cumulative-budget origin.
    module.validate_resume_checkpoint(
        manifest_path, ledger, schedule, manifest["schedule_sha256"],
        expected_input_hashes={"test-pinned-input": "test-pin-sha"},
        now_unix=manifest["wall_budget_started_unix"] + 61,
    )
    manifest["wall_budget_started_unix"] += 1.0
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(module.ProtocolError, match="resume_original_budget_start_pin_mismatch"):
        module.validate_resume_checkpoint(
            manifest_path, ledger, schedule, manifest["schedule_sha256"],
            expected_input_hashes={"test-pinned-input": "test-pin-sha"},
            now_unix=manifest["wall_budget_started_unix"] + 60,
        )


def test_later_resume_rejects_changed_original_ledger_byte_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    test_origin_manifest = "b" * 64
    test_origin_attempts = module._csv_prefix_sha256(ledger, 3)
    monkeypatch.setattr(module, "RESUME_ORIGINAL_MANIFEST_SHA256", test_origin_manifest)
    monkeypatch.setattr(module, "RESUME_ORIGINAL_ATTEMPTS_SHA256", test_origin_attempts)
    _pin_synthetic_resume_origin(module, ledger, manifest, test_origin_manifest)
    raw = ledger.read_bytes()
    assert b"{}" in raw
    ledger.write_bytes(raw.replace(b"{}", b"{ }", 1))
    manifest["attempts_sha256"] = module.sha256_file(ledger)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(module.ProtocolError, match="resume_original_attempts_byte_prefix_mismatch"):
        module.validate_resume_checkpoint(
            manifest_path, ledger, schedule, manifest["schedule_sha256"],
            expected_input_hashes={"test-pinned-input": "test-pin-sha"},
            now_unix=manifest["wall_budget_started_unix"] + 61,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"api_called": True},
        {"attempts_appended": 1},
        {"terminal_attempt_index": 4},
    ],
)
def test_append_only_resume_rejects_preflight_retry_that_changed_attempts(
    tmp_path: Path, changes: dict[str, object],
) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    invocation = {
        "invocation_index": 1,
        "from_attempt_count": 3,
        "first_attempt_index": 4,
        "terminal_attempt_index": 3,
        "attempts_appended": 0,
        "api_called": False,
        "prior_attempts_sha256": module.sha256_file(ledger),
        "status": "blocked_resume_preflight",
        "host_activity_preflight": {"gate": "busy"},
        "terminal_reason": "resume_host_activity_preflight_not_clear",
    }
    invocation.update(changes)
    manifest["resume_invocations"] = [invocation]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(module.ProtocolError, match="resume_preflight_block_changed_attempt_count"):
        _validate_synthetic_resume(module,
            manifest_path, ledger, schedule, manifest["schedule_sha256"],
            expected_input_hashes={"test-pinned-input": "test-pin-sha"},
            now_unix=manifest["wall_budget_started_unix"] + 61,
        )


def test_append_only_resume_does_not_accept_original_preflight_only_checkpoint(tmp_path: Path) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    manifest["status"] = "blocked_preflight"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(module.ProtocolError, match="resume_manifest_not_partial"):
        _validate_synthetic_resume(module,
            manifest_path, ledger, schedule, manifest["schedule_sha256"],
            expected_input_hashes={"test-pinned-input": "test-pin-sha"},
            now_unix=manifest["wall_budget_started_unix"] + 61,
        )


def test_append_only_resume_rejects_tampered_ledger_hash(tmp_path: Path) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    ledger.write_text(ledger.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(module.ProtocolError, match="resume_attempt_ledger_hash_mismatch"):
        _validate_synthetic_resume(module,
            manifest_path, ledger, schedule, manifest["schedule_sha256"],
            expected_input_hashes={"test-pinned-input": "test-pin-sha"},
            now_unix=manifest["wall_budget_started_unix"] + 61,
        )


def test_append_only_resume_rejects_tampered_frozen_schedule(tmp_path: Path) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    manifest["schedule"][3]["arm"] = "single_thread"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(module.ProtocolError, match="resume_frozen_schedule_mismatch"):
        _validate_synthetic_resume(module,
            manifest_path, ledger, schedule, manifest["schedule_sha256"],
            expected_input_hashes={"test-pinned-input": "test-pin-sha"},
            now_unix=manifest["wall_budget_started_unix"] + 61,
        )


def test_append_only_resume_rejects_last_timeout_even_with_rehashed_ledger(tmp_path: Path) -> None:
    module = runner_module()
    manifest_path, ledger, schedule, manifest = _write_resume_fixture(tmp_path, module)
    with ledger.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[-1]["status"] = "timeout"
    rows[-1]["error"] = "worker_timeout"
    with ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=module.ATTEMPT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest["attempts_sha256"] = module.sha256_file(ledger)
    manifest["terminal_counts"] = {"ok": 2, "timeout": 1}
    manifest["terminal_reason"] = "worker_timeout"
    manifest["last_attempt"] = rows[-1]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(module.ProtocolError, match="resume_disallowed_terminal_row"):
        _validate_synthetic_resume(module,
            manifest_path, ledger, schedule, manifest["schedule_sha256"],
            expected_input_hashes={"test-pinned-input": "test-pin-sha"},
            now_unix=manifest["wall_budget_started_unix"] + 61,
        )


def test_budget_completion_records_observable_overrun_and_decision_caveat() -> None:
    module = runner_module()
    fields = module._wall_budget_completion_fields(
        started_monotonic=10.0,
        finished_monotonic=3612.5,
        finished_unix=1_800_000_000.25,
    )
    assert fields == {
        "wall_budget_finished_unix": 1_800_000_000.25,
        "wall_budget_elapsed_seconds": 3602.5,
        "wall_budget_overrun_seconds": 2.5,
    }
    decision = (ROOT / "benchmark_v1/decisions/S84_MAESTRO_THREADPOOL_DIAGNOSTIC_20261002.md").read_text(
        encoding="utf-8"
    )
    assert "not a hard real-time wall cap" in decision
    assert "filesystem/fsync stall" in decision
    assert "observed overrun seconds" in decision


def test_terminal_error_and_timeout_attempts_are_durably_appended(tmp_path: Path) -> None:
    module = runner_module()
    path = tmp_path / "attempts.csv"
    base = {field: "" for field in module.ATTEMPT_FIELDS}
    module.append_terminal_attempt(path, {**base, "attempt_index": 1, "status": "error",
                                          "api_called": True, "error": "native_exception"})
    module.append_terminal_attempt(path, {**base, "attempt_index": 2, "status": "timeout",
                                          "api_called": True, "error": "worker_timeout"})
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["status"], row["error"]) for row in rows] == [
        ("error", "native_exception"), ("timeout", "worker_timeout"),
    ]


def test_output_directory_is_never_reused_or_overwritten(tmp_path: Path) -> None:
    module = runner_module()
    target = tmp_path / "new-run"
    module.create_output_dir(target)
    sentinel = target / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.create_output_dir(target)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
