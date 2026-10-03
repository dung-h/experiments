from __future__ import annotations

import csv
import contextlib
import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import run_family_aware_joint_mps_ladder_v1 as ladder  # noqa: E402


def test_cudaq_version_uses_installed_distribution_not_runtime_build_banner(monkeypatch):
    class Runtime:
        __name__ = "cudaq"
        __version__ = "CUDA-Q Version 0.15.1 (https://github.com/NVIDIA/...)"

    monkeypatch.setattr(ladder.importlib.metadata, "packages_distributions", lambda: {"cudaq": ["cuda-quantum-cu13"]})
    monkeypatch.setattr(ladder.importlib.metadata, "version", lambda name: "0.15.1" if name == "cuda-quantum-cu13" else None)

    assert ladder.cudaq_distribution_versions(Runtime()) == {"cuda-quantum-cu13": "0.15.1"}
    assert Runtime.__version__.startswith("CUDA-Q Version 0.15.1")


def test_cudaq_version_fails_closed_if_distribution_metadata_missing(monkeypatch):
    class Runtime:
        __name__ = "cudaq"

    monkeypatch.setattr(ladder.importlib.metadata, "packages_distributions", lambda: {"cudaq": []})
    with pytest.raises(RuntimeError, match="cannot resolve installed distribution version"):
        ladder.cudaq_distribution_versions(Runtime())


def test_preflight_materializes_exact150_c44_and_ladder_without_cuda(tmp_path, monkeypatch):
    def cuda_import_forbidden(*_args, **_kwargs):
        raise AssertionError("preflight must not import or initialize CUDA-Q")

    monkeypatch.setattr(ladder, "worker", cuda_import_forbidden)
    output = tmp_path / "preflight"
    manifest = ladder.make_plan(output)

    assert manifest["assigned_unique_exact_qasm_hashes"] == 150
    assert manifest["c44_unique_hash_fold_counts"] == {"0": 37, "1": 28, "2": 26, "3": 25, "4": 34}
    assert manifest["planned_rung_configurations"] == 900
    assert manifest["planned_session_attempts"] == 2700
    assert manifest["pilot_session_attempts"] == 180
    assert manifest["cudaq_imported"] is False
    assert manifest["timing_performed"] is False
    assert manifest["training_performed"] is False
    with (output / "attempt_manifest.csv").open(newline="") as handle:
        attempts = list(csv.DictReader(handle))
    assert len(attempts) == 2700
    assert "family" not in attempts[0]
    with (output / "hash_feature_manifest.csv").open(newline="") as handle:
        hashes = list(csv.DictReader(handle))
    assert len(hashes) == 150
    assert set(hashes[0]).issuperset(set(ladder.INPUT_FEATURES))
    assert "family_component_training_label_only" in hashes[0]
    assert ladder.read_plan(output)[0]["runner_id"] == ladder.RUNNER_ID


def test_family_component_split_is_deterministic_and_hash_grouped():
    panel = [
        {"stratum": "core_q2_q9", "qasm_sha256": "h1", "family": "qft"},
        {"stratum": "core_q2_q9", "qasm_sha256": "h1", "family": "qft_alias"},
        {"stratum": "core_q2_q9", "qasm_sha256": "h2", "family": "qft_alias"},
        {"stratum": "core_q2_q9", "qasm_sha256": "h2", "family": "fourier"},
        *[
            {"stratum": "core_q2_q9", "qasm_sha256": f"g{index}", "family": f"family-{index}"}
            for index in range(5)
        ],
    ]
    components = ladder.family_components(panel)
    assert components["h1"] == components["h2"]
    first = ladder.deterministic_family_folds(components)
    second = ladder.deterministic_family_folds(components)
    assert first == second
    assert first["h1"] == first["h2"]


def test_quality_label_requires_complete_ladder_and_uses_lowest_observed_passing_rung():
    digest = "hash-a"
    records = []
    values = {2: [0.96, 0.97, 0.98], 4: [0.991, 0.992, 0.993], 8: [0.989, 0.994, 0.995],
              16: [0.999, 0.999, 0.999], 32: [1.0, 1.0, 1.0], 64: [1.0, 1.0, 1.0]}
    for chi, fidelities in values.items():
        for index, fidelity in enumerate(fidelities, start=1):
            records.append({
                "source_qasm_sha256": digest, "max_bond": chi, "session_id": f"session-{index}",
                "status": "ok", "fidelity": fidelity, "warm_seconds": [chi + index / 10],
                "basis_canary_status": "PASS", "ir_sha256": "ir-a",
            })
    target = ladder.reduce_ladder(records, [digest])[0]
    assert target["ladder_status"] == "quality_target_observed"
    assert target["minimum_passing_max_bond"] == 4
    assert target["quality_constrained_warm_get_state_seconds"] == pytest.approx(4.2)
    incomplete = ladder.reduce_ladder(records[:-1], [digest])[0]
    assert incomplete["ladder_status"] == "incomplete_or_failed_ladder"
    assert incomplete["minimum_passing_max_bond"] == ""
    assert incomplete["quality_constrained_warm_get_state_seconds"] == ""


def test_no_pass_is_not_imputed_or_assigned_a_runtime():
    digest = "hash-b"
    records = []
    for chi in ladder.RUNG_CHI:
        for index in range(1, ladder.SESSIONS + 1):
            records.append({
                "source_qasm_sha256": digest, "max_bond": chi, "session_id": f"session-{index}",
                "status": "ok", "fidelity": 0.98, "warm_seconds": [0.2, 0.3],
                "basis_canary_status": "PASS", "ir_sha256": "ir-b",
            })
    target = ladder.reduce_ladder(records, [digest])[0]
    assert target["ladder_status"] == "quality_unattainable_on_declared_ladder"
    assert target["minimum_passing_max_bond"] == ""
    assert target["quality_constrained_warm_get_state_seconds"] == ""


def test_fidelity_adapter_returns_normalized_state_overlap():
    import numpy as np

    class FakeMps:
        def amplitudes(self, bitstrings):
            assert bitstrings == ["00", "10", "01", "11"]
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    class FakeDense:
        def to_numpy(self):
            return np.asarray([2.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    assert ladder.state_fidelity(FakeMps(), FakeDense(), width=2) == pytest.approx(1.0)


def test_full_measurement_cannot_start_without_matching_passing_pilot(tmp_path):
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "preflight_manifest.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="passing pilot_report"):
        ladder.execute_measurement(type("Args", (), {
            "plan_dir": plan_dir, "output": tmp_path / "out", "scope": "full",
            "pilot_report": None, "resume": False,
        })())


def test_attempt_ledger_rejects_duplicate_and_validates_three_fresh_sessions():
    records = []
    for index in range(1, 4):
        records.append({
            "source_qasm_sha256": "hash-a", "max_bond": 2, "session_id": f"session-{index}",
            "attempt_terminal": True, "status": "ok", "first_seconds": 0.1,
            "warm_seconds": [0.1] * 5, "fidelity": 0.999,
            "basis_canary_status": "PASS", "basis_canary_ir_sha256": "b" * 64,
        })
    assert len(ladder.validate_attempt_ledger(records)) == 3
    with pytest.raises(ValueError, match="duplicate"):
        ladder.validate_attempt_ledger(records + [records[0]])
    status, detail = ladder.classify_worker_status("ok", "session-1", {"session_id": "wrong-session"})
    assert status == "technical_error"
    assert "mismatch" in detail


def test_run_worker_process_reports_success_timeout_and_resource_limit(monkeypatch):
    class Done:
        returncode = 0

        def __init__(self, payload=None):
            self.payload = payload

        def poll(self):
            return 0

        def wait(self):
            return 0

        def kill(self):
            raise AssertionError("completed process must not be killed")

    def payload_process(payload):
        def factory(_command, stdout, **_kwargs):
            stdout.write("WORKER_RESULT=" + json.dumps(payload) + "\n")
            return Done(payload)
        return factory

    monkeypatch.setattr(ladder.subprocess, "Popen", payload_process({"status": "ok", "session": {"session_id": "session-1"}}))
    status, error, result, peak = ladder.run_worker_process(["mock"], timeout_seconds=1)
    assert (status, error, result["status"], peak) == ("ok", "", "ok", 0)

    monkeypatch.setattr(ladder.subprocess, "Popen", payload_process({"status": "resource_limit", "error": "mock OOM"}))
    status, error, result, _ = ladder.run_worker_process(["mock"], timeout_seconds=1)
    assert (status, error, result["status"]) == ("resource_limit", "mock OOM", "resource_limit")

    class Hung:
        returncode = None

        def poll(self):
            return None

        def wait(self):
            self.returncode = -9
            return -9

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(ladder.subprocess, "Popen", lambda *_args, **_kwargs: Hung())
    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(ladder.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(ladder.time, "sleep", lambda _seconds: None)
    heartbeats = []
    status, error, result, _ = ladder.run_worker_process(
        ["mock"], timeout_seconds=1, heartbeat=lambda current_peak: heartbeats.append(current_peak))
    assert status == "timeout"
    assert "exceeded 1 seconds" in error
    assert result is None
    assert heartbeats == [0]


def test_mocked_pilot_success_promotes_to_full_without_cuda(tmp_path, monkeypatch):
    plan_dir = tmp_path / "plan"
    ladder.make_plan(plan_dir)
    environment = {"interpreter": "mock-python", "gpu": "mock RTX",
                   "cudaq_expected_version": "mock-cudaq", "qiskit_expected_version": "mock-qiskit"}
    monkeypatch.setattr(ladder, "capture_execution_environment", lambda _args: environment)
    monkeypatch.setattr(ladder, "parse_gpu_memory", lambda: (15000, 1000))
    monkeypatch.setattr(ladder.s85, "exclusive_compute_lock", contextlib.nullcontext)

    class ImmediateSuccess:
        returncode = 0

        def __init__(self, command, stdout, **_kwargs):
            session_id = command[command.index("--session-id") + 1]
            result = {
                "status": "ok", "error": "", "build_seconds": 0.001,
                "cudaq_version": "mock-cudaq", "qiskit_version": "mock-qiskit",
                "ir_sha256": "a" * 64, "quality_reference_seconds": 0.2,
                "basis_canary": {"status": "PASS", "ir_sha256": "b" * 64},
                "basis_canary_seconds": 0.003,
                "session": {"session_id": session_id, "first_seconds": 0.1,
                            "warm_seconds": [0.1] * 5, "fidelity": 0.999,
                            "quality_extract_seconds": 0.01},
            }
            stdout.write("WORKER_RESULT=" + json.dumps(result) + "\n")

        def poll(self):
            return 0

        def wait(self):
            return 0

        def kill(self):
            raise AssertionError("mock successful process must not be killed")

    monkeypatch.setattr(ladder.subprocess, "Popen", ImmediateSuccess)
    common = {"plan_dir": plan_dir, "qiskit_site_packages": plan_dir,
              "resume": False, "pilot_report": None}
    pilot_dir = tmp_path / "pilot"
    pilot = ladder.execute_measurement(type("Args", (), {
        **common, "output": pilot_dir, "scope": "pilot10",
    })())
    assert pilot["status"] == "PASS"
    assert pilot["attempt_rows"] == 180
    assert pilot["target_hashes"] == 10
    assert pilot["technical_error_count"] == 0
    assert all(row["status"] == "ok" for row in (
        json.loads(line) for line in (pilot_dir / "attempt_records.jsonl").read_text().splitlines()
    ))
    pilot_rows = [json.loads(line) for line in (pilot_dir / "attempt_records.jsonl").read_text().splitlines()]
    bad_pilot = dict(pilot)
    bad_pilot["attempt_ledger_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ledger digest"):
        ladder.validate_pilot_for_full(bad_pilot, pilot["preflight_manifest_sha256"], environment,
                                       pilot_rows, pilot["pilot_hashes"])

    full = ladder.execute_measurement(type("Args", (), {
        **common, "output": tmp_path / "full", "scope": "full",
        "pilot_report": pilot_dir / "pilot_report.json",
    })())
    assert full["scope"] == "full"
    assert full["attempt_rows"] == 2700
    assert full["target_hashes"] == 150
    assert full["status"] == "PASS"


def test_resume_preserves_cumulative_budget_and_prior_failures(tmp_path, monkeypatch):
    plan_dir = tmp_path / "plan"
    plan = ladder.make_plan(plan_dir)
    environment = {"interpreter": "mock-python", "gpu": "mock RTX",
                   "cudaq_expected_version": "mock-cudaq", "qiskit_expected_version": "mock-qiskit"}
    monkeypatch.setattr(ladder, "capture_execution_environment", lambda _args: environment)
    monkeypatch.setattr(ladder, "parse_gpu_memory", lambda: (15000, 1000))
    monkeypatch.setattr(ladder.s85, "exclusive_compute_lock", contextlib.nullcontext)

    class ImmediateSuccess:
        returncode = 0

        def __init__(self, command, stdout, **_kwargs):
            session_id = command[command.index("--session-id") + 1]
            result = {"status": "ok", "error": "", "build_seconds": 0.001,
                      "cudaq_version": "mock-cudaq", "qiskit_version": "mock-qiskit",
                      "ir_sha256": "a" * 64,
                      "basis_canary": {"status": "PASS", "ir_sha256": "b" * 64},
                      "session": {"session_id": session_id, "first_seconds": 0.1,
                                  "warm_seconds": [0.1] * 5, "fidelity": 0.999,
                                  "quality_extract_seconds": 0.01}}
            stdout.write("WORKER_RESULT=" + json.dumps(result) + "\n")

        def poll(self):
            return 0

        def wait(self):
            return 0

        def kill(self):
            raise AssertionError("successful mock cannot be killed")

    monkeypatch.setattr(ladder.subprocess, "Popen", ImmediateSuccess)
    output = tmp_path / "resume-pilot"
    output.mkdir()
    hashes = sorted(row["source_qasm_sha256"] for row in ladder.read_plan(plan_dir)[1])
    prior = {"source_qasm_sha256": hashes[0], "max_bond": 2, "session_id": "session-1",
             "attempt_terminal": True, "status": "timeout", "error": "prior timeout"}
    (output / "attempt_records.jsonl").write_text(json.dumps(prior) + "\n", encoding="utf-8")
    run_manifest = {"runner_id": ladder.RUNNER_ID,
                    "runner_sha256": plan["input_hashes"]["measurement_runner_sha256"],
                    "protocol_sha256": plan["protocol_sha256"],
                    "preflight_manifest_sha256": ladder.sha256_file(plan_dir / "preflight_manifest.json"),
                    "scope": "pilot10", "execution_environment": environment,
                    "status": "failed", "accumulated_wall_seconds": 3599.0,
                    "max_gpu_memory_used_mib_observed": 1000, "attempt_count": 1}
    ladder.write_json(output / "run_manifest.json", run_manifest)
    summary = ladder.execute_measurement(type("Args", (), {
        "plan_dir": plan_dir, "qiskit_site_packages": plan_dir, "output": output,
        "scope": "pilot10", "pilot_report": None, "resume": True,
    })())
    assert summary["status"] == "FAIL"
    assert summary["attempt_rows"] == 180
    assert summary["technical_error_count"] == 1
    final_run = json.loads((output / "run_manifest.json").read_text())
    assert final_run["accumulated_wall_seconds"] >= 3599.0
    assert final_run["technical_error_count"] == 1
