from __future__ import annotations

import sys
import csv
import fcntl
import json
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import run_mps_fixed_chi16_runtime_adaptation_v1 as adaptation  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_gpu_lease_path(tmp_path, monkeypatch):
    # Unit tests must never acquire/create the real host GPU lease.
    monkeypatch.setattr(adaptation, "GPU_LEASE_PATH", tmp_path / "gpu-lease.lock")


def test_source_dag_features_strip_timing_ir_and_ignore_parameter_values():
    record = {
        "source_qasm_sha256": "hash-a",
        "num_qubits": 3,
        "representation": "source_qiskit_dag_topology_v1_not_Azizov_GNN_ready",
        "nodes": [
            {"node_id": 0, "operation": "h", "params_text": [], "qargs": [0]},
            {"node_id": 1, "operation": "cx", "params_text": [], "qargs": [0, 1]},
            {"node_id": 2, "operation": "rx", "params_text": ["3.14159"], "qargs": [1]},
            {"node_id": 3, "operation": "barrier", "params_text": [], "qargs": [0, 1]},
            {"node_id": 4, "operation": "measure", "params_text": [], "qargs": [0, 1]},
        ],
    }
    globals_, graph = adaptation.graph_features(record)

    assert globals_ == {
        "active_width": 2,
        "measurement_stripped_structural_depth": 2,
        "one_qubit_gate_count": 2,
        "two_qubit_gate_count": 1,
        "multi_qubit_gate_count": 0,
        "swap_like_gate_count": 0,
    }
    assert [node["operation"] for node in graph["nodes"]] == ["h", "cx", "rx"]
    assert graph["nodes"][2]["parameter_present"] is True
    assert "3.14159" not in repr(graph)
    assert graph["edge_src"] == [0, 1]
    assert graph["edge_dst"] == [1, 2]
    assert graph["removed_measurements"] == 1
    assert graph["removed_barriers"] == 1
    assert graph["nodes"][1]["qargs"] == [0, 1]


def test_node_encoder_preserves_ordered_qarg_roles():
    node = {"operation": "cx", "arity": 2, "parameter_present": False, "qargs": [2, 0]}
    encoded = adaptation.encode_node_features(node, {"cx": 0}, 1, 3)
    gate_width = 2

    assert encoded[0] == 1.0
    assert encoded[gate_width + 2] == 1.0  # two-qubit arity bucket
    assert encoded[gate_width + 5:gate_width + 7] == [1.0, 0.0]
    assert encoded[gate_width + 8:gate_width + 10] == [1.0, 1.0]


def test_exact_hash_target_reduction_keeps_quality_failed_and_does_not_impute_errors():
    panel = [
        {"stratum": "core_q2_q9", "qasm_sha256": "hash-a", "panel_member_id": "a-1", "family": "qft"},
        {"stratum": "core_q2_q9", "qasm_sha256": "hash-a", "panel_member_id": "a-2", "family": "qft_alias"},
        {"stratum": "core_q2_q9", "qasm_sha256": "hash-b", "panel_member_id": "b-1", "family": "grover"},
    ]
    raw = []
    for member, session, value in [
        ("a-1", "s1", "1"), ("a-1", "s1", "3"),
        ("a-2", "s1", "10"), ("a-1", "s2", "8"), ("a-2", "s2", "12"),
    ]:
        raw.append({
            "stratum": "core_q2_q9", "qasm_sha256": "hash-a", "observation_kind": "warm",
            "status": "quality_failed", "panel_member_id": member, "session_id": session,
            "fidelity": "0.8", "quality_threshold": "0.99", "t_warm_execution_s": value,
            "error": "quality threshold not met",
        })
    raw.append({
        "stratum": "core_q2_q9", "qasm_sha256": "hash-b", "observation_kind": "warm",
        "status": "adapter_error", "panel_member_id": "b-1", "session_id": "s1",
        "fidelity": "", "quality_threshold": "", "t_warm_execution_s": "", "error": "adapter failed",
    })

    rows = adaptation.aggregate_targets(
        raw, panel, {"hash-a": 0, "hash-b": 1}, expected_core_members=3, expected_hash_groups=2
    )
    by_hash = {row["source_qasm_sha256"]: row for row in rows}

    assert by_hash["hash-a"]["target_status"] == "runtime_observed"
    assert by_hash["hash-a"]["target_seconds"] == 9.0
    assert by_hash["hash-a"]["quality_status_separate"] == "quality_failed"
    assert by_hash["hash-a"]["quality_is_model_input"] is False
    assert by_hash["hash-b"]["target_status"] == "unavailable_adapter_error"
    assert by_hash["hash-b"]["target_seconds"] == ""


def test_fold_map_rejects_exact_hash_crossing_outer_folds():
    c44 = [
        {"stratum": "core_q2_q9", "source_sha256": "same", "fold": "0"},
        {"stratum": "core_q2_q9", "source_sha256": "same", "fold": "1"},
    ]

    with pytest.raises(ValueError, match="crosses folds"):
        adaptation.frozen_fold_map(c44)


def test_measurement_must_be_global_terminal_even_if_later_gate_uses_other_qubit():
    record = {
        "source_qasm_sha256": "hash-nonterminal",
        "num_qubits": 2,
        "representation": "source_qiskit_dag_topology_v1_not_Azizov_GNN_ready",
        "nodes": [
            {"node_id": 0, "operation": "h", "params_text": [], "qargs": [0]},
            {"node_id": 1, "operation": "measure", "params_text": [], "qargs": [0]},
            {"node_id": 2, "operation": "x", "params_text": [], "qargs": [1]},
        ],
    }

    with pytest.raises(ValueError, match="nonterminal measurement"):
        adaptation.graph_features(record)


def _authorized_gate_document():
    rules = dict(adaptation.FOLD0_CHECKPOINT_RULES)
    return {
        "default_execution_state": "not_authorized",
        "gates": [{
            "method_family": adaptation.MPS_METHOD_FAMILY,
            "method_id": adaptation.MPS_METHOD_ID,
            "target_clock_id": adaptation.TARGET_ID,
            "authorization_decision": str(adaptation.MPS_AUTH_DECISION.relative_to(adaptation.ROOT)),
            "state": adaptation.AUTHORIZED_MPS_GATE_STATE,
            "execution_authorization": {
                "status": "conditionally_authorized",
                "scope": "fixed_split_five_fold_oof_with_fold0_technical_checkpoint",
                "folds": [0, 1, 2, 3, 4],
                "seeds": list(adaptation.SEEDS),
                "prediction_methods": list(adaptation.AUTHORIZED_PREDICTION_METHODS),
                "family_ood": False,
                "five_fold_oof": True,
                "leaderboard_promotion": False,
                "new_simulator_timing": False,
                "requires_s84_complete": False,
                "requires_exclusive_timing_lock": True,
                "requires_host_gpu_lease": True,
                "requires_no_live_timing_worker": True,
                "fold0_checkpoint_rules": rules,
                "basis": "user conditional request plus scientific review recommendation",
            },
        }],
    }


def test_method_authorization_is_scoped_and_does_not_change_global_default():
    document = json.loads(adaptation.SIM_GATES.read_text(encoding="utf-8"))
    validated = adaptation.validate_mps_execution_authorization(document)

    assert document["default_execution_state"] == "not_authorized"
    assert validated["method_id"] == adaptation.MPS_METHOD_ID
    assert validated["folds"] == [0, 1, 2, 3, 4]
    assert validated["seeds"] == [42, 1234, 31415]
    assert validated["family_ood"] is False
    other_gates = [gate for gate in document["gates"] if gate.get("method_id") != adaptation.MPS_METHOD_ID]
    assert all("execution_authorization" not in gate for gate in other_gates)


@pytest.mark.parametrize("mutation, message", [
    ("global", "global execution default"),
    ("family", "method family"),
    ("target_clock", "target clock"),
    ("scope", "invalid scope"),
    ("missing_fold", "all five frozen folds"),
    ("family_ood", "exceeds the fixed-split five-fold"),
    ("metric_gate", "continuation rules"),
    ("completion_dependency", "timing exclusion without S84 completion"),
    ("missing_lock", "timing exclusion without S84 completion"),
    ("missing_gpu_lease", "timing exclusion without S84 completion"),
])
def test_authorization_fails_closed_when_scope_expands(mutation, message):
    document = _authorized_gate_document()
    gate = document["gates"][0]
    if mutation == "global":
        document["default_execution_state"] = "authorized"
    elif mutation == "family":
        gate["method_family"] = "another_method_family"
    elif mutation == "target_clock":
        gate["target_clock_id"] = "another_target_clock"
    elif mutation == "scope":
        gate["execution_authorization"]["scope"] = "all_folds"
    elif mutation == "missing_fold":
        gate["execution_authorization"]["folds"] = [0, 1]
    elif mutation == "family_ood":
        gate["execution_authorization"]["family_ood"] = True
    elif mutation == "metric_gate":
        gate["execution_authorization"]["fold0_checkpoint_rules"]["metrics_quality_inspected"] = True
    elif mutation == "completion_dependency":
        gate["execution_authorization"]["requires_s84_complete"] = True
    elif mutation == "missing_lock":
        gate["execution_authorization"]["requires_exclusive_timing_lock"] = False
    elif mutation == "missing_gpu_lease":
        gate["execution_authorization"]["requires_host_gpu_lease"] = False

    with pytest.raises(ValueError, match=message):
        adaptation.validate_mps_execution_authorization(document)


def test_partial_blocked_or_absent_s84_does_not_block_mps(tmp_path, monkeypatch):
    manifest_path = tmp_path / "run_manifest.json"
    assert adaptation.s84_status_audit(manifest_path)["status"] == "not_present"
    manifest_path.write_text(json.dumps({"status": "blocked_resume_preflight", "attempt_count": 3, "scheduled_attempts": 270}), encoding="utf-8")
    monkeypatch.setattr(adaptation, "live_timing_processes", lambda: [])
    with adaptation.exclusive_compute_lock(tmp_path / "compute.lock"):
        evidence = adaptation.s84_status_audit(manifest_path)
        assert evidence["completion_required"] is False
        assert evidence["attempt_count"] == 3
    manifest_path.write_text("incomplete write", encoding="utf-8")
    assert adaptation.s84_status_audit(manifest_path)["status"] == "audit_invalid"


def test_s84_audit_targets_append_only_attempt_003(tmp_path, monkeypatch):
    expected = adaptation.ROOT / "artifacts/benchmark_v3/simulator/maestro_threadpool_diagnostic_20261002/attempt_003/run_manifest.json"
    assert adaptation.s84_attempt_manifest_path() == expected

    # Audit the resumed attempt, without treating its completion as permission.
    attempt_dir = tmp_path / "attempt_003"
    attempt_dir.mkdir()
    manifest_path = attempt_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps({
        "status": "partial", "scheduled_attempts": 270,
        "attempt_count": 3,
    }), encoding="utf-8")
    monkeypatch.setattr(adaptation, "s84_attempt_manifest_path", lambda: manifest_path)

    evidence = adaptation.s84_status_audit()
    assert evidence["status"] == "partial"
    assert evidence["attempt_count"] == 3
    assert evidence["completion_required"] is False


def test_real_flock_blocks_active_timing_and_releases_on_error(tmp_path, monkeypatch):
    lock = tmp_path / "shared.lock"
    monkeypatch.setattr(adaptation, "live_timing_processes", lambda: [])
    with lock.open("a+") as timing_holder:
        fcntl.flock(timing_holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="lock is held"):
            with adaptation.exclusive_compute_lock(lock):
                pytest.fail("CUDA body must never start while timing holds the lock")
        fcntl.flock(timing_holder.fileno(), fcntl.LOCK_UN)

    with pytest.raises(RuntimeError, match="body failed"):
        with adaptation.exclusive_compute_lock(lock):
            # The same file is locked throughout activity, including nested folds.
            with adaptation.exclusive_compute_lock(lock):
                with lock.open("a+") as other:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise RuntimeError("body failed")
    with lock.open("a+") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
    with adaptation.GPU_LEASE_PATH.open("a+") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)


def test_gpu_timing_lease_blocks_when_maestro_lock_is_clear(tmp_path, monkeypatch):
    lock = tmp_path / "maestro.lock"
    gpu_lease = adaptation.GPU_LEASE_PATH
    monkeypatch.setattr(adaptation, "live_timing_processes", lambda: [])
    with gpu_lease.open("a+") as timing_holder:
        fcntl.flock(timing_holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="host GPU lease is held"):
            with adaptation.exclusive_compute_lock(lock):
                pytest.fail("CUDA body cannot overlap CUDA-Q timing")
        # The earlier Maestro lock must release even if GPU lease acquisition fails.
        with lock.open("a+") as holder:
            fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        fcntl.flock(timing_holder.fileno(), fcntl.LOCK_UN)

    with adaptation.exclusive_compute_lock(lock):
        with gpu_lease.open("a+") as holder:
            with pytest.raises(BlockingIOError):
                fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_live_controller_or_orphan_worker_blocks_cuda_before_body(tmp_path, monkeypatch):
    monkeypatch.setattr(adaptation, "live_timing_processes", lambda: [{"pid": 123, "kind": "possible_orphan_spawn_timing_worker"}])
    with pytest.raises(ValueError, match="orphan worker"):
        with adaptation.exclusive_compute_lock(tmp_path / "compute.lock"):
            pytest.fail("CUDA body must never start")


def test_process_guard_detects_s84_controller_and_spawn_worker(tmp_path):
    for pid, args in [
        (111, ["python", "/repo/run_maestro_threadpool_intervention.py", "--execute", "--resume"]),
        (112, ["python", "-c", "from multiprocessing.spawn import spawn_main; spawn_main(pipe_handle=7)"]),
        (113, ["python", "-m", "pytest"]),
    ]:
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in args) + b"\0")
    assert {row["pid"] for row in adaptation.live_timing_processes(tmp_path)} == {111, 112}


@pytest.mark.parametrize("action", ["preflight", "fit-five-fold-oof"])
def test_cli_cuda_actions_hold_lock_for_entire_body(tmp_path, monkeypatch, action):
    lock = tmp_path / "compute.lock"
    monkeypatch.setattr(adaptation, "SHARED_TIMING_LOCK", lock)
    monkeypatch.setattr(adaptation, "live_timing_processes", lambda: [])
    calls = []
    def body(output):
        with lock.open("a+") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with adaptation.GPU_LEASE_PATH.open("a+") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        calls.append(output)
    function = "materialize" if action == "preflight" else "run_five_fold_oof"
    monkeypatch.setattr(adaptation, function, adaptation.with_compute_exclusion(body))
    assert adaptation.main(["--action", action, "--output-dir", str(tmp_path / "out")]) == 0
    assert calls == [tmp_path / "out"]


def test_fold0_checkpoint_is_metric_blind_and_checks_exact_rows_predictions_and_environment(tmp_path):
    output = tmp_path
    fold_dir = output / "fold_0"
    fold_dir.mkdir()
    expected_cuda = {
        "python_executable": "/venv/bin/python",
        "torch": "2.7.1+cu128",
        "torch_cuda_runtime": "12.8",
        "torch_geometric": "2.6.1",
        "cuda_available": True,
        "device_count": 1,
        "device_name": "RTX 5070 Ti",
        "device_capability": [12, 0],
        "cuda_visible_devices": "<unset>",
        "pip_freeze_sha256": "a" * 64,
        "pip_freeze_tool": "uv-pip-freeze",
    }
    methods = ["train_fold_median", "ridge_alpha_1", *[f"graph_seed_{s}" for s in adaptation.SEEDS], "graph_median_three_seeds"]
    digest = "hash-fold0"
    prediction = {
        "source_qasm_sha256": digest,
        "fold": 0,
        "target_status": "runtime_observed",
        "target_seconds": "2.0",
        "quality_status_separate_audit_only": "quality_pass",
    }
    prediction.update({f"pred_{method}_seconds": "1.0" for method in methods})
    with (fold_dir / "fold0_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction))
        writer.writeheader()
        writer.writerow(prediction)
    prediction_sha256 = adaptation.sha256_file(fold_dir / "fold0_predictions.csv")
    qa = {
        "status": "PASS",
        "fold": 0,
        "cuda_pyg_smoke": "PASS",
        "train_test_hash_overlap": 0,
        "train_finite_hashes": 1,
        "test_assigned_hashes": 1,
        "test_finite_label_hashes": 1,
        "test_unavailable_hashes": 0,
        "prediction_methods": methods,
        "seed_set": list(adaptation.SEEDS),
        "predictions_cover_all_assigned_test_hashes": True,
        "prediction_sha256": prediction_sha256,
    }
    (fold_dir / "fold0_qa.json").write_text(json.dumps(qa), encoding="utf-8")
    (fold_dir / "environment.json").write_text(json.dumps({"cuda_training_environment": expected_cuda}), encoding="utf-8")
    # Deliberately terrible metric values cannot stop continuation; the technical
    # gate must not even consult this file.
    (fold_dir / "fold0_metrics.json").write_text(json.dumps({"mae_seconds": 1e99, "r2_seconds": -1e99}), encoding="utf-8")

    report = adaptation.validate_fold_integrity(
        output,
        0,
        {digest: 0, "hash-fold1": 1},
        {
            digest: {"target_status": "runtime_observed", "target_seconds": 2.0, "quality_status_separate": "quality_pass"},
            "hash-fold1": {"target_status": "runtime_observed", "target_seconds": 4.0, "quality_status_separate": "quality_pass"},
        },
        expected_cuda,
    )
    assert report["technical_qa"] == "PASS"
    assert report["metrics_quality_consulted"] is False


def test_fold_integrity_stops_on_missing_or_nonfinite_predictions(tmp_path):
    expected_cuda = {
        "python_executable": "/venv/bin/python", "torch": "2.7.1+cu128",
        "torch_cuda_runtime": "12.8", "torch_geometric": "2.6.1",
        "cuda_available": True, "device_count": 1, "device_name": "RTX 5070 Ti",
        "device_capability": [12, 0], "cuda_visible_devices": "<unset>",
        "pip_freeze_sha256": "a" * 64,
        "pip_freeze_tool": "uv-pip-freeze",
    }
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    methods = ["train_fold_median", "ridge_alpha_1", *[f"graph_seed_{s}" for s in adaptation.SEEDS], "graph_median_three_seeds"]
    digest = "hash-fold0"
    row = {"source_qasm_sha256": digest, "fold": 0, "target_status": "runtime_observed", "target_seconds": "2.0", "quality_status_separate_audit_only": "quality_pass"}
    row.update({f"pred_{method}_seconds": "1.0" for method in methods})
    row["pred_graph_seed_1234_seconds"] = "nan"
    with (fold_dir / "fold0_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    prediction_sha256 = adaptation.sha256_file(fold_dir / "fold0_predictions.csv")
    qa = {"status": "PASS", "fold": 0, "cuda_pyg_smoke": "PASS", "train_test_hash_overlap": 0,
          "train_finite_hashes": 1, "test_assigned_hashes": 1, "test_finite_label_hashes": 1,
          "test_unavailable_hashes": 0,
          "prediction_methods": methods, "seed_set": list(adaptation.SEEDS),
          "predictions_cover_all_assigned_test_hashes": True, "prediction_sha256": prediction_sha256}
    (fold_dir / "fold0_qa.json").write_text(json.dumps(qa), encoding="utf-8")
    (fold_dir / "environment.json").write_text(json.dumps({"cuda_training_environment": expected_cuda}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite/negative prediction"):
        adaptation.validate_fold_integrity(
            tmp_path, 0, {digest: 0, "hash-fold1": 1},
            {digest: {"target_status": "runtime_observed", "target_seconds": 2.0, "quality_status_separate": "quality_pass"},
             "hash-fold1": {"target_status": "runtime_observed", "target_seconds": 4.0, "quality_status_separate": "quality_pass"}},
            expected_cuda,
        )
