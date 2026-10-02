from __future__ import annotations

import fcntl
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import run_family_residual_fixed_mps_runtime_v1 as family  # noqa: E402
import run_mps_fixed_chi16_runtime_adaptation_v1 as s85  # noqa: E402


def test_family_aliases_are_connected_components_of_cooccurring_names():
    panel = [
        {"stratum": "core_q2_q9", "qasm_sha256": "h1", "family": "qft"},
        {"stratum": "core_q2_q9", "qasm_sha256": "h1", "family": "qft_alias"},
        {"stratum": "core_q2_q9", "qasm_sha256": "h2", "family": "qft_alias"},
        {"stratum": "core_q2_q9", "qasm_sha256": "h2", "family": "fourier_variant"},
        {"stratum": "core_q2_q9", "qasm_sha256": "h3", "family": "grover"},
        {"stratum": "frontier", "qasm_sha256": "outside", "family": "not_in_core"},
    ]

    component_by_hash, aliases_by_hash = family.family_alias_components(panel)

    assert component_by_hash["h1"] == component_by_hash["h2"]
    assert json.loads(component_by_hash["h1"]) == ["fourier_variant", "qft", "qft_alias"]
    assert component_by_hash["h3"] != component_by_hash["h1"]
    assert aliases_by_hash["h1"] == ["qft", "qft_alias"]
    assert "outside" not in repr(component_by_hash)


def test_groupkfold_crossfit_partitions_every_exact_hash_once_without_overlap():
    hashes = [f"hash-{index:02d}" for index in range(12)]

    splits = family.groupkfold_hash_splits(hashes, n_splits=4)

    predicted = []
    for split in splits:
        train = set(split["train_hashes"])
        held_out = set(split["predicted_hashes"])
        assert train.isdisjoint(held_out)
        assert train | held_out == set(hashes)
        predicted.extend(split["predicted_hashes"])
    assert len(predicted) == len(set(predicted)) == len(hashes)
    with pytest.raises(ValueError, match="unique hashes sorted"):
        family.groupkfold_hash_splits(["hash-b", "hash-a"], n_splits=2)


def test_feature_transform_is_log1p_and_uses_only_provided_training_rows():
    logged_train = np.log1p(np.asarray([[0.0, 2, 4, 8, 16, 32], [0.0, 6, 8, 10, 20, 64]]))
    logged_test = np.log1p(np.asarray([[100.0, 100, 100, 100, 100, 100]]))

    mean, scale = family.fit_feature_transform(logged_train)
    transformed_train = family.transform_features(logged_train, mean, scale)
    transformed_test = family.transform_features(logged_test, mean, scale)

    assert mean[0] == 0.0
    assert scale[0] == 1.0  # zero variance becomes unit scale
    assert np.allclose(transformed_train.mean(axis=0), 0.0, atol=1e-6)
    assert transformed_test[0, 0] == pytest.approx(np.log1p(100.0))  # fixed unit scale from training
    assert transformed_test[0, 1] > 5.0  # test values did not affect the training scaler


def test_runtime_unknown_family_is_exact_family_agnostic_bypass_and_shared_weights_match():
    import torch

    torch.manual_seed(17)
    SharedCore, FamilyRuntime, FamilyAgnostic = family._runtime_model_classes(torch)
    core = SharedCore()
    family_model = FamilyRuntime(core, class_count=3).eval()
    ablation_model = FamilyAgnostic(core).eval()

    assert family.shared_state_digest(core.shared_state()) == family.shared_state_digest(family_model.shared_state())
    assert family.shared_state_digest(core.shared_state()) == family.shared_state_digest(ablation_model.shared_state())
    assert torch.count_nonzero(family_model.family_residual.weight) == 0
    assert torch.count_nonzero(family_model.family_residual.bias) == 0
    x = torch.randn((5, 6), dtype=torch.float32)
    unknown = torch.full((5,), 3, dtype=torch.long)
    with torch.no_grad():
        expected = ablation_model(x)
        actual = family_model(x, unknown)
    assert torch.equal(actual, expected)

    ablation_before = {name: value.detach().clone() for name, value in ablation_model.shared_state().items()}
    optimizer = torch.optim.SGD(family_model.parameters(), lr=0.01)
    family_model.train()
    optimizer.zero_grad(set_to_none=True)
    training_ids = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    family_model(x, training_ids).sum().backward()
    optimizer.step()
    assert all(torch.equal(ablation_model.shared_state()[name], value) for name, value in ablation_before.items())


def test_registered_paired_hash_bootstrap_is_reproducible_and_uses_10000_replicates():
    registry = json.loads(family.SEED_REGISTRY.read_text(encoding="utf-8"))
    hashes = ["hash-a", "hash-b", "hash-c"]
    deltas = [-1.0, 0.5, 2.0]

    first = family.paired_hash_bootstrap(hashes, deltas, "finite|candidate|reference|mae_seconds", registry)
    second = family.paired_hash_bootstrap(hashes, deltas, "finite|candidate|reference|mae_seconds", registry)

    assert first == second
    assert first["replicates"] == 10_000
    assert first["bootstrap_groups"] == 3
    assert first["bootstrap_group_column"] == "source_qasm_sha256"
    assert first["observed_mae_difference_seconds"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="frozen at 10,000"):
        family.paired_hash_bootstrap(hashes, deltas, "finite|candidate|reference|mae_seconds", registry, replicates=999)


def test_cuda_path_reuses_s85_helper_and_acquires_both_nonblocking_locks(tmp_path, monkeypatch):
    timing_lock = tmp_path / "maestro.lock"
    gpu_lease = tmp_path / "gpu.lock"
    monkeypatch.setattr(s85, "SHARED_TIMING_LOCK", timing_lock)
    monkeypatch.setattr(s85, "GPU_LEASE_PATH", gpu_lease)
    monkeypatch.setattr(s85, "live_timing_processes", lambda: [])

    with s85.exclusive_compute_lock():
        assert timing_lock.exists()
        assert gpu_lease.exists()
        with gpu_lease.open("a+") as holder:
            with pytest.raises(BlockingIOError):
                fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    assert s85._COMPUTE_LOCK_OWNER_PID is None


def test_cuda_environment_parity_uses_s85_package_freeze_helper_and_tool(tmp_path, monkeypatch):
    import os
    import torch

    frozen_bytes = b"example-package==1.2.3\n"
    expected = {
        "python_executable": sys.executable,
        "torch": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda),
        "torch_geometric": "2.6.1",
        "cuda_available": True,
        "device_count": 1,
        "device_name": "test-device",
        "device_capability": [9, 0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "pip_freeze_sha256": family.sha256_bytes(frozen_bytes),
        "pip_freeze_tool": "uv-pip-freeze",
    }
    helper_calls = []
    validated_environments = []
    monkeypatch.setattr(s85, "cuda_training_environment", lambda: dict(expected))
    monkeypatch.setattr(s85, "package_freeze_snapshot", lambda: (helper_calls.append(True) or frozen_bytes, "uv-pip-freeze"))
    real_validate = s85.validate_cuda_environment_match

    def validate_spy(expected_environment, actual_environment):
        validated_environments.append(dict(actual_environment))
        return real_validate(expected_environment, actual_environment)

    monkeypatch.setattr(s85, "validate_cuda_environment_match", validate_spy)
    monkeypatch.setattr(family, "cuda_smoke_test", lambda _torch: {"status": "PASS"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "test-device")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (9, 0))
    output = tmp_path / "locked-setup"
    output.mkdir()

    actual, smoke = family._locked_cuda_setup(
        output,
        {"protocol_sha256": "protocol-hash", "input_sha256": {"input": "hash"}},
    )

    assert helper_calls == [True]
    assert actual == expected
    assert smoke == {"status": "PASS"}
    assert validated_environments[0]["pip_freeze_tool"] == "uv-pip-freeze"
    assert validated_environments[0]["pip_freeze_sha256"] == family.sha256_bytes(frozen_bytes)
    recorded = json.loads((output / "execution_environment.json").read_text(encoding="utf-8"))
    assert recorded["cuda_training_environment"]["pip_freeze_tool"] == "uv-pip-freeze"


def test_static_materialization_does_not_start_cuda_or_training(tmp_path):
    output = tmp_path / "preflight"
    family.materialize(output)

    manifest = json.loads((output / "materialization_manifest.json").read_text(encoding="utf-8"))
    table = family.read_csv(output / "hash_targets_and_static_features.csv")
    qa = family.validate_materialization(output)

    assert manifest["cuda_probe_status"] == "not_run_by_static_materialize_or_preflight"
    assert manifest["training_performed"] is False
    assert len(table) == 150
    assert sum(row["target_status"] == "runtime_observed" for row in table) == 144
    assert sum(row["target_status"] != "runtime_observed" for row in table) == 6
    assert qa["status"] == "PASS"
    assert qa["cuda_started"] is False
