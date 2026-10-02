from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import run_azizov_common_core_adaptation_v1 as adaptation  # noqa: E402


def _graph_record(digest: str, *, legacy: bool) -> dict:
    if legacy:
        nodes = [
            {"node_id": 0, "operation": "h", "qargs": [2], "cargs": [], "params_text": []},
            {"node_id": 1, "operation": "rx", "qargs": [2], "cargs": [], "params_text": [str(math.pi)]},
            {"node_id": 2, "operation": "measure", "qargs": [2], "cargs": [0], "params_text": []},
        ]
        edges = [
            {"source_node": 0, "target_node": 1, "wire_kind": "qubit", "wire_index": 2},
            {"source_node": 1, "target_node": 2, "wire_kind": "qubit", "wire_index": 2},
        ]
        return {
            "source_qasm_sha256": digest,
            "num_qubits": 3,
            "num_clbits": 1,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }
    nodes = [
        {
            "node_index": 0, "operation": "h", "qargs": [2], "cargs": [],
            "parameter_values_over_pi": [0.0, 0.0, 0.0, 0.0],
            "parameter_presence_masks": [0, 0, 0, 0],
            "quantum_arity_normalized": 1 / 127,
            "classical_arity_normalized": 0.0,
            "instruction_index_normalized": 0.0,
        },
        {
            "node_index": 1, "operation": "rx", "qargs": [2], "cargs": [],
            "parameter_values_over_pi": [1.0, 0.0, 0.0, 0.0],
            "parameter_presence_masks": [1, 0, 0, 0],
            "quantum_arity_normalized": 1 / 127,
            "classical_arity_normalized": 0.0,
            "instruction_index_normalized": 0.5,
        },
        {
            "node_index": 2, "operation": "measure", "qargs": [2], "cargs": [0],
            "parameter_values_over_pi": [0.0, 0.0, 0.0, 0.0],
            "parameter_presence_masks": [0, 0, 0, 0],
            "quantum_arity_normalized": 1 / 127,
            "classical_arity_normalized": 1 / 127,
            "instruction_index_normalized": 1.0,
        },
    ]
    return {
        "schema_version": "azizov_raw_graph_v1",
        "source_sha256": digest,
        "representation": "transpiled_graph",
        "num_qubits": 3,
        "num_clbits": 1,
        "node_count": len(nodes),
        "edge_count": 2,
        "nodes": nodes,
        "edges": [[0, 1], [1, 2]],
    }


def _small_bundle() -> adaptation.InputBundle:
    members = [
        {"panel_member_id": "m-a", "circuit_id": "a", "source_sha256": "a" * 64, "fold": 0, "observed_seconds": 1.0},
        {"panel_member_id": "m-a-alias", "circuit_id": "a2", "source_sha256": "a" * 64, "fold": 0, "observed_seconds": 3.0},
        {"panel_member_id": "m-b", "circuit_id": "b", "source_sha256": "b" * 64, "fold": 1, "observed_seconds": 2.0},
    ]
    hashes = {
        "a" * 64: {"source_sha256": "a" * 64, "fold": 0, "members": members[:2], "member_ids": ["m-a", "m-a-alias"], "target_seconds": 2.0},
        "b" * 64: {"source_sha256": "b" * 64, "fold": 1, "members": members[2:], "member_ids": ["m-b"], "target_seconds": 2.0},
    }
    return adaptation.InputBundle(
        artifact_dir=Path("."),
        protocol={},
        dictionary={},
        members=members,
        hashes=hashes,
        global_names={view: ("x",) for view in adaptation.VIEW_NAMES},
        global_values={view: {digest: (1.0,) for digest in hashes} for view in adaptation.VIEW_NAMES},
        graphs={"source": {}, "transpiled": {}},
        representation_status={"a" * 64: "available", "b" * 64: "unavailable"},
        input_hashes={},
        source_hashes_sha256="source-hash",
        representation_manifest_sha256="representation-hash",
    )


def test_inner_validation_split_is_hash_sorted_and_exact_twenty_percent():
    hashes = [f"{index:064x}" for index in range(11)]
    train, validation = adaptation.inner_validation_hashes(hashes)
    train_reversed, validation_reversed = adaptation.inner_validation_hashes(reversed(hashes))

    assert train == train_reversed
    assert validation == validation_reversed
    assert len(validation) == math.ceil(0.2 * len(hashes))
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(hashes)
    expected = sorted(hashes, key=lambda digest: __import__("hashlib").sha256(("azizov-inner-1234|" + digest).encode("ascii")).hexdigest())
    assert validation == expected[: math.ceil(0.2 * len(hashes))]


def test_train_only_standardizer_removes_constants_and_uses_population_scale():
    train = np.asarray([[1.0, 7.0, 2.0], [3.0, 7.0, 6.0]])
    scaler = adaptation.fit_global_scaler(train, ["vary_a", "constant", "vary_b"])
    transformed = scaler.transform([[1.0, 7.0, 2.0], [3.0, 7.0, 6.0], [1000.0, 7.0, 10.0]])

    assert scaler.removed_zero_variance_names == ("constant",)
    assert scaler.means == (2.0, 4.0)
    assert scaler.scales == (1.0, 2.0)
    assert transformed[0].tolist() == [-1.0, -1.0]
    assert transformed[1].tolist() == [1.0, 1.0]
    assert transformed[2, 0] == 998.0  # held-out extremes do not refit the train scaler


def test_all_constant_globals_become_one_explicit_zero_column():
    scaler = adaptation.fit_global_scaler([[2.0, 2.0], [2.0, 2.0]], ["a", "b"])
    assert scaler.all_constant
    assert scaler.retained_names == ("__all_constant_zero__",)
    assert scaler.transform([[2.0, 2.0], [200.0, 200.0]]).tolist() == [[0.0], [0.0]]


def test_graph_adapter_accepts_legacy_source_and_raw_transpiled_schemas():
    digest = "a" * 64
    source = adaptation.normalize_graph_record(_graph_record(digest, legacy=True))
    transpiled = adaptation.normalize_graph_record(_graph_record(digest, legacy=False))

    assert source.edges == ((0, 1), (1, 2))
    assert source.edges == transpiled.edges
    assert source.nodes[1]["parameter_values_over_pi"] == [1.0, 0.0, 0.0, 0.0]
    assert source.nodes[1]["parameter_presence_masks"] == [1.0, 0.0, 0.0, 0.0]
    assert source.nodes == transpiled.nodes


def test_graph_adapter_rejects_edge_leak_or_wrong_parameter_shape():
    record = _graph_record("a" * 64, legacy=False)
    record["edges"] = [[0, 2]]
    with pytest.raises(ValueError, match="edge list differs"):
        adaptation.normalize_graph_record(record)

    record = _graph_record("b" * 64, legacy=False)
    record["nodes"][1]["parameter_presence_masks"] = [1, 0]
    with pytest.raises(ValueError, match="parameter layout"):
        adaptation.normalize_graph_record(record)


def test_gate_vocabulary_is_train_only_and_unknown_operations_use_reserved_zero():
    train_graph = adaptation.normalize_graph_record(_graph_record("a" * 64, legacy=True))
    heldout_graph = adaptation.normalize_graph_record(_graph_record("b" * 64, legacy=True))
    heldout_graph = adaptation.NormalizedGraph(
        source_sha256=heldout_graph.source_sha256,
        num_qubits=heldout_graph.num_qubits,
        num_clbits=heldout_graph.num_clbits,
        nodes=tuple({**node, "operation": "heldout_gate" if i == 0 else node["operation"]} for i, node in enumerate(heldout_graph.nodes)),
        edges=heldout_graph.edges,
    )
    vocabulary = adaptation.fit_gate_vocabulary(
        {"source": {"a" * 64: train_graph, "b" * 64: heldout_graph}, "transpiled": {}},
        ["a" * 64],
    )
    encoded, edge_index, unknown_count = adaptation.encode_graph(heldout_graph, vocabulary)

    assert "heldout_gate" not in vocabulary
    assert vocabulary[adaptation.UNKNOWN_GATE] == 0
    assert unknown_count == 1
    assert encoded.shape == (3, len(vocabulary) + 265)
    assert encoded[0, 0] == 1.0
    assert edge_index.tolist() == [[0, 1], [1, 2]]


def test_protocol_result_family_count_is_nine_neural_plus_fifteen_classical():
    methods = adaptation.method_metadata()
    neural = [row for row in methods.values() if row["method_family"] == "azizov_gnn"]
    classical = [row for row in methods.values() if row["method_family"].startswith("classical_")]
    assert len(neural) == 3 * 3
    assert len(classical) == 5 * 3
    assert len(methods) == 24
    assert {row["seed"] for row in neural} == {42, 1234, 31415}
    assert {row["model"] for row in classical} == set(adaptation.CLASSICAL_NAMES)


def test_terminal_ledger_accounts_for_every_member_method_and_retains_unavailable():
    bundle = _small_bundle()
    ledger = adaptation.build_terminal_ledger(bundle, [])
    expected = len(bundle.members) * len(adaptation.METHOD_IDS)

    assert len(ledger) == expected
    assert len({(row["method_id"], row["panel_member_id"]) for row in ledger}) == expected
    unavailable = [row for row in ledger if row["source_sha256"] == "b" * 64]
    assert len(unavailable) == len(adaptation.METHOD_IDS)
    assert {row["status"] for row in unavailable} == {"unavailable_representation"}
    assert len([row for row in ledger if row["source_sha256"] == "a" * 64]) == 2 * len(adaptation.METHOD_IDS)


def test_fold0_gate_is_technical_only_and_requires_every_available_hash():
    bundle = _small_bundle()
    rows = []
    for method in adaptation.method_metadata().values():
        rows.extend(adaptation.terminal_rows_for_hashes(bundle, method, ["a" * 64], "technical_failure", "", {"a" * 64: 0.5}))
    qa = {"method_audit": {method_id: {"status": "PASS"} for method_id in adaptation.METHOD_IDS}}

    passed, reasons = adaptation.fold0_technical_gate(bundle, rows, qa)
    assert passed, reasons

    broken = [dict(row) for row in rows]
    broken[0]["status"] = "technical_failure"
    passed, reasons = adaptation.fold0_technical_gate(bundle, broken, qa)
    assert not passed
    assert any("missing finite prediction" in reason for reason in reasons)


def test_three_seed_median_is_created_only_from_all_three_successful_seeds():
    bundle = _small_bundle()
    rows = []
    methods = adaptation.method_metadata()
    for seed, prediction in zip(adaptation.NEURAL_SEEDS, (1.0, 3.0, 2.0)):
        method = methods[f"azizov_gnn_source_seed_{seed}"]
        rows.extend(adaptation.terminal_rows_for_hashes(bundle, method, ["a" * 64], "technical_failure", "", {"a" * 64: prediction}))
        rows.extend(adaptation.terminal_rows_for_hashes(bundle, method, ["b" * 64], "unavailable_representation", "E2 unavailable"))
    medians = adaptation._median_prediction_rows(bundle, rows)

    available = [row for row in medians if row["source_sha256"] == "a" * 64]
    unavailable = [row for row in medians if row["source_sha256"] == "b" * 64]
    assert len(available) == 2  # aliases expand the one exact-hash median
    assert {row["prediction_seconds"] for row in available} == {2.0}
    assert {row["successful_seed_count"] for row in available} == {3}
    assert {row["status"] for row in unavailable} == {"unavailable_representation"}

    partial = [row for row in rows if row["seed"] != 31415]
    partial_median = adaptation._median_prediction_rows(bundle, partial)
    assert {row["status"] for row in partial_median if row["source_sha256"] == "a" * 64} == {"incomplete_seed_set"}


def test_status_ledger_writer_includes_seed_count_and_blanks_non_median_rows(tmp_path):
    bundle = _small_bundle()
    seed_rows = []
    methods = adaptation.method_metadata()
    for seed, prediction in zip(adaptation.NEURAL_SEEDS, (1.0, 3.0, 2.0)):
        seed_rows.extend(adaptation.terminal_rows_for_hashes(
            bundle,
            methods[f"azizov_gnn_source_seed_{seed}"],
            ["a" * 64],
            "technical_failure",
            "",
            {"a" * 64: prediction},
        ))
    median_rows = [{**row, "seed": "median"} for row in adaptation._median_prediction_rows(bundle, seed_rows)]
    output_path = tmp_path / "coverage_and_terminal_statuses.csv"

    adaptation.write_csv(output_path, adaptation.STATUS_FIELDS, seed_rows + median_rows)

    fields, written = adaptation.read_csv(output_path)
    assert "successful_seed_count" in fields
    per_seed_rows = [row for row in written if row["seed"] != "median"]
    written_median_rows = [row for row in written if row["seed"] == "median"]
    assert per_seed_rows and all(row["successful_seed_count"] == "" for row in per_seed_rows)
    assert written_median_rows and {row["successful_seed_count"] for row in written_median_rows} == {"3"}


def test_primary_metrics_use_one_median_target_per_hash_and_member_metrics_stay_separate():
    bundle = _small_bundle()
    method = {
        "method_id": "classical_ridge_source",
        "method_family": "classical_ridge",
        "view": "source",
        "seed": 1234,
    }
    rows = adaptation.terminal_rows_for_hashes(
        bundle,
        method,
        ["a" * 64, "b" * 64],
        "technical_failure",
        "",
        {"a" * 64: 1.0, "b" * 64: 2.0},
    )
    metrics, pairs = adaptation._compute_metrics(bundle, rows)

    assert pairs == []
    assert metrics[0]["mae_seconds"] == 0.5  # hash a's protocol target is median(1, 3) = 2
    assert metrics[0]["member_mae_seconds"] == pytest.approx(2.0 / 3.0)


def test_runtime_lock_version_map_is_exact_and_gpu_independent():
    expected = {
        "expected_versions": {
            "python_version": "3.10.21",
            "numpy": "1.26.4",
            "scikit-learn": "1.7.2",
            "xgboost": "2.1.4",
            "torch": "2.7.1+cu128",
            "torch-geometric": "2.6.1",
            "cuda_runtime": "12.8",
        }
    }
    actual = dict(expected["expected_versions"])
    assert adaptation.runtime_version_mismatches(expected, actual) == []
    actual["torch"] = "2.7.1+cpu"
    assert adaptation.runtime_version_mismatches(expected, actual) == [
        "torch: expected 2.7.1+cu128, actual 2.7.1+cpu"
    ]


def test_runtime_lock_document_pins_versions_overlay_and_cuda_base_without_gpu():
    versions = {
        "python_version": "3.10.21",
        "numpy": "1.26.4",
        "scikit-learn": "1.7.2",
        "xgboost": "2.1.4",
        "torch": "2.7.1+cu128",
        "torch-geometric": "2.6.1",
        "cuda_runtime": "12.8",
    }
    lock = {
        "schema_version": adaptation.RUNTIME_LOCK_SCHEMA,
        "expected_versions": versions,
        "overlay_sha256": "a" * 64,
        "overlay_pip_freeze_sha256": "b" * 64,
        "base_cuda_lock_sha256": "c" * 64,
    }
    assert adaptation.validate_runtime_lock_document(
        lock,
        actual_versions=versions,
        overlay_sha256="a" * 64,
        overlay_freeze_sha256="b" * 64,
        base_cuda_lock_sha256="c" * 64,
    ) == []

    mismatches = adaptation.validate_runtime_lock_document(
        lock,
        actual_versions=versions,
        overlay_sha256="d" * 64,
        overlay_freeze_sha256="e" * 64,
        base_cuda_lock_sha256="f" * 64,
    )
    assert any(item.startswith("overlay_sha256:") for item in mismatches)
    assert any(item.startswith("overlay_pip_freeze_sha256:") for item in mismatches)
    assert any(item.startswith("base_cuda_lock_sha256:") for item in mismatches)


def test_runtime_lock_rejects_missing_schema_and_versions():
    mismatches = adaptation.validate_runtime_lock_document(
        {"expected_versions": {"numpy": "1.26.4"}},
        actual_versions={"numpy": "1.26.4"},
        overlay_sha256=None,
        overlay_freeze_sha256=None,
        base_cuda_lock_sha256="c" * 64,
    )
    assert any("schema_version" in item for item in mismatches)
    assert any("expected_versions missing required fields" in item for item in mismatches)
    assert any("must pin base_cuda_lock_sha256" in item for item in mismatches)


def test_pip_freeze_falls_back_to_uv_and_records_overlay_freeze(tmp_path, monkeypatch):
    overlay = tmp_path / "site-packages"
    overlay.mkdir()
    calls = []

    def fake_run(command, capture_output):
        calls.append(command)
        if "-m" in command and "pip" in command:
            return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"pip unavailable")
        if "--python" in command:
            return subprocess.CompletedProcess(command, 0, stdout=b"numpy==1.26.4\n", stderr=b"")
        if "--target" in command:
            return subprocess.CompletedProcess(command, 0, stdout=b"scikit-learn==1.7.2\n", stderr=b"")
        raise AssertionError(f"unexpected freeze command: {command}")

    monkeypatch.setattr(adaptation.subprocess, "run", fake_run)
    monkeypatch.setattr(adaptation.shutil, "which", lambda executable: "/fake/bin/uv" if executable == "uv" else None)

    environment, environment_tool, overlay_bytes, overlay_tool = adaptation.package_freeze_snapshot(overlay)

    assert environment == b"numpy==1.26.4\n"
    assert environment_tool == "uv-pip-freeze-python"
    assert overlay_bytes == b"scikit-learn==1.7.2\n"
    assert overlay_tool == "uv-pip-freeze-target"
    assert calls[1] == ["/fake/bin/uv", "pip", "freeze", "--python", adaptation.sys.executable]
    assert calls[2] == ["/fake/bin/uv", "pip", "freeze", "--target", str(overlay.resolve())]


def test_activation_puts_runtime_overlay_first_on_import_path(tmp_path, monkeypatch):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    old_path = list(adaptation.sys.path)
    monkeypatch.setattr(adaptation.sys, "path", old_path.copy())

    adaptation.activate_overlay(overlay)

    assert adaptation.sys.path[0] == str(overlay.resolve())


def test_overlay_tree_hash_is_stable_and_content_sensitive(tmp_path):
    first = tmp_path / "overlay"
    first.mkdir()
    (first / "package.dist-info").mkdir()
    (first / "package.dist-info" / "METADATA").write_text("Name: demo\nVersion: 1\n", encoding="utf-8")
    before = adaptation.sha256_tree(first)
    assert adaptation.sha256_tree(first) == before
    (first / "package.dist-info" / "METADATA").write_text("Name: demo\nVersion: 2\n", encoding="utf-8")
    assert adaptation.sha256_tree(first) != before


def test_e2_provenance_outputs_resolve_from_artifact_root_and_reject_escape(tmp_path):
    artifact = tmp_path / "azizov"
    nested = artifact / "compiled_qasm"
    nested.mkdir(parents=True)
    output = nested / "circuit.qasm"
    output.write_text("OPENQASM 2.0;\n", encoding="utf-8")
    manifest = {
        "outputs": {
            "compiled_qasm/circuit.qasm": {
                "sha256": adaptation.sha256_file(output),
                "bytes": output.stat().st_size,
            }
        }
    }

    adaptation._verify_provenance_outputs(artifact, manifest)
    manifest["outputs"] = {"../escape.qasm": {"sha256": "0" * 64}}
    with pytest.raises(ValueError, match="escapes its artifact root"):
        adaptation._verify_provenance_outputs(artifact, manifest)
