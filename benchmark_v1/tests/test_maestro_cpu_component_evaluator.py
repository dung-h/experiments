from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import evaluate_maestro_cpu_component_v1 as evaluator  # noqa: E402


def synthetic_context():
    fold_by_hash = {}
    targets = []
    features = {}
    counts = {0: 37, 1: 28, 2: 26, 3: 25, 4: 34}
    ordinal = 0
    for fold, count in counts.items():
        for _ in range(count):
            digest = hashlib.sha256(f"hash-{ordinal}".encode()).hexdigest()
            fold_by_hash[digest] = fold
            width = 2 + ordinal % 8
            features[digest] = {
                "globals": {
                    "active_width": width, "measurement_stripped_structural_depth": 3 + ordinal % 21,
                    "one_qubit_gate_count": 2 + ordinal % 15, "two_qubit_gate_count": 1 + ordinal % 9,
                    "multi_qubit_gate_count": 0, "swap_like_gate_count": ordinal % 3,
                },
                "graph": {"num_qubits": width, "nodes": [], "edge_src": [], "edge_dst": []},
            }
            runtime = 0.01 + 0.0002 * ordinal + 0.00001 * (ordinal % 7)
            status = "unavailable_timeout" if ordinal in {3, 49, 88, 120, 143} else "runtime_observed"
            targets.append({"source_qasm_sha256": digest, "outer_fold": str(fold),
                            "runtime_seconds": "" if status != "runtime_observed" else str(runtime),
                            "target_status": status, "normalized_qasm_sha256": "",
                            "normalized_width": "", "normalized_1q_gate_count": "",
                            "normalized_cx_gate_count": ""})
            ordinal += 1
    return fold_by_hash, targets, features


def test_target_contract_keeps_all_hashes_and_frozen_folds():
    folds, rows, _ = synthetic_context()
    targets = evaluator.validate_panel_targets(rows, folds)
    assert len(targets) == 150
    assert sum(value["runtime_seconds"] is None for value in targets.values()) == 5
    assert {fold: sum(value["fold"] == fold for value in targets.values()) for fold in range(5)} == {
        0: 37, 1: 28, 2: 26, 3: 25, 4: 34,
    }


def test_pinned_source_dag_inputs_are_materialized_for_full_panel():
    folds, graphs, features = evaluator.load_frozen_context()
    assert len(folds) == len(graphs) == len(features) == 150
    assert set(folds) == set(graphs) == set(features)


def test_target_contract_rejects_duplicates_fold_drift_and_fabricated_unavailable_label():
    folds, rows, _ = synthetic_context()
    with pytest.raises(ValueError, match="duplicate"):
        evaluator.validate_panel_targets(rows + [rows[0]], folds)
    changed = [dict(row) for row in rows]
    changed[0]["outer_fold"] = str((int(changed[0]["outer_fold"]) + 1) % 5)
    with pytest.raises(ValueError, match="fold"):
        evaluator.validate_panel_targets(changed, folds)
    changed = [dict(row) for row in rows]
    unavailable = next(row for row in changed if row["target_status"] != "runtime_observed")
    unavailable["runtime_seconds"] = "0.5"
    with pytest.raises(ValueError, match="must not carry"):
        evaluator.validate_panel_targets(changed, folds)
    changed = [dict(row) for row in rows]
    changed[0]["target_status"] = "made_up_status"
    with pytest.raises(ValueError, match="unknown target_status"):
        evaluator.validate_panel_targets(changed, folds)


def test_nested_ridge_uses_exact_hash_inner_folds_and_not_outer_test_targets():
    folds, rows, features = synthetic_context()
    targets = evaluator.validate_panel_targets(rows, folds)
    fold = 2
    pred_a, alpha_a, scores_a = evaluator.nested_ridge_fold(fold, targets, folds, features)
    assert len(pred_a) == 26
    assert alpha_a in evaluator.ALPHAS
    assert set(scores_a) == {f"{value:g}" for value in evaluator.ALPHAS}
    altered = {key: dict(value) for key, value in targets.items()}
    for digest, value in altered.items():
        if value["fold"] == fold:
            value["runtime_seconds"] = 1e6
    pred_b, alpha_b, scores_b = evaluator.nested_ridge_fold(fold, altered, folds, features)
    assert alpha_a == alpha_b
    assert scores_a == scores_b
    assert pred_a == pred_b


def test_component_calibrator_is_optional_and_uses_only_valid_width_knots():
    folds, rows, features = synthetic_context()
    targets = evaluator.validate_panel_targets(rows, folds)
    digest = next(iter(targets))
    absent = evaluator.component_predictions(targets, features, None, 1000)
    assert absent[digest] == (None, "calibration_model_not_provided")
    width = features[digest]["graph"]["num_qubits"]
    model = {width: {"status": "valid", "C1q": 0.01, "Ccx": 0.02, "b": 0.003, "m": 0.0001}}
    pred, status = evaluator.component_predictions(targets, features, model, 1000)[digest]
    assert (pred, status) == (None, "normalized_execution_metadata_unavailable")
    raw = next(row for row in rows if row["source_qasm_sha256"] == digest)
    raw.update({"normalized_qasm_sha256": hashlib.sha256(b"normalized-qasm").hexdigest(),
                "normalized_width": str(width), "normalized_1q_gate_count": "8",
                "normalized_cx_gate_count": "3"})
    normalized_targets = evaluator.validate_panel_targets(rows, folds)
    pred, status = evaluator.component_predictions(normalized_targets, features, model, 1000)[digest]
    expected = (2**width * (0.01 * 8 + 0.02 * 3) + 0.003 + 0.0001 * 1000)
    assert status == "available"
    assert pred == pytest.approx(expected)
    assert evaluator.component_predictions(normalized_targets, features,
                                           {width: {"status": "unavailable"}}, 1000)[digest] == (
        None, "invalid_or_missing_width_knot")


def test_component_uses_normalized_gate_counts_not_source_dag_globals_and_checks_width():
    folds, rows, features = synthetic_context()
    digest = rows[0]["source_qasm_sha256"]
    width = features[digest]["graph"]["num_qubits"]
    row = next(row for row in rows if row["source_qasm_sha256"] == digest)
    row.update({"normalized_qasm_sha256": hashlib.sha256(b"normalized-qasm-2").hexdigest(),
                "normalized_width": str(width), "normalized_1q_gate_count": "24",
                "normalized_cx_gate_count": "7"})
    targets = evaluator.validate_panel_targets(rows, folds)
    model = {width: {"status": "valid", "C1q": .1, "Ccx": .2, "b": .3, "m": 0.0}}
    before = evaluator.component_predictions(targets, features, model, 1000)[digest]
    features[digest]["globals"]["one_qubit_gate_count"] = 999999
    features[digest]["globals"]["two_qubit_gate_count"] = 999999
    after = evaluator.component_predictions(targets, features, model, 1000)[digest]
    assert before == after == (2**width * (.1 * 24 + .2 * 7) + .3, "available")
    targets[digest]["normalized_width"] = width + 1
    with pytest.raises(ValueError, match="disagrees with source-DAG width"):
        evaluator.component_predictions(targets, features, model, 1000)


def test_component_model_import_rejects_panel_overlap_and_bad_coefficients(tmp_path):
    folds, rows, _ = synthetic_context()
    target_hashes = set(folds)
    expected_hashes = evaluator.expected_calibration_hashes()
    assert len(expected_hashes) == 112
    valid = {"schema_id": "maestro-cpu-component-calibration-v1", "artifact_status": "PASS",
             "status": "complete",
             "protocol_sha256": evaluator.sha256_file(evaluator.PROTOCOL),
             "protocol_id": json.loads(evaluator.PROTOCOL.read_text())["protocol_id"],
             "optimizer_enabled_requested": None,
             "panel_targets_sha256": "e" * 64,
             "shot_count": 1000, "calibration_source_qasm_sha256": expected_hashes,
             "input_attempt_sha256": ["c" * 64], "attempts_sha256": "c" * 64,
             "coefficient_formula": "2^n*(C1q(n)*normalized_1q_count + Ccx(n)*normalized_cx_count) + b(n) + m(n)*shots",
             "width_coefficients": [{"width": n, "status": "valid", "C1q": .1,
                                     "Ccx": .1, "b": .1, "m": .1} for n in range(2, 10)]}
    path = tmp_path / "calibration_model.json"
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert set(evaluator.load_component_model(path, target_hashes,
                                               expected_panel_targets_sha256="e" * 64)) == set(range(2, 10))
    with pytest.raises(ValueError, match="panel_targets_sha256"):
        evaluator.load_component_model(path, target_hashes,
                                       expected_panel_targets_sha256="f" * 64)
    overlapped = dict(valid, calibration_source_qasm_sha256=[next(iter(target_hashes))] + expected_hashes[1:])
    path.write_text(json.dumps(overlapped), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        evaluator.load_component_model(path, target_hashes,
                                       expected_panel_targets_sha256="e" * 64)
    bad_hashes = dict(valid, calibration_source_qasm_sha256=["b" * 64] + expected_hashes[1:])
    path.write_text(json.dumps(bad_hashes), encoding="utf-8")
    with pytest.raises(ValueError, match="exact 112 synthetic"):
        evaluator.load_component_model(path, target_hashes,
                                       expected_panel_targets_sha256="e" * 64)
    bad = dict(valid, width_coefficients=[dict(row, C1q=-1) if row["width"] == 2 else row
                                         for row in valid["width_coefficients"]])
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid calibration"):
        evaluator.load_component_model(path, target_hashes,
                                       expected_panel_targets_sha256="e" * 64)


def test_partial_calibration_is_integrity_pass_and_preserves_unavailable_width(tmp_path):
    hashes = evaluator.expected_calibration_hashes()
    partial = {"schema_id": "maestro-cpu-component-calibration-v1", "artifact_status": "PASS",
               "status": "complete_with_unavailable_knots", "shot_count": 1000,
               "protocol_sha256": evaluator.sha256_file(evaluator.PROTOCOL),
               "protocol_id": json.loads(evaluator.PROTOCOL.read_text())["protocol_id"],
               "optimizer_enabled_requested": None,
               "panel_targets_sha256": "e" * 64,
               "coefficient_formula": (
                   "2^n*(C1q(n)*normalized_1q_count + Ccx(n)*normalized_cx_count) + b(n) + m(n)*shots"),
               "input_attempt_sha256": ["d" * 64], "attempts_sha256": "d" * 64,
               "calibration_source_qasm_sha256": hashes,
               "width_coefficients": [
                   ({"width": width, "status": "unavailable", "unavailable_reason": "unstable_fit"}
                    if width == 3 else
                    {"width": width, "status": "valid", "C1q": 0.1, "Ccx": 0.1, "b": 0.1, "m": 0.1})
                   for width in range(2, 10)]}
    path = tmp_path / "partial_calibration_model.json"
    path.write_text(json.dumps(partial), encoding="utf-8")
    model = evaluator.load_component_model(path, set(), expected_panel_targets_sha256="e" * 64)
    assert model[3]["status"] == "unavailable"
    assert model[2]["status"] == "valid"

    folds, rows, features = synthetic_context()
    for row in rows:
        digest = row["source_qasm_sha256"]
        width = features[digest]["graph"]["num_qubits"]
        row.update({"normalized_qasm_sha256": hashlib.sha256(("normalized-" + digest).encode()).hexdigest(),
                    "normalized_width": str(width), "normalized_1q_gate_count": "5",
                    "normalized_cx_gate_count": "2"})
    targets = evaluator.validate_panel_targets(rows, folds)
    predictions = evaluator.component_predictions(targets, features, model, 1000)
    unavailable_hashes = [digest for digest, item in features.items() if item["graph"]["num_qubits"] == 3]
    assert unavailable_hashes
    assert all(predictions[digest] == (None, "invalid_or_missing_width_knot") for digest in unavailable_hashes)


def test_oof_evaluation_emits_all_150_rows_same_fold_ids_and_coverage():
    folds, rows, features = synthetic_context()
    observed_graph_folds = []

    def mock_graph(fold, targets, fold_map, _features):
        observed_graph_folds.append(fold)
        train = {digest for digest, value in targets.items()
                 if value["fold"] != fold and value["runtime_seconds"] is not None}
        test = {digest for digest, value in targets.items() if value["fold"] == fold}
        assert train.isdisjoint(test)
        assert all(fold_map[digest] != fold for digest in train)
        return {seed: {digest: 0.25 + 0.01 * fold for digest in test}
                for seed in evaluator.SEEDS}

    oof, component, metrics, manifest = evaluator.evaluate(rows, folds, features, graph_runner=mock_graph)
    assert len(oof) == len(component) == 150
    assert len({row["source_qasm_sha256"] for row in oof}) == 150
    assert observed_graph_folds == [0, 1, 2, 3, 4]
    assert manifest["historical_Aer_CUDAQ_QPU_labels_read"] is False
    by_method_scope = {(row["method_id"], row["comparison_scope"]): row for row in metrics}
    assert {row["method_id"] for row in metrics} == set(evaluator.METHOD_IDS)
    assert by_method_scope[("maestro_style_cpu_sv_component_adaptation", "method_available_rows")]["n_evaluated"] == 0
    graph_available = by_method_scope[("source_dag_graph_adaptation_cuda", "method_available_rows")]
    assert graph_available["n_evaluated"] == 145
    assert graph_available["coverage_of_assigned"] == pytest.approx(145 / 150)
    assert graph_available["prediction_coverage_on_observed"] == pytest.approx(1.0)
    # No component calibration was supplied, so strict all-method comparison is empty;
    # it remains explicitly reported rather than using different row sets silently.
    learned_common = [row for row in metrics if row["comparison_scope"] == "common_learned_methods_intersection"]
    assert by_method_scope[("source_dag_graph_adaptation_cuda", "common_learned_methods_intersection")][
        "n_evaluated"] == 145
    assert by_method_scope[("maestro_style_cpu_sv_component_adaptation",
                            "common_learned_methods_intersection")]["n_evaluated"] == 0
    assert all(row["n_evaluated"] == 145 for row in learned_common
               if row["method_id"] != "maestro_style_cpu_sv_component_adaptation")
    assert all(row["n_evaluated"] == 0 for row in metrics
               if row["comparison_scope"] == "all_methods_common_successful_intersection")
    bootstrap = [row for row in metrics if row["comparison_scope"] == "paired_bootstrap_delta"]
    assert len(bootstrap) == 6
    graph_ridge = next(row for row in bootstrap if row["candidate_method_id"] == "nested_grouped_ridge"
                       and row["reference_method_id"] == "source_dag_graph_adaptation_cuda")
    assert graph_ridge["status"] == "evaluated"
    assert graph_ridge["bootstrap_replicates_requested"] == graph_ridge["bootstrap_replicates_completed"] == 10000
    assert graph_ridge["bootstrap_quantile_method"] == "linear"
    assert graph_ridge["bootstrap_rng_bit_generator"] == "PCG64"
    assert graph_ridge["observed_mae_difference_seconds"] is not None
    assert graph_ridge["bootstrap_mean_mae_difference_seconds"] is not None
    assert graph_ridge["bootstrap_ci_low_seconds"] <= graph_ridge["bootstrap_ci_high_seconds"]
    assert graph_ridge["prediction_coverage_on_observed"] == pytest.approx(1.0)
    assert graph_ridge["coverage_of_assigned"] == pytest.approx(145 / 150)
    # Fixed registered stream gives bit-for-bit reproducible paired summaries.
    graph_ridge_again = next(row for row in evaluator.score_oof(oof)
                             if row.get("candidate_method_id") == "nested_grouped_ridge"
                             and row.get("reference_method_id") == "source_dag_graph_adaptation_cuda")
    assert graph_ridge_again["bootstrap_seed"] == graph_ridge["bootstrap_seed"]
    assert graph_ridge_again["observed_mae_difference_seconds"] == graph_ridge["observed_mae_difference_seconds"]
    assert graph_ridge_again["bootstrap_mean_mae_difference_seconds"] == graph_ridge[
        "bootstrap_mean_mae_difference_seconds"]
    bootstrap_impl = manifest["paired_bootstrap_implementation"]
    assert bootstrap_impl["rng_bit_generator"] == "PCG64"
    assert bootstrap_impl["confidence_interval_quantile_method"] == "linear"
    assert bootstrap_impl["numpy_version"]


def test_graph_arm_refuses_cpu_fallback(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        evaluator.train_graph_fold(0, {}, {}, {})


def test_cuda_evaluation_holds_shared_compute_lease(monkeypatch):
    folds, rows, features = synthetic_context()
    events = []

    @contextmanager
    def lease():
        events.append("acquired")
        yield
        events.append("released")

    def cuda_identity():
        assert events == ["acquired"]
        return {"device": "cuda:0", "cuda_initialized_after_compute_lease": True}

    def mocked_evaluate(*args, **kwargs):
        assert events == ["acquired"]
        return [], [], [], {"status": "evaluation_complete"}

    monkeypatch.setattr(evaluator.s85, "exclusive_compute_lock", lease)
    monkeypatch.setattr(evaluator, "cuda_execution_identity", cuda_identity)
    monkeypatch.setattr(evaluator, "evaluate", mocked_evaluate)
    result = evaluator.evaluate_with_exclusive_compute(rows, folds, features)
    assert events == ["acquired", "released"]
    assert result[-1]["device"] == "cuda:0"
