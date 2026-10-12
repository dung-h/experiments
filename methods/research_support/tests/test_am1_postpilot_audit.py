"""Saved-evidence arithmetic and scope tests. No inference or measurement."""
import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def evidence():
    spec = importlib.util.spec_from_file_location("am1_audit", ROOT/"benchmark_v1/scripts/build_am1_postpilot_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit()


def rows(evidence, name):
    return list(csv.DictReader(io.StringIO(evidence[name])))


def test_raw_execution_reduction_and_counts(evidence):
    data = rows(evidence, "simulator_comparisons.csv")
    for context, graph, mlp in (("dense_fp32",27,20), ("dense_fp64",28,22)):
        chosen = [r for r in data if r["context"] == context]
        assert len({r["qasm_sha256"] for r in chosen}) == 30
        assert sum(r["graph_slower"] == "True" for r in chosen) == graph
        assert sum(r["mlp_slower"] == "True" for r in chosen) == mlp
        assert all(r["execution_repetitions"] == "15" and r["estimator_repetitions_per_method"] == "30" for r in chosen)
        assert all(r["comparison_class"].startswith("B ") for r in chosen)


def test_subset_and_full_metrics_remain_separate(evidence):
    data = rows(evidence, "accuracy_population_metrics.csv")
    assert len(data) == 22
    qpu = [r for r in data if r["context"] == "qpu"]
    assert {r["n"] for r in qpu if r["population"] == "full_frozen"} == {"4515"}
    assert {r["n"] for r in qpu if r["population"] == "exact_timing_subset"} == {"50"}
    maestro = [r for r in data if r["context"] == "maestro" and r["population"] == "exact_timing_subset"]
    assert {r["n"] for r in maestro} == {"29"}


def test_replay_denominator(evidence):
    data = rows(evidence, "replay_denominators.csv")
    assert sum(int(r["comparisons"]) for r in data) == 1050
    assert sum(int(r["passed"]) for r in data) == 1049
    assert sum(int(r["per_seed_comparisons"]) for r in data) == 660
    assert sum(int(r["ensemble_comparisons"]) for r in data) == 220
    assert sum(int(r["native_comparisons"]) for r in data) == 170


def test_timeout_is_not_attributed_to_scheduled_first_input(evidence):
    data = rows(evidence, "qiskit_selected_inputs.csv")
    assert len(data) == 50
    assert data[0]["observation_id"] == "mali_real_qpu|row91"
    assert all(r["observed_failing_id"].startswith("UNRESOLVED") for r in data)
    assert data[0]["parser_input_kind"] == "saved nominal compiled physical QASM3"


def test_no_experiment_or_full_closure_claim(evidence):
    validation = json.loads(evidence["provenance_manifest.json"])["validation"]
    assert not any(validation[k] for k in ("new_inference", "new_parser_call", "new_measurement", "training", "original_data_modified", "complete_Pareto_frontier", "full_RQ4_closed"))
    assert validation["stop_am1"]
    assert validation["paired_latency_contexts_verified"] == 3


def test_saved_audit_matches_derived_outputs(evidence):
    target = ROOT/"artifacts/am1_postpilot_audit/v1"
    for name, content in evidence.items():
        assert (target/name).read_text() == content, name
