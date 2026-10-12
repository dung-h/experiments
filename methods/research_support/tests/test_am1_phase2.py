"""Phase 2 saved-evidence tests. No parsing, model inference, training or timing."""
import ast
import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def evidence():
    spec = importlib.util.spec_from_file_location("phase2_analysis", ROOT / "benchmark_v1/scripts/analyze_am1_phase2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build()


def test_no_fit_or_workload_execution_in_harness():
    source = ROOT / "benchmark_v1/scripts/run_am1_phase2.py"
    tree = ast.parse(source.read_text())
    prohibited = {"fit", "train", "backward", "step", "transpile", "sample", "get_state", "contract", "simulate"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            assert name not in prohibited, name
    assert "best_model_state" in source.read_text()
    assert "model_state" in source.read_text()


def test_raw_input_gates_prevent_false_graph_or_dense_b2(evidence):
    m = json.loads(evidence["provenance_manifest.json"])
    v = m["validation"]
    assert v["raw_input_replay_requests"] == 210
    assert v["raw_input_replay_fail"] == 46
    assert v["numerical_replay_comparisons"] == v["numerical_replay_pass"] == 761
    assert v["gate_false_methods_not_timed"]
    assert v["requests_with_complete_B2"] == 45
    data = list(csv.DictReader(io.StringIO(evidence["raw_stage_timings.csv"])))
    assert {r["method"] for r in data if r["boundary"] == "B2"} == {"mali_global_mlp"}


def test_repetition_units_caps_and_coverage(evidence):
    v = json.loads(evidence["provenance_manifest.json"])["validation"]
    assert v["complete_B2_requests_including_repeats"] == 1350
    assert v["native_scalar_queries"] == 105
    assert v["raw_stage_rows"] == 12600
    assert v["ram_peak_bytes"] <= 8*1024**3
    assert v["vram_reserved_peak_bytes"] <= 8*1024**3
    assert v["phase2b_wall_seconds"] <= 7200
    assert 0 < v["preparation_fraction_same_requests"] < 1


def test_accuracy_never_claims_full_cohort_timing(evidence):
    m = json.loads(evidence["provenance_manifest.json"])
    metric = next(r for r in m["summary"]["measured_accuracy"] if r["method"] == "mali_global_mlp")
    assert metric["n"] == 45
    assert json.loads(metric["source_counts"]) == {"mali_real_qpu":38, "qonductor_single_circuit_ibm":7}
    assert metric["groups"] <= 45
    pairs = m["summary"]["paired_comparisons"]
    assert all(r["shared_queries"] == 0 for r in pairs if "mali_logical_graph" in r["comparison"] or "mali_style_dense_graph" in r["comparison"])
    assert not m["validation"]["full_Pareto_frontier"]
    assert not m["validation"]["full_RQ4_closed"]
    assert not m["validation"]["complete_graph_MLP_B2_comparison"]


def test_inventory_does_not_invent_missing_fitted_states(evidence):
    rows = list(csv.DictReader(io.StringIO(evidence["fitted_state_recovery_inventory.csv"])))
    assert len(rows) == 86
    assert {r["newly_recovered_competitive_fitted_state"] for r in rows} == {"false"}
    assert all(r["phase2_state_class"] == "predictions_or_parameters_insufficient" for r in rows if r["estimator_id"].startswith("qonductor_"))


def test_derived_reports_reproduce_exactly(evidence):
    out = ROOT / "artifacts/am1_phase2/v1"
    for name, text in evidence.items():
        assert (out/name).read_text() == text, name
    expected = {"README.md", "phase2a_feasibility.md", "fitted_state_recovery_inventory.csv",
        "pipeline_input_contracts.md", "predeclared_measurement_plan.md", "selected_variants.csv",
        "selected_workloads.csv", "environment_manifest.md", "raw_stage_timings.csv",
        "b2_pipeline_summary.md", "accuracy_cost_tradeoffs.md", "limitations_and_blockers.md",
        "rq4_decision.md", "validation_report.md"}
    assert expected <= {p.name for p in out.iterdir() if p.is_file()}
