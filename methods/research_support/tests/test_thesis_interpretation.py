"""Saved-evidence reporting checks only: no training, inference or new timing."""
import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_thesis_interpretation.py"
spec = importlib.util.spec_from_file_location("thesis_reporting", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture(scope="module")
def rendered():
    return report.render()


def rows(rendered, path):
    return list(csv.DictReader(io.StringIO(rendered[0][path])))


def test_required_outputs_and_no_extra_benchmark(rendered):
    files, counts = rendered
    assert all(name in files for name in ["estimator_fidelity_registry.md", "evidence_to_claims.md",
        "final_scientific_interpretation.md", "executive_summary.md", "thesis_claims_and_limitations.md", "presentation_brief.md"])
    assert counts["reused_derived_tables"] == 16
    assert counts["question_matrix_rows"] == 120
    assert counts["saved_transforms"] == 257
    assert counts["executive_summary_words"] <= 1000
    manifest = json.loads(files["provenance_manifest.json"])
    assert manifest["frozen_contract_rqs"] == 4
    assert manifest["reporting_questions"] == 5
    assert manifest["experimental_scope_unchanged"]


def test_all_saved_methods_are_mapped(rendered):
    index = rows(rendered, "tables/method_context_index.csv")
    assert len(index) == 34 + 49 + 3
    assert all(r["category"] in report.CATEGORIES for r in index)
    assert len({(r["track"],r["context_id"],r["method_id"]) for r in index}) == len(index)


def test_skill_formulas_match_independent_saved_tables(rendered):
    claims = {r["claim_id"]:r for r in rows(rendered,"evidence_to_claims.csv")}
    first = json.loads(claims["C01"]["exact_metrics_json"])
    second = json.loads(claims["C02"]["exact_metrics_json"])
    assert first["skill"] == pytest.approx(0.2671956039683391)
    assert second["skill"] == pytest.approx(0.0592371568058443)
    assert first["ci_high"] < 0
    assert second["ci_low"] < 0 < second["ci_high"]


def test_failure_is_supported_not_inconclusive(rendered):
    claims = {r["claim_id"]:r for r in rows(rendered,"evidence_to_claims.csv")}
    for cid in ["C14","C15","C27"]:
        assert claims[cid]["evidence_strength"] == "SUPPORTED"
    assert claims["C12"]["evidence_strength"] == "NOT EVALUATED"
    assert claims["C04"]["evidence_strength"] == "INCONCLUSIVE"


def test_pair_direction_and_small_support_are_preserved(rendered):
    r = rows(rendered,"tables/generalization_robustness.csv")
    assert len(r) == 42 and len({x["case_id"] for x in r}) == 21
    for x in r:
        partner = report.unique(r,case_id=x["case_id"],method_id=report.MLP if x["method_id"]==report.GRAPH else report.GRAPH)
        graph,mlp = (x,partner) if x["method_id"]==report.GRAPH else (partner,x)
        assert float(x["graph_minus_mlp_observed_mae_delta_seconds"]) == pytest.approx(float(graph["mae_seconds"])-float(mlp["mae_seconds"]))
        if int(x["structural_groups"]) < 10:
            assert x["graph_minus_mlp_ci_low_seconds"] == x["graph_minus_mlp_ci_high_seconds"] == ""
    kyoto = report.unique(r,case_id="backend_context:ibm_kyoto",method_id=report.MLP)
    assert int(kyoto["exact_consumed_input_collision_rows"]) == int(kyoto["n"]) == 148


def test_scholten_controls_and_tn_labels(rendered):
    claims = {r["claim_id"]:r for r in rows(rendered,"evidence_to_claims.csv")}
    scholten = json.loads(claims["C11"]["exact_metrics_json"])
    assert scholten["eligible_median"] == pytest.approx(.8200668937428495)
    assert scholten["eligible_source_median"] == pytest.approx(.692070418476177)
    tn = json.loads(claims["C28"]["exact_metrics_json"])
    assert tn["core_ratio"] == pytest.approx(3.7340122316333324)
    assert tn["frontier_ratio"] == pytest.approx(4.132415604942254)


def test_feature_and_qcec_boundaries(rendered):
    claims = {r["claim_id"]:r for r in rows(rendered,"evidence_to_claims.csv")}
    feature = json.loads(claims["C07"]["exact_metrics_json"])
    qcec = json.loads(claims["C20"]["exact_metrics_json"])
    assert (feature["raw"],feature["original_sim_support"],feature["current_parent"]) == (51,41,40)
    assert (qcec["prepared_proved"],qcec["measured_match"],qcec["unitary_proved"],qcec["unitary_counterexample"]) == (32,32,8,24)
    topology = report.unique(rows(rendered,"tables/feature_structural_contribution.csv"), control_id="coarse_topology")
    assert "mean of per-row predictions" in topology["control_type"]


def test_publication_is_not_experimental_completion(rendered):
    text = rendered[0]["README.md"] + rendered[0]["thesis_claims_and_limitations.md"]
    assert "private" in text.lower() and "release" in text.lower()
    assert "không mở experiment queue" in text
    for p in ["tables/track_a_performance.csv", "tables/track_b_findings.csv"]:
        assert rows(rendered,p)
