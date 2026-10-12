"""Saved-table interpretation checks only; no model fit, inference or simulation."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark_v1/scripts"))
import interpret_frozen_benchmark as M


def test_raw_ledger_order_and_classes():
    ledger = M.rows(M.OUT / "global_feature_ledger.csv")
    assert len(ledger) == 51 and [int(r["raw_index"]) for r in ledger] == list(range(51))
    assert Counter(r["feature_class"] for r in ledger) == {"gate_count":44,"width_depth":2,"SupermarQ_structural":5}


def test_original_supported_widths_and_exact_zero_fields():
    source = M.js(M.OUT / "original_corpus_support.json")
    assert (source["simulator_rows"],source["real_rows"]) == (3020,340)
    assert (source["sim_effective_width"],source["real_effective_width"]) == (41,40)
    zeros = {r["feature_name"] for r in source["support"] if not r["retained_original_sim_refinement"]}
    assert zeros == {"id","u0","csx","cu","rc3x","c3x","c3sqrtx","c4x","xx_plus_yy","ecr"}
    assert source["support"][37]["original_sim_nonzero_rows"] == 12


def test_notebook_default_and_recomputed_real_support_not_conflated():
    value = M.js(M.OUT / "upstream_pipeline_evidence.json")
    assert value["model_default_global_width"] == 41
    assert len(value["simulator_recorded_retained_indices"]) == 41
    assert len(value["real_recorded_retained_indices"]) == 12
    assert len(value["real_recomputed_published_340_indices"]) == 40
    assert value["no_notebook_cell_executed"] and value["no_checkpoint_loaded"]


def test_all_eleven_zero_positions_cannot_reach_current_model():
    ledger = M.rows(M.OUT / "global_feature_ledger.csv")
    zeros = [r for r in ledger if int(r["unified_nonzero_rows"]) == 0]
    assert len(zeros) == 11
    assert all(r["can_reach_any_current_model"] == "False" for r in zeros)


def test_all_transform_widths_and_size_constants():
    values = M.rows(M.OUT / "effective_input_widths.csv")
    assert len(values) == 257
    assert [int(v["effective_tensor_width"]) for v in values[:5]] == [40]*5
    final = [r for r in values if r["stage"] == "final_refit"]
    assert Counter(int(r["effective_tensor_width"]) for r in final) == {40:108,22:6,11:12}
    low = [r for r in values if r["case_id"] == "qpack_size:train_through_4" and r["stage"] == "selection"]
    assert len(low) == 6 and all((r["effective_tensor_width"],r["variable_width"]) == ("11","9") for r in low)


def test_random_support_loss_not_confused_with_eleven_global_zero_fields():
    rows = M.rows(M.OUT / "generalization_decomposition.csv")
    random = next(r for r in rows if r["case_id"] == "family:random")
    qnn = next(r for r in rows if r["case_id"] == "family:qnn")
    assert len(json.loads(random["held_out_only_global_indices_json"])) == 18
    assert json.loads(qnn["held_out_only_global_indices_json"]) == []
    assert all(float(r["skill"]) < -10 for r in M.rows(M.OUT / "generalization_predictive_skill.csv")
               if r["scope"] in {"family:qnn","family:random"}
               and r["method_id"] in {"mali_logical_graph","mali_global_mlp"}
               and r["reference_id"] == "train_global_median")


def test_skill_formula_and_source_information_disclosure():
    for name in ["qpu_predictive_skill.csv","simulator_predictive_skill.csv","analytical_predictive_skill.csv"]:
        for row in M.rows(M.OUT / name):
            assert abs(float(row["skill"]) - (1-float(row["mae_seconds"])/float(row["reference_mae_seconds"]))) < 1e-10
            if "source_median" in row["reference_id"]:
                assert row["reference_context_advantage"] == "True"


def test_macro_is_ratio_of_maes_not_mean_ratios():
    for row in M.rows(M.OUT / "qpu_source_balanced_skill.csv"):
        expected = 1-float(row["method_macro_mae_seconds"])/float(row["reference_macro_mae_seconds"])
        assert abs(float(row["source_balanced_skill"])-expected) < 1e-12


def test_all_eight_dimensions_have_saved_evidence():
    rows = M.rows(M.OUT / "benchmark_question_matrix.csv")
    assert len(rows) == 120
    assert len({r["dimension"] for r in rows}) == 8
    assert set(Counter(r["method_or_context"] for r in rows).values()) == {8}
    for r in rows:
        assert r["status"] in {"ANSWERED","PARTIAL","NOT ANSWERED","NOT APPLICABLE"}
        assert all((ROOT / p).is_file() for p in r["evidence_paths"].split(";"))


def test_shot_diversity_is_not_called_shot_transfer():
    rows = M.rows(M.OUT / "shot_diversity.csv")
    assert all(r["systematic_qpu_shot_holdout_status"] == "NOT ANSWERED" for r in rows)
    counts = {r["key"]:int(r["distinct_shot_values"]) for r in rows if r["stratum"] == "source_id"}
    assert counts == {"mali_real_qpu":1,"qonductor_single_circuit_ibm":6,"qpack_mcp":1}


def test_scholten_proxy_label_and_native_clock_boundaries():
    text = (M.OUT / "runtime_estimator_interpretation.md").read_text()
    assert "Scholten-inspired generalized" in text and "original not evaluable" in text
    assert "source median .69207s" in text and "train median .82007s" in text
    assert "additional predictor, not original" in text


def test_derived_question_tables_reproduce_without_experiments():
    for name, text in M.interpretation_tables().items():
        assert (M.OUT / name).read_text() == text


def test_reporting_decision_is_not_training_authorization():
    audit = (M.OUT / "mali_feature_schema_audit.md").read_text()
    assert "reporting fix only" in audit and "per-feature fit-only moments" in audit
    assert "Hard-code41?" in (M.OUT / "README.md").read_text()
    assert "not exact paper execution" in audit
