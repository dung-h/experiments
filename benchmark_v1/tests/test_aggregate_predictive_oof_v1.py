from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import aggregate_predictive_oof_v1 as aggregate  # noqa: E402

MEMBERS_BY_FOLD = {0: 39, 1: 30, 2: 27, 3: 28, 4: 38}


def _hash(letter: str) -> str:
    return letter * 64


def _record(digest: str, target, prediction, quality="not_applicable", fold=0):
    return {
        "source_sha256": digest,
        "fold": fold,
        "target_seconds": target,
        "quality_status": quality,
        "prediction_seconds": prediction,
        "method_terminal_status": "predicted" if prediction is not None else "not_run",
    }


def test_aliases_deduplicate_by_exact_hash_after_hash_and_fold_join():
    digest = _hash("a")
    labels = [
        {"panel_member_id": "member-a1", "source_sha256": digest, "fold": "0", "observed_seconds": "1", "target_clock": "warm_execution"},
        {"panel_member_id": "member-a2", "source_sha256": digest, "fold": "0", "observed_seconds": "3", "target_clock": "warm_execution"},
    ]
    ledger = [
        {"panel_member_id": "member-a1", "source_sha256": digest, "fold": "0", "method_id": "m", "target_seconds": "2", "prediction_seconds": "4", "status": "predicted", "terminal_reason": ""},
        {"panel_member_id": "member-a2", "source_sha256": digest, "fold": "0", "method_id": "m", "target_seconds": "2", "prediction_seconds": "4", "status": "predicted", "terminal_reason": ""},
    ]

    methods, coverage, clock = aggregate.collapse_aer_member_ledger(labels, ledger, validate_frozen_panel=False)

    assert clock == "warm_execution"
    assert list(methods["m"]) == [digest]
    assert methods["m"][digest]["target_seconds"] == 2.0  # median of alias observations
    assert methods["m"][digest]["prediction_seconds"] == 4.0
    assert methods["m"][digest]["panel_member_count"] == 2
    assert len(coverage) == 1
    assert coverage[0]["method_output_clock"] == aggregate.EXPECTED_AER_METHOD_OUTPUT_CLOCK
    assert coverage[0]["member_ids_json"] == '["member-a1","member-a2"]'


def test_aer_alias_prediction_conflict_fails_closed():
    digest = _hash("b")
    labels = [
        {"panel_member_id": member, "source_sha256": digest, "fold": "0", "observed_seconds": "2", "target_clock": "warm_execution"}
        for member in ("a", "b")
    ]
    ledger = [
        {"panel_member_id": member, "source_sha256": digest, "fold": "0", "method_id": "m", "target_seconds": "2", "prediction_seconds": prediction, "status": "predicted", "terminal_reason": ""}
        for member, prediction in (("a", "3"), ("b", "4"))
    ]
    with pytest.raises(ValueError, match="aliases disagree"):
        aggregate.collapse_aer_member_ledger(labels, ledger, validate_frozen_panel=False)


def test_mps_metrics_keep_150_144_142_2_6_denominators_separate():
    records = {}
    for index in range(150):
        digest = f"{index:064x}"
        if index < 142:
            target, quality = float(index + 1), "quality_pass"
        elif index < 144:
            target, quality = float(index + 1), "quality_failed"
        else:
            target, quality = None, "quality_unavailable"
        # Predictions can exist for an unavailable target; they remain unscored.
        records[digest] = _record(digest, target, float(index + 2), quality, fold=index % 5)

    all_finite = aggregate.summarize_method(
        domain_id="mps_fixed_chi16",
        source_artifact="fixture",
        source_method_column="prediction",
        method_id="m",
        records=records,
        target_clock="fixed_mps_target_clock",
        method_output_clock="predicted_fixed_mps_output_clock",
        population="all_finite_targets",
        quality_available=True,
    )
    quality_pass = aggregate.summarize_method(
        domain_id="mps_fixed_chi16",
        source_artifact="fixture",
        source_method_column="prediction",
        method_id="m",
        records=records,
        target_clock="fixed_mps_target_clock",
        method_output_clock="predicted_fixed_mps_output_clock",
        population="quality_pass_only",
        quality_available=True,
    )

    assert (all_finite["assigned_hashes"], all_finite["finite_target_hashes"]) == (150, 144)
    assert (all_finite["quality_pass_hashes"], all_finite["quality_failed_finite_hashes"], all_finite["target_unavailable_hashes"]) == (142, 2, 6)
    assert all_finite["prediction_hashes_assigned"] == 150
    assert all_finite["scored_hashes"] == 144
    assert all_finite["evaluation_target_clock"] == "fixed_mps_target_clock"
    assert all_finite["method_output_clock"] == "predicted_fixed_mps_output_clock"
    assert quality_pass["scored_hashes"] == 142
    assert quality_pass["mae_seconds"] == pytest.approx(1.0)


def test_bootstrap_seed_and_hash_resampling_are_reproducible_with_separate_point_and_mean():
    registry = {"root_seed": 20260925, "streams": {"bootstrap": "20260925-bootstrap"}}
    hashes = [_hash(letter) for letter in "abcd"]
    candidate = {digest: _record(digest, target, pred, fold=i) for i, (digest, target, pred) in enumerate(zip(hashes, (1, 2, 4, 8), (1, 2, 4, 20)))}
    reference = {digest: _record(digest, target, pred, fold=i) for i, (digest, target, pred) in enumerate(zip(hashes, (1, 2, 4, 8), (5, 2, 4, 8)))}
    kwargs = dict(
        domain_id="test_panel",
        target_clock="one_clock",
        method_output_clock="one_clock",
        population="all_finite_targets",
        comparison_label="test_panel|candidate-vs-reference|all_finite_targets",
        candidate_method_id="candidate",
        reference_method_id="reference",
        candidate=candidate,
        reference=reference,
        assigned_hashes=4,
        registry=registry,
    )

    first = aggregate.paired_bootstrap_row(**kwargs)
    second = aggregate.paired_bootstrap_row(**kwargs)

    assert first == second
    assert first["bootstrap_seed"] == aggregate.registered_seed(registry, comparison_label=kwargs["comparison_label"])
    assert first["bootstrap_replicates_completed"] == 10_000
    assert first["bootstrap_group_column"] == "source_sha256"
    assert first["method_output_clock"] == "one_clock"
    assert first["observed_mae_difference_seconds"] == pytest.approx(2.0)
    assert first["bootstrap_mean_mae_difference_seconds"] != ""
    assert first["bootstrap_ci_low_seconds"] != ""
    assert first["bootstrap_ci_high_seconds"] != ""
    assert first["shared_hashes"] == 4


def test_cross_configuration_comparison_is_rejected():
    registry = {"root_seed": 1, "streams": {"bootstrap": "20260925-bootstrap"}}
    with pytest.raises(ValueError, match="cross-configuration"):
        aggregate.build_comparison_rows(
            domain_id="aer_azizov_core_q9",
            methods={
                "aer": {_hash("a"): _record(_hash("a"), 1.0, 1.0)},
                "mps": {_hash("a"): _record(_hash("a"), 1.0, 1.0)},
            },
            target_clock="aer_clock",
            method_output_clock="aer_prediction_clock",
            comparisons=[
                {
                    "domain_id": "mps_fixed_chi16",
                    "comparison_label": "never_cross_clocks",
                    "candidate_method_id": "aer",
                    "reference_method_id": "mps",
                }
            ],
            registry=registry,
            quality_available=False,
        )


def test_aggregate_refuses_absent_e4_outputs():
    missing = Path("/tmp/e6-deliberately-missing-e4-fixture")
    with pytest.raises(FileNotFoundError, match="E4 Aer"):
        aggregate._require_files(missing, aggregate.E4_REQUIRED, role="E4 Aer")


def test_artifact_publication_is_idempotent_and_never_overwrites_different_bytes(tmp_path):
    output = tmp_path / "aggregate"
    first = {"table.csv": b"id,value\n1,2\n"}
    aggregate._write_new_artifact(output, first)
    aggregate._write_new_artifact(output, first)
    assert (output / "table.csv").read_bytes() == first["table.csv"]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        aggregate._write_new_artifact(output, {"table.csv": b"id,value\n1,999\n"})
    assert (output / "table.csv").read_bytes() == first["table.csv"]


def test_aggregate_requires_complete_five_fold_e4_oof():
    with pytest.raises(ValueError, match="complete five-fold E4 OOF"):
        aggregate.validate_e4_completion({"status": "stopped_fold0_technical_failure"})
    aggregate.validate_e4_completion({"status": "complete_five_fold_oof"})


def _complete_e4_manifest():
    audits = {}
    for fold, assigned_hashes in aggregate.EXPECTED_FOLDS.items():
        audits[str(fold)] = {
            "fold": fold,
            "assigned_members": MEMBERS_BY_FOLD[fold],
            "assigned_hashes": assigned_hashes,
            "eligible_hashes": assigned_hashes,
            "unavailable_representation_hashes": 0,
            "method_families": 24,
            "method_member_terminal_rows": 24 * MEMBERS_BY_FOLD[fold],
            "method_audit": {method_id: {"status": "PASS"} for method_id in aggregate.E4_BASE_METHOD_IDS},
        }
    audits["0"].update(
        {
            "technical_status": "PASS",
            "continuation_decision": "automatically_continue_folds_1_through_4",
            "continuation_reasons": [],
        }
    )
    return {
        "status": "complete_five_fold_oof",
        "folds": [0, 1, 2, 3, 4],
        "method_family_count": 24,
        "fold_audits": audits,
        "fold0_qa": dict(audits["0"]),
    }


def test_e4_manifest_proves_all_frozen_folds_and_fold0_continuation():
    manifest = _complete_e4_manifest()
    aggregate.validate_e4_completion(manifest)
    aggregate.validate_e4_fold_audits(manifest, MEMBERS_BY_FOLD)

    manifest["fold_audits"]["2"]["assigned_hashes"] = 24
    with pytest.raises(ValueError, match="fold 2 assigned_hashes"):
        aggregate.validate_e4_fold_audits(manifest, MEMBERS_BY_FOLD)

    manifest = _complete_e4_manifest()
    manifest["fold_audits"]["4"]["eligible_hashes"] = 33
    with pytest.raises(ValueError, match="fold 4 does not show all assigned hashes"):
        aggregate.validate_e4_fold_audits(manifest, MEMBERS_BY_FOLD)

    manifest = _complete_e4_manifest()
    manifest["fold_audits"]["1"]["assigned_members"] += 1
    with pytest.raises(ValueError, match="fold 1 assigned member count"):
        aggregate.validate_e4_fold_audits(manifest, MEMBERS_BY_FOLD)


def test_e4_manifest_rejects_missing_fold0_gnn_pass_or_continuation():
    manifest = _complete_e4_manifest()
    manifest["fold_audits"]["0"]["method_audit"]["azizov_gnn_source_seed_42"]["status"] = "FAIL"
    manifest["fold0_qa"]["method_audit"]["azizov_gnn_source_seed_42"]["status"] = "FAIL"
    with pytest.raises(ValueError, match="nine CUDA-inference GNN"):
        aggregate.validate_e4_fold_audits(manifest, MEMBERS_BY_FOLD)

    manifest = _complete_e4_manifest()
    manifest["fold_audits"]["0"]["continuation_decision"] = "stop_before_fold1"
    manifest["fold0_qa"]["continuation_decision"] = "stop_before_fold1"
    with pytest.raises(ValueError, match="automatic continuation"):
        aggregate.validate_e4_fold_audits(manifest, MEMBERS_BY_FOLD)


def test_later_fold_method_failure_is_preserved_and_checked_against_terminal_ledger():
    manifest = _complete_e4_manifest()
    failed_method = "classical_ridge_source"
    manifest["fold_audits"]["3"]["method_audit"][failed_method] = {"status": "FAIL"}
    aggregate.validate_e4_fold_audits(manifest, MEMBERS_BY_FOLD)
    ledger = []
    for fold in range(5):
        for method_id in aggregate.E4_BASE_METHOD_IDS:
            status = "technical_failure" if fold == 3 and method_id == failed_method else "predicted"
            ledger.extend({"fold": str(fold), "method_id": method_id, "status": status} for _ in range(MEMBERS_BY_FOLD[fold]))

    aggregate.validate_e4_terminal_ledger(manifest["fold_audits"], ledger, MEMBERS_BY_FOLD)
    bad_ledger = [dict(row, status="predicted") if int(row["fold"]) == 3 and row["method_id"] == failed_method else row for row in ledger]
    with pytest.raises(ValueError, match="audit/terminal-ledger status mismatch"):
        aggregate.validate_e4_terminal_ledger(manifest["fold_audits"], bad_ledger, MEMBERS_BY_FOLD)

    blocked_manifest = _complete_e4_manifest()
    blocked_manifest["fold_audits"]["4"]["method_audit"][failed_method] = {"status": "BLOCKED"}
    aggregate.validate_e4_fold_audits(blocked_manifest, MEMBERS_BY_FOLD)
    blocked_ledger = []
    for fold in range(5):
        for method_id in aggregate.E4_BASE_METHOD_IDS:
            terminal = "blocked_compute_lock_or_cuda" if fold == 4 and method_id == failed_method else "predicted"
            blocked_ledger.extend({"fold": str(fold), "method_id": method_id, "status": terminal} for _ in range(MEMBERS_BY_FOLD[fold]))
    aggregate.validate_e4_terminal_ledger(blocked_manifest["fold_audits"], blocked_ledger, MEMBERS_BY_FOLD)


def test_method_cards_distinguish_seed_diagnostics_medians_and_local_baselines():
    aer_ids = {
        "azizov_gnn_source_seed_42",
        "azizov_gnn_source_median_three_seeds",
        "classical_linear_regression_source",
    }
    mps_ids = {
        "e3_fixed_mps_train_fold_median",
        "e3_fixed_mps_ridge_alpha_1",
        "e3_fixed_mps_graph_median_three_seeds",
        "e5_family_residual_median_three_seeds",
        "e5_family_agnostic_median_three_seeds",
    }
    cards = aggregate.build_method_cards(
        aer_methods={method_id: {} for method_id in aer_ids},
        mps_methods={method_id: {} for method_id in mps_ids},
        aer_output_clock=aggregate.EXPECTED_AER_METHOD_OUTPUT_CLOCK,
        mps_output_clock=aggregate.EXPECTED_MPS_METHOD_OUTPUT_CLOCK,
    )

    assert set(cards) == aer_ids | mps_ids
    required_fields = {
        "method_id", "reader_label", "fidelity_class", "representation_stage",
        "method_output_clock", "claim_boundary",
    }
    assert all(set(card) == required_fields for card in cards.values())
    assert {card["fidelity_class"] for card in cards.values()} <= {"baseline", "adaptation"}
    assert "seed 42 individual output" in cards["azizov_gnn_source_seed_42"]["reader_label"]
    assert "not a selected seed" in cards["azizov_gnn_source_seed_42"]["reader_label"]
    assert "median of seeds 42, 1234, and 31415" in cards["azizov_gnn_source_median_three_seeds"]["reader_label"]
    assert cards["azizov_gnn_source_seed_42"]["fidelity_class"] == "adaptation"
    assert cards["classical_linear_regression_source"]["fidelity_class"] == "baseline"
    assert cards["classical_linear_regression_source"]["reader_label"].startswith("Local linear regression")
    assert cards["e3_fixed_mps_graph_median_three_seeds"]["fidelity_class"] == "adaptation"
    assert cards["e3_fixed_mps_ridge_alpha_1"]["fidelity_class"] == "baseline"
    assert cards["e3_fixed_mps_train_fold_median"]["fidelity_class"] == "baseline"
    assert cards["e5_family_residual_median_three_seeds"]["fidelity_class"] == "adaptation"
    assert cards["e5_family_residual_median_three_seeds"]["method_output_clock"] == aggregate.EXPECTED_MPS_METHOD_OUTPUT_CLOCK
    assert all(card["claim_boundary"].endswith(".") for card in cards.values())
    assert "evaluation_target_clock" in aggregate.METRICS_FIELDS
    assert "method_output_clock" in aggregate.METRICS_FIELDS
    assert "evaluation_target_clock" in aggregate.BOOTSTRAP_FIELDS
    assert "method_output_clock" in aggregate.BOOTSTRAP_FIELDS
