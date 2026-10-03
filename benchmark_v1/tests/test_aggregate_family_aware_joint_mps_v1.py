import hashlib
import json

import pytest

from benchmark_v1.scripts.aggregate_family_aware_joint_mps_v1 import (
    METHODS,
    metric_row,
    paired_group_bootstrap,
    select_hash_configurations,
    selection_metric_rows,
    validate_measurement_ledger,
)


def _full_attempt_fixture(status: str = "ok"):
    digest = "a" * 64
    attempts = [{"source_qasm_sha256": digest, "max_bond": chi,
                 "session_id": f"session-{session}", "status": status,
                 "attempt_terminal": True}
                for chi in (2, 4, 8, 16, 32, 64) for session in (1, 2, 3)]
    ledger_sha = hashlib.sha256(
        (json.dumps(attempts, sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()
    summary = {"scope": "full", "status": "PARTIAL" if status != "ok" else "PASS",
               "attempt_count": len(attempts), "technical_error_count": len(attempts) if status != "ok" else 0,
               "attempt_ledger_sha256": ledger_sha}
    return summary, attempts


def test_measurement_ledger_accepts_complete_partial_outcomes_without_imputation():
    summary, attempts = _full_attempt_fixture("adapter_error")

    result = validate_measurement_ledger(summary, attempts, {"a" * 64})

    assert result["measurement_campaign_status"] == "PARTIAL"
    assert result["measurement_attempt_count"] == 18
    assert result["measurement_target_hashes"] == 1
    assert result["measurement_technical_error_count"] == 18
    assert result["measurement_attempt_status_counts"] == {"adapter_error": 18}


def test_measurement_ledger_rejects_missing_identity_in_partial_run():
    summary, attempts = _full_attempt_fixture("timeout")
    attempts.pop()
    summary["attempt_count"] -= 1
    summary["technical_error_count"] -= 1
    summary["attempt_ledger_sha256"] = hashlib.sha256(
        (json.dumps(attempts, sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()

    with pytest.raises(ValueError, match="duplicate, missing, or extra identities"):
        validate_measurement_ledger(summary, attempts, {"a" * 64})


def test_measurement_ledger_rejects_pass_label_with_failure_rows():
    summary, attempts = _full_attempt_fixture("timeout")
    summary["status"] = "PASS"

    with pytest.raises(ValueError, match="labeled PASS despite"):
        validate_measurement_ledger(summary, attempts, {"a" * 64})


def test_family_joint_aggregate_reports_runtime_and_quality_on_valid_labels_only():
    rows = [
        {"target_status": "runtime_observed", "target_runtime_seconds": "1", "runtime_median_seed_seconds": "2",
         "target_quality_pass": "True", "quality_mean_seed_probability": "0.8"},
        {"target_status": "runtime_observed", "target_runtime_seconds": "3", "runtime_median_seed_seconds": "1",
         "target_quality_pass": "False", "quality_mean_seed_probability": "0.6"},
        {"target_status": "unavailable_technical", "target_runtime_seconds": "", "runtime_median_seed_seconds": "2.5",
         "target_quality_pass": "", "quality_mean_seed_probability": "0.5"},
    ]

    metrics = metric_row("c44", "family_conditioned", "all_rungs", rows)

    assert metrics["assigned_rows"] == 3
    assert metrics["runtime_evaluated_rows"] == 2
    assert metrics["runtime_coverage_of_assigned"] == pytest.approx(2 / 3)
    assert metrics["runtime_mae_seconds"] == pytest.approx(1.5)
    assert metrics["runtime_r2"] == pytest.approx(-1.5)
    assert metrics["quality_evaluated_rows"] == 2
    assert metrics["quality_brier_score"] == pytest.approx(0.2)
    assert metrics["quality_accuracy_at_0_5"] == pytest.approx(0.5)
    assert metrics["quality_precision_at_0_5"] == pytest.approx(0.5)


def test_selection_uses_mean_quality_probability_and_never_crosses_rungs():
    rows = []
    for method in METHODS:
        for digest, probabilities, minimum in (
            ("a" * 64, [0.2, 0.55, 0.8, 0.9, 0.9, 0.9], "8"),
            ("b" * 64, [0.1, 0.2, 0.3, 0.4, 0.49, 0.1], "NO_PASS"),
        ):
            for chi, probability in zip((2, 4, 8, 16, 32, 64), probabilities):
                rows.append({"method_id": method, "source_qasm_sha256": digest,
                    "max_bond": chi, "minimum_passing_chi": minimum,
                    "quality_mean_seed_probability": probability,
                    "target_status": "runtime_observed", "target_runtime_seconds": str(chi / 10),
                    "target_quality_pass": str(chi >= 8 and digest == "a" * 64),
                    "runtime_median_seed_seconds": str(chi / 8)})

    decisions = select_hash_configurations(rows)

    assert len(decisions) == 4
    selected = next(row for row in decisions if row["method_id"] == METHODS[0]
                    and row["source_qasm_sha256"] == "a" * 64)
    assert selected["selected_chi"] == 4
    assert selected["selected_actual_runtime_seconds"] == "0.4"
    assert selected["selected_predicted_runtime_seconds"] == "0.5"
    assert selected["actual_minimum_passing_chi"] == 8
    assert selected["selection_is_exact_minimum"] is False
    abstained = next(row for row in decisions if row["source_qasm_sha256"] == "b" * 64)
    assert abstained["selection_status"].startswith("abstain")

    summaries = selection_metric_rows("c44", decisions)
    assert len(summaries) == 2
    assert all(row["assigned_hashes"] == 2 and row["selected_hashes"] == 1 for row in summaries)
    assert all(row["selection_coverage"] == pytest.approx(0.5) for row in summaries)


def test_paired_uncertainty_resamples_exact_qasm_hash_clusters():
    losses = {"a" * 64: [(2.0, 1.0), (2.0, 1.0)],
              "b" * 64: [(4.0, 2.0)]}

    result = paired_group_bootstrap("c44", "same_actual_rung", "mean_absolute_error_seconds",
                                    losses, replicates=100)

    assert result["status"] == "evaluated"
    assert result["cluster_column"] == "source_qasm_sha256"
    assert result["shared_hashes"] == 2
    assert result["shared_rows"] == 3
    assert result["observed_delta_candidate_minus_reference"] == pytest.approx(4 / 3)
    assert result["replicates_completed"] == 100
