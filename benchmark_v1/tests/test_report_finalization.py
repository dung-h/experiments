"""Exercise final report data/clock boundaries, including rehashed corrupt outputs."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "benchmark_v1/scripts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def builder():
    return load("build_two_domain_scorecard_v2")


@pytest.fixture(scope="module")
def validator():
    return load("validate_two_domain_scorecard_v2")


@pytest.fixture(scope="module")
def report_pack(builder, tmp_path_factory):
    output = tmp_path_factory.mktemp("report-finalization") / "reader_tables"
    # Test output only; the actual canonical v3 remains agent F-C1's task.
    builder.REBUILD_ROOT = output.parent
    before = {p: digest(p) for p in (
        ROOT / "benchmark_v1/registry/CURRENT.json",
        ROOT / "artifacts/benchmark_v1/reproducibility/c4_release_receipt_20260930/receipt.json",
        builder.DEFAULT_OUT / "manifest.json")}
    builder.build(output, report_manifest=builder.DEFAULT_REPORT)
    assert all(digest(p) == value for p, value in before.items())
    return output


def test_frozen_contract_and_raw_terminal_evidence(builder):
    contract = builder.load_report_contract(builder.DEFAULT_REPORT)
    state, rows = builder.maestro_terminal_evidence(contract)
    assert state["score"] is None
    assert state["status_at_build"] == "pilot_gate_failed"
    assert (state["api_calls"], state["timed_calls"], state["untimed_calls"]) == (240, 150, 90)
    assert state["completed_cells"] + state["eligible_unstarted_cells"] + state["quarantined_calibration_cells"] == 432
    assert state["sv_stability_failed_cells"] == 9 and state["mps_stability_passed_cells"] == 1
    assert len(rows) == 10 and all(not row["score_eligible"] for row in rows)
    row = next(r for r in rows if r["candidate"] == "statevector"
               and r["operation_class"] == "one_qubit_noncommuting" and r["operation_repeats"] == "256")
    assert row["min_reported_seconds"] == pytest.approx(0.014905151)
    assert row["max_reported_seconds"] == pytest.approx(7.355305934)


def test_final_build_roundtrip_and_historical_snapshot(validator, report_pack):
    result = validator.validate(report_pack)
    assert result["status"] == "PASS", result
    assert result["maestro_status"] == "pilot_gate_failed"
    manifest = json.loads((report_pack / "manifest.json").read_text())
    assert manifest["scientific_coverage"] == "PARTIAL"
    assert manifest["simulator"]["maestro"]["panel_execution_started"] is False
    assert manifest["historical_availability_tables_are_snapshots"] is True
    assert "representative rz(0.3)/rx(0.2)" in (report_pack / "REPORT.md").read_text()


def test_historical_v2_remains_valid(validator, builder):
    result = validator.validate(builder.DEFAULT_OUT)
    assert result["status"] == "PASS", result
    assert result["maestro_status"] == "not_started_no_partial_run_manifest"
    assert len(result["historical_generator_verified_at"]) == 1


def test_refuses_overwrite(builder, report_pack):
    with pytest.raises(ValueError, match="refusing to overwrite"):
        builder.build(report_pack, report_manifest=builder.DEFAULT_REPORT)


def test_refuses_final_over_historical_output(builder):
    with pytest.raises(ValueError, match="canonical v3"):
        builder.build(builder.DEFAULT_OUT, report_manifest=builder.DEFAULT_REPORT)


@pytest.mark.parametrize("field,value", [
    ("experiment_or_training_authorized", True), ("public_release_approved", True),
    ("scientific_coverage", "COMPLETE"), ("wave4_five_fold_training_authorized", True)])
def test_contract_cannot_expand_authority(builder, tmp_path, field, value):
    contract = json.loads(builder.DEFAULT_REPORT.read_text())
    contract[field] = value
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="authority mismatch"):
        builder.load_report_contract(path)


def test_contract_source_population_not_just_hashes(builder, tmp_path):
    contract = json.loads(builder.DEFAULT_REPORT.read_text())
    contract["real_qpu"]["source_counts"]["qpack_mcp"] = 3944
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="source count mismatch"):
        builder.load_report_contract(path)


@pytest.mark.parametrize("field,value", [("score", 0.1), ("accepted_predictor", True),
                                        ("api_calls", 241), ("panel_execution_started", True)])
def test_rejects_fabricated_maestro_manifest(validator, report_pack, tmp_path, field, value):
    dest = tmp_path / "pack"
    shutil.copytree(report_pack, dest)
    path = dest / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["simulator"]["maestro"][field] = value
    path.write_text(json.dumps(manifest))
    assert validator.validate(dest)["status"] == "FAIL"


def test_rehashed_pilot_table_cannot_override_raw_stability(validator, report_pack, tmp_path):
    dest = tmp_path / "pack"
    shutil.copytree(report_pack, dest)
    path = dest / "simulator/maestro_pilot_summary.csv"
    rows = list(csv.DictReader(path.open()))
    rows[0]["stability_status"] = "ok"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mp = dest / "manifest.json"
    manifest = json.loads(mp.read_text())
    manifest["outputs"]["simulator/maestro_pilot_summary.csv"] = digest(path)
    mp.write_text(json.dumps(manifest))
    result = validator.validate(dest)
    assert result["status"] == "FAIL"
    assert any("raw-recomputed" in err for err in result["errors"])


def test_missing_required_output_pin_fails(validator, report_pack, tmp_path):
    dest = tmp_path / "pack"
    shutil.copytree(report_pack, dest)
    path = dest / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["outputs"]["simulator/maestro_pilot_summary.csv"]
    path.write_text(json.dumps(manifest))
    result = validator.validate(dest)
    assert result["status"] == "FAIL"
    assert "final output inventory mismatch" in result["errors"]


def test_terminal_expected_counts_cannot_be_relabelled(builder):
    contract = copy.deepcopy(builder.load_report_contract(builder.DEFAULT_REPORT))
    contract["maestro"]["expected"]["eligible_unstarted_cells"] = 0
    with pytest.raises(ValueError, match="expected outcome mismatch"):
        builder.maestro_terminal_evidence(contract)


def test_missing_evidence_is_fail_closed(validator, tmp_path):
    (tmp_path / "manifest.json").write_text('{}')
    assert validator.validate(tmp_path)["status"] == "FAIL"
