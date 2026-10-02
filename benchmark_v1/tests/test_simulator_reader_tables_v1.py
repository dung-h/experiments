"""Simulator reader tables v1 are evidence-only and fail-closed."""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/benchmark_v3/simulator/reader_tables_v1"
VALIDATOR = ROOT / "benchmark_v1/scripts/validate_simulator_reader_tables_v1.py"
CURRENT = ROOT / "benchmark_v1/registry/CURRENT.json"
HISTORICAL_ARCHIVE = ROOT / "benchmark_v1/registry/archive/CURRENT_pre_wave3_authority_switch_20261001.json"
PINNED_LIVE_CURRENT = "8a4ba7288e46324c49e290a1244c518b9f01b73a7f27a9b4a79984a13135c9e3"
PINNED_HISTORICAL_CURRENT = "fa840866ffa16b58d4fc0aabe2bc8b7a50023d38791a7cc20401989c5faa0680"
PINNED_FIDELITY_V2 = "98f3460c8d9a0cbabb424555dd695cee573409ed65facdb60557d836f1750810"
PINNED_SCORECARD = "f6d27336ecdb16a90efa4156a7e4d48dfbbea0be3c393b6e4a9a435641f8fc5c"


def validator_module():
    spec = importlib.util.spec_from_file_location("validate_simulator_reader_tables_v1", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_artifact(tmp_path: Path) -> Path:
    dest = tmp_path / "reader_tables_v1"
    shutil.copytree(ARTIFACT, dest)
    return dest


def _rewrite(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_validator_passes_real_artifact() -> None:
    module = validator_module()
    assert module.validate() == []
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    archive = json.loads(HISTORICAL_ARCHIVE.read_text(encoding="utf-8"))
    assert module.sha(CURRENT) == PINNED_LIVE_CURRENT
    assert module.sha(HISTORICAL_ARCHIVE) == PINNED_HISTORICAL_CURRENT
    assert current["status"] == "current"
    assert current["current_method_fidelity"]["path"] == "benchmark_v1/registry/method_fidelity_registry_v2.json"
    assert current["current_method_fidelity"]["sha256"] == PINNED_FIDELITY_V2
    assert current["reader_pointers"]["two_domain_scorecard_v1"]["sha256"] == PINNED_SCORECARD
    assert current["wave4_five_fold_training_authorized"] is False
    assert archive["current_method_fidelity"]["path"] == "benchmark_v1/registry/method_fidelity_registry_v1.json"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "field",
    (
        "status",
        "current_method_fidelity.path",
        "current_method_fidelity.sha256",
        "reader_pointers.two_domain_scorecard_v1.sha256",
        "wave4_five_fold_training_authorized",
    ),
)
def test_validator_rejects_live_authority_field_drift(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = validator_module()
    fake = tmp_path / "CURRENT.json"
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    if field == "status":
        current["status"] = "stale"
    elif field == "current_method_fidelity.path":
        current["current_method_fidelity"]["path"] = "benchmark_v1/registry/method_fidelity_registry_v1.json"
    elif field == "current_method_fidelity.sha256":
        current["current_method_fidelity"]["sha256"] = "0" * 64
    elif field == "reader_pointers.two_domain_scorecard_v1.sha256":
        current["reader_pointers"]["two_domain_scorecard_v1"]["sha256"] = "0" * 64
    else:
        current["wave4_five_fold_training_authorized"] = True
    write_json(fake, current)
    monkeypatch.setattr(module, "CURRENT", fake)
    monkeypatch.setattr(module, "PINNED_LIVE_CURRENT", module.sha(fake))
    assert any("live CURRENT" in item for item in module.validate())


def test_validator_rejects_live_current_hash_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = validator_module()
    fake = tmp_path / "CURRENT.json"
    shutil.copyfile(CURRENT, fake)
    fake.write_text(fake.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "CURRENT", fake)
    assert any("live CURRENT" in item for item in module.validate())


def test_validator_rejects_historical_pack_pin_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    module = validator_module()
    label, (path, digest) = next(iter(module.PACKS.items()))
    monkeypatch.setitem(module.PACKS, label, (path, "0" * 64))
    assert any("reader_table_pack_" in item for item in module.validate())


def test_panel_identity_allows_declared_qasm_aliases() -> None:
    with (ARTIFACT / "common_exact_qasm_panel.csv").open(newline="", encoding="utf-8") as handle:
        panel = list(csv.DictReader(handle))
    members = [row["panel_member_id"] for row in panel]
    qasms = {row["exact_qasm_sha256"] for row in panel}
    strata = {row["stratum"] for row in panel}
    assert len(panel) == 204
    assert len(set(members)) == 204
    assert len(qasms) == 191
    assert strata == {"core_q2_q9", "frontier_q10_q16"}


def test_validator_rejects_missing_source_hash(tmp_path: Path) -> None:
    module = validator_module()
    dest = _copy_artifact(tmp_path)
    path = dest / "per_configuration_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["source_sha256"] = "0" * 64
    _rewrite(path, rows)
    errors = module.validate(dest)
    assert any("source_sha256" in item for item in errors)


def test_validator_rejects_pooled_fp32_fp64(tmp_path: Path) -> None:
    module = validator_module()
    dest = _copy_artifact(tmp_path)
    path = dest / "per_configuration_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if "cudaq_nvidia_dense_fp32" in row["configuration_id"]:
            row["configuration_id"] = "cudaq_nvidia_dense_fp32_fp64"
            break
    _rewrite(path, rows)
    errors = module.validate(dest)
    assert any("fp32 and fp64 pooled" in item for item in errors)


def test_validator_rejects_qpu_clock_on_simulator_metric(tmp_path: Path) -> None:
    module = validator_module()
    dest = _copy_artifact(tmp_path)
    path = dest / "per_configuration_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["evaluation_target_clock"] = "archived_observed_service_execution_time"
    _rewrite(path, rows)
    errors = module.validate(dest)
    assert any("QPU clock" in item for item in errors)


def test_validator_rejects_duplicate_metric_keys(tmp_path: Path) -> None:
    module = validator_module()
    dest = _copy_artifact(tmp_path)
    path = dest / "per_configuration_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.append(dict(rows[0]))
    _rewrite(path, rows)
    errors = module.validate(dest)
    assert any("duplicate metric composite key" in item for item in errors)


def test_validator_rejects_imputed_maestro_runtime(tmp_path: Path) -> None:
    module = validator_module()
    dest = _copy_artifact(tmp_path)
    path = dest / "per_configuration_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fake = dict(rows[0])
    fake["configuration_id"] = "maestro_imputed"
    fake["method_id"] = "maestro_formula_reimplementation"
    fake["median_seconds"] = "1.0"
    fake["stratum"] = "core_q2_q9"
    fake["evaluation_target_clock"] = "warm_execution"
    fake["method_output_clock"] = "warm_execution"
    rows.append(fake)
    _rewrite(path, rows)
    errors = module.validate(dest)
    assert any("imputed Maestro runtime" in item for item in errors)


@pytest.mark.parametrize("forbidden", ["torch.cuda", "nvidia-smi", "cudaq.sample"])
def test_builder_and_validator_do_not_start_timing(forbidden: str) -> None:
    builder = (ROOT / "benchmark_v1/scripts/build_simulator_reader_tables_v1.py").read_text(encoding="utf-8")
    validator = VALIDATOR.read_text(encoding="utf-8")
    assert forbidden not in builder
    assert f"import {forbidden.split('.')[0]}" not in validator
    assert "Popen" not in builder and "Popen" not in validator


def test_aer_measurement_is_split_from_azizov_predictor() -> None:
    with (ARTIFACT / "per_configuration_metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    with (ARTIFACT / "availability_matrix.csv").open(newline="", encoding="utf-8") as handle:
        availability = list(csv.DictReader(handle))
    aer_metrics = [row for row in metrics if row["configuration_id"] == "aer_noisy_local_warm"]
    assert aer_metrics
    assert all(row["method_id"] == "qiskit_aer_noisy_local" for row in aer_metrics)
    assert all(row["fidelity_class"] == "local_measurement" for row in aer_metrics)
    assert {row["stratum"] for row in aer_metrics} == {"core_q2_q9", "frontier_q10_q16"}
    azizov_metrics = [row for row in metrics if row["method_id"] == "azizov_aer_independent"]
    assert azizov_metrics == []
    azizov_avail = [row for row in availability if row["method_id"] == "azizov_aer_independent"]
    assert azizov_avail
    assert all(int(row["attempted_cells"]) == 162 for row in azizov_avail)
    aer_avail = [row for row in availability if row["configuration_id"] == "aer_noisy_local_warm"]
    assert aer_avail
    assert all(row["method_id"] == "qiskit_aer_noisy_local" for row in aer_avail)
    assert all(int(row["attempted_cells"]) == 204 for row in aer_avail)


def test_validator_rejects_aer_labeled_as_azizov(tmp_path: Path) -> None:
    module = validator_module()
    dest = _copy_artifact(tmp_path)
    path = dest / "per_configuration_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mutated = False
    for row in rows:
        if row["configuration_id"] == "aer_noisy_local_warm":
            row["method_id"] = "azizov_aer_independent"
            row["fidelity_class"] = "independent_reimplementation"
            mutated = True
    assert mutated
    _rewrite(path, rows)
    errors = module.validate(dest)
    assert any("Aer measurement capsule labeled as Azizov predictor" in item for item in errors)


def test_one_cell_status_terminal_failure_dominates() -> None:
    spec = importlib.util.spec_from_file_location(
        "build_simulator_reader_tables_v1",
        ROOT / "benchmark_v1/scripts/build_simulator_reader_tables_v1.py",
    )
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    raw = [
        {
            "precision": "fp64",
            "panel_member_id": "m1",
            "session_id": "s1",
            "repetition": "1",
            "status": "ok",
            "observation_kind": "warm",
            "stratum": "core_q2_q9",
        },
        {
            "precision": "fp64",
            "panel_member_id": "m1",
            "session_id": "s1",
            "repetition": "0",
            "status": "adapter_error",
            "observation_kind": "warm",
            "stratum": "core_q2_q9",
        },
        {
            "precision": "fp64",
            "panel_member_id": "m2",
            "session_id": "s1",
            "repetition": "0",
            "status": "ok",
            "observation_kind": "warm",
            "stratum": "core_q2_q9",
        },
    ]
    reduced = builder.one_cell_status(raw, clock="warm", config_fields=("precision",))
    cells = reduced[("fp64",)]
    assert cells[("m1", "s1")]["status"] == "adapter_error"
    assert cells[("m2", "s1")]["status"] == "ok"
    reversed_raw = list(reversed(raw))
    again = builder.one_cell_status(reversed_raw, clock="warm", config_fields=("precision",))
    assert again[("fp64",)][("m1", "s1")]["status"] == "adapter_error"
