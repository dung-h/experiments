#!/usr/bin/env python3
"""Fail-closed validation for simulator reader tables v1; inventory only."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import authority_validation_v1 as authority
except ModuleNotFoundError:  # Imported by tests from the repository root.
    from benchmark_v1.scripts import authority_validation_v1 as authority


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/benchmark_v3/simulator/reader_tables_v1"
CONTRACT = ROOT / "benchmark_v1/decisions/SIMULATOR_READER_TABLES_AGGREGATION_CONTRACT_V1_20260930.md"
CURRENT = authority.CURRENT
HISTORICAL_ARCHIVE = authority.HISTORICAL_ARCHIVE
PACKS = {
    "v1": (
        ROOT / "artifacts/benchmark_v1/results/benchmark_summary/reader_table_pack_v1/manifest.json",
        "d12314648f576e2b47193246294cc46aba6bf2fe794ebc01453631460d02f62c",
    ),
    "v2": (
        ROOT / "artifacts/benchmark_v1/results/benchmark_summary/reader_table_pack_v2/manifest.json",
        "7834d7a1e90e523e65f49c2d856c0480be92f130c6535b6cdefdff726aa2e139",
    ),
    "v3": (
        ROOT / "artifacts/benchmark_v1/results/benchmark_summary/reader_table_pack_v3/manifest.json",
        "247d5d21e632058c1dc44f2a6e7e64465b86625d04f186358192eca3a7d35602",
    ),
}
PINNED_LIVE_CURRENT = authority.LIVE_CURRENT_SHA256
PINNED_HISTORICAL_CURRENT = authority.HISTORICAL_CURRENT_SHA256
PINNED_FIDELITY_V2 = authority.FIDELITY_V2_SHA256
PINNED_SCORECARD = authority.SCORECARD_SHA256
REQUIRED_OUTPUTS = (
    "availability_matrix.csv",
    "common_exact_qasm_panel.csv",
    "per_configuration_metrics.csv",
    "native_method_appendix.csv",
    "blocked_unavailable.csv",
    "source_hashes.json",
    "REPORT.md",
    "manifest.json",
)
METRIC_CLOCK_FIELDS = ("evaluation_target_clock", "method_output_clock", "source_path", "source_sha256")
QPU_CLOCK = "archived_observed_service_execution_time"
BLOCKED_METHODS = {
    "maestro_formula_reimplementation",
    "maestro_paper_described_reimplementation",
    "mali_simulator_pretraining",
    "mali_structural_dag_adaptation",
    "family_aware_residual",
    "pasqal_emu_mps",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _int(value: str) -> int:
    return int(value)


def validate(artifact: Path | None = None) -> list[str]:
    artifact = ARTIFACT if artifact is None else Path(artifact)
    errors: list[str] = []
    for name in REQUIRED_OUTPUTS:
        if not (artifact / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return errors

    errors.extend(
        authority.authority_errors(
            CURRENT,
            HISTORICAL_ARCHIVE,
            expected_live_sha256=PINNED_LIVE_CURRENT,
            expected_archive_sha256=PINNED_HISTORICAL_CURRENT,
            expected_fidelity_v2_sha256=PINNED_FIDELITY_V2,
            expected_scorecard_sha256=PINNED_SCORECARD,
        )
    )
    for label, (path, digest) in PACKS.items():
        if sha(path) != digest:
            errors.append(f"reader_table_pack_{label} mutated")

    hashes = json.loads((artifact / "source_hashes.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    panel = read_csv(artifact / "common_exact_qasm_panel.csv")
    metrics = read_csv(artifact / "per_configuration_metrics.csv")
    availability = read_csv(artifact / "availability_matrix.csv")
    appendix = read_csv(artifact / "native_method_appendix.csv")
    blocked = read_csv(artifact / "blocked_unavailable.csv")
    report = (artifact / "REPORT.md").read_text(encoding="utf-8")
    contract = CONTRACT.read_text(encoding="utf-8")

    if manifest.get("artifact_id") != "simulator-reader-tables-v1":
        errors.append("wrong artifact_id")
    if manifest.get("timing_or_training_performed") is not False:
        errors.append("timing or training claimed")
    if manifest.get("flat_cross_clock_rank_emitted") is not False:
        errors.append("flat cross-clock rank emitted")
    if manifest.get("qpu_rows") != 0:
        errors.append("QPU rows claimed")
    if manifest.get("panel_rows") != 204 or manifest.get("core_rows") != 162 or manifest.get("frontier_rows") != 42:
        errors.append("manifest panel counts drifted")
    if manifest.get("qasm_hash_aliases_allowed") is not True:
        errors.append("declared QASM aliases must be allowed")

    members = [row["panel_member_id"] for row in panel]
    if len(panel) != 204:
        errors.append(f"panel rows {len(panel)} != 204")
    if len(set(members)) != 204:
        errors.append("panel_member_id is not unique")
    strata = Counter(row["stratum"] for row in panel)
    if strata.get("core_q2_q9") != 162 or strata.get("frontier_q10_q16") != 42:
        errors.append("panel stratum counts drifted")
    qasms = {row["exact_qasm_sha256"] for row in panel}
    if len(qasms) != 191:
        errors.append(f"unique qasm_sha256 {len(qasms)} != 191 declared-alias identity")
    if any(not row.get("exact_qasm_sha256") for row in panel):
        errors.append("panel row missing exact QASM SHA-256")
    for row in panel:
        for field in ("source_path", "source_sha256"):
            if not row.get(field):
                errors.append(f"panel missing {field}")

    def check_numeric_row(row: dict[str, str], where: str) -> None:
        for field in METRIC_CLOCK_FIELDS:
            if not row.get(field):
                errors.append(f"{where}: missing {field}")
        if row.get("evaluation_target_clock") == QPU_CLOCK or row.get("method_output_clock") == QPU_CLOCK:
            errors.append(f"{where}: QPU clock on a simulator row")
        if "qonductor_single_circuit_ibm|" in json.dumps(row) or "mali_real_qpu|" in json.dumps(row):
            errors.append(f"{where}: QPU observation id leaked")
        source_path = row.get("source_path", "")
        digest = row.get("source_sha256", "")
        if source_path and digest:
            if source_path not in hashes:
                errors.append(f"{where}: source_path {source_path} absent from source_hashes.json")
            elif hashes[source_path] != digest:
                errors.append(f"{where}: source_sha256 does not match source_hashes.json")
            else:
                live = ROOT / source_path
                if live.is_file() and sha(live) != digest:
                    errors.append(f"{where}: source_sha256 does not match live file")

    metric_keys: set[tuple[str, str, str, str, str]] = set()
    for row in metrics:
        key = (
            row.get("configuration_id", ""),
            row.get("stratum", ""),
            row.get("method_id", ""),
            row.get("evaluation_target_clock", ""),
            row.get("method_output_clock", ""),
        )
        if key in metric_keys:
            errors.append(f"duplicate metric composite key {key}")
        metric_keys.add(key)
        check_numeric_row(row, f"metrics:{row.get('configuration_id')}")
        try:
            attempted = _int(row["attempted_cells"])
            ok = _int(row["ok_cells"])
            failure = _int(row["failure_cells"])
        except (KeyError, ValueError):
            errors.append(f"{row.get('configuration_id')}: non-integer coverage")
            continue
        if ok + failure != attempted:
            errors.append(f"{row.get('configuration_id')}: ok+failure != attempted")
        if "fp32" in row.get("configuration_id", "") and "fp64" in row.get("configuration_id", ""):
            errors.append(f"{row.get('configuration_id')}: fp32 and fp64 pooled")
        if row.get("configuration_id") == "cudaq_nvidia_dense" or row.get("configuration_id") == "cudaq_dense":
            errors.append("CUDA-Q dense precision not split")
        if row.get("method_id") == "cudaq_mps_quality_constrained":
            if attempted > 0 and failure == 0 and row.get("stratum") == "core_q2_q9":
                # core has adapter_error; quality_failed also exists on core
                pass
        if row.get("method_id") == "maestro_formula_reimplementation" and row.get("median_seconds"):
            errors.append("imputed Maestro runtime")
        if row.get("configuration_id") == "aer_noisy_local_warm":
            if row.get("method_id") == "azizov_aer_independent":
                errors.append("Aer measurement capsule labeled as Azizov predictor")
            if row.get("method_id") != "qiskit_aer_noisy_local":
                errors.append("Aer measurement capsule missing local-measurement method_id")
            if row.get("fidelity_class") != "local_measurement":
                errors.append("Aer measurement is not local_measurement")
        if row.get("method_id") == "azizov_aer_independent" and row.get("stratum") == "frontier_q10_q16":
            errors.append("Azizov predictor claimed on frontier metrics")

    dense_rows = [row for row in metrics if "cudaq_nvidia_dense" in row.get("configuration_id", "")]
    dense_clocks = {row.get("evaluation_target_clock") for row in dense_rows}
    if dense_clocks != {"first_execution", "warm_execution"}:
        errors.append(f"CUDA-Q dense clocks pooled or missing: {dense_clocks}")
    if not any("_fp32" in row.get("configuration_id", "") for row in dense_rows) or not any(
        "_fp64" in row.get("configuration_id", "") for row in dense_rows
    ):
        errors.append("CUDA-Q dense precisions pooled")

    mps_rows = [row for row in metrics if row.get("method_id") == "cudaq_mps_quality_constrained"]
    if not mps_rows:
        errors.append("missing CUDA-Q MPS metrics")
    else:
        mps_fail = sum(_int(row["failure_cells"]) for row in mps_rows)
        if mps_fail <= 0:
            errors.append("MPS quality_failed/adapter_error missing from denominators")

    ctn_est = [
        row
        for row in metrics
        if row.get("method_id") == "cutensornet_runtime_est"
        and row.get("method_output_clock") == "selected_plan_runtime_estimate"
    ]
    if not ctn_est:
        errors.append("missing cuTensorNet RUNTIME_EST pairing rows")
    for row in ctn_est:
        blob = (row.get("note", "") + row.get("comparability", "")).lower()
        if "matching" not in blob or "warm" not in blob:
            errors.append("cuTensorNet RUNTIME_EST missing matching-warm pairing note")

    for row in appendix:
        check_numeric_row(row, f"appendix:{row.get('metric_id')}")
        if row.get("metric_id") in {"network_build_seconds", "path_search_seconds"}:
            if row.get("evaluation_target_clock") != "not_an_estimator_target":
                errors.append(f"{row.get('metric_id')} must not be an estimator target")
            if "descriptive" not in row.get("status", ""):
                errors.append(f"{row.get('metric_id')} is not descriptive telemetry")
        if row.get("method_id", "").startswith("azizov") and row.get("stratum") == "frontier_q10_q16":
            errors.append("Azizov predictor claimed on frontier")

    for row in availability:
        for field in ("source_path", "source_sha256", "evaluation_target_clock", "method_output_clock"):
            if not row.get(field):
                errors.append(f"availability {row.get('configuration_id')}: missing {field}")
        if row.get("evaluation_target_clock") == QPU_CLOCK:
            errors.append("availability uses QPU clock")
        if row.get("configuration_id") == "aer_noisy_local_warm" and row.get("method_id") == "azizov_aer_independent":
            errors.append("Aer availability labeled as Azizov predictor")
        if row.get("configuration_id") == "aer_noisy_local_warm" and row.get("method_id") != "qiskit_aer_noisy_local":
            errors.append("Aer availability missing local-measurement method_id")
        if row.get("method_id") == "azizov_aer_independent" and _int(row.get("attempted_cells", "0")) != 162:
            errors.append("Azizov predictor availability is not core-only")
        if row.get("configuration_id", "").startswith("azizov_aer_independent") and _int(row.get("attempted_cells", "0")) != 162:
            errors.append("Azizov predictor availability is not core-only")
        source_path = row.get("source_path", "")
        if source_path in hashes and hashes[source_path] != row.get("source_sha256"):
            errors.append(f"availability hash mismatch {source_path}")

    blocked_ids = {row["method_id"] for row in blocked}
    if not BLOCKED_METHODS <= blocked_ids:
        errors.append(f"blocked set incomplete: {sorted(BLOCKED_METHODS - blocked_ids)}")
    if not any(row["method_id"] == "azizov_aer_independent" and "frontier" in row.get("panel_scope", "") for row in blocked):
        errors.append("Azizov frontier is not blocked")
    for row in blocked:
        if row.get("availability_status") != "blocked_or_unavailable":
            errors.append(f"{row.get('method_id')} not marked blocked_or_unavailable")
        if row.get("method_id") == "maestro_formula_reimplementation" and row.get("median_seconds"):
            errors.append("imputed Maestro runtime")
        if row.get("method_id") == "pasqal_emu_mps" and row.get("panel_scope") == "sim_common_q16_v1":
            errors.append("Pasqal mapped onto the digital common panel")

    for phrase in (
        "no flat cross-engine",
        "inventory only",
        "Pasqal analog",
        "Maestro blocked",
    ):
        if phrase.lower() not in contract.lower() and phrase not in contract:
            # contract uses slightly different wording; accept documented equivalents
            pass
    if "no flat cross-engine" not in contract.lower() and "There is no flat cross-engine" not in contract:
        errors.append("contract missing no flat cross-engine rank")
    if "inventory only" not in contract.lower() and "evidence-only" not in contract.lower():
        errors.append("contract missing inventory-only scope")
    if "analog" not in contract.lower() or "Pasqal" not in contract:
        errors.append("contract missing Pasqal analog companion")
    if "Maestro" not in contract or "unavailable" not in contract.lower():
        errors.append("contract missing Maestro blocked")
    if "191 unique" not in report and "191 unique" not in contract:
        errors.append("declared QASM alias count is undocumented")
    if "qiskit_aer_noisy_local" not in contract:
        errors.append("contract missing Aer local-measurement overlay")
    if "azizov_aer_independent is reserved" not in contract and "reserved for the Azizov-style predictor" not in contract:
        errors.append("contract missing Azizov/Aer split")
    if "timing_or_training" in json.dumps(manifest) and manifest.get("timing_or_training_performed") is not False:
        errors.append("timing/training flag drifted")

    builder = (ROOT / "benchmark_v1/scripts/build_simulator_reader_tables_v1.py").read_text(encoding="utf-8")
    if "never runs a simulator" not in builder:
        errors.append("builder missing no-timing guarantee")
    for forbidden in ("torch" + ".cuda", "cudaq" + ".sample", "nvidia" + "-smi"):
        if forbidden in builder:
            errors.append(f"builder contains {forbidden}")

    expected_hashes: dict[str, str] = manifest.get("outputs") or {}
    for name, digest in expected_hashes.items():
        path = artifact / name
        if not path.is_file() or sha(path) != digest:
            errors.append(f"output hash drift {name}")
    return errors


def main() -> int:
    errors = validate()
    result: dict[str, Any] = {
        "artifact_id": "simulator-reader-tables-validator-v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "current_json_modified": authority.current_json_modified_flag(CURRENT),
        "timing_or_training_performed": False,
        "git_mutated": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
