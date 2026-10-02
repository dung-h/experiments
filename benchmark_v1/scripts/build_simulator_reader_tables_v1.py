#!/usr/bin/env python3
"""Materialize conservative, evidence-only simulator reader tables v1.

This builder reads frozen local artifacts only.  It never runs a simulator, starts
CUDA, trains a model, or rewrites raw evidence.  Existing public output is never
overwritten; an explicitly supplied temporary --output-dir is the test-only
exception.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/benchmark_v3/simulator/reader_tables_v1"
PANEL = ROOT / "artifacts/benchmark_v1/sim_common_q16_manifest_20260927/sim_common_q16_manifest.csv"
AER_DIR = ROOT / "artifacts/benchmark_v1/c44_aer_q16_full_panel_evaluation_20260927"
AER_RAW = AER_DIR / "aer_q16_reduced_warm.csv"
AER_CORE_METRICS = AER_DIR / "core_metrics.csv"
AER_CORE_OOF = AER_DIR / "core_oof_predictions.csv"
AZIZOV_DIR = ROOT / "artifacts/benchmark_v1/c77_v5_comparison_cells_v5_20260929/cells/sim_aer_q2_q9_warm_execution/members"
AZIZOV_GBR = AZIZOV_DIR / "azizov_aer_independent_compiled_gbr.csv"
AZIZOV_RIDGE = AZIZOV_DIR / "azizov_aer_independent_logical_ridge.csv"
DENSE_DIR = ROOT / "artifacts/benchmark_v1/cudaq_dense_common_q16_bridge_full_v1_20260928"
DENSE_RAW = DENSE_DIR / "cudaq_dense_common_raw.csv"
DENSE_QA = ROOT / "artifacts/benchmark_v1/cudaq_dense_common_q16_bridge_qa_v1_20260928/qa.json"
MPS_DIR = ROOT / "artifacts/benchmark_v1/cudaq_mps_common_q16_bridge_full_v1_20260928"
MPS_RAW = MPS_DIR / "cudaq_mps_common_raw.csv"
MPS_QA = ROOT / "artifacts/benchmark_v1/cudaq_mps_common_q16_bridge_qa_v1_20260928/qa.json"
CTN_DIR = ROOT / "artifacts/benchmark_v1/cutensornet_common_q16_full_v1_20260928"
CTN_RAW = CTN_DIR / "cutensornet_scalar_common_raw.csv"
CTN_QA = ROOT / "artifacts/benchmark_v1/cutensornet_common_q16_qa_v3_20260928/qa.json"
CTN_INTERFERENCE = CTN_DIR / "MEASUREMENT_INTERFERENCE.json"
CTN_RATIO = ROOT / "artifacts/benchmark_v1/c86_cutensornet_ratio_diagnostic_v2_20260929/paired_ratio_rows.csv"
MAESTRO = ROOT / "artifacts/benchmark_v1/c18_maestro_selector_matrix_20260926/run_manifest.json"
MAESTRO_QA = ROOT / "artifacts/benchmark_v1/c25_qa_aggregate_v4_after_s39b_20260928/c18_cell_summary.csv"
PASQAL = ROOT / "artifacts/benchmark_v1/pasqal_emu_mps_formula_crossed_pilot_v1r1_20260926/run_manifest.json"
REGISTRY = ROOT / "benchmark_v1/registry/method_fidelity_registry_v1.json"
CONTRACT = ROOT / "benchmark_v1/decisions/SIMULATOR_READER_TABLES_AGGREGATION_CONTRACT_V1_20260930.md"
ADJUDICATION = ROOT / "benchmark_v1/S40_S42_SIMULATOR_PROVENANCE_ADJUDICATION_V1_20260928.md"
PROTOCOL = ROOT / "benchmark_v1/protocol/simulator_predictor_cells_v2.json"
FAIL_STATUSES = frozenset({"adapter_error", "quality_failed", "timeout"})
# Registry V1 has no distinct Aer measurement ID. Azizov is the predictor only.
LOCAL_FIDELITY = {
    "qiskit_aer_noisy_local": {
        "reader_label": "Local noisy-Aer warm measurement capsule",
        "fidelity_class": "local_measurement",
        "claim_boundary": (
            "Local engine/configuration measurement, not Azizov predictor "
            "coverage and not archived QPU time."
        ),
    }
}
PINNED = {
    "dense_cells_per_clock": 1224,
    "dense_ok": 1188,
    "dense_adapter_error": 36,
    "mps_cells_per_clock": 612,
    "mps_ok": 573,
    "mps_quality_failed": 21,
    "mps_adapter_error": 18,
    "ctn_cells_per_clock": 612,
    "ctn_ok": 603,
    "ctn_timeout": 9,
    "panel_rows": 204,
    "core_rows": 162,
    "frontier_rows": 42,
}


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, data: list[dict[str, object]]) -> None:
    if not data:
        raise ValueError(f"refusing to write empty table {path.name}")
    fields = list(data[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(data)


def source(path: Path) -> tuple[str, str]:
    return rel(path), sha(path)


def summary(values: list[float]) -> tuple[str, str, str]:
    if not values:
        return "", "", ""
    return f"{median(values):.12g}", f"{mean(values):.12g}", f"{min(values):.12g}"


def registry() -> dict[str, dict[str, str]]:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {item["method_id"]: item for item in payload["methods"]}


def fidelity_card(registry_cards: dict[str, dict[str, str]], method_id: str) -> dict[str, str]:
    if method_id in LOCAL_FIDELITY:
        return LOCAL_FIDELITY[method_id]
    return registry_cards[method_id]


def metric_row(
    *,
    configuration_id: str,
    engine_id: str,
    method_id: str,
    fidelity: dict[str, str],
    stratum: str,
    evaluation_target_clock: str,
    method_output_clock: str,
    values: list[float],
    status_counts: Counter[str],
    source_file: Path,
    quality_policy_id: str,
    comparability: str,
    note: str,
) -> dict[str, object]:
    path, digest = source(source_file)
    attempted = sum(status_counts.values())
    ok = int(status_counts.get("ok", 0))
    med, avg, low = summary(values)
    return {
        "configuration_id": configuration_id,
        "engine_id": engine_id,
        "method_id": method_id,
        "reader_label": fidelity["reader_label"],
        "fidelity_class": fidelity["fidelity_class"],
        "claim_boundary": fidelity["claim_boundary"],
        "stratum": stratum,
        "evaluation_target_clock": evaluation_target_clock,
        "method_output_clock": method_output_clock,
        "quality_policy_id": quality_policy_id,
        "attempted_cells": attempted,
        "ok_cells": ok,
        "failure_cells": attempted - ok,
        "ok_coverage": f"{ok / attempted:.12g}" if attempted else "0",
        "median_seconds": med,
        "mean_seconds": avg,
        "min_seconds": low,
        "comparability": comparability,
        "note": note,
        "source_path": path,
        "source_sha256": digest,
    }


def one_cell_status(
    raw: list[dict[str, str]],
    *,
    clock: str,
    config_fields: tuple[str, ...],
) -> dict[tuple[str, ...], dict[tuple[str, str], dict[str, str]]]:
    """One terminal status per (config, panel member, session) for a single clock.

    Terminal failures dominate any later or earlier ok.  Input order is sorted so
    reruns are byte-stable.
    """
    grouped: dict[tuple[str, ...], dict[tuple[str, str], dict[str, str]]] = defaultdict(dict)
    ordered = sorted(
        raw,
        key=lambda row: (
            row.get("precision", ""),
            row.get("panel_member_id", ""),
            row.get("session_id", ""),
            row.get("repetition", ""),
            row.get("status", ""),
        ),
    )
    for row in ordered:
        if row["observation_kind"] != clock:
            continue
        config = tuple(row[field] for field in config_fields)
        key = (row["panel_member_id"], row["session_id"])
        old = grouped[config].get(key)
        if old is None:
            grouped[config][key] = row
            continue
        old_fail = old["status"] in FAIL_STATUSES
        new_fail = row["status"] in FAIL_STATUSES
        if new_fail and not old_fail:
            grouped[config][key] = row
    return grouped


def build(out: Path) -> None:
    fidelity = registry()
    panel = rows(PANEL)
    if len(panel) != PINNED["panel_rows"]:
        raise ValueError("panel row count drifted")
    members = [row["panel_member_id"] for row in panel]
    if len(members) != len(set(members)):
        raise ValueError("duplicate panel_member_id")
    strata = Counter(row["stratum"] for row in panel)
    if strata["core_q2_q9"] != PINNED["core_rows"] or strata["frontier_q10_q16"] != PINNED["frontier_rows"]:
        raise ValueError("panel stratum counts drifted")
    panel_source, panel_hash = source(PANEL)
    write_csv(
        out / "common_exact_qasm_panel.csv",
        [
            {
                "panel_id": row["panel_id"],
                "panel_member_id": row["panel_member_id"],
                "exact_qasm_sha256": row["qasm_sha256"],
                "hash_alias_count": row["hash_alias_count"],
                "hash_group_id": row["hash_group_id"],
                "family": row["family"],
                "width_qubits": row["width_qubits"],
                "stratum": row["stratum"],
                "source_path": panel_source,
                "source_sha256": panel_hash,
            }
            for row in panel
        ],
    )

    metrics: list[dict[str, object]] = []
    availability: list[dict[str, object]] = []

    def add_avail(
        *,
        configuration_id: str,
        engine_id: str,
        method_id: str,
        status: str,
        attempted: int,
        ok: int,
        src: Path,
        evaluation_target_clock: str,
        method_output_clock: str,
        reason: str,
    ) -> None:
        path, digest = source(src)
        card = fidelity_card(fidelity, method_id)
        availability.append(
            {
                "configuration_id": configuration_id,
                "engine_id": engine_id,
                "method_id": method_id,
                "reader_label": card["reader_label"],
                "fidelity_class": card["fidelity_class"],
                "availability_status": status,
                "panel_id": "sim_common_q16_v1",
                "attempted_cells": attempted,
                "ok_cells": ok,
                "failure_cells": attempted - ok,
                "evaluation_target_clock": evaluation_target_clock,
                "method_output_clock": method_output_clock,
                "reason": reason,
                "source_path": path,
                "source_sha256": digest,
            }
        )

    aer_raw = rows(AER_RAW)
    if len(aer_raw) != PINNED["panel_rows"]:
        raise ValueError("Aer reduced warm capsule is not 204 rows")
    aer_strata = Counter(row["stratum"] for row in aer_raw)
    if aer_strata["core_q2_q9"] != 162 or aer_strata["frontier_q10_q16"] != 42:
        raise ValueError("Aer stratum counts drifted")
    for stratum in ("core_q2_q9", "frontier_q10_q16"):
        part = [row for row in aer_raw if row["stratum"] == stratum]
        metrics.append(
            metric_row(
                configuration_id="aer_noisy_local_warm",
                engine_id="qiskit_aer_noisy",
                method_id="qiskit_aer_noisy_local",
                fidelity=fidelity_card(fidelity, "qiskit_aer_noisy_local"),
                stratum=stratum,
                evaluation_target_clock="warm_execution",
                method_output_clock="warm_execution",
                values=[float(row["observed_seconds"]) for row in part],
                status_counts=Counter(ok=len(part)),
                source_file=AER_RAW,
                quality_policy_id="exact_noisy_aer_configuration",
                comparability="engine_configuration_descriptive_only",
                note="Local noisy-Aer reduced warm measurement capsule; not QPU time and not Azizov predictor coverage.",
            )
        )
    add_avail(
        configuration_id="aer_noisy_local_warm",
        engine_id="qiskit_aer_noisy",
        method_id="qiskit_aer_noisy_local",
        status="available",
        attempted=len(aer_raw),
        ok=len(aer_raw),
        src=AER_RAW,
        evaluation_target_clock="warm_execution",
        method_output_clock="warm_execution",
        reason="Full-panel local noisy-Aer timing capsule. Azizov predictor OOF is a separate core-only configuration.",
    )

    for config_id, path, label in (
        ("azizov_aer_independent_compiled_gbr", AZIZOV_GBR, "compiled GBR"),
        ("azizov_aer_independent_logical_ridge", AZIZOV_RIDGE, "logical ridge"),
    ):
        predictor = rows(path)
        if len(predictor) != PINNED["core_rows"]:
            raise ValueError(f"{path.name} is not 162-row core OOF")
        add_avail(
            configuration_id=config_id,
            engine_id="qiskit_aer_noisy",
            method_id="azizov_aer_independent",
            status="coverage_qualified",
            attempted=len(predictor),
            ok=len(predictor),
            src=path,
            evaluation_target_clock="warm_execution",
            method_output_clock="warm_execution",
            reason=f"Azizov-style {label} OOF is coverage-qualified on the 162 core members only; frontier is blocked, not imputed.",
        )

    def gpu_metrics(
        raw_file: Path,
        *,
        method_id: str,
        engine_id: str,
        clocks: list[tuple[str, str, str]],
        quality: str,
        comparability: str,
        expected_per_clock: int,
        expected_ok: int,
        expected_fail: dict[str, int],
        note: str,
    ) -> None:
        raw = rows(raw_file)
        for raw_clock, output_clock, value_col in clocks:
            reduced = one_cell_status(raw, clock=raw_clock, config_fields=("precision",))
            clock_rows = [cell for cells in reduced.values() for cell in cells.values()]
            if len(clock_rows) != expected_per_clock:
                raise ValueError(f"{raw_file.name} {raw_clock} unique-cell count drifted: {len(clock_rows)}")
            counts = Counter(row["status"] for row in clock_rows)
            if counts.get("ok", 0) != expected_ok:
                raise ValueError(f"{raw_file.name} {raw_clock} ok count drifted: {counts}")
            for status, expected in expected_fail.items():
                if counts.get(status, 0) != expected:
                    raise ValueError(f"{raw_file.name} {raw_clock} {status} drifted: {counts}")
            for config, cells in sorted(reduced.items()):
                precision = config[0]
                configuration_id = f"{engine_id}_{precision}"
                entries = list(cells.values())
                add_avail(
                    configuration_id=configuration_id,
                    engine_id=engine_id,
                    method_id=method_id,
                    status="available_with_terminal_failures" if any(row["status"] != "ok" for row in entries) else "available",
                    attempted=len(entries),
                    ok=sum(row["status"] == "ok" for row in entries),
                    src=raw_file,
                    evaluation_target_clock=output_clock,
                    method_output_clock=output_clock,
                    reason="Separate local engine/precision/clock cells; terminal failures retained; no cross-engine rank.",
                )
                for stratum in ("core_q2_q9", "frontier_q10_q16"):
                    part = [row for row in entries if row["stratum"] == stratum]
                    status_counts = Counter(row["status"] for row in part)
                    values = [
                        float(row[value_col])
                        for row in part
                        if row["status"] == "ok" and row[value_col] != ""
                    ]
                    metrics.append(
                        metric_row(
                            configuration_id=configuration_id,
                            engine_id=engine_id,
                            method_id=method_id,
                            fidelity=fidelity[method_id],
                            stratum=stratum,
                            evaluation_target_clock=output_clock,
                            method_output_clock=output_clock,
                            values=values,
                            status_counts=status_counts,
                            source_file=raw_file,
                            quality_policy_id=quality,
                            comparability=comparability,
                            note=note,
                        )
                    )

    gpu_metrics(
        DENSE_RAW,
        method_id="cudaq_dense_local",
        engine_id="cudaq_nvidia_dense",
        clocks=[("first", "first_execution", "t_first_execution_s"), ("warm", "warm_execution", "t_warm_execution_s")],
        quality="sample_api_fp32_or_fp64",
        comparability="same_engine_precision_clock_only",
        expected_per_clock=PINNED["dense_cells_per_clock"],
        expected_ok=PINNED["dense_ok"],
        expected_fail={"adapter_error": PINNED["dense_adapter_error"]},
        note="Terminal adapter_error cells remain in the denominator; FP32/FP64 and first/warm are never pooled.",
    )
    gpu_metrics(
        MPS_RAW,
        method_id="cudaq_mps_quality_constrained",
        engine_id="cudaq_nvidia_mps",
        clocks=[("first", "first_execution", "t_first_execution_s"), ("warm", "warm_execution", "t_warm_execution_s")],
        quality="fp64_bond16_cutoff1e-10_gesvdj_fidelity_ge_0.99",
        comparability="same_engine_fixed_quality_clock_only",
        expected_per_clock=PINNED["mps_cells_per_clock"],
        expected_ok=PINNED["mps_ok"],
        expected_fail={
            "quality_failed": PINNED["mps_quality_failed"],
            "adapter_error": PINNED["mps_adapter_error"],
        },
        note="MPS runtime is quality-gated; quality_failed and adapter_error remain in the denominator. Not a dense-equivalence claim.",
    )

    ctn = rows(CTN_RAW)
    ctn_cells = one_cell_status(ctn, clock="warm", config_fields=("precision",))
    for config, cells in sorted(ctn_cells.items()):
        precision = config[0]
        entries = list(cells.values())
        if len(entries) != PINNED["ctn_cells_per_clock"]:
            raise ValueError("cuTensorNet unique warm-cell count drifted")
        counts_all = Counter(row["status"] for row in entries)
        if counts_all.get("ok", 0) != PINNED["ctn_ok"] or counts_all.get("timeout", 0) != PINNED["ctn_timeout"]:
            raise ValueError(f"cuTensorNet warm status drifted: {counts_all}")
        add_avail(
            configuration_id=f"cutensornet_scalar_{precision}",
            engine_id="cutensornet_scalar_network",
            method_id="cutensornet_runtime_est",
            status="available_with_planner_timeouts",
            attempted=len(entries),
            ok=int(counts_all["ok"]),
            src=CTN_RAW,
            evaluation_target_clock="warm_execution",
            method_output_clock="selected_plan_runtime_estimate",
            reason="Native pair only against matching selected-plan warm scalar contraction; path/build are descriptive telemetry.",
        )
        for stratum in ("core_q2_q9", "frontier_q10_q16"):
            part = [row for row in entries if row["stratum"] == stratum]
            counts = Counter(row["status"] for row in part)
            for output_clock, column in (
                ("warm_execution", "t_warm_contract_s"),
                ("selected_plan_runtime_estimate", "runtime_est_s"),
            ):
                values = [
                    float(row[column])
                    for row in part
                    if row["status"] == "ok" and row[column] != ""
                ]
                metrics.append(
                    metric_row(
                        configuration_id=f"cutensornet_scalar_{precision}",
                        engine_id="cutensornet_scalar_network",
                        method_id="cutensornet_runtime_est",
                        fidelity=fidelity["cutensornet_runtime_est"],
                        stratum=stratum,
                        evaluation_target_clock="warm_execution",
                        method_output_clock=output_clock,
                        values=values,
                        status_counts=counts,
                        source_file=CTN_RAW,
                        quality_policy_id="same_selected_plan_workspace_precision",
                        comparability="paired_native_estimate_vs_matching_warm_scalar_contraction_only",
                        note="Same selected plan/workspace/precision/session required. Not a generic end-to-end simulator score.",
                    )
                )

    metrics.sort(
        key=lambda row: (
            str(row["configuration_id"]),
            str(row["stratum"]),
            str(row["evaluation_target_clock"]),
            str(row["method_output_clock"]),
        )
    )
    availability.sort(
        key=lambda row: (
            str(row["configuration_id"]),
            str(row["evaluation_target_clock"]),
            str(row["method_output_clock"]),
        )
    )
    write_csv(out / "availability_matrix.csv", availability)
    write_csv(out / "per_configuration_metrics.csv", metrics)

    appendix: list[dict[str, object]] = []
    ratio_rows = rows(CTN_RATIO)
    ratio_path, ratio_hash = source(CTN_RATIO)
    for stratum in ("core_q2_q9", "frontier_q10_q16"):
        part = [row for row in ratio_rows if row["stratum"] == stratum and row["status"] == "ok"]
        appendix.append(
            {
                "section": "paired_native_estimator",
                "configuration_id": "cutensornet_scalar_complex64",
                "method_id": "cutensornet_runtime_est",
                "stratum": stratum,
                "metric_id": "runtime_est_to_matching_warm_contraction_ratio",
                "evaluation_target_clock": "warm_execution",
                "method_output_clock": "selected_plan_runtime_estimate",
                "n_successful_pairs": len(part),
                "median_value": f"{median(float(row['ratio']) for row in part):.12g}" if part else "",
                "status": "paired_diagnostic_only",
                "note": "Selected-plan RUNTIME_EST versus matching warm scalar contraction only.",
                "source_path": ratio_path,
                "source_sha256": ratio_hash,
            }
        )
    ctn_path, ctn_hash = source(CTN_RAW)
    for metric_id, column, clock, status in (
        ("network_build_seconds", "t_network_build_s", "network_build", "descriptive_telemetry"),
        ("path_search_seconds", "t_path_search_s", "path_search", "descriptive_interference_flagged"),
    ):
        values = [
            float(row[column])
            for row in ctn
            if row["observation_kind"] == "first" and row["status"] == "ok" and row[column] != ""
        ]
        appendix.append(
            {
                "section": "descriptive_telemetry",
                "configuration_id": "cutensornet_scalar_complex64",
                "method_id": "cutensornet_runtime_est",
                "stratum": "all_panel",
                "metric_id": metric_id,
                "evaluation_target_clock": "not_an_estimator_target",
                "method_output_clock": clock,
                "n_successful_pairs": len(values),
                "median_value": f"{median(values):.12g}" if values else "",
                "status": status,
                "note": "Not ranked or summarized as a predictor error; S40--S42 applies.",
                "source_path": ctn_path,
                "source_sha256": ctn_hash,
            }
        )
    for config_id, path, label in (
        ("azizov_aer_independent_compiled_gbr", AZIZOV_GBR, "compiled GBR"),
        ("azizov_aer_independent_logical_ridge", AZIZOV_RIDGE, "logical ridge"),
    ):
        predictor = rows(path)
        errors = [abs(float(row["predicted_seconds"]) - float(row["actual_seconds"])) for row in predictor]
        pred_path, pred_hash = source(path)
        appendix.append(
            {
                "section": "coverage_qualified_predictor",
                "configuration_id": config_id,
                "method_id": "azizov_aer_independent",
                "stratum": "core_q2_q9",
                "metric_id": "oof_mae_seconds",
                "evaluation_target_clock": "warm_execution",
                "method_output_clock": "warm_execution",
                "n_successful_pairs": len(predictor),
                "median_value": f"{mean(errors):.12g}",
                "status": "coverage_qualified_core_only",
                "note": f"Azizov-style {label} OOF MAE on 162 core members; frontier is not imputed.",
                "source_path": pred_path,
                "source_sha256": pred_hash,
            }
        )
    write_csv(out / "native_method_appendix.csv", appendix)

    blocked_specs = [
        (
            "azizov_aer_independent",
            "qiskit_aer_noisy",
            "frontier_q10_q16 of sim_common_q16_v1",
            "Azizov-style predictor OOF is core-only; frontier is blocked rather than imputed",
            AZIZOV_GBR,
        ),
        (
            "maestro_formula_reimplementation",
            "maestro_paper_described_candidates",
            "sim_common_q16_v1",
            "missing Composer estimator implementation and complete held-out common-panel candidate matrix",
            MAESTRO,
        ),
        (
            "maestro_paper_described_reimplementation",
            "maestro_paper_described_candidates",
            "sim_common_q16_v1",
            "candidate grid is archived-QPU-context evidence, not a complete common-panel selector evaluation",
            MAESTRO_QA,
        ),
        (
            "mali_simulator_pretraining",
            "mali_simulator_source_native",
            "sim_common_q16_v1",
            "no row-aligned frozen local common-panel evidence supplied",
            PROTOCOL,
        ),
        (
            "mali_structural_dag_adaptation",
            "ma_li_structural_sim_adaptation",
            "sim_common_q16_v1",
            "no row-aligned frozen local common-panel evidence supplied",
            PROTOCOL,
        ),
        (
            "family_aware_residual",
            "local_approximation_ladder",
            "sim_common_q16_v1",
            "quality-ladder labels absent on common local panel",
            PROTOCOL,
        ),
        (
            "pasqal_emu_mps",
            "pasqal_emu_mps",
            "separate_analog_pulse_layout_panel",
            "separate analog pulse/layout pilot; digital OpenQASM mapping is prohibited",
            PASQAL,
        ),
    ]
    blocked: list[dict[str, object]] = []
    for method_id, engine_id, scope, reason, path in blocked_specs:
        file_path, digest = source(path)
        card = fidelity[method_id]
        blocked.append(
            {
                "method_id": method_id,
                "reader_label": card["reader_label"],
                "fidelity_class": card["fidelity_class"],
                "claim_boundary": card["claim_boundary"],
                "engine_id": engine_id,
                "panel_scope": scope,
                "availability_status": "blocked_or_unavailable",
                "evaluation_target_clock": "",
                "method_output_clock": "",
                "reason": reason,
                "source_path": file_path,
                "source_sha256": digest,
            }
        )
    write_csv(out / "blocked_unavailable.csv", blocked)

    inputs = [
        PANEL,
        AER_RAW,
        AER_CORE_METRICS,
        AER_CORE_OOF,
        AZIZOV_GBR,
        AZIZOV_RIDGE,
        DENSE_RAW,
        DENSE_QA,
        MPS_RAW,
        MPS_QA,
        CTN_RAW,
        CTN_QA,
        CTN_RATIO,
        CTN_INTERFERENCE,
        MAESTRO,
        MAESTRO_QA,
        PASQAL,
        REGISTRY,
        CONTRACT,
        ADJUDICATION,
        PROTOCOL,
    ]
    hashes = {rel(path): sha(path) for path in inputs}
    (out / "source_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = """# Simulator reader tables v1

This evidence-only reader pack preserves local simulator configurations and their
native clocks. It contains no QPU rows and no flat cross-engine or cross-clock
ranking. `sim_common_q16_v1` contains 204 unique panel members: 162 core and 42
frontier. Declared QASM hash aliases are allowed (191 unique `qasm_sha256`
values). Identity is the panel-member row plus that row's exact QASM SHA-256.
Terminal failures remain in denominators.

The Aer reduced-warm capsule is a local noisy-Aer measurement (204/204) under
reader overlay `qiskit_aer_noisy_local` (`local_measurement`). Registry V1 has
no distinct Aer measurement ID; `azizov_aer_independent` is reserved for the
Azizov-style predictor and is coverage-qualified on the 162 core members only;
frontier predictor rows are blocked, not imputed. CUDA-Q first and warm clocks, and
FP32/FP64 dense configurations, are separate. Unique-cell pins: dense 1224 per
clock (1188 ok, 36 adapter_error); MPS 612 per clock (573 ok, 21 quality_failed,
18 adapter_error); cuTensorNet 612 per clock (603 ok, 9 timeout). MPS is
quality-qualified (FP64, bond 16, cutoff 1e-10, gesvdj). cuTensorNet RUNTIME_EST
is only a selected-plan pair against matching warm scalar contraction;
network-build and interference-flagged path-search values are appendix
telemetry. Maestro and unavailable adaptations are blocked rather than
substituted; Pasqal is kept on its separate analog pulse/layout panel.
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    outputs = [
        "availability_matrix.csv",
        "common_exact_qasm_panel.csv",
        "per_configuration_metrics.csv",
        "native_method_appendix.csv",
        "blocked_unavailable.csv",
        "source_hashes.json",
        "REPORT.md",
    ]
    manifest = {
        "artifact_id": "simulator-reader-tables-v1",
        "contract": rel(CONTRACT),
        "panel_id": "sim_common_q16_v1",
        "panel_rows": PINNED["panel_rows"],
        "core_rows": PINNED["core_rows"],
        "frontier_rows": PINNED["frontier_rows"],
        "unique_panel_member_ids": PINNED["panel_rows"],
        "unique_qasm_sha256": len({row["qasm_sha256"] for row in panel}),
        "qasm_hash_aliases_allowed": True,
        "qpu_rows": 0,
        "timing_or_training_performed": False,
        "flat_cross_clock_rank_emitted": False,
        "unique_cells": {
            "cudaq_dense_per_clock": {
                "cells": PINNED["dense_cells_per_clock"],
                "ok": PINNED["dense_ok"],
                "adapter_error": PINNED["dense_adapter_error"],
            },
            "cudaq_mps_per_clock": {
                "cells": PINNED["mps_cells_per_clock"],
                "ok": PINNED["mps_ok"],
                "quality_failed": PINNED["mps_quality_failed"],
                "adapter_error": PINNED["mps_adapter_error"],
            },
            "cutensornet_per_clock": {
                "cells": PINNED["ctn_cells_per_clock"],
                "ok": PINNED["ctn_ok"],
                "timeout": PINNED["ctn_timeout"],
            },
        },
        "outputs": {name: sha(out / name) for name in outputs},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    supplied = "--output-dir" in sys.argv
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        test_temp = supplied and (out.name.startswith("tmp") or "pytest" in str(out) or "test" in out.name)
        if not test_temp:
            raise SystemExit(
                "refusing to overwrite existing public reader_tables_v1; "
                "use an explicit temporary --output-dir for tests"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    build(out)
    print(
        json.dumps(
            {"status": "PASS", "output_dir": rel(out), "timing_or_training_performed": False},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
