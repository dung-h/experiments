#!/usr/bin/env python3
"""Check source/output pins and core scope invariants for reader tables v2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK = ROOT / "artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v2"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate(pack: Path) -> dict[str, object]:
    errors: list[str] = []
    manifest_path = pack / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "FAIL", "errors": ["manifest.json is missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    is_final = manifest.get("artifact_id") == "two-domain-benchmark-reader-tables-v3"
    if manifest.get("artifact_id") not in {"two-domain-benchmark-reader-tables-v2", "two-domain-benchmark-reader-tables-v3"}:
        errors.append("unexpected artifact_id")
    expected_status = "BUILT_FROM_PINNED_INPUTS_MAESTRO_PILOT_FAILED" if is_final else "BUILT_FROM_PINNED_INPUTS_MAESTRO_NOT_STARTED"
    if manifest.get("status") != expected_status:
        errors.append("unexpected build status")
    if manifest.get("current_json_modified") is not False:
        errors.append("CURRENT mutation is not explicitly false")
    if manifest.get("historical_packs_modified") is not False:
        errors.append("historical-pack mutation is not explicitly false")
    if manifest.get("no_flat_cross_domain_or_cross_clock_rank") is not True:
        errors.append("cross-domain/cross-clock rank boundary is missing")
    historical_code_verified: list[str] = []
    for name, expected in manifest.get("source_hashes", {}).items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            # v2 pins the original generator, not its subsequent extension.
            # Only this exact archived generator may satisfy that historical pin.
            archive = ROOT / "work/repo_finalization/release_candidate" / name
            if (not is_final and name == "benchmark_v1/scripts/build_two_domain_scorecard_v2.py"
                    and expected == "8a91eaa9e709d10fe5a17b195cb66ec346a8ea7a95b1bd603b15f11bd6598077"
                    and archive.is_file() and sha(archive) == expected):
                historical_code_verified.append(str(archive.relative_to(ROOT)))
            else:
                errors.append(f"source hash mismatch: {name}")
    for name, expected in manifest.get("outputs", {}).items():
        path = pack / name
        if not path.is_file() or sha(path) != expected:
            errors.append(f"output hash mismatch: {name}")

    coverage = read_csv(pack / "qpu/archived_method_coverage.csv")
    if len(coverage) != 24 or any(int(row["assigned_n"]) != 8767 for row in coverage):
        errors.append("C4 coverage is not 24 methods × 8,767 assigned rows")
    pair_rows = read_csv(pack / "qpu/archived_method_pairwise_comparisons.csv")
    if len(pair_rows) != 72:
        errors.append("C4 source-local pair table does not contain 72 pairs")
    simulator = read_csv(pack / "simulator/aer_mali_graph_c3a_diagnostic.csv")
    if len(simulator) != 1:
        errors.append("C3a diagnostic row count differs from one")
    else:
        row = simulator[0]
        if row.get("n") != "162" or row.get("unique_qasm_hashes") != "150":
            errors.append("C3a population/hash count mismatch")
        if row.get("registry_method_id") or row.get("fidelity_class"):
            errors.append("C3a supplementary row must not claim a registered fidelity class")
        if row.get("metrics_recomputed_from_oof_rows") != "True":
            errors.append("C3a metrics are not marked raw-derived")
    maestro = manifest.get("simulator", {}).get("maestro", {})
    expected_maestro_status = "pilot_gate_failed" if is_final else "not_started_no_partial_run_manifest"
    if maestro.get("status_at_build") != expected_maestro_status or maestro.get("score") is not None:
        errors.append("Maestro no-score state is missing or fabricated")
    if is_final:
        spec = importlib.util.spec_from_file_location("report_builder", ROOT / "benchmark_v1/scripts/build_two_domain_scorecard_v2.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        authority = manifest["report_contract"]
        contract_path = ROOT / authority["path"]
        if contract_path.resolve() != builder.DEFAULT_REPORT.resolve() or sha(contract_path) != authority["sha256"]:
            errors.append("final report contract identity mismatch")
        contract = builder.load_report_contract(contract_path)
        if (manifest.get("scientific_coverage") != "PARTIAL"
                or manifest.get("training_or_timing_performed") is not False
                or manifest.get("public_release_approved") is not False
                or manifest.get("wave4_five_fold_training_authorized") is not False
                or manifest.get("historical_availability_tables_are_snapshots") is not True):
            errors.append("final scope/authority boundary mismatch")
        for name, expected in contract["source_hashes"].items():
            if manifest["source_hashes"].get(name) != expected:
                errors.append(f"final manifest omitted or changed authority pin: {name}")
        actual_maestro, pilot_rows = builder.maestro_terminal_evidence(contract)
        if maestro != actual_maestro:
            errors.append("Maestro manifest counters/status differ from raw evidence")
        summary = read_csv(pack / "simulator/maestro_pilot_summary.csv")
        expected_rows = [{k: str(v) for k, v in row.items()} for row in pilot_rows]
        if summary != expected_rows:
            errors.append("Maestro pilot table differs from raw-recomputed medians/MADs")
        if read_csv(pack / "method_scope.csv") != [{k: str(v) for k, v in row.items()} for row in contract["method_scope"]]:
            errors.append("original approach scope table differs from report contract")
        if read_csv(pack / "qpu/source_native_scope.csv") != [{k: str(v) for k, v in row.items()} for row in contract["source_native_appendix"]]:
            errors.append("source-native scope pointers differ from report contract")
        aer_rows, _ = builder.aer_diagnostic()
        if simulator != [{k: str(v) for k, v in row.items()} for row in aer_rows]:
            errors.append("Aer diagnostic table differs from raw OOF metrics")
        copied_pairs = [(builder.C4 / src, f"qpu/{dst}") for src, dst in zip(builder.C4_FILES, (
            "archived_method_coverage.csv", "archived_method_source_metrics.csv", "archived_method_pairwise_comparisons.csv"))]
        copied_pairs += [(builder.LEARNED / src, f"qpu/{dst}") for src, dst in (
            ("own_success_metrics.csv", "unified_learned_metrics.csv"), ("pair_metrics.csv", "unified_learned_pairs.csv"),
            ("bootstrap.csv", "unified_learned_bootstrap.csv"), ("coverage.csv", "unified_learned_coverage.csv"),
            ("source_stratified.csv", "unified_learned_source_metrics.csv"), ("unavailable.csv", "unified_learned_unavailable.csv"),
            ("tail_diagnostics.csv", "unified_learned_tail_diagnostics.csv"))]
        copied_pairs += [(builder.SIM_PACK / name, f"simulator/{name}") for name in builder.SIM_FILES]
        for source, name in copied_pairs:
            if sha(pack / name) != sha(source):
                errors.append(f"final table is not the pinned source copy: {name}")
        required_outputs = set(builder.OUTPUTS) | {name for _, name in copied_pairs} | {
            "simulator/aer_mali_graph_c3a_by_fold.csv", "simulator/maestro_pilot_summary.csv", "method_scope.csv", "qpu/source_native_scope.csv"}
        if set(manifest["outputs"]) != required_outputs:
            errors.append("final output inventory mismatch")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "source_count": len(manifest.get("source_hashes", {})),
        "output_count": len(manifest.get("outputs", {})),
        "qpu_methods": len(coverage),
        "qpu_source_pairs": len(pair_rows),
        "c3a_rows": manifest.get("simulator", {}).get("c3a_oof_rows"),
        "maestro_status": maestro.get("status_at_build"),
        "historical_generator_verified_at": historical_code_verified,
    }


def validate(pack: Path) -> dict[str, object]:
    try:
        return _validate(pack)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "FAIL", "errors": [f"invalid or missing reader evidence: {exc}"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    args = parser.parse_args()
    pack = args.pack if args.pack.is_absolute() else ROOT / args.pack
    result = validate(pack)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
