#!/usr/bin/env python3
"""Read-only evidence/table replay; --seal only prints a new manifest patch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import interpret_frozen_benchmark as build

ROOT, OUT = build.ROOT, build.OUT

HISTORICAL = [
    "artifacts/mali/graph_intervention_audit_20260921/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/topology_controls_validation/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/topology_controls_validation/summary.json",
    "artifacts/validation/mali_physical_dag_v1/feature_ablation/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/feature_ablation/metrics.csv",
    "artifacts/validation/mali_physical_dag_v1/critical_subgraph_evaluation/REPORT.md",
    "artifacts/validation/mali_physical_dag_v1/critical_subgraph_evaluation/metrics.csv",
    "artifacts/validation/mali_physical_dag_v1/qwalk_forensics/REPORT.md",
    "artifacts/validation/mali_strict_transfer_logical_v1/evaluation/REPORT.md",
    "artifacts/validation/mali_strict_transfer_logical_v1/evaluation/summary.json",
    "artifacts/mali/MA_LI_TRANSFER_BASELINE_COMPARISON.md",
    "artifacts/mali/MA_LI_REPLICATION_REPORT.md",
    "artifacts/mali/full_result.json",
    "artifacts/cudaq_runtime/launch_scaling_v1_20260922/analysis/REPORT.md",
    "artifacts/cudaq_runtime/launch_scaling_v1_20260922/analysis/shot_sweep.csv",
    "experiments/simulator_runtime_v2/README.md",
    "artifacts/benchmark_deep_dive/tables/qpu_shared_reference_comparisons_v1.csv",
    "artifacts/benchmark_deep_dive/tables/qpu_source_balanced_paired_v2.csv",
    "artifacts/benchmark_deep_dive/tables/analytical_controls_v1.csv",
    "artifacts/benchmark_deep_dive/tables/analytical_shared_comparisons_v1.csv",
    "artifacts/benchmark_deep_dive/tables/simulator_paired_v1.csv",
    "artifacts/benchmark_deep_dive/tables/mps_selection_safety_v1.csv",
    "artifacts/real_qpu/qualified_logical_panel/analytical/qcre_source_receipt.json",
    "artifacts/real_qpu/qualified_logical_panel/analytical/attempts.csv",
    "artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/reader_tables/paired_comparisons.csv",
    "artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/reader_tables/source_metrics.csv",
    "artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/reader_tables/input_support.csv",
    "benchmark_v1/scripts/mali_full_features.py",
    "benchmark_v1/scripts/run_mali_full_features.py",
    "benchmark_v1/scripts/run_qualified_logical_benchmark.py",
    "benchmark_v1/scripts/run_generalization_neural.py",
    "benchmark_v1/scripts/qonductor_native_adapter.py",
    "tracks/mali/overlay/data_preparation/build_training_data_from_logs.py",
    "benchmark_v1/protocol/qualified_logical_benchmark.json",
    "benchmark_v1/protocol/real_qpu_generalization.json",
    "benchmark_v1/protocol/strict_method_completion.json",
    "docs/research_contract.md",
    "artifacts/final_research/manifest.json",
    "artifacts/final_research/evidence.json",
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replay_tables():
    pins, outputs = {}, {}
    for function in (build.feature_tables, build.skill_tables, build.shots,
                     build.generalization_tables, build.interpretation_tables, build.upstream_pipeline):
        build.PINS.clear()
        values = function()
        pins.update(build.PINS)
        for name, expected in values.items():
            assert (OUT / name).read_text() == expected, "table_replay:" + name
            outputs[name] = len(expected.encode())
    return pins, outputs


def verify_upstream(original):
    """Batch Git-blob verification, without re-parsing circuits or loading models."""
    expected = dict(original["source_qasm_sha256"])
    for device in ("washington", "sherbrooke", "osaka", "kyoto"):
        relative = f"data/{device}_time_taken.csv"
        expected[relative] = original["source_pins"][str(build.UPSTREAM / relative)]
    expected["data_preparation/helper.py"] = original["source_code_pins"]["helper_sha256"]
    expected["data_preparation/utils.py"] = original["source_code_pins"]["utils_sha256"]
    expected.update(build.js(OUT / "upstream_pipeline_evidence.json")["git_blob_sha256"])
    relative_paths = sorted(expected)
    payload = "".join(f"{build.REV}:{p}\n" for p in relative_paths).encode()
    result = subprocess.check_output(["git", "-C", str(build.UPSTREAM), "cat-file", "--batch"], input=payload)
    offset = 0
    for relative in relative_paths:
        end = result.index(b"\n", offset)
        header = result[offset:end].split()
        assert len(header) == 3 and header[1] == b"blob", relative
        size = int(header[2]); data = result[end + 1:end + 1 + size]
        assert build.sha(data) == expected[relative], "upstream_blob:" + relative
        offset = end + size + 2
    assert offset == len(result)
    return len(relative_paths)


def invariants():
    original = build.js(OUT / "original_corpus_support.json")
    assert original["sim_effective_width"] == 41 and original["real_effective_width"] == 40
    assert len(original["source_qasm_sha256"]) == 1510
    ledger = build.rows(OUT / "global_feature_ledger.csv")
    assert len(ledger) == 51 and [int(r["raw_index"]) for r in ledger] == list(range(51))
    assert {r["feature_name"] for r in ledger if int(r["original_sim_nonzero_rows"]) == 0} == {
        "id", "u0", "csx", "cu", "rc3x", "c3x", "c3sqrtx", "c4x", "xx_plus_yy", "ecr"}
    rccx = ledger[37]
    assert rccx["feature_name"] == "rccx" and int(rccx["original_sim_nonzero_rows"]) == 12
    assert int(rccx["unified_nonzero_rows"]) == 0
    widths = build.rows(OUT / "effective_input_widths.csv")
    assert len(widths) == 257 and {r["effective_tensor_width"] for r in widths[:5]} == {"40"}
    matrix = build.rows(OUT / "benchmark_question_matrix.csv")
    assert len(matrix) == 120 and set(Counter(r["method_or_context"] for r in matrix).values()) == {8}
    for r in matrix:
        assert r["status"] in {"ANSWERED", "PARTIAL", "NOT ANSWERED", "NOT APPLICABLE"}
        assert all((ROOT / p).is_file() for p in r["evidence_paths"].split(";"))
    decomposition = build.rows(OUT / "generalization_decomposition.csv")
    assert len(decomposition) == 42 and len({r["case_id"] for r in decomposition}) == 21
    for name in ("qpu_predictive_skill.csv", "generalization_predictive_skill.csv",
                 "simulator_predictive_skill.csv", "analytical_predictive_skill.csv"):
        for row in build.rows(OUT / name):
            reference = float(row["reference_mae_seconds"])
            assert abs(float(row["skill"]) - (1-float(row["mae_seconds"])/reference)) < 1e-10
            if row["reference_id"] in {"outer_train_source_median", "train_source_median", "eligible_source_median"}:
                assert row["reference_context_advantage"] == "True"
    return dict(raw_fields=51, upstream_sim_width=41, upstream_real_width=40,
                transforms=257, question_rows=120, case_model_rows=42,
                static_upstream_blobs_verified=verify_upstream(original))


def seal():
    pins, outputs = replay_tables()
    original = build.js(OUT / "original_corpus_support.json")
    pins.update(original["source_pins"])
    for relative in HISTORICAL:
        path = ROOT / relative
        pins[str(path)] = digest(path)
    # The old freeze seal is its authority; this is not a new scientific result seal.
    old = build.js(ROOT / "artifacts/final_research/manifest.json")
    for entry in old["payload"]:
        path = ROOT / entry["path"]
        assert digest(path) == entry["sha256"], "frozen_payload_changed:" + str(path)
        pins[str(path)] = entry["sha256"]
    for row in build.rows(OUT / "benchmark_question_matrix.csv"):
        for relative in row["evidence_paths"].split(";"):
            path = ROOT / relative
            if not path.is_relative_to(OUT):
                pins[str(path)] = digest(path)
    payload = sorted([p for p in OUT.iterdir() if p.is_file() and p.name not in {"manifest.json", "validation.json"}]
                     + [Path(__file__), ROOT / "benchmark_v1/scripts/interpret_frozen_benchmark.py",
                        ROOT / "benchmark_v1/tests/test_benchmark_interpretation.py",
                        ROOT / "docs/benchmark_interpretation.md"])
    packages = {}
    for name in ("numpy", "qiskit", "scikit-learn", "pytest"):
        packages[name] = importlib.metadata.version(name)
    value = dict(schema="frozen_benchmark_interpretation_addendum", decision="B_REPORTING_FIX_ONLY",
        scientific_authority="Final Scientific Freeze v2; completed21-case RQ4 grid",
        experimental_scope_unchanged=True, source_pins=pins,
        payload=[dict(path=str(p.relative_to(ROOT)), sha256=digest(p), bytes=p.stat().st_size) for p in payload],
        generated_table_bytes=outputs, environment=dict(python=platform.python_version(), packages=packages),
        upstream_revision=build.REV, original_support_verification="pinned static feature re-extraction; no author processed tensor available",
        excluded_from_payload_hashes=["manifest.json (no self-hash)", "validation.json (records this manifest's hash)"],
        report_class="derived_interpretation_not_new_experiment_or_publication_authorization",
        permissions=dict(training=False,inference=False,measurement=False,qpu=False,qcec=False,commit_push=False,canva=False))
    build.patch({"manifest.json": build.json_text(value)})


def validate():
    manifest = build.js(OUT / "manifest.json")
    for path, expected in manifest["source_pins"].items():
        assert digest(path) == expected, "source_hash:" + path
    for row in manifest["payload"]:
        path = ROOT / row["path"]
        assert digest(path) == row["sha256"] and path.stat().st_size == row["bytes"], "payload_hash:" + str(path)
    _, outputs = replay_tables()
    counts = invariants()
    import consolidate_final_scientific_reporting as frozen
    authority = frozen.validate()
    assert authority["status"] == "PASS"
    return dict(status="PASS", decision="B_REPORTING_FIX_ONLY", counts=counts,
        source_pins_checked=len(manifest["source_pins"]), payload_files_checked=len(manifest["payload"]),
        derived_tables_replayed=len(outputs), all_source_outputs_and_frozen_authority_unchanged=True,
        frozen_authority_validation=authority, manifest_sha256=digest(OUT / "manifest.json"),
        scope="SAVED_EVIDENCE_AND_STATIC_SOURCE_VERIFICATION_NOT_NEW_EXPERIMENT",
        training_inference_measurement_QCEC=False, commit_push_Canva=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--receipt-patch", action="store_true")
    args = parser.parse_args()
    if args.seal:
        seal()
    else:
        result = validate()
        if args.receipt_patch:
            build.patch({"validation.json": build.json_text(result)})
        else:
            print(build.json_text(result))
