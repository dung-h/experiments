#!/usr/bin/env python3
"""Build the current two-domain reader tables from pinned benchmark outputs.

This builder creates one new semantic output directory. It never overwrites
historical reader packs, writes CURRENT, trains a model, starts CUDA, or runs a
simulator. Archived-QPU tables are copied from the hash-pinned C4 aggregation;
the separate C3a Aer graph diagnostic is recomputed from its frozen OOF rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v2"
C4 = ROOT / "artifacts/benchmark_v3/real_qpu/unified_method_comparisons_v2"
LEARNED = ROOT / "artifacts/benchmark_v3/real_qpu/unified_learned_derived_comparisons_v1"
CANONICAL = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
SIM_PACK = ROOT / "artifacts/benchmark_v3/simulator/simulator_reader_pack_v1"
AER = ROOT / "artifacts/benchmark_v1/aer_mali_graph_adaptation_v1_20261001"
PARTIAL_MAESTRO = ROOT / "benchmark_v1/execution/manifests/maestro_common_panel_completion_partial.json"
MAESTRO_V3 = ROOT / "artifacts/benchmark_v3/simulator/maestro_candidate_runtime_v3/calibration/run_manifest.json"
REGISTRY = ROOT / "benchmark_v1/registry/method_fidelity_registry_v2.json"

PINS = {
    "c4_manifest": "91449edbe653017b9dd54f79d465ec136f3495c9306fa42a49c3a64e9d8d4d83",
    "canonical_observations": "920d745e03dd00e8155118d9323baea18ff865e4df2620f3a059bd5503a05ba4",
    "fidelity_registry_v2": "98f3460c8d9a0cbabb424555dd695cee573409ed65facdb60557d836f1750810",
    "learned_manifest": "60d18c8cc5fe8f3e31a2ae04fd1d88e76bb581ad337d6003bc52a51f7e2eee13",
    "simulator_pack_manifest": "07b59ac1dc7ffa16e37ae4f638f93c4331df8318a0d2277e9ba7649db4e54d91",
    "aer_training_manifest": "84de8ebb2a501ba09233261329b708371bf06be59afcda4a23f5231edd3a2082",
    "aer_overall_qa": "083dc30ae81dddd979b3b09535a36baf4f2ca621ebc4e31b8b66fd44e64b6649",
    "aer_oof_predictions": "02bacf27c3a7cecde07283eda6518eadb1e1bca79eeb14508e7a69e2bca320a6",
    "aer_core_metrics": "320cc482d6fdabfc4c214fc01c2d4c5bfcf719b19d175a09f9eaaeb6a3cab074",
    "partial_maestro_policy": "3675c80f370f36aafec0dda3aa6814d5348e5f942400166f046d74388a256f5b",
    "failed_maestro_v3": "9e60bb86d08c0db51f76d1b50cc03bde68f6b69d1a7aa0e5f34ec2089c9a30c6",
}

C4_FILES = ("method_metrics.csv", "source_metrics.csv", "pairwise_comparisons.csv")
LEARNED_FILES = ("own_success_metrics.csv", "pair_metrics.csv", "bootstrap.csv")
SIM_FILES = (
    "per_configuration_metrics.csv",
    "native_method_appendix.csv",
    "blocked_unavailable.csv",
    "availability_matrix.csv",
    "common_exact_qasm_panel.csv",
)
OUTPUTS = (
    "README.md",
    "REPORT.md",
    "qpu/archived_method_coverage.csv",
    "qpu/archived_method_source_metrics.csv",
    "qpu/archived_method_pairwise_comparisons.csv",
    "qpu/unified_learned_metrics.csv",
    "qpu/unified_learned_pairs.csv",
    "qpu/unified_learned_bootstrap.csv",
    "simulator/per_configuration_metrics.csv",
    "simulator/native_method_appendix.csv",
    "simulator/blocked_unavailable.csv",
    "simulator/availability_matrix.csv",
    "simulator/common_exact_qasm_panel.csv",
    "simulator/aer_mali_graph_c3a_diagnostic.csv",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to emit empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def require_pin(path: Path, expected: str, label: str) -> str:
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual}")
    return actual


def check_manifest_files(folder: Path, manifest: dict) -> None:
    output_map = manifest.get("output_hashes", manifest.get("outputs", {}))
    for name, expected in output_map.items():
        path = folder / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"manifest output hash mismatch: {path}")


def check_input_hashes(manifest: dict) -> None:
    for name, expected in manifest.get("input_hashes", {}).items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"manifest input hash mismatch: {name}")
    inputs = manifest.get("inputs", {})
    if isinstance(inputs, dict):
        for item in inputs.values():
            if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
                continue
            path = ROOT / item["path"]
            if not path.is_file() or sha(path) != item["sha256"]:
                raise ValueError(f"manifest input hash mismatch: {item['path']}")


def copy_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    return sha(destination)


def compute_metrics(rows: list[dict[str, str]]) -> dict[str, float | int]:
    actual = [float(row["actual_seconds"]) for row in rows]
    predicted = [float(row["predicted_seconds"]) for row in rows]
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    if not errors:
        raise ValueError("cannot compute metrics for empty prediction set")
    log_errors = [abs(math.log1p(a) - math.log1p(p)) for a, p in zip(actual, predicted)]
    center = mean(actual)
    denominator = sum((value - center) ** 2 for value in actual)
    r2 = 1.0 - sum((a - p) ** 2 for a, p in zip(actual, predicted)) / denominator if denominator else float("nan")
    ordered = sorted(errors)

    def quantile(q: float) -> float:
        position = (len(ordered) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "n": len(rows),
        "mae_seconds": mean(errors),
        "medae_seconds": median(errors),
        "mae_log1p_seconds": mean(log_errors),
        "r2_seconds": r2,
        "p90_absolute_error_seconds": quantile(0.90),
        "p99_absolute_error_seconds": quantile(0.99),
        "max_absolute_error_seconds": max(errors),
    }


def aer_diagnostic() -> tuple[list[dict[str, object]], dict[str, object]]:
    prediction_path = AER / "core_oof_predictions.csv"
    metrics_path = AER / "core_metrics.csv"
    qa_path = AER / "overall_qa.json"
    training_path = AER / "training_manifest.json"
    for name, path in (
        ("aer_training_manifest", training_path),
        ("aer_overall_qa", qa_path),
        ("aer_oof_predictions", prediction_path),
        ("aer_core_metrics", metrics_path),
    ):
        require_pin(path, PINS[name], name)
    training = read_json(training_path)
    qa = read_json(qa_path)
    if training.get("status") != "PASS" or qa.get("status") != "PASS":
        raise ValueError("C3a training/QA status is not PASS")
    predictions = read_csv(prediction_path)
    if len(predictions) != 162 or len({r["observation_id"] for r in predictions}) != 162:
        raise ValueError("C3a must retain exactly 162 unique core OOF rows")
    if len({r["source_sha256"] for r in predictions}) != 150:
        raise ValueError("C3a QASM-hash count differs from frozen QA")
    if {int(r["fold"]) for r in predictions} != set(range(5)):
        raise ValueError("C3a fold coverage mismatch")
    for row in predictions:
        actual, predicted = float(row["actual_seconds"]), float(row["predicted_seconds"])
        if actual < 0 or predicted < 0 or not math.isfinite(actual) or not math.isfinite(predicted):
            raise ValueError(f"C3a contains an invalid runtime row: {row['observation_id']}")
        if not math.isclose(abs(actual - predicted), float(row["absolute_error_seconds"]), rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"C3a absolute error mismatch: {row['observation_id']}")
    overall = compute_metrics(predictions)
    published = next(r for r in read_csv(metrics_path) if r["scope"] == "all_core_oof")
    for field, key in (
        ("mae_seconds", "mae_seconds"),
        ("medae_seconds", "medae_seconds"),
        ("mae_log1p_seconds", "mae_log1p_seconds"),
        ("r2_seconds", "r2_seconds"),
        ("p90_absolute_error_seconds", "p90_absolute_error_seconds"),
        ("p99_absolute_error_seconds", "p99_absolute_error_seconds"),
        ("max_absolute_error_seconds", "max_absolute_error_seconds"),
    ):
        published_field = "mae_log1p_seconds" if field == "mae_log1p_seconds" else field
        if not math.isclose(float(published[published_field]), float(overall[key]), rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"C3a raw-derived metric differs from pinned core metrics: {published_field}")
    row = {
        "method_id": "c3a_mali_style_aer_graph_local_diagnostic",
        "reader_label": "S71-style Ma-Li graph + global-feature adaptation on local Aer warm labels",
        "registry_method_id": "",
        "fidelity_class": "",
        "registry_note": "No method ID/card exists for this C3a diagnostic in fidelity registry v2; do not treat as a registered paper-method comparison.",
        "claim_boundary": "Local C3a architecture adaptation; not original Ma-Li, not Azizov GNN, and not a full 204-row panel method.",
        "evaluation_target_clock": "warm_execution",
        "method_output_clock": "warm_execution",
        "population": "sim_common_q16_v1_core_only",
        "unique_qasm_hashes": len({r["source_sha256"] for r in predictions}),
        **overall,
        "frontier_rows_in_accuracy": 0,
        "quality_policy": "not an approximation-quality-constrained simulator; Aer observed warm runtime labels",
        "training_manifest_sha256": PINS["aer_training_manifest"],
        "overall_qa_sha256": PINS["aer_overall_qa"],
        "prediction_sha256": PINS["aer_oof_predictions"],
        "metrics_recomputed_from_oof_rows": True,
    }
    fold_rows = []
    for fold in range(5):
        fold_set = [r for r in predictions if int(r["fold"]) == fold]
        fold_rows.append({"fold": fold, **compute_metrics(fold_set)})
    return [row], {"fold_rows": fold_rows, "status": qa["status"]}


def build(output: Path) -> dict[str, object]:
    output = output if output.is_absolute() else ROOT / output
    if output.resolve() != DEFAULT_OUT.resolve():
        raise ValueError("v2 reader tables use the single canonical output path")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {output}")

    c4_manifest_path = C4 / "manifest.json"
    learned_manifest_path = LEARNED / "manifest.json"
    sim_manifest_path = SIM_PACK / "manifest.json"
    c4_manifest = read_json(c4_manifest_path)
    learned_manifest = read_json(learned_manifest_path)
    sim_manifest = read_json(sim_manifest_path)
    require_pin(c4_manifest_path, PINS["c4_manifest"], "C4 manifest")
    require_pin(CANONICAL, PINS["canonical_observations"], "canonical corpus")
    require_pin(REGISTRY, PINS["fidelity_registry_v2"], "fidelity registry v2")
    require_pin(learned_manifest_path, PINS["learned_manifest"], "unified learned manifest")
    require_pin(sim_manifest_path, PINS["simulator_pack_manifest"], "simulator reader pack manifest")
    require_pin(PARTIAL_MAESTRO, PINS["partial_maestro_policy"], "Maestro partial policy")
    require_pin(MAESTRO_V3, PINS["failed_maestro_v3"], "failed Maestro v3 manifest")
    if c4_manifest.get("status") != "PASS" or c4_manifest.get("method_count") != 24:
        raise ValueError("C4 is not the pinned 24-method PASS aggregation")
    if learned_manifest.get("status") != "PASS":
        raise ValueError("unified learned-QPU reader source is not PASS")
    if (
        sim_manifest.get("artifact_id") != "simulator-reader-pack-v1"
        or sim_manifest.get("panel_rows") != 204
        or sim_manifest.get("unique_qasm_sha256") != 191
        or sim_manifest.get("flat_cross_clock_rank_emitted") is not False
    ):
        raise ValueError("simulator reader source does not satisfy the pinned 204/191 contract")
    check_manifest_files(C4, c4_manifest)
    check_manifest_files(LEARNED, learned_manifest)
    check_manifest_files(SIM_PACK, sim_manifest)
    check_input_hashes(c4_manifest)
    check_input_hashes(learned_manifest)

    canonical_rows = read_csv(CANONICAL)
    if len(canonical_rows) != 8767:
        raise ValueError("canonical QPU corpus must contain 8,767 observations")
    canonical = {r["canonical_row_id"]: r for r in canonical_rows}
    if len(canonical) != len(canonical_rows):
        raise ValueError("canonical row IDs are not unique")
    attempts = read_csv(C4 / "attempts.csv")
    if len(attempts) != 24 * 8767:
        raise ValueError(f"C4 attempt envelope mismatch: {len(attempts)}")
    by_method: dict[str, set[str]] = {}
    for row in attempts:
        rid = row["canonical_row_id"]
        if rid not in canonical:
            raise ValueError(f"C4 attempt has unknown canonical row {rid}")
        base = canonical[rid]
        if row["source_id"] != base["source_id"] or not math.isclose(
            float(row["actual_seconds"]), float(base["target_seconds"]), rel_tol=0, abs_tol=1e-10
        ):
            raise ValueError(f"C4 attempt/canonical label mismatch {rid}")
        by_method.setdefault(row["method_id"], set()).add(rid)
    if len(by_method) != 24 or any(ids != set(canonical) for ids in by_method.values()):
        raise ValueError("C4 methods do not each retain the same full 8,767-row envelope")

    # Complete all scientific input checks before creating the output directory.
    aer_rows, aer_extra = aer_diagnostic()
    partial = read_json(PARTIAL_MAESTRO)
    old_v3 = read_json(MAESTRO_V3)

    output.mkdir(parents=True, exist_ok=True)
    qpu = output / "qpu"
    sim = output / "simulator"
    copied: dict[str, str] = {}
    for name, target in zip(C4_FILES, (
        "archived_method_coverage.csv", "archived_method_source_metrics.csv", "archived_method_pairwise_comparisons.csv"
    )):
        copied[f"qpu/{target}"] = copy_file(C4 / name, qpu / target)
    for name, target in zip(LEARNED_FILES, (
        "unified_learned_metrics.csv", "unified_learned_pairs.csv", "unified_learned_bootstrap.csv"
    )):
        copied[f"qpu/{target}"] = copy_file(LEARNED / name, qpu / target)
    for name in SIM_FILES:
        copied[f"simulator/{name}"] = copy_file(SIM_PACK / name, sim / name)

    write_csv(sim / "aer_mali_graph_c3a_diagnostic.csv", aer_rows)
    copied["simulator/aer_mali_graph_c3a_diagnostic.csv"] = sha(sim / "aer_mali_graph_c3a_diagnostic.csv")
    write_csv(sim / "aer_mali_graph_c3a_by_fold.csv", aer_extra["fold_rows"])
    copied["simulator/aer_mali_graph_c3a_by_fold.csv"] = sha(sim / "aer_mali_graph_c3a_by_fold.csv")

    # The Maestro partial policy has not produced any new timed result at build time.
    # Carry only its governing identity and an explicit no-score status.
    maestro_status = {
        "partial_experiment_id": partial.get("manifest_id", ""),
        "partial_manifest_path": rel(PARTIAL_MAESTRO),
        "partial_manifest_sha256": sha(PARTIAL_MAESTRO),
        "status_at_build": "not_started_no_partial_run_manifest",
        "score": None,
        "failed_v3_artifact_status": old_v3.get("status", "unknown"),
        "failed_v3_is_resumed": False,
        "interpretation": "No new partial-calibration or panel timing score is included; preserve v3 failure and do not report this as an estimator failure on the common panel.",
    }

    c3a = aer_rows[0]
    report = f"""# Two-domain benchmark reader tables v2

Built from pinned archived real-QPU aggregation and local simulator artifacts.
This is a reader surface, not a flat leaderboard: QPU source labels, scheduled
outputs, and local simulator clocks remain separate.

## Real-QPU domain

The canonical population is **8,767 archived circuit observations**. The C4
attempt ledger has 24 method variants × 8,767 assigned rows (210,408 attempts).
Read `qpu/archived_method_coverage.csv` for assigned/predicted/unavailable
counts, `qpu/archived_method_source_metrics.csv` for source-local errors and
the 46-workflow QPack diagnostic, and `qpu/archived_method_pairwise_comparisons.csv`
for paired source-local contrasts. There is no cross-source pooled QPU score.
The three unified learned-adaptation summaries and their compatible-row pairs
are separate in `qpu/unified_learned_*.csv`.

## Local-simulator domain

The fixed panel has 204 members / 191 exact QASM hashes (162 core, 42 frontier,
q=2–16). Engine, precision, first/warm clock, approximation quality and
failure states stay separate in the five copied simulator tables. The C3a
local Aer graph adaptation is reported separately from the Azizov GNN: on 162
core OOF rows / 150 hashes its raw-derived MAE is {c3a['mae_seconds']:.6f} s,
MedAE {c3a['medae_seconds']:.6f} s, log1p-MAE {c3a['mae_log1p_seconds']:.6f},
R² {c3a['r2_seconds']:.6f}; q10–16 are excluded. It is an unregistered
supplementary architecture adaptation, not original Ma–Li or Azizov.

Maestro has **no score in this pack**. The previous v3 preparation failure is
preserved; the separately governed partial run was not started at this build.
This is not a measured common-panel prediction failure.

## Scope limits

No cross-domain or cross-clock ranking is emitted. `evaluation_target_clock`
and `method_output_clock` remain distinct in the source tables. QPack replay
circuits are reconstruction-qualified; they do not change the archived labels.
The package is a local review artifact, not a public-release approval. Root
license/citation and external redistribution-rights decisions remain open.
"""
    readme = """# Two-domain benchmark tables v2

Start with `REPORT.md`, then use `qpu/` and `simulator/` as separate evidence
domains. This pack replaces neither CURRENT nor any historical reader pack.
There is intentionally no single cross-domain ranking: clocks and targets
differ. `manifest.json` pins the source/output hashes and the explicit Maestro
no-score status at build time.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "README.md").write_text(readme, encoding="utf-8")

    source_paths = [
        Path(__file__).resolve(),
        C4 / "manifest.json", C4 / "attempts.csv", C4 / "method_metrics.csv",
        C4 / "source_metrics.csv", C4 / "pairwise_comparisons.csv",
        CANONICAL, REGISTRY, LEARNED / "manifest.json", SIM_PACK / "manifest.json",
        AER / "training_manifest.json", AER / "overall_qa.json",
        AER / "core_oof_predictions.csv", AER / "core_metrics.csv",
        PARTIAL_MAESTRO, MAESTRO_V3,
    ]
    source_hashes = {rel(path): sha(path) for path in source_paths}
    output_hashes = {name: sha(output / name) for name in (*OUTPUTS, "simulator/aer_mali_graph_c3a_by_fold.csv")}
    payload = {
        "artifact_id": "two-domain-benchmark-reader-tables-v2",
        "status": "BUILT_FROM_PINNED_INPUTS_MAESTRO_NOT_STARTED",
        "domains": ["archived_real_qpu", "local_simulator"],
        "real_qpu": {
            "canonical_observations": 8767,
            "c4_method_variants": len(by_method),
            "c4_assigned_attempts": len(attempts),
            "cross_source_pooled_error_metrics": False,
            "source_pair_table": "qpu/archived_method_pairwise_comparisons.csv",
        },
        "simulator": {
            "panel_members": 204,
            "unique_qasm_hashes": 191,
            "core_members": 162,
            "frontier_members": 42,
            "c3a_oof_rows": c3a["n"],
            "c3a_unique_qasm_hashes": c3a["unique_qasm_hashes"],
            "c3a_metrics_recomputed_from_raw_oof": True,
            "maestro": maestro_status,
        },
        "no_flat_cross_domain_or_cross_clock_rank": True,
        "current_json_modified": False,
        "historical_packs_modified": False,
        "training_or_timing_performed": False,
        "public_release_approved": False,
        "source_hashes": source_hashes,
        "copied_or_derived_output_hashes": copied,
        "outputs": output_hashes,
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = build(args.output_dir)
    print(json.dumps({"artifact_id": payload["artifact_id"], "status": payload["status"],
                      "output_dir": str(args.output_dir), "c4_method_variants": payload["real_qpu"]["c4_method_variants"],
                      "c3a_rows": payload["simulator"]["c3a_oof_rows"],
                      "maestro_status": payload["simulator"]["maestro"]["status_at_build"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
