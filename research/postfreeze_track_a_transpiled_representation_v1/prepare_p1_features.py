#!/usr/bin/env python3
"""Freeze label-blind feature views for the post-freeze transpiled study."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/postfreeze_track_a_transpiled_representation_v1"
P0 = ART / "p0_coverage"
PANEL = ROOT / "artifacts/real_qpu/qualified_logical_panel/panel.csv"
OUTER = ROOT / "artifacts/real_qpu/qualified_logical_panel/outer_splits.csv"
INNER = ROOT / "artifacts/real_qpu/qualified_logical_panel/inner_splits.csv"

# Fixed before fitting.  These are semantic opcode names, not a test-derived
# vocabulary.  All remaining saved opcodes map to native_other_count.
NATIVE_OPCODE_SCHEMA = (
    "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg",
    "rx", "ry", "rz", "p", "u", "u1", "u2", "u3", "cx", "cz", "ecr",
    "swap", "measure", "reset", "barrier", "delay",
)
SOURCE_FIELDS = tuple(f"g{i:02d}" for i in range(51))
STATIC_FIELDS = (
    "active_physical_width", "physical_depth", "native_operation_count",
    "native_2q_operation_count", "native_measurement_count", "native_swap_like_count",
    "dag_node_count", "dag_edge_count",
)
TIMING_FIELDS = ("nominal_single_shot_duration_seconds",)
CONTEXT_FIELDS = ("log1p_shots",)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, values: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader(); writer.writerows(values)


def main() -> int:
    import math
    panel = rows(PANEL)
    p0 = rows(P0 / "observation_coverage.csv")
    outer, inner = rows(OUTER), rows(INNER)
    p0_by_id = {r["canonical_observation_id"]: r for r in p0}
    if len(panel) != 4515 or len(p0_by_id) != 4515 or set(r["canonical_observation_id"] for r in panel) != set(p0_by_id):
        raise RuntimeError("p0_panel_identity_mismatch")
    if any(r["terminal_status"] != "AVAILABLE" for r in p0):
        raise RuntimeError("p0_common_success_not_complete")
    # Inner assignments exist once for every outer-training membership, hence
    # four records per observation rather than a duplicate outer-test record.
    if len(outer) != 4515 or len(inner) != 4515 * 4:
        raise RuntimeError("frozen_split_row_count_mismatch")
    if any(field not in panel[0] for field in SOURCE_FIELDS):
        raise RuntimeError("source_static_schema_missing")

    feature_dir = ART / "features"
    source_rows: list[dict[str, str]] = []
    static_rows: list[dict[str, str]] = []
    timing_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    for row in panel:
        cid = row["canonical_observation_id"]
        physical = p0_by_id[cid]
        histogram = json.loads(physical["native_gate_histogram"])
        unknown = sum(int(v) for key, v in histogram.items() if key not in NATIVE_OPCODE_SCHEMA)
        source_rows.append({"canonical_observation_id": cid, **{field: row[field] for field in SOURCE_FIELDS},
                            "log1p_shots": repr(math.log1p(float(row["shots"])))})
        static = {"canonical_observation_id": cid,
                  "active_physical_width": physical["active_physical_width"],
                  "physical_depth": physical["physical_depth"],
                  "native_operation_count": physical["native_operation_count"],
                  "native_2q_operation_count": physical["native_2q_operation_count"],
                  "native_measurement_count": str(histogram.get("measure", 0)),
                  "native_swap_like_count": str(histogram.get("swap", 0)),
                  "dag_node_count": physical["dag_node_count"], "dag_edge_count": physical["dag_edge_count"],
                  **{f"native_{opcode}_count": str(histogram.get(opcode, 0)) for opcode in NATIVE_OPCODE_SCHEMA},
                  "native_other_count": str(unknown), "log1p_shots": repr(math.log1p(float(row["shots"]))) }
        static_rows.append(static)
        timing_rows.append({"canonical_observation_id": cid,
                            "nominal_single_shot_duration_seconds": physical["weighted_critical_path"],
                            "log1p_shots": repr(math.log1p(float(row["shots"])))})
        manifest_rows.append({"canonical_observation_id": cid, "group_id": row["group_id"], "source_id": row["source_id"],
                              "backend": row["backend"], "structure_provenance": physical["structure_provenance"],
                              "timing_provenance": physical["timing_provenance"], "compiled_hash": physical["compiled_hash"],
                              "physical_graph_key": physical["physical_graph_key"]})

    static_fields = list(static_rows[0])
    write_csv(feature_dir / "source_static.csv", source_rows, list(source_rows[0]))
    write_csv(feature_dir / "transpiled_static.csv", static_rows, static_fields)
    write_csv(feature_dir / "timing_features.csv", timing_rows, list(timing_rows[0]))
    write_csv(feature_dir / "representation_manifest.csv", manifest_rows, list(manifest_rows[0]))
    dictionary = {
        "schema_version": "p1-label-blind-v1", "source_static": {"fields": list(SOURCE_FIELDS), "transform": "g00-g43 log1p; g44-g45 log1p; g46-g50 natural scale; outer-train scaler"},
        "transpiled_static": {"base_fields": list(STATIC_FIELDS), "native_opcode_schema": list(NATIVE_OPCODE_SCHEMA), "other_policy": "sum saved opcodes outside fixed semantic schema", "transform": "all counts/depth log1p; outer-train scaler"},
        "timing": {"fields": list(TIMING_FIELDS), "provenance": "saved nominal single-shot proxy; log1p then outer-train scaler", "not_claimed": "historical execution-day timing"},
        "shared_context": {"fields": list(CONTEXT_FIELDS), "transform": "already log1p(shots); outer-train scaler"},
        "forbidden_primary": ["source_id", "backend", "snapshot_id", "hashes", "provenance flags", "target", "fold"],
        "views": {"S0": "source_static + log1p_shots", "P0": "transpiled_static + log1p_shots", "P1": "transpiled_static + timing + log1p_shots", "H0": "source_static + transpiled_static + log1p_shots", "H1": "source_static + transpiled_static + timing + log1p_shots"},
    }
    (feature_dir / "feature_dictionary.json").write_text(json.dumps(dictionary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    feature_paths = [feature_dir / x for x in ("source_static.csv", "transpiled_static.csv", "timing_features.csv", "representation_manifest.csv", "feature_dictionary.json")]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in feature_paths}
    (feature_dir / "feature_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    p1_manifest = {
        "status": "FROZEN_BEFORE_P1_FITTING", "cohort": {"rows": 4515, "groups": 165, "source_counts": {"mali_real_qpu": 340, "qonductor_single_circuit_ibm": 230, "qpack_mcp": 3945}},
        "feature_hashes": hashes, "split_hashes": {str(OUTER.relative_to(ROOT)): sha(OUTER), str(INNER.relative_to(ROOT)): sha(INNER)},
        "panel_sha256": sha(PANEL), "target_column": "target_seconds", "target_transform": "log1p(seconds)",
        "views": dictionary["views"], "primary_learner": {"name": "Ridge", "alpha_grid": [1e-4,1e-3,1e-2,1e-1,1,10,100,1000], "selection": "four inner group folds, group-balanced MAE in raw seconds"},
        "secondary_learner": {"name": "HistGradientBoostingRegressor", "selection": "same inner folds; fixed small predeclared grid"},
        "bootstrap": {"unit": "structural_group", "replicates": 10000, "seed": 42}, "no_test_feature_selection": True,
    }
    manifest_path = ART / "p1_manifest.json"
    manifest_path.write_text(json.dumps(p1_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ART / "p1_manifest.sha256").write_text(sha(manifest_path) + "  p1_manifest.json\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "manifest_sha256": sha(manifest_path), "feature_hashes": hashes}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
