#!/usr/bin/env python3
"""Audit original Ma-Li source joins and freeze a fresh grouped panel; no fit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "benchmark_v1/protocol/strict_logical_benchmark.json"
SOURCE = ROOT.parent / "Quantum-Execution-Time-Prediction"
CANONICAL = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
FEATURE_ROOT = ROOT / "artifacts/real_qpu/logical_inputs"
REP_ROOT = ROOT / "artifacts/benchmark_v3/real_qpu/compiled_unified"
REGISTRY = ROOT / "artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def unique_index(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate_identity:{key}")
    return result


def source_blob(revision, path):
    return subprocess.check_output(["git", "-C", str(SOURCE), "show", f"{revision}:{path}"])


def join_timing(canonical, timing, source_hash):
    """Name+device binds a label to source QASM; no family/width inference."""
    expected = "ibm_" + timing["device"]
    name = timing["quantum_circuit"] + ".qasm"
    if canonical["backend"] != expected or canonical["qasm_path_or_member"] != name:
        raise ValueError("source_timing_identity_mismatch")
    if canonical["qasm_bytes_sha256"] != source_hash:
        raise ValueError("source_qasm_hash_mismatch")
    if float(canonical["shots"]) != 1024:
        raise ValueError("source_shots_mismatch")
    a, b = float(canonical["target_seconds"]), float(timing["time_taken"])
    if not math.isfinite(a) or a <= 0 or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("source_target_mismatch")
    return abs(a - b)


def graph_signature(record, globals_):
    """Exact declared model input without calibration, independent of backend."""
    import numpy as np
    h = hashlib.sha256()
    global_array = np.asarray(globals_, dtype="<f8")
    global_array[global_array == 0] = 0  # Canonicalize signed zeros.
    h.update(global_array.tobytes())
    for key in ("node_type", "wire0", "wire1", "node_index", "edge_index"):
        value = np.asarray(record[key])
        value = value.astype(value.dtype.newbyteorder("<"), copy=False)
        h.update(json.dumps([key, list(value.shape), value.dtype.str]).encode())
        h.update(value.tobytes(order="C"))
    return h.hexdigest()


def close_groups(rows):
    parents = {r["canonical_observation_id"]: r["canonical_observation_id"] for r in rows}
    def find(x):
        while x != parents[x]:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    witnesses = {}
    for row in sorted(rows, key=lambda r: r["canonical_observation_id"]):
        rid = row["canonical_observation_id"]
        for field in ("logical_qasm_sha256", "backend_independent_model_signature"):
            key = (field, row[field])
            if key in witnesses:
                a, b = find(rid), find(witnesses[key])
                parents[max(a, b)] = min(a, b)
            witnesses[key] = rid
    components = defaultdict(list)
    for rid in sorted(parents):
        components[find(rid)].append(rid)
    return {"logical:" + sha("\n".join(ids).encode()): ids for ids in components.values()}


def assign_groups(components, backends, folds, seed):
    if len(components) < folds:
        raise ValueError("insufficient_group_support")
    totals = Counter(backends[rid] for ids in components.values() for rid in ids)
    loads = [Counter() for _ in range(folds)]
    result = {}
    ordered = sorted(components, key=lambda g: (
        -max(Counter(backends[rid] for rid in components[g])[b] / totals[b] for b in totals),
        -len(components[g]), g))
    for group in ordered:
        vector = Counter(backends[rid] for rid in components[group])
        def score(chosen):
            cost = sum(((loads[f][b] + (vector[b] if f == chosen else 0) - totals[b] / folds)
                        / max(totals[b] / folds, 1)) ** 2 for f in range(folds) for b in totals)
            return cost, sha(f"{seed}:{group}:{chosen}".encode())
        chosen = min(range(folds), key=score)
        result[group] = chosen
        loads[chosen].update(vector)
    if any(set(load) != set(totals) for load in loads):
        raise ValueError("backend_support_missing_in_partition")
    return result, [dict(x) for x in loads]


def build_splits(rows, components):
    backends = {r["canonical_observation_id"]: r["backend"] for r in rows}
    outer_assignment, counts = assign_groups(components, backends, 5, 42)
    outer, inner, support = [], [], {}
    for group, ids in sorted(components.items()):
        outer.extend({"canonical_observation_id": rid, "group_id": group,
                      "outer_fold": outer_assignment[group]} for rid in ids)
    for f in range(5):
        train = {g: ids for g, ids in components.items() if outer_assignment[g] != f}
        inner_assignment, inner_counts = assign_groups(train, backends, 4, 43 + f)
        support[str(f)] = inner_counts
        for group, ids in sorted(train.items()):
            inner.extend({"canonical_observation_id": rid, "group_id": group,
                          "outer_fold": f, "inner_fold": inner_assignment[group]} for rid in ids)
    validate_splits(outer, inner)
    return outer, inner, {"outer_backend_counts": counts, "inner_backend_counts": support}


def validate_splits(outer, inner):
    index = unique_index(outer, "canonical_observation_id")
    for f in range(5):
        test = {r["canonical_observation_id"] for r in outer if int(r["outer_fold"]) == f}
        inside = [r for r in inner if int(r["outer_fold"]) == f]
        train = unique_index(inside, "canonical_observation_id")
        if set(train) != set(index) - test:
            raise ValueError("inner_outer_membership_mismatch")
        group_test = {index[rid]["group_id"] for rid in test}
        if group_test & {r["group_id"] for r in inside}:
            raise ValueError("outer_group_leakage")
        for v in range(4):
            valid = {r["group_id"] for r in inside if int(r["inner_fold"]) == v}
            fit = {r["group_id"] for r in inside if int(r["inner_fold"]) != v}
            if not valid or not fit or valid & fit:
                raise ValueError("inner_group_leakage_or_empty")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze(out):
    import numpy as np
    import mali_full_features as mf
    if out.exists() and any(out.iterdir()):
        raise ValueError("refuse_nonempty_output")
    contract = json.loads(CONTRACT.read_text())
    if contract["training_authorized"] or contract["source"]["expected_rows"] != 340:
        raise ValueError("unexpected_preparation_contract")
    canonical = read_csv(CANONICAL)
    if len(unique_index(canonical, "canonical_row_id")) != 8767:
        raise ValueError("canonical_identity_count_mismatch")
    selected = [r for r in canonical if r["source_id"] == "mali_real_qpu"]
    if len(selected) != 340:
        raise ValueError("source_cohort_count_mismatch")
    timing = {}
    source_hashes = {}
    revision = contract["source"]["revision"]
    for path in contract["source"]["timing_files"]:
        blob = source_blob(revision, path)
        source_hashes[path] = sha(blob)
        for r in csv.DictReader(io.StringIO(blob.decode())):
            key = ("ibm_" + r["device"], r["quantum_circuit"] + ".qasm")
            if key in timing:
                raise ValueError("duplicate_source_timing_key")
            timing[key] = r
    for path in ("main.ipynb", "data_preparation/execution.py"):
        source_hashes[path] = sha(source_blob(revision, path))
    if set(timing) != {(r["backend"], r["qasm_path_or_member"]) for r in selected}:
        raise ValueError("source_canonical_timing_keys_differ")
    features = unique_index(read_csv(FEATURE_ROOT / "feature_attempts.csv"), "canonical_observation_id")
    rep = unique_index(read_csv(REP_ROOT / "representation_rows.csv"), "canonical_row_id")
    registry = unique_index(read_csv(REGISTRY), "snapshot_id")
    extractor, helper_pins = mf.load_pinned_global_extractor()
    rows, source_cache, graph_hashes, snapshot_hashes, compiled_hashes = [], {}, {}, {}, {}
    max_delta = 0.0
    for c in selected:
        rid, name = c["canonical_row_id"], c["qasm_path_or_member"]
        path = "data/quantum_circuits/" + name
        if path not in source_cache:
            data = source_blob(revision, path)
            qasm_hash = sha(data)
            circuit, _ = mf.parse_qasm(data)
            values = mf.extract_global_features(circuit, extractor)
            source_cache[path] = (qasm_hash, circuit, values)
            source_hashes[path] = qasm_hash
        qasm_hash, circuit, values = source_cache[path]
        max_delta = max(max_delta, join_timing(c, timing[(c["backend"], name)], qasm_hash))
        fr, cr = features[rid], rep[rid]
        if (fr["source_qasm_sha256"] != qasm_hash or fr["logical_input_tier"] != "source_has_logical_input"
                or fr["status"] != "available" or cr["source_qasm_sha256"] != qasm_hash):
            raise ValueError(f"cached_input_provenance_mismatch:{rid}")
        observed = np.array([float(fr[f"g{i:02d}"]) for i in range(51)])
        if not np.array_equal(np.asarray(values), observed):
            raise ValueError(f"global_reextraction_mismatch:{rid}")
        snapshot = registry[cr["target_snapshot_id"]]
        for asset, expected in (("configuration", cr["target_configuration_sha256"]),
                                ("properties", cr["target_properties_sha256"])):
            asset_path = Path(snapshot[f"{asset}_path"])
            if file_sha(asset_path) != expected or snapshot[f"{asset}_sha256"] != expected:
                raise ValueError(f"snapshot_hash_mismatch:{rid}:{asset}")
            snapshot_hashes[str(asset_path)] = expected
        if (fr["snapshot_id"] != cr["target_snapshot_id"]
                or fr["properties_sha256"] != cr["target_properties_sha256"]):
            raise ValueError(f"snapshot_context_mismatch:{rid}")
        calibration, _ = mf.snapshot_t1_t2(cr, registry)
        rebuilt, meta = mf.compact_graph_record(circuit, calibration)
        graph_path = FEATURE_ROOT / fr["graph_file"]
        actual_hash = file_sha(graph_path)
        if actual_hash != fr["graph_sha256"]:
            raise ValueError(f"graph_cache_hash_mismatch:{rid}")
        with np.load(graph_path, allow_pickle=False) as graph:
            if set(graph.files) != set(rebuilt) or any(not np.array_equal(graph[k], rebuilt[k]) for k in rebuilt):
                raise ValueError(f"graph_reencoding_mismatch:{rid}")
        graph_hashes[str(graph_path.relative_to(ROOT))] = actual_hash
        compiled_path = REP_ROOT / cr["compiled_qasm3_file"]
        if file_sha(compiled_path) != cr["compiled_qasm3_sha256"]:
            raise ValueError(f"compiled_qasm_cache_hash_mismatch:{rid}")
        compiled_hashes[str(compiled_path.relative_to(ROOT))] = cr["compiled_qasm3_sha256"]
        rows.append({"canonical_observation_id": rid, "source_id": c["source_id"], "backend": c["backend"],
                     "logical_qasm_source_path": path, "logical_qasm_sha256": qasm_hash,
                     "backend_independent_model_signature": graph_signature(rebuilt, values),
                     "target_seconds": c["target_seconds"], "shots": c["shots"],
                     "target_source_key": c["backend"] + ":" + name,
                     "snapshot_id": fr["snapshot_id"], "properties_sha256": fr["properties_sha256"],
                     "graph_file": str(graph_path.relative_to(ROOT)), "graph_sha256": actual_hash,
                     "node_count": meta["node_count"], **{f"g{i:02d}": values[i] for i in range(51)}})
    if len(source_cache) != 192:
        raise ValueError("source_qasm_count_mismatch")
    exact_hash_count = len({value[0] for value in source_cache.values()})
    if exact_hash_count != contract["source"]["expected_exact_qasm_hashes"]:
        raise ValueError("source_unique_qasm_hash_count_mismatch")
    components = close_groups(rows)
    outer, inner, support = build_splits(rows, components)
    group_by_id = {rid: g for g, ids in components.items() for rid in ids}
    for r in rows:
        r["group_id"] = group_by_id[r["canonical_observation_id"]]
    cohort = [{k: r[k] for k in r if not (k.startswith("g") and k[1:].isdigit())} for r in rows]
    excluded = [{"source_id": "qonductor_single_circuit_ibm", "rows": 4482,
                 "reason": "original_logical_instance_link_not_accepted"},
                {"source_id": "qpack_mcp", "rows": 3945,
                 "reason": "representative_angles_not_original_instance"}]
    input_paths = [CANONICAL, FEATURE_ROOT / "feature_attempts.csv", REP_ROOT / "representation_rows.csv",
                   REGISTRY, CONTRACT, Path(__file__), ROOT / "benchmark_v1/scripts/mali_full_features.py",
                   ROOT / "benchmark_v1/protocol/mali_full_features.json",
                   ROOT / "benchmark_v1/protocol/qonductor_native_features.json",
                   ROOT / "benchmark_v1/protocol/scholten_unified_nominal.json",
                   ROOT / "benchmark_v1/protocol/analytical_extension.json",
                   ROOT / "benchmark_v1/protocol/common_panel_completion.json"]
    pins = {str(p.relative_to(ROOT)): file_sha(p) for p in input_paths}
    pins.update(graph_hashes)
    pins.update(compiled_hashes)
    out.mkdir(parents=True, exist_ok=True)
    for filename, content in (("cohort.csv", cohort), ("logical_features.csv", rows),
                              ("outer_splits.csv", outer), ("inner_splits.csv", inner),
                              ("exclusions.csv", excluded)):
        write_csv(out / filename, content)
    method_contracts = {"protocol_sha256": file_sha(CONTRACT), "contract": contract,
                        "input_pins": pins, "source_git_blob_hashes": source_hashes,
                        "snapshot_asset_hashes": snapshot_hashes, "feature_helper_source_pins": helper_pins}
    (out / "execution_contract.json").write_text(json.dumps(method_contracts, indent=2, sort_keys=True) + "\n")
    manifest = {"status": "PASS_scientific_freeze_no_training", "rows": 340,
                "backend_counts": dict(Counter(r["backend"] for r in rows)),
                "logical_qasm_files": len(source_cache), "exact_logical_qasm_hashes": exact_hash_count,
                "logical_groups": len(components),
                "source_timing_join": "340/340", "target_max_abs_source_difference": max_delta,
                "global_reextraction": "340/340", "graph_reencoding": "340/340",
                "parameter_and_context_safe_grouping": True, "train_test_group_overlap": 0,
                "split_assignment_consumed_target_values": False,
                "training_authorized": False, "training_started": False,
                "GPU_used": False, "original_repeated_measurements_recomputed": False,
                "source_runtime_semantics": contract["target"], **support,
                "output_hashes": {p.name: file_sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/real_qpu/strict_logical_panel")
    args = parser.parse_args()
    manifest = freeze(args.output)
    print(json.dumps({k: v for k, v in manifest.items() if k != "output_hashes"}, indent=2))


if __name__ == "__main__":
    main()
