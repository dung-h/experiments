#!/usr/bin/env python3
"""Prepare, profile and train the frozen logical-input common panel."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmark_v1/scripts"
CONTRACT = ROOT / "benchmark_v1/protocol/common_panel_completion.json"
PANEL = ROOT / "artifacts/real_qpu/common_panel"
FEATURE_ROOT = ROOT / "artifacts/real_qpu/logical_inputs"
OUTPUT = PANEL / "mali"
STATUS = PANEL / "run_status.json"
sys.path.insert(0, str(SCRIPT))

import run_mali_full_features as base  # noqa: E402
import run_mali_batched as batching  # noqa: E402

METHODS = {
    "mali_logical_graph": base.METHODS[0],
    "mali_global_mlp": base.METHODS[1],
}
SEEDS = (42, 1234, 31415)
EPOCHS = 500
NODE_CANDIDATES = (250_000, 500_000)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(100_000_000)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse_empty_csv:{path}")
    base.write_csv_atomic(path, rows, list(rows[0]))


def contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def pins() -> dict[str, str]:
    return {name: digest(ROOT / item["path"]) for name, item in contract()["input_pins"].items()}


def _rows_by_id(rows: list[dict[str, str]], key: str, label: str) -> dict[str, dict[str, str]]:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate_{label}_identity")
    return result


def prepare() -> dict[str, Any]:
    import numpy as np

    manifest = json.loads((PANEL / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("rows") != 7350 or manifest.get("status") != "scientific_design_locked_execution_pending":
        raise RuntimeError("panel_manifest_identity_mismatch")
    panel_rows = read_csv(PANEL / "panel.csv")
    targets = _rows_by_id(read_csv(PANEL / "targets.csv"), "canonical_observation_id", "target")
    ledger_rows = read_csv(PANEL / "mali_feature_ledger.csv")
    ledger = _rows_by_id(ledger_rows, "canonical_row_id", "neural_feature")
    source_features = _rows_by_id(read_csv(FEATURE_ROOT / "feature_attempts.csv"),
                                  "canonical_observation_id", "logical_feature")
    panel_ids = {row["canonical_observation_id"] for row in panel_rows}
    if (len(panel_ids) != 7350 or set(targets) != panel_ids or set(ledger) != panel_ids
            or not panel_ids <= set(source_features)):
        raise RuntimeError("common_panel_join_not_7350_to_7350")
    if pins() != {name: item["sha256"] for name, item in contract()["input_pins"].items()}:
        raise RuntimeError("frozen_input_pin_mismatch")

    graph_meta: dict[str, tuple[int, int]] = {}
    graph_hashes: dict[str, str] = {}
    index_rows = []
    for row in panel_rows:
        identity = row["canonical_observation_id"]
        feature = ledger[identity]
        source = source_features[identity]
        target = targets[identity]
        raw = [float(feature[f"global_{i:02d}"]) for i in range(51)]
        source_globals = [float(source[f"g{i:02d}"]) for i in range(51)]
        seconds, shots = float(target["target_seconds"]), float(target["shots"])
        if feature["global_status"] != "available" or feature["graph_status"] != "available":
            raise RuntimeError(f"neural_input_unavailable:{identity}")
        if (not np.isfinite(raw).all() or not np.isfinite(source_globals).all()
                or raw != source_globals or not np.isfinite(seconds) or seconds <= 0
                or not np.isfinite(shots) or shots <= 0):
            raise RuntimeError(f"nonfinite_or_nonpositive_input:{identity}")
        relative = Path(feature["graph_artifact"])
        graph_path = (FEATURE_ROOT / relative).resolve()
        if FEATURE_ROOT.resolve() not in graph_path.parents or not graph_path.is_file():
            raise RuntimeError(f"graph_path_missing_or_outside_root:{identity}")
        actual_hash = graph_hashes.get(relative.as_posix())
        if actual_hash is None:
            actual_hash = digest(graph_path)
            graph_hashes[relative.as_posix()] = actual_hash
        if actual_hash != feature["graph_artifact_sha256"]:
            raise RuntimeError(f"graph_file_hash_mismatch:{identity}")
        if (source["status"] != "available" or source["graph_file"] != relative.as_posix()
                or source["graph_sha256"] != actual_hash or int(source["node_count"]) <= 0):
            raise RuntimeError(f"logical_source_feature_identity_mismatch:{identity}")
        if relative.as_posix() not in graph_meta:
            with np.load(graph_path, allow_pickle=False) as data:
                record = {name: data[name] for name in data.files}
                dense = base.expand_dense_x(record)
                edge_index = record["edge_index"]
                if (dense.shape != (len(record["node_type"]), 178) or not np.isfinite(dense).all()
                        or edge_index.ndim != 2 or edge_index.shape[0] != 2
                        or (edge_index.size and (edge_index.min() < 0 or edge_index.max() >= dense.shape[0]))):
                    raise RuntimeError(f"graph_schema_invalid:{identity}")
                if dense.shape[0] != int(source["node_count"]):
                    raise RuntimeError(f"graph_node_count_differs_from_source_ledger:{identity}")
                graph_meta[relative.as_posix()] = (int(dense.shape[0]), int(edge_index.shape[1]))
        nodes, edges = graph_meta[relative.as_posix()]
        index_rows.append({"canonical_observation_id": identity, "source_id": target["source_id"],
                           "logical_input_tier": feature["logical_input_tier"],
                           "graph_artifact": relative.as_posix(), "graph_sha256": actual_hash,
                           "graph_node_count": nodes, "graph_edge_count": edges})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT / "input_index.csv"
    if index_path.exists():
        old = read_csv(index_path)
        if old != index_rows:
            old_without_targets = [{key: row.get(key, "") for key in index_rows[0]} for row in old]
            expected_wire = [{key: str(value) for key, value in row.items()} for row in index_rows]
            if old_without_targets != expected_wire or len(old) != len(index_rows):
                raise RuntimeError("existing_input_index_differs_refuse_overwrite")
            write_csv(index_path, index_rows)
    else:
        write_csv(index_path, index_rows)
    result = {"status": "PASS", "rows": 7350, "unique_graph_files": len(graph_meta),
              "largest_graph_nodes": max(nodes for nodes, _edges in graph_meta.values()),
              "largest_graph_edges": max(edges for _nodes, edges in graph_meta.values()),
        "input_index": index_path.relative_to(ROOT).as_posix(), "input_index_sha256": digest(index_path),
              "input_pins": pins(), "runner_sha256": digest(Path(__file__))}
    base.atomic_json(OUTPUT / "input_index_manifest.json", result)
    update_status(stage="inputs", status="PASS", completed_rows=7350,
                  input_index_sha256=result["input_index_sha256"])
    print(json.dumps(result, indent=2), flush=True)
    return result


def load_context() -> dict[str, Any]:
    index_path = OUTPUT / "input_index.csv"
    manifest_path = OUTPUT / "input_index_manifest.json"
    if not index_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("run_stage_prepare_before_cuda_work")
    receipt = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (receipt.get("status") != "PASS" or receipt.get("input_pins") != pins()
            or receipt.get("runner_sha256") != digest(Path(__file__))
            or digest(index_path) != receipt.get("input_index_sha256")):
        raise RuntimeError("input_index_receipt_or_source_pin_invalid")
    expected_pins = {name: item["sha256"] for name, item in contract()["input_pins"].items()}
    if pins() != expected_pins:
        raise RuntimeError("frozen_input_pin_mismatch")
    neural_rows = _rows_by_id(read_csv(PANEL / "mali_feature_ledger.csv"), "canonical_row_id", "neural_feature")
    index_rows = _rows_by_id(read_csv(index_path), "canonical_observation_id", "input_index")
    panel_rows = read_csv(PANEL / "panel.csv")
    panel_by_id = _rows_by_id(panel_rows, "canonical_observation_id", "panel")
    if set(neural_rows) != set(index_rows) or set(index_rows) != set(panel_by_id) or len(index_rows) != 7350:
        raise RuntimeError("input_index_join_mismatch")
    ledger = {}
    for identity, row in neural_rows.items():
        idx = index_rows[identity]
        if idx["graph_sha256"] != row["graph_artifact_sha256"] or idx["source_id"] != panel_by_id[identity]["source_id"]:
            raise RuntimeError(f"input_index_source_row_mismatch:{identity}")
        ledger[identity] = {**row, "graph_node_count": idx["graph_node_count"],
                            "graph_edge_count": idx["graph_edge_count"]}
    outer = read_csv(PANEL / "outer_splits.csv")
    inner = read_csv(PANEL / "inner_splits.csv")
    if len(outer) != 7350 or {row["canonical_observation_id"] for row in outer} != set(ledger):
        raise RuntimeError("outer_split_coverage_invalid")
    context = {"feature_root": FEATURE_ROOT, "ledger": ledger,
        "source_by_id": {identity: row["source_id"] for identity, row in panel_by_id.items()}, "outer": outer,
        "inner": inner, "technical": {"activation_recomputation_enabled": True},
        "pins": {"contract": digest(CONTRACT), "input_index": digest(index_path),
                 "input_manifest": digest(manifest_path), **pins()}}
    return context


def make_transform(context: dict[str, Any], parts: dict[str, list[str]], fold: int) -> dict[str, Any]:
    import numpy as np

    raw = {identity: [float(context["ledger"][identity][f"global_{i:02d}"]) for i in range(51)]
           for identity in parts["fit"]}
    global_transform = base.fit_transform_global(raw, parts["fit"])
    paths = [(FEATURE_ROOT / context["ledger"][identity]["graph_artifact"]).resolve()
             for identity in parts["fit"]]
    node_transform = base.fit_node_transform(paths, parts["fit"])
    if (global_transform["fit_ids_sha256"] != base.stable_hash(parts["fit"])
            or node_transform["fit_ids_sha256"] != base.stable_hash(parts["fit"])
            or not np.isfinite(global_transform["mean"]).all()
            or not np.isfinite(node_transform["mean"]).all()):
        raise RuntimeError(f"train_only_transform_invalid:{fold}")
    return {"global": global_transform, "node": node_transform, "outer_fold": fold,
            "fit_ids_sha256": base.stable_hash(parts["fit"]), "validation_ids_sha256": base.stable_hash(parts["validation"]),
            "panel_manifest_sha256": pins()["panel_manifest"], "logical_feature_manifest_sha256": pins()["logical_feature_manifest"]}


def partitions(context: dict[str, Any], fold: int) -> dict[str, list[str]]:
    outer = _rows_by_id(context["outer"], "canonical_observation_id", "outer_split")
    if len(outer) != 7350:
        raise RuntimeError("outer_split_not_7350")
    tests = sorted(identity for identity, row in outer.items() if int(row["outer_fold"]) == fold)
    train_ids = set(outer) - set(tests)
    inner_rows = [row for row in context["inner"] if int(row["outer_fold"]) == fold]
    assignments = {row["canonical_observation_id"]: row for row in inner_rows}
    if set(assignments) != train_ids or len(assignments) != len(inner_rows):
        raise RuntimeError(f"inner_fold_assignment_not_exact:{fold}")
    fit = sorted(identity for identity, row in assignments.items() if int(row["inner_fold"]) != 0)
    valid = sorted(identity for identity, row in assignments.items() if int(row["inner_fold"]) == 0)
    group = {identity: row["group_id"] for identity, row in outer.items()}
    if ({group[i] for i in fit} & {group[i] for i in valid + tests}
            or {group[i] for i in valid} & {group[i] for i in tests}):
        raise RuntimeError(f"leakage_group_overlap:{fold}")
    if not fit or not valid or not tests:
        raise RuntimeError(f"empty_partition:{fold}")
    return {"fit": fit, "validation": valid, "test": tests}


def status_load() -> dict[str, Any]:
    if STATUS.is_file():
        state = json.loads(STATUS.read_text(encoding="utf-8"))
        if (state.get("contract_sha256") != digest(CONTRACT) or state.get("input_pins") != pins()):
            if state.get("completed_cells_by_id") or any(
                    int(value.get("completed_epoch", 0)) > 0 for value in state.get("cells", {}).values()
                    if isinstance(value, dict)):
                raise RuntimeError("run_status_identity_changed_after_training_started")
            return {"artifact_id": "real-qpu-common-panel-run", "status": "initialized",
                    "contract_sha256": digest(CONTRACT), "input_pins": pins(), "cells": {},
                    "superseded_contract_sha256": state.get("contract_sha256"),
                    "superseded_stage": state.get("stage")}
        return state
    return {"artifact_id": "real-qpu-common-panel-run", "status": "initialized",
            "contract_sha256": digest(CONTRACT), "input_pins": pins(), "cells": {}}


def update_status(**updates: Any) -> None:
    state = status_load()
    state.update(updates)
    state["updated_at_unix"] = time.time()
    base.atomic_json(STATUS, state)


def _new_model(method: str, transform: dict[str, Any], torch):
    internal = METHODS[method]
    model_class, source_hash = base.load_pinned_model()
    model = base.make_model(model_class, internal, len(transform["global"]["retained_indices"]), checkpoint_layers=True)
    if internal == base.METHODS[0]:
        model.mask = model.mask.to("cuda:0")
    return model.cuda(), source_hash


def technical_profile(context: dict[str, Any], torch, gpu_budget: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    contract_data = contract()
    parts = partitions(context, 0)
    transform = make_transform(context, parts, 0)
    transform_path = OUTPUT / "fold_0" / "transform.json"
    transform_path.parent.mkdir(parents=True, exist_ok=True)
    if transform_path.exists() and json.loads(transform_path.read_text()) != transform:
        raise RuntimeError("fold0_transform_changed_refuse_resume")
    if not transform_path.exists():
        base.atomic_json(transform_path, transform)

    cache = batching.ExampleCache(context, transform, 8 * 1024**3)
    small = sorted(parts["fit"], key=lambda i: int(context["ledger"][i]["graph_node_count"]))[:2]
    parity = batching.gradient_gate(context, transform, cache, small, torch)
    if parity.get("status") != "PASS":
        raise RuntimeError("logical_graph_microbatch_parity_failed")
    order = np.random.default_rng(20261006).permutation(len(parts["fit"]))
    permutation = [parts["fit"][int(i)] for i in order]
    largest = max(parts["fit"], key=lambda i: int(context["ledger"][i]["graph_node_count"]))
    if largest not in permutation[:4096]:
        permutation[4095] = largest
    sample_sets = {n: permutation[:n] for n in sorted({512, 2048, 4096}) if len(permutation) >= n}
    if not all(n in sample_sets for n in (2048, 4096)):
        raise RuntimeError("fold0_fit_too_small_for_frozen_batch_candidates")

    largest_graph = [largest]
    model, _ = _new_model("mali_logical_graph", transform, torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    try:
        with torch.no_grad():
            item = cache.get(largest, METHODS["mali_logical_graph"])
            batch = base._batch_to_device([item], "cuda:0", True)
        del batch, item
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        loss = batching.loss_step(model, optimizer, largest_graph, METHODS["mali_logical_graph"],
                                 250_000, cache, {largest: 0.0}, torch)
        torch.cuda.synchronize()
        if not np.isfinite(loss):
            raise RuntimeError("largest_logical_graph_loss_nonfinite")
        largest_probe = {"status": "PASS", "canonical_observation_id": largest,
                         "nodes": int(context["ledger"][largest]["graph_node_count"]),
                         "edges": int(context["ledger"][largest]["graph_edge_count"]),
                         "loss": loss, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "free_after_bytes": torch.cuda.mem_get_info()[0]}
    finally:
        del model, optimizer
        torch.cuda.empty_cache()

    records = []
    for effective_batch in contract_data["profiling"]["graph_batch_candidates"]:
        n = effective_batch if effective_batch > 512 else 512
        ids = sample_sets[n]
        for node_limit in NODE_CANDIDATES:
            repetitions = []
            failed = None
            for repeat in range(3):
                base.seed_all(20261006 + repeat)
                model, _ = _new_model("mali_logical_graph", transform, torch)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
                synthetic = {identity: 0.0 for identity in ids}
                # Keep file decoding/cache warm-up separate from the timed CUDA work.
                preload_started = time.monotonic()
                for identity in ids:
                    cache.get(identity, METHODS["mali_logical_graph"])
                preload_seconds = time.monotonic() - preload_started
                try:
                    warm_ids = ids[:min(2, len(ids))]
                    batching.loss_step(model, optimizer, warm_ids, METHODS["mali_logical_graph"],
                                       node_limit, cache, synthetic, torch)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.monotonic()
                    losses = [batching.loss_step(model, optimizer, ids[start:start + effective_batch],
                               METHODS["mali_logical_graph"], node_limit, cache, synthetic, torch)
                              for start in range(0, len(ids), effective_batch)]
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - started
                    free = int(torch.cuda.mem_get_info()[0])
                    if not np.isfinite(losses).all() or free < 4 * 1024**3:
                        raise RuntimeError("profile_nonfinite_or_reserve_below_4gib")
                    repetitions.append({"repeat": repeat, "seconds": elapsed,
                        "observations_per_second": len(ids) / elapsed, "preload_seconds": preload_seconds,
                        "optimizer_steps": len(losses), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "free_after_bytes": free})
                except torch.cuda.OutOfMemoryError as exc:
                    failed = f"cuda_oom:{type(exc).__name__}"
                    torch.cuda.empty_cache()
                    break
                finally:
                    del model, optimizer
                    torch.cuda.empty_cache()
            record = {"method": "mali_logical_graph", "effective_batch_size": effective_batch,
                      "node_limit": node_limit, "sample_rows": len(ids), "sample_ids_sha256": base.stable_hash(ids),
                      "uses_observed_targets": False, "repeats": repetitions,
                      "status": "PASS" if len(repetitions) == 3 else "OOM",
                      "error": failed or "", "median_observations_per_second": (
                          float(np.median([r["observations_per_second"] for r in repetitions]))
                          if repetitions else None)}
            records.append(record)
            base.atomic_json(OUTPUT / "batch_probe.json", {"status": "profiling", "records": records,
                          "largest_graph": largest_probe, "microbatch_parity": parity})
            print(json.dumps({"profile": effective_batch, "node_limit": node_limit,
                              "status": record["status"], "median_rows_per_second": record["median_observations_per_second"]}), flush=True)

    successful = [row for row in records if row["status"] == "PASS"]
    if not successful:
        raise RuntimeError("no_successful_graph_batch_profile")
    best = max(row["median_observations_per_second"] for row in successful)
    mlp_records = []
    for effective_batch in contract_data["profiling"]["mlp_batch_candidates"]:
        ids = sample_sets[effective_batch if effective_batch > 512 else 512]
        repetitions = []
        failed = None
        for repeat in range(3):
            base.seed_all(20261016 + repeat)
            model, _ = _new_model("mali_global_mlp", transform, torch)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
            synthetic = {identity: 0.0 for identity in ids}
            try:
                preload_started = time.monotonic()
                for identity in ids:
                    cache.get(identity, METHODS["mali_global_mlp"])
                preload_seconds = time.monotonic() - preload_started
                batching.loss_step(model, optimizer, ids[:min(2, len(ids))], METHODS["mali_global_mlp"],
                                   0, cache, synthetic, torch)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                losses = [batching.loss_step(model, optimizer, ids[start:start + effective_batch],
                           METHODS["mali_global_mlp"], 0, cache, synthetic, torch)
                          for start in range(0, len(ids), effective_batch)]
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                free = int(torch.cuda.mem_get_info()[0])
                if not np.isfinite(losses).all() or free < 4 * 1024**3:
                    raise RuntimeError("mlp_profile_nonfinite_or_reserve_below_4gib")
                repetitions.append({"repeat": repeat, "seconds": elapsed,
                    "observations_per_second": len(ids) / elapsed, "preload_seconds": preload_seconds,
                    "optimizer_steps": len(losses), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "free_after_bytes": free})
            except torch.cuda.OutOfMemoryError as exc:
                failed = f"cuda_oom:{type(exc).__name__}"
                torch.cuda.empty_cache()
                break
            finally:
                del model, optimizer
                torch.cuda.empty_cache()
        mlp_records.append({"method": "mali_global_mlp", "effective_batch_size": effective_batch,
            "sample_rows": len(ids), "sample_ids_sha256": base.stable_hash(ids),
            "uses_observed_targets": False, "repeats": repetitions,
            "status": "PASS" if len(repetitions) == 3 else "OOM", "error": failed or "",
            "median_observations_per_second": (float(np.median([r["observations_per_second"] for r in repetitions]))
                                               if repetitions else None)})
        base.atomic_json(OUTPUT / "batch_probe.json", {"status": "profiling", "records": records,
                      "mlp_records": mlp_records, "largest_graph": largest_probe, "microbatch_parity": parity})
        print(json.dumps({"profile": "mali_global_mlp", "effective_batch_size": effective_batch,
                          "status": mlp_records[-1]["status"]}), flush=True)
    near_best = [row for row in successful if row["median_observations_per_second"] >= best * 0.95]
    feasible_batches = {row["effective_batch_size"] for row in mlp_records if row["status"] == "PASS"}
    candidates = [row for row in near_best if row["effective_batch_size"] in feasible_batches]
    if not candidates:
        raise RuntimeError("no_shared_graph_mlp_batch_profile_within_five_percent")
    chosen = min(candidates,
                 key=lambda row: (row["effective_batch_size"], row["node_limit"]))
    profile = {"status": "PASS", "contract_sha256": digest(CONTRACT), "input_pins": context["pins"],
        "gpu": gpu_budget, "method_pair": list(METHODS),
        "effective_batch_size": chosen["effective_batch_size"], "graph_node_limit": chosen["node_limit"],
        "host_cache_gib": 8, "epochs": EPOCHS, "seeds": list(SEEDS),
        "selection_uses_observed_targets": False, "largest_graph": largest_probe,
        "microbatch_parity": parity, "candidate_results": records, "mlp_candidate_results": mlp_records,
        "sample_policy": contract_data["profiling"]["sample_policy"],
        "runner_sha256": digest(Path(__file__)), "batch_helper_sha256": digest(Path(batching.__file__)),
        "model_runner_sha256": digest(Path(base.__file__))}
    base.atomic_json(OUTPUT / "batch_profile.json", profile)
    base.atomic_json(OUTPUT / "batch_probe.json", {"status": "PASS", "profile_sha256": digest(OUTPUT / "batch_profile.json"),
                  "largest_graph": largest_probe, "microbatch_parity": parity})
    update_status(stage="cuda_canary_and_profile", status="PASS", profile_sha256=digest(OUTPUT / "batch_profile.json"))
    print(json.dumps({"status": "PASS", "effective_batch_size": chosen["effective_batch_size"],
                      "node_limit": chosen["node_limit"], "profile": str(OUTPUT / "batch_profile.json")}, indent=2), flush=True)
    return profile


def validate_fold_zero_gate() -> dict[str, Any]:
    check_path = OUTPUT / "fold_0_gate_receipt.json"
    if not check_path.is_file():
        raise RuntimeError("independent_fold_zero_receipt_missing_run_validator_before_later_folds")
    independent = json.loads(check_path.read_text(encoding="utf-8"))
    if (independent.get("status") != "PASS"
            or independent.get("contract_sha256") != digest(CONTRACT)
            or independent.get("profile_sha256") != digest(OUTPUT / "batch_profile.json")
            or independent.get("validator_sha256") != digest(SCRIPT / "validate_common_panel_execution.py")
            or independent.get("prediction_replay", {}).get("status") != "PASS"):
        raise RuntimeError("independent_fold_zero_gate_failed")
    return independent


def validate_profile_gate() -> dict[str, Any]:
    gate_path = OUTPUT / "profile_validation.json"
    profile_path = OUTPUT / "batch_profile.json"
    if not gate_path.is_file() or not profile_path.is_file():
        raise RuntimeError("independent_cuda_profile_validation_missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (gate.get("status") != "PASS" or gate.get("contract_sha256") != digest(CONTRACT)
            or gate.get("profile_sha256") != digest(profile_path)
            or gate.get("runner_sha256") != digest(Path(__file__))
            or gate.get("validator_sha256") != digest(SCRIPT / "validate_common_panel_execution.py")
            or gate.get("input_index_sha256") != digest(OUTPUT / "input_index.csv")):
        raise RuntimeError("independent_cuda_profile_validation_pin_mismatch")
    return gate


def fit_cell(context: dict[str, Any], parts: dict[str, list[str]], transform: dict[str, Any],
             method: str, fold: int, seed: int, profile: dict[str, Any],
             all_targets: dict[str, float], torch) -> dict[str, Any]:
    import numpy as np

    internal = METHODS[method]
    graph = internal == base.METHODS[0]
    identity = {"method": method, "internal_model_variant": internal, "outer_fold": fold, "seed": seed,
        "contract_sha256": digest(CONTRACT), "input_pins": context["pins"],
        "fit_ids_sha256": base.stable_hash(parts["fit"]),
        "validation_ids_sha256": base.stable_hash(parts["validation"]),
        "test_ids_sha256": base.stable_hash(parts["test"]), "transform_sha256": base.stable_hash(transform),
        "profile_sha256": digest(OUTPUT / "batch_profile.json"),
        "runner_sha256": digest(Path(__file__)), "batch_helper_sha256": digest(Path(batching.__file__)),
        "model_runner_sha256": digest(Path(base.__file__)),
        "source_model_sha256": base.load_pinned_model()[1],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "precision": "float32"}
    folder = OUTPUT / f"fold_{fold}" / method / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "checkpoint.pt"
    progress_file = folder / "cell_progress.json"
    manifest_file = folder / "cell_manifest.json"
    predictions_file = folder / "predictions.csv"
    failures_file = folder / "test_failures.csv"
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise RuntimeError(f"completed_cell_identity_mismatch:{method}:{fold}:{seed}")
        for name, expected in manifest["output_hashes"].items():
            if digest(folder / name) != expected:
                raise RuntimeError(f"completed_cell_output_hash_mismatch:{method}:{fold}:{seed}:{name}")
        return manifest

    base.seed_all(seed)
    model, source_model_hash = _new_model(method, transform, torch)
    if source_model_hash != identity["source_model_sha256"]:
        raise RuntimeError("source_model_hash_changed_during_cell_setup")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    start_epoch, best_loss, best_epoch, best_state, prior_seconds = 0, float("inf"), -1, None, 0.0
    if checkpoint.exists():
        saved = base.load_saved_checkpoint(checkpoint, identity, torch)
        if (saved.get("fit_ids_sha256") != identity["fit_ids_sha256"]
                or saved.get("validation_ids_sha256") != identity["validation_ids_sha256"]):
            raise RuntimeError("checkpoint_partition_identity_mismatch")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        base.restore_rng_state(saved["rng_state"])
        start_epoch = int(saved["next_epoch"])
        best_loss = float(saved["best_validation_mse"])
        best_epoch = int(saved["best_epoch"])
        best_state = saved["best_model_state"]
        prior_seconds = float(saved["elapsed_seconds"])
    cache = batching.ExampleCache(context, transform, 8 * 1024**3)
    targets = {identity_: all_targets[identity_] for identity_ in parts["fit"] + parts["validation"]}
    started = time.monotonic()
    batch_size = int(profile["effective_batch_size"])
    node_limit = int(profile["graph_node_limit"])
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        order = np.random.default_rng(int(base.stable_hash([fold, seed, epoch])[:16], 16)).permutation(len(parts["fit"]))
        ids = [parts["fit"][int(i)] for i in order]
        losses = []
        for begin in range(0, len(ids), batch_size):
            group = ids[begin:begin + batch_size]
            losses.append(batching.loss_step(model, optimizer, group, internal, node_limit, cache, targets, torch))
        scheduler.step()
        train_loss = float(np.mean(losses)) if losses else float("nan")
        model.eval()
        val_error = 0.0
        with torch.no_grad():
            for group in batching.physical_groups(parts["validation"], context["ledger"], node_limit, graph):
                examples = [cache.get(identity_, internal) for identity_ in group]
                batch = base._batch_to_device(examples, "cuda:0", graph)
                prediction = base._prediction(model, batch)
                actual = torch.tensor([targets[i] for i in group], dtype=torch.float32, device="cuda:0")
                val_error += float(torch.nn.functional.mse_loss(prediction, actual, reduction="sum").cpu())
                del batch, prediction, actual, examples
        validation_mse = val_error / len(parts["validation"])
        if not np.isfinite(train_loss) or not np.isfinite(validation_mse):
            raise RuntimeError(f"nonfinite_train_or_validation_loss:{method}:{fold}:{seed}:{epoch + 1}")
        if validation_mse < best_loss:
            best_loss, best_epoch = validation_mse, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        completed = epoch + 1
        elapsed = prior_seconds + time.monotonic() - started
        torch.cuda.synchronize()
        payload = {"identity": identity, "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "rng_state": base.capture_rng_state(), "completed_epoch": completed, "next_epoch": completed,
            "best_validation_mse": best_loss, "best_epoch": best_epoch, "best_model_state": best_state,
            "fit_ids_sha256": identity["fit_ids_sha256"], "validation_ids_sha256": identity["validation_ids_sha256"],
            "transform": transform, "elapsed_seconds": elapsed, "train_loss": train_loss,
            "validation_mse": validation_mse}
        base.save_checkpoint(checkpoint, payload)
        epoch_record = {"status": "training", "method": method, "outer_fold": fold, "seed": seed,
            "completed_epoch": completed, "epochs": EPOCHS, "best_epoch": best_epoch,
            "train_mse_raw_seconds_squared": train_loss, "validation_mse_raw_seconds_squared": validation_mse,
            "epoch_seconds": time.monotonic() - epoch_start, "elapsed_seconds": elapsed,
            "effective_batch_size": batch_size, "graph_node_limit": node_limit,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "host_cache_bytes": cache.bytes,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "checkpoint_sha256": digest(checkpoint), "identity": identity}
        base.atomic_json(progress_file, epoch_record)
        state = status_load()
        state.setdefault("cells", {})[f"{method}/fold_{fold}/seed_{seed}"] = epoch_record
        state["stage"] = "training"
        state["status"] = "running"
        base.atomic_json(STATUS, state)
        print(json.dumps({"cell": f"{method}/fold_{fold}/seed_{seed}", "epoch": completed,
                          "validation_mse": validation_mse, "epoch_seconds": epoch_record["epoch_seconds"]}), flush=True)

    if best_state is None or best_epoch < 0:
        raise RuntimeError("best_validation_state_missing")
    model.load_state_dict(best_state)
    model.eval()
    predictions, failures = [], []
    # Test inference sees features and IDs; target seconds enter only in later scoring.
    with torch.no_grad():
        for identity_ in parts["test"]:
            if not graph or context["ledger"][identity_]["graph_status"] == "available":
                try:
                    example = cache.get(identity_, internal)
                    batch = base._batch_to_device([example], "cuda:0", graph)
                    value = float(base._prediction(model, batch).detach().cpu().reshape(-1)[0])
                    if not np.isfinite(value):
                        raise RuntimeError("nonfinite_test_prediction")
                    predictions.append({"canonical_observation_id": identity_, "outer_fold": fold,
                                        "method_id": method, "seed": seed, "predicted_seconds": value,
                                        "status": "predicted", "terminal_reason": ""})
                    del example, batch
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    failures.append({"canonical_observation_id": identity_, "outer_fold": fold,
                                     "method_id": method, "seed": seed, "status": "resource_failure",
                                     "terminal_reason": "cuda_oom_singleton_test_inference"})
            else:
                failures.append({"canonical_observation_id": identity_, "outer_fold": fold,
                                 "method_id": method, "seed": seed, "status": "unavailable",
                                 "terminal_reason": "logical_graph_input_unavailable"})
    prediction_fields = ["canonical_observation_id", "outer_fold", "method_id", "seed", "predicted_seconds", "status", "terminal_reason"]
    failure_fields = ["canonical_observation_id", "outer_fold", "method_id", "seed", "status", "terminal_reason"]
    base.write_csv_atomic(predictions_file, predictions, prediction_fields)
    base.write_csv_atomic(failures_file, failures, failure_fields)
    output_hashes = {name: digest(folder / name) for name in ("checkpoint.pt", "predictions.csv", "test_failures.csv")}
    result = {"status": "complete", "terminal_status": "completed", "method": method, "outer_fold": fold,
        "seed": seed, "identity": identity, "epochs": EPOCHS, "best_epoch": best_epoch,
        "best_validation_mse_raw_seconds_squared": best_loss, "n_fit": len(parts["fit"]),
        "n_validation": len(parts["validation"]), "n_test_assigned": len(parts["test"]),
        "n_test_predicted": len(predictions), "n_test_failures": len(failures),
        "fit_ids_sha256": identity["fit_ids_sha256"], "validation_ids_sha256": identity["validation_ids_sha256"],
        "test_ids_sha256": identity["test_ids_sha256"], "elapsed_seconds": prior_seconds + time.monotonic() - started,
        "output_hashes": output_hashes}
    base.atomic_json(manifest_file, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "profile", "train"))
    parser.add_argument("--folds", type=int, nargs="+", choices=range(5), default=[0])
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
        return
    context = load_context()
    if args.stage == "profile":
        profile_path = OUTPUT / "batch_profile.json"
        if profile_path.is_file():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            if (profile.get("status") == "PASS" and profile.get("contract_sha256") == digest(CONTRACT)
                    and profile.get("runner_sha256") == digest(Path(__file__))
                    and (OUTPUT / "profile_validation.json").is_file()):
                print(json.dumps({"status": "profile_already_present_use_validated_profile",
                                  "profile_sha256": digest(profile_path)}, flush=True))
                return
            raise RuntimeError("refuse_to_overwrite_existing_batch_profile")
        os.environ["OMP_NUM_THREADS"] = "12"
        os.environ["MKL_NUM_THREADS"] = "12"
        import torch
        torch.set_num_threads(12)
        torch.set_num_interop_threads(4)
        from gpu_lease import acquire_gpu_lease
        lease = acquire_gpu_lease("common-panel-mali-profile")
        try:
            gpu_budget = base.validate_gpu_budget(torch, budget_gib=12, reserve_gib=4)
            technical_profile(context, torch, gpu_budget)
        finally:
            lease.release()
        return
    if args.folds != [0] and args.folds != [1, 2, 3, 4]:
        raise SystemExit("train fold must be [0] before [1,2,3,4]")
    if not (OUTPUT / "batch_profile.json").is_file():
        raise RuntimeError("batch_profile_missing")
    profile = json.loads((OUTPUT / "batch_profile.json").read_text(encoding="utf-8"))
    if profile.get("status") != "PASS" or profile.get("input_pins") != context["pins"]:
        raise RuntimeError("batch_profile_pin_mismatch")
    validate_profile_gate()
    if args.folds != [0]:
        validate_fold_zero_gate()
    os.environ["OMP_NUM_THREADS"] = "12"
    os.environ["MKL_NUM_THREADS"] = "12"
    import torch
    torch.set_num_threads(12)
    torch.set_num_interop_threads(4)
    from gpu_lease import acquire_gpu_lease
    lease = acquire_gpu_lease("common-panel-mali-training")
    try:
        resource_receipt = base.validate_gpu_budget(torch, budget_gib=12, reserve_gib=4)
        state = status_load()
        state.update({"stage": "training", "status": "running", "gpu": resource_receipt,
                      "expected_neural_cells": 30, "completed_cells": sum(
                          1 for p in OUTPUT.glob("fold_*/mali_*/seed_*/cell_manifest.json")),
                      "profile_sha256": digest(OUTPUT / "batch_profile.json")})
        base.atomic_json(STATUS, state)
        for fold in args.folds:
            parts = partitions(context, fold)
            training_ids = set(parts["fit"]) | set(parts["validation"])
            all_targets = {row["canonical_observation_id"]: float(row["target_seconds"])
                           for row in read_csv(PANEL / "targets.csv")
                           if row["canonical_observation_id"] in training_ids}
            if set(all_targets) != training_ids:
                raise RuntimeError(f"fold_training_targets_not_exact:{fold}")
            transform = make_transform(context, parts, fold)
            transform_file = OUTPUT / f"fold_{fold}" / "transform.json"
            transform_file.parent.mkdir(parents=True, exist_ok=True)
            if transform_file.exists() and json.loads(transform_file.read_text(encoding="utf-8")) != transform:
                raise RuntimeError(f"fold_transform_identity_mismatch:{fold}")
            if not transform_file.exists():
                base.atomic_json(transform_file, transform)
            fold_results = []
            for method in METHODS:
                for seed in SEEDS:
                    result = fit_cell(context, parts, transform, method, fold, seed, profile, all_targets, torch)
                    fold_results.append(result)
                    state = status_load()
                    state.setdefault("completed_cells_by_id", {})[f"{method}/fold_{fold}/seed_{seed}"] = {
                        "status": result["status"], "identity": result["identity"],
                        "manifest_sha256": digest(OUTPUT / f"fold_{fold}" / method / f"seed_{seed}" / "cell_manifest.json")}
                    state["completed_cells"] = len(state["completed_cells_by_id"])
                    base.atomic_json(STATUS, state)
            if len(fold_results) != 6 or any(r.get("status") != "complete" for r in fold_results):
                raise RuntimeError(f"fold_cell_completion_mismatch:{fold}")
            if len({r["fit_ids_sha256"] for r in fold_results}) != 1 or len({r["validation_ids_sha256"] for r in fold_results}) != 1:
                raise RuntimeError(f"matched_graph_mlp_partition_hash_mismatch:{fold}")
            if fold == 0:
                receipt = {"status": "TRAINED_AWAITING_INDEPENDENT_TECHNICAL_VALIDATION", "outer_fold": 0,
                           "cell_count": 6, "cell_manifests": [
                               {"path": str(OUTPUT / f"fold_0" / r["method"] / f"seed_{r['seed']}" / "cell_manifest.json"),
                                "sha256": digest(OUTPUT / "fold_0" / r["method"] / f"seed_{r['seed']}" / "cell_manifest.json")}
                               for r in fold_results], "fit_ids_sha256": fold_results[0]["fit_ids_sha256"],
                           "validation_ids_sha256": fold_results[0]["validation_ids_sha256"],
                           "profile_sha256": digest(OUTPUT / "batch_profile.json"), "contract_sha256": digest(CONTRACT)}
                base.atomic_json(OUTPUT / "fold_0_technical_qa.json", receipt)
                state = status_load()
                state.update({"stage": "awaiting_independent_fold0_validation", "status": "awaiting_validation",
                              "fold0_technical_qa_sha256": digest(OUTPUT / "fold_0_technical_qa.json")})
                base.atomic_json(STATUS, state)
                if args.folds == [0]:
                    print(json.dumps({"status": "awaiting_independent_fold0_validation",
                                      "automatic_next_folds": [1, 2, 3, 4]}, flush=True))
                    return
    finally:
        lease.release()


if __name__ == "__main__":
    main()
