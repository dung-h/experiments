#!/usr/bin/env python3
"""Independently check common-panel input, profile and saved neural cells."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmark_v1/scripts"
CONTRACT = ROOT / "benchmark_v1/protocol/common_panel_completion.json"
PANEL = ROOT / "artifacts/real_qpu/common_panel"
OUTPUT = PANEL / "mali"
sys.path.insert(0, str(SCRIPT))

import run_common_panel_mali as run  # noqa: E402
import run_mali_batched as batching  # noqa: E402
import run_mali_full_features as model_runner  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_torch(torch) -> None:
    try:
        torch.set_num_threads(12)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(4)
    except RuntimeError:
        pass


def check_pins() -> dict[str, str]:
    spec = run.contract()
    actual = {key: sha(ROOT / value["path"]) for key, value in spec["input_pins"].items()}
    expected = {key: value["sha256"] for key, value in spec["input_pins"].items()}
    if actual != expected:
        raise RuntimeError("frozen_input_hash_mismatch")
    if sha(ROOT / spec["precedence"]["parent_contract"]) != spec["precedence"]["parent_sha256"]:
        raise RuntimeError("parent_panel_contract_hash_mismatch")
    return actual


def audit_inputs() -> dict[str, Any]:
    spec = run.contract()
    pins = check_pins()
    panel = csv_rows(PANEL / "panel.csv")
    targets = csv_rows(PANEL / "targets.csv")
    outer = csv_rows(PANEL / "outer_splits.csv")
    inner = csv_rows(PANEL / "inner_splits.csv")
    index_path = OUTPUT / "input_index.csv"
    index_manifest = read_json(OUTPUT / "input_index_manifest.json")
    index = csv_rows(index_path)
    id_field = "canonical_observation_id"
    def keyed(rows, name):
        result = {row[id_field]: row for row in rows}
        if len(rows) != len(result):
            raise RuntimeError(f"duplicate_{name}_identity")
        return result
    p, t, i, o = keyed(panel, "panel"), keyed(targets, "target"), keyed(index, "input_index"), keyed(outer, "outer")
    ids = set(p)
    if len(ids) != 7350 or set(t) != ids or set(i) != ids or set(o) != ids:
        raise RuntimeError("input_coverage_not_7350")
    if index_manifest.get("status") != "PASS" or index_manifest.get("input_pins") != pins or index_manifest.get("input_index_sha256") != sha(index_path):
        raise RuntimeError("input_index_manifest_mismatch")
    if index_manifest.get("runner_sha256") != sha(SCRIPT / "run_common_panel_mali.py"):
        raise RuntimeError("input_index_builder_hash_mismatch")
    forbidden = {"target_seconds", "actual_seconds", "shots"}
    if forbidden & set(index[0]):
        raise RuntimeError("label_or_shot_column_leaked_into_batch_profile_index")
    groups = {identity: row["group_id"] for identity, row in o.items()}
    sources = {identity: row["source_id"] for identity, row in p.items()}
    required_sources = {row["source_id"] for row in panel}
    frozen_panel_manifest = read_json(PANEL / "manifest.json")
    if len({row["group_id"] for row in o.values()}) != frozen_panel_manifest.get("groups"):
        raise RuntimeError("panel_group_count_mismatch")
    for fold in range(5):
        test = {identity for identity, row in o.items() if int(row["outer_fold"]) == fold}
        train = ids - test
        if ({sources[identity] for identity in test} != required_sources
                or {sources[identity] for identity in train} != required_sources):
            raise RuntimeError(f"source_missing_from_outer_partition:{fold}")
        assignments = [row for row in inner if int(row["outer_fold"]) == fold]
        inner_by_id = {row[id_field]: row for row in assignments}
        if set(inner_by_id) != ids - test or len(inner_by_id) != len(assignments):
            raise RuntimeError(f"inner_partition_identity_mismatch:{fold}")
        if {groups[x] for x in test} & {row["group_id"] for row in assignments}:
            raise RuntimeError(f"outer_group_leakage:{fold}")
        for held in range(4):
            left = {row["group_id"] for row in assignments if int(row["inner_fold"]) == held}
            right = {row["group_id"] for row in assignments if int(row["inner_fold"]) != held}
            if left & right:
                raise RuntimeError(f"inner_group_leakage:{fold}:{held}")
            held_ids = {row[id_field] for row in assignments if int(row["inner_fold"]) == held}
            fit_ids = {row[id_field] for row in assignments if int(row["inner_fold"]) != held}
            if ({sources[identity] for identity in held_ids} != required_sources
                    or {sources[identity] for identity in fit_ids} != required_sources):
                raise RuntimeError(f"source_missing_from_inner_partition:{fold}:{held}")
    checked_graphs: set[str] = set()
    graph_root = ROOT / "artifacts/real_qpu/logical_inputs"
    for identity in ids:
        expected, observed = p[identity], i[identity]
        if (expected["source_id"] != observed["source_id"]
                or expected["logical_input_tier"] != observed["logical_input_tier"]
                or expected["graph_sha256"] != observed["graph_sha256"]
                or expected["group_id"] != o[identity]["group_id"]):
            raise RuntimeError(f"panel_input_index_join_mismatch:{identity}")
        y, shots = float(t[identity]["target_seconds"]), float(t[identity]["shots"])
        if not math.isfinite(y) or y <= 0 or not math.isfinite(shots) or shots <= 0:
            raise RuntimeError(f"invalid_target_or_shots:{identity}")
        relative = observed["graph_artifact"]
        if relative not in checked_graphs:
            graph_path = (graph_root / relative).resolve()
            if graph_root.resolve() not in graph_path.parents or not graph_path.is_file():
                raise RuntimeError(f"logical_graph_missing_or_outside_root:{identity}")
            if sha(graph_path) != observed["graph_sha256"]:
                raise RuntimeError(f"logical_graph_file_hash_mismatch:{identity}")
            with np.load(graph_path, allow_pickle=False) as archive:
                record = {name: archive[name] for name in archive.files}
                dense = model_runner.expand_dense_x(record)
                edges = record["edge_index"]
                if (dense.shape != (int(observed["graph_node_count"]), 178)
                        or not np.isfinite(dense).all() or edges.ndim != 2 or edges.shape[0] != 2
                        or (edges.size and (edges.min() < 0 or edges.max() >= dense.shape[0]))
                        or int(edges.shape[1]) != int(observed["graph_edge_count"])):
                    raise RuntimeError(f"logical_graph_schema_invalid:{identity}")
            checked_graphs.add(relative)
    feature_path = ROOT / "artifacts/real_qpu/logical_inputs/feature_attempts.csv"
    feature_rows = csv_rows(feature_path)
    features = {row["canonical_observation_id"]: row for row in feature_rows}
    ledger_rows = csv_rows(PANEL / "mali_feature_ledger.csv")
    ledger = {row["canonical_row_id"]: row for row in ledger_rows}
    if (len(features) != len(feature_rows) or not ids <= set(features)
            or len(ledger) != 7350 or len(ledger) != len(ledger_rows)):
        raise RuntimeError("logical_input_source_feature_coverage_invalid")
    for identity in ids:
        source, derived = features[identity], ledger[identity]
        if (source["status"] != "available" or source["graph_file"] != i[identity]["graph_artifact"]
                or source["graph_sha256"] != i[identity]["graph_sha256"]
                or int(source["node_count"]) != int(i[identity]["graph_node_count"])):
            raise RuntimeError(f"logical_graph_source_identity_mismatch:{identity}")
        raw = [float(source[f"g{k:02d}"]) for k in range(51)]
        copied = [float(derived[f"global_{k:02d}"]) for k in range(51)]
        if raw != copied or not all(math.isfinite(value) for value in raw):
            raise RuntimeError(f"51_global_feature_copy_mismatch:{identity}")
    source_counts = {}
    for row in panel:
        source_counts[row["source_id"]] = source_counts.get(row["source_id"], 0) + 1
    if source_counts != {"mali_real_qpu": 340, "qonductor_single_circuit_ibm": 3065, "qpack_mcp": 3945}:
        raise RuntimeError("source_denominator_mismatch")
    largest_identity = max(i, key=lambda identity: int(i[identity]["graph_node_count"]))
    return {"status": "PASS", "rows": 7350, "source_counts": source_counts,
            "groups": len({row["group_id"] for row in outer}),
            "unique_graph_files": len(checked_graphs),
            "largest_graph_id": largest_identity,
            "largest_graph_nodes": int(i[largest_identity]["graph_node_count"]),
            "outer_inner_group_leakage": False,
            "profile_index_has_no_targets_or_shots": True,
            "input_index_sha256": sha(index_path), "input_pins": pins}


def audit_profile() -> dict[str, Any]:
    inputs = audit_inputs()
    profile_path = OUTPUT / "batch_profile.json"
    profile = read_json(profile_path)
    spec = run.contract()
    profile_sha = sha(profile_path)
    if profile.get("status") != "PASS" or profile.get("contract_sha256") != sha(CONTRACT):
        raise RuntimeError("batch_profile_contract_mismatch")
    if profile.get("input_pins") != {"contract": sha(CONTRACT), "input_index": inputs["input_index_sha256"],
                                     "input_manifest": sha(OUTPUT / "input_index_manifest.json"), **inputs["input_pins"]}:
        raise RuntimeError("batch_profile_input_identity_mismatch")
    if (profile.get("runner_sha256") != sha(SCRIPT / "run_common_panel_mali.py")
            or profile.get("batch_helper_sha256") != sha(SCRIPT / "run_mali_batched.py")
            or profile.get("model_runner_sha256") != sha(SCRIPT / "run_mali_full_features.py")):
        raise RuntimeError("batch_profile_implementation_hash_mismatch")
    candidates = spec["profiling"]["graph_batch_candidates"]
    records = profile.get("candidate_results", [])
    if {(int(r["effective_batch_size"]), int(r["node_limit"])) for r in records} != {
            (int(batch), int(nodes)) for batch in candidates for nodes in spec["profiling"]["node_budget_candidates"]}:
        raise RuntimeError("graph_batch_profile_candidate_grid_incomplete")
    context = run.load_context()
    parts = run.partitions(context, 0)
    permutation = np.random.default_rng(spec["profiling"]["probe_seed"]).permutation(len(parts["fit"]))
    ordered_ids = [parts["fit"][int(index)] for index in permutation]
    largest_fit = max(parts["fit"], key=lambda identity: int(context["ledger"][identity]["graph_node_count"]))
    if largest_fit not in ordered_ids[:4096]:
        ordered_ids[4095] = largest_fit
    samples = {size: ordered_ids[:size] for size in (512, 2048, 4096)}
    for record in records:
        if record.get("status") not in {"PASS", "OOM"} or (
                record.get("status") == "OOM" and not record.get("error")):
            raise RuntimeError("graph_batch_profile_has_unclassified_candidate_state")
        size = int(record["effective_batch_size"])
        expected_ids = samples[size if size > 512 else 512]
        if (int(record["sample_rows"]) != len(expected_ids)
                or record["sample_ids_sha256"] != model_runner.stable_hash(expected_ids)
                or record.get("uses_observed_targets") is not False):
            raise RuntimeError("graph_batch_profile_sample_identity_or_label_leakage")
        if record["status"] == "PASS":
            reps = record.get("repeats", [])
            if len(reps) != 3 or any(
                    not math.isfinite(float(row["observations_per_second"]))
                    or float(row["observations_per_second"]) <= 0
                    or not math.isfinite(float(row["seconds"])) or float(row["seconds"]) <= 0
                    or int(row["free_after_bytes"]) < 4 * 1024**3 for row in reps):
                raise RuntimeError("graph_batch_profile_repetition_or_vram_reserve_failure")
    mlp = profile.get("mlp_candidate_results", [])
    if {int(row["effective_batch_size"]) for row in mlp} != set(spec["profiling"]["mlp_batch_candidates"]):
        raise RuntimeError("mlp_batch_profile_grid_incomplete")
    for record in mlp:
        if record.get("status") not in {"PASS", "OOM"} or (
                record.get("status") == "OOM" and not record.get("error")):
            raise RuntimeError("mlp_batch_profile_has_unclassified_candidate_state")
        size = int(record["effective_batch_size"])
        expected_ids = samples[size if size > 512 else 512]
        if (int(record["sample_rows"]) != len(expected_ids)
                or record["sample_ids_sha256"] != model_runner.stable_hash(expected_ids)
                or record.get("uses_observed_targets") is not False):
            raise RuntimeError("mlp_batch_profile_sample_identity_or_label_leakage")
        if record["status"] == "PASS":
            reps = record.get("repeats", [])
            if len(reps) != 3 or any(float(row["seconds"]) <= 0
                                      or float(row["observations_per_second"]) <= 0
                                      or int(row["free_after_bytes"]) < 4 * 1024**3
                                      for row in reps):
                raise RuntimeError("mlp_batch_profile_repetition_or_vram_reserve_failure")
    selected = int(profile["effective_batch_size"])
    if not any(int(row["effective_batch_size"]) == selected and row["status"] == "PASS" for row in mlp):
        raise RuntimeError("selected_batch_not_feasible_for_matched_mlp")
    if int(profile["graph_node_limit"]) not in spec["profiling"]["node_budget_candidates"]:
        raise RuntimeError("selected_node_budget_outside_contract")
    graph_success = [row for row in records if row["status"] == "PASS"]
    if not graph_success:
        raise RuntimeError("no_successful_graph_profile_candidate")
    best = max(float(row["median_observations_per_second"]) for row in graph_success)
    mlp_feasible = {int(row["effective_batch_size"]) for row in mlp if row["status"] == "PASS"}
    expected_choice = min(
        (row for row in graph_success
         if float(row["median_observations_per_second"]) >= best * 0.95
         and int(row["effective_batch_size"]) in mlp_feasible),
        key=lambda row: (int(row["effective_batch_size"]), int(row["node_limit"])),
    )
    if (selected != int(expected_choice["effective_batch_size"])
            or int(profile["graph_node_limit"]) != int(expected_choice["node_limit"])):
        raise RuntimeError("batch_profile_selection_rule_mismatch")
    if (profile.get("largest_graph", {}).get("status") != "PASS"
            or int(profile["largest_graph"]["nodes"]) != inputs["largest_graph_nodes"]
            or profile["largest_graph"].get("canonical_observation_id") != inputs["largest_graph_id"]):
        raise RuntimeError("largest_graph_canary_missing")
    parity = profile.get("microbatch_parity", {})
    expected_fixture = sorted(parts["fit"], key=lambda identity: int(context["ledger"][identity]["graph_node_count"]))[:2]
    if (parity.get("status") != "PASS" or not parity.get("checkpoint_resume")
            or parity.get("fixture_ids") != expected_fixture
            or float(parity.get("gradient_max_abs_difference", math.inf)) > 1e-4
            or float(parity.get("optimizer_max_abs_difference", math.inf)) > 1e-4):
        raise RuntimeError("microbatch_parity_or_resume_check_missing")
    return {"status": "PASS", "input_audit": inputs, "profile_sha256": profile_sha,
            "selected_batch": selected, "node_budget": profile["graph_node_limit"],
            "profile_candidates": len(records), "mlp_candidates": len(mlp),
            "largest_graph_nodes": profile["largest_graph"]["nodes"],
            "gradient_max_abs_difference": parity.get("gradient_max_abs_difference"),
            "optimizer_max_abs_difference": parity.get("optimizer_max_abs_difference")}


def independent_cuda_profile_gate() -> dict[str, Any]:
    """Re-run numerical parity and the real largest-graph resource check independently."""
    import copy
    import torch
    configure_torch(torch)
    import torch.nn.functional as F

    from gpu_lease import acquire_gpu_lease

    configure_torch(torch)
    lease = acquire_gpu_lease("common-panel-independent-profile-validation")
    try:
        gpu = model_runner.validate_gpu_budget(torch, budget_gib=12, reserve_gib=4)
        context = run.load_context()
        parts = run.partitions(context, 0)
        transform = run.make_transform(context, parts, 0)
        transform_path = OUTPUT / "fold_0" / "transform.json"
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        if transform_path.exists() and read_json(transform_path) != transform:
            raise RuntimeError("independently_recomputed_fold0_transform_mismatch")
        if not transform_path.exists():
            model_runner.atomic_json(transform_path, transform)

        small = sorted(parts["fit"], key=lambda identity: int(context["ledger"][identity]["graph_node_count"]))[:2]
        cache = batching.ExampleCache(context, transform, 8 * 1024**3)
        model_class, model_hash = model_runner.load_pinned_model()
        variant = run.METHODS["mali_logical_graph"]

        def new_model():
            model = model_runner.make_model(model_class, variant,
                len(transform["global"]["retained_indices"]), checkpoint_layers=True).cuda()
            model.mask = model.mask.to("cuda:0")
            return model

        targets = {identity: float(index + 1) for index, identity in enumerate(small)}

        def independent_step(model, optimizer, batches):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for ids in batches:
                examples = [cache.get(identity, variant) for identity in ids]
                data = model_runner._batch_to_device(examples, "cuda:0", True)
                predicted = model_runner._prediction(model, data)
                actual = torch.tensor([targets[cid] for cid in ids], dtype=torch.float32, device="cuda:0")
                loss = F.mse_loss(predicted, actual, reduction="sum") / len(small)
                loss.backward()
                total += float(loss.detach().cpu())
                del examples, data, predicted, actual, loss
            optimizer.step()
            return total

        model_runner.seed_all(42)
        full, micro = new_model(), None
        micro = copy.deepcopy(full)
        optimizer_full = torch.optim.Adam(full.parameters(), lr=0.0005, weight_decay=0.0001)
        optimizer_micro = torch.optim.Adam(micro.parameters(), lr=0.0005, weight_decay=0.0001)
        full_loss = independent_step(full, optimizer_full, [small])
        micro_loss = independent_step(micro, optimizer_micro, [[identity] for identity in small])
        torch.cuda.synchronize()
        max_gradient = 0.0
        max_parameter = 0.0
        for (name, left), (_, right) in zip(full.named_parameters(), micro.named_parameters()):
            max_gradient = max(max_gradient, float((left.grad - right.grad).abs().max()))
            max_parameter = max(max_parameter, float((left - right).abs().max()))
            if (not torch.allclose(left.grad, right.grad, rtol=1e-5, atol=1e-6)
                    or not torch.allclose(left, right, rtol=1e-5, atol=1e-5)):
                raise RuntimeError(f"independent_microbatch_parity_failed:{name}")
        if not math.isclose(full_loss, micro_loss, rel_tol=1e-5, abs_tol=1e-6):
            raise RuntimeError("independent_microbatch_loss_mismatch")

        # Verify optimizer/RNG continuation from a real-feature one-step state.
        identity = {"gate": "common-panel", "fixture_ids": small}
        payload = {"identity": identity, "model_state": full.state_dict(),
                   "optimizer_state": optimizer_full.state_dict(), "rng_state": model_runner.capture_rng_state()}
        import tempfile
        with tempfile.TemporaryDirectory(prefix="qre-common-panel-resume-") as temp:
            checkpoint = Path(temp) / "checkpoint.pt"
            model_runner.save_checkpoint(checkpoint, payload)
            independent_step(full, optimizer_full, [[identity_] for identity_ in small])
            expected_state = {key: value.detach().cpu().clone() for key, value in full.state_dict().items()}
            resumed = new_model()
            optimizer_resumed = torch.optim.Adam(resumed.parameters(), lr=0.0005, weight_decay=0.0001)
            saved = model_runner.load_saved_checkpoint(checkpoint, identity, torch)
            resumed.load_state_dict(saved["model_state"])
            optimizer_resumed.load_state_dict(saved["optimizer_state"])
            model_runner.restore_rng_state(saved["rng_state"])
            independent_step(resumed, optimizer_resumed, [[identity_] for identity_ in small])
            resume_max = max(float((expected_state[key] - value.detach().cpu()).abs().max())
                             for key, value in resumed.state_dict().items())
            if any(not torch.allclose(expected_state[key], value.detach().cpu(), rtol=1e-5, atol=1e-6)
                   for key, value in resumed.state_dict().items()):
                raise RuntimeError("independent_checkpoint_resume_mismatch")

        all_ids = set(context["ledger"])
        largest_nodes = max(int(context["ledger"][identity_]["graph_node_count"]) for identity_ in all_ids)
        largest = min(identity_ for identity_ in all_ids
                      if int(context["ledger"][identity_]["graph_node_count"]) == largest_nodes)
        targets[largest] = 0.0
        largest_model = new_model()
        largest_optimizer = torch.optim.Adam(largest_model.parameters(), lr=0.0005, weight_decay=0.0001)
        cache.get(largest, variant)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        independent_step(largest_model, largest_optimizer, [[largest]])
        torch.cuda.synchronize()
        free, _ = torch.cuda.mem_get_info()
        largest_record = {"canonical_observation_id": largest,
            "nodes": int(context["ledger"][largest]["graph_node_count"]),
            "edges": int(context["ledger"][largest]["graph_edge_count"]),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "free_after_bytes": int(free)}
        input_audit = audit_inputs()
        if free < 4 * 1024**3 or largest_record["nodes"] != input_audit["largest_graph_nodes"]:
            raise RuntimeError(f"independent_largest_graph_resource_gate_failed:free={free}:nodes={largest_record['nodes']}:expected={input_audit['largest_graph_nodes']}")
        profile = audit_profile()
        receipt = {"status": "PASS", "validator_sha256": sha(Path(__file__)),
            "runner_sha256": sha(SCRIPT / "run_common_panel_mali.py"),
            "contract_sha256": sha(CONTRACT), "profile_sha256": profile["profile_sha256"],
            "input_index_sha256": sha(OUTPUT / "input_index.csv"),
            "source_model_sha256": model_hash, "gpu": gpu, "fixture_ids": small,
            "full_loss": full_loss, "microbatch_loss": micro_loss,
            "max_gradient_abs_difference": max_gradient,
            "max_parameter_abs_difference": max_parameter, "checkpoint_resume_max_abs_difference": resume_max,
            "checkpoint_resume": True, "largest_graph": largest_record, "validated_at_unix": time.time()}
        model_runner.atomic_json(OUTPUT / "profile_validation.json", receipt)
        return receipt
    finally:
        lease.release()


def audit_transform(fold: int, context: dict[str, Any], parts: dict[str, list[str]]) -> dict[str, Any]:
    expected = run.make_transform(context, parts, fold)
    path = OUTPUT / f"fold_{fold}" / "transform.json"
    if not path.is_file() or read_json(path) != expected:
        raise RuntimeError(f"train_only_transform_recomputation_mismatch:{fold}")
    if (expected["fit_ids_sha256"] != model_runner.stable_hash(parts["fit"])
            or expected["global"]["fit_ids_sha256"] != expected["fit_ids_sha256"]
            or expected["node"]["fit_ids_sha256"] != expected["fit_ids_sha256"]):
        raise RuntimeError(f"train_only_transform_partition_mismatch:{fold}")
    return expected


def audit_cell(fold: int, method: str, seed: int, target_rows: dict[str, dict[str, str]],
               index_rows: dict[str, dict[str, str]], outer_rows: dict[str, dict[str, str]],
               context: dict[str, Any], parts: dict[str, list[str]], transform: dict[str, Any]) -> dict[str, Any]:
    method_dir = OUTPUT / f"fold_{fold}" / method / f"seed_{seed}"
    manifest_path = method_dir / "cell_manifest.json"
    manifest = read_json(manifest_path)
    identity = manifest.get("identity", {})
    if identity.get("method") != method or int(identity.get("outer_fold", -1)) != fold or int(identity.get("seed", -1)) != seed:
        raise RuntimeError(f"cell_model_fold_seed_identity_mismatch:{method}:{fold}:{seed}")
    expected_identity = {
        "method": method, "internal_model_variant": run.METHODS[method], "outer_fold": fold, "seed": seed,
        "contract_sha256": sha(CONTRACT), "input_pins": context["pins"],
        "fit_ids_sha256": model_runner.stable_hash(parts["fit"]),
        "validation_ids_sha256": model_runner.stable_hash(parts["validation"]),
        "test_ids_sha256": model_runner.stable_hash(parts["test"]),
        "transform_sha256": model_runner.stable_hash(transform),
        "profile_sha256": sha(OUTPUT / "batch_profile.json"),
        "runner_sha256": sha(SCRIPT / "run_common_panel_mali.py"),
        "batch_helper_sha256": sha(SCRIPT / "run_mali_batched.py"),
        "model_runner_sha256": sha(SCRIPT / "run_mali_full_features.py"),
    }
    for key, value in expected_identity.items():
        if identity.get(key) != value:
            raise RuntimeError(f"cell_identity_mismatch:{method}:{fold}:{seed}:{key}")
    import torch
    if (identity.get("torch") != torch.__version__ or identity.get("cuda") != torch.version.cuda
            or identity.get("precision") != "float32"):
        raise RuntimeError(f"cell_environment_identity_mismatch:{method}:{fold}:{seed}")
    for key, ids in (("fit_ids_sha256", parts["fit"]), ("validation_ids_sha256", parts["validation"]), ("test_ids_sha256", parts["test"])):
        if identity.get(key) != model_runner.stable_hash(ids) or manifest.get(key) != identity.get(key):
            raise RuntimeError(f"cell_{key}_mismatch:{method}:{fold}:{seed}")
    if manifest.get("status") != "complete" or manifest.get("epochs") != 500 or not (0 <= int(manifest.get("best_epoch", -1)) < 500):
        raise RuntimeError(f"cell_training_terminal_state_invalid:{method}:{fold}:{seed}")
    if not math.isfinite(float(manifest["best_validation_mse_raw_seconds_squared"])):
        raise RuntimeError(f"cell_validation_mse_nonfinite:{method}:{fold}:{seed}")
    expected_output_names = {"checkpoint.pt", "predictions.csv", "test_failures.csv"}
    if set(manifest.get("output_hashes", {})) != expected_output_names:
        raise RuntimeError(f"cell_output_hash_set_mismatch:{method}:{fold}:{seed}")
    for name, expected in manifest["output_hashes"].items():
        if sha(method_dir / name) != expected:
            raise RuntimeError(f"cell_output_hash_mismatch:{method}:{fold}:{seed}:{name}")
    predictions = csv_rows(method_dir / "predictions.csv")
    failures = csv_rows(method_dir / "test_failures.csv")
    pred_ids = [row["canonical_observation_id"] for row in predictions]
    fail_ids = [row["canonical_observation_id"] for row in failures]
    if (len(pred_ids) != len(set(pred_ids)) or len(fail_ids) != len(set(fail_ids))
            or set(pred_ids) & set(fail_ids) or set(pred_ids) | set(fail_ids) != set(parts["test"])):
        raise RuntimeError(f"cell_prediction_terminal_identity_mismatch:{method}:{fold}:{seed}")
    for row in predictions:
        pred = float(row["predicted_seconds"])
        if row["status"] != "predicted" or not math.isfinite(pred):
            raise RuntimeError(f"nonfinite_or_bad_prediction:{method}:{fold}:{seed}")
        if int(row["outer_fold"]) != fold or row["method_id"] != method or int(row["seed"]) != seed:
            raise RuntimeError(f"prediction_metadata_mismatch:{method}:{fold}:{seed}")
        if "actual_seconds" in row or "target_seconds" in row:
            raise RuntimeError("outer_test_target_written_into_prediction_file")
        if row["canonical_observation_id"] not in target_rows or row["canonical_observation_id"] not in outer_rows:
            raise RuntimeError("prediction_identity_absent_from_evaluation_ledger")
    for row in failures:
        if row["status"] not in {"resource_failure", "unavailable"} or not row.get("terminal_reason"):
            raise RuntimeError(f"unknown_terminal_failure_state:{row['status']}")
    checkpoint = method_dir / "checkpoint.pt"
    if manifest["output_hashes"].get("checkpoint.pt") != sha(checkpoint):
        raise RuntimeError("terminal_checkpoint_hash_mismatch")
    configure_torch(torch)
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        saved = torch.load(checkpoint, map_location="cpu")
    if (saved.get("identity") != identity or saved.get("completed_epoch") != 500
            or saved.get("next_epoch") != 500
            or saved.get("best_model_state") is None
            or not saved.get("optimizer_state") or not saved.get("scheduler_state") or not saved.get("rng_state")
            or int(saved.get("best_epoch", -1)) != int(manifest["best_epoch"])
            or not math.isclose(float(saved.get("best_validation_mse", math.nan)),
                                float(manifest["best_validation_mse_raw_seconds_squared"]), rel_tol=0, abs_tol=1e-12)
            or saved.get("transform") != transform):
        raise RuntimeError(f"checkpoint_state_or_transform_mismatch:{method}:{fold}:{seed}")
    if (identity.get("torch") != torch.__version__ or identity.get("cuda") != torch.version.cuda
            or identity.get("precision") != "float32"):
        raise RuntimeError(f"cell_recorded_environment_mismatch:{method}:{fold}:{seed}")
    if set(saved["best_model_state"]) != set(saved["model_state"]):
        raise RuntimeError(f"checkpoint_best_state_schema_mismatch:{method}:{fold}:{seed}")
    if (int(manifest.get("n_test_assigned", -1)) != len(parts["test"])
            or int(manifest.get("n_test_predicted", -1)) != len(pred_ids)
            or int(manifest.get("n_test_failures", -1)) != len(fail_ids)):
        raise RuntimeError(f"cell_manifest_count_mismatch:{method}:{fold}:{seed}")
    for key, tensor in saved["best_model_state"].items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"checkpoint_best_state_nonfinite:{method}:{fold}:{seed}:{key}")
    return {"path": manifest_path.relative_to(ROOT).as_posix(), "sha256": sha(manifest_path),
            "status": "PASS", "method": method, "fold": fold, "seed": seed,
            "fit": len(parts["fit"]), "validation": len(parts["validation"]),
            "test": len(parts["test"]), "predicted": len(pred_ids), "failures": len(fail_ids)}


def audit_predictions_cuda(fold_cells: list[tuple[int, str, int]],
                           context: dict[str, Any], parts_by_fold: dict[int, dict[str, list[str]]],
                           transforms: dict[int, dict[str, Any]]) -> dict[str, Any]:
    import torch
    from gpu_lease import acquire_gpu_lease

    configure_torch(torch)
    lease = acquire_gpu_lease("common-panel-independent-prediction-validation")
    total = compared = 0
    try:
        model_runner.validate_gpu_budget(torch, budget_gib=12, reserve_gib=4)
        model_class, source_hash = model_runner.load_pinned_model()
        device_name = torch.cuda.get_device_name(0)
        profile = read_json(OUTPUT / "batch_profile.json")
        active_fold = None
        cache = None
        for fold, method, seed in fold_cells:
            if fold != active_fold:
                cache = batching.ExampleCache(context, transforms[fold], 8 * 1024**3)
                active_fold = fold
            internal = run.METHODS[method]
            graph = internal == model_runner.METHODS[0]
            folder = OUTPUT / f"fold_{fold}" / method / f"seed_{seed}"
            checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
            identity = checkpoint["identity"]
            if (identity.get("source_model_sha256") != source_hash
                    or identity.get("device") != device_name):
                raise RuntimeError(f"prediction_replay_source_model_mismatch:{method}:{fold}:{seed}")
            model = model_runner.make_model(model_class, internal,
                len(transforms[fold]["global"]["retained_indices"]), checkpoint_layers=True).cuda()
            if graph:
                model.mask = model.mask.to("cuda:0")
            model.load_state_dict(checkpoint["best_model_state"])
            model.eval()
            expected_rows = {row["canonical_observation_id"]: row
                             for row in csv_rows(folder / "predictions.csv")}
            failures = csv_rows(folder / "test_failures.csv")
            if len(expected_rows) + len(failures) != len(parts_by_fold[fold]["test"]):
                raise RuntimeError(f"prediction_replay_denominator_mismatch:{method}:{fold}:{seed}")
            node_limit = int(profile["graph_node_limit"]) if graph else 0
            test_ids = [identity_ for identity_ in parts_by_fold[fold]["test"] if identity_ in expected_rows]
            groups = batching.physical_groups(test_ids, context["ledger"], node_limit, graph)
            with torch.no_grad():
                for group in groups:
                    examples = [cache.get(identity_, internal) for identity_ in group]
                    data = model_runner._batch_to_device(examples, "cuda:0", graph)
                    values = model_runner._prediction(model, data).detach().reshape(-1).cpu().tolist()
                    for identity_, prediction in zip(group, values):
                        observed = float(expected_rows[identity_]["predicted_seconds"])
                        if not math.isclose(float(prediction), observed, rel_tol=1e-5, abs_tol=1e-6):
                            raise RuntimeError(f"prediction_replay_value_mismatch:{method}:{fold}:{seed}:{identity_}")
                    compared += len(group)
                    total += len(group)
                    del examples, data
            del model
            torch.cuda.empty_cache()
        if cache is not None:
            del cache
        return {"status": "PASS", "cells": len(fold_cells), "predictions_replayed": compared,
                "prediction_count_expected": total, "source_model_sha256": source_hash,
                "relative_tolerance": 1e-5, "absolute_tolerance": 1e-6}
    finally:
        lease.release()


def audit_fold0() -> dict[str, Any]:
    profile = audit_profile()
    profile_validation_path = OUTPUT / "profile_validation.json"
    if not profile_validation_path.is_file():
        raise RuntimeError("independent_cuda_profile_validation_missing")
    profile_validation = read_json(profile_validation_path)
    if (profile_validation.get("status") != "PASS"
            or profile_validation.get("contract_sha256") != sha(CONTRACT)
            or profile_validation.get("profile_sha256") != profile["profile_sha256"]
            or profile_validation.get("runner_sha256") != sha(SCRIPT / "run_common_panel_mali.py")):
        raise RuntimeError("independent_cuda_profile_validation_receipt_mismatch")
    context = run.load_context()
    parts = run.partitions(context, 0)
    transform = audit_transform(0, context, parts)
    targets = {row["canonical_observation_id"]: row for row in csv_rows(PANEL / "targets.csv")}
    indices = {row["canonical_observation_id"]: row for row in csv_rows(OUTPUT / "input_index.csv")}
    outer = {row["canonical_observation_id"]: row for row in csv_rows(PANEL / "outer_splits.csv")}
    cells = [audit_cell(0, method, seed, targets, indices, outer, context, parts, transform)
             for method in run.METHODS for seed in run.SEEDS]
    if len(cells) != 6 or any(row["status"] != "PASS" for row in cells):
        raise RuntimeError("independent_fold0_validation_failed")
    if any(row["failures"] or row["predicted"] != row["test"] for row in cells):
        raise RuntimeError("fold0_did_not_produce_finite_prediction_for_every_assigned_row")
    prediction_audit = audit_predictions_cuda([(0, method, seed) for method in run.METHODS for seed in run.SEEDS],
        context, {0: parts}, {0: transform})
    receipt = {"status": "PASS", "validator_sha256": sha(Path(__file__)),
        "contract_sha256": sha(CONTRACT), "profile_sha256": profile["profile_sha256"],
        "input_index_sha256": sha(OUTPUT / "input_index.csv"), "cells": cells,
        "profile_validation_sha256": sha(profile_validation_path), "prediction_replay": prediction_audit,
        "same_fit_and_validation_per_fold": True, "outer_test_labels_absent_from_prediction_outputs": True,
        "score_gate": False, "validated_at_unix": time.time()}
    model_runner.atomic_json(OUTPUT / "fold_0_gate_receipt.json", receipt)
    state = run.status_load()
    state.update({"stage": "fold0_gate_passed", "status": "PASS",
                  "fold0_gate_receipt_sha256": sha(OUTPUT / "fold_0_gate_receipt.json")})
    model_runner.atomic_json(PANEL / "run_status.json", state)
    return receipt


def audit_final() -> dict[str, Any]:
    audit_inputs()
    audit_profile()
    profile_validation_path = OUTPUT / "profile_validation.json"
    if not profile_validation_path.is_file():
        raise RuntimeError("independent_cuda_profile_validation_missing")
    profile_validation = read_json(profile_validation_path)
    if (profile_validation.get("status") != "PASS"
            or profile_validation.get("contract_sha256") != sha(CONTRACT)
            or profile_validation.get("profile_sha256") != sha(OUTPUT / "batch_profile.json")
            or profile_validation.get("runner_sha256") != sha(SCRIPT / "run_common_panel_mali.py")
            or profile_validation.get("validator_sha256") != sha(Path(__file__))):
        raise RuntimeError("independent_cuda_profile_validation_receipt_mismatch")
    fold0_gate_path = OUTPUT / "fold_0_gate_receipt.json"
    if not fold0_gate_path.is_file():
        raise RuntimeError("independent_fold0_gate_missing")
    fold0_gate = read_json(fold0_gate_path)
    if (fold0_gate.get("status") != "PASS" or fold0_gate.get("contract_sha256") != sha(CONTRACT)
            or fold0_gate.get("profile_sha256") != sha(OUTPUT / "batch_profile.json")
            or fold0_gate.get("validator_sha256") != sha(Path(__file__))):
        raise RuntimeError("independent_fold0_gate_receipt_mismatch")
    targets = {row["canonical_observation_id"]: row for row in csv_rows(PANEL / "targets.csv")}
    indices = {row["canonical_observation_id"]: row for row in csv_rows(OUTPUT / "input_index.csv")}
    outer = {row["canonical_observation_id"]: row for row in csv_rows(PANEL / "outer_splits.csv")}
    context = run.load_context()
    parts_by_fold = {fold: run.partitions(context, fold) for fold in range(5)}
    transforms = {fold: audit_transform(fold, context, parts_by_fold[fold]) for fold in range(5)}
    cells = [audit_cell(fold, method, seed, targets, indices, outer, context,
                        parts_by_fold[fold], transforms[fold])
             for fold in range(5) for method in run.METHODS for seed in run.SEEDS]
    if len(cells) != 30 or any(row["status"] != "PASS" for row in cells):
        raise RuntimeError("final_neural_cell_validation_failed")
    prediction_audit = audit_predictions_cuda([(fold, method, seed) for fold in range(5)
        for method in run.METHODS for seed in run.SEEDS], context, parts_by_fold, transforms)
    receipt = {"status": "PASS", "contract_sha256": sha(CONTRACT),
        "validator_sha256": sha(Path(__file__)), "cell_count": len(cells), "cells": cells,
        "prediction_replay": prediction_audit,
        "assigned_rows_per_method_fold": 7350, "validated_at_unix": time.time()}
    model_runner.atomic_json(OUTPUT / "final_validation.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("inputs", "profile", "fold0", "final"), required=True)
    args = parser.parse_args()
    if args.phase in {"profile", "fold0", "final"}:
        import torch
        configure_torch(torch)
    if args.phase == "inputs":
        result = audit_inputs()
    elif args.phase == "profile":
        audit_profile()
        result = independent_cuda_profile_gate()
    elif args.phase == "fold0":
        result = audit_fold0()
    else:
        result = audit_final()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
