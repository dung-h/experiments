#!/usr/bin/env python3
"""Nested group-subsampled A1 transfer extension.

This module imports the frozen A1 runner and changes only the fit-group
membership and output namespace.  It never changes the frozen panel, folds,
model definition, checkpoint, or 100%-QPU artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
import time
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OLD = REPO / "artifacts/postfreeze_track_a_transfer_v1"
OUT = REPO / "artifacts/postfreeze_track_a_transfer_sample_efficiency_v1"
RUNNER = OLD / "run_qpu_factorial.py"
FRACTIONS = (0.10, 0.25, 0.50)
SUBSET_SEEDS = (20261019, 20261020)
SEEDS = (42, 1234, 31415)
TREATMENTS = ("M-S", "M-P", "G-S", "G-P")


def load_runner():
    spec = importlib.util.spec_from_file_location("frozen_a1_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot_load_frozen_a1_runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.REPO = REPO
    module.QUALIFICATION = OLD / "checkpoint_qualification.json"
    module.STRICT_PRETRAIN = OLD / "pretraining/valid_pinned_simple_model"
    module.FACTORIAL = OUT / "qpu_factorial"
    install_graph_cache(module)
    return module


def install_graph_cache(base):
    """Cache deterministic graph parsing/normalization within this process.

    The frozen A1 runner is intentionally simple and reparses every graph each
    epoch.  Caching only the identical CPU `Data` objects changes no feature,
    split, optimizer, seed, or model operation; it removes repeated file I/O
    from this extension's implementation path.
    """
    import numpy as np
    import torch
    from torch_geometric.data import Data
    original_load = base.load_graph
    raw = OrderedDict()
    normalized = OrderedDict()
    lock = threading.RLock()
    used = {"raw": 0, "normalized": 0}
    budgets = {"raw": 2 * 1024**3, "normalized": 2 * 1024**3}
    stats = {"raw_misses": 0, "raw_hits": 0, "normalized_hits": 0,
             "normalized_misses": 0, "load_seconds": 0.0,
             "graph_item_seconds": 0.0, "prefetch_workers": 4}
    base._sample_graph_cache = normalized
    base._sample_preparation_stats = stats

    def insert(cache, key, value, kind, size):
        with lock:
            if key in cache:
                return cache[key][0]
            if size <= budgets[kind]:
                while cache and used[kind] + size > budgets[kind]:
                    _, (_, old_size) = cache.popitem(last=False)
                    used[kind] -= old_size
                cache[key] = (value, size)
                used[kind] += size
            return value

    def cached_load(row):
        key = (row["graph_file"], row["graph_sha256"])
        with lock:
            if key in raw:
                stats["raw_hits"] += 1
                raw.move_to_end(key)
                return raw[key][0]
        start = time.perf_counter()
        value = original_load(row)
        with lock:
            stats["raw_misses"] += 1
            stats["load_seconds"] += time.perf_counter() - start
        return insert(raw, key, value, "raw", sum(a.nbytes for a in value))

    def cached(row, gmean, gstd, nmean, nstd):
        start = time.perf_counter()
        # Node tensors depend on graph content and train-fitted node transforms,
        # not on observation ID. Repeats share tensors, never target values.
        key = (row["graph_file"], row["graph_sha256"], nmean.tobytes(), nstd.tobytes())
        with lock:
            entry = normalized.get(key)
            if entry is not None:
                normalized.move_to_end(key)
                stats["normalized_hits"] += 1
        if entry is None:
            x, e = cached_load(row)
            tensors = (torch.as_tensor((x-nmean)/nstd), torch.as_tensor(e, dtype=torch.long))
            size = sum(t.numel() * t.element_size() for t in tensors)
            tensors = insert(normalized, key, tensors, "normalized", size)
            stats["normalized_misses"] += 1
        else:
            tensors = entry[0]
        g = (np.asarray([float(row[f"g{j:02d}"]) for j in range(51)], dtype=np.float32)-gmean)/gstd
        result = Data(x=tensors[0], edge_index=tensors[1],
                      global_features=torch.as_tensor(g).reshape(1,51),
                      y=torch.tensor(float(row["target_seconds"]), dtype=torch.float32))
        stats["graph_item_seconds"] += time.perf_counter() - start
        return result

    def prefetch(rows):
        unique = {(r["graph_file"], r["graph_sha256"]): r for r in rows}
        start = time.perf_counter()
        # Threads only read/expand immutable inputs. Seeded model computation,
        # batch order, optimizer steps and train-only transforms stay serial.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for _ in pool.map(cached_load, unique.values()):
                pass
        stats["prefetch_seconds"] = time.perf_counter() - start
        stats["unique_prefetched_graphs"] = len(unique)

    base.load_graph = cached_load
    base.graph_item = cached
    base._sample_prefetch_graphs = prefetch
    from torch_geometric.data import Batch
    original_batch = Batch.from_data_list.__func__
    stats["collation_seconds"] = 0.0
    stats["collation_calls"] = 0

    def timed_batch(cls, data_list, *args, **kwargs):
        start = time.perf_counter()
        result = original_batch(cls, data_list, *args, **kwargs)
        stats["collation_seconds"] += time.perf_counter() - start
        stats["collation_calls"] += 1
        return result

    Batch.from_data_list = classmethod(timed_batch)
    # Batch cache stays disabled; random training packs must not accumulate.
    base._sample_cpu_batch_cache = {}



def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha_lines(values):
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def subset_plan(base, fraction: float, subset_seed: int, fold: int):
    panel = read_csv(base.PANEL)
    panel_by = {r["canonical_observation_id"]: r for r in panel}
    outer = read_csv(base.OUTER)
    inner = read_csv(base.INNER)
    full = base.partitions(fold, panel_by, outer, inner)
    fit_groups = sorted({panel_by[i]["group_id"] for i in full["fit"]})
    rng = random.Random(subset_seed * 1009 + fold)
    shuffled = list(fit_groups)
    rng.shuffle(shuffled)
    if fraction >= 0.999999:
        selected = set(fit_groups)
    else:
        selected = set(shuffled[: max(1, math.ceil(fraction * len(fit_groups)))])
    fit = sorted(i for i in full["fit"] if panel_by[i]["group_id"] in selected)
    return {"fit": fit, "validation": full["validation"], "test": full["test"]}, {
        "outer_fold": fold,
        "fraction": fraction,
        "subset_seed": subset_seed,
        "full_fit_group_count": len(fit_groups),
        "selected_fit_group_count": len(selected),
        "fit_observation_count": len(fit),
        "selected_group_sha256": sha_lines(sorted(selected)),
        "fit_id_sha256": sha_lines(fit),
        "validation_id_sha256": sha_lines(full["validation"]),
        "test_id_sha256": sha_lines(full["test"]),
    }


def run_cell(base, treatment, fold, seed, fraction, subset_seed):
    base.FACTORIAL = OUT / "qpu_factorial" / f"fraction_{int(round(fraction * 100)):03d}" / f"subset_{subset_seed}"
    original_partitions = base.partitions
    part, receipt = subset_plan(base, fraction, subset_seed, fold)
    cell = base.FACTORIAL / treatment / f"fold_{fold}" / f"seed_{seed}"
    if treatment.startswith("G") and not (cell / "terminal_receipt.json").exists():
        base._sample_prefetch_graphs(read_csv(base.PANEL))

    def patched_partitions(_fold, panel_by, outer, inner):
        if _fold != fold:
            raise RuntimeError("unexpected_fold_in_subset_run")
        return part

    base.partitions = patched_partitions
    try:
        result = base.fit_cell(treatment, fold, seed)
    finally:
        base.partitions = original_partitions
    cell = base.FACTORIAL / treatment / f"fold_{fold}" / f"seed_{seed}"
    for name in ("identity_manifest.json", "terminal_receipt.json"):
        path = cell / name
        if path.exists():
            payload = json.loads(path.read_text())
            payload["study_id"] = "postfreeze_track_a_transfer_sample_efficiency_v1"
            payload["parent_study_id"] = "postfreeze_track_a_transfer_v1"
            payload["fraction"] = fraction
            payload["subset_seed"] = subset_seed
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    cell_receipt = dict(result)
    cell_receipt.update({"fraction": fraction, "subset_seed": subset_seed, "subset": receipt})
    (cell / "subset_receipt.json").write_text(json.dumps(cell_receipt, indent=2, sort_keys=True) + "\n")
    profile = dict(base._sample_preparation_stats)
    profile["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    profile["cache_budgets_gib"] = {"raw": 2, "normalized": 2}
    (cell / "preparation_profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    base._sample_cpu_batch_cache.clear()
    return cell_receipt


def copy_manifest():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "study_id": "postfreeze_track_a_transfer_sample_efficiency_v1",
        "frozen_parent": str(OLD.relative_to(REPO)),
        "fractions": list(FRACTIONS) + [1.0],
        "subset_seeds": list(SUBSET_SEEDS),
        "model_seeds": list(SEEDS),
        "treatments": list(TREATMENTS),
        "nested_group_sampling": True,
        "selection_target": "outer-training groups only; no labels used",
        "parent_runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        "status": "RUNNING",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # The 100% predictions are reused, never retrained here.
    reuse = {
        "fraction": 1.0,
        "source": str((OLD / "per_seed_predictions.csv").relative_to(REPO)),
        "status": "REUSED_FROZEN_A1",
        "sha256": hashlib.sha256((OLD / "per_seed_predictions.csv").read_bytes()).hexdigest(),
    }
    (OUT / "reuse_100_receipt.json").write_text(json.dumps(reuse, indent=2, sort_keys=True) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fraction", type=float, choices=FRACTIONS)
    p.add_argument("--subset-seed", type=int, choices=SUBSET_SEEDS)
    p.add_argument("--fold", type=int, choices=range(5))
    p.add_argument("--treatment", choices=TREATMENTS)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--only-subset-seed", type=int, choices=SUBSET_SEEDS)
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    base = load_runner()
    copy_manifest()
    if not a.all and None in (a.fraction, a.subset_seed, a.fold, a.treatment, a.seed):
        p.error("provide a complete cell or --all")
    if a.all:
        for fraction in FRACTIONS:
            subset_seeds = (a.only_subset_seed,) if a.only_subset_seed is not None else SUBSET_SEEDS
            for subset_seed in subset_seeds:
                for fold in range(5):
                    for treatment in TREATMENTS:
                        for seed in SEEDS:
                            print(json.dumps(run_cell(base, treatment, fold, seed, fraction, subset_seed), sort_keys=True), flush=True)
    else:
        print(json.dumps(run_cell(base, a.treatment, a.fold, a.seed, a.fraction, a.subset_seed), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
