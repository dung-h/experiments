#!/usr/bin/env python3
"""Preflight or fit the frozen graph-free global-metadata QPU baseline.

Preflight is CPU-only and read-only. Fit requires CUDA, runs one outer fold,
and writes only to a new caller-selected output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
DEFAULT_OUTER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv"
DEFAULT_INNER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_inner_split_v2.csv"
DEFAULT_GRAPH_OOF = ROOT / "artifacts/benchmark_v3/real_qpu/unified_graph_v3_large_oof_20260930/predictions.csv"
SEED_REGISTRY = ROOT / "benchmark_v1/registry/seed_registry.json"
PROTOCOL = ROOT / "benchmark_v1/protocol/qpu_global_metadata_mlp_baseline_v1.json"

METHOD_ID = "qpu_global_metadata_mlp_baseline_v1"
FEATURE_FIELDS = (
    "active_width", "structural_depth", "one_qubit_count", "two_qubit_count",
    "swap_like_count", "measurement_count", "shots",
)
SEEDS = (42, 1234, 31415)
UNAVAILABLE_GRAPH_ID = "qonductor_single_circuit_ibm|row4477"
EXPECTED_SOURCE_COUNTS = {
    "mali_real_qpu": 340,
    "qonductor_single_circuit_ibm": 4482,
    "qpack_mcp": 3945,
}
EXPECTED_TEST_COUNTS = {0: 1778, 1: 2206, 2: 1547, 3: 1467, 4: 1769}
EXPECTED_MATCHED_TRAIN_COUNTS = {0: 6989, 1: 6560, 2: 7219, 3: 7299, 4: 6997}
EXPECTED_HASHES = {
    "canonical": "920d745e03dd00e8155118d9323baea18ff865e4df2620f3a059bd5503a05ba4",
    "outer": "d72b7ba6f90d5bf6ed2f3b602196a05bee11a0f441ee4e5caaca6978b6b0e25a",
    "inner": "aaff2b0950379ac7ad8992698bc140687a4cd1b6696b8abeff7cb96270af3cab",
    "seed_registry": "ae48f16dac27cdf3c9f42ce89acf4a67cb1b318f9c23f32e88d2c61d4323e864",
    "graph_oof": "3c179ae464fc25167ae0f1aca028cf5ff1117409dce9d15f6945e4f621420d2c",
    "features": "cb233f00258cd8189da1a3b17363386ff19618e84778956ed8de9f352eb6117a",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {path}")
        rows = list(reader)
    return rows


def unique_index(rows: Iterable[dict[str, str]], key: str, label: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if not value or value in out:
            raise ValueError(f"{label}: empty or duplicate {key}: {value!r}")
        out[value] = row
    return out


def global_log_features(row: dict[str, str]) -> tuple[float, ...]:
    """Read only the seven S71 global scalars; no targets or context fields."""
    values: list[float] = []
    for field in FEATURE_FIELDS:
        raw = row.get(field, "")
        if raw == "":
            raise ValueError(f"missing allowed global feature {field}")
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid allowed global feature {field}={raw!r}")
        values.append(math.log1p(value))
    return tuple(values)


def validate_fold_assignments(
    canonical: dict[str, dict[str, str]],
    outer: dict[str, dict[str, str]],
    inner_rows: list[dict[str, str]],
    exclude_from_all_training: set[str] | None = None,
) -> dict[int, dict[str, object]]:
    if set(canonical) != set(outer):
        raise ValueError("canonical and frozen outer split ID sets differ")
    inner_by_fold: dict[int, dict[str, dict[str, str]]] = {fold: {} for fold in range(5)}
    for row in inner_rows:
        cid = row.get("canonical_observation_id", "")
        if cid not in canonical:
            raise ValueError(f"inner split contains unknown ID {cid!r}")
        fold = int(row["outer_fold"])
        if fold not in inner_by_fold or cid in inner_by_fold[fold]:
            raise ValueError(f"invalid or duplicate inner assignment for {cid}, outer={fold}")
        inner_by_fold[fold][cid] = row

    excluded = exclude_from_all_training or set()
    if not excluded.issubset(canonical):
        raise ValueError("training-support exclusions contain unknown IDs")
    output: dict[int, dict[str, object]] = {}
    for fold in range(5):
        expected_ids = {cid for cid, row in outer.items() if int(row["outer_fold"]) == fold}
        fold_inner = inner_by_fold[fold]
        if set(fold_inner) != set(canonical):
            raise ValueError(f"outer fold {fold}: inner rows do not cover canonical IDs exactly")
        train_ids = (set(canonical) - expected_ids) - excluded
        test_ids = expected_ids
        train_groups = {outer[cid]["unified_leakage_group_id"] for cid in train_ids}
        test_groups = {outer[cid]["unified_leakage_group_id"] for cid in test_ids}
        overlap = train_groups & test_groups
        if overlap:
            raise ValueError(f"outer fold {fold}: leakage groups cross train/test: {len(overlap)}")
        seen_inner: dict[str, set[str]] = {}
        for cid, row in fold_inner.items():
            role = "test" if cid in test_ids else "train"
            value = row.get("inner_fold", "")
            if role == "test" and value != "":
                raise ValueError(f"outer fold {fold}: test ID has inner fold: {cid}")
            if role == "train":
                if value not in {"0", "1", "2", "3"}:
                    raise ValueError(f"outer fold {fold}: train ID lacks frozen inner fold: {cid}")
                group = outer[cid]["unified_leakage_group_id"]
                seen_inner.setdefault(group, set()).add(value)
        if any(len(folds) != 1 for folds in seen_inner.values()):
            raise ValueError(f"outer fold {fold}: an inner leakage group is split across folds")
        inner_counts = Counter(fold_inner[cid]["inner_fold"] for cid in train_ids)
        if any(inner_counts.get(str(inner), 0) <= 0 for inner in range(4)):
            raise ValueError(f"outer fold {fold}: empty inner validation partition")
        output[fold] = {
            "train_ids": sorted(train_ids),
            "test_ids": sorted(test_ids),
            "train_group_count": len(train_groups),
            "test_group_count": len(test_groups),
            "group_overlap_count": 0,
            "inner_validation_rows": {str(k): inner_counts[str(k)] for k in range(4)},
        }
    return output


def validate_feature_coverage(
    canonical: dict[str, dict[str, str]], features: dict[str, dict[str, str]],
) -> None:
    if set(canonical) != set(features):
        raise ValueError("feature sidecar and canonical ID sets differ")
    for cid, row in features.items():
        if row.get("availability_status") != "available":
            raise ValueError(f"feature row not available: {cid}: {row.get('terminal_reason')}")
        if row.get("circuit_count") != "1":
            raise ValueError(f"not a one-circuit observation: {cid}")
        # Strict allow-list access: source/backend/lifecycle/target metadata is
        # never serialized as model input, even when present in this sidecar.
        global_log_features(row)


def validate_graph_oof(path: Path, canonical: dict[str, dict[str, str]]) -> dict[str, object]:
    graph_rows = read_csv(path)
    graph = unique_index(graph_rows, "canonical_observation_id", "Graph V3-large OOF")
    expected = set(canonical) - {UNAVAILABLE_GRAPH_ID}
    if set(graph) != expected:
        raise ValueError("Graph V3-large OOF IDs differ from 8,767 minus row4477")
    return {"graph_oof_rows": len(graph), "graph_unavailable_id": UNAVAILABLE_GRAPH_ID,
            "paired_rows_available": len(graph)}


def hash_inputs(paths: dict[str, Path], verify: bool) -> dict[str, dict[str, str]]:
    results = {}
    for name, path in paths.items():
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"required input not found: {path}")
        actual = sha256(path)
        expected = EXPECTED_HASHES.get(name)
        if verify and expected and actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
        results[name] = {"path": str(path), "sha256": actual}
    return results


def load_and_preflight(args) -> dict[str, object]:
    paths = {
        "canonical": args.canonical,
        "outer": args.outer_split,
        "inner": args.inner_split,
        "seed_registry": SEED_REGISTRY,
        "graph_oof": args.graph_predictions,
        "features": args.features,
    }
    hashes = hash_inputs(paths, verify=args.verify_authority_hashes)
    canonical_rows = read_csv(args.canonical)
    outer_rows = read_csv(args.outer_split)
    inner_rows = read_csv(args.inner_split)
    feature_rows = read_csv(args.features)
    canonical = unique_index(canonical_rows, "canonical_row_id", "canonical corpus")
    outer = unique_index(outer_rows, "canonical_observation_id", "outer split")
    features = unique_index(feature_rows, "canonical_row_id", "C134 feature sidecar")
    if len(canonical) != 8767:
        raise ValueError(f"canonical row count is {len(canonical)}, expected 8767")
    if Counter(row["source_id"] for row in canonical_rows) != EXPECTED_SOURCE_COUNTS:
        raise ValueError("canonical per-source counts differ from frozen protocol")
    if Counter(row["source_id"] for row in outer_rows) != EXPECTED_SOURCE_COUNTS:
        raise ValueError("outer split per-source counts differ from frozen protocol")
    for cid, row in canonical.items():
        if row["source_id"] != outer[cid]["source_id"]:
            raise ValueError(f"source_id changed between canonical and outer assignment for {cid}")
        target = float(row["target_seconds"])
        if not math.isfinite(target) or target < 0:
            raise ValueError(f"invalid target label for {cid}")
    validate_feature_coverage(canonical, features)
    folds = validate_fold_assignments(canonical, outer, inner_rows,
                                      exclude_from_all_training={UNAVAILABLE_GRAPH_ID})
    for fold, summary in folds.items():
        if len(summary["test_ids"]) != EXPECTED_TEST_COUNTS[fold]:
            raise ValueError(f"fold {fold}: unexpected test rows {len(summary['test_ids'])}")
        if len(summary["train_ids"]) != EXPECTED_MATCHED_TRAIN_COUNTS[fold]:
            raise ValueError(f"fold {fold}: train support differs from Graph V3-large eligible rows")
    if int(outer[UNAVAILABLE_GRAPH_ID]["outer_fold"]) != 0:
        raise ValueError("known graph-unavailable row4477 changed fold")
    graph = validate_graph_oof(args.graph_predictions, canonical)
    registry = json.loads(SEED_REGISTRY.read_text(encoding="utf-8"))
    preserved = set(int(x) for x in registry.get("historical_seeds_preserved", []))
    if not set(SEEDS).issubset(preserved):
        raise ValueError("the frozen graph seeds are absent from the seed registry")
    return {
        "status": "PASS",
        "mode": "metadata_preflight_only_no_fit",
        "method_id": METHOD_ID,
        "canonical_rows": len(canonical),
        "feature_rows": len(features),
        "feature_columns_used_in_order": list(FEATURE_FIELDS),
        "feature_rows_available": len(features),
        "circuit_count_asserted_one": True,
        "feature_firewall": "passed; only seven scalar columns passed to the model transform",
        "source_counts": dict(Counter(row["source_id"] for row in canonical_rows)),
        "outer_fold_summary": {
            str(fold): {"train_rows": len(info["train_ids"]), "test_rows": len(info["test_ids"]),
                        "train_groups": info["train_group_count"], "test_groups": info["test_group_count"],
                        "group_overlap": info["group_overlap_count"],
                        "inner_validation_rows": info["inner_validation_rows"]}
            for fold, info in folds.items()
        },
        "graph_comparison": graph,
        "training_support": "row4477 excluded from all outer-train sets to exactly match Graph V3-large",
        "supplementary_prediction": "fold-0 row4477 may be predicted from its held-out seven global features; label is not used to fit this fold and its score is excluded from the primary graph-paired table",
        "seeds": list(SEEDS),
        "hyperparameter_selection": "none; fixed schedule inherited from graph V3-large contract; inner folds audited only",
        "input_hashes": hashes,
    }


def make_model_class(torch):
    class GlobalMetadataMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gf1 = torch.nn.Linear(7, 64)
            self.gf2 = torch.nn.Linear(64, 64)
            self.head1 = torch.nn.Linear(128, 512)
            self.head2 = torch.nn.Linear(512, 512)
            self.head3 = torch.nn.Linear(512, 128)
            self.head4 = torch.nn.Linear(128, 1)

        def forward(self, global_features):
            gf = torch.relu(self.gf1(global_features))
            gf = torch.relu(self.gf2(gf))
            graph_placeholder = torch.zeros((gf.shape[0], 64), device=gf.device, dtype=gf.dtype)
            value = torch.cat([graph_placeholder, gf], dim=1)
            value = torch.relu(self.head1(value))
            value = torch.relu(self.head2(value))
            value = torch.relu(self.head3(value))
            return self.head4(value).reshape(-1)

    return GlobalMetadataMLP


def cuda_device_receipt() -> dict[str, str | int]:
    """Capture the physical GPU and driver visible to the fit process."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("nvidia-smi returned no visible GPU inventory")
    fields = [part.strip() for part in lines[0].split(",")]
    if len(fields) != 3 or not fields[0] or not fields[1]:
        raise RuntimeError(f"could not parse nvidia-smi inventory: {lines[0]!r}")
    return {"gpu_name": fields[0], "gpu_driver_version": fields[1],
            "gpu_total_memory_mib": int(fields[2])}


CHECKPOINT_INTERVAL_EPOCHS = 50


def write_run_manifest(out: Path, manifest: dict[str, object]) -> None:
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def execute_fit_with_lock(args, preflight: dict[str, object], lock_factory, trainer) -> dict[str, object]:
    """Create an interruption-visible run envelope, then acquire S85 lock."""
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory {out}")
    out.mkdir(parents=True)
    manifest: dict[str, object] = {
        "artifact_id": f"{METHOD_ID}-fold-{args.fold}",
        "status": "running", "stage": "waiting_for_compute_lock",
        "fit_completed": False, "fit_started": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method_id": METHOD_ID, "outer_fold": args.fold,
        "command": sys.argv, "inputs": preflight["input_hashes"],
        "compute_lock": {"shared_timing_and_host_gpu_lease_held": False},
        "checkpoint_policy": {
            "interval_epochs": CHECKPOINT_INTERVAL_EPOCHS,
            "state": ["model", "optimizer", "python_rng", "numpy_rng", "torch_rng", "cuda_rng", "completed_epoch"],
            "automatic_resume_supported": False,
        },
    }
    write_run_manifest(out, manifest)
    try:
        with lock_factory():
            manifest.update({"stage": "compute_lock_acquired", "fit_started": True,
                             "compute_lock": {"shared_timing_and_host_gpu_lease_held": True}})
            write_run_manifest(out, manifest)
            return trainer(args, preflight, out)
    except Exception as exc:
        current = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        current.update({"status": "failed", "stage": "exception",
                        "error_type": type(exc).__name__, "error": str(exc),
                        "fit_completed": False})
        write_run_manifest(out, current)
        raise


def fit_fold(args, preflight: dict[str, object]) -> dict[str, object]:
    if args.output_dir.resolve().exists():
        raise FileExistsError(f"refusing to overwrite existing output directory {args.output_dir.resolve()}")
    def lock_factory():
        try:
            from benchmark_v1.scripts.run_mps_fixed_chi16_runtime_adaptation_v1 import exclusive_compute_lock
        except ImportError as exc:
            raise RuntimeError("fit requires the S85 shared timing + host-GPU compute lock") from exc
        return exclusive_compute_lock()
    return execute_fit_with_lock(args, preflight, lock_factory, _fit_fold_locked)


def _fit_fold_locked(args, preflight: dict[str, object], out: Path) -> dict[str, object]:
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("fit requires numpy and a CUDA-enabled PyTorch install") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for neural fitting; refusing CPU fallback")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    device = torch.device("cuda:0")
    gpu_receipt = cuda_device_receipt()

    canonical_rows = read_csv(args.canonical)
    outer_rows = read_csv(args.outer_split)
    feature_rows = read_csv(args.features)
    canonical = unique_index(canonical_rows, "canonical_row_id", "canonical corpus")
    outer = unique_index(outer_rows, "canonical_observation_id", "outer split")
    features = unique_index(feature_rows, "canonical_row_id", "C134 feature sidecar")
    fold_rows = validate_fold_assignments(canonical, outer, read_csv(args.inner_split),
                                          exclude_from_all_training={UNAVAILABLE_GRAPH_ID})[args.fold]
    train_ids = fold_rows["train_ids"]
    test_ids = fold_rows["test_ids"]
    x_raw_train = np.asarray([global_log_features(features[cid]) for cid in train_ids], dtype=np.float32)
    x_raw_test = np.asarray([global_log_features(features[cid]) for cid in test_ids], dtype=np.float32)
    mean = x_raw_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_raw_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    x_train = (x_raw_train - mean) / std
    x_test = (x_raw_test - mean) / std
    y_train = np.asarray([math.log1p(float(canonical[cid]["target_seconds"])) for cid in train_ids], dtype=np.float32)
    Model = make_model_class(torch)
    # CUDA smoke test is mandatory and recorded before model fitting.
    smoke = Model().to(device)
    smoke_x = torch.zeros((2, 7), dtype=torch.float32, device=device)
    smoke_y = smoke(smoke_x).sum()
    smoke_y.backward()
    if not math.isfinite(float(smoke_y.detach().cpu())) or any(
        p.grad is None or not torch.isfinite(p.grad).all() for p in smoke.parameters()
    ):
        raise RuntimeError("CUDA forward/backward smoke failed")
    del smoke, smoke_x, smoke_y
    torch.cuda.synchronize()

    tx_train = torch.tensor(x_train, dtype=torch.float32)
    ty_train = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tx_train, ty_train), batch_size=32, shuffle=True)
    predictions_by_seed: list[list[float]] = []
    seed_summaries = []
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir()
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = Model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(500):
            batch_losses = []
            for bx, by in loader:
                bx = bx.to(device, non_blocking=False)
                by = by.to(device, non_blocking=False)
                optimizer.zero_grad(set_to_none=True)
                pred = model(bx)
                loss = torch.nn.functional.mse_loss(pred, by)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            if not batch_losses or not math.isfinite(sum(batch_losses) / len(batch_losses)):
                raise RuntimeError(f"non-finite or empty loss in seed={seed}, epoch={epoch}")
            completed_epoch = epoch + 1
            if completed_epoch % CHECKPOINT_INTERVAL_EPOCHS == 0 or completed_epoch == 500:
                checkpoint_path = checkpoint_dir / f"seed_{seed}_epoch_{completed_epoch:04d}.pt"
                torch.save({
                    "artifact_id": f"{METHOD_ID}-fold-{args.fold}-seed-{seed}",
                    "seed": seed, "completed_epoch": completed_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                    "input_hashes": preflight["input_hashes"],
                    "protocol_sha256": sha256(PROTOCOL),
                }, checkpoint_path)
                progress_manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
                progress_manifest.update({
                    "status": "running", "stage": "training",
                    "last_checkpoint": str(checkpoint_path.relative_to(out)),
                    "last_checkpoint_seed": seed,
                    "last_checkpoint_completed_epoch": completed_epoch,
                    "automatic_resume_supported": False,
                })
                write_run_manifest(out, progress_manifest)
        model.eval()
        with torch.no_grad():
            pred_log = model(torch.tensor(x_test, dtype=torch.float32, device=device))
            pred_seconds = torch.expm1(pred_log).clamp_min(0).detach().cpu().numpy().astype(float).tolist()
        if len(pred_seconds) != len(test_ids) or not all(math.isfinite(x) for x in pred_seconds):
            raise RuntimeError(f"invalid seed={seed} OOF predictions")
        predictions_by_seed.append(pred_seconds)
        seed_summaries.append({
            "seed": seed,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        })
        del model, optimizer
        torch.cuda.empty_cache()
    pred_median = np.median(np.asarray(predictions_by_seed, dtype=np.float64), axis=0)
    fields = ["canonical_observation_id", "source_id", "unified_leakage_group_id", "outer_fold",
              "comparison_scope", "actual_seconds", "predicted_seconds", "absolute_error_seconds"] + [f"predicted_seed_{s}_seconds" for s in SEEDS]
    pred_path = out / "predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, cid in enumerate(test_ids):
            prediction = float(pred_median[i])
            # Keep the archived target at source precision. This value is for
            # the result ledger only; float32 remains the model's training
            # precision. Casting the target through float32 here can alter
            # archived decimal labels by a few microseconds and fail exact
            # lineage validation.
            actual = float(canonical[cid]["target_seconds"])
            row = {"canonical_observation_id": cid, "source_id": outer[cid]["source_id"],
                   "unified_leakage_group_id": outer[cid]["unified_leakage_group_id"],
                   "outer_fold": args.fold,
                   "comparison_scope": "metadata_only_supplementary" if cid == UNAVAILABLE_GRAPH_ID else "primary_graph_shared_panel",
                   "actual_seconds": actual,
                   "predicted_seconds": prediction,
                   "absolute_error_seconds": abs(prediction - actual)}
            row.update({f"predicted_seed_{seed}_seconds": predictions_by_seed[j][i]
                        for j, seed in enumerate(SEEDS)})
            writer.writerow(row)
    attempt_path = out / "attempts.csv"
    with attempt_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["canonical_observation_id", "source_id", "unified_leakage_group_id", "outer_fold", "status", "reason"])
        writer.writeheader()
        writer.writerows({"canonical_observation_id": cid, "source_id": outer[cid]["source_id"],
                          "unified_leakage_group_id": outer[cid]["unified_leakage_group_id"],
                          "outer_fold": args.fold,
                          "status": "predicted_supplementary" if cid == UNAVAILABLE_GRAPH_ID else "predicted",
                          "reason": "graph_v3_large_unavailable; metadata-only prediction excluded from paired primary score" if cid == UNAVAILABLE_GRAPH_ID else ""}
                         for cid in test_ids)
    device_name = torch.cuda.get_device_name(device)
    summary = {
        "artifact_id": f"{METHOD_ID}-fold-{args.fold}", "status": "PASS", "fit_ran": True,
        "outer_fold": args.fold, "method_id": METHOD_ID, "n_train": len(train_ids),
        "n_test_assigned": len(test_ids),
        "n_test_primary_paired": sum(cid != UNAVAILABLE_GRAPH_ID for cid in test_ids),
        "n_test_supplementary": sum(cid == UNAVAILABLE_GRAPH_ID for cid in test_ids),
        "train_group_count": fold_rows["train_group_count"], "test_group_count": fold_rows["test_group_count"],
        "group_overlap_count": fold_rows["group_overlap_count"], "features": list(FEATURE_FIELDS),
        "all_finite_sidecar_rows_used_for_train_or_test": True,
        "row4477_role": "excluded from train in every fold; fold-0 output is supplementary, not part of primary paired score",
        "selected_seed": None, "seed_reduction": "rowwise_median_of_all_three",
        "device": device_name, **gpu_receipt,
        "cuda_version": torch.version.cuda, "torch_version": torch.__version__,
        "precision": "float32", "cuda_forward_backward_smoke": "PASS",
        "cpu_threads": torch.get_num_threads(), "seeds": seed_summaries,
        "checkpoint_interval_epochs": CHECKPOINT_INTERVAL_EPOCHS,
        "automatic_resume_supported": False,
        "target_transform": "log1p_seconds_fit_and_expm1_clip_nonnegative_for_report",
        "normalization_mean_log_features": mean.tolist(), "normalization_std_log_features": std.tolist(),
        "inner_folds_audited_not_used_for_selection": True,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint_hashes = {
        str(path.relative_to(out)): sha256(path)
        for path in sorted(checkpoint_dir.glob("seed_*_epoch_*.pt"))
    }
    manifest = {
        "artifact_id": summary["artifact_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS", "method_id": METHOD_ID, "command": sys.argv,
        "python": sys.version, "platform": platform.platform(), "inputs": preflight["input_hashes"],
        "source_hashes": {"runner": sha256(Path(__file__).resolve()),
                          "protocol": sha256(PROTOCOL)},
        "outputs": {"predictions.csv": sha256(pred_path), "attempts.csv": sha256(attempt_path),
                    "summary.json": sha256(summary_path), **checkpoint_hashes},
        "contract": "benchmark_v1/protocol/qpu_global_metadata_mlp_baseline_v1.json",
        "outer_fold": args.fold, "device": device_name, **gpu_receipt,
        "cuda": torch.version.cuda,
        "torch": torch.__version__, "cuda_smoke": "PASS", "fit_completed": True,
        "status": "PASS", "stage": "completed",
        "compute_lock": {"shared_timing_and_host_gpu_lease_held_for_cuda_operations": True},
        "checkpoint_policy": {"interval_epochs": CHECKPOINT_INTERVAL_EPOCHS,
                              "automatic_resume_supported": False},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def validate_fit_output(args) -> dict[str, object]:
    """CPU-only technical gate for one fold's completed attempt envelope."""
    out = args.output_dir.resolve()
    manifest_path, summary_path = out / "run_manifest.json", out / "summary.json"
    prediction_path, attempts_path = out / "predictions.csv", out / "attempts.csv"
    required = (manifest_path, summary_path, prediction_path, attempts_path)
    if any(not path.is_file() for path in required):
        raise ValueError("fold output is incomplete: manifest/summary/predictions/attempts required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or not manifest.get("fit_completed"):
        raise ValueError("fit manifest is not successfully completed")
    if manifest.get("cuda_smoke") != "PASS" or not manifest.get("compute_lock", {}).get(
        "shared_timing_and_host_gpu_lease_held_for_cuda_operations"
    ):
        raise ValueError("CUDA smoke or S85 compute-lock receipt missing")
    for field in ("device", "gpu_driver_version", "gpu_total_memory_mib", "cuda", "torch"):
        if not manifest.get(field):
            raise ValueError(f"fit manifest lacks hardware/runtime field {field}")
    for path in (summary_path, prediction_path, attempts_path):
        relative = path.name
        if manifest.get("outputs", {}).get(relative) != sha256(path):
            raise ValueError(f"output hash missing/mismatched: {relative}")
    predictions = unique_index(read_csv(prediction_path), "canonical_observation_id", "fold predictions")
    attempts = unique_index(read_csv(attempts_path), "canonical_observation_id", "fold attempts")
    canonical = unique_index(read_csv(args.canonical), "canonical_row_id", "canonical corpus")
    outer = unique_index(read_csv(args.outer_split), "canonical_observation_id", "outer split")
    expected = {cid for cid, row in outer.items() if int(row["outer_fold"]) == args.fold}
    if set(predictions) != expected or set(attempts) != expected:
        raise ValueError("fold attempt/prediction IDs do not equal frozen assigned test IDs")
    paired = expected - {UNAVAILABLE_GRAPH_ID}
    if args.fold != 0 and UNAVAILABLE_GRAPH_ID in expected:
        raise ValueError("row4477 unexpectedly moved from frozen fold 0")
    for cid, row in predictions.items():
        if row["source_id"] != outer[cid]["source_id"] or row["unified_leakage_group_id"] != outer[cid]["unified_leakage_group_id"]:
            raise ValueError(f"fold output identity metadata mismatch: {cid}")
        actual, predicted = float(row["actual_seconds"]), float(row["predicted_seconds"])
        if not math.isfinite(actual) or not math.isfinite(predicted) or actual != float(canonical[cid]["target_seconds"]):
            raise ValueError(f"non-finite or label-mismatched prediction: {cid}")
        scope = "metadata_only_supplementary" if cid == UNAVAILABLE_GRAPH_ID else "primary_graph_shared_panel"
        if row["comparison_scope"] != scope:
            raise ValueError(f"incorrect comparison scope for {cid}")
        for seed in SEEDS:
            if not math.isfinite(float(row[f"predicted_seed_{seed}_seconds"])):
                raise ValueError(f"non-finite seed prediction for {cid}, seed={seed}")
        seed_values = [float(row[f"predicted_seed_{seed}_seconds"]) for seed in SEEDS]
        expected_median = sorted(seed_values)[1]
        if not math.isclose(predicted, expected_median, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"rowwise three-seed median mismatch: {cid}")
        if not math.isclose(float(row["absolute_error_seconds"]), abs(predicted - actual), rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"absolute error mismatch: {cid}")
    for cid, row in attempts.items():
        expected_status = "predicted_supplementary" if cid == UNAVAILABLE_GRAPH_ID else "predicted"
        if row["status"] != expected_status:
            raise ValueError(f"attempt status/supplementary labeling mismatch: {cid}")
    checkpoints = sorted((out / "checkpoints").glob("seed_*_epoch_*.pt"))
    expected_names = {f"seed_{seed}_epoch_{epoch:04d}.pt" for seed in SEEDS for epoch in range(50, 501, 50)}
    if {path.name for path in checkpoints} != expected_names:
        raise ValueError("checkpoint set is incomplete (expected all three seeds at epochs 50..500)")
    for path in checkpoints:
        relative = str(path.relative_to(out))
        if manifest.get("outputs", {}).get(relative) != sha256(path):
            raise ValueError(f"checkpoint hash missing/mismatched: {relative}")
    if summary.get("n_test_assigned") != len(expected) or summary.get("n_test_primary_paired") != len(paired):
        raise ValueError("summary fold counts do not match frozen technical gate")
    if summary.get("group_overlap_count") != 0 or summary.get("cuda_forward_backward_smoke") != "PASS":
        raise ValueError("summary group/CUDA smoke technical check failed")
    return {"status": "PASS", "outer_fold": args.fold, "assigned_predictions": len(expected),
            "primary_paired_predictions": len(paired),
            "supplementary_predictions": int(UNAVAILABLE_GRAPH_ID in expected),
            "finite_predictions": True, "frozen_ids_match": True,
            "S85_compute_lock_recorded": True, "checkpoints_verified": len(checkpoints)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "fit", "validate_fold"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), help="Required with --stage fit/validate_fold")
    parser.add_argument("--features", type=Path, required=True,
                        help="Pinned C134 V3 feature sidecar; currently outside this candidate package")
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--outer-split", type=Path, default=DEFAULT_OUTER)
    parser.add_argument("--inner-split", type=Path, default=DEFAULT_INNER)
    parser.add_argument("--graph-predictions", type=Path, default=DEFAULT_GRAPH_OOF)
    parser.add_argument("--output-dir", type=Path, help="New output directory for fit; existing fold directory for validate_fold")
    parser.add_argument("--skip-authority-hash-check", action="store_true",
                        help="Diagnostic only; never use for a released score")
    args = parser.parse_args(argv)
    if args.stage in {"fit", "validate_fold"} and (args.fold is None or args.output_dir is None):
        parser.error(f"--stage {args.stage} requires --fold and --output-dir")
    if args.stage == "preflight" and (args.fold is not None or args.output_dir is not None):
        parser.error("--stage preflight does not accept --fold or --output-dir")
    if args.stage == "fit" and args.output_dir is not None and args.output_dir.exists():
        parser.error(f"refusing existing fit output directory before preflight/fit: {args.output_dir}")
    args.verify_authority_hashes = not args.skip_authority_hash_check
    result = load_and_preflight(args)
    if args.stage == "fit":
        result = fit_fold(args, result)
    elif args.stage == "validate_fold":
        result = validate_fit_output(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
