#!/usr/bin/env python3
"""Fit the frozen Family-Aware-inspired joint runtime/quality adaptation.

One invocation fits one split/fold and all three registered seeds. Run only
after the full MPS measurement ledger exists and the timing worker has exited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark_v1/scripts"))
import run_mps_fixed_chi16_runtime_adaptation_v1 as s85

PROTOCOL_PATH = ROOT / "benchmark_v1/protocol/family_aware_joint_runtime_quality_mps_v1.json"
SEED_REGISTRY = ROOT / "benchmark_v1/registry/seed_registry.json"
DEFAULT_PLAN = ROOT / "work/family_aware_joint_mps_ladder_v1/preflight"
DEFAULT_MEASUREMENTS = ROOT / "artifacts/benchmark_v3/simulator/family_aware_joint_mps_ladder_v1/full"
DEFAULT_OUTPUT = ROOT / "artifacts/benchmark_v3/simulator/family_aware_joint_mps_ladder_v1/training"
FEATURES = ["active_width", "measurement_stripped_structural_depth", "one_qubit_gate_count",
            "two_qubit_gate_count", "multi_qubit_gate_count", "swap_like_gate_count"]
CHI = [2, 4, 8, 16, 32, 64]
SEEDS = [42, 1234, 31415]
THRESHOLD = 0.99
METHODS = ("family_conditioned", "family_agnostic_ablation")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def attach_minimum_passing_chi(labels: dict[tuple[str, int], dict[str, Any]], hashes: list[str]) -> None:
    """Only derive a circuit-level minimum after its entire six-rung ladder is valid."""
    for qhash in hashes:
        complete_ladder = all(labels[(qhash, chi)]["target_status"] == "runtime_observed" for chi in CHI)
        passing = [chi for chi in CHI if labels[(qhash, chi)]["quality_pass"] is True] if complete_ladder else []
        minimum = min(passing) if passing else None
        value: Any = minimum if minimum is not None else ("NO_PASS" if complete_ladder else "")
        for chi in CHI:
            labels[(qhash, chi)]["minimum_passing_chi"] = value


def static_features(row: dict[str, str]) -> list[float]:
    values = [max(0.0, float(row[name])) for name in FEATURES]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite static feature for {row['source_qasm_sha256']}")
    return [math.log1p(value) for value in values]


def load_training_data(plan_dir: Path, measure_dir: Path) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    plan_manifest_path = plan_dir / "preflight_manifest.json"
    plan_manifest = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
    if plan_manifest.get("protocol_sha256") != sha(PROTOCOL_PATH):
        raise ValueError("Family plan protocol hash changed")
    for filename, expected in plan_manifest["files"].items():
        if sha(plan_dir / filename) != expected:
            raise ValueError(f"Family plan materialization hash mismatch: {filename}")
    feature_rows = read_csv(plan_dir / "hash_feature_manifest.csv")
    if len(feature_rows) != 150 or len({row["source_qasm_sha256"] for row in feature_rows}) != 150:
        raise ValueError("training requires the frozen 150-hash feature manifest")
    full_summary_path = measure_dir / "run_summary.json"
    attempt_path = measure_dir / "attempt_records.jsonl"
    if not full_summary_path.is_file() or not attempt_path.is_file():
        raise FileNotFoundError("full MPS run_summary.json and attempt_records.jsonl are required")
    summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
    if summary.get("scope") != "full" or summary.get("timing_performed") is not True:
        raise ValueError("refusing pilot data: joint-model training requires full-panel measurements")
    attempts = [json.loads(line) for line in attempt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: set[tuple[str, int, str]] = set()
    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        key = (str(row["source_qasm_sha256"]), int(row["max_bond"]), str(row["session_id"]))
        if key in seen:
            raise ValueError(f"duplicate measured rung/session: {key}")
        seen.add(key)
        if row.get("attempt_terminal") is not True:
            raise ValueError(f"nonterminal measurement attempt: {key}")
        by_cell[key[:2]].append(row)
    feature_by_hash = {row["source_qasm_sha256"]: row for row in feature_rows}
    if set(feature_by_hash) != {row["source_qasm_sha256"] for row in attempts}:
        raise ValueError("measurement IDs differ from the frozen 150-hash panel")
    required = {(digest, chi, f"session-{session}") for digest in feature_by_hash
                for chi in CHI for session in (1, 2, 3)}
    if seen != required:
        raise ValueError(f"full attempt identity set incomplete: {len(seen)}/{len(required)}")

    labels: dict[tuple[str, int], dict[str, Any]] = {}
    hash_ir: dict[str, set[str]] = defaultdict(set)
    for (qhash, chi), rows in by_cell.items():
        successful = sorted(rows, key=lambda row: row["session_id"])
        valid = len(successful) == 3 and all(
            row.get("status") == "ok" and row.get("basis_canary_status") == "PASS" and
            len(row.get("warm_seconds", [])) == 5 and math.isfinite(float(row.get("fidelity", "nan")))
            for row in successful)
        if valid:
            hash_ir[qhash].update(str(row.get("ir_sha256")) for row in successful)
            session_medians = [statistics.median(float(value) for value in row["warm_seconds"])
                               for row in successful]
            runtime = statistics.median(session_medians)
            fidelity = [float(row["fidelity"]) for row in successful]
            if any(not math.isfinite(value) or value < 0 for value in [runtime, *fidelity]):
                valid = False
        else:
            runtime, fidelity = None, []
        labels[(qhash, chi)] = {
            "target_status": "runtime_observed" if valid else "unavailable_technical",
            "runtime_seconds": runtime,
            "quality_pass": bool(valid and all(value >= THRESHOLD for value in fidelity)) if valid else None,
            "minimum_passing_chi": "",
            "session_fidelity": fidelity,
        }
    for qhash, ir_hashes in hash_ir.items():
        if len(ir_hashes) != 1 or "None" in ir_hashes:
            for chi in CHI:
                labels[(qhash, chi)].update(target_status="unavailable_ir_mismatch", runtime_seconds=None,
                                           quality_pass=None)
    attach_minimum_passing_chi(labels, list(feature_by_hash))

    return {"protocol": protocol, "plan_manifest": plan_manifest, "measurement_summary": summary,
            "feature_rows": feature_rows, "feature_by_hash": feature_by_hash, "labels": labels,
            "attempt_sha256": sha(attempt_path), "plan_sha256": sha(plan_manifest_path),
            "measurement_summary_sha256": sha(full_summary_path), "attempt_rows": len(attempts)}


def prepare_fold(data: dict[str, Any], split_id: str, fold: int) -> dict[str, Any]:
    features = data["feature_by_hash"]
    fold_col = "c44_fold" if split_id == "c44" else "family_holdout_fold_diagnostic_only"
    train_hashes = sorted(qhash for qhash, row in features.items() if int(row[fold_col]) != fold)
    test_hashes = sorted(qhash for qhash, row in features.items() if int(row[fold_col]) == fold)
    if not train_hashes or not test_hashes or set(train_hashes) & set(test_hashes):
        raise ValueError(f"invalid {split_id} fold {fold} assignment")
    components = {qhash: features[qhash]["family_component_training_label_only"] for qhash in features}
    vocab = sorted({components[qhash] for qhash in train_hashes})
    raw = {qhash: static_features(features[qhash]) for qhash in features}
    x_mean = np.mean(np.asarray([raw[qhash] for qhash in train_hashes]), axis=0)
    x_scale = np.std(np.asarray([raw[qhash] for qhash in train_hashes]), axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    x_scaled = {qhash: ((np.asarray(raw[qhash]) - x_mean) / x_scale).astype(np.float32) for qhash in features}
    train_cells = [(qhash, chi) for qhash in train_hashes for chi in CHI
                   if data["labels"][(qhash, chi)]["runtime_seconds"] is not None]
    if not train_cells:
        raise ValueError(f"no technically complete training rung labels in {split_id} fold {fold}")
    log_y = np.asarray([math.log1p(float(data["labels"][cell]["runtime_seconds"])) for cell in train_cells])
    y_mean, y_scale = float(log_y.mean()), float(log_y.std())
    if y_scale < 1e-12:
        y_scale = 1.0

    def make_x(qhash: str, chi: int) -> np.ndarray:
        return np.concatenate((x_scaled[qhash], np.asarray([math.log2(chi)], dtype=np.float32)))

    # For every registered model seed, each primary training family input is
    # supplied by a same-seed classifier that excluded that exact QASM hash.
    inner_predictions_by_seed: dict[int, dict[str, str]] = {}
    test_predictions_by_seed: dict[int, dict[str, str]] = {}
    classifier_payloads: dict[int, dict[str, Any]] = {}
    family_skipped = len(train_hashes) < 2 or len(vocab) < 2
    if not family_skipped:
        inner_count = min(4, len(train_hashes))
        groups = np.asarray(train_hashes)
        y_family = np.asarray([components[qhash] for qhash in train_hashes])
        for seed in SEEDS:
            inner_predictions: dict[str, str] = {}
            inner_classifier_records = []
            splitter = GroupKFold(n_splits=inner_count)
            for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(
                    splitter.split(groups, y_family, groups=groups)):
                inner_train = [train_hashes[index] for index in inner_train_idx]
                inner_valid = [train_hashes[index] for index in inner_valid_idx]
                inner_vocab = sorted({components[qhash] for qhash in inner_train})
                if len(inner_vocab) == 1:
                    for qhash in inner_valid:
                        inner_predictions[qhash] = inner_vocab[0]
                    inner_classifier_records.append({"inner_fold": inner_fold,
                        "train_hashes": inner_train, "validation_hashes": inner_valid,
                        "vocabulary": inner_vocab, "constant_prediction": inner_vocab[0],
                        "mean": None, "scale": None, "state_dict": None})
                    continue
                inner_mean = np.mean(np.asarray([raw[qhash] for qhash in inner_train]), axis=0)
                inner_scale = np.std(np.asarray([raw[qhash] for qhash in inner_train]), axis=0)
                inner_scale[inner_scale < 1e-12] = 1.0
                inner_x = np.asarray([(np.asarray(raw[qhash]) - inner_mean) / inner_scale
                                      for qhash in inner_train], dtype=np.float32)
                inner_y = np.asarray([inner_vocab.index(components[qhash]) for qhash in inner_train], dtype=np.int64)
                classifier = fit_family_classifier(inner_x, inner_y, inner_vocab, seed)
                valid_x = np.asarray([(np.asarray(raw[qhash]) - inner_mean) / inner_scale
                                      for qhash in inner_valid], dtype=np.float32)
                predicted = predict_family(classifier, valid_x, inner_vocab)
                inner_predictions.update(dict(zip(inner_valid, predicted)))
                inner_classifier_records.append({"inner_fold": inner_fold,
                    "train_hashes": inner_train, "validation_hashes": inner_valid,
                    "vocabulary": inner_vocab, "constant_prediction": None,
                    "mean": inner_mean.tolist(), "scale": inner_scale.tolist(),
                    "state_dict": classifier_state_dict(classifier)})
            if set(inner_predictions) != set(train_hashes):
                raise ValueError("inner GroupKFold family predictions do not cover all outer-training hashes")
            outer_classifier = fit_family_classifier(
                np.asarray([x_scaled[qhash] for qhash in train_hashes], dtype=np.float32),
                np.asarray([vocab.index(components[qhash]) for qhash in train_hashes], dtype=np.int64),
                vocab, seed)
            test_family_predictions = predict_family(outer_classifier,
                np.asarray([x_scaled[qhash] for qhash in test_hashes], dtype=np.float32), vocab)
            inner_predictions_by_seed[seed] = inner_predictions
            test_predictions_by_seed[seed] = dict(zip(test_hashes, test_family_predictions))
            classifier_payloads[seed] = {
                "seed": seed, "inner_group_kfold": inner_classifier_records,
                "outer_test_classifier": {
                    "train_hashes": train_hashes, "test_hashes": test_hashes,
                    "vocabulary": vocab, "mean": x_mean.tolist(), "scale": x_scale.tolist(),
                    "state_dict": classifier_state_dict(outer_classifier),
                },
            }
    else:
        for seed in SEEDS:
            inner_predictions_by_seed[seed] = {qhash: "__unknown__" for qhash in train_hashes}
            test_predictions_by_seed[seed] = {qhash: "__unknown__" for qhash in test_hashes}
            classifier_payloads[seed] = {"seed": seed, "family_conditioning_skipped": True,
                                         "reason": "fewer_than_two_training_hashes_or_family_classes"}

    family_indices = {name: index for index, name in enumerate(vocab)}
    train_family_by_seed = {
        seed: np.asarray([family_indices.get(inner_predictions_by_seed[seed][qhash], len(vocab))
                          for qhash, _ in train_cells], dtype=np.int64)
        for seed in SEEDS
    }
    return {"split_id": split_id, "fold": fold, "fold_column": fold_col,
            "train_hashes": train_hashes, "test_hashes": test_hashes,
            "train_cells": train_cells, "components": components, "vocab": vocab,
            "family_skipped": family_skipped, "test_family_prediction_by_seed": test_predictions_by_seed,
            "x_scaled": x_scaled, "x_mean": x_mean.tolist(), "x_scale": x_scale.tolist(),
            "y_mean": y_mean, "y_scale": y_scale,
            "train_x": np.asarray([make_x(qhash, chi) for qhash, chi in train_cells], dtype=np.float32),
            "train_family_by_seed": train_family_by_seed,
            "train_y": ((log_y - y_mean) / y_scale).astype(np.float32),
            "train_quality": np.asarray([int(data["labels"][(qhash, chi)]["quality_pass"]) for qhash, chi in train_cells], dtype=np.float32),
            "test_x": {(qhash, chi): make_x(qhash, chi) for qhash in test_hashes for chi in CHI},
            "family_indices": family_indices, "labels": data["labels"],
            "classifier_payloads": classifier_payloads}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def classifier_state_dict(model) -> dict[str, torch.Tensor] | None:
    if isinstance(model, dict):
        return None
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def model_classes():
    class Classifier(torch.nn.Module):
        def __init__(self, class_count: int):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.SiLU(),
                                           torch.nn.Linear(64, 64), torch.nn.SiLU(),
                                           torch.nn.Linear(64, class_count))
        def forward(self, x):
            return self.net(x)

    class FamilyJoint(torch.nn.Module):
        def __init__(self, family_count: int):
            super().__init__()
            self.backbone = torch.nn.Sequential(torch.nn.Linear(7, 128), torch.nn.SiLU(),
                torch.nn.Dropout(0.2), torch.nn.Linear(128, 64), torch.nn.SiLU(), torch.nn.Dropout(0.2))
            self.embedding = torch.nn.Embedding(family_count + 1, 64)
            self.family_branch = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU())
            self.gamma = torch.nn.Linear(64, 64)
            self.beta = torch.nn.Linear(64, 64)
            self.runtime_main = torch.nn.Linear(64, 1)
            self.runtime_shortcut = torch.nn.Linear(7, 1)
            self.runtime_family = torch.nn.Linear(64, 1)
            self.quality_main = torch.nn.Linear(64, 1)
            self.quality_shortcut = torch.nn.Linear(7, 1)
            torch.nn.init.zeros_(self.runtime_family.weight)
            torch.nn.init.zeros_(self.runtime_family.bias)
        def forward(self, x, family):
            f = self.family_branch(self.embedding(family))
            h = self.backbone(x) * (1.0 + self.gamma(f)) + self.beta(f)
            return (self.runtime_main(h) + self.runtime_shortcut(x) + self.runtime_family(f)).squeeze(-1), \
                   (self.quality_main(h) + self.quality_shortcut(x)).squeeze(-1)

    class Ablation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(torch.nn.Linear(7, 128), torch.nn.SiLU(),
                torch.nn.Dropout(0.2), torch.nn.Linear(128, 64), torch.nn.SiLU(), torch.nn.Dropout(0.2))
            self.runtime_main = torch.nn.Linear(64, 1)
            self.runtime_shortcut = torch.nn.Linear(7, 1)
            self.quality_main = torch.nn.Linear(64, 1)
            self.quality_shortcut = torch.nn.Linear(7, 1)
        def forward(self, x, family=None):
            h = self.backbone(x)
            return (self.runtime_main(h) + self.runtime_shortcut(x)).squeeze(-1), \
                   (self.quality_main(h) + self.quality_shortcut(x)).squeeze(-1)
    return Classifier, FamilyJoint, Ablation


def fit_family_classifier(x: np.ndarray, y: np.ndarray, vocab: list[str], seed: int):
    if len(vocab) == 1:
        return {"constant": vocab[0]}
    Classifier, _, _ = model_classes()
    seed_everything(seed)
    model = Classifier(len(vocab)).cuda()
    tx = torch.as_tensor(x, dtype=torch.float32, device="cuda")
    ty = torch.as_tensor(y, dtype=torch.long, device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    model.train()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(tx), ty)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite family-classifier loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("non-finite family-classifier gradient")
        optimizer.step()
    model.eval()
    return model


def predict_family(model, x: np.ndarray, vocab: list[str]) -> list[str]:
    if isinstance(model, dict):
        return [str(model["constant"])] * len(x)
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32, device="cuda"))
        index = logits.argmax(dim=1).cpu().tolist()
    return [vocab[int(i)] if 0 <= int(i) < len(vocab) else "__unknown__" for i in index]


def train_joint_fold(fold_data: dict[str, Any], split_id: str, fold: int, output: Path) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this model; CPU fallback is forbidden")
    Classifier, FamilyJoint, Ablation = model_classes()
    method_models: dict[str, list[dict[str, Any]]] = {name: [] for name in METHODS}
    tx = torch.as_tensor(fold_data["train_x"], dtype=torch.float32, device="cuda")
    ty = torch.as_tensor(fold_data["train_y"], dtype=torch.float32, device="cuda")
    tq = torch.as_tensor(fold_data["train_quality"], dtype=torch.float32, device="cuda")
    predictions: dict[str, dict[tuple[str, int], dict[str, float]]] = {name: {} for name in METHODS}
    for seed in SEEDS:
        tf = torch.as_tensor(fold_data["train_family_by_seed"][seed], dtype=torch.long, device="cuda")
        seed_everything(seed)
        family_model = (Ablation() if fold_data["family_skipped"] else FamilyJoint(len(fold_data["vocab"]))).cuda()
        seed_everything(seed)
        ablation_model = Ablation().cuda()
        # Copy common initialization so the ablation starts from exactly the
        # shared weights used by the family model.
        base = family_model.state_dict()
        other = ablation_model.state_dict()
        common = ("backbone.", "runtime_main.", "runtime_shortcut.", "quality_main.", "quality_shortcut.")
        for name, value in base.items():
            if name.startswith(common) and name in other and other[name].shape == value.shape:
                other[name] = value.detach().clone()
        ablation_model.load_state_dict(other)
        for method_id, model in ((METHODS[0], family_model), (METHODS[1], ablation_model)):
            seed_everything(seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160)
            model.train()
            epoch_records = []
            for epoch in range(160):
                optimizer.zero_grad(set_to_none=True)
                pred_y, pred_q = model(tx, tf) if method_id == METHODS[0] else model(tx)
                runtime_loss = torch.nn.functional.mse_loss(pred_y, ty)
                quality_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_q, tq)
                loss = runtime_loss + quality_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite joint loss: {split_id} fold={fold} seed={seed} method={method_id}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError("non-finite joint-model gradient")
                optimizer.step()
                scheduler.step()
                if (epoch + 1) % 20 == 0 or epoch == 159:
                    epoch_records.append({"epoch": epoch + 1, "total_loss": float(loss.detach().cpu()),
                                          "runtime_loss": float(runtime_loss.detach().cpu()),
                                          "quality_loss": float(quality_loss.detach().cpu())})
            model.eval()
            with torch.no_grad():
                for cell, x_np in fold_data["test_x"].items():
                    x = torch.as_tensor(x_np, dtype=torch.float32, device="cuda").reshape(1, -1)
                    if method_id == METHODS[0]:
                        predicted_name = fold_data["test_family_prediction_by_seed"][seed][cell[0]]
                        family_idx = fold_data["family_indices"].get(predicted_name, len(fold_data["vocab"]))
                        family_tensor = torch.tensor([family_idx], dtype=torch.long, device="cuda")
                        raw_runtime, quality_logit = model(x, family_tensor) if not fold_data["family_skipped"] else model(x)
                    else:
                        raw_runtime, quality_logit = model(x)
                    predicted_log = raw_runtime * fold_data["y_scale"] + fold_data["y_mean"]
                    seconds = max(0.0, math.expm1(float(predicted_log.item())))
                    probability = float(torch.sigmoid(quality_logit).item())
                    if not math.isfinite(seconds) or not math.isfinite(probability):
                        raise RuntimeError("non-finite prediction")
                    predictions[method_id].setdefault(cell, {})[str(seed)] = {
                        "runtime_seconds": seconds, "quality_pass_probability": probability}
            checkpoint = output / "checkpoints" / f"{method_id}_seed_{seed}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temp = checkpoint.with_suffix(".pt.tmp")
            torch.save({"seed": seed, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "completed_epoch": 160,
                        "protocol_sha256": sha(PROTOCOL_PATH), "fold": fold, "split_id": split_id}, temp)
            os.replace(temp, checkpoint)
            write_json(output / "training_log" / f"{method_id}_seed_{seed}.json", epoch_records)
    rows = []
    for method_id, cells in predictions.items():
        for qhash in fold_data["test_hashes"]:
            for chi in CHI:
                cell = (qhash, chi)
                label = fold_data["labels"][cell]
                by_seed = cells[cell]
                row = {"method_id": method_id, "split_id": split_id, "fold": fold,
                       "source_qasm_sha256": qhash, "max_bond": chi,
                       "target_status": label["target_status"],
                       "target_runtime_seconds": label["runtime_seconds"] if label["runtime_seconds"] is not None else "",
                       "target_quality_pass": label["quality_pass"] if label["quality_pass"] is not None else "",
                       "minimum_passing_chi": label["minimum_passing_chi"],
                       "family_component_diagnostic_only": fold_data["components"][qhash],
                       "family_conditioning_skipped": fold_data["family_skipped"]}
                for seed in SEEDS:
                    row[f"predicted_family_seed_{seed}"] = fold_data["test_family_prediction_by_seed"][seed][qhash]
                for seed in SEEDS:
                    row[f"runtime_seed_{seed}_seconds"] = by_seed[str(seed)]["runtime_seconds"]
                    row[f"quality_seed_{seed}_probability"] = by_seed[str(seed)]["quality_pass_probability"]
                row["runtime_median_seed_seconds"] = statistics.median(row[f"runtime_seed_{seed}_seconds"] for seed in SEEDS)
                row["quality_mean_seed_probability"] = statistics.mean(row[f"quality_seed_{seed}_probability"] for seed in SEEDS)
                row["quality_predicted_pass"] = row["quality_mean_seed_probability"] >= 0.5
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--measure-dir", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("c44", "family_holdout"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"refusing existing output directory: {output}")
    data = load_training_data(args.plan_dir.resolve(), args.measure_dir.resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU training fallback")
    torch.set_num_threads(2)
    output.mkdir(parents=True)
    manifest = {"artifact_id": f"family-joint-{args.split}-fold-{args.fold}", "status": "running",
                "created_utc": datetime.now(timezone.utc).isoformat(), "split_id": args.split,
                "fold": args.fold, "protocol_sha256": sha(PROTOCOL_PATH),
                "plan_sha256": data["plan_sha256"], "measurement_attempts_sha256": data["attempt_sha256"],
                "measurement_summary_sha256": data["measurement_summary_sha256"],
                "runner_sha256": sha(Path(__file__).resolve()), "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0),
                "seeds": SEEDS, "device": "cuda", "fit_started": False,
                "training_hashes": None, "test_hashes": None, "prediction_sha256": None}
    write_json(output / "run_manifest.json", manifest)
    try:
        with s85.exclusive_compute_lock():
            manifest["fit_started"] = True
            manifest["compute_lock_held"] = True
            write_json(output / "run_manifest.json", manifest)
            fold_data = prepare_fold(data, args.split, args.fold)
            feature_state = {"x_mean": fold_data["x_mean"], "x_scale": fold_data["x_scale"],
                             "runtime_log_mean": fold_data["y_mean"], "runtime_log_scale": fold_data["y_scale"],
                             "family_vocabulary": fold_data["vocab"], "train_hashes": fold_data["train_hashes"],
                             "test_hashes": fold_data["test_hashes"], "family_classifier_crossfit": True,
                             "family_classifier_prediction_seeds": list(SEEDS)}
            write_json(output / "fold_transform_and_support.json", feature_state)
            classifier_dir = output / "family_classifier_checkpoints"
            classifier_dir.mkdir(parents=True, exist_ok=True)
            for seed, payload in fold_data["classifier_payloads"].items():
                temp = classifier_dir / f"seed_{seed}.pt.tmp"
                torch.save(payload, temp)
                os.replace(temp, classifier_dir / f"seed_{seed}.pt")
            rows = train_joint_fold(fold_data, args.split, args.fold, output)
            write_csv(output / "oof_predictions.csv", rows)
            if len(rows) != len(fold_data["test_hashes"]) * len(CHI) * len(METHODS):
                raise ValueError("OOF prediction denominator mismatch")
            if not all(math.isfinite(float(row["runtime_median_seed_seconds"])) and
                       math.isfinite(float(row["quality_mean_seed_probability"])) for row in rows):
                raise ValueError("non-finite OOF prediction")
            manifest.update({"status": "completed_fold_technical_qa_pass", "training_hashes": len(fold_data["train_hashes"]),
                             "test_hashes": len(fold_data["test_hashes"]), "training_rung_rows": len(fold_data["train_cells"]),
                             "prediction_rows": len(rows), "quality_labels_used_as_inputs": False,
                             "test_runtime_or_fidelity_used_for_fit": False,
                             "compute_lock_held": True, "prediction_sha256": sha(output / "oof_predictions.csv"),
                             "transform_sha256": sha(output / "fold_transform_and_support.json"),
                             "checkpoints": {p.name: sha(p) for p in sorted((output / "checkpoints").glob("*.pt"))},
                             "family_classifier_checkpoints": {p.name: sha(p) for p in sorted(classifier_dir.glob("*.pt"))},
                             "training_logs": {p.name: sha(p) for p in sorted((output / "training_log").glob("*.json"))},
                             "hardware": {"device": torch.cuda.get_device_name(0),
                                          "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                                          "cpu_threads": torch.get_num_threads(),
                                          "python_executable": sys.executable,
                                          "python_version": sys.version,
                                          "torch_geometric_version": __import__("importlib.metadata", fromlist=["version"]).version("torch-geometric")}})
            write_json(output / "run_manifest.json", manifest)
    except BaseException as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}".splitlines()[0][:1000]})
        write_json(output / "run_manifest.json", manifest)
        raise
    print(json.dumps({"status": manifest["status"], "split": args.split, "fold": args.fold,
                      "train_hashes": manifest["training_hashes"], "test_hashes": manifest["test_hashes"],
                      "prediction_rows": manifest["prediction_rows"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
