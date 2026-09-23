"""Audit the public Quantum Rings protocol beyond the paper's headline table.

This is a follow-up audit, not a replacement for ``replicate_paper.py``.  It
uses the same public JSON/QASM data and circuit-level folds to answer four
questions raised during review:

* do runtime tree regressors provide a competitive baseline?
* do the incremental feature/family/loss ablations reproduce the paper's
  direction of improvement?
* are aggregate metrics hiding a CPU/GPU or precision failure mode?
* does the model transfer to a repository family that was absent from train?

The public forward runtime is available only at the challenge's 0.99-selected
threshold.  ``threshold_conditioning`` is therefore reported as a diagnostic
on that public target; it is not a reconstruction of the paper's missing
0.75 forward-runtime labels.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn


_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ["QUANTUM_RINGS_ROOT"]).resolve() if os.environ.get("QUANTUM_RINGS_ROOT") else _SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.replicate_paper import (
    CLASS_NAMES,
    FAMILY_UNKNOWN,
    LADDER,
    FamilyAwareNet,
    _decayed_threshold_loss,
    _scaled_features,
    _threshold_metrics,
    extract_features,
    runtime_target,
)


FEATURE_COLUMNS = [f"f{index:02d}" for index in range(32)]
# Basic/gate/complexity/local/fingerprint features + context, excluding the
# five RCM/cut-pressure graph features (indices 23..27).
GLOBAL_INDICES = list(range(23)) + list(range(28, 32))


def repository_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def threshold_value(result: dict, target: float) -> int:
    for sweep in sorted(result.get("threshold_sweep", []), key=lambda item: item["threshold"]):
        fidelity = sweep.get("sdk_get_fidelity")
        if (
            sweep.get("threshold") in LADDER
            and fidelity is not None
            and sweep.get("returncode", 0) == 0
            and float(fidelity) >= target
        ):
            return int(sweep["threshold"])
    return 0


def load_rows(target: float) -> tuple[pd.DataFrame, dict[str, int]]:
    data = json.loads((ROOT / "data" / "hackathon_public.json").read_text())
    circuit_meta = {item["file"]: item for item in data["circuits"]}
    cache: dict[str, object] = {}
    rows: list[dict] = []
    for result in data["results"]:
        file_name = result["file"]
        circuit = cache.get(file_name)
        if circuit is None:
            from qiskit import QuantumCircuit

            circuit = QuantumCircuit.from_qasm_file(ROOT / "circuits" / file_name)
            cache[file_name] = circuit
        base = extract_features(circuit, "CPU", "single")[:28]
        context = extract_features(circuit, result["backend"], result["precision"])[28:]
        runtime, runtime_source = runtime_target(result, target)
        rows.append(
            {
                "file": file_name,
                "family": circuit_meta[file_name]["family"],
                "backend": result["backend"],
                "precision": result["precision"],
                "threshold_value": threshold_value(result, target),
                "threshold_class": CLASS_NAMES.index(str(threshold_value(result, target)))
                if threshold_value(result, target)
                else 0,
                "runtime_s": runtime,
                "runtime_source": runtime_source,
                "features": np.concatenate([base, context]),
            }
        )
    frame = pd.DataFrame(rows)
    matrix = np.vstack(frame.pop("features").to_numpy())
    for index in range(matrix.shape[1]):
        frame[f"f{index:02d}"] = matrix[:, index]
    families = sorted(frame["family"].unique())
    family_to_index = {family: index for index, family in enumerate(families)}
    family_to_index[FAMILY_UNKNOWN] = len(family_to_index)
    return frame, family_to_index


def regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    y_true_seconds = np.exp(y_true_log)
    y_pred_seconds = np.exp(np.clip(y_pred_log, -20.0, 20.0))
    return {
        "r2_log_runtime": float(r2_score(y_true_log, y_pred_log)),
        "r2_runtime_seconds": float(r2_score(y_true_seconds, y_pred_seconds)),
        "mae_runtime_seconds": float(mean_absolute_error(y_true_seconds, y_pred_seconds)),
        "median_relative_error": float(
            np.median(np.abs(y_pred_seconds - y_true_seconds) / np.maximum(y_true_seconds, 1e-12))
        ),
    }


def folds_for(frame: pd.DataFrame, seed: int):
    files = np.asarray(sorted(frame["file"].unique()))
    return KFold(n_splits=5, shuffle=True, random_state=seed).split(files), files


def tree_baselines(frame: pd.DataFrame, seed: int) -> dict:
    all_predictions = {
        "random_forest": [],
        "extra_trees": [],
        "gradient_boosting": [],
    }
    threshold_predictions: list[np.ndarray] = []
    truths: list[tuple[np.ndarray, np.ndarray]] = []
    contexts: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    fold_iter, files = folds_for(frame, seed)
    for train_idx, test_idx in fold_iter:
        train_files, test_files = files[train_idx], files[test_idx]
        train = frame[frame.file.isin(train_files)]
        test = frame[frame.file.isin(test_files)]
        x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
        x_test = test[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train_log = np.log(np.maximum(train.runtime_s.to_numpy(dtype=float), 1e-9))
        y_test_log = np.log(np.maximum(test.runtime_s.to_numpy(dtype=float), 1e-9))
        y_train_threshold = train.threshold_class.to_numpy(dtype=int)
        y_test_threshold = test.threshold_class.to_numpy(dtype=int)

        classifier = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
        classifier.fit(x_train, y_train_threshold)
        threshold_predictions.append(classifier.predict(x_test))
        truths.append((y_test_threshold, y_test_log))

        models = {
            "random_forest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1),
            "extra_trees": ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=1),
            "gradient_boosting": GradientBoostingRegressor(random_state=seed),
        }
        for name, model in models.items():
            model.fit(x_train, y_train_log)
            all_predictions[name].append(model.predict(x_test))

        fold_contexts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, group in test.groupby(["backend", "precision"]):
            indices = group.index.to_numpy()
            local = test.index.get_indexer(indices)
            fold_contexts["+".join(key)] = (local, y_test_log[local])
        contexts.append(fold_contexts)

    y_threshold = np.concatenate([item[0] for item in truths])
    y_runtime = np.concatenate([item[1] for item in truths])
    result = {
        "threshold_random_forest": _threshold_metrics(
            y_threshold, np.concatenate(threshold_predictions)
        ),
        "runtime": {},
        "per_context_random_forest": {},
    }
    for name, predictions in all_predictions.items():
        result["runtime"][name] = regression_metrics(y_runtime, np.concatenate(predictions))

    # Re-run the RF predictions in fold order for transparent per-context rows.
    rf_predictions = np.concatenate(all_predictions["random_forest"])
    offsets = 0
    per_context: dict[str, tuple[list[float], list[float]]] = {}
    for fold_context, (_, y_test_log) in zip(contexts, truths):
        fold_n = len(y_test_log)
        fold_pred = rf_predictions[offsets : offsets + fold_n]
        offsets += fold_n
        for key, (local, truth) in fold_context.items():
            bucket = per_context.setdefault(key, ([], []))
            bucket[0].extend(truth.tolist())
            bucket[1].extend(fold_pred[local].tolist())
    for key, (truth, pred) in sorted(per_context.items()):
        result["per_context_random_forest"][key] = regression_metrics(
            np.asarray(truth), np.asarray(pred)
        )
    return result


def train_ablation_model(
    x_train: np.ndarray,
    y_threshold: np.ndarray,
    y_runtime_normalized: np.ndarray,
    family_train: np.ndarray,
    n_families: int,
    use_family: bool,
    decayed_ce: bool,
    seed: int,
) -> FamilyAwareNet:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = FamilyAwareNet(x_train.shape[1], n_families, len(CLASS_NAMES), use_family)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160)
    features = torch.tensor(x_train, dtype=torch.float32)
    thresholds = torch.tensor(y_threshold, dtype=torch.long)
    runtimes = torch.tensor(y_runtime_normalized, dtype=torch.float32)
    families = torch.tensor(family_train, dtype=torch.long)
    model.train()
    for _ in range(160):
        optimizer.zero_grad()
        logits, runtime = model(features, families)
        threshold_loss = (
            _decayed_threshold_loss(logits, thresholds)
            if decayed_ce
            else torch.nn.functional.cross_entropy(logits, thresholds)
        )
        loss = threshold_loss + torch.mean((runtime - runtimes) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return model.eval()


def train_family_classifier(train: pd.DataFrame, test: pd.DataFrame, scaler: StandardScaler, seed: int):
    train_unique = train.drop_duplicates("file")
    test_unique = test.drop_duplicates("file")
    classifier = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")
        classifier.fit(
            scaler.transform(_scaled_features(train_unique[FEATURE_COLUMNS].to_numpy()))[:, :28],
            train_unique.family,
        )
    predicted = classifier.predict(
        scaler.transform(_scaled_features(test_unique[FEATURE_COLUMNS].to_numpy()))[:, :28]
    )
    return dict(zip(test_unique.file, predicted)), accuracy_score(test_unique.family, predicted)


def mlp_ablations(frame: pd.DataFrame, family_to_index: dict[str, int], seed: int) -> dict:
    names = {
        "baseline_mlp": (GLOBAL_INDICES, False, False),
        "plus_graph_features": (list(range(32)), False, False),
        "plus_family_classifier": (list(range(32)), True, False),
        "plus_decayed_ce": (list(range(32)), True, True),
    }
    predictions = {name: [] for name in names}
    truths: list[tuple[np.ndarray, np.ndarray]] = []
    fold_iter, files = folds_for(frame, seed)
    family_accuracy: list[float] = []
    for fold, (train_idx, test_idx) in enumerate(fold_iter):
        train_files, test_files = files[train_idx], files[test_idx]
        train = frame[frame.file.isin(train_files)].copy()
        test = frame[frame.file.isin(test_files)].copy()
        full_train = _scaled_features(train[FEATURE_COLUMNS].to_numpy(dtype=float))
        full_test = _scaled_features(test[FEATURE_COLUMNS].to_numpy(dtype=float))
        scaler = StandardScaler().fit(full_train)
        full_train = scaler.transform(full_train)
        full_test = scaler.transform(full_test)
        y_threshold_train = train.threshold_class.to_numpy(dtype=int)
        y_threshold_test = test.threshold_class.to_numpy(dtype=int)
        y_runtime_train = np.log(np.maximum(train.runtime_s.to_numpy(dtype=float), 1e-9))
        y_runtime_test = np.log(np.maximum(test.runtime_s.to_numpy(dtype=float), 1e-9))
        center = float(y_runtime_train.mean())
        scale = float(y_runtime_train.std()) or 1.0
        y_runtime_train_norm = (y_runtime_train - center) / scale
        family_train = np.asarray([family_to_index[x] for x in train.family], dtype=int)
        family_predicted_by_file, family_acc = train_family_classifier(train, test, scaler, seed)
        family_accuracy.append(float(family_acc))
        family_test = np.asarray(
            [family_to_index[family_predicted_by_file[file_name]] for file_name in test.file],
            dtype=int,
        )
        truths.append((y_threshold_test, y_runtime_test))
        for name, (indices, use_family, decayed_ce) in names.items():
            x_train = full_train[:, indices]
            x_test = full_test[:, indices]
            model = train_ablation_model(
                x_train,
                y_threshold_train,
                y_runtime_train_norm,
                family_train,
                len(family_to_index),
                use_family,
                decayed_ce,
                seed + fold + (100 if use_family else 0) + (1000 if decayed_ce else 0),
            )
            with torch.no_grad():
                logits, runtime = model(
                    torch.tensor(x_test, dtype=torch.float32),
                    torch.tensor(family_test, dtype=torch.long),
                )
            predictions[name].append(
                (logits.argmax(1).numpy(), runtime.numpy() * scale + center)
            )
    y_threshold = np.concatenate([item[0] for item in truths])
    y_runtime = np.concatenate([item[1] for item in truths])
    result = {"family_classifier_accuracy": float(np.mean(family_accuracy)), "models": {}}
    for name, fold_predictions in predictions.items():
        pred_threshold = np.concatenate([item[0] for item in fold_predictions])
        pred_runtime = np.concatenate([item[1] for item in fold_predictions])
        result["models"][name] = {
            "threshold": _threshold_metrics(y_threshold, pred_threshold),
            "runtime": regression_metrics(y_runtime, pred_runtime),
        }
    return result


def leave_one_family_out(frame: pd.DataFrame, seed: int) -> dict:
    all_truth_threshold: list[np.ndarray] = []
    all_pred_threshold: list[np.ndarray] = []
    all_truth_runtime: list[np.ndarray] = []
    all_pred_runtime: list[np.ndarray] = []
    per_family: dict[str, dict] = {}
    for family in sorted(frame.family.unique()):
        train = frame[frame.family != family]
        test = frame[frame.family == family]
        x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
        x_test = test[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train_threshold = train.threshold_class.to_numpy(dtype=int)
        y_test_threshold = test.threshold_class.to_numpy(dtype=int)
        y_train_runtime = np.log(np.maximum(train.runtime_s.to_numpy(dtype=float), 1e-9))
        y_test_runtime = np.log(np.maximum(test.runtime_s.to_numpy(dtype=float), 1e-9))
        classifier = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
        regressor = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1)
        classifier.fit(x_train, y_train_threshold)
        regressor.fit(x_train, y_train_runtime)
        pred_threshold = classifier.predict(x_test)
        pred_runtime = regressor.predict(x_test)
        all_truth_threshold.append(y_test_threshold)
        all_pred_threshold.append(pred_threshold)
        all_truth_runtime.append(y_test_runtime)
        all_pred_runtime.append(pred_runtime)
        per_family[family] = {
            "n_circuits": int(test.file.nunique()),
            "n_rows": int(len(test)),
            "threshold": _threshold_metrics(y_test_threshold, pred_threshold),
            "runtime": regression_metrics(y_test_runtime, pred_runtime),
        }
    truth_threshold = np.concatenate(all_truth_threshold)
    pred_threshold = np.concatenate(all_pred_threshold)
    truth_runtime = np.concatenate(all_truth_runtime)
    pred_runtime = np.concatenate(all_pred_runtime)
    return {
        "n_families": len(per_family),
        "aggregate": {
            "threshold": _threshold_metrics(truth_threshold, pred_threshold),
            "runtime": regression_metrics(truth_runtime, pred_runtime),
        },
        "per_family": per_family,
    }


def threshold_conditioning(frame: pd.DataFrame, seed: int) -> dict:
    """Compare runtime RF with true versus predicted threshold as an input."""
    no_threshold: list[np.ndarray] = []
    oracle_threshold: list[np.ndarray] = []
    predicted_threshold: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    fold_iter, files = folds_for(frame, seed)
    for train_idx, test_idx in fold_iter:
        train_files, test_files = files[train_idx], files[test_idx]
        train = frame[frame.file.isin(train_files)]
        test = frame[frame.file.isin(test_files)]
        x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
        x_test = test[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train_log = np.log(np.maximum(train.runtime_s.to_numpy(dtype=float), 1e-9))
        y_test_log = np.log(np.maximum(test.runtime_s.to_numpy(dtype=float), 1e-9))
        threshold_train = np.log1p(train.threshold_value.to_numpy(dtype=float)).reshape(-1, 1)
        threshold_true_test = np.log1p(test.threshold_value.to_numpy(dtype=float)).reshape(-1, 1)
        classifier = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
        classifier.fit(x_train, train.threshold_class.to_numpy(dtype=int))
        predicted_class = classifier.predict(x_test)
        predicted_rung = np.asarray(
            [0.0 if index == 0 else float(LADDER[index - 1]) for index in predicted_class]
        ).reshape(-1, 1)
        models = {
            "no_threshold": (x_train, x_test),
            "oracle_threshold": (
                np.hstack([x_train, threshold_train]),
                np.hstack([x_test, threshold_true_test]),
            ),
            "predicted_threshold": (
                np.hstack([x_train, threshold_train]),
                np.hstack([x_test, predicted_rung]),
            ),
        }
        for name, (train_x, test_x) in models.items():
            regressor = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1)
            regressor.fit(train_x, y_train_log)
            prediction = regressor.predict(test_x)
            if name == "no_threshold":
                no_threshold.append(prediction)
            elif name == "oracle_threshold":
                oracle_threshold.append(prediction)
            else:
                predicted_threshold.append(prediction)
        truths.append(y_test_log)
    truth = np.concatenate(truths)
    return {
        "public_target": 0.99,
        "runtime_source": "forward_at_public_0.99_selection",
        "models": {
            "no_threshold": regression_metrics(truth, np.concatenate(no_threshold)),
            "oracle_threshold": regression_metrics(truth, np.concatenate(oracle_threshold)),
            "predicted_threshold": regression_metrics(truth, np.concatenate(predicted_threshold)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("paper", "challenge", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "protocol_audit.json")
    args = parser.parse_args()
    targets = {"paper": [0.75], "challenge": [0.99], "both": [0.75, 0.99]}[args.target]
    output = {
        "repo_commit": repository_revision(),
        "audit_recorded": datetime.now(timezone.utc).date().isoformat(),
        "seed": args.seed,
        "protocols": {},
    }
    for target in targets:
        frame, family_to_index = load_rows(target)
        protocol = {
            "target": target,
            "n_rows": int(len(frame)),
            "n_circuits": int(frame.file.nunique()),
            "tree_baselines": tree_baselines(frame, args.seed),
            "mlp_ablations": mlp_ablations(frame, family_to_index, args.seed),
            "leave_one_family_out": leave_one_family_out(frame, args.seed),
        }
        if target == 0.99:
            protocol["threshold_conditioning"] = threshold_conditioning(frame, args.seed)
        output["protocols"][str(target)] = protocol
        print(json.dumps({str(target): protocol}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
