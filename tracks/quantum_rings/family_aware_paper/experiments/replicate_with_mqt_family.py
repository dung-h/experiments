"""Re-run the performance predictor with a public MQT-pretrained classifier.

The classifier is trained independently by ``train_mqt_family_classifier.py``
and is never updated on the 36 Quantum Rings evaluation circuits.  This is the
closest public approximation to the paper's frozen ~200-circuit family
classifier.  It is not claimed to be the authors' private classifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ["QUANTUM_RINGS_ROOT"]).resolve() if os.environ.get("QUANTUM_RINGS_ROOT") else _SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.replicate_paper import (  # noqa: E402
    CLASS_NAMES,
    FAMILY_UNKNOWN,
    FamilyAwareNet,
    _decayed_threshold_loss,
    _runtime_metrics,
    _scaled_features,
    _threshold_metrics,
    load_dataset,
)


FEATURE_COLUMNS = [f"f{index:02d}" for index in range(32)]
MISSING_FAMILY = "UNKNOWN"


def evaluation_family(name: str) -> str:
    """Map repository labels to the paper classifier's public 10-class vocabulary."""
    mapping = {
        "Deutsch_Jozsa": "DJ",
        "GHZ": "GHZ",
        "W_State": "W-State",
        "GraphState": "Graph State",
        "Grover_NoAncilla": "Grover",
        "Grover_V_Chain": "Grover",
        "QFT": "QFT",
        "QFT_Entangled": "QFT Entangled",
        "QPE_Exact": "QPE",
        "QNN": "QNN",
        "VQE": "VQE",
        "Portfolio_VQE": "VQE",
    }
    return mapping.get(name, MISSING_FAMILY)


def load_classifier(path: Path):
    artifact = joblib.load(path)
    return artifact["model"], artifact["scaler"], artifact["families"]


def predict_families(frame: pd.DataFrame, model, scaler, classes: list[str]) -> dict[str, str]:
    unique = frame.drop_duplicates("file").copy()
    raw = _scaled_features(unique[FEATURE_COLUMNS].to_numpy(dtype=float))[:, :28]
    predicted = model.predict(scaler.transform(raw)).astype(int)
    names = [classes[index] for index in predicted]
    return dict(zip(unique.file, names))


def train_network(
    x_train: np.ndarray,
    y_threshold: np.ndarray,
    y_runtime: np.ndarray,
    family_train: np.ndarray,
    n_families: int,
    seed: int,
) -> FamilyAwareNet:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = FamilyAwareNet(x_train.shape[1], n_families, len(CLASS_NAMES), use_family=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160)
    features = torch.tensor(x_train, dtype=torch.float32)
    thresholds = torch.tensor(y_threshold, dtype=torch.long)
    runtimes = torch.tensor(y_runtime, dtype=torch.float32)
    families = torch.tensor(family_train, dtype=torch.long)
    model.train()
    for _ in range(160):
        optimizer.zero_grad()
        threshold_logits, runtime = model(features, families)
        # Match the paper replication's rung-aware objective: a near miss is
        # penalized less than a prediction several threshold bins away.
        loss = _decayed_threshold_loss(threshold_logits, thresholds)
        loss = loss + torch.mean((runtime - runtimes) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return model.eval()


def run_cv(frame: pd.DataFrame, family_index: dict[str, int], predicted_names: dict[str, str], oracle_names: dict[str, str], seed: int):
    files = np.asarray(sorted(frame.file.unique()))
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    predictions = {"mqt_predicted_family": [], "mqt_oracle_family": []}
    truths = []
    for fold, (train_idx, test_idx) in enumerate(folds.split(files)):
        train_files, test_files = files[train_idx], files[test_idx]
        train = frame[frame.file.isin(train_files)]
        test = frame[frame.file.isin(test_files)]
        raw_train = _scaled_features(train[FEATURE_COLUMNS].to_numpy(dtype=float))
        raw_test = _scaled_features(test[FEATURE_COLUMNS].to_numpy(dtype=float))
        scaler = StandardScaler().fit(raw_train)
        x_train, x_test = scaler.transform(raw_train), scaler.transform(raw_test)
        y_threshold_train = train.threshold_class.to_numpy(dtype=int)
        y_threshold_test = test.threshold_class.to_numpy(dtype=int)
        y_runtime_train = np.log(np.maximum(train.runtime_s.to_numpy(dtype=float), 1e-9))
        y_runtime_test = np.log(np.maximum(test.runtime_s.to_numpy(dtype=float), 1e-9))
        center = float(y_runtime_train.mean())
        scale = float(y_runtime_train.std()) or 1.0
        y_runtime_train = (y_runtime_train - center) / scale
        predicted_train = np.asarray([family_index[predicted_names[file_name]] for file_name in train.file])
        predicted_test = np.asarray([family_index[predicted_names[file_name]] for file_name in test.file])
        oracle_train = np.asarray([family_index[oracle_names[file_name]] for file_name in train.file])
        oracle_test = np.asarray([family_index[oracle_names[file_name]] for file_name in test.file])
        truths.append((y_threshold_test, y_runtime_test))
        for label, train_family, test_family, seed_offset in (
            ("mqt_predicted_family", predicted_train, predicted_test, 0),
            ("mqt_oracle_family", oracle_train, oracle_test, 200),
        ):
            model = train_network(
                x_train,
                y_threshold_train,
                y_runtime_train,
                train_family,
                len(family_index),
                seed + seed_offset + fold,
            )
            with torch.no_grad():
                logits, runtime = model(
                    torch.tensor(x_test, dtype=torch.float32),
                    torch.tensor(test_family, dtype=torch.long),
                )
            predictions[label].append(
                (logits.argmax(1).numpy(), runtime.numpy() * scale + center)
            )
    y_threshold = np.concatenate([item[0] for item in truths])
    y_runtime = np.concatenate([item[1] for item in truths])
    result = {}
    for label, fold_predictions in predictions.items():
        pred_threshold = np.concatenate([item[0] for item in fold_predictions])
        pred_runtime = np.concatenate([item[1] for item in fold_predictions])
        result[label] = {
            "threshold": _threshold_metrics(y_threshold, pred_threshold),
            "runtime": _runtime_metrics(y_runtime, pred_runtime),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=Path, default=ROOT / "results" / "mqt_family_classifier" / "mqt_family_classifier.joblib")
    parser.add_argument("--target", choices=("paper", "challenge", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "mqt_pretrained_family_replication.json")
    args = parser.parse_args()
    classifier_path = args.classifier.resolve()
    model, scaler, classes = load_classifier(classifier_path)
    targets = {"paper": [0.75], "challenge": [0.99], "both": [0.75, 0.99]}[args.target]
    output = {
        "recorded": datetime.now(timezone.utc).date().isoformat(),
        "classifier": str(classifier_path.relative_to(ROOT)),
        "classifier_classes": classes,
        "family_vocab": classes + [MISSING_FAMILY],
        "protocols": {},
    }
    for target in targets:
        frame, _, _ = load_dataset(target)
        predicted_by_file = predict_families(frame, model, scaler, classes)
        true_by_file = {
            file_name: evaluation_family(family)
            for file_name, family in zip(frame.file, frame.family)
        }
        true_by_file = dict(true_by_file)
        family_index = {name: index for index, name in enumerate(classes + [MISSING_FAMILY])}
        mapped_true = [true_by_file[file_name] for file_name in sorted(true_by_file)]
        predicted_eval = [predicted_by_file[file_name] for file_name in sorted(true_by_file)]
        known = [name != MISSING_FAMILY for name in mapped_true]
        output["protocols"][str(target)] = {
            "n_rows": int(len(frame)),
            "n_circuits": int(frame.file.nunique()),
            "evaluation_family_accuracy_all": float(accuracy_score(mapped_true, predicted_eval)),
            "evaluation_family_accuracy_known_only": float(
                accuracy_score(np.asarray(mapped_true)[known], np.asarray(predicted_eval)[known])
            )
            if any(known)
            else None,
            "known_family_coverage": float(np.mean(known)),
            "models": run_cv(frame, family_index, predicted_by_file, true_by_file, args.seed),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
