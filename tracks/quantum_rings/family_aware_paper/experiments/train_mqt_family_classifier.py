"""Train and persist the public MQT Bench family classifier.

This follows the paper's stated setup more closely than the fold-local
surrogate: the classifier is trained on a separate 200-circuit MQT set and is
then frozen before it is used by the evaluation predictor.  The model is still
an approximation because the paper's exact 200 circuits and classifier weights
are not public.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import os

import joblib
import numpy as np
from qiskit import QuantumCircuit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ["QUANTUM_RINGS_ROOT"]).resolve() if os.environ.get("QUANTUM_RINGS_ROOT") else _SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.replicate_paper import extract_features, _scaled_features


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "mqt_family_pretraining",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "mqt_family_classifier",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    features = []
    labels = []
    files = []
    for row in manifest["circuits"]:
        path = args.data_dir / "circuits" / row["file"]
        circuit = QuantumCircuit.from_qasm_file(path)
        features.append(extract_features(circuit, "CPU", "single")[:28])
        labels.append(row["family"])
        files.append(row["file"])
    x = np.asarray(features, dtype=float)
    x = _scaled_features(x)
    classes = sorted(set(labels))
    class_to_index = {name: index for index, name in enumerate(classes)}
    y = np.asarray([class_to_index[name] for name in labels], dtype=int)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )
    scaler = StandardScaler().fit(x[train_idx])
    x_train = scaler.transform(x[train_idx])
    x_test = scaler.transform(x[test_idx])
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=32,
        learning_rate_init=5e-3,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=50,
        random_state=args.seed,
    )
    mlp.fit(x_train, y[train_idx])
    rf = RandomForestClassifier(n_estimators=300, random_state=args.seed, n_jobs=1)
    rf.fit(x_train, y[train_idx])
    mlp_pred = mlp.predict(x_test).astype(int)
    rf_pred = rf.predict(x_test).astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": mlp,
            "scaler": scaler,
            "feature_count": 28,
            "feature_transform": "log1p count-like features then StandardScaler",
            "families": sorted(set(labels)),
            "manifest_sha256": sha256_file(manifest_path),
            "training_seed": args.seed,
        },
        args.output_dir / "mqt_family_classifier.joblib",
    )
    result = {
        "recorded": datetime.now(timezone.utc).date().isoformat(),
        "data_manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "n_rows": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_families": int(len(set(labels))),
        "features": 28,
        "split": "stratified circuit-level 80/20; one row per generated QASM",
        "mlp": {
            "accuracy": float(accuracy_score(y[test_idx], mlp_pred)),
            "n_iter": int(mlp.n_iter_),
            "confusion_matrix": confusion_matrix(y[test_idx], mlp_pred, labels=mlp.classes_).tolist(),
            "classes": classes,
        },
        "random_forest_reference": {
            "accuracy": float(accuracy_score(y[test_idx], rf_pred)),
        },
        "paper_boundary": "public MQT approximation; not the authors' private 200-circuit set",
    }
    (args.output_dir / "mqt_family_classifier.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "heldout_predictions.csv").write_text(
        "file,true_family,mlp_prediction,rf_prediction\n"
        + "\n".join(
            f"{files[index]},{classes[y[index]]},{classes[mlp_pred[position]]},{classes[rf_pred[position]]}"
            for position, index in enumerate(test_idx)
        )
        + "\n"
    )
    print(json.dumps({k: result[k] for k in ("recorded", "n_rows", "n_train", "n_test", "n_families", "mlp", "random_forest_reference")}, indent=2))


if __name__ == "__main__":
    main()
