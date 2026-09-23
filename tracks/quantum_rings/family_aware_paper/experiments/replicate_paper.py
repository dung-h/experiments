"""Reproduce the paper-style modeling experiment from the public dataset.

The challenge repository contains labels and circuits, but not the authors'
training code or their private ~200-circuit family-classifier pretraining set.
This script therefore implements a transparent, paper-aligned reproduction:

* 32 structural/context features (including graph and fingerprint features),
* circuit-level 5-fold cross-validation (all four contexts stay together),
* a family-conditioned residual/FiLM MLP,
* a no-family MLP and Random-Forest baselines, and
* both the paper target (mirror fidelity >= 0.75) and the challenge target
  (mirror fidelity >= 0.99).

For the paper target, the public forward run was selected using 0.99 rather
than 0.75.  The script consequently uses the mirror-sweep wall time at the
0.75 crossing as a clearly marked runtime proxy; it does not claim this is a
true 10,000-shot forward label at the paper-selected threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
import torch
from qiskit import QuantumCircuit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch import nn


_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ["QUANTUM_RINGS_ROOT"]).resolve() if os.environ.get("QUANTUM_RINGS_ROOT") else _SCRIPT_DIR.parents[1]
LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CLASS_NAMES = ("star",) + tuple(str(x) for x in LADDER)
FAMILY_UNKNOWN = "UNKNOWN"


def repository_revision() -> str:
    """Return the checked-out source revision without making git mandatory."""
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


def _entropy(values: Iterable[int]) -> float:
    counts = np.bincount(np.asarray(list(values), dtype=int))
    counts = counts[counts > 0].astype(float)
    if not len(counts):
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _instruction_parts(instruction):
    """Return operation and qubits across Qiskit 1.x/2.x instruction APIs."""
    operation = getattr(instruction, "operation", None)
    qubits = getattr(instruction, "qubits", None)
    if operation is None:
        operation = instruction[0]
        qubits = instruction[1]
    return operation, list(qubits)


def extract_features(circuit: QuantumCircuit, backend: str, precision: str) -> np.ndarray:
    """Extract the 32 feature groups described in Section III-B of the paper."""
    gate_names = ("h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "cx", "cz", "swap")
    records = []
    for instruction in circuit.data:
        operation, qubits = _instruction_parts(instruction)
        if getattr(operation, "name", "") not in {"barrier", "measure"}:
            records.append((operation, qubits))
    operations = [operation for operation, _ in records]
    counts = {name: 0 for name in gate_names}
    for operation in operations:
        if operation.name in counts:
            counts[operation.name] += 1

    graph = nx.Graph()
    graph.add_nodes_from(range(circuit.num_qubits))
    spans: list[int] = []
    for operation, operation_qubits in records:
        qubits = [circuit.find_bit(q).index for q in operation_qubits]
        if len(qubits) >= 2:
            for left, right in zip(qubits, qubits[1:]):
                if left != right:
                    graph.add_edge(left, right)

    edge_count = graph.number_of_edges()
    two_qubit_count = sum(1 for _, operation_qubits in records if len(operation_qubits) == 2)
    total_gate_count = len(operations)
    width = circuit.num_qubits
    depth = circuit.depth()
    graph_density = nx.density(graph) if width > 1 else 0.0
    degrees = [degree for _, degree in graph.degree()]

    # RCM and cut-pressure proxies.  They are intentionally defined without
    # simulator internals so they can be evaluated on hidden QASM as well.
    try:
        ordering = list(nx.utils.reverse_cuthill_mckee_ordering(graph))
    except Exception:
        ordering = list(range(width))
    order_index = {node: index for index, node in enumerate(ordering)}
    cut_counts = []
    for boundary in range(1, max(1, width)):
        left = set(ordering[:boundary])
        cut_counts.append(sum((u in left) != (v in left) for u, v in graph.edges()))
    for left, right in graph.edges():
        spans.append(abs(order_index[left] - order_index[right]))

    # Heuristic fingerprints from gate composition.  The scores are bounded
    # and deliberately simple, matching the small-data motivation in the
    # paper rather than pretending to be a learned circuit recognizer.
    total = max(1, total_gate_count)
    text_gate_names = {operation.name for operation in operations}
    arithmetic_count = sum(
        1 for operation in operations if operation.name in {"ccx", "mcx", "cswap"}
    )
    t_count = counts["t"] + sum(1 for operation in operations if operation.name in {"tdg", "u1"})
    arithmetic_score = min(1.0, (arithmetic_count / total) * (1.0 + min(1.0, t_count / total)))
    controlled_phase = sum(
        1 for operation in operations if operation.name in {"cp", "cu1", "mcp"}
    )
    qft_score = min(1.0, (controlled_phase / total) * 4.0 + (counts["h"] / total) * 0.5)
    parameterized = sum(
        1
        for operation in operations
        if operation.name in {"rx", "ry", "rz", "u", "u1", "u2", "u3"}
        and getattr(operation, "params", [])
    )
    qnn_score = min(1.0, parameterized / total * 4.0)

    basic = [depth, width, total_gate_count]
    gate_vector = [counts[name] for name in gate_names]
    complexity = [
        two_qubit_count / total,
        graph_density,
        depth / max(1, width),
    ]
    local = [max(degrees, default=0), _entropy(degrees)]
    fingerprints = [arithmetic_score, qft_score, qnn_score]
    graph_features = [
        float(nx.average_clustering(graph)) if width else 0.0,
        float(nx.number_connected_components(graph)) if width else 0.0,
        float(max(cut_counts, default=0)),
        float(np.mean(cut_counts) if cut_counts else 0.0),
        float(max(spans, default=0)),
    ]
    context = [
        float(backend == "CPU"),
        float(backend == "GPU"),
        float(precision == "single"),
        float(precision == "double"),
    ]
    features = np.asarray(basic + gate_vector + complexity + local + fingerprints + graph_features + context, dtype=float)
    if features.shape != (32,):
        raise AssertionError(f"Expected 32 features, got {features.shape}")
    return features


def _scaled_features(features: np.ndarray) -> np.ndarray:
    """Apply the paper's log transform to unbounded/count-like features."""
    transformed = features.copy()
    # basic stats, gate counts, max degree, component/cut/span counts
    for index in list(range(15)) + [18, 24, 25, 26, 27]:
        transformed[:, index] = np.log1p(np.maximum(0.0, transformed[:, index]))
    return transformed


def derive_threshold(result: dict, target: float) -> int:
    for sweep in sorted(result.get("threshold_sweep", []), key=lambda item: item["threshold"]):
        fidelity = sweep.get("sdk_get_fidelity")
        if (
            sweep.get("threshold") in LADDER
            and fidelity is not None
            and sweep.get("returncode", 0) == 0
            and float(fidelity) >= target
        ):
            return CLASS_NAMES.index(str(sweep["threshold"]))
    return 0  # star/no threshold reached


def runtime_target(result: dict, target: float) -> tuple[float, str]:
    """Return runtime and provenance for challenge or paper protocol."""
    if target == 0.99:
        return float(result["forward"]["run_wall_s"]), "forward_at_public_0.99_selection"
    # The public forward run is selected at 0.99.  Use the sweep runtime at
    # the first 0.75 crossing as a proxy, explicitly recording the limitation.
    for sweep in sorted(result.get("threshold_sweep", []), key=lambda item: item["threshold"]):
        fidelity = sweep.get("sdk_get_fidelity")
        if (
            sweep.get("threshold") in LADDER
            and fidelity is not None
            and sweep.get("returncode", 0) == 0
            and float(fidelity) >= target
        ):
            return float(sweep["run_wall_s"]), "mirror_sweep_at_0.75_crossing_proxy"
    return float(result["forward"]["run_wall_s"]), "forward_fallback_no_0.75_crossing"


def load_dataset(target: float) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    data = json.loads((ROOT / "data" / "hackathon_public.json").read_text())
    circuit_meta = {item["file"]: item for item in data["circuits"]}
    feature_cache: dict[str, np.ndarray] = {}
    rows = []
    for result in data["results"]:
        file_name = result["file"]
        if file_name not in feature_cache:
            circuit = QuantumCircuit.from_qasm_file(ROOT / "circuits" / file_name)
            feature_cache[file_name] = extract_features(circuit, "CPU", "single")[:28]
        base = feature_cache[file_name]
        context = extract_features(
            QuantumCircuit.from_qasm_file(ROOT / "circuits" / file_name),
            result["backend"],
            result["precision"],
        )[28:]
        runtime, runtime_source = runtime_target(result, target)
        rows.append(
            {
                "file": file_name,
                "family": circuit_meta[file_name]["family"],
                "backend": result["backend"],
                "precision": result["precision"],
                "threshold_class": derive_threshold(result, target),
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
    return frame, family_to_index, families


class FamilyAwareNet(nn.Module):
    def __init__(self, n_features: int, n_families: int, n_classes: int, use_family: bool = True):
        super().__init__()
        self.use_family = use_family
        self.backbone = nn.Sequential(
            nn.Linear(n_features, 128), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.SiLU(), nn.Dropout(0.2),
        )
        self.family_embedding = nn.Embedding(n_families, 64)
        self.family_mlp = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.film_gamma = nn.Linear(64, 64)
        self.film_beta = nn.Linear(64, 64)
        self.threshold_head = nn.Linear(64, n_classes)
        self.runtime_head = nn.Linear(64, 1)
        self.threshold_shortcut = nn.Linear(n_features, n_classes)
        self.runtime_shortcut = nn.Linear(n_features, 1)
        self.family_threshold_residual = nn.Linear(64, n_classes)
        self.family_runtime_residual = nn.Linear(64, 1)
        nn.init.zeros_(self.family_threshold_residual.weight)
        nn.init.zeros_(self.family_threshold_residual.bias)
        nn.init.zeros_(self.family_runtime_residual.weight)
        nn.init.zeros_(self.family_runtime_residual.bias)

    def forward(self, features: torch.Tensor, families: torch.Tensor):
        backbone = self.backbone(features)
        if self.use_family:
            family = self.family_mlp(self.family_embedding(families))
            hidden = backbone * (1.0 + self.film_gamma(family)) + self.film_beta(family)
            threshold = self.threshold_head(hidden) + self.threshold_shortcut(features) + self.family_threshold_residual(family)
            runtime = self.runtime_head(hidden) + self.runtime_shortcut(features) + self.family_runtime_residual(family)
        else:
            threshold = self.threshold_head(backbone) + self.threshold_shortcut(features)
            runtime = self.runtime_head(backbone) + self.runtime_shortcut(features)
        # The public set is intentionally tiny and includes hour-scale
        # outliers. Bound the normalized log prediction to prevent a single
        # extrapolating shortcut from dominating a held-out fold.
        return threshold, torch.clamp(runtime.squeeze(-1), -4.0, 4.0)


def _decayed_threshold_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    true_probability = probabilities[torch.arange(len(labels)), labels]
    loss = -torch.log(true_probability.clamp_min(1e-8))
    for distance in range(1, logits.shape[1]):
        higher = torch.clamp(labels + distance, max=logits.shape[1] - 1)
        active = (labels != 0) & (labels + distance < logits.shape[1])
        loss = loss + active * (0.5**distance) * probabilities[torch.arange(len(labels)), higher]
    return loss.mean()


def train_network(
    x_train,
    y_threshold,
    y_runtime,
    family_train,
    n_families: int,
    use_family: bool,
    seed: int,
) -> FamilyAwareNet:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = FamilyAwareNet(x_train.shape[1], n_families, len(CLASS_NAMES), use_family)
    # Full-batch training is used because each fold has only ~115 rows.  A
    # slightly conservative learning rate keeps the direct runtime shortcut
    # stable on the small public set while remaining close to the paper's
    # AdamW setup.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160)
    model.train()
    for _ in range(160):
        optimizer.zero_grad()
        threshold_logits, runtime_log = model(x_train, family_train)
        loss = _decayed_threshold_loss(threshold_logits, y_threshold) + torch.mean((runtime_log - y_runtime) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return model.eval()


def _threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "exact_accuracy": float(accuracy_score(y_true, y_pred)),
        "within_one_rung": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "mean_rung_distance": float(np.mean(np.abs(y_true - y_pred))),
    }


def _runtime_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    true_seconds = np.exp(y_true_log)
    # Defensive clipping keeps an unstable fold from turning the aggregate
    # report into inf if a tiny training split produces an extreme log output.
    pred_seconds = np.exp(np.clip(y_pred_log, -20.0, 20.0))
    return {
        "r2_log_runtime": float(r2_score(y_true_log, y_pred_log)),
        "r2_runtime_seconds": float(r2_score(true_seconds, pred_seconds)),
        "mae_runtime_seconds": float(mean_absolute_error(true_seconds, pred_seconds)),
        "median_relative_error": float(np.median(np.abs(pred_seconds - true_seconds) / np.maximum(true_seconds, 1e-12))),
    }


def run_cv(frame: pd.DataFrame, family_to_index: dict[str, int], target: float, seed: int) -> dict[str, object]:
    feature_columns = [f"f{index:02d}" for index in range(32)]
    files = np.asarray(sorted(frame["file"].unique()))
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    all_predictions = {"rf": [], "mlp_no_family": [], "family_predicted": [], "family_oracle": []}
    all_truth = []
    family_truth = []
    family_predicted = []
    for fold, (train_file_idx, test_file_idx) in enumerate(folds.split(files)):
        train_files, test_files = files[train_file_idx], files[test_file_idx]
        train = frame[frame.file.isin(train_files)].copy()
        test = frame[frame.file.isin(test_files)].copy()
        x_train = _scaled_features(train[feature_columns].to_numpy())
        x_test = _scaled_features(test[feature_columns].to_numpy())
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        y_train_threshold = train.threshold_class.to_numpy(dtype=int)
        y_test_threshold = test.threshold_class.to_numpy(dtype=int)
        y_train_runtime = np.log(np.maximum(train.runtime_s.to_numpy(dtype=float), 1e-9))
        y_test_runtime = np.log(np.maximum(test.runtime_s.to_numpy(dtype=float), 1e-9))
        runtime_center = float(y_train_runtime.mean())
        runtime_scale = float(y_train_runtime.std()) or 1.0
        y_train_runtime_normalized = (y_train_runtime - runtime_center) / runtime_scale
        family_train = np.asarray([family_to_index[x] for x in train.family], dtype=int)
        family_test_oracle = np.asarray([family_to_index[x] for x in test.family], dtype=int)

        # A lightweight family classifier stands in for the unavailable private
        # 200-circuit pretraining set. It is fit only on the training files.
        train_unique = train.drop_duplicates("file")
        test_unique = test.drop_duplicates("file")
        family_classifier = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The number of unique classes is greater than 50%",
            )
            family_classifier.fit(
                scaler.transform(_scaled_features(train_unique[feature_columns].to_numpy()))[:, :28],
                train_unique.family,
            )
            family_test_names = family_classifier.predict(
                scaler.transform(_scaled_features(test_unique[feature_columns].to_numpy()))[:, :28]
            )
        family_by_file = dict(zip(test_unique.file, family_test_names))
        family_test_predicted = np.asarray([family_to_index[family_by_file[file_name]] for file_name in test.file], dtype=int)
        family_truth.extend(test_unique.family.tolist())
        family_predicted.extend(family_test_names.tolist())

        tx = torch.tensor(x_train, dtype=torch.float32)
        ty_threshold = torch.tensor(y_train_threshold, dtype=torch.long)
        ty_runtime = torch.tensor(y_train_runtime_normalized, dtype=torch.float32)
        tfamily = torch.tensor(family_train, dtype=torch.long)
        vx = torch.tensor(x_test, dtype=torch.float32)
        vfamily_predicted = torch.tensor(family_test_predicted, dtype=torch.long)
        vfamily_oracle = torch.tensor(family_test_oracle, dtype=torch.long)

        rf_classifier = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
        rf_classifier.fit(x_train, y_train_threshold)
        rf_regressor = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1)
        rf_regressor.fit(x_train, y_train_runtime)
        all_predictions["rf"].append((rf_classifier.predict(x_test), rf_regressor.predict(x_test)))

        no_family = train_network(
            tx, ty_threshold, ty_runtime, tfamily, len(family_to_index), False, seed + fold
        )
        with torch.no_grad():
            pred_threshold, pred_runtime = no_family(vx, vfamily_predicted)
        all_predictions["mlp_no_family"].append(
            (pred_threshold.argmax(1).numpy(), pred_runtime.numpy() * runtime_scale + runtime_center)
        )

        family_predicted_model = train_network(
            tx, ty_threshold, ty_runtime, tfamily, len(family_to_index), True, seed + 100 + fold
        )
        with torch.no_grad():
            pred_threshold, pred_runtime = family_predicted_model(vx, vfamily_predicted)
        all_predictions["family_predicted"].append(
            (pred_threshold.argmax(1).numpy(), pred_runtime.numpy() * runtime_scale + runtime_center)
        )

        family_oracle_model = train_network(
            tx, ty_threshold, ty_runtime, tfamily, len(family_to_index), True, seed + 200 + fold
        )
        with torch.no_grad():
            pred_threshold, pred_runtime = family_oracle_model(vx, vfamily_oracle)
        all_predictions["family_oracle"].append(
            (pred_threshold.argmax(1).numpy(), pred_runtime.numpy() * runtime_scale + runtime_center)
        )
        all_truth.append((y_test_threshold, y_test_runtime))

    y_threshold = np.concatenate([item[0] for item in all_truth])
    y_runtime = np.concatenate([item[1] for item in all_truth])
    result = {
        "target": target,
        "n_rows": int(len(frame)),
        "n_circuits": int(frame.file.nunique()),
        "family_classifier": {"accuracy": float(accuracy_score(family_truth, family_predicted)), "n_circuits": len(family_truth)},
        "models": {},
    }
    for name, predictions in all_predictions.items():
        threshold = np.concatenate([item[0] for item in predictions])
        runtime = np.concatenate([item[1] for item in predictions])
        result["models"][name] = {
            "threshold": _threshold_metrics(y_threshold, threshold),
            "runtime": _runtime_metrics(y_runtime, runtime),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("paper", "challenge", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "paper_replication.json")
    args = parser.parse_args()
    targets = {"paper": [0.75], "challenge": [0.99], "both": [0.75, 0.99]}[args.target]
    output = {
        "repo_commit": repository_revision(),
        "paper": "arXiv:2606.11620",
        "replication_recorded": datetime.now(timezone.utc).date().isoformat(),
        "protocols": {},
    }
    for target in targets:
        frame, family_to_index, families = load_dataset(target)
        cv = run_cv(frame, family_to_index, target, args.seed)
        cv["families"] = families
        cv["runtime_sources"] = frame.runtime_source.value_counts().to_dict()
        output["protocols"][str(target)] = cv
        print(json.dumps({str(target): cv}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
