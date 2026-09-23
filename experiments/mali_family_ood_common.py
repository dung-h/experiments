#!/usr/bin/env python3
"""Shared constants and helpers for the Ma--Li family-OOD audits.

These helpers do not load FakeBackend objects. Logical-circuit features are
computed from QASM; runtime labels are never pooled across sources.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SELECTED_FAMILIES = (
    "qft",
    "qftentangled",
    "qpeexact",
    "qpeinexact",
    "qnn",
    "random",
    "su2random",
    "realamprandom",
    "twolocalrandom",
    "qwalk-noancilla",
)
MISSING_FAMILIES = (
    "ghz",
    "wstate",
    "graphstate",
    "dj",
    "ae",
    "vqe",
    "qaoa",
    "portfoliovqe",
    "portfolioqaoa",
    "grover-noancilla",
    "grover-v-chain",
    "qwalk-v-chain",
)
FAMILY_ALIASES = {
    "qwalk_noancilla": "qwalk-noancilla",
    "grover_noancilla": "grover-noancilla",
    "grover_v_chain": "grover-v-chain",
    "qwalk_v_chain": "qwalk-v-chain",
}
WIDTH_BUCKETS = (
    ("2-16", 2, 16),
    ("17-50", 17, 50),
    ("51-90", 51, 90),
    ("91-127", 91, 127),
)
LOGICAL_FEATURE_KEYS = (
    "logical_width",
    "logical_depth",
    "logical_two_qubit_depth",
    "logical_two_qubit_count",
    "logical_gate_count",
    "logical_parallelism",
    "interaction_n_edges",
    "interaction_density",
    "interaction_mean_degree",
    "interaction_max_degree",
)
PHYSICAL_FEATURE_KEYS = (
    "physical_depth",
    "physical_two_qubit_depth",
    "physical_two_qubit_gate_count",
    "physical_swap_count",
    "physical_gate_count",
    "qcre_weighted_critical_path_seconds",
    "routing_depth_ratio",
    "routing_two_qubit_ratio",
)
TWO_QUBIT_OPS = {"cx", "cz", "ecr", "swap", "rzz", "iswap", "cp", "cu1", "cu", "cs"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filename_family(circuit_name: str) -> str:
    stem = Path(circuit_name).stem
    family = stem.split("_indep_")[0] if "_indep_" in stem else stem.split("_")[0]
    return FAMILY_ALIASES.get(family, family.replace("_", "-"))


def width_from_name(circuit_name: str) -> int:
    return int(Path(circuit_name).stem.split("_")[-1])


def width_bucket(width: int) -> str:
    for label, lo, hi in WIDTH_BUCKETS:
        if lo <= width <= hi:
            return label
    return "other"


def group_of(family: str) -> str:
    if family in SELECTED_FAMILIES:
        return "selected"
    if family in MISSING_FAMILIES:
        return "missing"
    return "other"


def component_families(component: str) -> tuple[str, ...]:
    parts = [FAMILY_ALIASES.get(part, part.replace("_", "-")) for part in component.split("+")]
    return tuple(parts)


def iter_ops(circuit):
    for entry in circuit.data:
        operation = getattr(entry, "operation", None)
        if operation is None:
            operation, qargs, _cargs = entry
        else:
            qargs = entry.qubits
        yield operation, qargs


def qarg_indices(circuit, qargs) -> tuple[int, ...]:
    return tuple(circuit.find_bit(qubit).index for qubit in qargs)


def two_qubit_depth(circuit) -> int:
    clocks = [0] * circuit.num_qubits
    for instruction, qargs in iter_ops(circuit):
        if len(qargs) < 2 or getattr(instruction, "_directive", False):
            continue
        locations = qarg_indices(circuit, qargs)
        start = max((clocks[index] for index in locations), default=0)
        finish = start + 1
        for index in locations:
            clocks[index] = finish
    return max(clocks, default=0)


def logical_features(circuit) -> dict[str, float]:
    counts = circuit.count_ops()
    two_qubit_count = float(
        sum(count for name, count in counts.items() if name in TWO_QUBIT_OPS)
    )
    gate_count = float(len(circuit.data))
    depth = float(circuit.depth() or 0)
    edges: set[frozenset[int]] = set()
    for instruction, qargs in iter_ops(circuit):
        if len(qargs) < 2 or getattr(instruction, "_directive", False):
            continue
        locations = qarg_indices(circuit, qargs)
        if len(locations) >= 2:
            for left in range(len(locations)):
                for right in range(left + 1, len(locations)):
                    edges.add(frozenset((locations[left], locations[right])))
    width = float(circuit.num_qubits)
    n_edges = float(len(edges))
    possible = width * (width - 1) / 2.0 if width > 1 else 0.0
    degrees = [0.0] * int(width)
    for edge in edges:
        for qubit in edge:
            degrees[qubit] += 1.0
    return {
        "logical_width": width,
        "logical_depth": depth,
        "logical_two_qubit_depth": float(two_qubit_depth(circuit)),
        "logical_two_qubit_count": two_qubit_count,
        "logical_gate_count": gate_count,
        "logical_parallelism": (gate_count / depth) if depth else 0.0,
        "interaction_n_edges": n_edges,
        "interaction_density": (n_edges / possible) if possible else 0.0,
        "interaction_mean_degree": float(np.mean(degrees) if degrees else 0.0),
        "interaction_max_degree": float(max(degrees) if degrees else 0.0),
    }


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if len(target) == 0:
        return {
            "n": 0,
            "mae_seconds": float("nan"),
            "medae_seconds": float("nan"),
            "rmse_seconds": float("nan"),
            "r2_seconds": float("nan"),
            "mae_log1p_seconds": float("nan"),
            "r2_log1p_seconds": float("nan"),
        }
    prediction = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    target = np.asarray(target, dtype=float)
    error = prediction - target
    target_log = np.log1p(target)
    prediction_log = np.log1p(prediction)
    seconds_variance = float(np.sum((target - target.mean()) ** 2))
    log_variance = float(np.sum((target_log - target_log.mean()) ** 2))
    return {
        "n": int(len(target)),
        "mae_seconds": float(np.mean(np.abs(error))),
        "medae_seconds": float(np.median(np.abs(error))),
        "rmse_seconds": float(np.sqrt(np.mean(error**2))),
        "r2_seconds": float(1.0 - np.sum(error**2) / seconds_variance) if seconds_variance else float("nan"),
        "mae_log1p_seconds": float(np.mean(np.abs(prediction_log - target_log))),
        "r2_log1p_seconds": float(1.0 - np.sum((prediction_log - target_log) ** 2) / log_variance)
        if log_variance else float("nan"),
    }


def ridge_log1p(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Ridge on log1p(y) with intercept unpenalized. Column 0 is intercept."""
    train = np.array(x_train, dtype=float, copy=True)
    test = np.array(x_test, dtype=float, copy=True)
    if train.shape[1] > 1:
        mean = train[:, 1:].mean(axis=0)
        scale = train[:, 1:].std(axis=0)
        scale[scale == 0] = 1.0
        train[:, 1:] = (train[:, 1:] - mean) / scale
        test[:, 1:] = (test[:, 1:] - mean) / scale
    penalty = np.eye(train.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        train.T @ train + penalty,
        train.T @ np.log1p(np.maximum(y_train, 0.0)),
    )
    return np.expm1(np.maximum(test @ coefficients, -0.999999))


def affine_map(source: np.ndarray, target: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Fit target_log ≈ a + b * source_log on train, apply to query."""
    source_log = np.log1p(np.maximum(source, 0.0))
    query_log = np.log1p(np.maximum(query, 0.0))
    design = np.column_stack((np.ones(len(source_log)), source_log))
    penalty = np.eye(2, dtype=float) * 1e-8
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ np.log1p(np.maximum(target, 0.0)),
    )
    return np.expm1(np.maximum(np.column_stack((np.ones(len(query_log)), query_log)) @ coefficients, -0.999999))


def matrix_from_rows(
    rows: list[dict[str, object]],
    keys: tuple[str, ...],
    backend_key: str | None = None,
) -> np.ndarray:
    numeric = np.asarray(
        [[float(row[key]) for key in keys] for row in rows],
        dtype=float,
    )
    numeric = np.log1p(np.maximum(numeric, 0.0))
    intercept = np.ones((len(rows), 1), dtype=float)
    if backend_key is None:
        return np.hstack((intercept, numeric))
    backend = np.asarray(
        [[1.0 if str(row.get(backend_key, "")) in {"osaka", "ibm_osaka"} else 0.0] for row in rows],
        dtype=float,
    )
    return np.hstack((intercept, numeric, backend))


def ks_median(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    from scipy.stats import ks_2samp

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) == 0 or len(right) == 0:
        return {
            "n_left": float(len(left)),
            "n_right": float(len(right)),
            "median_left": float("nan"),
            "median_right": float("nan"),
            "median_ratio_right_over_left": float("nan"),
            "ks_statistic": float("nan"),
            "ks_pvalue": float("nan"),
        }
    result = ks_2samp(left, right, alternative="two-sided", mode="auto")
    median_left = float(np.median(left))
    median_right = float(np.median(right))
    return {
        "n_left": float(len(left)),
        "n_right": float(len(right)),
        "median_left": median_left,
        "median_right": median_right,
        "median_ratio_right_over_left": (
            float(median_right / median_left) if median_left not in {0.0, -0.0} else float("nan")
        ),
        "ks_statistic": float(result.statistic),
        "ks_pvalue": float(result.pvalue),
    }


def standardized_centroid_distance(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    pooled = np.vstack((left, right)) if len(left) and len(right) else pooled_empty(left, right)
    if pooled.size == 0:
        return {"centroid_euclidean": float("nan"), "n_left": float(len(left)), "n_right": float(len(right))}
    mean = pooled.mean(axis=0)
    scale = pooled.std(axis=0)
    scale[scale == 0] = 1.0
    left_z = (left - mean) / scale if len(left) else left
    right_z = (right - mean) / scale if len(right) else right
    left_c = left_z.mean(axis=0) if len(left) else np.zeros(pooled.shape[1])
    right_c = right_z.mean(axis=0) if len(right) else np.zeros(pooled.shape[1])
    return {
        "centroid_euclidean": float(np.linalg.norm(left_c - right_c)),
        "n_left": float(len(left)),
        "n_right": float(len(right)),
    }


def pooled_empty(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left):
        return left
    if len(right):
        return right
    return np.zeros((0, 1), dtype=float)


def nearest_centroid_group_accuracy(
    features: np.ndarray,
    families: list[str],
) -> dict[str, object]:
    """Leave-one-family-out nearest selected/missing centroid on family means."""
    unique = sorted(set(families))
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    z = (features - mean) / scale
    family_centroid = {}
    family_group = {}
    for family in unique:
        mask = np.asarray([item == family for item in families])
        family_centroid[family] = z[mask].mean(axis=0)
        family_group[family] = group_of(family)
    rows = []
    correct = 0
    for family in unique:
        others_selected = [item for item in unique if item != family and family_group[item] == "selected"]
        others_missing = [item for item in unique if item != family and family_group[item] == "missing"]
        if not others_selected or not others_missing:
            continue
        selected_c = np.mean([family_centroid[item] for item in others_selected], axis=0)
        missing_c = np.mean([family_centroid[item] for item in others_missing], axis=0)
        held = family_centroid[family]
        d_selected = float(np.linalg.norm(held - selected_c))
        d_missing = float(np.linalg.norm(held - missing_c))
        predicted = "selected" if d_selected <= d_missing else "missing"
        true = family_group[family]
        ok = predicted == true
        correct += int(ok)
        rows.append(
            {
                "family": family,
                "true_group": true,
                "predicted_group": predicted,
                "distance_to_selected": d_selected,
                "distance_to_missing": d_missing,
                "correct": ok,
            }
        )
    return {
        "n_families": len(rows),
        "accuracy": (correct / len(rows)) if rows else float("nan"),
        "rows": rows,
    }
