"""Static, pre-execution features for the dense-statevector v2 study."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def static_features(spec: Any, precision: str) -> dict[str, Any]:
    """Return circuit/context features without running a simulator.

    ``spec`` is deliberately duck-typed so this module can be shared by the
    v2 runner and the corpus builder while the direct statevector kernel stays
    in v1.  Every quantity here is available before the state is allocated.
    """
    width = int(spec.num_qubits)
    gates = tuple(spec.gates)
    pair_gates = [gate for gate in gates if len(gate.qubits) == 2]
    activity = [0] * width
    pair_degree = [0] * width
    edges: set[tuple[int, int]] = set()
    spans: list[int] = []
    for gate in gates:
        for qubit in gate.qubits:
            activity[qubit] += 1
    for gate in pair_gates:
        first, second = sorted(gate.qubits)
        edges.add((first, second))
        pair_degree[first] += 1
        pair_degree[second] += 1
        spans.append(abs(first - second))

    statevector_bytes = (2**width) * (8 if precision == "complex64" else 16)
    gate_work = max(1, len(gates)) * (2**width)
    canonical_template = {
        "family": spec.family,
        "num_qubits": width,
        "depth_parameter": int(spec.depth_parameter),
        "gates": [
            {"name": gate.name, "qubits": tuple(gate.qubits)}
            for gate in gates
        ],
    }
    template_id = hashlib.sha256(
        json.dumps(canonical_template, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    mean_activity = sum(activity) / width
    mean_degree = sum(pair_degree) / width
    return {
        "circuit_template_id": template_id,
        "statevector_bytes": statevector_bytes,
        "log2_statevector_bytes": float(math.log2(statevector_bytes)),
        "analytical_gate_work": gate_work,
        "log2_analytical_gate_work": float(math.log2(gate_work)),
        "interaction_unique_edge_count": len(edges),
        "interaction_density": (len(edges) / (width * (width - 1) / 2)) if width > 1 else 0.0,
        "two_qubit_span_mean": (sum(spans) / len(spans)) if spans else 0.0,
        "two_qubit_span_max": max(spans) if spans else 0,
        "qubit_activity_mean": mean_activity,
        "qubit_activity_max": max(activity),
        "qubit_activity_std": math.sqrt(sum((value - mean_activity) ** 2 for value in activity) / width),
        "two_qubit_degree_max": max(pair_degree),
        "two_qubit_degree_std": math.sqrt(sum((value - mean_degree) ** 2 for value in pair_degree) / width),
    }
