#!/usr/bin/env python3
"""Generate a target-aware CUDA-Q simulator runtime matrix.

This script is intentionally small and explicit about timing semantics.  It
does not turn CUDA-Q's compilation or simulation time into a QPU/cloud label.
The first call and the warmed calls are recorded as separate targets because
CUDA-Q may lazily compile a kernel and initialise the backend on first use.

Run this file with the isolated CUDA-Q environment (CUDA-Q 0.15.1 in the
current capsule), for example::

    /path/to/.venv-cudaq313/bin/python experiments/cudaq_runtime/run_cudaq_matrix.py \
        --widths 16,20,24 --families ghz,hea,qaoa_cycle,random_brickwork \
        --repeats 3 --output artifacts/cudaq_runtime/cudaq_matrix.csv

The output is a CSV plus a JSON manifest.  Rows from different targets are
not intended to be pooled without keeping ``simulator_target`` and
``runtime_semantics`` as columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cudaq


TARGETS = {
    "cpu_fp64": ("qpp-cpu", None, "qpp-cpu", "fp64"),
    "gpu_fp32": ("nvidia", "fp32", "nvidia", "fp32"),
    "gpu_fp64": ("nvidia", "fp64", "nvidia", "fp64"),
}

FAMILY_NAMES = ("ghz", "hea", "qaoa_cycle", "random_brickwork")


@cudaq.kernel
def ghz_kernel(num_qubits: int):
    q = cudaq.qvector(num_qubits)
    h(q[0])
    for i in range(num_qubits - 1):
        x.ctrl(q[i], q[i + 1])


@cudaq.kernel
def hea_kernel(num_qubits: int):
    q = cudaq.qvector(num_qubits)
    for layer in range(3):
        for i in range(num_qubits):
            h(q[i])
            rz(0.17 + 0.03 * layer, q[i])
            rx(0.31 + 0.02 * layer, q[i])
        for i in range(num_qubits - 1):
            x.ctrl(q[i], q[i + 1])


@cudaq.kernel
def qaoa_cycle_kernel(num_qubits: int):
    q = cudaq.qvector(num_qubits)
    for i in range(num_qubits):
        h(q[i])
    # One MaxCut-like ring layer.  The interaction is deliberately fixed so
    # that structural features, rather than random parameter draws, explain
    # the measured variation.
    for i in range(num_qubits - 1):
        x.ctrl(q[i], q[i + 1])
        rz(0.41, q[i + 1])
        x.ctrl(q[i], q[i + 1])
    x.ctrl(q[num_qubits - 1], q[0])
    rz(0.41, q[0])
    x.ctrl(q[num_qubits - 1], q[0])


@cudaq.kernel
def random_brickwork_kernel(num_qubits: int):
    q = cudaq.qvector(num_qubits)
    # Deterministic brickwork with two alternating matchings.  This is a
    # reproducible structural family, not a claim about a random-circuit
    # distribution.
    for layer in range(4):
        for i in range(num_qubits):
            if (i + layer) % 2 == 0:
                rz(0.11 + 0.07 * layer, q[i])
                rx(0.23 + 0.05 * (i % 3), q[i])
        start = layer % 2
        for i in range(start, num_qubits - 1, 2):
            x.ctrl(q[i], q[i + 1])


KERNELS: dict[str, Callable] = {
    "ghz": ghz_kernel,
    "hea": hea_kernel,
    "qaoa_cycle": qaoa_cycle_kernel,
    "random_brickwork": random_brickwork_kernel,
}


def structural_features(family: str, n: int) -> dict[str, int | float]:
    """Return deterministic logical features for the kernel definitions."""

    if family == "ghz":
        return {"logical_depth": n, "gate_count": n, "two_qubit_gates": n - 1}
    if family == "hea":
        # Three layers, 3 single-qubit gates per qubit and a linear entangler.
        return {
            "logical_depth": 3 * 3 + 3 * (n - 1),
            "gate_count": 3 * 3 * n + 3 * (n - 1),
            "two_qubit_gates": 3 * (n - 1),
        }
    if family == "qaoa_cycle":
        # H on every qubit and two CX + one RZ per ring edge.
        edges = n
        return {
            "logical_depth": 1 + 3 * (n - 1) + 3,
            "gate_count": n + 3 * edges,
            "two_qubit_gates": 2 * edges,
        }
    if family == "random_brickwork":
        two_q = sum((n - (layer % 2) - 1) // 2 for layer in range(4))
        one_q = sum((n + 1) // 2 for layer in range(4))
        return {
            "logical_depth": 4 * 2,
            "gate_count": one_q + two_q,
            "two_qubit_gates": two_q,
        }
    raise ValueError(f"unknown family: {family}")


def parse_csv(value: str, allowed: tuple[str, ...] | None = None) -> list[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("empty comma-separated argument")
    if allowed:
        bad = sorted(set(items) - set(allowed))
        if bad:
            raise ValueError(f"unsupported values: {bad}; allowed={allowed}")
    return items


def gpu_info() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
        return " | ".join(line.strip() for line in out.splitlines() if line.strip())
    except Exception:
        return "unavailable"


def set_target(target_name: str) -> tuple[str, str, str]:
    target, option, simulator, precision = TARGETS[target_name]
    if option is None:
        cudaq.set_target(target)
    else:
        cudaq.set_target(target, option=option)
    return target, simulator, precision


def measure_row(
    target_name: str,
    family: str,
    n: int,
    shots: int,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    target, simulator, precision = set_target(target_name)
    kernel = KERNELS[family]
    features = structural_features(family, n)
    precision_bytes = 16 if precision == "fp64" else 8
    row: dict[str, object] = {
        "target_name": target_name,
        "cudaq_target": target,
        "simulator": simulator,
        "precision": precision,
        "family": family,
        "width": n,
        "shots": shots,
        "logical_depth": features["logical_depth"],
        "gate_count": features["gate_count"],
        "two_qubit_gates": features["two_qubit_gates"],
        "statevector_bytes": (2**n) * precision_bytes,
        "runtime_semantics": "CUDA-Q sample wall-clock; first and warm calls separate; no queue/QPU time",
        "status": "ok",
    }
    try:
        first_start = time.perf_counter()
        counts = cudaq.sample(kernel, n, shots_count=shots)
        first_s = time.perf_counter() - first_start
        row["first_sample_s"] = first_s
        row["first_shots_observed"] = int(sum(counts.values()))
        for _ in range(warmups):
            cudaq.sample(kernel, n, shots_count=shots)
        warm = []
        for _ in range(repeats):
            start = time.perf_counter()
            counts = cudaq.sample(kernel, n, shots_count=shots)
            warm.append(time.perf_counter() - start)
            if int(sum(counts.values())) != shots:
                raise RuntimeError("CUDA-Q returned an unexpected shot count")
        warm_sorted = sorted(warm)
        row["warm_sample_median_s"] = warm_sorted[len(warm_sorted) // 2]
        row["warm_sample_min_s"] = min(warm)
        row["warm_sample_max_s"] = max(warm)
        row["warm_sample_repeats"] = json.dumps(warm)
    except Exception as exc:  # retain resource/unsupported rows explicitly
        row["status"] = "error"
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc).splitlines()[0][:500]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="cpu_fp64,gpu_fp32,gpu_fp64")
    parser.add_argument("--families", default=",".join(FAMILY_NAMES))
    parser.add_argument("--widths", default="16,20,24")
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    targets = parse_csv(args.targets, tuple(TARGETS))
    families = parse_csv(args.families, FAMILY_NAMES)
    widths = [int(x) for x in parse_csv(args.widths)]
    if any(n < 1 for n in widths):
        raise ValueError("width must be positive")
    if args.shots < 1 or args.repeats < 1 or args.warmups < 0:
        raise ValueError("shots/repeats must be positive and warmups non-negative")

    rows = []
    for target_name in targets:
        for family in families:
            for n in widths:
                print(f"[cudaq] target={target_name} family={family} width={n}", flush=True)
                rows.append(measure_row(target_name, family, n, args.shots, args.warmups, args.repeats))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "cudaq_version": str(cudaq.__version__),
        "cudaq_git_revision": "aca5853a76d499ecc3d5f97c2e06163ae99d9c75",
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "gpu": gpu_info(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "targets": targets,
        "families": families,
        "widths": widths,
        "shots": args.shots,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "target_contract": {
            "first_sample_s": "wall clock around first cudaq.sample call; may include lazy compilation/initialisation",
            "warm_sample_median_s": "median wall clock of post-warmup cudaq.sample calls; includes simulation and result extraction",
            "excluded": ["QPU execution", "queue waiting", "cloud turnaround", "cross-framework equivalence"],
        },
        "csv": str(args.output),
    }
    manifest_path = args.output.with_suffix(".environment.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "csv": str(args.output), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
