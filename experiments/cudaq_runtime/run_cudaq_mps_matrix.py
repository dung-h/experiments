#!/usr/bin/env python3
"""Run a target-aware CUDA-Q MPS canary.

CUDA-Q reads the MPS controls from environment variables when the target is
initialised.  Therefore one invocation represents one predeclared
``max_bond``/cutoff configuration.  Run separate invocations for different
caps and merge their CSVs afterwards.

The runner records two distinct local-simulator targets:

* ``state_*_s``: wall clock around ``cudaq.get_state``;
* ``sample_*_s``: wall clock around ``cudaq.sample``.

The MPS tensors returned by ``get_state`` expose their extents, allowing an
observed maximum bond and tensor-storage lower bound to be recorded without
materialising a dense statevector.  These are post-run diagnostics, not
pre-run features.  In particular, ``observed_max_bond`` must not be used as a
static estimator input.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cudaq


FAMILIES = ("ghz", "hea", "qaoa_cycle", "random_brickwork")


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
    for layer in range(4):
        for i in range(num_qubits):
            if (i + layer) % 2 == 0:
                rz(0.11 + 0.07 * layer, q[i])
                rx(0.23 + 0.05 * (i % 3), q[i])
        for i in range(layer % 2, num_qubits - 1, 2):
            x.ctrl(q[i], q[i + 1])


KERNELS = {
    "ghz": ghz_kernel,
    "hea": hea_kernel,
    "qaoa_cycle": qaoa_cycle_kernel,
    "random_brickwork": random_brickwork_kernel,
}


def features(family: str, n: int) -> dict[str, int]:
    if family == "ghz":
        return {"logical_depth": n, "gate_count": n, "two_qubit_gates": n - 1}
    if family == "hea":
        return {
            "logical_depth": 9 + 3 * (n - 1),
            "gate_count": 9 * n + 3 * (n - 1),
            "two_qubit_gates": 3 * (n - 1),
        }
    if family == "qaoa_cycle":
        edges = n
        return {"logical_depth": 1 + 3 * n, "gate_count": n + 3 * edges, "two_qubit_gates": 2 * edges}
    if family == "random_brickwork":
        two_q = sum((n - (layer % 2) - 1) // 2 for layer in range(4))
        one_q = sum((n + 1) // 2 for layer in range(4))
        return {"logical_depth": 8, "gate_count": one_q + two_q, "two_qubit_gates": two_q}
    raise ValueError(family)


def parse_csv(value: str, allowed: tuple[str, ...] | None = None) -> list[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("empty comma-separated argument")
    if allowed:
        bad = sorted(set(items) - set(allowed))
        if bad:
            raise ValueError(f"unsupported values: {bad}; allowed={allowed}")
    return items


def state_stats(state) -> dict[str, int]:
    tensors = state.getTensors()
    extents = [list(t.extents) for t in tensors]
    bonds = [int(x) for shape in extents for x in shape]
    return {
        "mps_tensor_count": len(tensors),
        "observed_max_bond": max(bonds) if bonds else 0,
        "mps_tensor_elements": int(sum(t.get_num_elements() for t in tensors)),
        "mps_tensor_bytes": int(sum(t.get_num_elements() * t.get_element_size() for t in tensors)),
    }


def basis_strings(n: int) -> list[str]:
    return [format(i, f"0{n}b") for i in range(2**n)]


def fidelity_against_dense(kernel, n: int, mps_state, max_width: int) -> float | None:
    """Compare amplitudes for small states only; avoid dense materialisation above max_width."""

    if n > max_width:
        return None
    # CUDA-Q's MPS tensor order is little-endian while ``State.to_numpy`` is
    # exposed in the dense vector convention. Reverse each basis label before
    # comparing; symmetric GHZ circuits would otherwise hide this mismatch.
    basis = basis_strings(n)
    mps_amp = mps_state.amplitudes([label[::-1] for label in basis])
    cudaq.set_target("nvidia", option="fp64")
    dense_state = cudaq.get_state(kernel, n)
    dense_amp = dense_state.to_numpy()
    # CUDA-Q's basis-string ordering and dense vector ordering are both
    # lexicographic for this interface.  Normalize defensively before overlap.
    import numpy as np

    mps_vec = np.asarray(mps_amp, dtype=np.complex128)
    dense_vec = np.asarray(dense_amp, dtype=np.complex128)
    mps_vec /= np.linalg.norm(mps_vec)
    dense_vec /= np.linalg.norm(dense_vec)
    return float(abs(np.vdot(dense_vec, mps_vec)) ** 2)


def set_mps_environment(max_bond: int, abs_cutoff: float, svd_algo: str) -> None:
    os.environ["CUDAQ_MPS_MAX_BOND"] = str(max_bond)
    os.environ["CUDAQ_MPS_ABS_CUTOFF"] = str(abs_cutoff)
    os.environ["CUDAQ_MPS_SVD_ALGO"] = svd_algo


def measure_row(args, family: str, n: int) -> dict[str, object]:
    kernel = KERNELS[family]
    f = features(family, n)
    row: dict[str, object] = {
        "target_name": "cudaq_tensornet_mps",
        "simulator": "tensornet_mps",
        "precision": args.precision,
        "family": family,
        "width": n,
        "shots": args.shots,
        "mps_max_bond_config": args.max_bond,
        "mps_abs_cutoff_config": args.abs_cutoff,
        "mps_svd_algo": args.svd_algo,
        "logical_depth": f["logical_depth"],
        "gate_count": f["gate_count"],
        "two_qubit_gates": f["two_qubit_gates"],
        "statevector_bytes": (2**n) * (16 if args.precision == "fp64" else 8),
        "runtime_semantics": "CUDA-Q MPS get_state/sample wall-clock; first and warm calls separate; no queue/QPU time",
        "status": "ok",
    }
    try:
        option = "fp64" if args.precision == "fp64" else "fp32"
        cudaq.set_target("tensornet-mps", option=option)
        start = time.perf_counter()
        state = cudaq.get_state(kernel, n)
        row["state_first_s"] = time.perf_counter() - start
        row.update(state_stats(state))
        row["fidelity_vs_dense"] = fidelity_against_dense(kernel, n, state, args.reference_width_max)
        # The small-width fidelity check temporarily switches to the dense
        # NVIDIA target. Restore the declared MPS target before timing warm and
        # sampling calls.
        option = "fp64" if args.precision == "fp64" else "fp32"
        cudaq.set_target("tensornet-mps", option=option)
        state_warm = []
        for _ in range(args.warmups):
            cudaq.get_state(kernel, n)
        for _ in range(args.repeats):
            start = time.perf_counter()
            state = cudaq.get_state(kernel, n)
            state_warm.append(time.perf_counter() - start)
        row["state_warm_median_s"] = sorted(state_warm)[len(state_warm) // 2]
        row["state_warm_repeats"] = json.dumps(state_warm)
        sample_start = time.perf_counter()
        counts = cudaq.sample(kernel, n, shots_count=args.shots)
        row["sample_first_s"] = time.perf_counter() - sample_start
        row["sample_first_shots_observed"] = int(sum(counts.values()))
        sample_warm = []
        for _ in range(args.warmups):
            cudaq.sample(kernel, n, shots_count=args.shots)
        for _ in range(args.repeats):
            start = time.perf_counter()
            counts = cudaq.sample(kernel, n, shots_count=args.shots)
            sample_warm.append(time.perf_counter() - start)
            if int(sum(counts.values())) != args.shots:
                raise RuntimeError("unexpected shot count")
        row["sample_warm_median_s"] = sorted(sample_warm)[len(sample_warm) // 2]
        row["sample_warm_repeats"] = json.dumps(sample_warm)
    except Exception as exc:
        row["status"] = "error"
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc).splitlines()[0][:500]
    return row


def gpu_info() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        ).strip()
    except Exception:
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--widths", default="16,20,24,28")
    parser.add_argument("--max-bond", type=int, default=64)
    parser.add_argument("--abs-cutoff", type=float, default=1e-12)
    parser.add_argument("--svd-algo", default="gesvdj")
    parser.add_argument("--precision", choices=["fp32", "fp64"], default="fp64")
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reference-width-max", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.families = parse_csv(args.families, FAMILIES)
    args.widths = [int(x) for x in parse_csv(args.widths)]
    if args.max_bond < 1 or not 0.0 < args.abs_cutoff < 1.0:
        raise ValueError("max-bond must be positive and abs-cutoff must be in (0,1)")
    set_mps_environment(args.max_bond, args.abs_cutoff, args.svd_algo)

    rows = []
    for family in args.families:
        for n in args.widths:
            print(f"[cudaq-mps] cap={args.max_bond} family={family} width={n}", flush=True)
            rows.append(measure_row(args, family, n))

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
        "gpu": gpu_info(),
        "target": "tensornet-mps",
        "precision": args.precision,
        "max_bond": args.max_bond,
        "abs_cutoff": args.abs_cutoff,
        "svd_algo": args.svd_algo,
        "families": args.families,
        "widths": args.widths,
        "shots": args.shots,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "reference_width_max": args.reference_width_max,
        "target_semantics": {
            "state": "cudaq.get_state wall-clock and returned MPS tensor extents",
            "sample": "cudaq.sample wall-clock with first/warm calls separate",
            "post_run_only": ["observed_max_bond", "mps_tensor_bytes", "fidelity_vs_dense"],
            "excluded": ["QPU execution", "queue waiting", "cloud turnaround"],
        },
        "csv": str(args.output),
    }
    args.output.with_suffix(".environment.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "csv": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
