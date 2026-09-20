#!/usr/bin/env python3
"""Run a small, reproducible EMU-MPS GPU frontier benchmark.

The benchmark is intentionally a separate neutral-atom/MPS track.  Its target
is EMU-MPS wall-clock runtime and per-step statistics, not Qiskit Aer runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import torch
from emu_mps import MPSBackend, MPSConfig
from pulser import Pulse, Register, Sequence
from pulser.devices import DigitalAnalogDevice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pasqal_emu_mps/gpu_frontier.csv"),
    )
    parser.add_argument("--duration-ns", type=int, default=200)
    parser.add_argument("--dt-ns", type=float, default=10.0)
    parser.add_argument("--max-bond-dim", type=int, default=256)
    parser.add_argument("--max-krylov-dim", type=int, default=40)
    return parser.parse_args()


def make_sequence(rows: int, cols: int, duration_ns: int) -> Sequence:
    register = Register.rectangle(rows, cols, spacing=5.0, prefix="q")
    sequence = Sequence(register, DigitalAnalogDevice)
    sequence.declare_channel("rydberg_global", "rydberg_global")
    sequence.add(
        Pulse.ConstantPulse(duration_ns, 1.0, 0.0, 0.0),
        "rydberg_global",
    )
    return sequence


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this script requires a GPU.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = [("n16", 4, 4), ("n24", 4, 6), ("n30", 5, 6)]
    rows: list[dict[str, object]] = []
    for case, nrows, ncols in cases:
        n_qubits = nrows * ncols
        sequence = make_sequence(nrows, ncols, args.duration_ns)
        config = MPSConfig(
            dt=args.dt_ns,
            precision=1e-5,
            max_bond_dim=args.max_bond_dim,
            max_krylov_dim=args.max_krylov_dim,
            num_gpus_to_use=1,
            log_level=20,
        )
        print(
            f"BEGIN {case} n={n_qubits} device={torch.cuda.get_device_name(0)}",
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = MPSBackend(sequence, config=config).run()
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - start
        peak_gpu_allocated_mb = torch.cuda.max_memory_allocated() / 2**20
        peak_gpu_reserved_mb = torch.cuda.max_memory_reserved() / 2**20
        statistics = result.get_tagged_results()["statistics"]
        rows.append(
            {
                "case": case,
                "n_qubits": n_qubits,
                "rows": nrows,
                "cols": ncols,
                "duration_ns": args.duration_ns,
                "dt_ns": args.dt_ns,
                "num_steps": len(statistics),
                "precision": config.precision,
                "max_bond_dim_limit": config.max_bond_dim,
                "max_krylov_dim": config.max_krylov_dim,
                "max_bond_dimension_observed": max(
                    item["max_bond_dimension"] for item in statistics
                ),
                "peak_rss_mb": max(item["RSS"] for item in statistics),
                "peak_state_memory_mb": max(
                    item["memory_footprint"] for item in statistics
                ),
                "peak_gpu_allocated_mb": peak_gpu_allocated_mb,
                "peak_gpu_reserved_mb": peak_gpu_reserved_mb,
                "sum_step_seconds": sum(item["duration"] for item in statistics),
                "wall_seconds": wall_seconds,
                "gpu": torch.cuda.get_device_name(0),
            }
        )
        print(
            f"END {case} wall={wall_seconds:.3f}s "
            f"steps={len(statistics)} "
            f"max_chi={rows[-1]['max_bond_dimension_observed']} "
            f"peak_rss_mb={rows[-1]['peak_rss_mb']:.3f}",
            flush=True,
        )

    fieldnames = list(rows[0])
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "emu_mps": "2.9.1",
        "pulser_core": "1.9.1",
        "config": vars(args),
    }
    env_path = args.output.with_suffix(".environment.json")
    env_path.write_text(json.dumps(environment, indent=2, default=str) + "\n")
    print(f"WROTE {args.output}")
    print(f"WROTE {env_path}")


if __name__ == "__main__":
    main()
