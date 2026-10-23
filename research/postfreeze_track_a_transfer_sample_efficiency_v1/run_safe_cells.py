#!/usr/bin/env python3
"""Run the sample-efficiency extension one cell per subprocess with hard guards.

This wrapper does not change the frozen A1 model or training contract. It only
prevents concurrent cells and bounds process-level resources so a failed graph
cell cannot take down the full run. Every cell receives its own Python/CUDA
allocator state and writes a status receipt outside the scientific predictions.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts/postfreeze_track_a_transfer_sample_efficiency_v1"
PYTHON = Path(sys.executable)
FRACTIONS = (0.10, 0.25, 0.50)
SUBSET_SEEDS = (20261019, 20261020)
SEEDS = (42, 1234, 31415)
TREATMENTS = ("M-S", "M-P", "G-S", "G-P")


def run_one(fraction: float, subset_seed: int, fold: int, treatment: str, seed: int,
            timeout_s: int, memory_fraction: float, ram_limit_gib: float,
            label: str) -> dict:
    cell = OUT / "qpu_factorial" / f"fraction_{int(round(fraction * 100)):03d}" / f"subset_{subset_seed}" / treatment / f"fold_{fold}" / f"seed_{seed}"
    status_dir = OUT / "safe_cell_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    key = f"{label}_fraction_{int(round(fraction * 100)):03d}_subset_{subset_seed}_{treatment}_fold_{fold}_seed_{seed}"
    status = status_dir / f"{key}.json"
    terminal = cell / "terminal_receipt.json"
    if terminal.is_file():
        payload = {"status": "ALREADY_COMPLETE", "cell": key, "terminal_receipt": str(terminal.relative_to(REPO))}
        status.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
        "LASER_CUDA_MEMORY_FRACTION": str(memory_fraction),
    })
    args = [
        "--fraction", str(fraction), "--subset-seed", str(subset_seed),
        "--fold", str(fold), "--treatment", treatment, "--seed", str(seed),
    ]
    # Import torch first, set the per-process fraction, then execute the exact
    # existing single-cell runner. No model code is duplicated here.
    guard = (
        "import os,runpy,sys,torch; "
        "torch.set_num_threads(1); torch.set_num_interop_threads(1); "
        "torch.cuda.set_per_process_memory_fraction(float(os.environ['LASER_CUDA_MEMORY_FRACTION'])); "
        "sys.argv=['run_sample_efficiency.py']+os.environ['LASER_CELL_ARGS'].split(); "
        "runpy.run_path(os.environ['LASER_CELL_SCRIPT'],run_name='__main__')"
    )
    env["LASER_CELL_ARGS"] = " ".join(args)
    env["LASER_CELL_SCRIPT"] = str(REPO / "research/postfreeze_track_a_transfer_sample_efficiency_v1/run_sample_efficiency.py")
    log = OUT / "logs" / f"safe_{key}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    peak_rss = 0
    limit_bytes = int(ram_limit_gib * (1024 ** 3))

    def rss_bytes(pid: int) -> int:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (FileNotFoundError, OSError, ValueError):
            return 0
        return 0

    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen([str(PYTHON), "-c", guard], cwd=REPO, env=env,
                                stdout=fh, stderr=subprocess.STDOUT)
        started = time.time()
        reason = None
        while proc.poll() is None:
            current = rss_bytes(proc.pid)
            peak_rss = max(peak_rss, current)
            if current > limit_bytes:
                reason = "RAM_LIMIT"
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            if time.time() - started > timeout_s:
                reason = "TIMEOUT"
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            time.sleep(0.5)
        returncode = proc.poll()
    if reason is not None:
        result = {"status": reason, "returncode": returncode}
    else:
        result = {"status": "COMPLETE" if returncode == 0 and terminal.is_file() else "FAILED", "returncode": returncode}
    result.update({"cell": key, "fraction": fraction, "subset_seed": subset_seed,
                   "fold": fold, "treatment": treatment, "seed": seed,
                   "timeout_s": timeout_s, "memory_fraction": memory_fraction,
                   "ram_limit_gib": ram_limit_gib,
                   "peak_rss_gib": round(peak_rss / (1024 ** 3), 3),
                   "elapsed_s": round(time.time() - started, 3),
                   "log": str(log.relative_to(REPO))})
    status.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subset-seed", type=int, choices=SUBSET_SEEDS, default=20261019)
    p.add_argument("--fraction", type=float, choices=FRACTIONS)
    p.add_argument("--fold", type=int, choices=range(5))
    p.add_argument("--treatment", choices=TREATMENTS)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--all", action="store_true")
    p.add_argument("--timeout-s", type=int, default=900)
    p.add_argument("--memory-fraction", type=float, default=0.55)
    p.add_argument("--ram-limit-gib", type=float, default=28.0)
    p.add_argument("--label", default="safe")
    a = p.parse_args()
    if not a.all and None in (a.fraction, a.fold, a.treatment, a.seed):
        p.error("provide a complete cell or --all")
    cells = []
    if a.all:
        for fraction in FRACTIONS:
            for fold in range(5):
                for treatment in TREATMENTS:
                    for seed in SEEDS:
                        cells.append((fraction, a.subset_seed, fold, treatment, seed))
    else:
        cells = [(a.fraction, a.subset_seed, a.fold, a.treatment, a.seed)]
    for cell in cells:
        result = run_one(*cell, timeout_s=a.timeout_s, memory_fraction=a.memory_fraction,
                         ram_limit_gib=a.ram_limit_gib, label=a.label)
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status"] in {"FAILED", "TIMEOUT", "RAM_LIMIT"}:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
