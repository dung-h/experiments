#!/usr/bin/env python3
"""Run a timeout-bounded cuTensorNet feasibility frontier one cell at a time.

Each cell is a separate process so that an OOM, CUDA failure, or timeout is a
recorded observation rather than a lost long-running batch. Successful rows
retain the runtime semantics produced by ``run_cutensornet_runtime_benchmark``;
this orchestrator only adds feasibility status and writes a separate failures
table.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "tracks" / "cutensornet" / "code" / "run_cutensornet_runtime_benchmark.sh"
DEFAULT_OUTPUT = ROOT / "run-output" / "cutensornet" / "feasibility_frontier"


def comma_values(value: str, convert: type[int] | type[float] | type[str]) -> list[int] | list[float] | list[str]:
    return [convert(item.strip()) for item in value.split(",") if item.strip()]


def cell_name(family: str, qubits: int, multiplier: float) -> str:
    return f"{family}__q{qubits:02d}__x{multiplier:g}".replace(".", "p")


def read_success(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default="ghz,hea,qaoa_cycle,random_brickwork,qft")
    parser.add_argument("--qubits", default="16,20,24,28,30")
    parser.add_argument("--depth-multipliers", default="0.5,1,2,4")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cell-timeout-s", type=float, default=60.0)
    parser.add_argument("--memory-limit", default="70%")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not RUNNER.is_file():
        raise SystemExit(f"Benchmark runner not found: {RUNNER}")
    families = comma_values(args.families, str)
    widths = comma_values(args.qubits, int)
    multipliers = comma_values(args.depth_multipliers, float)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cell_root = args.output_dir / "cells"
    successes: list[dict[str, str]] = []
    failures: list[dict[str, object]] = []
    manifest = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "runner": str(RUNNER.relative_to(ROOT)),
        "families": families,
        "qubits": widths,
        "depth_multipliers": multipliers,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "cell_timeout_s": args.cell_timeout_s,
        "memory_limit": args.memory_limit,
        "seed": args.seed,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    cells = [
        (family, width, multiplier)
        for family in families
        for width in widths
        for multiplier in ((1.0,) if family in {"ghz", "qft"} else multipliers)
    ]
    for index, (family, width, multiplier) in enumerate(cells, start=1):
        name = cell_name(family, width, multiplier)
        output = cell_root / name
        command = [
            str(RUNNER), "--families", family, "--qubits", str(width),
            "--depth-multipliers", str(multiplier), "--warmups", str(args.warmups),
            "--repeats", str(args.repeats), "--memory-limit", args.memory_limit,
            "--seed", str(args.seed), "--output-dir", str(output),
        ]
        print(f"[{index}/{len(cells)}] {name}", flush=True)
        base_failure = {
            "family": family, "num_qubits": width, "depth_multiplier": multiplier,
            "cell": name, "command": command,
        }
        try:
            completed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, timeout=args.cell_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            failures.append({
                **base_failure, "status": "timeout", "timeout_s": args.cell_timeout_s,
                "stdout": (exc.stdout or "")[-4000:], "stderr": (exc.stderr or "")[-4000:],
            })
            print(f"  TIMEOUT after {args.cell_timeout_s:g}s", file=sys.stderr, flush=True)
            continue
        rows = read_success(output / "cutensornet_runtime_benchmark.csv")
        if completed.returncode == 0 and rows:
            for row in rows:
                row["feasibility_status"] = "ok"
                row["cell_timeout_s"] = str(args.cell_timeout_s)
                successes.append(row)
            continue
        error_file = output / "cutensornet_runtime_benchmark_failures.json"
        runner_failures = json.loads(error_file.read_text(encoding="utf-8")) if error_file.exists() else []
        failures.append({
            **base_failure,
            "status": "error" if completed.returncode else "no_success_row",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "runner_failures": runner_failures,
        })
        print("  FAILED; preserved in failures JSON", file=sys.stderr, flush=True)

    write_csv(args.output_dir / "cutensornet_feasibility_frontier.csv", successes)
    failure_path = args.output_dir / "cutensornet_feasibility_frontier_failures.json"
    if failures:
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    else:
        failure_path.unlink(missing_ok=True)
    print(f"Completed: {len(successes)} successful, {len(failures)} unsuccessful cells")


if __name__ == "__main__":
    main()
