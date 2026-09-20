#!/usr/bin/env python3
"""Run a width-10--16 canary with process isolation and resource sampling.

Each backend/circuit row invokes the checkpointed matrix runner in a fresh
process. This keeps a memory-heavy row from taking down the complete canary.
The safety RSS cap is deliberately separate from the paper's 900-second
timeout; canary rows stopped by either cap are retained as censored rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


BACKENDS = ("FakeWashingtonV2", "FakeSherbrooke")
FIELDS = [
    "circuit_id", "family", "source_file", "source_sha256", "backend_class",
    "optimization_level", "shots", "logical_width", "logical_depth", "logical_ops",
    "logical_two_qubit_ops", "physical_width", "physical_depth", "physical_ops",
    "physical_two_qubit_ops", "swap_count", "dag_node_count", "t_transpile_s",
    "t_exec_s", "t_total_s", "status", "error", "peak_rss_mb",
    "peak_cpu_single_core_pct", "peak_cpu_host_pct", "monitor_wall_s", "stop_reason",
    "returncode",
]


def proc_stat(pid: int):
    """Return (ppid, cpu_ticks) for a live Linux process."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        tail = raw.rsplit(")", 1)[1].split()
        # tail[0] is state, tail[1] is ppid, tail[11]/tail[12] are utime/stime.
        return int(tail[1]), int(tail[11]) + int(tail[12])
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def live_tree(root_pid: int):
    parents = {}
    for path in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(path.name)
        except ValueError:
            continue
        item = proc_stat(pid)
        if item is not None:
            parents[pid] = item[0]
    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parents.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return sorted(tree)


def sample_tree(root_pid: int, previous=None):
    pids = live_tree(root_pid)
    rss_kb = 0
    cpu_ticks = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss_kb += int(line.split()[1])
                    break
        except (FileNotFoundError, ProcessLookupError):
            pass
        item = proc_stat(pid)
        if item is not None:
            cpu_ticks += item[1]
    now = time.monotonic()
    single_pct = 0.0
    host_pct = 0.0
    if previous is not None:
        old_time, old_ticks = previous
        elapsed = max(now - old_time, 1e-9)
        cpu_seconds = max(cpu_ticks - old_ticks, 0) / os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        single_pct = 100.0 * cpu_seconds / elapsed
        host_pct = single_pct / max(os.cpu_count() or 1, 1)
    return (now, cpu_ticks), rss_kb / 1024.0, single_pct, host_pct


def kill_group(proc: subprocess.Popen):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_row(args, source, backend_name, temp_root):
    temp_root.mkdir(parents=True, exist_ok=True)
    manifest = temp_root / "manifest.csv"
    output = temp_root / "row.csv"
    fields = list(source.keys())
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(source)
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_matrix.py")),
        "--mali-root", str(args.mali_root),
        "--manifest", str(manifest),
        "--output", str(output),
        "--max-qubits", str(max(int(source["logical_width"]), args.max_qubits)),
        "--shots", str(args.shots),
        "--timeout-s", str(args.timeout_s),
        "--backends", backend_name,
        "--optimization-levels", str(args.optimization_level),
        "--max-parallel-threads", str(args.max_parallel_threads),
        "--max-parallel-shots", str(args.max_parallel_shots),
        "--max-parallel-experiments", str(args.max_parallel_experiments),
    ]
    started = time.monotonic()
    label = f"{backend_name} {source['circuit_id']}"
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    previous = None
    peak_rss = peak_single = peak_host = 0.0
    stop_reason = "completed"
    last_progress = 0.0
    try:
        while proc.poll() is None:
            previous, rss, single, host = sample_tree(proc.pid, previous)
            peak_rss = max(peak_rss, rss)
            peak_single = max(peak_single, single)
            peak_host = max(peak_host, host)
            elapsed = time.monotonic() - started
            if elapsed - last_progress >= args.progress_interval_s:
                print(
                    f"progress {label} elapsed={elapsed:.0f}s rss={rss / 1024:.2f}GiB "
                    f"peak_rss={peak_rss / 1024:.2f}GiB cpu_host={host:.1f}%",
                    flush=True,
                )
                last_progress = elapsed
            if peak_rss >= args.max_rss_gib * 1024:
                stop_reason = f"rss_cap>{args.max_rss_gib:g}GiB"
                kill_group(proc)
                break
            if elapsed >= args.wall_timeout_s:
                stop_reason = f"wall_timeout>{args.wall_timeout_s:g}s"
                kill_group(proc)
                break
            time.sleep(args.sample_interval_s)
    except KeyboardInterrupt:
        kill_group(proc)
        proc.wait(timeout=10)
        raise
    output_text, _ = proc.communicate(timeout=10)
    # One-row run_matrix output has one data row when it completes normally.
    measured = []
    if output.is_file():
        with output.open(newline="") as handle:
            measured = list(csv.DictReader(handle))
    if measured:
        row = dict(measured[-1])
    else:
        row = {
            **source,
            "backend_class": backend_name,
            "optimization_level": str(args.optimization_level),
            "shots": str(args.shots),
            "status": "resource_limit" if stop_reason != "completed" else "error",
            "error": output_text[-1000:].replace("\n", " "),
        }
    if stop_reason.startswith("rss_cap"):
        terminal_status = "resource_limit"
    elif stop_reason.startswith("wall_timeout"):
        terminal_status = "timeout"
    else:
        terminal_status = "error"
    row.update({
        "status": terminal_status,
        "peak_rss_mb": f"{peak_rss:.2f}",
        "peak_cpu_single_core_pct": f"{peak_single:.2f}",
        "peak_cpu_host_pct": f"{peak_host:.2f}",
        "monitor_wall_s": f"{time.monotonic() - started:.3f}",
        "stop_reason": stop_reason,
        "returncode": str(proc.returncode),
    })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mali-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--environment", type=Path, required=True)
    ap.add_argument("--min-width", type=int, default=10)
    ap.add_argument("--max-width", type=int, default=16)
    ap.add_argument("--max-qubits", type=int, default=16)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--optimization-level", type=int, default=0)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--wall-timeout-s", type=float, default=180)
    ap.add_argument("--max-rss-gib", type=float, default=16.0)
    ap.add_argument("--sample-interval-s", type=float, default=0.5)
    ap.add_argument("--progress-interval-s", type=float, default=10.0)
    ap.add_argument("--max-parallel-threads", type=int, default=1)
    ap.add_argument("--max-parallel-shots", type=int, default=1)
    ap.add_argument("--max-parallel-experiments", type=int, default=1)
    ap.add_argument("--include-circuit", action="append", default=[])
    ap.add_argument("--exclude-circuit", action="append", default=[])
    args = ap.parse_args()
    with args.manifest.open(newline="") as handle:
        sources = [
            row for row in csv.DictReader(handle)
            if row.get("source_status") == "ok"
            and row.get("logical_width", "").isdigit()
            and args.min_width <= int(row["logical_width"]) <= args.max_width
            and (not args.include_circuit or row["circuit_id"] in set(args.include_circuit))
            and row["circuit_id"] not in set(args.exclude_circuit)
        ]
    sources.sort(key=lambda row: (row["family"], int(row["logical_width"]), row["circuit_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.environment.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="azizov_resource_canary_") as tmp:
        temp_root = Path(tmp)
        for backend in BACKENDS:
            for source in sources:
                print(backend, source["circuit_id"], flush=True)
                rows.append(run_row(args, source, backend, temp_root / f"{backend}_{source['circuit_id'].replace(':', '_')}"))
                with args.output.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
    args.environment.write_text(json.dumps({
        "python": sys.version,
        "platform": __import__("platform").platform(),
        "cpu_count": os.cpu_count(),
        "mali_root": str(args.mali_root),
        "min_width": args.min_width,
        "max_width": args.max_width,
        "source_rows": len(sources),
        "backend_count": len(BACKENDS),
        "optimization_level": args.optimization_level,
        "shots": args.shots,
        "paper_timeout_s": args.timeout_s,
        "wall_timeout_s": args.wall_timeout_s,
        "max_rss_gib": args.max_rss_gib,
        "sample_interval_s": args.sample_interval_s,
        "progress_interval_s": args.progress_interval_s,
        "max_parallel_threads": args.max_parallel_threads,
        "max_parallel_shots": args.max_parallel_shots,
        "max_parallel_experiments": args.max_parallel_experiments,
        "measurement": "sum RSS and CPU ticks over process tree sampled from /proc",
    }, indent=2) + "\n")
    print(f"completed {len(rows)} rows; output={args.output}")


if __name__ == "__main__":
    main()
