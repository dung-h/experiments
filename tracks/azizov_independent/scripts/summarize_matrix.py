#!/usr/bin/env python3
"""Summarize a checkpointed P1/P2 runtime matrix without pandas."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def numeric(rows, key):
    return [float(r[key]) for r in rows if r.get(key) not in (None, "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--label", default="P1 matrix")
    args = ap.parse_args()
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ok = [r for r in rows if r.get("status") == "ok"]
    status = Counter(r.get("status", "") for r in rows)
    lines = [
        f"# Azizov independent reproduction — {args.label}",
        "",
        "The target is local Qiskit Aer noisy-simulator execution after backend-aware transpilation.",
        "This report describes the local matrix only; it is not an exact reconstruction of the unavailable author artifact.",
        "",
        "## Coverage",
        "",
        f"- Total rows: **{len(rows)}**.",
        f"- Successful rows: **{len(ok)}**.",
        f"- Status counts: `{dict(status)}`.",
        f"- Unique circuits: **{len({r['circuit_id'] for r in rows})}**.",
        f"- Families: **{len({r['family'] for r in rows})}**.",
        f"- Backends: {', '.join(sorted({r['backend_class'] for r in rows}))}.",
        f"- Optimization levels: {', '.join(sorted({r['optimization_level'] for r in rows}))}.",
        "",
        "## Runtime summary",
        "",
        "| Backend | Opt | n successful | median T_transpile | median T_exec | p95 T_exec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    groups = defaultdict(list)
    for row in ok:
        groups[(row["backend_class"], row["optimization_level"])].append(row)
    for key, group in sorted(groups.items()):
        execs = sorted(numeric(group, "t_exec_s"))
        transpile = numeric(group, "t_transpile_s")
        p95 = execs[min(len(execs) - 1, max(0, int(round(0.95 * len(execs))) - 1))]
        lines.append(f"| {key[0]} | {key[1]} | {len(group)} | {statistics.median(transpile):.4f} | {statistics.median(execs):.4f} | {p95:.4f} |")
    if ok:
        lines += [
            "", "## Observed boundaries", "",
            f"- Successful `T_exec` range: {min(numeric(ok, 't_exec_s')):.4f}–{max(numeric(ok, 't_exec_s')):.4f} s.",
            f"- Maximum logical width in successful rows: {max(int(r['logical_width']) for r in ok)}.",
            f"- Maximum transpiled DAG row size: {max(int(r['dag_node_count']) for r in ok)} nodes.",
        ]
    lines += [
        "", "The matrix is checkpointed row-by-row. Timeout rows are retained "
        "as censored observations and must not be treated as ordinary zero or "
        "median targets in model training.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
