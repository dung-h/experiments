#!/usr/bin/env python3
"""Create a compact, provenance-aware P0 report from the runtime CSV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def floats(rows, key):
    return [float(r[key]) for r in rows if r.get(key) not in (None, "")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ablation", type=Path, default=None)
    args = ap.parse_args()
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ok = [r for r in rows if r.get("status") == "ok"]
    backends = sorted({r["backend_class"] for r in rows})
    families = sorted({r["family"] for r in rows})
    lines = [
        "# Azizov independent reproduction — P0 smoke report",
        "",
        "This is a bounded smoke run, not the paper's full 1,402-circuit result.",
        "The target is one timed `AerSimulator.run(...).result()` after one",
        "untimed backend warm-up; transpilation is measured separately.",
        "",
        "## Coverage",
        "",
        f"- Rows: **{len(rows)}** ({len(ok)} successful, {len(rows) - len(ok)} errors).",
        f"- Backends: {', '.join(backends)}.",
        f"- Families represented: **{len(families)}**.",
        f"- Optimization levels: {', '.join(sorted({r['optimization_level'] for r in rows}))}.",
        f"- Shots: {rows[0]['shots'] if rows else 'n/a'}.",
        "- The 22nd source family (`twolocalrandom`) has no unique circuit with "
        "eight or fewer logical qubits after source-hash deduplication, so it is "
        "not present in this P0 selection.",
        "",
        "## Runtime summary (seconds)",
        "",
        "| Backend | Opt | n | median T_transpile | median T_exec | p95 T_exec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    groups = defaultdict(list)
    for row in ok:
        groups[(row["backend_class"], row["optimization_level"])].append(row)
    for (backend, opt), group in sorted(groups.items()):
        execs = sorted(floats(group, "t_exec_s"))
        trans = floats(group, "t_transpile_s")
        p95 = execs[min(len(execs) - 1, max(0, int(round(0.95 * len(execs))) - 1))]
        lines.append(f"| {backend} | {opt} | {len(group)} | {statistics.median(trans):.4f} | {statistics.median(execs):.4f} | {p95:.4f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The P0 run verifies that the public Ma–Li QASM pool can be loaded, "
        "transpiled against both current fake-provider snapshots, and executed "
        "with Aer noise at 1,024 shots. It also exposes a non-trivial family "
        "and backend spread in `T_exec`. These numbers are local measurements "
        "under Qiskit 2.5.2/Aer 0.17.2 and must not be compared as exact values "
        "to the paper's HPC measurements.",
        "",
        "Next: run the full P1 matrix with explicit timeout/censoring records, "
        "then fit grouped baselines and compare source versus transpiled feature "
        "blocks.",
    ]
    if args.ablation and args.ablation.is_file():
        ablation = json.loads(args.ablation.read_text())
        lines += ["", "## Source versus compiled feature smoke ablation", "",
                  "The same circuit-ID grouped split is reused for every block.",
                  "These results are P0 diagnostics, not paper-scale metrics.", "",
                  "| Block | Model | R² log | R² seconds | RMSE seconds | MAE seconds |",
                  "|---|---|---:|---:|---:|---:|"]
        for block, block_data in ablation["blocks"].items():
            for model, metric in block_data["models"].items():
                lines.append(
                    f"| {block} | {model} | {metric['r2_log']:.4f} | "
                    f"{metric['r2_seconds']:.4f} | {metric['rmse_seconds']:.4f} | "
                    f"{metric['mae_seconds']:.4f} |"
                )
        lines += ["", "On this small subset, compiled global features improve Ridge "
                  "over source-only features, while the hybrid block does not "
                  "improve every model. This is the intended hypothesis check; "
                  "the full P1/P2 runs must determine whether the pattern is stable."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
