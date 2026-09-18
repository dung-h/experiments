#!/usr/bin/env python3
"""Create an auditable local-machine screening subset from the source manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--eligible", type=Path, required=True)
    ap.add_argument("--rejected", type=Path, required=True)
    ap.add_argument("--max-width", type=int, default=9)
    ap.add_argument("--max-ops", type=int, default=4000)
    ap.add_argument("--max-depth", type=int, default=4000)
    args = ap.parse_args()
    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible, rejected = [], []
    for row in rows:
        reasons = []
        if row.get("source_status") != "ok":
            reasons.append("source_parse_error")
        else:
            if int(row["logical_width"]) > args.max_width:
                reasons.append(f"width>{args.max_width}")
            if int(row["logical_ops"]) > args.max_ops:
                reasons.append(f"ops>{args.max_ops}")
            if int(row["logical_depth"]) > args.max_depth:
                reasons.append(f"depth>{args.max_depth}")
        row = dict(row)
        row["screening_status"] = "eligible" if not reasons else "rejected"
        row["screening_reason"] = ";".join(reasons)
        (eligible if not reasons else rejected).append(row)
    fields = list(rows[0]) + ["screening_status", "screening_reason"] if rows else ["screening_status", "screening_reason"]
    for path, output in ((args.eligible, eligible), (args.rejected, rejected)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output)
    print(f"eligible={len(eligible)} ({len(eligible) / len(rows) * 100:.1f}%) rejected={len(rejected)} ({len(rejected) / len(rows) * 100:.1f}%)")
    print(f"thresholds: width<={args.max_width}, logical_ops<={args.max_ops}, logical_depth<={args.max_depth}")


if __name__ == "__main__":
    main()
