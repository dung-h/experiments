#!/usr/bin/env python3
"""Replace workstation-specific paths in a completed VQCSim run with run-relative paths.

The upstream runner emits absolute paths to generated QASM and result files.
They are useful during execution but make a committed archival artifact less
portable. This post-processing step changes only path strings under the given
run root; timing, features and terminal status are not modified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PATH_FIELDS = (
    "qasm3_path",
    "qasm3_bound_path",
    "qasm2_parameterized_path",
    "qasm2_bound_path",
)


def replace_prefix(value: Any, root: str) -> Any:
    if isinstance(value, str):
        return value[len(root) + 1:] if value.startswith(root + "/") else value
    if isinstance(value, list):
        return [replace_prefix(item, root) for item in value]
    if isinstance(value, dict):
        return {key: replace_prefix(item, root) for key, item in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    root_text = str(run_root)
    csv_path = run_root / "records.csv"
    json_path = run_root / "records.json"
    if not csv_path.is_file() or not json_path.is_file():
        raise FileNotFoundError("expected records.csv and records.json under --run-root")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames
    if not fields:
        raise RuntimeError("records.csv has no header")
    changed = 0
    for row in rows:
        for field in PATH_FIELDS:
            original = row.get(field, "")
            relative = replace_prefix(original, root_text)
            if relative != original:
                row[field] = relative
                changed += 1
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    records = json.loads(json_path.read_text(encoding="utf-8"))
    json_path.write_text(
        json.dumps(replace_prefix(records, root_text), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name in ("manifest.json", "run_summary.json", "progress.json"):
        path = run_root / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            path.write_text(
                json.dumps(replace_prefix(payload, root_text), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    print(f"rewrote {changed} CSV path fields under {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
