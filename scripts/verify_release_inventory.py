#!/usr/bin/env python3
"""Verify every committed candidate file against docs/release_inventory.csv."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/release_inventory.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    with INVENTORY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    include: dict[str, dict[str, str]] = {}
    materialized: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for row in rows:
        if row.get("action") not in {"include", "materialize"}:
            continue
        rel = row.get("path", "")
        if not rel or rel in include or rel in materialized:
            errors.append(f"missing or duplicate inventory path: {rel!r}")
            continue
        (include if row["action"] == "include" else materialized)[rel] = row

    try:
        actual = set(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "ls-files", "-z"], text=False
            ).decode("utf-8").rstrip("\0").split("\0")
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return fail([f"cannot inspect candidate Git index: {exc}"])

    expected = set(include)
    for rel in sorted(expected - actual):
        errors.append(f"included inventory path is not tracked: {rel}")
    for rel in sorted(actual - expected):
        errors.append(f"tracked path is missing from inventory: {rel}")

    verified = 0
    verified_bytes = 0
    for rel, row in {**include, **materialized}.items():
        expected_hash = row.get("sha256", "")
        if not expected_hash:  # The inventory intentionally does not hash itself.
            if rel != INVENTORY.relative_to(ROOT).as_posix():
                errors.append(f"missing SHA-256: {rel}")
            continue
        path = (ROOT / rel).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            errors.append(f"inventory path escapes candidate: {rel}")
            continue
        if not path.is_file():
            errors.append(f"file missing: {rel}; run python3 scripts/materialize_csv_parts.py" if rel in materialized else f"included file missing: {rel}")
            continue
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            errors.append(f"SHA-256 mismatch: {rel}")
            continue
        verified += 1
        verified_bytes += path.stat().st_size

    result = {
        "artifact_id": "release-inventory-verification-v1",
        "status": "PASS" if not errors else "FAIL",
        "tracked_paths": len(actual),
        "hashes_verified": verified,
        "materialized_paths": len(materialized),
        "verified_bytes": verified_bytes,
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


def fail(errors: list[str]) -> int:
    print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
