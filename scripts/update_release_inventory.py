#!/usr/bin/env python3
"""Refresh the review-package inventory from the Git index.

The inventory is intentionally a release review artifact: every tracked path
is listed, existing explanatory fields are preserved, and newly staged files
receive a narrow E1--E6 review-package description. The inventory itself has
no self-hash, as verified by ``verify_release_inventory.py``.
"""
from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/release_inventory.csv"
FIELDS = [
    "path", "action", "reason", "required_by", "sha256", "bytes",
    "rights_status", "delivery",
]
DEFAULT = {
    "action": "include",
    "reason": "E1--E6 review-package evidence or maintained code",
    "required_by": "Current reader tables, reproducibility commands, or regression suite",
    "rights_status": "review-branch publication requested by author 2026-10-02; no new license grant; complete external source archives excluded",
    "delivery": "Git on presentation-review; no Git LFS; not a tagged public release",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_paths() -> list[str]:
    output = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"])
    return sorted(path for path in output.decode("utf-8").split("\0") if path)


def read_rows_from_head() -> dict[str, dict[str, str]]:
    """Keep explicit external/materialized exclusions across inventory refreshes."""
    try:
        payload = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", "HEAD:docs/release_inventory.csv"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return {}
    return {row["path"]: row for row in csv.DictReader(payload.splitlines())}


def main() -> None:
    with INVENTORY.open(newline="", encoding="utf-8") as handle:
        working = {row["path"]: row for row in csv.DictReader(handle)}
    historical = {**read_rows_from_head(), **working}

    rows: list[dict[str, str]] = []
    tracked = tracked_paths()
    for rel in tracked:
        path = ROOT / rel
        row = {**DEFAULT, **historical.get(rel, {}), "path": rel}
        if rel == INVENTORY.relative_to(ROOT).as_posix():
            row["sha256"] = ""
            row["bytes"] = ""
        else:
            row["sha256"] = sha256(path)
            row["bytes"] = str(path.stat().st_size)
        rows.append({field: row.get(field, "") for field in FIELDS})

    # A materialized CSV, external source boundary, or archive-local record is
    # deliberately not a tracked path. Preserve those explicit decisions.
    retained = [
        row for path, row in historical.items()
        if path not in set(tracked) and row.get("action") != "include"
    ]
    rows.extend({field: row.get(field, "") for field in FIELDS} for row in retained)
    rows.sort(key=lambda row: row["path"])

    with INVENTORY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
