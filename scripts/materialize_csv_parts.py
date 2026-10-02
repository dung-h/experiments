#!/usr/bin/env python3
"""Restore frozen CSVs from row-aligned parts without changing their bytes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "artifacts/benchmark_v3/real_qpu/csv_parts_manifest.json"
SOURCES = (
    "artifacts/benchmark_v3/real_qpu/unified_method_comparisons_v2/attempts.csv",
    "artifacts/benchmark_v3/real_qpu/hyb_hanas_bounded_repair_v1_20261001/merged_analytical_attempts_for_c4.csv",
    "artifacts/benchmark_v3/real_qpu/archived_analytical_wave_v3_20260930/attempts.csv",
)
PART_LIMIT = 48 * 1024 * 1024


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def inside(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    if path == root.resolve():
        raise ValueError("manifest path cannot be the repository root")
    return path


class RecordLines:
    """Let csv.reader find complete records while retaining original bytes."""

    def __init__(self, stream):
        self.stream = stream
        self.lines: list[bytes] = []

    def __iter__(self):
        return self

    def __next__(self):
        line = self.stream.readline()
        if not line:
            raise StopIteration
        self.lines.append(line)
        return line.decode("utf-8")

    def take(self) -> bytes:
        record = b"".join(self.lines)
        self.lines.clear()
        return record


def split_csv(root: Path, name: str, limit: int = PART_LIMIT) -> dict:
    source = inside(root, name)
    parts_dir = source.with_suffix(source.suffix + ".parts")
    parts_dir.mkdir()  # Refuse to overwrite an earlier partition.
    parts: list[dict] = []
    rows_total = 0
    with source.open("rb") as stream:
        lines = RecordLines(stream)
        reader = csv.reader(lines)
        next(reader)
        header = lines.take()
        output = None
        count = 0

        def close_part():
            if output is not None:
                output.close()
                parts.append({
                    "path": path.relative_to(root).as_posix(),
                    "rows": count,
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                })

        try:
            for _ in reader:
                record = lines.take()
                if len(header) + len(record) > limit:
                    raise ValueError(f"single CSV record exceeds part limit: {name}")
                if output is None or output.tell() + len(record) > limit:
                    close_part()
                    path = parts_dir / f"part-{len(parts) + 1:03d}.csv"
                    output = path.open("xb")
                    output.write(header)
                    count = 0
                output.write(record)
                count += 1
                rows_total += 1
            if output is None:  # Header-only CSV.
                path = parts_dir / "part-001.csv"
                output = path.open("xb")
                output.write(header)
            close_part()
        finally:
            if output is not None and not output.closed:
                output.close()
    return {
        "path": name,
        "bytes": source.stat().st_size,
        "sha256": digest(source),
        "rows": rows_total,
        "header_bytes": len(header),
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "parts": parts,
    }


def materialize(root: Path, item: dict) -> str:
    target = inside(root, item["path"])
    if target.exists():
        if target.stat().st_size == item["bytes"] and digest(target) == item["sha256"]:
            return "already_verified"
        raise ValueError(f"refusing to overwrite mismatched CSV: {item['path']}")
    if not item["parts"] or sum(part["rows"] for part in item["parts"]) != item["rows"]:
        raise ValueError(f"invalid part row counts: {item['path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".csv-join-", delete=False) as out:
            temporary = Path(out.name)
            for index, part in enumerate(item["parts"]):
                path = inside(root, part["path"])
                if path.stat().st_size != part["bytes"] or digest(path) != part["sha256"]:
                    raise ValueError(f"CSV part hash/size mismatch: {part['path']}")
                with path.open("rb") as stream:
                    header = stream.read(item["header_bytes"])
                    if hashlib.sha256(header).hexdigest() != item["header_sha256"]:
                        raise ValueError(f"CSV part header mismatch: {part['path']}")
                    if index == 0:
                        out.write(header)
                    shutil.copyfileobj(stream, out, length=1 << 20)
        if temporary.stat().st_size != item["bytes"] or digest(temporary) != item["sha256"]:
            raise ValueError(f"reassembled CSV hash/size mismatch: {item['path']}")
        os.replace(temporary, target)
        temporary = None
        return "restored_and_verified"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="store_true", help="maintainer: create new parts from the three original CSVs")
    args = parser.parse_args()
    manifest = ROOT / MANIFEST
    if args.split:
        if manifest.exists():
            raise ValueError("partition manifest already exists; refusing to overwrite")
        payload = {"format_version": 1, "part_limit_bytes": PART_LIMIT,
                   "files": [split_csv(ROOT, name) for name in SOURCES]}
        manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"files": len(payload["files"]), "parts": sum(len(item["parts"]) for item in payload["files"])}))
    else:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload["format_version"] != 1:
            raise ValueError("unsupported partition format")
        for item in payload["files"]:
            print(f"{materialize(ROOT, item)}: {item['path']}")


if __name__ == "__main__":
    main()
