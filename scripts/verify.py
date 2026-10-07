#!/usr/bin/env python3
"""Check published files, documentation links and data-selection boundaries."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path = ROOT) -> int:
    root = root.resolve()
    index = json.loads((root / "provenance/files.json").read_text())["files"]
    expected = {entry["path"] for entry in index} | {"provenance/files.json"}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and ".git" not in path.relative_to(root).parts}
    if len(expected) != len(index) + 1 or actual != expected:
        raise ValueError(f"inventory mismatch: extra={sorted(actual-expected)}, missing={sorted(expected-actual)}")
    for entry in index:
        path = root / entry["path"]
        if (not path.resolve().is_relative_to(root)
                or any(part.is_symlink() for part in [path, *path.parents]
                       if part != root and part.is_relative_to(root))):
            raise ValueError(f"unsafe file: {entry['path']}")
        if (path.stat().st_size != entry["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]):
            raise ValueError(f"hash mismatch: {entry['path']}")
        if (entry["path"].startswith(("data/real_qpu/", "data/simulator/aer/graphs/"))
                or path.suffix in {".qasm", ".sqlite", ".db", ".parquet", ".pyc"}
                or "__pycache__" in path.parts):
            raise ValueError(f"excluded input or cache: {entry['path']}")
    publication = json.loads((root / "provenance/publication.json").read_text())
    for key in ("third_party_qpu_observations_included",
                "third_party_qpu_row_level_derivatives_included", "source_circuit_bytes_included"):
        if publication.get(key) is not False:
            raise ValueError(f"unexpected publication scope: {key}")
    for path in root.rglob("*.md"):
        if ".git" in path.parts:
            continue
        for target in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", path.read_text()):
            target = target.strip("<>").split("#", 1)[0]
            if not target or target.startswith(("https:", "http:", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError(f"broken documentation link: {path.relative_to(root)} -> {target}")
    return len(index)


if __name__ == "__main__":
    print(f"Verified {verify()} published files; no bundled QPU observations")
