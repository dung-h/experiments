#!/usr/bin/env python3
"""Check published files, documentation links and data-selection boundaries."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify_qpu_dataset(root: Path) -> None:
    """Check public aggregate selection accounting, not excluded source labels."""
    path = root / "protocol/real_qpu_dataset.json"
    if not path.exists():
        return  # Small publication-verifier fixtures need no experiment metadata.
    dataset = json.loads(path.read_text())
    sources = dataset["sources"]
    expected = {"mali_real_qpu": 340, "qonductor_single_circuit_ibm": 230,
                "qpack_mcp": 3945}
    if dataset["observations"] != 4515 or {
            key: value["observations"] for key, value in sources.items()} != expected:
        raise ValueError("QPU dataset selection counts disagree")
    if sum(value["groups"] for value in sources.values()) != dataset["groups"] or dataset["groups"] != 165:
        raise ValueError("QPU dataset group counts disagree")
    q = dataset["source_filters"]["qonductor_single_circuit_ibm"]
    classes = q["logical_recovery_classifications"]
    if (sum(classes.values()) != q["ibm_source_screen"]
            or classes["archive_resolved_recipe"] != q["retained"]
            or q["excluded"] + q["retained"] != q["ibm_source_screen"]
            or sum(q["retained_families"].values()) != 230):
        raise ValueError("Qonductor source-filter accounting disagrees")
    folds = dataset["split"]["outer_test_source_counts"]
    if len(folds) != 5 or any(set(fold) != set(expected) or min(fold.values()) <= 0 for fold in folds):
        raise ValueError("QPU split source support disagrees")
    if {key: sum(fold[key] for fold in folds) for key in expected} != expected:
        raise ValueError("QPU split observation counts disagree")
    if sum(dataset["backend_counts"].values()) != 4515:
        raise ValueError("QPU backend counts disagree")
    contract = json.loads((root / "protocol/real_qpu_common_panel.json").read_text())
    if contract["assigned_rows"] != 4515 or contract["source_counts"] != expected:
        raise ValueError("QPU public execution summary disagrees")
    with (root / "results/real_qpu/dataset_profile.csv").open(newline="") as stream:
        profile = {row["source_id"]: row for row in csv.DictReader(stream)}
    if set(profile) != set(expected):
        raise ValueError("QPU aggregate profile source coverage disagrees")
    for key, value in sources.items():
        if (int(profile[key]["observations"]) != value["observations"]
                or int(profile[key]["groups"]) != value["groups"]):
            raise ValueError("QPU aggregate profile counts disagree")
    with (root / "results/real_qpu/source_filtering.csv").open(newline="") as stream:
        filtering = {row["source_id"]: row for row in csv.DictReader(stream)}
    if set(filtering) != set(expected):
        raise ValueError("QPU source-filter table coverage disagrees")
    for key, count in expected.items():
        row = filtering[key]
        if int(row["retained_n"]) != count or int(row["source_screen_n"]) != (
                count + int(row["excluded_within_screen_n"])):
            raise ValueError("QPU source-filter table counts disagree")
    if dataset["selection"]["exact_historical_logical_instance_claim"]:
        raise ValueError("Unsupported exact historical logical-instance claim")
    if dataset["status"]["scores_from_other_training_populations_reused"]:
        raise ValueError("Unsupported reuse of fitted QPU scores")
    if any(not re.fullmatch(r"[0-9a-f]{64}", digest)
           for digest in dataset["evidence_sha256"].values()):
        raise ValueError("Invalid QPU evidence digest")


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
    verify_qpu_dataset(root)
    return len(index)


if __name__ == "__main__":
    print(f"Verified {verify()} published files; no bundled QPU observations")
