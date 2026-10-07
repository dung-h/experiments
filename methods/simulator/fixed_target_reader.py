"""Read the fixed-bond MPS target table without requiring joint-MPS files.

The package includes the recovered fixed-chi table, but not the source R9
target manifest or every input required to refit the historical model. This
reader verifies the packaged table, row schema, quality separation and frozen
C44 assignment pin. It does not claim source-level lineage or enable fitting.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


TARGET_PATH = "data/simulator/mps/recovered_targets/fixed_chi16_targets.csv"
RESULT_MANIFEST_PATH = "data/simulator/mps/fixed_recovered/aggregate/manifest.json"
FILE_INDEX_PATH = "provenance/files.json"
EXPECTED_FOLD_COUNTS = {0: 37, 1: 28, 2: 26, 3: 25, 4: 34}
ALLOWED_QUALITY = {"quality_pass", "quality_failed", "quality_mixed", "quality_unavailable"}


def _safe_path(root: Path, relative: str) -> Path:
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or not name.parts:
        raise ValueError(f"unsafe package path: {relative}")
    root = root.resolve()
    path = root
    for part in name.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed for package input: {relative}")
    try:
        path.resolve(strict=True).relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"missing or escaping package input: {relative}") from exc
    return path


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _assignment_hash(folds: dict[str, int]) -> str:
    rows = [{"source_qasm_sha256": value, "fold": folds[value]} for value in sorted(folds)]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_fixed_targets(package_root: Path) -> dict[str, Any]:
    """Validate the included fixed-target table against package pins.

    The result manifest is used only for the split and target-revision pins.
    The file inventory supplies the target-table byte hash. The joint target
    table and its measurement ledger are intentionally never opened here.
    """
    package_root = package_root.resolve()
    index_path = _safe_path(package_root, FILE_INDEX_PATH)
    target_path = _safe_path(package_root, TARGET_PATH)
    result_manifest_path = _safe_path(package_root, RESULT_MANIFEST_PATH)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index.get("files")
    if not isinstance(entries, list):
        raise ValueError("package file index has no files list")
    matches = [entry for entry in entries if entry.get("path") == TARGET_PATH]
    if len(matches) != 1:
        raise ValueError("fixed target table must have exactly one package-index entry")
    expected = matches[0].get("sha256")
    if not isinstance(expected, str) or _sha256(target_path) != expected:
        raise ValueError("fixed target table differs from the published byte hash")

    manifest = json.loads(result_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "five_fold_oof_complete_not_promoted":
        raise ValueError("fixed-MPS result manifest is not the reviewed completed revision")
    revision = manifest.get("target_revision_id")
    if not isinstance(revision, str) or not revision:
        raise ValueError("fixed-MPS result manifest has no target revision")
    split_pin = manifest.get("split_assignment_sha256")
    if not isinstance(split_pin, str) or len(split_pin) != 64:
        raise ValueError("fixed-MPS result manifest has no C44 split pin")

    with target_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 150:
        raise ValueError(f"fixed-MPS target table must contain 150 hashes; found {len(rows)}")
    folds: dict[str, int] = {}
    quality_counts: Counter[str] = Counter()
    for row in rows:
        qhash = row.get("source_qasm_sha256", "")
        if len(qhash) != 64 or any(ch not in "0123456789abcdef" for ch in qhash) or qhash in folds:
            raise ValueError("fixed-MPS target table has a malformed or duplicate QASM hash")
        try:
            fold = int(row["fold"])
            seconds = float(row["target_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid fixed-MPS label or fold for {qhash}") from exc
        if fold not in EXPECTED_FOLD_COUNTS:
            raise ValueError(f"fixed-MPS target has an invalid C44 fold for {qhash}")
        if row.get("target_status") != "runtime_observed" or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"fixed-MPS target is not a finite observed runtime for {qhash}")
        if row.get("target_revision_id") != revision:
            raise ValueError(f"fixed-MPS target revision differs from result manifest for {qhash}")
        quality = row.get("quality_status_separate", "")
        if quality not in ALLOWED_QUALITY:
            raise ValueError(f"unknown separate MPS quality status for {qhash}")
        if row.get("quality_is_model_input", "").strip().lower() not in {"false", "0"}:
            raise ValueError("MPS quality status must remain separate from model inputs")
        folds[qhash] = fold
        quality_counts[quality] += 1

    if dict(sorted(Counter(folds.values()).items())) != EXPECTED_FOLD_COUNTS:
        raise ValueError("fixed-MPS target table differs from frozen C44 fold counts")
    actual_split = _assignment_hash(folds)
    if actual_split != split_pin:
        raise ValueError("fixed-MPS target table differs from the result manifest C44 split pin")
    if sum(quality_counts.values()) != 150:
        raise ValueError("fixed-MPS quality statuses do not cover the target table")

    return {
        "status": "verified_packaged_fixed_targets",
        "source_target_manifest_included": False,
        "source_refit_enabled": False,
        "target_path": TARGET_PATH,
        "target_sha256": expected,
        "target_revision_id": revision,
        "split_assignment_sha256": actual_split,
        "assigned_hashes": len(rows),
        "observed_runtime_labels": len(rows),
        "quality_status_counts": dict(sorted(quality_counts.items())),
        "joint_targets_read": False,
    }
