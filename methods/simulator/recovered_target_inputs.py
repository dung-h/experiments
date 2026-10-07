#!/usr/bin/env python3
"""Strict reader for the immutable R9 recovered MPS target revision.

This module only validates derived target inputs.  It does not materialize,
rewrite, or train on them.  The R9 adapters import this reader before every
fit and aggregate stage so a changed manifest or target table fails closed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHI = (2, 4, 8, 16, 32, 64)
EXPECTED_C44_FOLD_COUNTS = {0: 37, 1: 28, 2: 26, 3: 25, 4: 34}
EXPECTED_FAMILY_FOLD_COUNTS = {0: 32, 1: 25, 2: 31, 3: 31, 4: 31}
EXPECTED_HASHES = 150
EXPECTED_TARGET_MANIFEST_SHA256 = "c7a66fe6195daaf1af74c2da5582a06a260469d227682d2634276fdd0a4fdbeb"
FIXED_FILENAME = "fixed_chi16_targets.csv"
JOINT_FILENAME = "joint_rung_targets.csv"
JOINT_LEDGER = "joint_measurements/attempt_records.jsonl"
JOINT_SUMMARY = "joint_measurements/run_summary.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite_nonnegative(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return parsed


def canonical_assignment_sha256(
    fold_by_hash: dict[str, int],
    *,
    expected_counts: dict[int, int] | None = None,
) -> str:
    if (len(fold_by_hash) != EXPECTED_HASHES
            or set(fold_by_hash.values()) - set(EXPECTED_C44_FOLD_COUNTS)
            or any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in fold_by_hash)):
        raise ValueError("split assignment must cover 150 hashes and folds 0..4")
    if expected_counts is not None and dict(sorted(Counter(fold_by_hash.values()).items())) != expected_counts:
        raise ValueError("split assignment fold counts differ from its frozen contract")
    rows = [{"source_qasm_sha256": digest, "fold": int(fold_by_hash[digest])}
            for digest in sorted(fold_by_hash)]
    return sha256_bytes(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())


def _file_sha_pin(manifest: dict[str, Any], filename: str) -> str:
    """Read a filename pin from the R2 manifest's explicit file registries."""
    candidates: list[Any] = []
    for key in ("files_sha256", "files"):
        registry = manifest.get(key)
        if isinstance(registry, dict):
            candidates.append(registry.get(filename))
            candidates.append(registry.get(f"targets_recovered_v1/{filename}"))
    for candidate in candidates:
        if isinstance(candidate, str) and len(candidate) == 64:
            return candidate
        if isinstance(candidate, dict):
            digest = candidate.get("sha256") or candidate.get("sha256_file")
            if isinstance(digest, str) and len(digest) == 64:
                return digest
    raise ValueError(f"manifest has no SHA-256 file pin for {filename}")


def _row_count_pin(manifest: dict[str, Any], filename: str) -> int | None:
    for key in ("files_row_counts", "row_counts"):
        registry = manifest.get(key)
        if isinstance(registry, dict):
            value = registry.get(filename)
            if value is None:
                value = registry.get(f"targets_recovered_v1/{filename}")
            if isinstance(value, int):
                return value
    registry = manifest.get("files")
    if isinstance(registry, dict):
        value = registry.get(filename)
        if isinstance(value, dict) and isinstance(value.get("row_count"), int):
            return value["row_count"]
    return None


def _split_pins(manifest: dict[str, Any]) -> tuple[str, str]:
    splits = manifest.get("split_assignments")
    if not isinstance(splits, dict):
        raise ValueError("manifest is missing split_assignments")
    c44 = splits.get("c44_split_assignment_sha256")
    family = splits.get("family_holdout_split_assignment_sha256")
    if not all(isinstance(x, str) and len(x) == 64 for x in (c44, family)):
        raise ValueError("manifest split assignment hashes are missing or malformed")
    return c44, family


@dataclass(frozen=True)
class RecoveredTargets:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    target_revision_id: str
    manifest: dict[str, Any]
    fixed_rows: tuple[dict[str, str], ...]
    joint_rows: tuple[dict[str, str], ...]
    fixed_fold_by_hash: dict[str, int]
    family_fold_by_hash: dict[str, int]
    fixed_table_sha256: str
    joint_table_sha256: str
    runner_lineage: dict[str, str]


def load_recovered_targets(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str | None = EXPECTED_TARGET_MANIFEST_SHA256,
) -> RecoveredTargets:
    """Validate all R9 target files and frozen split identities, fail closed."""
    manifest_path = manifest_path.resolve()
    if manifest_path.name != "manifest.json":
        raise ValueError("R9 target manifest must be named manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = sha256_bytes(manifest_bytes)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise ValueError("R9 target manifest differs from the explicitly pinned revision")
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("R9 target manifest must be a JSON object")
    if manifest.get("artifact_id") != "family-mps-r9-recovered-targets-v1":
        raise ValueError("unexpected R9 target artifact_id")
    if manifest.get("status") != "PASS" or manifest.get("target_revision_id") != "mps-r9-recovered-v1":
        raise ValueError("R9 target manifest is not a passing recovered target revision")
    revision = str(manifest["target_revision_id"])
    root = manifest_path.parent
    paths = {FIXED_FILENAME: root / FIXED_FILENAME, JOINT_FILENAME: root / JOINT_FILENAME}
    hashes: dict[str, str] = {}
    tables: dict[str, list[dict[str, str]]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"R9 target table is missing: {path}")
        actual = sha256_file(path)
        if actual != _file_sha_pin(manifest, name):
            raise ValueError(f"R9 target table hash differs from manifest: {name}")
        rows = read_csv(path)
        row_pin = _row_count_pin(manifest, name)
        if row_pin is not None and row_pin != len(rows):
            raise ValueError(f"R9 target row count differs from manifest: {name}")
        hashes[name] = actual
        tables[name] = rows

    fixed_rows = tables[FIXED_FILENAME]
    if len(fixed_rows) != EXPECTED_HASHES:
        raise ValueError(f"fixed target table must have 150 rows, got {len(fixed_rows)}")
    fixed_fold: dict[str, int] = {}
    allowed_quality = {"quality_pass", "quality_failed", "quality_mixed", "quality_unavailable"}
    for row in fixed_rows:
        digest = row.get("source_qasm_sha256", "")
        if len(digest) != 64 or digest in fixed_fold:
            raise ValueError("fixed target table has a malformed or duplicate source-QASM hash")
        try:
            fold = int(row["fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"fixed target has an invalid fold for {digest}") from exc
        if row.get("target_status") != "runtime_observed":
            raise ValueError(f"fixed R9 target must be runtime_observed for {digest}")
        finite_nonnegative(row.get("target_seconds"), field=f"fixed target seconds {digest}")
        if row.get("quality_status_separate") not in allowed_quality:
            raise ValueError(f"fixed target has an unknown separate quality status for {digest}")
        if row.get("quality_is_model_input", "false").strip().lower() not in {"false", "0"}:
            raise ValueError("fixed target quality must remain separate from model inputs")
        if row.get("target_revision_id") != revision:
            raise ValueError(f"fixed target row revision mismatch for {digest}")
        fixed_fold[digest] = fold
    if canonical_assignment_sha256(fixed_fold, expected_counts=EXPECTED_C44_FOLD_COUNTS) != \
            manifest.get("split_assignments", {}).get("c44_split_assignment_sha256"):
        raise ValueError("manifest C44 split pin does not match fixed target rows")

    joint_rows = tables[JOINT_FILENAME]
    if len(joint_rows) != EXPECTED_HASHES * len(CHI):
        raise ValueError(f"joint target table must have 900 rows, got {len(joint_rows)}")
    joint_fold: dict[str, int] = {}
    joint_family_fold: dict[str, int] = {}
    by_hash: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in joint_rows:
        digest = row.get("source_qasm_sha256", "")
        if digest not in fixed_fold:
            raise ValueError(f"joint rung hash is outside the fixed panel: {digest}")
        try:
            chi = int(row["max_bond"])
            c44 = int(row["c44_fold"])
            family = int(row["family_holdout_fold_diagnostic_only"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid joint target identity/split for {digest}") from exc
        if chi not in CHI or chi in by_hash[digest]:
            raise ValueError(f"invalid or duplicate joint rung for {digest}/{chi}")
        if row.get("target_status") != "runtime_observed":
            raise ValueError(f"joint R9 runtime target must be runtime_observed for {digest}/{chi}")
        finite_nonnegative(row.get("target_runtime_seconds"), field=f"joint runtime {digest}/{chi}")
        if row.get("target_quality_pass", "").strip().lower() not in {"true", "false"}:
            raise ValueError(f"joint quality label must be true/false for {digest}/{chi}")
        if int(row.get("session_count", "0")) != 3:
            raise ValueError(f"joint rung must use exactly three sessions for {digest}/{chi}")
        threshold = finite_nonnegative(row.get("quality_threshold"), field=f"quality threshold {digest}/{chi}")
        if not math.isclose(threshold, 0.99, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("joint target quality threshold differs from frozen 0.99")
        if row.get("target_revision_id", revision) != revision:
            raise ValueError(f"joint target row revision mismatch for {digest}/{chi}")
        if c44 != fixed_fold[digest]:
            raise ValueError(f"joint target C44 fold differs from fixed target for {digest}")
        previous_c44 = joint_fold.setdefault(digest, c44)
        previous_family = joint_family_fold.setdefault(digest, family)
        if previous_c44 != c44 or previous_family != family:
            raise ValueError(f"joint target has inconsistent fold assignment across rungs for {digest}")
        by_hash[digest][chi] = row
    if set(joint_fold) != set(fixed_fold) or any(set(cells) != set(CHI) for cells in by_hash.values()):
        raise ValueError("joint target hash×rung coverage differs from the complete 150×6 panel")
    c44_pin, family_pin = _split_pins(manifest)
    if canonical_assignment_sha256(fixed_fold, expected_counts=EXPECTED_C44_FOLD_COUNTS) != c44_pin:
        raise ValueError("manifest C44 split pin does not match fixed target rows")
    family_counts = {int(key): int(value) for key, value in
                     manifest.get("split_assignments", {}).get("family_holdout_hashes_by_fold", {}).items()}
    if family_counts != EXPECTED_FAMILY_FOLD_COUNTS:
        raise ValueError("manifest family-component fold sizes differ from the frozen assignment")
    if canonical_assignment_sha256(joint_family_fold, expected_counts=EXPECTED_FAMILY_FOLD_COUNTS) != family_pin:
        raise ValueError("manifest family-component split pin does not match joint target rows")

    # A hash may have a quality-passing rung only when its recorded minimum is
    # one of the passing rungs; NO_PASS means the complete six-rung ladder has
    # no passing rung. The target is repeated identically at each rung.
    for digest, cells in by_hash.items():
        minima = {str(row.get("minimum_passing_chi", "")) for row in cells.values()}
        if len(minima) != 1:
            raise ValueError(f"minimum_passing_chi differs across rungs for {digest}")
        minimum = next(iter(minima))
        passing = [chi for chi, row in cells.items() if row["target_quality_pass"].strip().lower() == "true"]
        expected = str(min(passing)) if passing else "NO_PASS"
        if minimum != expected:
            raise ValueError(f"minimum_passing_chi label is inconsistent with quality labels for {digest}")

    source_code = manifest.get("source_code_hashes")
    input_hashes = manifest.get("input_hashes")
    if not isinstance(source_code, dict) or not isinstance(input_hashes, dict):
        raise ValueError("R9 manifest must preserve source-code and input-hash lineage")
    runner_lineage = {
        "old_joint_measurement_runner_sha256": str(source_code.get("old_joint_measurement_runner", "")),
        "r4_recovery_runner_sha256": str(source_code.get("joint_measurement_runner", "")),
        "joint_training_runner_sha256": str(input_hashes.get("joint_trainer_candidate", "")),
    }
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in runner_lineage.values()):
        raise ValueError("R9 manifest has incomplete joint runner lineage pins")
    if runner_lineage["old_joint_measurement_runner_sha256"] == runner_lineage["r4_recovery_runner_sha256"]:
        raise ValueError("historical and R4 measurement runner identities must remain distinct")
    if input_hashes.get("old_joint_measurement_runner") != runner_lineage["old_joint_measurement_runner_sha256"]:
        raise ValueError("R9 old joint runner lineage disagrees across manifest sections")
    if input_hashes.get("joint_measurement_runner") != runner_lineage["r4_recovery_runner_sha256"]:
        raise ValueError("R9 R4 runner lineage disagrees across manifest sections")

    return RecoveredTargets(
        root=root, manifest_path=manifest_path,
        manifest_sha256=manifest_sha, target_revision_id=revision,
        manifest=manifest, fixed_rows=tuple(fixed_rows), joint_rows=tuple(joint_rows),
        fixed_fold_by_hash=fixed_fold, family_fold_by_hash=joint_family_fold,
        fixed_table_sha256=hashes[FIXED_FILENAME], joint_table_sha256=hashes[JOINT_FILENAME],
        runner_lineage=runner_lineage,
    )


def verify_manifest_file(bundle: RecoveredTargets, relative_path: str) -> str:
    """Verify one materializer output/input path against its manifest file pin."""
    registry = bundle.manifest.get("files_sha256")
    if not isinstance(registry, dict):
        raise ValueError("R9 manifest has no files_sha256 registry")
    expected = registry.get(relative_path)
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"R9 manifest has no exact SHA-256 pin for {relative_path}")
    path = bundle.root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"R9 manifest-pinned file is missing: {relative_path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"R9 manifest-pinned file hash mismatch: {relative_path}")
    return actual
