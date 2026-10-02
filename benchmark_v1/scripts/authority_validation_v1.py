#!/usr/bin/env python3
"""Shared validation primitives for historical and live authority epochs.

The benchmark has two intentionally different authority epochs.  Historical
artifacts keep their receipt-time/build-time pins, while reader-facing
post-Wave-3C validation checks the live applied v2 authority.  This module is
read-only and never writes CURRENT.json or any artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / "benchmark_v1/registry/CURRENT.json"
HISTORICAL_ARCHIVE = ROOT / "benchmark_v1/registry/archive/CURRENT_pre_wave3_authority_switch_20261001.json"
FIDELITY_V1_PATH = "benchmark_v1/registry/method_fidelity_registry_v1.json"
FIDELITY_V2_PATH = "benchmark_v1/registry/method_fidelity_registry_v2.json"
SEMANTIC_V2_PATH = "artifacts/benchmark_v1/results/SEMANTIC_INDEX_MANIFEST_v2.json"

HISTORICAL_CURRENT_SHA256 = "fa840866ffa16b58d4fc0aabe2bc8b7a50023d38791a7cc20401989c5faa0680"
LIVE_CURRENT_SHA256 = "8a4ba7288e46324c49e290a1244c518b9f01b73a7f27a9b4a79984a13135c9e3"
FIDELITY_V1_SHA256 = "df2fb465de487ef077b519c9f044241d4c466d69ffde654ae0fd7093f90ffbbe"
FIDELITY_V2_SHA256 = "98f3460c8d9a0cbabb424555dd695cee573409ed65facdb60557d836f1750810"
SEMANTIC_V2_SHA256 = "cb41e06740001fee9fd4a3b3fa2b463c45aaa13159cc9f900b9a5de30e928b26"
SCORECARD_SHA256 = "f6d27336ecdb16a90efa4156a7e4d48dfbbea0be3c393b6e4a9a435641f8fc5c"
C4_RECEIPT_SHA256 = "e6e30da64c71ae79c75d1382779fe4910972992986a57368b49a5c0802c5a059"
C4_RECORDED_CURRENT_SHA256 = "f1d428eae1731656470a0b23a0ac966e012cb6e43db2e599771e8716521cec27"
C4_RECORDED_FIDELITY_SHA256 = "ce1ff119506da21b874af1f199f0a0572ed6a05912f3623f6e680947881aefbf"


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest without changing the file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fidelity_path(data: dict[str, Any]) -> str | None:
    return (data.get("current_method_fidelity") or {}).get("path")


def archive_is_historical_v1(
    path: Path | None = None,
    *,
    expected_sha256: str = HISTORICAL_CURRENT_SHA256,
) -> bool:
    """Whether *path* is the pinned pre-switch CURRENT archive."""
    path = HISTORICAL_ARCHIVE if path is None else Path(path)
    if not path.is_file() or sha256(path) != expected_sha256:
        return False
    try:
        data = load_json(path)
    except (OSError, ValueError, TypeError):
        return False
    return _fidelity_path(data) == FIDELITY_V1_PATH


def live_is_applied_v2(
    path: Path | None = None,
    *,
    expected_sha256: str = LIVE_CURRENT_SHA256,
    expected_fidelity_sha256: str = FIDELITY_V2_SHA256,
    expected_scorecard_sha256: str = SCORECARD_SHA256,
) -> bool:
    """Whether *path* is the pinned, post-Wave-3C live authority."""
    path = CURRENT if path is None else Path(path)
    if not path.is_file() or sha256(path) != expected_sha256:
        return False
    try:
        data = load_json(path)
    except (OSError, ValueError, TypeError):
        return False
    fidelity = data.get("current_method_fidelity") or {}
    pointer = (data.get("reader_pointers") or {}).get("two_domain_scorecard_v1") or {}
    return (
        data.get("status") == "current"
        and _fidelity_path(data) == FIDELITY_V2_PATH
        and fidelity.get("sha256") == expected_fidelity_sha256
        and pointer.get("sha256") == expected_scorecard_sha256
        and data.get("wave4_five_fold_training_authorized") is False
    )


def authority_errors(
    current_path: Path | None = None,
    archive_path: Path | None = None,
    *,
    expected_live_sha256: str = LIVE_CURRENT_SHA256,
    expected_archive_sha256: str = HISTORICAL_CURRENT_SHA256,
    expected_fidelity_v2_sha256: str = FIDELITY_V2_SHA256,
    expected_scorecard_sha256: str = SCORECARD_SHA256,
) -> list[str]:
    """Return fail-closed errors for both authority epochs.

    The archive check is deliberately independent of the live check.  A live
    v2 switch therefore cannot make a historical v1 artifact appear mutated,
    and a historical archive cannot authorize a stale live CURRENT file.
    """
    current_path = CURRENT if current_path is None else Path(current_path)
    archive_path = HISTORICAL_ARCHIVE if archive_path is None else Path(archive_path)
    errors: list[str] = []
    if not archive_is_historical_v1(archive_path, expected_sha256=expected_archive_sha256):
        errors.append("historical CURRENT archive missing, hash-drifted, or not fidelity v1")
    if not live_is_applied_v2(
        current_path,
        expected_sha256=expected_live_sha256,
        expected_fidelity_sha256=expected_fidelity_v2_sha256,
        expected_scorecard_sha256=expected_scorecard_sha256,
    ):
        errors.append("live CURRENT is missing, hash-drifted, or not applied fidelity v2")
    return errors


def current_json_modified_flag(path: Path | None = None) -> bool:
    """Return whether the post-switch live authority is invalid."""
    return not live_is_applied_v2(path)


def c4_receipt_errors(
    receipt_path: Path,
    *,
    expected_receipt_sha256: str = C4_RECEIPT_SHA256,
    expected_recorded_current_sha256: str = C4_RECORDED_CURRENT_SHA256,
    expected_recorded_fidelity_sha256: str = C4_RECORDED_FIDELITY_SHA256,
) -> list[str]:
    """Validate C4's historical receipt-time authority without live coupling."""
    errors: list[str] = []
    if not receipt_path.is_file():
        return ["C4 receipt is missing"]
    if sha256(receipt_path) != expected_receipt_sha256:
        errors.append("C4 receipt hash drifted")
    try:
        receipt = load_json(receipt_path)
    except (OSError, ValueError, TypeError):
        return errors + ["C4 receipt is not valid JSON"]
    if receipt.get("current_json_sha256") != expected_recorded_current_sha256:
        errors.append("C4 receipt recorded CURRENT hash drifted")
    labels = {row.get("label"): row for row in receipt.get("current_json_verification", [])}
    fidelity = labels.get("current_method_fidelity") or {}
    if fidelity.get("sha256") != expected_recorded_fidelity_sha256:
        errors.append("C4 receipt recorded fidelity hash drifted")
    if fidelity.get("status") != "MATCH":
        errors.append("C4 receipt recorded fidelity verification is not MATCH")
    return errors
