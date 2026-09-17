#!/usr/bin/env python3
"""Validate the versioned four-track capsule without external dependencies."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "README.md",
    "RESULTS.md",
    "REPRODUCTION.md",
    "upstream/upstream.lock.json",
    "artifacts/manifest.json",
    "tracks/qonductor/qonductor_local_replication.patch",
    "tracks/mali/README.md",
    "tracks/cdaa_qcre/README.md",
    "tracks/cdaa_qcre/data/instruction_durations/SNAPSHOT_MANIFEST.csv",
    "tracks/cdaa_qcre/independent/sqgm/exp-in/exp_eagle.json",
    "tracks/cdaa_qcre/independent/sqgm/exp-in/exp_heron.json",
    "tracks/cutensornet/code/run_cutensornet_runtime_benchmark.py",
    "artifacts/mali/fold10_qwalk_outlier.json",
    "artifacts/qonductor/reproduction_metrics.json",
    "artifacts/cdaa_qcre/independent_all_metrics.csv",
    "artifacts/cutensornet/cutensornet_runtime_benchmark.csv",
)

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def main() -> int:
    for relative in REQUIRED:
        require((ROOT / relative).is_file(), f"missing required artifact: {relative}")

    lock = json.loads((ROOT / "upstream/upstream.lock.json").read_text())
    require(set(lock["tracks"]) == {"mali", "qonductor", "cdaa", "qcre", "cutensornet"},
            "unexpected upstream lock tracks")
    artifact_manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
    require(len(artifact_manifest["artifacts"]) == 6,
            "unexpected replication artifact manifest size")

    qonductor = json.loads((ROOT / "artifacts/qonductor/reproduction_metrics.json").read_text())
    execution = qonductor["execution_time"]
    require(abs(execution["qonductor_vs_real"]["r2"] - 0.9386347722408748) < 1e-12,
            "Qonductor regression R2 fixture changed")
    require(abs(execution["dag_vs_real"]["r2"] - 0.8848516371292685) < 1e-12,
            "Qonductor DAG R2 fixture changed")

    with (ROOT / "artifacts/cutensornet/cutensornet_runtime_benchmark.csv").open(newline="") as handle:
        cutn_rows = list(csv.DictReader(handle))
    require(len(cutn_rows) == 25, "unexpected cuTensorNet baseline row count")
    require({"cutensornet_runtime_est_s", "contract_gpu_median_s", "end_to_end_first_s"}
            <= set(cutn_rows[0]), "cuTensorNet target columns changed")

    with (ROOT / "artifacts/cdaa_qcre/independent_all_metrics.csv").open(newline="") as handle:
        cdaa_rows = list(csv.DictReader(handle))
    require(len(cdaa_rows) == 315, "unexpected CDAA/QCRE independent metric row count")

    with (ROOT / "tracks/cdaa_qcre/data/instruction_durations/SNAPSHOT_MANIFEST.csv").open(
        newline=""
    ) as handle:
        snapshots = list(csv.DictReader(handle))
    require(len(snapshots) == 6, "unexpected instruction-duration snapshot count")
    for row in snapshots:
        path = ROOT / "tracks/cdaa_qcre/data/instruction_durations" / row["packaged_file"]
        require(path.is_file(), f"missing duration snapshot: {path.name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == row["sha256"], f"duration snapshot hash changed: {path.name}")

    for path in (ROOT / "tracks/cdaa_qcre/independent").rglob("*.py"):
        require(
            "/home/server/Documents/cdaa" not in path.read_text(encoding="utf-8"),
            f"non-portable absolute CDAA path remains: {path.relative_to(ROOT)}",
        )

    outlier = json.loads((ROOT / "artifacts/mali/fold10_qwalk_outlier.json").read_text())
    require(outlier["split"]["seed"] == 1234, "Ma-Li outlier seed changed")
    require(outlier["split"]["fold"] == 10, "Ma-Li outlier fold changed")
    require(outlier["circuit"] == "qwalk-noancilla_indep_qiskit_9",
            "Ma-Li outlier circuit changed")
    require(abs(outlier["predictions_seconds"]["pretrained_osaka_test"] - 36.0312) < 1e-4,
            "Ma-Li pretrained fixture changed")
    require(abs(outlier["predictions_seconds"]["from_scratch_osaka_test"] - 12.8724) < 1e-4,
            "Ma-Li from-scratch fixture changed")

    print("OK: four-track reports, fixtures, provenance lock and Ma-Li seed-1234 diagnostic verified")
    print("This check does not rerun GPU timing, retrain models or contact external services.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
