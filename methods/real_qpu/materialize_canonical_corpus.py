#!/usr/bin/env python3
"""Materialize the source-preserving canonical 8,767-row corpus (C14)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "benchmark_v1/registry/data_registry.json"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def raw_json(row: dict) -> str:
    return json.dumps({str(k): json_value(v) for k, v in row.items()}, sort_keys=True, separators=(",", ":"), default=str)


def split_for(path: Path) -> pd.DataFrame:
    result = pd.read_csv(path)
    if "source_row_index" not in result:
        raise ValueError(f"split lacks source_row_index: {path}")
    return result.reset_index(drop=True)


def source_frame(source: dict, split: pd.DataFrame) -> pd.DataFrame:
    data_path = Path(source["path"])
    raw = pd.read_parquet(data_path).reset_index(drop=True)
    if len(raw) != len(split):
        raise ValueError(f"{source['id']} data/split mismatch: {len(raw)} != {len(split)}")
    if not (split["source_row_index"].astype(int).to_numpy() == range(len(raw))).all():
        raise ValueError(f"{source['id']} split ordering is not source row order")
    source_id = source["id"]
    if source_id == "mali_real_qpu":
        target = raw["time_taken"].astype(float)
        native = raw["time_taken"].astype(float)
        family = raw["algorithm_family"].astype(str)
        backend = raw["backend"].astype(str)
        width = raw["num_qubits"].astype(int)
        depth = pd.Series([None] * len(raw))
        twoq = raw["two_qubit_gates"].astype(int)
        qasm_exact = True
        workflow = pd.Series([None] * len(raw))
    elif source_id == "qonductor_single_circuit_ibm":
        target = raw["taken_time_seconds"].astype(float)
        native = raw["taken_time_seconds"].astype(float)
        family = raw["primary_algorithm_family"].astype(str)
        backend = raw["backend_name"].astype(str)
        width = raw["sum_qubits"].astype(int)
        depth = raw["sum_depth"].astype(int)
        twoq = raw["sum_2q_gates"].astype(int)
        qasm_exact = True
        workflow = pd.Series([None] * len(raw))
    elif source_id == "qpack_mcp":
        target = raw["circuit_execution_duration_ms"].astype(float) / 1000.0
        native = raw["circuit_execution_duration_ms"].astype(float)
        family = raw["problem"].astype(str)
        backend = raw["backend_name"].astype(str)
        width = raw["num_qubits"].astype(int)
        depth = raw["depth"].astype(float)
        twoq = pd.Series([None] * len(raw))
        qasm_exact = False
        workflow = split["workflow_group_hash"].astype(str)
    else:
        raise ValueError(source_id)

    metadata = pd.DataFrame({
        "source_id": source_id,
        "source_row_index": range(len(raw)),
        "canonical_row_id": [f"{source_id}|row{i}" for i in range(len(raw))],
        "source_row_id": split["row_id"].astype(str),
        "identity_kind": split["identity_kind"].astype(str),
        "qasm_path_or_member": split["qasm_path_or_member"],
        "qasm_bytes_sha256": split["qasm_bytes_sha256"],
        "circuit_group_hash": split["circuit_group_hash"],
        "workflow_group_hash": workflow,
        "source_primary_fold": split["primary_fold"],
        "workflow_fold": split.get("workflow_fold", pd.Series([None] * len(raw))),
        "family": family,
        "family_component": split["family_component_holdout_id"].astype(str),
        "backend": backend,
        "width_qubits": width,
        "depth": depth,
        "two_qubit_gates": twoq,
        "shots": raw["shots"].astype(int),
        "target_seconds": target,
        "target_native": native,
        "target_native_unit": "milliseconds" if source_id == "qpack_mcp" else "seconds",
        "runtime_semantics": raw["runtime_semantics"].astype(str),
        "observation_unit": raw["observation_unit"].astype(str),
        "backend_holdout_id": split["backend_holdout_id"].astype(str),
        "timestamp_utc": split["timestamp_utc"],
        "qasm_exact": qasm_exact,
        "qasm_dependent_method_eligible": split["qasm_dependent_method_eligible"].astype(bool),
        "primary_benchmark_eligible": split["primary_benchmark_eligible"].astype(bool),
    })
    raw_payload = [raw_json(row) for row in raw.to_dict("records")]
    metadata["raw_row_json"] = raw_payload
    metadata["raw_row_sha256"] = [hashlib.sha256(value.encode()).hexdigest() for value in raw_payload]
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    all_frames = []
    source_summary = []
    split_map = {
        "mali_real_qpu": ROOT / "benchmark_v1/splits/mali_real_qpu.csv",
        "qonductor_single_circuit_ibm": ROOT / "benchmark_v1/splits/qonductor_single_circuit_ibm.csv",
        "qpack_mcp": ROOT / "benchmark_v1/splits/qpack_workflow_secondary.csv",
    }
    for source in registry["sources"]:
        if source["id"] not in split_map:
            continue
        split = split_for(split_map[source["id"]])
        frame = source_frame(source, split)
        all_frames.append(frame)
        source_summary.append({"source_id": source["id"], "rows": len(frame), "source_path": source["path"], "source_sha256": sha(Path(source["path"])), "split_path": str(split_map[source["id"]].relative_to(ROOT)), "split_sha256": sha(split_map[source["id"]])})
    corpus = pd.concat(all_frames, ignore_index=True)
    if len(corpus) != 8767:
        raise ValueError(f"canonical corpus expected 8767 rows, got {len(corpus)}")
    if corpus["canonical_row_id"].duplicated().any():
        raise ValueError("canonical row IDs are duplicated")
    parquet = args.output_dir / "canonical_observations.parquet"
    csv_path = args.output_dir / "canonical_observations.csv"
    corpus.to_parquet(parquet, index=False)
    corpus.to_csv(csv_path, index=False)
    for source_id, frame in corpus.groupby("source_id", sort=True):
        frame.to_csv(args.output_dir / f"{source_id}.csv", index=False)
    manifest = {"corpus_id": "canonical-archived-real-qpu-v1-8767", "protocol_id": "qre-benchmark-v1", "rows": len(corpus), "source_rows": {row["source_id"]: row["rows"] for row in source_summary}, "source_summary": source_summary, "identity_policy": "source_row_id plus exact QASM hash where available; QPack workflow hash only", "target_unit": "seconds", "qpack_target_conversion": "circuit_execution_duration_ms / 1000", "outputs": {"parquet": {"path": parquet.name, "sha256": sha(parquet)}, "csv": {"path": csv_path.name, "sha256": sha(csv_path)}}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
