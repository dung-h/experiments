#!/usr/bin/env python3
"""Build FakeWashingtonV2/FakeSherbrooke native DAGs for Ma-Li QASMs.

This is the Azizov substitution in Ma-Li's encoder: transpile to the target
backend, then store the native-operation DAG. Labels remain the public
Washington/Sherbrooke time_taken CSVs and are not written into the NPZ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from build_mali_physical_dag import (  # noqa: E402
    build_graph,
    graph_id,
    layout_metadata,
    save_graph,
    snapshot_id,
    snapshot_payload,
)
from mali_family_ood_common import filename_family, sha256_file  # noqa: E402

_WORKER_BACKENDS = {}
_WORKER_SNAPSHOTS = {}
_WORKER_T1T2 = {}


def resolve_mali_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    for candidate in (ROOT / "work" / "mali", ROOT.parent / "Quantum-Execution-Time-Prediction"):
        if (candidate / "data" / "quantum_circuits").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Ma-Li checkout not found")


def qubit_t1_t2(backend, n_qubits: int = 127):
    import numpy as np

    t1 = np.zeros(n_qubits, dtype=np.float32)
    t2 = np.zeros(n_qubits, dtype=np.float32)
    properties = backend.properties()
    for index in range(int(backend.num_qubits)):
        t1[index] = float(properties.t1(index))
        t2[index] = float(properties.t2(index))
    return t1, t2


def _init_worker() -> None:
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeWashingtonV2

    global _WORKER_BACKENDS, _WORKER_SNAPSHOTS, _WORKER_T1T2
    backends = {"washington": FakeWashingtonV2(), "sherbrooke": FakeSherbrooke()}
    _WORKER_BACKENDS = backends
    _WORKER_SNAPSHOTS = {}
    _WORKER_T1T2 = {}
    for name, backend in backends.items():
        payload = snapshot_payload(backend)
        _WORKER_SNAPSHOTS[name] = (snapshot_id(payload), payload)
        _WORKER_T1T2[name] = qubit_t1_t2(backend)


def _worker(payload: dict) -> dict:
    from qiskit import QuantumCircuit, transpile

    started = time.time()
    backend_name = payload["backend"]
    seed = int(payload["seed"])
    optimization_level = int(payload["optimization_level"])
    snapshot_digest, _ = _WORKER_SNAPSHOTS[backend_name]
    gid = graph_id(payload["qasm_sha256"], backend_name, snapshot_digest, seed, optimization_level)
    graph_path = Path(payload["graph_path"])
    row = {
        "circuit": payload["circuit"],
        "backend": backend_name,
        "qasm_sha256": payload["qasm_sha256"],
        "graph_id": gid,
        "snapshot_id": snapshot_digest,
        "priority": payload["priority"],
        "status": "ok",
        "error": "",
        "graph_path": str(graph_path),
    }
    try:
        if graph_path.exists():
            import numpy as np

            with np.load(graph_path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
            row["status"] = "cached"
        else:
            original = QuantumCircuit.from_qasm_file(str(payload["abs_path"]))
            compiled = transpile(
                original,
                backend=_WORKER_BACKENDS[backend_name],
                optimization_level=optimization_level,
                seed_transpiler=seed,
            )
            arrays, metadata = build_graph(original, compiled, _WORKER_BACKENDS[backend_name])
            t1, t2 = _WORKER_T1T2[backend_name]
            arrays["qubit_t1"] = t1
            arrays["qubit_t2"] = t2
            metadata.update(
                {
                    "graph_id": gid,
                    "backend": backend_name,
                    "qasm_sha256": payload["qasm_sha256"],
                    "circuit": payload["circuit"],
                    "snapshot_id": snapshot_digest,
                    "seed_transpiler": seed,
                    "optimization_level": optimization_level,
                    "provenance_class": "CURRENT_FAKE_SNAPSHOT_PROXY",
                    "layout": layout_metadata(original, compiled),
                    "fake_backend_class": type(_WORKER_BACKENDS[backend_name]).__name__,
                }
            )
            save_graph(graph_path, arrays, metadata)
            row["status"] = "built"
        row.update(
            {
                "n_nodes": metadata["n_nodes"],
                "n_edges": metadata["n_edges"],
                "active_physical_width": metadata["active_physical_width"],
                "compiled_depth": metadata["compiled_depth"],
                "native_two_qubit_gate_count": metadata.get("native_two_qubit_gate_count", ""),
                "qcre_critical_path_seconds": metadata.get("qcre_critical_path_seconds", ""),
            }
        )
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["error_trace"] = traceback.format_exc(limit=4)
    row["wall_seconds"] = round(time.time() - started, 4)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mali-root", type=str, default=None)
    parser.add_argument(
        "--qpu-manifest",
        type=Path,
        default=ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "work/mali_azizov_transpiled_dag_v1",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "artifacts/validation/mali_azizov_transpiled_dag_v1",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--priority-only", action="store_true")
    args = parser.parse_args()

    mali_root = resolve_mali_root(args.mali_root)
    qasm_dir = mali_root / "data" / "quantum_circuits"
    priority = set()
    if args.qpu_manifest.is_file():
        with args.qpu_manifest.open(newline="", encoding="utf-8") as handle:
            priority = {row["qasm_sha256"] for row in csv.DictReader(handle)}

    jobs = []
    for path in sorted(qasm_dir.glob("*.qasm")):
        digest = sha256_file(path)
        is_priority = digest in priority
        if args.priority_only and not is_priority:
            continue
        for backend in ("washington", "sherbrooke"):
            gid_placeholder = hashlib.sha256(f"{digest}|{backend}".encode()).hexdigest()[:24]
            jobs.append(
                {
                    "circuit": path.stem,
                    "abs_path": str(path),
                    "qasm_sha256": digest,
                    "backend": backend,
                    "priority": is_priority,
                    "seed": args.seed,
                    "optimization_level": args.optimization_level,
                    "family": filename_family(path.name),
                    "graph_path": "",  # filled after snapshot ids exist
                    "_digest": digest,
                    "_gid_placeholder": gid_placeholder,
                }
            )

    jobs.sort(key=lambda item: (not item["priority"], item["circuit"], item["backend"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = args.output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    backend_dir = args.output_dir / "backend"
    backend_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot ids once in the parent so graph_id matches workers.
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeWashingtonV2

    backends = {"washington": FakeWashingtonV2(), "sherbrooke": FakeSherbrooke()}
    snapshot_ids = {}
    for name, backend in backends.items():
        payload = snapshot_payload(backend)
        digest = snapshot_id(payload)
        snapshot_ids[name] = digest
        path = backend_dir / f"{name}_{digest}.json"
        if not path.exists():
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for job in jobs:
        gid = graph_id(job["qasm_sha256"], job["backend"], snapshot_ids[job["backend"]], args.seed, args.optimization_level)
        job["graph_path"] = str(graph_dir / f"{gid}.npz")
        job["graph_id"] = gid

    csv_path = args.artifact_dir / "ws_graph_records.csv"
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    existing = {}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") in {"ok", "built", "cached"}:
                    existing[row["graph_id"]] = row

    pending = [job for job in jobs if job["graph_id"] not in existing or not Path(job["graph_path"]).exists()]
    rows = list(existing.values())
    started = time.time()
    progress_path = args.artifact_dir / "build_progress.json"

    def dump_progress(done: int, last: dict | None = None) -> None:
        payload = {
            "done": done,
            "target": len(jobs),
            "pending": len(pending),
            "ok": sum(1 for row in rows if row.get("status") in {"ok", "built", "cached"}),
            "error": sum(1 for row in rows if row.get("status") == "error"),
            "elapsed_s": round(time.time() - started, 1),
        }
        if last:
            payload["last_circuit"] = f"{last.get('circuit')}@{last.get('backend')}"
            payload["last_status"] = last.get("status")
            payload["last_nodes"] = last.get("n_nodes")
        progress_path.write_text(json.dumps(payload, indent=2) + "\n")

    dump_progress(len(jobs) - len(pending))
    if pending:
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context, initializer=_init_worker) as pool:
            futures = {pool.submit(_worker, job): job for job in pending}
            finished = 0
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                finished += 1
                if finished % 10 == 0 or row.get("status") == "error":
                    write_csv(csv_path, rows)
                    dump_progress(len(jobs) - len(pending) + finished, row)
                    print(
                        f"ws-dag {len(jobs) - len(pending) + finished}/{len(jobs)} "
                        f"{row.get('status')} {row.get('circuit')}@{row.get('backend')} "
                        f"nodes={row.get('n_nodes')} {row.get('error','')[:120]}",
                        flush=True,
                    )

    write_csv(csv_path, rows)
    ok_rows = [row for row in rows if row.get("status") in {"ok", "built", "cached"}]
    summary = {
        "n_jobs": len(jobs),
        "n_ok": len(ok_rows),
        "n_error": sum(1 for row in rows if row.get("status") == "error"),
        "n_priority_ok": sum(1 for row in ok_rows if str(row.get("priority")).lower() in {"true", "1"}),
        "elapsed_s": round(time.time() - started, 1),
        "optimization_level": args.seed and args.optimization_level,
        "seed": args.seed,
        "graph_semantics": "transpiled native-operation DAG, current FakeWashingtonV2/FakeSherbrooke",
        "csv": str(csv_path),
    }
    (args.artifact_dir / "ws_build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    dump_progress(len(jobs), ok_rows[-1] if ok_rows else None)
    print(json.dumps(summary, indent=2))
    return 0 if summary["n_error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
