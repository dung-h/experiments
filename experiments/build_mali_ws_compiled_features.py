#!/usr/bin/env python3
"""Build FakeWashingtonV2/FakeSherbrooke compiled features for Ma--Li QASMs.

This is the simulator-domain half of the Azizov-style compiled transfer
experiment. Labels stay the public Washington/Sherbrooke ``time_taken`` CSVs.
The compile target is the current fake snapshot, not the historical backend
that produced those labels.

Priority jobs are QASMs that also appear in the 340 Osaka/Kyoto rows, so the
same-circuit transfer evaluator can start before the remaining 1,170 circuits
finish.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from mali_family_ood_common import (
    LOGICAL_FEATURE_KEYS,
    PHYSICAL_FEATURE_KEYS,
    filename_family,
    group_of,
    sha256_file,
    two_qubit_depth,
    width_bucket,
    width_from_name,
)

FIELDS = [
    "circuit",
    "backend",
    "qasm_sha256",
    "family",
    "family_group",
    "logical_width",
    "width_bucket",
    "hw_sampled",
    "priority",
    *LOGICAL_FEATURE_KEYS,
    *PHYSICAL_FEATURE_KEYS,
    "unsupported_duration_ops",
    "transpile_seed",
    "optimization_level",
    "status",
    "error",
    "wall_seconds",
    "fake_backend_class",
    "fake_backend_dt_seconds",
]

_WORKER_BACKENDS = {}
_WORKER_META = {}


def resolve_mali_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    for candidate in (ROOT / "work" / "mali", ROOT.parent / "Quantum-Execution-Time-Prediction"):
        if (candidate / "data" / "quantum_circuits").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Ma-Li checkout with data/quantum_circuits not found")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _init_worker() -> None:
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeWashingtonV2

    global _WORKER_BACKENDS, _WORKER_META
    backends = {"washington": FakeWashingtonV2(), "sherbrooke": FakeSherbrooke()}
    _WORKER_BACKENDS = backends
    _WORKER_META = {
        name: {
            "class": type(backend).__name__,
            "dt_seconds": float(backend.dt),
            "num_qubits": int(backend.num_qubits),
        }
        for name, backend in backends.items()
    }


def _physical_worker(payload: dict[str, object]) -> dict[str, object]:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd() / "experiments"))
    from qiskit import QuantumCircuit, transpile
    from mali_family_ood_common import logical_features, two_qubit_depth
    from mali_qcre_proxy import weighted_critical_path

    started = time.time()
    backend_name = str(payload["backend"])
    seed = int(payload["seed"])
    optimization_level = int(payload["optimization_level"])
    row = {
        "circuit": payload["circuit"],
        "backend": backend_name,
        "qasm_sha256": payload["qasm_sha256"],
        "family": payload["family"],
        "family_group": payload["family_group"],
        "logical_width": payload["logical_width"],
        "width_bucket": payload["width_bucket"],
        "hw_sampled": payload["hw_sampled"],
        "priority": payload["priority"],
        "transpile_seed": seed,
        "optimization_level": optimization_level,
        "status": "ok",
        "error": "",
        "unsupported_duration_ops": "",
        "fake_backend_class": _WORKER_META[backend_name]["class"],
        "fake_backend_dt_seconds": _WORKER_META[backend_name]["dt_seconds"],
    }
    for key in LOGICAL_FEATURE_KEYS:
        row[key] = payload.get(key, "")
    try:
        original = QuantumCircuit.from_qasm_file(str(payload["abs_path"]))
        backend = _WORKER_BACKENDS[backend_name]
        compiled = transpile(
            original,
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=seed,
        )
        weighted_dt, unsupported = weighted_critical_path(compiled, backend)
        counts = compiled.count_ops()
        physical_two = sum(
            count for name, count in counts.items() if name in {"cx", "cz", "ecr", "swap"}
        )
        logical_depth = float(row["logical_depth"] or original.depth() or 0)
        logical_two = float(row["logical_two_qubit_count"] or 0.0)
        physical_depth = float(compiled.depth() or 0)
        row.update(
            {
                "logical_width": int(original.num_qubits),
                "logical_depth": float(original.depth() or 0),
                "physical_depth": physical_depth,
                "physical_two_qubit_depth": float(two_qubit_depth(compiled)),
                "physical_two_qubit_gate_count": float(physical_two),
                "physical_swap_count": float(counts.get("swap", 0)),
                "physical_gate_count": float(len(compiled.data)),
                "qcre_weighted_critical_path_seconds": float(weighted_dt) * float(backend.dt),
                "routing_depth_ratio": (physical_depth / logical_depth) if logical_depth else 0.0,
                "routing_two_qubit_ratio": (float(physical_two) / logical_two) if logical_two else 0.0,
                "unsupported_duration_ops": ",".join(sorted(set(unsupported))),
            }
        )
        row.update(logical_features(original))
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["error_trace"] = traceback.format_exc(limit=3)
        for key in PHYSICAL_FEATURE_KEYS:
            row.setdefault(key, "")
    row["wall_seconds"] = round(time.time() - started, 4)
    return row


def index_qasms(mali_root: Path) -> list[dict[str, object]]:
    directory = mali_root / "data" / "quantum_circuits"
    rows = []
    for path in sorted(directory.glob("*.qasm")):
        family = filename_family(path.name)
        width = width_from_name(path.name)
        rows.append(
            {
                "circuit": path.stem,
                "abs_path": path,
                "family": family,
                "family_group": group_of(family),
                "logical_width": width,
                "width_bucket": width_bucket(width),
            }
        )
    return rows


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
        default=ROOT / "artifacts/validation/mali_azizov_compiled_transfer_v1",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--priority-only", action="store_true")
    args = parser.parse_args()

    mali_root = resolve_mali_root(args.mali_root)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "washington_sherbrooke_compiled_features.csv"
    progress_path = output_dir / "compile_progress.json"

    qpu_rows = read_csv(args.qpu_manifest)
    priority_names = {row["circuit"] for row in qpu_rows}
    qasms = index_qasms(mali_root)

    done_rows = read_csv(csv_path)
    done_keys = {(row["circuit"], row["backend"]) for row in done_rows if row.get("status")}
    jobs = []
    backends = ("washington", "sherbrooke")
    for item in qasms:
        priority = item["circuit"] in priority_names
        if args.priority_only and not priority:
            continue
        digest = sha256_file(item["abs_path"])
        for backend in backends:
            key = (item["circuit"], backend)
            if key in done_keys:
                continue
            jobs.append(
                {
                    **item,
                    "backend": backend,
                    "qasm_sha256": digest,
                    "hw_sampled": str(priority),
                    "priority": str(priority),
                    "seed": args.seed,
                    "optimization_level": args.optimization_level,
                }
            )
    jobs.sort(key=lambda job: (job["priority"] != "True", int(job["logical_width"]), job["circuit"], job["backend"]))

    total_target = len(done_rows) + len(jobs)
    started = time.time()
    print(
        f"resume={len(done_rows)} pending={len(jobs)} priority_pending="
        f"{sum(1 for job in jobs if job['priority']=='True')} mali_root={mali_root}",
        flush=True,
    )
    if not jobs:
        progress_path.write_text(json.dumps({"done": len(done_rows), "pending": 0, "status": "complete"}, indent=2))
        print("nothing pending", flush=True)
        return 0

    completed = list(done_rows)
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=ctx, initializer=_init_worker) as pool:
        futures = [pool.submit(_physical_worker, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            completed.append(row)
            if index % 10 == 0 or index == len(jobs):
                write_csv(csv_path, completed)
                ok = sum(1 for item in completed if item.get("status") == "ok")
                err = sum(1 for item in completed if item.get("status") != "ok")
                pri_done = sum(1 for item in completed if str(item.get("priority")) == "True")
                payload = {
                    "done": len(completed),
                    "target": total_target,
                    "pending": len(jobs) - index,
                    "ok": ok,
                    "error": err,
                    "priority_done": pri_done,
                    "last_circuit": f"{row.get('circuit')}@{row.get('backend')}",
                    "last_status": row.get("status"),
                    "elapsed_s": round(time.time() - started, 1),
                }
                progress_path.write_text(json.dumps(payload, indent=2))
                print(
                    f"[{index}/{len(jobs)}] ok={ok} err={err} pri={pri_done} "
                    f"{payload['last_circuit']} {row.get('status')} {row.get('wall_seconds')}s",
                    flush=True,
                )

    write_csv(csv_path, completed)
    ok = sum(1 for item in completed if item.get("status") == "ok")
    summary = {
        "n_rows": len(completed),
        "n_ok": ok,
        "n_error": len(completed) - ok,
        "priority_ok": sum(1 for item in completed if item.get("status") == "ok" and str(item.get("priority")) == "True"),
        "optimization_level": args.optimization_level,
        "seed": args.seed,
        "proxy_semantics": (
            "current FakeWashingtonV2/FakeSherbrooke transpile + target-duration "
            "weighted path; labels remain Ma-Li washington/sherbrooke time_taken"
        ),
        "elapsed_s": round(time.time() - started, 1),
        "csv": str(csv_path),
    }
    (output_dir / "compile_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
