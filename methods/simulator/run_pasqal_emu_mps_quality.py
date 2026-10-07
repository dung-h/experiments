#!/usr/bin/env python3
"""Run the source-pinned Pasqal EMU-MPS quality pilot.

The execution clock contains a run with no result observable and therefore no
state copy or fidelity callback. State capture and overlap are repeated after
timing as separately recorded quality-control work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SCRIPTS = ROOT / "tracks/pasqal_emu_mps/scripts"
sys.path.insert(0, str(SOURCE_SCRIPTS))
from official_source_programs import source_sequence

DEFAULT_MANIFEST = ROOT / "benchmark_v1/execution/manifests/pasqal_emu_mps_official_source_pilot.json"


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def memtotal_bytes() -> int | None:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value


def descriptor(manifest: dict[str, Any], family: str, rows: int, columns: int) -> dict[str, Any]:
    source = manifest["source_programs"]
    program = SOURCE_SCRIPTS / "official_source_programs.py"
    sequence = source_sequence(family, rows, columns)
    duration_ns = int(sequence.get_duration())
    dt_ns = float(manifest["simulator"]["dt_ns"])
    return {
        "source_repository": source["repository"], "source_commit": source["commit"],
        "source_document_sha256": source["document_sha256"],
        "reconstruction_module_sha256": sha256(program.read_bytes()),
        "family": family, "rows": rows, "columns": columns,
        "sequence_duration_ns": duration_ns,
        "num_steps_estimate": int(round(duration_ns / dt_ns)),
        "step_count_definition": "round(Pulser Sequence.get_duration() / configured EMU-MPS dt_ns); record for per-step formula response",
        "canonical_tensor_ordering": "Pulser Register.rectangle order; optimize_qubit_ordering=false",
    }


def config(manifest: dict[str, Any], max_bond: int, observables: list[Any]) -> Any:
    from emu_mps import MPSConfig
    sim = manifest["simulator"]
    return MPSConfig(
        dt=sim["dt_ns"], precision=sim["precision"], max_bond_dim=max_bond,
        max_krylov_dim=sim["max_krylov_dim"], num_gpus_to_use=1,
        interaction_cutoff=sim["interaction_cutoff"],
        optimize_qubit_ordering=sim["optimize_qubit_ordering"], observables=observables,
        log_level=40,
    )


def worker(args: argparse.Namespace) -> None:
    import torch
    from emu_mps import MPSBackend, StateResult

    manifest = load(Path(args.input_manifest))
    payload: dict[str, Any] = {"status": "ok", "error": None, "warm_execute_seconds": [], "warm_network_build_seconds": []}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        sequence = source_sequence(args.family, args.rows, args.columns)
        # EMU-MPS observable times are normalized to the sequence interval
        # [0, 1], rather than Pulser's nanosecond duration. `1.0` is the
        # final simulated state for both source programs.
        end_us = 1.0
        quality = manifest["quality"]
        chi256, chi512 = quality["reference_rungs"]

        # Reference validation is deliberate non-target work.
        started = time.perf_counter()
        ref512_result = MPSBackend(sequence, config=config(manifest, chi512, [StateResult(evaluation_times=[end_us])])).run()
        payload["reference_512_seconds"] = time.perf_counter() - started
        ref512 = ref512_result.get_result("state", end_us)
        started = time.perf_counter()
        ref256_result = MPSBackend(sequence, config=config(manifest, chi256, [StateResult(evaluation_times=[end_us])])).run()
        payload["reference_256_seconds"] = time.perf_counter() - started
        ref256 = ref256_result.get_result("state", end_us)
        started = time.perf_counter()
        reference_fidelity = float(ref512.overlap(ref256))
        payload["reference_overlap_seconds"] = time.perf_counter() - started
        payload["reference_fidelity_256_vs_512"] = reference_fidelity
        payload["quality_reference_hash"] = digest({"descriptor": descriptor(manifest, args.family, args.rows, args.columns), "reference_rungs": [chi256, chi512], "metric": "Fidelity"})
        if reference_fidelity < 0.9999:
            payload.update(status="quality_failed", error=f"reference non-convergent: chi256-vs-chi512 fidelity={reference_fidelity}")
        else:
            # First construction and execution clocks are explicitly split.
            # The timed runs have no observables: no state copy or Fidelity
            # callback is part of this execution-runtime estimand.
            started = time.perf_counter()
            backend = MPSBackend(sequence, config=config(manifest, args.max_bond, []))
            payload["network_build_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            backend.run()
            payload["first_execute_seconds"] = time.perf_counter() - started
            for _ in range(args.warmups):
                MPSBackend(sequence, config=config(manifest, args.max_bond, [])).run()
            for _ in range(args.repetitions):
                started = time.perf_counter()
                warm_backend = MPSBackend(sequence, config=config(manifest, args.max_bond, []))
                payload["warm_network_build_seconds"].append(time.perf_counter() - started)
                started = time.perf_counter()
                warm_backend.run()
                payload["warm_execute_seconds"].append(time.perf_counter() - started)
            # Run once more only for quality. The clock above has already
            # stopped; state capture and direct overlap remain visible here.
            started = time.perf_counter()
            candidate_result = MPSBackend(sequence, config=config(manifest, args.max_bond, [StateResult(evaluation_times=[end_us])])).run()
            payload["candidate_quality_state_capture_seconds"] = time.perf_counter() - started
            candidate_state = candidate_result.get_result("state", end_us)
            started = time.perf_counter()
            candidate_fidelity = float(ref256.overlap(candidate_state))
            payload["candidate_overlap_seconds"] = time.perf_counter() - started
            payload["candidate_fidelity"] = candidate_fidelity
            payload["quality_status"] = "ok" if candidate_fidelity >= quality["candidate_threshold"] else "quality_failed"
    except MemoryError as exc:
        payload.update(status="resource_limit", error=f"MemoryError: {exc}")
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}".splitlines()[0][:1000]
        status = "unsupported" if "cuda unavailable" in text.lower() else "resource_limit" if "out of memory" in text.lower() else "error"
        payload.update(status=status, error=text)
        payload["error_trace"] = traceback.format_exc(limit=8)
    payload["torch_cuda_available"] = bool(torch.cuda.is_available())
    Path(args.worker_output).write_text(json.dumps(payload, sort_keys=True))


def environment(manifest: dict[str, Any]) -> dict[str, Any]:
    import emu_mps, pulser, torch
    gpu = command_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"])
    fields = [x.strip() for x in gpu.splitlines()[0].split(",")] if not gpu.startswith("unavailable:") else []
    freeze = command_output([sys.executable, "-m", "pip", "freeze", "--all"])
    sim, quality = manifest["simulator"], manifest["quality"]
    source_cases = [
        descriptor(manifest, family, rows, columns)
        for family in manifest["source_programs"]["families"]
        for rows, columns in manifest["source_programs"]["layouts"]
    ]
    return {
        "host_id": platform.node(), "os": platform.platform(),
        "cpu_model": command_output(["bash", "-lc", "lscpu | awk -F: '/Model name/{sub(/^ /, \"\", $2); print $2; exit}'"]),
        "ram_bytes": memtotal_bytes(), "python_version": sys.version, "package_lock_sha256": sha256(freeze.encode()),
        "thread_policy": "one sequential EMU-MPS GPU worker; no concurrent timing workload",
        "gpu_model": fields[0] if len(fields)>0 else gpu, "gpu_uuid_or_redacted_stable_id": fields[1] if len(fields)>1 else "unavailable",
        "vram_bytes": int(fields[2].replace(" MiB", ""))*1024*1024 if len(fields)>2 and fields[2].endswith(" MiB") else None,
        "driver_version": fields[3] if len(fields)>3 else "unavailable", "cuda_version": torch.version.cuda, "gpu_concurrency": 1,
        "emu_mps_version": getattr(emu_mps, "__version__", "unknown"), "pulser_version": getattr(pulser, "__version__", "unknown"), "torch_version": torch.__version__,
        "precision": sim["precision"], "max_bond_dimension": ",".join(map(str, sim["max_bond_dimensions"])), "cutoff": sim["interaction_cutoff"],
        "pulse_hash": digest(source_cases), "atom_layout_hash": digest(manifest["source_programs"]["layouts"]),
        "quality_reference_hash": digest({"source": manifest["source_programs"], "reference": quality["reference_rungs"]}), "quality_threshold": quality["candidate_threshold"],
        "sessions": manifest["measurement"]["sessions"], "timed_warm_repetitions_per_session": manifest["measurement"]["timed_warm_repetitions_per_session"],
    }


def base(run_id: str, manifest: dict[str, Any], family: str, rows: int, columns: int, bond: int, session: str) -> dict[str, Any]:
    spec = descriptor(manifest, family, rows, columns)
    pulse_hash = digest(spec)
    return {
        "protocol_id": "qre-benchmark-v1", "run_id": run_id, "row_id": f"{family}|{rows}x{columns}|chi{bond}", "circuit_hash": pulse_hash,
        "context_id": f"pasqal-emu-mps-gpu-chi{bond}-precision{manifest['simulator']['precision']:g}", "session_id": session,
        "seed_record": {"seed_not_required": "fixed source-pinned deterministic pulse program"}, "input_hashes": {"pulse_hash": pulse_hash, "atom_layout_hash": digest({"rows":rows,"columns":columns})},
        "metadata": {**spec, "pulse_hash": pulse_hash, "atom_layout_hash": digest({"rows":rows,"columns":columns}), "max_bond_dimension": bond, "quality_threshold": manifest["quality"]["candidate_threshold"], "execution_clock": "host perf_counter around MPSBackend.run with observables=[]; constructor clock separate; state capture/overlap are post-timer quality work"},
    }


def run_parent(args: argparse.Namespace) -> None:
    manifest, artifact = load(args.input_manifest), args.artifact_dir.resolve()
    raw_path = artifact / "raw_records.jsonl"
    if not raw_path.exists() or not (artifact / "run_manifest.json").exists(): raise ValueError("initialize run first")
    measure = manifest["measurement"]
    if (args.sessions,args.warmups,args.repetitions) != (measure["sessions"],measure["untimed_warmups_per_session"],measure["timed_warm_repetitions_per_session"]): raise ValueError("measurement values must equal frozen manifest")
    (artifact/"environment.json").write_text(json.dumps(environment(manifest),indent=2,sort_keys=True)+"\n")
    payload_dir=artifact/"worker_payloads"; payload_dir.mkdir(exist_ok=True)
    done={(x["row_id"],x["context_id"],x["session_id"]) for x in (json.loads(line) for line in raw_path.read_text().splitlines() if line) if x["stage"]=="first_execute"}
    for family in manifest["source_programs"]["families"]:
      for rows,columns in manifest["source_programs"]["layouts"]:
       for bond in manifest["simulator"]["max_bond_dimensions"]:
        for index in range(args.sessions):
            session=f"session-{index+1}"; record=base(args.run_id,manifest,family,rows,columns,bond,session); key=(record["row_id"],record["context_id"],session)
            if key in done: continue
            payload_path=payload_dir/f"{family}_{rows}x{columns}_chi{bond}_{session}.json"
            command=[sys.executable,str(Path(__file__).resolve()),"--worker","--input-manifest",str(args.input_manifest),"--family",family,"--rows",str(rows),"--columns",str(columns),"--max-bond",str(bond),"--warmups",str(args.warmups),"--repetitions",str(args.repetitions),"--worker-output",str(payload_path)]
            print(f"[pasqal-mps] {record['row_id']} {session}",flush=True)
            try:
                process=subprocess.run(command,text=True,capture_output=True,timeout=args.cell_timeout,check=False)
                payload=json.loads(payload_path.read_text()) if payload_path.exists() else {"status":"error","error":f"worker exit={process.returncode}; {process.stderr[-600:]}"}
            except subprocess.TimeoutExpired: payload={"status":"timeout","error":f"cell exceeded frozen timeout {args.cell_timeout}s"}
            status=payload.get("status","error")
            if status=="ok": status=payload.get("quality_status","error")
            quality={key:payload.get(key) for key in ("reference_fidelity_256_vs_512","candidate_fidelity","quality_reference_hash","reference_512_seconds","reference_256_seconds","reference_overlap_seconds","candidate_quality_state_capture_seconds","candidate_overlap_seconds")}
            rows_to_write=[{**record,"stage":"network_build","repetition":0,"status":status,"elapsed_seconds":payload.get("network_build_seconds"),"quality":quality,"error":payload.get("error")}, {**record,"stage":"first_execute","repetition":0,"status":status,"elapsed_seconds":payload.get("first_execute_seconds"),"quality":quality,"error":payload.get("error")}]
            warms=payload.get("warm_execute_seconds",[]) if isinstance(payload.get("warm_execute_seconds"),list) else []
            for repetition in range(args.repetitions): rows_to_write.append({**record,"stage":"warm_execute","repetition":repetition,"observation_kind":"raw_repetition","status":status,"elapsed_seconds":warms[repetition] if repetition<len(warms) else None,"quality":quality,"error":payload.get("error")})
            with raw_path.open("a") as handle:
                for item in rows_to_write: handle.write(json.dumps(item,sort_keys=True)+"\n")


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--worker",action="store_true");p.add_argument("--input-manifest",type=Path,default=DEFAULT_MANIFEST);p.add_argument("--write-environment",type=Path);p.add_argument("--artifact-dir",type=Path);p.add_argument("--run-id");p.add_argument("--sessions",type=int,default=1);p.add_argument("--warmups",type=int,default=3);p.add_argument("--repetitions",type=int,default=1);p.add_argument("--cell-timeout",type=int,default=900);p.add_argument("--family");p.add_argument("--rows",type=int);p.add_argument("--columns",type=int);p.add_argument("--max-bond",type=int);p.add_argument("--worker-output")
    a=p.parse_args()
    if a.worker:
        if any(getattr(a,k) is None for k in ("family","rows","columns","max_bond","worker_output")): p.error("worker arguments missing")
    elif a.write_environment is None and (a.artifact_dir is None or not a.run_id): p.error("--artifact-dir and --run-id required")
    return a


if __name__ == "__main__":
    args=parse_args()
    if args.worker: worker(args)
    elif args.write_environment is not None:
        args.write_environment.parent.mkdir(parents=True,exist_ok=True);args.write_environment.write_text(json.dumps(environment(load(args.input_manifest)),indent=2,sort_keys=True)+"\n")
    else: run_parent(args)
