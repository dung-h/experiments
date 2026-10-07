#!/usr/bin/env python3
"""Strict R9 wrapper for the frozen joint runtime/quality MPS fits.

The candidate trainer and aggregator stay byte-pinned and unmodified. This
adapter verifies exact recovered-target/ledger parity, records the manifest
SHA at each fit/aggregate boundary, and writes only under an explicit new
output root. It never performs simulator timing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "work/family_aware_joint_mps_ladder_v1/preflight"
CANDIDATE_ROOT = ROOT
TRAINER = CANDIDATE_ROOT / "benchmark_v1/scripts/train_family_aware_joint_mps_v1.py"
AGGREGATOR = CANDIDATE_ROOT / "benchmark_v1/scripts/aggregate_family_aware_joint_mps_v1.py"
PROTOCOL = ROOT / "benchmark_v1/protocol/family_aware_joint_runtime_quality_mps_v1.json"
CANDIDATE_PROTOCOL = CANDIDATE_ROOT / "benchmark_v1/protocol/family_aware_joint_runtime_quality_mps_v1.json"
SEED_REGISTRY = ROOT / "benchmark_v1/registry/seed_registry.json"
RECOVERY_PLAN = ROOT / "docs/benchmark_recovery_plan.md"
R9_ADJUDICATION: Path | None = None
STATE_PATH: Path | None = None
S85_DIR = ROOT / "benchmark_v1/scripts"
if str(S85_DIR) not in sys.path:
    sys.path.insert(0, str(S85_DIR))
import run_mps_fixed_chi16_runtime_adaptation_v1 as s85  # noqa: E402

from recovered_target_inputs import (  # noqa: E402
    CHI,
    EXPECTED_C44_FOLD_COUNTS,
    EXPECTED_FAMILY_FOLD_COUNTS,
    EXPECTED_HASHES,
    EXPECTED_TARGET_MANIFEST_SHA256,
    JOINT_LEDGER,
    JOINT_SUMMARY,
    RecoveredTargets,
    load_recovered_targets,
    read_csv,
    read_json,
    sha256_file,
    verify_manifest_file,
)


METHODS = ("family_conditioned", "family_agnostic_ablation")
SEEDS = (42, 1234, 31415)
SESSIONS = tuple(f"session-{i}" for i in (1, 2, 3))
EXPECTED_ATTEMPTS = EXPECTED_HASHES * len(CHI) * len(SESSIONS)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _path_label(path: Path) -> str:
    """Use checkout-relative lineage when possible; otherwise preserve the explicit input path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _load_attempts(bundle: RecoveredTargets) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    ledger_sha = verify_manifest_file(bundle, JOINT_LEDGER)
    summary_sha = verify_manifest_file(bundle, JOINT_SUMMARY)
    ledger_path = bundle.root / JOINT_LEDGER
    summary_path = bundle.root / JOINT_SUMMARY
    attempts = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = read_json(summary_path)
    if len(attempts) != EXPECTED_ATTEMPTS or summary.get("attempt_count") != EXPECTED_ATTEMPTS:
        raise ValueError("R9 joint ledger must contain exactly 2,700 terminal hash×chi×session rows")
    if summary.get("scope") != "full" or summary.get("status") != "PASS" or summary.get("timing_performed") is not True:
        raise ValueError("R9 joint derived ledger is not a complete full-panel measurement input")
    if summary.get("training_performed") is not False:
        raise ValueError("R9 materializer ledger must not contain training")
    normalized_ledger_sha = hashlib.sha256(
        (json.dumps(attempts, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    ).hexdigest()
    if summary.get("attempt_ledger_sha256") != normalized_ledger_sha:
        raise ValueError("R9 joint summary normalized-ledger digest mismatch")
    if summary.get("technical_error_count") != 0:
        raise ValueError("R9 joint recovered ledger has technical failures")
    return attempts, summary, ledger_sha, summary_sha


def verify_joint_target_parity(bundle: RecoveredTargets, plan_dir: Path = DEFAULT_PLAN) -> dict[str, Any]:
    """Cross-check all 2,700 attempts, 900 reduced targets and frozen splits."""
    plan_dir = plan_dir.resolve()
    plan_manifest_path = plan_dir / "preflight_manifest.json"
    feature_path = plan_dir / "hash_feature_manifest.csv"
    plan_sha = sha256_file(plan_manifest_path)
    feature_sha = sha256_file(feature_path)
    input_hashes = bundle.manifest.get("input_hashes", {})
    if input_hashes.get("joint_plan_manifest") != plan_sha:
        raise ValueError("R9 materializer pins a different frozen joint plan manifest")
    if input_hashes.get("joint_plan_features") != feature_sha:
        raise ValueError("R9 materializer pins a different frozen joint feature/split table")
    if input_hashes.get("joint_protocol") != sha256_file(PROTOCOL):
        raise ValueError("R9 materializer pins a different joint MPS protocol")
    if sha256_file(CANDIDATE_PROTOCOL) != input_hashes.get("joint_protocol"):
        raise ValueError("candidate trainer/aggregator protocol differs from the R9-pinned protocol")
    plan_manifest = read_json(plan_manifest_path)
    for filename, expected in plan_manifest.get("files", {}).items():
        candidate = plan_dir / filename
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"frozen joint plan file pin mismatch: {filename}")
    feature_rows = read_csv(feature_path)
    feature_by_hash = {row["source_qasm_sha256"]: row for row in feature_rows}
    expected_hashes = set(bundle.fixed_fold_by_hash)
    if len(feature_rows) != EXPECTED_HASHES or len(feature_by_hash) != EXPECTED_HASHES or set(feature_by_hash) != expected_hashes:
        raise ValueError("frozen joint feature manifest must cover the exact 150 target hashes once")

    for digest, row in feature_by_hash.items():
        if int(row["c44_fold"]) != bundle.fixed_fold_by_hash[digest]:
            raise ValueError(f"C44 fold differs between target revision and frozen feature manifest: {digest}")
        if int(row["family_holdout_fold_diagnostic_only"]) != bundle.family_fold_by_hash[digest]:
            raise ValueError(f"family-component fold differs between target revision and frozen feature manifest: {digest}")
    if dict(sorted(Counter(int(row["c44_fold"]) for row in feature_rows).items())) != EXPECTED_C44_FOLD_COUNTS:
        raise ValueError("joint feature table C44 fold counts differ from the frozen 37/28/26/25/34 assignment")
    if dict(sorted(Counter(int(row["family_holdout_fold_diagnostic_only"]) for row in feature_rows).items())) != EXPECTED_FAMILY_FOLD_COUNTS:
        raise ValueError("joint feature table family split counts differ from frozen 32/25/31/31/31 assignment")

    attempts, summary, ledger_sha, summary_sha = _load_attempts(bundle)
    exact_keys: set[tuple[str, int, str]] = set()
    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    ir_by_hash: dict[str, set[str]] = defaultdict(set)
    for row in attempts:
        try:
            digest = str(row["source_qasm_sha256"])
            chi = int(row["max_bond"])
            session = str(row["session_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("R9 joint attempt has malformed exact hash/rung/session identity") from exc
        key = (digest, chi, session)
        if key in exact_keys:
            raise ValueError(f"duplicate R9 joint attempt identity: {key}")
        exact_keys.add(key)
        if digest not in expected_hashes or chi not in CHI or session not in SESSIONS:
            raise ValueError(f"R9 joint attempt outside the frozen 150×6×3 panel: {key}")
        if row.get("attempt_terminal") is not True or row.get("status") != "ok" or row.get("basis_canary_status") != "PASS":
            raise ValueError(f"R9 joint attempt is not a successful terminal canary pass: {key}")
        if int(row.get("c44_fold", -1)) != bundle.fixed_fold_by_hash[digest]:
            raise ValueError(f"attempt C44 fold differs from frozen assignment: {key}")
        warm = row.get("warm_seconds")
        if not isinstance(warm, list) or len(warm) != 5:
            raise ValueError(f"R9 joint attempt must retain five warm samples: {key}")
        try:
            warm_values = [float(value) for value in warm]
            fidelity = float(row["fidelity"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"R9 joint attempt has nonnumeric warm timing/fidelity: {key}") from exc
        if any(not math.isfinite(value) or value < 0 for value in warm_values) or not math.isfinite(fidelity):
            raise ValueError(f"R9 joint attempt has nonfinite or negative target inputs: {key}")
        ir_by_hash[digest].add(str(row.get("ir_sha256", "")))
        by_cell[(digest, chi)].append(row)
    expected_keys = {(digest, chi, session) for digest in expected_hashes for chi in CHI for session in SESSIONS}
    if exact_keys != expected_keys or len(exact_keys) != EXPECTED_ATTEMPTS:
        raise ValueError(f"R9 joint exact attempt-key set mismatch: {len(exact_keys)}/{len(expected_keys)}")
    if any(len(hashes) != 1 or "" in hashes or "None" in hashes for hashes in ir_by_hash.values()):
        raise ValueError("R9 joint normalized circuit IR differs across rung/session for a source hash")

    target_by_key = {(row["source_qasm_sha256"], int(row["max_bond"])): row for row in bundle.joint_rows}
    if len(target_by_key) != EXPECTED_HASHES * len(CHI):
        raise ValueError("R9 joint target table does not have one row per exact hash and rung")
    for key, rows in by_cell.items():
        if len(rows) != 3 or {str(row["session_id"]) for row in rows} != set(SESSIONS):
            raise ValueError(f"R9 joint cell does not have exactly three distinct sessions: {key}")
        ordered = sorted(rows, key=lambda row: str(row["session_id"]))
        session_medians = [statistics.median(float(value) for value in row["warm_seconds"]) for row in ordered]
        expected_runtime = statistics.median(session_medians)
        quality_pass = all(float(row["fidelity"]) >= 0.99 for row in ordered)
        target = target_by_key[key]
        if not math.isclose(float(target["target_runtime_seconds"]), expected_runtime, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"R9 rung runtime differs from the exact median-of-session-medians reducer: {key}")
        if target["target_quality_pass"].strip().lower() != str(quality_pass).lower():
            raise ValueError(f"R9 rung quality differs from the all-three-sessions >=0.99 rule: {key}")
        if int(target["session_count"]) != 3 or not math.isclose(float(target["quality_threshold"]), 0.99, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"R9 rung target changed the session count/quality threshold: {key}")

    # Match the frozen trainer's 150-hash minimum-passing-chi reduction.
    quality_pass_by_hash: dict[str, list[int]] = defaultdict(list)
    for (digest, chi), row in target_by_key.items():
        if row["target_quality_pass"].strip().lower() == "true":
            quality_pass_by_hash[digest].append(chi)
    for digest in expected_hashes:
        expected_minimum = str(min(quality_pass_by_hash[digest])) if quality_pass_by_hash[digest] else "NO_PASS"
        for chi in CHI:
            row = target_by_key[(digest, chi)]
            if str(row["minimum_passing_chi"]) != expected_minimum:
                raise ValueError(f"R9 minimum-passing-chi target is inconsistent for {digest}")

    return {
        "target_manifest_sha256": bundle.manifest_sha256,
        "target_revision_id": bundle.target_revision_id,
        "fixed_target_table_sha256": bundle.fixed_table_sha256,
        "joint_target_table_sha256": bundle.joint_table_sha256,
        "joint_attempt_ledger_sha256": ledger_sha,
        "joint_run_summary_sha256": summary_sha,
        "joint_attempt_rows": len(attempts),
        "joint_attempt_normalized_sha256": summary["attempt_ledger_sha256"],
        "joint_target_rows": len(target_by_key),
        "unique_hashes": len(feature_by_hash),
        "c44_fold_counts": {str(key): value for key, value in EXPECTED_C44_FOLD_COUNTS.items()},
        "family_holdout_fold_counts": {str(key): value for key, value in EXPECTED_FAMILY_FOLD_COUNTS.items()},
        "old_and_r4_runner_lineage": bundle.runner_lineage,
        "plan_manifest_sha256": plan_sha,
        "plan_feature_table_sha256": feature_sha,
        "protocol_sha256": sha256_file(PROTOCOL),
        "candidate_trainer_sha256": sha256_file(TRAINER),
        "candidate_aggregator_sha256": sha256_file(AGGREGATOR),
        "quality_threshold": 0.99,
        "sessions_per_hash_rung": 3,
        "runtime_reducer": "median of five warm repetitions within each session, then median of the three session medians",
        "quality_rule": "rung passes only if all three session fidelities are >= 0.99",
        "training_started": False,
        "gpu_used": False,
    }


def preflight(target_manifest: Path, output_root: Path, plan_dir: Path) -> dict[str, Any]:
    if R9_ADJUDICATION is None or not R9_ADJUDICATION.is_file():
        raise FileNotFoundError("pass --r9-adjudication with the reviewed R9 execution decision")
    if not RECOVERY_PLAN.is_file():
        raise FileNotFoundError(f"checkout is missing the frozen recovery plan: {RECOVERY_PLAN}")
    bundle = load_recovered_targets(target_manifest, expected_manifest_sha256=EXPECTED_TARGET_MANIFEST_SHA256)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite R9 joint output root: {output_root}")
    manifest = verify_joint_target_parity(bundle, plan_dir)
    manifest.update({"artifact_id": "r9-joint-mps-recovered-preflight-v1",
                     "status": "CPU_preflight_PASS_training_not_started",
                     "created_utc": datetime.now(timezone.utc).isoformat(),
                     "adapter_sha256": sha256_file(Path(__file__).resolve()),
                     "execution_authority": {
                         "recovery_plan_path": _path_label(RECOVERY_PLAN),
                         "recovery_plan_sha256": sha256_file(RECOVERY_PLAN),
                         "r9_target_adjudication_path": _path_label(R9_ADJUDICATION),
                         "r9_target_adjudication_sha256": sha256_file(R9_ADJUDICATION),
                     },
                     "target_manifest_path": _path_label(bundle.manifest_path),
                     "plan_path": str(plan_dir.resolve().relative_to(ROOT))})
    output_root.mkdir(parents=True)
    _write_json(output_root / "preflight_manifest.json", manifest)
    return manifest


def _read_preflight(target_manifest: Path, output_root: Path, plan_dir: Path) -> tuple[RecoveredTargets, dict[str, Any]]:
    if R9_ADJUDICATION is None or not R9_ADJUDICATION.is_file():
        raise FileNotFoundError("pass --r9-adjudication with the reviewed R9 execution decision")
    bundle = load_recovered_targets(target_manifest, expected_manifest_sha256=EXPECTED_TARGET_MANIFEST_SHA256)
    preflight_path = output_root / "preflight_manifest.json"
    if not preflight_path.is_file():
        raise ValueError("run R9 joint CPU preflight before fitting or aggregation")
    manifest = read_json(preflight_path)
    if manifest.get("target_manifest_sha256") != bundle.manifest_sha256:
        raise ValueError("R9 joint preflight is pinned to another recovered target manifest")
    if manifest.get("adapter_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("R9 joint preflight was produced by another adapter revision")
    expected_authority = {
        "recovery_plan_path": _path_label(RECOVERY_PLAN),
        "recovery_plan_sha256": sha256_file(RECOVERY_PLAN),
        "r9_target_adjudication_path": _path_label(R9_ADJUDICATION),
        "r9_target_adjudication_sha256": sha256_file(R9_ADJUDICATION),
    }
    if manifest.get("execution_authority") != expected_authority:
        raise ValueError("R9 joint preflight does not pin the active recovery authorization")
    current = verify_joint_target_parity(bundle, plan_dir)
    for name, value in current.items():
        if manifest.get(name) != value:
            raise ValueError(f"R9 joint frozen preflight input changed: {name}")
    return bundle, manifest


def _terminal_queue_task(value: Any) -> bool:
    """A failed but explicitly recorded upstream branch is terminal, not active."""
    if not isinstance(value, str):
        return False
    status = value.strip().lower()
    if status.startswith("completed_validated"):
        return True
    if status.startswith("explicit_terminal:"):
        return bool(status.split(":", 1)[1].strip())
    if status.startswith("failed_technical") or status.startswith("completed_with_terminal_failures"):
        return bool(status.partition(";")[2].strip())
    return False


def require_gpu_queue_release(state_path: Path | None = None) -> dict[str, str]:
    """Fail closed until R7/R8 are terminal, inactive, and their coordinator exited."""
    state_path = state_path or STATE_PATH
    if state_path is None or not state_path.is_file():
        raise FileNotFoundError("joint fold fitting requires an explicit --state queue receipt")
    state = read_json(state_path)
    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("R9 joint fit requires R7/R8 task states")
    statuses = {task: tasks.get(task) for task in ("R7", "R8")}
    pending = [task for task, value in statuses.items() if not _terminal_queue_task(value)]
    if pending:
        raise ValueError("R9 joint fit is ordered after terminal R7/R8; pending: " + ", ".join(pending))
    active_gpu_job = state.get("active_gpu_job")
    # The serial R9 coordinator marks its own child as active before invoking
    # this gate. Accept that explicit R9 lease only after R7/R8 are terminal;
    # any other active label still blocks execution.
    if active_gpu_job and not str(active_gpu_job).startswith("R9 "):
        raise ValueError("R9 joint fit blocked by an active GPU job outside its own queue")
    r7_queue_pid = state.get("r7_execution", {}).get("queue_pid")
    if r7_queue_pid:
        proc = Path(f"/proc/{int(r7_queue_pid)}")
        if proc.exists():
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if "run_r7_queue.py" in cmdline:
                raise ValueError("R9 joint fit waits for the R7/R8 coordinator process to exit")
    return {task: str(value) for task, value in statuses.items()}


def _load_candidate_trainer():
    """Import only after the caller holds S85's timing and GPU locks."""
    module_name = "r9_locked_candidate_joint_mps_trainer"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, TRAINER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned joint trainer {TRAINER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _invoke_candidate_trainer_locked(trainer_args: list[str]) -> None:
    """Keep module import and CUDA availability probes inside both locks."""
    with s85.exclusive_compute_lock():
        trainer = _load_candidate_trainer()
        previous_argv = sys.argv
        sys.argv = [str(TRAINER), *trainer_args]
        try:
            returncode = trainer.main()
        finally:
            sys.argv = previous_argv
    if returncode not in (None, 0):
        raise RuntimeError(f"pinned joint trainer exited with status {returncode}")


def _validate_fold_output(bundle: RecoveredTargets, split: str, fold: int, fold_dir: Path,
                          preflight_manifest: dict[str, Any]) -> dict[str, Any]:
    run_manifest_path = fold_dir / "run_manifest.json"
    prediction_path = fold_dir / "oof_predictions.csv"
    transform_path = fold_dir / "fold_transform_and_support.json"
    run = read_json(run_manifest_path)
    if run.get("status") != "completed_fold_technical_qa_pass" or run.get("split_id") != split or int(run.get("fold", -1)) != fold:
        raise ValueError(f"joint trainer did not complete technical QA for {split}/fold_{fold}")
    expected_runner_sha = preflight_manifest["candidate_trainer_sha256"]
    if run.get("runner_sha256") != expected_runner_sha:
        raise ValueError("joint fold was produced by another frozen trainer source")
    if run.get("protocol_sha256") != preflight_manifest["protocol_sha256"]:
        raise ValueError("joint fold protocol hash differs from R9 preflight")
    if run.get("plan_sha256") != preflight_manifest["plan_manifest_sha256"]:
        raise ValueError("joint fold split/feature plan differs from R9 preflight")
    if run.get("measurement_attempts_sha256") != preflight_manifest["joint_attempt_ledger_sha256"]:
        raise ValueError("joint fold used another measurement-attempt ledger")
    if run.get("measurement_summary_sha256") != preflight_manifest["joint_run_summary_sha256"]:
        raise ValueError("joint fold used another measurement summary")
    if run.get("quality_labels_used_as_inputs") is not False or run.get("test_runtime_or_fidelity_used_for_fit") is not False:
        raise ValueError("joint fold does not attest train/test target separation")
    if run.get("seeds") != list(SEEDS) or run.get("device") != "cuda":
        raise ValueError("joint fold seed set or required CUDA training device changed")
    if run.get("r9_target_manifest_sha256") != bundle.manifest_sha256:
        raise ValueError("joint fold run manifest is not explicitly bound to the recovered target SHA")
    if sha256_file(transform_path) != run.get("transform_sha256"):
        raise ValueError("joint fold transform/support file hash mismatch")
    transform = read_json(transform_path)
    fold_column = "c44_fold" if split == "c44" else "family_holdout_fold_diagnostic_only"
    expected_test = {digest for digest, assigned in (
        bundle.fixed_fold_by_hash.items() if split == "c44" else bundle.family_fold_by_hash.items()) if assigned == fold}
    train_hashes = set(transform.get("train_hashes", []))
    test_hashes = set(transform.get("test_hashes", []))
    if test_hashes != expected_test or train_hashes != set(bundle.fixed_fold_by_hash) - expected_test:
        raise ValueError(f"joint fold does not preserve the frozen {split} hash assignment")
    if train_hashes & test_hashes or train_hashes | test_hashes != set(bundle.fixed_fold_by_hash):
        raise ValueError(f"joint fold train/test partition leaks or drops hashes")
    rows = read_csv(prediction_path)
    expected_cells = {(digest, chi, method) for digest in expected_test for chi in CHI for method in METHODS}
    actual_cells = {(row["source_qasm_sha256"], int(row["max_bond"]), row["method_id"]) for row in rows}
    if len(rows) != len(expected_cells) or actual_cells != expected_cells:
        raise ValueError(f"joint fold prediction identities differ from the frozen {split} test fold")
    targets = {(row["source_qasm_sha256"], int(row["max_bond"])): row for row in bundle.joint_rows}
    for row in rows:
        key = (row["source_qasm_sha256"], int(row["max_bond"]))
        target = targets[key]
        if row.get("target_status") != target["target_status"]:
            raise ValueError(f"joint fold target status mismatch for {key}")
        if not math.isclose(float(row["target_runtime_seconds"]), float(target["target_runtime_seconds"]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"joint fold runtime target mismatch for {key}")
        if row.get("target_quality_pass", "").strip().lower() != target["target_quality_pass"].strip().lower():
            raise ValueError(f"joint fold quality target mismatch for {key}")
        for seed in SEEDS:
            if not math.isfinite(float(row[f"runtime_seed_{seed}_seconds"])) or not math.isfinite(float(row[f"quality_seed_{seed}_probability"])):
                raise ValueError(f"joint fold has a nonfinite seed prediction for {key}")
    return {"split": split, "fold": fold,
            "target_manifest_sha256": bundle.manifest_sha256,
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "prediction_sha256": sha256_file(prediction_path),
            "transform_sha256": sha256_file(transform_path),
            "test_hashes": len(expected_test), "prediction_rows": len(rows),
            "technical_qa": "PASS"}


def fit_fold(target_manifest: Path, output_root: Path, plan_dir: Path, split: str, fold: int) -> dict[str, Any]:
    bundle, preflight_manifest = _read_preflight(target_manifest, output_root, plan_dir)
    if split not in {"c44", "family_holdout"} or fold not in range(5):
        raise ValueError("split/fold must be c44 or family_holdout and fold 0..4")
    # Recheck the full source-manifest and attempt keys immediately before the
    # frozen trainer starts, not merely at the earlier preflight checkpoint.
    current = verify_joint_target_parity(bundle, plan_dir)
    if current["target_manifest_sha256"] != preflight_manifest["target_manifest_sha256"]:
        raise ValueError("R9 joint target manifest changed before fit")
    fold_dir = output_root / "training" / split / f"fold_{fold}"
    if fold_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing joint fold attempt: {fold_dir}")
    fold_dir.parent.mkdir(parents=True, exist_ok=True)
    require_gpu_queue_release()
    trainer_args = [
        "--plan-dir", str(plan_dir.resolve()),
        "--measure-dir", str(bundle.root / "joint_measurements"),
        "--output-dir", str(fold_dir), "--split", split, "--fold", str(fold),
    ]
    _invoke_candidate_trainer_locked(trainer_args)
    # Re-open exact R9 pins after training. No model result can be attributed
    # to a different manifest if a target changed during the process.
    after = load_recovered_targets(target_manifest, expected_manifest_sha256=EXPECTED_TARGET_MANIFEST_SHA256)
    if after.manifest_sha256 != bundle.manifest_sha256:
        raise ValueError("R9 target manifest changed while the joint fold was running")
    run_path = fold_dir / "run_manifest.json"
    run = read_json(run_path)
    run["r9_target_manifest_sha256"] = bundle.manifest_sha256
    run["r9_target_revision_id"] = bundle.target_revision_id
    run["r9_joint_target_table_sha256"] = bundle.joint_table_sha256
    run["r9_joint_attempt_ledger_sha256"] = preflight_manifest["joint_attempt_ledger_sha256"]
    run["r9_source_runner_lineage"] = bundle.runner_lineage
    _write_json(run_path, run)
    report = _validate_fold_output(bundle, split, fold, fold_dir, preflight_manifest)
    receipt_path = fold_dir / "r9_target_receipt.json"
    _write_json(receipt_path, {"artifact_id": "r9-joint-fold-target-binding-v1",
                               "status": "PASS", **report,
                               "target_revision_id": bundle.target_revision_id,
                               "joint_target_table_sha256": bundle.joint_table_sha256,
                               "attempt_ledger_sha256": preflight_manifest["joint_attempt_ledger_sha256"],
                               "source_runner_lineage": bundle.runner_lineage})
    return report


def validate_fold(target_manifest: Path, output_root: Path, plan_dir: Path,
                  split: str, fold: int) -> dict[str, Any]:
    """Independently validate an existing R9-bound fold for safe queue resume."""
    bundle, preflight_manifest = _read_preflight(target_manifest, output_root, plan_dir)
    if split not in {"c44", "family_holdout"} or fold not in range(5):
        raise ValueError("split/fold must be c44 or family_holdout and fold 0..4")
    fold_dir = output_root / "training" / split / f"fold_{fold}"
    report = _validate_fold_output(bundle, split, fold, fold_dir, preflight_manifest)
    receipt = read_json(fold_dir / "r9_target_receipt.json")
    if (receipt.get("status") != "PASS"
            or receipt.get("target_manifest_sha256") != bundle.manifest_sha256
            or receipt.get("run_manifest_sha256") != report["run_manifest_sha256"]
            or receipt.get("prediction_sha256") != report["prediction_sha256"]
            or receipt.get("transform_sha256") != report["transform_sha256"]):
        raise ValueError(f"existing joint fold receipt is missing or stale: {split}/fold_{fold}")
    return report


def aggregate(target_manifest: Path, output_root: Path, plan_dir: Path) -> dict[str, Any]:
    bundle, preflight_manifest = _read_preflight(target_manifest, output_root, plan_dir)
    for split in ("c44", "family_holdout"):
        for fold in range(5):
            fold_dir = output_root / "training" / split / f"fold_{fold}"
            receipt_path = fold_dir / "r9_target_receipt.json"
            if not receipt_path.is_file():
                raise ValueError(f"cannot aggregate before every R9-bound fold passes: {split}/fold_{fold}")
            receipt = read_json(receipt_path)
            if receipt.get("status") != "PASS" or receipt.get("target_manifest_sha256") != bundle.manifest_sha256:
                raise ValueError(f"joint fold receipt is not bound to the current target: {split}/fold_{fold}")
            if receipt.get("prediction_sha256") != sha256_file(fold_dir / "oof_predictions.csv"):
                raise ValueError(f"joint fold prediction changed after R9 binding: {split}/fold_{fold}")
    output_dir = output_root / "aggregate"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite R9 joint aggregate: {output_dir}")
    command = [sys.executable, str(AGGREGATOR),
               "--plan-dir", str(plan_dir.resolve()),
               "--measure-dir", str(bundle.root / "joint_measurements"),
               "--training-root", str((output_root / "training").resolve()),
               "--output-dir", str(output_dir)]
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"frozen joint aggregator failed ({completed.returncode}): {completed.stderr[-4000:]}")
    current = load_recovered_targets(target_manifest, expected_manifest_sha256=EXPECTED_TARGET_MANIFEST_SHA256)
    if current.manifest_sha256 != bundle.manifest_sha256:
        raise ValueError("R9 target manifest changed during joint aggregation")
    aggregate_manifest_path = output_dir / "manifest.json"
    aggregate_manifest = read_json(aggregate_manifest_path)
    aggregate_manifest["r9_target_manifest_sha256"] = bundle.manifest_sha256
    aggregate_manifest["r9_target_revision_id"] = bundle.target_revision_id
    aggregate_manifest["r9_joint_target_table_sha256"] = bundle.joint_table_sha256
    _write_json(aggregate_manifest_path, aggregate_manifest)
    aggregate_manifest_sha = sha256_file(aggregate_manifest_path)
    receipt = {"artifact_id": "r9-joint-mps-recovered-aggregate-binding-v1",
               "status": "PASS",
               "target_revision_id": bundle.target_revision_id,
               "target_manifest_sha256": bundle.manifest_sha256,
               "joint_target_table_sha256": bundle.joint_table_sha256,
               "joint_attempt_ledger_sha256": preflight_manifest["joint_attempt_ledger_sha256"],
               "joint_aggregate_manifest_sha256": aggregate_manifest_sha,
               "files_sha256": {path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.is_file()},
               "split_assignment_sha256": bundle.manifest["split_assignments"],
               "method_ids": list(METHODS), "seeds": list(SEEDS),
               "c44_folds": list(range(5)), "family_holdout_folds": list(range(5)),
               "quality_is_predictor_input": False,
               "runtime_quality_slice_policy": "all finite runtime labels are trained/evaluated; quality remains a separate target/slice",
               "source_runner_lineage": bundle.runner_lineage}
    _write_json(output_dir / "r9_target_binding.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    global R9_ADJUDICATION, STATE_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "fit-fold", "validate-fold", "aggregate"), required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--r9-adjudication", type=Path, required=True,
                        help="Reviewed execution decision; not bundled as an implicit authorization")
    parser.add_argument("--state", type=Path,
                        help="Terminal R7/R8 queue state required for fit-fold")
    parser.add_argument("--split", choices=("c44", "family_holdout"))
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args(argv)
    R9_ADJUDICATION = args.r9_adjudication.resolve()
    STATE_PATH = args.state.resolve() if args.state is not None else None
    if args.action == "preflight":
        result = preflight(args.target_manifest, args.output_root, args.plan_dir)
    elif args.action == "fit-fold":
        if args.split is None or args.fold is None:
            parser.error("fit-fold requires --split and --fold")
        result = fit_fold(args.target_manifest, args.output_root, args.plan_dir, args.split, args.fold)
    elif args.action == "validate-fold":
        if args.split is None or args.fold is None:
            parser.error("validate-fold requires --split and --fold")
        result = validate_fold(args.target_manifest, args.output_root, args.plan_dir, args.split, args.fold)
    else:
        result = aggregate(args.target_manifest, args.output_root, args.plan_dir)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
