import json
import io
import csv
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

from benchmark_v1.scripts import run_maestro_cpu_component_v1 as runner
from benchmark_v1.scripts.prepare_maestro_cpu_component import calibration_cells


def test_runner_cli_help_exposes_one_protocol_override():
    result = subprocess.run([sys.executable, str(Path(runner.__file__).resolve()), "--help"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--protocol" in result.stdout
    assert result.stdout.count("Full frozen or resolved runtime protocol") == 1


def test_runner_script_cli_resolves_repository_package_without_pythonpath():
    work = runner.ROOT / "work"
    work.mkdir(exist_ok=True)
    missing_plan = work / "test_missing_maestro_plan_for_cli_import.json"
    output = work / "test_maestro_cli_import_preflight"
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(Path(runner.__file__).resolve()), "--action", "preflight",
         "--plan", str(missing_plan), "--output-dir", str(output)],
        cwd=runner.ROOT, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "FileNotFoundError" in result.stderr
    assert "No module named 'benchmark_v1'" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("body,classical_width,expected", [
    ("h q[0]; measure q[0] -> c[1];", 3, [(0, 1)]),
    ("h q[0]; measure q[0] -> c[2]; measure q[1] -> c[0];", 4, [(0, 2), (1, 0)]),
    ("measure q[0] -> c[0]; barrier q;", 2, [(0, 0)]),
])
def test_terminal_subset_and_unused_classical_bits_preserve_mapping(tmp_path, body, classical_width, expected):
    from qiskit import qasm2
    raw = f'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[{classical_width}]; {body}'
    source = tmp_path / "source.qasm"
    source.write_text(raw)
    normalized = runner.qasm_for({"source_file": str(source), "qasm_sha256": runner.sha_bytes(raw.encode())})
    circuit = qasm2.loads(normalized)
    assert circuit.num_qubits == 2 and circuit.num_clbits == classical_width
    assert [(circuit.find_bit(inst.qubits[0]).index, circuit.find_bit(inst.clbits[0]).index)
            for inst in circuit.data if inst.operation.name == "measure"] == expected


@pytest.mark.parametrize("body", ["measure q[0] -> c[0]; x q[1];",
    "measure q[0] -> c[0]; measure q[0] -> c[1];",
    "measure q[0] -> c[0]; measure q[1] -> c[0];", "h q[0];"])
def test_unsupported_measurement_semantics_are_not_rewritten(tmp_path, body):
    raw = f'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; {body}'
    source = tmp_path / "source.qasm"
    source.write_text(raw)
    with pytest.raises(runner.UnsupportedCircuit):
        runner.qasm_for({"source_file": str(source), "qasm_sha256": runner.sha_bytes(raw.encode())})


def _context(arm="primary_native_threads_1"):
    env = {"thread_arm": arm, "native": {"native_module": "/pinned/maestro.so",
                                           "boost_library": "/pinned/libboost_json.so"}}
    return {"protocol_sha256": "p" * 64, "plan_sha256": "x" * 64,
            "worker_sha256": "w" * 64, "environment": env,
            "environment_sha256": runner.canonical_sha(env)}


def test_registry_seed_is_deterministic_and_identity_sensitive():
    registry = runner.read_json(runner.SEED_REGISTRY_PATH)
    a = runner.registry_seed("a" * 64, "cpu|shots=1000", 1, registry)
    assert a == runner.registry_seed("a" * 64, "cpu|shots=1000", 1, registry)
    assert 0 <= a < 2**32
    assert a != runner.registry_seed("a" * 64, "cpu|shots=1", 1, registry)
    assert a != runner.registry_seed("b" * 64, "cpu|shots=1000", 1, registry)


def test_primary_thread_policy_sets_hard_openmp_thread_limit():
    assert runner.THREAD_VARS == ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
                                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@pytest.mark.parametrize("requested,expected", [(False, False), (None, None)])
def test_worker_records_optimizer_request_and_resolved_config(monkeypatch, tmp_path, requested, expected):
    module_path = tmp_path / "maestro.so"
    module_path.write_bytes(b"fake extension identity")
    observed = {}
    module = types.ModuleType("maestro")
    module.__file__ = str(module_path)
    module.SimulatorType = types.SimpleNamespace(QCSim=types.SimpleNamespace(value="QCSim"))
    module.SimulationType = types.SimpleNamespace(Statevector=types.SimpleNamespace(value="Statevector"))
    module._optimizer_profile = lambda _qasm: None

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.optimize_circuit = None

    def simple_execute(_qasm, config, shots):
        observed["optimizer"] = config.optimize_circuit
        observed["shots"] = shots
        return {"simulator": "QCSim", "method": "Statevector", "time_taken": 0.25}

    module.SimulatorConfig = Config
    module.simple_execute = simple_execute
    monkeypatch.setitem(__import__("sys").modules, "maestro", module)
    payload = {"qasm": "OPENQASM 2.0;", "shots": 1000, "seed": 42,
               "native_module": str(module_path), "native_build_dir": str(tmp_path),
               "boost_library": str(tmp_path / "libboost_json.so"), "thread_arm": "primary_native_threads_1",
               "optimize_circuit": requested}

    result = runner._worker(payload)

    assert result["status"] == "ok"
    assert result["optimizer_enabled_requested"] is expected
    assert result["optimizer_enabled_resolved"] is expected
    assert observed == {"optimizer": expected, "shots": 1000}


def test_subprocess_thread_arms_set_or_unset_openmp_limits(monkeypatch):
    captured = {}

    class FakeProcess:
        returncode = 0
        def communicate(self, _input, timeout):
            assert timeout == 2
            return ('{"status":"ok","engine_reported_seconds":0.1,"host_wall_seconds":0.2}', "")

    def fake_popen(_command, **kwargs):
        captured.clear()
        captured.update(kwargs["env"])
        return FakeProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("OMP_THREAD_LIMIT", "8")
    base = {"native_module": "/pinned/maestro.so", "boost_library": "/pinned/libboost_json.so"}

    primary = runner.subprocess_attempt({**base, "thread_arm": "primary_native_threads_1"}, 2)
    assert primary["status"] == "ok"
    assert captured["OMP_NUM_THREADS"] == captured["OMP_THREAD_LIMIT"] == "1"

    unset = runner.subprocess_attempt({**base, "thread_arm": "unset_native_thread_environment"}, 2)
    assert unset["status"] == "ok"
    assert "OMP_NUM_THREADS" not in captured and "OMP_THREAD_LIMIT" not in captured


def test_source_qasm_parser_uses_frozen_legacy_instruction_map():
    class FakeQasm2:
        LEGACY_CUSTOM_INSTRUCTIONS = object()
        def loads(self, raw, **kwargs):
            assert raw == "legacy-qasm"
            assert kwargs["strict"] is True
            assert kwargs["custom_instructions"] is self.LEGACY_CUSTOM_INSTRUCTIONS
            return "parsed"

    assert runner.load_source_qasm("legacy-qasm", FakeQasm2()) == "parsed"


def test_static_plan_allows_same_sampling_qasm_at_distinct_shot_knots(tmp_path):
    from benchmark_v1.scripts.prepare_maestro_cpu_component import PROTOCOL, ROOT
    from benchmark_v1.scripts import prepare_maestro_cpu_component as preparer
    import json
    protocol = json.loads(PROTOCOL.read_text())
    plan_path = tmp_path / "plan.json"
    plan = {
        "protocol_sha256": runner.sha_file(runner.PROTOCOL_PATH),
        "preparer_sha256": runner.sha_file(runner.PREPARER_PATH),
        "calibration_cells": preparer.calibration_cells(protocol),
    }
    groups = {}
    for cell in plan["calibration_cells"]:
        groups.setdefault(cell["qasm_sha256"], []).append(cell)
    repeated = [rows for rows in groups.values() if len(rows) > 1]
    assert len(repeated) == 8
    assert all({r["shots"] for r in rows} == {1, 10, 100, 1000, 10000} for rows in repeated)
    assert len({r["cell_id"] for r in plan["calibration_cells"]}) == 112
    assert len(runner.validate_calibration_cell_hashes(plan["calibration_cells"])) == 80

    broken = [dict(cell) for cell in plan["calibration_cells"]]
    broken[1]["qasm"] = broken[0]["qasm"]
    broken[1]["qasm_sha256"] = broken[0]["qasm_sha256"]
    broken[1]["cell_id"] = runner.canonical_sha({k: v for k, v in broken[1].items()
                                                   if k not in {"qasm", "cell_id"}})
    with pytest.raises(ValueError, match="repeated calibration QASM"):
        runner.validate_calibration_cell_hashes(broken)


def test_normalized_execution_identity_records_exact_rx_rz_cx_representation():
    qasm = """OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[2];
creg c[2];
rx(0.125) q[0];
rz(0.25) q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""
    identity = runner.normalized_execution_identity(qasm)
    assert identity == {
        "normalized_qasm_sha256": runner.sha_bytes(qasm.encode()),
        "normalized_width": 2,
        "normalized_1q_gate_count": 2,
        "normalized_cx_gate_count": 1,
    }


def test_mocked_attempts_append_start_then_terminal_and_resume_without_retry(monkeypatch):
    work = runner.ROOT / "work"
    work.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="maestro_runner_test_", dir=work))
    monkeypatch.setattr(runner, "ROOT", runner.ROOT)
    spec = {"cell_id": "c1", "qasm_sha256": "a" * 64, "qasm": "OPENQASM 2.0;",
            "shots": 1, "outer_fold": None, "scope": "calibration"}
    protocol = {"context": {"seed_context_id": "cpu|shots={cell_shots}"},
                "measurement": {"timeout_seconds": 3}}
    contexts = {"primary_native_threads_1": _context()}
    calls = []

    def mocked(payload, timeout):
        calls.append((payload, timeout))
        return {"status": "ok", "engine_reported_seconds": 0.01,
                "host_wall_seconds": 0.02, "outer_wall_seconds": 0.03}

    try:
        # Keep this test small by replacing the frozen session/repetition loops.
        monkeypatch.setattr(runner, "registry_seed", lambda *a, **k: 7)
        spec_list = [spec]
        runner.call_specs(spec_list, "full", ["primary_native_threads_1"], out,
                          protocol, "x" * 64, contexts, call=mocked)
        ledger = out / "attempts.jsonl"
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert len(rows) == 30  # 15 immutable starts + 15 terminal records
        assert all(rows[i]["record_type"] == "attempt_started" and rows[i+1]["record_type"] == "terminal"
                   for i in range(0, len(rows), 2))
        assert len(calls) == 15
        runner.call_specs(spec_list, "full", ["primary_native_threads_1"], out,
                          protocol, "x" * 64, contexts, call=mocked)
        assert len(calls) == 15
        assert len(ledger.read_text().splitlines()) == 30
        changed = _context()
        changed["environment_sha256"] = "d" * 64
        with pytest.raises(ValueError, match="worker/environment identity differs"):
            runner.call_specs(spec_list, "full", ["primary_native_threads_1"], out,
                              protocol, "x" * 64, {"primary_native_threads_1": changed}, call=mocked)
        with ledger.open("a") as handle:
            handle.write(ledger.read_text().splitlines()[-1] + "\n")
        with pytest.raises(ValueError, match="duplicate terminal identity"):
            runner.read_ledger(ledger)
    finally:
        shutil.rmtree(out)


def test_panel_normalization_failure_becomes_explicit_unsupported_attempt(monkeypatch):
    work = runner.ROOT / "work"
    work.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="maestro_unsupported_test_", dir=work))
    spec = {"cell_id": "panel-c1", "qasm_sha256": "e" * 64, "qasm": None,
            "_prep_error": "normalization_failed:QASMError:unsupported", "shots": 1000,
            "outer_fold": 0, "scope": "panel"}
    protocol = {"context": {"seed_context_id": "cpu|shots={cell_shots}"},
                "measurement": {"timeout_seconds": 1}}
    contexts = {"primary_native_threads_1": _context()}
    calls = []
    try:
        runner.call_specs([spec], "full", ["primary_native_threads_1"], out, protocol,
                          "x" * 64, contexts, call=lambda *args: calls.append(args))
        rows = [json.loads(line) for line in (out / "attempts.jsonl").read_text().splitlines()]
        terminal = [r for r in rows if r["record_type"] == "terminal"]
        assert len(terminal) == 15
        assert all(r["status"] == "unsupported" and "normalization_failed" in r["error"] for r in terminal)
        assert calls == []
    finally:
        shutil.rmtree(out)


def test_pilot_gate_is_technical_only_and_rejects_unstable_or_failed_cells():
    cells = [{"cell_id": f"c{i}"} for i in range(10)]
    plan = {"pilot_cell_ids": [c["cell_id"] for c in cells]}
    rows = []
    for cell in cells:
        for arm in ("unset_native_thread_environment", "primary_native_threads_1"):
            for session in (1, 2, 3):
                for repetition in (1, 2, 3, 4, 5):
                    rows.append({"record_type": "terminal", "cell_id": cell["cell_id"],
                                 "thread_arm": arm, "session": session, "repetition": repetition,
                                 "status": "ok", "engine_reported_seconds": 0.01,
                                 "outer_wall_seconds": 1.0})
    gate = runner.pilot_gate(plan, rows)
    assert gate["status"] == "pass"
    assert gate["accuracy_gate_applied"] is False
    assert gate["projected_3930_assigned_calls_from_scratch_seconds"] == 3930.0
    assert gate["projected_3780_new_calls_after_150_pilot_primary_reuse_seconds"] == 3780.0
    next(r for r in rows if r["thread_arm"] == "primary_native_threads_1")["status"] = "timeout"
    assert runner.pilot_gate(plan, rows)["status"] == "fail"


def test_calibration_reducer_identifies_coefficients_and_unavailable_states():
    protocol = runner.read_json(runner.PROTOCOL_PATH)
    cells = calibration_cells(protocol)
    plan = {"calibration_cells": cells}
    attempts = []
    for cell in cells:
        n = cell["width"]
        for session in (1, 2, 3):
            for repetition in (1, 2, 3, 4, 5):
                if cell["kind"] == "sampling":
                    value = 0.0001 + cell["shots"] * 0.000001
                else:
                    slope = {"one_qubit_noncommuting": 0.00002,
                             "two_qubit_wrapper_control": 0.000005,
                             "two_qubit_cx_interleaved": 0.000015}[cell["operation"]]
                    value = 0.01 + n * 0.0001 + slope * cell["repeats"]
                attempts.append({"record_type": "terminal", "cell_id": cell["cell_id"],
                                 "session": session, "repetition": repetition,
                                 "status": "ok", "engine_reported_seconds": value})
    model = runner.reduce_calibration(plan, attempts)
    assert model["status"] == "complete"
    assert len(model["knots"]) == 8
    assert all(k["valid"] for k in model["knots"])
    assert all(k["C1q_seconds_per_gate_amplitude_scale"] > 0 for k in model["knots"])
    assert all(k["Ccx_seconds_per_CX_amplitude_scale"] > 0 for k in model["knots"])
    # A terminal timeout is preserved as missing support, not substituted or retried.
    attempts = [r for r in attempts if not (r["cell_id"] == cells[0]["cell_id"] and r["session"] == 1)]
    model2 = runner.reduce_calibration(plan, attempts)
    assert model2["status"] == "complete_with_unavailable_knots"
    assert "unstable_or_incomplete_cell" in " ".join(model2["knots"][0]["unavailable_reasons"])


def test_identified_increment_gate_rejects_tiny_negative_and_unstable_values():
    assert runner._identified_increment([0.001, 0.00101, 0.00099])["pass"]
    assert "median_increment_not_resolved_above_absolute_floor" in runner._identified_increment(
        [1e-6, 2e-6, 3e-6])["reasons"]
    assert "nonpositive_session_increment" in runner._identified_increment(
        [0.001, -0.001, 0.001])["reasons"]
    assert "session_increment_unstable" in runner._identified_increment(
        [0.0001, 0.001, 0.003])["reasons"]


def test_subprocess_timeout_and_resource_limit_are_terminal(monkeypatch):
    class TimeoutProc:
        returncode = None
        def __init__(self, *a, **kw): pass
        def communicate(self, *a, **kw):
            if kw.get("timeout"):
                raise runner.subprocess.TimeoutExpired("cmd", kw["timeout"])
            return ("", "")
        def kill(self): self.returncode = -9

    monkeypatch.setattr(runner.subprocess, "Popen", TimeoutProc)
    result = runner.subprocess_attempt({"thread_arm": "primary_native_threads_1",
        "native_module": "/native/maestro.so", "boost_library": "/boost/libboost.so"}, 1)
    assert result["status"] == "timeout"

    class KilledProc:
        returncode = -9
        def __init__(self, *a, **kw): pass
        def communicate(self, *a, **kw): return ("", "")
    monkeypatch.setattr(runner.subprocess, "Popen", KilledProc)
    result = runner.subprocess_attempt({"thread_arm": "primary_native_threads_1",
        "native_module": "/native/maestro.so", "boost_library": "/boost/libboost.so"}, 1)
    assert result["status"] == "resource_limit"


def test_pilot_report_rejects_tampered_ledger_or_context():
    work = runner.ROOT / "work"
    work.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="maestro_pilot_report_test_", dir=work))
    try:
        cells = [f"c{i}" for i in range(10)]
        rows = []
        for cell in cells:
            for arm in ("unset_native_thread_environment", "primary_native_threads_1"):
                for session in (1, 2, 3):
                    for repetition in (1, 2, 3, 4, 5):
                        rows.append({"record_type": "terminal", "cell_id": cell, "thread_arm": arm,
                                     "session": session, "repetition": repetition, "status": "ok"})
        ledger = out / "attempts.jsonl"
        ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
        ctx = _context()
        report = {"status": "pass", "plan_sha256": "p" * 64,
                  "primary_environment_sha256": ctx["environment_sha256"],
                  "runner_sha256": ctx["worker_sha256"], "pilot_cell_ids": cells,
                  "terminal_attempts_sha256": runner.sha_file(ledger)}
        report_path = out / "pilot_report.json"
        report_path.write_text(json.dumps(report))
        _, primary = runner.validate_pilot_report(report_path, "p" * 64, ctx)
        assert len(primary) == 150
        report["primary_environment_sha256"] = "tampered"
        report_path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match="environment/runner"):
            runner.validate_pilot_report(report_path, "p" * 64, ctx)
        report["primary_environment_sha256"] = ctx["environment_sha256"]
        report["terminal_attempts_sha256"] = "bad"
        report_path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match="ledger changed"):
            runner.validate_pilot_report(report_path, "p" * 64, ctx)
    finally:
        shutil.rmtree(out)


def test_hidden_worker_dispatch_does_not_require_public_cli_arguments(monkeypatch):
    monkeypatch.setattr(runner.sys, "stdin", io.StringIO('{"dummy": true}'))
    monkeypatch.setattr(runner, "_worker", lambda payload: {"status": "ok", "seen": payload["dummy"]})
    outputs = []
    monkeypatch.setattr("builtins.print", lambda value: outputs.append(value))
    assert runner.main(["--_worker"]) == 0
    assert json.loads(outputs[-1]) == {"status": "ok", "seen": True}


def test_aggregate_requires_complete_pinned_attempts_and_emits_evaluator_target_contract(monkeypatch):
    protocol = runner.read_json(runner.PROTOCOL_PATH)
    calibration = calibration_cells(protocol)
    fold_counts = {0: 37, 1: 28, 2: 26, 3: 25, 4: 34}
    panel = []
    index = 0
    for fold, count in fold_counts.items():
        for _ in range(count):
            digest = runner.sha_bytes(f"panel-{index}".encode())
            panel.append({"source_qasm_sha256": digest, "outer_fold": fold,
                              "width": 2 + index % 8, "source_file": "unused",
                              "aliases": [f"alias-{index}"], "shots": 1000,
                              "measurement_map": [[0, 0], [1, 1]],
                              "measurement_map_sha256": runner.canonical_sha([[0, 0], [1, 1]]),
                              "measured_qubit_count": 2, "measured_classical_bit_count": 2,
                              "measured_classical_extent": 2, "declared_classical_width": 2,
                              "measurement_terminal": True, "measurement_map_injective": True})
            index += 1
    plan = {"calibration_cells": calibration, "panel": panel}
    plan_hash = "f" * 64
    monkeypatch.setattr(runner, "expected_plan_identity", lambda path: (protocol, plan, plan_hash))
    normalized_panel_qasm = """OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[2];
creg c[2];
rx(0.125) q[0];
rz(0.25) q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""
    monkeypatch.setattr(runner, "qasm_for", lambda spec: normalized_panel_qasm)
    work = runner.ROOT / "work"
    work.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="maestro_aggregate_test_", dir=work))
    try:
        ledger = out / "attempts.jsonl"
        rows = []
        ordinal = 0
        hash_by_cell = {c["cell_id"]: c["qasm_sha256"] for c in calibration}
        hash_by_cell.update({p["source_qasm_sha256"]: p["source_qasm_sha256"] for p in panel})
        normalized_by_cell = {
            c["cell_id"]: runner.normalized_execution_identity(c["qasm"])
            for c in calibration
        }
        normalized_by_cell.update({
            p["source_qasm_sha256"]: runner.normalized_execution_identity(normalized_panel_qasm)
            for p in panel
        })
        cell_specs = [(c["cell_id"], c["qasm_sha256"], c["shots"], c) for c in calibration]
        cell_specs.extend((p["source_qasm_sha256"], p["source_qasm_sha256"], 1000, p) for p in panel)
        for cell_id, qhash, shots, spec in cell_specs:
            for session in (1, 2, 3):
                for repetition in (1, 2, 3, 4, 5):
                    ordinal += 1
                    base = {"phase": "full", "cell_id": cell_id, "qasm_sha256": qhash,
                            "outer_fold": spec.get("outer_fold"), "thread_arm": "primary_native_threads_1",
                            "session": session, "repetition": repetition, "attempt_ordinal": ordinal,
                            "seed_uint32": 17, "protocol_sha256": "a"*64, "plan_sha256": plan_hash,
                            "worker_sha256": "b"*64, "environment_sha256": "c"*64,
                            "simulator_id": "QCSim", "method_id": "Statevector",
                            **normalized_by_cell[cell_id]}
                    rows.append({**base, "record_type": "attempt_started", "status": "started"})
                    if "kind" not in spec:
                        seconds = 0.1 + ordinal * 1e-8
                    elif spec["kind"] == "sampling":
                        seconds = 0.0001 + shots * 0.000001
                    else:
                        slope = {"one_qubit_noncommuting": .00002,
                                 "two_qubit_wrapper_control": .000005,
                                 "two_qubit_cx_interleaved": .000015}[spec["operation"]]
                        seconds = .01 + spec["width"] * .0001 + slope * spec["repeats"]
                    rows.append({**base, "record_type": "terminal", "status": "ok",
                                 "engine_reported_seconds": seconds, "host_wall_seconds": seconds * 2,
                                 "outer_wall_seconds": seconds * 3, "error": None})
        ledger.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
        (out / "run_manifest.json").write_text(json.dumps({"phase": "full", "status": "completed",
            "plan_sha256": plan_hash, "attempts_sha256": runner.sha_file(ledger)}))
        result = runner.aggregate(out, Path("ignored-plan.json"))
        assert result["panel_complete"] == 150
        model = json.loads((out / "calibration_model.json").read_text())
        assert model["schema_id"] == "maestro-cpu-component-calibration-v1"
        assert model["artifact_status"] == "PASS"
        assert model["status"] == "complete"
        assert model["input_attempt_sha256"] == [runner.sha_file(ledger)]
        assert model["attempts_sha256"] == runner.sha_file(ledger)
        assert model["panel_targets_sha256"] == runner.sha_file(out / "panel_targets.csv")
        assert len(model["calibration_source_qasm_sha256"]) == 112
        assert len(model["width_coefficients"]) == 8
        targets_rows = list(csv.DictReader((out / "panel_targets.csv").open()))
        assert len(targets_rows) == 150
        assert all(row["target_status"] == "runtime_observed" for row in targets_rows)
        assert all(row["normalized_qasm_sha256"] == runner.sha_bytes(normalized_panel_qasm.encode())
                   and row["normalized_width"] == "2"
                   and row["normalized_1q_gate_count"] == "2"
                   and row["normalized_cx_gate_count"] == "1" for row in targets_rows)

        # The evaluator accepts target labels only when exact hashes/folds are present.
        import sys
        scripts_dir = str(runner.ROOT / "benchmark_v1/scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import evaluate_maestro_cpu_component_v1 as evaluator
        fold_by_hash = {p["source_qasm_sha256"]: p["outer_fold"] for p in panel}
        accepted = evaluator.validate_panel_targets(evaluator.read_csv(out / "panel_targets.csv"), fold_by_hash)
        assert len(accepted) == 150
        accepted_knots = evaluator.load_component_model(
            out / "calibration_model.json", set(fold_by_hash),
            expected_panel_targets_sha256=runner.sha_file(out / "panel_targets.csv"))
        assert set(accepted_knots) == set(range(2, 10))

        # Aggregate refuses tampering and orphan/missing attempt pairs before emitting fresh outputs.
        manifest = runner.read_json(out / "run_manifest.json")
        manifest["attempts_sha256"] = "tampered"
        (out / "run_manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="ledger changed"):
            runner.aggregate(out, Path("ignored-plan.json"))
        manifest["attempts_sha256"] = runner.sha_file(ledger)
        (out / "run_manifest.json").write_text(json.dumps(manifest))
        original = ledger.read_text()
        ledger.write_text("\n".join(original.splitlines()[:-1]) + "\n")
        manifest["attempts_sha256"] = runner.sha_file(ledger)
        (out / "run_manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="incomplete, duplicated, or orphaned"):
            runner.aggregate(out, Path("ignored-plan.json"))
    finally:
        shutil.rmtree(out)


def test_protocol_seed_pin_is_derived_not_hardcoded():
    protocol = runner.read_json(runner.PROTOCOL_PATH)
    assert runner.sha_file(runner.SEED_REGISTRY_PATH) == protocol["context"]["seed_registry_sha256"]
    assert runner.sha_file(runner.PROTOCOL_PATH) == runner.sha_file(runner.PROTOCOL_PATH)
