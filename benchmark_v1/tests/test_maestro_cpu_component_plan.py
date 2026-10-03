import json

import pytest

from benchmark_v1.scripts.prepare_maestro_cpu_component import (
    PROTOCOL,
    ROOT,
    calibration_cells,
    synthetic_qasm,
    validate_seed_registry_pin,
)


def test_plan_size_and_source_independent_descriptor():
    cfg = json.loads(PROTOCOL.read_text())
    cells = calibration_cells(cfg)
    assert len(cells) == 112
    assert sum(c["kind"] == "operation" for c in cells) == 72
    assert sum(c["kind"] == "sampling" for c in cells) == 40
    assert len({c["cell_id"] for c in cells}) == 112
    pilot = [c for c in cells if c["width"] == 9 and (c["kind"] == "operation" or c["shots"] == 1000)]
    assert len(pilot) == 10
    assert all("runtime_label" not in c for c in cells)


def test_matched_wrapper_has_same_local_word():
    wrapper = synthetic_qasm(4, "two_qubit_wrapper_control", 3)
    cx = synthetic_qasm(4, "two_qubit_cx_interleaved", 3)
    assert [l for l in wrapper.splitlines() if not l.startswith("cx ")] == [l for l in cx.splitlines() if not l.startswith("cx ")]
    assert cx.count("cx ") - wrapper.count("cx ") == 3
    assert "qreg q[4]" in wrapper and wrapper.endswith("measure q -> c;\n")


def test_zero_sample_does_not_invent_unitary_width_control():
    qasm = synthetic_qasm(9, "zero_gate_sample", 0)
    assert "qreg q[9]" in qasm and "measure q -> c;" in qasm
    assert not any(gate in qasm for gate in ("rx(", "rz(", "cx "))
    with pytest.raises(ValueError):
        synthetic_qasm(9, "zero_gate_sample", 1)


def test_seed_registry_pin_matches_and_rejects_stale_protocol_hash():
    cfg = json.loads(PROTOCOL.read_text())
    assert validate_seed_registry_pin(cfg, ROOT) == cfg["context"]["seed_registry_sha256"]
    cfg["context"]["seed_registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="seed registry hash"):
        validate_seed_registry_pin(cfg, ROOT)
