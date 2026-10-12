"""Preprocessing regression tests: no model, inference or latency execution."""
from pathlib import Path
import base64
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark_v1/scripts"))
import am1_preprocessing_compat as compat
import report_am1_preprocessing_recovery as report
from recover_am1_preprocessing import OUT, PHASE2, read, rows, verify_pins


def test_environment_is_explicit():
    assert compat.require_environment() == {"qiskit":"2.5.2", "numpy":"2.2.6"}


def test_only_static_source_functions_reside_in_setup_cache():
    compat.pinned_extractor.cache_clear()
    first=compat.pinned_extractor()
    assert compat.pinned_extractor() is first
    assert callable(first["create_feature_dict"])
    assert compat.pinned_extractor.cache_info().currsize==1
    assert compat.pinned_extractor.cache_info().hits==1


def test_environment_rejects_legacy(monkeypatch):
    monkeypatch.setattr(compat.importlib.metadata, "version", lambda k:"0.44.2" if k=="qiskit" else "2.2.6")
    with pytest.raises(RuntimeError,match="historical_preprocessing_environment_required"):
        compat.require_environment()


def test_array_transport_preserves_exact_dtype_shape():
    arrays = dict(x=np.asarray([[0.0,-1,0.25]],dtype=np.float32),
                  edge_index=np.empty((2,0),dtype=np.int64),
                  global_features=np.asarray([[1,2]],dtype=np.float32))
    result = compat.unpack(compat.pack(arrays))
    assert set(result)==set(arrays)
    for k,v in result.items():
        assert v.shape==arrays[k].shape and v.dtype==arrays[k].dtype
        assert np.array_equal(v,arrays[k])


def test_missing_source_not_generated():
    with pytest.raises(ValueError,match="no_serialized_raw_QASM_contract"):
        compat.parse_qpu(b"", "qpack_mcp")


def test_nominal_properties_preserve_wire_and_unit():
    props = {"qubits":[[{"name":"T1","value":2e-4,"unit":"s"},
                         {"name":"T2","value":3e5,"unit":"ns"}]]}
    assert compat.calibration(json.dumps(props).encode()) == {0:(200.0,300.0)}


def test_frozen_qpu_transform_and_positional_feature():
    old=read(OUT/"diagnostic_phase2.json")
    hist=read(OUT/"diagnostic_historical.json")
    by_id={r["id"]:r for r in hist["qpu"]}
    assert len(old["qpu"])==45
    for r in old["qpu"]:
        h=by_id[r["id"]]
        assert r["canonical_graph_hash"]==h["canonical_graph_hash"]
        assert r["raw_summary"]["sequence_hash"]==h["raw_summary"]["sequence_hash"]
        assert not h["mismatched_fields"] and h["model_ready_exact"]
        assert r["model_ready_exact"] == (not bool(r["mismatched_fields"]))


def test_reproduces_exact_original_failure_IDs():
    old=read(OUT/"diagnostic_phase2.json")
    failure={r["id"] for r in old["qpu"] if r["mismatched_fields"]}
    assert len(failure)==38
    assert failure=={r["id"] for r in old["qpu"] if r["source"]=="mali_real_qpu"}
    for r in old["qpu"]:
        if r["id"] in failure:
            hist=next(h for h in read(OUT/"diagnostic_historical.json")["qpu"] if h["id"]==r["id"])
            assert r["barrier_sequence_hash"]!=hist["barrier_sequence_hash"]
        assert r["global51_match"]


def test_dense_complete_contexts_no_success_subset():
    old=read(OUT/"diagnostic_phase2.json")["dense"]
    hist=read(OUT/"diagnostic_historical.json")["dense"]
    for context,failed in (("dense_fp32",3),("dense_fp64",1)):
        original=[r for r in old if r["context"]==context]
        restored=[r for r in hist if r["context"]==context]
        assert len(original)==len(restored)==30
        assert {(r['id'],r['fold']) for r in original}=={(r['id'],r['fold']) for r in restored}
        assert sum(r["status"]=="MISMATCH" for r in original)==failed
        assert all(r["status"]=="EXACT MATCH" for r in restored)


def test_RCCX_static_full_matrix_positive_and_negative_controls():
    evidence=report.static_matrix_checks(read(OUT/"diagnostic_phase2.json"),read(OUT/"diagnostic_historical.json"))
    assert evidence["status"]=="PASS"
    assert evidence["errors"]["negative_control_CCX_difference"]>1
    assert evidence["errors"]["old_vs_restored_definition"]<1e-14


def test_dense_unvalidated_RCCX_definition_not_overwritten():
    # Test the original lowerer's definition guard directly. The frozen raw
    # panel has only library RCCX, not arbitrary same-named custom QASM gates.
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    from cudaq_qiskit_adapter import _lower_circuit
    definition=QuantumCircuit(3)
    definition.p(0.3,2)
    gate=Gate("rccx",3,[])
    gate.definition=definition
    circuit=QuantumCircuit(3)
    circuit.append(gate,[0,1,2])
    with pytest.raises(ValueError,match="outside the validated"):
        _lower_circuit(circuit)


def test_independent_original_model_encoders_and_bridge():
    receipt=read(OUT/"bridge_validation.json")
    assert len(receipt["receipts"])==210
    assert receipt["model_forward_calls"]==receipt["CUDA_calls"]==0
    for r in receipt["receipts"]:
        assert r["exact_model_ready_match"]
        assert r["actual_hash"]==r["expected_hash"]
        assert set(r["cpu_tensor_devices"].values())=={"cpu"}
    assert len({(r["context"],r["id"],r["model"]) for r in receipt["receipts"]})==210


def test_original_MLP_accounting_is_ratio_of_sums():
    a=report.accounting()
    assert a["paired_request_repetitions"]==1350
    assert a["fraction"]==0.9813402497333147
    assert a["numerator_nanoseconds"]<a["denominator_nanoseconds"]
    assert a["includes_H2D"] and not a["includes_model_load_or_Graph_preparation"]


def test_frozen_artifacts_and_QPack_unavailable_unchanged():
    assert verify_pins()>200
    s=rows(PHASE2/"selected_workloads.csv")
    assert len(s)==110 and sum(not r["raw_path"] for r in s)==5
    assert all(r["b2_eligibility"]=="unavailable" for r in s if not r["raw_path"])
