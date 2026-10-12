"""CPU-only synthetic tests of Phase4 tooling, not saved science PASS substitutes."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
import run_am1_phase4 as run
import analyze_am1_phase4 as analysis


def test_balance():
    for i in range(45):
        assert sum(run.order(i,r)[0]==run.METHODS[0] for r in range(30))==15


def test_tolerance_fixed():
    assert run.passing(1,1)
    assert run.passing(1.00001,1)
    assert not run.passing(1.001,1)
    assert not run.passing(float('nan'),1)


def test_cluster_pairs():
    data=[dict(group="a",graph_abs_error_seconds=1,mlp_abs_error_seconds=2,graph_B2_median_ms=4,mlp_B2_median_ms=2),
          dict(group="a",graph_abs_error_seconds=1,mlp_abs_error_seconds=2,graph_B2_median_ms=4,mlp_B2_median_ms=2),
          dict(group="b",graph_abs_error_seconds=1,mlp_abs_error_seconds=2,graph_B2_median_ms=4,mlp_B2_median_ms=2)]
    receipt=analysis.cluster_bootstrap(data,reps=20)
    assert receipt["groups"]==2
    assert receipt["mae_difference_graph_minus_mlp_seconds"]==[-1,-1]
    assert receipt["marginal_median_latency_ratio"]==[2,2]


def test_metrics():
    assert analysis.metrics([1,2],[1,2])["r2"]==1
    assert analysis.metrics([1,1],[2,2])["r2"] is None
    assert analysis.metrics([1,2],[2,3])["mae_seconds"]==1


def test_partition_does_not_double_count_cpu_wait():
    record=dict(cpu_parse_QASM_ns=10,cpu_array_serialization_ns=3,cpu_payload_JSON_encode_ns=2,host_request_serialization_ns=4,
        host_response_JSON_decode_ns=1,host_typed_array_decode_ns=1,host_tensor_construction_ns=1,host_batch_and_H2D_ns=1,
        host_three_seed_inference_inverse_and_D2H_ns=2,host_host_median_aggregation_ns=1,IPC_unattributed_residual_ns=5,coordinator_unattributed_ns=1,
        host_response_wait_inclusive_ns=20,total_B2_ns=32)
    assert sum(analysis.stage_partition(record).values())==32


def test_integrity_mismatch(tmp_path):
    f=tmp_path/"input";f.write_text("original")
    digest=run.sha(f)
    assert run.verify({str(f):digest})==1
    f.write_text("changed")
    with pytest.raises(RuntimeError,match="integrity_mismatch"):
        run.verify({str(f):digest})


def test_exclusive_writes(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"OUT",tmp_path)
    run.write_json("receipt.json",{"a":1})
    with pytest.raises(FileExistsError):run.write_json("receipt.json",{"a":2})


def test_no_experiment_training_calls():
    text=Path(run.__file__).read_text()
    assert "optimizer.step" not in text
    assert "transpile(" not in text
    assert "get_state(" not in text
    assert "torch.no_grad()" in text
