"""Synthetic controls plus CPU-only raw-to-model-ready checks, no inference."""
from pathlib import Path
import sys
import base64

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import run_am1_phase5_dense_b2 as run
from analyze_am1_phase4 import metrics,cluster_bootstrap,stage_partition


@pytest.mark.parametrize("value,expected",[("341.0",341),("9",9),("0.00",0)])
def test_exact_count(value,expected): assert run.count(value)==expected


@pytest.mark.parametrize("value",["nan","-1","1.2","Infinity"])
def test_reject_noncount(value):
    with pytest.raises(ValueError):run.count(value)


def test_balanced_order():
    for i in range(30):assert sum(run.order(i,r)[0]==run.METHODS[0] for r in range(30))==15


def test_no_forbidden_calls():
    text=Path(run.__file__).read_text()
    assert "optimizer.step" not in text and "transpile(" not in text and "get_state(" not in text
    assert "torch.no_grad()" in text and "torch.expm1" in text and "no_CPU_fallback" in text


def test_metric_and_interval_units():
    assert metrics([.001,.002],[.001,.002])["r2"]==1
    result=cluster_bootstrap([dict(group="a",graph_abs_error_seconds=.001,mlp_abs_error_seconds=.002,graph_B2_median_ms=4,mlp_B2_median_ms=2)],reps=10)
    assert result["mae_difference_graph_minus_mlp_seconds"]==[-.001,-.001]
    assert result["marginal_median_latency_ratio"]==[2,2]


def test_raw_mlp_skips_dag_and_matches_frozen(monkeypatch):
    import am1_preprocessing_compat as compat
    import materialize_cudaq_dense_ir_features_v1 as producer
    import recover_am1_preprocessing as recovery
    r=next(r for r in run.rows(run.PHASE2/"selected_workloads.csv") if r["context"]=="dense_fp32" and r["raw_path"])
    contract=run.read(run.folder("fp32",int(r["fold"]))/"fold_inputs.json")["feature_contracts"][run.METHODS[1]]
    def fail(_):raise AssertionError("MLP must not construct DAG")
    monkeypatch.setattr(producer,"graph_record",fail)
    arrays=run.preprocess_request(dict(graph=False,raw_qasm_base64=base64.b64encode(Path(r["raw_path"]).read_bytes()).decode(),contract=contract),lambda name,fn:fn())
    receipt=next(x for x in run.read(run.PHASE3/"bridge_validation.json")["receipts"] if x["context"]=="dense_fp32" and x["id"]==r["id"] and x["model"]=="MLP")
    assert recovery.array_hash(arrays)==receipt["expected_hash"]
    assert set(arrays)=={"global_features"}


def test_exclusive_namespace_writes(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"OUT",tmp_path);run.write_json("receipt.json",dict(status="PASS"))
    with pytest.raises(FileExistsError):run.write_json("receipt.json",dict(status="FAIL"))
