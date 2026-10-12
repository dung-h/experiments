"""Saved-evidence checks only. No estimator/framework import or execution."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"artifacts/am1_overhead_pilot/v1"


def records(name):
    with (OUT/name).open(newline="") as f:return list(csv.DictReader(f))


def test_selected_denominators():
    assert Counter(r["context"] for r in records("selected_workloads.csv"))=={"qualified_qpu_4515":50,"dense_fp32":30,"dense_fp64":30,"maestro":30}


def test_frozen_seed_coverage():
    groups={}
    for r in records("replay_validation.csv"):
        if r["seed"]!="native":groups.setdefault((r["context"],r["method"],r["id"]),set()).add(r["seed"])
    assert len(groups)==220
    assert all(s=={"42","1234","31415","median3"} for s in groups.values())


def test_replay_and_unavailable():
    assert Counter(r["status"] for r in records("replay_validation.csv"))=={"PASS":1049,"UNAVAILABLE_SAVED_OUTPUT":1}


def test_no_timeout_promoted_as_native_latency():
    assert not any(r["context"]=="analytical" for r in records("raw_timings.csv"))
    assert json.loads((OUT/"native_replay_timeout_receipt.json").read_text())["prediction_parity"]=="NOT COMPLETED"


def test_latency_and_exact_saved_target_join():
    timing={(r["context"],r["method"],r["id"]):r for r in records("timing_by_workload.csv") if r["boundary"]=="B1"}
    for r in records("workload_accuracy_cost.csv"):
        assert float(r["B1_median_seconds"])==float(timing[(r["context"],r["method"],r["id"])]["median_seconds"])
        if r["saved_target_seconds"]!="NA":
            assert float(r["B1_over_historical_execution"])==float(r["B1_median_seconds"])/float(r["saved_target_seconds"])


def test_boundaries_and_no_full_frontier():
    assert not any(r["boundary"]=="B2" for r in records("raw_timings.csv"))
    v=json.loads((OUT/"provenance_manifest.json").read_text())["validation"]
    assert not v["full_B2_executed"] and not v["complete_Pareto_frontier"] and not v["commit_push"]
