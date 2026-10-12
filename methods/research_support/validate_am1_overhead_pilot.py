#!/usr/bin/env python3
"""Independent stdlib saved-evidence validation, never executes estimators."""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/"artifacts/am1_overhead_pilot/v1"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


def rows(name):
    with (P/name).open(newline="") as f:return list(csv.DictReader(f))


def main():
    m=json.loads((P/"provenance_manifest.json").read_text())
    actual_payload={str(p.relative_to(P)) for p in P.rglob("*") if p.is_file() and p.name!="provenance_manifest.json"}
    assert actual_payload==set(m["payload"]),"untracked or missing pilot payload"
    for p,s in m["payload"].items():assert sha(P/p)==s,p
    for p,s in m["source_pins"].items():assert sha(p)==s,p
    assert sha(ROOT/"benchmark_v1/scripts/summarize_am1_overhead_pilot.py")==m["report_builder_sha256"]
    for p in m["validation"]["required_artifacts"]:assert (P/p).is_file(),p
    plan=json.loads((P/"selection_manifest.json").read_text())
    assert sha(P/"selected_workloads.csv")==plan["selected_workloads_sha256"]
    assert sha(P/"selected_variants.csv")==plan["selected_variants_sha256"]
    replay=rows("replay_validation.csv")
    for r in replay:
        if r["status"]=="PASS":
            diff=abs(float(r["replayed_seconds"])-float(r["saved_seconds"]))
            assert diff<=float(r["atol_seconds"])+float(r["rtol"])*abs(float(r["saved_seconds"]))
            assert abs(diff-float(r["abs_diff_seconds"]))<1e-14
    counts=defaultdict(list)
    for r in rows("raw_timings.csv"):
        assert int(r["end_ns"])-int(r["start_ns"])==int(r["elapsed_ns"])>=0
        if r["boundary"] in ("B1","P1","CONTROL"):
            rid=r["id"] if r["boundary"]!="CONTROL" else r["id"]+"/fold"+r["fold"]
            counts[(r["context"],r["method"],r["boundary"],rid)].append(int(r["elapsed_ns"])/1e9)
    for k,v in counts.items():assert len(v)==30,(k,len(v))
    for r in rows("timing_by_workload.csv"):
        assert statistics.median(counts[(r["context"],r["method"],r["boundary"],r["id"])])==float(r["median_seconds"])
    assert m["validation"]["rss_peak_bytes"]<8*1024**3
    assert m["validation"]["cuda_reserved_peak_bytes"]<8*1024**3
    assert sum(p.stat().st_size for p in P.rglob("*") if p.is_file())<256*1024**2
    closure=json.loads((P/"resource_closure.json").read_text())
    assert closure["elapsed_since_predeclaration_seconds"]<closure["wall_budget_seconds"]
    assert closure["active_am1_worker_pids"]==[]
    timings={(r["context"],r["method"],r["id"]):float(r["median_seconds"]) for r in rows("timing_by_workload.csv") if r["boundary"]=="B1"}
    for r in rows("workload_accuracy_cost.csv"):
        value=timings[(r["context"],r["method"],r["id"])]
        assert float(r["B1_median_seconds"])==value
        if r["saved_target_seconds"]!="NA":
            assert float(r["B1_over_historical_execution"])==value/float(r["saved_target_seconds"])
    print(json.dumps(dict(status="PASS",source_pins=len(m["source_pins"]),payload_files=len(m["payload"]),replay_rows=len(replay),request_timing_groups=len(counts))))


if __name__=="__main__":main()
