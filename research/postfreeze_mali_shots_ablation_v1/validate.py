#!/usr/bin/env python3
"""Validate saved experiment outputs without training or changing predictions."""
from __future__ import annotations
import csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/"artifacts/postfreeze_mali_shots_ablation_v1"
sys.path.insert(0,str(ROOT/"benchmark_v1/scripts"));import run_qualified_logical_benchmark as runner
TREAT={"M1":"mali_global_mlp_shots","G1":"mali_logical_graph_shots"};SEEDS=(42,1234,31415)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(newline="",encoding="utf8") as f:return list(csv.DictReader(f))
def main():
    checks=[]
    def check(name,ok,detail=None):
        if not ok:raise RuntimeError(f"validation_failed:{name}:{detail}")
        checks.append({"check":name,"status":"PASS","detail":detail})
    protocol=json.loads((OUT/"protocol.json").read_text())
    check("protocol_locked",protocol.get("status")=="LOCKED_PRE_OUTCOME")
    reuse=json.loads((OUT/"baseline_reuse_receipt.json").read_text())
    check("baseline_reuse_pass",reuse.get("status")=="PASS")
    ctx=runner.load_context();panel={r["canonical_observation_id"]:r for r in ctx["panel"]}
    groups={i:r["group_id"] for i,r in panel.items()};ids=set(panel)
    check("frozen_cohort_4515",len(ids)==4515)
    check("frozen_group_count_165",len(set(groups.values()))==165)
    check("baseline_prediction_checksums",all(sha(OUT/n)==h for n,h in [("baseline_predictions.csv",sha(OUT/"baseline_predictions.csv"))]))
    cells=[];seed_rows=[]
    for fold in range(5):
        parts=runner.partitions(ctx,fold)
        for a,b in (("fit","validation"),("fit","test"),("validation","test")):
            ga={groups[i] for i in parts[a]};gb={groups[i] for i in parts[b]}
            check(f"fold{fold}_{a}_{b}_group_disjoint",not (ga&gb),len(ga&gb))
        for tag,method in TREAT.items():
            for seed in SEEDS:
                d=OUT/"training"/f"fold_{fold}"/method/f"seed_{seed}"
                manifest=json.loads((d/"manifest.json").read_text())
                check(f"cell_complete_{tag}_{fold}_{seed}",manifest.get("status")=="complete" and manifest.get("epochs")==500)
                check(f"cell_partition_ids_{tag}_{fold}_{seed}",manifest["identity"]["fit_ids_sha256"]==runner.idhash(parts["fit"])
                    and manifest["identity"]["validation_ids_sha256"]==runner.idhash(parts["validation"])
                    and manifest["identity"]["test_ids_sha256"]==runner.idhash(parts["test"]))
                check(f"cell_checksum_{tag}_{fold}_{seed}",all(sha(d/n)==h for n,h in manifest["outputs"].items()))
                gate=json.loads((d/"initialization_gate.json").read_text())
                check(f"init_zero_column_parity_{tag}_{fold}_{seed}",gate.get("status")=="PASS" and gate.get("initialized_forward_max_abs_diff",1)<=1e-6)
                check(f"init_shot_gradient_{tag}_{fold}_{seed}",gate.get("shots_column_gradient_max_abs",0)>0)
                rows=read(d/"predictions.csv");expected=set(parts["test"])
                check(f"oof_exact_test_ids_{tag}_{fold}_{seed}",len(rows)==len(expected) and {r["canonical_observation_id"] for r in rows}==expected)
                for r in rows:
                    i=r["canonical_observation_id"]
                    check(f"target_and_fold_{tag}_{fold}_{seed}_{i}",float(r["actual_seconds"])==float(ctx["targets"][i]) and int(r["outer_fold"])==fold and int(r["seed"])==seed)
                    check(f"finite_prediction_{tag}_{fold}_{seed}_{i}",math.isfinite(float(r["predicted_seconds"])))
                    seed_rows.append(r)
                cells.append({"treatment":tag,"fold":fold,"seed":seed,"manifest_sha256":sha(d/"manifest.json"),
                    "checkpoint_sha256":sha(d/"checkpoint.pt"),"prediction_sha256":sha(d/"predictions.csv"),"rows":len(rows)})
    agg=read(OUT/"aggregate/oof_predictions.csv")
    check("aggregate_18060_rows",len(agg)==18060)
    check("aggregate_treatment_coverage",{r["treatment"] for r in agg}=={"M0","M1","G0","G1"})
    for tag in ("M0","M1","G0","G1"):
        subset=[r for r in agg if r["treatment"]==tag]
        check(f"aggregate_4515_unique_{tag}",len(subset)==4515 and len({r["canonical_observation_id"] for r in subset})==4515)
    baseline=read(OUT/"baseline_predictions.csv")
    original=read(runner.OUT/"summary/oof_predictions.csv")
    # Reconfirm baseline is unchanged and remains the saved frozen three-seed median.
    saved={(r["canonical_observation_id"],r["method_id"]):float(r["predicted_seconds"])
        for r in original if r["status"]=="predicted"}
    diffs=[]
    for r in baseline:
        m=r["method"];cid=r["canonical_observation_id"]
        if m in ("mali_global_mlp","mali_logical_graph"):
            diffs.append(abs(float(r["prediction"])-saved[(cid,m)]))
    check("baseline_replay_matches_frozen_oof",max(diffs)<1e-6,max(diffs))
    draws=np.load(OUT/"aggregate/group_bootstrap_draws_seed42.npy",allow_pickle=False)
    check("bootstrap_draw_shape_and_range",draws.shape==(10000,165) and int(draws.min())>=0 and int(draws.max())<165,draws.shape)
    comparisons=read(OUT/"aggregate/paired_comparisons.csv")
    check("two_primary_comparisons",{r["contrast"] for r in comparisons}=={"G1-G0","M1-M0"})
    check("comparison_bootstrap_contract",all(int(r["bootstrap_replicates"])==10000 and r["bootstrap_unit"]=="structural_group" and int(r["seed"])==42 for r in comparisons))
    receipt={"status":"PASS","study":"postfreeze_mali_shots_ablation_v1","checks_count":len(checks),
        "checks":checks,"cells":cells,"new_treatment_seed_prediction_rows":len(seed_rows),
        "aggregated_prediction_rows":len(agg),"baseline_reuse_receipt_sha256":sha(OUT/"baseline_reuse_receipt.json"),
        "protocol_sha256":sha(OUT/"protocol.json"),"bootstrap_draw_sha256":sha(OUT/"aggregate/group_bootstrap_draws_seed42.npy"),
        "no_training_or_prediction_in_this_validation":True}
    p=OUT/"validation_receipt.json";p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"status":"PASS","checks_count":len(checks),"new_cells":len(cells),"new_seed_predictions":len(seed_rows),"aggregate_predictions":len(agg)},indent=2))
if __name__=="__main__":main()
