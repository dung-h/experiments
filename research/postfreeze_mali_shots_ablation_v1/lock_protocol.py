#!/usr/bin/env python3
"""Freeze the shots-ablation protocol and fit-only transforms before training."""
from __future__ import annotations
import hashlib, json, sys, platform
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/"artifacts/postfreeze_mali_shots_ablation_v1"
sys.path.insert(0,str(ROOT/"benchmark_v1/scripts"))
import run_qualified_logical_benchmark as runner
import run_mali_full_features as base
from shots_core import fit_shot_transform

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".partial");tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n");tmp.replace(path)

def main():
    prior=OUT/"protocol.json"
    amendment_version=0
    if prior.exists():
        training=OUT/"training"
        if training.exists() and any(training.rglob("checkpoint.pt")):
            raise SystemExit("refuse_protocol_change_after_any_training_checkpoint")
        versions=[]
        for item in OUT.glob("protocol_preoutcome_attempt_v*.json"):
            m=re.search(r"_v(\d+)\.json$",item.name)
            if m:versions.append(int(m.group(1)))
        amendment_version=max(versions,default=-1)+1
        archive=OUT/f"protocol_preoutcome_attempt_v{amendment_version}.json"
        prior.replace(archive)
        doc=OUT/"PROTOCOL.md"
        if doc.exists():doc.replace(OUT/f"PROTOCOL_PREOUTCOME_ATTEMPT_V{amendment_version}.md")
    reuse=json.loads((OUT/"baseline_reuse_receipt.json").read_text())
    if reuse.get("status")!="PASS":raise RuntimeError("baseline_reuse_not_validated")
    ctx=runner.load_context();ctx["panel_by_id"]={r["canonical_observation_id"]:r for r in ctx["panel"]}
    if len(ctx["panel"])!=4515 or len({r["group_id"] for r in ctx["panel"]})!=165:raise RuntimeError("frozen_cohort_mismatch")
    contract=json.loads(runner.CONTRACT.read_text())["neural"]
    fold_records=[]
    for fold in range(5):
        parts=runner.partitions(ctx,fold)
        raw=runner.neural_transform(ctx,parts)
        shot=fit_shot_transform([float(ctx["panel_by_id"][i]["shots"]) for i in parts["fit"]])
        transform={"global":raw["global_transform"],"node":raw["node_transform"],"shots":shot,
                   "fit_ids_sha256":runner.idhash(parts["fit"]),"fold":fold}
        p=OUT/"training"/f"fold_{fold}"/"transform.json"
        if p.exists():
            if json.loads(p.read_text())!=transform:raise RuntimeError(f"previous_preoutcome_transform_mismatch:{p}")
        else:write(p,transform)
        fold_records.append({"fold":fold,"fit_rows":len(parts["fit"]),"validation_rows":len(parts["validation"]),
            "test_rows":len(parts["test"]),"fit_ids_sha256":runner.idhash(parts["fit"]),
            "validation_ids_sha256":runner.idhash(parts["validation"]),"test_ids_sha256":runner.idhash(parts["test"]),
            "transform_sha256":sha(p),"shots_transform":shot})
    source_paths=[runner.OUT/"manifest.json",runner.OUT/"panel.csv",runner.OUT/"outer_splits.csv",runner.OUT/"inner_splits.csv",
        runner.OUT/"summary/oof_predictions.csv",runner.CONTRACT,ROOT/"artifacts/final_research/manifest.json",
        OUT/"baseline_reuse_receipt.json",OUT/"baseline_predictions.csv",Path(__file__),HERE/"run.py",HERE/"shots_core.py",
        HERE/"aggregate.py",HERE/"validate.py",HERE/"preflight.py",HERE/"test_shots.py"]
    protocol={"study":"postfreeze_mali_shots_ablation_v1","status":"LOCKED_PRE_OUTCOME",
        "scientific_question":"Does recorded shots input improve prediction within the same frozen Track-A architecture?",
        "cohort":{"observations":4515,"structural_groups":165,"outer_folds":5},
        "treatments":{"M0":"frozen global MLP, no added shots","M1":"same MLP plus fit-standardized raw shots",
            "G0":"frozen logical-DAG Graph+MLP, no added shots","G1":"same Graph+MLP plus fit-standardized raw shots"},
        "new_training_cells":30,"new_treatments":["M1","G1"],"outer_folds":[0,1,2,3,4],"seeds":[42,1234,31415],
        "epochs":500,"effective_batch":32,"graph_node_budget":250000,"precision":"float32_no_amp",
        "optimizer":{"name":"Adam","lr":0.0005,"weight_decay":0.0001},"loss":"unweighted raw-seconds MSE",
        "checkpoint_selection":"minimum inner-fold-0 validation MSE over all 500 epochs; earliest tie",
        "split_rule":"reuse frozen outer folds; inner fold 0 validation; inner folds 1-3 fit; no test-driven selection",
        "shot_feature":{"value":"raw shots","transform":"(shots - inner-fit mean) / inner-fit sample std; float64 fit, ddof=1, float32 model input",
            "constant_std_le_1e-6":"zero and record constant support","fit_only":True,"source":"qualified panel.csv",
            "not_used":"log shots, shot multiplier, shot curve, source/backend IDs"},
        "architecture_change":"append one zero-initialized input column to gf_linear1; copy existing weights/bias exactly; all other tensors unchanged; fresh optimizer",
        "primary_metric":"group-balanced MAE seconds","primary_contrasts":["G1-G0","M1-M0"],
        "bootstrap":{"unit":"165 structural groups","replicates":10000,"seed":42,"intervals":[0.95,0.975],
            "same_draws_across_contrasts":True,"95_percent_ci_descriptive; 97.5_percent_interval_familywise_for_two_primary_contrasts":True},
        "seed_reduction":"median of 3 seed predictions per row only with complete coverage",
        "baseline_reuse_receipt_sha256":sha(OUT/"baseline_reuse_receipt.json"),
        "baseline_predictions_sha256":sha(OUT/"baseline_predictions.csv"),
        "panel_sha256":sha(runner.OUT/"panel.csv"),"outer_split_sha256":sha(runner.OUT/"outer_splits.csv"),
        "inner_split_sha256":sha(runner.OUT/"inner_splits.csv"),"target_oof_source_sha256":sha(runner.OUT/"summary/oof_predictions.csv"),
        "freeze_manifest_sha256":sha(ROOT/"artifacts/final_research/manifest.json"),
        "qualified_protocol_sha256":sha(runner.CONTRACT),"runner_sha256":sha(HERE/"run.py"),"helpers_sha256":sha(HERE/"shots_core.py"),
        "baseline_model_source_sha256":base.load_pinned_model()[1],"source_files":{str(p.relative_to(ROOT)):sha(p) for p in source_paths},
        "folds":fold_records,"environment_lock":{"python":sys.executable,"python_version":sys.version,
            "platform":platform.platform(),"git_head":__import__("subprocess").check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()},
        "pre_outcome_amendment":{"version":amendment_version,"reason":"The launcher rejected the locked outer-fold representation during startup validation. The canary stopped before model construction or training; this corrected lock supersedes the archived attempt.","prior_protocol_sha256":sha(OUT/f"protocol_preoutcome_attempt_v{amendment_version}.json") if amendment_version else None},
        "preservation":{"no_frozen_artifact_write":True,"no_simulation_or_qpu_job":True,"no_model_search":True}}
    write(OUT/"protocol.json",protocol)
    lines=["# Ma–Li shots ablation protocol", "", "Status: `LOCKED_PRE_OUTCOME`. This is a bounded post-freeze extension. It does not change Scientific Freeze v2, saved A1 results, targets, or folds.",
        "", "## Question and comparison", "", "Test whether adding the recorded shot count improves OOF runtime prediction inside the same MLP and Graph+MLP architecture. M0/G0 are reused frozen baselines; only M1/G1 are newly trained.",
        "", "## Locked data and training", "", "- 4,515 observations, 165 structural groups, five existing outer folds.",
        "- Inner fold 0 selects checkpoints; inner folds 1–3 fit transforms and train. Outer test is untouched until prediction.",
        "- 500 epochs, batch 32, Adam (0.0005, weight decay 0.0001), float32, raw-seconds MSE; minimum validation MSE, earliest tie.",
        "- Seeds 42, 1234, 31415. New cells: 30 (2 architectures × 5 folds × 3 seeds).",
        "- Shots are raw counts standardized using only the fold's inner-fit mean and ddof=1 sample standard deviation. No log transform, multiplier, shot curve, source ID, or backend ID.",
        "- Only `gf_linear1` gains one input weight per output; the new column is zero at initialization and all existing tensors are copied unchanged. Fresh Adam optimizer.",
        "", "## Primary endpoint", "", "Group-balanced MAE in seconds. Primary paired contrasts: G1−G0 and M1−M0; negative favors adding shots. Use 10,000 paired structural-group bootstrap replicates with seed 42. Report 95% percentile CIs and 97.5% intervals for the two-contrast family. No equivalence claim when an interval includes zero.",
        "", "## Reuse and provenance", "", "The 30 frozen baseline cells were checksum/split/OOF validated before reuse. Their receipt and predictions are preserved in this output namespace. Detailed immutable inputs, fold transform hashes, and protocol pins are in `protocol.json`.",
        "", "## Shot-response limitation", "", "Ma–Li and QPack have fixed shot levels; pooled gains may reflect source/context separation. Qonductor has only 11 groups. Within-hash/backend shot diagnostics will use held-out OOF predictions only and will be described as observational, not causal.",
        "", "## Stop rule", "", "No result-driven architecture or hyperparameter search. Incomplete cells or missing paired predictions block interpretation. No QPU/simulator execution, no frozen artifact modification, no commit or push.", ""]
    (OUT/"PROTOCOL.md").write_text("\n".join(lines))
    print(json.dumps({"status":"LOCKED_PRE_OUTCOME","protocol_sha256":sha(OUT/"protocol.json"),"folds":fold_records},indent=2))
if __name__=="__main__":main()
